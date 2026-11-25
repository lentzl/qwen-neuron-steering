"""Plan or run frozen-weight inference and residual capture. Defaults to OFFLINE DRY RUN.

No remote-machine connection or scheduling is implemented here.
GPU dependencies are imported only after an explicit --execute.
"""
import argparse
import copy
import datetime
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
from common import ROOT, dataset_fingerprint, digest, file_sha, load_cases, read_json, write_json


def blocks_of(model):
    """Resolve decoder blocks explicitly; never silently hook the vision encoder."""
    for path in ("model.language_model.layers", "language_model.model.layers",
                 "language_model.layers", "model.layers"):
        obj = model
        try:
            for name in path.split("."):
                obj = getattr(obj, name)
        except AttributeError:
            continue
        if len(obj) and all(hasattr(block, "register_forward_hook") for block in obj):
            return path, obj
    raise RuntimeError("Unrecognized decoder layout. Inspect the architecture and add an audited block path.")


def output_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def install_hooks(blocks, torch, capture, vector=None, layer=None, strength=0.0):
    """Capture the first prefill's last position, before that block's intervention.

    Higher layers in a steered run can already reflect the upstream intervention.
    Only unsteered runs are eligible for fitting.

    A custom intervention applies at one decoder-block OUTPUT to the final
    prompt position and each generation step. It does NOT modify all prompt
    positions. This is deliberately different from the current UI's token mode.
    """
    handles = []
    for index, block in enumerate(blocks):
        def hook(module, inputs, output, index=index):
            hidden = output_tensor(output)
            if not torch.is_tensor(hidden) or hidden.ndim != 3 or hidden.shape[0] != 1:
                raise RuntimeError("Expected decoder output [1, sequence, hidden].")
            if index not in capture:
                capture[index] = hidden[0, -1].detach().float().cpu().numpy().copy()
            if vector is None or strength == 0 or index != layer:
                return output
            direction = torch.as_tensor(vector, device=hidden.device, dtype=torch.float32)
            if direction.numel() != hidden.shape[-1]:
                raise RuntimeError("Steering dimension does not match residual width.")
            norm = torch.linalg.vector_norm(direction)
            if not bool(torch.isfinite(norm)) or float(norm) == 0:
                raise RuntimeError("Steering vector is zero or nonfinite.")
            direction = direction / norm
            # Modify only the final position. Clone avoids in-place view/cache errors.
            result = hidden.clone()
            last = hidden[:, -1, :].float()
            delta = strength * torch.linalg.vector_norm(last, dim=-1, keepdim=True) * direction
            result[:, -1, :] = (last + delta).to(hidden.dtype)
            if isinstance(output, tuple):
                return (result,) + output[1:]
            return result
        handles.append(block.register_forward_hook(hook))
    return handles


def make_plan(args):
    if args.suite == "preposition" and args.vector:
        raise ValueError("Preposition controls are an unsteered wording diagnostic; do not pool them into direction fitting.")
    cases = load_cases(args.suite, args.split, args.pilot)
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError("No cases selected. Reporting tasks use validation or test, not discovery.")
    if args.pilot and args.suite != "pairs":
        raise ValueError("--pilot is defined only for the paired discovery suite.")
    if args.split == "test" and not args.frozen_selection:
        raise ValueError("Held-out tests require --frozen-selection with a validation-selected, fixed intervention manifest.")
    intervention = {"kind": "none", "strength": 0.0}
    if args.vector:
        if not args.vector.is_file():
            raise ValueError("Vector file does not exist.")
        if args.layer is None:
            raise ValueError("Specify --layer explicitly; decoder outputs use 0-based indices.")
        if not args.vector.with_suffix(".json").is_file():
            raise ValueError("Missing vector provenance sidecar.")
        intervention = {"kind": args.direction_kind, "strength": args.strength,
                        "layer": args.layer, "vector_file": str(args.vector.resolve()),
                        "vector_sha256": file_sha(args.vector),
                        "provenance_sha256": file_sha(args.vector.with_suffix(".json")),
                        "scope": "last prompt position and each generated position"}
    elif args.strength != 0 or args.direction_kind != "candidate":
        raise ValueError("A nonzero intervention or random control requires --vector.")
    plan = {"suite": args.suite, "split": args.split, "pilot": args.pilot,
            "model": args.model, "revision": args.revision, "dtype": args.dtype,
            "device_map": args.device_map, "seed": args.seed,
            "generation": {"temperature": args.temperature, "max_new_tokens": args.max_new_tokens,
                           "do_sample": args.temperature > 0, "top_p": 1.0, "top_k": 0,
                           "enable_thinking": False, "system_prompt": None},
            "capture": "All decoder block outputs at the last position of first prefill (assistant boundary, not necessarily final user token), before intervention at that block. Higher blocks may inherit steering.",
            "intervention": intervention, "datasets": dataset_fingerprint(args.suite), "cases": cases}
    if args.frozen_selection:
        frozen = read_json(args.frozen_selection)
        for key in ("model", "revision", "dtype", "seed", "generation", "datasets"):
            if frozen.get(key) != plan[key]:
                raise ValueError("Frozen selection mismatch: " + key)
        allowed = frozen.get("allowed_interventions", [])
        signature = {k: intervention.get(k) for k in ("kind", "strength", "layer", "vector_sha256")}
        if signature not in allowed:
            raise ValueError("Intervention not listed in frozen selection (including zero and random controls).")
        if frozen.get("chosen_from") != "validation" or not frozen.get("reviewer"):
            raise ValueError("Frozen selection must name a reviewer and validation provenance.")
        plan["frozen_selection_sha256"] = file_sha(args.frozen_selection)
    plan["plan_sha256"] = digest(plan)
    return plan


def execute(args, plan):
    if args.out is None:
        raise ValueError("--execute requires a new --out directory.")
    if args.out.exists():
        raise ValueError("Output already exists; use a new directory. Partial runs are never silently overwritten or resumed.")
    args.out.mkdir(parents=True)
    write_json(args.out / "plan.json", plan)
    write_json(args.out / "status.json", {"status": "starting", "no_gpu_allocation_claimed": True})
    if not args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch
    import transformers
    from transformers import AutoConfig, AutoTokenizer
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("No CUDA GPU available. Refusing an accidental large CPU run; --allow-cpu is explicit.")
    local = not args.allow_download
    kwargs = {"revision": args.revision, "local_files_only": local, "trust_remote_code": False}
    config = AutoConfig.from_pretrained(args.model, **kwargs)
    dtype = getattr(torch, args.dtype)
    if getattr(config, "model_type", "") in ("qwen3_5", "qwen3_5_moe"):
        model_class = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_class is None:
            raise RuntimeError("This runtime lacks Qwen3.5/3.6 multimodal support. Use a separate compatible environment.")
    else:
        model_class = transformers.AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model, **kwargs)
    model = model_class.from_pretrained(args.model, dtype=dtype, device_map=args.device_map,
                                        low_cpu_mem_usage=True, **kwargs)
    model.eval()
    path, blocks = blocks_of(model)
    config_dict = config.to_dict()
    model_identity = {"requested_model": args.model, "revision": args.revision,
                      "resolved_commit": getattr(config, "_commit_hash", None),
                      "config_sha256": digest(config_dict), "dtype": args.dtype,
                      "decoder_blocks": path, "layer_count": len(blocks)}
    # A local path is mutable: checksum the weights before using it as an identity.
    local_path = Path(args.model)
    if local_path.is_dir():
        weights = sorted(local_path.glob("*.safetensors")) or sorted(local_path.glob("pytorch_model*.bin"))
        if not weights:
            raise RuntimeError("No auditable weight files found in the local checkpoint directory.")
        model_identity["weight_sha256"] = {p.name: file_sha(p) for p in weights}
    elif not model_identity["resolved_commit"]:
        raise RuntimeError("Remote checkpoint did not resolve an immutable commit identity.")
    if args.frozen_selection and read_json(args.frozen_selection).get("model_identity") != model_identity:
        raise RuntimeError("Frozen selection checkpoint identity mismatch.")
    runtime = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
               "transformers": transformers.__version__, "numpy": np.__version__,
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "device_map": {k: str(v) for k, v in getattr(model, "hf_device_map", {}).items()},
               "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    vector = None
    if args.vector:
        provenance = read_json(args.vector.with_suffix(".json"))
        if provenance["model_identity"] != model_identity:
            raise RuntimeError("Vector/checkpoint identity mismatch; never transfer a vector across checkpoints silently.")
        if provenance["layer"] != args.layer:
            raise RuntimeError("Vector layer mismatch.")
        if provenance.get("fit_split") != "discovery" or provenance.get("datasets") != plan["datasets"]:
            raise RuntimeError("Vector must come from this dataset's discovery split.")
        if provenance.get("vector_sha256") != file_sha(args.vector):
            raise RuntimeError("Vector provenance checksum mismatch.")
        with np.load(args.vector, allow_pickle=False) as data:
            vector = data["direction"].astype(np.float32)
        if args.direction_kind == "random":
            vector = np.random.default_rng(args.seed).normal(size=vector.shape).astype(np.float32)
    write_json(args.out / "runtime.json", {"model_identity": model_identity, "runtime": runtime,
                                           "config": config_dict})
    (args.out / "activations").mkdir()
    generation = copy.deepcopy(model.generation_config)
    generation.do_sample = args.temperature > 0
    generation.temperature = args.temperature if args.temperature > 0 else 1.0
    generation.top_p = 1.0; generation.top_k = 0
    generation.max_new_tokens = args.max_new_tokens
    generation.num_beams = 1; generation.num_return_sequences = 1
    generation.repetition_penalty = 1.0
    if generation.pad_token_id is None:
        generation.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    write_json(args.out / "effective_generation.json", generation.to_dict())
    with (args.out / "responses.jsonl").open("x") as stream:
        for case in plan["cases"]:
            # Matched sampling starts from the same seed in each condition. Greedy
            # decoding still is not a guarantee of cross-device bitwise identity.
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            messages = [{"role": "user", "content": case["prompt"]}]
            encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                     enable_thinking=False, return_tensors="pt")
            device = model.get_input_embeddings().weight.device
            input_ids = encoded.to(device)
            attention_mask = torch.ones_like(input_ids)
            capture = {}
            handles = install_hooks(blocks, torch, capture, vector, args.layer, args.strength)
            try:
                with torch.inference_mode():
                    output = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                            generation_config=generation, use_cache=True)
            finally:
                for handle in handles:
                    handle.remove()
            if sorted(capture) != list(range(len(blocks))):
                raise RuntimeError("Not all decoder outputs were captured.")
            hidden = np.stack([capture[i] for i in range(len(blocks))])
            if not np.isfinite(hidden).all():
                raise RuntimeError("Nonfinite captured activations.")
            key = case["case_id"].replace("::", "__")
            activation_path = "activations/" + key + ".npz"
            np.savez_compressed(args.out / activation_path, hidden=hidden)
            generated = output[0, input_ids.shape[1]:].tolist()
            eos = generation.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            row = {"case_id": case["case_id"], "suite": case["suite"], "split": case["split"],
                   "prompt": case["prompt"], "response": tokenizer.decode(generated, skip_special_tokens=True),
                   "input_token_ids": input_ids[0].tolist(), "generated_token_ids": generated,
                   "decoded_prompt": tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False),
                   "chat_template_sha256": digest(tokenizer.chat_template),
                   "capture_position": input_ids.shape[1] - 1, "activation_file": activation_path,
                   "activation_sha256": file_sha(args.out / activation_path),
                   "generated_count": len(generated),
                   "stop_reason": "eos" if generated and generated[-1] in eos_ids else "length_or_other",
                   "model_identity": model_identity, "generation": plan["generation"],
                   "intervention": plan["intervention"], "plan_sha256": plan["plan_sha256"],
                   "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n"); stream.flush()
            print(json.dumps({"completed": case["case_id"], "generated_tokens": len(generated)}), flush=True)
    write_json(args.out / "status.json", {"status": "completed", "case_count": len(plan["cases"]),
                                         "responses_sha256": file_sha(args.out / "responses.jsonl")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["pairs", "reporting", "preposition"], default="pairs")
    parser.add_argument("--split", choices=["discovery", "validation", "test"], default="discovery")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--revision", default="main", help="Resolved commit and local weight hashes are recorded at execution")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--vector", type=Path)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--strength", type=float, default=0.0)
    parser.add_argument("--direction-kind", choices=["candidate", "random"], default="candidate")
    parser.add_argument("--frozen-selection", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.temperature) or not math.isfinite(args.strength) or args.temperature < 0 or args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1) or (args.layer is not None and args.layer < 0):
        parser.error("Invalid generation settings.")
    output_existed = bool(args.out and args.out.exists())
    try:
        plan = make_plan(args)
        if not args.execute:
            print(json.dumps({"status": "dry_run_no_model_loaded", "case_count": len(plan["cases"]), "plan": plan}, indent=2))
            if args.out:
                if args.out.exists():
                    raise ValueError("Dry-run output exists; refusing overwrite.")
                write_json(args.out / "plan.json", plan)
            return
        execute(args, plan)
    except Exception as error:
        if args.execute and not output_existed and args.out and args.out.exists() and (args.out / "status.json").exists():
            write_json(args.out / "status.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
