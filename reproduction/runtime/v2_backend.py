"""Teacher-forced response extraction and controlled generation on the pinned model."""
import copy
import datetime
import json
import os
from pathlib import Path
import re
import time
from common import digest, file_sha, write_json
from runner import blocks_of, output_tensor

MODEL = "Qwen/Qwen3.6-27B"
REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"


def normalized_claim(text):
    return re.sub(r"\s+", " ", text.replace("*", "").replace("`", "")).strip().casefold()


def apply_delta(hidden, direction, strength, torch, scope="last", units="residual_fraction"):
    """Zero is an exact identity; clone only for nonzero interventions."""
    if strength == 0:
        return hidden
    if scope not in {"last", "all"} or units not in {"residual_fraction", "raw"}:
        raise ValueError("Unknown intervention convention")
    vector = torch.as_tensor(direction, device=hidden.device, dtype=torch.float32)
    if vector.ndim != 1 or vector.numel() != hidden.shape[-1] or not bool(torch.isfinite(vector).all()):
        raise ValueError("Invalid steering direction")
    norm = torch.linalg.vector_norm(vector)
    if float(norm) <= 1e-12:
        raise ValueError("Zero steering direction")
    selected = hidden[:, -1:, :].float() if scope == "last" else hidden.float()
    if units == "residual_fraction":
        delta = strength * torch.linalg.vector_norm(selected, dim=-1, keepdim=True) * vector / norm
    else:
        delta = strength * vector
    result = hidden.clone()
    if scope == "last":
        result[:, -1:, :] = (selected + delta).to(hidden.dtype)
    else:
        result = (selected + delta).to(hidden.dtype)
    return result


class Session:
    def __init__(self, snapshot, runtime_out):
        import numpy as np
        import torch
        import transformers
        from transformers import AutoConfig, AutoTokenizer, AutoModelForImageTextToText
        self.np = np; self.torch = torch
        snapshot = Path(snapshot)
        if snapshot.name != REVISION or len(list(snapshot.glob("*.safetensors"))) != 15:
            raise ValueError("Expected complete pinned Qwen3.6 snapshot")
        if torch.cuda.device_count() != 2:
            raise RuntimeError("This study requires its two allocated GPUs")
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        self.config = AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        print(json.dumps({"event": "loading_model", "revision": REVISION}), flush=True)
        started = time.monotonic()
        self.model = AutoModelForImageTextToText.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
            device_map="balanced", max_memory={0: "43GiB", 1: "43GiB", "cpu": "40GiB"},
            attn_implementation="sdpa", low_cpu_mem_usage=True)
        self.model.eval()
        self.block_path, self.blocks = blocks_of(self.model)
        if len(self.blocks) != 64:
            raise RuntimeError("Checkpoint decoder block count differs from plan")
        placement = {k: str(v) for k, v in self.model.hf_device_map.items()}
        if any(v in {"cpu", "disk"} for v in placement.values()):
            raise RuntimeError("Unexpected CPU/disk model offload; inspect allocation before inference")
        self.device = self.model.get_input_embeddings().weight.device
        self.identity = {"model": MODEL, "revision": REVISION, "dtype": "bfloat16", "decoder_blocks": self.block_path,
                         "layer_count": len(self.blocks), "config_sha256": digest(self.config.to_dict()),
                         "tokenizer_config_sha256": file_sha(snapshot / "tokenizer_config.json"),
                         "tokenizer_json_sha256": file_sha(snapshot / "tokenizer.json"),
                         "weight_index_sha256": file_sha(snapshot / "model.safetensors.index.json")}
        runtime = {"identity": self.identity, "torch": torch.__version__, "transformers": transformers.__version__,
                   "numpy": np.__version__, "model_load_seconds": time.monotonic() - started, "device_map": placement,
                   "GPUs": [torch.cuda.get_device_name(i) for i in range(2)], "base_runtime_modified": False,
                   "backend_sha256": file_sha(Path(__file__))}
        write_json(runtime_out, runtime)
        print(json.dumps({"event": "model_loaded", "seconds": runtime["model_load_seconds"], "device_map": placement}), flush=True)

    def replay_seed(self, seed, out):
        np = self.np; torch = self.torch
        if out.exists():
            raise ValueError("Replay output already exists")
        out.mkdir(parents=True)
        prompt_ids = seed["prompt_token_ids"]
        expected = self.tokenizer.apply_chat_template([{"role": "user", "content": seed["prompt"]}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        if list(expected) != prompt_ids:
            write_json(out / "tokenizer_mismatch.json", {"exported_ids": prompt_ids, "current_template_ids": list(expected),
                        "exported_decode": self.tokenizer.decode(prompt_ids), "current_decode": self.tokenizer.decode(expected)})
            raise RuntimeError("Exported prompt IDs differ from pinned tokenizer/template; inspect before replay")
        summaries = {}; details = {}
        for label, response in seed["responses"].items():
            ids = prompt_ids + response["token_ids"]
            mask = response["claim_span"]["sequence_token_indices"]
            selected_decode = self.tokenizer.decode([ids[i] for i in mask], skip_special_tokens=False)
            if normalized_claim(response["claim_span"]["text"]) not in normalized_claim(selected_decode):
                raise RuntimeError("Selected claim IDs do not decode to their annotated text: " + selected_decode)
            capture = {}; handles = []
            for layer, block in enumerate(self.blocks):
                def hook(module, inputs, output, layer=layer):
                    hidden = output_tensor(output)
                    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != len(ids):
                        raise RuntimeError("Unexpected replay activation shape")
                    selected = hidden[0, mask, :].detach()
                    if not bool(torch.isfinite(selected).all()):
                        raise RuntimeError("Nonfinite replay activations")
                    capture[layer] = {"mean": selected.float().mean(0).cpu().numpy(),
                                      "last": selected[-1].float().cpu().numpy(),
                                      "prompt_last": hidden[0, len(prompt_ids) - 1].float().cpu().numpy(),
                                      "claim_tokens": selected.to(torch.float16).cpu().numpy()}
                    return output
                handles.append(block.register_forward_hook(hook))
            try:
                with torch.inference_mode():
                    result = self.model(input_ids=torch.tensor([ids], device=self.device),
                                        attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device=self.device),
                                        use_cache=False, logits_to_keep=1, return_dict=True)
                del result
            finally:
                for handle in handles:
                    handle.remove()
            if sorted(capture) != list(range(64)):
                raise RuntimeError("Missing decoder blocks in replay")
            arrays = {key: np.stack([capture[i][key] for i in range(64)]) for key in ("mean", "last", "prompt_last", "claim_tokens")}
            np.savez_compressed(out / (label + ".npz"), **arrays)
            summaries[label] = arrays
            details[label] = {"selected_decode": selected_decode, "complete_decode": self.tokenizer.decode(ids, skip_special_tokens=False),
                              "sequence_ids": ids, "mask": mask, "source_had_steering": response["source_had_steering"],
                              "replay_had_steering": False, "activation_sha256": file_sha(out / (label + ".npz"))}
            print(json.dumps({"event": "replay_captured", "label": label, "claim_tokens": len(mask)}), flush=True)
        raw = summaries["false"]["mean"] - summaries["truthful"]["mean"]
        last = summaries["false"]["last"] - summaries["truthful"]["last"]
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        last_norms = np.linalg.norm(last, axis=1, keepdims=True)
        if not np.isfinite(raw).all() or (norms <= 1e-12).any() or (last_norms <= 1e-12).any():
            raise RuntimeError("Invalid replay contrast")
        prefix_difference = float(np.max(np.abs(summaries["false"]["prompt_last"] - summaries["truthful"]["prompt_last"])))
        if prefix_difference > .05:
            raise RuntimeError("Shared prompt activations diverge unexpectedly; inspect causal replay: " + str(prefix_difference))
        np.savez_compressed(out / "directions.npz", mean_raw=raw, mean_unit=raw/norms, last_raw=last, last_unit=last/last_norms)
        write_json(out / "provenance.json", {"identity": self.identity, "seed_sha256": digest(seed), "fit_split": "discovery",
                   "estimator": "false technical-claim mean minus truthful technical-claim mean", "direction_sign": "false minus truthful",
                   "details": details, "prefix_last_max_absolute_difference": prefix_difference,
                   "directions_sha256": file_sha(out / "directions.npz"), "pair_count": 1,
                   "status": "exploratory_single_pair_direction"})
        return raw / norms

    def encode(self, case):
        tokenizer = self.tokenizer
        if case.get("format") == "raw":
            text = case["raw_text"]
        else:
            messages = case.get("messages") or [{"role": "user", "content": case["prompt"]}]
            if case.get("assistant_prefix") is not None:
                messages = list(messages) + [{"role": "assistant", "content": case["assistant_prefix"]}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                                    continue_final_message=True, enable_thinking=False)
            else:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return tokenizer.encode(text, add_special_tokens=False), text

    def generate(self, case, direction=None, layer=None, strength=0, seed=1729, temperature=0,
                 max_new_tokens=256, scope="last", units="residual_fraction"):
        torch = self.torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        ids, formatted = self.encode(case)
        if not ids:
            raise ValueError("Empty model input")
        handles = []; calls = 0
        if direction is not None:
            if layer is None or not 0 <= layer < len(self.blocks):
                raise ValueError("Invalid intervention layer")
            def hook(module, inputs, output):
                nonlocal calls
                calls += 1
                hidden = output_tensor(output)
                modified = apply_delta(hidden, direction, strength, torch, scope, units)
                if strength == 0 and modified is not hidden:
                    raise RuntimeError("Zero-strength hook is not an identity")
                return (modified,) + output[1:] if isinstance(output, tuple) else modified
            handles.append(self.blocks[layer].register_forward_hook(hook))
        config = copy.deepcopy(self.model.generation_config)
        config.do_sample = temperature > 0; config.temperature = temperature if temperature > 0 else 1.
        config.top_p = 1.; config.top_k = 0; config.max_new_tokens = max_new_tokens
        config.num_beams = 1; config.num_return_sequences = 1; config.repetition_penalty = 1.
        config.pad_token_id = self.tokenizer.pad_token_id
        started = time.monotonic()
        try:
            with torch.inference_mode():
                result = self.model.generate(input_ids=torch.tensor([ids], device=self.device),
                         attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device=self.device),
                         generation_config=config, use_cache=True)
        finally:
            for handle in handles:
                handle.remove()
        generated = result[0, len(ids):].tolist()
        eos = config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        return {"case_id": case["case_id"], "prompt": case.get("prompt"), "response": self.tokenizer.decode(generated, skip_special_tokens=True),
                "input_token_ids": ids, "generated_token_ids": generated, "formatted_input": formatted,
                "generated_tokens": len(generated), "stop_reason": "eos" if generated and generated[-1] in eos_ids else "length_or_other",
                "seconds": time.monotonic() - started, "layer": layer, "strength": strength,
                "units": units, "scope": scope, "hook_calls": calls, "seed": seed,
                "temperature": temperature, "max_new_tokens": max_new_tokens,
                "identity": self.identity, "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
