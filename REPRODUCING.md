# Reproducing the study

There are three distinct levels of reproduction.

## 1. Recompute reported results from saved records

No model, network access or GPU is required:

```bash
python3 reproduction/check_results.py
python3 reproduction/verify_release.py
```

The first command checks engagement, correctness, coding scores and selected persona probabilities against the released records. It does not replace the manual judgments in the engagement/correctness labels. Full answers, rubrics and rationales are supplied so others can regrade them. Reviews are not represented as independent or fully blind when they were not.

The second checks every release-file hash, article links and the basic portable-file constraints. `provenance/exports.json` distinguishes original source hashes from privacy-normalized export hashes. Embedded hashes inside historical records usually identify **original** bytes, not the exported file's bytes.

## 2. Rerun the fixed intervention

The original run uses two NVIDIA RTX A6000 GPUs, BF16, PyTorch **2.11.0+cu128**, Transformers **5.6.2**, NumPy **2.3.5**, SDPA, and a balanced two-GPU placement. The preserved backend requires two visible CUDA GPUs and 15 model shards. Core library versions and the actual device map are recorded; a complete original environment lock was not retained. `accelerate` is a loader dependency whose original version is not asserted here. Differences in kernels, hardware or dependencies can change numerical results.

```bash
python3 -m venv .venv
.venv/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements-inference.txt
```

Download `Qwen/Qwen3.6-27B`, revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, using `huggingface_hub.snapshot_download`. Keep its returned snapshot directory (whose final component is the revision hash). Downloading approximately 54 GB of weights is separate from this repository.

```python
from huggingface_hub import snapshot_download
snapshot = snapshot_download(
    'Qwen/Qwen3.6-27B',
    revision='6a9e13bd6fc8f0983b9b99948120bc37f49c13e9',
    cache_dir='hf-cache',
)
print(snapshot)
```

Run with that path:

```bash
.venv/bin/python reproduction/run_inference.py \
  --snapshot PATH_TO_SNAPSHOT \
  --prompt 'Chinese CNC machine tools in Russian military production' \
  --arms native positive negative --out outputs/example

.venv/bin/python reproduction/run_inference.py \
  --snapshot PATH_TO_SNAPSHOT \
  --cases evidence/engagement/reserved/model-prompts-v1.json \
  --arms native positive negative --out outputs/reserved
```

Greedy decoding uses seed 1729 and a 2,048-token cap by default. The historical coding run uses a 4,096-token cap and its saved prompts. The final coefficient is **2.484356321608939**, not simply 2. The `.pt` file's `variants['mlp__complement']` contains the 892 selected coordinates. The other partitions are retained as part of the exact artifact.

The operator adds FP32 `alpha * delta` to selected BF16 MLP activations before `down_proj`, rounds back to BF16, and touches positions 3 through the final serialized prompt token. It removes its hooks after the single prefill. There are no decoding edits, updates to weights, or truthful donor prompts at inference.

The entry point checks model/tokenizer identity and coefficient fingerprints, writes runtime differences explicitly, preserves all generated token IDs, and refuses to overwrite existing outputs. The historical serialized configuration hash includes the installation path; the portable entry point instead verifies the exact pinned `config.json` bytes (SHA256 `69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c`). The original archived backend and hook are unchanged. It does not assert bitwise equivalence across different machines. This publication preparation runs CPU checks; it does not rerun the 27B evaluations on a GPU.

For a lightweight test of the actual hook, install CPU-compatible PyTorch 2.11.0 and run:

```bash
python3 reproduction/test_operator.py
```

This tests positive, negative and zero doses, position selection, removal before decoding, error cleanup and every selected coefficient fingerprint using synthetic activations. It is not a model-behavior test.

### Coding and persona evaluation

`evidence/coding` contains all 328 completions, paired scoring records and the task-32 evaluator correction. The final counts use the corrected oracle on the same saved completions, not regenerated answers. Original scoring and finalization code is supplied. Use isolated execution when rerunning generated code. The evaluator is EvalPlus commit `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`, HumanEvalPlus v0.1.10. Benchmark results can be recomputed from saved pass/fail records without executing any generated code.

`evidence/persona` contains prepared prompts, 135 × 16 items, native/positive/negative/random-sign readouts and label-order controls. These are next-token conditional Yes/No measurements, **not free-text declarations by the model**. Original `run_persona.py` and `analyze_persona.py` are included under `archive/persona`; they document the exact measurement and control definitions but retain historical environment assumptions. The portable inference CLI generates answers; it does not silently substitute generation for the persona measurement.

## 3. Reconstruct discovery and coordinate selection

`archive/construction` and `evidence/construction` document donor replacement, fixed memory-writer offsets, the key/value-only test, gradient scoring, selection of 1,024 coordinates, and exclusion of 132 coordinates in layers 19–20. The article explains the method step by step.

The code snapshots record the exact proxy, sign filter, normalization order and selection logic. The retained discovery artifacts are now also supplied under `artifacts/discovery/`: the memory-writer coefficients, top 4,096 ranked candidates with scores and per-topic mean deltas, the intermediate causal-offset package, and the quality-evaluation offset package. They are recovered from previously saved backups, and their hashes match the historical construction records. See [recovery provenance](provenance/discovery-recovery.json).

With PyTorch installed, verify the deterministic construction from these retained candidates through to the published intervention:

```bash
python3 reproduction/check_discovery_lineage.py
```

This checks candidate ordering and donor-delta averaging, the 256/1,024/4,096 candidate packages, the quality package and its random control, all six layer partitions, the final 892 coordinates and their exact coefficients, and the dose calculation. It runs on CPU and makes no model calls.

Full transient activation/gradient tensors and the all-coordinate ranking pool were not retained. Regenerating the original gradients and ranking from the base model therefore requires rerunning the documented captures. The historical launch/audit scripts also retain environment assumptions and are not a turnkey pipeline. We have verified the saved-candidate-to-final-intervention lineage; we have not rerun discovery end to end from the base weights. The initial release's statement about missing upstream binary artifacts is superseded by this recovery.
