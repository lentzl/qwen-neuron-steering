# Making Qwen’s “crimethink” public

**A fixed, prompt-only intervention that reduces Qwen3.6-27B’s evasion of topics sensitive to the Chinese government.**

By **lentzl**, with **GPT-6 Astra through Codex** providing research, analysis and writing assistance.

[Read the investigation](ARTICLE.md) · [Reproduce the results](REPRODUCING.md) · [Evidence guide](EVIDENCE.md) · [Acknowledgments](ACKNOWLEDGMENTS.md)

We begin with J-Lens steering experiments in Neuronpedia, follow a contrast involving Chinese fibre-optic cables in Russian drones and CNC machine tools in Russian military production, and develop a fixed intervention on **892 MLP neurons across 20 layers**. It acts during prompt processing; the weights remain frozen and no donor prompt is needed during inference.

The main outcome is greater substantive engagement with sensitive questions:

| Panel | Native | Intervention |
|---|---:|---:|
| Previously explored topics, 32 questions | 25/32 | 30/32 |
| Earlier transfer panel, 16 questions | 6/16 | 16/16 |
| Reserved-event panel, 12 questions | 9/12 | 12/12 |

The panels have different exposure histories and are **not pooled**. Greater engagement does not guarantee a correct answer. The article separately reports factual quality, a small coding-score decrease (HumanEval+ **157/164 → 154/164**), and selected persona shifts resembling effects reported in prior honesty-steering work. These results do not establish a general honesty axis, a fully identified native mechanism, or an equivalence guarantee for other capabilities.

## What is released

- The illustrated article in Markdown and standalone HTML (`index.html`).
- The exact **95,163-byte offset artifact**, coordinate fingerprints and original inference hook.
- Full saved prompts, answers, token IDs, labels, scoring records and persona readouts supporting the reported results.
- Construction records, gradient-selection source and intermediate causal-experiment evidence.
- Recovered discovery tensors, with historical hash matches and a CPU check of the candidate-to-final-intervention lineage.
- A portable GPU inference entry point, CPU result checks and an operator test.
- A file manifest and explicit privacy-export provenance.

The base model is downloaded separately at its pinned revision. The release does **not** include transient full activation/gradient tensors that were not retained, unrelated experiments, credentials or operational chat history. Historical construction scripts are evidence snapshots; they are not a one-command reconstruction pipeline. See [reproduction scope and limits](REPRODUCING.md).

## Quick start: inspect results without a GPU

```bash
python3 reproduction/check_results.py
python3 reproduction/verify_release.py
```

To rebuild the reading copy:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-analysis.txt
.venv/bin/python reproduction/build_article.py
```

## Release policy and licenses

This repository is published as an archived, read-only research snapshot. It can be cloned and forked. A future update can be published by unarchiving and issuing another version.

Original code is under the [MIT license](LICENSE). The article, illustration and original research records are under [CC BY 4.0](LICENSE-DATA), attributed to **lentzl**. Third-party material retains its own terms; see [THIRD_PARTY.md](THIRD_PARTY.md). Model weights are not redistributed.
