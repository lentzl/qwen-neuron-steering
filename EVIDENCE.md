# Evidence guide

The article links directly to its evidence records. This release preserves successes, failures, alternative grading and control arms where available.

| Question | Location |
|---|---|
| What exactly is the intervention? | `artifacts/`, `evidence/cited/recipe.json`, `reproduction/runtime/causal_patch.py` |
| What motivates the investigation? | `evidence/cited/lhasa.json`, `taiwan.json`, `fibre.json` |
| What are the full source-topic answers? | `evidence/cited/source-answers.json` |
| What happens on the fixed 32-question panel? | `evidence/cited/explored-answers.json`, `explored.json`, `evidence/engagement/fixed/` |
| What happens on the earlier 16-question panel? | `evidence/cited/earlier-labels.json`, `earlier.json`, `evidence/engagement/earlier/` |
| What happens on the reserved 12-question panel? | `evidence/cited/reserved.json`, `evidence/engagement/reserved/` |
| How is the intervention constructed? | `evidence/construction/`, `archive/construction/` |
| Can the saved candidates be traced to the final offsets? | `artifacts/discovery/`, `reproduction/check_discovery_lineage.py`, `provenance/discovery-recovery.json` |
| What are the coding completions and corrected test outcomes? | `evidence/coding/`, `evidence/cited/coding.json` |
| What are all persona measurements, including controls? | `evidence/persona/`, `evidence/cited/persona.json`, `persona-items.json` |
| Which source bytes were privacy-normalized? | `provenance/exports.json` |

The raw engagement runs include insensitive/control questions as well as the sensitive subsets summarized in the article. Do not count every raw record as a new sensitive-topic replicate. The reserved panel has four event clusters, with three sensitive forms per event; the earlier panel has four clusters and four sensitive forms per event. The fixed panel concerns previously explored topics. Reports retain their historical primary/secondary analysis labels; the article presents engagement first and factual correctness separately.

## Privacy normalization

This is a clean export with new Git history. Personal filesystem names, private machine paths, process/host identifiers and account metadata are removed or replaced. No credentials, private conversations, unrelated project history or base model weights are included. The illustration has text/EXIF/time metadata removed without changing pixels.

Scientific prompts, answers, token IDs, numerical measurements and labels are preserved. Generated model answers may contain factual errors, including the failures discussed in the article; they are evidence, not editorial assertions. Historical `ARCHIVE`, `WORKSPACE` and `REMOTE_HOME` strings are normalized provenance paths, not live links. Use this guide, article references and the release manifest to navigate the portable package.
