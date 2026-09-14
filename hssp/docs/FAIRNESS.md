# Baseline fairness: what this repository establishes

The frozen assignments and membership hashes specify a common dataset contract. They do not establish that every historical baseline execution followed that contract. No baseline table values are fabricated or imported into this release as proof of uniform reruns.

For each compared run, retain the following evidence:

| Field | Required evidence |
|---|---|
| Data | processed-file manifest SHA256, scenario SHA256 |
| Model assignment | exact backbone IDs/revisions per client |
| Task schedule | order, rounds per task, local-step budget |
| Optimization | batch/accumulation, initialization and seed; method-specific differences disclosed |
| Evaluation | final-round/checkpoint rule, evaluator code SHA, tokenizer identifiers, Self/Others membership |
| Output identity | per-task prediction/result keys and missing-key validation |

Only data identity has been freshly checked here against all 80 retained files. Full baseline run equivalence remains an audit task. A shared scenario filename or a main-table entry is not sufficient evidence.

The historical open-ended metric depends on receiver tokenization; preserve its exact implementation when comparing existing results. Do not silently replace it with a different F1 implementation and continue to use old numbers.

## Submission wording

Supported: “We release the task assignment, recovered preprocessing code, evaluation contract, and frozen split manifests.”

Not established by this package alone: “All baselines were independently rerun with identical configurations” or “the full raw-download-to-result pipeline was reproduced.”
