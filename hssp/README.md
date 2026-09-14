# HSSP: Heterogeneous Specialty Stream Protocol

HSSP organizes eleven public task sources into forty sequential client–task units for heterogeneous federated vision-language learning. This repository contains the recovered construction code, frozen task assignments, sample membership hashes, and evaluation contract supporting the paper.

**Verified on 2026-09-14:** all 80 processed train/test JSON files matched the frozen 2026-08-15 SHA256 manifest. All referenced image paths existed. This verifies the retained processed artifacts, not a fresh download-to-training reproduction.

| Property | Frozen protocol |
|---|---:|
| Source datasets / domains | 11 / 6 |
| Clients: 1B / 3B | 6 / 4 |
| Ordered tasks per client | 4 |
| Open-ended / multiple-choice task units | 31 / 9 |
| Training records | 151,744 |
| Unique test records | 10,240 |
| Scheduled sample-level test uses | 39,240 |

The shared test uses are not independent additional examples. HSSP is a task-stream protocol over existing data, not a newly collected image dataset.

## Start here: no data download required

Python 3.10 or newer is sufficient to inspect the release:

```bash
python scripts/verify_release.py
```

This validates hashes, sample ordering, subset sizes, train-subset disjointness, train/test image-path separation, and scheduled versus unique counts. For the retained processed files:

```bash
python scripts/verify_release.py --data-root /path/to/fedmosaic-root
```

The latter verifies the bytes of all 80 JSON files. Image existence was checked in the recorded export; this command does not hash image pixels.

## Three different reproducibility operations

1. **Verify the paper split:** use the frozen manifests. This is the release's primary evidence.
2. **Restore the paper split from normalized candidates:** match exact record hashes and original order; requires the original normalized record content. No images or question/answer text are bundled.
3. **Create a new seeded split:** use the portable converter. This is a new partition, not a claim to regenerate the historical partition.

```bash
# Check whether existing normalized candidates can reconstruct every frozen JSON.
# The input contains <source>/train/dataset-*.json and <source>/test/dataset-*.json.
python scripts/replay_frozen.py --candidates /path/to/dataset --check-only

# Restore into a new directory; existing output directories are rejected.
python scripts/replay_frozen.py --candidates /path/to/dataset --output-root outputs/replay

# Convert source archives and build a NEW partition using an explicit seed.
python -m pip install -r requirements.txt
python scripts/prepare_sources.py --raw-root raw --output-root outputs/new-seed-1 --seed 1
```

Raw source layouts must match [the construction notes](docs/CONSTRUCTION.md). Hugging Face repositories may have changed formats since the historical download. Source URLs are listed in [configs/sources.json](configs/sources.json); missing historical revisions are explicitly null. For a recorded Hugging Face ID, a new download requires a chosen full commit SHA:

```bash
python scripts/download_source.py --source DrivingVQA --revision FULL_40_CHARACTER_COMMIT_SHA --raw-root raw
```

This pins a new download. It does not establish that the commit was used for the paper.

## Rescore saved predictions

For a single task, supply an array of records containing `sentence` and `gt_sentence`; choice questions also require `input`. Remove aggregate-only summary records first. Choice parsing is extracted from the existing evaluator, including its historical heuristics.

```bash
python scripts/evaluate_saved.py --mode choice --predictions predictions.json
pip install -r requirements-eval.txt
python scripts/evaluate_saved.py --mode open --predictions predictions.json --tokenizer /path/to/exact/receiver/tokenizer
```

Open-ended scoring requires the same receiver tokenizer used in the experiment. The metric pools token-set overlaps across examples within a task; it is not mean per-example F1. This utility does not regenerate predictions or validate training configurations.

## Included files

- `configs/scenario_original.json`: historical client/task/model assignment.
- `configs/scenario.json`: the same assignment with two legacy metric labels corrected to F1.
- `configs/sources.json`: eleven source pointers and historical revision status.
- `manifests/tasks.csv`: all forty ordered client–task units and taxonomy.
- `manifests/membership.jsonl.gz`: row positions, generated IDs, image paths, and canonical record SHA256, without raw images or QA text.
- `manifests/frozen_files.json`: current-file comparison against the historical SHA256 manifest.
- `scripts/prepare_sources.py`: portable converter derived from the recovered script.
- `scripts/export_frozen_manifest.py`: read-only metadata export from retained processed data.
- `scripts/replay_frozen.py`: exact normalized-record replay.
- `src/hssp/metrics.py`: historical token-set F1 and task macro aggregation.
- `docs/FAIRNESS.md`: common protocol requirements and limits of baseline-fairness evidence.

## Scope and attribution

The historical builder did not set a random seed, and used filesystem/set iteration. The original download directory and exact Hugging Face commit revisions were not recovered. Consequently, a new raw download followed by reseeding is not certified to recreate the paper's sample membership. The frozen membership manifest prevents this gap from being hidden.

The portable converter adds explicit roots/seed, deterministic iteration where revised, safe archive checks, and output validation. Full raw-format conversion has not been re-executed in this release because the historical raw archives were unavailable at their recorded location.

Original source data retain their respective ownership and terms. This release does not redistribute images, question/answer bodies, or checkpoints, and does not apply a blanket license to third-party data. Historical scripts are included for attribution and inspection; this package does not assert a new license grant over code of unresolved ownership.

## Paper wording

> We provide the HSSP task assignments, recovered preprocessing code, evaluation contract, and hash-indexed frozen split manifests to support inspection of the experimental protocol.

Do not replace this with “fully reproducible from a single raw-data download” until raw source revisions and complete download-to-frozen replay have been verified. Add the actual repository URL to the paper only after publication.

中文说明：[HSSP构建说明](docs/中文构建说明.md) · [公平性与复核边界](docs/FAIRNESS.md)
