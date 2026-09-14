# Recovered construction

The recovered `prepare_vqa_rad.py` on the workstation and server has SHA256 `68c868219d424de49bb1c4214b9a8a1a563e0e6b43ca33eb9a07ab1bb80daaf9`. Despite its name it handles all eleven sources. The separate `prepare_slake.py` normalizes paths and includes a broader historical whitelist; MedXpertQA is not part of the forty-unit frozen HSSP scenario.

## Dataflow

Public source archive → source-specific parser → RGB JPEG and LLaVA-style two-turn conversation → train/test pool → answer-conditioned Dirichlet subsets → frozen client/stage assignment.

Parsers select a single answer when a source supplies multiple answers. Multiple-choice options are rendered into the prompt and answers normalized to option letters. InspecSafe-V1 is converted from image annotations into an observation question and a textual list of labels; it is not an untouched original VQA benchmark.

## Raw input layouts expected by the recovered parsers

| Source | Input under `raw/<source>/` | Subsets |
|---|---|---:|
| vqa-rad | `data/*.parquet` with image bytes and question/answer columns | 1 |
| malaria-microscopy-vqa | `data/*.parquet`; choices and correct_answer are used | 4 |
| path-vqa | `data/*.parquet` | 5 |
| DocumentVQA | `data/*.parquet` | 4 |
| ChartQA | `data/*.parquet` | 4 |
| InfoVQA | `data/*.parquet` | 4 |
| ChemVQA-2K | `ChemVQA_2K_full.csv`, `ChemVQA_2K_images.zip` | 4 |
| MMAD | `metadata.csv`, image ZIP archives | 4 |
| DrivingVQA | `train.json` and test/val JSON, `images.zip` | 1 |
| InspecSafe-V1 | `train.tar`, `test.tar` containing annotations and images | 2 |
| Traffic-VQA | train/test JSON and image files under the source directory | 7 |

Source format checks are necessary before conversion. A current Hugging Face snapshot is not guaranteed to use this historical layout. Unsupported or missing input must not be interpreted as successful reproduction.

## Partition mechanism

- The twenty most frequent normalized answer strings define answer classes; other answers form a residual class.
- A Dirichlet distribution with alpha 0.5 supplies per-class subset probabilities.
- Samples are assigned without replacement, with a cap of 4,000 training samples per subset.
- A minimum-size repair moves examples from the largest subset to a small subset. The effective configured minimum is 500. It does not synthesize duplicates.
- A source's test pool is capped at 1,000 records and reused across its task subsets.
- ChemVQA-2K and MMAD pools are shuffled and divided 80/20 before subset allocation. Other parsers follow their source split files (validation files may enter the test pool).
- Historical random state, file order, and set order were not frozen by the builder. The recovered final membership, not a guessed seed, is the authority for the paper split.

## Tasks, sizes, and difficulty

`configs/scenario_original.json` assigns model IDs and four ordered source/subset entries to each client. The assignment is explicit, not inferred from model capacity during training. `manifests/tasks.csv` includes the zero-based stage and its final communication round (5, 10, 15, 20).

Difficulty labels are predefined ordinal protocol labels: simple, medium, complex, extreme. They are not calibrated psychometric measurements and are coupled to task and model assignments. Comparisons across these groups are descriptive.

The original InspecSafe metric metadata contained two Rouge-L labels. The corrected scenario changes only these labels to the actually used F1 contract; it changes neither samples nor predictions.

## Evaluation

Multiple-choice units use option accuracy under the experiment's answer extractor. Open-ended units encode the stripped prediction and single stored reference using the receiver tokenizer, form unique token-ID sets, sum intersections and set sizes across the task, then compute precision/recall F1. Tokenizer special-token defaults are retained. There is no best-of-multiple-references aggregation. This is not the mean of per-example F1, nor the separate caption-metric pipeline.

Self macro-averages the final client's own task keys. Others evaluates other clients' task keys. Shared canonical test samples must not be counted as independent new observations when estimating uncertainty.
