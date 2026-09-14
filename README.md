# TGA: Selective Knowledge Admission

Code and research materials for **Selective Knowledge Admission for Heterogeneous Multi-Task Federated Learning**.

TGA asks which compatible sources should contribute to a receiving client. It selects a small set of sources within each parameter family's feasible pool, then normalizes their task-relevance weights. The receiver retains its local adapter and combines local and collaborative outputs through the existing backend.

## What is included

- `tga/admission.py`: source selection and weighting functions extracted without changing their bodies from the existing implementation.
- `examples/admission_demo.py`: a CPU example showing separate cross-size and same-model source pools.
- `configs/`: the TGA policy settings and DRAKE, HFLB, and HSSP client/task assignments.
- `hssp/`: source registry, recovered construction tools, frozen split manifests, and evaluation utilities for HSSP.
- `analysis/task_deltas.csv`: 116 task-level TGA/FedMosaic comparisons used for the paper visualization.

This release runs the admission example and HSSP verification independently. This is **not a standalone end-to-end training package**. Full VLM training additionally requires the original model/trainer stack, base models, prepared task data, and local adapters. No model weights or source images are included.

## Quick start

Use Python 3.10 or newer from the repository root:

```bash
python -m pip install -r requirements.txt
python -m examples.admission_demo
```

The example receives a synthetic relevance vector. For receiver 0, it selects clients 2 and 1 in the cross-size pool, and clients 1 and 4 in the same-model pool. Each family's weights sum to one; unselected sources have zero weight. This is an illustration of admission, not a training result.

## How TGA works

1. Form the feasible source pool for each receiver and parameter family. Selected-layer rank-space parameters support transfer across the configured 1B/3B models; ordinary LoRA parameters at other layers require the same model assignment.
2. Use the backend's completed task-relevance scores to select up to two sources in each pool.
3. Apply softmax with temperature 0.5 over the admitted sources, separately for each family.
4. Transfer the admitted adapter states through the backend. Preserve the receiver's local adapter and combine local and collaborative outputs through its gate.

The relation scores come from gradient-based task summaries in the shared representation model. The standalone admission module accepts these scores; it does not recompute task summaries. A same-model source may participate in both families.

### Policy configuration

The default configuration is `configs/tga.json`. Its integration settings are:

```text
--fedmosaic_native_sparse_sources True
--fedmosaic_tg_source_policy similarity
--fedmosaic_tg_topk 2
--fedmosaic_tg_min_sim -1.0
--fedmosaic_tg_hetero_quota 0
--fedmosaic_tg_source_scope all
--fedmosaic_tg_weight_policy auto
--softmax_temp 0.5
```

For the matched-random control, set `fedmosaic_tg_source_policy` to `random_matched`; `auto` then uses uniform weights. This matches same-model/cross-size source counts on the control's own trajectory. For uniform relevance top-2, retain `similarity` and set `fedmosaic_tg_weight_policy` to `uniform`.

The paper setting uses seed 1, rank 128, four tasks per client, five rounds per task, 94 local iterations per round, and effective batch size four.

## HSSP protocol

HSSP organizes 11 public sources from six professional domains into 40 client-task units:

| Item | Value |
|---|---:|
| Clients | 10 (six 1B, four 3B) |
| Tasks per client | 4 |
| Open-ended / multiple-choice units | 31 / 9 |
| Training records | 151,744 |
| Unique test records | 10,240 |

Validate the bundled metadata without downloading datasets:

```bash
python hssp/scripts/verify_release.py
```

For construction, download specifications, frozen-split replay, and saved-prediction scoring, see [the HSSP guide](hssp/README.md). The release includes sample identities and hashes, not question/answer bodies or images. Fresh seeded construction and replay of the historical frozen split are separate operations described there. Respect each source dataset's access terms and license.

## Evaluation

**Self** measures performance on each client's own tasks; **Others** measures performance on other clients' tasks. Multiple-choice tasks use accuracy. Open-ended tasks use the recorded token-set F1 implementation. The HSSP guide documents tokenization and averaging details and provides a scoring entry point.

The task-level CSV stores normalized scores and percentage-point differences. It supports recreating the task-difference analysis without training. The small differences between rounded table arithmetic and means computed from task scores are due to rounding order.

## Attribution

TGA uses the Co-LoRA/FedMosaic parameterization and transport implementation. Please credit [Co-LoRA: Collaborative Model Personalization on Heterogeneous Multi-Modal Clients](https://openreview.net/forum?id=0g5Dk4Qfh0) together with TGA when using this work. Third-party code and datasets retain their original terms; this repository does not grant new rights over them.

A formal paper citation will be added when publication metadata is available. Repository: https://github.com/zgh1216820-coder/TGA
