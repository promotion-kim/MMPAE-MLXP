# MMPAE Scale-Up Experiment Sheet

This file is intentionally limited to training configurations and evaluation tables for the midterm report.

## Training Configurations

### E1. Large 12h Pilot

Use this as the first required scale-up run.

| Field | Value |
|---|---|
| Exp ID | `scaleup-h200-1gpu-bs128-e90` |
| Job template | `k8s/mmpae-scaleup-12h-job.yaml` |
| Model | HMMPAE + InfoNCE |
| Model size | `large` |
| Params | about `0.354B` |
| Dataset | polyOne, 14 parquet shards |
| Train split | `ar, az, bg, bo, bq, em, fx, gk, hk, ho` |
| Validation split | `hu, hv` |
| Test split | `hw, hx` |
| GPU | 1 x H200 |
| Batch size | `128` |
| Eval batch size | `512` |
| Epochs | `90` |
| Steps / epoch | `1024` |
| Eval interval | `30` |
| Learning rate | `1e-4` |
| Weight decay | `1e-4` |
| Optimizer | AdamW |
| Decoder layers | `12` |
| Loss type | `CwA` |
| Alpha | `100` |
| Beta | `1000` |
| Temperature | `0.2` |
| Expected runtime | about 10.5-12h plus cluster/image overhead |
| Purpose | Validate stable 0.354B training within about 12 hours |

### E2. Large Full Reproduction

Use this after E1 is stable.

| Field | Value |
|---|---|
| Exp ID | `scaleup-h200-1gpu-bs128` |
| Job template | `k8s/mmpae-scaleup-job.yaml` |
| Model | HMMPAE + InfoNCE |
| Model size | `large` |
| Params | about `0.354B` |
| Dataset | polyOne, 14 parquet shards |
| GPU | 1 x H200 |
| Batch size | `128` |
| Eval batch size | `512` |
| Epochs | `200` |
| Steps / epoch | `1024` |
| Eval interval | `10` |
| Learning rate | `1e-4` |
| Weight decay | `1e-4` |
| Optimizer | AdamW |
| Decoder layers | `12` |
| Loss type | `CwA` |
| Alpha | `100` |
| Beta | `1000` |
| Temperature | `0.2` |
| Expected runtime | about 24-48h depending on evaluation overhead |
| Purpose | Full single-GPU reproduction run |

### Optional Follow-Up Runs

Run these only after E1 succeeds.

| Exp ID | Change from E1 | Purpose | Expected runtime |
|---|---|---|---|
| `scaleup-large-bs128-e90-beta500` | `beta=500` | InfoNCE weight sensitivity | about 12h |
| `scaleup-large-bs128-e90-beta2000` | `beta=2000` | InfoNCE weight sensitivity | about 12h |
| `scaleup-large-bs128-e90-tau01` | `temperature=0.1` | Contrastive temperature sensitivity | about 12h |
| `scaleup-base-bs256-e120` | `model_size=base`, `batch_size=256`, `epochs=120` | Smaller control model | TBD |

## Evaluation Metrics

These tables are kept because they are the minimum set needed to compare runs and fill the midterm report. Setup, data-preparation, and execution commands are intentionally kept in `README.md`.

### 1. Training Configuration Record

| Exp ID | Model size | Params | GPUs | Batch | Eval batch | Epochs | Steps/epoch | Eval interval | Alpha | Beta | Temp. | LR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scaleup-h200-1gpu-bs128-e90` | large | 0.354B | 1 | 128 | 512 | 90 | 1024 | 30 | 100 | 1000 | 0.2 | 1e-4 |
| `scaleup-h200-1gpu-bs128` | large | 0.354B | 1 | 128 | 512 | 200 | 1024 | 10 | 100 | 1000 | 0.2 | 1e-4 |
|  |  |  |  |  |  |  |  |  |  |  |  |  |

### 2. Training Losses

| Exp ID | Epoch | total_loss | ce_loss | mse_loss | eos_loss | contrast_loss | grad_norm | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
|  | 1 |  |  |  |  |  |  |  |
|  | 10 |  |  |  |  |  |  |  |
|  | 30 |  |  |  |  |  |  |  |
|  | 60 |  |  |  |  |  |  |  |
|  | 90 |  |  |  |  |  |  |  |
|  | 200 |  |  |  |  |  |  |  |

### 3. Validation/Test Metrics

| Exp ID | Eval epoch | Prop RMSE ↓ | Prop R2 ↑ | Validity ↑ | Tanimoto ↑ | Target RMSE ↓ | Target R2 ↑ | Eval CSV |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `scaleup-h200-1gpu-bs128-e90` | 30 |  |  |  |  |  |  |  |
| `scaleup-h200-1gpu-bs128-e90` | 60 |  |  |  |  |  |  |  |
| `scaleup-h200-1gpu-bs128-e90` | 90 |  |  |  |  |  |  |  |
| `scaleup-h200-1gpu-bs128` | 200 |  |  |  |  |  |  |  |

### 4. Scale-Up Summary

| Model | Params | Training status | Best Prop RMSE ↓ | Best Prop R2 ↑ | Best Target RMSE ↓ | Best Target R2 ↑ | Key observation |
|---|---:|---|---:|---:|---:|---:|---|
| MMPAE reference small | 0.13B | reference |  |  |  |  |  |
| HMMPAE reference large | 0.355B | reference |  |  |  |  |  |
| Current E1 pilot | 0.354B |  |  |  |  |  |  |
| Current E2 full | 0.354B |  |  |  |  |  |  |
