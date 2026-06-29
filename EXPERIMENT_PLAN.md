# MMPAE Scale-Up Experiment Plan

Each teammate owns one model size, runs the same training recipe for 200 epochs, and fills the evaluation table after the run.

## Training Configs

Common settings for all runs:

| Field | Value |
|---|---|
| Dataset | polyOne tokenized shards in `/data/polyone_tokenized` |
| Tokenizer | `/data/polyBERT` |
| Predictor checkpoint | `/data/ckpt/PolyBert_Regressor.pt` |
| Epochs | `200` |
| Steps / epoch | `1024` |
| Full eval | final epoch only |
| Inference interval | `10` epochs |
| Inference batches | `10` |
| Batch size | `128` |
| Eval batch size | `512` |
| Decoder layers | `12` |
| Training mode | MM-CwA (`--inverse` omitted/False) |
| Loss type | `CwA` |
| Alpha | `100` |
| Beta | `1000` |
| Temperature | `0.2` |
| Learning rate | `1e-4` |
| Weight decay | `1e-4` |
| GPU | `1 x H200` |

| Exp ID | Owner | Config file | Target size | `d_model` | `nhead` | `dim_feedforward` | Encoder layers | Decoder layers | Epochs | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `mmpae-0p35B-e200` |  | `configs/Inverse_CwA_0p35B.yaml` | 0.35B | 1024 | 16 | 2048 | 24 | 12 | 200 | Complete |
| `mmpae-1p5B-e200` |  | `configs/Inverse_CwA_1p5B.yaml` | 1.5B | 1792 | 28 | 3584 | 40 | 12 | 200 |  |
| `mmpae-3B-e200` |  | `configs/Inverse_CwA_3B.yaml` | 3B | 2304 | 36 | 4608 | 52 | 12 | 200 |  |
| `mmpae-8B-e200` |  | `configs/Inverse_CwA_8B.yaml` | 8B | 3584 | 56 | 7168 | 60 | 12 | 200 |  |

Use the same job template for every run and change only these two arguments:

```bash
--config_path configs/Inverse_CwA_1p5B.yaml
--exp_name mmpae-1p5B-e200
```

The `Target RMSE/R2` columns are computed by decoding from target properties and scoring generated PSMILES with the frozen PolyBERT property predictor. Do not pass `--inverse True` for these rows; that flag disables property prediction metrics in the current training code.

## Evaluation Metrics

Fill one row per completed run.

| Exp ID | Actual params from log | Final epoch | Runtime | Peak GPU memory | Final total_loss | Final ce_loss | Final mse_loss | Final contrast_loss | Best Prop RMSE ↓ | Best Prop R2 ↑ | Validity ↑ | Tanimoto ↑ | Target RMSE ↓ | Target R2 ↑ | Run directory | Notes |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `mmpae-0p35B-e200` | 354497835 | 200 | 23h 49m 30s | 54490.3 MB | 1837.9699 | 0.0642 | 1.2396 | 1.7139 | 0.2417 | 0.9453 | 0.9758 | 0.5631 | 0.3607 | 0.8712 | `/data/runs/mmpae-0p35B-e200` | Complete; metrics from `Eval_0200_metrics.json` |
| `mmpae-1p5B-e260` | 1495526443 | 260 | 2d 8h 58m 17s | 35349 MB observed | 3446.8454 | 0.0620 | 1.1143 | 1.6677 | 0.2058 | 0.9547 | 0.9773 | 0.5963 | 0.3166 | 0.8863 | `/data/runs/mmpae-1p5B-e260-2gpu-bs128-lr5e5-b2000` | Complete; metrics from `Eval_0260_metrics.json`; actual run id `mmpae-1p5B-e260-2gpu-bs128-lr5e5-b2000`; improved via `lr=5e-5`, `beta=2000`, `epochs=260`, `alpha=100`, `2 x H200`, `data_parallel=True`, `batch_size=128`, `eval_batch_size=512`, `infer_steps=10`; uploaded to `luca0621/mmpae-1p5B-e260-2gpu-bs128-lr5e5-b2000` |
| `mmpae-3B-e200` |  | 200 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `mmpae-8B-e200` |  | 200 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Previous 100-Epoch Result

| Exp ID | Actual params from log | Final epoch | Runtime | Peak GPU memory | Final total_loss | Final ce_loss | Final mse_loss | Final contrast_loss | Best Prop RMSE ↓ | Best Prop R2 ↑ | Validity ↑ | Tanimoto ↑ | Target RMSE ↓ | Target R2 ↑ | Run directory | Notes |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `mmpae-0p35B-e100` | 354497835 | 100 |  |  | 1324.6245 | 0.1203 | 3.1903 | 1.0055 |  |  |  |  |  |  |  | legacy 100-epoch run |
