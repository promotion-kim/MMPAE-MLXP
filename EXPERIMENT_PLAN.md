# MMPAE Scale-Up Experiment Plan

Each teammate owns one model size, runs the same training recipe for 100 epochs, and fills the evaluation table after the run.

## Training Configs

Common settings for all runs:

| Field | Value |
|---|---|
| Dataset | polyOne tokenized shards in `/data/polyone_tokenized` |
| Tokenizer | `/data/polyBERT` |
| Predictor checkpoint | `/data/ckpt/PolyBert_Regressor.pt` |
| Epochs | `100` |
| Steps / epoch | `1024` |
| Eval interval | `10` |
| Batch size | `128` |
| Eval batch size | `512` |
| Decoder layers | `12` |
| Loss type | `CwA` |
| Alpha | `100` |
| Beta | `1000` |
| Temperature | `0.2` |
| Learning rate | `1e-4` |
| Weight decay | `1e-4` |
| GPU | `1 x H200` |

| Exp ID | Owner | Config file | Target size | `d_model` | `nhead` | `dim_feedforward` | Encoder layers | Decoder layers | Epochs | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `mmpae-0p35B-e100` |  | `configs/Inverse_CwA_0p35B.yaml` | 0.35B | 1024 | 16 | 2048 | 24 | 12 | 100 |  |
| `mmpae-1B-e100` |  | `configs/Inverse_CwA_1B.yaml` | 1B | 1536 | 24 | 3072 | 32 | 12 | 100 |  |
| `mmpae-2B-e100` |  | `configs/Inverse_CwA_2B.yaml` | 2B | 2048 | 32 | 4096 | 40 | 12 | 100 |  |
| `mmpae-4B-e100` |  | `configs/Inverse_CwA_4B.yaml` | 4B | 2560 | 32 | 5120 | 58 | 12 | 100 |  |

Use the same job template for every run and change only these two arguments:

```bash
--config_path configs/Inverse_CwA_1B.yaml
--exp_name mmpae-1B-e100
```

## Evaluation Metrics

Fill one row per completed run.

| Exp ID | Actual params from log | Final epoch | Runtime | Peak GPU memory | Final total_loss | Final ce_loss | Final mse_loss | Final contrast_loss | Best Prop RMSE ↓ | Best Prop R2 ↑ | Validity ↑ | Tanimoto ↑ | Target RMSE ↓ | Target R2 ↑ | Run directory | Notes |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `mmpae-0p35B-e100` | 354497835 | 100 |  |  | 1324.6245 | 0.1203 | 3.1903 | 1.0055 |  |  |  |  |  |  |  |  |
| `mmpae-1B-e100` |  | 100 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `mmpae-2B-e100` |  | 100 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `mmpae-4B-e100` |  | 100 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
