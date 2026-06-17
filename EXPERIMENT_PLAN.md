# MMPAE Scale-Up Experiment Plan

이 문서는 동료들이 같은 형식으로 scale-up 실험을 나누어 실행하고, 결과를 바로 채울 수 있도록 만든 기록 템플릿이다.

## 1. 실험 목표

- 0.354B급 HMMPAE+InfoNCE large 모델이 MLXP/H200 환경에서 실제 학습 가능한지 검증한다.
- `batch_size=512` OOM 이후 안정적으로 동작한 `batch_size=128` 설정을 기준으로 training curve와 평가 지표를 확보한다.
- 중간보고서에는 0.13B -> 0.355B 확장 근거와 현재 MLXP 재현 결과를 연결해서 보고한다.

## 2. 공통 준비물

| 항목 | 위치/값 | 확인 여부 | 담당자 | 비고 |
|---|---|---|---|---|
| GitHub repo clone | `https://github.com/promotion-kim/MMPAE-MLXP` |  |  |  |
| Docker image | `docker.io/<DH_USER>/mmpae:<TAG>` |  |  |  |
| kubeconfig | 로컬 경로 |  |  | commit 금지 |
| namespace | `<NAMESPACE>` |  |  |  |
| PVC | `<PVC_NAME>` mounted at `/data` |  |  |  |
| raw polyOne | `/data/polyone/*.parquet` |  |  | 14 shards |
| tokenized polyOne | `/data/polyone_tokenized/*.parquet` |  |  | 14 shards |
| PolyBERT tokenizer | `/data/polyBERT/` |  |  | 별도 전달 필요 |
| predictor checkpoint | `/data/ckpt/PolyBert_Regressor.pt` |  |  | Zenodo downloader |
| patched train script | `/data/scripts/train_HMMPAE.py` |  |  |  |

## 3. 권장 역할 분담

| 역할 | 담당자 | 작업 | 산출물 |
|---|---|---|---|
| 환경/재현 담당 |  | clone, image build/push, manifest render, smoke/load check | smoke/load-check 로그 |
| 데이터 담당 |  | raw polyOne download, PolyBERT copy, tokenization | `/data/polyone_tokenized` 검증 로그 |
| 12h pilot 담당 |  | `mmpae-scaleup-12h` 실행/모니터링 | epoch loss, final eval, GPU 사용량 |
| full run 담당 |  | `mmpae-scaleup` 200 epoch 실행 | final checkpoint, eval CSV |
| 보고서 담당 |  | 표 취합, 그래프, 중간보고서 문장화 | report tables/figures |

## 4. Training Configs

### E1. 12h Pilot Scale-Up

목적: full model size는 유지하면서 하루 미만으로 학습 가능성을 확인한다.

| 항목 | 값 |
|---|---|
| Job template | `k8s/mmpae-scaleup-12h-job.yaml` |
| Job name | `mmpae-scaleup-12h` |
| Exp name | `scaleup-h200-1gpu-bs128-e90` |
| Model | HMMPAE+InfoNCE large |
| Params | about `0.354B` |
| GPU | 1 x H200 |
| Dataset | polyOne 14 shards |
| Train split | ar, az, bg, bo, bq, em, fx, gk, hk, ho |
| Validation split | hu, hv |
| Test split | hw, hx |
| Batch size | 128 |
| Eval batch size | 512 |
| Epochs | 90 |
| Steps per epoch | 1024 |
| Eval interval | 30 |
| Learning rate | 1e-4 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| d_model / latent_dim | 1024 / 1024 |
| Encoder depth | 24 |
| Decoder depth | 12 |
| Attention heads | 16 |
| FFN dim | 2048 |
| Dropout | 0.0 |
| Loss type | CwA |
| Alpha / Beta / Temperature | 100 / 1000 / 0.2 |
| Expected runtime | about 10.5-12h plus cluster/image overhead |

### E2. Full 200-Epoch Reproduction

목적: 논문 구현 세팅과 더 가까운 200 epoch 결과를 확보한다.

| 항목 | 값 |
|---|---|
| Job template | `k8s/mmpae-scaleup-job.yaml` |
| Job name | `mmpae-scaleup` |
| Exp name | `scaleup-h200-1gpu-bs128` |
| Model | HMMPAE+InfoNCE large |
| Params | about `0.354B` |
| GPU | 1 x H200 |
| Batch size | 128 |
| Epochs | 200 |
| Steps per epoch | 1024 |
| Eval interval | 10 |
| Expected runtime | about 24-48h depending on eval overhead |

### Optional Follow-Up Configs

| Exp ID | Purpose | Key Change | Expected Runtime | Notes |
|---|---|---|---|---|
| `scaleup-base-bs256-e120` | smaller baseline | `model_size=base`, `batch_size=256`, `epochs=120` | TBD | 비교용, scale-up 주장은 약함 |
| `scaleup-large-bs128-e90-beta500` | InfoNCE sensitivity | `beta=500` | about 12h | 성능 민감도 |
| `scaleup-large-bs128-e90-beta2000` | InfoNCE sensitivity | `beta=2000` | about 12h | 성능 민감도 |
| `scaleup-large-bs128-e90-tau01` | temperature sensitivity | `temperature=0.1` | about 12h | contrastive alignment |

## 5. 실행 기록표

| Exp ID | 담당자 | Git commit | Image tag | Job name | Start time | End time | Status | Runtime | Notes |
|---|---|---:|---|---|---|---|---|---|---|
| `scaleup-h200-1gpu-bs128-e90` |  |  |  | `mmpae-scaleup-12h` |  |  |  |  |  |
| `scaleup-h200-1gpu-bs128` |  |  |  | `mmpae-scaleup` |  |  |  |  |  |

## 6. Resource / Throughput Table

| Exp ID | GPU type | #GPU | CPU | Memory | Peak GPU Memory | GPU Util. | it/s | sec/epoch | Image pull time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `scaleup-h200-1gpu-bs128-e90` | H200 | 1 | 16 | 128Gi |  |  |  |  |  |
| `scaleup-h200-1gpu-bs128` | H200 | 1 | 16 | 128Gi |  |  |  |  |  |

## 7. Training Loss Table

| Exp ID | Epoch | total_loss | ce_loss | mse_loss | eos_loss | contrast_loss | grad_norm | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
|  | 1 |  |  |  |  |  |  |  |
|  | 10 |  |  |  |  |  |  |  |
|  | 30 |  |  |  |  |  |  |  |
|  | 60 |  |  |  |  |  |  |  |
|  | 90 |  |  |  |  |  |  |  |
|  | 200 |  |  |  |  |  |  |  |

## 8. Evaluation Metrics Table

논문/계획서 기준 핵심 지표는 property prediction의 RMSE/R2, inverse design의 Validity, Tanimoto similarity, Target RMSE, Target R2다.

| Exp ID | Eval epoch | Prop RMSE ↓ | Prop R2 ↑ | Validity ↑ | Tanimoto ↑ | Target RMSE ↓ | Target R2 ↑ | Eval CSV |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `scaleup-h200-1gpu-bs128-e90` | 30 |  |  |  |  |  |  |  |
| `scaleup-h200-1gpu-bs128-e90` | 60 |  |  |  |  |  |  |  |
| `scaleup-h200-1gpu-bs128-e90` | 90 |  |  |  |  |  |  |  |
| `scaleup-h200-1gpu-bs128` | 200 |  |  |  |  |  |  |  |

## 9. 중간보고서 문장 템플릿

```text
본 중간 실험에서는 기존 MMPAE/HMMPAE 코드를 MLXP Kubernetes Job 형태로 컨테이너화하고,
polyOne 데이터셋 기반 전처리, PolyBERT predictor checkpoint 로딩, GPU smoke/load check를
거쳐 0.354B급 HMMPAE+InfoNCE large 모델의 실제 학습 구동을 확인하였다.

초기 batch_size=512 설정은 1 x H200 환경에서 CUDA OOM이 발생하여, batch_size=128로 조정하였다.
조정 후 동일한 large 모델 구조, full polyOne split, InfoNCE beta=1000, temperature=0.2 설정으로
학습이 안정적으로 진행되었으며, 12시간 pilot run은 90 epoch까지의 training/evaluation curve 확보를
목표로 한다.
```

## 10. 실행 명령 요약

```bash
export REPO_DIR=/path/to/MMPAE
export DH_USER=<your-dockerhub-user>
export TAG=<your-image-tag>
export KUBECONFIG=/path/to/your-kubeconfig.yaml
export NAMESPACE=<your-namespace>
export PVC_NAME=<your-pvc-name>

cd "$REPO_DIR"
mkdir -p k8s.local
for f in k8s/*.yaml; do
  sed \
    -e "s|__NAMESPACE__|$NAMESPACE|g" \
    -e "s|__IMAGE__|docker.io/$DH_USER/mmpae:$TAG|g" \
    -e "s|__PVC_NAME__|$PVC_NAME|g" \
    "$f" > "k8s.local/$(basename "$f")"
done

kubectl -n "$NAMESPACE" delete job mmpae-scaleup-12h --ignore-not-found
kubectl apply -f k8s.local/mmpae-scaleup-12h-job.yaml
kubectl -n "$NAMESPACE" logs -f job/mmpae-scaleup-12h -c main
```
