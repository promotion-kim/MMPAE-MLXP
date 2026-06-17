# MMPAE Reproducible Run Notes

This repository contains the MMPAE/HMMPAE training code plus container and Kubernetes manifests for reproducing a single-GPU MLXP training run.

The checked-in Kubernetes manifests are templates. Each user should set their own Docker Hub user, image tag, namespace, PVC, and kubeconfig path before applying them.

## What Is Included

This repository includes:

- MMPAE/HMMPAE source code
- Dockerfile and Conda environment file
- Kubernetes Job/Pod templates
- polyOne raw-data downloader
- PolyBERT regressor checkpoint downloader
- preprocessing, smoke-test, load-check, and scale-up workflows

This repository does not include:

- kubeconfig files
- raw polyOne parquet files
- tokenized parquet files
- model checkpoints
- run outputs
- the local `polyBERT` tokenizer folder

The `polyBERT` tokenizer folder must be obtained separately and copied to `/data/polyBERT`. Without it, preprocessing and training will fail.

## Verified Run Shape

This workflow is not the full original paper-scale default. It is a practical 1 x H200 scale-up check that reaches real training:

- image: `docker.io/$DH_USER/mmpae:$TAG`
- namespace: `$NAMESPACE`
- PVC: `$PVC_NAME`, mounted at `/data`
- data: 14 polyOne parquet shards under `/data/polyone`
- tokenized data: `/data/polyone_tokenized`
- tokenizer: local `polyBERT` folder copied to `/data/polyBERT`
- predictor checkpoint: `/data/ckpt/PolyBert_Regressor.pt`
- training job: `k8s.local/mmpae-scaleup-job.yaml`
- training config: `large`, 1 GPU, `batch_size=128`, `eval_batch_size=512`, `epochs=200`, `steps=1024`, `interval=10`

`Property_Transformer.pt` from Zenodo record `17665048` is not the predictor checkpoint used by `train_HMMPAE.py`; it contains AE training state such as `AE`, `AE_ema`, and optimizer state. The predictor checkpoint used for this workflow is `PolyBert_Regressor.pt`.

## User-Specific Variables

Set these once in each shell session:

```bash
export REPO_DIR=/path/to/MMPAE
export DH_USER=<your-dockerhub-user>
export TAG=mmpae-YYYYMMDD
export KUBECONFIG=/path/to/your-kubeconfig.yaml
export NAMESPACE=<your-kubernetes-namespace>
export PVC_NAME=<your-pvc-name>
```

Example values are intentionally not hard-coded in this README. Do not commit kubeconfig files, raw data, checkpoints, or run outputs.

If a shared image is already available and public or pullable from MLXP, set `DH_USER` and `TAG` to that image owner and tag, then skip the build step. Otherwise, each user should build and push their own image.

Before submitting scale-up training, verify that all of these exist:

```text
docker.io/$DH_USER/mmpae:$TAG
$KUBECONFIG
$NAMESPACE
$PVC_NAME mounted at /data
/data/polyone/*.parquet
/data/polyone_tokenized/*.parquet
/data/polyBERT/
/data/ckpt/PolyBert_Regressor.pt
/data/scripts/train_HMMPAE.py
```

## Local Smoke Test

```bash
cd "$REPO_DIR"
conda activate main
python smoke_test.py
```

Expected result:

```text
smoke test passed
device=cpu
```

## Build and Push the Image

If Docker daemon access is available:

```bash
cd "$REPO_DIR"
docker login
docker build -t docker.io/$DH_USER/mmpae:$TAG -t docker.io/$DH_USER/mmpae:latest .
docker push docker.io/$DH_USER/mmpae:$TAG
docker push docker.io/$DH_USER/mmpae:latest
```

On servers without Docker daemon access, rootless Podman can be used:

```bash
cd "$REPO_DIR"

export XDG_RUNTIME_DIR=/tmp/run-user-$(id -u)-mmpae-overlay
export PODMAN_ROOT=/tmp/containers-$(id -u)-mmpae-overlay/storage
export PODMAN_RUNROOT=/tmp/run-user-$(id -u)-mmpae-overlay/containers
mkdir -p "$XDG_RUNTIME_DIR" "$PODMAN_ROOT" "$PODMAN_RUNROOT"

podman \
  --root "$PODMAN_ROOT" \
  --runroot "$PODMAN_RUNROOT" \
  --storage-driver overlay \
  --storage-opt mount_program=/usr/bin/fuse-overlayfs \
  --storage-opt ignore_chown_errors=true \
  build \
  -t docker.io/$DH_USER/mmpae:$TAG \
  -t docker.io/$DH_USER/mmpae:latest \
  .

podman login docker.io -u "$DH_USER"
podman \
  --root "$PODMAN_ROOT" \
  --runroot "$PODMAN_RUNROOT" \
  --storage-driver overlay \
  --storage-opt mount_program=/usr/bin/fuse-overlayfs \
  --storage-opt ignore_chown_errors=true \
  push docker.io/$DH_USER/mmpae:$TAG

podman \
  --root "$PODMAN_ROOT" \
  --runroot "$PODMAN_RUNROOT" \
  --storage-driver overlay \
  --storage-opt mount_program=/usr/bin/fuse-overlayfs \
  --storage-opt ignore_chown_errors=true \
  push docker.io/$DH_USER/mmpae:latest
```

## Render Kubernetes Manifests

The files in `k8s/` contain placeholders:

- `__NAMESPACE__`
- `__IMAGE__`
- `__PVC_NAME__`

Render user-specific manifests into ignored local files:

```bash
cd "$REPO_DIR"
mkdir -p k8s.local

for f in k8s/*.yaml; do
  sed \
    -e "s|__NAMESPACE__|$NAMESPACE|g" \
    -e "s|__IMAGE__|docker.io/$DH_USER/mmpae:$TAG|g" \
    -e "s|__PVC_NAME__|$PVC_NAME|g" \
    "$f" > "k8s.local/$(basename "$f")"
done
```

Check access:

```bash
kubectl -n "$NAMESPACE" get pods
kubectl -n "$NAMESPACE" get pvc "$PVC_NAME"
```

## GPU Smoke Test

```bash
kubectl -n "$NAMESPACE" delete job mmpae-smoke --ignore-not-found
kubectl apply -f k8s.local/mmpae-smoke-job.yaml
kubectl -n "$NAMESPACE" logs -f job/mmpae-smoke -c main
```

Expected result:

```text
smoke test passed
device=cuda
```

The first image pull can take several minutes because the image is about 5 GB.

## PVC Shell Pod

Create a helper pod that mounts `$PVC_NAME` at `/data`:

```bash
kubectl -n "$NAMESPACE" delete pod mmpae-pvc-shell --ignore-not-found
kubectl apply -f k8s.local/mmpae-pvc-shell-pod.yaml
kubectl -n "$NAMESPACE" wait --for=condition=Ready pod/mmpae-pvc-shell --timeout=300s
kubectl -n "$NAMESPACE" exec mmpae-pvc-shell -c main -- ls -lah /data
```

Use `/data` on the PVC for this workflow. Avoid relying on machine-local disks unless your cluster guarantees they are mounted in the training pod.

## End-to-End PVC Preparation

Run the following steps inside the MLXP helper pod workflow before submitting training:

1. Download raw polyOne parquet shards to `/data/polyone`.
2. Copy the local `polyBERT` tokenizer folder to `/data/polyBERT`.
3. Download `PolyBert_Regressor.pt` to `/data/ckpt`.
4. Tokenize raw polyOne into `/data/polyone_tokenized`.
5. Run `mmpae-load-check` before training.

The final PVC layout should be:

```text
/data/polyone/*.parquet
/data/polyone_tokenized/*.parquet
/data/polyBERT/
/data/ckpt/PolyBert_Regressor.pt
/data/scripts/polyone_token_extract.py
/data/scripts/train_HMMPAE.py
/data/runs/
```

## Raw polyOne Data

Copy and run the downloader inside the PVC shell pod:

```bash
kubectl -n "$NAMESPACE" exec mmpae-pvc-shell -c main -- mkdir -p /data/scripts /data/polyone
kubectl -n "$NAMESPACE" cp scripts/download_polyone_raw.py mmpae-pvc-shell:/data/scripts/download_polyone_raw.py -c main
kubectl -n "$NAMESPACE" exec -it mmpae-pvc-shell -c main -- \
  /opt/conda/envs/main/bin/python /data/scripts/download_polyone_raw.py --dest /data/polyone
```

Validate:

```bash
kubectl -n "$NAMESPACE" exec mmpae-pvc-shell -c main -- ls -lh /data/polyone
kubectl -n "$NAMESPACE" exec mmpae-pvc-shell -c main -- du -sh /data/polyone
```

The verified dataset is about 1.5 GB and contains these shards:

```text
polyOne_ar.parquet  polyOne_az.parquet  polyOne_bg.parquet  polyOne_bo.parquet
polyOne_bq.parquet  polyOne_em.parquet  polyOne_fx.parquet  polyOne_gk.parquet
polyOne_hk.parquet  polyOne_ho.parquet  polyOne_hu.parquet  polyOne_hv.parquet
polyOne_hw.parquet  polyOne_hx.parquet
```

Each full shard has 500,000 rows. `polyOne_hx.parquet` has 202,698 rows in the verified download.

## PolyBERT Tokenizer

The current workflow uses a local tokenizer folder named `polyBERT`. This repository does not provide a public downloader for it because public Hugging Face access to `kuelumbus/polyBERT` returned authorization errors during setup. Obtain the folder from the code/data owner or a shared internal storage location, then copy it into the PVC:

```bash
kubectl -n "$NAMESPACE" exec mmpae-pvc-shell -c main -- mkdir -p /data/polyBERT
kubectl -n "$NAMESPACE" cp ./polyBERT/. mmpae-pvc-shell:/data/polyBERT -c main
kubectl -n "$NAMESPACE" exec mmpae-pvc-shell -c main -- ls -lh /data/polyBERT
```

Expected files:

```text
added_tokens.json
config.json
special_tokens_map.json
spm.model
tokenizer.json
tokenizer_config.json
```

This folder does not include pretrained `AutoModel` weights. The training code falls back to constructing the encoder from config and then loads weights from `PolyBert_Regressor.pt`.

Validate tokenizer loading:

```bash
kubectl -n "$NAMESPACE" exec -i mmpae-pvc-shell -c main -- \
  /opt/conda/envs/main/bin/python - <<'PY'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/data/polyBERT")
print(type(tok).__name__, len(tok), tok.pad_token_id, tok.eos_token_id)
PY
```

Expected output:

```text
DebertaV2TokenizerFast 270 267 266
```

## Predictor Checkpoint

Download the predictor checkpoint from Zenodo record `17665048` using the included script:

```bash
kubectl -n "$NAMESPACE" exec mmpae-pvc-shell -c main -- mkdir -p /data/scripts /data/ckpt
kubectl -n "$NAMESPACE" cp scripts/download_polybert_regressor.py mmpae-pvc-shell:/data/scripts/download_polybert_regressor.py -c main
kubectl -n "$NAMESPACE" exec -it mmpae-pvc-shell -c main -- \
  /opt/conda/envs/main/bin/python /data/scripts/download_polybert_regressor.py
kubectl -n "$NAMESPACE" exec mmpae-pvc-shell -c main -- ls -lh /data/ckpt/PolyBert_Regressor.pt
```

Expected size is about 293 MiB.

Do not use `Property_Transformer.pt` as `--predictor_checkpoint` for `train_HMMPAE.py`; it is an AE training-state checkpoint, not the PolyBERT regressor state dict expected by the trainer.

## Tokenize polyOne

Copy the current preprocessing script to the PVC and submit the preprocessing job. This converts raw PSMILES strings to numeric token IDs and attention masks, which are the Transformer input format:

```bash
kubectl -n "$NAMESPACE" exec mmpae-pvc-shell -c main -- mkdir -p /data/scripts
kubectl -n "$NAMESPACE" cp polyone_token_extract.py mmpae-pvc-shell:/data/scripts/polyone_token_extract.py -c main

kubectl -n "$NAMESPACE" delete job mmpae-preprocess --ignore-not-found
kubectl apply -f k8s.local/mmpae-preprocess-job.yaml
kubectl -n "$NAMESPACE" logs -f job/mmpae-preprocess -c main
```

The script converts PSMILES strings to numeric token IDs and stores tokenized parquet files in `/data/polyone_tokenized`. With the optimized script, already-completed shards are skipped, and per-file batch progress is printed.

Validate:

```bash
kubectl -n "$NAMESPACE" exec mmpae-pvc-shell -c main -- ls -lh /data/polyone_tokenized
```

Each tokenized row contains:

- `smiles`
- `token_ids`
- `mask`
- `properties`

The verified token length is `160`; property vectors have length `37`, and training uses the first `29` properties.

Quick content check:

```bash
kubectl -n "$NAMESPACE" exec -i mmpae-pvc-shell -c main -- \
  /opt/conda/envs/main/bin/python - <<'PY'
import pandas as pd
from pathlib import Path
path = sorted(Path("/data/polyone_tokenized").glob("*.parquet"))[0]
df = pd.read_parquet(path)
print(path.name, df.shape)
print(df.columns.tolist())
print(len(df.iloc[0]["token_ids"]), len(df.iloc[0]["properties"]))
PY
```

Expected token/property lengths:

```text
160 37
```

## Load Check

Run this before submitting a long training job:

```bash
kubectl -n "$NAMESPACE" delete job mmpae-load-check --ignore-not-found
kubectl apply -f k8s.local/mmpae-load-check-job.yaml
kubectl -n "$NAMESPACE" logs -f job/mmpae-load-check -c main
```

Expected result:

```text
predictor.load_state_dict OK
load check passed
```

## Scale-Up Job YAML Format

Use `k8s/mmpae-scaleup-job.yaml` as the canonical scale-up job format. The checked-in file uses placeholders so each user can render it for their own environment:

- `__NAMESPACE__`: Kubernetes namespace
- `__IMAGE__`: pushed MMPAE image, for example `docker.io/$DH_USER/mmpae:$TAG`
- `__PVC_NAME__`: PVC that will be mounted at `/data`

The scale-up job should have this structure:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: mmpae-scaleup
  namespace: __NAMESPACE__
spec:
  backoffLimit: 0
  template:
    metadata:
      labels:
        private-h200-aipr-0-pod-default: "true"
    spec:
      restartPolicy: Never
      containers:
        - name: main
          image: __IMAGE__
          imagePullPolicy: Always
          command: ["/bin/bash", "-c"]
          args:
            - |
              set -euo pipefail
              export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
              cd /workspace
              if [ -f /data/scripts/train_HMMPAE.py ]; then
                cp /data/scripts/train_HMMPAE.py /workspace/train_HMMPAE.py
              fi
              /opt/conda/envs/main/bin/python smoke_test.py --device cuda
              /opt/conda/envs/main/bin/python train_HMMPAE.py \
                --data_path /data/polyone_tokenized \
                --predictor_checkpoint /data/ckpt/PolyBert_Regressor.pt \
                --tokenizer_path /data/polyBERT \
                --prefix /data/runs \
                --model_size large \
                --loss_type CwA \
                --temperature 0.2 \
                --beta 1000 \
                --alpha 100 \
                --epochs 200 \
                --batch_size 128 \
                --eval_batch_size 512 \
                --steps 1024 \
                --interval 10 \
                --dec_layers 12 \
                --exp_name scaleup-h200-1gpu-bs128
          resources:
            requests:
              cpu: "16"
              memory: "128Gi"
            limits:
              cpu: "16"
              memory: "128Gi"
              nvidia.com/gpu: 1
          volumeMounts:
            - name: data
              mountPath: /data
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: __PVC_NAME__
```

Cluster-specific scheduling fields may need to change. In this MLXP setup, the label `private-h200-aipr-0-pod-default: "true"` selects the H200 private zone through admission defaults. Other workspaces may need a different label, node selector, toleration, or priority class.

The required data layout inside the PVC is:

```text
/data/polyone_tokenized/*.parquet
/data/polyBERT/
/data/ckpt/PolyBert_Regressor.pt
/data/scripts/train_HMMPAE.py
/data/runs/
```

## Scale-Up Training

Copy the current training script into the PVC so the job uses the patched tokenizer/config fallback:

```bash
kubectl -n "$NAMESPACE" cp train_HMMPAE.py mmpae-pvc-shell:/data/scripts/train_HMMPAE.py -c main
```

Submit:

```bash
kubectl -n "$NAMESPACE" delete job mmpae-scaleup --ignore-not-found
kubectl apply -f k8s.local/mmpae-scaleup-job.yaml
kubectl -n "$NAMESPACE" get pods | grep mmpae-scaleup
kubectl -n "$NAMESPACE" logs -f job/mmpae-scaleup -c main
```

The current manifest runs:

```bash
python train_HMMPAE.py \
  --data_path /data/polyone_tokenized \
  --predictor_checkpoint /data/ckpt/PolyBert_Regressor.pt \
  --tokenizer_path /data/polyBERT \
  --prefix /data/runs \
  --model_size large \
  --loss_type CwA \
  --temperature 0.2 \
  --beta 1000 \
  --alpha 100 \
  --epochs 200 \
  --batch_size 128 \
  --eval_batch_size 512 \
  --steps 1024 \
  --interval 10 \
  --dec_layers 12 \
  --exp_name scaleup-h200-1gpu-bs128
```

`batch_size=512` caused CUDA OOM on 1 x H200. `batch_size=128` has been verified to start real training.

Outputs are written under:

```text
/data/runs/scaleup-h200-1gpu-bs128
```

For a shorter pilot run that keeps the same 0.354B large model but targets roughly 12 hours, use:

```bash
kubectl -n "$NAMESPACE" delete job mmpae-scaleup-12h --ignore-not-found
kubectl apply -f k8s.local/mmpae-scaleup-12h-job.yaml
kubectl -n "$NAMESPACE" logs -f job/mmpae-scaleup-12h -c main
```

The 12-hour template changes:

- `epochs`: `200` -> `90`
- `interval`: `10` -> `30`
- `exp_name`: `scaleup-h200-1gpu-bs128-e90`

This is a pilot setting for reporting training feasibility and early evaluation curves. Use the 200-epoch job for the fuller reproduction run.

## Monitoring

```bash
kubectl -n "$NAMESPACE" get pods | grep mmpae-scaleup
kubectl -n "$NAMESPACE" logs job/mmpae-scaleup -c main --tail=80
kubectl -n "$NAMESPACE" exec <mmpae-scaleup-pod> -c main -- nvidia-smi
```

In the verified run, epoch 1 took about 7 minutes for 1,024 training steps on one H200. The 200-epoch training loop is therefore roughly 23 hours before evaluation overhead. Because evaluation runs every 10 epochs and can be expensive, budget about 1 to 2 days for the full job unless the evaluation path is reduced.

## Notes for Reuse

- Set `DH_USER`, `TAG`, `NAMESPACE`, `PVC_NAME`, and `KUBECONFIG` for each user's environment.
- Render `k8s.local/*.yaml` before applying manifests.
- Update node selectors, labels, tolerations, and resource requests if another MLXP workspace uses different scheduling rules.
- Do not commit raw data, checkpoints, run outputs, kubeconfig files, or local model artifacts.
- This repository currently verifies single-GPU training. Multi-GPU/DDP scale-up requires a separate job design.
