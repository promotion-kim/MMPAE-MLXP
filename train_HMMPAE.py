import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoTokenizer

from models.MMTransformer import MMTransformerAR

import os

import math
import shutil
import argparse
import time
import json
import copy
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from contextlib import contextmanager

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger

import yaml
from dotted_dict import DottedDict
from torch.amp import autocast, GradScaler
from sklearn.preprocessing import StandardScaler

from libs.ldm.modules.ema import LitEma

from rdkit import Chem
from rdkit.Chem.BRICS import BRICSDecompose
from rdkit.Chem.Recap import RecapDecompose

import random
import pickle

from utils import (
    init_wandb,
    Standardize,
    save_checkpoint,
    seed_everything,
    logging_from_dict,
    decode_with_eos,
    compute_rmse,
    compute_r2,
    compute_grad_norm
)

def optional_path_arg(value):
    if value in [None, False]:
        return None
    if str(value).lower() in ["false", "none", "null", "0"]:
        return None
    return str(value)


parser = argparse.ArgumentParser(
    description="Arguments for Train HMMPAE"
)
parser.add_argument("--epochs", type=int, default=200, help="total epochs for training")
parser.add_argument("--start_epoch", type=int, default=0, help="start epochs for training")
parser.add_argument("--interval", type=int, default=10, help="epochs interval for evaluation")
parser.add_argument("--steps", type=int, default=1024, help="train steps per epoch")
parser.add_argument("--eval_steps", type=int, default=None, help="max eval batches; <= 0 means full eval loader")
parser.add_argument("--infer_steps", type=int, default=10, help="max inference batches for reconstruction metrics; <= 0 means full inference loader")
parser.add_argument("--final_eval_only", default=False, type=lambda s: s in ["True", "true", "1", 1, True], help="skip interval evals and evaluate only at the final epoch")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="save checkpoint every N epochs; <= 0 disables interval checkpoints")
parser.add_argument("--run_final_test", default=False, type=lambda s: s in ["True", "true", "1", 1, True], help="run a final test-set evaluation after training")
parser.add_argument("--hf_repo_id", type=str, default=None, help="optional Hugging Face Hub repo id for checkpoint uploads, e.g. user/repo")
parser.add_argument("--hf_upload_every", type=int, default=100, help="upload checkpoint every N epochs when --hf_repo_id is set; <= 0 disables uploads")
parser.add_argument("--hf_private", default=False, type=lambda s: s in ["True", "true", "1", 1, True], help="create Hugging Face repo as private; default is public")
parser.add_argument("--hf_token_env", type=str, default="HF_TOKEN", help="environment variable containing the Hugging Face write/read token")
parser.add_argument("--hf_upload_path", type=str, default="checkpoints", help="directory path inside the Hugging Face repo for uploaded checkpoints")
parser.add_argument("--hf_checkpoint_repo_id", type=str, default=None, help="Hugging Face repo id to download a checkpoint from for inference-only runs")
parser.add_argument("--hf_checkpoint_filename", type=str, default=None, help="checkpoint filename/path inside the Hugging Face repo")
parser.add_argument("--hf_checkpoint_revision", type=str, default=None, help="optional Hugging Face checkpoint revision")
parser.add_argument("--inference_only", default=False, type=lambda s: s in ["True", "true", "1", 1, True], help="load a checkpoint and run evaluation/inference without training")
parser.add_argument("--workers", default=8, type=int)

parser.add_argument("--batch_size", type=int, default=512, help="batch size per device for training Unet model", )
parser.add_argument("--eval_batch_size", type=int, default=2048, help="batch size per device for training Unet model", )
parser.add_argument("--data_size", type=str, default='base', choices=['small', 'base', 'middle', 'large'])

parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--wd", type=float, default=1e-4, help="weight decay degree")
parser.add_argument("--drop_rate", type=float, default=0.5, help="drop rate for train MMCwA", )

parser.add_argument("--prefix", type=str, default="./runs")
parser.add_argument("--data_path", type=str, default=None, help="directory containing preprocessed PolyOne parquet shards")
parser.add_argument("--predictor_checkpoint", type=str, default=None, help="PolyBERT property regressor checkpoint path")
parser.add_argument("--tokenizer_path", type=str, default=None, help="Hugging Face id or local PolyBERT tokenizer/model path")
parser.add_argument('--save', default=True, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--resume', default=None, type=optional_path_arg, help="local checkpoint path to resume/evaluate")
parser.add_argument('--exp_name', default='temp', type=str, required=False)

# AE related params
parser.add_argument('--AR', default=True, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--L2', default=True, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--fullrep', default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--ema', default=True, type=lambda s: s in ["True", "true", 1])

parser.add_argument('--num_properties', default=29, type=int)
parser.add_argument('--n_samples', default=10, type=int)

parser.add_argument("--model_size", type=str, default='base', choices=['small', 'base', 'middle', 'large'])
parser.add_argument('--property', default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--layer', type=str, default="multi")
parser.add_argument('--config_name', default='Inverse_CwA', type=str)
parser.add_argument("--config_path", type=str, default=None, help="explicit model config yaml path")

parser.add_argument("--wandb", default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument("--debug", default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument("--seed", default=1004, type=int)

parser.add_argument("--alpha", type=float, default=100.0, help="coefficient of MSE loss")
parser.add_argument("--beta", type=float, default=1000.0, help="coefficient of CwA loss")
parser.add_argument("--gamma", type=float, default=0.0, help="coefficient of eos loss")
parser.add_argument("--temperature", type=float, default=0.1, help="temperature of CwA loss")
parser.add_argument('--loss_type', type=str, default="CwA", help="[None, CwA, CwAsym] (both lower and upper cases are handled).")
parser.add_argument("--inverse", default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument("--deepp", default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--dec_layers', default=12, type=int)


parser.add_argument('--GC', default=True, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--bf16', default=True, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--decompose', default=False, type=lambda s: s in ["True", "true", 1])

args = parser.parse_args()
config = None
RDLogger.DisableLog('rdApp.*')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_config(args):
    fname = args.config_path or f'./configs/{args.config_name}_{args.model_size}.yaml'
    with open(fname, 'r') as y_file:
        yaml_file = yaml.load(y_file, Loader=yaml.FullLoader)
        config = DottedDict(dict(yaml_file))

    keys = list(args.__dict__.keys())
    values = list(args.__dict__.values())
    [setattr(config, keys[i], values[i]) for i in range(len(keys))]

    if args.tokenizer_path is not None:
        config.model.tokenizer_path = args.tokenizer_path

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.amp_dtype = torch.bfloat16 if config.bf16 and torch.cuda.is_bf16_supported() else torch.float16

    return config


def get_limited_steps(data_loader, max_steps):
    total_steps = len(data_loader)
    if max_steps is None or max_steps <= 0:
        return total_steps
    return min(total_steps, max_steps)


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes:d}m {seconds:02d}s"


def should_run_eval(config):
    if config.final_eval_only:
        return config.epoch == config.epochs
    interval_due = config.interval > 0 and config.epoch % config.interval == 0
    return interval_due or config.epoch == config.epochs


def should_run_inference(config):
    interval_due = config.interval > 0 and config.epoch % config.interval == 0
    return interval_due or config.epoch == config.epochs


def should_save_epoch_checkpoint(config):
    interval_due = config.checkpoint_interval > 0 and config.epoch % config.checkpoint_interval == 0
    return interval_due or config.epoch == config.epochs


def should_upload_epoch_checkpoint(config):
    if not config.hf_repo_id:
        return False
    interval_due = config.hf_upload_every > 0 and config.epoch % config.hf_upload_every == 0
    return interval_due or config.epoch == config.epochs


def checkpoint_payload(AE, AE_ema, optimizer, config):
    return {
        "epoch": config.epoch,
        "global_step": global_step,
        "AE": AE.state_dict(),
        "AE_ema": AE_ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": {
            "config_path": config.config_path,
            "tokenizer_path": config.model.tokenizer_path,
            "num_properties": config.num_properties,
            "dec_layers": config.dec_layers,
            "loss_type": config.loss_type,
            "alpha": config.alpha,
            "beta": config.beta,
            "gamma": config.gamma,
            "temperature": config.temperature,
            "inverse": config.inverse,
            "property": config.property,
            "deepp": config.deepp,
        },
    }


def upload_checkpoint_to_hf(config, checkpoint_path):
    if not should_upload_epoch_checkpoint(config):
        return

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as exc:
        print(f"[HF Upload] skipped: huggingface_hub is not installed ({exc})", flush=True)
        return

    token = os.environ.get(config.hf_token_env)
    repo_id = config.hf_repo_id
    remote_name = f"{config.hf_upload_path.rstrip('/')}/{Path(checkpoint_path).name}"

    print(f"[HF Upload] uploading {checkpoint_path} to {repo_id}/{remote_name}", flush=True)
    create_repo(repo_id=repo_id, token=token, private=config.hf_private, exist_ok=True, repo_type="model")
    HfApi(token=token).upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo=remote_name,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"[HF Upload] uploaded {repo_id}/{remote_name}", flush=True)


def resolve_checkpoint_path(config):
    if config.resume:
        return config.resume

    if not config.hf_checkpoint_repo_id or not config.hf_checkpoint_filename:
        return None

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Install huggingface_hub to download checkpoints from Hugging Face.") from exc

    token = os.environ.get(config.hf_token_env)
    print(
        f"[HF Download] downloading {config.hf_checkpoint_repo_id}/{config.hf_checkpoint_filename}",
        flush=True,
    )
    return hf_hub_download(
        repo_id=config.hf_checkpoint_repo_id,
        filename=config.hf_checkpoint_filename,
        revision=config.hf_checkpoint_revision,
        token=token,
        repo_type="model",
    )


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def peak_gpu_memory_mb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def experiment_plan_metrics(config, params, runtime_seconds, eval_metrics):
    return {
        "Exp ID": config.exp_name,
        "Actual params from log": int(params),
        "Final epoch": int(config.epoch),
        "Runtime": format_duration(runtime_seconds),
        "Peak GPU memory": peak_gpu_memory_mb(),
        "Final total_loss": eval_metrics.get("total_loss"),
        "Final ce_loss": eval_metrics.get("ce_loss"),
        "Final mse_loss": eval_metrics.get("mse_loss"),
        "Final contrast_loss": eval_metrics.get("contrast_loss"),
        "Best Prop RMSE ↓": eval_metrics.get("prop_rmse"),
        "Best Prop R2 ↑": eval_metrics.get("prop_r2"),
        "Validity ↑": eval_metrics.get("validity"),
        "Tanimoto ↑": eval_metrics.get("sim_score"),
        "Target RMSE ↓": eval_metrics.get("inv_rmse"),
        "Target R2 ↑": eval_metrics.get("inv_r2"),
        "Run directory": str(config.save_path),
        "Notes": f"infer_steps={config.infer_steps}, eval_steps={config.eval_steps}",
    }


def write_epoch_metrics(config, params, run_start_time, train_metrics, eval_metrics, score=None, is_best=False, checkpoint_path=None, prefix="Eval"):
    if not config.save:
        return

    runtime_seconds = time.time() - run_start_time
    best_checkpoint_path = Path(config.save_path, "Checkpoint_BEST.pt")
    record = {
        "exp_id": config.exp_name,
        "phase": prefix,
        "epoch": int(config.epoch),
        "epochs": int(config.epochs),
        "global_step": int(global_step),
        "actual_params": int(params),
        "runtime_seconds": runtime_seconds,
        "runtime": format_duration(runtime_seconds),
        "peak_gpu_memory_mb": peak_gpu_memory_mb(),
        "train": train_metrics or {},
        "eval": eval_metrics,
        "score": score,
        "is_best": bool(is_best),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path.exists() else None,
        "experiment_plan_metrics": experiment_plan_metrics(config, params, runtime_seconds, eval_metrics),
    }
    record = json_safe(record)

    metrics_path = Path(config.exp_path, f"{prefix}_{config.epoch:04d}_metrics.json")
    latest_path = Path(config.exp_path, f"latest_{prefix.lower()}_metrics.json")
    jsonl_path = Path(config.exp_path, "metrics.jsonl")

    with open(metrics_path, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(latest_path, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"==> Metrics saved: {metrics_path}", flush=True)


def metric_score(metrics, config):
    if config.property:
        return metrics["prop_r2"] - metrics["prop_rmse"]
    return metrics["inv_r2"] - metrics["inv_rmse"]


class PolyBertRegresser(nn.Module):
    def __init__(self, encoder, tokenizer, num_properties=37):
        super(PolyBertRegresser, self).__init__()

        self.tokenizer = tokenizer
        self.polybert = encoder
        self.polybert.resize_token_embeddings(len(tokenizer))

        if config.layer == "linear":
            self.Regressor = nn.Linear(self.polybert.config.hidden_size, num_properties)

        else:
            self.Regressor = nn.Sequential(
                nn.Linear(self.polybert.config.hidden_size, self.polybert.config.hidden_size),
                nn.GELU(),
                nn.Linear(self.polybert.config.hidden_size, num_properties),
            )

    def forward(self, input_ids, mask=None):
        outputs = self.polybert(
            input_ids=input_ids, attention_mask=mask
        )
        logits = outputs.last_hidden_state[:, 0, :]
        output = self.Regressor(logits)
        return output

    def forward_with_psmiles_batch(self, psmiles_list):
        encoding = self.tokenizer(
            psmiles_list,
            padding="max_length",
            max_length=160,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].cuda()
        attention_mask = encoding["attention_mask"].cuda()
        return self.forward(input_ids, attention_mask)


class EvalDataset(Dataset):
    def __init__(self):
        self.origin_psmiles = []
        self.recon_psmiles = []
        self.properties = []
        self.pred_properties = []

    def __len__(self):
        return len(self.origin_psmiles)

    def __getitem__(self, idx):
        origin_psmiles = self.origin_psmiles[idx]
        recon_psmiles = self.recon_psmiles[idx]
        properties = np.array(self.properties[idx])
        pred_properties = np.array(self.pred_properties[idx])

        return origin_psmiles, recon_psmiles, properties, pred_properties


class DummyDataset(Dataset):
    def __init__(self):
        self.psmiles = []
        self.properties = []
        self.token_indices = []
        self.masks = []

    def __len__(self):
        return len(self.psmiles)

    def __getitem__(self, idx):
        psmiles = self.psmiles[idx]
        psmiles = self.get_frag(psmiles)

        properties = np.array(self.properties[idx])
        token_idx = self.token_indices[idx]
        mask = self.masks[idx]

        return psmiles, properties, token_idx.astype(np.int16), mask.astype(np.bool_)


    def get_frag(self, psmiles):
        if config.decompose:

            m = Chem.MolFromSmiles(psmiles)
            res = RecapDecompose(m, minFragmentSize=1).GetAllChildren()
            res = list(res.keys())

            if len(res) >0:
                return random.choice(res)
            else:
                return psmiles

        else:
            return psmiles


def load_parquet_files_with_dataset(file_list, drop_cols=None):
    dataset = DummyDataset()

    for f in tqdm(file_list):
        df = pd.read_parquet(f, engine='pyarrow')
        if drop_cols:
            df = df.drop(columns=drop_cols, errors='ignore')

        dataset.psmiles.extend(list(df['smiles'].values))
        dataset.token_indices.extend(list(df['token_ids'].values))
        dataset.properties.extend(list(np.stack(df['properties'].values, axis=0)[:, :29]))
        dataset.masks.extend(list(np.arange(160) < df['mask'].values[:, None]))

    return dataset


def get_dataset(config):
    if config.data_path is None:
        raise ValueError("Set --data_path to the directory containing preprocessed PolyOne parquet files.")

    data_path = config.data_path

    train_files = ['polyOne_ar.parquet', 'polyOne_az.parquet', 'polyOne_bg.parquet', 'polyOne_bo.parquet', 'polyOne_bq.parquet', 'polyOne_em.parquet', 'polyOne_fx.parquet', 'polyOne_gk.parquet', 'polyOne_hk.parquet', 'polyOne_ho.parquet']
    val_files = ['polyOne_hu.parquet', 'polyOne_hv.parquet']
    test_files = ['polyOne_hw.parquet', 'polyOne_hx.parquet']

    train_files = [os.path.join(data_path, f) for f in train_files]
    val_files = [os.path.join(data_path, f) for f in val_files]
    test_files = [os.path.join(data_path, f) for f in test_files]

    train_dataset = load_parquet_files_with_dataset(train_files, drop_cols=["index", 'level_0'])
    val_dataset = load_parquet_files_with_dataset(val_files, drop_cols=["index", 'level_0'])
    test_dataset = load_parquet_files_with_dataset(test_files, drop_cols=["index", 'level_0'])

    train_prop, val_prop, test_prop = Standardize(train_dataset.properties, val_dataset.properties, test_dataset.properties)
    train_dataset.properties = train_prop
    val_dataset.properties = val_prop
    test_dataset.properties = test_prop

    train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True, num_workers=config.workers)
    val_loader = DataLoader(val_dataset, config.eval_batch_size, shuffle=False, num_workers=config.workers)
    test_loader = DataLoader(test_dataset, config.eval_batch_size, shuffle=False, num_workers=config.workers)
    infer_loader = DataLoader(val_dataset, config.batch_size, shuffle=True, num_workers=config.workers)

    return train_loader, val_loader, test_loader, infer_loader


def get_data_generator(loader):
    while True:
        try:
            psmiles, properties, token_idx, mask = next(data_iter)
        except:
            data_iter = iter(loader)
            psmiles, properties, token_idx, mask = next(data_iter)
        finally:
            psmiles, properties, token_idx, mask = psmiles, properties.cuda(), token_idx.long().cuda(), mask.bool().cuda()

        batch = {"psmiles": psmiles, "properties": properties, "input_ids": token_idx, "mask": mask}
        yield batch


@contextmanager
def ema_scope(AE, AE_ema):
    AE_ema.store(AE.parameters())
    AE_ema.copy_to(AE)
    AE_ema.restore(AE.parameters())


def on_train_batch_end(AE, AE_ema):
    AE_ema(AE)


def restoring(models, optimizer, config, load_optimizer=True):
    load_path = resolve_checkpoint_path(config)

    if load_path:
        assert Path(load_path).exists(), f"checkpoint not found: {load_path}"

        print("\n==> Restoring state dict from {}".format(load_path))
        AE, AE_ema, tokenizer, _ = models
        load_dict = torch.load(load_path, weights_only=False)
        AE.load_state_dict(load_dict['AE'])

        ema_dict = load_dict['AE_ema']

        if config.resume and 'MP' in config.resume:
            for k in list(ema_dict.keys()):
                if k.startswith('module'):
                    ema_dict[k[len('module'):]] = ema_dict[k]
                    del ema_dict[k]

        AE_ema.load_state_dict(ema_dict)

        if load_optimizer and optimizer is not None and 'optimizer' in load_dict:
            optimizer.load_state_dict(load_dict['optimizer'])

        global global_step
        global_step = load_dict.get('global_step', 0)
        config.start_epoch = load_dict['epoch']
        config.epoch = config.start_epoch

        return models, optimizer

    else:
        return models, optimizer


def init_train_setting(AE, config):
    optimizer = torch.optim.AdamW(AE.parameters(), lr=config.lr, weight_decay=config.wd)

    criterion = None
    scaler = GradScaler()
    wandb = init_wandb(config, project_id='project id', run_prefix=None)

    config.rdFingerprintGen = None

    return criterion, optimizer, scaler, wandb


def build_model(config):
    model_config = config.model

    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_path)
    params = model_config.params

    print('Model params:')
    for key in params.keys():
        print(key, ':', params[key])
    print()

    model = MMTransformerAR(
        tokenizer=tokenizer,
        vocab_size=len(tokenizer),
        latent_dim=params.latent_dim,
        d_model=params.d_model,
        nhead=params.nhead,
        dim_feedforward=params.dim_feedforward,
        num_layers=params.num_layers,
        dec_layers=config.dec_layers,
        activation=params.activation,
        bias=params.bias,
        norm_first=params.norm_first,
        dropout=params.dropout,
        pad_token_id=tokenizer.pad_token_id,
        alpha=config.alpha,
        beta=config.beta,
        gamma=config.gamma,
        temperature=config.temperature,
        num_properties=config.num_properties,
        fullrep=config.fullrep,
        L2=config.L2,
        loss_type=config.loss_type,
        inverse=config.inverse,
        property=config.property,
        deepp=config.deepp,
    ).to(config.device)

    model_ema = None
    if config.ema:
        model_ema = LitEma(model).to(config.device)

    print(f"CONFIG.property: {config.property}")
    print(f"CONFIG.inverse: {config.inverse}")
    from transformers import AutoModel

    try:
        encoder = AutoModel.from_pretrained(model_config.tokenizer_path).to(config.device)
    except OSError as exc:
        print(f"AutoModel.from_pretrained failed for {model_config.tokenizer_path}: {exc}")
        print("Falling back to AutoModel.from_config; predictor checkpoint must provide encoder weights.")
        encoder_config = AutoConfig.from_pretrained(model_config.tokenizer_path)
        encoder = AutoModel.from_config(encoder_config).to(config.device)
    predictor = PolyBertRegresser(encoder=encoder, tokenizer=tokenizer, num_properties=config.num_properties).to(config.device)

    if config.predictor_checkpoint is None:
        raise ValueError("Set --predictor_checkpoint to the PolyBERT property regressor checkpoint.")

    load_path = Path(config.predictor_checkpoint)
    if not load_path.exists():
        raise FileNotFoundError(f"Predictor checkpoint not found: {load_path}")

    predictor.load_state_dict(torch.load(load_path, map_location=config.device)['state_dict'])

    predictor.eval()
    predictor.requires_grad_(False)

    return model, model_ema, tokenizer, predictor


def compute_inverse_metrics(predictor, origin_psmiles, recon_psmiles, property_values, pred_property_values, config):
    dataset = EvalDataset()
    dataset.origin_psmiles = origin_psmiles
    dataset.recon_psmiles = recon_psmiles
    dataset.properties = np.array(property_values)
    dataset.pred_properties = np.array(pred_property_values)

    temp_loader = DataLoader(dataset, config.batch_size, shuffle=False, num_workers=config.workers)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    inv_rmse, inv_r2 = 0, 0
    valid_sample = 0
    
    prop_rmse, prop_r2 = 0, 0
    prop_sample = 0

    sim_scores = []

    with tqdm(total=len(temp_loader), desc=f"Metric epoch {config.epoch}", dynamic_ncols=True) as pbar:
        for idx, (origin_p, recon_p, prop, pred_prop) in enumerate(temp_loader):
            properties = prop.cuda()
            pred_properties = pred_prop.cuda()
            batch_valid_mask = []

            # Compute Property Prediction Metrics
            if not config.inverse:
                prop_rmse += compute_rmse(pred_properties, properties).item() * len(properties)
                prop_r2 += compute_r2(pred_properties, properties).item() * len(properties)
                prop_sample += len(properties)

            # Compute Property Prediction Metrics
            for gt_p, gen_p in zip(origin_p, recon_p):
                try:
                    mol_gen = Chem.MolFromSmiles(gen_p)
                    fp_gen = generator.GetFingerprint(mol_gen)

                    mol_gt = Chem.MolFromSmiles(gt_p)
                    fp_gt = generator.GetFingerprint(mol_gt)

                    sim_score = DataStructs.TanimotoSimilarity(fp_gt, fp_gen)
                    sim_scores.append(sim_score)

                    batch_valid_mask.append(True)

                except:
                    batch_valid_mask.append(False)

            valid_idx = [i for i, valid in enumerate(batch_valid_mask) if valid]
            if len(valid_idx) != 0:
                valid_sample += len(valid_idx)

                valid_gen = [recon_p[i] for i in valid_idx]
                valid_properties = properties[valid_idx]

                gen_pred = predictor.forward_with_psmiles_batch(valid_gen).squeeze()
                inv_rmse += compute_rmse(gen_pred, valid_properties).item() * len(valid_idx)
                inv_r2 += compute_r2(gen_pred, valid_properties).item() * len(valid_idx)

            pbar.set_postfix(valid=valid_sample)
            pbar.update(1)


    if not config.inverse and prop_sample > 0:
        prop_rmse = prop_rmse / prop_sample
        prop_r2 = prop_r2 / prop_sample

    inv_rmse = inv_rmse / valid_sample if valid_sample != 0 else 0
    inv_r2 = inv_r2 / valid_sample if valid_sample != 0 else 0

    validity = valid_sample / len(recon_psmiles) if len(recon_psmiles) > 0 else 0
    sim_score = np.mean(sim_scores) if len(sim_scores) > 0 else 0.0

    return validity, sim_score, inv_rmse, inv_r2, prop_rmse, prop_r2


def inference(models, data_loader, config):
    AE, AE_ema, tokenizer, predictor = models

    AE.eval()
    AE_ema.eval()

    origin_psmiles, recon_psmiles = [], []
    property_values = []
    pred_property_values = []

    infer_steps = get_limited_steps(data_loader, config.infer_steps)
    infer_start = time.time()
    max_decode_steps = getattr(AE, "max_sequence_length", 0)
    total_decode_steps = infer_steps * max_decode_steps if max_decode_steps else 0
    last_decode_log = 0.0

    print(f"\n## Inference ({infer_steps}/{len(data_loader)} batches)", flush=True)
    with torch.no_grad():
        with AE_ema.ema_scope(AE):
            with tqdm(total=infer_steps, desc=f"Infer epoch {config.epoch}", dynamic_ncols=True) as pbar:
                for idx, batch in enumerate(data_loader):
                    if idx >= infer_steps:
                        break

                    psmiles, properties, input_ids, mask = batch[0], batch[1].cuda(), batch[2].long().cuda(), batch[3].cuda()
                    batch_start = time.time()
                    print(
                        f"[Inference] batch {idx + 1}/{infer_steps} start "
                        f"(batch_size={len(psmiles)}, elapsed={format_duration(batch_start - infer_start)})",
                        flush=True,
                    )

                    def log_decode_progress(step, total):
                        nonlocal last_decode_log
                        now = time.time()
                        if step != 1 and step != total and now - last_decode_log < 30:
                            return

                        decode_done = idx * total + step
                        elapsed = now - infer_start
                        eta = 0.0
                        if total_decode_steps and decode_done > 0:
                            eta = elapsed * (total_decode_steps - decode_done) / decode_done

                        print(
                            f"[Inference] batch {idx + 1}/{infer_steps} decode {step}/{total} "
                            f"(decode_steps={decode_done}/{total_decode_steps}, "
                            f"elapsed={format_duration(elapsed)}, eta={format_duration(eta)})",
                            flush=True,
                        )
                        last_decode_log = now

                    with autocast(device_type='cuda', dtype=config.amp_dtype):
                        pred_tokens = AE(properties, drop_rate=0.0, mode='infer_psmiles', progress_callback=log_decode_progress)
                        pred_tokens = [torch.tensor(seq, dtype=torch.long, device='cuda') for seq in pred_tokens]
                        recon = decode_with_eos(pred_tokens, tokenizer, tokenizer.eos_token_id, tokenizer.pad_token_id)

                        if not config.inverse:
                            pred_properties = AE(token_ids=input_ids, drop_rate=0.0, mode='infer_properties')
                        else:
                            pred_properties = torch.zeros_like(properties, dtype=torch.float32, device=properties.device)

                    origin_psmiles.extend(list(psmiles))
                    recon_psmiles.extend(recon)
                    property_values.extend(properties.detach().cpu().numpy())
                    pred_property_values.extend(pred_properties.detach().cpu().float().numpy())

                    pbar.set_postfix(samples=len(origin_psmiles))
                    pbar.update(1)
                    done_batches = idx + 1
                    elapsed = time.time() - infer_start
                    eta = elapsed * (infer_steps - done_batches) / done_batches
                    print(
                        f"[Inference] batch {done_batches}/{infer_steps} done "
                        f"(batch_time={format_duration(time.time() - batch_start)}, "
                        f"samples={len(origin_psmiles)}, elapsed={format_duration(elapsed)}, "
                        f"eta={format_duration(eta)})",
                        flush=True,
                    )

    infer_seconds = time.time() - infer_start
    print(
        f"## Inference done ({infer_steps}/{len(data_loader)} batches, "
        f"samples={len(origin_psmiles)}, elapsed={format_duration(infer_seconds)})",
        flush=True,
    )

    validity, sim_score, inv_rmse, inv_r2, prop_rmse, prop_r2 = compute_inverse_metrics(predictor, origin_psmiles, recon_psmiles, property_values, pred_property_values, config)

    return origin_psmiles, recon_psmiles, validity, sim_score, inv_rmse, inv_r2, prop_rmse, prop_r2, infer_steps, infer_seconds


def save_reconstruction_outputs(config, origin_psmiles, recon_psmiles, prefix):
    data = dict(psmiles=origin_psmiles, recon_psmiles=recon_psmiles)
    df = pd.DataFrame(data, index=range(len(origin_psmiles)))
    if config.save:
        df.to_csv(os.path.join(config.exp_path, f"{prefix}_{config.epoch:04d}.csv"), index=False)

    sample_count = min(len(origin_psmiles), 5)
    if sample_count == 0:
        return

    for idx in list(np.random.choice(len(origin_psmiles), sample_count, replace=False)):
        print("\n[GT]    PSMILES:", origin_psmiles[idx])
        print("[Recon] PSMILES:", recon_psmiles[idx])


def inference_evaluation(models, infer_loader, config, prefix="Infer"):
    origin_psmiles, recon_psmiles, validity, sim_score, inv_rmse, inv_r2, prop_rmse, prop_r2, infer_batches, infer_seconds = inference(models, infer_loader, config)
    save_reconstruction_outputs(config, origin_psmiles, recon_psmiles, prefix)

    return dict(
        validity=validity,
        sim_score=sim_score,
        inv_rmse=inv_rmse,
        inv_r2=inv_r2,
        prop_rmse=prop_rmse,
        prop_r2=prop_r2,
        infer_batches=infer_batches,
        infer_seconds=infer_seconds,
        infer_samples=len(origin_psmiles),
    )


def evaluation(models, data_loader, infer_loader, config):
    AE, AE_ema, tokenizer, predictor = models

    AE.eval()
    AE_ema.eval()
    total_loss, ce_total, mse_total, eos_total, contrast_total = 0, 0, 0, 0, 0

    eval_start = time.time()
    eval_steps = get_limited_steps(data_loader, config.eval_steps)
    print(f"\n## Evaluation ({eval_steps}/{len(data_loader)} batches)")
    n_batches = 0
    with torch.no_grad():
        with AE_ema.ema_scope(AE):
            with tqdm(total=eval_steps, desc=f"Eval epoch {config.epoch}", dynamic_ncols=True) as pbar:
                for idx, batch in enumerate(data_loader):
                    if idx >= eval_steps:
                        break

                    psmiles, properties, input_ids, mask = batch[0], batch[1].cuda(), batch[2].long().cuda(), batch[3].cuda()

                    with autocast(device_type='cuda', dtype=config.amp_dtype):
                        logits, predict_prop, zf, zs = AE(properties, drop_rate=0.0, token_ids=input_ids)
                        loss, ce, mse, contrast, eos = AE.compute_loss_with_logits(input_ids=input_ids,
                                                                                    properties=properties,
                                                                                    pad_token_id=tokenizer.pad_token_id,
                                                                                    eos_token_id=tokenizer.eos_token_id,
                                                                                    logits=logits,
                                                                                    predict_prop=predict_prop,
                                                                                    zf=zf, zs=zs)
                    total_loss += loss.item()
                    ce_total += ce.item()
                    mse_total += mse.item()
                    eos_total += eos.item()
                    contrast_total += contrast.item()
                    n_batches += 1

                    pbar.set_postfix(loss=total_loss / n_batches)
                    pbar.update(1)

    if n_batches == 0:
        raise ValueError("Evaluation ran zero batches. Check --eval_steps and the eval dataset.")

    total_loss /= n_batches
    ce_total /= n_batches
    mse_total /= n_batches
    eos_total /= n_batches
    contrast_total /= n_batches

    eval_seconds = time.time() - eval_start
    origin_psmiles, recon_psmiles, validity, sim_score, inv_rmse, inv_r2, prop_rmse, prop_r2, infer_batches, infer_seconds = inference(models, infer_loader, config)

    save_reconstruction_outputs(config, origin_psmiles, recon_psmiles, "Eval")

    out_dict = dict(total_loss=total_loss, ce_loss=ce_total, mse_loss=mse_total, eos_loss=eos_total, contrast_loss=contrast_total,
                    validity=validity, sim_score=sim_score, inv_rmse=inv_rmse, inv_r2=inv_r2, prop_rmse=prop_rmse, prop_r2=prop_r2,
                    eval_batches=eval_steps, eval_seconds=eval_seconds, infer_batches=infer_batches,
                    infer_seconds=infer_seconds, infer_samples=len(origin_psmiles))
    return out_dict



def prop_evaluation(models, data_loader, infer_loader, config):
    AE, AE_ema, tokenizer, _ = models

    AE.eval()
    AE_ema.eval()
    total_loss, ce_total, mse_total, eos_total, contrast_total = 0, 0, 0, 0, 0
    prop_rmse, prop_r2 = 0.0, 0.0
    n_total = 0

    eval_steps = get_limited_steps(data_loader, config.eval_steps)
    print(f"\n## Evaluation Property ({eval_steps}/{len(data_loader)} batches)")
    with torch.no_grad():
        with AE_ema.ema_scope(AE):
            with tqdm(total=eval_steps, desc=f"Eval property epoch {config.epoch}", dynamic_ncols=True) as pbar:
                for idx, batch in enumerate(data_loader):
                    if idx >= eval_steps:
                        break

                    psmiles, properties, input_ids, mask = batch[0], batch[1].cuda(), batch[2].long().cuda(), batch[3].cuda()

                    with autocast(device_type='cuda', dtype=config.amp_dtype):
                        pred_properties = AE(properties, drop_rate=0.0, token_ids=input_ids, mode='infer_properties')
                        mse_loss = torch.mean(torch.sum((properties - pred_properties)**2, dim=-1), dim=0)

                    prop_rmse += compute_rmse(pred_properties, properties).item() * len(properties)
                    prop_r2 += compute_r2(pred_properties, properties).item() * len(properties)

                    mse_total += mse_loss.item() * len(properties)
                    n_total += len(properties)

                    pbar.set_postfix(samples=n_total)
                    pbar.update(1)

    if n_total == 0:
        raise ValueError("Property evaluation ran zero samples. Check --eval_steps and the eval dataset.")

    mse_total = mse_total / n_total
    prop_rmse = prop_rmse / n_total
    prop_r2 = prop_r2 / n_total


    out_dict = dict(total_loss=mse_total, prop_rmse=prop_rmse, prop_r2=prop_r2)
    return out_dict



def train_one_epoch(models, data_generator, optimizer, scaler, config):
    AE, _, tokenizer, predictor = models
    AE.train()
    total_loss, ce_total, mse_total, contrast_total, eos_total = 0, 0, 0, 0, 0
    grad_norm = 0

    global global_step
    pbar = tqdm(total=config.steps)

    for step, batch_idx in enumerate(range(config.steps)):
        batch = next(data_generator)

        properties = batch['properties'].cuda()
        input_ids = batch["input_ids"].long().cuda()
        mask = batch["mask"].bool().cuda()

        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=config.amp_dtype):
            logits, predict_prop, zf, zs = AE(properties, drop_rate=config.drop_rate, token_ids=input_ids)
            loss, ce, mse, contrast, eos = AE.compute_loss_with_logits(input_ids=input_ids,
                                                                       properties=properties,
                                                                       pad_token_id=tokenizer.pad_token_id,
                                                                       eos_token_id=tokenizer.eos_token_id,
                                                                       logits=logits,
                                                                       predict_prop=predict_prop,
                                                                       zf=zf, zs=zs)
            
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        grad_norm = grad_norm + compute_grad_norm(AE)

        if config.GC:
            torch.nn.utils.clip_grad_norm_(AE.parameters(), max_norm=1.0) 

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        ce_total += ce.item()
        mse_total += mse.item()
        eos_total += eos.item()
        contrast_total += contrast.item()

        txt = (
            f"Epoch: [{config.epoch:>4d}/{config.epochs:>4d}]  "
            f"Train Iter: {global_step + 1:3}/{config.steps * config.epochs:4}. lr: {optimizer.param_groups[0]['lr']:>.5f}. "
            f"loss: {total_loss / (step + 1):>.4f}. ce_loss: {ce_total / (step + 1):>.4f}. mse_loss: {mse_total / (step + 1):>.4f}. "
            f"contrast_loss: {contrast_total / (step + 1):>.4f}"
        )

        pbar.set_description(txt)
        pbar.update()

        global_step += 1

    total_loss = total_loss / config.steps
    ce_total = ce_total / config.steps
    mse_total = mse_total / config.steps
    eos_total = eos_total / config.steps
    contrast_total = contrast_total / config.steps
    grad_norm = grad_norm / config.steps

    out_dict = dict(total_loss=total_loss, ce_loss=ce_total, mse_loss=mse_total, eos_loss=eos_total, contrast_loss=contrast_total, grad_norm=grad_norm)

    return out_dict


def main():
    global config
    config = set_config(args)
    run_start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("==> Get model")
    AE, AE_ema, tokenizer, predictor = build_model(config)
    models = (AE, AE_ema, tokenizer, predictor)
    print("...DONE\n")

    params = sum(p.numel() for p in AE.parameters())
    print("Num. of params: {} (about {:.3f}B)".format(params, params/1000000000))

    print("==> Get dataset")
    train_loader, val_loader, test_loader, infer_loader = get_dataset(config)
    data_generator = get_data_generator(train_loader)
    print("...DONE\n")

    print("==> Init training settings")
    criterion, optimizer, scaler, wandb = init_train_setting(AE, config)
    print("...DONE\n")

    global global_step
    global_step = 0

    models, optimizer = restoring(models, optimizer, config, load_optimizer=not config.inference_only)

    eval_fn = prop_evaluation if config.property else evaluation
    if config.inference_only:
        if not (config.resume or (config.hf_checkpoint_repo_id and config.hf_checkpoint_filename)):
            raise ValueError("Set --resume or --hf_checkpoint_repo_id/--hf_checkpoint_filename for --inference_only True.")
        print("==> Start inference-only evaluation")
        out_dict_test = eval_fn(models, test_loader, infer_loader, config)
        logging_from_dict(prefix='Test', out_dict=out_dict_test, wandb=wandb, config=config)
        write_epoch_metrics(
            config,
            params,
            run_start_time,
            train_metrics=None,
            eval_metrics=out_dict_test,
            score=None,
            is_best=False,
            checkpoint_path=resolve_checkpoint_path(config),
            prefix="Eval",
        )
        if config.wandb:
            wandb.finish()
        return

    print("==> Start Training")
    print("=" * 100, "\n")

    state_dict = copy.deepcopy(AE.state_dict())
    max_score = -np.inf
    for epoch in range(config.start_epoch, config.epochs):
        config.epoch = epoch + 1
        is_best = False
        out_dict_metrics = None
        metrics_prefix = None
        score = None

        out_dict_train = train_one_epoch(models, data_generator, optimizer, scaler, config)
        logging_from_dict(prefix='Train', out_dict=out_dict_train, wandb=wandb, config=config)
        on_train_batch_end(AE, AE_ema)
        torch.cuda.empty_cache()

        if should_run_eval(config):
            out_dict_metrics = eval_fn(models, val_loader, infer_loader, config)
            metrics_prefix = "Eval"
            logging_from_dict(prefix='Valid', out_dict=out_dict_metrics, wandb=wandb, config=config)
        elif should_run_inference(config):
            out_dict_metrics = inference_evaluation(models, infer_loader, config, prefix="Infer")
            metrics_prefix = "Infer"
            logging_from_dict(prefix='Infer', out_dict=out_dict_metrics, wandb=wandb, config=config)

        if out_dict_metrics is not None:
            score = metric_score(out_dict_metrics, config)
            if  score > max_score:
                max_score = score
                state_dict = copy.deepcopy(AE.state_dict())
                is_best = True

        checkpoint_path = None
        if should_save_epoch_checkpoint(config) or is_best:
            checkpoint_path = os.path.join(config.save_path, f"Polyone_AE_{config.epoch:04d}.pt")
            save_checkpoint(
                config,
                checkpoint_payload(AE, AE_ema, optimizer, config),
                checkpoint_path,
                is_best
            )
            print(f"==> Model saved at epoch {config.epoch:04d}")
            if Path(checkpoint_path).exists():
                upload_checkpoint_to_hf(config, checkpoint_path)

        if out_dict_metrics is not None:
            write_epoch_metrics(
                config,
                params,
                run_start_time,
                train_metrics=out_dict_train,
                eval_metrics=out_dict_metrics,
                score=score,
                is_best=is_best,
                checkpoint_path=checkpoint_path,
                prefix=metrics_prefix,
            )

        torch.cuda.empty_cache()
        print("=" * 100, "\n")

    if config.run_final_test:
        AE.load_state_dict(state_dict)
        out_dict_test = eval_fn(models, test_loader, infer_loader, config)
        logging_from_dict(prefix='Test', out_dict=out_dict_test, wandb=wandb, config=config)
        write_epoch_metrics(
            config,
            params,
            run_start_time,
            train_metrics=None,
            eval_metrics=out_dict_test,
            score=max_score,
            is_best=False,
            checkpoint_path=Path(config.save_path, "Checkpoint_BEST.pt"),
            prefix="Test",
        )

    if config.wandb:
        wandb.finish()


if __name__ == '__main__':
    seed_everything(seed=args.seed)

    args.save_path = Path(args.prefix, args.exp_name)
    if args.save: args.save_path.mkdir(exist_ok=True, parents=True)

    args.exp_path = Path(args.save_path, 'eval')
    if args.exp_path: args.exp_path.mkdir(exist_ok=True, parents=True)

    main()
