import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoTokenizer

from models.MMTransformer import MMTransformerAR

import os

import math
import shutil
import argparse
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

parser = argparse.ArgumentParser(
    description="Arguments for Train HMMPAE"
)
parser.add_argument("--epochs", type=int, default=1000, help="total epochs for training")
parser.add_argument("--start_epoch", type=int, default=0, help="start epochs for training")
parser.add_argument("--interval", type=int, default=10, help="epochs interval for evaluation")
parser.add_argument("--steps", type=int, default=1024, help="train steps per epoch")
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
parser.add_argument('--resume', default=False, type=lambda s: s in ["True", "true", 1])
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
    fname = f'./configs/Inverse_CwA_large.yaml'
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


def restoring(models, optimizer, config):
    load_path = config.resume

    if load_path:
        assert Path(load_path).exists()

        print("\n==> Restoring state dict from {}".format(load_path))
        AE, AE_ema, tokenizer, _ = models
        load_dict = torch.load(load_path, weights_only=False)
        AE.load_state_dict(load_dict['AE'])

        ema_dict = load_dict['AE_ema']

        if 'MP' in config.resume:
            for k in list(ema_dict.keys()):
                if k.startswith('module'):
                    ema_dict[k[len('module'):]] = ema_dict[k]
                    del ema_dict[k]

        AE_ema.load_state_dict(ema_dict)

        optimizer.load_state_dict(load_dict['optimizer'])

        global global_step
        global_step = load_dict['global_step']
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

    print("CONFIG.property: True")
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
        if len(valid_idx) == 0:
            continue

        valid_sample += len(valid_idx)

        valid_gen = [recon_p[i] for i in valid_idx]
        valid_properties = properties[valid_idx]

        gen_pred = predictor.forward_with_psmiles_batch(valid_gen).squeeze()
        inv_rmse += compute_rmse(gen_pred, valid_properties).item() * len(valid_idx)
        inv_r2 += compute_r2(gen_pred, valid_properties).item() * len(valid_idx)


    if not config.inverse:
        prop_rmse = prop_rmse / prop_sample
        prop_r2 = prop_r2 / prop_sample

    inv_rmse = inv_rmse / valid_sample if valid_sample != 0 else 0
    inv_r2 = inv_r2 / valid_sample if valid_sample != 0 else 0

    validity = valid_sample / len(recon_psmiles)
    sim_score = np.mean(sim_scores) if len(sim_scores) > 0 else 0.0

    return validity, sim_score, inv_rmse, inv_r2, prop_rmse, prop_r2


def inference(models, data_loader, config):
    AE, AE_ema, tokenizer, predictor = models

    AE.eval()
    AE_ema.eval()

    origin_psmiles, recon_psmiles = [], []
    property_values = []
    pred_property_values = []

    print("\n## Inference")
    pbar = tqdm(total=config.n_samples)
    with torch.no_grad():
        with AE_ema.ema_scope(AE):
            for idx, batch in enumerate(data_loader):
                psmiles, properties, input_ids, mask = batch[0], batch[1].cuda(), batch[2].long().cuda(), batch[3].cuda()

                with autocast(device_type='cuda', dtype=config.amp_dtype):
                    pred_tokens = AE(properties, drop_rate=0.0, mode='infer_psmiles')
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

                pbar.update()

    validity, sim_score, inv_rmse, inv_r2, prop_rmse, prop_r2 = compute_inverse_metrics(predictor, origin_psmiles, recon_psmiles, property_values, pred_property_values, config)

    return origin_psmiles, recon_psmiles, validity, sim_score, inv_rmse, inv_r2, prop_rmse, prop_r2


def evaluation(models, data_loader, infer_loader, config):
    AE, AE_ema, tokenizer, predictor = models

    AE.eval()
    AE_ema.eval()
    total_loss, ce_total, mse_total, eos_total, contrast_total = 0, 0, 0, 0, 0

    print("\n## Evaluation")
    with torch.no_grad():
        with AE_ema.ema_scope(AE):
            for idx, batch in enumerate(tqdm(data_loader)):
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


    total_loss /= (idx+1)
    ce_total /= (idx+1)
    mse_total /= (idx+1)
    eos_total /= (idx+1)
    contrast_total /= (idx+1)

    origin_psmiles, recon_psmiles, validity, sim_score, inv_rmse, inv_r2, prop_rmse, prop_r2 = inference(models, infer_loader, config)

    data = dict(psmiles=origin_psmiles, recon_psmiles=recon_psmiles)
    df = pd.DataFrame(data, index=range(len(origin_psmiles)))
    if config.save: df.to_csv(os.path.join(config.exp_path, f"Eval_{config.epoch:04d}.csv"), index=False)

    for idx in list(np.random.choice(len(origin_psmiles), 5, replace=False)):
        print("\n[GT]    PSMILES:", origin_psmiles[idx])
        print("[Recon] PSMILES:", recon_psmiles[idx])

    out_dict = dict(total_loss=total_loss, ce_loss=ce_total, mse_loss=mse_total, eos_loss=eos_total, contrast_loss=contrast_total,
                    validity=validity, sim_score=sim_score, inv_rmse=inv_rmse, inv_r2=inv_r2, prop_rmse=prop_rmse, prop_r2=prop_r2)
    return out_dict



def prop_evaluation(models, data_loader, infer_loader, config):
    AE, AE_ema, tokenizer, _ = models

    AE.eval()
    AE_ema.eval()
    total_loss, ce_total, mse_total, eos_total, contrast_total = 0, 0, 0, 0, 0
    prop_rmse, prop_r2 = 0.0, 0.0
    n_total = 0

    print("\n## Evaluation Property")
    with torch.no_grad():
        with AE_ema.ema_scope(AE):
            for idx, batch in enumerate(tqdm(data_loader)):
                psmiles, properties, input_ids, mask = batch[0], batch[1].cuda(), batch[2].long().cuda(), batch[3].cuda()

                with autocast(device_type='cuda', dtype=config.amp_dtype):
                    pred_properties = AE(properties, drop_rate=0.0, token_ids=input_ids, mode='infer_properties')
                    mse_loss = torch.mean(torch.sum((properties - pred_properties)**2, dim=-1), dim=0)
                    
                prop_rmse += compute_rmse(pred_properties, properties).item() * len(properties)
                prop_r2 += compute_r2(pred_properties, properties).item() * len(properties)

                mse_total += mse_loss.item() * len(properties)
                n_total += len(properties)

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

    # config.resume = 'resume_path'
    models, optimizer = restoring(models, optimizer, config)

    global global_step
    global_step = 0

    print("==> Start Training")
    print("=" * 100, "\n")


    eval_fn = prop_evaluation if config.property else evaluation

    state_dict = AE.state_dict()
    max_score = -np.inf
    for epoch in range(config.start_epoch, config.epochs):
        config.epoch = epoch + 1
        is_best = False

        out_dict_train = train_one_epoch(models, data_generator, optimizer, scaler, config)
        logging_from_dict(prefix='Train', out_dict=out_dict_train, wandb=wandb, config=config)
        on_train_batch_end(AE, AE_ema)
        torch.cuda.empty_cache()

        if config.epoch % config.interval == 0 or config.epoch == config.epochs:
            out_dict_val = eval_fn(models, val_loader, infer_loader, config)
            logging_from_dict(prefix='Valid', out_dict=out_dict_val, wandb=wandb, config=config)

            score = out_dict_val['prop_r2'] - out_dict_val['prop_rmse'] if config.property else out_dict_val['inv_r2'] - out_dict_val['inv_rmse'] 
            if  score > max_score:
                max_score = score
                state_dict = AE.state_dict()
                is_best = True

            save_checkpoint(
                config,
                {
                    "epoch": config.epoch,
                    "global_step": global_step,
                    "AE": AE.state_dict(),
                    "AE_ema": AE_ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, os.path.join(config.save_path, f"Polyone_AE_{config.epoch:04d}.pt"), is_best
            )
            print(f"==> Model saved at epoch {config.epoch:04d}")

        torch.cuda.empty_cache()
        print("=" * 100, "\n")

    AE.load_state_dict(state_dict)
    out_dict_test = eval_fn(models, test_loader, infer_loader, config)
    logging_from_dict(prefix='Test', out_dict=out_dict_test, wandb=wandb, config=config)

    if config.wandb:
        wandb.finish()


if __name__ == '__main__':
    seed_everything(seed=args.seed)

    args.save_path = Path(args.prefix, args.exp_name)
    if args.save: args.save_path.mkdir(exist_ok=True, parents=True)

    args.exp_path = Path(args.save_path, 'eval')
    if args.exp_path: args.exp_path.mkdir(exist_ok=True, parents=True)

    main()
