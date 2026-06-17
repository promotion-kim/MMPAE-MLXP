import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from models.MMTransformer_HMoE import MMTransformerAR

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

from contextlib import contextmanager, nullcontext

import yaml
from dotted_dict import DottedDict
from torch.amp import autocast, GradScaler

from libs.ldm.modules.ema import LitEma

from utils import (
    init_wandb,
    Standardize,
    save_checkpoint,
    seed_everything,
    logging_from_dict,
    sigmoid_beta_annealing,
    compute_similarity, #tanimoto_similarity,
    decode_with_eos
)

import subprocess
import pickle

parser = argparse.ArgumentParser(
    description="Arguments for Evaluating MMPAE on inverse design task"
)
parser.add_argument("--T", type=int, default=1000, help="timesteps for Unet model")
parser.add_argument("--droprate", type=float, default=0.1, help="dropout rate for model")
parser.add_argument("--dtype", default=torch.float32)
parser.add_argument("--workers", default=8, type=int)

parser.add_argument("--epochs", type=int, default=100, help="total epochs for training")
parser.add_argument("--start_epoch", type=int, default=0, help="start epochs for training")
parser.add_argument("--interval", type=int, default=100, help="epochs interval for evaluation")
parser.add_argument("--steps", type=int, default=1024, help="train steps per epoch")

parser.add_argument("--batch_size", type=int, default=512, help="batch size per device for training Unet model",)
parser.add_argument("--eval_batch_size", type=int, default=512, help="batch size per device for training Unet model",)
parser.add_argument("--data_size", type=str, default='base', choices=['base', 'middle', 'large'])

parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--wd", type=float, default=1e-4, help="weight decay degree")
parser.add_argument("--drop_rate", type=float, default=0.1, help="drop rate for DownstreamRegressionModel",)

parser.add_argument("--prefix", type=str, default="/root path for saving experiments")
parser.add_argument('--save', default=True, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--resume', default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--exp_name', default='temp', type=str, required=False)

# AE related params
parser.add_argument("--pretrain", default=True, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--ema', default=True, type=lambda s: s in ["True", "true", 1])

parser.add_argument('--AR', default=True, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--L2', default=True, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--fullrep', default=False, type=lambda s: s in ["True", "true", 1])

parser.add_argument('--num_properties', default=29, type=int)
parser.add_argument('--n_samples', default=10, type=int)
parser.add_argument('--dec_layers', default=12, type=int)

parser.add_argument('--beta_update', default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--sim', default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--lp', default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--validity', default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--config_name', default='Inverse_CwA', type=str)

parser.add_argument("--wandb", default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument("--debug", default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument("--seed", default=1004, type=int)

parser.add_argument("--model_size", default='base', type=str, choices=['small', 'base', 'large', 'huge'])
parser.add_argument("--alpha", type=float, default=100, help="coefficient of MSE loss")
parser.add_argument("--beta", type=float, default=1000, help="coefficient of CwA loss")
parser.add_argument("--gamma", type=float, default=0.1, help="coefficient of EOS loss")
parser.add_argument("--temperature", type=float, default=0.05, help="temperature of CwA loss")
parser.add_argument('--property', default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--loss_type', type=str, default="None", help="[None, CwA, CwAsym] (both lower and upper cases are handled).")

args = parser.parse_args()
config = None


def set_config(args):
    fname = f'./configs/Inverse_CwA_large.yaml'
    with open(fname, 'r') as y_file:
        yaml_file = yaml.load(y_file, Loader=yaml.FullLoader)
        config = DottedDict(dict(yaml_file))

    keys = list(args.__dict__.keys())
    values = list(args.__dict__.values())
    [setattr(config, keys[i], values[i]) for i in range(len(keys))]

    config.model.params.num_properties = config.num_properties
    config.model.AR = config.AR
    config.model.params.beta = config.beta
    config.model.params.temperature = config.temperature

    config.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    return config

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
        properties = np.array(self.properties[idx])
        token_idx = self.token_indices[idx]
        mask = self.masks[idx]

        return psmiles, properties, token_idx.astype(np.int64), mask.astype(np.bool_)
    

def load_parquet_files_with_dataset(file_list, drop_cols=None):
    dataset = DummyDataset()

    for f in tqdm(file_list):
        df = pd.read_parquet(f, engine='pyarrow')
        if drop_cols:
            df = df.drop(columns=drop_cols, errors='ignore')

        dataset.psmiles.extend(list(df['smiles'].values))

        dataset.properties.extend(list(np.stack(df['properties'].values, axis=0)[:, :29]))
        dataset.token_indices.extend(list(df['token_ids'].values))
        dataset.masks.extend(list(df['mask'].values))
        del df

    return dataset

def get_dataset(config):
    data_path = "data root path"

    train_files = ['polyOne_ar.parquet', 'polyOne_az.parquet', 'polyOne_bg.parquet', 'polyOne_bo.parquet', 'polyOne_bq.parquet', 'polyOne_em.parquet', 'polyOne_fx.parquet', 'polyOne_gk.parquet', 'polyOne_hk.parquet', 'polyOne_ho.parquet']
    val_files = ['polyOne_hu.parquet', 'polyOne_hv.parquet']
    test_files = ['polyOne_hw.parquet', 'polyOne_hx.parquet']

    train_files = [os.path.join(data_path, f) for f in train_files]
    val_files = [os.path.join(data_path, f) for f in val_files]
    test_files = [os.path.join(data_path, f) for f in test_files]

    train_dataset = load_parquet_files_with_dataset(train_files, drop_cols=["index", 'level_0'])
    val_dataset = load_parquet_files_with_dataset(val_files, drop_cols=["index", 'level_0'])
    test_dataset = load_parquet_files_with_dataset(test_files, drop_cols=["index", 'level_0'])

    _, _, test_prop = Standardize(train_dataset.properties, val_dataset.properties, test_dataset.properties)
    test_dataset.properties = test_prop

    eval_loader = DataLoader(test_dataset, config.eval_batch_size, shuffle=False, num_workers=config.workers)

    return eval_loader


def get_data_generator(loader):
    while True:
        try:
            psmiles, properties, token_idx, mask = next(data_iter)
        except:
            data_iter = iter(loader)
            psmiles, properties, token_idx, mask  = next(data_iter)
        finally:
            psmiles, properties, token_idx, mask  = psmiles, properties.cuda(), token_idx.long().cuda(), mask.bool().cuda()

        batch = {"psmiles": psmiles, "properties": properties, "input_ids": token_idx, "mask": mask}
        yield batch


def init_train_setting(AE, config):
    optimizer = torch.optim.AdamW(AE.parameters(), lr=config.lr, weight_decay=config.wd)

    criterion = None
    scaler = GradScaler()
    wandb = init_wandb(config, project_id='project id', run_prefix='run prefix')

    config.rdFingerprintGen = None

    return criterion, optimizer, scaler, wandb


def build_model():
    model_config = config.model

    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_path)
    params = model_config.params
    params.loss_type = config.loss_type
    params.L2 = config.L2
    
    print('Model params:')
    for key in params.keys():
        print(key, ':', params[key])
    print()


    model_path = 'Model checkpoint path for loading pretrained MMPAE model'
    config.csv_name = 'CSV namd for saving inference result'
    assert params.L2 is True and config.dec_layers == 12

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
        loss_type=config.loss_type
    ).cuda()

    print("==> Load pre-trained AR model from:", model_path)
    ddict = torch.load(model_path, weights_only=False)
    print("==> State dict epoch:", ddict['epoch'])
    model.load_state_dict(ddict['AE'], strict=False)

    return model, tokenizer


def inference(models, data_loader, drop_rate):
    AE, tokenizer = models

    AE.eval()

    ddict = {'origin_psmiles': None, 'gen_psmiles': None, 'properties': None}
    origin_psmiles, recon_psmiles = [], []
    property_values = []

    print(f"\n## Inference with {drop_rate} drop rate")
    print("### Total Inference iter:", len(data_loader))
    print()

    INTERVAL = (len(data_loader) // 10)

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader)):
            psmiles, properties, input_ids, mask = batch[0], batch[1].cuda(), batch[2].long().cuda(), batch[3].cuda()

            with autocast(device_type='cuda', dtype=config.amp_dtype):
                pred_tokens = AE(properties, drop_rate=drop_rate, mode='infer_psmiles')
                pred_tokens = [torch.tensor(seq, dtype=torch.long, device='cuda') for seq in pred_tokens]
            
                recon = decode_with_eos(pred_tokens, tokenizer, tokenizer.eos_token_id, tokenizer.pad_token_id)

            origin_psmiles.extend(list(psmiles))
            recon_psmiles.extend(recon)
            property_values.extend(properties.detach().cpu().numpy())

            if idx % INTERVAL == 0:
                ddict['origin_psmiles'] = origin_psmiles
                ddict['gen_psmiles'] = recon_psmiles
                ddict['properties'] = property_values
                df = pd.DataFrame(data=ddict, index=range(len(origin_psmiles)))
                df.to_parquet(f'{config.infer_root}/{config.csv_name}/{config.csv_name}_drop{drop_rate}.parquet', index=False)

        ddict['origin_psmiles'] = origin_psmiles
        ddict['gen_psmiles'] = recon_psmiles
        ddict['properties'] = property_values
        df = pd.DataFrame(data=ddict, index=range(len(origin_psmiles)))
        df.to_parquet(f'{config.infer_root}/{config.csv_name}/{config.csv_name}_drop{drop_rate}.parquet', index=False)


    out_dict = {"len_psmiles": len(origin_psmiles)}
    return out_dict



def main():
    global config
    config = set_config(args)

    print("==> Get model")
    AE, tokenizer = build_model()
    models = (AE, tokenizer)
    print("...DONE\n")

    print("==> Get dataset")
    test_loader = get_dataset()
    print("...DONE\n")

    print("==> Init training settings")
    # criterion, optimizer, scaler, wandb = init_train_setting(AE, config)
    print("...DONE\n")

    global global_step
    global_step = 0
    
    print("==> Start Training")
    print("=" * 100, "\n")

    prefix = 'Inference'
    config.infer_root = './inference_ablation_result'
    config.infer_root = './inference_result'
    Path(config.infer_root, config.csv_name).mkdir(exist_ok=True, parents=True)

    drops = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    for drop in drops:
        out_dict_val = inference(models, test_loader, drop)
        logging_from_dict(prefix=prefix, out_dict=out_dict_val, wandb=False, config=config)

    return


if __name__ == '__main__':
    seed_everything(seed=args.seed)

    args.save_path = Path(args.prefix, 'folder path to save')
    if args.save: args.save_path.mkdir(exist_ok=True, parents=True)

    args.exp_path = Path(args.save_path, 'experiment name')
    if args.exp_path: args.exp_path.mkdir(exist_ok=True, parents=True)

    main()
