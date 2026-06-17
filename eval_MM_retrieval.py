import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

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

from contextlib import contextmanager, nullcontext

import yaml
from dotted_dict import DottedDict
from torch.amp import autocast, GradScaler

from libs.ldm.modules.ema import LitEma

import pickle

import subprocess

import os
import argparse
import random
from pathlib import Path
from glob import glob
import warnings
warnings.filterwarnings(action='ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical
from tqdm import tqdm
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

from transformers import AutoTokenizer, WordpieceTokenizer

from dotted_dict import DottedDict
import pickle
import yaml


from utils import (
    Standardize,
)


from utils import (
    decode_with_eos,
    compute_rmse, compute_r2,
)



parser = argparse.ArgumentParser(
    description="Arguments for Evaluating MMPAE on cross-modal retrieval task"
)
parser.add_argument("--T", type=int, default=1000, help="timesteps for Unet model")
parser.add_argument("--droprate", type=float, default=0.1, help="dropout rate for model")
parser.add_argument("--dtype", default=torch.float32)
parser.add_argument("--workers", default=0, type=int)

parser.add_argument("--epochs", type=int, default=100, help="total epochs for training")
parser.add_argument("--start_epoch", type=int, default=0, help="start epochs for training")
parser.add_argument("--interval", type=int, default=5, help="epochs interval for evaluation")
parser.add_argument("--steps", type=int, default=1024, help="train steps per epoch")

parser.add_argument("--batch_size", type=int, default=512, help="batch size per device for training Unet model",)
parser.add_argument("--eval_batch_size", type=int, default=1024, help="batch size per device for training Unet model",)
parser.add_argument("--data_size", type=str, default='base', choices=['base', 'middle', 'large'])

parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--wd", type=float, default=1e-4, help="weight decay degree")
parser.add_argument("--drop_rate", type=float, default=0.1, help="drop rate for DownstreamRegressionModel",)

parser.add_argument("--prefix", type=str, default="/root path for saving experiments")
parser.add_argument('--save', default=True, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--resume', default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--exp_name', default='temp', type=str, required=False)

# AE related params
parser.add_argument("--model_size", type=str, default='base', choices=['small', 'base', 'large', 'huge'])
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

parser.add_argument("--alpha", type=float, default=1, help="coefficient of CwA loss")
parser.add_argument("--beta", type=float, default=0.1, help="coefficient of CwA loss")
parser.add_argument("--gamma", type=float, default=0.0, help="coefficient of CwA loss")
parser.add_argument("--temperature", type=float, default=0.05, help="temperature of CwA loss")
parser.add_argument('--property', default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--inverse', default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--attn_pool', default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--rep_load', default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--loss_type', type=str, default="None", help="[None, CwA, CwAsym] (both lower and upper cases are handled).")


args = parser.parse_args()
config = None

amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    config.devic = device

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
    

class TokenDataset(Dataset):
    def __init__(self):
        self.token_features = []

    def __len__(self):
        return len(self.token_features)

    def __getitem__(self, idx):
        token_feat = self.token_features[idx]

        return torch.tensor(token_feat)
    

class PropDataset(Dataset):
    def __init__(self):
        self.prop_features = []

    def __len__(self):
        return len(self.prop_features)

    def __getitem__(self, idx):
        prop_feat = self.prop_features[idx]

        return torch.tensor(prop_feat)
    

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
    wandb = None

    config.rdFingerprintGen = None
    
    Path('./inference_result', config.csv_name).mkdir(exist_ok=True, parents=True)

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



@torch.no_grad()
def feature_extractor(models, data_loader, drop=0.0, token_features=None, prop_features=None):
    AE, AE_ema, tokenizer = models
    AE.eval()
    AE_ema.eval()

    token_zf= torch.FloatTensor()
    prop_zf = torch.FloatTensor()

    token_zs= torch.FloatTensor()
    prop_zs = torch.FloatTensor()

    MAX_ITER = len(data_loader)  

    cnt = AE_ema.ema_scope(AE) if config.ema else nullcontext()

    with torch.no_grad():
        with cnt:
            if token_features is None and prop_features is None:
                for idx, batch in enumerate(tqdm(data_loader)):
                    psmiles, properties, input_ids, mask = batch[0], batch[1].cuda(), batch[2].long().cuda(), batch[3].cuda()

                    with autocast(device_type='cuda', dtype=amp_dtype):
                        token_z = AE.encode_tokens(token_ids=input_ids, drop_rate=0.0)
                        prop_z = AE.encode_properties(properties=properties, drop_rate=0.0)

                    token_zf = torch.cat((token_zf, token_z.float().squeeze().detach().cpu()), dim=0)
                    prop_zf = torch.cat((prop_zf, prop_z.float().squeeze().detach().cpu()), dim=0)

                torch.save({'token_features': token_zf, 'prop_features': prop_zf}, args.save_path)

            else:
                token_zf = token_features[0]
                prop_zf = prop_features[0]


            if drop != 0.0: 
                for idx, batch in enumerate(tqdm(data_loader)):
                    psmiles, properties, input_ids, mask = batch[0], batch[1].cuda(), batch[2].long().cuda(), batch[3].cuda()

                    with autocast(device_type='cuda', dtype=amp_dtype):
                        token_z = AE.encode_tokens(token_ids=input_ids, drop_rate=drop)
                        prop_z = AE.encode_properties(properties=properties, drop_rate=drop)

                    token_zs = torch.cat((token_zs, token_z.float().squeeze().detach().cpu()), dim=0)
                    prop_zs = torch.cat((prop_zs, prop_z.float().squeeze().detach().cpu()), dim=0)

            else:
                token_zs, prop_zs = token_zf.clone(), prop_zf.clone()

    return token_zf, token_zs, prop_zf, prop_zs


@torch.no_grad()
def ranks_streaming(A: torch.Tensor, B: torch.Tensor, drop_rate=0.0, row_bs=256, col_bs=1024, device=None, tie_rule="min") -> torch.Tensor:
    assert A.size(1) == B.size(1), "Dim mismatch"
    if device is None: device = A.device

    A = A.to(device, dtype=amp_dtype)
    B = B.to(device, dtype=amp_dtype)

    N, D = A.shape
    M = B.size(0)
    B_t = B.t().contiguous()

    greater = torch.zeros(N, dtype=torch.long, device=device)
    equal   = torch.zeros(N, dtype=torch.long, device=device)

    eps = 1e-6
    pbar = tqdm(total=(N // row_bs + 1), desc="Ranks rows")

    top1_acc = 0.0
    top5_acc = 0.0
    n_samples = 0

    for r0 in range(0, N, row_bs):
        r1 = min(r0 + row_bs, N)
        q = A[r0:r1]
        gt_sim = (q * B[r0:r1]).sum(dim=1)

        for c0 in range(0, M, col_bs):
            c1 = min(c0 + col_bs, M)
            sb = q @ B_t[:, c0:c1]
            self_cols = torch.arange(r0, r1, device=device)
            in_block  = (self_cols >= c0) & (self_cols < c1)
            if in_block.any():
                loc = (self_cols[in_block] - c0)
                sb[in_block, loc] = float('-inf')

            greater[r0:r1] += (sb > gt_sim[:, None]).sum(dim=1)
            equal  [r0:r1] += (torch.abs(sb - gt_sim[:, None]) <= eps).sum(dim=1)


        ranks_blk = greater[r0:r1] + (equal[r0:r1] // 2)

        res_blk = recall_from_ranks(ranks_blk, ks=(1,5))
        blk = (r1 - r0)  

        top1_acc += res_blk['R@1'] * blk
        top5_acc += res_blk['R@5'] * blk
        n_samples += blk

        txt = (
            f"Drop: [{drop_rate}]  "
            f"R@1: {top1_acc / n_samples:>.4f}. R@5: {top5_acc / n_samples:>.4f}."
        )
        pbar.set_description(txt)
        pbar.update()

    if tie_rule == "min":
        ranks = greater
    elif tie_rule == "max":
        ranks = greater + equal
    else:
        ranks = greater + (equal // 2)

    return ranks


@torch.no_grad()
def recall_from_ranks(ranks: torch.Tensor, ks=(1,5,10)):
    return {f'R@{k}': float((ranks < k).float().mean().item()) for k in ks}


@torch.no_grad()
def Retrive(token_features, prop_features, drop_rate=0.0, ks=(1, 5, 10), both_directions=True, row_bs=512, col_bs=2048, use_exact_ranks=True):
    assert token_features[0].size(1) == prop_features[0].size(1), "Shape mismatch"

    token_zf, token_zs = token_features
    prop_zf, prop_zs = prop_features

    results = {}
    # A -> B
    ranks_AB = ranks_streaming(token_zs, prop_zf, drop_rate=drop_rate, row_bs=row_bs, col_bs=col_bs)
    results.update({f"[PSMILES to Property]  {k}": v for k, v in recall_from_ranks(ranks_AB, ks).items()})

    results[":=============="] = ""
    # B -> A
    ranks_BA = ranks_streaming(prop_zs, token_zf, drop_rate=drop_rate, row_bs=row_bs, col_bs=col_bs)
    results.update({f"[Property to PSMILES]  {k}": v for k, v in recall_from_ranks(ranks_BA, ks).items()})

    return results


def main():
    global config
    config = set_config(args)

    # reproducibility
    seed = random.randint(0, 1000)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    cudnn.benchmark = True

    print("==> Get model")
    AE, AE_ema, tokenizer = build_model()
    models = (AE, AE_ema, tokenizer)
    print("...DONE\n")

    params = sum(p.numel() for p in AE.parameters())
    print("Num. of params: {} (about {:.3f}B)".format(params, params/1000000000))

    print("==> Get dataset")
    eval_loader = get_dataset()
    print("...DONE\n")

    print("==> Init training settings")
    criterion, optimizer, scaler, wandb = init_train_setting(AE, config)
    print("...DONE\n")

    global global_step
    global_step = 0

    print("==> Start Training")
    print("=" * 100, "\n")

    save_root = f'path for saving extracted features'
    args.save_path = os.path.join(save_root, f'{config.csv_name}.pt')
    Path(save_root).mkdir(parents=True, exist_ok=True)

    eval_root = 'path for saving evaluation results'
    eval_folder = os.path.join(eval_root, config.csv_name)
    Path(eval_folder).mkdir(parents=True, exist_ok=True)

    drops = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    if Path(args.save_path).exists() and args.rep_load:
        print("==> Load pre-extracted features from:", args.save_path)
        ddict = torch.load(args.save_path, map_location='cpu')
        token_features = (ddict['token_features'], ddict['token_features'])
        prop_features = (ddict['prop_features'], ddict['prop_features'])
    else:
        token_features = None
        prop_features = None


    for drop in drops:
        print(f"==> [Drop rate {drop}] Extract PSMILES token and Property features")

        token_zf, token_zs, prop_zf, prop_zs = feature_extractor(models, eval_loader, drop, token_features, prop_features)
        token_features, prop_features = (token_zf, token_zs), (prop_zf, prop_zs)
        print("\nTotal Searching Space:", len(token_zf))

        print("\n==> Retrieve both A to B and B to A")
        out_dict_val = Retrive(token_features, prop_features, drop_rate=drop, ks=(1, 3, 5), both_directions=True, row_bs=512, col_bs=4096)


        for k, v in out_dict_val.items():
            if isinstance(v, str):
                print(f"{k}: {v}")
            else:
                print(f"{k}: {v:.4f}")

        out_path = os.path.join(eval_folder, f"Retrieval_Drop{drop}.txt")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(str(out_dict_val))

        print(f"Eval result saved at: {out_path}")
        print("=" * 100, "\n\n")

        if not Path(args.save_path).exists():
            torch.save({'token_features': token_zf, 'prop_features': prop_zf}, args.save_path)

    return


if __name__ == '__main__':
    args.save_path = Path(args.prefix, 'folder path to save')
    if args.save: args.save_path.mkdir(exist_ok=True, parents=True)

    args.exp_path = Path(args.save_path, 'experiment name')
    if args.exp_path: args.exp_path.mkdir(exist_ok=True, parents=True)

    main()
