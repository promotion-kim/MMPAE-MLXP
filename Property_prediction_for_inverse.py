import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
)

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

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger

from contextlib import contextmanager

import wandb
import yaml
from dotted_dict import DottedDict
from torch.amp import autocast, GradScaler

# from ldm.modules.losses import LPIPSWithDiscriminator
from libs.ldm.modules.ema import LitEma

from sklearn.preprocessing import StandardScaler


from utils import (
    init_wandb,
    seed_everything,
    logging_from_dict,
)



parser = argparse.ArgumentParser(
    description="Arguments for Property prediction"
)
parser.add_argument("--T", type=int, default=1000, help="timesteps for Unet model")
parser.add_argument("--droprate", type=float, default=0.1, help="dropout rate for model")
parser.add_argument("--dtype", default=torch.float32)
parser.add_argument("--workers", default=16, type=int)

parser.add_argument("--epochs", type=int, default=100, help="total epochs for training")
parser.add_argument("--start_epoch", type=int, default=0, help="start epochs for training")
parser.add_argument("--interval", type=int, default=5, help="epochs interval for evaluation")
parser.add_argument("--steps", type=int, default=1024, help="train steps per epoch")

parser.add_argument("--batch_size", type=int, default=128, help="batch size per device for training Unet model",)
parser.add_argument("--eval_batch_size", type=int, default=1024, help="batch size per device for training Unet model",)
parser.add_argument("--data_size", type=str, default='base', choices=['base', 'middle', 'large'])

parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--wd", type=float, default=1e-4, help="weight decay degree")
parser.add_argument("--drop_rate", type=float, default=0.1, help="drop rate for DownstreamRegressionModel",)

parser.add_argument("--prefix", type=str, default="path for saving result")
parser.add_argument('--save', default=True, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--resume', default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--exp_name', default=None, type=str, required=False)

# AE related params
parser.add_argument("--num_properties", default=29, type=int)
parser.add_argument("--layer", default='multi', type=str)
parser.add_argument("--polybert", default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument("--pretrain", default=True, type=lambda s: s in ["True", "true", 1])

parser.add_argument('--L2', default=True, type=lambda s: s in ["True", "true", 1])
parser.add_argument("--model_size", default='large', type=str, choices=['small', 'base', 'large', 'huge'])
parser.add_argument('--fullrep', default=False, type=lambda s: s in ["True", "true", 1])

parser.add_argument('--AR', default=True, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--batch', default=True, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--ema', default=True, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--beta_update', default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--sim', default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--lp', default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--validity', default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--config_name', default='Inverse_CwA', type=str)
parser.add_argument('--target_folder', nargs='+', type=str, required=True)

parser.add_argument("--wandb", default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument("--debug", default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument("--rdkit_logging", default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument("--seed", default=1004, type=int)


parser.add_argument("--alpha", type=float, default=0.1, help="coefficient of CwA loss")
parser.add_argument("--beta", type=float, default=0.1, help="coefficient of CwA loss")
parser.add_argument("--gamma", type=float, default=0.1, help="coefficient of CwA loss")
parser.add_argument("--temperature", type=float, default=0.05, help="temperature of CwA loss")
parser.add_argument('--loss_type', type=str, default="None", help="[None, CwA, CwAsym] (both lower and upper cases are handled).")

args = parser.parse_args()
config = None

RDLogger.DisableLog('rdApp.*')
os.environ['TOKENIZERS_PARALLELISM'] = 'False'


amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

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
        self.psmiles = []
        self.gen_psmiles = []
        self.properties = []

    def __len__(self):
        return len(self.properties)

    def __getitem__(self, idx):
        psmiles = self.psmiles[idx]
        gen_psmiles = self.gen_psmiles[idx]
        properties = self.properties[idx]

        return psmiles, gen_psmiles, torch.tensor(properties)
    

class DummyDataset(Dataset):
    def __init__(self):
        self.psmiles = []
        self.gen_smiles = []
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

        return psmiles, properties, token_idx, mask.astype(np.bool_)
    
    def get_prop(self, psmiles):
        idx = self.psmiles.index(psmiles)
        properties = np.array(self.properties[idx])
        properties = torch.tensor(properties)

        return idx, properties
    
    def get_token_from_idx(self, idx):
        token_idx = self.token_indices[idx]

        return token_idx
        
    
def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))

def compute_r2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    return 1 - ss_res / ss_tot


def set_config(args):
    fname = f'path of config file.   For example: ./configs/Inverse_CwA.yaml'
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

    return config


def load_parquet_files_with_dataset(file_list, drop_cols=None):
    dataset = DummyDataset()

    for f in tqdm(file_list):
        df = pd.read_parquet(f, engine='pyarrow')
        if drop_cols:
            df = df.drop(columns=drop_cols, errors='ignore')

        dataset.psmiles.extend(list(df['smiles'].values))
        dataset.properties.extend(list(np.stack(df['properties'].values)[:, :config.num_properties]))
        dataset.token_indices.extend(list(df['token_ids'].values))
        dataset.masks.extend(list(df['mask'].values))

        del df

    return dataset


def load_eval_dataset(file_name, drop_cols=None):
    dataset = EvalDataset()
    df = pd.read_parquet(file_name, engine='pyarrow')
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')

    dataset.psmiles.extend(list(df['origin_psmiles'].values))
    dataset.gen_psmiles.extend(list(df['gen_psmiles'].values))
    dataset.properties.extend(list(np.stack(df['properties'].values)))

    del df

    return dataset


def init_train_setting(model, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)

    criterion = nn.MSELoss()
    scaler = GradScaler()
    wandb = init_wandb(args)
    config.generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    return criterion, optimizer, scaler, wandb


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


def get_dataset(file_path):
    print("Load evaluation data from:", file_path)
    eval_dset = load_eval_dataset(file_path, drop_cols=["index", 'level_0'])
    eval_loader = DataLoader(eval_dset, batch_size=config.eval_batch_size, shuffle=False, num_workers=config.workers)

    return eval_dset, eval_loader


def build_model(config):
    model_config = config.model

    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_path)
    params = model_config.params
    params.loss_type = config.loss_type
    params.L2 = config.L2
    
    from models.MMTransformer_HMoE import MMTransformerAR

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
        dec_layers=8,
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
    ).to(device)

    model_path = 'path of Property Transformer checkpoint'

    ddict = torch.load(model_path, weights_only=False)
    model.load_state_dict(ddict['AE'], strict=False)

    return model, tokenizer


def tokenizer_encode(tokenizer, psmiles_list):
    encoding = tokenizer(
        psmiles_list,
        padding="max_length",
        max_length=160,
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].cuda()
    attention_mask = encoding["attention_mask"].cuda()

    return input_ids, attention_mask


def evaluation_AR_Batch(models, eval_loader, eval_df, criterion):
    model, tokenizer = models

    model.eval()

    total_loss, gen_rmse, gen_r2, gt_rmse, gt_r2 = 0., 0., 0., 0., 0.
    sim_scores, exact_match = 0., 0.
    n_sample = 0
    valid_sample = 0

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    pbar = tqdm(total=len(eval_loader))
    print("Total Iter (batches):", len(eval_loader))
    
    with torch.no_grad():
        for idx, batch in enumerate(eval_loader):
            # batch: (gt_psmiles, gen_psmiles, properties)
            gt_psmiles_list, gen_psmiles_list, properties = batch
            properties = properties.cuda()

            batch_size = len(gt_psmiles_list)
            n_sample += batch_size

            batch_valid_mask = []
            sim_score_batch = []
            exact_match_batch = []

            # SMILES validity & similarity 
            for gt_psmiles, gen_psmiles in zip(gt_psmiles_list, gen_psmiles_list):
                try:
                    mol_gt = Chem.MolFromSmiles(gt_psmiles)
                    mol_gen = Chem.MolFromSmiles(gen_psmiles)
                    if mol_gt is None or mol_gen is None:
                        batch_valid_mask.append(False)
                        sim_score_batch.append(0.0)
                        exact_match_batch.append(0.0)
                        continue

                    fp_gt = generator.GetFingerprint(mol_gt)
                    fp_gen = generator.GetFingerprint(mol_gen)
                    sim = DataStructs.TanimotoSimilarity(fp_gt, fp_gen)

                    batch_valid_mask.append(True)
                    sim_score_batch.append(sim)
                    exact_match_batch.append(1.0 if sim == 1.0 else 0.0)
                except:
                    batch_valid_mask.append(False)
                    sim_score_batch.append(0.0)
                    exact_match_batch.append(0.0)

            valid_idx = [i for i, valid in enumerate(batch_valid_mask) if valid]
            if len(valid_idx) == 0:
                continue

            valid_sample += len(valid_idx)

            valid_gt = [gt_psmiles_list[i] for i in valid_idx]
            valid_gen = [gen_psmiles_list[i] for i in valid_idx]
            valid_properties = properties[valid_idx]

            with autocast(device_type='cuda', dtype=amp_dtype):
                valid_gt_tokens, gt_mask = tokenizer_encode(tokenizer, valid_gt)
                valid_gen_tokens, gen_mask = tokenizer_encode(tokenizer, valid_gen)

                # forward
                gt_pred = model(token_ids=valid_gt_tokens, drop_rate=0.0, mode='infer_properties')
                gen_pred = model(token_ids=valid_gen_tokens, drop_rate=0.0, mode='infer_properties')

                loss = criterion(gen_pred, valid_properties)

            total_loss = total_loss + loss.item()
            gen_rmse = gen_rmse + compute_rmse(gen_pred, valid_properties).item() * len(valid_idx)
            gen_r2 = gen_r2 + compute_r2(gen_pred, valid_properties).item() * len(valid_idx)

            gt_rmse = gt_rmse + compute_rmse(gt_pred, valid_properties).item() * len(valid_idx)
            gt_r2 = gt_r2 + compute_r2(gt_pred, valid_properties).item() * len(valid_idx)


            sim_scores = sim_scores + sum([sim_score_batch[i] for i in valid_idx])
            exact_match = exact_match + sum([exact_match_batch[i] for i in valid_idx])

            txt = (
                f"loss: {total_loss / (idx+1):>.4f}.  validity: {(valid_sample / n_sample):>.4f}.  "
                f"sim_score: {(sim_scores / valid_sample):>.4f}.  exact_match: {(exact_match / valid_sample):>.4f}.  "
                f"gen_rmse: {gen_rmse / valid_sample:>.4f}.  gen_r2: {gen_r2 / valid_sample:>.4f}.  "
                f"gt_rmse: {gt_rmse / valid_sample:>.4f}.  gt_r2: {gt_r2 / valid_sample:>.4f}.  "
            )
            pbar.set_description(txt)
            pbar.update()



    total_loss = total_loss / (idx+1)
    sim_scores = sim_scores / valid_sample
    validity = valid_sample / n_sample

    exact_match = exact_match / valid_sample
    gen_rmse = gen_rmse / valid_sample
    gen_r2 = gen_r2 / valid_sample
    gt_rmse = gt_rmse / valid_sample
    gt_r2 = gt_r2 /valid_sample

    out_dict = dict(
        total_loss=total_loss,
        validity=validity,
        exact_match=exact_match,
        sim_score=sim_scores,
        gen_rmse=gen_rmse,
        gen_r2=gen_r2,
        gt_rmse=gt_rmse,
        gt_r2=gt_r2
    )
    return out_dict


def main():
    global config
    config = set_config(args)

    print("==> Get model")
    model, tokenizer = build_model(config=config)
    models = (model, tokenizer)
    print("...DONE\n")

    print("==> Init training settings")
    criterion, optimizer, scaler, wandb = init_train_setting(model, config)
    config.epoch = None
    print("...DONE\n")

    eval_root = 'path for saving evaluation result'

    target_folder = args.target_folder
    for folder in target_folder:

        inference_folder = os.path.join(inference_root, folder)
        assert os.path.exists(inference_folder), f'please select --target_folder option in {os.listdir(inference_root)}'

        eval_fn = evaluation_AR_Batch
        
        files = [x for x in os.listdir(inference_folder) if 'parquet' in x]
        files.sort()

        eval_folder = os.path.join(eval_root, 'Property_' + folder)
        Path(eval_folder).mkdir(parents=True, exist_ok=True)

        print("==> Start Evaluating")
        print("=" * 100, "\n")
        print(f"==> using {amp_dtype} for compute")


        for file_name in files:
            fname = '.'.join(file_name.split('.')[:-1])
            txt_name = f'[Eval result]_{fname}.txt'

            print(f"==> Get dataset from:{file_name}")
            file_path = os.path.join(inference_folder, file_name)
            eval_df, eval_loader = get_dataset(file_path)
            print("...DONE\n")

            out_dict_val = eval_fn(models, eval_loader, eval_df, criterion)
            logging_from_dict(prefix='Evaluation', out_dict=out_dict_val, wandb=False, config=config)
            
            with open(os.path.join(eval_folder, txt_name), 'w+') as f:
                f.write(str(out_dict_val))
            f.close()
            print(f"Eval result saved at.. {os.path.join(eval_folder, txt_name)}")
            print("=" * 100, "\n\n")

    torch.cuda.empty_cache()
    os._exit(0)
    return



if __name__ == '__main__':
    seed_everything(seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inference_root = 'path of inverse design inferenced result (Generated PSMILES with properties)'

    main()