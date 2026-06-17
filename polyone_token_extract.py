import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
from pathlib import Path
from multiprocessing import Pool, set_start_method
import os
import argparse
import math

from rdkit import Chem


# ───── argparse ──────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--pretrain', default=True, type=lambda s: s in ['True', 'true', 1])
parser.add_argument('--source', default='./data/polyone', help='directory containing raw polyOne parquet shards')
parser.add_argument('--dest', default='./data/polyone_tokenized', help='directory to write tokenized polyOne parquet shards')
parser.add_argument('--tokenizer_path', default='kuelumbus/polyBERT', help='Hugging Face id or local PolyBERT tokenizer path')
parser.add_argument('--overwrite', default=False, type=lambda s: s in ['True', 'true', 1], help='reprocess files that already exist')
args = parser.parse_args()



source = args.source
dest = args.dest
Path(dest).mkdir(parents=True, exist_ok=True)


prop_list = ['Tg', 'Tm', 'Td',
             'Cp', 'Eat', 'LOI', 'Xc', 'Xe', 'rho',
             'Egc', 'Egb', 'Eea', 'Ei', 'Eib', 'CED',
             'YM', 'TSy', 'TSb', 'epsb',
             'permO2', 'permCO2', 'permN2', 'permH2', 'permHe', 'permCH4',
             'nc', 'ne', 'epsc', 'epse_6.0', 'epse_1.78', 'epse_2.0', 'epse_3.0', 'epse_4.0', 'epse_5.0', 'epse_7.0', 'epse_9.0', 'epse_15.0']

# ───── Dataset clas ──────────────────────────────────────
class PolyDataset(Dataset):
    def __init__(self, df, tokenizer, max_token_len=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        content = list(self.df.loc[idx])
        # psmiles = content[2]
        psmiles = content[0]
        targets = np.array(content[1:])
        targets = torch.from_numpy(targets)
        return psmiles, targets


# ───── Mean Pooling func ──────────────────────────────────────
def mean_pooling(model_output, attention_mask):
    if isinstance(model_output, tuple):
        token_embeddings = model_output[0]
    else:
        token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# ───── subprocess ───────────────────────
def token_extraction(file_path):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    fname = os.path.basename(file_path)
    dest_path = os.path.join(dest, fname)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0 and not args.overwrite:
        return f"Skipped existing: {fname}"

    df = pd.read_parquet(file_path).reset_index(drop=True)
    missing_cols = [col for col in ["smiles", *prop_list] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{fname} is missing columns: {missing_cols}")

    psmiles = df["smiles"].astype(str).tolist()
    properties = df[prop_list].to_numpy(dtype=np.float32)

    token_indices = []
    masks = []
    total_batches = math.ceil(len(psmiles) / args.batch_size)

    for batch_idx, start in enumerate(range(0, len(psmiles), args.batch_size), start=1):
        if batch_idx == 1 or batch_idx % 25 == 0 or batch_idx == total_batches:
            print(f"{fname}: batch {batch_idx}/{total_batches} ({start}/{len(psmiles)} rows)", flush=True)
        batch_psmiles = psmiles[start:start + args.batch_size]
        encoded_batch = tokenizer(
            batch_psmiles,
            padding="max_length",
            max_length=160,
            truncation=True,
            return_tensors="np",
        )
        input_ids = encoded_batch["input_ids"].astype(np.int16)
        attention_mask = encoded_batch["attention_mask"].astype(np.int16)

        token_indices.extend([row for row in input_ids])
        masks.extend(attention_mask.sum(axis=1).astype(np.int16).tolist())


    new_len = len(token_indices)
    df = {'smiles': psmiles,
          'token_ids': token_indices,
          'mask': masks,
          'properties': [row for row in properties],
         }
    df = pd.DataFrame.from_dict(data=df, orient="columns")
    df.to_parquet(dest_path, index=False)

    return f"Processed: {fname} ({new_len} valid sequences)"


# ───── Main  ──────────────────────────────────────
def main():
    files = sorted(glob(os.path.join(source, '*.parquet')))

    if args.workers == 1:
        for file_path in tqdm(files, total=len(files), desc="files"):
            print(token_extraction(file_path), flush=True)
        return

    with Pool(processes=args.workers) as pool:
        results = list(tqdm(pool.imap_unordered(token_extraction, files), total=len(files), desc="files"))
    for result in results:
        print(result, flush=True)


# ───── Entry Point ──────────────────────────────────────
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
