import os
import math
import shutil
import wandb
import numpy as np
from datetime import datetime

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import torch
from torch.nn.utils.rnn import pad_sequence

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Draw

import pickle

import wandb


def init_wandb(config, project_id='project id', run_prefix='run prefix'):
    if config.wandb:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_prefix is None:
            run_id = f"{config.exp_name}_{timestamp}"
        else:
            run_id = f"{run_prefix}_{config.exp_name}_{timestamp}"

        wandb.init(entity='entity', project=project_id, id=run_id, dir=config.prefix)
        wandb.config.update(config.__dict__, allow_val_change=True)
        return wandb
    
    return None


def compute_similarity(origin_psmiles, recon_psmiles, n_samples, config):
    validity, sim_scores = 0, 0.0

    n_samples = min(len(recon_psmiles), n_samples)
    selected = list(np.random.choice(len(origin_psmiles), n_samples, replace=False))
    print(f"Validity and similarity check for {n_samples} samples")

    if config.rdFingerprintGen == None:
        config.rdFingerprintGen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    for idx in tqdm(selected):
        sim_score = tanimoto_similarity(origin_psmiles[idx], recon_psmiles[idx], config.rdFingerprintGen)

        if sim_score is not None:
            validity += 1
            sim_scores += sim_score

    sim_scores = sim_scores / validity if validity > 0 else 0
    validity = validity / n_samples
    print(f"[Epoch: {config.epoch}]  Validity: {validity:.3f}")
    print(f"[Epoch: {config.epoch}]  Similarity: {sim_scores:.3f}")

    return validity, sim_scores


def tanimoto_similarity(smi1, smi2, generator):
    try:
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)

        fp1 = generator.GetFingerprint(mol1)
        fp2 = generator.GetFingerprint(mol2)

        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception as e:
        return None
    

def decode_with_eos(pred_ids, tokenizer, eos_token_id, pad_token_id=None):
    pred_ids_trimmed = []
    for seq in pred_ids:
        eos_pos = (seq == eos_token_id).nonzero(as_tuple=True)[0]
        cutoff = eos_pos[0].item() + 1 if len(eos_pos) > 0 else len(seq)
        pred_ids_trimmed.append(seq[:cutoff])

    if pad_token_id is not None:
        pred_ids_trimmed = pad_sequence(pred_ids_trimmed, batch_first=True, padding_value=pad_token_id)

    decoded = tokenizer.batch_decode(pred_ids_trimmed, skip_special_tokens=True)
    return decoded
    


def decode_with_cleanup(pred_ids, tokenizer, eos_token_id, pad_token_id=None):
    """
    pred_ids: list[Tensor] or Tensor [B, L]
    tokenizer: HuggingFace tokenizer
    eos_token_id: int
    pad_token_id: int or None
    """
    pred_ids_trimmed = []
    for seq in pred_ids:
        
        eos_pos = (seq == eos_token_id).nonzero(as_tuple=True)[0]
        cutoff = eos_pos[0].item() + 1 if len(eos_pos) > 0 else len(seq)
        trimmed = seq[:cutoff]

        if pad_token_id is not None:
            trimmed = trimmed[trimmed != pad_token_id]

        pred_ids_trimmed.append(trimmed)

    if pad_token_id is not None:
        pred_ids_trimmed = pad_sequence(pred_ids_trimmed, batch_first=True, padding_value=pad_token_id)

    decoded = tokenizer.batch_decode(pred_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    decoded = [s.replace(" ", "") for s in decoded]
    decoded = [s.replace("<pad>", "") for s in decoded]

    return decoded


def Standardize(train, valid, test, precomputed_path=None):
    """
    :param data:Input data
    :return:normalized data
    """
    if precomputed_path is not None:
        print("Statistics loaded from:", precomputed_path)
        with open(precomputed_path, 'rb') as f:
            scaler = pickle.load(f)
    else:
        print("Statistics is not available")
        scaler = StandardScaler()
        scaler.fit(train)
    a = scaler.transform(train)
    b = scaler.transform(valid)
    c = scaler.transform(test)
    return a.astype(np.float32), b.astype(np.float32), c.astype(np.float32)


def logging_from_dict(prefix: str, out_dict, wandb, config):
    content = {}
    for k, v in out_dict.items():
        content[k] = v


    out_str = f"[Epoch {config.epoch}] [{prefix}]" if hasattr(config, 'epoch') else f"[{prefix}]"
    for k, v in content.items():
        out_str += f"  {k}: {v:.4f}"
    print(out_str)

    ddict = {prefix: content}
    if config.wandb: wandb.log(ddict, step=config.epoch)


def seed_everything(seed=1062):
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def save_checkpoint(config, state_dict, save_path, is_best=False):
    if config.save:
        torch.save(state_dict, save_path)

        if is_best:
            shutil.copy(save_path, os.path.join(config.save_path, 'Checkpoint_BEST.pt'))


class DiagonalGaussianDistribution:
    """Diagonal Gaussian distribution with mean and logvar parameters.

    Adapted from: https://github.com/CompVis/latent-diffusion, with modifications for our tensors,
    which are of shape (N, d) instead of (B, H, W, d) for 2D images.
    """

    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)  # split along channel dim
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=1
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=1,
                )

    def mode(self):
        return self.mean

    def __repr__(self):
        return f"DiagonalGaussianDistribution(mean={self.mean}, logvar={self.logvar})"


def compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2

    total_norm = total_norm ** 0.5

    return total_norm


def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))

def compute_r2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    return 1 - ss_res / ss_tot