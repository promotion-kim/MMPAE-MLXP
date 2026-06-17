import argparse
from types import SimpleNamespace

import torch

from models.MMTransformer import MMTransformerAR


class DummyTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __len__(self):
        return 64


def build_batch(batch_size: int, sequence_length: int, num_properties: int, device: torch.device):
    tokenizer = DummyTokenizer()
    input_ids = torch.full(
        (batch_size, sequence_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    input_ids[:, 0] = tokenizer.bos_token_id
    input_ids[:, 1:8] = torch.randint(3, len(tokenizer), (batch_size, 7), device=device)
    input_ids[:, 8] = tokenizer.eos_token_id

    properties = torch.randn(batch_size, num_properties, device=device)
    return tokenizer, properties, input_ids


def run_smoke(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    num_properties = 29
    tokenizer, properties, input_ids = build_batch(
        args.batch_size,
        args.sequence_length,
        num_properties,
        device,
    )

    model = MMTransformerAR(
        tokenizer=tokenizer,
        vocab_size=len(tokenizer),
        latent_dim=args.d_model,
        d_model=args.d_model,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        num_layers=args.num_layers,
        dec_layers=args.dec_layers,
        activation="gelu",
        bias=True,
        norm_first=True,
        pad_token_id=tokenizer.pad_token_id,
        dropout=0.0,
        alpha=1.0,
        beta=1.0,
        gamma=0.0,
        temperature=0.1,
        num_properties=num_properties,
        fullrep=False,
        L2=True,
        loss_type="CwA",
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    logits, predict_prop, zf, zs = model(
        properties=properties,
        token_ids=input_ids,
        drop_rate=0.0,
        mode="train",
    )
    loss, ce_loss, mse_loss, contrast_loss, eos_loss = model.compute_loss_with_logits(
        input_ids=input_ids,
        properties=properties,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        logits=logits,
        predict_prop=predict_prop,
        zf=zf,
        zs=zs,
    )

    loss.backward()
    optimizer.step()

    print("smoke test passed")
    print(f"device={device}")
    print(f"logits_shape={tuple(logits.shape)}")
    print(f"predict_prop_shape={tuple(predict_prop.shape)}")
    print(f"loss={loss.item():.6f}")
    print(f"ce_loss={ce_loss.item():.6f}")
    print(f"mse_loss={mse_loss.item():.6f}")
    print(f"contrast_loss={contrast_loss.item():.6f}")
    print(f"eos_loss={eos_loss.item():.6f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic HMMPAE smoke test")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--sequence_length", type=int, default=160)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dec_layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1004)
    return parser.parse_args(namespace=SimpleNamespace())


if __name__ == "__main__":
    run_smoke(parse_args())
