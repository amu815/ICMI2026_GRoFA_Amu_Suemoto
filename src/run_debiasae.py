#!/usr/bin/env python3
"""debiaSAE reproduction for embedding-space bias mitigation.

Based on: Kuleen Sasse et al., "debiaSAE: Benchmarking and Mitigating Vision-Language
Model Bias", arXiv:2410.13146 (2024-2025). Official repo: https://github.com/KuleenS/VLMBiasEval

Adaptation: VLMBiasEval's original code relies on gemma-scope pretrained SAEs for
Paligemma/LLM internals. Since we operate on ViT CLS embeddings from BLIP/CLIP/DINOv2,
we train a small Sparse Autoencoder per backbone on clean training embeddings, then
apply the debiaSAE intervention (zero-clamp bias-correlated SAE features at inference)
identically to the original method.

Pipeline:
  1. Load base embeddings (frozen backbone CLS) + sensitive-attribute labels.
  2. Train SAE on clean (noise_level==0) train embeddings.
  3. Identify top-K bias features via |Spearman rho| with gender label.
  4. Intervene on ALL embeddings (clean + noisy): encode -> zero top-K -> decode.
  5. Save debiased train/test npz matching the baseline pipeline schema.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SparseAE(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.enc = nn.Linear(d_in, d_hidden, bias=True)
        self.dec = nn.Linear(d_hidden, d_in, bias=True)
        nn.init.kaiming_uniform_(self.enc.weight, a=0)
        nn.init.kaiming_uniform_(self.dec.weight, a=0)
        # Tie dec weight to transpose of enc (common SAE init)
        with torch.no_grad():
            self.dec.weight.copy_(self.enc.weight.t())

    def encode(self, x):
        return F.relu(self.enc(x))

    def decode(self, f):
        return self.dec(f)

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


def train_sae(
    emb: np.ndarray,
    d_hidden: int = 4096,
    l1: float = 1e-3,
    epochs: int = 60,
    lr: float = 1e-3,
    batch_size: int = 512,
    seed: int = 0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d_in = emb.shape[1]
    sae = SparseAE(d_in, d_hidden).to(DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    x = torch.from_numpy(emb.astype(np.float32)).to(DEVICE)
    ds = TensorDataset(x)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for ep in range(epochs):
        total_recon, total_l1 = 0.0, 0.0
        for (batch,) in dl:
            opt.zero_grad()
            x_hat, f = sae(batch)
            recon = F.mse_loss(x_hat, batch)
            sparsity = f.abs().mean()
            loss = recon + l1 * sparsity
            loss.backward()
            opt.step()
            total_recon += recon.item() * batch.size(0)
            total_l1 += sparsity.item() * batch.size(0)
        n = len(ds)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  ep {ep:3d}: recon={total_recon/n:.5f} l1={total_l1/n:.5f}")
    return sae


def identify_bias_features(
    sae: SparseAE,
    clean_emb: np.ndarray,
    sensitive: np.ndarray,
    top_k: int = 50,
):
    """Find SAE features most correlated with sensitive attribute via point-biserial
    correlation (or Spearman for multi-class). Returns indices of top-K |rho|."""
    with torch.no_grad():
        x = torch.from_numpy(clean_emb.astype(np.float32)).to(DEVICE)
        feats = sae.encode(x).cpu().numpy()  # (N, d_hidden)
    # Normalize sensitive to integer codes
    uniq = sorted(set(sensitive.tolist()))
    mapping = {v: i for i, v in enumerate(uniq)}
    s = np.array([mapping[v] for v in sensitive.tolist()], dtype=np.float32)
    # Compute Pearson correlation per feature (fast vectorized)
    f_mean = feats.mean(axis=0, keepdims=True)
    f_std = feats.std(axis=0, keepdims=True) + 1e-8
    s_mean = s.mean()
    s_std = s.std() + 1e-8
    rho = ((feats - f_mean) * (s - s_mean).reshape(-1, 1)).mean(axis=0) / (f_std.squeeze() * s_std)
    order = np.argsort(-np.abs(rho))
    return order[:top_k], rho[order[:top_k]]


def intervene(
    sae: SparseAE,
    emb: np.ndarray,
    bias_idx: np.ndarray,
    batch_size: int = 512,
):
    """Apply debiaSAE intervention: encode -> zero out bias features -> decode."""
    sae.eval()
    out = np.empty_like(emb, dtype=np.float32)
    mask = torch.ones(sae.enc.out_features, device=DEVICE)
    mask[bias_idx] = 0.0
    with torch.no_grad():
        for i in range(0, emb.shape[0], batch_size):
            chunk = torch.from_numpy(emb[i : i + batch_size].astype(np.float32)).to(DEVICE)
            f = sae.encode(chunk)
            f_debiased = f * mask
            x_hat = sae.decode(f_debiased)
            out[i : i + batch_size] = x_hat.cpu().numpy()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_train", required=True, type=Path)
    p.add_argument("--base_test", required=True, type=Path)
    p.add_argument("--out_train", required=True, type=Path)
    p.add_argument("--out_test", required=True, type=Path)
    p.add_argument("--sensitive", default="gender",
                   choices=["gender", "race", "age"])
    p.add_argument("--d_hidden", type=int, default=4096)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--l1", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"[debiaSAE] loading {args.base_train}")
    tr = dict(np.load(args.base_train, allow_pickle=True))
    te = dict(np.load(args.base_test, allow_pickle=True))
    emb_tr = tr["embeddings"]
    emb_te = te["embeddings"]

    # Clean-only subset for SAE training and bias identification
    nl_tr = tr["noise_level"].astype(int) if "noise_level" in tr else np.zeros(len(emb_tr), dtype=int)
    clean_mask = nl_tr == 0
    emb_clean = emb_tr[clean_mask]
    sens_clean = tr[args.sensitive][clean_mask]
    print(f"[debiaSAE] clean train: {emb_clean.shape}, full train: {emb_tr.shape}, test: {emb_te.shape}")

    print(f"[debiaSAE] training SAE (d_hidden={args.d_hidden}, epochs={args.epochs})")
    sae = train_sae(emb_clean, d_hidden=args.d_hidden, l1=args.l1, epochs=args.epochs, seed=args.seed)

    print(f"[debiaSAE] identifying top-{args.top_k} bias features (sensitive={args.sensitive})")
    bias_idx, rhos = identify_bias_features(sae, emb_clean, sens_clean, top_k=args.top_k)
    print(f"  max |rho|={np.abs(rhos).max():.4f}, min |rho|={np.abs(rhos).min():.4f}")

    print(f"[debiaSAE] intervening on all train/test embeddings")
    emb_tr_debiased = intervene(sae, emb_tr, bias_idx)
    emb_te_debiased = intervene(sae, emb_te, bias_idx)

    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    # Preserve all metadata fields from base
    save_tr = {k: tr[k] for k in tr.keys() if k != "embeddings"}
    save_tr["embeddings"] = emb_tr_debiased
    save_te = {k: te[k] for k in te.keys() if k != "embeddings"}
    save_te["embeddings"] = emb_te_debiased
    np.savez_compressed(args.out_train, **save_tr)
    np.savez_compressed(args.out_test, **save_te)
    print(f"[debiaSAE] saved {args.out_train} and {args.out_test}")


if __name__ == "__main__":
    main()
