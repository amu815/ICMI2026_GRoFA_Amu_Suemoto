#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src3/train_neurips_v60.py (ICMI 2026 — RTDAR-G v60: v55b on IPMix LoRA Backbone)

Based on v55b with one key change:
  - Uses IPMix LoRA-finetuned BLIP as the frozen backbone (instead of vanilla BLIP)
  - Both base_model and teacher use the IPMix LoRA backbone
  - This combines IPMix's noise-robust features with DART's adapter+noise gate

Rationale: IPMix+WiSE-FT achieves 50/66 BM> and beats DART on absolute accuracy
in many conditions. By training DART's adapter on top of IPMix's stronger backbone,
we get both methods' advantages: IPMix's augmentation robustness + DART's noise gate.

All other components inherited from v55b:
  - Task-Rotating Batch Distillation
  - Per-Task HSIC + Clean Anchoring
  - AAAC + MGP + TDARRouter
"""
import argparse
import json
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from transformers import BlipModel
from transformers import BlipForConditionalGeneration
from peft import PeftModel
from models import (
    NeurIPSModelV28, TDARRouter,
)

# --- External Loss Libraries ---
from lightly.loss import NTXentLoss
from pytorch_metric_learning.losses import (
    MultiSimilarityLoss,
    ArcFaceLoss,
    TripletMarginLoss,
)
from pytorch_metric_learning.miners import TripletMarginMiner

# --- Custom Losses ---
from losses import (
    HSICLoss,
    BarlowTwinsLoss,
    VICRegLoss,
    CenterLoss,
)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_lora_backbone(lora_ckpt: str, lora_scale: float = 1.0):
    """Load BLIP vision model with IPMix LoRA merged.

    Args:
        lora_ckpt: Path to LoRA adapter directory (must contain adapter_config.json)
        lora_scale: WiSE-FT alpha for LoRA (0=pretrained, 1=fully finetuned)
    Returns:
        Merged BLIP vision model with LoRA applied
    """
    base_id = "Salesforce/blip-image-captioning-base"
    full = BlipForConditionalGeneration.from_pretrained(base_id)
    vision = full.vision_model
    vision = PeftModel.from_pretrained(vision, lora_ckpt)

    if lora_scale != 1.0:
        with torch.no_grad():
            for name, param in vision.named_parameters():
                if "lora_B" in name:
                    param.data *= lora_scale
        print(f"  -> LoRA backbone scaled by α={lora_scale}")

    vision = vision.merge_and_unload()
    print(f"  -> LoRA backbone merged from {lora_ckpt}")
    return vision

# ==========================================
# Loss Pool Configuration (17 losses)
# ==========================================
PER_SAMPLE_NAMES = ["MSE", "Cosine", "SmoothL1", "KL_Div", "FFL", "L1", "SSIM"]
BATCH_NAMES = ["NT-Xent", "MultiSim", "Triplet", "ArcFace", "Center", "Barlow", "VICReg"]
CLS_NAMES = ["CE_Race", "CE_Gender", "CE_Age"]
LOSS_NAMES = PER_SAMPLE_NAMES + BATCH_NAMES + CLS_NAMES
DIST_NAMES = PER_SAMPLE_NAMES + BATCH_NAMES
NUM_PER_SAMPLE = len(PER_SAMPLE_NAMES)     # 7
NUM_BATCH = len(BATCH_NAMES)                # 7
NUM_CLS = len(CLS_NAMES)                    # 3
NUM_DIST = NUM_PER_SAMPLE + NUM_BATCH       # 14
NUM_TOTAL = NUM_DIST + NUM_CLS              # 17


# ==========================================
# Per-Sample Loss Functions (indices 0-6)
# ==========================================
def compute_per_sample_losses(z_noisy, z_clean):
    """Compute 7 per-sample distillation losses. Returns: (B, 7) tensor."""
    losses = []
    losses.append(((z_noisy - z_clean) ** 2).mean(dim=-1))
    losses.append(1.0 - F.cosine_similarity(z_noisy, z_clean, dim=-1))
    losses.append(F.smooth_l1_loss(z_noisy, z_clean, reduction="none").mean(dim=-1))
    tau = 2.0
    log_p = F.log_softmax(z_noisy / tau, dim=-1)
    q = F.softmax(z_clean.detach() / tau, dim=-1)
    losses.append(F.kl_div(log_p, q, reduction="none").sum(dim=-1) * (tau ** 2))
    f_n = torch.fft.rfft(z_noisy, dim=-1, norm="ortho")
    f_c = torch.fft.rfft(z_clean, dim=-1, norm="ortho")
    diff = (f_n - f_c).abs()
    focal_weight = diff.detach() + 1e-8
    losses.append((focal_weight * diff).mean(dim=-1))
    losses.append(F.l1_loss(z_noisy, z_clean, reduction="none").mean(dim=-1))
    mu_x = z_noisy.mean(dim=-1)
    mu_y = z_clean.mean(dim=-1)
    sig_x = z_noisy.var(dim=-1)
    sig_y = z_clean.var(dim=-1)
    sig_xy = (z_noisy * z_clean).mean(dim=-1) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sig_x + sig_y + c2)
    )
    losses.append(1.0 - ssim)
    result = torch.stack(losses, dim=-1)
    result = torch.nan_to_num(result, nan=0.0, posinf=100.0, neginf=0.0)
    return result


# ==========================================
# Batch-Level Loss Helpers
# ==========================================
def _safe_loss(loss_val, device, max_val=100.0):
    if torch.isnan(loss_val) or torch.isinf(loss_val):
        return torch.tensor(0.0, device=device, requires_grad=True)
    return loss_val.clamp(max=max_val)


def _triplet_loss(z, y, miner_fn, crit, device):
    indices = miner_fn(z, y)
    if indices[0].numel() > 0:
        return _safe_loss(crit(z, y, indices), device)
    return torch.tensor(0.0, device=device, requires_grad=True)


# ==========================================
# R-TDAR Core: Task-Decomposed Alignment (v29, with v32 AAAC)
# ==========================================
def compute_tda_scores(z_out, per_sample_losses, batch_losses,
                       logits_r, logits_g, logits_a,
                       y_race, y_gender, y_age, crit_ce, tau_task=1.0,
                       delta_age=None):
    """
    Compute Task-Decomposed Alignment (TDA) scores.

    Key difference from v28 compute_alignment_scores:
      - Computes per-TASK gradients (g_race, g_gender, g_age) separately
      - Task difficulty weighting: lambda_t = softmax(L_t / tau_task)
      - Per-task alignment: cosine(g_k, g_t) for each task t
      - Weighted aggregation: a_k = sum_t lambda_t * cosine(g_k, g_t)

    v32 AAAC: If delta_age is not None, clips age alignment scores at
    delta_age to prevent over-suppression of reconstruction losses.

    Args:
        z_out: (B, 768) adapter output (requires_grad=True)
        per_sample_losses: (B, 7) per-sample losses
        batch_losses: list of 7 scalar tensors (batch-level losses)
        logits_r, logits_g, logits_a: classification logits
        y_race, y_gender, y_age: labels
        crit_ce: nn.CrossEntropyLoss()
        tau_task: temperature for task difficulty weighting
        delta_age: if not None, clip age alignment scores at this minimum value

    Returns:
        alignment: (B, 14) weighted task-decomposed alignment, DETACHED
        task_weights: (3,) difficulty weights [race, gender, age], DETACHED
        per_task_align: (B, 14, 3) per-task alignment scores, DETACHED
    """
    assert z_out.requires_grad, "z_out must require grad for TDA alignment"

    B, D = z_out.shape

    # === Per-task losses and gradients ===
    L_race = crit_ce(logits_r, y_race)
    L_gender = crit_ce(logits_g, y_gender)
    L_age = crit_ce(logits_a, y_age)

    g_race = torch.autograd.grad(
        L_race, z_out, retain_graph=True, create_graph=False
    )[0]  # (B, D)
    g_gender = torch.autograd.grad(
        L_gender, z_out, retain_graph=True, create_graph=False
    )[0]  # (B, D)
    g_age = torch.autograd.grad(
        L_age, z_out, retain_graph=True, create_graph=False
    )[0]  # (B, D)

    # Stack per-task gradients: (B, 3, D)
    g_tasks = torch.stack([
        F.normalize(g_race, dim=-1),
        F.normalize(g_gender, dim=-1),
        F.normalize(g_age, dim=-1),
    ], dim=1)

    # Task difficulty weighting: lambda_t = softmax(L_t / tau_task)
    with torch.no_grad():
        task_losses = torch.stack([L_race, L_gender, L_age])  # (3,)
        task_weights = F.softmax(task_losses / tau_task, dim=0)  # (3,)

    # === Distillation loss gradients ===
    g_dist_list = []

    # Per-sample loss gradients (losses 0-6)
    for k in range(7):
        g_k = torch.autograd.grad(
            per_sample_losses[:, k].sum(), z_out,
            retain_graph=True, create_graph=False
        )[0]  # (B, D)
        g_dist_list.append(F.normalize(g_k, dim=-1))

    # Batch loss gradients (losses 7-13)
    for k in range(7):
        if batch_losses[k].requires_grad:
            g_k = torch.autograd.grad(
                batch_losses[k], z_out,
                retain_graph=True, create_graph=False,
                allow_unused=True,
            )[0]
            if g_k is None:
                g_k = torch.zeros_like(z_out)
        else:
            g_k = torch.zeros_like(z_out)
        g_dist_list.append(F.normalize(g_k + 1e-8, dim=-1))

    # Stack distillation gradients: (B, 14, D)
    g_dist = torch.stack(g_dist_list, dim=1)

    # Per-task alignment: cosine(g_k, g_t) -> (B, 14, 3)
    # einsum: (B, K, D) x (B, T, D) -> (B, K, T)
    per_task_align = torch.einsum('bkd,btd->bkt', g_dist, g_tasks)

    # v32 AAAC: Age-Aware Alignment Clipping
    # Clips age alignment (index 2) at delta_age to prevent over-suppression
    if delta_age is not None:
        per_task_align[:, :, 2] = torch.clamp(per_task_align[:, :, 2], min=delta_age)

    # Weighted aggregation: sum_t lambda_t * cosine(g_k, g_t) -> (B, 14)
    # einsum: (B, K, T) x (T,) -> (B, K)
    alignment = torch.einsum('bkt,t->bk', per_task_align, task_weights)

    alignment = torch.nan_to_num(alignment, nan=0.0)
    return alignment.detach(), task_weights.detach(), per_task_align.detach()


# ==========================================
# Dataset (Same as v28)
# ==========================================
class NeurIPSData(Dataset):
    def __init__(self, jsonl_path, transform=None):
        self.jsonl_path = Path(jsonl_path)
        self.data_root = self.jsonl_path.parent.parent
        with open(jsonl_path, "r") as f:
            self.rows = [json.loads(l) for l in f]
        self.transform = transform
        self.gender_map = {"male": 0, "female": 1}

        all_races = sorted(set(row["race"] for row in self.rows))
        self.race_map = {race: i for i, race in enumerate(all_races)}
        self.num_groups = len(self.race_map)
        print(f"  [Dataset] Dynamic race_map ({self.num_groups} races): {self.race_map}")

        all_ages = sorted(set(row.get("age", "unknown") for row in self.rows))
        self.age_map = {age: i for i, age in enumerate(all_ages)}
        self.num_ages = len(self.age_map)
        print(f"  [Dataset] Dynamic age_map ({self.num_ages} ages): {self.age_map}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        item = self.rows[idx]
        noisy_path = self.data_root / item["image"]
        if item["noise_type"] == "clean":
            clean_path = noisy_path
        else:
            p = Path(item["image"])
            clean_dir = p.parent.parent / "clean"
            clean_fname = p.name.replace(
                f"_{item['noise_type']}_lv{item['noise_level']}", ""
            )
            clean_path = self.data_root / clean_dir / clean_fname
        try:
            img_n = Image.open(noisy_path).convert("RGB")
            img_c = Image.open(clean_path).convert("RGB")
        except Exception:
            img_n = Image.new("RGB", (224, 224))
            img_c = Image.new("RGB", (224, 224))
        if self.transform:
            img_n = self.transform(img_n)
            img_c = self.transform(img_c)
        g_label = self.gender_map.get(item.get("gender"), 0)
        r_label = self.race_map.get(item.get("race"), 0)
        a_label = self.age_map.get(item.get("age", "unknown"), 0)
        noise_level = int(item.get("noise_level", 0))
        return {
            "img_noisy": img_n, "img_clean": img_c,
            "gender": torch.tensor(g_label, dtype=torch.long),
            "race": torch.tensor(r_label, dtype=torch.long),
            "age": torch.tensor(a_label, dtype=torch.long),
            "noise_type": item["noise_type"],
            "noise_level": torch.tensor(noise_level, dtype=torch.long),
        }


def get_transforms():
    import torchvision.transforms as T
    return T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize(mean=[0.481, 0.458, 0.408], std=[0.268, 0.261, 0.275]),
    ])


# ==========================================
# Visualization
# ==========================================
def save_weight_trajectory(weight_history, dist_names, out_dir):
    """Save weight trajectory over batches."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_steps = len(weight_history)
    if n_steps == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    arr = np.array(weight_history)
    for k, name in enumerate(dist_names):
        ax.plot(range(n_steps), arr[:, k], label=name, linewidth=1.0, alpha=0.8)
    ax.set_xlabel("Step (logged every 10 batches)", fontsize=12)
    ax.set_ylabel("Router Weight", fontsize=12)
    ax.set_title("R-TDAR: Distillation Loss Weight Trajectory", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "weight_trajectory.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_accuracy_curves(acc_history, out_dir):
    """Save validation accuracy curves per task."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_epochs = len(acc_history)
    task_names = ["Race", "Gender", "Age"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for t, task_name in enumerate(task_names):
        vals = [acc_history[ep][t] for ep in range(n_epochs)]
        ax.plot(range(1, n_epochs + 1), vals, label=task_name, linewidth=2)
    mean_vals = [np.mean(acc_history[ep]) for ep in range(n_epochs)]
    ax.plot(range(1, n_epochs + 1), mean_vals, label="Mean", linewidth=2,
            linestyle="--", color="black")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("R-TDAR: Validation Accuracy", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "val_accuracy_curves.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_alignment_heatmap(alignment_history, dist_names, out_dir):
    """Save mean alignment score per loss per epoch as heatmap."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_epochs = len(alignment_history)
    if n_epochs == 0:
        return

    arr = np.array(alignment_history)  # (n_epochs, 14)
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(arr.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_yticks(range(len(dist_names)))
    ax.set_yticklabels(dist_names, fontsize=8)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_title("R-TDAR: Mean Alignment Score a_k(x) per Epoch", fontsize=14)
    plt.colorbar(im, ax=ax, shrink=0.8, label="cosine(g_k, g_task)")
    plt.tight_layout()
    plt.savefig(out_dir / "alignment_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_tau_trajectory(tau_history, out_dir):
    """Save temperature tau trajectory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(tau_history) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(tau_history) + 1), tau_history, linewidth=2.5,
            color="crimson", marker="o", markersize=4)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Temperature tau", fontsize=12)
    ax.set_title("R-TDAR: Routing Temperature", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=6.5, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='tau_init=6.5')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "tau_trajectory.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_prior_heatmap(prior_b, dist_names, race_names, out_dir):
    """Save final prior b matrix as heatmap."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(prior_b.T, aspect="auto", cmap="RdBu_r")
    ax.set_xticks(range(len(race_names)))
    ax.set_xticklabels([race_names[i] for i in range(len(race_names))], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(dist_names)))
    ax.set_yticklabels(dist_names, fontsize=8)
    ax.set_title("R-TDAR: Learned Prior b^(g) (per-group, per-loss)", fontsize=14)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Prior bias")
    # Annotate cell values
    for i in range(prior_b.shape[0]):
        for j in range(prior_b.shape[1]):
            val = prior_b[i, j]
            color = "white" if abs(val) > 0.3 else "black"
            ax.text(i, j, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)
    plt.tight_layout()
    plt.savefig(out_dir / "prior_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_noise_trajectory(noise_history, out_dir):
    """Save noise score trajectory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(noise_history) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(noise_history) + 1), noise_history, linewidth=2.5,
            color="darkcyan", marker="D", markersize=4, label="Mean noise_score")
    ax.axhline(y=0.0, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='s=0 (clean)')
    ax.axhline(y=1.0, color='red', linestyle=':', linewidth=1, alpha=0.5, label='s=1 (noisy)')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Noise Score", fontsize=12)
    ax.set_title("R-TDAR: Noise Score (batch mean)", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "noise_score_trajectory.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_fairness_gap_trajectory(gap_history, out_dir):
    """Save fairness gap trajectory per task."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(gap_history) == 0:
        return

    arr = np.array(gap_history)  # (n_epochs, 3)
    task_names = ["Race", "Gender", "Age"]
    epsilons = [0.03, 0.15, 0.10]

    fig, ax = plt.subplots(figsize=(10, 5))
    for t, task_name in enumerate(task_names):
        ax.plot(range(1, len(gap_history) + 1), arr[:, t],
                linewidth=2, label=f"{task_name} gap")
        ax.axhline(y=epsilons[t], color=f"C{t}", linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy Gap (max - min across groups)", fontsize=12)
    ax.set_title("R-TDAR: Fairness Gap per Task", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fairness_gap_trajectory.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_task_weight_trajectory(task_weight_history, out_dir):
    """Save TDA task difficulty weight trajectory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(task_weight_history) == 0:
        return

    arr = np.array(task_weight_history)  # (n_epochs, 3)
    task_names = ["Race", "Gender", "Age"]

    fig, ax = plt.subplots(figsize=(10, 4))
    for t, task_name in enumerate(task_names):
        ax.plot(range(1, len(task_weight_history) + 1), arr[:, t],
                linewidth=2.5, marker="s", markersize=4, label=task_name)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Task Difficulty Weight (lambda_t)", fontsize=12)
    ax.set_title("R-TDAR: Task Difficulty Weights (softmax of losses)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "task_weight_trajectory.png", dpi=200, bbox_inches="tight")
    plt.close()


# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="RTDAR-G v55: Task-Rotating Batch Distillation for Multi-Task Fairness"
    )
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--lora_ckpt", type=str, default=None,
                        help="Path to LoRA adapter dir for backbone (e.g., IPMix LoRA)")
    parser.add_argument("--lora_scale", type=float, default=1.0,
                        help="WiSE-FT alpha for backbone LoRA (0=pretrained, 1=fully finetuned)")

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cls_ratio", type=float, default=0.7,
                        help="Legacy: uniform cls weight (used if cls_w_* not set)")
    parser.add_argument("--cls_w_race", type=float, default=None,
                        help="Per-task CE weight for race")
    parser.add_argument("--cls_w_gender", type=float, default=None,
                        help="Per-task CE weight for gender")
    parser.add_argument("--cls_w_age", type=float, default=None,
                        help="Per-task CE weight for age")
    parser.add_argument("--lambda_hsic", type=float, default=None,
                        help="(deprecated) Global HSIC weight")
    parser.add_argument("--lambda_hsic_race", type=float, default=0.5,
                        help="HSIC independence weight for race")
    parser.add_argument("--lambda_hsic_gender", type=float, default=0.3,
                        help="HSIC independence weight for gender")
    parser.add_argument("--lambda_anchor", type=float, default=0.5,
                        help="Clean anchoring loss weight")
    parser.add_argument("--lambda_noise", type=float, default=0.1)
    parser.add_argument("--batch_loss_warmup", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)

    # Training settings
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    # Ablation modes
    parser.add_argument("--ablation_mode", type=str, default="rtdar_g",
                        choices=[
                            "rtdar_v29_repro",
                            "rtdar_ac",
                            "rtdar_mgp",
                            "rtdar_g",
                        ])

    # R-TDAR specific hyperparameters
    parser.add_argument("--tau_min", type=float, default=3.0,
                        help="Minimum temperature for TDARRouter (prevents argmax collapse)")
    parser.add_argument("--alpha", type=float, default=5.0,
                        help="Alignment signal scaling factor")
    parser.add_argument("--lambda_l2", type=float, default=0.01,
                        help="L2 regularization weight for prior b")
    parser.add_argument("--lambda_ent", type=float, default=0.1,
                        help="Entropy regularization weight (maximizes weight entropy)")
    parser.add_argument("--tau_task", type=float, default=1.0,
                        help="Temperature for task difficulty weighting in TDA")

    # Supporting components
    parser.add_argument("--use_tgh", action="store_true", default=False,
                        help="Enable Task-Gradient Harmonization (PCGrad)")

    # v32: Age-Aware Alignment Clipping (AAAC)
    parser.add_argument("--delta_age", type=float, default=-0.01,
                        help="Age alignment clip floor (AAAC). -0.01 prevents over-suppression.")

    # v32: Max-Gap Penalty (MGP)
    parser.add_argument("--lambda_mgp", type=float, default=0.5,
                        help="MGP penalty strength (fixed, no Lagrangian)")
    parser.add_argument("--eps_race", type=float, default=0.04,
                        help="MGP tolerance for race gap")
    parser.add_argument("--eps_gender", type=float, default=0.15,
                        help="MGP tolerance for gender gap")
    parser.add_argument("--eps_age", type=float, default=0.12,
                        help="MGP tolerance for age gap")
    parser.add_argument("--mgp_warmup", type=int, default=3,
                        help="Epochs before activating MGP")

    # v55: Reduced fixed ArcFace weight (v53=0.5, v55=0.15) to prevent race dominance
    parser.add_argument("--arc_fixed_weight", type=float, default=0.15,
                        help="Weight for fixed ArcFace loss (v55 default 0.15, v53 was 0.5)")
    parser.add_argument("--adv_weight", type=float, default=0.2,
                        help="Weight for domain adversarial loss (default 0.2)")
    parser.add_argument("--arc_margin", type=float, default=0.35,
                        help="ArcFace margin (default 0.35)")
    parser.add_argument("--arc_scale", type=float, default=48.0,
                        help="ArcFace scale (default 48.0)")
    parser.add_argument("--rotation_pattern", type=int, nargs='+', default=[0, 1, 2],
                        help="Task rotation: 0=race, 1=age, 2=gender (default: [0,1,2])")
    parser.add_argument("--gate_ceiling", type=float, default=1.0,
                        help="Max gate value for noise-gated adapter (1.0=no limit, 0.3=cap at 30%%)")
    parser.add_argument("--arf_floor", type=float, default=0.0,
                        help="Min gate value (0.0=full suppression on noisy, 0.05=small contribution)")

    args = parser.parse_args()

    # Default per-task CE weights from cls_ratio if not specified
    default_w = args.cls_ratio / 3.0
    if args.cls_w_race is None:
        args.cls_w_race = default_w
    if args.cls_w_gender is None:
        args.cls_w_gender = default_w
    if args.cls_w_age is None:
        args.cls_w_age = default_w

    # Seed for reproducibility
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"  [Seed] Set to {args.seed} (deterministic mode)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = args.out_dir / "analysis_logs"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # --- Mode Configuration (v32: all modes use RTDAR) ---
    use_rtdar = True
    use_router = True
    # v32 feature flags based on ablation mode
    use_aaac = args.ablation_mode in ("rtdar_ac", "rtdar_g")
    use_mgp = args.ablation_mode in ("rtdar_mgp", "rtdar_g")
    delta_age_val = args.delta_age if use_aaac else None

    # --- Datasets ---
    ds_train = NeurIPSData(args.train_jsonl, transform=get_transforms())
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=8, drop_last=True)
    ds_val = NeurIPSData(args.val_jsonl, transform=get_transforms())
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, drop_last=False)

    num_groups = ds_train.num_groups
    num_ages = ds_train.num_ages
    race_names = {v: k for k, v in ds_train.race_map.items()}

    # --- Models ---
    print("[Init] Loading Models...")
    if args.lora_ckpt:
        print(f"[v60] Using LoRA backbone: {args.lora_ckpt} (scale={args.lora_scale})")
        base_model = load_lora_backbone(args.lora_ckpt, args.lora_scale)
        teacher = load_lora_backbone(args.lora_ckpt, args.lora_scale).to(DEVICE).eval()
    else:
        _full = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        base_model = _full.vision_model
        _full_t = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        teacher = _full_t.vision_model.to(DEVICE).eval()

    model = NeurIPSModelV28(
        base_model, num_race=num_groups, num_gender=2, num_age=num_ages,
        hidden_dim=384, arf_floor=args.arf_floor, gate_ceiling=args.gate_ceiling,
    ).to(DEVICE)

    # Explicit zero-init adapter output layer
    nn.init.zeros_(model.adapter.adapter[5].weight)
    nn.init.zeros_(model.adapter.adapter[5].bias)
    print("[v32] Zero-initialized adapter output layer")

    # --- Router (v32: always TDARRouter) ---
    router = TDARRouter(
        num_losses=NUM_DIST, num_groups=num_groups,
        tau_min=args.tau_min, tau_max=10.0, alpha=args.alpha,
    ).to(DEVICE)
    router_params = sum(p.numel() for p in router.parameters())
    router_type = "TDARRouter"
    print(f"[v32] TDARRouter (task-decomposed alignment): {router_params:,} params "
          f"(G={num_groups}, K={NUM_DIST})")
    print(f"  Mode: {args.ablation_mode}")
    print(f"  tau_min={args.tau_min}, alpha={args.alpha}, "
          f"lambda_l2={args.lambda_l2}, lambda_ent={args.lambda_ent}, "
          f"tau_task={args.tau_task}")
    print(f"  AAAC: {'ON (delta_age=' + str(args.delta_age) + ')' if use_aaac else 'OFF'}")
    print(f"  MGP: {'ON (lambda=' + str(args.lambda_mgp) + ', eps=[' + str(args.eps_race) + ',' + str(args.eps_gender) + ',' + str(args.eps_age) + '], warmup=' + str(args.mgp_warmup) + ')' if use_mgp else 'OFF'}")

    # --- Batch-Level Loss Criteria ---
    crit_ntx = NTXentLoss(temperature=0.07)
    crit_ms = MultiSimilarityLoss()
    crit_trip = TripletMarginLoss(margin=0.4)
    miner = TripletMarginMiner(margin=0.15, type_of_triplets="hard")
    # v55: Task-specific ArcFace (race + gender; age uses Center instead)
    crit_arc_race = ArcFaceLoss(num_classes=num_groups, embedding_size=768).to(DEVICE)
    crit_arc_gender = ArcFaceLoss(num_classes=2, embedding_size=768).to(DEVICE)
    crit_cen = CenterLoss()
    crit_barlow = BarlowTwinsLoss()
    crit_vic = VICRegLoss()

    # --- Fixed Losses ---
    crit_hsic = HSICLoss()
    crit_ce = nn.CrossEntropyLoss()

    # --- v51 NEW: Separate ArcFace (fixed weight, not routed) ---
    from train_neurips_v49 import ArcFace as ArcFaceFixed, FocalLoss, GRL
    arcface_fixed = ArcFaceFixed(768, num_groups, m=args.arc_margin, s=args.arc_scale).to(DEVICE)
    dom_head = nn.Linear(768, 2).to(DEVICE)
    focal_adv = FocalLoss()

    # --- v51 NEW: EMA model ---
    import copy
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters(): p.requires_grad_(False)

    # --- Optimizer ---
    param_groups = [
        {"params": filter(lambda p: p.requires_grad, model.parameters()), "lr": args.lr},
        # v55: Task-specific ArcFace instances
        {"params": crit_arc_race.parameters(), "lr": args.lr},
        {"params": crit_arc_gender.parameters(), "lr": args.lr},
        # RTDAR router uses 10x LR for faster prior adaptation
        {"params": router.parameters(), "lr": args.lr * 10},
        # v51: fixed ArcFace + domain head
        {"params": arcface_fixed.parameters(), "lr": args.lr},
        {"params": dom_head.parameters(), "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # --- State Variables ---
    training_logs = []
    best_val_metric = -float("inf")
    patience_counter = 0
    acc_history = []
    weight_log = []
    tau_history = []
    alignment_epoch_history = []  # (n_epochs, 14) mean alignment per loss
    noise_score_history = []
    noise_loss_history = []
    gap_history = []
    task_weight_history = []     # v29: (n_epochs, 3) task difficulty weights
    prior_norm_history = []      # v29: (n_epochs,) prior b norm
    current_gaps = None  # Updated at validation, used during training

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = model_params + router_params

    # --- Print config ---
    mode_desc = {
        "rtdar_v29_repro": "RTDAR-v29-Repro: v29 reproduction (no AAAC, no MGP)",
        "rtdar_ac": "RTDAR-AC: Age-Aware Alignment Clipping only",
        "rtdar_mgp": "RTDAR-MGP: Max-Gap Penalty only",
        "rtdar_g": "RTDAR-G: Full (AAAC + MGP)",
    }

    print(f"\n=== RTDAR-G v55 (Task-Rotating Batch Distillation) ===")
    print(f"  Ablation Mode: {args.ablation_mode} -- {mode_desc[args.ablation_mode]}")
    print(f"  Num Groups: {num_groups} ({', '.join(race_names[i] for i in range(num_groups))})")
    print(f"  Num Ages: {num_ages}")
    print(f"  Loss Pool: {NUM_TOTAL} losses ({NUM_PER_SAMPLE} per-sample + {NUM_BATCH} batch + {NUM_CLS} cls)")
    print(f"  v55 Rotation: batch_idx%3 -> [race, age, gender]")
    print(f"    ArcFace: race(routed, {num_groups}cls), gender(routed, 2cls), age(Center instead)")
    print(f"    Fixed ArcFace weight: {args.arc_fixed_weight} (v53 was 0.5)")
    print(f"  Trainable params (model): {model_params:,}")
    print(f"  Trainable params (router): {router_params:,}")
    print(f"  Trainable params (total): {total_params:,}")
    print(f"  Epochs: {args.epochs} (patience={args.patience})")
    print(f"  lr={args.lr}, cls_ratio={args.cls_ratio}, lambda_hsic_race={args.lambda_hsic_race}, lambda_hsic_gender={args.lambda_hsic_gender}, lambda_anchor={args.lambda_anchor}")
    print(f"  lambda_noise={args.lambda_noise}")
    print(f"  R-TDAR: tau_min={args.tau_min}, alpha={args.alpha}, "
          f"lambda_l2={args.lambda_l2}, lambda_ent={args.lambda_ent}, tau_task={args.tau_task}")
    print(f"  Router: {router_type} (ON)")
    print(f"  TGH: {'ON' if args.use_tgh else 'OFF'}")
    print(f"  AAAC: {'ON (delta_age=' + str(args.delta_age) + ')' if use_aaac else 'OFF'}")
    print(f"  MGP: {'ON (lambda=' + str(args.lambda_mgp) + ', warmup=' + str(args.mgp_warmup) + ')' if use_mgp else 'OFF'}")
    print()

    for ep in range(args.epochs):
        model.train()
        if router is not None:
            router.train()
        total_loss = 0
        total_cls_loss = 0
        total_distill_loss = 0
        epoch_logs = []
        epoch_noise_score_sum = 0.0
        epoch_noise_loss_sum = 0.0
        epoch_noise_count = 0
        epoch_alignment_sum = np.zeros(NUM_DIST)  # for alignment tracking
        epoch_alignment_count = 0
        epoch_task_weight_sum = np.zeros(3)       # v29: task weights
        epoch_task_weight_count = 0
        epoch_per_task_align_sum = np.zeros((NUM_DIST, 3))  # v29: per-task alignment

        # Batch loss warmup
        if args.batch_loss_warmup > 0 and ep < args.batch_loss_warmup:
            batch_loss_scale = ep / args.batch_loss_warmup
        else:
            batch_loss_scale = 1.0

        # ==================================
        # Training Loop
        # ==================================
        pbar = tqdm(dl_train, ncols=140, desc=f"Ep {ep+1}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            img_n = batch["img_noisy"].to(DEVICE)
            img_c = batch["img_clean"].to(DEVICE)
            y_race = batch["race"].to(DEVICE)
            y_gen = batch["gender"].to(DEVICE)
            y_age = batch["age"].to(DEVICE)
            noise_level = batch["noise_level"].to(DEVICE).float()

            optimizer.zero_grad()

            # --- Teacher (frozen) ---
            with torch.no_grad():
                z_clean = teacher(pixel_values=img_c).last_hidden_state[:, 0, :]

            # --- Student (v28/v29: noise-modulated adapter) ---
            z_out, z_base, logits_r, logits_g, logits_a, noise_score = model(img_n)

            # ==============================
            # Classification losses
            # ==============================
            loss_ce_race = crit_ce(logits_r, y_race)
            loss_ce_gender = crit_ce(logits_g, y_gen)
            loss_ce_age = crit_ce(logits_a, y_age)
            cls_loss = (args.cls_w_race * loss_ce_race
                        + args.cls_w_gender * loss_ce_gender
                        + args.cls_w_age * loss_ce_age)

            # ==============================
            # Per-sample distillation losses (0-6)
            # ==============================
            L_ps = compute_per_sample_losses(z_out, z_clean)  # (B, 7)

            # ==============================
            # Batch-level distillation losses (7-13)
            # v55: Task-rotating labels for labeled batch losses
            # ==============================
            z_norm = F.normalize(z_out, dim=1)

            # v55: Rotate task labels per batch
            _rot = args.rotation_pattern
            task_idx = _rot[batch_idx % len(_rot)]
            if task_idx == 0:
                y_batch = y_race
                _arc_fn = lambda: crit_arc_race(z_out, y_race)
            elif task_idx == 1:
                y_batch = y_age
                _arc_fn = lambda: crit_cen(z_out, y_age)  # Center for age (ordinal, no ArcFace)
            else:
                y_batch = y_gen
                _arc_fn = lambda: crit_arc_gender(z_out, y_gen)

            batch_crits = [
                ("NT-Xent", lambda: crit_ntx(z_out, z_clean)),
                ("MultiSim", lambda: crit_ms(z_norm, y_batch)),
                ("Triplet", lambda: _triplet_loss(z_norm, y_batch, miner, crit_trip, DEVICE)),
                ("ArcFace", _arc_fn),
                ("Center", lambda: crit_cen(z_out, y_batch)),
                ("Barlow", lambda: crit_barlow(z_out, z_clean)),
                ("VICReg", lambda: crit_vic(z_out, z_clean)),
            ]
            batch_losses = []
            for name, loss_fn in batch_crits:
                try:
                    l_b = _safe_loss(loss_fn(), DEVICE)
                    batch_losses.append(l_b)
                except Exception:
                    batch_losses.append(torch.tensor(0.0, device=DEVICE, requires_grad=True))

            # ==============================
            # Router: Compute distillation weights
            # ==============================
            dist_budget = 1.0 - (args.cls_w_race + args.cls_w_gender + args.cls_w_age)

            if use_rtdar:
                # === R-TDAR: Compute task-decomposed alignment scores ===
                alignment, task_weights_t, per_task_align = compute_tda_scores(
                    z_out, L_ps, batch_losses,
                    logits_r, logits_g, logits_a,
                    y_race, y_gen, y_age, crit_ce,
                    tau_task=args.tau_task,
                    delta_age=delta_age_val,  # v32: AAAC (None if disabled)
                )  # alignment: (B, 14) detached
                batch_task_weights = task_weights_t.cpu().numpy()
                epoch_per_task_align_sum += per_task_align.mean(dim=0).cpu().numpy()

                router_weights, rtdar_diag = router(alignment, y_race)  # (B, 14)

                # Track alignment & task weights
                epoch_alignment_sum += alignment.mean(dim=0).cpu().numpy()
                epoch_alignment_count += 1
                epoch_task_weight_sum += batch_task_weights
                epoch_task_weight_count += 1

                # Per-sample losses: element-wise weighting
                w_ps = router_weights[:, :NUM_PER_SAMPLE]  # (B, 7)
                loss_distill_ps = (w_ps * L_ps).sum(dim=-1).mean()

                # Batch losses: batch-averaged weights
                w_batch = router_weights[:, NUM_PER_SAMPLE:].mean(dim=0)  # (7,)
                loss_distill_batch = sum(
                    w_batch[i] * batch_losses[i] for i in range(NUM_BATCH)
                )

                loss_distill = dist_budget * (
                    loss_distill_ps + batch_loss_scale * loss_distill_batch
                )

                weights_np = router_weights.mean(dim=0).detach().cpu().numpy()

            # ==============================
            # HSIC fairness regularization (v55b: per-task weights)
            # ==============================
            l_hsic_gen = crit_hsic(z_out, y_gen)
            l_hsic_race = crit_hsic(z_out, y_race)
            loss_hsic = args.lambda_hsic_race * l_hsic_race + args.lambda_hsic_gender * l_hsic_gen

            # ==============================
            # Clean anchoring: penalize adapter deviation on clean inputs
            # ==============================
            clean_mask = (noise_level == 0)
            if clean_mask.any():
                loss_anchor = ((z_out[clean_mask] - z_base[clean_mask]) ** 2).mean()
            else:
                loss_anchor = torch.tensor(0.0, device=DEVICE)

            # ==============================
            # Noise supervision
            # ==============================
            target_s = noise_level / 3.0  # 0=clean, 0.33, 0.67, 1.0=noisy
            loss_noise = F.binary_cross_entropy(noise_score.squeeze(-1), target_s)
            epoch_noise_score_sum += noise_score.detach().mean().item()
            epoch_noise_loss_sum += loss_noise.item()
            epoch_noise_count += 1

            # ==============================
            # v32: Max-Gap Penalty (MGP)
            # ==============================
            loss_mgp = torch.tensor(0.0, device=DEVICE)
            if use_mgp and current_gaps is not None and ep >= args.mgp_warmup:
                epsilon_t = torch.tensor(
                    [args.eps_race, args.eps_gender, args.eps_age],
                    device=DEVICE
                )
                violations = (current_gaps - epsilon_t).clamp(min=0.0)
                loss_mgp = args.lambda_mgp * violations.max()

            # ==============================
            # R-TDAR Regularization
            # ==============================
            loss_prior_l2 = args.lambda_l2 * router.compute_prior_regularization()
            loss_entropy = args.lambda_ent * router.compute_weight_entropy(router_weights)

            # ==============================
            # v51 NEW: Fixed ArcFace + Domain Adversarial
            # ==============================
            loss_arc_fixed = arcface_fixed(z_out, y_race) * args.arc_fixed_weight

            # Domain adversary: z_clean (teacher, clean) vs z_out (student, noisy)
            z_dom = torch.cat([z_clean.detach(), z_out])
            dom_labels = torch.cat([torch.zeros(len(z_clean)), torch.ones(len(z_out))]).long().to(DEVICE)
            loss_adv = focal_adv(dom_head(GRL.apply(z_dom, 0.6)), dom_labels) * args.adv_weight

            # ==============================
            # Total Loss
            # ==============================
            final_loss = (loss_distill + cls_loss
                          + loss_hsic  # per-task weights already applied
                          + args.lambda_anchor * loss_anchor
                          + args.lambda_noise * loss_noise
                          + loss_mgp
                          + loss_prior_l2 + loss_entropy
                          + loss_arc_fixed + loss_adv)

            if torch.isnan(final_loss) or torch.isinf(final_loss):
                optimizer.zero_grad()
                if batch_idx % 50 == 0:
                    print(f"  [WARN] NaN/Inf loss at batch {batch_idx}, skipping")
                continue

            final_loss.backward()
            all_params = (list(model.parameters()) +
                          list(crit_arc_race.parameters()) + list(crit_arc_gender.parameters()) +
                          list(router.parameters()) + list(arcface_fixed.parameters()) +
                          list(dom_head.parameters()))
            torch.nn.utils.clip_grad_norm_(all_params, args.max_grad_norm)
            optimizer.step()

            # v51 NEW: EMA update
            with torch.no_grad():
                for (k1,p1),(k2,p2) in zip(model.named_parameters(), ema_model.named_parameters()):
                    if k1==k2: p2.data.mul_(0.99).add_(p1.data, alpha=0.01)

            total_loss += final_loss.item()
            total_distill_loss += loss_distill.item()
            total_cls_loss += cls_loss.item()

            # Logging
            if batch_idx % 20 == 0:
                log_entry = {
                    "Epoch": ep + 1, "Batch": batch_idx,
                    "Loss": final_loss.item(),
                    "DistillLoss": loss_distill.item(),
                    "ClsLoss": cls_loss.item(),
                    "HSIC": loss_hsic.item(),
                    "AnchorLoss": loss_anchor.item(),
                    "NoiseLoss": loss_noise.item(),
                    "NoiseScore": noise_score.detach().mean().item(),
                }
                if use_rtdar:
                    log_entry["tau"] = rtdar_diag["tau"]
                    log_entry["alpha"] = rtdar_diag["alpha"]
                    log_entry["prior_norm"] = rtdar_diag["prior_norm"]
                    log_entry["weight_entropy"] = rtdar_diag["weight_entropy"]
                    log_entry["alignment_mean"] = alignment.mean().item()
                    log_entry["alignment_std"] = alignment.std().item()
                    log_entry["loss_prior_l2"] = loss_prior_l2.item()
                    log_entry["loss_entropy"] = loss_entropy.item()
                    for ti, tname in enumerate(["race", "gender", "age"]):
                        log_entry[f"task_weight_{tname}"] = float(batch_task_weights[ti])
                if use_mgp:
                    log_entry["MGP_penalty"] = loss_mgp.item()
                for i, name in enumerate(DIST_NAMES):
                    log_entry[f"W_{name}"] = float(weights_np[i])
                epoch_logs.append(log_entry)

                weight_log.append(weights_np.copy())

            # Progress bar
            mgp_str = f" mgp={loss_mgp.item():.3f}" if use_mgp and loss_mgp.item() > 0 else ""
            pbar.set_postfix_str(
                f"L={final_loss.item():.2f} D={loss_distill.item():.2f} "
                f"C={cls_loss.item():.2f} tau={rtdar_diag['tau']:.2f} "
                f"a={alignment.mean().item():.3f} "
                f"ent={rtdar_diag['weight_entropy']:.2f} "
                f"s={noise_score.detach().mean().item():.2f}{mgp_str}"
            )

        training_logs.extend(epoch_logs)

        # --- Epoch summary ---
        n_batches = max(len(dl_train), 1)
        wu_str = f" [BATCH_WU={batch_loss_scale:.2f}]" if batch_loss_scale < 1.0 else ""

        # Track router histories
        tau_history.append(router.tau.item())
        prior_norm_history.append(router.b.detach().norm().item())
        if epoch_alignment_count > 0:
            alignment_epoch_history.append(
                (epoch_alignment_sum / epoch_alignment_count).tolist()
            )
        if epoch_task_weight_count > 0:
            task_weight_history.append(
                (epoch_task_weight_sum / epoch_task_weight_count).tolist()
            )
        if epoch_noise_count > 0:
            noise_score_history.append(epoch_noise_score_sum / epoch_noise_count)
            noise_loss_history.append(epoch_noise_loss_sum / epoch_noise_count)

        print(
            f"Ep {ep+1}/{args.epochs}{wu_str} "
            f"Loss={total_loss/n_batches:.4f} "
            f"Distill={total_distill_loss/n_batches:.4f} "
            f"Cls={total_cls_loss/n_batches:.4f} "
            f"LR={scheduler.get_last_lr()[0]:.6f}"
        )

        # Print router info
        top3_idx = np.argsort(weights_np)[-3:][::-1]
        bot3_idx = np.argsort(weights_np)[:3]
        print(f"  [RTDAR] tau={router.tau.item():.3f} alpha={args.alpha}")
        print(f"  [RTDAR] Top-3: {', '.join(f'{DIST_NAMES[i]}={weights_np[i]:.4f}' for i in top3_idx)}")
        print(f"  [RTDAR] Bot-3: {', '.join(f'{DIST_NAMES[i]}={weights_np[i]:.4f}' for i in bot3_idx)}")
        if alignment_epoch_history:
            a_mean = alignment_epoch_history[-1]
            print(f"  [RTDAR] Mean alignment: {np.mean(a_mean):.4f} "
                  f"(range [{np.min(a_mean):.3f}, {np.max(a_mean):.3f}])")
        w_ent = -np.sum(weights_np * np.log(weights_np + 1e-8))
        max_ent = np.log(NUM_DIST)
        print(f"  [RTDAR] Weight entropy: {w_ent:.3f} / {max_ent:.3f} (ratio={w_ent/max_ent:.3f})")
        # Prior b stats
        b_np = router.b.detach().cpu().numpy()
        print(f"  [RTDAR] Prior b: mean={b_np.mean():.4f} std={b_np.std():.4f} "
              f"range=[{b_np.min():.3f}, {b_np.max():.3f}] norm={np.linalg.norm(b_np):.4f}")
        if task_weight_history:
            tw = task_weight_history[-1]
            print(f"  [RTDAR] Task weights: race={tw[0]:.3f} gender={tw[1]:.3f} age={tw[2]:.3f}")

        if noise_score_history:
            print(f"  [Noise] score={noise_score_history[-1]:.4f} loss={noise_loss_history[-1]:.4f}")

        # ==================================
        # Validation
        # ==================================
        model.eval()
        router.eval()

        # Per-group accuracy tracking (for MGP)
        val_cls_correct = {
            "race": np.zeros(num_groups),
            "gender": np.zeros(num_groups),
            "age": np.zeros(num_groups),
        }
        val_cls_count = np.zeros(num_groups)

        with torch.no_grad():
            for batch in dl_val:
                img_n = batch["img_noisy"].to(DEVICE)
                y_race_v = batch["race"].to(DEVICE)
                y_gen_v = batch["gender"].to(DEVICE)
                y_age_v = batch["age"].to(DEVICE)

                z_out, z_base, lr_v, lg_v, la_v, ns_v = model(img_n)

                # Per-group accuracy
                for g in range(num_groups):
                    mask = (y_race_v == g)
                    n_g = mask.sum().item()
                    if n_g > 0:
                        val_cls_count[g] += n_g
                        val_cls_correct["race"][g] += (lr_v.argmax(1)[mask] == y_race_v[mask]).sum().item()
                        val_cls_correct["gender"][g] += (lg_v.argmax(1)[mask] == y_gen_v[mask]).sum().item()
                        val_cls_correct["age"][g] += (la_v.argmax(1)[mask] == y_age_v[mask]).sum().item()

        # Overall accuracy
        total_correct_r = val_cls_correct["race"].sum()
        total_correct_g = val_cls_correct["gender"].sum()
        total_correct_a = val_cls_correct["age"].sum()
        total_n = val_cls_count.sum()

        acc_r = total_correct_r / max(total_n, 1)
        acc_g = total_correct_g / max(total_n, 1)
        acc_a = total_correct_a / max(total_n, 1)
        val_metric = (acc_r + acc_g + acc_a) / 3.0

        acc_history.append([acc_r, acc_g, acc_a])

        # Per-group accuracy for fairness gaps
        race_acc_pg = val_cls_correct["race"] / np.maximum(val_cls_count, 1)
        gender_acc_pg = val_cls_correct["gender"] / np.maximum(val_cls_count, 1)
        age_acc_pg = val_cls_correct["age"] / np.maximum(val_cls_count, 1)

        gap_race = race_acc_pg.max() - race_acc_pg.min()
        gap_gender = gender_acc_pg.max() - gender_acc_pg.min()
        gap_age = age_acc_pg.max() - age_acc_pg.min()

        current_gaps = torch.tensor([gap_race, gap_gender, gap_age],
                                     dtype=torch.float32, device=DEVICE)
        gap_history.append([gap_race, gap_gender, gap_age])

        print(f"  Val ClsAcc: race={acc_r:.3f} gender={acc_g:.3f} "
              f"age={acc_a:.3f} | Mean={val_metric:.3f}")
        print(f"  Val Gaps: race={gap_race:.3f} gender={gap_gender:.3f} age={gap_age:.3f}")

        # v32: Log MGP status
        if use_mgp:
            if ep >= args.mgp_warmup and current_gaps is not None:
                epsilon_t = torch.tensor([args.eps_race, args.eps_gender, args.eps_age], device=DEVICE)
                violations = (current_gaps - epsilon_t).clamp(min=0.0)
                print(f"  [MGP] max_violation={violations.max().item():.4f} "
                      f"(race={violations[0].item():.4f}, gender={violations[1].item():.4f}, age={violations[2].item():.4f})")
            else:
                print(f"  [MGP] warmup ({ep+1}/{args.mgp_warmup})")

        # --- Early Stopping ---
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            patience_counter = 0
            save_dict = {
                "model": model.state_dict(),
                "rtdar_router": router.state_dict(),
            }
            torch.save(save_dict, args.out_dir / "model_best.pt")
            print(f"  -> New best model (acc={val_metric:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"  [Early Stop] No improvement for {args.patience} epochs. Stopping.")
                break

        scheduler.step()

    # --- Save training dynamics ---
    pd.DataFrame(training_logs).to_csv(analysis_dir / "training_dynamics.csv", index=False)

    # --- Save visualizations ---
    if weight_log:
        save_weight_trajectory(weight_log, DIST_NAMES, analysis_dir)
        np.save(analysis_dir / "weight_history.npy", np.array(weight_log))

    if tau_history:
        save_tau_trajectory(tau_history, analysis_dir)
        np.save(analysis_dir / "tau_history.npy", np.array(tau_history))

    if alignment_epoch_history:
        save_alignment_heatmap(alignment_epoch_history, DIST_NAMES, analysis_dir)
        np.save(analysis_dir / "alignment_history.npy", np.array(alignment_epoch_history))

    if noise_score_history:
        save_noise_trajectory(noise_score_history, analysis_dir)
        np.save(analysis_dir / "noise_score_history.npy", np.array(noise_score_history))
        np.save(analysis_dir / "noise_loss_history.npy", np.array(noise_loss_history))

    if gap_history:
        save_fairness_gap_trajectory(gap_history, analysis_dir)
        np.save(analysis_dir / "gap_history.npy", np.array(gap_history))

    if acc_history:
        save_accuracy_curves(acc_history, analysis_dir)
        np.save(analysis_dir / "acc_history.npy", np.array(acc_history))

    # v29: Save task weight trajectory
    if task_weight_history:
        save_task_weight_trajectory(task_weight_history, analysis_dir)
        np.save(analysis_dir / "task_weight_history.npy", np.array(task_weight_history))

    # v29: Save prior norm history
    if prior_norm_history:
        np.save(analysis_dir / "prior_norm_history.npy", np.array(prior_norm_history))

    # Save prior b heatmap
    b_np = router.b.detach().cpu().numpy()
    save_prior_heatmap(b_np, DIST_NAMES, race_names, analysis_dir)
    np.save(analysis_dir / "prior_b.npy", b_np)

    # --- Save config ---
    config = {
        "version": "v55",
        "method": "RTDAR-G",
        "seed": args.seed,
        "v55_rotation": "round_robin (batch_idx%3: race/age/gender)",
        "gate_ceiling": args.gate_ceiling,
        "arf_floor": args.arf_floor,
        "v55_arc_fixed_weight": args.arc_fixed_weight,
        "ablation_mode": args.ablation_mode,
        "router_type": router_type,
        "use_aaac": use_aaac,
        "use_mgp": use_mgp,
        "delta_age": args.delta_age if use_aaac else None,
        "lambda_mgp": args.lambda_mgp if use_mgp else None,
        "eps_race": args.eps_race,
        "eps_gender": args.eps_gender,
        "eps_age": args.eps_age,
        "mgp_warmup": args.mgp_warmup,
        "use_tgh": args.use_tgh,
        "num_groups": num_groups,
        "num_ages": num_ages,
        "race_map": ds_train.race_map,
        "age_map": ds_train.age_map,
        "num_losses": NUM_TOTAL,
        "num_dist_losses": NUM_DIST,
        "num_cls_losses": NUM_CLS,
        "loss_names": LOSS_NAMES,
        "epochs": args.epochs,
        "epochs_actual": ep + 1,
        "patience": args.patience,
        "lr": args.lr,
        "cls_ratio": args.cls_ratio,
        "lambda_hsic_race": args.lambda_hsic_race,
        "lambda_hsic_gender": args.lambda_hsic_gender,
        "lambda_anchor": args.lambda_anchor,
        "lambda_noise": args.lambda_noise,
        "batch_loss_warmup": args.batch_loss_warmup,
        "tau_min": args.tau_min,
        "alpha": args.alpha,
        "lambda_l2": args.lambda_l2,
        "lambda_ent": args.lambda_ent,
        "tau_task": args.tau_task,
        "tau_history": tau_history,
        "alignment_epoch_history": alignment_epoch_history,
        "task_weight_history": task_weight_history,
        "prior_norm_history": prior_norm_history,
        "noise_score_history": noise_score_history,
        "noise_loss_history": noise_loss_history,
        "gap_history": gap_history,
        "best_val_metric": best_val_metric,
        "model_params": model_params,
        "router_params": router_params,
        "total_params": total_params,
    }
    with open(args.out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n[Done] Training complete.")
    print(f"  Best val metric (mean acc): {best_val_metric:.4f}")
    print(f"  Stopped at epoch: {ep + 1}/{args.epochs}")
    print(f"  Model saved to: {args.out_dir / 'model_best.pt'}")
    print(f"  Analysis logs: {analysis_dir}")


if __name__ == "__main__":
    main()
