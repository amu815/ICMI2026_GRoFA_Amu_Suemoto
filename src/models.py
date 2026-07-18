#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models.py — GRoFA model definitions

SSD-Net: Spectral-Spatial Dual-Gated Network
Two loss weighting modes:
  - "kg": Kendall & Gal uncertainty + sample-adaptive delta (v4 legacy)
  - "mol": Mixture-of-Losses Router - per-sample dynamic loss routing (v5)

Gate collapse prevention via balanced normalization + entropy regularization.
Frequency path: FFT filter + LayerNorm (gradient balance fix)
Spatial path: MLP adapter + LayerNorm
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Mixture-of-Losses Router (v5 contribution)
# ============================================================
class MoLRouter(nn.Module):
    """
    Mixture-of-Losses Router: per-sample dynamic loss selection and weighting.

    Inspired by Mixture-of-Experts (MoE) routing (Shazeer et al., 2017;
    Fedus et al., 2022 Switch Transformer), applied to the loss function space.

    Given input features, the router predicts a weight distribution over
    candidate losses for each sample. This enables:
      - Sample-adaptive loss emphasis (e.g., frequency loss for JPEG, MSE for Gaussian)
      - Automatic discovery of optimal loss combinations
      - Sparse routing via optional top-k selection

    Args:
        input_dim: dimension of input features
        num_losses: number of candidate losses in the pool
        top_k: if set, only top-k losses are active per sample (sparse routing)
    """

    def __init__(self, input_dim, num_losses, top_k=None):
        super().__init__()
        self.num_losses = num_losses
        self.top_k = top_k

        self.router = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_losses),
        )
        # Initialize near-uniform: zeros → softmax → 1/num_losses each
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

        # Learnable temperature for softmax sharpness
        # Starts at 1.0 (log(1)=0), learns to sharpen or soften
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        """
        Args:
            z: (B, D) input features (typically z_base before adapter)
        Returns:
            weights: (B, num_losses) per-sample loss weights (sum to 1)
            balance_loss: scalar, load balancing auxiliary loss
        """
        temperature = self.log_temperature.exp().clamp(min=0.1, max=10.0)
        logits = self.router(z)  # (B, num_losses)

        if self.top_k is not None and self.top_k < self.num_losses:
            # Sparse routing: only top-k losses active
            _, topk_idx = torch.topk(logits, self.top_k, dim=-1)
            mask = torch.zeros_like(logits)
            mask.scatter_(1, topk_idx, 1.0)
            logits_for_weights = logits.masked_fill(mask == 0, float("-inf"))
        else:
            logits_for_weights = logits

        weights = F.softmax(logits_for_weights / temperature, dim=-1)

        # Load balancing auxiliary loss (Switch Transformer style)
        # Prevents collapse: encourages all losses to be used across the batch
        # f_i = fraction of batch that uses loss i (above threshold)
        # P_i = mean softmax probability for loss i
        full_probs = F.softmax(logits / temperature, dim=-1)
        f = (weights > 0.01).float().mean(dim=0)  # (num_losses,)
        P = full_probs.mean(dim=0)  # (num_losses,)
        balance_loss = self.num_losses * (f * P).sum()

        return weights, balance_loss


# ============================================================
# Residual Router (v9 contribution)
# ============================================================
class ResidualRouter(nn.Module):
    """
    Residual Router: Static Tuned weights as frozen base + learnable per-sample delta.

    Instead of learning routing weights from scratch (cold start problem),
    this router starts from the v6-discovered Static Tuned weights and learns
    a per-sample residual correction. At initialization, delta_head outputs zero,
    so the router exactly reproduces Static Tuned behavior.

    Args:
        input_dim: dimension of input features
        num_losses: number of candidate losses in the pool
        static_weights: (num_losses,) tensor of Static Tuned weights
        top_k: if set, only top-k losses are active per sample (sparse routing)
    """

    def __init__(self, input_dim, num_losses, static_weights, top_k=None):
        super().__init__()
        self.num_losses = num_losses
        self.top_k = top_k

        # Frozen base: log(static_weights) as logit-space anchor
        # Clamp to avoid log(0) = -inf
        static_clamped = static_weights.detach().clamp(min=1e-4)
        self.register_buffer("static_bias", torch.log(static_clamped))

        # Learnable delta head: outputs per-sample correction in logit space
        self.delta_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_losses),
        )
        # Small initialization: xavier with gain=0.1 for first layer
        nn.init.xavier_uniform_(self.delta_head[0].weight, gain=0.1)
        nn.init.zeros_(self.delta_head[0].bias)
        # Zero initialization for final layer → delta starts at 0 → Static Tuned
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

        # Learnable temperature
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        """
        Args:
            z: (B, D) input features
        Returns:
            weights: (B, num_losses) per-sample loss weights (sum to 1)
            balance_loss: scalar, load balancing auxiliary loss
        """
        temperature = self.log_temperature.exp().clamp(min=0.1, max=10.0)
        delta = self.delta_head(z)  # (B, num_losses)
        logits = self.static_bias + delta  # (B, num_losses)

        if self.top_k is not None and self.top_k < self.num_losses:
            _, topk_idx = torch.topk(logits, self.top_k, dim=-1)
            mask = torch.zeros_like(logits)
            mask.scatter_(1, topk_idx, 1.0)
            logits_for_weights = logits.masked_fill(mask == 0, float("-inf"))
        else:
            logits_for_weights = logits

        weights = F.softmax(logits_for_weights / temperature, dim=-1)

        # Load balancing loss (same as MoLRouter)
        full_probs = F.softmax(logits / temperature, dim=-1)
        f = (weights > 0.01).float().mean(dim=0)
        P = full_probs.mean(dim=0)
        balance_loss = self.num_losses * (f * P).sum()

        return weights, balance_loss


# ============================================================
# Group-Conditional Router (v11 contribution)
# ============================================================
class GroupConditionalRouter(nn.Module):
    """
    Group-Conditional Loss Router: group-specific loss weighting + per-sample delta.

    Each demographic group gets its own learned loss weight distribution via a
    (num_groups, num_losses) logit table. A per-sample delta MLP adds fine-grained
    adaptation within each group (e.g., for different noise types).

    Designed for fairness-aware knowledge distillation: the outer-loop training
    optimizes group_logits to equalize distillation quality across demographic groups.

    Args:
        input_dim: dimension of input features
        num_losses: number of candidate losses in the pool
        num_groups: number of demographic groups (e.g., 5 for UTKFace, 7 for FairFace)
        init_weights: (num_losses,) tensor of initial weights (e.g., from Optuna search)
    """

    def __init__(self, input_dim, num_losses, num_groups, init_weights=None,
                 uncertainty_guided=False):
        super().__init__()
        self.num_losses = num_losses
        self.num_groups = num_groups
        self.uncertainty_guided = uncertainty_guided

        # (A) Group-specific base logits: each group gets its own loss weight distribution
        self.group_logits = nn.Parameter(torch.zeros(num_groups, num_losses))
        if init_weights is not None:
            # Initialize all groups from the same Optuna weights → diverge during training
            init_clamped = init_weights.detach().clamp(min=1e-4)
            logits = torch.log(init_clamped)
            self.group_logits.data = logits.unsqueeze(0).expand(num_groups, -1).clone()

        # (B) Per-sample delta: residual correction for within-group variation
        # v12: uncertainty_guided adds entropy feature → input_dim+1
        delta_input_dim = input_dim + 1 if uncertainty_guided else input_dim
        self.delta_net = nn.Sequential(
            nn.Linear(delta_input_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_losses),
        )
        # Small initialization → delta starts near zero → group_logits dominate initially
        nn.init.xavier_uniform_(self.delta_net[0].weight, gain=0.1)
        nn.init.zeros_(self.delta_net[0].bias)
        nn.init.zeros_(self.delta_net[-1].weight)
        nn.init.zeros_(self.delta_net[-1].bias)

        # Learnable temperature for softmax sharpness
        self.log_temperature = nn.Parameter(torch.zeros(1))

        # v12: Learnable residual scale (was fixed 0.1 in v11)
        # Initialized at 0.3 for stronger per-sample adaptation
        self.log_delta_scale = nn.Parameter(torch.tensor(math.log(0.3)))

    @property
    def delta_scale(self):
        """Learnable residual scale, clamped to [0.05, 1.0]."""
        return self.log_delta_scale.exp().clamp(min=0.05, max=1.0)

    def forward(self, z, group_labels):
        """
        Args:
            z: (B, D) or (B, D+1) input features (D+1 when uncertainty_guided)
            group_labels: (B,) integer group labels (0..num_groups-1)
        Returns:
            weights: (B, num_losses) per-sample loss weights (sum to 1)
            balance_loss: scalar, load balancing auxiliary loss
        """
        temperature = self.log_temperature.exp().clamp(min=0.1, max=10.0)
        delta_scale = self.delta_scale

        # Group-specific base + per-sample delta
        group_part = self.group_logits[group_labels]            # (B, num_losses)
        sample_delta = self.delta_net(z)                        # (B, num_losses)
        logits = group_part + delta_scale * sample_delta        # (B, num_losses)

        weights = F.softmax(logits / temperature, dim=-1)       # (B, num_losses)

        # Load balancing auxiliary loss (Switch Transformer style)
        full_probs = F.softmax(logits / temperature, dim=-1)
        f = (weights > 0.01).float().mean(dim=0)   # (num_losses,)
        P = full_probs.mean(dim=0)                  # (num_losses,)
        balance_loss = self.num_losses * (f * P).sum()

        return weights, balance_loss


# ============================================================
# SSD-Net: Spectral-Spatial Dual-Gated Network
# ============================================================
class SSDNet(nn.Module):
    """
    Spectral-Spatial Dual-Gated Network.

    Two processing paths for noise robustness:
      - Frequency path: learnable spectral filter (FFT domain)
      - Spatial path: MLP residual adapter
    Combined via a learned gate with entropy regularization.

    Loss weighting modes:
      - "kg": K&G log-variance + sample-adaptive delta (v4)
      - "mol": MoL Router per-sample routing (v5)
    """

    def __init__(self, input_dim, num_losses, mode="kg", top_k=None,
                 uncertainty_guided=False, high_freq_input=False,
                 static_weights=None, num_groups=None):
        super().__init__()
        self.input_dim = input_dim
        self.num_losses = num_losses
        self.mode = mode
        self.uncertainty_guided = uncertainty_guided
        self.high_freq_input = high_freq_input

        # =====================
        # High-Frequency Input (v8)
        # =====================
        if high_freq_input:
            # AvgPool1d for high-pass filter: x_hp = x - AvgPool(x)
            self.avg_pool = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)

        # =====================
        # Frequency Branch
        # =====================
        self.freq_dim = input_dim // 2 + 1
        self.complex_weight = nn.Parameter(
            torch.view_as_real(torch.ones(self.freq_dim, dtype=torch.complex64))
        )
        self.freq_norm = nn.LayerNorm(input_dim)

        # =====================
        # Spatial Branch
        # =====================
        self.spatial_adapter = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.GELU(),
            nn.Linear(input_dim // 4, input_dim),
            nn.LayerNorm(input_dim),
        )

        # =====================
        # Gate (Freq vs Spatial)
        # =====================
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # Gate bias = 2.2 → sigmoid(2.2) ≈ 0.9, spatial-heavy start
        nn.init.constant_(self.gate_net[-1].bias, 2.2)

        # =====================
        # Loss Weighting
        # =====================
        if mode == "mol":
            # Uncertainty-Guided Router (v8): input_dim+1 when guided
            router_input_dim = input_dim + 1 if uncertainty_guided else input_dim
            self.mol_router = MoLRouter(router_input_dim, num_losses, top_k=top_k)
        elif mode == "residual_mol":
            # Residual Router (v9): Static Tuned base + per-sample delta
            assert static_weights is not None, "residual_mol mode requires static_weights"
            router_input_dim = input_dim + 1 if uncertainty_guided else input_dim
            self.mol_router = ResidualRouter(
                router_input_dim, num_losses, static_weights, top_k=top_k,
            )
        elif mode == "group_mol":
            # Group-Conditional Router (v11→v12): per-group loss weighting + per-sample delta
            assert num_groups is not None, "group_mol mode requires num_groups"
            self.mol_router = GroupConditionalRouter(
                input_dim, num_losses, num_groups,
                init_weights=static_weights,
                uncertainty_guided=uncertainty_guided,  # v12: pass through
            )
        elif mode == "kg":
            self.log_var = nn.Parameter(torch.zeros(num_losses))
            self.delta_head = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, num_losses),
            )
            nn.init.zeros_(self.delta_head[-1].weight)
            nn.init.zeros_(self.delta_head[-1].bias)

    def forward(self, x, gate_override=None, group_labels=None):
        """
        Args:
            x: (B, D) base embeddings from frozen backbone
            gate_override: if not None, use this fixed gate value
            group_labels: (B,) integer group labels (required for group_mol mode)
        Returns (mode="mol"/"residual_mol"/"group_mol"):
            z_out, gate, mol_weights, balance_loss
        Returns (mode="kg"):
            z_out, gate, log_var, delta
        """
        # 1. Frequency Path
        # High-Freq Input (v8): use high-pass filtered input for frequency path
        if self.high_freq_input:
            # x: (B, D) -> unsqueeze for AvgPool1d -> (B, 1, D)
            x_smooth = self.avg_pool(x.unsqueeze(1)).squeeze(1)  # (B, D)
            x_freq_input = x - x_smooth  # high-pass filtered
        else:
            x_freq_input = x

        x_freq = torch.fft.rfft(x_freq_input, dim=-1, norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x_freq_filt = torch.fft.irfft(
            x_freq * weight, n=x.shape[-1], dim=-1, norm="ortho"
        )
        x_freq_filt = self.freq_norm(x_freq_filt)

        # 2. Spatial Path (always uses raw x)
        x_spatial = self.spatial_adapter(x)

        # 3. Gating
        gate_learned = torch.sigmoid(self.gate_net(x))  # (B, 1)
        if gate_override is not None:
            gate = torch.full_like(gate_learned, gate_override)
        else:
            gate = gate_learned

        # 4. Feature Fusion (with residual)
        z_out = (1 - gate) * x_freq_filt + gate * x_spatial + x

        # 5. Loss Weighting
        if self.mode in ("mol", "residual_mol"):
            # Uncertainty-Guided Router (v8/v9): append teacher entropy to router input
            if self.uncertainty_guided:
                p = F.softmax(x, dim=-1)  # (B, D)
                H = -(p * torch.log(p + 1e-8)).sum(dim=-1, keepdim=True)  # (B, 1)
                H_norm = H / torch.log(torch.tensor(float(self.input_dim), device=x.device))  # [0, 1]
                router_input = torch.cat([x, H_norm], dim=-1)  # (B, D+1)
            else:
                router_input = x
            mol_weights, balance_loss = self.mol_router(router_input)
            return z_out, gate_learned, mol_weights, balance_loss
        elif self.mode == "group_mol":
            # Group-Conditional Router (v12): group labels + uncertainty guidance
            assert group_labels is not None, "group_mol mode requires group_labels"
            # v12: Uncertainty-Guided delta_net input (same as mol/residual_mol)
            if self.uncertainty_guided:
                p = F.softmax(x, dim=-1)  # (B, D)
                H = -(p * torch.log(p + 1e-8)).sum(dim=-1, keepdim=True)  # (B, 1)
                H_norm = H / torch.log(torch.tensor(float(self.input_dim), device=x.device))
                router_input = torch.cat([x, H_norm], dim=-1)  # (B, D+1)
            else:
                router_input = x
            mol_weights, balance_loss = self.mol_router(router_input, group_labels)
            return z_out, gate_learned, mol_weights, balance_loss
        else:
            raw_delta = self.delta_head(x)
            delta = 0.1 * torch.tanh(raw_delta)
            return z_out, gate_learned, self.log_var, delta


# ============================================================
# Full Model: Frozen BLIP + SSD-Net
# ============================================================
# ============================================================
# FAMO Weighter (v13 contribution — NeurIPS 2023 extension)
# ============================================================
class FairDualWeighter(nn.Module):
    """
    FairDual: Dual-Rate Multi-Loss Optimization for Fair Representation Learning.

    Two-head structure with dual-rate updates:
      - alpha_dist (G, K_d): distillation loss weight logits (inner FAMO update)
      - alpha_cls  (G, K_c): classification loss weight logits (inner FAMO update)
      - ratio_logit (G,):    distillation vs classification budget ratio (outer update)

    Inner loop (batch-level, high-frequency):
      FAMO-style update every M batches — equalizes loss decrease rates across losses.
    Outer loop (epoch-level, low-frequency):
      Performance-gap guided update — shifts budget toward cls for underperforming groups.

    Weight computation:
      ratio_g = sigmoid(ratio_logit[g])
      w_dist = ratio_g * softmax(alpha_dist[g])      # (K_d,)
      w_cls  = (1-ratio_g) * softmax(alpha_cls[g])    # (K_c,)
      w = concat(w_dist, w_cls)                        # (K,)

    Args:
        num_dist_losses: number of distillation losses (K_d = 14)
        num_cls_losses: number of classification losses (K_c = 3)
        num_groups: number of demographic groups (G = 7)
        inner_lr: FAMO batch-level update learning rate
        outer_lr: performance-gap guided update learning rate
        gamma: weight decay on logits
        update_interval: batches between inner FAMO updates (M)
    """

    def __init__(self, num_dist_losses=14, num_cls_losses=3, num_groups=7,
                 inner_lr=0.1, outer_lr=0.02, gamma=5e-3, update_interval=50):
        super().__init__()
        self.num_dist = num_dist_losses
        self.num_cls = num_cls_losses
        self.num_total = num_dist_losses + num_cls_losses
        self.num_groups = num_groups
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.gamma = gamma
        self.update_interval = update_interval

        # Two-head logits
        self.alpha_dist = nn.Parameter(torch.zeros(num_groups, num_dist_losses))
        self.alpha_cls = nn.Parameter(torch.zeros(num_groups, num_cls_losses))
        self.ratio_logit = nn.Parameter(torch.zeros(num_groups))  # sigmoid → distill fraction

        # Sliding window for batch-level FAMO
        self.register_buffer('loss_sum', torch.zeros(num_groups, num_dist_losses + num_cls_losses))
        self.register_buffer('group_count', torch.zeros(num_groups))
        self.register_buffer('prev_losses', torch.zeros(num_groups, num_dist_losses + num_cls_losses))
        self.register_buffer('batch_counter', torch.tensor(0))
        self.register_buffer('has_prev', torch.tensor(False))

    def get_weights(self, group_labels):
        """
        Get per-sample loss weights based on group membership.

        Args:
            group_labels: (B,) integer group labels (0..num_groups-1)
        Returns:
            weights: (B, K_d + K_c) per-sample loss weights
        """
        ratio = torch.sigmoid(self.ratio_logit)  # (G,) distill fraction
        w_dist = ratio.unsqueeze(-1) * F.softmax(self.alpha_dist, dim=-1)  # (G, K_d)
        w_cls = (1.0 - ratio).unsqueeze(-1) * F.softmax(self.alpha_cls, dim=-1)  # (G, K_c)
        all_weights = torch.cat([w_dist, w_cls], dim=-1)  # (G, K)
        return all_weights[group_labels]  # (B, K)

    def get_group_weights(self):
        """Get per-group weight distributions. Returns: (G, K) weights."""
        ratio = torch.sigmoid(self.ratio_logit)
        w_dist = ratio.unsqueeze(-1) * F.softmax(self.alpha_dist, dim=-1)
        w_cls = (1.0 - ratio).unsqueeze(-1) * F.softmax(self.alpha_cls, dim=-1)
        return torch.cat([w_dist, w_cls], dim=-1)

    def get_ratios(self):
        """Get per-group distillation ratios. Returns: (G,) values in [0,1]."""
        return torch.sigmoid(self.ratio_logit)

    @torch.no_grad()
    def accumulate(self, losses_per_sample, group_labels):
        """
        Accumulate per-group losses for batch-level FAMO sliding window.

        Args:
            losses_per_sample: (B, K) per-sample loss values
            group_labels: (B,) integer group labels
        """
        for g in range(self.num_groups):
            mask = (group_labels == g)
            if mask.sum() > 0:
                self.loss_sum[g] += losses_per_sample[mask].sum(dim=0)
                self.group_count[g] += mask.sum().float()
        self.batch_counter += 1

    @torch.no_grad()
    def maybe_inner_update(self):
        """
        FAMO-style inner update every M batches.
        Updates alpha_dist and alpha_cls based on loss decrease rates.
        Returns True if update was performed.
        """
        if self.batch_counter < self.update_interval:
            return False

        # Compute window-average losses
        safe_count = self.group_count.clamp(min=1).unsqueeze(-1)  # (G, 1)
        curr_losses = self.loss_sum / safe_count  # (G, K)

        # Reset sliding window
        self.loss_sum.zero_()
        self.group_count.zero_()
        self.batch_counter.zero_()

        if not self.has_prev:
            self.prev_losses.copy_(curr_losses)
            self.has_prev.fill_(True)
            return False

        # Loss decrease rate (relative)
        scale = self.prev_losses.clamp(min=1e-6)
        decrease_rate = (self.prev_losses - curr_losses) / scale  # (G, K)

        # Split into distill and cls parts
        dr_dist = decrease_rate[:, :self.num_dist]  # (G, K_d)
        dr_cls = decrease_rate[:, self.num_dist:]   # (G, K_c)

        # Center and update alpha_dist
        mean_dist = dr_dist.mean(dim=-1, keepdim=True)
        centered_dist = dr_dist - mean_dist
        self.alpha_dist.data -= self.inner_lr * centered_dist

        # Center and update alpha_cls
        mean_cls = dr_cls.mean(dim=-1, keepdim=True)
        centered_cls = dr_cls - mean_cls
        self.alpha_cls.data -= self.inner_lr * centered_cls

        # Mild weight decay
        self.alpha_dist.data *= (1.0 - self.gamma)
        self.alpha_cls.data *= (1.0 - self.gamma)

        self.prev_losses.copy_(curr_losses)
        return True

    @torch.no_grad()
    def outer_update(self, group_accs, threshold=0.01):
        """
        Outer loop: performance-gap guided update on ratio_logit and alpha_cls.

        Args:
            group_accs: (G, 3) per-group classification accuracy [race, gender, age]
            threshold: minimum gap to trigger update
        """
        # Mean accuracy per group across tasks
        mean_per_group = group_accs.mean(dim=-1)  # (G,)
        overall_mean = mean_per_group.mean()       # scalar

        perf_gap = overall_mean - mean_per_group   # (G,) positive = underperforming

        for g in range(self.num_groups):
            gap = perf_gap[g].item()
            if gap > threshold:
                # Underperforming: decrease ratio_logit → more cls budget
                self.ratio_logit.data[g] -= self.outer_lr * gap
                # Boost worst task in alpha_cls
                worst_task = group_accs[g].argmin()
                self.alpha_cls.data[g, worst_task] += self.outer_lr * gap
            elif gap < -threshold:
                # Overperforming: increase ratio_logit → more distill budget
                self.ratio_logit.data[g] += self.outer_lr * abs(gap)


class FAMOWeighter(nn.Module):
    """
    Per-Group FAMO (Fast Adaptive Multitask Optimization).

    Extension of Liu et al. (NeurIPS 2023) for fairness-aware multi-loss optimization.
    Maintains per-group loss weight logits α ∈ R^{G × K} and updates them based on
    loss decrease rates. Losses that decrease slowly get higher weights automatically.

    Key properties:
      - O(1) overhead: no per-loss backward passes needed
      - Automatic loss pruning: weights → ~0 for conflicting/unhelpful losses
      - Per-group specialization: each demographic group discovers its own optimal loss mix

    Args:
        num_losses: number of candidate losses (K)
        num_groups: number of demographic groups (G)
        w_lr: learning rate for weight logit updates
        gamma: weight decay on logits (prevents divergence)
    """

    def __init__(self, num_losses, num_groups, w_lr=0.025, gamma=1e-3):
        super().__init__()
        self.num_losses = num_losses
        self.num_groups = num_groups
        self.w_lr = w_lr
        self.gamma = gamma

        # Per-group weight logits: softmax → per-group loss weights
        self.logits = nn.Parameter(torch.zeros(num_groups, num_losses))

        # Buffers for FAMO meta-update (not part of model parameters)
        self.register_buffer('prev_losses', torch.zeros(num_groups, num_losses))
        self.register_buffer('epoch_count', torch.tensor(0))

        # Epoch-level loss accumulation buffers
        self.register_buffer('epoch_loss_sum', torch.zeros(num_groups, num_losses))
        self.register_buffer('epoch_group_count', torch.zeros(num_groups))

    def get_weights(self, group_labels):
        """
        Get per-sample loss weights based on group membership.

        Args:
            group_labels: (B,) integer group labels (0..num_groups-1)
        Returns:
            weights: (B, num_losses) per-sample loss weights (sum to 1 per sample)
        """
        all_weights = F.softmax(self.logits, dim=-1)  # (G, K)
        return all_weights[group_labels]  # (B, K)

    def get_group_weights(self):
        """Get per-group weight distributions. Returns: (G, K) softmax weights."""
        return F.softmax(self.logits, dim=-1)

    @torch.no_grad()
    def accumulate_batch(self, losses_per_sample, group_labels):
        """
        Accumulate per-group losses during an epoch (called per batch).

        Args:
            losses_per_sample: (B, K) per-sample loss values
            group_labels: (B,) integer group labels
        """
        for g in range(self.num_groups):
            mask = (group_labels == g)
            if mask.sum() > 0:
                self.epoch_loss_sum[g] += losses_per_sample[mask].sum(dim=0)
                self.epoch_group_count[g] += mask.sum().float()

    @torch.no_grad()
    def epoch_update(self):
        """
        FAMO meta-update at epoch end: increase weight for slow-decreasing losses.

        Uses epoch-level average losses (not noisy batch-level). Called once per epoch.
        """
        self.epoch_count += 1

        # Compute epoch mean losses per group
        safe_count = self.epoch_group_count.clamp(min=1).unsqueeze(-1)  # (G, 1)
        curr_losses = self.epoch_loss_sum / safe_count  # (G, K)

        # Reset accumulation
        self.epoch_loss_sum.zero_()
        self.epoch_group_count.zero_()

        if self.epoch_count <= 1:
            # First epoch: just record losses, no update
            self.prev_losses.copy_(curr_losses)
            return

        # Loss decrease rate: positive = loss decreased
        decrease_rate = self.prev_losses - curr_losses  # (G, K)

        # Normalize by loss scale (relative decrease rate)
        # This makes comparison across losses with different scales fair
        scale = self.prev_losses.clamp(min=1e-6)
        relative_decrease = decrease_rate / scale  # (G, K) fractional decrease

        # Center: subtract per-group mean
        mean_decrease = relative_decrease.mean(dim=-1, keepdim=True)  # (G, 1)
        centered = relative_decrease - mean_decrease  # (G, K)

        # Update logits: decrease logit for fast-decreasing losses
        # increase logit for slow-decreasing losses (need more attention)
        self.logits.data -= self.w_lr * centered

        # Mild weight decay to prevent divergence (per-epoch, not per-step)
        self.logits.data *= (1.0 - self.gamma)

        # Update previous losses
        self.prev_losses.copy_(curr_losses)


# ============================================================
# AdaSelect Weighter (v15 — Temperature-Annealed Loss Gates)
# ============================================================
class AdaSelectWeighter(nn.Module):
    """
    AdaSelect: Accuracy-Driven Adaptive Loss Selection.

    14 distillation losses get learnable binary gates (sigmoid with temperature
    annealing). 3 classification losses are always active (ungated).

    Temperature anneals from init_temp (soft, exploratory) to final_temp
    (hard, near-binary), enabling the model to discover and prune harmful losses.

    Weight computation:
      gate_probs = sigmoid(gate_logits / τ)
      dist_w = (1 - cls_ratio) * gate_probs / sum(gate_probs)
      cls_w  = cls_ratio / 3
      weights = cat([dist_w, cls_w * ones(3)])

    Args:
        num_dist_losses: number of distillation losses (14)
        num_cls_losses: number of classification losses (3)
        cls_ratio: fraction of total budget for classification (0.7)
        init_temp: initial gate temperature (1.0)
        final_temp: final gate temperature (0.1)
        sparsity_lambda: L1 penalty coefficient on gate probs
    """

    def __init__(self, num_dist_losses=14, num_cls_losses=3,
                 cls_ratio=0.7, init_temp=1.0, final_temp=0.1,
                 sparsity_lambda=0.01):
        super().__init__()
        self.num_dist = num_dist_losses
        self.num_cls = num_cls_losses
        self.num_total = num_dist_losses + num_cls_losses
        self.cls_ratio = cls_ratio
        self.init_temp = init_temp
        self.final_temp = final_temp
        self.sparsity_lambda = sparsity_lambda

        # Learnable gate logits for distillation losses (shared across groups)
        self.gate_logits = nn.Parameter(torch.zeros(num_dist_losses))

        # Temperature buffer (annealed externally)
        self.register_buffer('temperature', torch.tensor(init_temp))

    def set_temperature(self, progress):
        """
        Set temperature based on training progress.

        Args:
            progress: float in [0, 1], where 0 = start, 1 = end
        """
        tau = self.init_temp + (self.final_temp - self.init_temp) * progress
        self.temperature.fill_(tau)

    def get_weights(self):
        """
        Compute loss weights with gated distillation losses.

        Returns:
            weights: (num_total,) loss weights
            sparsity_loss: scalar L1 penalty on gate probabilities
        """
        gate_probs = torch.sigmoid(self.gate_logits / self.temperature)  # (14,)

        # Distillation weights: normalized gate probs scaled by (1 - cls_ratio)
        gate_sum = gate_probs.sum().clamp(min=1e-6)
        dist_w = (1.0 - self.cls_ratio) * gate_probs / gate_sum  # (14,)

        # Classification weights: equal per task, scaled by cls_ratio
        cls_w = torch.full((self.num_cls,), self.cls_ratio / self.num_cls,
                           device=gate_probs.device)  # (3,)

        weights = torch.cat([dist_w, cls_w])  # (17,)

        # L1 sparsity loss on gate probs (encourages pruning)
        sparsity_loss = self.sparsity_lambda * gate_probs.sum()

        return weights, sparsity_loss

    def get_gate_probs(self):
        """Get current gate probabilities. Returns: (14,) values in [0, 1]."""
        return torch.sigmoid(self.gate_logits / self.temperature)


# ============================================================
# ============================================================
# PILR: Phase-Interference Loss Router (v19 contribution)
# ============================================================
class PILRRouter(nn.Module):
    """
    Phase-Interference Loss Router (PILR).

    Quantum-inspired wave interference framework for dynamic loss weighting.
    Each of K losses is represented as a complex wave:
        Ψ_k(x) = A_k(x) * exp(i * φ_k(x))

    A learnable complex synergy matrix S ∈ C^{K×K} captures pairwise loss
    interactions. The interfered state is:
        I_k(x) = Σ_j S_{k,j} * Ψ_j(x)

    Final weights are Temperature-scaled Born probabilities:
        w_k(x) = |I_k(x)|^(2/τ) / Σ_m |I_m(x)|^(2/τ)
    where τ controls sharpness (τ→∞: uniform, τ=1: original Born)

    Constructive interference (aligned phases) → synergistic loss combinations
    Destructive interference (opposed phases) → conflict resolution

    Args:
        input_dim: dimension of input features (768 for BLIP)
        num_losses: number of candidate losses (K=14)
        hidden_dim: router MLP hidden dimension (128)
    """

    def __init__(self, input_dim, num_losses, hidden_dim=128, synergy_off_diag_std=0.1):
        super().__init__()
        self.num_losses = num_losses

        # Router: predicts amplitude (K) + phase (K) = 2K outputs
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_losses * 2),
        )
        # Zero-init last layer → A = softplus(0) ≈ 0.693, φ = tanh(0)*π = 0
        # → uniform weights at initialization (matches Zero-Init philosophy)
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

        # Complex Synergy Matrix S ∈ C^{K×K}
        # Initialize near identity: stable start, off-diagonal learns cross-loss synergies
        # v20: increased off-diagonal std (0.1 vs v19's 0.01) for stronger excitation
        S_real = torch.eye(num_losses) + synergy_off_diag_std * torch.randn(num_losses, num_losses)
        S_imag = synergy_off_diag_std * torch.randn(num_losses, num_losses)
        self.synergy = nn.Parameter(torch.complex(S_real, S_imag))

    def forward(self, z, tau=1.0, gamma=0.0, use_phase=True, use_synergy=True):
        """
        Args:
            z: (B, D) input features
            tau: temperature for Born probability (τ→∞: uniform, τ=1: original Born)
            gamma: BEC chemical potential threshold (γ>0: sparse, γ=0: no filtering)
            use_phase: if False, set φ=0 (amplitude-only routing)
            use_synergy: if False, use S=Identity (no cross-loss interaction)
        Returns:
            weights: (B, K) per-sample loss weights (sum to 1, real-valued)
            diagnostics: dict with amplitude, phase, interference magnitude
        """
        out = self.router(z)  # (B, 2K)

        # Amplitude: softplus ensures A > 0
        A = F.softplus(out[:, :self.num_losses])  # (B, K)

        # Phase: tanh * π bounds to [-π, π]
        if use_phase:
            phi = torch.tanh(out[:, self.num_losses:]) * math.pi  # (B, K)
        else:
            phi = torch.zeros_like(A)  # ablation: no phase

        # Complex wave: Ψ_k = A_k * exp(i * φ_k)
        psi = A * torch.exp(1j * phi)  # (B, K) complex

        # Interference: I_k = Σ_j S_{k,j} * Ψ_j
        if use_synergy:
            S = self.synergy  # (K, K) complex
        else:
            S = torch.eye(self.num_losses, dtype=torch.cfloat, device=z.device)

        # I = S @ Ψ^T : (K, K) @ (B, K, 1) → (B, K, 1) → (B, K)
        I = torch.matmul(S, psi.unsqueeze(-1)).squeeze(-1)  # (B, K)

        # Temperature-scaled Born probability: E_k = |I_k|^(2/τ)
        E = torch.abs(I) ** (2.0 / tau)  # (B, K) real

        # BEC: Chemical potential threshold
        if gamma > 0:
            mu = gamma * E.mean(dim=-1, keepdim=True)  # (B, 1)
            E_tilde = F.relu(E - mu)                    # (B, K)
            # Safety: all losses zeroed → uniform fallback
            dead_mask = (E_tilde.sum(dim=-1) < 1e-10)   # (B,)
            if dead_mask.any():
                E_tilde[dead_mask] = 1.0 / self.num_losses
        else:
            E_tilde = E

        weights = E_tilde / (E_tilde.sum(dim=-1, keepdim=True) + 1e-8)  # (B, K)

        # Sparsity diagnostics
        num_active = (E_tilde > 1e-10).float().sum(dim=-1)   # (B,)
        sparsity = 1.0 - num_active / self.num_losses          # (B,)

        diagnostics = {
            "amplitude": A.detach(),
            "phase": phi.detach(),
            "mag_scaled": E.detach(),
            "tau": tau,
            "gamma": gamma,
            "sparsity": sparsity.detach(),
            "num_active": num_active.detach(),
        }

        return weights, diagnostics


# ============================================================
# PWDR Router (v22 — Particle-Wave Dual Routing)
# ============================================================
class PWDRRouter(nn.Module):
    """
    Particle-Wave Dual Routing (PWDR) for per-sample loss weighting.

    Key insight: Decouple amplitude (from spatial/robust features) and phase
    (from frequency/detail features) into separate heads with separate inputs.

    - Amplitude A_k from z_spatial (frozen BLIP CLS token — robust semantics)
    - Phase φ_k from z_freq (adapter delta — learned high-frequency details)

    This breaks cosine-collapse for Age (wrinkles/texture need high-freq phase)
    and provides natural noise robustness (phase randomization under noise).

    Modes:
      "dual":    A from z_spatial, φ from z_freq (default, v22 proposal)
      "spatial": A and φ from z_spatial (ablation: separate heads, same input)
      "unified": A and φ from z_spatial (v21 PILRRouter equivalent, single concat)

    Args:
        input_dim: dimension of input features (768 for BLIP)
        num_losses: number of candidate losses (K=14)
        hidden_dim: router MLP hidden dimension (128)
        synergy_off_diag_std: std for off-diagonal synergy initialization
    """

    def __init__(self, input_dim=768, num_losses=14, hidden_dim=128, synergy_off_diag_std=0.1):
        super().__init__()
        self.num_losses = num_losses

        # Amplitude head: z_spatial → A_k (always from spatial)
        self.amp_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_losses),
        )
        # Zero-init last layer → softplus(0) ≈ 0.693 → near-uniform at init
        nn.init.zeros_(self.amp_head[-1].weight)
        nn.init.zeros_(self.amp_head[-1].bias)

        # Phase head: z_freq → φ_k (from freq in dual mode, spatial otherwise)
        self.phase_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_losses),
        )
        # Zero-init last layer → tanh(0)*π = 0 → aligned phases at init
        nn.init.zeros_(self.phase_head[-1].weight)
        nn.init.zeros_(self.phase_head[-1].bias)

        # Complex Synergy Matrix S ∈ C^{K×K}
        S_real = torch.eye(num_losses) + synergy_off_diag_std * torch.randn(num_losses, num_losses)
        S_imag = synergy_off_diag_std * torch.randn(num_losses, num_losses)
        self.synergy = nn.Parameter(torch.complex(S_real, S_imag))

    def forward(self, z_spatial, z_freq=None, tau=1.0, gamma=0.0,
                use_phase=True, use_synergy=True, mode="dual"):
        """
        Args:
            z_spatial: (B, D) spatial features (frozen BLIP CLS token)
            z_freq: (B, D) frequency features (adapter delta = z_out - z_base)
            tau: temperature for Born probability
            gamma: BEC chemical potential threshold
            use_phase: if False, set φ=0 (amplitude-only routing)
            use_synergy: if False, use S=Identity
            mode: "dual" | "spatial" | "unified"
        Returns:
            weights: (B, K) per-sample loss weights (sum to 1)
            diagnostics: dict
        """
        # Amplitude: always from z_spatial
        A = F.softplus(self.amp_head(z_spatial))  # (B, K)

        # Phase: depends on mode
        if use_phase:
            if mode == "dual" and z_freq is not None:
                phi = torch.tanh(self.phase_head(z_freq)) * math.pi  # (B, K)
            elif mode == "spatial":
                phi = torch.tanh(self.phase_head(z_spatial)) * math.pi
            else:  # "unified" or fallback
                phi = torch.tanh(self.phase_head(z_spatial)) * math.pi
        else:
            phi = torch.zeros_like(A)

        # Complex wave: Ψ_k = A_k * exp(i * φ_k)
        psi = A * torch.exp(1j * phi)  # (B, K) complex

        # Interference: I_k = Σ_j S_{k,j} * Ψ_j
        if use_synergy:
            S = self.synergy  # (K, K) complex
        else:
            S = torch.eye(self.num_losses, dtype=torch.cfloat, device=z_spatial.device)

        I = torch.matmul(S, psi.unsqueeze(-1)).squeeze(-1)  # (B, K)

        # Temperature-scaled Born probability: E_k = |I_k|^(2/τ)
        E = torch.abs(I) ** (2.0 / tau)  # (B, K) real

        # BEC: Chemical potential threshold
        if gamma > 0:
            mu = gamma * E.mean(dim=-1, keepdim=True)  # (B, 1)
            E_tilde = F.relu(E - mu)                    # (B, K)
            dead_mask = (E_tilde.sum(dim=-1) < 1e-10)   # (B,)
            if dead_mask.any():
                E_tilde[dead_mask] = 1.0 / self.num_losses
        else:
            E_tilde = E

        weights = E_tilde / (E_tilde.sum(dim=-1, keepdim=True) + 1e-8)  # (B, K)

        # Sparsity diagnostics
        num_active = (E_tilde > 1e-10).float().sum(dim=-1)
        sparsity = 1.0 - num_active / self.num_losses

        diagnostics = {
            "amplitude": A.detach(),
            "phase": phi.detach(),
            "mag_scaled": E.detach(),
            "tau": tau,
            "gamma": gamma,
            "sparsity": sparsity.detach(),
            "num_active": num_active.detach(),
            "mode": mode,
        }

        return weights, diagnostics


# ============================================================
# QER Router (v23 — Quantum Entanglement Routing)
# ============================================================
class QERouter(nn.Module):
    """
    Quantum Entanglement Routing (QER) for per-sample loss weighting.

    Key innovations over PWDR (v22):
      1. Bilinear Entanglement Gates: cross-modal gating between spatial/freq
         z_amp   = z_spatial * sigmoid(gate_amp(z_freq))
         z_phase = z_freq * sigmoid(gate_phase(z_spatial))
      2. Sample-Adaptive Thermodynamics: tau(x) and gamma(x) predicted per-sample
         tau(x) in [1.0, 10.0], gamma(x) in [0.0, 0.8]

    Args:
        input_dim: dimension of input features (768 for BLIP)
        num_losses: number of candidate losses (K=14)
        hidden_dim: router MLP hidden dimension (128)
        synergy_off_diag_std: std for off-diagonal synergy initialization
    """

    def __init__(self, input_dim=768, num_losses=14, hidden_dim=128, synergy_off_diag_std=0.1):
        super().__init__()
        self.num_losses = num_losses

        # Entanglement gates (zero-init → identity at start)
        self.gate_amp = nn.Linear(input_dim, input_dim)
        nn.init.zeros_(self.gate_amp.weight)
        nn.init.zeros_(self.gate_amp.bias)

        self.gate_phase = nn.Linear(input_dim, input_dim)
        nn.init.zeros_(self.gate_phase.weight)
        nn.init.zeros_(self.gate_phase.bias)

        # Amplitude head: z_amp → A_k
        self.amp_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_losses),
        )
        nn.init.zeros_(self.amp_head[-1].weight)
        nn.init.zeros_(self.amp_head[-1].bias)

        # Phase head: z_phase → φ_k
        self.phase_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_losses),
        )
        nn.init.zeros_(self.phase_head[-1].weight)
        nn.init.zeros_(self.phase_head[-1].bias)

        # Sample-adaptive tau head: z_fused → tau(x) in [1.0, 10.0]
        self.tau_head = nn.Linear(input_dim, 1)
        nn.init.zeros_(self.tau_head.weight)
        nn.init.zeros_(self.tau_head.bias)

        # Sample-adaptive gamma head: z_fused → gamma(x) in [0.0, 0.8]
        self.gamma_head = nn.Linear(input_dim, 1)
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)

        # Complex Synergy Matrix S ∈ C^{K×K}
        S_real = torch.eye(num_losses) + synergy_off_diag_std * torch.randn(num_losses, num_losses)
        S_imag = synergy_off_diag_std * torch.randn(num_losses, num_losses)
        self.synergy = nn.Parameter(torch.complex(S_real, S_imag))

    def forward(self, z_spatial, z_freq=None, tau=None, gamma=None,
                use_entanglement=True, use_adaptive=True,
                use_phase=True, use_synergy=True):
        """
        Args:
            z_spatial: (B, D) spatial features (frozen BLIP CLS token)
            z_freq: (B, D) frequency features (adapter delta = z_out - z_base)
            tau: if provided, use fixed tau (scalar). Otherwise predict per-sample.
            gamma: if provided, use fixed gamma (scalar). Otherwise predict per-sample.
            use_entanglement: if True, apply bilinear entanglement gates
            use_adaptive: if True, predict tau/gamma per-sample (ignored if tau/gamma given)
            use_phase: if False, set φ=0 (amplitude-only routing)
            use_synergy: if False, use S=Identity
        Returns:
            weights: (B, K) per-sample loss weights (sum to 1)
            diagnostics: dict
        """
        if z_freq is None:
            z_freq = z_spatial

        # Entanglement gates
        if use_entanglement:
            z_amp = z_spatial * torch.sigmoid(self.gate_amp(z_freq))      # (B, D)
            z_phase = z_freq * torch.sigmoid(self.gate_phase(z_spatial))  # (B, D)
        else:
            z_amp = z_spatial
            z_phase = z_freq

        # Amplitude
        A = F.softplus(self.amp_head(z_amp))  # (B, K)

        # Phase
        if use_phase:
            phi = torch.tanh(self.phase_head(z_phase)) * math.pi  # (B, K)
        else:
            phi = torch.zeros_like(A)

        # Complex wave: Ψ_k = A_k * exp(i * φ_k)
        psi = A * torch.exp(1j * phi)  # (B, K) complex

        # Interference: I_k = Σ_j S_{k,j} * Ψ_j
        if use_synergy:
            S = self.synergy  # (K, K) complex
        else:
            S = torch.eye(self.num_losses, dtype=torch.cfloat, device=z_spatial.device)

        I = torch.matmul(S, psi.unsqueeze(-1)).squeeze(-1)  # (B, K)

        # Sample-adaptive thermodynamics
        z_fused = z_amp + z_phase  # (B, D)

        if tau is not None:
            # Fixed tau (scalar or tensor)
            tau_val = tau
            tau_per_sample = torch.full((z_spatial.size(0), 1), float(tau),
                                        device=z_spatial.device)
        elif use_adaptive:
            # Predicted per-sample: tau(x) in [1.0, 10.0]
            tau_per_sample = 1.0 + 9.0 * torch.sigmoid(self.tau_head(z_fused))  # (B, 1)
            tau_val = tau_per_sample
        else:
            tau_val = 3.0
            tau_per_sample = torch.full((z_spatial.size(0), 1), 3.0,
                                        device=z_spatial.device)

        if gamma is not None:
            gamma_val = gamma
            gamma_per_sample = torch.full((z_spatial.size(0), 1), float(gamma),
                                          device=z_spatial.device)
        elif use_adaptive:
            # Predicted per-sample: gamma(x) in [0.0, 0.8]
            gamma_per_sample = 0.8 * torch.sigmoid(self.gamma_head(z_fused))  # (B, 1)
            gamma_val = gamma_per_sample
        else:
            gamma_val = 0.5
            gamma_per_sample = torch.full((z_spatial.size(0), 1), 0.5,
                                          device=z_spatial.device)

        # Temperature-scaled Born probability: E_k = |I_k|^(2/τ)
        # tau_val can be scalar or (B, 1)
        if isinstance(tau_val, (int, float)):
            E = torch.abs(I) ** (2.0 / tau_val)  # (B, K) real
        else:
            E = torch.abs(I) ** (2.0 / tau_val)  # (B, K) with broadcasting

        # BEC: Chemical potential threshold (per-sample gamma)
        if isinstance(gamma_val, (int, float)):
            if gamma_val > 0:
                mu = gamma_val * E.mean(dim=-1, keepdim=True)  # (B, 1)
                E_tilde = F.relu(E - mu)
                dead_mask = (E_tilde.sum(dim=-1) < 1e-10)
                if dead_mask.any():
                    E_tilde[dead_mask] = 1.0 / self.num_losses
            else:
                E_tilde = E
        else:
            # Per-sample gamma: gamma_per_sample is (B, 1)
            mu = gamma_per_sample * E.mean(dim=-1, keepdim=True)  # (B, 1)
            E_tilde = F.relu(E - mu)
            dead_mask = (E_tilde.sum(dim=-1) < 1e-10)
            if dead_mask.any():
                E_tilde[dead_mask] = 1.0 / self.num_losses

        weights = E_tilde / (E_tilde.sum(dim=-1, keepdim=True) + 1e-8)  # (B, K)

        # Sparsity diagnostics
        num_active = (E_tilde > 1e-10).float().sum(dim=-1)
        sparsity = 1.0 - num_active / self.num_losses

        diagnostics = {
            "amplitude": A.detach(),
            "phase": phi.detach(),
            "mag_scaled": E.detach(),
            "tau": tau_per_sample.detach(),
            "gamma": gamma_per_sample.detach(),
            "sparsity": sparsity.detach(),
            "num_active": num_active.detach(),
        }

        return weights, diagnostics


class AQRouter(nn.Module):
    """
    Anisotropic Quantum Routing (AQR) for per-sample loss weighting.

    Key innovations over QERouter (v23):
      1. Residual Entanglement: z_amp = z_spatial + η_A * (z_spatial ⊙ σ(gate_amp(z_freq)))
         Starts as identity (η=0), learns safe entanglement amount.
      2. Anisotropic Thermodynamics: Per-loss τ_k, γ_k ∈ R^K
         Different losses at different temperatures.

    Args:
        input_dim: dimension of input features (768 for BLIP)
        num_losses: number of candidate losses (K=14)
        hidden_dim: router MLP hidden dimension (128)
        synergy_off_diag_std: std for off-diagonal synergy initialization
        anisotropic: if True, tau/gamma are per-loss (B,K); if False, scalar (B,1)
    """

    def __init__(self, input_dim=768, num_losses=14, hidden_dim=128,
                 synergy_off_diag_std=0.1, anisotropic=True):
        super().__init__()
        self.num_losses = num_losses
        self.anisotropic = anisotropic

        # Residual entanglement scalars (init 0 → identity at start, like v22)
        self.eta_A = nn.Parameter(torch.tensor(0.0))
        self.eta_phi = nn.Parameter(torch.tensor(0.0))

        # Entanglement gates (zero-init → identity at start)
        self.gate_amp = nn.Linear(input_dim, input_dim)
        nn.init.zeros_(self.gate_amp.weight)
        nn.init.zeros_(self.gate_amp.bias)

        self.gate_phase = nn.Linear(input_dim, input_dim)
        nn.init.zeros_(self.gate_phase.weight)
        nn.init.zeros_(self.gate_phase.bias)

        # Amplitude head: z_amp → A_k
        self.amp_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_losses),
        )
        nn.init.zeros_(self.amp_head[-1].weight)
        nn.init.zeros_(self.amp_head[-1].bias)

        # Phase head: z_phase → φ_k
        self.phase_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_losses),
        )
        nn.init.zeros_(self.phase_head[-1].weight)
        nn.init.zeros_(self.phase_head[-1].bias)

        # Tau/Gamma head output dim: K for anisotropic, 1 for isotropic
        tau_out = num_losses if anisotropic else 1
        gamma_out = num_losses if anisotropic else 1

        # Sample-adaptive tau head: z_fused → tau(x) in [1.0, 10.0]
        self.tau_head = nn.Linear(input_dim, tau_out)
        nn.init.zeros_(self.tau_head.weight)
        nn.init.zeros_(self.tau_head.bias)

        # Sample-adaptive gamma head: z_fused → gamma(x) in [0.0, 0.8]
        self.gamma_head = nn.Linear(input_dim, gamma_out)
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)

        # Complex Synergy Matrix S ∈ C^{K×K}
        S_real = torch.eye(num_losses) + synergy_off_diag_std * torch.randn(num_losses, num_losses)
        S_imag = synergy_off_diag_std * torch.randn(num_losses, num_losses)
        self.synergy = nn.Parameter(torch.complex(S_real, S_imag))

    def forward(self, z_spatial, z_freq=None, tau=None, gamma=None,
                use_adaptive=True, use_phase=True, use_synergy=True):
        """
        Args:
            z_spatial: (B, D) spatial features (frozen BLIP CLS token)
            z_freq: (B, D) frequency features (adapter delta = z_out - z_base)
            tau: if provided, use fixed tau (scalar). Otherwise predict per-sample.
            gamma: if provided, use fixed gamma (scalar). Otherwise predict per-sample.
            use_adaptive: if True, predict tau/gamma per-sample (ignored if tau/gamma given)
            use_phase: if False, set φ=0 (amplitude-only routing)
            use_synergy: if False, use S=Identity
        Returns:
            weights: (B, K) per-sample loss weights (sum to 1)
            diagnostics: dict
        """
        if z_freq is None:
            z_freq = z_spatial

        # Residual Entanglement (key change from QERouter):
        # η=0 at init → z_amp = z_spatial (pure v22 behavior)
        # η learns how much entanglement is safe
        z_amp = z_spatial + self.eta_A * (z_spatial * torch.sigmoid(self.gate_amp(z_freq)))      # (B, D)
        z_phase = z_freq + self.eta_phi * (z_freq * torch.sigmoid(self.gate_phase(z_spatial)))   # (B, D)

        # Amplitude
        A = F.softplus(self.amp_head(z_amp))  # (B, K)

        # Phase
        if use_phase:
            phi = torch.tanh(self.phase_head(z_phase)) * math.pi  # (B, K)
        else:
            phi = torch.zeros_like(A)

        # Complex wave: Ψ_k = A_k * exp(i * φ_k)
        psi = A * torch.exp(1j * phi)  # (B, K) complex

        # Interference: I_k = Σ_j S_{k,j} * Ψ_j
        if use_synergy:
            S = self.synergy  # (K, K) complex
        else:
            S = torch.eye(self.num_losses, dtype=torch.cfloat, device=z_spatial.device)

        I = torch.matmul(S, psi.unsqueeze(-1)).squeeze(-1)  # (B, K)

        # Sample-adaptive thermodynamics
        z_fused = z_amp + z_phase  # (B, D)

        if tau is not None:
            # Fixed tau (scalar)
            tau_per_sample = torch.full((z_spatial.size(0), 1), float(tau),
                                        device=z_spatial.device)
        elif use_adaptive:
            # Predicted per-sample: tau(x) in [1.0, 10.0]
            # Shape: (B, K) for anisotropic, (B, 1) for isotropic
            tau_per_sample = 1.0 + 9.0 * torch.sigmoid(self.tau_head(z_fused))
        else:
            tau_per_sample = torch.full((z_spatial.size(0), 1), 3.0,
                                        device=z_spatial.device)

        if gamma is not None:
            gamma_per_sample = torch.full((z_spatial.size(0), 1), float(gamma),
                                          device=z_spatial.device)
        elif use_adaptive:
            # Predicted per-sample: gamma(x) in [0.0, 0.8]
            # Shape: (B, K) for anisotropic, (B, 1) for isotropic
            gamma_per_sample = 0.8 * torch.sigmoid(self.gamma_head(z_fused))
        else:
            gamma_per_sample = torch.full((z_spatial.size(0), 1), 0.5,
                                          device=z_spatial.device)

        # Temperature-scaled Born probability: E_k = |I_k|^(2/τ_k)
        # tau_per_sample: (B, K) or (B, 1) — broadcasts correctly with (B, K)
        E = torch.abs(I) ** (2.0 / tau_per_sample)  # (B, K) real

        # BEC: Chemical potential threshold
        # gamma_per_sample: (B, K) or (B, 1) — broadcasts correctly
        mu = gamma_per_sample * E.mean(dim=-1, keepdim=True)  # (B, 1)
        E_tilde = F.relu(E - mu)
        dead_mask = (E_tilde.sum(dim=-1) < 1e-10)
        if dead_mask.any():
            E_tilde[dead_mask] = 1.0 / self.num_losses

        weights = E_tilde / (E_tilde.sum(dim=-1, keepdim=True) + 1e-8)  # (B, K)

        # Sparsity diagnostics
        num_active = (E_tilde > 1e-10).float().sum(dim=-1)
        sparsity = 1.0 - num_active / self.num_losses

        diagnostics = {
            "amplitude": A.detach(),
            "phase": phi.detach(),
            "mag_scaled": E.detach(),
            "tau": tau_per_sample.detach(),
            "gamma": gamma_per_sample.detach(),
            "sparsity": sparsity.detach(),
            "num_active": num_active.detach(),
            "eta_A": self.eta_A.detach(),
            "eta_phi": self.eta_phi.detach(),
        }

        return weights, diagnostics


# Theory-Guided Loss Grouping (v25 — TGLG)
# ============================================================
class TGLGRouter(nn.Module):
    """
    Theory-Guided Loss Grouping (TGLG) Router for per-sample loss weighting.

    Key innovation over AQR (v24):
      Groups 14 distillation losses into 3 theory-motivated groups,
      sharing τ and γ within each group to reduce over-parameterization.

    Groups:
      0 (Global Semantic):     Cosine, NT-Xent, Barlow, VICReg
      1 (Local Magnitude):     MSE, SmoothL1, KL_Div, L1
      2 (High-Freq Structural): FFL, SSIM, MultiSim, Triplet, ArcFace, Center

    Args:
        input_dim: dimension of input features (768 for BLIP)
        num_losses: number of candidate losses (K=14)
        num_groups: number of theory-guided groups (G=3)
        hidden_dim: router MLP hidden dimension (128)
        synergy_off_diag_std: std for off-diagonal synergy initialization
    """

    # Loss index → group mapping (constant)
    # MSE(0)→1, Cosine(1)→0, SmoothL1(2)→1, KL_Div(3)→1, FFL(4)→2,
    # L1(5)→1, SSIM(6)→2, NT-Xent(7)→0, MultiSim(8)→2, Triplet(9)→2,
    # ArcFace(10)→2, Center(11)→2, Barlow(12)→0, VICReg(13)→0
    LOSS_TO_GROUP = [1, 0, 1, 1, 2, 1, 2, 0, 2, 2, 2, 2, 0, 0]

    GROUP_NAMES = ["Global Semantic", "Local Magnitude", "High-Freq Structural"]

    def __init__(self, input_dim=768, num_losses=14, num_groups=3, hidden_dim=128,
                 synergy_off_diag_std=0.1):
        super().__init__()
        self.num_losses = num_losses
        self.num_groups = num_groups

        # Register loss→group mapping as buffer (not a parameter)
        self.register_buffer(
            "loss_to_group",
            torch.tensor(self.LOSS_TO_GROUP, dtype=torch.long),
        )

        # Residual entanglement scalars (init 0 → identity at start)
        self.eta_A = nn.Parameter(torch.tensor(0.0))
        self.eta_phi = nn.Parameter(torch.tensor(0.0))

        # Entanglement gates (zero-init → identity at start)
        self.gate_amp = nn.Linear(input_dim, input_dim)
        nn.init.zeros_(self.gate_amp.weight)
        nn.init.zeros_(self.gate_amp.bias)

        self.gate_phase = nn.Linear(input_dim, input_dim)
        nn.init.zeros_(self.gate_phase.weight)
        nn.init.zeros_(self.gate_phase.bias)

        # Amplitude head: z_amp → A_k (per-loss)
        self.amp_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_losses),
        )
        nn.init.zeros_(self.amp_head[-1].weight)
        nn.init.zeros_(self.amp_head[-1].bias)

        # Phase head: z_phase → φ_k (per-loss)
        self.phase_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_losses),
        )
        nn.init.zeros_(self.phase_head[-1].weight)
        nn.init.zeros_(self.phase_head[-1].bias)

        # Group-level tau head: z_fused → τ_G ∈ R^3 (broadcast to K=14)
        self.tau_head = nn.Linear(input_dim, num_groups)
        nn.init.zeros_(self.tau_head.weight)
        nn.init.zeros_(self.tau_head.bias)

        # Group-level gamma head: z_fused → γ_G ∈ R^3 (broadcast to K=14)
        self.gamma_head = nn.Linear(input_dim, num_groups)
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)

        # Complex Synergy Matrix S ∈ C^{K×K}
        S_real = torch.eye(num_losses) + synergy_off_diag_std * torch.randn(num_losses, num_losses)
        S_imag = synergy_off_diag_std * torch.randn(num_losses, num_losses)
        self.synergy = nn.Parameter(torch.complex(S_real, S_imag))

    def forward(self, z_spatial, z_freq=None, tau=None, gamma=None,
                use_adaptive=True, use_phase=True, use_synergy=True):
        """
        Args:
            z_spatial: (B, D) spatial features (frozen BLIP CLS token)
            z_freq: (B, D) frequency features (adapter delta = z_out - z_base)
            tau: if provided, use fixed tau (scalar). Otherwise predict per-group.
            gamma: if provided, use fixed gamma (scalar). Otherwise predict per-group.
            use_adaptive: if True, predict tau/gamma per-group (ignored if tau/gamma given)
            use_phase: if False, set φ=0 (amplitude-only routing)
            use_synergy: if False, use S=Identity
        Returns:
            weights: (B, K) per-sample loss weights (sum to 1)
            diagnostics: dict
        """
        if z_freq is None:
            z_freq = z_spatial

        # Residual Entanglement
        z_amp = z_spatial + self.eta_A * (z_spatial * torch.sigmoid(self.gate_amp(z_freq)))
        z_phase = z_freq + self.eta_phi * (z_freq * torch.sigmoid(self.gate_phase(z_spatial)))

        # Amplitude
        A = F.softplus(self.amp_head(z_amp))  # (B, K)

        # Phase
        if use_phase:
            phi = torch.tanh(self.phase_head(z_phase)) * math.pi  # (B, K)
        else:
            phi = torch.zeros_like(A)

        # Complex wave: Ψ_k = A_k * exp(i * φ_k)
        psi = A * torch.exp(1j * phi)  # (B, K) complex

        # Interference
        if use_synergy:
            S = self.synergy
        else:
            S = torch.eye(self.num_losses, dtype=torch.cfloat, device=z_spatial.device)

        I = torch.matmul(S, psi.unsqueeze(-1)).squeeze(-1)  # (B, K)

        # Group-level thermodynamics
        z_fused = z_amp + z_phase  # (B, D)

        if tau is not None:
            tau_per_sample = torch.full((z_spatial.size(0), 1), float(tau),
                                        device=z_spatial.device)
        elif use_adaptive:
            # Predict per-group: τ_G ∈ [1.0, 10.0], shape (B, G)
            tau_group = 1.0 + 9.0 * torch.sigmoid(self.tau_head(z_fused))  # (B, 3)
            # Broadcast to per-loss: (B, K) via group mapping
            tau_per_sample = tau_group[:, self.loss_to_group]  # (B, 14)
        else:
            tau_per_sample = torch.full((z_spatial.size(0), 1), 3.0,
                                        device=z_spatial.device)

        if gamma is not None:
            gamma_per_sample = torch.full((z_spatial.size(0), 1), float(gamma),
                                          device=z_spatial.device)
        elif use_adaptive:
            # Predict per-group: γ_G ∈ [0.0, 0.8], shape (B, G)
            gamma_group = 0.8 * torch.sigmoid(self.gamma_head(z_fused))  # (B, 3)
            # Broadcast to per-loss: (B, K) via group mapping
            gamma_per_sample = gamma_group[:, self.loss_to_group]  # (B, 14)
        else:
            gamma_per_sample = torch.full((z_spatial.size(0), 1), 0.5,
                                          device=z_spatial.device)

        # Temperature-scaled Born probability: E_k = |I_k|^(2/τ_k)
        E = torch.abs(I) ** (2.0 / tau_per_sample)  # (B, K)

        # BEC: Chemical potential threshold
        mu = gamma_per_sample * E.mean(dim=-1, keepdim=True)
        E_tilde = F.relu(E - mu)
        dead_mask = (E_tilde.sum(dim=-1) < 1e-10)
        if dead_mask.any():
            E_tilde[dead_mask] = 1.0 / self.num_losses

        weights = E_tilde / (E_tilde.sum(dim=-1, keepdim=True) + 1e-8)  # (B, K)

        # Sparsity diagnostics
        num_active = (E_tilde > 1e-10).float().sum(dim=-1)
        sparsity = 1.0 - num_active / self.num_losses

        # Build diagnostics — include group-level τ/γ
        diagnostics = {
            "amplitude": A.detach(),
            "phase": phi.detach(),
            "mag_scaled": E.detach(),
            "tau": tau_per_sample.detach(),
            "gamma": gamma_per_sample.detach(),
            "sparsity": sparsity.detach(),
            "num_active": num_active.detach(),
            "eta_A": self.eta_A.detach(),
            "eta_phi": self.eta_phi.detach(),
        }
        # Add group-level diagnostics when adaptive
        if use_adaptive and tau is None:
            diagnostics["tau_group"] = tau_group.detach()   # (B, 3)
            diagnostics["gamma_group"] = gamma_group.detach()  # (B, 3)

        return weights, diagnostics


class TSRouter(nn.Module):
    """
    Thermodynamic Superposition Routing (TSR) for per-sample loss weighting.

    Key innovation over TGLG (v25):
      Instead of hard-coding loss→group mapping, learns a soft superposition
      matrix P_raw ∈ R^{14×3}. Each loss is a weighted mixture of 3 basis states.
      Smart-initialized to match v25's hard grouping (softmax(5,0,0) ≈ 0.993),
      then learns elastic grouping during training.

    Basis states (same as TGLG groups, used for initialization only):
      0 (Global Semantic):     Cosine, NT-Xent, Barlow, VICReg
      1 (Local Magnitude):     MSE, SmoothL1, KL_Div, L1
      2 (High-Freq Structural): FFL, SSIM, MultiSim, Triplet, ArcFace, Center

    Args:
        input_dim: dimension of input features (768 for BLIP)
        num_losses: number of candidate losses (K=14)
        num_bases: number of basis states (G=3)
        hidden_dim: router MLP hidden dimension (128)
        synergy_off_diag_std: std for off-diagonal synergy initialization
    """

    # Used only for smart initialization of P_raw
    INIT_LOSS_TO_GROUP = [1, 0, 1, 1, 2, 1, 2, 0, 2, 2, 2, 2, 0, 0]

    BASIS_NAMES = ["Global Semantic", "Local Magnitude", "High-Freq Structural"]

    def __init__(self, input_dim=768, num_losses=14, num_bases=3, hidden_dim=128,
                 synergy_off_diag_std=0.1):
        super().__init__()
        self.num_losses = num_losses
        self.num_bases = num_bases

        # Learnable superposition matrix: softmax(P_raw, dim=1) → M ∈ [0,1]^{14×3}
        # Smart init: P_raw[k, group_k] = 5.0, rest = 0.0
        # → softmax(5, 0, 0) ≈ (0.993, 0.003, 0.003) → starts like v25
        P_init = torch.zeros(num_losses, num_bases)
        for k, g in enumerate(self.INIT_LOSS_TO_GROUP):
            P_init[k, g] = 5.0
        self.P_raw = nn.Parameter(P_init)

        # Residual entanglement scalars (init 0 → identity at start)
        self.eta_A = nn.Parameter(torch.tensor(0.0))
        self.eta_phi = nn.Parameter(torch.tensor(0.0))

        # Entanglement gates (zero-init → identity at start)
        self.gate_amp = nn.Linear(input_dim, input_dim)
        nn.init.zeros_(self.gate_amp.weight)
        nn.init.zeros_(self.gate_amp.bias)

        self.gate_phase = nn.Linear(input_dim, input_dim)
        nn.init.zeros_(self.gate_phase.weight)
        nn.init.zeros_(self.gate_phase.bias)

        # Amplitude head: z_amp → A_k (per-loss)
        self.amp_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_losses),
        )
        nn.init.zeros_(self.amp_head[-1].weight)
        nn.init.zeros_(self.amp_head[-1].bias)

        # Phase head: z_phase → φ_k (per-loss)
        self.phase_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_losses),
        )
        nn.init.zeros_(self.phase_head[-1].weight)
        nn.init.zeros_(self.phase_head[-1].bias)

        # Basis-level tau head: z_fused → τ_basis ∈ R^3
        self.tau_head = nn.Linear(input_dim, num_bases)
        nn.init.zeros_(self.tau_head.weight)
        nn.init.zeros_(self.tau_head.bias)

        # Basis-level gamma head: z_fused → γ_basis ∈ R^3
        self.gamma_head = nn.Linear(input_dim, num_bases)
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)

        # Complex Synergy Matrix S ∈ C^{K×K}
        S_real = torch.eye(num_losses) + synergy_off_diag_std * torch.randn(num_losses, num_losses)
        S_imag = synergy_off_diag_std * torch.randn(num_losses, num_losses)
        self.synergy = nn.Parameter(torch.complex(S_real, S_imag))

    def forward(self, z_spatial, z_freq=None, tau=None, gamma=None,
                use_adaptive=True, use_phase=True, use_synergy=True):
        """
        Args:
            z_spatial: (B, D) spatial features (frozen BLIP CLS token)
            z_freq: (B, D) frequency features (adapter delta = z_out - z_base)
            tau: if provided, use fixed tau (scalar). Otherwise predict per-basis.
            gamma: if provided, use fixed gamma (scalar). Otherwise predict per-basis.
            use_adaptive: if True, predict tau/gamma per-basis (ignored if tau/gamma given)
            use_phase: if False, set φ=0 (amplitude-only routing)
            use_synergy: if False, use S=Identity
        Returns:
            weights: (B, K) per-sample loss weights (sum to 1)
            diagnostics: dict
        """
        if z_freq is None:
            z_freq = z_spatial

        # Superposition matrix: soft assignment of losses to bases
        M = F.softmax(self.P_raw, dim=1)  # (14, 3) each row sums to 1

        # Residual Entanglement
        z_amp = z_spatial + self.eta_A * (z_spatial * torch.sigmoid(self.gate_amp(z_freq)))
        z_phase = z_freq + self.eta_phi * (z_freq * torch.sigmoid(self.gate_phase(z_spatial)))

        # Amplitude
        A = F.softplus(self.amp_head(z_amp))  # (B, K)

        # Phase
        if use_phase:
            phi = torch.tanh(self.phase_head(z_phase)) * math.pi  # (B, K)
        else:
            phi = torch.zeros_like(A)

        # Complex wave: Ψ_k = A_k * exp(i * φ_k)
        psi = A * torch.exp(1j * phi)  # (B, K) complex

        # Interference
        if use_synergy:
            S = self.synergy
        else:
            S = torch.eye(self.num_losses, dtype=torch.cfloat, device=z_spatial.device)

        I = torch.matmul(S, psi.unsqueeze(-1)).squeeze(-1)  # (B, K)

        # Basis-level thermodynamics mixed via superposition matrix
        z_fused = z_amp + z_phase  # (B, D)

        if tau is not None:
            tau_per_sample = torch.full((z_spatial.size(0), 1), float(tau),
                                        device=z_spatial.device)
        elif use_adaptive:
            # Predict per-basis: τ_basis ∈ [1.0, 10.0], shape (B, 3)
            tau_base = 1.0 + 9.0 * torch.sigmoid(self.tau_head(z_fused))  # (B, 3)
            # Mix to per-loss via superposition: (B,3) @ (3,14) → (B,14)
            tau_per_sample = torch.matmul(tau_base, M.T)  # (B, 14)
        else:
            tau_per_sample = torch.full((z_spatial.size(0), 1), 3.0,
                                        device=z_spatial.device)

        if gamma is not None:
            gamma_per_sample = torch.full((z_spatial.size(0), 1), float(gamma),
                                          device=z_spatial.device)
        elif use_adaptive:
            # Predict per-basis: γ_basis ∈ [0.0, 0.8], shape (B, 3)
            gamma_base = 0.8 * torch.sigmoid(self.gamma_head(z_fused))  # (B, 3)
            # Mix to per-loss via superposition: (B,3) @ (3,14) → (B,14)
            gamma_per_sample = torch.matmul(gamma_base, M.T)  # (B, 14)
        else:
            gamma_per_sample = torch.full((z_spatial.size(0), 1), 0.5,
                                          device=z_spatial.device)

        # Temperature-scaled Born probability: E_k = |I_k|^(2/τ_k)
        E = torch.abs(I) ** (2.0 / tau_per_sample)  # (B, K)

        # BEC: Chemical potential threshold
        mu = gamma_per_sample * E.mean(dim=-1, keepdim=True)
        E_tilde = F.relu(E - mu)
        dead_mask = (E_tilde.sum(dim=-1) < 1e-10)
        if dead_mask.any():
            E_tilde[dead_mask] = 1.0 / self.num_losses

        weights = E_tilde / (E_tilde.sum(dim=-1, keepdim=True) + 1e-8)  # (B, K)

        # Sparsity diagnostics
        num_active = (E_tilde > 1e-10).float().sum(dim=-1)
        sparsity = 1.0 - num_active / self.num_losses

        diagnostics = {
            "amplitude": A.detach(),
            "phase": phi.detach(),
            "mag_scaled": E.detach(),
            "tau": tau_per_sample.detach(),
            "gamma": gamma_per_sample.detach(),
            "sparsity": sparsity.detach(),
            "num_active": num_active.detach(),
            "eta_A": self.eta_A.detach(),
            "eta_phi": self.eta_phi.detach(),
            "M": M.detach(),  # (14, 3) superposition matrix
        }
        # Add basis-level diagnostics when adaptive
        if use_adaptive and tau is None:
            diagnostics["tau_base"] = tau_base.detach()    # (B, 3)
            diagnostics["gamma_base"] = gamma_base.detach()  # (B, 3)

        return weights, diagnostics


class NCCRouter(nn.Module):
    """
    Noise-Conditioned Curriculum Routing (NCCR) for per-sample loss weighting.

    Key innovation over TSR (v26):
      Explicitly observes noise severity s(x) via a noise_head, then gates
      temperature τ and chemical potential γ so that:
        - Clean (s≈0): sharp routing (τ∈[1,5], γ∈[0,0.8]) for task-specific SOTA
        - Noisy (s≈1): forced uniform (τ→10, γ→0) for noise robustness SOTA

    Uses hard groups from TGLG (v25) — v26 showed M matrix barely learned.
    The noise_head receives auxiliary supervision L_noise = BCE(s(x), target_s).

    Groups (same as TGLG):
      0 (Global Semantic):     Cosine, NT-Xent, Barlow, VICReg
      1 (Local Magnitude):     MSE, SmoothL1, KL_Div, L1
      2 (High-Freq Structural): FFL, SSIM, MultiSim, Triplet, ArcFace, Center

    Args:
        input_dim: dimension of input features (768 for BLIP)
        num_losses: number of candidate losses (K=14)
        num_groups: number of theory-guided groups (G=3)
        hidden_dim: router MLP hidden dimension (128)
        synergy_off_diag_std: std for off-diagonal synergy initialization
    """

    LOSS_TO_GROUP = [1, 0, 1, 1, 2, 1, 2, 0, 2, 2, 2, 2, 0, 0]
    GROUP_NAMES = ["Global Semantic", "Local Magnitude", "High-Freq Structural"]

    def __init__(self, input_dim=768, num_losses=14, num_groups=3, hidden_dim=128,
                 synergy_off_diag_std=0.1):
        super().__init__()
        self.num_losses = num_losses
        self.num_groups = num_groups

        # Register loss→group mapping as buffer
        self.register_buffer(
            "loss_to_group",
            torch.tensor(self.LOSS_TO_GROUP, dtype=torch.long),
        )

        # === Noise Observation Head ===
        # Input: [z_spatial, z_freq] concatenated (B, 2*input_dim)
        # Output: s(x) ∈ [0, 1] — noise severity scalar
        self.noise_head = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Residual entanglement scalars (init 0 → identity at start)
        self.eta_A = nn.Parameter(torch.tensor(0.0))
        self.eta_phi = nn.Parameter(torch.tensor(0.0))

        # Entanglement gates (zero-init → identity at start)
        self.gate_amp = nn.Linear(input_dim, input_dim)
        nn.init.zeros_(self.gate_amp.weight)
        nn.init.zeros_(self.gate_amp.bias)

        self.gate_phase = nn.Linear(input_dim, input_dim)
        nn.init.zeros_(self.gate_phase.weight)
        nn.init.zeros_(self.gate_phase.bias)

        # Amplitude head: z_amp → A_k (per-loss)
        self.amp_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_losses),
        )
        nn.init.zeros_(self.amp_head[-1].weight)
        nn.init.zeros_(self.amp_head[-1].bias)

        # Phase head: z_phase → φ_k (per-loss)
        self.phase_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_losses),
        )
        nn.init.zeros_(self.phase_head[-1].weight)
        nn.init.zeros_(self.phase_head[-1].bias)

        # Group-level tau head: z_fused → τ_raw ∈ R^3
        self.tau_head = nn.Linear(input_dim, num_groups)
        nn.init.zeros_(self.tau_head.weight)
        nn.init.zeros_(self.tau_head.bias)

        # Group-level gamma head: z_fused → γ_raw ∈ R^3
        self.gamma_head = nn.Linear(input_dim, num_groups)
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)

        # Complex Synergy Matrix S ∈ C^{K×K}
        S_real = torch.eye(num_losses) + synergy_off_diag_std * torch.randn(num_losses, num_losses)
        S_imag = synergy_off_diag_std * torch.randn(num_losses, num_losses)
        self.synergy = nn.Parameter(torch.complex(S_real, S_imag))

    def forward(self, z_spatial, z_freq=None, tau=None, gamma=None,
                use_adaptive=True, use_phase=True, use_synergy=True):
        """
        Args:
            z_spatial: (B, D) spatial features (frozen BLIP CLS token)
            z_freq: (B, D) frequency features (adapter delta = z_out - z_base)
            tau: if provided, use fixed tau (scalar). Otherwise noise-gated per-group.
            gamma: if provided, use fixed gamma (scalar). Otherwise noise-gated per-group.
            use_adaptive: if True, predict tau/gamma per-group (ignored if tau/gamma given)
            use_phase: if False, set φ=0 (amplitude-only routing)
            use_synergy: if False, use S=Identity
        Returns:
            weights: (B, K) per-sample loss weights (sum to 1)
            diagnostics: dict (includes noise_severity, tau_group, gamma_group)
        """
        if z_freq is None:
            z_freq = z_spatial

        # === Noise Observation ===
        noise_input = torch.cat([z_spatial, z_freq], dim=-1)  # (B, 2D)
        s = self.noise_head(noise_input)  # (B, 1) — noise severity ∈ [0, 1]

        # Residual Entanglement
        z_amp = z_spatial + self.eta_A * (z_spatial * torch.sigmoid(self.gate_amp(z_freq)))
        z_phase = z_freq + self.eta_phi * (z_freq * torch.sigmoid(self.gate_phase(z_spatial)))

        # Amplitude
        A = F.softplus(self.amp_head(z_amp))  # (B, K)

        # Phase
        if use_phase:
            phi = torch.tanh(self.phase_head(z_phase)) * math.pi  # (B, K)
        else:
            phi = torch.zeros_like(A)

        # Complex wave: Ψ_k = A_k * exp(i * φ_k)
        psi = A * torch.exp(1j * phi)  # (B, K) complex

        # Interference
        if use_synergy:
            S = self.synergy
        else:
            S = torch.eye(self.num_losses, dtype=torch.cfloat, device=z_spatial.device)

        I = torch.matmul(S, psi.unsqueeze(-1)).squeeze(-1)  # (B, K)

        # === Noise-Gated Group-Level Thermodynamics ===
        z_fused = z_amp + z_phase  # (B, D)

        if tau is not None:
            tau_per_sample = torch.full((z_spatial.size(0), 1), float(tau),
                                        device=z_spatial.device)
            tau_group = None
        elif use_adaptive:
            # Noise-gated τ: clean→[1,5] (sharp), noisy→10 (forced uniform)
            tau_raw = self.tau_head(z_fused)  # (B, 3)
            tau_group = (1.0 - s) * (1.0 + 4.0 * torch.sigmoid(tau_raw)) + s * 10.0  # (B, 3)
            tau_per_sample = tau_group[:, self.loss_to_group]  # (B, 14)
        else:
            tau_per_sample = torch.full((z_spatial.size(0), 1), 3.0,
                                        device=z_spatial.device)
            tau_group = None

        if gamma is not None:
            gamma_per_sample = torch.full((z_spatial.size(0), 1), float(gamma),
                                          device=z_spatial.device)
            gamma_group = None
        elif use_adaptive:
            # Noise-gated γ: clean→[0,0.8] (BEC sparsity active), noisy→0 (no filtering)
            gamma_raw = self.gamma_head(z_fused)  # (B, 3)
            gamma_group = (1.0 - s) * 0.8 * torch.sigmoid(gamma_raw)  # (B, 3)
            gamma_per_sample = gamma_group[:, self.loss_to_group]  # (B, 14)
        else:
            gamma_per_sample = torch.full((z_spatial.size(0), 1), 0.5,
                                          device=z_spatial.device)
            gamma_group = None

        # Temperature-scaled Born probability: E_k = |I_k|^(2/τ_k)
        E = torch.abs(I) ** (2.0 / tau_per_sample)  # (B, K)

        # BEC: Chemical potential threshold
        mu = gamma_per_sample * E.mean(dim=-1, keepdim=True)
        E_tilde = F.relu(E - mu)
        dead_mask = (E_tilde.sum(dim=-1) < 1e-10)
        if dead_mask.any():
            E_tilde[dead_mask] = 1.0 / self.num_losses

        weights = E_tilde / (E_tilde.sum(dim=-1, keepdim=True) + 1e-8)  # (B, K)

        # Sparsity diagnostics
        num_active = (E_tilde > 1e-10).float().sum(dim=-1)
        sparsity = 1.0 - num_active / self.num_losses

        diagnostics = {
            "amplitude": A.detach(),
            "phase": phi.detach(),
            "mag_scaled": E.detach(),
            "tau": tau_per_sample.detach(),
            "gamma": gamma_per_sample.detach(),
            "sparsity": sparsity.detach(),
            "num_active": num_active.detach(),
            "eta_A": self.eta_A.detach(),
            "eta_phi": self.eta_phi.detach(),
            "noise_severity": s,  # (B, 1) — NOT detached, needs grad for L_noise
        }
        if use_adaptive and tau is None and tau_group is not None:
            diagnostics["tau_group"] = tau_group.detach()    # (B, 3)
            diagnostics["gamma_group"] = gamma_group.detach()  # (B, 3)

        return weights, diagnostics


# Simplified Adapter (v13 — FFT/gate全削除)
# ============================================================
class SimplifiedAdapter(nn.Module):
    """
    MLP Adapter: 768 → 384 → 384 → 768 with residual connection.

    Replaces v12's SSD-Net (FFT branch + spatial branch + gate).
    ~590K params (2× v12's SSD-Net), all parameters are useful (no dead FFT branch).

    Args:
        input_dim: dimension of input embeddings (768 for BLIP)
        hidden_dim: dimension of hidden layers (384)
        zero_init: if True, zero-init the final Linear layer so adapter(x) ≈ 0 at start
                   → z_out = 0 + x = x (pure baseline at epoch 1)
    """

    def __init__(self, input_dim=768, hidden_dim=384, zero_init=False):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim),
        )

        if zero_init:
            # Zero-init the last Linear (index 5) so output starts at ~0
            nn.init.zeros_(self.adapter[5].weight)
            nn.init.zeros_(self.adapter[5].bias)

    def forward(self, x):
        """
        Args:
            x: (B, D) input embeddings
        Returns:
            z_out: (B, D) adapted embeddings with residual
        """
        return self.adapter(x) + x


# ============================================================
# Model v13: Frozen BLIP + SimplifiedAdapter + Classification Heads
# ============================================================
class GRoFAModelV13(nn.Module):
    """
    v13 Model: Frozen BLIP backbone + SimplifiedAdapter + Classification Heads.

    Key differences from v12 (GRoFAModel):
      - No SSD-Net (no FFT, no gate, no router MLP)
      - SimplifiedAdapter: pure MLP with residual (~590K params)
      - Classification heads: directly optimize linear probe accuracy
        (breaks teacher ceiling by optimizing the actual evaluation metric)
      - FAMO weighting is external (FAMOWeighter handles loss weights)

    Args:
        base_model: BLIP vision model (frozen)
        num_race: number of race classes (7 for FairFace)
        num_gender: number of gender classes (2)
        num_age: number of age classes (9 for FairFace)
        hidden_dim: adapter hidden dimension (384)
    """

    def __init__(self, base_model, num_race=7, num_gender=2, num_age=9, hidden_dim=384):
        super().__init__()
        self.base_model = base_model

        # Freeze backbone
        for p in self.base_model.parameters():
            p.requires_grad = False

        # Trainable adapter
        self.adapter = SimplifiedAdapter(768, hidden_dim)

        # Classification heads (key to breaking teacher ceiling)
        self.cls_race = nn.Linear(768, num_race)
        self.cls_gender = nn.Linear(768, num_gender)
        self.cls_age = nn.Linear(768, num_age)

    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) input images
        Returns:
            z_out: (B, 768) adapted embeddings
            z_base: (B, 768) frozen base embeddings (for distillation)
            logits_race: (B, num_race) race classification logits
            logits_gender: (B, num_gender) gender classification logits
            logits_age: (B, num_age) age classification logits
        """
        # Extract frozen features
        with torch.no_grad():
            z_base = self.base_model(pixel_values=x).last_hidden_state[:, 0, :]

        # Adapt
        z_out = self.adapter(z_base)

        # Classify
        logits_race = self.cls_race(z_out)
        logits_gender = self.cls_gender(z_out)
        logits_age = self.cls_age(z_out)

        return z_out, z_base, logits_race, logits_gender, logits_age


# ============================================================
# Noise Gate MLP (v17 — ParetoFair)
# ============================================================
class NoiseGateMLP(nn.Module):
    """
    Noise-Gated Adapter control: estimates noise level from frozen features
    and gates adapter application accordingly.

    clean → noise_score ≈ 0 → z_out ≈ z_base (bypass adapter)
    noisy → noise_score ≈ 1 → z_out = z_base + adapter(z_base)

    ~50K parameters.

    Args:
        input_dim: dimension of input embeddings (768 for BLIP)
        hidden_dim: hidden layer dimension (64)
    """

    def __init__(self, input_dim=768, hidden_dim=64, init_bias=None,
                 zero_init_weight=False):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        # v48: Explicit initialization for NoiseGate last layer
        # zero_init_weight=True + init_bias=-0.85 → ns = sigmoid(-0.85) ≈ 0.30
        # CONSTANT for all inputs (input-independent), preventing rapid drift
        if init_bias is not None:
            with torch.no_grad():
                self.gate[2].bias.fill_(init_bias)
        if zero_init_weight:
            with torch.no_grad():
                self.gate[2].weight.zero_()

    def forward(self, z_base):
        """
        Args:
            z_base: (B, 768) frozen backbone features
        Returns:
            noise_score: (B, 1) in [0, 1]
        """
        return self.gate(z_base)


# ============================================================
# Augmented Lagrangian Fairness (v17 — ParetoFair)
# ============================================================
class AugmentedLagrangianFairness(nn.Module):
    """
    Augmented Lagrangian penalty for multi-task fairness constraints.

    For each task t (race, gender, age):
      gap_t = max_g(acc_g) - min_g(acc_g)
      violation_t = max(gap_t - epsilon, 0)
      L_AL = sum_t [mu_t * violation_t + (rho/2) * violation_t^2]

    Dual variables mu_t are updated at each epoch end.

    Args:
        num_tasks: number of classification tasks (3: race, gender, age)
        epsilon: fairness gap tolerance
        rho_init: initial penalty coefficient
        rho_max: maximum penalty coefficient
        rho_mult: multiplicative increase for rho each epoch
    """

    def __init__(self, num_tasks=3, epsilon=0.05, rho_init=0.1,
                 rho_max=10.0, rho_mult=1.5):
        super().__init__()
        self.num_tasks = num_tasks
        self.rho_max = rho_max
        self.rho_mult = rho_mult

        # Support per-task epsilon (v28+): scalar or list/tuple
        if isinstance(epsilon, (list, tuple)):
            self.register_buffer('epsilon', torch.tensor(epsilon, dtype=torch.float32))
        else:
            self.register_buffer('epsilon', torch.tensor([epsilon] * num_tasks, dtype=torch.float32))

        self.register_buffer('mu', torch.zeros(num_tasks))
        self.register_buffer('rho', torch.tensor(rho_init))

    def compute_penalty(self, gaps):
        """
        Compute Augmented Lagrangian penalty from accuracy gaps.

        Args:
            gaps: (num_tasks,) accuracy gap per task (max_group - min_group)
        Returns:
            penalty: scalar AL penalty loss
        """
        violations = torch.clamp(gaps - self.epsilon, min=0.0)
        penalty = (self.mu * violations).sum() + \
                  0.5 * self.rho * (violations ** 2).sum()
        return penalty

    @torch.no_grad()
    def update_dual(self, gaps):
        """
        Update dual variables at epoch end.

        Args:
            gaps: (num_tasks,) accuracy gap per task
        """
        violations = torch.clamp(gaps - self.epsilon, min=0.0)
        self.mu.copy_(torch.clamp(self.mu + self.rho * violations, min=0.0))
        self.rho.copy_(torch.clamp(self.rho * self.rho_mult, max=self.rho_max))


# ============================================================
# Model v17: Frozen BLIP + Noise-Gated Adapter + Classification
# ============================================================
class GRoFAModelV17(nn.Module):
    """
    v17 ParetoFair Model: Frozen BLIP + Noise-Gated Adapter + Classification.

    Key innovations:
      - NoiseGateMLP: gates adapter application based on estimated noise level
      - SimplifiedAdapter with zero-init: baseline quality guaranteed at epoch 1
      - Classification heads for 3 tasks (race, gender, age)

    Forward returns 6 values: z_out, z_base, logits_r, logits_g, logits_a, noise_score

    Args:
        base_model: BLIP vision model (frozen)
        num_race: number of race classes (7 for FairFace)
        num_gender: number of gender classes (2)
        num_age: number of age classes (9 for FairFace)
        hidden_dim: adapter hidden dimension (384)
    """

    def __init__(self, base_model, num_race=7, num_gender=2, num_age=9, hidden_dim=384):
        super().__init__()
        self.base_model = base_model

        # Freeze backbone
        for p in self.base_model.parameters():
            p.requires_grad = False

        # Trainable adapter (zero-init applied externally or via flag)
        self.adapter = SimplifiedAdapter(768, hidden_dim, zero_init=True)

        # Noise gate
        self.noise_gate = NoiseGateMLP(768, hidden_dim=64)

        # Classification heads
        self.cls_race = nn.Linear(768, num_race)
        self.cls_gender = nn.Linear(768, num_gender)
        self.cls_age = nn.Linear(768, num_age)

    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) input images
        Returns:
            z_out: (B, 768) noise-gated adapted embeddings
            z_base: (B, 768) frozen base embeddings
            logits_race: (B, num_race)
            logits_gender: (B, num_gender)
            logits_age: (B, num_age)
            noise_score: (B, 1) estimated noise level
        """
        with torch.no_grad():
            z_base = self.base_model(pixel_values=x).last_hidden_state[:, 0, :]

        # Noise-gated adapter
        noise_score = self.noise_gate(z_base)          # (B, 1)
        z_adapted = self.adapter.adapter(z_base)       # adapter without residual
        z_out = z_base + noise_score * z_adapted       # gated residual

        # Classify
        logits_race = self.cls_race(z_out)
        logits_gender = self.cls_gender(z_out)
        logits_age = self.cls_age(z_out)

        return z_out, z_base, logits_race, logits_gender, logits_age, noise_score


# ============================================================
# TALRouter — v28 Task-Aligned Loss Router (primary contribution)
# ============================================================
class TALRouter(nn.Module):
    """
    Task-Aligned Loss Router (TALR) — v28 primary contribution.

    Per-sample distillation loss weighting via gradient-geometry alignment.
    Weight_k(x) = softmax( (alignment_k(x) + prior_k^(g)) / tau )

    alignment_k(x) = cosine(grad_k(x), grad_task(x))
      - grad_k: gradient of distillation loss k w.r.t. z_out
      - grad_task: gradient of classification losses w.r.t. z_out

    Parameters: G*K + 1 = 7*14 + 1 = 99 scalars.
    Compare: v26 TSRouter ~1.4M params, v27 NCCRouter ~1.4M params.
    """

    def __init__(self, num_losses=14, num_groups=7):
        super().__init__()
        self.num_losses = num_losses
        self.num_groups = num_groups

        # Per-group learnable prior bias (init 0 = uniform)
        self.b = nn.Parameter(torch.zeros(num_groups, num_losses))

        # Learnable temperature: tau = 1 + 9 * sigmoid(tau_raw)
        # tau_raw=0 -> sigmoid(0)=0.5 -> tau=5.5 (moderate)
        self.tau_raw = nn.Parameter(torch.tensor(0.0))

    @property
    def tau(self):
        return 1.0 + 9.0 * torch.sigmoid(self.tau_raw)

    def forward(self, alignment_scores, group_labels):
        """
        Args:
            alignment_scores: (B, K) cosine similarity in [-1, 1].
                              Computed EXTERNALLY via compute_alignment_scores().
                              Must be .detach()'ed — not part of backprop graph.
            group_labels: (B,) integer group labels 0..G-1
        Returns:
            weights: (B, K) per-sample loss weights summing to 1
            diagnostics: dict for logging
        """
        prior = self.b[group_labels]  # (B, K)
        logits = (alignment_scores + prior) / self.tau
        weights = F.softmax(logits, dim=-1)

        diagnostics = {
            "alignment": alignment_scores.detach(),
            "prior": prior.detach(),
            "weights": weights.detach(),
            "tau": self.tau.detach().item(),
        }
        return weights, diagnostics


# ============================================================
# Model v28 (GRoFA): Frozen BLIP + Noise-Modulated Adapter + Classification
# ============================================================
class GRoFAModelV28(nn.Module):
    """
    v28 Model: Frozen BLIP + Noise-Modulated SimplifiedAdapter + Classification Heads.

    Differences from GRoFAModelV13:
      - NoiseGateMLP gates the adapter residual:
        z_out = z_base + gate * adapter(z_base)
        gate = noise_score (invert_gate=True) or 1-noise_score (default)
      - invert_gate=True: adapter active on noisy, bypassed on clean (v57+)
      - invert_gate=False: adapter active on clean, bypassed on noisy (v28-v56)
      - Returns 6 values (same signature as GRoFAModelV17)

    Args:
        base_model: BLIP vision model (frozen)
        num_race: number of race classes (7 for FairFace)
        num_gender: number of gender classes (2)
        num_age: number of age classes (9 for FairFace)
        hidden_dim: adapter hidden dimension (384)
    """

    def __init__(self, base_model, num_race=7, num_gender=2, num_age=9,
                 hidden_dim=384, arf_floor=0.0, gate_ceiling=1.0,
                 detach_noise_score=False,
                 ng_init_bias=None, ng_zero_init_weight=False,
                 invert_gate=False):
        super().__init__()
        self.base_model = base_model
        self.arf_floor = arf_floor
        self.gate_ceiling = gate_ceiling
        self.detach_noise_score = detach_noise_score
        self.invert_gate = invert_gate

        for p in self.base_model.parameters():
            p.requires_grad = False

        self.adapter = SimplifiedAdapter(768, hidden_dim, zero_init=True)
        self.ng_zero_init_weight = ng_zero_init_weight
        self.noise_gate = NoiseGateMLP(768, hidden_dim=64, init_bias=ng_init_bias,
                                       zero_init_weight=ng_zero_init_weight)

        self.cls_race = nn.Linear(768, num_race)
        self.cls_gender = nn.Linear(768, num_gender)
        self.cls_age = nn.Linear(768, num_age)

    def forward(self, x):
        """
        Returns:
            z_out: (B, 768) — noise-gated adapted embeddings
            z_base: (B, 768) — frozen base embeddings
            logits_race, logits_gender, logits_age
            noise_score: (B, 1) — estimated noise level (0=clean, 1=noisy)
        """
        with torch.no_grad():
            z_base = self.base_model(pixel_values=x).last_hidden_state[:, 0, :]

        noise_score = self.noise_gate(z_base)          # (B, 1)
        z_adapted = self.adapter.adapter(z_base)       # raw adapter output (no residual)
        # v44: detach noise_score from main task gradient (only noise_loss trains NoiseGateMLP)
        ns_for_gate = noise_score.detach() if self.detach_noise_score else noise_score
        if self.invert_gate:
            gate = torch.clamp(ns_for_gate, min=self.arf_floor, max=self.gate_ceiling)
        else:
            gate = torch.clamp(1.0 - ns_for_gate, min=self.arf_floor, max=self.gate_ceiling)
        z_out = z_base + gate * z_adapted              # noise-gated residual

        logits_race = self.cls_race(z_out)
        logits_gender = self.cls_gender(z_out)
        logits_age = self.cls_age(z_out)

        return z_out, z_base, logits_race, logits_gender, logits_age, noise_score


class GRoFAModelV59(nn.Module):
    """
    v59: Per-Task Noise Gates.

    Three independent NoiseGateMLP instances produce task-specific gated embeddings.
    Each task (race, gender, age) has its own noise gate with independent arf_floor,
    allowing different annealing schedules per task.

    Shared adapter and backbone. Classification heads use task-specific z_out.
    Compatible with warm-starting from v28 checkpoints (single noise_gate -> 3 gates).
    """

    def __init__(self, base_model, num_race=7, num_gender=2, num_age=9,
                 hidden_dim=384,
                 arf_floors=(0.0, 0.0, 0.0),
                 gate_ceiling=1.0,
                 invert_gate=False):
        super().__init__()
        self.base_model = base_model
        self.arf_floors = list(arf_floors)  # [race, gender, age] — mutable for annealing
        self.gate_ceiling = gate_ceiling
        self.invert_gate = invert_gate

        for p in self.base_model.parameters():
            p.requires_grad = False

        self.adapter = SimplifiedAdapter(768, hidden_dim, zero_init=True)

        # Per-task noise gates
        self.noise_gate_race = NoiseGateMLP(768, hidden_dim=64)
        self.noise_gate_gender = NoiseGateMLP(768, hidden_dim=64)
        self.noise_gate_age = NoiseGateMLP(768, hidden_dim=64)

        self.cls_race = nn.Linear(768, num_race)
        self.cls_gender = nn.Linear(768, num_gender)
        self.cls_age = nn.Linear(768, num_age)

    def forward(self, x):
        """
        Returns:
            z_out_mean: (B, 768) — mean-gated embeddings (for distillation / inference)
            z_base: (B, 768) — frozen base embeddings
            logits_race, logits_gender, logits_age — from task-specific z_outs
            noise_score_mean: (B, 1) ��� mean of 3 task noise scores
            extras: dict with per-task z_outs and gates
        """
        with torch.no_grad():
            z_base = self.base_model(pixel_values=x).last_hidden_state[:, 0, :]

        z_adapted = self.adapter.adapter(z_base)  # raw adapter output (no residual)

        # Per-task noise scores
        ns_race = self.noise_gate_race(z_base)      # (B, 1)
        ns_gender = self.noise_gate_gender(z_base)   # (B, 1)
        ns_age = self.noise_gate_age(z_base)         # (B, 1)

        # Per-task gates
        if self.invert_gate:
            gate_race = torch.clamp(ns_race, min=self.arf_floors[0], max=self.gate_ceiling)
            gate_gender = torch.clamp(ns_gender, min=self.arf_floors[1], max=self.gate_ceiling)
            gate_age = torch.clamp(ns_age, min=self.arf_floors[2], max=self.gate_ceiling)
        else:
            gate_race = torch.clamp(1.0 - ns_race, min=self.arf_floors[0], max=self.gate_ceiling)
            gate_gender = torch.clamp(1.0 - ns_gender, min=self.arf_floors[1], max=self.gate_ceiling)
            gate_age = torch.clamp(1.0 - ns_age, min=self.arf_floors[2], max=self.gate_ceiling)

        # Per-task z_outs
        z_out_race = z_base + gate_race * z_adapted
        z_out_gender = z_base + gate_gender * z_adapted
        z_out_age = z_base + gate_age * z_adapted

        # Mean z_out for distillation/inference
        gate_mean = (gate_race + gate_gender + gate_age) / 3.0
        z_out_mean = z_base + gate_mean * z_adapted

        # Classification from task-specific z_outs
        logits_race = self.cls_race(z_out_race)
        logits_gender = self.cls_gender(z_out_gender)
        logits_age = self.cls_age(z_out_age)

        noise_score_mean = (ns_race + ns_gender + ns_age) / 3.0

        extras = {
            "z_out_race": z_out_race,
            "z_out_gender": z_out_gender,
            "z_out_age": z_out_age,
            "z_adapted": z_adapted,
            "gate_race": gate_race,
            "gate_gender": gate_gender,
            "gate_age": gate_age,
            "ns_race": ns_race,
            "ns_gender": ns_gender,
            "ns_age": ns_age,
        }

        return z_out_mean, z_base, logits_race, logits_gender, logits_age, noise_score_mean, extras

    @classmethod
    def from_v28_checkpoint(cls, base_model, checkpoint_path, device="cuda", **kwargs):
        """Load a v28 checkpoint into v59, duplicating the single noise_gate to 3 task gates."""
        import copy
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt["model"] if "model" in ckpt else ckpt

        # Determine architecture from checkpoint
        num_race = state["cls_race.weight"].shape[0]
        num_gender = state["cls_gender.weight"].shape[0]
        num_age = state["cls_age.weight"].shape[0]

        model = cls(base_model, num_race=num_race, num_gender=num_gender,
                     num_age=num_age, **kwargs)

        # Build new state dict: map noise_gate.* -> 3 task gates
        new_state = {}
        for k, v in state.items():
            if k.startswith("noise_gate."):
                suffix = k[len("noise_gate."):]
                new_state[f"noise_gate_race.{suffix}"] = v.clone()
                new_state[f"noise_gate_gender.{suffix}"] = v.clone()
                new_state[f"noise_gate_age.{suffix}"] = v.clone()
            elif k.startswith("base_model."):
                continue  # Skip frozen backbone (loaded separately)
            else:
                new_state[k] = v

        missing, unexpected = model.load_state_dict(new_state, strict=False)
        print(f"[v59] Loaded v28 checkpoint -> v59 model")
        print(f"  Missing (expected base_model): {len(missing)}")
        if unexpected:
            print(f"  Unexpected: {unexpected}")

        router_state = ckpt.get("rtdar_router", None)
        return model, router_state


# ============================================================
# R-TDAR Router (v29): Regularized Task-Decomposed Alignment
# ============================================================
class TDARRouter(nn.Module):
    """
    Task-Decomposed Alignment Router (R-TDAR) — v29. 99 params.

    Fixes 3 failure modes from v28 TALRouter:
      1. tau_min=3.0 prevents argmax collapse (vs TALRouter's tau_min=1.0)
      2. alpha scaling amplifies alignment signal before prior addition
      3. L2 + entropy regularization on prior b prevents weight collapse

    w_k(x) = softmax( (alpha * a_k(x) + b_k^(g)) / tau )
    where a_k(x) = sum_t task_weight_t * cosine(g_k, g_t)  (task-decomposed)

    Parameters: G*K + 1 = 7*14 + 1 = 99 scalars (same as TALRouter).
    """

    def __init__(self, num_losses=14, num_groups=7, tau_min=3.0, tau_max=10.0, alpha=5.0):
        super().__init__()
        self.num_losses, self.num_groups = num_losses, num_groups
        self.tau_min, self.tau_max, self.alpha = tau_min, tau_max, alpha

        # Per-group learnable prior bias (init 0 = uniform)
        self.b = nn.Parameter(torch.zeros(num_groups, num_losses))

        # Learnable temperature: tau = tau_min + (tau_max - tau_min) * sigmoid(tau_raw)
        # tau_raw=0 -> sigmoid(0)=0.5 -> tau ≈ 6.5
        self.tau_raw = nn.Parameter(torch.tensor(0.0))

    @property
    def tau(self):
        return self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.tau_raw)

    def compute_prior_regularization(self):
        """L2 regularization on prior b. Returns scalar."""
        return (self.b ** 2).sum()

    def compute_weight_entropy(self, weights):
        """Negative entropy of batch-averaged weights. Adding to loss MAXIMIZES entropy."""
        w_bar = weights.mean(dim=0)  # (K,)
        return (w_bar * torch.log(w_bar + 1e-8)).sum()  # -H(w_bar)

    def forward(self, alignment_scores, group_labels):
        """
        Args:
            alignment_scores: (B, K) task-decomposed alignment in [-1, 1], DETACHED.
            group_labels: (B,) integer group labels 0..G-1
        Returns:
            weights: (B, K) per-sample loss weights summing to 1
            diagnostics: dict for logging
        """
        prior = self.b[group_labels]  # (B, K)
        logits = (self.alpha * alignment_scores + prior) / self.tau
        weights = F.softmax(logits, dim=-1)

        diagnostics = {
            "alignment": alignment_scores.detach(),
            "prior": prior.detach(),
            "weights": weights.detach(),
            "tau": self.tau.detach().item(),
            "alpha": self.alpha,
            "prior_norm": self.b.detach().norm().item(),
            "weight_entropy": -(weights.mean(0) * torch.log(weights.mean(0) + 1e-8)).sum().item(),
        }
        return weights, diagnostics


class TDARRouterV30(nn.Module):
    """
    Task-Decomposed Alignment Router v30. 99 params.

    Changes from v29 TDARRouter:
      1. tau_min=2.0 (vs 3.0): allows sharper routing without argmax collapse
      2. No compute_weight_entropy() for loss: entropy reg structurally removed
      3. compute_prior_l2() renamed for clarity (same as compute_prior_regularization)
      4. weight_entropy still in diagnostics for MONITORING only

    w_k(x) = softmax( (alpha * a_k(x) + b_k^(g)) / tau )
    Parameters: G*K + 1 = 7*14 + 1 = 99 scalars.
    """

    def __init__(self, num_losses=14, num_groups=7, tau_min=2.0, tau_max=10.0, alpha=5.0):
        super().__init__()
        self.num_losses, self.num_groups = num_losses, num_groups
        self.tau_min, self.tau_max, self.alpha = tau_min, tau_max, alpha

        # Per-group learnable prior bias (init 0 = uniform)
        self.b = nn.Parameter(torch.zeros(num_groups, num_losses))

        # Learnable temperature: tau = tau_min + (tau_max - tau_min) * sigmoid(tau_raw)
        # tau_raw=0 -> sigmoid(0)=0.5 -> tau = 2.0 + 4.0 = 6.0
        self.tau_raw = nn.Parameter(torch.tensor(0.0))

    @property
    def tau(self):
        return self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.tau_raw)

    def compute_prior_l2(self):
        """L2 regularization on prior b. Returns scalar."""
        return (self.b ** 2).sum()

    def forward(self, alignment_scores, group_labels):
        """
        Args:
            alignment_scores: (B, K) task-decomposed alignment in [-1, 1], DETACHED.
            group_labels: (B,) integer group labels 0..G-1
        Returns:
            weights: (B, K) per-sample loss weights summing to 1
            diagnostics: dict for logging
        """
        prior = self.b[group_labels]  # (B, K)
        logits = (self.alpha * alignment_scores + prior) / self.tau
        weights = F.softmax(logits, dim=-1)

        w_mean = weights.mean(0)
        weight_entropy = -(w_mean * torch.log(w_mean + 1e-8)).sum().item()

        diagnostics = {
            "alignment": alignment_scores.detach(),
            "prior": prior.detach(),
            "weights": weights.detach(),
            "tau": self.tau.detach().item(),
            "alpha": self.alpha,
            "prior_norm": self.b.detach().norm().item(),
            "weight_entropy": weight_entropy,
            "max_weight": w_mean.max().item(),
            "min_weight": w_mean.min().item(),
            "weight_std": w_mean.std().item(),
        }
        return weights, diagnostics


class NAARRouter(nn.Module):
    """Noise-Adaptive Alignment Router — v31.

    ZERO learnable parameters.  Per-sample temperature is a deterministic
    function of the noise score:
        τ(x) = τ_clean + (τ_noisy - τ_clean) · s(x)
    where s(x) ∈ [0, 1] is the detached noise score from NoiseGateMLP.

    Clean samples (s≈0) → low τ → sharp routing (specialize).
    Noisy samples (s≈1) → high τ → uniform routing (hedge).
    """

    def __init__(self, tau_clean: float = 1.5, tau_noisy: float = 6.0,
                 alpha: float = 5.0):
        super().__init__()
        # Fixed hyper-parameters — no nn.Parameter
        self.tau_clean = tau_clean
        self.tau_noisy = tau_noisy
        self.alpha = alpha

    def forward(self, alignment_scores, noise_scores):
        """
        Args:
            alignment_scores: (B, K) task-decomposed alignment, DETACHED.
            noise_scores: (B,) or (B, 1) detached noise scores in [0, 1].
        Returns:
            weights: (B, K) per-sample loss weights summing to 1.
            diagnostics: dict for logging.
        """
        # Handle (B,1) → (B,)
        if noise_scores.dim() == 2:
            noise_scores = noise_scores.squeeze(-1)

        # Per-sample temperature: τ(x) ∈ [τ_clean, τ_noisy]
        tau = self.tau_clean + (self.tau_noisy - self.tau_clean) * noise_scores  # (B,)

        # Routing logits and weights
        logits = self.alpha * alignment_scores / tau.unsqueeze(1)  # (B, K)
        weights = F.softmax(logits, dim=-1)  # (B, K)

        # ── Diagnostics ──
        w_mean = weights.mean(0)
        weight_entropy = -(w_mean * torch.log(w_mean + 1e-8)).sum().item()

        # Per-noise-regime entropy
        clean_mask = noise_scores < 0.3
        noisy_mask = noise_scores > 0.5
        n_clean = clean_mask.sum().item()
        n_noisy = noisy_mask.sum().item()

        def _regime_entropy(mask, n):
            if n < 2:
                return float('nan')
            wm = weights[mask].mean(0)
            return -(wm * torch.log(wm + 1e-8)).sum().item()

        entropy_clean = _regime_entropy(clean_mask, n_clean)
        entropy_noisy = _regime_entropy(noisy_mask, n_noisy)

        diagnostics = {
            "weights": weights.detach(),
            "tau_mean": tau.mean().item(),
            "tau_std": tau.std().item(),
            "weight_entropy": weight_entropy,
            "entropy_clean": entropy_clean,
            "entropy_noisy": entropy_noisy,
            "n_clean": n_clean,
            "n_noisy": n_noisy,
            "max_weight": w_mean.max().item(),
            "min_weight": w_mean.min().item(),
        }
        return weights, diagnostics


# ============================================================
# Full Model (v4-v12): Frozen BLIP + SSD-Net
# ============================================================
class GRoFAModel(nn.Module):
    """
    Full model: Frozen BLIP backbone + trainable SSD-Net.

    Args:
        base_model: BLIP vision model (frozen)
        num_losses: number of loss functions
        mode: "kg" (v4) or "mol" (v5)
        top_k: for MoL sparse routing (None = soft routing)
        uncertainty_guided: if True, append teacher entropy to router input (v8)
        high_freq_input: if True, use high-pass filtered input for frequency path (v8)
    """

    def __init__(self, base_model, num_losses=12, mode="kg", top_k=None,
                 uncertainty_guided=False, high_freq_input=False,
                 static_weights=None, num_groups=None):
        super().__init__()
        self.base_model = base_model
        self.mode = mode

        # Freeze Teacher (Base Model)
        for p in self.base_model.parameters():
            p.requires_grad = False

        # The Student (Trainable)
        self.ssd_net = SSDNet(
            input_dim=768, num_losses=num_losses, mode=mode, top_k=top_k,
            uncertainty_guided=uncertainty_guided, high_freq_input=high_freq_input,
            static_weights=static_weights, num_groups=num_groups,
        )

    def forward(self, x, gate_override=None, group_labels=None):
        # 1. Extract Frozen Features
        with torch.no_grad():
            out = self.base_model(pixel_values=x)
            z_base = out.last_hidden_state[:, 0, :]  # [CLS] token

        # 2. Refine with SSD-Net
        results = self.ssd_net(z_base, gate_override=gate_override,
                               group_labels=group_labels)

        if self.mode in ("mol", "residual_mol", "group_mol"):
            z_new, gate, mol_weights, balance_loss = results
            return {
                "z": z_new,
                "gate": gate,
                "mol_weights": mol_weights,
                "balance_loss": balance_loss,
            }
        else:
            z_new, gate, log_var, delta = results
            return {
                "z": z_new,
                "gate": gate,
                "log_var": log_var,
                "delta": delta,
            }
