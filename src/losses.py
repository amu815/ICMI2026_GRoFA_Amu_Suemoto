#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src3/losses.py (The Ultimate Edition)
GRoFA loss pool (routed distillation/alignment losses)
物理、統計、情報理論、そして最新の自己教師あり学習理論を統合した特殊Loss関数群。
Standard Losses (MSE, KL, ArcFace, etc.) are used directly from torch/libraries.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

# ==========================================
# 1. Frequency & Physics (物理・信号処理)
# ==========================================
class FocalFrequencyLoss(nn.Module):
    """
    [Legacy+New] FFL: 周波数領域での整合性を強制。
    画像がぼやけるのを防ぎ、鋭いエッジ（高周波成分）を復元するのに不可欠。
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, z_pred, z_target):
        # 埋め込みベクトルzに対しても周波数解析は有効（次元間の周期性など）
        z_pred_freq = torch.fft.rfft(z_pred, dim=-1, norm='ortho')
        z_target_freq = torch.fft.rfft(z_target, dim=-1, norm='ortho')
        
        diff = torch.abs(z_pred_freq - z_target_freq) ** 2
        # 難易度の高い周波数成分（誤差が大きい部分）に重みを置くFocal的な重み付け
        weight = (diff + 1e-8) ** self.alpha
        return (weight * diff).mean()

class CosineDirectionLoss(nn.Module):
    """
    [Legacy: Cosine] ベクトルの「向き」の一致を強制。
    MSE(大きさ+向き)とは異なり、大きさの変動を許容するため学習が安定しやすい。
    """
    def forward(self, z_pred, z_target):
        return (1.0 - F.cosine_similarity(z_pred, z_target, dim=-1)).mean()

# ==========================================
# 2. Statistics & Structure (統計・構造)
# ==========================================
class StatSSIMLoss(nn.Module):
    """
    [Current: SSIM] Embeddingの統計的構造（平均・分散・共分散）の一致度。
    画像のSSIMを特徴量ベクトル用に一次元化したもの。
    """
    def forward(self, x, y):
        mu_x, mu_y = x.mean(1), y.mean(1)
        sig_x, sig_y = x.var(1), y.var(1)
        sig_xy = (x * y).mean(1) - mu_x * mu_y
        
        c1, c2 = 0.01**2, 0.03**2
        ssim = ((2*mu_x*mu_y + c1) * (2*sig_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sig_x + sig_y + c2))
        return 1.0 - ssim.mean()

class CenterLoss(nn.Module):
    """
    [Legacy: Center] クラス内分散の最小化。
    ArcFaceが「角度」で分離するのに対し、これは「ユークリッド距離」で凝集させる。
    """
    def forward(self, z, labels):
        # シンプルな実装: 各クラスの重心との距離の分散
        unique_labels = torch.unique(labels)
        loss = torch.tensor(0.0, device=z.device)
        count = 0
        for l in unique_labels:
            mask = (labels == l)
            if mask.sum() > 1:
                cluster = z[mask]
                center = cluster.mean(dim=0, keepdim=True)
                # 重心と各点の距離の二乗平均
                loss += ((cluster - center) ** 2).sum(dim=1).mean()
                count += 1
        return loss / max(count, 1)

# ==========================================
# 3. Fairness & Independence (公平性・独立性)
# ==========================================
class HSICLoss(nn.Module):
    """
    [New SOTA] HSIC: Hilbert-Schmidt Independence Criterion
    特徴量(z)とセンシティブ属性(s)の「非線形な独立性」を強制する最強の公平性Loss。
    Adversarial Trainingよりも学習が安定する。
    """
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def _kernel(self, X, sigma):
        X = X.view(len(X), -1)
        dist_sq = torch.cdist(X, X, p=2) ** 2
        return torch.exp(-dist_sq / (2 * sigma ** 2))

    def forward(self, z, s):
        N = z.size(0)
        if N < 2: return torch.tensor(0.0, device=z.device)
        # sがラベルならOne-hot化、連続値ならそのまま
        if s.dim() == 1 or (s.dim()==2 and s.size(1)==1):
            if s.dtype == torch.long:
                num_cls = int(s.max().item()) + 1
                s = F.one_hot(s, num_classes=num_cls).float()
            else:
                s = s.float().view(-1, 1)
        
        K = self._kernel(z, self.sigma)
        L = self._kernel(s, 1.0) # 属性空間のカーネル
        
        # Centering Matrix H
        H = torch.eye(N, device=z.device) - (1.0 / N) * torch.ones((N, N), device=z.device)
        
        # HSIC = Trace(KHLH) / (n-1)^2
        return torch.trace(K @ H @ L @ H) / ((N - 1) ** 2)

class OrthogonalLoss(nn.Module):
    """
    [Current: Ortho] 特徴量の次元間の直交化。
    異なる次元が同じ情報を重複して持つのを防ぐ（Disentanglement）。
    """
    def forward(self, z1, z2):
        # z1とz2の内積（相関）を最小化する
        # ここではBatch方向ではなく、Feature方向の相関を見る場合もあるが、
        # User実装に合わせて「ペア間の直交性」とするなら以下
        return (F.normalize(z1, dim=-1) * F.normalize(z2, dim=-1)).sum(dim=-1).abs().mean()

class MMDLoss(nn.Module):
    """
    [Current: MMD] 最大平均不一致。
    CleanとNoisyの分布（Distribution）自体を一致させる。
    """
    def forward(self, source, target):
        # 線形カーネル版の簡易実装（平均の一致）
        return ((source.mean(0) - target.mean(0)) ** 2).mean()

# ==========================================
# 4. SOTA Self-Supervised (最新トレンド)
# ==========================================
# ==========================================
# 5. Fairness-Aware Training (v11)
# ==========================================
def compute_group_equity_loss(per_sample_loss, group_labels, num_groups):
    """
    Per-group loss standard deviation minimization.
    Ensures equal distillation quality across all demographic groups.

    Args:
        per_sample_loss: (B,) scalar loss per sample
        group_labels: (B,) integer group labels (0..num_groups-1)
        num_groups: total number of groups

    Returns:
        equity_loss: scalar, std of per-group mean losses
    """
    group_losses = []
    for g in range(num_groups):
        mask = (group_labels == g)
        if mask.sum() > 0:
            group_losses.append(per_sample_loss[mask].mean())
    if len(group_losses) < 2:
        return torch.tensor(0.0, device=per_sample_loss.device)
    return torch.stack(group_losses).std()


class BarlowTwinsLoss(nn.Module):
    """
    [New SOTA] Barlow Twins: Redundancy Reduction
    特徴量の相関行列を単位行列に近づけることで、
    「次元ごとの独立性」と「不変性」を同時に獲得する。
    """
    def __init__(self, lambda_coeff=5e-3):
        super().__init__()
        self.lambda_coeff = lambda_coeff

    def forward(self, z1, z2):
        # Normalize along batch dimension
        N, D = z1.shape
        z1_norm = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
        z2_norm = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)

        # Cross-correlation matrix (D x D)
        c = (z1_norm.T @ z2_norm) / N

        # Loss: diagonal terms -> 1, off-diagonal -> 0
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        
        return on_diag + self.lambda_coeff * off_diag

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class VICRegLoss(nn.Module):
    """
    [New SOTA] VICReg: Variance-Invariance-Covariance Regularization
    Contrastive Learningの欠点（負例が必要）を克服。
    1. Variance: 特徴が潰れる(Collapse)のを防ぐ
    2. Invariance: CleanとNoisyを近づける (MSE)
    3. Covariance: 次元間の相関を消す
    """
    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, x, y):
        # 1. Invariance (MSE)
        repr_loss = F.mse_loss(x, y)

        # 2. Variance (Hinge Loss on Std)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        # 3. Covariance
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        cov_x = (x.T @ x) / (x.size(0) - 1)
        cov_y = (y.T @ y) / (y.size(0) - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum() / x.size(1) + \
                   off_diagonal(cov_y).pow_(2).sum() / y.size(1)

        return (
            self.sim_coeff * repr_loss +
            self.std_coeff * std_loss +
            self.cov_coeff * cov_loss
        )