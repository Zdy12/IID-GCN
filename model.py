# -*- coding: utf-8 -*-
"""
RUS-GCN:
Three-Channel GCN + Residual Edge Gate
+ Mutual Cross-Attention-Based Information Fusion
+ ROI Attention
+ Soft Node Grouping
+ Channel Attention
for PED-based fMRI disease classification.

This version continues from:
    Three-Channel GCN
    + Channel Attention
    + ROI Attention Readout
    + Residual Edge Gate
    + Soft Node Grouping

New module added:
    Mutual Cross-Attention-Based Information Fusion

Added for interpretability:
    Save subject-level ROI attention on the held-out test set.
    Each subject saves alpha_R, alpha_U, alpha_S, where:
        alpha_R: ROI importance in redundancy channel
        alpha_U: ROI importance in uniqueness channel
        alpha_S: ROI importance in synergy channel

Core idea:
    After three channel-specific GCN encoders, for each ROI i we have:
        h_i^R, h_i^U, h_i^S

    The new module performs mutual cross-attention among the three channels.
    For a target channel c, H^c is used as Query and the other two channels
    are used as Key/Value. The enhanced representation is:
        H_new^c = H^c + sum_{s != c} Attn_{c<-s}

Why this module:
    It lets redundancy, uniqueness, and synergy complement each other at the
    ROI level, instead of only fusing them at the final graph-level readout.

Input files:
    ~/Desktop/TimeHOIAD_bivariate_PED/AD_bivariate_PED.npz
    ~/Desktop/TimeHOIAD_bivariate_PED/CN_bivariate_PED.npz
    ~/Desktop/TimeHOIAD_bivariate_PED/MCI_bivariate_PED.npz

Each npz file must contain:
    ped: shape = [n_subject, 90, 90, 3]

Channel definition:
    ped[:, :, :, 0] = redundancy
    ped[:, :, :, 1] = unique
    ped[:, :, :, 2] = synergy
"""

import os
import random
import copy
import csv
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 0. Config
# ============================================================

@dataclass
class Config:
    ped_dir: str = os.path.expanduser("~/Desktop/TimeHOIAD_bivariate_PED")
    ad_file: str = "AD_bivariate_PED.npz"
    cn_file: str = "CN_bivariate_PED.npz"
    mci_file: str = "MCI_bivariate_PED.npz"

    num_nodes: int = 90
    num_channels: int = 3
    num_classes: int = 3

    # Keep this the same as your previous best run for fair comparison.
    topk_per_node: Optional[int] = 10

    hidden_dim: int = 32
    num_groups: int = 7
    dropout: float = 0.10

    n_splits: int = 10
    val_ratio_in_trainval: float = 1 / 9
    batch_size: int = 16
    max_epochs: int = 300
    patience: int = 70
    lr: float = 1e-3
    weight_decay: float = 5e-4
    seed: int = 42

    # Residual edge gate.
    # In the paper notation:
    #     eta^C = eta_max * tanh(rho^C), rho^C initialized as 0.
    # Keeping gate_gamma_init only as a backward-compatible name for rho_init.
    gate_gamma_init: float = 0.0
    gate_eta_max: float = 0.5

    # Optional weak regularization for residual gate.
    # Keep 0 for first run. If gate becomes too strong, try 1e-4.
    lambda_gate_deviation: float = 1e-3

    # Optional regularization for ROI-wise cross-information attention.
    # Keep 0 first. If alpha collapses to one channel too early, try 1e-4.
    lambda_cross_entropy: float = 0.0

    # Debug: set to 1 first; set None for full 10-fold CV.
    max_folds_to_run: Optional[int] = None

    save_dir: str = os.path.expanduser("~/Desktop/RUS_GCN_results")


CFG = Config()


# ============================================================
# 1. Reproducibility
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. Data loading and preprocessing
# ============================================================

def load_ped_feature(npz_path: str) -> np.ndarray:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Cannot find file: {npz_path}")

    data = np.load(npz_path)

    if "ped" not in data:
        raise KeyError(
            f"File {npz_path} does not contain key 'ped'. "
            f"Available keys: {list(data.keys())}"
        )

    ped = data["ped"].astype(np.float32)

    if ped.ndim != 4 or ped.shape[-1] != 3:
        raise ValueError(f"Expected ped shape [B, N, N, 3], got {ped.shape}")

    return ped


def preprocess_ped_no_subject_scaling(ped: np.ndarray) -> np.ndarray:
    """
    Convert [B, N, N, 3] to [B, 3, N, N].

    No per-subject max scaling here.
    Scaling will be done inside each fold using training subjects only.
    """

    x = np.transpose(ped, (0, 3, 1, 2)).astype(np.float32)  # [B, 3, N, N]
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    x = np.clip(x, 0.0, None)

    # Symmetrize.
    x = 0.5 * (x + np.transpose(x, (0, 1, 3, 2)))

    # Remove diagonal.
    n = x.shape[-1]
    diag_idx = np.arange(n)
    x[:, :, diag_idx, diag_idx] = 0.0

    return x.astype(np.float32)


def load_all_data(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    ad_path = os.path.join(cfg.ped_dir, cfg.ad_file)
    cn_path = os.path.join(cfg.ped_dir, cfg.cn_file)
    mci_path = os.path.join(cfg.ped_dir, cfg.mci_file)

    ped_ad = load_ped_feature(ad_path)
    ped_cn = load_ped_feature(cn_path)
    ped_mci = load_ped_feature(mci_path)

    print("Raw ped_AD shape :", ped_ad.shape)
    print("Raw ped_CN shape :", ped_cn.shape)
    print("Raw ped_MCI shape:", ped_mci.shape)

    x_ad = preprocess_ped_no_subject_scaling(ped_ad)
    x_cn = preprocess_ped_no_subject_scaling(ped_cn)
    x_mci = preprocess_ped_no_subject_scaling(ped_mci)

    y_ad = np.zeros(x_ad.shape[0], dtype=np.int64)       # AD  -> 0
    y_cn = np.ones(x_cn.shape[0], dtype=np.int64)        # CN  -> 1
    y_mci = np.ones(x_mci.shape[0], dtype=np.int64) * 2  # MCI -> 2

    x = np.concatenate([x_ad, x_cn, x_mci], axis=0)
    y = np.concatenate([y_ad, y_cn, y_mci], axis=0)

    print("X raw shape:", x.shape)
    print("y shape    :", y.shape)
    print("Class counts [AD, CN, MCI]:", np.bincount(y, minlength=cfg.num_classes))

    return x, y


def fold_channel_max_scale(x: np.ndarray, train_idx: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    x: [B, 3, N, N]

    Use only training subjects to estimate one scale for each channel.
    """

    train_x = x[train_idx]
    channel_max = train_x.max(axis=(0, 2, 3), keepdims=True) + eps  # [1, 3, 1, 1]

    out = x / channel_max
    out = np.clip(out, 0.0, 1.0)

    return out.astype(np.float32)


def keep_topk_edges(x: np.ndarray, k: Optional[int]) -> np.ndarray:
    """
    x: [B, 3, N, N]

    Keep top-k strongest neighbors per node in each channel.
    Symmetrize the mask to keep an undirected graph.
    """

    if k is None:
        return x

    if k <= 0:
        raise ValueError("topk_per_node must be positive or None.")

    B, C, N, _ = x.shape
    k = min(k, N - 1)

    out = np.zeros_like(x, dtype=np.float32)

    for b in range(B):
        for c in range(C):
            A = x[b, c].copy()
            np.fill_diagonal(A, 0.0)

            mask = np.zeros((N, N), dtype=np.float32)

            idx = np.argpartition(-A, kth=k, axis=1)[:, :k]
            rows = np.arange(N)[:, None]
            mask[rows, idx] = 1.0

            # Undirected mask.
            mask = np.maximum(mask, mask.T)

            A_sparse = A * mask
            A_sparse = 0.5 * (A_sparse + A_sparse.T)
            np.fill_diagonal(A_sparse, 0.0)

            out[b, c] = A_sparse

    return out.astype(np.float32)


class PEDGraphDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class IndexedSubset(Dataset):
    """
    A subset wrapper that returns the original subject index.

    This is used only for interpretability saving. It does not affect
    training or normal evaluation.
    """

    def __init__(self, base_dataset: Dataset, indices):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        original_idx = int(self.indices[i])
        x, y = self.base_dataset[original_idx]
        return x, y, original_idx


# ============================================================
# 3. Model modules
# ============================================================

def normalize_adj(adj: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Symmetric normalized adjacency with self-loops.

    adj: [B, N, N]
    """

    B, N, _ = adj.shape

    eye = torch.eye(N, device=adj.device, dtype=adj.dtype).unsqueeze(0).expand(B, -1, -1)
    A = adj + eye

    deg = A.sum(dim=-1).clamp_min(eps)
    deg_inv_sqrt = torch.pow(deg, -0.5)

    A_norm = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)

    return A_norm


class ResidualEdgeGate(nn.Module):
    """
    Residual edge gate implemented according to the paper formulation.

    For each subject i and channel C:
        G_i^C = sigmoid(s^C F_i^C + M^C)
        eta^C = eta_max * tanh(rho^C)
        A_tilde_i^C = A_i^C * [1 + eta^C * (2 * G_i^C - 1)]

    In this code, x is the input adjacency tensor currently passed to the
    model. Since the original training pipeline passes the top-k sparse
    PED graph into the model, x is used as both the gate input and the
    adjacency to be recalibrated. This keeps the rest of the code unchanged.

    Interpretability outputs:
        raw_gate      : G_i^C
        modulation    : 1 + eta^C * (2 * G_i^C - 1)
        edge_delta    : A_tilde_i^C - A_i^C
        global_M      : symmetric zero-diagonal M^C
        eta           : eta^C
    """

    def __init__(
        self,
        num_channels: int,
        num_nodes: int,
        eta_max: float = 0.5,
        rho_init: float = 0.0,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_nodes = num_nodes
        self.eta_max = float(eta_max)

        # s^C in the paper, one learnable scale for each information channel.
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))

        # Raw parameter of M^C. It is symmetrized and forced to have zero
        # diagonal inside forward(), matching the paper definition.
        self.global_M_raw = nn.Parameter(torch.zeros(1, num_channels, num_nodes, num_nodes))

        # rho^C. rho_init=0 gives eta^C=0 and therefore A_tilde=A at initialization.
        self.rho = nn.Parameter(torch.ones(num_channels) * rho_init)

        eye = torch.eye(num_nodes).view(1, 1, num_nodes, num_nodes)
        self.register_buffer("offdiag_mask", 1.0 - eye)

    def get_symmetric_global_M(self) -> torch.Tensor:
        """
        Return M^C as a symmetric zero-diagonal matrix.

        Shape:
            [1, C, N, N]
        """
        M = 0.5 * (self.global_M_raw + self.global_M_raw.transpose(-1, -2))
        M = M * self.offdiag_mask
        return M

    def forward(self, x: torch.Tensor):
        """
        x: [B, C, N, N]

        return:
            x_gated:   [B, C, N, N], recalibrated adjacency A_tilde
            raw_gate:  [B, C, N, N], G_i^C
            eta:       [C],          eta^C
            modulation:[B, C, N, N], multiplicative residual factor
            edge_delta:[B, C, N, N], A_tilde - A
            global_M:  [C, N, N],    symmetric zero-diagonal M^C
        """

        B, C, N, _ = x.shape
        if C != self.num_channels or N != self.num_nodes:
            raise ValueError(
                f"Expected x shape [B,{self.num_channels},{self.num_nodes},{self.num_nodes}], "
                f"got {tuple(x.shape)}"
            )

        eye = torch.eye(N, device=x.device, dtype=x.dtype).view(1, 1, N, N)
        offdiag = 1.0 - eye

        # Keep the input graph symmetric and zero-diagonal before gating.
        A = 0.5 * (x + x.transpose(-1, -2))
        A = A * offdiag

        # M^C in the paper: symmetric global edge modulation matrix with zero diagonal.
        global_M = self.get_symmetric_global_M().to(device=x.device, dtype=x.dtype)

        # G_i^C = sigmoid(s^C F_i^C + M^C).
        # The current pipeline provides A as model input, so A is used as F here
        # to avoid changing the rest of the data interface.
        raw_gate = torch.sigmoid(self.scale * A + global_M)
        raw_gate = 0.5 * (raw_gate + raw_gate.transpose(-1, -2))
        raw_gate = raw_gate * offdiag

        # eta^C = eta_max * tanh(rho^C), initialized at 0.
        eta = self.eta_max * torch.tanh(self.rho)
        eta_view = eta.view(1, self.num_channels, 1, 1)

        # A_tilde_i^C = A_i^C * [1 + eta^C * (2G_i^C - 1)].
        modulation = 1.0 + eta_view * (2.0 * raw_gate - 1.0)
        modulation = modulation * offdiag

        x_gated = A * modulation
        x_gated = torch.clamp(x_gated, min=0.0)
        x_gated = 0.5 * (x_gated + x_gated.transpose(-1, -2))
        x_gated = x_gated * offdiag

        edge_delta = x_gated - A

        return (
            x_gated,
            raw_gate,
            eta,
            modulation,
            edge_delta,
            global_M.squeeze(0),
        )


class GCNLayer(nn.Module):
    """
    Basic GCN layer:
        H' = A_norm H W
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        A_norm = normalize_adj(adj)
        out = torch.bmm(A_norm, h)
        out = self.linear(out)
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out


class SingleChannelGCNEncoder(nn.Module):
    """
    One GCN encoder for one PED channel.

    Initial node feature:
        each node's original R/U/S connectivity profile, i.e., one row of the
        input channel matrix before residual edge gating.

    Message-passing adjacency:
        the gated adjacency matrix produced by the residual edge gate.

    Output:
        node features h: [B, N, H]
    """

    def __init__(self, num_nodes: int, hidden_dim: int, dropout: float):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gcn1 = GCNLayer(hidden_dim, hidden_dim, dropout)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim, dropout)

    def forward(self, adj_c: torch.Tensor, node_feat_c: torch.Tensor):
        """
        adj_c:
            [B, N, N], gated adjacency matrix used for GCN message passing.

        node_feat_c:
            [B, N, N], original input R/U/S matrix of this channel, used as
            the initial node feature matrix. Each row is the connectivity
            profile of one ROI.
        """

        # Use the original R/U/S channel matrix as node features.
        # Do NOT use the gated adjacency matrix as the initial node feature.
        h = self.input_proj(node_feat_c)  # [B, N, H]

        # Use the gated adjacency matrix only for graph propagation.
        h = self.gcn1(h, adj_c)
        h = self.gcn2(h, adj_c)

        return h


class NodeWiseCrossInformationInteraction(nn.Module):
    """
    Mutual cross-attention-based information fusion among redundancy,
    unique-information, and synergy channels.

    This module follows the design:
        for each target channel c, H_c is used as Query;
        the other two channels are used as Key/Value;
        the enhanced feature is obtained by direct residual addition:

            H_c_new = H_c + sum_{s != c} Attn_{c<-s}

    where
        Attn_{c<-s} = softmax((H_c W_Q^c)(H_s W_K^s)^T / sqrt(d_k)) (H_s W_V^s)

    Input:
        h_list: list of 3 tensors, each [B, N, H]

    Output:
        h_refined:      list of 3 tensors, each [B, N, H]
        cross_attn:     [B, C, C, N, N]
                        cross_attn[:, c, s] is the attention matrix when
                        target channel c attends to source channel s.
        cross_strength: [B, C, C]
                        message strength of each target-source pair.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_channels: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.scale = float(hidden_dim) ** 0.5

        self.q_proj = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_channels)
        ])

        self.k_proj = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_channels)
        ])

        self.v_proj = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_channels)
        ])

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, h_list):
        """
        h_list:
            [
                H_R: [B, N, H],
                H_U: [B, N, H],
                H_S: [B, N, H]
            ]
        """

        C = self.num_channels
        B, N, H = h_list[0].shape

        cross_attn = h_list[0].new_zeros(B, C, C, N, N)
        cross_strength = h_list[0].new_zeros(B, C, C)

        h_refined = []

        for tgt in range(C):
            Q = self.q_proj[tgt](h_list[tgt])  # [B, N, H]

            # Start from the original target-channel feature matrix.
            h_new = h_list[tgt]

            for src in range(C):
                if src == tgt:
                    continue

                K = self.k_proj[src](h_list[src])  # [B, N, H]
                V = self.v_proj[src](h_list[src])  # [B, N, H]

                # Target channel as Query; source channel as Key/Value.
                attn_score = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [B, N, N]
                attn = F.softmax(attn_score, dim=-1)                       # [B, N, N]
                attn = self.attn_dropout(attn)

                # Cross-attended complementary feature from source channel.
                msg = torch.bmm(attn, V)                                   # [B, N, H]

                # Direct residual addition:
                # H_tgt_new = H_tgt + Attn_{tgt<-src1} + Attn_{tgt<-src2}
                h_new = h_new + msg

                cross_attn[:, tgt, src] = attn
                cross_strength[:, tgt, src] = msg.norm(p=2, dim=-1).mean(dim=-1)

            h_refined.append(h_new)

        return h_refined, cross_attn, cross_strength


class ROIAttentionReadout(nn.Module):
    """
    ROI attention readout.

    Given node features h_i:
        alpha_i = softmax(score(h_i))
        g_att = sum_i alpha_i h_i
    """

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h: torch.Tensor):
        scores = self.score(h).squeeze(-1)  # [B, N]
        alpha = torch.softmax(scores, dim=-1)

        att_pool = torch.sum(alpha.unsqueeze(-1) * h, dim=1)

        return att_pool, alpha


class SoftNodeGroupingReadout(nn.Module):
    """
    Soft node grouping pooling.

    Given ROI node features H:
        P = softmax(MLP(H))       # [B, N, K]
        H_g = P^T H / sum_i P_ik # [B, K, H]

    Return group_mean and group_max.
    """

    def __init__(self, hidden_dim: int, num_groups: int, dropout: float):
        super().__init__()
        self.num_groups = num_groups

        self.assign = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_groups),
        )

    def forward(self, h: torch.Tensor):
        """
        h: [B, N, H]

        return:
            group_mean: [B, H]
            group_max:  [B, H]
            P:          [B, N, K]
        """
        logits = self.assign(h)          # [B, N, K]
        P = torch.softmax(logits, dim=-1)

        P_t = P.transpose(1, 2)          # [B, K, N]
        group_size = P_t.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        group_h = torch.bmm(P_t, h) / group_size   # [B, K, H]

        group_mean = group_h.mean(dim=1)
        group_max = group_h.max(dim=1).values

        return group_mean, group_max, P


class ChannelBranchReadout(nn.Module):
    """
    Readout for one channel after cross-information refinement.

    Readout:
        concat(mean_pool, max_pool, ROI-attention_pool, group_mean, group_max)
    """

    def __init__(self, hidden_dim: int, num_groups: int, dropout: float):
        super().__init__()

        self.roi_attention = ROIAttentionReadout(hidden_dim, dropout)
        self.group_readout = SoftNodeGroupingReadout(hidden_dim, num_groups, dropout)

    def forward(self, h: torch.Tensor):
        mean_pool = h.mean(dim=1)
        max_pool = h.max(dim=1).values
        att_pool, roi_alpha = self.roi_attention(h)
        group_mean, group_max, group_assign = self.group_readout(h)

        graph_emb = torch.cat(
            [mean_pool, max_pool, att_pool, group_mean, group_max],
            dim=-1
        )  # [B, 5H]

        return graph_emb, roi_alpha, group_assign


class ChannelAttentionFusion(nn.Module):
    """
    Learn graph-level weights for R/U/S branches.
    """

    def __init__(self, branch_dim: int, dropout: float):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(branch_dim, branch_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(branch_dim // 2, 1),
        )

    def forward(self, branch_embeds: torch.Tensor):
        scores = self.score(branch_embeds).squeeze(-1)  # [B, 3]
        beta = torch.softmax(scores, dim=-1)            # [B, 3]

        fused = torch.sum(beta.unsqueeze(-1) * branch_embeds, dim=1)  # [B, D]

        return fused, beta


class RUSGCN(nn.Module):
    """
    RUS-GCN:
        Residual Edge Gate
        + three channel-specific GCN encoders
        + node-wise cross-information interaction
        + ROI attention readout
        + soft node grouping readout
        + channel attention fusion
    """

    def __init__(
        self,
        num_nodes: int = 90,
        num_channels: int = 3,
        hidden_dim: int = 32,
        num_groups: int = 12,
        num_classes: int = 3,
        dropout: float = 0.30,
        gate_gamma_init: float = 0.0,
        gate_eta_max: float = 0.5,
    ):
        super().__init__()

        assert num_channels == 3

        self.num_channels = num_channels

        self.edge_gate = ResidualEdgeGate(
            num_channels=num_channels,
            num_nodes=num_nodes,
            eta_max=gate_eta_max,
            rho_init=gate_gamma_init,
        )

        self.encoders = nn.ModuleList([
            SingleChannelGCNEncoder(num_nodes, hidden_dim, dropout)
            for _ in range(num_channels)
        ])

        self.cross_info = NodeWiseCrossInformationInteraction(
            hidden_dim=hidden_dim,
            num_channels=num_channels,
            dropout=dropout,
        )

        self.readouts = nn.ModuleList([
            ChannelBranchReadout(hidden_dim, num_groups, dropout)
            for _ in range(num_channels)
        ])

        # Each branch outputs mean + max + ROI-attention + group_mean + group_max = 5H.
        branch_dim = 5 * hidden_dim

        self.channel_attention = ChannelAttentionFusion(branch_dim, dropout)

        # concat three branches: 3 * branch_dim
        # attention fused: branch_dim
        clf_in_dim = 4 * branch_dim

        self.classifier = nn.Sequential(
            nn.Linear(clf_in_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        x: [B, 3, N, N]
        """

        x_gated, raw_gate, eta, gate_modulation, edge_delta, global_M = self.edge_gate(x)

        h_list = []
        for c in range(self.num_channels):
            # GCN message-passing adjacency: residual-gated R/U/S graph.
            adj_c = x_gated[:, c]

            # Initial node features: original input R/U/S matrix before edge gating.
            # Each ROI uses its original connectivity profile as node features.
            node_feat_c = x[:, c]

            h_c = self.encoders[c](
                adj_c=adj_c,
                node_feat_c=node_feat_c,
            )
            h_list.append(h_c)

        h_refined_list, cross_attn, cross_strength = self.cross_info(h_list)

        branch_list = []
        roi_alpha_list = []
        group_assign_list = []

        for c in range(self.num_channels):
            emb_c, roi_alpha_c, group_assign_c = self.readouts[c](h_refined_list[c])
            branch_list.append(emb_c)
            roi_alpha_list.append(roi_alpha_c)
            group_assign_list.append(group_assign_c)

        # [B, 3, D]
        branch_stack = torch.stack(branch_list, dim=1)

        # [B, D], [B, 3]
        att_fused, beta = self.channel_attention(branch_stack)

        concat_emb = torch.cat(branch_list, dim=-1)
        final_emb = torch.cat([concat_emb, att_fused], dim=-1)

        logits = self.classifier(final_emb)

        roi_alpha = torch.stack(roi_alpha_list, dim=1)          # [B, 3, N]
        group_assign = torch.stack(group_assign_list, dim=1)    # [B, 3, N, K]

        if return_attention:
            return logits, {
                "beta": beta,
                "roi_alpha": roi_alpha,
                "group_assign": group_assign,
                "cross_alpha": cross_strength,  # [B, 3, 3], target-source message strength
                "cross_attn": cross_attn,        # [B, 3, 3, N, N], cross-attention matrices
                "raw_gate": raw_gate,
                "gate_modulation": gate_modulation,
                "edge_delta": edge_delta,
                "global_M": global_M,
                "eta": eta,
                # Keep the old key name for compatibility with the existing
                # training/evaluation print logic. It now denotes eta^C.
                "gamma": eta,
                "x_gated": x_gated,
            }

        return logits


# ============================================================
# 4. Train / evaluate
# ============================================================

def make_class_weight(y_train: np.ndarray, device: torch.device) -> torch.Tensor:
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    full_weights = np.ones(int(classes.max()) + 1, dtype=np.float32)

    for cls, w in zip(classes, weights):
        full_weights[int(cls)] = w

    return torch.tensor(full_weights, dtype=torch.float32, device=device)


def gate_deviation_loss(raw_gate: torch.Tensor) -> torch.Tensor:
    """
    Keep off-diagonal raw gate values close to 0.5 if regularization is used.

    raw_gate: [B, 3, N, N]

    The diagonal entries are structurally zero in the paper-style gate, so they
    should not be included in this regularization term.
    """
    B, C, N, _ = raw_gate.shape
    eye = torch.eye(N, device=raw_gate.device, dtype=raw_gate.dtype).view(1, 1, N, N)
    mask = 1.0 - eye
    loss = ((raw_gate - 0.52) ** 2) * mask
    return loss.sum() / mask.sum().clamp_min(1.0) / B / C


def cross_entropy_balance_loss(cross_alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Optional entropy regularization for mutual cross-attention message strengths.

    cross_alpha: [B, C, C]
        cross_alpha[:, tgt, src] is the message strength from source channel src
        to target channel tgt. The diagonal is zero.

    This loss normalizes the two source-channel strengths for each target channel
    and returns negative entropy. Minimizing it with a positive lambda encourages
    each target channel not to collapse to only one source channel too early.
    """
    B, C, _ = cross_alpha.shape
    device = cross_alpha.device
    dtype = cross_alpha.dtype

    mask = (1.0 - torch.eye(C, device=device, dtype=dtype)).unsqueeze(0)
    strength = cross_alpha * mask

    prob = strength / strength.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -torch.sum(prob * torch.log(prob + eps) * mask, dim=-1)  # [B, C]
    return -entropy.mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> Dict:
    model.eval()

    y_true_list = []
    y_pred_list = []
    y_prob_list = []
    beta_list = []
    roi_alpha_list = []
    group_assign_list = []
    cross_alpha_list = []
    gate_mean_list = []
    gate_modulation_mean_list = []
    edge_delta_mean_list = []
    x_gated_mean_list = []
    gamma_list = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits, aux = model(x, return_attention=True)

        prob = F.softmax(logits, dim=-1)
        pred = prob.argmax(dim=-1)

        y_true_list.append(y.detach().cpu().numpy())
        y_pred_list.append(pred.detach().cpu().numpy())
        y_prob_list.append(prob.detach().cpu().numpy())

        beta_list.append(aux["beta"].detach().cpu().numpy())
        roi_alpha_list.append(aux["roi_alpha"].detach().cpu().numpy())
        group_assign_list.append(aux["group_assign"].detach().cpu().numpy())
        cross_alpha_list.append(aux["cross_alpha"].detach().cpu().numpy())
        gate_mean_list.append(aux["raw_gate"].detach().cpu().numpy().mean(axis=0))  # [3,N,N]
        gate_modulation_mean_list.append(aux["gate_modulation"].detach().cpu().numpy().mean(axis=0))  # [3,N,N]
        edge_delta_mean_list.append(aux["edge_delta"].detach().cpu().numpy().mean(axis=0))  # [3,N,N]
        x_gated_mean_list.append(aux["x_gated"].detach().cpu().numpy().mean(axis=0))  # [3,N,N]
        gamma_list.append(aux["gamma"].detach().cpu().numpy())

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    y_prob = np.concatenate(y_prob_list)

    beta_all = np.concatenate(beta_list, axis=0)                    # [B, 3]
    roi_alpha_all = np.concatenate(roi_alpha_list, axis=0)          # [B, 3, N]
    group_assign_all = np.concatenate(group_assign_list, axis=0)    # [B, 3, N, K]
    cross_alpha_all = np.concatenate(cross_alpha_list, axis=0)      # [B, 3, 3]

    gate_mean = np.mean(np.stack(gate_mean_list, axis=0), axis=0)   # [3,N,N]
    gate_modulation_mean = np.mean(np.stack(gate_modulation_mean_list, axis=0), axis=0)  # [3,N,N]
    edge_delta_mean = np.mean(np.stack(edge_delta_mean_list, axis=0), axis=0)  # [3,N,N]
    x_gated_mean = np.mean(np.stack(x_gated_mean_list, axis=0), axis=0)  # [3,N,N]
    gamma_mean = np.mean(np.stack(gamma_list, axis=0), axis=0)      # [3], eta in paper notation

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    auc = np.nan
    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
    except Exception:
        pass

    return {
        "acc": acc,
        "prec": prec,
        "recall": rec,
        "f1": f1,
        "macro_f1": macro_f1,
        "auc": auc,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "beta": beta_all,
        "roi_alpha": roi_alpha_all,
        "group_assign": group_assign_all,
        "cross_alpha": cross_alpha_all,
        "gate_mean": gate_mean,
        "gate_modulation_mean": gate_modulation_mean,
        "edge_delta_mean": edge_delta_mean,
        "x_gated_mean": x_gated_mean,
        "eta_mean": gamma_mean,
        "gamma_mean": gamma_mean,
    }


@torch.no_grad()
def save_test_subject_edge_gate(
    model: nn.Module,
    dataset: Dataset,
    test_idx: np.ndarray,
    fold_id: int,
    cfg: Config,
    device: torch.device,
    save_root: Optional[str] = None,
):
    """
    Save subject-level residual edge-gate outputs on the held-out test set.

    Saved for each subject:
        input_adj        : A_i^C, shape [3, N, N]
        gated_adj        : A_tilde_i^C, shape [3, N, N]
        raw_gate         : G_i^C, shape [3, N, N]
        gate_modulation  : 1 + eta^C(2G_i^C - 1), shape [3, N, N]
        edge_delta       : A_tilde_i^C - A_i^C, shape [3, N, N]
        eta              : eta^C, shape [3]
        global_M         : M^C, shape [3, N, N]

    It also writes compact CSV files containing the top enhanced and suppressed
    edges for each test subject. These CSV files are easier to inspect than a
    full edge-level long table.
    """

    model.eval()

    if save_root is None:
        save_root = os.path.join(os.getcwd(), "edge_gate_test_subjects")

    fold_dir = os.path.join(save_root, f"fold_{fold_id:02d}")
    os.makedirs(fold_dir, exist_ok=True)

    indexed_test_dataset = IndexedSubset(dataset, test_idx)
    loader = DataLoader(
        indexed_test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    channel_names = ["R", "U", "S"]
    topk_edges_to_csv = 30
    rows = []
    saved_count = 0

    for x, y, original_idx in loader:
        x = x.to(device)
        y = y.to(device)

        logits, aux = model(x, return_attention=True)
        prob = F.softmax(logits, dim=-1)
        pred = prob.argmax(dim=-1)

        input_adj = x.detach().cpu().numpy()                         # [B, 3, N, N]
        gated_adj = aux["x_gated"].detach().cpu().numpy()             # [B, 3, N, N]
        raw_gate = aux["raw_gate"].detach().cpu().numpy()             # [B, 3, N, N]
        modulation = aux["gate_modulation"].detach().cpu().numpy()    # [B, 3, N, N]
        edge_delta = aux["edge_delta"].detach().cpu().numpy()         # [B, 3, N, N]
        eta = aux["eta"].detach().cpu().numpy()                       # [3]
        global_M = aux["global_M"].detach().cpu().numpy()             # [3, N, N]

        y_np = y.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        prob_np = prob.detach().cpu().numpy()
        idx_np = original_idx.detach().cpu().numpy()

        for b in range(input_adj.shape[0]):
            save_dict = {
                "subject_index": np.array(idx_np[b], dtype=np.int64),
                "fold_id": np.array(fold_id, dtype=np.int64),
                "true_label": np.array(y_np[b], dtype=np.int64),
                "pred_label": np.array(pred_np[b], dtype=np.int64),
                "correct": np.array(y_np[b] == pred_np[b], dtype=np.int64),
                "prob": prob_np[b].astype(np.float32),
                "input_adj": input_adj[b].astype(np.float32),
                "gated_adj": gated_adj[b].astype(np.float32),
                "raw_gate": raw_gate[b].astype(np.float32),
                "gate_modulation": modulation[b].astype(np.float32),
                "edge_delta": edge_delta[b].astype(np.float32),
                "eta": eta.astype(np.float32),
                "global_M": global_M.astype(np.float32),
                "channel_names": np.array(channel_names),
            }

            for c, name in enumerate(channel_names):
                save_dict[f"input_adj_{name}"] = input_adj[b, c].astype(np.float32)
                save_dict[f"gated_adj_{name}"] = gated_adj[b, c].astype(np.float32)
                save_dict[f"raw_gate_{name}"] = raw_gate[b, c].astype(np.float32)
                save_dict[f"gate_modulation_{name}"] = modulation[b, c].astype(np.float32)
                save_dict[f"edge_delta_{name}"] = edge_delta[b, c].astype(np.float32)

            file_name = (
                f"fold_{fold_id:02d}"
                f"_sub_{int(idx_np[b]):04d}"
                f"_true_{int(y_np[b])}"
                f"_pred_{int(pred_np[b])}"
                f"_edge_gate.npz"
            )
            save_path = os.path.join(fold_dir, file_name)
            np.savez_compressed(save_path, **save_dict)
            saved_count += 1

            # Compact subject-level top-edge CSV.
            N = edge_delta.shape[-1]
            triu = np.triu_indices(N, k=1)
            for c, name in enumerate(channel_names):
                vals = edge_delta[b, c, triu[0], triu[1]]
                gate_vals = raw_gate[b, c, triu[0], triu[1]]
                mod_vals = modulation[b, c, triu[0], triu[1]]
                in_vals = input_adj[b, c, triu[0], triu[1]]
                out_vals = gated_adj[b, c, triu[0], triu[1]]

                enhanced_order = np.argsort(-vals)[:topk_edges_to_csv]
                suppressed_order = np.argsort(vals)[:topk_edges_to_csv]

                for direction, order in [
                    ("enhanced", enhanced_order),
                    ("suppressed", suppressed_order),
                ]:
                    for rank, pos in enumerate(order, start=1):
                        rows.append({
                            "fold_id": int(fold_id),
                            "subject_index": int(idx_np[b]),
                            "true_label": int(y_np[b]),
                            "pred_label": int(pred_np[b]),
                            "correct": int(y_np[b] == pred_np[b]),
                            "channel": name,
                            "direction": direction,
                            "rank": int(rank),
                            "roi_p": int(triu[0][pos]),
                            "roi_q": int(triu[1][pos]),
                            "input_weight": float(in_vals[pos]),
                            "gated_weight": float(out_vals[pos]),
                            "edge_delta": float(vals[pos]),
                            "raw_gate": float(gate_vals[pos]),
                            "gate_modulation": float(mod_vals[pos]),
                            "eta": float(eta[c]),
                        })

    if len(rows) > 0:
        fieldnames = list(rows[0].keys())

        fold_csv_path = os.path.join(fold_dir, f"fold_{fold_id:02d}_edge_gate_top_edges.csv")
        with open(fold_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        aggregate_csv_path = os.path.join(save_root, "all_folds_edge_gate_top_edges.csv")
        aggregate_exists = os.path.exists(aggregate_csv_path)
        with open(aggregate_csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not aggregate_exists:
                writer.writeheader()
            writer.writerows(rows)

        print(
            f"[Fold {fold_id:02d}] Saved edge-gate top-edge CSV to:\n"
            f"  {fold_csv_path}"
        )
        print(
            f"[Fold {fold_id:02d}] Updated aggregate edge-gate CSV:\n"
            f"  {aggregate_csv_path}"
        )

    print(
        f"[Fold {fold_id:02d}] Saved {saved_count} test-subject "
        f"edge-gate files to:\n  {fold_dir}"
    )
@torch.no_grad()
def save_test_subject_group_assignment(
    model: nn.Module,
    dataset: Dataset,
    test_idx: np.ndarray,
    fold_id: int,
    cfg: Config,
    device: torch.device,
    save_root: Optional[str] = None,
):
    """
    Save subject-level soft node-group assignment matrices on the held-out test set.

    Saved assignment:
        group_assign: [3, N, K]

    Channel indices:
        0 = R, redundancy
        1 = U, uniqueness
        2 = S, synergy

    For each subject:
        group_assign_R: [N, K]
        group_assign_U: [N, K]
        group_assign_S: [N, K]

    hard_group_R/U/S:
        argmax over K for each node, shape [N]
    """

    model.eval()

    if save_root is None:
        save_root = os.path.join(os.getcwd(), "group_assignment_test_subjects")

    fold_dir = os.path.join(save_root, f"fold_{fold_id:02d}")
    os.makedirs(fold_dir, exist_ok=True)

    indexed_test_dataset = IndexedSubset(dataset, test_idx)
    loader = DataLoader(
        indexed_test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    channel_names = ["R", "U", "S"]
    rows = []
    saved_count = 0

    for x, y, original_idx in loader:
        x = x.to(device)
        y = y.to(device)

        logits, aux = model(x, return_attention=True)
        prob = F.softmax(logits, dim=-1)
        pred = prob.argmax(dim=-1)

        group_assign = aux["group_assign"].detach().cpu().numpy()  # [B, 3, N, K]

        y_np = y.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        prob_np = prob.detach().cpu().numpy()
        idx_np = original_idx.detach().cpu().numpy()

        for b in range(group_assign.shape[0]):
            assign_b = group_assign[b].astype(np.float32)  # [3, N, K]
            hard_b = np.argmax(assign_b, axis=-1).astype(np.int64)  # [3, N]

            save_dict = {
                "subject_index": np.array(idx_np[b], dtype=np.int64),
                "fold_id": np.array(fold_id, dtype=np.int64),
                "true_label": np.array(y_np[b], dtype=np.int64),
                "pred_label": np.array(pred_np[b], dtype=np.int64),
                "correct": np.array(y_np[b] == pred_np[b], dtype=np.int64),
                "prob": prob_np[b].astype(np.float32),

                "group_assign": assign_b,      # [3, N, K]
                "hard_group": hard_b,          # [3, N]

                "group_assign_R": assign_b[0],
                "group_assign_U": assign_b[1],
                "group_assign_S": assign_b[2],

                "hard_group_R": hard_b[0],
                "hard_group_U": hard_b[1],
                "hard_group_S": hard_b[2],

                "channel_names": np.array(channel_names),
            }

            file_name = (
                f"fold_{fold_id:02d}"
                f"_sub_{int(idx_np[b]):04d}"
                f"_true_{int(y_np[b])}"
                f"_pred_{int(pred_np[b])}"
                f"_group_assignment.npz"
            )

            save_path = os.path.join(fold_dir, file_name)
            np.savez_compressed(save_path, **save_dict)
            saved_count += 1

            N = assign_b.shape[1]
            K = assign_b.shape[2]

            for c, cname in enumerate(channel_names):
                for roi in range(N):
                    row = {
                        "fold_id": int(fold_id),
                        "subject_index": int(idx_np[b]),
                        "true_label": int(y_np[b]),
                        "pred_label": int(pred_np[b]),
                        "correct": int(y_np[b] == pred_np[b]),
                        "channel": cname,
                        "roi_index": int(roi),
                        "hard_group": int(hard_b[c, roi]),
                    }

                    for k in range(K):
                        row[f"group_prob_{k}"] = float(assign_b[c, roi, k])

                    rows.append(row)

    if len(rows) > 0:
        fieldnames = list(rows[0].keys())

        fold_csv_path = os.path.join(
            fold_dir,
            f"fold_{fold_id:02d}_group_assignment_long.csv"
        )

        with open(fold_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        aggregate_csv_path = os.path.join(save_root, "all_folds_group_assignment_long.csv")
        aggregate_exists = os.path.exists(aggregate_csv_path)

        with open(aggregate_csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not aggregate_exists:
                writer.writeheader()
            writer.writerows(rows)

        print(
            f"[Fold {fold_id:02d}] Saved fold-level group-assignment CSV to:\n"
            f"  {fold_csv_path}"
        )

        print(
            f"[Fold {fold_id:02d}] Updated aggregate group-assignment CSV:\n"
            f"  {aggregate_csv_path}"
        )

    print(
        f"[Fold {fold_id:02d}] Saved {saved_count} test-subject "
        f"group-assignment files to:\n  {fold_dir}"
    )
@torch.no_grad()
def save_test_subject_cross_attention(
    model: nn.Module,
    dataset: Dataset,
    test_idx: np.ndarray,
    fold_id: int,
    cfg: Config,
    device: torch.device,
    save_root: Optional[str] = None,
):
    """
    Save the six subject-level cross-information attention matrices on the
    test set for one fold.

    The returned cross-attention tensor of the model has shape:
        [B, 3, 3, N, N]

    Channel indices:
        0 = R, redundancy
        1 = U, uniqueness
        2 = S, synergy

    Saved matrices for each subject:
        R_from_U: target R attends to source U
        R_from_S: target R attends to source S
        U_from_R: target U attends to source R
        U_from_S: target U attends to source S
        S_from_R: target S attends to source R
        S_from_U: target S attends to source U

    Each saved matrix has shape [N, N]. Rows are target-channel ROIs and
    columns are source-channel ROIs.
    """

    model.eval()

    if save_root is None:
        save_root = os.path.join(os.getcwd(), "cross_attention_test_subjects")

    fold_dir = os.path.join(save_root, f"fold_{fold_id:02d}")
    os.makedirs(fold_dir, exist_ok=True)

    indexed_test_dataset = IndexedSubset(dataset, test_idx)
    loader = DataLoader(
        indexed_test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    channel_names = ["R", "U", "S"]
    relation_index = {
        "R_from_U": (0, 1),
        "R_from_S": (0, 2),
        "U_from_R": (1, 0),
        "U_from_S": (1, 2),
        "S_from_R": (2, 0),
        "S_from_U": (2, 1),
    }

    saved_count = 0

    for x, y, original_idx in loader:
        x = x.to(device)
        y = y.to(device)

        logits, aux = model(x, return_attention=True)
        prob = F.softmax(logits, dim=-1)
        pred = prob.argmax(dim=-1)

        cross_attn = aux["cross_attn"].detach().cpu().numpy()  # [B, 3, 3, N, N]
        y_np = y.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        prob_np = prob.detach().cpu().numpy()
        idx_np = original_idx.detach().cpu().numpy()

        for b in range(cross_attn.shape[0]):
            save_dict = {
                "subject_index": np.array(idx_np[b], dtype=np.int64),
                "fold_id": np.array(fold_id, dtype=np.int64),
                "true_label": np.array(y_np[b], dtype=np.int64),
                "pred_label": np.array(pred_np[b], dtype=np.int64),
                "prob": prob_np[b].astype(np.float32),
                "channel_names": np.array(channel_names),
            }

            for rel_name, (tgt, src) in relation_index.items():
                save_dict[rel_name] = cross_attn[b, tgt, src].astype(np.float32)

            file_name = (
                f"fold_{fold_id:02d}"
                f"_sub_{int(idx_np[b]):04d}"
                f"_true_{int(y_np[b])}"
                f"_pred_{int(pred_np[b])}.npz"
            )
            save_path = os.path.join(fold_dir, file_name)
            np.savez_compressed(save_path, **save_dict)
            saved_count += 1

    print(
        f"[Fold {fold_id:02d}] Saved {saved_count} test-subject "
        f"cross-attention files to:\n  {fold_dir}"
    )


@torch.no_grad()
def save_test_subject_channel_attention(
    model: nn.Module,
    dataset: Dataset,
    test_idx: np.ndarray,
    fold_id: int,
    cfg: Config,
    device: torch.device,
    save_root: Optional[str] = None,
):
    """
    Save subject-level channel-attention weights beta_R, beta_U, beta_S
    on the held-out test set for one fold.

    This is used for information-channel contribution analysis:

        beta_i^C, C in {R, U, S}

    Channel indices:
        0 = R, redundancy
        1 = U, uniqueness
        2 = S, synergy

    Saved files:
        1) One .npz file for each test subject:
           fold_XX/fold_XX_sub_XXXX_true_Y_pred_Z_channel_attention.npz

        2) One fold-level CSV:
           fold_XX/fold_XX_channel_attention.csv

        3) One aggregate CSV across folds:
           all_folds_channel_attention.csv

    The aggregate CSV is convenient for the independent violin-plot script.
    """

    model.eval()

    if save_root is None:
        save_root = os.path.join(os.getcwd(), "channel_attention_test_subjects")

    fold_dir = os.path.join(save_root, f"fold_{fold_id:02d}")
    os.makedirs(fold_dir, exist_ok=True)

    indexed_test_dataset = IndexedSubset(dataset, test_idx)
    loader = DataLoader(
        indexed_test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    channel_names = ["R", "U", "S"]
    rows = []
    saved_count = 0

    for x, y, original_idx in loader:
        x = x.to(device)
        y = y.to(device)

        logits, aux = model(x, return_attention=True)
        prob = F.softmax(logits, dim=-1)
        pred = prob.argmax(dim=-1)

        beta = aux["beta"].detach().cpu().numpy()  # [B, 3]
        y_np = y.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        prob_np = prob.detach().cpu().numpy()
        idx_np = original_idx.detach().cpu().numpy()

        for b in range(beta.shape[0]):
            beta_b = beta[b].astype(np.float32)

            row = {
                "fold_id": int(fold_id),
                "subject_index": int(idx_np[b]),
                "true_label": int(y_np[b]),
                "pred_label": int(pred_np[b]),
                "correct": int(y_np[b] == pred_np[b]),
                "beta_R": float(beta_b[0]),
                "beta_U": float(beta_b[1]),
                "beta_S": float(beta_b[2]),
            }

            # Save class probabilities as prob_0, prob_1, ...
            for c in range(prob_np.shape[1]):
                row[f"prob_{c}"] = float(prob_np[b, c])

            rows.append(row)

            save_dict = {
                "subject_index": np.array(idx_np[b], dtype=np.int64),
                "fold_id": np.array(fold_id, dtype=np.int64),
                "true_label": np.array(y_np[b], dtype=np.int64),
                "pred_label": np.array(pred_np[b], dtype=np.int64),
                "correct": np.array(y_np[b] == pred_np[b], dtype=np.int64),
                "prob": prob_np[b].astype(np.float32),
                "beta": beta_b,  # [3]
                "beta_R": np.array(beta_b[0], dtype=np.float32),
                "beta_U": np.array(beta_b[1], dtype=np.float32),
                "beta_S": np.array(beta_b[2], dtype=np.float32),
                "channel_names": np.array(channel_names),
            }

            file_name = (
                f"fold_{fold_id:02d}"
                f"_sub_{int(idx_np[b]):04d}"
                f"_true_{int(y_np[b])}"
                f"_pred_{int(pred_np[b])}"
                f"_channel_attention.npz"
            )
            save_path = os.path.join(fold_dir, file_name)
            np.savez_compressed(save_path, **save_dict)
            saved_count += 1

    if len(rows) > 0:
        fieldnames = list(rows[0].keys())

        fold_csv_path = os.path.join(fold_dir, f"fold_{fold_id:02d}_channel_attention.csv")
        with open(fold_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        aggregate_csv_path = os.path.join(save_root, "all_folds_channel_attention.csv")
        aggregate_exists = os.path.exists(aggregate_csv_path)
        with open(aggregate_csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not aggregate_exists:
                writer.writeheader()
            writer.writerows(rows)

        print(
            f"[Fold {fold_id:02d}] Saved fold-level channel-attention CSV to:\n"
            f"  {fold_csv_path}"
        )
        print(
            f"[Fold {fold_id:02d}] Updated aggregate channel-attention CSV:\n"
            f"  {aggregate_csv_path}"
        )

    print(
        f"[Fold {fold_id:02d}] Saved {saved_count} test-subject "
        f"channel-attention files to:\n  {fold_dir}"
    )


@torch.no_grad()
def save_test_subject_roi_attention(
    model: nn.Module,
    dataset: Dataset,
    test_idx: np.ndarray,
    fold_id: int,
    cfg: Config,
    device: torch.device,
    save_root: Optional[str] = None,
):
    """
    Save subject-level ROI attention weights alpha_R, alpha_U, alpha_S
    on the held-out test set for one fold.

    This is used for brain-region importance analysis:

        alpha_{i,p}^C, C in {R, U, S}

    Channel indices:
        0 = R, redundancy
        1 = U, uniqueness
        2 = S, synergy

    Saved files:
        1) One .npz file for each test subject:
           fold_XX/fold_XX_sub_XXXX_true_Y_pred_Z_roi_attention.npz

        2) One fold-level long-format CSV:
           fold_XX/fold_XX_roi_attention_long.csv

        3) One aggregate long-format CSV across folds:
           all_folds_roi_attention_long.csv

    Long-format CSV columns:
        fold_id, subject_index, true_label, pred_label, correct,
        roi_index, alpha_R, alpha_U, alpha_S,
        beta_R, beta_U, beta_S, prob_0, prob_1, prob_2
    """

    model.eval()

    if save_root is None:
        save_root = os.path.join(os.getcwd(), "roi_attention_test_subjects")

    fold_dir = os.path.join(save_root, f"fold_{fold_id:02d}")
    os.makedirs(fold_dir, exist_ok=True)

    indexed_test_dataset = IndexedSubset(dataset, test_idx)
    loader = DataLoader(
        indexed_test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    channel_names = ["R", "U", "S"]
    rows = []
    saved_count = 0

    for x, y, original_idx in loader:
        x = x.to(device)
        y = y.to(device)

        logits, aux = model(x, return_attention=True)
        prob = F.softmax(logits, dim=-1)
        pred = prob.argmax(dim=-1)

        roi_alpha = aux["roi_alpha"].detach().cpu().numpy()  # [B, 3, N]
        beta = aux["beta"].detach().cpu().numpy()            # [B, 3]

        y_np = y.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        prob_np = prob.detach().cpu().numpy()
        idx_np = original_idx.detach().cpu().numpy()

        for b in range(roi_alpha.shape[0]):
            alpha_b = roi_alpha[b].astype(np.float32)  # [3, N]
            beta_b = beta[b].astype(np.float32)

            row_base = {
                "fold_id": int(fold_id),
                "subject_index": int(idx_np[b]),
                "true_label": int(y_np[b]),
                "pred_label": int(pred_np[b]),
                "correct": int(y_np[b] == pred_np[b]),
                "beta_R": float(beta_b[0]),
                "beta_U": float(beta_b[1]),
                "beta_S": float(beta_b[2]),
            }

            for c in range(prob_np.shape[1]):
                row_base[f"prob_{c}"] = float(prob_np[b, c])

            save_dict = {
                "subject_index": np.array(idx_np[b], dtype=np.int64),
                "fold_id": np.array(fold_id, dtype=np.int64),
                "true_label": np.array(y_np[b], dtype=np.int64),
                "pred_label": np.array(pred_np[b], dtype=np.int64),
                "correct": np.array(y_np[b] == pred_np[b], dtype=np.int64),
                "prob": prob_np[b].astype(np.float32),

                # ROI attention: [3, N]
                "roi_alpha": alpha_b,
                "alpha_R": alpha_b[0],
                "alpha_U": alpha_b[1],
                "alpha_S": alpha_b[2],

                # Channel attention beta: [3]
                "beta": beta_b,
                "beta_R": np.array(beta_b[0], dtype=np.float32),
                "beta_U": np.array(beta_b[1], dtype=np.float32),
                "beta_S": np.array(beta_b[2], dtype=np.float32),

                "channel_names": np.array(channel_names),
            }

            file_name = (
                f"fold_{fold_id:02d}"
                f"_sub_{int(idx_np[b]):04d}"
                f"_true_{int(y_np[b])}"
                f"_pred_{int(pred_np[b])}"
                f"_roi_attention.npz"
            )
            save_path = os.path.join(fold_dir, file_name)
            np.savez_compressed(save_path, **save_dict)
            saved_count += 1

            # Long-format CSV: one row per subject per ROI.
            N = alpha_b.shape[1]
            for roi in range(N):
                row = dict(row_base)
                row.update({
                    "roi_index": int(roi),
                    "alpha_R": float(alpha_b[0, roi]),
                    "alpha_U": float(alpha_b[1, roi]),
                    "alpha_S": float(alpha_b[2, roi]),
                })
                rows.append(row)

    if len(rows) > 0:
        fieldnames = list(rows[0].keys())

        fold_csv_path = os.path.join(
            fold_dir,
            f"fold_{fold_id:02d}_roi_attention_long.csv"
        )
        with open(fold_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        aggregate_csv_path = os.path.join(save_root, "all_folds_roi_attention_long.csv")
        aggregate_exists = os.path.exists(aggregate_csv_path)
        with open(aggregate_csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not aggregate_exists:
                writer.writeheader()
            writer.writerows(rows)

        print(
            f"[Fold {fold_id:02d}] Saved fold-level ROI-attention CSV to:\n"
            f"  {fold_csv_path}"
        )
        print(
            f"[Fold {fold_id:02d}] Updated aggregate ROI-attention CSV:\n"
            f"  {aggregate_csv_path}"
        )

    print(
        f"[Fold {fold_id:02d}] Saved {saved_count} test-subject "
        f"ROI-attention files to:\n  {fold_dir}"
    )


def print_metrics(prefix: str, metrics: Dict):
    auc_str = "nan" if np.isnan(metrics["auc"]) else f"{metrics['auc']:.4f}"
    print(
        f"{prefix} | "
        f"Acc={metrics['acc']:.4f}, "
        f"Prec={metrics['prec']:.4f}, "
        f"Recall={metrics['recall']:.4f}, "
        f"F1={metrics['f1']:.4f}, "
        f"MacroF1={metrics['macro_f1']:.4f}, "
        f"AUC={auc_str}"
    )


def get_top_rois_from_alpha(roi_alpha: np.ndarray, topk: int = 10):
    """
    roi_alpha: [B, 3, N]
    """
    mean_alpha = roi_alpha.mean(axis=0)  # [3, N]
    top_dict = {}
    names = ["R", "U", "S"]

    for c, name in enumerate(names):
        idx = np.argsort(-mean_alpha[c])[:topk]
        vals = mean_alpha[c, idx]
        top_dict[name] = (idx, vals)

    return top_dict, mean_alpha


def get_group_assignment_summary(group_assign: np.ndarray, topk_rois: int = 8):
    """
    group_assign: [B, 3, N, K]

    For each channel and each group, return top ROI indices by mean assignment.
    """
    mean_assign = group_assign.mean(axis=0)  # [3, N, K]
    mean_assign_g = np.transpose(mean_assign, (0, 2, 1))  # [3, K, N]

    names = ["R", "U", "S"]
    summary = {}

    for c, name in enumerate(names):
        channel_groups = []
        K = mean_assign_g.shape[1]
        for k in range(K):
            vals = mean_assign_g[c, k]
            idx = np.argsort(-vals)[:topk_rois]
            channel_groups.append((k, idx, vals[idx]))
        summary[name] = channel_groups

    return summary, mean_assign


def get_top_edges_from_gate(gate_mean: np.ndarray, topk: int = 10):
    """
    gate_mean: [3, N, N]

    Return top edges per channel by raw gate value.
    """
    C, N, _ = gate_mean.shape
    triu = np.triu_indices(N, k=1)
    names = ["R", "U", "S"]

    top_dict = {}

    for c, name in enumerate(names):
        vals = gate_mean[c, triu[0], triu[1]]
        order = np.argsort(-vals)[:topk]
        edges = list(zip(triu[0][order], triu[1][order], vals[order]))
        top_dict[name] = edges

    return top_dict




def get_top_edges_from_delta(edge_delta_mean: np.ndarray, topk: int = 10):
    """
    edge_delta_mean: [3, N, N]

    Return top enhanced and suppressed edges per channel by signed edge
    modulation Delta = A_tilde - A.
    """
    C, N, _ = edge_delta_mean.shape
    triu = np.triu_indices(N, k=1)
    names = ["R", "U", "S"]

    out = {}
    for c, name in enumerate(names):
        vals = edge_delta_mean[c, triu[0], triu[1]]
        enhanced_order = np.argsort(-vals)[:topk]
        suppressed_order = np.argsort(vals)[:topk]
        out[name] = {
            "enhanced": list(zip(triu[0][enhanced_order], triu[1][enhanced_order], vals[enhanced_order])),
            "suppressed": list(zip(triu[0][suppressed_order], triu[1][suppressed_order], vals[suppressed_order])),
        }

    return out

def get_cross_info_summary(cross_alpha: np.ndarray):
    """
    Summarize mutual cross-attention message strengths.

    cross_alpha: [B, 3, 3]
        cross_alpha[:, tgt, src] denotes the mean strength of the message
        from source channel src to target channel tgt. The diagonal is zero.

    Return:
        source_mean: [3]
            average contribution of each source channel to other target channels.
        pair_mean: [3, 3]
            target-source message strength matrix averaged over subjects.
    """
    pair_mean = cross_alpha.mean(axis=0)  # [3, 3], target x source

    # Average over target channels for each source channel, excluding diagonal.
    C = pair_mean.shape[0]
    mask = 1.0 - np.eye(C, dtype=np.float32)
    source_sum = (pair_mean * mask).sum(axis=0)
    source_count = mask.sum(axis=0).clip(min=1.0)
    source_mean = source_sum / source_count

    return source_mean, pair_mean


def train_one_fold(
    dataset: Dataset,
    labels_np: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    fold_id: int,
    cfg: Config,
    device: torch.device,
) -> Tuple[Dict, str]:

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = RUSGCN(
        num_nodes=cfg.num_nodes,
        num_channels=cfg.num_channels,
        hidden_dim=cfg.hidden_dim,
        num_groups=cfg.num_groups,
        num_classes=cfg.num_classes,
        dropout=cfg.dropout,
        gate_gamma_init=cfg.gate_gamma_init,
        gate_eta_max=cfg.gate_eta_max,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=8,
        verbose=False,
    )

    class_weight = make_class_weight(labels_np[train_idx], device)

    best_val_macro_f1 = -1.0
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()

        epoch_losses = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits, aux = model(x, return_attention=True)
            loss = F.cross_entropy(logits, y, weight=class_weight)

            if cfg.lambda_gate_deviation > 0:
                loss = loss + cfg.lambda_gate_deviation * gate_deviation_loss(aux["raw_gate"])

            if cfg.lambda_cross_entropy > 0:
                loss = loss + cfg.lambda_cross_entropy * cross_entropy_balance_loss(aux["cross_alpha"])

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))

        # Use validation set for early stopping and learning-rate scheduling.
        val_metrics = evaluate(model, val_loader, device, cfg.num_classes)

        scheduler.step(val_metrics["macro_f1"])

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0:
            beta_mean = val_metrics["beta"].mean(axis=0)
            gamma_mean = val_metrics["gamma_mean"]
            cross_mean, _ = get_cross_info_summary(val_metrics["cross_alpha"])

            print(
                f"Fold {fold_id:02d} | Epoch {epoch:03d} | "
                f"Loss={np.mean(epoch_losses):.4f} | "
                f"Val Acc={val_metrics['acc']:.4f} | "
                f"Val MacroF1={val_metrics['macro_f1']:.4f} | "
                f"Best MacroF1={best_val_macro_f1:.4f}@{best_epoch} | "
                f"Beta[R,U,S]={beta_mean} | "
                f"CrossStrength[source R,U,S]={cross_mean} | "
                f"Eta[R,U,S]={gamma_mean}"
            )

        if patience_counter >= cfg.patience:
            print(
                f"Fold {fold_id:02d} early stopped at epoch {epoch}. "
                f"Best epoch = {best_epoch}."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save subject-level residual edge-gate outputs on the held-out test set.
    # Output directory: ./edge_gate_test_subjects/fold_XX/
    save_test_subject_edge_gate(
        model=model,
        dataset=dataset,
        test_idx=test_idx,
        fold_id=fold_id,
        cfg=cfg,
        device=device,
        save_root=os.path.join(os.getcwd(), "edge_gate_test_subjects"),
    )
    # Save subject-level soft node-group assignment matrices on the held-out test set.
    # Output directory: ./group_assignment_test_subjects/fold_XX/
    save_test_subject_group_assignment(
        model=model,
        dataset=dataset,
        test_idx=test_idx,
        fold_id=fold_id,
        cfg=cfg,
        device=device,
        save_root=os.path.join(os.getcwd(), "group_assignment_test_subjects"),
    )
    # Save subject-level channel-attention weights beta_R, beta_U, beta_S
    # on the held-out test set for information-channel contribution analysis.
    # Output directory: ./channel_attention_test_subjects/fold_XX/
    save_test_subject_channel_attention(
        model=model,
        dataset=dataset,
        test_idx=test_idx,
        fold_id=fold_id,
        cfg=cfg,
        device=device,
        save_root=os.path.join(os.getcwd(), "channel_attention_test_subjects"),
    )

    # Save subject-level ROI-attention weights alpha_R, alpha_U, alpha_S
    # on the held-out test set for brain-region importance analysis.
    # Output directory: ./roi_attention_test_subjects/fold_XX/
    save_test_subject_roi_attention(
        model=model,
        dataset=dataset,
        test_idx=test_idx,
        fold_id=fold_id,
        cfg=cfg,
        device=device,
        save_root=os.path.join(os.getcwd(), "roi_attention_test_subjects"),
    )

    # Save subject-level cross-attention matrices on the held-out test set.
    # Output directory: ./cross_attention_test_subjects/fold_XX/
    save_test_subject_cross_attention(
        model=model,
        dataset=dataset,
        test_idx=test_idx,
        fold_id=fold_id,
        cfg=cfg,
        device=device,
        save_root=os.path.join(os.getcwd(), "cross_attention_test_subjects"),
    )

    train_metrics = evaluate(model, train_loader, device, cfg.num_classes)
    val_metrics = evaluate(model, val_loader, device, cfg.num_classes)
    test_metrics = evaluate(model, test_loader, device, cfg.num_classes)

    print_metrics(f"Fold {fold_id:02d} Train", train_metrics)
    print_metrics(f"Fold {fold_id:02d} Val  ", val_metrics)
    print_metrics(f"Fold {fold_id:02d} Test ", test_metrics)

    beta_mean = test_metrics["beta"].mean(axis=0)
    beta_std = test_metrics["beta"].std(axis=0)

    gamma_mean = test_metrics["gamma_mean"]

    cross_mean, cross_pair_mean = get_cross_info_summary(test_metrics["cross_alpha"])

    print(f"Fold {fold_id:02d} Test Channel Attention Beta Mean [R,U,S]: {beta_mean}")
    print(f"Fold {fold_id:02d} Test Channel Attention Beta Std  [R,U,S]: {beta_std}")
    print(f"Fold {fold_id:02d} Test Cross-Information Alpha Mean [R,U,S]: {cross_mean}")
    print(f"Fold {fold_id:02d} Test Edge Gate Eta [R,U,S]: {gamma_mean}")

    top_roi_dict, roi_alpha_mean = get_top_rois_from_alpha(test_metrics["roi_alpha"], topk=10)

    print("\nTop ROI indices by ROI attention:")
    for name, (idx, vals) in top_roi_dict.items():
        print(f"Channel {name}:")
        print("  ROI indices:", idx)
        print("  Attention  :", vals)

    group_summary, group_assign_mean = get_group_assignment_summary(test_metrics["group_assign"], topk_rois=8)

    print("\nTop ROI indices in each learned group by soft assignment:")
    for name, groups in group_summary.items():
        print(f"Channel {name} groups:")
        for group_id, idx, vals in groups[:5]:
            print(f"  group={group_id}, top_rois={idx}, assign={vals}")

    # Target-source cross-attention message strength.
    print("\nCross-attention message strength matrix (target rows, source columns):")
    names = ["R", "U", "S"]
    print("      source:      R          U          S")
    for t, name in enumerate(names):
        print(
            f"target {name}: "
            f"{cross_pair_mean[t, 0]:.6f}  {cross_pair_mean[t, 1]:.6f}  {cross_pair_mean[t, 2]:.6f}"
        )

    top_edge_dict = get_top_edges_from_gate(test_metrics["gate_mean"], topk=10)

    print("\nTop edges by residual edge gate raw value:")
    for name, edges in top_edge_dict.items():
        print(f"Channel {name}:")
        for i, j, v in edges:
            print(f"  edge=({i}, {j}), gate={v:.6f}")

    top_delta_dict = get_top_edges_from_delta(test_metrics["edge_delta_mean"], topk=10)

    print("\nTop enhanced/suppressed edges by signed modulation Delta = A_tilde - A:")
    for name, directions in top_delta_dict.items():
        print(f"Channel {name} enhanced:")
        for i, j, v in directions["enhanced"]:
            print(f"  edge=({i}, {j}), delta={v:.6f}")
        print(f"Channel {name} suppressed:")
        for i, j, v in directions["suppressed"]:
            print(f"  edge=({i}, {j}), delta={v:.6f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"]))

    print("Pred class counts [AD, CN, MCI]:")
    print(np.bincount(test_metrics["y_pred"], minlength=cfg.num_classes))

    print("True class counts [AD, CN, MCI]:")
    print(np.bincount(test_metrics["y_true"], minlength=cfg.num_classes))

    print(
        classification_report(
            test_metrics["y_true"],
            test_metrics["y_pred"],
            target_names=["AD", "CN", "MCI"],
            zero_division=0,
        )
    )

    os.makedirs(cfg.save_dir, exist_ok=True)
    model_path = os.path.join(cfg.save_dir, f"rus_gcn_fold_{fold_id:02d}.pt")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "test_metrics": {
                k: v for k, v in test_metrics.items()
                if not k.startswith("y_") and k not in [
                    "beta", "roi_alpha", "group_assign", "cross_alpha",
                    "gate_mean", "gate_modulation_mean", "edge_delta_mean",
                    "x_gated_mean", "eta_mean", "gamma_mean"
                ]
            },
            "test_beta_mean": beta_mean,
            "test_beta_std": beta_std,
            "roi_alpha_mean": roi_alpha_mean,
            "group_assign_mean": group_assign_mean,
            "cross_source_strength_mean": cross_mean,
            "cross_pair_strength_mean": cross_pair_mean,
            "gate_mean": test_metrics["gate_mean"],
            "gate_modulation_mean": test_metrics["gate_modulation_mean"],
            "edge_delta_mean": test_metrics["edge_delta_mean"],
            "x_gated_mean": test_metrics["x_gated_mean"],
            "eta_mean": gamma_mean,
            "gamma_mean": gamma_mean,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
        },
        model_path,
    )

    scalar_metrics = {
        "acc": test_metrics["acc"],
        "prec": test_metrics["prec"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "macro_f1": test_metrics["macro_f1"],
        "auc": test_metrics["auc"],
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "beta_R": beta_mean[0],
        "beta_U": beta_mean[1],
        "beta_S": beta_mean[2],
        "cross_R": cross_mean[0],
        "cross_U": cross_mean[1],
        "cross_S": cross_mean[2],
        "eta_R": gamma_mean[0],
        "eta_U": gamma_mean[1],
        "eta_S": gamma_mean[2],
    }

    return scalar_metrics, model_path


def summarize_results(all_fold_metrics: List[Dict]):
    print("\n" + "#" * 80)
    print("Final Cross Validation Results: RUS-GCN")
    print("#" * 80)

    for i, m in enumerate(all_fold_metrics, start=1):
        auc_str = "nan" if np.isnan(m["auc"]) else f"{m['auc']:.4f}"
        print(
            f"Fold {i:02d}: "
            f"Acc={m['acc']:.4f}, "
            f"Prec={m['prec']:.4f}, "
            f"Recall={m['recall']:.4f}, "
            f"F1={m['f1']:.4f}, "
            f"MacroF1={m['macro_f1']:.4f}, "
            f"AUC={auc_str}, "
            f"Beta[R,U,S]=[{m['beta_R']:.4f}, {m['beta_U']:.4f}, {m['beta_S']:.4f}], "
            f"CrossStrength[source R,U,S]=[{m['cross_R']:.4f}, {m['cross_U']:.4f}, {m['cross_S']:.4f}], "
            f"Eta[R,U,S]=[{m['eta_R']:.4f}, {m['eta_U']:.4f}, {m['eta_S']:.4f}], "
            f"BestEpoch={m['best_epoch']}"
        )

    print("\nMean ± Std:")
    keys = [
        "acc", "prec", "recall", "f1", "macro_f1", "auc",
        "beta_R", "beta_U", "beta_S",
        "cross_R", "cross_U", "cross_S",
        "eta_R", "eta_U", "eta_S",
    ]
    for k in keys:
        vals = np.array([m[k] for m in all_fold_metrics], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            print(f"{k:10s}: nan")
        else:
            print(f"{k:10s}: {vals.mean():.4f} ± {vals.std():.4f}")


def run_10fold(cfg: Config):
    set_seed(cfg.seed)

    os.makedirs(cfg.save_dir, exist_ok=True)

    # Clean the aggregate channel-attention CSV from previous runs.
    # Fold-level CSV files will be overwritten fold by fold.
    channel_save_root = os.path.join(os.getcwd(), "channel_attention_test_subjects")
    os.makedirs(channel_save_root, exist_ok=True)
    aggregate_channel_csv = os.path.join(channel_save_root, "all_folds_channel_attention.csv")
    if os.path.exists(aggregate_channel_csv):
        os.remove(aggregate_channel_csv)

    # Clean the aggregate ROI-attention CSV from previous runs.
    # Fold-level CSV files will be overwritten fold by fold.
    roi_save_root = os.path.join(os.getcwd(), "roi_attention_test_subjects")
    os.makedirs(roi_save_root, exist_ok=True)
    aggregate_roi_csv = os.path.join(roi_save_root, "all_folds_roi_attention_long.csv")
    if os.path.exists(aggregate_roi_csv):
        os.remove(aggregate_roi_csv)

    # Clean the aggregate edge-gate CSV from previous runs.
    edge_gate_save_root = os.path.join(os.getcwd(), "edge_gate_test_subjects")
    os.makedirs(edge_gate_save_root, exist_ok=True)
    aggregate_edge_gate_csv = os.path.join(edge_gate_save_root, "all_folds_edge_gate_top_edges.csv")
    if os.path.exists(aggregate_edge_gate_csv):
        os.remove(aggregate_edge_gate_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    x_raw, y = load_all_data(cfg)

    kf = StratifiedKFold(
        n_splits=cfg.n_splits,
        shuffle=True,
        random_state=cfg.seed,
    )

    all_fold_metrics = []
    model_paths = []

    for fold, (train_val_idx, test_idx) in enumerate(kf.split(x_raw, y), start=1):

        if cfg.max_folds_to_run is not None and fold > cfg.max_folds_to_run:
            break

        print("\n" + "=" * 80)
        print(f"Processing Fold {fold}/{cfg.n_splits}")
        print("=" * 80)

        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=cfg.val_ratio_in_trainval,
            stratify=y[train_val_idx],
            random_state=cfg.seed + fold,
        )

        print("Train class counts:", np.bincount(y[train_idx], minlength=cfg.num_classes))
        print("Val class counts  :", np.bincount(y[val_idx], minlength=cfg.num_classes))
        print("Test class counts :", np.bincount(y[test_idx], minlength=cfg.num_classes))

        # Fold-wise scaling using training subjects only.
        x_fold = fold_channel_max_scale(x_raw, train_idx)

        # Optional graph sparsification.
        x_fold = keep_topk_edges(x_fold, cfg.topk_per_node)

        dataset = PEDGraphDataset(x_fold, y)

        metrics, model_path = train_one_fold(
            dataset=dataset,
            labels_np=y,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            fold_id=fold,
            cfg=cfg,
            device=device,
        )

        all_fold_metrics.append(metrics)
        model_paths.append(model_path)

    summarize_results(all_fold_metrics)

    summary_path = os.path.join(cfg.save_dir, "rus_gcn_summary.npz")
    np.savez_compressed(
        summary_path,
        fold_metrics=np.array([
            [
                m["acc"], m["prec"], m["recall"], m["f1"], m["macro_f1"], m["auc"],
                m["beta_R"], m["beta_U"], m["beta_S"],
                m["cross_R"], m["cross_U"], m["cross_S"],
                m["eta_R"], m["eta_U"], m["eta_S"],
            ]
            for m in all_fold_metrics
        ], dtype=np.float32),
        metric_names=np.array([
            "acc", "prec", "recall", "f1", "macro_f1", "auc",
            "beta_R", "beta_U", "beta_S",
            "cross_R", "cross_U", "cross_S",
            "eta_R", "eta_U", "eta_S",
        ]),
        model_paths=np.array(model_paths),
    )

    print("\nSaved summary to:", summary_path)


if __name__ == "__main__":
    run_10fold(CFG)
