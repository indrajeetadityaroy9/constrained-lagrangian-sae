"""Constraint violation functions: faithfulness, drift, orthogonality."""

import torch
from torch import Tensor

from src.constants import EPS_NUM


def compute_faithfulness_violation(
    mahal_sq: Tensor,
    tau_faith: float,
) -> Tensor:
    """Whitened-metric faithfulness violation: E[||x - x_hat||_M^2] - tau."""
    return mahal_sq.mean() - tau_faith


def compute_drift_violation(
    W_dec_A: Tensor,
    W_vocab: Tensor,
    tau_drift: float,
) -> Tensor:
    """Anchored decoder drift: ||W_dec_A - W_vocab||_F^2 - tau_drift."""
    return (W_dec_A - W_vocab).pow(2).sum() - tau_drift


def compute_orthogonality_violation(
    z: Tensor,
    W_dec_A: Tensor,
    W_dec_B: Tensor,
    tau_ortho: float,
    gamma_init_mean: float,
) -> Tensor:
    """Differentiable co-activation orthogonality violation via column sampling.

    Samples k columns to avoid materializing O(F²) Gram matrices.
    Provides an unbiased estimate of the activation-weighted off-diagonal coherence.
    """
    F_total = W_dec_A.shape[1] + W_dec_B.shape[1]
    k = min(F_total, int(F_total**0.5) + 1)

    # Sample column indices (uniform without replacement).
    idx = torch.randperm(F_total, device=W_dec_A.device)[:k]

    V = W_dec_A.shape[1]
    a_idx = idx.clamp(max=V - 1)
    b_idx = (idx - V).clamp(min=0)
    cols_a = W_dec_A[:, a_idx]          # [d, k], differentiable
    cols_b = W_dec_B[:, b_idx]          # [d, k], differentiable
    is_a = (idx < V).unsqueeze(0)       # [1, k] broadcast mask
    W_sub = torch.where(is_a, cols_a, cols_b)  # [d, k], differentiable to both
    col_norms = W_sub.norm(dim=0, keepdim=True)
    W_hat = W_sub / col_norms

    G = W_hat.T @ W_hat          # [k, k]
    G_sq = G.pow(2)

    # Masking inactive features avoids injecting orthogonality gradients through dead units.
    temperature = (2.0 * gamma_init_mean) ** 0.5
    z_sub = z[:, idx]
    active = torch.sigmoid(z_sub / temperature) * (z_sub > 0).float()

    C = active.T @ active         # [k, k]

    weighted = G_sq * C
    numerator = weighted.sum() - weighted.diagonal().sum()
    denominator = C.sum() - C.diagonal().sum() + EPS_NUM

    return numerator / denominator - tau_ortho
