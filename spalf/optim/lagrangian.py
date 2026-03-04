"""AL-CoLe smooth penalty and augmented Lagrangian objective."""

import torch
from torch import Tensor


def compute_augmented_lagrangian(
    l0_corr: Tensor,
    violations: Tensor,
    lambdas: Tensor,
    rhos: Tensor,
) -> Tensor:
    """Compute l0_corr + sum_i rho_i * Ψ(violations_i, lambda_i / rho_i).

    AL-CoLe smooth penalty: Ψ(g, y) = (max(0, 2g + y)^2 - y^2) / 4
    (Boero et al., arXiv 2510.20995, Eq. 3).
    """
    y = lambdas / rhos
    inner = 2.0 * violations + y
    psi = (torch.clamp(inner, min=0.0).pow(2) - y.pow(2)) / 4.0
    return l0_corr + (rhos * psi).sum()
