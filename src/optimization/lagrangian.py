"""AL-CoLe smooth penalty and augmented Lagrangian objective."""

import torch
from torch import Tensor


def al_cole_psi(g: Tensor, y: Tensor) -> Tensor:
    """AL-CoLe smooth penalty: Ψ(g, y) = (max(0, 2g + y)^2 - y^2) / 4."""
    inner = 2.0 * g + y
    return (torch.clamp(inner, min=0.0).pow(2) - y.pow(2)) / 4.0


def compute_augmented_lagrangian(
    l0_corr: Tensor,
    violations: Tensor,
    lambdas: Tensor,
    rhos: Tensor,
) -> Tensor:
    """Compute l0_corr + sum_i rho_i * Ψ(violations_i, lambda_i / rho_i)."""
    y = lambdas / rhos
    psi_values = al_cole_psi(violations, y)
    constraint_penalty = (rhos * psi_values).sum()
    return l0_corr + constraint_penalty
