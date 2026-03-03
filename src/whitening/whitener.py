"""Soft-ZCA whitening: online covariance estimation and isotropic preconditioning.

Regularization via Oracle Approximating Shrinkage (OAS):
    Chen, Wiesel, Eldar, Hero (2010).
    lambda_reg_i = (1 - rho) * lambda_i + rho * mu  where mu = mean(lambda), rho = OAS shrinkage intensity.
"""

import math

import torch
from torch import Tensor

from src.constants import DEVICE


class OnlineCovariance:
    """Welford online covariance estimation with snapshot-based convergence."""

    def __init__(self, d: int) -> None:
        self.d = d
        self.check_interval = min(math.ceil(d**2), 1_000_000)
        self._device = DEVICE

        self._n = 0
        self._mean = torch.zeros(d, dtype=torch.float64, device=self._device)
        self._M2 = torch.zeros(d, d, dtype=torch.float64, device=self._device)

        self._snapshot_cov: torch.Tensor | None = None
        self._n_at_snapshot = 0
        self._converged = False

    def update(self, x: torch.Tensor) -> None:
        """Update statistics with a batch [batch, d]."""
        x = x.to(dtype=torch.float64)
        batch_size = x.shape[0]

        batch_mean = x.mean(dim=0)
        batch_n = batch_size

        delta = batch_mean - self._mean
        new_n = self._n + batch_n
        new_mean = self._mean + delta * (batch_n / new_n)

        batch_centered = x - batch_mean
        batch_M2 = batch_centered.T @ batch_centered
        self._M2 += batch_M2 + torch.outer(delta, delta) * (
            self._n * batch_n / new_n
        )

        self._mean = new_mean
        self._n = new_n

        if (
            not self._converged
            and self._n - self._n_at_snapshot >= self.check_interval
        ):
            self._check_convergence()

    def _check_convergence(self) -> None:
        current_cov = self.get_covariance()

        if self._snapshot_cov is not None:
            diff_norm = torch.linalg.norm(current_cov - self._snapshot_cov)
            current_norm = torch.linalg.norm(current_cov)
            relative_change = diff_norm / current_norm

            if relative_change < 1.0 / self.d:
                self._converged = True

        self._snapshot_cov = current_cov
        self._n_at_snapshot = self._n

    @property
    def converged(self) -> bool:
        return self._converged

    def get_mean(self) -> torch.Tensor:
        return self._mean.clone()

    def get_covariance(self) -> torch.Tensor:
        return self._M2 / (self._n - 1)

    @property
    def n_samples(self) -> int:
        return self._n


class SoftZCAWhitener:
    """Frozen Soft-ZCA whitening transform with OAS-optimal regularization."""

    def __init__(
        self,
        mean: Tensor,
        eigenvalues: Tensor,
        eigenvectors: Tensor,
        rho_oas: float,
    ) -> None:
        self.d = mean.shape[0]
        self.rho_oas = rho_oas

        self.mean = mean.float()
        self._eigenvalues = eigenvalues.float()
        self._eigenvectors = eigenvectors.float()

        mu = self._eigenvalues.mean()
        reg_eigenvalues = (1 - rho_oas) * self._eigenvalues + rho_oas * mu

        if rho_oas < 1.0:
            cutoff = rho_oas * mu / (1 - rho_oas)
            self._k = (self._eigenvalues > cutoff).sum().item()
        else:
            self._k = self.d

        self.is_low_rank = self._k < self.d // 4

        if self.is_low_rank:
            self._U_k = self._eigenvectors[:, : self._k]
            self._Lambda_k = reg_eigenvalues[: self._k]

            self._lambda_bar = float(reg_eigenvalues[self._k :].mean())

            self._scale_k = self._Lambda_k.rsqrt()
            self._scale_tail = 1.0 / self._lambda_bar ** 0.5
            self._diff_scale = self._scale_k - self._scale_tail
            self._diff_scale_sq = self._scale_k.pow(2) - self._scale_tail ** 2
            self._scale_tail_sq = self._scale_tail ** 2
        else:
            scales = reg_eigenvalues.rsqrt()
            U = self._eigenvectors
            self._W_white = U @ torch.diag(scales) @ U.T

            self._precision = self._W_white.T @ self._W_white

    @classmethod
    def from_covariance(cls, cov: OnlineCovariance) -> "SoftZCAWhitener":
        """Build whitener from a converged covariance estimate. OAS shrinkage is computed from the spectrum."""
        Sigma = cov.get_covariance()
        mean = cov.get_mean()

        eigenvalues, eigenvectors = torch.linalg.eigh(Sigma)

        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)

        eigenvalues = eigenvalues.clamp(min=1e-12)

        d = mean.shape[0]
        n_samples = cov.n_samples
        trace_s = eigenvalues.sum().item()
        trace_s2 = eigenvalues.pow(2).sum().item()
        num = (1 - 2.0 / d) * trace_s2 + trace_s ** 2
        denom = (n_samples + 1 - 2.0 / d) * (trace_s2 - trace_s ** 2 / d)
        rho_oas = max(0.0, min(num / (denom + 1e-12), 1.0))

        return cls(
            mean=mean,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            rho_oas=rho_oas,
        )

    def to(self, device: torch.device) -> "SoftZCAWhitener":
        """Move all tensors to a device."""
        self.mean = self.mean.to(device)
        self._eigenvalues = self._eigenvalues.to(device)
        self._eigenvectors = self._eigenvectors.to(device)

        if self.is_low_rank:
            self._U_k = self._U_k.to(device)
            self._Lambda_k = self._Lambda_k.to(device)
            self._scale_k = self._scale_k.to(device)
            self._diff_scale = self._diff_scale.to(device)
            self._diff_scale_sq = self._diff_scale_sq.to(device)
        else:
            self._W_white = self._W_white.to(device)
            self._precision = self._precision.to(device)

        return self

    def forward(self, x: Tensor) -> Tensor:
        """Whiten activations."""
        centered = x - self.mean

        if self.is_low_rank:
            proj = centered @ self._U_k
            return (proj * self._diff_scale) @ self._U_k.T + centered * self._scale_tail
        else:
            return centered @ self._W_white.T

    def compute_mahalanobis_sq(self, diff: Tensor) -> Tensor:
        """Compute ||diff||^2 in the regularized precision metric."""
        if self.is_low_rank:
            proj = diff @ self._U_k
            return (proj.pow(2) * self._diff_scale_sq).sum(dim=1) + self._scale_tail_sq * diff.pow(2).sum(dim=1)
        else:
            return (diff @ self._precision * diff).sum(dim=1)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean,
            "eigenvalues": self._eigenvalues,
            "eigenvectors": self._eigenvectors,
            "rho_oas": torch.tensor(self.rho_oas),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.__init__(
            mean=sd["mean"],
            eigenvalues=sd["eigenvalues"],
            eigenvectors=sd["eigenvectors"],
            rho_oas=sd["rho_oas"].item(),
        )
