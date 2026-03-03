"""Soft-ZCA whitening transform: full and low-rank variants.

Regularization via Oracle Approximating Shrinkage (OAS):
    Chen, Wiesel, Eldar, Hero (2010).
    λ_reg_i = (1 - ρ) · λ_i + ρ · μ  where μ = mean(λ), ρ = OAS shrinkage intensity.
"""

import torch
from torch import Tensor

from spalf.whitening.covariance import OnlineCovariance


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

        self.effective_rank = self._k
        self.is_low_rank = self._k < self.d // 4

        if self.is_low_rank:
            self._U_k = self._eigenvectors[:, : self._k]
            self._Lambda_k = reg_eigenvalues[: self._k]

            self._lambda_bar = float(reg_eigenvalues[self._k :].mean())

            self._scale_k = self._Lambda_k.rsqrt()
            self._scale_tail = 1.0 / self._lambda_bar ** 0.5

            self._inv_scale_k = self._Lambda_k.sqrt()
            self._inv_scale_tail = self._lambda_bar ** 0.5
        else:
            scales = reg_eigenvalues.rsqrt()
            U = self._eigenvectors
            self._W_white = U @ torch.diag(scales) @ U.T

            inv_scales = reg_eigenvalues.sqrt()
            self._W_white_inv = U @ torch.diag(inv_scales) @ U.T

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
        rho_oas = max(0.0, min(num / denom, 1.0))

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
            self._inv_scale_k = self._inv_scale_k.to(device)
        else:
            self._W_white = self._W_white.to(device)
            self._W_white_inv = self._W_white_inv.to(device)
            self._precision = self._precision.to(device)

        return self

    def forward(self, x: Tensor) -> Tensor:
        """Whiten activations."""
        centered = x - self.mean

        if self.is_low_rank:
            proj = centered @ self._U_k
            whitened_proj = proj * self._scale_k
            result_top = whitened_proj @ self._U_k.T

            complement = centered - (proj @ self._U_k.T)
            result_tail = complement * self._scale_tail

            return result_top + result_tail
        else:
            return centered @ self._W_white.T

    def inverse(self, x_tilde: Tensor) -> Tensor:
        """Map whitened activations back to original space."""
        if self.is_low_rank:
            proj = x_tilde @ self._U_k
            unwhitened_proj = proj * self._inv_scale_k
            result_top = unwhitened_proj @ self._U_k.T

            complement = x_tilde - (proj @ self._U_k.T)
            result_tail = complement * self._inv_scale_tail

            return result_top + result_tail + self.mean
        else:
            return x_tilde @ self._W_white_inv.T + self.mean

    def compute_mahalanobis_sq(self, diff: Tensor) -> Tensor:
        """Compute ||diff||^2 in the regularized precision metric."""
        if self.is_low_rank:
            proj = diff @ self._U_k
            scaled_proj = proj * self._scale_k
            top_term = (scaled_proj**2).sum(dim=1)

            complement = diff - (proj @ self._U_k.T)
            tail_term = (complement**2).sum(dim=1) * (self._scale_tail**2)

            return top_term + tail_term
        else:
            return (diff @ self._precision * diff).sum(dim=1)

    def state_dict(self) -> dict:
        """Serialize whitener state for checkpointing."""
        return {
            "mean": self.mean,
            "eigenvalues": self._eigenvalues,
            "eigenvectors": self._eigenvectors,
            "rho_oas": torch.tensor(self.rho_oas),
        }

    def load_state_dict(self, sd: dict) -> None:
        """Restore whitener from checkpoint state."""
        self.__init__(
            mean=sd["mean"],
            eigenvalues=sd["eigenvalues"],
            eigenvectors=sd["eigenvectors"],
            rho_oas=sd["rho_oas"].item(),
        )
