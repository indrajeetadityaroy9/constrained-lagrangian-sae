"""Stratified Sparse Autoencoder: V anchored + (F-V) free decoder columns.

JumpReLU gating with learnable log-thresholds and Moreau bandwidth (γ).
Forward/backward logic lives in the fused Triton kernel (spalf/model/kernel.py).
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from spalf.model.kernel import FusedJumpReLUFunction

# IQR-to-σ ratio for the standard normal: 2·Φ⁻¹(0.75) = 2·√2·erfinv(0.5).
_IQR_SIGMA: float = 2.0 * math.sqrt(2.0) * math.erfinv(0.5)


class StratifiedSAE(nn.Module):
    """Sparse autoencoder with anchored and free decoder strata."""

    def __init__(self, d: int, F: int, V: int) -> None:
        super().__init__()
        self.d = d
        self.F = F
        self.V = V
        self.F_free = F - V

        self.W_enc = nn.Parameter(torch.empty(F, d))
        self.b_enc = nn.Parameter(torch.zeros(F))

        self.W_dec_A = nn.Parameter(torch.empty(d, V))
        self.W_dec_B = nn.Parameter(torch.empty(d, self.F_free))

        self.b_dec = nn.Parameter(torch.zeros(d))

        self.log_threshold = nn.Parameter(torch.zeros(F))
        self.register_buffer("gamma", torch.ones(F))
        self.register_buffer("gamma_init", torch.ones(F))
        self.register_buffer("gamma_init_mean", torch.tensor(1.0))
        self.register_buffer("dead_counter", torch.zeros(F, dtype=torch.long))

    def decode(self, z: Tensor) -> Tensor:
        """Decode to original space."""
        z_A = z[:, : self.V]
        z_B = z[:, self.V :]
        return z_A @ self.W_dec_A.T + z_B @ self.W_dec_B.T + self.b_dec

    def forward(self, x_tilde: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Fused encode/decode via Triton kernel."""
        pre_act = x_tilde @ self.W_enc.T + self.b_enc
        z, gate_mask, l0_probs = FusedJumpReLUFunction.apply(
            pre_act, self.log_threshold, self.gamma
        )
        x_hat = self.decode(z)

        # Discretization penalty: u²/γ in the Moreau transition zone.
        # Normalized by γ for scale invariance (dimensionless, O(1)).
        theta = self.log_threshold.detach().exp()
        u = pre_act - theta
        bandwidth = (2.0 * self.gamma).sqrt()
        in_zone = (u > -bandwidth) & (u <= 0.0)
        disc_penalty = (u.pow(2) / self.gamma) * in_zone.float()

        return x_hat, z, gate_mask, l0_probs, disc_penalty

    @torch.no_grad()
    def recalibrate_gamma(self, pre_act: Tensor) -> None:
        """Recalibrate Moreau bandwidth via density-matched transition zone.

        Sets each feature's transition zone width so that its probability mass
        under a Gaussian reference equals the per-feature firing rate q = L0/F.
        Derivation: c = q / (IQR_σ * phi(z_q)) where IQR_σ = 2·Φ⁻¹(0.75) ≈ 1.349.
        """
        thresholds = self.log_threshold.exp()

        # Data-driven window: global IQR per feature (robust, no σ assumption).
        global_q75 = torch.quantile(pre_act, 0.75, dim=0)
        global_q25 = torch.quantile(pre_act, 0.25, dim=0)
        global_iqr = global_q75 - global_q25

        lower = thresholds - global_iqr
        upper = thresholds + global_iqr
        mask = (pre_act > lower.unsqueeze(0)) & (pre_act < upper.unsqueeze(0))

        masked = pre_act.clone()
        masked[~mask] = float("nan")
        q75 = torch.nanquantile(masked, 0.75, dim=0)
        q25 = torch.nanquantile(masked, 0.25, dim=0)
        iqr = q75 - q25
        # Fallback: if local IQR is NaN (empty window), use global IQR.
        iqr = torch.where(iqr.isnan(), global_iqr, iqr)

        # Per-feature firing rate → density-matched scale factor.
        q = (pre_act > thresholds.unsqueeze(0)).float().mean(dim=0)
        q = q.clamp(min=1.0 / self.F)  # floor for dead features
        z_q = torch.erfinv(1.0 - 2.0 * q) * math.sqrt(2)
        phi_zq = torch.exp(-z_q.pow(2) / 2.0) / math.sqrt(2 * math.pi)
        c = (q / (_IQR_SIGMA * phi_zq)).clamp(max=1.0)  # cap: zone <= IQR

        self.gamma.copy_((c * iqr).pow(2) / 2.0)

    @torch.no_grad()
    def normalize_free_decoder(self) -> None:
        """Project free decoder columns to unit norm. Called after each optimizer step."""
        norms = self.W_dec_B.norm(dim=0, keepdim=True)
        self.W_dec_B.div_(norms)

    @torch.no_grad()
    def update_dead_counts(self, gate_mask: Tensor) -> None:
        """Track per-feature consecutive non-firing steps."""
        fired = gate_mask.any(dim=0)
        self.dead_counter[fired] = 0
        self.dead_counter[~fired] += 1

    @torch.no_grad()
    def resample_dead_features(
        self, x_tilde: Tensor, x_hat: Tensor,
        dead_threshold: int, L0_target: int,
    ) -> int:
        """Reinitialize dead FREE features toward reconstruction error."""
        dead_mask = self.dead_counter > dead_threshold
        dead_mask[:self.V] = False
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        n_dead = dead_indices.shape[0]
        if n_dead == 0:
            return 0

        residuals = x_tilde - x_hat
        error_norms = residuals.norm(dim=1)
        k = min(n_dead, residuals.shape[0])
        _, top_idx = error_norms.topk(k)

        directions = residuals[top_idx]
        directions = directions / directions.norm(dim=1, keepdim=True)

        active = dead_indices[:k]
        self.W_enc.data[active] = directions
        self.W_dec_B.data[:, active - self.V] = directions.T

        quantile = 1.0 - L0_target / self.F
        pre_acts = x_tilde @ directions.T                       # [B, k]
        thresholds = torch.quantile(pre_acts, quantile, dim=0)  # [k]
        thresholds = thresholds.clamp(min=torch.finfo(pre_acts.dtype).eps)
        self.log_threshold.data[active] = thresholds.log()
        self.dead_counter[active] = 0

        return n_dead
