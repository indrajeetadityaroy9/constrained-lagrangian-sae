"""Stratified Sparse Autoencoder: V anchored + (F-V) free decoder columns.

JumpReLU gating with learnable log-thresholds and Moreau bandwidth (γ).
Forward/backward logic lives in the fused Triton kernel (src/model/kernel.py).
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from src.model.kernel import FusedJumpReLUFunction


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

        nn.init.xavier_uniform_(self.W_enc)
        nn.init.xavier_uniform_(self.W_dec_A)
        nn.init.xavier_uniform_(self.W_dec_B)

    def decode(self, z: Tensor) -> Tensor:
        """Decode to original space."""
        z_A = z[:, : self.V]
        z_B = z[:, self.V :]
        return z_A @ self.W_dec_A.T + z_B @ self.W_dec_B.T + self.b_dec

    def forward(
        self, x_tilde: Tensor, lambda_disc: float = 0.0
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Fused encode/decode via Triton kernel with optional discretization correction."""
        pre_act = x_tilde @ self.W_enc.T + self.b_enc
        z, gate_mask, l0_probs, disc_raw = FusedJumpReLUFunction.apply(
            pre_act, self.log_threshold, self.gamma, lambda_disc
        )
        x_hat = self.decode(z)
        return x_hat, z, gate_mask, l0_probs, disc_raw

    @torch.no_grad()
    def recalibrate_gamma(self, pre_act: Tensor) -> None:
        """Recalibrate Moreau bandwidth via density-matched transition zone.

        Sets each feature's transition zone width so that its probability mass
        under a Gaussian reference equals the per-feature firing rate q = L0/F.
        Derivation: c = q / (1.35 * phi(z_q)) where z_q = Phi^{-1}(1-q).
        """
        thresholds = self.log_threshold.exp()
        std_all = pre_act.std(dim=0)

        # Threshold-local IQR (±2σ window).
        lower = thresholds - 2.0 * std_all
        upper = thresholds + 2.0 * std_all
        mask = (pre_act > lower.unsqueeze(0)) & (pre_act < upper.unsqueeze(0))

        masked = pre_act.clone()
        masked[~mask] = float("nan")
        q75 = torch.nanquantile(masked, 0.75, dim=0)
        q25 = torch.nanquantile(masked, 0.25, dim=0)
        iqr = q75 - q25
        iqr = torch.where(iqr.isnan(), std_all, iqr)

        # Per-feature firing rate → density-matched scale factor.
        q = (pre_act > thresholds.unsqueeze(0)).float().mean(dim=0)
        q = q.clamp(min=1.0 / self.F)  # floor for dead features
        z_q = torch.erfinv(1.0 - 2.0 * q) * math.sqrt(2)
        phi_zq = torch.exp(-z_q.pow(2) / 2.0) / math.sqrt(2 * math.pi)
        c = (q / (1.35 * phi_zq)).clamp(max=1.0)  # cap: zone <= IQR

        self.gamma.copy_((c * iqr).pow(2) / 2.0)

    @torch.no_grad()
    def normalize_free_decoder(self) -> None:
        """Project free decoder columns to unit norm. Called after each optimizer step."""
        norms = self.W_dec_B.norm(dim=0, keepdim=True)
        self.W_dec_B.div_(norms)
