"""SAE initialization: matched-filter encoder, decoder init, threshold calibration."""

import torch
from torch import Tensor

from spalf.model.sae import StratifiedSAE
from spalf.whitening.whitener import SoftZCAWhitener

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spalf.config import CalibrationResult
    from spalf.data.activation_store import ActivationStore


def initialize_sae(
    sae: StratifiedSAE,
    whitener: SoftZCAWhitener,
    W_vocab: Tensor,
    activation_sample: Tensor,
    L0_target: int,
) -> None:
    """Initialize SAE weights, thresholds, and bandwidths."""
    d, V = W_vocab.shape
    F = sae.F
    device = W_vocab.device

    with torch.no_grad():
        sae.W_dec_A.copy_(W_vocab)

        free_cols = torch.randn(d, sae.F_free, device=device)
        free_cols = free_cols / free_cols.norm(dim=0, keepdim=True)
        sae.W_dec_B.copy_(free_cols)

        if whitener.is_low_rank:
            # Matched filter in whitened space: w_enc_j = Σ^{-1/2} w_j.
            # Low-rank Σ^{-1/2} applied column-wise to avoid materializing dense dxd.
            W_enc_A = torch.zeros(V, d, device=device)
            for j in range(V):
                w_j = W_vocab[:, j]
                proj = w_j @ whitener._U_k                       # [k]
                top = (proj * whitener._scale_k) @ whitener._U_k.T  # Λ_k^{-1/2} in top-k
                complement = w_j - proj @ whitener._U_k.T
                tail = complement * whitener._scale_tail           # λ̄^{-1/2} in tail
                W_enc_A[j] = top + tail
            sae.W_enc.data[:V] = W_enc_A
        else:
            sae.W_enc.data[:V] = (whitener._W_white @ W_vocab).T

        W_enc_A = sae.W_enc.data[:V]
        W_enc_B = torch.randn(sae.F_free, d, device=device) / (d**0.5)

        n_orthogonal = min(sae.F_free, d - V)
        if n_orthogonal > 0:
            Q, _ = torch.linalg.qr(W_enc_A.T)

            for i in range(n_orthogonal):
                row = W_enc_B[i]
                proj = Q @ (Q.T @ row)
                row = row - proj
                W_enc_B[i] = row / row.norm()

        for i in range(n_orthogonal, sae.F_free):
            W_enc_B[i] /= W_enc_B[i].norm()

        sae.W_enc.data[V:] = W_enc_B

        _calibrate_thresholds(sae, whitener, activation_sample, L0_target)


def _calibrate_thresholds(
    sae: StratifiedSAE,
    whitener: SoftZCAWhitener,
    activation_sample: Tensor,
    L0_target: int,
) -> None:
    """Calibrate JumpReLU thresholds and bandwidths from an activation sample."""
    F = sae.F

    x_tilde = whitener.forward(activation_sample)
    pre_act = x_tilde @ sae.W_enc.T + sae.b_enc

    quantile = 1.0 - L0_target / F
    thresholds = torch.quantile(pre_act, quantile, dim=0)
    sae.log_threshold.data = thresholds.log()

    sae.recalibrate_gamma(pre_act)


def initialize_from_calibration(
    cal: "CalibrationResult",
    store: "ActivationStore",
) -> StratifiedSAE:
    """Create and initialize the SAE from calibration outputs.

    Also measures initial orthogonality to set cal.tau_ortho (mutated in-place).
    """
    from spalf.model.constraints import compute_orthogonality_violation

    device = cal.W_vocab.device

    sae = StratifiedSAE(cal.d, cal.F, cal.V).to(device)

    samples = []
    n_needed = min(max(100 * cal.F // cal.L0_target, 10_000), store.batch_size * 20)
    while sum(s.shape[0] for s in samples) < n_needed:
        samples.append(store.next_batch())
    activation_sample = torch.cat(samples, dim=0)[:n_needed].to(device)

    initialize_sae(
        sae=sae,
        whitener=cal.whitener,
        W_vocab=cal.W_vocab,
        activation_sample=activation_sample,
        L0_target=cal.L0_target,
    )

    # Set tau_ortho from initialized geometry to keep the first constraint scale data-driven.
    with torch.no_grad():
        batch_size = min(activation_sample.shape[0], 4096)
        x_sample = activation_sample[:batch_size]
        x_tilde = cal.whitener.forward(x_sample)
        _, z_init, _, _, _ = sae(x_tilde)
        raw_ortho = compute_orthogonality_violation(
            z_init, sae.W_dec_A, sae.W_dec_B, 0.0, sae.gamma
        ).item()
    cal.tau_ortho = max(raw_ortho, 1.0 / cal.d)

    return sae
