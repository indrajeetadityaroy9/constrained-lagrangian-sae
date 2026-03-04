"""SAE initialization: matched-filter encoder, decoder init, threshold calibration."""

import torch
from torch import Tensor

from spalf.data.store import ActivationStore
from spalf.model.constraints import compute_orthogonality_violation
from spalf.model.sae import StratifiedSAE
from spalf.whitening import SoftZCAWhitener


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

    with torch.no_grad():
        sae.W_dec_A.copy_(W_vocab)
        sae.b_dec.data.copy_(whitener.mean)

        free_cols = torch.randn(d, sae.F_free, device="cuda")
        free_cols = free_cols / free_cols.norm(dim=0, keepdim=True)
        sae.W_dec_B.copy_(free_cols)

        if whitener.is_low_rank:
            # Matched filter in whitened space: w_enc_j = Σ^{-1/2} w_j.
            # Batched: W_vocab.T @ U_k gives [V, k] projections, then apply
            # per-eigenvalue scaling and reconstruct via U_k.T.
            proj = W_vocab.T @ whitener._U_k                       # [V, k]
            top = (proj * whitener._scale_k) @ whitener._U_k.T     # [V, d]
            complement = W_vocab.T - proj @ whitener._U_k.T        # [V, d]
            tail = complement * whitener._scale_tail                # [V, d]
            sae.W_enc.data[:V] = top + tail
        else:
            sae.W_enc.data[:V] = (whitener._W_white @ W_vocab).T

        W_enc_A = sae.W_enc.data[:V]
        W_enc_B = torch.randn(sae.F_free, d, device="cuda") / (d**0.5)

        n_orthogonal = min(sae.F_free, d - V)
        if n_orthogonal > 0:
            Q, _ = torch.linalg.qr(W_enc_A.T)
            proj = W_enc_B[:n_orthogonal] @ Q
            W_enc_B[:n_orthogonal] -= proj @ Q.T
            norms = W_enc_B[:n_orthogonal].norm(dim=1, keepdim=True)
            W_enc_B[:n_orthogonal] /= norms

        if n_orthogonal < sae.F_free:
            norms = W_enc_B[n_orthogonal:].norm(dim=1, keepdim=True)
            W_enc_B[n_orthogonal:] /= norms

        sae.W_enc.data[V:] = W_enc_B

        _calibrate_thresholds(sae, whitener, activation_sample, L0_target)

        sae.gamma_init.copy_(sae.gamma)
        sae.gamma_init_mean.fill_(sae.gamma.mean().item())


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
    # Floor at eps: features with negative quantiles get near-zero thresholds
    # (always fire), which is correct — they lack selectivity at this sparsity.
    thresholds = thresholds.clamp(min=torch.finfo(thresholds.dtype).eps)
    sae.log_threshold.data = thresholds.log()

    sae.recalibrate_gamma(pre_act)


def initialize_from_calibration(
    cal: dict,
    store: ActivationStore,
) -> StratifiedSAE:
    """Create and initialize the SAE from calibration outputs.

    Also measures initial orthogonality to set cal["tau_ortho"] (mutated in-place).
    """
    sae = StratifiedSAE(cal["d"], cal["F"], cal["V"]).cuda()

    samples = []
    # F expected active observations per feature (F²/L0 total samples).
    # Floor: at least one sample per feature. Ceiling: buffer_size tokens (memory).
    n_stat = cal["F"] * cal["F"] // cal["L0_target"]
    n_needed = min(max(n_stat, cal["F"]), cal["buffer_size"])
    while sum(s.shape[0] for s in samples) < n_needed:
        samples.append(store.next_batch())
    activation_sample = torch.cat(samples, dim=0)[:n_needed]

    initialize_sae(
        sae=sae,
        whitener=cal["whitener"],
        W_vocab=cal["W_vocab"],
        activation_sample=activation_sample,
        L0_target=cal["L0_target"],
    )

    # Set tau_ortho from initialized geometry to keep the first constraint scale data-driven.
    with torch.no_grad():
        x_tilde = cal["whitener"].forward(activation_sample)
        _, z_init, _, _, _ = sae(x_tilde)
        raw_ortho = compute_orthogonality_violation(
            z_init, sae.W_dec_A, sae.W_dec_B, 0.0, sae.gamma_init_mean.item()
        ).item()
    cal["tau_ortho"] = max(raw_ortho, 1.0 / cal["d"])

    return sae
