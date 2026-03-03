"""Calibration routines."""


import math

from spalf.config import CalibrationResult, SPALFConfig
from spalf.data.activation_store import ActivationStore
from spalf.whitening.covariance import OnlineCovariance
from spalf.whitening.whitener import SoftZCAWhitener


def run_calibration(
    config: SPALFConfig,
    store: ActivationStore,
) -> CalibrationResult:
    """Build covariance, whitener, vocabulary slice, and constraint thresholds."""
    d = store.d_model

    F = config.F if config.F > 0 else 32 * d
    L0_target = config.L0_target if config.L0_target is not None else math.ceil(F / 400)

    cov = OnlineCovariance(d)

    while not cov.converged:
        batch = store.next_batch()
        cov.update(batch)

    whitener = SoftZCAWhitener.from_covariance(cov)
    whitener = whitener.to(store.device)

    W_vocab_full = store.get_unembedding_matrix()
    if config.V_cap is not None and config.V_cap < W_vocab_full.shape[1]:
        norms = W_vocab_full.norm(dim=0)
        _, top_indices = norms.topk(config.V_cap)
        top_indices = top_indices.sort().values
        W_vocab = W_vocab_full[:, top_indices]
    else:
        W_vocab = W_vocab_full

    V = W_vocab.shape[1]

    tau_faith = (1.0 - config.R2_target) * d
    tau_drift = config.delta_drift**2 * W_vocab.pow(2).sum().item()
    tau_ortho = 0.0

    return CalibrationResult(
        whitener=whitener,
        W_vocab=W_vocab,
        d=d,
        V=V,
        F=F,
        L0_target=L0_target,
        tau_faith=tau_faith,
        tau_drift=tau_drift,
        tau_ortho=tau_ortho,
    )
