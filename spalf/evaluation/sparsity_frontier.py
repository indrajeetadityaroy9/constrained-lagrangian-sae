"""Sparsity frontier: L0 vs Loss Recovered Pareto curve via threshold sweeping.

Evaluation budget: ~2M tokens (2^21) per phase, matching Rajamanoharan et al.
(2024) and Lieberum et al. (2024). Multipliers are log-uniform in the native
parameter space of log_threshold.
"""

import math

import torch
import torch.nn.functional as F

from spalf.constants import EPS_NUM
from spalf.data.activation_store import ActivationStore
from spalf.data.patching import run_patched_forward, run_zero_ablation_forward
from spalf.evaluation import EVAL_TOKENS
from spalf.model import StratifiedSAE
from spalf.whitening.whitener import SoftZCAWhitener


@torch.no_grad()
def compute_sparsity_frontier(
    sae: StratifiedSAE,
    whitener: SoftZCAWhitener,
    store: ActivationStore,
) -> list[dict[str, float]]:
    """Sweep threshold multipliers to trace the L0-vs-Loss-Recovered frontier."""
    device = next(sae.parameters()).device
    tokens_per_batch = store.batch_size * store.seq_len
    n_eval_batches = EVAL_TOKENS // tokens_per_batch

    # Log-uniform spacing: uniform in log_threshold's native space.
    multipliers = torch.logspace(math.log10(0.5), math.log10(3.0), 9).tolist()

    # --- Baseline: zero-ablation and original CE ---
    total_ce_zero = 0.0
    total_ce_orig = 0.0
    total_tokens_baseline = 0

    baseline_iter = store.token_iterator()
    for _ in range(n_eval_batches):
        tokens = next(baseline_iter).to(device)
        B, S = tokens.shape
        labels = tokens[:, 1:]
        n_tok = B * (S - 1)

        orig_logits = store.model(tokens).logits
        zero_logits = run_zero_ablation_forward(store, tokens)

        orig_flat = orig_logits[:, :-1].reshape(-1, orig_logits.shape[-1]).float()
        zero_flat = zero_logits[:, :-1].reshape(-1, zero_logits.shape[-1]).float()
        labels_flat = labels.reshape(-1)

        ce_orig = F.cross_entropy(orig_flat, labels_flat).item()
        ce_zero = F.cross_entropy(zero_flat, labels_flat).item()

        total_ce_orig += ce_orig * n_tok
        total_ce_zero += ce_zero * n_tok
        total_tokens_baseline += n_tok

    avg_ce_orig = total_ce_orig / total_tokens_baseline
    avg_ce_zero = total_ce_zero / total_tokens_baseline
    denominator = max(avg_ce_zero - avg_ce_orig, EPS_NUM)

    # --- Sweep multipliers ---
    orig_log_threshold = sae.log_threshold.data.clone()
    results = []

    for mult in multipliers:
        sae.log_threshold.data = orig_log_threshold + torch.tensor(mult).log()

        total_l0 = 0.0
        total_mse = 0.0
        total_var = 0.0
        total_ce_patched = 0.0
        total_tokens = 0
        total_activations = 0

        token_iter = store.token_iterator()

        for _ in range(n_eval_batches):
            tokens = next(token_iter).to(device)
            B, S = tokens.shape

            orig_logits, patched_logits, x_flat, x_hat_flat, gate_mask = (
                run_patched_forward(store, sae, whitener, tokens)
            )

            labels = tokens[:, 1:]
            n_tok = B * (S - 1)
            n_act = x_flat.shape[0]

            ce_patched = F.cross_entropy(
                patched_logits[:, :-1].reshape(-1, patched_logits.shape[-1]).float(),
                labels.reshape(-1),
            )

            total_l0 += gate_mask.sum(dim=1).mean().item() * n_act
            total_mse += (x_flat - x_hat_flat).pow(2).sum(dim=1).mean().item() * n_act
            x_centered = x_flat - x_flat.mean(dim=0)
            total_var += x_centered.pow(2).sum(dim=1).mean().item() * n_act
            total_ce_patched += ce_patched.item() * n_tok
            total_tokens += n_tok
            total_activations += n_act

        avg_ce_patched = total_ce_patched / total_tokens
        loss_recovered = (avg_ce_zero - avg_ce_patched) / denominator

        avg_mse = total_mse / total_activations
        point = {
            "multiplier": mult,
            "l0": total_l0 / total_activations,
            "ce_loss_increase": avg_ce_patched - avg_ce_orig,
            "loss_recovered": loss_recovered,
            "mse": avg_mse,
            "fvu": avg_mse / (total_var / total_activations),
            "eval_tokens": total_tokens,
        }
        results.append(point)

    sae.log_threshold.data = orig_log_threshold

    return results
