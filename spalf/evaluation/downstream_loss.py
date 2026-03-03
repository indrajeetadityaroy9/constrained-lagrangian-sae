"""Downstream loss evaluation: delta CE loss and KL divergence.

Evaluation budget: ~2M tokens (2^21), matching Rajamanoharan et al. (2024)
and Lieberum et al. (2024). Zero-ablation and loss recovered are computed
in the sparsity frontier module.
"""


import torch
import torch.nn.functional as F

from spalf.data.activation_store import ActivationStore
from spalf.data.patching import run_patched_forward
from spalf.evaluation import EVAL_TOKENS
from spalf.model import StratifiedSAE
from spalf.whitening.whitener import SoftZCAWhitener


@torch.no_grad()
def evaluate_downstream_loss(
    sae: StratifiedSAE,
    whitener: SoftZCAWhitener,
    store: ActivationStore,
) -> dict[str, float]:
    """Evaluate delta CE loss and KL divergence under SAE patching."""
    device = next(sae.parameters()).device
    tokens_per_batch = store.batch_size * store.seq_len
    n_eval_batches = EVAL_TOKENS // tokens_per_batch

    total_ce_orig = 0.0
    total_ce_patched = 0.0
    total_kl = 0.0
    total_tokens = 0

    token_iter = store.token_iterator()

    for _ in range(n_eval_batches):
        tokens = next(token_iter).to(device)
        B, S = tokens.shape
        labels = tokens[:, 1:]
        n_tok = B * (S - 1)

        orig_logits, patched_logits, _, _, _ = run_patched_forward(
            store, sae, whitener, tokens
        )

        orig_shifted = orig_logits[:, :-1].reshape(-1, orig_logits.shape[-1]).float()
        patched_shifted = patched_logits[:, :-1].reshape(-1, patched_logits.shape[-1]).float()
        labels_flat = labels.reshape(-1)

        ce_orig = F.cross_entropy(orig_shifted, labels_flat)
        ce_patched = F.cross_entropy(patched_shifted, labels_flat)

        log_p = F.log_softmax(orig_shifted, dim=-1)
        log_q = F.log_softmax(patched_shifted, dim=-1)
        kl = F.kl_div(log_q, log_p, reduction="batchmean", log_target=True)

        total_ce_orig += ce_orig.item() * n_tok
        total_ce_patched += ce_patched.item() * n_tok
        total_kl += kl.item() * n_tok
        total_tokens += n_tok

    avg_ce_orig = total_ce_orig / total_tokens
    avg_ce_patched = total_ce_patched / total_tokens
    avg_kl = total_kl / total_tokens

    return {
        "ce_loss_orig": avg_ce_orig,
        "ce_loss_patched": avg_ce_patched,
        "ce_loss_increase": avg_ce_patched - avg_ce_orig,
        "kl_div": avg_kl,
        "eval_tokens": total_tokens,
    }
