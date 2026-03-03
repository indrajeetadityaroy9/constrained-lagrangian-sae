"""SAE patching and zero-ablation forward passes through the transformer."""


import torch
from torch import Tensor

from spalf.data.activation_store import ActivationStore
from spalf.model import StratifiedSAE
from spalf.whitening.whitener import SoftZCAWhitener


@torch.no_grad()
def run_patched_forward(
    store: ActivationStore,
    sae: StratifiedSAE,
    whitener: SoftZCAWhitener,
    tokens: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Run original and SAE-patched forwards and return logits plus SAE intermediates."""
    model = store.model

    orig_logits = model(tokens).logits
    x_raw = store.last_activations
    B, S, d = x_raw.shape

    x_flat = x_raw.reshape(-1, d).float()
    x_tilde = whitener.forward(x_flat)
    x_hat_flat, _, gate_mask, _, _ = sae(x_tilde)
    x_hat = x_hat_flat.reshape(B, S, d).to(x_raw.dtype)

    def inject_hook(_module, _input, output):
        return (x_hat,) + output[1:]

    handle = store.swap_hook(inject_hook)
    patched_logits = model(tokens).logits
    handle.remove()
    store.restore_hook()

    return orig_logits, patched_logits, x_flat, x_hat_flat, gate_mask


@torch.no_grad()
def run_zero_ablation_forward(
    store: ActivationStore,
    tokens: Tensor,
) -> Tensor:
    """Zero-ablation forward: replace hook point activations with zeros. Returns logits."""
    model = store.model

    def inject_zero(_module, _input, output):
        return (torch.zeros_like(output[0]),) + output[1:]

    handle = store.swap_hook(inject_zero)
    zero_logits = model(tokens).logits
    handle.remove()
    store.restore_hook()

    return zero_logits
