"""SPALF evaluation: patching utilities, checkpoint loading, self-terminating metrics.

Shared inference utilities (run_patched_forward, compute_kl) are used by both
the training loop (KL constraint) and post-hoc evaluation.
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F_fn
from omegaconf import DictConfig
from safetensors.torch import load_file
from torch import Tensor

from spalf.data.store import ActivationStore
from spalf.data.buffer import ActivationBuffer
from spalf.model.sae import StratifiedSAE
from spalf.whitening import SoftZCAWhitener


@torch.no_grad()
def run_patched_forward(
    store: ActivationStore, sae: StratifiedSAE,
    whitener: SoftZCAWhitener, tokens: Tensor,
) -> tuple[Tensor, Tensor]:
    """Run original and SAE-patched forwards, return (orig_logits, patched_logits)."""
    model = store.model

    orig_logits = model(tokens).logits
    x_raw = store._captured_activations
    B, S, d = x_raw.shape

    x_flat = x_raw.reshape(-1, d).float()
    x_tilde = whitener.forward(x_flat)
    x_hat_flat, _, _, _, _ = sae(x_tilde)
    x_hat = x_hat_flat.reshape(B, S, d).to(x_raw.dtype)

    def inject_hook(_module, _input, output):
        return (x_hat,) + output[1:]

    handle = store.swap_hook(inject_hook)
    patched_logits = model(tokens).logits
    handle.remove()
    store.restore_hook()

    return orig_logits, patched_logits


def compute_kl(orig_logits: Tensor, patched_logits: Tensor) -> Tensor:
    """KL divergence between original and patched next-token distributions."""
    V_vocab = orig_logits.shape[-1]
    log_p = F_fn.log_softmax(orig_logits[:, :-1].float(), dim=-1).reshape(-1, V_vocab)
    log_q = F_fn.log_softmax(patched_logits[:, :-1].float(), dim=-1).reshape(-1, V_vocab)
    return F_fn.kl_div(log_q, log_p, reduction="batchmean", log_target=True)


class WelfordMean:
    """Online mean + relative standard error via Welford's algorithm."""

    def __init__(self) -> None:
        self.n = 0
        self._mean = 0.0
        self._m2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self._mean
        self._mean += delta / self.n
        delta2 = x - self._mean
        self._m2 += delta * delta2

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def rse(self) -> float:
        """Relative standard error: SE / |mean|."""
        if self.n < 2 or self._mean == 0.0:
            return float("inf")
        variance = self._m2 / (self.n - 1)
        se = math.sqrt(variance / self.n)
        return se / abs(self._mean)


def _load_checkpoint(
    checkpoint_path: str,
) -> tuple[StratifiedSAE, SoftZCAWhitener, torch.Tensor, dict]:
    """Load SAE, whitener, W_vocab, and metadata from a checkpoint directory."""
    ckpt = Path(checkpoint_path)

    with open(ckpt / "metadata.json") as f:
        metadata = json.load(f)

    cal = metadata["calibration"]
    sae = StratifiedSAE(cal["d"], cal["F"], cal["V"]).cuda()

    training_state = torch.load(
        ckpt / "training_state.pt", map_location="cuda", weights_only=False,
    )
    sae.load_state_dict(training_state["sae"])
    sae.eval()

    cal_sd = load_file(str(ckpt / "calibration.safetensors"), device="cuda")
    whitener = SoftZCAWhitener(
        mean=cal_sd["mean"],
        eigenvalues=cal_sd["eigenvalues"],
        eigenvectors=cal_sd["eigenvectors"],
        reg_eigenvalues=cal_sd["reg_eigenvalues"],
        n_samples=int(cal_sd["n_samples"].item()),
        total_trace=float(cal_sd["total_trace"].item()),
    )
    W_vocab = cal_sd["W_vocab"]

    return sae, whitener, W_vocab, metadata


@torch.no_grad()
def evaluate_checkpoint(config: DictConfig) -> dict:
    """Evaluate a trained SAE checkpoint with self-terminating convergence."""
    sae, whitener, W_vocab, metadata = _load_checkpoint(config.checkpoint_path)
    cal = metadata["calibration"]
    d = cal["d"]
    convergence_threshold = 1.0 / d

    store = ActivationStore(
        model_name=config.model_name,
        hook_point=config.hook_point,
        dataset_name=config.dataset,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        text_column=config.text_column,
        dataset_split=config.dataset_split,
        dataset_config=config.dataset_config,
        seed=config.seed + 1,
    )

    buffer = ActivationBuffer(store, buffer_size=config.seq_len * config.batch_size)

    r2_tracker = WelfordMean()
    l0_tracker = WelfordMean()
    kl_tracker = WelfordMean()
    feature_counts = torch.zeros(cal["F"], device="cuda")
    total_samples = 0

    min_batches = math.ceil(cal["F"] / config.batch_size)
    batch_num = 0

    token_iter = store._token_generator(config.batch_size)

    while True:
        batch_num += 1

        x = buffer.next_batch(config.batch_size)
        x_tilde = whitener.forward(x)
        x_hat, z, gate_mask, _, _ = sae(x_tilde)

        diff = x - x_hat
        mse = diff.pow(2).sum(dim=1).mean().item()
        x_centered = x - x.mean(dim=0)
        x_var = x_centered.pow(2).sum(dim=1).mean().item()
        r2_tracker.update(1.0 - mse / x_var)

        l0_tracker.update(gate_mask.sum(dim=1).float().mean().item())

        feature_counts += gate_mask.any(dim=0).long()
        total_samples += 1

        tokens = next(token_iter).cuda()
        orig_logits, patched_logits = run_patched_forward(
            store, sae, whitener, tokens,
        )
        kl_val = compute_kl(orig_logits, patched_logits).item()
        kl_tracker.update(kl_val)

        if batch_num >= min_batches:
            if all(t.rse < convergence_threshold for t in (r2_tracker, l0_tracker, kl_tracker)):
                break

        if batch_num >= d:
            break

    dead_fraction = (feature_counts == 0).float().mean().item()

    return {
        "r2": r2_tracker.mean,
        "l0": l0_tracker.mean,
        "kl_divergence": kl_tracker.mean,
        "dead_fraction": dead_fraction,
    }
