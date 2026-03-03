"""SPALF training pipeline: calibration, initialization, constrained AL loop.

Single-file pipeline that reads top-to-bottom:
  calibration helpers -> patching/KL helpers -> checkpoint helper -> train()
"""

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_fn
from torch import Tensor

from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file, save_file

from src.constants import BETA_SLOW, DEVICE, EPS_NUM
from src.activations.activation_store import ActivationStore
from src.activations.buffer import ActivationBuffer
from src.model import StratifiedSAE
from src.model.constraints import (
    compute_drift_violation,
    compute_faithfulness_violation,
    compute_orthogonality_violation,
)
from src.model.initialization import initialize_from_calibration
from src.optimization.dual_controller import DualController
from src.optimization.lagrangian import compute_augmented_lagrangian
from src.whitening.whitener import OnlineCovariance, SoftZCAWhitener


def _run_calibration(config: DictConfig, store: ActivationStore) -> dict:
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

    return {
        "whitener": whitener, "W_vocab": W_vocab,
        "d": d, "V": V, "F": F, "L0_target": L0_target,
        "tau_faith": tau_faith, "tau_drift": tau_drift,
        "tau_ortho": 0.0, "tau_kl": None,
    }


# Sentinel violation for inactive KL constraint (before onset).
# -1.0: Psi(-1,0)=0 (no AL penalty), lambda stays at 0 (clamped), CAPU v_bar->1.0 (stable rho).
_KL_SENTINEL = -1.0


@torch.no_grad()
def _run_patched_forward(
    store: ActivationStore, sae: StratifiedSAE,
    whitener: SoftZCAWhitener, tokens: Tensor,
) -> tuple[Tensor, Tensor]:
    """Run original and SAE-patched forwards, return (orig_logits, patched_logits)."""
    model = store.model

    orig_logits = model(tokens).logits
    x_raw = store.last_activations
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


def _compute_kl(orig_logits: Tensor, patched_logits: Tensor) -> Tensor:
    """KL divergence between original and patched next-token distributions."""
    V_vocab = orig_logits.shape[-1]
    log_p = F_fn.log_softmax(orig_logits[:, :-1].float(), dim=-1).reshape(-1, V_vocab)
    log_q = F_fn.log_softmax(patched_logits[:, :-1].float(), dim=-1).reshape(-1, V_vocab)
    return F_fn.kl_div(log_q, log_p, reduction="batchmean", log_target=True)


def _save_checkpoint(
    output_dir: str | Path,
    sae: StratifiedSAE,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    controller: DualController,
    whitener: SoftZCAWhitener,
    W_vocab: Tensor,
    cal: dict,
    config: DictConfig,
    step: int,
    onset_step: int,
    lambda_disc_prev: float,
) -> None:
    """Save training state and calibration artifacts to a checkpoint directory."""
    ckpt_dir = Path(output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "sae": sae.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "controller": controller.state_dict(),
    }, ckpt_dir / "training_state.pt")

    sd = whitener.state_dict()
    sd["W_vocab"] = W_vocab
    save_file(sd, str(ckpt_dir / "calibration.safetensors"))

    metadata = {
        "step": step,
        "onset_step": onset_step,
        "lambda_disc_prev": lambda_disc_prev,
        "config": OmegaConf.to_container(config, resolve=True),
        "calibration": {
            "d": cal["d"], "V": cal["V"], "F": cal["F"],
            "L0_target": cal["L0_target"],
            "tau_faith": cal["tau_faith"], "tau_drift": cal["tau_drift"],
            "tau_ortho": cal["tau_ortho"], "tau_kl": cal["tau_kl"],
        },
    }
    with open(ckpt_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def train(config: DictConfig) -> StratifiedSAE:
    """Full SPALF pipeline: seed -> calibrate -> initialize -> constrained AL loop."""


    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)


    store = ActivationStore(
        model_name=config.model_name,
        hook_point=config.hook_point,
        dataset_name=config.dataset,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        text_column=config.text_column,
        dataset_split=config.dataset_split,
        dataset_config=config.dataset_config,
        seed=config.seed,
    )
    buffer = ActivationBuffer(store, buffer_size=2**20)


    cal = _run_calibration(config, store)
    whitener = cal["whitener"]
    W_vocab = cal["W_vocab"]


    sae = initialize_from_calibration(cal, store)
    sae = sae.to(DEVICE)
    sae = torch.compile(sae, mode="max-autotune")


    with torch.no_grad():
        accum = torch.zeros(3, device=DEVICE)
        for _ in range(8):
            x = buffer.next_batch(config.batch_size)
            x_tilde = whitener.forward(x)
            x_hat, z, _, _, _ = sae(x_tilde)
            mahal_sq = whitener.compute_mahalanobis_sq(x - x_hat)
            v_faith = compute_faithfulness_violation(mahal_sq, cal["tau_faith"])
            v_drift = compute_drift_violation(sae.W_dec_A, W_vocab, cal["tau_drift"])
            v_ortho = compute_orthogonality_violation(
                z, sae.W_dec_A, sae.W_dec_B, cal["tau_ortho"],
                sae.gamma_init_mean.item(),
            )
            accum += torch.stack([v_faith, v_drift, v_ortho]).abs()
        initial_violations = torch.cat([accum / 8, torch.ones(1, device=DEVICE)])

    rho_0 = 1.0 / initial_violations.abs().mean().item()
    controller = DualController(
        initial_violations=initial_violations,
        rho_0=rho_0,
        beta=BETA_SLOW,
        eps_num=EPS_NUM,
    )


    total_steps = config.total_tokens // config.batch_size
    optimizer = torch.optim.Adam(sae.parameters(), lr=config.lr, betas=(0.9, 0.999), fused=True)

    def _lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / config.warmup_steps
        progress = (step - config.warmup_steps) / (total_steps - config.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.lr_min_ratio + (1.0 - config.lr_min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


    start_step = 0
    onset_step = total_steps
    lambda_disc_prev = 0.0

    if config.resume_from_checkpoint:
        ckpt_path = Path(config.resume_from_checkpoint)
        training_state = torch.load(
            ckpt_path / "training_state.pt", map_location=DEVICE, weights_only=False,
        )
        sae.load_state_dict(training_state["sae"])
        optimizer.load_state_dict(training_state["optimizer"])
        scheduler.load_state_dict(training_state["scheduler"])
        controller.load_state_dict(training_state["controller"])

        cal_sd = load_file(str(ckpt_path / "calibration.safetensors"), device=str(DEVICE))
        whitener.load_state_dict({
            "mean": cal_sd["mean"],
            "eigenvalues": cal_sd["eigenvalues"],
            "eigenvectors": cal_sd["eigenvectors"],
            "rho_oas": cal_sd["rho_oas"],
        })
        whitener = whitener.to(DEVICE)
        W_vocab = cal_sd["W_vocab"]

        with open(ckpt_path / "metadata.json") as f:
            ckpt_meta = json.load(f)
        start_step = ckpt_meta["step"]
        onset_step = ckpt_meta["onset_step"]
        lambda_disc_prev = ckpt_meta["lambda_disc_prev"]
        ckpt_cal = ckpt_meta["calibration"]
        cal["tau_faith"] = ckpt_cal["tau_faith"]
        cal["tau_drift"] = ckpt_cal["tau_drift"]
        cal["tau_ortho"] = ckpt_cal["tau_ortho"]
        cal["tau_kl"] = ckpt_cal["tau_kl"]


    whitener.forward = torch.compile(whitener.forward)
    whitener.compute_mahalanobis_sq = torch.compile(whitener.compute_mahalanobis_sq)

    slow_update_interval = round(1.0 / (1.0 - BETA_SLOW))
    kl_batch_size = min(256, config.batch_size)
    token_iter = buffer.store.token_iterator(kl_batch_size)
    tau_kl = cal["tau_kl"]
    kl_active = tau_kl is not None

    tau_vec = torch.tensor(
        [cal["tau_faith"], cal["tau_drift"], cal["tau_ortho"]], device=DEVICE
    )
    alpha_floor = (cal["d"] / cal["F"]) ** 2
    kl_sentinel = torch.tensor(_KL_SENTINEL, device=DEVICE)
    gamma_init_mean_val = sae.gamma_init_mean.item()

    for step in range(start_step, total_steps):
        # KL measurement (only after onset).
        kl_div = None
        if kl_active:
            tokens = next(token_iter).to(DEVICE)
            with torch.no_grad():
                orig_logits, patched_logits = _run_patched_forward(
                    store, sae, whitener, tokens
                )
                kl_div = _compute_kl(orig_logits, patched_logits)

        # Forward pass.
        x = buffer.next_batch(config.batch_size)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            x_tilde = whitener.forward(x)
            # RCMA discretization ratchet: lambda_disc driven by residual.
            # Before onset: forced to 0. After onset: monotonically non-decreasing.
            if not kl_active:
                lambda_disc = 0.0
            else:
                r_disc = controller.compute_residual(tau_vec)
                raw = 1.0 - r_disc
                lambda_disc = max(raw, lambda_disc_prev)
                lambda_disc_prev = lambda_disc
            x_hat, z, gate_mask, l0_probs, disc_raw = sae(x_tilde, lambda_disc)

        # Constraint violations (float32).
        mahal_sq = whitener.compute_mahalanobis_sq(x - x_hat)
        v_faith = compute_faithfulness_violation(mahal_sq, cal["tau_faith"])
        v_drift = compute_drift_violation(sae.W_dec_A, W_vocab, cal["tau_drift"])
        v_ortho = compute_orthogonality_violation(
            z, sae.W_dec_A, sae.W_dec_B, cal["tau_ortho"],
            gamma_init_mean_val,
        )

        if kl_active:
            v_kl = kl_div - tau_kl
            violations = torch.stack([v_faith, v_drift, v_ortho, v_kl])
        else:
            violations = torch.stack([
                v_faith, v_drift, v_ortho, kl_sentinel,
            ])

        controller.update(violations)

        # AL objective.
        l0_corr = l0_probs.mean() + disc_raw.mean()
        lagrangian = compute_augmented_lagrangian(
            l0_corr=l0_corr,
            violations=violations,
            lambdas=controller.lambdas,
            rhos=controller.rhos,
        )

        optimizer.zero_grad(set_to_none=True)
        lagrangian.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Slow-timescale updates (RCMA-coupled).
        if step % slow_update_interval == 0:
            r_ema = controller.compute_residual(tau_vec)
            r_inst = controller.compute_residual_instantaneous(violations, tau_vec)
            r_gamma = max(r_ema, r_inst)

            controller.step(r_ema, BETA_SLOW)

            with torch.no_grad():
                scale = max(r_gamma, alpha_floor)
                sae.gamma.copy_(sae.gamma_init * scale)

        sae.normalize_free_decoder()

        # KL onset detection.
        if not kl_active:
            if (controller.v_ema[:3] < 0).all().item():
                onset_step = step
                tokens = next(token_iter).to(DEVICE)
                with torch.no_grad():
                    orig_logits, patched_logits = _run_patched_forward(
                        store, sae, whitener, tokens
                    )
                    kl_init = _compute_kl(orig_logits, patched_logits)

                tau_kl = kl_init.item()
                cal["tau_kl"] = tau_kl
                controller.recalibrate(3, tau_kl)
                kl_active = True

                print(json.dumps({
                    "event": "kl_onset", "step": step, "tau_kl": tau_kl,
                    "v_ema_faith": controller.v_ema[0].item(),
                    "v_ema_drift": controller.v_ema[1].item(),
                    "v_ema_ortho": controller.v_ema[2].item(),
                }, sort_keys=True), flush=True)

        # Checkpointing.
        if config.checkpoint_interval > 0 and step % config.checkpoint_interval == 0:
            _save_checkpoint(
                Path(config.output_dir) / f"checkpoint_step{step}",
                sae, optimizer, scheduler, controller,
                whitener, W_vocab, cal, config, step, onset_step, lambda_disc_prev,
            )

        # Logging.
        if step % 100 == 0:
            l0_mean = gate_mask.sum(dim=1).mean().item()
            mse = (x - x_hat).pow(2).sum(dim=1).mean().item()
            x_centered = x - x.mean(dim=0)
            x_var = x_centered.pow(2).sum(dim=1).mean().item()

            metrics = {
                "event": "train_step", "step": step,
                "l0": l0_mean, "r2": 1.0 - mse / x_var,
                "r": controller.compute_residual(tau_vec),
                "lambda_disc": lambda_disc,
                "gamma_mean": sae.gamma.mean().item(),
                "v_faith": controller.v_ema[0].item(),
                "v_drift": controller.v_ema[1].item(),
                "v_ortho": controller.v_ema[2].item(),
            }
            if kl_active:
                metrics["v_kl"] = controller.v_ema[3].item()

            print(json.dumps(metrics, sort_keys=True), flush=True)

    # Final checkpoint.
    _save_checkpoint(
        Path(config.output_dir) / f"checkpoint_step{total_steps}",
        sae, optimizer, scheduler, controller,
        whitener, W_vocab, cal, config, total_steps, onset_step, lambda_disc_prev,
    )

    return sae
