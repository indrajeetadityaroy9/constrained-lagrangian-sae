"""Single continuous training loop with constraint-triggered KL onset."""


import json
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import Tensor

from spalf.checkpoint import save_calibration_state
from spalf.config import CalibrationResult, SPALFConfig
from spalf.constants import BETA_SLOW, DEVICE, EPS_NUM
from spalf.data.activation_store import ActivationStore
from spalf.data.buffer import ActivationBuffer
from spalf.data.patching import run_patched_forward
from spalf.model.constraints import (
    compute_drift_violation,
    compute_faithfulness_violation,
    compute_orthogonality_violation,
)
from spalf.optimization.discretization import DiscretizationSchedule
from spalf.optimization.dual_controller import DualController
from spalf.optimization.lagrangian import compute_augmented_lagrangian
from spalf.model import StratifiedSAE
from spalf.whitening.whitener import SoftZCAWhitener

# Sentinel violation for inactive KL constraint (before onset).
# -1.0: Ψ(-1,0)=0 (no AL penalty), lambda stays at 0 (clamped), CAPU v_bar→1.0 (stable rho).
_KL_SENTINEL = -1.0


def _compute_kl(orig_logits: Tensor, patched_logits: Tensor) -> Tensor:
    """KL divergence between original and patched next-token distributions."""
    V_vocab = orig_logits.shape[-1]
    log_p_orig = F.log_softmax(orig_logits[:, :-1].float(), dim=-1).reshape(-1, V_vocab)
    log_p_patched = F.log_softmax(patched_logits[:, :-1].float(), dim=-1).reshape(-1, V_vocab)
    return F.kl_div(log_p_patched, log_p_orig, reduction="batchmean", log_target=True)


def run_training_loop(
    sae: StratifiedSAE,
    whitener: SoftZCAWhitener,
    W_vocab: Tensor,
    buffer: ActivationBuffer,
    config: SPALFConfig,
    controller: DualController,
    disc_schedule: DiscretizationSchedule,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cal: CalibrationResult,
    accelerator: Accelerator,
    start_step: int = 0,
    store: ActivationStore | None = None,
) -> None:
    """Run training to completion."""
    total_steps = config.total_tokens // config.batch_size
    slow_update_interval = round(1.0 / (1.0 - BETA_SLOW))
    token_iter = None
    tau_kl = cal.tau_kl

    # KL onset state: either from checkpoint (tau_kl is not None) or detected dynamically.
    kl_active = tau_kl is not None
    kl_onset_step = disc_schedule.onset_step if disc_schedule.onset_step < total_steps else None

    # Dead latent tracking.
    fire_count = torch.zeros(sae.F, device=DEVICE)

    for step in range(start_step, total_steps):
        # --- KL measurement (only after onset) ---
        kl_div = None
        if kl_active:
            if token_iter is None:
                token_iter = store.token_iterator()
            tokens = next(token_iter).to(DEVICE)
            with torch.no_grad():
                orig_logits, patched_logits, _, _, _ = run_patched_forward(
                    store, sae, whitener, tokens
                )
                kl_div = _compute_kl(orig_logits, patched_logits)

        # --- Forward pass ---
        x = buffer.next_batch(config.batch_size).to(DEVICE)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            x_tilde = whitener.forward(x)
            lambda_disc = disc_schedule.get_lambda(step)
            x_hat, z, gate_mask, l0_probs, disc_raw = sae(x_tilde, lambda_disc)

        # --- Constraint violations (float32) ---
        mahal_sq = whitener.compute_mahalanobis_sq(x - x_hat)
        v_faith = compute_faithfulness_violation(mahal_sq, cal.tau_faith)
        v_drift = compute_drift_violation(sae.W_dec_A, W_vocab, cal.tau_drift)
        v_ortho = compute_orthogonality_violation(
            z.detach(), sae.W_dec_A, sae.W_dec_B, cal.tau_ortho, sae.gamma
        )

        if kl_active and kl_div is not None:
            v_kl = kl_div - tau_kl
            violations = torch.stack([v_faith, v_drift, v_ortho, v_kl])
        else:
            violations = torch.stack([
                v_faith, v_drift, v_ortho,
                torch.tensor(_KL_SENTINEL, device=DEVICE),
            ])

        controller.update(violations)

        # --- AL objective ---
        disc_correction = disc_raw.mean()
        l0_loss = l0_probs.mean()
        l0_corr = l0_loss + disc_correction

        lagrangian = compute_augmented_lagrangian(
            l0_corr=l0_corr,
            violations=violations,
            lambdas=controller.lambdas,
            rhos=controller.rhos,
        )

        optimizer.zero_grad()
        lagrangian.backward()
        optimizer.step()
        scheduler.step()

        # --- Slow-timescale updates ---
        if step % slow_update_interval == 0:
            controller.step()
            with torch.no_grad():
                pre_act = x_tilde @ sae.W_enc.T + sae.b_enc
                sae.recalibrate_gamma(pre_act)

        sae.normalize_free_decoder()

        # --- Dead latent resampling ---
        fire_count += gate_mask.sum(dim=0)
        if (
            config.resample_interval > 0
            and step > 0
            and step % config.resample_interval == 0
        ):
            dead_mask = fire_count == 0
            n_resampled = sae.resample_dead_free(dead_mask, x_tilde, optimizer)
            if n_resampled > 0:
                print(
                    json.dumps(
                        {"event": "resample", "step": step, "n_resampled": n_resampled},
                        sort_keys=True,
                    ),
                    flush=True,
                )
            fire_count.zero_()

        # --- KL onset detection ---
        if not kl_active:
            if (controller.v_ema[:3] < 0).all().item():
                kl_onset_step = step
                disc_schedule.set_onset(step)

                # Measure initial KL for tau_kl calibration.
                if token_iter is None:
                    token_iter = store.token_iterator()
                tokens = next(token_iter).to(DEVICE)
                with torch.no_grad():
                    orig_logits, patched_logits, _, _, _ = run_patched_forward(
                        store, sae, whitener, tokens
                    )
                    kl_init = _compute_kl(orig_logits, patched_logits)

                tau_kl = kl_init.item()
                cal.tau_kl = tau_kl
                controller.recalibrate(3, tau_kl)
                kl_active = True

                print(
                    json.dumps(
                        {
                            "event": "kl_onset",
                            "step": step,
                            "tau_kl": tau_kl,
                            "v_ema_faith": controller.v_ema[0].item(),
                            "v_ema_drift": controller.v_ema[1].item(),
                            "v_ema_ortho": controller.v_ema[2].item(),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        # --- Checkpointing ---
        if (
            config.checkpoint_interval > 0
            and step % config.checkpoint_interval == 0
        ):
            ckpt_dir = Path(config.output_dir) / f"checkpoint_step{step}"
            accelerator.save_state(str(ckpt_dir))
            save_calibration_state(ckpt_dir, whitener, W_vocab, cal, config, step)

        # --- Logging ---
        if step % 100 == 0:
            l0_mean = gate_mask.sum(dim=1).mean().item()
            mse = (x - x_hat).pow(2).sum(dim=1).mean().item()
            x_centered = x - x.mean(dim=0)
            x_var = x_centered.pow(2).sum(dim=1).mean().item()
            r_squared = 1.0 - mse / (x_var + EPS_NUM)
            kl_div_value = kl_div.item() if kl_div is not None else 0.0
            lam = controller.lambdas

            metrics = {
                "event": "train_step",
                "step": step,
                "lr": optimizer.param_groups[0]["lr"],
                "l0": l0_mean,
                "mse": mse,
                "r2": r_squared,
                "kl": kl_div_value,
                "v_faith": controller.v_ema[0].item(),
                "v_drift": controller.v_ema[1].item(),
                "v_ortho": controller.v_ema[2].item(),
                "lambda_faith": lam[0].item(),
                "lambda_drift": lam[1].item(),
                "lambda_ortho": lam[2].item(),
                "rho_faith": controller.rhos[0].item(),
                "rho_drift": controller.rhos[1].item(),
                "rho_ortho": controller.rhos[2].item(),
            }
            if kl_active:
                metrics["v_kl"] = controller.v_ema[3].item()
                metrics["lambda_kl"] = lam[3].item()
                metrics["rho_kl"] = controller.rhos[3].item()

            print(json.dumps(metrics, sort_keys=True), flush=True)

    final_dir = Path(config.output_dir) / f"checkpoint_step{total_steps}"
    accelerator.save_state(str(final_dir))
    save_calibration_state(final_dir, whitener, W_vocab, cal, config, total_steps)
