"""SPALF training pipeline: calibration, initialization, RC²MA constrained AL loop.

Single-file pipeline that reads top-to-bottom:
  calibration helpers -> checkpoint helper -> train()

RC²MA (Rate-Certified RCMA) augmentations:
  A1: Rate-clamped gamma schedule (Böhm & Wright, 2003.07612)
  A2: Non-monotone CAPU (Dolgopolik, 2412.14269) — in spalf/optim/controller.py
  A3: CUSUM-gated KL onset (Page, 1954)
"""

import json
import math
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file, save_file

from spalf.data.store import ActivationStore
from spalf.data.buffer import ActivationBuffer
from spalf.evaluation import run_patched_forward, compute_kl
from spalf.model.sae import StratifiedSAE
from spalf.model.constraints import compute_orthogonality_violation
from spalf.model.initialization import initialize_from_calibration
from spalf.optim.controller import DualController
from spalf.optim.lagrangian import compute_augmented_lagrangian
from spalf.whitening import FrequentDirections, SoftZCAWhitener


def _derive_hyperparameters(
    whitener: SoftZCAWhitener,
    W_vocab: Tensor,
    d: int,
    V: int,
    total_tokens: int,
    batch_size: int,
    seq_len: int,
) -> dict:
    """Derive all training hyperparameters from data statistics."""
    eff_rank = whitener.effective_rank

    # Dictionary size: V anchored + eff_rank free features.
    F = V + eff_rank
    # Sparsity: each token activates eff_rank features (its effective dimensionality).
    L0_target = max(1, eff_rank)

    # Faithfulness: noise fraction of covariance (variance below rank-k cutoff).
    # Floor at 1/d: one dimension of tolerance.
    tau_faith = max(whitener.noise_fraction, 1.0 / d) * d
    # Drift: per-dimension drift budget (dimension-normalized vocab norm).
    tau_drift = W_vocab.pow(2).sum().item() / d

    total_steps = total_tokens // batch_size

    # Learning rate: 1/sqrt(n_params) (McCandlish et al. 2018 gradient noise scaling).
    n_params = 2 * F * d + 2 * F + d
    lr = 1.0 / n_params ** 0.5

    # Warmup: sqrt(total_steps) steps (sublinear in T).
    warmup_steps = round(total_steps ** 0.5)

    # Calibration: one forward pass per feature.
    n_cal_batches = math.ceil(F / batch_size)
    # Buffer: seq_len batches for temporal decorrelation.
    buffer_size = seq_len * batch_size
    # rho_0 scale: match penalty to L0 objective magnitude.
    l0_scale = L0_target / F

    derived = {
        "F": F, "L0_target": L0_target, "eff_rank": eff_rank,
        "tau_faith": tau_faith, "tau_drift": tau_drift,
        "total_steps": total_steps, "lr": lr,
        "warmup_steps": warmup_steps,
        "n_cal_batches": n_cal_batches, "buffer_size": buffer_size,
        "l0_scale": l0_scale,
    }

    print(json.dumps({"event": "derived_hyperparameters", **derived}, sort_keys=True), flush=True)

    return derived


def _run_calibration(config: DictConfig, store: ActivationStore) -> dict:
    """Build FD sketch, whitener, vocabulary slice, and constraint thresholds."""
    d = store.model.config.hidden_size

    sketch = FrequentDirections(d)
    while not sketch._converged:
        batch = store.next_batch()
        sketch.update(batch)

    whitener = SoftZCAWhitener.from_sketch(sketch)

    W_vocab_full = store.get_unembedding_matrix()
    V_cap = config.V_cap
    if V_cap > 0 and V_cap < W_vocab_full.shape[1]:
        norms = W_vocab_full.norm(dim=0)
        _, top_indices = norms.topk(V_cap)
        top_indices = top_indices.sort().values
        W_vocab = W_vocab_full[:, top_indices]
    else:
        W_vocab = W_vocab_full

    V = W_vocab.shape[1]

    hp = _derive_hyperparameters(
        whitener, W_vocab, d, V,
        config.total_tokens, config.batch_size, config.seq_len,
    )

    # User overrides (explicit config values take precedence over derived).
    F = config.F if config.F > 0 else hp["F"]
    L0_target = config.L0_target if config.L0_target > 0 else hp["L0_target"]

    return {
        **hp,
        "whitener": whitener, "W_vocab": W_vocab,
        "d": d, "V": V, "F": F, "L0_target": L0_target,
        "tau_ortho": 0.0, "tau_kl": None,
        "primal_tau_names": ["faith", "drift", "ortho"],
        "onset_tau_names": ["kl"],
    }


def _make_grad_clipper(sae: StratifiedSAE, optimizer: torch.optim.Optimizer) -> Callable[[], None]:
    """AdaGC: clips per-tensor grad norm to EMA of historical norms (arXiv:2502.11034).

    Beta matches Adam's second-moment decay (Kingma & Ba 2015): both track
    second-order gradient statistics, so the same timescale applies.

    GPU-resident: EMA norms bootstrap from parameter norms (first-step clip
    ceiling), then blend toward gradient norms. No .item() CUDA syncs.
    """
    beta = optimizer.defaults["betas"][1]
    params = [p for p in sae.parameters() if p.data.ndim > 1]
    # Bootstrap EMA from parameter norms: first-step ceiling = param magnitude.
    ema_norms = [p.data.norm().detach().clone() for p in params]

    def clip() -> None:
        for i, p in enumerate(params):
            if p.grad is None:
                continue
            grad_norm = p.grad.data.norm()
            # Clip against current EMA (before incorporating this step's gradient).
            torch.where(
                grad_norm > ema_norms[i],
                p.grad.data * (ema_norms[i] / grad_norm),
                p.grad.data,
                out=p.grad.data,
            )
            # Update EMA after clipping (tracks unclipped distribution per AdaGC).
            ema_norms[i].mul_(beta).add_(grad_norm, alpha=1.0 - beta)

    return clip


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
    lambda_disc: float,
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
        "config": OmegaConf.to_container(config, resolve=True),
        "calibration": {
            "d": cal["d"], "V": cal["V"], "F": cal["F"],
            "L0_target": cal["L0_target"],
            "tau_faith": cal["tau_faith"], "tau_drift": cal["tau_drift"],
            "tau_ortho": cal["tau_ortho"], "tau_kl": cal["tau_kl"],
        },
        "lambda_disc": lambda_disc,
    }
    with open(ckpt_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


@torch.no_grad()
def _calibrate_initial_violations(
    buffer: ActivationBuffer,
    whitener: SoftZCAWhitener,
    sae: StratifiedSAE,
    W_vocab: Tensor,
    cal: dict,
    batch_size: int,
) -> Tensor:
    """Measure initial constraint violations over n_cal_batches forward passes."""
    n_primal = len(cal["primal_tau_names"])
    n_onset = len(cal["onset_tau_names"])
    accum = torch.zeros(n_primal, device="cuda")
    for _ in range(cal["n_cal_batches"]):
        x = buffer.next_batch(batch_size)
        x_tilde = whitener.forward(x)
        x_hat, z, _, _, _ = sae(x_tilde)
        mahal_sq = whitener.compute_mahalanobis_sq(x - x_hat)
        v_faith = mahal_sq.mean() - cal["tau_faith"]
        v_drift = (sae.W_dec_A - W_vocab).pow(2).sum() - cal["tau_drift"]
        v_ortho = compute_orthogonality_violation(
            z, sae.W_dec_A, sae.W_dec_B, cal["tau_ortho"],
            sae.gamma_init_mean.item(),
        )
        accum += torch.stack([v_faith, v_drift, v_ortho]).abs()
    return torch.cat([
        accum / cal["n_cal_batches"],
        torch.ones(n_onset, device="cuda"),
    ])


def _resume_from_checkpoint(
    config: DictConfig,
    sae: StratifiedSAE,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    controller: DualController,
    whitener: SoftZCAWhitener,
    W_vocab: Tensor,
    cal: dict,
) -> tuple[int, int, SoftZCAWhitener, Tensor]:
    """Restore training state from a checkpoint directory.

    Returns (start_step, onset_step, whitener, W_vocab).
    """
    ckpt_path = Path(config.resume_from_checkpoint)
    training_state = torch.load(
        ckpt_path / "training_state.pt", map_location="cuda", weights_only=False,
    )
    sae.load_state_dict(training_state["sae"])
    optimizer.load_state_dict(training_state["optimizer"])
    scheduler.load_state_dict(training_state["scheduler"])
    controller.load_state_dict(training_state["controller"])

    cal_sd = load_file(str(ckpt_path / "calibration.safetensors"), device="cuda")
    whitener.load_state_dict({
        "mean": cal_sd["mean"],
        "eigenvalues": cal_sd["eigenvalues"],
        "eigenvectors": cal_sd["eigenvectors"],
        "reg_eigenvalues": cal_sd["reg_eigenvalues"],
        "n_samples": cal_sd["n_samples"],
        "total_trace": cal_sd["total_trace"],
    })
    W_vocab = cal_sd["W_vocab"]

    with open(ckpt_path / "metadata.json") as f:
        ckpt_meta = json.load(f)

    # Restore tau values from checkpoint (not re-derived).
    ckpt_cal = ckpt_meta["calibration"]
    cal["tau_faith"] = ckpt_cal["tau_faith"]
    cal["tau_drift"] = ckpt_cal["tau_drift"]
    cal["tau_ortho"] = ckpt_cal["tau_ortho"]
    cal["tau_kl"] = ckpt_cal["tau_kl"]

    return (
        ckpt_meta["step"],
        ckpt_meta["onset_step"],
        whitener,
        W_vocab,
    )


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

    cal = _run_calibration(config, store)
    whitener = cal["whitener"]
    W_vocab = cal["W_vocab"]

    buffer = ActivationBuffer(store, buffer_size=cal["buffer_size"])

    sae = initialize_from_calibration(cal, store)
    sae = torch.compile(sae, mode="max-autotune")

    initial_violations = _calibrate_initial_violations(
        buffer, whitener, sae, W_vocab, cal, config.batch_size,
    )

    total_steps = cal["total_steps"]

    rho_0 = cal["l0_scale"] / initial_violations.abs().mean().item()
    controller = DualController(
        initial_violations=initial_violations,
        rho_0=rho_0,
        total_steps=total_steps,
        n_primal=len(cal["primal_tau_names"]),
    )

    optimizer = torch.optim.Adam(
        sae.parameters(), lr=cal["lr"], fused=True,
    )

    grad_clipper = _make_grad_clipper(sae, optimizer)

    warmup = cal["warmup_steps"]
    # WSD (Warmup-Stable-Decay): decay phase = warmup (both are √T).
    # The stable phase keeps primal LR constant while dual variables evolve,
    # avoiding the confound of simultaneously decaying LR and adjusting penalties.
    # arXiv:2602.06797 (Li et al. 2026); ICLR 2025 river valley analysis.
    decay_steps = warmup
    stable_end = total_steps - decay_steps

    def _lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        if step < stable_end:
            return 1.0
        return (total_steps - step) / decay_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    start_step = 0
    onset_step = total_steps

    if config.resume_from_checkpoint:
        start_step, onset_step, whitener, W_vocab = (
            _resume_from_checkpoint(
                config, sae, optimizer, scheduler, controller,
                whitener, W_vocab, cal,
            )
        )
        with open(Path(config.resume_from_checkpoint) / "metadata.json") as f:
            lambda_disc = json.load(f)["lambda_disc"]

    whitener.forward = torch.compile(whitener.forward)
    whitener.compute_mahalanobis_sq = torch.compile(whitener.compute_mahalanobis_sq)

    token_iter = store._token_generator(config.kl_batch_size)
    tau_kl = cal["tau_kl"]
    kl_active = tau_kl is not None

    tau_vec = torch.tensor(
        [cal["tau_faith"], cal["tau_drift"], cal["tau_ortho"]], device="cuda"
    )
    alpha_floor = (cal["d"] / cal["F"]) ** 2
    kl_sentinel = -tau_vec.sum()
    gamma_init_mean_val = sae.gamma_init_mean.item()
    c_rate = gamma_init_mean_val ** (1.0 / 3.0)
    controller.init_cusum(tau_vec)
    if not config.resume_from_checkpoint:
        lambda_disc = 0.0

    # Dead threshold: Poisson-derived, floored at n_cal_batches.
    # P(0 firings in N steps) = exp(-N * batch_size * L0 / F) < 1/F
    # => N > F * ln(F) / (batch_size * L0)
    dead_threshold = max(
        math.ceil(cal["F"] * math.log(cal["F"]) / (config.batch_size * cal["L0_target"])),
        cal["n_cal_batches"],
    )

    for step in range(start_step, total_steps):

        # ── Forward pass + constraint violations ──────────────────

        kl_div = None
        if kl_active:
            tokens = next(token_iter).cuda()
            with torch.no_grad():
                orig_logits, patched_logits = run_patched_forward(
                    store, sae, whitener, tokens
                )
                kl_div = compute_kl(orig_logits, patched_logits)

        x = buffer.next_batch(config.batch_size)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            x_tilde = whitener.forward(x)
            x_hat, z, gate_mask, l0_probs, disc_penalty = sae(x_tilde)

        mahal_sq = whitener.compute_mahalanobis_sq(x - x_hat)
        v_faith = mahal_sq.mean() - cal["tau_faith"]
        v_drift = (sae.W_dec_A - W_vocab).pow(2).sum() - cal["tau_drift"]
        v_ortho = compute_orthogonality_violation(
            z, sae.W_dec_A, sae.W_dec_B, cal["tau_ortho"],
            gamma_init_mean_val,
        )

        v_kl = (kl_div - tau_kl) if kl_active else kl_sentinel
        violations = torch.stack([v_faith, v_drift, v_ortho, v_kl])

        controller.update(violations)

        # ── AL objective + optimization step ──────────────────────

        l0_corr = l0_probs.mean() + lambda_disc * disc_penalty.mean()
        lagrangian = compute_augmented_lagrangian(
            l0_corr=l0_corr,
            violations=violations,
            lambdas=controller._lambdas,
            rhos=controller._rhos,
        )

        optimizer.zero_grad(set_to_none=True)
        lagrangian.backward()
        grad_clipper()
        optimizer.step()
        scheduler.step()
        sae.update_dead_counts(gate_mask)

        # ── Slow-timescale updates (RCMA + KL onset) ─────────────

        if controller.should_do_slow_update(step):
            r_ema = controller.compute_residual(tau_vec)
            r_inst = controller.compute_residual_instantaneous(violations, tau_vec)
            r_gamma = max(r_ema, r_inst)

            controller.step()

            if kl_active:
                lambda_disc = max(1.0 - r_ema, lambda_disc)

            with torch.no_grad():
                gamma_rcma = sae.gamma_init * max(r_gamma, alpha_floor)
                gamma_floor_t = gamma_init_mean_val * max(
                    alpha_floor, c_rate * (step + 1) ** (-1.0 / 3.0)
                )
                sae.gamma.copy_(torch.clamp(gamma_rcma, min=gamma_floor_t))

        sae.normalize_free_decoder()

        if kl_active and step % dead_threshold == 0:
            n_resampled = sae.resample_dead_features(
                x_tilde, x_hat, dead_threshold, cal["L0_target"],
            )

        if not kl_active:
            controller.update_cusum(violations)
            if controller.check_onset():
                onset_step = step
                tokens = next(token_iter).cuda()
                with torch.no_grad():
                    orig_logits, patched_logits = run_patched_forward(
                        store, sae, whitener, tokens
                    )
                    kl_init = compute_kl(orig_logits, patched_logits)

                tau_kl = kl_init.item()
                cal["tau_kl"] = tau_kl
                controller.recalibrate(controller.n_primal, tau_kl)
                kl_active = True

                print(json.dumps({
                    "event": "kl_onset", "step": step, "tau_kl": tau_kl,
                }, sort_keys=True), flush=True)

        # ── Checkpointing + logging ──────────────────────────────

        if config.checkpoint_interval > 0 and step % config.checkpoint_interval == 0:
            _save_checkpoint(
                Path(config.output_dir) / f"checkpoint_step{step}",
                sae, optimizer, scheduler, controller,
                whitener, W_vocab, cal, config, step, onset_step,
                lambda_disc,
            )

        if step % config.log_interval == 0:
            # Single CUDA sync: stack all scalars, transfer once.
            diff = x - x_hat
            x_centered = x - x.mean(dim=0)
            v_ema = controller.v_ema
            n_primal = controller.n_primal
            log_scalars = torch.stack([
                gate_mask.sum(dim=1).mean(),
                diff.pow(2).sum(dim=1).mean(),
                x_centered.pow(2).sum(dim=1).mean(),
                *[v_ema[i] for i in range(n_primal)],
            ])
            vals = log_scalars.tolist()  # single sync
            l0_mean, mse, x_var = vals[0], vals[1], vals[2]

            metrics = {
                "event": "train_step", "step": step,
                "l0": l0_mean, "r2": 1.0 - mse / x_var,
            }
            for i, name in enumerate(cal["primal_tau_names"]):
                metrics[f"v_{name}"] = vals[3 + i]
            if kl_active:
                metrics["v_kl"] = v_ema[n_primal].item()

            print(json.dumps(metrics, sort_keys=True), flush=True)

    # Final checkpoint.
    _save_checkpoint(
        Path(config.output_dir) / f"checkpoint_step{total_steps}",
        sae, optimizer, scheduler, controller,
        whitener, W_vocab, cal, config, total_steps, onset_step,
        lambda_disc,
    )

    return sae
