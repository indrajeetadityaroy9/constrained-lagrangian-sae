"""Top-level SPALF training pipeline with Accelerate-managed checkpointing."""

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from spalf.config import CalibrationResult, SPALFConfig
from spalf.constants import (
    BETA_SLOW,
    DEVICE,
    EPS_NUM,
)
from spalf.data.activation_store import ActivationStore
from spalf.data.buffer import ActivationBuffer
from spalf.model import StratifiedSAE
from spalf.model.constraints import (
    compute_drift_violation,
    compute_faithfulness_violation,
    compute_orthogonality_violation,
)
from spalf.model.initialization import initialize_from_calibration
from spalf.optimization.discretization import DiscretizationSchedule
from spalf.optimization.dual_controller import DualController
from spalf.training.calibration import run_calibration
from spalf.training.loop import run_training_loop
from spalf.whitening.whitener import SoftZCAWhitener


@torch.no_grad()
def _measure_initial_violations(
    sae: StratifiedSAE,
    whitener: SoftZCAWhitener,
    W_vocab: torch.Tensor,
    buffer: ActivationBuffer,
    cal: CalibrationResult,
    batch_size: int,
    n_batches: int = 8,
) -> torch.Tensor:
    """Measure initial constraint violations for CAPU calibration (multi-batch average)."""
    accum = torch.zeros(3, device=DEVICE)
    for _ in range(n_batches):
        x = buffer.next_batch(batch_size).to(DEVICE)
        x_tilde = whitener.forward(x)
        x_hat, z, _, _, _ = sae(x_tilde)

        mahal_sq = whitener.compute_mahalanobis_sq(x - x_hat)
        v_faith = compute_faithfulness_violation(mahal_sq, cal.tau_faith)
        v_drift = compute_drift_violation(sae.W_dec_A, W_vocab, cal.tau_drift)
        v_ortho = compute_orthogonality_violation(
            z, sae.W_dec_A, sae.W_dec_B, cal.tau_ortho, sae.gamma
        )
        accum += torch.stack([v_faith, v_drift, v_ortho]).abs()

    # 4th element = 1.0 (neutral placeholder for KL, calibrated at onset).
    base = accum / n_batches
    return torch.cat([base, torch.ones(1, device=DEVICE)])


def train(config: SPALFConfig) -> StratifiedSAE:
    """Run full training pipeline: calibration, initialization, and constrained optimization."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    project_config = ProjectConfiguration(
        project_dir=config.output_dir,
        automatic_checkpoint_naming=True,
        total_limit=3,
    )
    accelerator = Accelerator(
        mixed_precision="no",
        project_configuration=project_config,
    )

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

    cal = run_calibration(config, store)

    sae = initialize_from_calibration(cal, store)
    sae = sae.to(DEVICE)
    sae = torch.compile(sae, mode="max-autotune")

    initial_violations = _measure_initial_violations(
        sae, cal.whitener, cal.W_vocab, buffer, cal, config.batch_size
    )
    rho_0 = 1.0 / (initial_violations.abs().mean().item() + EPS_NUM)

    controller = DualController(
        initial_violations=initial_violations,
        rho_0=rho_0,
        beta=BETA_SLOW,
        eps_num=EPS_NUM,
    )

    total_steps = config.total_tokens // config.batch_size
    disc_schedule = DiscretizationSchedule(T_total=total_steps)

    optimizer = torch.optim.Adam(sae.parameters(), lr=config.lr, betas=(0.9, 0.999))

    def _lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(total_steps - config.warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.lr_min_ratio + (1.0 - config.lr_min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    sae, optimizer, scheduler = accelerator.prepare(sae, optimizer, scheduler)
    accelerator.register_for_checkpointing(controller, disc_schedule)

    start_step = 0
    if config.resume_from_checkpoint:
        accelerator.load_state(config.resume_from_checkpoint)
        with open(Path(config.resume_from_checkpoint) / "metadata.json") as f:
            ckpt_meta = json.load(f)
        start_step = ckpt_meta["step"]
        ckpt_cal = ckpt_meta["calibration"]
        if "tau_faith" in ckpt_cal:
            cal.tau_faith = ckpt_cal["tau_faith"]
        if "tau_drift" in ckpt_cal:
            cal.tau_drift = ckpt_cal["tau_drift"]
        if "tau_ortho" in ckpt_cal:
            cal.tau_ortho = ckpt_cal["tau_ortho"]
        if "tau_kl" in ckpt_cal:
            cal.tau_kl = ckpt_cal["tau_kl"]

    run_training_loop(
        sae=sae,
        whitener=cal.whitener,
        W_vocab=cal.W_vocab,
        buffer=buffer,
        config=config,
        controller=controller,
        disc_schedule=disc_schedule,
        optimizer=optimizer,
        scheduler=scheduler,
        cal=cal,
        accelerator=accelerator,
        start_step=start_step,
        store=store,
    )

    return sae
