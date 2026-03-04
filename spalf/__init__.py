"""SPALF: Spectrally-Preconditioned Augmented Lagrangian on Stratified Feature Manifolds."""

import os

import torch


def configure_cuda() -> None:
    """Set CUDA performance flags. Called once from script entrypoints."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
