"""Numerical constants for SPALF."""

import torch

EPS_NUM: float = 1e-8
BETA_SLOW: float = 0.99  # EMA timescale (~100 steps).
DEVICE = torch.device("cuda")
