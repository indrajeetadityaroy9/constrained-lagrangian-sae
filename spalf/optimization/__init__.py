"""Augmented Lagrangian optimization: penalties and dual control."""

from spalf.optimization.discretization import DiscretizationSchedule
from spalf.optimization.dual_controller import DualController
from spalf.optimization.lagrangian import compute_augmented_lagrangian

__all__ = [
    "compute_augmented_lagrangian",
    "DualController",
    "DiscretizationSchedule",
]
