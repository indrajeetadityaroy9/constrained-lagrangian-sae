"""Unified dual controller: EMA filtering, PI dual update, monotone CAPU.

Replaces the three-class stack (DualRateEMA, NuPIDualUpdater, MonotoneCAPU)
with a single object. The dual-rate EMA split is removed: CAPU's internal
v_bar (beta=0.99) already smooths squared violations, making the fast EMA
pre-filter redundant.

PI dual update: lambda = clamp(lambda + rho*(2v - v_prev), 0).
kappa_p = 1.0 follows from PI/ALM equivalence (arXiv 2509.22500,
Theorem 2): optimism omega must equal penalty coefficient c for
dual optimistic ascent to be equivalent to GDA on the augmented
Lagrangian.
"""

import torch
from torch import Tensor


class DualController:
    """Single-rate EMA + PI dual + monotone CAPU in one object."""

    def __init__(
        self,
        initial_violations: Tensor,
        rho_0: float,
        beta: float = 0.99,
        eps_num: float = 1e-8,
        device: torch.device | None = None,
    ) -> None:
        self.n_constraints = initial_violations.shape[0]
        self.beta = beta
        self.eps_num = eps_num
        self._device = device or torch.device("cuda")

        # EMA state (single-rate, bias-corrected).
        self._v_ema_raw = torch.zeros(self.n_constraints, device=self._device)
        self._n_updates = 0

        # PI dual state.
        self._lambdas = torch.zeros(self.n_constraints, device=self._device)
        self._v_prev = torch.zeros(self.n_constraints, device=self._device)

        # CAPU state.
        iv = initial_violations.abs().to(self._device)
        self._etas = 1.0 / (iv + eps_num).sqrt()
        self._v_bar = torch.ones(self.n_constraints, device=self._device)
        self._rhos = torch.full((self.n_constraints,), rho_0, device=self._device)

    def update(self, violations: Tensor) -> None:
        """Per-step EMA and v_bar update from raw violations."""
        v = violations.detach()
        self._n_updates += 1
        self._v_ema_raw = self.beta * self._v_ema_raw + (1.0 - self.beta) * v
        self._v_bar = self.beta * self._v_bar + (1.0 - self.beta) * v.pow(2)

    def step(self) -> None:
        """Slow-timescale PI dual update and monotone rho update."""
        v = self.v_ema
        # PI dual: I-term + P-term (kappa_p = 1.0, ALM equivalence).
        self._lambdas = torch.clamp(
            self._lambdas + self._rhos * (2.0 * v - self._v_prev),
            min=0.0,
        )
        self._v_prev = v.clone()
        # Monotone CAPU.
        target = self._etas / (self._v_bar + self.eps_num).sqrt()
        self._rhos = torch.max(self._rhos, target)

    def recalibrate(self, index: int, initial_violation: float) -> None:
        """Recalibrate a constraint's eta and v_bar (e.g., at KL onset)."""
        self._etas[index] = 1.0 / (abs(initial_violation) + self.eps_num) ** 0.5
        self._v_bar[index] = 1.0

    @property
    def v_ema(self) -> Tensor:
        """Bias-corrected EMA of violations [n_constraints]."""
        return self._v_ema_raw / (1.0 - self.beta ** self._n_updates)

    @property
    def lambdas(self) -> Tensor:
        return self._lambdas

    @property
    def rhos(self) -> Tensor:
        return self._rhos

    def state_dict(self) -> dict:
        return {
            "v_ema_raw": self._v_ema_raw,
            "n_updates": torch.tensor(self._n_updates, device=self._device),
            "lambdas": self._lambdas,
            "v_prev": self._v_prev,
            "etas": self._etas,
            "v_bar": self._v_bar,
            "rhos": self._rhos,
        }

    def load_state_dict(self, sd: dict) -> None:
        self._v_ema_raw = sd["v_ema_raw"].to(self._device)
        self._n_updates = sd["n_updates"].item()
        self._lambdas = sd["lambdas"].to(self._device)
        self._v_prev = sd["v_prev"].to(self._device)
        self._etas = sd["etas"].to(self._device)
        self._v_bar = sd["v_bar"].to(self._device)
        self._rhos = sd["rhos"].to(self._device)
