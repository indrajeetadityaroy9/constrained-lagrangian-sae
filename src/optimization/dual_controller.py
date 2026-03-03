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

RCMA integration: compute_residual() provides the squared normalized
constraint residual r(t) that governs gamma coupling and lambda_disc
ratchet. step() includes a stationarity gate that defers dual updates
when the primal subproblem is actively improving within tolerance.
"""

import torch
from torch import Tensor

from src.constants import DEVICE


class DualController:
    """Single-rate EMA + PI dual + monotone CAPU in one object."""

    def __init__(
        self,
        initial_violations: Tensor,
        rho_0: float,
        beta: float = 0.99,
        eps_num: float = 1e-8,
    ) -> None:
        self.n_constraints = initial_violations.shape[0]
        self.beta = beta
        self.eps_num = eps_num
        self._device = DEVICE

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

        # RCMA stationarity gate state.
        self._r_prev = 1.0

    def update(self, violations: Tensor) -> None:
        """Per-step EMA and v_bar update from raw violations."""
        v = violations.detach()
        self._n_updates += 1
        self._v_ema_raw = self.beta * self._v_ema_raw + (1.0 - self.beta) * v
        self._v_bar = self.beta * self._v_bar + (1.0 - self.beta) * v.pow(2)

    def compute_residual(self, taus: Tensor) -> float:
        """Squared normalized constraint residual r(t) in [0, 1].

        r = max_i clamp(max(0, v_ema_i)^2 / (tau_i^2 + eps), 0, 1)

        Squared form ensures the Moreau bandwidth s = sqrt(2*gamma) satisfies
        s = O(||v||) for optimal Smoothed Sharp ALM convergence
        (arXiv 2410.03050, Theorem 3.1). Linear form gives s = O(||v||^{1/2}),
        which over-smooths near feasibility.
        """
        v = self.v_ema[:3]
        v_pos = torch.clamp(v, min=0.0)
        r_per = v_pos.pow(2) / taus.pow(2)
        return r_per.clamp(max=1.0).max().item()

    def compute_residual_instantaneous(self, violations: Tensor, taus: Tensor) -> float:
        """Instantaneous residual from raw (non-EMA) violations.

        Used for conservative gamma coupling: r_gamma = max(r_ema, r_inst).
        This addresses EMA lag during violent transients (KL onset, resampling)
        by ensuring gamma widens immediately on violation spikes.
        """
        v = violations[:3].detach()
        v_pos = torch.clamp(v, min=0.0)
        r_per = v_pos.pow(2) / taus.pow(2)
        return r_per.clamp(max=1.0).max().item()

    def step(self, r: float, beta_slow: float) -> bool:
        """Slow-timescale PI dual update with stationarity gate.

        The gate defers dual tightening when the primal is actively improving
        and violations are within tolerance, enforcing the approximate-minimizer
        precondition for ALM convergence (Nocedal & Wright, Ch. 17).

        Returns True if the update was executed, False if deferred.
        """
        # Stationarity gate: check if primal progress has stalled.
        # stalled = True when r decreased by less than (1-beta)% since last check,
        # meaning the primal subproblem is no longer making sufficient progress.
        # Guard: at feasibility (r < eps), defer dual to prevent over-tightening.
        if r < self.eps_num:
            stalled = False
        else:
            stalled = self._r_prev <= 0.0 or r / (self._r_prev + self.eps_num) >= beta_slow
        beyond_tolerance = r > 1.0
        self._r_prev = r

        if not stalled and not beyond_tolerance:
            return False  # defer: primal improving AND within tolerance

        # PI dual: I-term + P-term (kappa_p = 1.0, ALM equivalence).
        v = self.v_ema
        self._lambdas = torch.clamp(
            self._lambdas + self._rhos * (2.0 * v - self._v_prev),
            min=0.0,
        )
        self._v_prev = v
        # Monotone CAPU.
        target = self._etas / (self._v_bar + self.eps_num).sqrt()
        self._rhos = torch.max(self._rhos, target)
        return True

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
            "n_updates": self._n_updates,
            "lambdas": self._lambdas,
            "v_prev": self._v_prev,
            "etas": self._etas,
            "v_bar": self._v_bar,
            "rhos": self._rhos,
            "r_prev": self._r_prev,
        }

    def load_state_dict(self, sd: dict) -> None:
        self._v_ema_raw = sd["v_ema_raw"].to(self._device)
        self._n_updates = int(sd["n_updates"])
        self._lambdas = sd["lambdas"].to(self._device)
        self._v_prev = sd["v_prev"].to(self._device)
        self._etas = sd["etas"].to(self._device)
        self._v_bar = sd["v_bar"].to(self._device)
        self._rhos = sd["rhos"].to(self._device)
        self._r_prev = float(sd["r_prev"])
