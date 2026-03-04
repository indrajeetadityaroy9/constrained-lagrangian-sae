"""Unified dual controller: EMA filtering, PI dual update, non-monotone CAPU.

PI dual update: lambda = clamp(lambda + rho*(2v - v_prev), 0).
kappa_p = 1.0 follows from PI/ALM equivalence (arXiv 2509.22500,
Theorem 2).

Non-monotone CAPU (Dolgopolik, 2412.14269): rho can decrease when
constraints are satisfied. Per-constraint rho_floor (arXiv 2508.15695)
scales inversely with initial violation magnitude for convergence.

CUSUM onset detector (Page, 1954): provably optimal sequential test
for KL constraint activation.
"""

import math

import torch
from torch import Tensor

class DualController:
    """EMA + PI dual + non-monotone CAPU + CUSUM onset detector."""

    def __init__(
        self,
        initial_violations: Tensor,
        rho_0: float,
        total_steps: int,
        n_primal: int,
    ) -> None:
        self.n_constraints = initial_violations.shape[0]
        self.n_primal = n_primal
        self._total_steps = total_steps

        # TEMA state (bias-corrected triple EMA, Peleg et al. 2306.01423).
        self._ema1 = torch.zeros(self.n_constraints, device="cuda")
        self._ema2 = torch.zeros(self.n_constraints, device="cuda")
        self._ema3 = torch.zeros(self.n_constraints, device="cuda")
        self._n_updates = 0

        # Log-space running product for bias correction with time-varying beta.
        self._log_bc = 0.0

        # PI dual state.
        self._lambdas = torch.zeros(self.n_constraints, device="cuda")
        self._v_prev = torch.zeros(self.n_constraints, device="cuda")

        # Non-monotone CAPU state (Dolgopolik, 2412.14269).
        iv = initial_violations.abs()
        self._etas = iv.rsqrt()
        self._v_bar = torch.ones(self.n_constraints, device="cuda")
        self._rhos = torch.full((self.n_constraints,), rho_0, device="cuda")
        # Per-constraint rho_floor (arXiv 2508.15695): inversely proportional
        # to initial violation magnitude. rho_floor.mean() ≈ rho_0 by construction.
        self._rho_0 = rho_0
        self._rho_floor = rho_0 * (iv.mean() / iv)

        # Stationarity gate initial damping rate (Cauchy CDF form).
        # 1/(1+ρ₀) maps ρ₀ ∈ [0,∞) → τ₀ ∈ (0,1] monotonically.
        self._tau_0 = 1.0 / (1.0 + rho_0)

        # Adaptive dual update interval (triangular spacing).
        self._n_dual_updates = 0
        self._next_slow_step = 0

        # CUSUM onset detector state (initialized by init_cusum).
        self._cusum = torch.zeros(0, device="cuda")
        self._cusum_h = torch.zeros(0, device="cuda")

    def _adaptive_beta(self) -> float:
        """Harmonic gain with natural burn-in.

        beta_t = max(t, n_constraints) / (max(t, n_constraints) + 1).
        The burn-in offset n_constraints ensures the initial window covers
        at least one observation of each constraint before EMA contributes.
        Asymptotics: beta_t -> t/(t+1) as t >> n_constraints (Robbins-Monro).
        """
        t = max(self._n_updates, self.n_constraints)
        return t / (t + 1)

    def update(self, violations: Tensor) -> None:
        """Per-step EMA and v_bar update from raw violations."""
        v = violations.detach()
        self._n_updates += 1
        beta = self._adaptive_beta()
        self._log_bc += math.log(beta)

        self._ema1 = beta * self._ema1 + (1.0 - beta) * v
        self._ema2 = beta * self._ema2 + (1.0 - beta) * self._ema1
        self._ema3 = beta * self._ema3 + (1.0 - beta) * self._ema2
        self._v_bar = beta * self._v_bar + (1.0 - beta) * v.pow(2)

    def _residual(self, v: Tensor, taus: Tensor) -> float:
        """Squared normalized constraint residual in [0, 1].

        r = max_i clamp(max(0, v_i)^2 / tau_i^2, 0, 1)

        Squared form ensures the Moreau bandwidth s = sqrt(2*gamma) satisfies
        s = O(||v||) for optimal Smoothed Sharp ALM convergence
        (arXiv 2410.03050, Theorem 3.1). Linear form gives s = O(||v||^{1/2}),
        which over-smooths near feasibility.
        """
        v_pos = torch.clamp(v[:self.n_primal], min=0.0)
        r_per = v_pos.pow(2) / taus.pow(2)
        return r_per.clamp(max=1.0).max().item()

    def compute_residual(self, taus: Tensor) -> float:
        """EMA-smoothed residual."""
        return self._residual(self.v_ema, taus)

    def compute_residual_instantaneous(self, violations: Tensor, taus: Tensor) -> float:
        """Instantaneous residual from raw violations.

        Used for conservative gamma coupling: r_gamma = max(r_ema, r_inst).
        Addresses EMA lag during violent transients (KL onset, resampling).
        """
        return self._residual(violations.detach(), taus)

    def step(self) -> None:
        """Slow-timescale PI dual update + stationarity gate + non-monotone CAPU."""
        v = self.v_ema

        # Standard PI dual update: I-term + P-term (kappa_p = 1.0, ALM equivalence).
        pi_update = torch.clamp(
            self._lambdas + self._rhos * (2.0 * v - self._v_prev),
            min=0.0,
        )

        # Stationarity gate (arXiv 2504.12759): damp when feasible AND improving.
        # tau_t decays as O(1/k), recovering exact KKT asymptotically.
        tau_t = self._tau_0 / (1.0 + self._n_dual_updates)
        feasible_improving = (v < 0) & (v < self._v_prev)
        damped = self._lambdas * (1.0 - tau_t)
        self._lambdas = torch.where(feasible_improving, damped, pi_update)

        self._v_prev = v.clone()

        # Non-monotone CAPU (Dolgopolik, 2412.14269): rho can decrease,
        # lower-bounded by rho_floor for convergence guarantee.
        target = self._etas * self._v_bar.rsqrt()
        self._rhos = torch.clamp(target, min=self._rho_floor)

    def recalibrate(self, index: int, initial_violation: float) -> None:
        """Recalibrate a constraint's eta, v_bar, rho_floor, and EMA state (e.g., at KL onset).

        Resets _ema1 and _v_prev for the recalibrated constraint to avoid sentinel
        contamination in the PI dual update (prevents ~3500-step activation delay
        and spurious lambda spike from stale _v_prev).
        """
        self._etas[index] = abs(initial_violation) ** -0.5
        self._v_bar[index] = 1.0
        self._rho_floor[index] = self._rho_0
        self._ema1[index] = 0.0
        self._ema2[index] = 0.0
        self._ema3[index] = 0.0
        self._v_prev[index] = 0.0

    def init_cusum(self, taus: Tensor) -> None:
        """Initialize CUSUM onset detector (Page, 1954).

        Wald threshold: h = tau * sqrt(2 * ln(T)).
        Expected <= 1 false alarm over T steps (Wald, 1945).
        The 2.0 inside the sqrt is from the Gaussian MGF coefficient.
        """
        self._cusum = torch.zeros(taus.shape[0], device="cuda")
        self._cusum_h = taus * math.sqrt(2.0 * math.log(self._total_steps))

    def update_cusum(self, violations: Tensor) -> None:
        """Accumulate negative-shift evidence from raw violations."""
        n = self._cusum.shape[0]
        self._cusum = torch.clamp(self._cusum - violations[:n].detach(), min=0.0)

    def check_onset(self) -> bool:
        """True when all CUSUM statistics exceed their thresholds."""
        return (self._cusum > self._cusum_h).all().item()

    def should_do_slow_update(self, step: int) -> bool:
        """Dual update interval = n_dual_updates (triangular spacing).

        Total updates over T steps ~ sqrt(2T).
        Interval grows with problem experience, not an arbitrary schedule.
        """
        if step >= self._next_slow_step:
            self._n_dual_updates += 1
            self._next_slow_step = step + self._n_dual_updates
            return True
        return False

    @property
    def v_ema(self) -> Tensor:
        """Bias-corrected TEMA of violations (Peleg et al. 2306.01423).

        TEMA = 3·EMA₁ − 3·EMA₂ + EMA₃ where coefficients are the unique
        solution to the lag-minimization problem min_{a,b,c} E[(TEMA - v)²]
        subject to a+b+c = 1 (unbiasedness). Lag: O(1/(1-β)³) vs O(1/(1-β)).
        """
        bc = 1.0 - math.exp(self._log_bc)
        e1 = self._ema1 / bc
        e2 = self._ema2 / bc
        e3 = self._ema3 / bc
        return 3.0 * e1 - 3.0 * e2 + e3

    def state_dict(self) -> dict:
        return {
            "ema1": self._ema1,
            "ema2": self._ema2,
            "ema3": self._ema3,
            "n_updates": self._n_updates,
            "log_bc": self._log_bc,
            "lambdas": self._lambdas,
            "v_prev": self._v_prev,
            "etas": self._etas,
            "v_bar": self._v_bar,
            "rhos": self._rhos,
            "rho_0": self._rho_0,
            "rho_floor": self._rho_floor,
            "n_dual_updates": self._n_dual_updates,
            "next_slow_step": self._next_slow_step,
            "total_steps": self._total_steps,
            "cusum": self._cusum,
            "cusum_h": self._cusum_h,
            "n_primal": self.n_primal,
            "tau_0": self._tau_0,
        }

    def load_state_dict(self, sd: dict) -> None:
        self._ema1 = sd["ema1"].cuda()
        self._ema2 = sd["ema2"].cuda()
        self._ema3 = sd["ema3"].cuda()
        self._n_updates = int(sd["n_updates"])
        self._log_bc = float(sd["log_bc"])
        self._lambdas = sd["lambdas"].cuda()
        self._v_prev = sd["v_prev"].cuda()
        self._etas = sd["etas"].cuda()
        self._v_bar = sd["v_bar"].cuda()
        self._rhos = sd["rhos"].cuda()
        self._rho_0 = float(sd["rho_0"])
        self._rho_floor = sd["rho_floor"].cuda()
        self._n_dual_updates = int(sd["n_dual_updates"])
        self._next_slow_step = int(sd["next_slow_step"])
        self._total_steps = int(sd["total_steps"])
        self._cusum = sd["cusum"].cuda()
        self._cusum_h = sd["cusum_h"].cuda()
        self.n_primal = int(sd["n_primal"])
        self._tau_0 = float(sd["tau_0"])
