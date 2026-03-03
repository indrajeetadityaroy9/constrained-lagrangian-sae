"""CAGE-inspired discretization weight schedule with dynamic onset."""

import torch


class DiscretizationSchedule:
    """Linear ramp from onset to T_total.

    Weight converges to 1.0: disc penalty matches L0 objective in scale
    at convergence (both are batch-averaged). Onset is set dynamically
    when all base constraints are satisfied.
    """

    def __init__(self, T_total: int) -> None:
        self.T_total = T_total
        self.onset_step = T_total

    def set_onset(self, step: int) -> None:
        """Set the step at which disc ramp begins."""
        self.onset_step = step

    def get_lambda(self, step: int) -> float:
        if step <= self.onset_step:
            return 0.0
        remaining = self.T_total - self.onset_step
        if remaining <= 0:
            return 1.0
        return (step - self.onset_step) / remaining

    def state_dict(self) -> dict:
        return {"onset_step": torch.tensor(self.onset_step)}

    def load_state_dict(self, sd: dict) -> None:
        self.onset_step = sd["onset_step"].item()
