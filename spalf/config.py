"""SPALF experiment configuration: minimal research-relevant parameters only."""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml
from torch import Tensor


@dataclass
class SPALFConfig:
    """Experiment configuration.

    Auto-scaling defaults (applied when left at zero/None):
        F:         32 * d_model  (dictionary size = 32x expansion)
        L0_target: ceil(F / 400) (target active features per token)
    """

    model_name: str = "EleutherAI/pythia-1.4b"
    hook_point: str = "gpt_neox.layers.6"

    dataset: str = "monology/pile-uncopyrighted"
    dataset_config: str = ""
    text_column: str = "text"
    dataset_split: str = "train"
    total_tokens: int = 1_000_000_000
    batch_size: int = 4096
    seq_len: int = 128

    F: int = 0
    L0_target: int | None = None
    R2_target: float = 0.97
    V_cap: int | None = None
    lr: float = 3e-4
    warmup_steps: int = 1000
    lr_min_ratio: float = 0.1

    delta_drift: float = 0.1
    resample_interval: int = 25000

    seed: int = 42
    output_dir: str = "runs/default"

    resume_from_checkpoint: str = ""
    checkpoint_interval: int = 5000

    checkpoint: str = ""

    def save(self, path: str | Path) -> None:
        """Save config to YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> SPALFConfig:
        """Load config from YAML."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


@dataclass
class CalibrationResult:
    """Calibration outputs shared across training and checkpointing."""

    whitener: "SoftZCAWhitener"
    W_vocab: Tensor
    d: int
    V: int
    F: int
    L0_target: int
    tau_faith: float
    tau_drift: float
    tau_ortho: float
    tau_kl: float | None = None
