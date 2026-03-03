"""SPALF training entrypoint."""

import os

import hydra
import torch
from omegaconf import DictConfig

from src.training.train import train

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
