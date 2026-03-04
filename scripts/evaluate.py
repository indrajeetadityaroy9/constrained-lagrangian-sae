"""SPALF evaluation entrypoint."""

import json

import hydra
from omegaconf import DictConfig

from spalf import configure_cuda
from spalf.evaluation import evaluate_checkpoint


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    configure_cuda()
    results = evaluate_checkpoint(cfg)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
