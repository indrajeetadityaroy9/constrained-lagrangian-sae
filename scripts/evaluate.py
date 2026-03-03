"""SPALF evaluation entrypoint. Usage: spalf-eval path/to/config.yaml"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from spalf.checkpoint import load_checkpoint
from spalf.config import SPALFConfig
from spalf.data.activation_store import ActivationStore
from spalf.evaluation import (
    evaluate_downstream_loss,
    compute_sparsity_frontier,
    drift_fidelity,
    feature_absorption_rate,
    count_dead_latents,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def run_eval(config: SPALFConfig) -> dict:
    """Run all evaluation suites."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    sae, whitener, W_vocab = load_checkpoint(config.checkpoint)

    ckpt_meta_path = Path(config.checkpoint) / "metadata.json"
    with open(ckpt_meta_path) as f:
        ckpt_metadata = json.load(f)

    store = ActivationStore(
        model_name=config.model_name,
        hook_point=config.hook_point,
        dataset_name=config.dataset,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        text_column=config.text_column,
        dataset_split=config.dataset_split,
        dataset_config=config.dataset_config,
        seed=config.seed,
    )

    cal = ckpt_metadata["calibration"]
    results = {
        "_metadata": {
            "model_name": config.model_name,
            "hook_point": config.hook_point,
            "d": cal["d"],
            "F": cal["F"],
            "V": cal["V"],
            "checkpoint": config.checkpoint,
            "seed": config.seed,
        },
    }

    results["downstream_loss"] = evaluate_downstream_loss(sae, whitener, store)
    results["sparsity_frontier"] = compute_sparsity_frontier(sae, whitener, store)
    results["drift_fidelity"] = drift_fidelity(sae.W_dec_A, W_vocab)
    results["feature_absorption"] = feature_absorption_rate(sae.W_dec_B, W_vocab)
    results["dead_latents"] = count_dead_latents(sae, whitener, store)

    return results


def write_results(config: SPALFConfig, results: dict) -> None:
    """Write evaluation outputs: metrics.json + config stamp."""
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config.save(out_dir / "eval_config.yaml")

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, default=str)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="SPALF evaluation: load checkpoint, run eval suites, write metrics"
    )
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    config = SPALFConfig.load(args.config)
    results = run_eval(config)
    write_results(config, results)


if __name__ == "__main__":
    main()
