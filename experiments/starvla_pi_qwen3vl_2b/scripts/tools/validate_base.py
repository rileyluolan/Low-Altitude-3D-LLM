#!/usr/bin/env python3
"""Fast structural validation for the materialized StarVLA-PI-2B base."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml


from experiment_paths import ROOT, WORKSPACE, STARVLA, HUGEBENCH, GAUSSIAN, HUGE_DATA, MODEL, VENV, BASE, config_dict
DEFAULT_RUN = BASE.parent.parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(32 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--verify-hash", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint = run_dir / "checkpoints/base_pytorch_model.pt"
    manifest = json.loads((run_dir / "conversion_manifest.json").read_text())
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    stats = json.loads((run_dir / "dataset_statistics.json").read_text())

    require(checkpoint.stat().st_size == manifest["target"]["checkpoint_size"], "Checkpoint size mismatch")
    if args.verify_hash:
        require(sha256_file(checkpoint) == manifest["target"]["checkpoint_sha256"], "Checkpoint hash mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    require(len(state) == 1154, f"Expected 1154 tensors, got {len(state)}")
    require(tuple(state["action_model.action_encoder.layer1.weight"].shape) == (1024, 4), "Bad action encoder")
    require(tuple(state["action_model.action_decoder.layer2.weight"].shape) == (4, 1024), "Bad action decoder")
    require(tuple(state["project_layers.27.1.weight"].shape) == (1024, 2048), "Bad last projector")
    require(not any(key.startswith("project_layers.28.") for key in state), "Unexpected projector layer 28")
    require(not any("language_model.layers.28." in key for key in state), "Unexpected VLM layer 28")

    action_cfg = config["framework"]["action_model"]
    require(config["framework"]["name"] == "QwenPI_v3", "Wrong framework")
    require(action_cfg["action_dim"] == action_cfg["state_dim"] == 4, "Wrong state/action dimension")
    require(action_cfg["action_horizon"] == 20, "Wrong action horizon")
    huge_stats = stats["new_embodiment"]
    require(len(huge_stats["state"]["q01"]) == 4, "Bad state statistics")
    require(len(huge_stats["action"]["q99"]) == 4, "Bad action statistics")
    require(huge_stats["num_transitions"] == 1_720_096, "Bad training-frame count")
    print(
        f"Base validation passed: {manifest['target']['parameters']:,} parameters, "
        f"{len(state)} tensors, sha256={manifest['target']['checkpoint_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
