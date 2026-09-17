#!/usr/bin/env python3
"""Derive the HUGE-Bench 4D/20-step base from the mature 2B OXE base."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from omegaconf import OmegaConf

from starVLA.model.framework.base_framework import build_framework


from checkpoint_transfer import CHANNELS, transfer_huge
from experiment_paths import ROOT, load_config
SOURCE_ROOT = ROOT / "artifacts/base/qwen3vl_2b_pi_v3_bridge_rt1_mature"
SOURCE = SOURCE_ROOT / "checkpoints/base_pytorch_model.pt"
TARGET_CONFIG = ROOT / "configs/qwen3vl_2b_hugebench_base.yaml"
TARGET_ROOT = ROOT / "artifacts/base/qwen3vl_2b_pi_v3_hugebench_mature_base"
HUGE_STATS = ROOT / "assets/hugebench_dataset_statistics.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(32 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for path in (SOURCE, TARGET_CONFIG, HUGE_STATS):
        if not path.is_file():
            raise FileNotFoundError(path)
    training = json.loads((SOURCE_ROOT / "training_manifest.json").read_text())
    if len(training.get("verified_stages", [])) != 2:
        raise ValueError("Run promote_oxe_base.py after verifying both stages")
    source_sha = sha256_file(SOURCE)
    if source_sha != training["checkpoint"]["sha256"]:
        raise ValueError("Mature OXE checkpoint differs from its training manifest")
    if TARGET_ROOT.exists() and any(TARGET_ROOT.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{TARGET_ROOT} already exists; validate it or pass --overwrite")
        shutil.rmtree(TARGET_ROOT)
    checkpoint_dir = TARGET_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    torch.manual_seed(42)
    model = build_framework(load_config(TARGET_CONFIG)).to(device="cpu", dtype=torch.bfloat16)
    target = model.state_dict()
    source = torch.load(SOURCE, map_location="cpu", weights_only=True, mmap=True)
    target, transferred = transfer_huge(source, target)
    initialized = {}
    model.load_state_dict(target, strict=True)
    checkpoint = checkpoint_dir / "base_pytorch_model.pt"
    torch.save(model.state_dict(), checkpoint)
    OmegaConf.save(load_config(TARGET_CONFIG), TARGET_ROOT / "config.yaml", resolve=True)
    OmegaConf.save(load_config(TARGET_CONFIG), TARGET_ROOT / "config.full.yaml", resolve=True)
    shutil.copy2(HUGE_STATS, TARGET_ROOT / "dataset_statistics.json")
    manifest = {
        "schema_version": 1,
        "identity": "mature StarVLA-PI Qwen3-VL-2B HUGE-Bench base",
        "status": "OXE-pretrained base; requires HUGE-Bench fine-tuning and evaluation",
        "source": {
            "checkpoint": str(SOURCE),
            "checkpoint_sha256": source_sha,
            "training": "5k projector alignment + 50k joint Bridge+RT-1 steps",
            "action_dim": 7,
            "action_horizon": 16,
        },
        "target": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_size": checkpoint.stat().st_size,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "action_dim": 4,
            "action_horizon": 20,
        },
        "transfer": {
            "action_channel_map": {
                "source": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
                "target": ["dx", "dy", "dz", "dyaw"],
                "source_indices": CHANNELS,
            },
            "transferred_tensor_count": len(transferred),
            "initialized_tensor_count": len(initialized),
            "transferred_tensors": transferred,
            "initialized_tensors": initialized,
            "notes": [
                "The Qwen3-VL-2B backbone and all 28 trained 2048-to-1024 projectors are preserved.",
                "All tensors transferred; four state/action tensors explicitly sliced to channels [0,1,2,5].",
                "This base still requires HUGE-Bench fine-tuning before evaluation.",
            ],
        },
    }
    (TARGET_ROOT / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Mature HUGE base: {checkpoint}")
    print(f"SHA256: {manifest['target']['checkpoint_sha256']}")
    print(f"Transferred: {len(transferred)}, initialized: {len(initialized)} tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
