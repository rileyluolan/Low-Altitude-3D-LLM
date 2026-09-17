#!/usr/bin/env python3
"""Create the 7D/16-step Qwen3-VL-2B PI-v3 initialization for OXE training."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from omegaconf import OmegaConf

from starVLA.model.framework.base_framework import build_framework


from experiment_paths import ROOT, MODEL, load_config
SOURCE_ROOT = ROOT / "artifacts/source/StarVLA-Qwen3VL-PI_v3-Bridge-RT_1"
SOURCE = SOURCE_ROOT / "checkpoints/steps_50000_pytorch_model.pt"
SOURCE_SHA256 = "7f59a5d0fa9c167fabd941bca8e606bdf5597bfb4f99ca83e345672dd9c345ed"
QWEN = MODEL / "model.safetensors"
QWEN_SHA256 = "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"
CONFIG = ROOT / "configs/qwen3vl_2b_oxe_init.yaml"
OUTPUT = ROOT / "artifacts/base/qwen3vl_2b_pi_v3_bridge_rt1_init"
from checkpoint_transfer import transfer_oxe


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
    for path in (SOURCE, QWEN, CONFIG, SOURCE_ROOT / "dataset_statistics.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(SOURCE) != SOURCE_SHA256:
        raise ValueError("Source StarVLA checkpoint SHA256 mismatch")
    if sha256_file(QWEN) != QWEN_SHA256:
        raise ValueError("Qwen3-VL-2B checkpoint SHA256 mismatch")
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{OUTPUT} is not empty; pass --overwrite")
        shutil.rmtree(OUTPUT)
    checkpoint_dir = OUTPUT / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    torch.manual_seed(42)
    cfg = load_config(CONFIG)
    model = build_framework(cfg).to(device="cpu", dtype=torch.bfloat16)
    target = model.state_dict()
    source = torch.load(SOURCE, map_location="cpu", weights_only=True, mmap=True)
    target, transferred, initialized, mapping, source_depth, target_depth = transfer_oxe(source, target)
    model.load_state_dict(target, strict=True)
    checkpoint = checkpoint_dir / "base_pytorch_model.pt"
    torch.save(model.state_dict(), checkpoint)
    OmegaConf.save(cfg, OUTPUT / "config.yaml", resolve=True)
    OmegaConf.save(cfg, OUTPUT / "config.full.yaml", resolve=True)
    shutil.copy2(SOURCE_ROOT / "dataset_statistics.json", OUTPUT / "dataset_statistics.json")

    action_keys = [key for key in target if key.startswith("action_model.")]
    action_transferred = [key for key in action_keys if key in transferred]
    manifest = {
        "schema_version": 1,
        "identity": "StarVLA-PI Qwen3-VL-2B Bridge+RT-1 alignment initialization",
        "status": "requires Bridge+RT-1 alignment and joint training",
        "source": {
            "repo_id": "StarVLA/Qwen3VL-PI_v3-Bridge-RT_1",
            "revision": "72618cfb5084e020b30ad6fd77c288da0c41ed39",
            "checkpoint_sha256": SOURCE_SHA256,
            "action_dim": 7,
            "action_horizon": 16,
            "dit_depth": source_depth,
        },
        "target": {
            "qwen_repo_id": "Qwen/Qwen3-VL-2B-Instruct",
            "qwen_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
            "qwen_sha256": QWEN_SHA256,
            "checkpoint": str(checkpoint),
            "checkpoint_size": checkpoint.stat().st_size,
            "checkpoint_sha256": sha256_file(checkpoint),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "action_dim": 7,
            "action_horizon": 16,
            "dit_depth": target_depth,
        },
        "transfer": {
            "dit_layer_map_target_to_source": mapping,
            "action_tensor_count": len(action_keys),
            "action_transferred_tensor_count": len(action_transferred),
            "action_transferred_numel": sum(target[key].numel() for key in action_transferred),
            "action_total_numel": sum(target[key].numel() for key in action_keys),
            "transferred_tensors": transferred,
            "initialized_tensors": initialized,
        },
    }
    (OUTPUT / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"OXE initialization: {checkpoint}")
    print(f"Parameters: {manifest['target']['parameters']:,}")
    print(f"SHA256: {manifest['target']['checkpoint_sha256']}")
    print(
        f"Action transfer: {len(action_transferred)}/{len(action_keys)} tensors, "
        f"{manifest['transfer']['action_transferred_numel']:,}/"
        f"{manifest['transfer']['action_total_numel']:,} parameters"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
