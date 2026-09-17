#!/usr/bin/env python3
"""Fail closed if the Qwen3-VL-2B OXE maturity recipe drifts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from omegaconf import OmegaConf


from experiment_paths import ROOT, STARVLA, MODEL, load_config
EXPECTED_COT = (
    "Your task is {instruction}. To identify the key objects for your task. "
    "Locate their bounding boxes in [x1,y1,x2,y2] format."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    model_cfg = json.loads((MODEL / "config.json").read_text())
    text_cfg = model_cfg["text_config"]
    assert text_cfg["hidden_size"] == 2048
    assert text_cfg["num_hidden_layers"] == 28

    align = OmegaConf.load(ROOT / "configs/qwen3vl_2b_oxe_align.yaml")
    joint = OmegaConf.load(ROOT / "configs/qwen3vl_2b_oxe_joint.yaml")
    for cfg in (align, joint):
        assert cfg.framework.name == "QwenPI_v3"
        assert cfg.framework.action_model.action_dim == 7
        assert cfg.framework.action_model.action_horizon == 16
        assert cfg.framework.action_model.diffusion_model_cfg.action_dit_hidden_dim == 1024
        assert cfg.datasets.vla_data.data_mix == "bridge_rt_1"
        assert cfg.datasets.vla_data.balance_dataset_weights is False
        assert cfg.datasets.vla_data.balance_trajectory_weights is False
        assert cfg.datasets.vla_data.filter_trajectories_to_cached_steps is True
        assert cfg.datasets.vla_data.CoT_prompt == EXPECTED_COT
    assert align.trainer.max_train_steps == 5000
    assert align.trainer.freeze_modules == "qwen_vl_interface,action_model"
    assert align.trainer.learning_rate.project_layers == 1e-4
    assert joint.trainer.max_train_steps == 50000
    assert joint.trainer.freeze_modules == ""
    assert joint.trainer.learning_rate.qwen_vl_interface == 1e-5
    assert joint.trainer.learning_rate.project_layers == 1e-4
    assert joint.trainer.learning_rate.action_model == 1e-4

    ds = json.loads((ROOT / "configs/ds_zero2_oxe.json").read_text())
    assert ds["zero_optimization"]["stage"] == 2
    assert ds["gradient_accumulation_steps"] == 16
    assert 1 * 8 * ds["gradient_accumulation_steps"] == 128
    assert align.trainer.gradient_accumulation_steps == joint.trainer.gradient_accumulation_steps == 16

    conversion = json.loads(
        (
            ROOT
            / "artifacts/base/qwen3vl_2b_pi_v3_bridge_rt1_init/conversion_manifest.json"
        ).read_text()
    )
    transfer = conversion["transfer"]
    assert transfer["action_transferred_tensor_count"] == transfer["action_tensor_count"] == 416
    assert transfer["action_transferred_numel"] == transfer["action_total_numel"]
    assert conversion["source"]["dit_depth"] == 36
    assert conversion["target"]["dit_depth"] == 28
    non_backbone_new = [
        key
        for key in transfer["initialized_tensors"]
        if not key.startswith(("qwen_vl_interface.", "project_layers."))
    ]
    assert not non_backbone_new, non_backbone_new

    lock = OmegaConf.load(ROOT / "SOURCE.lock.yaml")
    head = subprocess.check_output(
        ["git", "-C", str(STARVLA), "rev-parse", "HEAD"], text=True
    ).strip()
    assert head == lock.starvla.commit
    for patch in lock.starvla.local_patches:
        assert sha256_file(STARVLA / patch.path) == patch.sha256

    print("OXE recipe validation: PASS")
    print("Qwen3-VL-2B: hidden=2048, layers=28")
    print("Action transfer: 416/416 tensors (100%)")
    print("Schedule: 5k projector alignment + 50k joint, effective batch=128")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
