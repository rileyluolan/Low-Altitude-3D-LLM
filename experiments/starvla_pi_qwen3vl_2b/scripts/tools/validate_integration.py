#!/usr/bin/env python3
"""Cheap preflight: validate source pin, model metadata, registry, and data schema."""

import json
import subprocess
from pathlib import Path

import yaml


from experiment_paths import ROOT, WORKSPACE, STARVLA, HUGEBENCH, GAUSSIAN, HUGE_DATA, MODEL, VENV, BASE, config_dict
EXPECTED_COMMIT = "2f17402a5ccaa09907516ae5e542b0fa6ee5d155"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


head = subprocess.check_output(
    ["git", "-C", str(STARVLA), "rev-parse", "HEAD"], text=True
).strip()
require(head == EXPECTED_COMMIT, f"StarVLA commit mismatch: {head}")

cfg = config_dict(ROOT / "configs/qwen3vl_2b_hugebench_smoke.yaml")
framework_cfg = cfg["framework"]
action_cfg = framework_cfg["action_model"]
dataset_cfg = cfg["datasets"]["vla_data"]
require(framework_cfg["name"] == "QwenPI_v3", "Framework must be QwenPI_v3")
require(action_cfg["action_dim"] == 4, "HUGE-Bench action_dim must be 4")
require(action_cfg["state_dim"] == 4, "HUGE-Bench state_dim must be 4")
require(action_cfg["action_horizon"] == 20, "Action horizon must be 20")
require(dataset_cfg["action_mode"] == "abs", "Stored deltas must not be differenced again")

info_path = Path(dataset_cfg["data_root_dir"]) / "hugebench_train" / "meta" / "info.json"
with info_path.open() as handle:
    info = json.load(handle)
require(info["features"]["state"]["shape"] == [4], "Dataset state shape is not [4]")
require(info["features"]["actions"]["shape"] == [4], "Dataset action shape is not [4]")
require(info["features"]["image"]["shape"] == [256, 256, 3], "Unexpected current image shape")
require(info["features"]["first_image"]["shape"] == [256, 256, 3], "Unexpected first image shape")

from starVLA.dataloader.gr00t_lerobot.registry import (  # noqa: E402
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
)

require("hugebench_train" in DATASET_NAMED_MIXTURES, "HUGE-Bench mixture not discovered")
data_cfg = ROBOT_TYPE_CONFIG_MAP["hugebench_uav"]
require(len(data_cfg.action_indices) == 20, "Registry action chunk is not 20")
require(data_cfg.video_keys == ["video.first_image", "video.current_image"], "Image order changed")

model_config = Path(framework_cfg["qwenvl"]["base_vlm"]) / "config.json"
if model_config.exists():
    with model_config.open() as handle:
        model_meta = json.load(handle)
    text_cfg = model_meta.get("text_config", model_meta)
    require(text_cfg["hidden_size"] == 2048, "Unexpected Qwen3-VL-2B hidden size")
    require(text_cfg["num_hidden_layers"] == 28, "Unexpected Qwen3-VL-2B layer count")
    model_status = "present (2048 hidden, 28 layers)"
else:
    model_status = "not downloaded; structural metadata check skipped"

print("Integration validation passed")
print(f"StarVLA commit: {head}")
print(f"Qwen3-VL-2B:    {model_status}")
print(f"HUGE-Bench:     {info['total_episodes']} episodes, {info['total_frames']} frames")
