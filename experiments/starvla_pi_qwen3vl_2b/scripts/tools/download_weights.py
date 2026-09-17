#!/usr/bin/env python3
"""Download the two public, pinned inputs; checkpoint construction verifies SHA256."""
from huggingface_hub import snapshot_download
from experiment_paths import ROOT, MODEL

snapshot_download("Qwen/Qwen3-VL-2B-Instruct", revision="89644892e4d85e24eaac8bacfd4f463576704203", local_dir=MODEL)
snapshot_download("StarVLA/Qwen3VL-PI_v3-Bridge-RT_1", revision="72618cfb5084e020b30ad6fd77c288da0c41ed39",
    local_dir=ROOT / "artifacts/source/StarVLA-Qwen3VL-PI_v3-Bridge-RT_1",
    allow_patterns=["checkpoints/steps_50000_pytorch_model.pt", "config.yaml", "config.full.yaml", "dataset_statistics.json", "README.md"])
