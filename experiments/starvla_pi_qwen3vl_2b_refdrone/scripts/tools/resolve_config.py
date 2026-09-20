#!/usr/bin/env python3
"""Resolve portable templates and keep Accelerate/DeepSpeed accumulation identical."""
import argparse
import json
from pathlib import Path
from omegaconf import OmegaConf
from experiment_paths import ROOT, load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("profile", choices=["train", "smoke", "oxe_align", "oxe_joint"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-processes", type=int, default=8)
    a = p.parse_args()
    oxe = a.profile.startswith("oxe_")
    name = a.profile if oxe else "hugebench_" + a.profile
    cfg = load_config(ROOT / f"configs/qwen3vl_2b_{name}.yaml")
    accel = load_config(ROOT / f"configs/deepspeed_zero2_{'oxe' if oxe else a.profile}.yaml")
    ds = json.loads(Path(accel.deepspeed_config.deepspeed_config_file).read_text())
    if oxe:
        if a.num_processes not in (4, 8):
            p.error("OXE supports 8 GPUs (historical recipe) or 4 GPUs (same global batch).")
        accel.num_processes = a.num_processes
        cfg.trainer.gradient_accumulation_steps = 128 // a.num_processes
    ds["gradient_accumulation_steps"] = cfg.trainer.gradient_accumulation_steps
    batch = accel.num_processes * cfg.datasets.vla_data.per_device_batch_size * cfg.trainer.gradient_accumulation_steps
    expected = 128 if oxe else (512 if a.profile == "train" else 8)
    if batch != expected:
        raise ValueError(f"Global batch {batch} != recipe {expected}")
    a.output.mkdir(parents=True, exist_ok=True)
    ds_path = (a.output / "deepspeed.json").resolve()
    ds_path.write_text(json.dumps(ds, indent=2) + "\n")
    accel.deepspeed_config.deepspeed_config_file = str(ds_path)
    OmegaConf.save(cfg, a.output / "train.yaml", resolve=True)
    OmegaConf.save(accel, a.output / "accelerate.yaml", resolve=True)
    print(f"Resolved {a.profile}: {accel.num_processes} GPUs, batch={batch}, steps={cfg.trainer.max_train_steps}")

if __name__ == "__main__":
    main()
