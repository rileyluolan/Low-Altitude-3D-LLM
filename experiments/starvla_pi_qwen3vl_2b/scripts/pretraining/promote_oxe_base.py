#!/usr/bin/env python3
"""Promote the completed Bridge+RT-1 run to the immutable mature-base layout."""

import hashlib
import json
import shutil
from pathlib import Path


from experiment_paths import ROOT, config_dict
RUN = ROOT / "artifacts/checkpoints/qwen3vl_2b_pi_v3_bridge_rt1_joint"
SOURCE = RUN / "final_model/pytorch_model.pt"
OUTPUT = ROOT / "artifacts/base/qwen3vl_2b_pi_v3_bridge_rt1_mature"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(32 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    from verify_stage import verify_stage
    verified = [verify_stage("align"), verify_stage("joint")]
    cfg = config_dict(RUN / "config.full.yaml")
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    checkpoint_dir = OUTPUT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = checkpoint_dir / "base_pytorch_model.pt"
    if target.exists():
        if sha256_file(target) != sha256_file(SOURCE):
            raise FileExistsError(f"Existing mature base differs: {target}")
    else:
        shutil.copy2(SOURCE, target)
    for name in ("config.yaml", "config.full.yaml", "dataset_statistics.json"):
        source = RUN / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, OUTPUT / name)
    manifest = {
        "schema_version": 1,
        "verified_stages": verified,
        "identity": "mature StarVLA-PI Qwen3-VL-2B Bridge+RT-1 base",
        "status": "training_complete; downstream benchmark validation required",
        "training": {
            "alignment_run": "qwen3vl_2b_pi_v3_bridge_rt1_align",
            "alignment_steps": 5000,
            "alignment_trainable": ["project_layers"],
            "joint_run": "qwen3vl_2b_pi_v3_bridge_rt1_joint",
            "joint_steps": 50000,
            "mixture": {"bridge_orig": 0.5, "fractal_rt1": 0.5},
            "effective_batch": 128,
            "seed": cfg["seed"],
            "learning_rates": {
                key: cfg["trainer"]["learning_rate"][key]
                for key in ("qwen_vl_interface", "project_layers", "action_model")
            },
            "valid_language_data": {
                "bridge": {"episodes": 38660, "frames": 1305714},
                "rt1": {"episodes": 87204, "frames": 3786152},
            },
        },
        "evaluation": {
            "simpler_env_widowx": "pending",
            "note": "Training maturity does not imply equal 2B/4B benchmark accuracy.",
        },
        "checkpoint": {
            "path": str(target),
            "size": target.stat().st_size,
            "sha256": sha256_file(target),
        },
    }
    (OUTPUT / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Mature OXE base: {target}")
    print(f"SHA256: {manifest['checkpoint']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
