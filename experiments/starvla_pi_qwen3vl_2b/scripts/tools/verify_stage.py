#!/usr/bin/env python3
"""Check a final checkpoint against its saved recipe and terminal step record."""
import argparse
import json
from pathlib import Path
from experiment_paths import ROOT, config_dict


def verify_stage(phase):
    expected = {"align": 5000, "joint": 50000}[phase]
    run = ROOT / "artifacts/checkpoints" / f"qwen3vl_2b_pi_v3_bridge_rt1_{phase}"
    final = run / "final_model/pytorch_model.pt"
    if not final.is_file() or final.stat().st_size == 0:
        raise FileNotFoundError(final)
    cfg = config_dict(run / "config.full.yaml")
    if cfg["trainer"]["max_train_steps"] != expected:
        raise ValueError(f"{phase}: recipe does not specify {expected} steps")
    steps = [json.loads(s)["steps"] for s in (run / "summary.jsonl").read_text().splitlines() if s.strip()]
    if not steps or steps[-1] != expected:
        raise ValueError(f"{phase}: no completed {expected}-step record")
    if phase == "align" and cfg["trainer"]["freeze_modules"] != "qwen_vl_interface,action_model":
        raise ValueError("Alignment must train projectors only")
    if phase == "joint" and cfg["trainer"]["freeze_modules"]:
        raise ValueError("Joint phase must train all modules")
    recipe = json.loads((run / "launcher_recipe.json").read_text())
    if recipe["num_processes"] * cfg["datasets"]["vla_data"]["per_device_batch_size"] * cfg["trainer"]["gradient_accumulation_steps"] != 128:
        raise ValueError("OXE global batch must be 128")
    return {"phase": phase, "steps": expected, "checkpoint": str(final.relative_to(ROOT)),
            "checkpoint_bytes": final.stat().st_size, "global_batch": 128, "launcher": recipe}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["align", "joint"])
    print(json.dumps(verify_stage(p.parse_args().phase), indent=2))
