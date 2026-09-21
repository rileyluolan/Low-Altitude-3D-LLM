#!/usr/bin/env python3
"""Paired pi0.5 state ablation. `plan` needs only Python; execution needs OpenPI."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys

from setup_sources import verify

ROOT = Path(__file__).resolve().parents[1]


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "prepare", "check", "train", "rollout"])
    parser.add_argument("--variant", choices=["state", "no_state"], default="state")
    parser.add_argument("--epochs", type=int, choices=[1, 5], default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fsdp-devices", type=int, choices=[1, 2, 4, 8], default=8)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--source-root", type=Path, default=ROOT / "third_party/XPolicyLab")
    parser.add_argument("--data-root", type=Path, help="HUGE_data/data_traj, containing train/test_seen/test_unseen")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "runtime")
    parser.add_argument("--run-name", help="One path component; defaults to paired_<epochs>ep_seed<seed>")
    parser.add_argument("--base-weights", default="gs://openpi-assets/checkpoints/pi05_base/params")
    parser.add_argument("--resume", action="store_true", help="Explicitly resume an identical recorded training run")
    parser.add_argument("--checkpoint", type=Path, help="Concrete checkpoint directory for rollout")
    parser.add_argument("--eval-tag", default="full", help="Separate smoke/full/repeated evaluation outputs")
    parser.add_argument("--host", default="127.0.0.1", help="Already running HUGE 3DGS renderer")
    parser.add_argument("--port", type=int, default=5550)
    parser.add_argument("--split", choices=["test_seen", "test_unseen", "test_seen,test_unseen"], default="test_seen,test_unseen")
    parser.add_argument("--num-trajs", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args(argv)
    if args.run_name and (Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}):
        parser.error("--run-name must be a single path component")
    if not args.eval_tag or Path(args.eval_tag).name != args.eval_tag or args.eval_tag in {".", ".."}:
        parser.error("--eval-tag must be a single path component")
    if args.num_workers < 0 or args.num_trajs < 0:
        parser.error("worker and trajectory counts must be nonnegative")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("Require 0 <= shard-index < num-shards")
    return args


def plan(args):
    name = "pi05_overall_state" if args.variant == "state" else "pi05_overall"
    if args.epochs == 5:
        name += "_5ep"
    work = args.work_dir.resolve()
    run_name = args.run_name or f"paired_{args.epochs}ep_seed{args.seed}"
    lock = json.loads((ROOT / "SOURCE.lock.json").read_text())
    return {
        "schema_version": 1,
        "config_name": name,
        "run_name": run_name,
        "discrete_state_input": args.variant == "state",
        "images": ["first_image", "image"],
        "state_fields": ["x", "y", "z", "yaw_rad"],
        "state_bins": 256,
        "prompt_from_task": True,
        "action_fields": ["dx", "dy", "dz", "dyaw"],
        "action_horizon": 20,
        "model_action_dim": 32,
        "extra_delta_transform": False,
        "dataset": "task_overall/train",
        "train_episodes": 5175,
        "train_frames": 1720096,
        "data_root": str(args.data_root.resolve()) if args.data_root else None,
        "initialization": args.base_weights,
        "seed": args.seed,
        "epochs_label": args.epochs,
        # Preserve the historical recipes (their epoch counts are approximate).
        "num_train_steps": 13438 * args.epochs,
        "batch_size": 128,
        "fsdp_devices": args.fsdp_devices,
        "num_workers": args.num_workers,
        "learning_rate": {"warmup_steps": 1000, "peak": 2.5e-5, "end": 2.5e-6},
        "checkpoint_root": str(work / "checkpoints" / name / run_name),
        "assets_root": str(work / "assets"),
        "normalization_sha256": lock["norm_stats"]["sha256"],
        "source_commit": lock["commit"],
        "source_patch_sha256": lock["patch_sha256"],
        "wandb_enabled": False,
    }


def prepare(args, recipe):
    if args.data_root is None:
        raise RuntimeError("--data-root must identify HUGE_data/data_traj")
    data_root = args.data_root.resolve()
    info = json.loads((data_root / "train/meta/info.json").read_text())
    expected = {"total_episodes": 5175, "total_frames": 1720096, "fps": 5}
    if any(info.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Dataset counts/frequency differ from the locked HUGE train split")
    for key, shape in {"state": [4], "actions": [4], "image": [256, 256, 3], "first_image": [256, 256, 3]}.items():
        if info["features"][key]["shape"] != shape:
            raise RuntimeError(f"Unexpected {key} shape")
    norm = ROOT / "assets/norm_stats.json"
    if hashlib.sha256(norm.read_bytes()).hexdigest() != recipe["normalization_sha256"]:
        raise RuntimeError("Normalization statistics differ from the source manifest")
    work = args.work_dir.resolve()
    link_root = work / "data/task_overall"
    link_root.mkdir(parents=True, exist_ok=True)
    required = ["train"] if args.action != "rollout" else ["train", *args.split.split(",")]
    for split in required:
        source = data_root / split
        if not (source / "meta/info.json").is_file():
            raise RuntimeError(f"Missing dataset: {source}")
        target = link_root / split
        if target.is_symlink() or target.exists():
            if target.resolve() != source:
                raise RuntimeError(f"Refusing to replace dataset path: {target}")
        else:
            target.symlink_to(source, target_is_directory=True)
    # Both arms use byte-identical statistics; no validation/test statistics are computed.
    target = Path(recipe["assets_root"]) / recipe["config_name"] / "task_overall/train/norm_stats.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != norm.read_bytes():
        raise RuntimeError(f"Refusing to replace different statistics: {target}")
    if not target.exists():
        shutil.copyfile(norm, target)


def load_config(args, recipe):
    verify(args.source_root.resolve())
    source = args.source_root.resolve() / "policy/Pi_05/openpi"
    sys.path.insert(0, str(source / "src"))
    work = args.work_dir.resolve()
    os.environ.pop("LEROBOT_HOME", None)
    os.environ["HF_LEROBOT_HOME"] = str(work / "data")
    os.environ["OPENPI_DATA_HOME"] = str(work / "cache/openpi")
    os.environ["HF_HUB_CACHE"] = str(work / "cache/huggingface/hub")
    os.environ["HF_DATASETS_CACHE"] = str(work / "cache/hf-datasets")
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(work / "cache/jax")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from openpi.training import config, weight_loaders

    # Derive both arms from one recipe so the treatment is only the state flag.
    base = config.get_config("pi05_overall")
    cfg = dataclasses.replace(
        base,
        name=recipe["config_name"],
        exp_name=recipe["run_name"],
        model=dataclasses.replace(base.model, discrete_state_input=recipe["discrete_state_input"]),
        data=dataclasses.replace(base.data, assets=config.AssetsConfig()),
        weight_loader=weight_loaders.CheckpointWeightLoader(recipe["initialization"]),
        seed=recipe["seed"],
        batch_size=recipe["batch_size"],
        fsdp_devices=recipe["fsdp_devices"],
        num_workers=recipe["num_workers"],
        num_train_steps=recipe["num_train_steps"],
        lr_schedule=dataclasses.replace(base.lr_schedule, decay_steps=recipe["num_train_steps"]),
        assets_base_dir=recipe["assets_root"],
        checkpoint_base_dir=str(work / "checkpoints"),
        checkpoint_dir_override=None,
        keep_period=4000 if args.epochs == 1 else 10000,
        overwrite=False,
        resume=args.resume,
        wandb_enabled=False,
    )
    config._CONFIGS_DICT[cfg.name] = cfg
    return cfg, source


def record_training(args, recipe):
    record = args.work_dir.resolve() / "records" / recipe["config_name"] / recipe["run_name"] / "recipe.json"
    checkpoint = Path(recipe["checkpoint_root"])
    if args.resume:
        if not record.is_file() or json.loads(record.read_text()) != recipe or not checkpoint.is_dir():
            raise RuntimeError("Resume requires an existing checkpoint and the identical recorded recipe")
    else:
        if record.exists() or checkpoint.exists():
            raise RuntimeError("Run already exists; choose --run-name or explicitly --resume")
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(recipe, indent=2) + "\n")


def main(argv=None):
    args = arguments(argv)
    recipe = plan(args)
    print(json.dumps(recipe, indent=2), flush=True)
    if args.action == "plan":
        return
    prepare(args, recipe)
    if args.action == "prepare":
        return
    cfg, source = load_config(args, recipe)
    if args.action == "check":
        data = cfg.data.create(cfg.assets_dirs, cfg.model)
        if set(data.norm_stats) != {"state", "actions"}:
            raise RuntimeError("Expected both state and action normalization statistics")
        print(f"Config and transforms resolved: {cfg.name}; no training was started")
    elif args.action == "train":
        record_training(args, recipe)
        runpy.run_path(str(source / "scripts/train.py"))["main"](cfg)
    else:
        if not args.checkpoint:
            raise RuntimeError("rollout requires --checkpoint pointing to this arm's trained checkpoint")
        checkpoint = args.checkpoint.resolve()
        if not (checkpoint / "params").is_dir() or not (checkpoint / "assets").is_dir():
            raise RuntimeError("Checkpoint requires params/ and assets/")
        record = args.work_dir.resolve() / "records" / cfg.name / cfg.exp_name / "recipe.json"
        if not record.is_file() or json.loads(record.read_text()) != recipe:
            raise RuntimeError("Rollout must match a recorded training recipe")
        if checkpoint.parent != Path(recipe["checkpoint_root"]) or not checkpoint.name.isdigit():
            raise RuntimeError("Checkpoint belongs to a different run or arm")
        output = args.work_dir.resolve() / "results" / cfg.name / cfg.exp_name / checkpoint.name / args.eval_tag / f"shard_{args.shard_index}"
        if output.exists():
            raise RuntimeError(f"Rollout output already exists: {output}")
        sys.argv = [str(source / "scripts/action_infer.py"),
                    "--task_id", "overall", "--config_name", cfg.name,
                    "--checkpoint_dir", str(checkpoint), "--out_dir", str(output),
                    "--splits", args.split, "--host", args.host, "--port", str(args.port),
                    "--exec_steps", "10", "--seed", str(args.seed),
                    "--num_trajs", str(args.num_trajs), "--num_shards", str(args.num_shards),
                    "--shard_index", str(args.shard_index)]
        runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
