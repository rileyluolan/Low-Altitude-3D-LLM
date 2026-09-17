#!/usr/bin/env python3
"""Install OXE metadata overlays and cheap caches without scanning parquet."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path


from experiment_paths import ROOT, STARVLA, MODEL, load_config
DATA_ROOT = ROOT / "data/oxe"
SOURCE_ROOT = ROOT / "artifacts/source/StarVLA-Qwen3VL-PI_v3-Bridge-RT_1"
SPECS = {
    "bridge_orig_1.0.0_lerobot": {
        "tag": "oxe_bridge",
        "modality": STARVLA / "examples/simBenchmarks/SimplerEnv/train_files/modality.json",
    },
    "fractal20220817_data_0.1.0_lerobot": {
        "tag": "oxe_rt1",
        "modality": STARVLA / "examples/simBenchmarks/SimplerEnv/train_files/fractal_modality.json",
    },
}


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def prepare_dataset(name: str, spec: dict, source_stats: dict, require_full: bool) -> None:
    dataset = DATA_ROOT / name
    meta = dataset / "meta"
    info_path = meta / "info.json"
    episodes_path = meta / "episodes.jsonl"
    for path in (info_path, episodes_path, meta / "tasks.jsonl"):
        if not path.is_file():
            raise FileNotFoundError(path)
    info = json.loads(info_path.read_text())
    shutil.copy2(spec["modality"], meta / "modality.json")

    stats = source_stats[spec["tag"]]
    write_json_atomic(
        meta / "stats_gr00t.json",
        {
            "__format_version": 2,
            "__cache_config": {"mode": "abs"},
            "statistics": {
                "observation.state": {
                    key: value for key, value in stats["state"].items() if key != "mask"
                },
                "action": {
                    key: value for key, value in stats["action"].items() if key != "mask"
                },
            },
        },
    )

    episodes = []
    nonempty = 0
    with episodes_path.open() as stream:
        for line in stream:
            episode = json.loads(line)
            tasks = episode.get("tasks", [])
            if tasks and str(tasks[0]).strip():
                nonempty += 1
                episodes.append((int(episode["episode_index"]), int(episode["length"])))
    expected_transitions = int(source_stats[spec["tag"]]["num_transitions"])
    expected_trajectories = nonempty

    steps_path = meta / "steps_data_index.pkl"
    expected_steps = sum(length for _, length in episodes)
    if expected_steps != expected_transitions:
        raise ValueError(f"{name}: {expected_steps} valid frames != source {expected_transitions}")
    rebuild = True
    if steps_path.is_file():
        try:
            with steps_path.open("rb") as stream:
                cached = pickle.load(stream)
            rebuild = int(cached.get("total_steps", -1)) != expected_steps
        except Exception:
            rebuild = True
    if rebuild:
        steps = [(episode, step) for episode, length in episodes for step in range(length)]
        temporary = steps_path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(
                {
                    "config_key": "experiment_prevalidated",
                    "steps": steps,
                    "num_trajectories": len(episodes),
                    "total_steps": len(steps),
                    "delete_pause_frame": False,
                },
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        os.replace(temporary, steps_path)

    if require_full:
        video_key = next(iter(json.loads((meta / "modality.json").read_text())["video"]))
        chunks_size = int(info["chunks_size"])
        data_template = info["data_path"]
        video_template = info["video_path"]
        missing_data = []
        missing_video = []
        for episode, _ in episodes:
            values = {
                "episode_chunk": episode // chunks_size,
                "episode_index": episode,
                "video_key": f"observation.images.{video_key}",
            }
            if not (dataset / data_template.format(**values)).is_file():
                missing_data.append(episode)
            if not (dataset / video_template.format(**values)).is_file():
                missing_video.append(episode)
        if missing_data or missing_video:
            raise RuntimeError(
                f"{name}: incomplete valid-language episodes, "
                f"missing parquet={len(missing_data)}/{expected_trajectories} "
                f"(first={missing_data[:5]}), missing selected videos="
                f"{len(missing_video)}/{expected_trajectories} (first={missing_video[:5]})"
            )
    print(f"{name}: {len(episodes):,} episodes, {expected_steps:,} indexed frames")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-full-data", action="store_true")
    args = parser.parse_args()
    source_stats = json.loads((SOURCE_ROOT / "dataset_statistics.json").read_text())
    for name, spec in SPECS.items():
        prepare_dataset(name, spec, source_stats, args.require_full_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
