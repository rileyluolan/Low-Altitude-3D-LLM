#!/usr/bin/env python3
"""Build StarVLA's step index from trusted LeRobot episode metadata."""

import argparse
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("dataset_view", type=Path)
args = parser.parse_args()
meta = args.dataset_view / "meta"

with (meta / "tasks.jsonl").open() as handle:
    tasks = [json.loads(line) for line in handle if line.strip()]
if not tasks or any(not str(task.get("task", "")).strip() for task in tasks):
    raise RuntimeError("tasks.jsonl contains a missing or empty instruction")

steps = []
with (meta / "episodes.jsonl").open() as handle:
    episodes = [json.loads(line) for line in handle if line.strip()]
for episode in episodes:
    episode_index = int(episode["episode_index"])
    steps.extend((episode_index, index) for index in range(int(episode["length"])))

config = {"delete_pause_frame": False, "dataset_name": args.dataset_view.name}
config_key = hashlib.md5(str(sorted(config.items())).encode()).hexdigest()[:12]
payload = {
    "config_key": config_key,
    "steps": steps,
    "num_trajectories": len(episodes),
    "total_steps": len(steps),
    "computed_timestamp": datetime.now(timezone.utc).isoformat(),
    "delete_pause_frame": False,
}
with (meta / "steps_data_index.pkl").open("wb") as handle:
    pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Indexed {len(steps)} steps from {len(episodes)} episodes")
