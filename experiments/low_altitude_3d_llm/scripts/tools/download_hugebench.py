#!/usr/bin/env python3
"""Download pinned HUGE trajectories and/or seven official 3D scene archives."""
import argparse
import json
import tarfile
from pathlib import Path
from huggingface_hub import snapshot_download
from experiment_paths import ROOT, HUGE_DATA, config_dict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--part", choices=["all", "train", "eval"], default="all")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    lock = config_dict(ROOT / "SOURCE.lock.yaml")
    trajectories, scenes = lock["hugebench_trajectories"], lock["hugebench_scenes"]
    splits = ["train"] if a.part == "train" else (["test_seen", "test_unseen"] if a.part == "eval" else trajectories["splits"])
    snapshot_download(trajectories["repository"], repo_type="dataset", revision=trajectories["revision"],
                      local_dir=HUGE_DATA / "data_traj", allow_patterns=[s + "/**" for s in splits], max_workers=a.workers)
    if a.part != "train":
        archives = ROOT / "downloads/3DGS_Mesh_Envs"
        names = [f"archives/3DGS_Mesh_Envs_{s}.tar" for s in scenes["scenes"]]
        snapshot_download(scenes["repository"], repo_type="dataset", revision=scenes["revision"],
                          local_dir=archives, allow_patterns=names, max_workers=a.workers)
        for name in names:
            with tarfile.open(archives / name) as stream:
                members = stream.getmembers()
                for member in members:
                    target = (HUGE_DATA / member.name).resolve()
                    if not target.is_relative_to((HUGE_DATA / "data_3d").resolve()) or not (member.isfile() or member.isdir()):
                        raise ValueError(f"Unexpected archive member: {member.name}")
                files = [m for m in members if m.isfile()]
                if not all((HUGE_DATA / m.name).is_file() and (HUGE_DATA / m.name).stat().st_size == m.size for m in files):
                    stream.extractall(HUGE_DATA, members=members, filter="data")
                if not all((HUGE_DATA / m.name).stat().st_size == m.size for m in files):
                    raise ValueError(f"Incomplete extraction: {name}")
    (HUGE_DATA / "download_revisions.json").write_text(json.dumps({"trajectories": trajectories, "scenes": scenes, "requested_splits": splits}, indent=2) + "\n")
    print(f"HUGE data ready at {HUGE_DATA}")

if __name__ == "__main__":
    main()
