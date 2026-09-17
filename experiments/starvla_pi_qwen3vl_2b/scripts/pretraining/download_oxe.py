#!/usr/bin/env python3
"""Download only the OXE files consumed by the Bridge+RT-1 training recipe.

The Bridge repository contains four camera streams, while QwenPI_v3 uses only
image_0.  Skipping the other three cameras saves roughly 16 GB without changing
the released StarVLA input contract.  Downloads are revision-pinned, atomic,
parallel, and resumable.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path



from experiment_paths import ROOT, STARVLA, MODEL, load_config
DATA_ROOT = ROOT / "data/oxe"
DATASETS = {
    "bridge": {
        "repo": "IPEC-COMMUNITY/bridge_orig_lerobot",
        "revision": "0e9d76d07e9df3ea3eba257b2520d4913833fad2",
        "directory": "bridge_orig_1.0.0_lerobot",
        "video_key": "observation.images.image_0",
    },
    "rt1": {
        "repo": "IPEC-COMMUNITY/fractal20220817_data_lerobot",
        "revision": "91bf7d7f7ce50770a1ba5c6db14b8d1c0815122e",
        "directory": "fractal20220817_data_0.1.0_lerobot",
        "video_key": "observation.images.image",
    },
}
def download_one(repo: str, revision: str, relative: str, target: Path) -> int:
    # Hub manages revision metadata, atomic writes, resume and expected file size.
    # Do not promote an arbitrary .part file after a Range/416 response.
    from huggingface_hub import hf_hub_download
    before = target.stat().st_size if target.is_file() else 0
    root = target.parents[len(Path(relative).parts) - 1]
    path = Path(hf_hub_download(repo_id=repo, repo_type="dataset", revision=revision,
                               filename=relative, local_dir=root))
    return path.stat().st_size if not before else 0


def valid_episodes(dataset_root: Path) -> list[int]:
    episodes = []
    with (dataset_root / "meta/episodes.jsonl").open() as stream:
        for line in stream:
            episode = json.loads(line)
            tasks = episode.get("tasks", [])
            if tasks and str(tasks[0]).strip():
                episodes.append(int(episode["episode_index"]))
    return episodes


def expected_files(spec: dict, dataset_root: Path, episodes: list[int]):
    with (dataset_root / "meta/info.json").open() as stream:
        info = json.load(stream)
    for episode in episodes:
        chunk = episode // int(info["chunks_size"])
        yield (
            f"data/chunk-{chunk:03d}/episode_{episode:06d}.parquet",
            dataset_root / f"data/chunk-{chunk:03d}/episode_{episode:06d}.parquet",
        )
        video_key = spec["video_key"]
        yield (
            f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode:06d}.mp4",
            dataset_root
            / f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode:06d}.mp4",
        )


def batched(iterator, size: int):
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def download_dataset(name: str, spec: dict, workers: int) -> None:
    dataset_root = DATA_ROOT / spec["directory"]
    if not (dataset_root / "meta/info.json").is_file():
        raise FileNotFoundError(
            f"Missing metadata for {name}; run scripts/download_oxe.sh first"
        )
    episodes = valid_episodes(dataset_root)
    total = 2 * len(episodes)
    completed = 0
    downloaded_bytes = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for batch in batched(expected_files(spec, dataset_root, episodes), 2048):
            args = [
                (spec["repo"], spec["revision"], relative, target)
                for relative, target in batch
            ]
            for size in pool.map(lambda values: download_one(*values), args):
                completed += 1
                downloaded_bytes += size
            elapsed = max(time.time() - started, 1.0)
            print(
                f"[{name}] {completed:,}/{total:,} files; "
                f"new {downloaded_bytes / 2**30:.2f} GiB; {completed / elapsed:.1f} files/s",
                flush=True,
            )
    print(f"[{name}] complete: {total:,} training files", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["bridge", "rt1", "both"], default="both")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    selected = DATASETS if args.dataset == "both" else {args.dataset: DATASETS[args.dataset]}
    for name, spec in selected.items():
        download_dataset(name, spec, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
