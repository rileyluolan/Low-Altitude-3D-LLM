#!/usr/bin/env python3
"""Download the small, revision-pinned metadata files for both OXE datasets."""

from pathlib import Path

from huggingface_hub import hf_hub_download


from experiment_paths import ROOT, STARVLA, MODEL, load_config
ITEMS = [
    (
        "IPEC-COMMUNITY/bridge_orig_lerobot",
        "0e9d76d07e9df3ea3eba257b2520d4913833fad2",
        "bridge_orig_1.0.0_lerobot",
    ),
    (
        "IPEC-COMMUNITY/fractal20220817_data_lerobot",
        "91bf7d7f7ce50770a1ba5c6db14b8d1c0815122e",
        "fractal20220817_data_0.1.0_lerobot",
    ),
]
FILES = ["README.md", "meta/info.json", "meta/stats.json", "meta/tasks.jsonl", "meta/episodes.jsonl"]


def main() -> int:
    for repo, revision, directory in ITEMS:
        for filename in FILES:
            path = hf_hub_download(
                repo_id=repo,
                repo_type="dataset",
                revision=revision,
                filename=filename,
                local_dir=ROOT / "data/oxe" / directory,
            )
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
