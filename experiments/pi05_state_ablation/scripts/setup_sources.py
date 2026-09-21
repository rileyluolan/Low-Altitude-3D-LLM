#!/usr/bin/env python3
"""Restore the pinned, experiment-owned OpenPI source; does not start training."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def verify(target):
    lock = json.loads((ROOT / "SOURCE.lock.json").read_text())
    head = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    if head != lock["commit"]:
        raise RuntimeError(f"Source revision differs: {head}")
    for entry in lock["snapshot_files"]:
        path = target / "policy/Pi_05/openpi" / entry["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Source snapshot differs: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=ROOT / "third_party/XPolicyLab")
    args = parser.parse_args()
    target = args.target.resolve()
    lock = json.loads((ROOT / "SOURCE.lock.json").read_text())
    patch = ROOT / "patches/0001-hugebench-snapshot.patch"
    if hashlib.sha256(patch.read_bytes()).hexdigest() != lock["patch_sha256"]:
        raise RuntimeError("Source patch checksum differs")
    if target.exists():
        verify(target)
        print(f"Existing source verified: {target}")
        return
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", lock["repository"], str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "sparse-checkout", "set", "policy/Pi_05/openpi"], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", lock["commit"]], check=True)
    subprocess.run(["git", "-C", str(target), "apply", "--check", str(patch)], check=True)
    subprocess.run(["git", "-C", str(target), "apply", str(patch)], check=True)
    verify(target)
    print(f"Source snapshot restored and verified: {target}")


if __name__ == "__main__":
    main()
