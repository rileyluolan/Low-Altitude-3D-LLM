#!/usr/bin/env python3
"""Restore Release asset chunks into their original experiments/*/artifacts paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import urllib.parse
import urllib.request

BLOCK = 8 * 1024 * 1024
EXPERIMENTS = {'starvla_pi_qwen3vl_2b', 'low_altitude_3d_llm'}


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(BLOCK), b''):
            h.update(block)
    return h.hexdigest()


def restore(manifest, workspace, base_url=None):
    if manifest['repository'] != 'rileyluolan/Low-Altitude-3D-LLM':
        raise ValueError('Unexpected repository in manifest.')
    base = base_url or ('https://github.com/' + manifest['repository'] + '/releases/download/'
                        + urllib.parse.quote(manifest['tag'], safe='') + '/')
    root = workspace.resolve()
    for entry in manifest['files']:
        rel = PurePosixPath(entry['path'])
        if (rel.is_absolute() or '..' in rel.parts or len(rel.parts) < 4
                or rel.parts[0] != 'experiments' or rel.parts[1] not in EXPERIMENTS
                or rel.parts[2] != 'artifacts'):
            raise ValueError(f'Unsafe artifact path: {rel}')
        dest = root.joinpath(*rel.parts)
        temp = dest.with_name(dest.name + '.release-partial')
        if not dest.resolve().is_relative_to(root) or not temp.resolve().is_relative_to(root):
            raise ValueError('Path escapes workspace through a symlink.')
        if dest.exists():
            if dest.stat().st_size != entry['size'] or checksum(dest) != entry['sha256']:
                raise ValueError(f'Refusing to overwrite different existing file: {dest}')
            print(f'Already verified: {rel}', flush=True)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        total, whole = 0, hashlib.sha256()
        # A failed file restarts at its first part; completed files are skipped after verification.
        with temp.open('wb') as out:
            for part in entry['parts']:
                url = base + urllib.parse.quote(part['asset'], safe='')
                h, size = hashlib.sha256(), 0
                with urllib.request.urlopen(url, timeout=120) as response:
                    for block in iter(lambda: response.read(BLOCK), b''):
                        size += len(block)
                        if size > part['size']:
                            raise ValueError('Downloaded part exceeds expected size.')
                        out.write(block)
                        h.update(block)
                        whole.update(block)
                if size != part['size'] or h.hexdigest() != part['sha256']:
                    raise ValueError(f'Part checksum mismatch: {part["asset"]}')
                total += size
        if total != entry['size'] or whole.hexdigest() != entry['sha256']:
            raise ValueError(f'Full file checksum mismatch: {rel}')
        # Exclusive creation prevents replacing a file written while the download ran.
        os.link(temp, dest)
        temp.unlink()
        print(f'Restored: {rel}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--workspace', type=Path, required=True)
    a = p.parse_args()
    restore(json.loads(a.manifest.read_text()), a.workspace)
