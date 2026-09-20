#!/usr/bin/env python3
"""Fetch catalogued model files from Blob and verify their SHA256."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(32 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('groups', nargs='*', help='Group IDs shown by --list')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--config', type=Path, help='Private Blob JSON config; never commit it')
    parser.add_argument('--metadata-only', action='store_true', help='Skip .pt and .safetensors')
    parser.add_argument('--verify-only', action='store_true', help='Only check existing local files')
    args = parser.parse_args()
    catalog = json.loads(Path(__file__).with_name('manifest.json').read_text())
    if args.list:
        for name, group in catalog['groups'].items():
            size = sum(item['bytes'] for item in group['files'])
            print(f'{name}: {size / 1e9:.3f} GB — {group["description"]}')
        for name, asset in catalog['generated_assets'].items():
            print(f'{name}: rebuild locally — {asset["recipe"]}')
        return
    if not args.groups:
        parser.error('Select at least one group, or use --list')
    unknown = set(args.groups) - catalog['groups'].keys()
    if unknown:
        parser.error('Unknown group(s): ' + ', '.join(sorted(unknown)))
    workspace = args.workspace.expanduser().resolve()
    config_path = args.config or Path(os.environ.get('BLOB_CONFIG', workspace / '.blob_config.json'))
    config = json.loads(config_path.expanduser().read_text()) if config_path.is_file() else {}
    container = os.environ.get('BLOB_SAS_URL', config.get('sas_url', catalog['container_url']))
    token = os.environ.get('BLOB_SAS_TOKEN', config.get('sas_token', '')).lstrip('?')
    if not args.verify_only and not token:
        parser.error('Provide --config, BLOB_CONFIG, or BLOB_SAS_TOKEN')
    container = container.split('?')[0].rstrip('/')
    selected = {item['path']: item for name in args.groups for item in catalog['groups'][name]['files']}
    for relative, item in selected.items():
        if args.metadata_only and Path(relative).suffix in ('.pt', '.safetensors'):
            continue
        target = (workspace / relative).resolve()
        if not target.is_relative_to(workspace):
            raise RuntimeError('Catalog path escapes the selected workspace')
        if target.exists():
            if target.stat().st_size != item['bytes'] or sha256(target) != item['sha256']:
                raise RuntimeError(f'Existing file differs; preserved without changes: {relative}')
            print(f'Verified existing: {relative}', flush=True)
            continue
        if args.verify_only:
            raise RuntimeError(f'Missing: {relative}')
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + '.partial')
        url = container + '/' + quote(item['blob'], safe='/') + '?' + token
        for attempt in range(4):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == item['bytes']:
                break
            if offset > item['bytes']:
                raise RuntimeError(f'Oversized partial file: {relative}')
            try:
                with urlopen(Request(url, headers={'Range': f'bytes={offset}-'} if offset else {}), timeout=90) as response:
                    if response.status != (206 if offset else 200):
                        raise RuntimeError('Server did not honor the requested range')
                    with partial.open('ab' if offset else 'wb') as stream:
                        for block in iter(lambda: response.read(8 * 1024 * 1024), b''):
                            stream.write(block)
                if partial.stat().st_size != item['bytes']:
                    raise RuntimeError('Incomplete transfer')
                break
            except (HTTPError, URLError, OSError, RuntimeError) as exc:
                if attempt == 3:
                    detail = f'HTTP {exc.code}' if isinstance(exc, HTTPError) else type(exc).__name__
                    raise RuntimeError(f'Download failed: {relative} ({detail})') from None
                time.sleep(attempt + 1)
        if sha256(partial) != item['sha256']:
            raise RuntimeError(f'SHA256 mismatch; partial file retained: {relative}')
        partial.replace(target)
        print(f'Downloaded and verified: {relative}', flush=True)


if __name__ == '__main__':
    main()
