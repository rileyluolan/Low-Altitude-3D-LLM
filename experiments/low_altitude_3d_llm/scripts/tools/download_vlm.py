#!/usr/bin/env python3
"""Download the selected RefDrone VLM without exposing Blob credentials in logs."""
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(os.environ.get('REFDRONE_RAW_DIR', ROOT / 'models/qwen3vl_stage2_raw')).expanduser().resolve()
PREFIX = 'output/liyan/lnj/images/0729/qwen3vl_stage2/'


def main():
    config = {}
    config_path = Path(os.environ.get('BLOB_CONFIG', str(Path.home() / '.blob_config.json'))).expanduser()
    if config_path.is_file():
        config = json.loads(config_path.read_text())
    sas_url = os.environ.get('BLOB_SAS_URL', config.get('sas_url', ''))
    base = sas_url.split('?')[0].rstrip('/')
    token = os.environ.get('BLOB_SAS_TOKEN', config.get('sas_token', sas_url.partition('?')[2])).lstrip('?')
    if not base.startswith('https://') or not token:
        raise RuntimeError('Set BLOB_SAS_URL/BLOB_SAS_TOKEN or BLOB_CONFIG with Blob container access')
    inventory = json.loads((ROOT / 'assets/refdrone_vlm.lock.json').read_text())
    assert inventory['source_blob_prefix'] == PREFIX
    entries = inventory['files']
    DEST.mkdir(parents=True, exist_ok=True)
    records = []
    for item in entries:
        name = item['name']
        if '/' in name or name.startswith('.'):
            raise ValueError('Unexpected candidate filename')
        target = DEST / name
        url = base + '/' + quote(PREFIX + name, safe='/') + '?' + token
        for attempt in range(5):
            try:
                with urlopen(Request(url, method='HEAD'), timeout=60) as response:
                    etag = response.headers['ETag']
                    remote_md5 = response.headers.get('Content-MD5')
                    assert int(response.headers['Content-Length']) == item['bytes']
                break
            except Exception as exc:
                if attempt == 4:
                    raise RuntimeError(f'Cannot read metadata for {name}: {type(exc).__name__}') from None
                time.sleep(2)
        if not target.exists():
            partial = target.with_name(name + '.partial')
            tagfile = partial.with_name(partial.name + '.etag')
            if partial.exists() and (not tagfile.exists() or tagfile.read_text() != etag):
                raise RuntimeError('Partial download does not match the current Blob ETag')
            tagfile.write_text(etag)
            for attempt in range(5):
                offset = partial.stat().st_size if partial.exists() else 0
                if offset == item['bytes']:
                    break
                headers = {'If-Match': etag}
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                try:
                    with urlopen(Request(url, headers=headers), timeout=90) as response:
                        if response.status != (206 if offset else 200):
                            raise ValueError('Unexpected range response')
                        with partial.open('ab' if offset else 'wb') as stream:
                            printed = offset
                            start = time.monotonic()
                            while chunk := response.read(8 * 1024 * 1024):
                                stream.write(chunk)
                                offset += len(chunk)
                                if offset - printed >= 128 * 1024 * 1024:
                                    print(f'{name}: {offset/1e9:.2f}/{item["bytes"]/1e9:.2f} GB', flush=True)
                                    printed = offset
                            stream.flush()
                            os.fsync(stream.fileno())
                    if offset != item['bytes']:
                        raise ValueError('Incomplete response')
                    break
                except Exception as exc:
                    if attempt == 4:
                        raise RuntimeError(f'Download failed for {name}: {type(exc).__name__}') from None
                    print(f'Retrying {name} after {type(exc).__name__}', flush=True)
                    time.sleep(2)
            if partial.stat().st_size != item['bytes']:
                raise RuntimeError(f'Wrong download length: {name}')
            partial.replace(target)
            tagfile.unlink()
        if target.stat().st_size != item['bytes']:
            raise RuntimeError(f'Existing file has wrong length: {name}')
        sha = hashlib.sha256()
        md5 = hashlib.md5()
        with target.open('rb') as stream:
            for chunk in iter(lambda: stream.read(32 * 1024 * 1024), b''):
                sha.update(chunk)
                md5.update(chunk)
        if remote_md5 and base64.b64encode(md5.digest()).decode() != remote_md5:
            raise RuntimeError(f'Blob MD5 mismatch: {name}')
        if sha.hexdigest() != item['sha256']:
            raise RuntimeError(f'Pinned SHA256 mismatch: {name}')
        records.append({'name': name, 'bytes': item['bytes'], 'sha256': sha.hexdigest(),
                        'blob_etag': etag, 'blob_md5_verified': bool(remote_md5)})
        print(f'Verified {name}: {sha.hexdigest()}', flush=True)
    result = {'source_blob_prefix': PREFIX, 'completed_utc': datetime.now(timezone.utc).isoformat(),
              'files': records, 'status': 'complete'}
    (DEST / 'DOWNLOAD_MANIFEST.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
