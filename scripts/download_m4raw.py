"""Fetch exactly the frozen real-MRI selection; validate every source SHA256."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import requests


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch_one(root, manifest, item):
    root = Path(root).resolve()
    path = (root / item['relative_path']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Source path escapes data directory')
    if path.exists() and path.stat().st_size == int(item['bytes']) and sha256(path) == item['sha256']:
        return 'cached'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.h5.part')
    url = f"https://huggingface.co/datasets/{manifest['dataset']}/resolve/{manifest['revision']}/{item['relative_path']}"
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with temporary.open('wb') as stream:
                    for block in response.iter_content(1024 * 1024):
                        stream.write(block)
            if temporary.stat().st_size != int(item['bytes']) or sha256(temporary) != item['sha256']:
                raise ValueError('Source file failed size/SHA256 validation')
            temporary.replace(path)
            return 'downloaded'
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='research/real-acquisition/manifest.json')
    parser.add_argument('--output', default='tmp/m4raw/data')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error('Use 1-4 download workers')
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    print(f"Frozen selection: {len(manifest['files'])} files, {manifest['total_bytes']} bytes", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(lambda item: fetch_one(args.output, manifest, item), manifest['files']), 1):
            print(f"{index}/{len(manifest['files'])} {result}", flush=True)
    print('All selected source files verified', flush=True)


if __name__ == '__main__':
    main()
