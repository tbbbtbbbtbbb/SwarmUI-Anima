"""Download only the pinned models used by this extension."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def sha256(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-dir', type=Path, required=True, help='Your ComfyUI models directory')
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name('models.json').read_text(encoding='utf-8'))
    for model in manifest:
        path = args.models_dir / model['path']
        if path.exists() and path.stat().st_size == model['size'] and sha256(path) == model['sha256']:
            print('Verified:', model['path'])
            continue
        if path.exists():
            raise RuntimeError(f'{path} already exists but differs from the supported model. Move it aside before installing.')
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + '.download')
        url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{model['source']}"
        print('Downloading:', model['path'], flush=True)
        with urllib.request.urlopen(url, timeout=180) as response, temp.open('wb') as file:
            while chunk := response.read(1024 * 1024):
                file.write(chunk)
        if temp.stat().st_size != model['size'] or sha256(temp) != model['sha256']:
            raise RuntimeError(f'Incomplete or corrupt download: {temp}. Run the installer again.')
        temp.replace(path)
        print('Verified:', model['path'])


if __name__ == '__main__':
    main()
