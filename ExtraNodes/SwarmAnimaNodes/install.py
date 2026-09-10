"""First-start setup for the bundled Anima nodes."""
import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, distributions, version
from importlib.util import find_spec
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request

from filelock import FileLock
from packaging.version import Version


def ensure_dependencies():
    # Serialize backends sharing an interpreter. Never replace Torch or its dependencies.
    key = hashlib.sha256(sys.executable.encode()).hexdigest()[:16]
    with FileLock(str(Path(tempfile.gettempdir()) / f'swarm-anima-python-{key}.lock')):
        # Swarm supplies these; plain external ComfyUI installs may not have them.
        missing = [package for module, package in [('cv2', 'opencv-python-headless'), ('onnxruntime', 'onnxruntime')]
                   if find_spec(module) is None]
        if missing:
            print(f'[SwarmAnima] Installing image dependencies: {", ".join(missing)}...', flush=True)
            with tempfile.TemporaryDirectory(prefix='swarm-anima-deps-') as temp:
                constraints = Path(temp) / 'constraints.txt'
                constraints.write_text('\n'.join(f'{p.metadata["Name"]}=={p.version}' for p in distributions()), encoding='utf-8')
                subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-input',
                                '--only-binary=:all:', '--constraint', str(constraints), *missing], check=True, timeout=600)
        try:
            if Version(version('rtmlib')) >= Version('0.0.16'):
                return
        except PackageNotFoundError:
            pass
        print('[SwarmAnima] Installing pose dependency...', flush=True)
        # rtmlib lists multiple OpenCV distributions; reuse the cv2 provider above.
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                        '--no-input', '--no-deps', 'rtmlib==0.0.16'], check=True, timeout=600)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_models(models_dir):
    models_dir = Path(models_dir)
    manifest = json.loads(Path(__file__).with_name('models.json').read_text(encoding='utf-8'))
    lock = models_dir / 'anima/setup.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock)):
        for model in manifest:
            path = models_dir / model['path']
            if path.exists():
                if path.stat().st_size == model['size'] and sha256(path) == model['sha256']:
                    continue
                raise RuntimeError(f'{path} differs from the supported model; it was not overwritten.')
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(path.suffix + '.download')
            url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{model['source']}"
            print(f"[SwarmAnima] Downloading {model['path']} ({model['size'] / 1024**2:.0f} MB)...", flush=True)
            try:
                with urllib.request.urlopen(url, timeout=180) as response, temp.open('wb') as file:
                    downloaded = 0
                    while chunk := response.read(1024 * 1024):
                        file.write(chunk)
                        downloaded += len(chunk)
                        if downloaded // (64 * 1024**2) != (downloaded - len(chunk)) // (64 * 1024**2):
                            print(f"[SwarmAnima] {model['path']}: {downloaded / model['size']:.0%}", flush=True)
                if temp.stat().st_size != model['size'] or sha256(temp) != model['sha256']:
                    raise RuntimeError(f"Incomplete or corrupt download: {model['path']}")
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)
            print(f"[SwarmAnima] Verified {model['path']}", flush=True)
    print('[SwarmAnima] Model assets ready.', flush=True)


def setup(models_dir):
    ensure_dependencies()
    ensure_models(models_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-dir', type=Path, required=True)
    setup(parser.parse_args().models_dir)
