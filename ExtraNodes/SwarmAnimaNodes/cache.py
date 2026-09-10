import hashlib
import json
import logging
import os
from pathlib import Path
import time
import uuid

from filelock import FileLock
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file


class ArtifactCache:
    """One bounded disk cache, shared by the pose and reference nodes."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(self.directory / '.lock')
        config = self.directory.parent / 'cache.json'
        defaults = dict(max_mb=1024, max_entries=2000, max_age_days=0)
        if not config.exists():
            with self.lock:
                if not config.exists():
                    config.write_text(json.dumps(defaults, indent=2), encoding='utf-8')
        self.policy = defaults | json.loads(config.read_text(encoding='utf-8'))
        if any(not isinstance(v, (int, float)) or v < 0 for v in self.policy.values()):
            raise ValueError(f'Invalid Anima cache limits in {config}')

    def prune(self):
        files = sorted(self.directory.glob('*.safetensors'), key=lambda p: p.stat().st_mtime)
        sizes = {p: p.stat().st_size for p in files}
        total, count = sum(sizes.values()), len(files)
        age = self.policy['max_age_days'] * 86400
        for path in files:
            expired = age > 0 and time.time() - path.stat().st_mtime > age
            if expired or total > self.policy['max_mb'] * 1024**2 or (self.policy['max_entries'] > 0 and count > self.policy['max_entries']):
                path.unlink()
                total -= sizes[path]
                count -= 1

    def get(self, kind, key, compute):
        path = self.directory / f'{kind}-{key}.safetensors'
        # Serializes misses too, preventing duplicate encoding across worker processes.
        with self.lock:
            self.prune()
            if path.exists():
                try:
                    data = load_file(path)
                except SafetensorError:
                    path.unlink()
                else:
                    os.utime(path, None)
                    logging.info('SwarmAnima %s cache hit: %s', kind, key[:12])
                    return data
            data = {k: v.detach().cpu().contiguous() for k, v in compute().items()}
            if self.policy['max_mb'] > 0:
                temp = self.directory / f'{uuid.uuid4().hex}.tmp'
                try:
                    save_file(data, temp)
                    temp.replace(path)
                finally:
                    temp.unlink(missing_ok=True)
                self.prune()
            logging.info('SwarmAnima %s cache miss: %s', kind, key[:12])
            return data


def image_key(image, version, files=()):
    digest = hashlib.sha256()
    digest.update(version.encode())
    digest.update(str(image.shape).encode())
    digest.update(image.contiguous().numpy().tobytes())
    for path in files:
        path = Path(path).resolve()
        stat = path.stat()
        digest.update(f'{path}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
    return digest.hexdigest()
