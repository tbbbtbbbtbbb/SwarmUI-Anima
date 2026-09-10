"""Inspect or clear this extension's derived-artifact cache while ComfyUI is stopped."""
import argparse
from pathlib import Path

from filelock import FileLock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--user-dir', type=Path, required=True, help='Your ComfyUI user directory')
    parser.add_argument('--clear', action='store_true')
    args = parser.parse_args()
    root = args.user_dir.resolve() / 'default/swarm_anima'
    cache = root / 'cache'
    if not cache.is_dir():
        print('Cache is empty.')
        return
    with FileLock(cache / '.lock'):
        files = list(cache.glob('*.safetensors'))
        print(f'{len(files)} entries, {sum(p.stat().st_size for p in files) / 1024**2:.1f} MiB')
        if args.clear:
            for path in files:
                path.unlink()
            print('Cache cleared.')
        config = root / 'cache.json'
        if config.exists():
            print(f'Limits: {config}')
            print(config.read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
