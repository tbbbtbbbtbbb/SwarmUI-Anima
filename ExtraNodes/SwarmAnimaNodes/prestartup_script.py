"""ComfyUI runs this before importing custom nodes, including Swarm's ExtraNodes."""
from pathlib import Path
import runpy

import folder_paths

try:
    runpy.run_path(str(Path(__file__).with_name('install.py')))['setup'](folder_paths.models_dir)
except Exception as error:
    raise RuntimeError(f'SwarmAnima automatic setup failed: {error}. Restart the backend to retry.') from error
