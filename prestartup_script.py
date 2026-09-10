"""Startup entry point when installed directly in ComfyUI/custom_nodes."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).parent / 'ExtraNodes/SwarmAnimaNodes/prestartup_script.py'))
