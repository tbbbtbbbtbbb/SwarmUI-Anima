"""Optional command-line entry point; normal installs run setup automatically."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).parent / 'ExtraNodes/SwarmAnimaNodes/install.py'), run_name='__main__')
