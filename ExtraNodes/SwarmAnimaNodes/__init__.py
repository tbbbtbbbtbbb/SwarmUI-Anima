from pathlib import Path

import folder_paths

folder_paths.folder_names_and_paths.setdefault('ipadapter', ([str(Path(folder_paths.models_dir) / 'ipadapter')], {'.safetensors'}))

from .adapters import SwarmAnimaPoseApply, SwarmAnimaReference
from .pose import SwarmAnimaPose
from .reference import SwarmAnimaEncode

NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (SwarmAnimaEncode, SwarmAnimaReference, SwarmAnimaPose, SwarmAnimaPoseApply)}
NODE_DISPLAY_NAME_MAPPINGS = {
    'SwarmAnimaEncode': 'Anima Reference Encode',
    'SwarmAnimaReference': 'Anima Reference Apply',
    'SwarmAnimaPose': 'Anima Pose Detect',
    'SwarmAnimaPoseApply': 'Anima Pose Apply',
}
