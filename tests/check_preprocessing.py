"""Check real preprocessing cache hits without loading the diffusion model or using a GPU."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfyui', type=Path, required=True)
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--image', type=Path, default=Path(__file__).parent / 'results/reference-and-pose.png')
    args = parser.parse_args()
    sys.path[:0] = [str(args.comfyui.resolve()), str(Path(__file__).parents[1] / 'ExtraNodes')]
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import folder_paths
    folder_paths.models_dir = str((args.models_dir or args.comfyui / 'models').resolve())
    import numpy as np
    from PIL import Image
    import torch
    from SwarmAnimaNodes import reference, pose
    from SwarmAnimaNodes.cache import ArtifactCache

    image = torch.from_numpy(np.asarray(Image.open(args.image).convert('RGB')).copy()).float()[None] / 255
    checks = {}
    with tempfile.TemporaryDirectory() as temp, torch.inference_mode():
        cache = ArtifactCache(Path(temp) / 'cache')
        with patch.object(reference, 'cache', return_value=cache):
            encoder = reference.SwarmAnimaEncode()
            cold = encoder.encode(image)[0]
            with patch.object(reference.SiglipVisionModel, 'from_pretrained', side_effect=AssertionError('Cache hit loaded encoder')):
                warm = encoder.encode(image)[0]
            assert torch.equal(cold, warm)
            checks['reference'] = dict(bit_identical=True, shape=list(cold.shape), skips_model_load=True)
        with patch.object(pose, 'cache', return_value=cache):
            detector = pose.SwarmAnimaPose()
            cold = detector.detect(image, 1024, 1024)[0]
            with patch.object(pose, 'Wholebody', side_effect=AssertionError('Cache hit loaded detector')):
                warm = detector.detect(image, 1024, 1024)[0]
            assert torch.equal(cold, warm)
            checks['pose'] = dict(bit_identical=True, shape=list(cold.shape), skips_model_load=True)
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
