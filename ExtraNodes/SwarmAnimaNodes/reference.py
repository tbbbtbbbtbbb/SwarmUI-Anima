from pathlib import Path
from importlib.metadata import version

import numpy as np
from PIL import Image
import torch
from transformers import SiglipVisionModel

import folder_paths
from .cache import ArtifactCache, image_key


def cache():
    return ArtifactCache(Path(folder_paths.get_user_directory()) / 'default/swarm_anima/cache')


class SwarmAnimaEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',)}}

    RETURN_TYPES = ('ANIMA_FEATURES',)
    FUNCTION = 'encode'
    CATEGORY = 'SwarmUI/Anima'

    def encode(self, image):
        directory = Path(folder_paths.models_dir) / 'siglip2/siglip2-base-patch16-512'
        files = [directory / 'config.json', directory / 'model.safetensors']
        key = image_key(image[:1].cpu(), f'siglip2-last-black-pad-512-bilinear-v1-{version("transformers")}', files)

        def compute():
            # This small encoder runs on CPU, leaving the GPU available to the main model.
            encoder = SiglipVisionModel.from_pretrained(directory, local_files_only=True).eval()
            pixels = (image[0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            original = Image.fromarray(pixels[:, :, :3])
            ratio = 512 / max(original.size)
            size = tuple(max(1, round(d * ratio)) for d in original.size)
            resized = original.resize(size, Image.Resampling.BILINEAR)
            padded = Image.new('RGB', (512, 512))
            padded.paste(resized, ((512 - size[0]) // 2, (512 - size[1]) // 2))
            inputs = torch.from_numpy(np.array(padded)).permute(2, 0, 1).float()[None] / 127.5 - 1
            features = encoder(inputs, interpolate_pos_encoding=True).last_hidden_state
            return {'features': features}

        return (cache().get('reference', key, compute)['features'],)
