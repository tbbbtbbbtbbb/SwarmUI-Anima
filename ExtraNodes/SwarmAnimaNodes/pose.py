from pathlib import Path

import cv2
import numpy as np
import torch
from rtmlib import Wholebody

import folder_paths
from .cache import image_key
from .reference import cache


LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16),
         (5, 6), (11, 12), (5, 11), (6, 12), (0, 5), (0, 6)]
COLORS = [(255, 0, 0), (255, 128, 0), (255, 255, 0), (128, 255, 0), (0, 255, 0),
          (0, 255, 128), (0, 255, 255), (0, 128, 255), (0, 0, 255), (128, 0, 255),
          (255, 0, 255), (255, 0, 128), (180, 180, 180), (220, 220, 220)]


def render_pose(points, scores, width, height):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for (a, b), color in zip(LIMBS, COLORS):
        if scores[a] >= 0.3 and scores[b] >= 0.3:
            cv2.line(canvas, tuple(points[a].astype(int)), tuple(points[b].astype(int)), color, 2)
    for i, (point, score) in enumerate(zip(points, scores)):
        if score >= 0.3:
            radius = 3 if i < 17 else 1 if i < 23 else 2
            color = (0, 255, 255) if i >= 91 else (255, 255, 255)
            cv2.circle(canvas, tuple(point.astype(int)), radius, color, -1)
    return canvas


class SwarmAnimaPose:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',), 'width': ('INT', {'default': 1024, 'min': 256, 'max': 4096, 'step': 16}),
                             'height': ('INT', {'default': 1024, 'min': 256, 'max': 4096, 'step': 16})}}

    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'detect'
    CATEGORY = 'SwarmUI/Anima'

    def detect(self, image, width, height):
        directory = Path(folder_paths.models_dir) / 'anima/detector'
        files = [directory / 'yolox.onnx', directory / 'dwpose.onnx']
        key = image_key(image[:1].cpu(), f'dwpose-wholebody133-r0-v1-{width}x{height}', files)

        def compute():
            estimator = Wholebody(det=str(files[0]), det_input_size=(640, 640), pose=str(files[1]),
                                 pose_input_size=(192, 256), to_openpose=False, backend='onnxruntime', device='cpu')
            pixels = (image[0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            pixels = cv2.resize(pixels[:, :, :3], (width, height))
            points, scores = estimator(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
            if points is None or len(points) == 0 or np.count_nonzero(scores[0, :17] >= 0.3) < 3:
                raise ValueError('No person detected in Pose Image. Choose an image with a clearly visible body.')
            skeleton = render_pose(points[0], scores[0], width, height)
            return {'skeleton': torch.from_numpy(skeleton), 'points': torch.from_numpy(points[0]), 'scores': torch.from_numpy(scores[0])}

        data = cache().get('pose', key, compute)
        return (data['skeleton'].float()[None] / 255,)
