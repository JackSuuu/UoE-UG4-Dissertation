"""Strict pretrained DINOv2 encoding of policy RGB images; no random fallback."""
import hashlib
from pathlib import Path
import sys

import numpy as np


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class FrozenVisualEncoder:
    feature_dim = 384
    transform = 'policy_rgb_pil_bicubic_224_imagenet_l2_v1'

    def __init__(self, checkpoint, source, device='cpu', expected_sha=None):
        import torch
        self.checkpoint, self.source = Path(checkpoint).resolve(), Path(source).resolve()
        self.device = device
        digest = file_sha(self.checkpoint)
        if expected_sha is not None and digest != expected_sha:
            raise ValueError('Visual checkpoint hash mismatch')
        sys.path.insert(0, str(self.source))
        from dinov2.hub.backbones import dinov2_vits14
        self.model = dinov2_vits14(pretrained=False)
        self.model.load_state_dict(torch.load(self.checkpoint, map_location='cpu', weights_only=True), strict=True)
        self.model.eval().requires_grad_(False).to(device)
        code_hash = hashlib.sha256()
        for path in sorted((self.source/'dinov2').rglob('*.py')):
            code_hash.update(str(path.relative_to(self.source)).encode())
            code_hash.update(path.read_bytes())
        self.provenance = dict(name='dinov2_vits14', checkpoint=str(self.checkpoint),
            sha256=digest, source=str(self.source), source_sha256=code_hash.hexdigest(),
            transform=self.transform, feature_dim=self.feature_dim, pretrained=True)

    def encode_batch(self, images):
        import torch
        from PIL import Image
        arrays = []
        for image in images:
            image = np.asarray(image)
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError('Expected uint8 policy RGB image, not raw flipped simulator pixels')
            arrays.append(np.asarray(Image.fromarray(image).resize((224, 224), Image.Resampling.BICUBIC)).copy())
        if not arrays:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        x = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float().to(self.device) / 255.
        mean = x.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
        std = x.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
        with torch.inference_mode():
            features = self.model((x-mean)/std)
            features = torch.nn.functional.normalize(features.float(), dim=-1)
        return features.cpu().numpy()

    def encode(self, image):
        return self.encode_batch([image])[0]
