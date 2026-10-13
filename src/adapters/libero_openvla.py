"""Full task-finetuned OpenVLA, native H1 actions, official LIBERO preprocessing."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from adapters import tf5_compat


def policy_image(raw):
    import tensorflow as tf
    # Official libero_utils: rotate, JPEG round trip, antialiased Lanczos resize.
    image = np.ascontiguousarray(raw[::-1, ::-1])
    image = tf.io.decode_image(tf.image.encode_jpeg(image), expand_animations=False, dtype=tf.uint8)
    image = tf.image.resize(image, (224, 224), method='lanczos3', antialias=True)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    image = tf.image.convert_image_dtype(image, tf.float32)
    side = tf.sqrt(tf.constant(0.9, tf.float32))
    lo = (1 - side) / 2
    boxes = tf.reshape(tf.stack([lo, lo, lo + side, lo + side]), (1, 4))
    image = tf.image.crop_and_resize(image[None], boxes, [0], (224, 224))[0]
    image = tf.image.convert_image_dtype(tf.clip_by_value(image, 0, 1), tf.uint8, saturate=True)
    return Image.fromarray(image.numpy()).convert('RGB')


class LiberoOpenVLA:
    def __init__(self, snapshot, device='cuda:0'):
        from safetensors.torch import load_file
        from transformers import AutoConfig
        self.device = device
        snapshot = str(Path(snapshot).resolve())
        tf5_compat.setup_hf_offline()
        cfg = AutoConfig.from_pretrained(snapshot, trust_remote_code=True)
        # Finetuned repo references the base repo's remote code; weights/config
        # remain task-specific. Resolve the already-cached base code explicitly.
        code_snapshot = tf5_compat.snapshot_dir()
        self.model, self.build = tf5_compat.build_architecture(cfg, code_snapshot)
        weights = {}
        for shard in sorted(Path(snapshot).glob('*.safetensors')):
            weights.update(load_file(str(shard), device='cpu'))
        self.model.load_state_dict(weights, strict=True, assign=True)
        del weights
        self.model = self.model.to(device=device, dtype=torch.bfloat16).eval()
        self.model.requires_grad_(False)
        self.processor = tf5_compat.load_processor(snapshot)
        stats = json.loads((Path(snapshot) / 'dataset_statistics.json').read_text())
        self.key = 'libero_10' if 'libero_10' in stats else 'libero_10_no_noops'
        self.stats = stats[self.key]['action']
        assert len(self.stats['q01']) == 7
        self.build.update(unnorm_key=self.key, snapshot=snapshot, code_snapshot=code_snapshot,
                          decode='uncached_full_multimodal_greedy_tf5', H=1,
                          official_tf4401_decode_parity_verified=False)

    @torch.inference_mode()
    def act(self, raw_image, language):
        image = policy_image(raw_image)
        prompt = f'In: What action should the robot take to {language.lower()}?\nOut:'
        inputs = self.processor(prompt, image).to(self.device)
        # The existing reference appends prompt-end and builds its matching mask.
        tokens = tf5_compat.greedy_decode(self.model, inputs['input_ids'],
                                          inputs['pixel_values'].to(torch.bfloat16))
        bins = tf5_compat.tokens_to_bins(tokens.cpu().numpy(), self.model.vocab_size)
        native = tf5_compat.bins_to_action(bins, self.stats['q01'], self.stats['q99'],
                                          self.stats.get('mask'))
        action = native.copy()
        action[-1] = -np.sign(2 * action[-1] - 1)  # official normalize, binarize, invert
        assert action.shape == (7,) and np.isfinite(action).all()
        return action, dict(tokens=tokens.tolist(), native_action=native.tolist()), image
