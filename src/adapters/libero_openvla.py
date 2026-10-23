"""Full task-finetuned OpenVLA, native H1 actions, official LIBERO preprocessing."""
import json
from pathlib import Path

import numpy as np
import torch

from . import tf5_compat
from .libero_image import policy_image


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
