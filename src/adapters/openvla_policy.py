"""Frozen official OpenVLA H1 interface for the task-agnostic runtime.

Uses the checkpoint's predict_action implementation with transformers 4.40.1.
Policy image preprocessing is owned by the LIBERO bridge and applied once.
"""
from io import BytesIO

from ..interfaces import Proposal


class OpenVLAPolicy:
    def __init__(self, checkpoint, *, unnorm_key, device='cuda:0', revision=None):
        import torch
        import transformers
        from transformers import AutoModelForVision2Seq, AutoProcessor
        if transformers.__version__ != '4.40.1':
            raise RuntimeError('This official OpenVLA backend requires transformers==4.40.1')
        self.device, self.key = device, unnorm_key
        self.processor = AutoProcessor.from_pretrained(checkpoint, revision=revision, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(checkpoint, revision=revision,
            trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            attn_implementation='eager').to(device).eval()
        self.model.requires_grad_(False)
        if unnorm_key not in self.model.norm_stats:
            raise ValueError(f'Unknown action normalization key: {unnorm_key}')
        self.provenance = dict(checkpoint=str(checkpoint), revision=revision,
                               resolved_revision=getattr(self.model.config, '_commit_hash', None),
                               unnorm_key=unnorm_key, frozen=True, horizon=1,
                               transformers=transformers.__version__)

    def propose(self, image_png, instruction, *, sample=False, seed=0):
        import numpy as np
        import torch
        from PIL import Image
        image = Image.open(BytesIO(image_png)).convert('RGB')
        prompt = f'In: What action should the robot take to {instruction.lower()}?\nOut:'
        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        options = dict(do_sample=sample)
        if sample:
            options.update(temperature=.8, top_p=.95)
        devices = [torch.device(self.device).index or 0] if str(self.device).startswith('cuda') else []
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(seed)
            native = self.model.predict_action(**inputs, unnorm_key=self.key, **options)
        action = np.asarray(native, dtype=float).copy()
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError('OpenVLA must return a finite H1 action with shape (7,)')
        # Official LIBERO action-space mapping, not a repair edit.
        action[-1] = -np.sign(2*action[-1]-1)
        return Proposal(tuple(action.tolist()), dict(self.provenance, native_action=np.asarray(native).tolist(),
                                                      sample=sample, seed=seed))
