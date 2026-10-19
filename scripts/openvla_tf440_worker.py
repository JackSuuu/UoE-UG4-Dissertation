"""Pinned TF4 inference subprocess: PNG+language in, native action+tokens out.

This process receives no simulator state or repair trajectory. Model code is loaded
from the pinned local base snapshot and weights from the shared LIBERO10 checkpoint.
"""
import argparse
import base64
import io
import json
import sys
import time

import numpy as np
from PIL import Image
import torch
import transformers
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def main(snapshot, code_snapshot):
    assert transformers.__version__ == '4.40.1'
    torch.manual_seed(0)
    torch.set_num_threads(4)
    cfg_cls = get_class_from_dynamic_module('configuration_prismatic.OpenVLAConfig', code_snapshot)
    cls = get_class_from_dynamic_module('modeling_prismatic.OpenVLAForActionPrediction', code_snapshot)
    proc_cls = get_class_from_dynamic_module('processing_prismatic.PrismaticProcessor', code_snapshot)
    cfg = cfg_cls.from_pretrained(snapshot)
    model = cls.from_pretrained(snapshot, config=cfg, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation='sdpa').to('cuda:0').eval()
    model.requires_grad_(False)
    processor = proc_cls.from_pretrained(snapshot, trust_remote_code=True)
    from pathlib import Path
    model.norm_stats = json.loads((Path(snapshot)/'dataset_statistics.json').read_text())
    key = 'libero_10' if 'libero_10' in model.norm_stats else 'libero_10_no_noops'
    original_generate = model.generate
    generated = []

    def capture(*args, **kwargs):
        result = original_generate(*args, **kwargs)
        generated[:] = result[0, -7:].tolist()
        return result

    model.generate = capture
    print('RESULT '+json.dumps(dict(ready=True, snapshot=snapshot, code_snapshot=code_snapshot,
        unnorm_key=key, transformers=transformers.__version__, torch=torch.__version__,
        decode='official_predict_action_cached_greedy', H=1)), flush=True)
    with torch.inference_mode():
        for line in sys.stdin:
            request = json.loads(line)
            image = Image.open(io.BytesIO(base64.b64decode(request['png']))).convert('RGB')
            prompt = f"In: What action should the robot take to {request['language'].lower()}?\nOut:"
            inputs = processor(prompt, image).to('cuda:0', dtype=torch.bfloat16)
            start = time.monotonic()
            native = model.predict_action(**inputs, unnorm_key=key, do_sample=False)
            action = native.copy()
            action[-1] = -np.sign(2*action[-1]-1)
            print('RESULT '+json.dumps(dict(action=action.tolist(), native_action=native.tolist(),
                tokens=generated.copy(), model_inference_s=time.monotonic()-start)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('snapshot')
    p.add_argument('code_snapshot')
    a = p.parse_args()
    main(a.snapshot, a.code_snapshot)
