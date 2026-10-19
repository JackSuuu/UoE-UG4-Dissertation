"""Separate official TF4.40.1 frozen actor worker, sequential batch-one candidates."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch
import transformers
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def main(snapshot, code_snapshot):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '2'
    assert os.environ['MUJOCO_EGL_DEVICE_ID'] == '2'
    assert transformers.__version__ == '4.40.1'
    torch.manual_seed(0)
    torch.set_num_threads(4)
    cfg_cls = get_class_from_dynamic_module('configuration_prismatic.OpenVLAConfig', code_snapshot)
    cls = get_class_from_dynamic_module('modeling_prismatic.OpenVLAForActionPrediction', code_snapshot)
    proc_cls = get_class_from_dynamic_module('processing_prismatic.PrismaticProcessor', code_snapshot)
    model = cls.from_pretrained(snapshot, config=cfg_cls.from_pretrained(snapshot), torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation='sdpa').to('cuda:0').eval()
    model.requires_grad_(False)
    processor = proc_cls.from_pretrained(snapshot, trust_remote_code=True)
    model.norm_stats = json.loads((Path(snapshot)/'dataset_statistics.json').read_text())
    key = 'libero_10' if 'libero_10' in model.norm_stats else 'libero_10_no_noops'
    original_generate, generated = model.generate, []
    def capture(*args, **kwargs):
        result = original_generate(*args, **kwargs)
        generated[:] = result[0, -7:].tolist()
        return result
    model.generate = capture
    print('RESULT '+json.dumps(dict(ready=True, snapshot=snapshot, code_snapshot=code_snapshot,
        unnorm_key=key, transformers=transformers.__version__, torch=torch.__version__, batch=1,
        frozen=True, H=1, official_predict_action=True)), flush=True)
    with torch.inference_mode():
        for line in sys.stdin:
            req = json.loads(line)
            torch.manual_seed(req['seed'])
            image = Image.open(io.BytesIO(base64.b64decode(req['png']))).convert('RGB')
            inputs = processor(f"In: What action should the robot take to {req['language'].lower()}?\nOut:", image).to('cuda:0', dtype=torch.bfloat16)
            sampled = req['source'] == 'sample'
            kwargs = dict(do_sample=True, temperature=.7, top_p=.95) if sampled else dict(do_sample=False)
            start = time.monotonic()
            native = model.predict_action(**inputs, unnorm_key=key, **kwargs)
            torch.cuda.synchronize()
            action = native.copy()
            action[-1] = -np.sign(2*action[-1]-1)
            print('RESULT '+json.dumps(dict(action=action.tolist(), native_action=native.tolist(),
                tokens=generated.copy(), model_inference_s=time.monotonic()-start,
                source=req['source'], seed=req['seed'], sampling=kwargs)), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
