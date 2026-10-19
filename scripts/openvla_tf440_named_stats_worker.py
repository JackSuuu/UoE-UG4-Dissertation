"""Official TF4 action inference with an explicit, unmodified dataset stats key."""
import argparse
import base64
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import transformers
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def main(snapshot, code_snapshot, key):
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
    model.norm_stats = json.loads((Path(snapshot)/'dataset_statistics.json').read_text())
    assert key in model.norm_stats and len(model.norm_stats[key]['action']['q01']) == 7
    original = model.generate
    tokens = []
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        tokens[:] = result[0,-7:].tolist()
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
            inputs = processor(prompt,image).to('cuda:0',dtype=torch.bfloat16)
            start = time.monotonic()
            native = model.predict_action(**inputs,unnorm_key=key,do_sample=False)
            action = native.copy()
            action[-1] = -np.sign(2*action[-1]-1)
            print('RESULT '+json.dumps(dict(action=action.tolist(),native_action=native.tolist(),
                tokens=tokens.copy(),model_inference_s=time.monotonic()-start)),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('snapshot')
    p.add_argument('code_snapshot')
    p.add_argument('key')
    a=p.parse_args()
    main(a.snapshot,a.code_snapshot,a.key)
