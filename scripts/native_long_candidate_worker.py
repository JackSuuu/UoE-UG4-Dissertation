"""Official frozen TF4.40.1 H1 actor; one persistent, sequential batch-one worker."""
import argparse
import base64
import ctypes
import ctypes.util
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True
import numpy as np
from PIL import Image
import torch
import transformers
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def device_identity(physical_gpu, expected_uuid, egl_device):
    """Check the selected physical UUID against both nvidia-smi and CUDA ordinal 0."""
    if physical_gpu not in (2, 3) or egl_device != {2: 1, 3: 0}[physical_gpu]:
        raise ValueError('Expected physical GPU2/EGL1 or GPU3/EGL0')
    raw = subprocess.check_output(['nvidia-smi', '-i', str(physical_gpu),
        '--query-gpu=index,uuid,pci.bus_id', '--format=csv,noheader'], text=True).strip()
    index, actual_uuid, bus = [part.strip() for part in raw.split(',')]
    if int(index) != physical_gpu or actual_uuid != expected_uuid:
        raise ValueError(f'Physical GPU identity mismatch: {raw}')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != expected_uuid:
        raise ValueError('Worker must be masked by the selected physical UUID')
    if os.environ.get('MUJOCO_EGL_DEVICE_ID') != str(egl_device):
        raise ValueError('Worker EGL selection differs from the audited mapping')
    # CUDA driver ordinals respect CUDA_VISIBLE_DEVICES. Do not infer identity
    # from a device name or from the physical numeric index after masking.
    driver = ctypes.CDLL(ctypes.util.find_library('cuda') or 'libcuda.so.1')
    device, count = ctypes.c_int(), ctypes.c_int()
    identifier = (ctypes.c_ubyte * 16)()
    for result in (driver.cuInit(0), driver.cuDeviceGetCount(ctypes.byref(count)),
                   driver.cuDeviceGet(ctypes.byref(device), 0)):
        if result != 0:
            raise RuntimeError(f'CUDA driver identity check failed: {result}')
    get_uuid = getattr(driver, 'cuDeviceGetUuid_v2', driver.cuDeviceGetUuid)
    if get_uuid(ctypes.byref(identifier), device) != 0:
        raise RuntimeError('CUDA UUID lookup failed')
    cuda_uuid = 'GPU-' + str(uuid.UUID(bytes=bytes(identifier)))
    if count.value != 1 or cuda_uuid != expected_uuid or torch.cuda.device_count() != 1:
        raise ValueError(f'CUDA ordinal identity mismatch: {cuda_uuid}, count={count.value}')
    return dict(physical_gpu=physical_gpu, physical_uuid=actual_uuid,
        cuda_ordinal=0, cuda_uuid=cuda_uuid, egl_device=egl_device, pci_bus_id=bus)


def main(args):
    if transformers.__version__ != '4.40.1':
        raise RuntimeError('This worker requires transformers==4.40.1')
    identity = device_identity(args.gpu, args.physical_uuid, args.egl_device)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    cfg_cls = get_class_from_dynamic_module('configuration_prismatic.OpenVLAConfig', args.code_snapshot)
    cls = get_class_from_dynamic_module('modeling_prismatic.OpenVLAForActionPrediction', args.code_snapshot)
    proc_cls = get_class_from_dynamic_module('processing_prismatic.PrismaticProcessor', args.code_snapshot)
    model = cls.from_pretrained(args.snapshot, config=cfg_cls.from_pretrained(args.snapshot),
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation='sdpa').to('cuda:0').eval()
    model.requires_grad_(False)
    processor = proc_cls.from_pretrained(args.snapshot, trust_remote_code=True)
    model.norm_stats = json.loads((Path(args.snapshot)/'dataset_statistics.json').read_text())
    key = 'libero_10'
    if key not in model.norm_stats:
        raise ValueError('Frozen L2 protocol requires libero_10 normalization')
    original_generate, generated = model.generate, []
    resolved = {}
    original_prepare = model._prepare_generation_config

    def capture_config(*a, **kw):
        # Observe TF4's real resolution, including official predict_action's own
        # max_new_tokens=7. Return its original objects without modifying them.
        config, model_kwargs = original_prepare(*a, **kw)
        resolved.clear()
        resolved.update(config.to_dict())
        return config, model_kwargs

    def capture(*a, **kw):
        result = original_generate(*a, **kw)
        generated[:] = result[0, -7:].tolist()
        return result

    model.generate = capture
    model._prepare_generation_config = capture_config
    print('RESULT '+json.dumps(dict(ready=True, snapshot=args.snapshot,
        code_snapshot=args.code_snapshot, unnorm_key=key, transformers=transformers.__version__,
        torch=torch.__version__, batch=1, frozen=True, H=1, official_predict_action=True,
        device_identity=identity, worker_pid=os.getpid(),
        generation_kwargs={source: generation_kwargs(source) for source in ('greedy', 'sample')},
        greedy='official_predict_action_explicit_single_beam_single_return',
        generation_config=model.generation_config.to_dict())), flush=True)
    with torch.inference_mode():
        for query_index, line in enumerate(sys.stdin):
            req = json.loads(line)
            if set(req) != {'png', 'language', 'source', 'seed', 'query_index'}:
                raise ValueError('Only RGB, original language, and decode metadata are accepted')
            if req['source'] not in ('greedy', 'sample') or req['query_index'] != query_index:
                raise ValueError('Invalid policy or nonsequential query')
            if not isinstance(req['seed'], int) or not 0 <= req['seed'] < 2**32:
                raise ValueError('Invalid per-step seed')
            if req['source'] == 'greedy' and req['seed'] != 0:
                raise ValueError('Nominal greedy request seed must remain zero')
            torch.manual_seed(req['seed'])
            png = base64.b64decode(req['png'], validate=True)
            image = Image.open(io.BytesIO(png)).convert('RGB')
            prompt = f"In: What action should the robot take to {req['language'].lower()}?\nOut:"
            inputs = processor(prompt, image).to('cuda:0', dtype=torch.bfloat16)
            kwargs = generation_kwargs(req['source'])
            generated.clear()
            start = time.monotonic()
            native = model.predict_action(**inputs, unnorm_key=key, **kwargs)
            torch.cuda.synchronize()
            native = np.asarray(native)
            action = native.copy()
            if action.shape == (7,):
                action[-1] = -np.sign(2*action[-1]-1)
            valid = (action.shape == native.shape == (7,) and np.isfinite(action).all()
                     and np.isfinite(native).all() and len(generated) == 7)
            print('RESULT '+json.dumps(dict(action=action.tolist(), native_action=native.tolist(),
                tokens=generated.copy(), model_inference_s=time.monotonic()-start,
                source=req['source'], seed=req['seed'], sampling=kwargs, query_index=query_index,
                valid=bool(valid), invalid_reason=None if valid else 'invalid_official_action_or_tokens',
                resolved_generation_config=resolved.copy(),
                policy_png_sha256=hashlib.sha256(png).hexdigest(),
                policy_rgb_sha256=hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
                prompt=prompt)), flush=True)
            if not valid:
                return


def generation_kwargs(source):
    settings = dict(do_sample=source == 'sample', top_k=50, num_beams=1, num_return_sequences=1)
    if source == 'sample':
        settings.update(temperature=.7, top_p=.95)
    return settings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot')
    parser.add_argument('code_snapshot')
    parser.add_argument('--gpu', type=int, choices=[2, 3], required=True)
    parser.add_argument('--physical-uuid', required=True)
    parser.add_argument('--egl-device', type=int, choices=[0, 1], required=True)
    main(parser.parse_args())
