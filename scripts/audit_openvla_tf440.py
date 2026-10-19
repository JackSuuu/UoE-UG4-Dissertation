"""Offline pinned-transformers inference audit on fixed R2 observations.

Run --export in the existing environment, then run in an isolated TF4.40.1 venv.
No simulator actions or policy/model changes are made by this diagnostic.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CASES = [('seed0_closed_actor', 0), ('seed0_closed_actor', 200),
         ('seed0_open_actor', 0), ('seed0_open_actor', 300),
         ('seed0_open_framework', 245), ('seed0_open_framework', 400)]


def export(root, out):
    sys.path.insert(0, str(REPO/'src'))
    from adapters.libero_openvla import policy_image
    from adapters import tf5_compat
    from transformers import AutoProcessor, LlamaForCausalLM
    tf5_compat.setup_hf_offline()
    out.mkdir(parents=True, exist_ok=False)
    status = json.loads((root/CASES[0][0]/'status.json').read_text())
    processor = tf5_compat.load_processor(status['args']['snapshot'])
    manifest = dict(snapshot=status['args']['snapshot'], code_snapshot=tf5_compat.snapshot_dir(),
                    source_root=str(root), cases=[], scope='fixed_offline_decode_audit')
    for name in sorted(set(name for name, _ in CASES)):
        rows = [json.loads(l) for l in (root/name/'steps.jsonl').read_text().splitlines()]
        with np.load(root/name/'images.npz') as data:
            images = data['agent']
        for _, step in [case for case in CASES if case[0] == name]:
            row = rows[step]
            image = policy_image(images[step])
            key = f'{name}_{step}'
            image.save(out/(key+'.png'))
            prompt = f"In: What action should the robot take to {row['language'].lower()}?\nOut:"
            inputs = processor(prompt, image)
            np.savez(out/(key+'.npz'), input_ids=inputs['input_ids'].numpy(),
                     pixel_values=inputs['pixel_values'].numpy())
            manifest['cases'].append(dict(key=key, prompt=prompt, recorded_tokens=row['tokens'],
                recorded_native=row['native_action'], recorded_proposal=row['proposal'],
                raw_image_sha256=hashlib.sha256(images[step].tobytes()).hexdigest()))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


def audit(out, uncached=False):
    import torch
    import transformers
    from PIL import Image
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    assert transformers.__version__ == '4.40.1'
    torch.set_num_threads(4)
    torch.manual_seed(0)
    manifest = json.loads((out/'manifest.json').read_text())
    base, snapshot = manifest['code_snapshot'], manifest['snapshot']
    config_class = get_class_from_dynamic_module('configuration_prismatic.OpenVLAConfig', base)
    model_class = get_class_from_dynamic_module('modeling_prismatic.OpenVLAForActionPrediction', base)
    processor_class = get_class_from_dynamic_module('processing_prismatic.PrismaticProcessor', base)
    config = config_class.from_pretrained(snapshot)
    model = model_class.from_pretrained(snapshot, config=config, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation='sdpa').to('cuda:0').eval()
    processor = processor_class.from_pretrained(snapshot, trust_remote_code=True)
    stats = json.loads((Path(snapshot)/'dataset_statistics.json').read_text())
    model.norm_stats = stats
    key = 'libero_10' if 'libero_10' in stats else 'libero_10_no_noops'
    report = dict(scope='pinned_transformers_offline_prediction_not_full_simulator_parity',
        use_cache=not uncached,
        versions={k: importlib.metadata.version(k) for k in ('torch', 'transformers', 'timm', 'tokenizers')},
        unnorm_key=key, cases=[])
    generate = model.generate
    generated = []

    def capture(*args, **kwargs):
        result = generate(*args, **kwargs)
        generated.append(result[0, -7:].tolist())
        return result

    model.generate = capture
    with torch.inference_mode():
        for case in manifest['cases']:
            inputs = processor(case['prompt'], Image.open(out/(case['key']+'.png')).convert('RGB'))
            with np.load(out/(case['key']+'.npz')) as current:
                ids_equal = np.array_equal(current['input_ids'], inputs['input_ids'].numpy())
                pixels_gap = float(np.abs(current['pixel_values']-inputs['pixel_values'].numpy()).max())
            start = time.monotonic()
            inputs = inputs.to('cuda:0', dtype=torch.bfloat16)
            native = model.predict_action(**inputs, unnorm_key=key, do_sample=False, use_cache=not uncached)
            action = native.copy()
            action[-1] = -np.sign(2*action[-1]-1)
            result = dict(key=case['key'], processor_ids_equal=ids_equal,
                processor_pixels_max_gap=pixels_gap, tokens=generated[-1],
                recorded_tokens=case['recorded_tokens'], tokens_equal=generated[-1] == case['recorded_tokens'],
                native=native.tolist(), action=action.tolist(),
                max_action_gap=float(np.abs(action-case['recorded_proposal']).max()),
                inference_s=time.monotonic()-start)
            report['cases'].append(result)
            print(json.dumps(result), flush=True)
    report['all_exact'] = all(r['tokens_equal'] and r['processor_ids_equal'] and
        r['processor_pixels_max_gap'] == 0 and r['max_action_gap'] == 0 for r in report['cases'])
    name = 'report_uncached.json' if uncached else 'report.json'
    (out/name).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('out', type=Path)
    p.add_argument('--export', type=Path)
    p.add_argument('--uncached', action='store_true')
    a = p.parse_args()
    if a.export:
        export(a.export.resolve(), a.out.resolve())
    else:
        audit(a.out.resolve(), a.uncached)
