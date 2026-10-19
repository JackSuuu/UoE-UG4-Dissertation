"""Public metadata-only pretrained/data audit. Never fetch weights or run remote code."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import urllib.request


REPOS = [
    ('datasets','yifengzhu-hf/LIBERO-datasets'),
    ('datasets','jesbu1/libero_90_rlds'),
    ('models','VQ-VLA/openvla-7b-finetuned-libero-90'),
    ('models','Stanford-ILIAD/minivla-libero90-prismatic'),
    ('models','lerobot/smolvla_robocasa'),
    ('datasets','pepijn223/robocasa_CloseFridge'),
    ('datasets','lerobot/robocasa_target_human_unified'),
    ('models','aleleanza/diffusion-policy-vac-fridge'),
    ('models','ryota-tron/test_act_open_fridge'),
]


def get(url):
    request = urllib.request.Request(url, headers={'User-Agent':'shared-actor-data-audit/1.0'})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read(5_000_001)
    if len(data)>5_000_000:
        raise RuntimeError('Metadata response exceeds 5 MB cap')
    return data


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    def audit(pair):
        kind, repo = pair
        dest = out/(kind+'__'+repo.replace('/','__'))
        dest.mkdir()
        url = f'https://huggingface.co/api/{kind}/{repo}?blobs=true'
        result = dict(kind=kind, repo=repo, api_url=url, accessed=time.time(), files={})
        try:
            raw = get(url)
            (dest/'api.json').write_bytes(raw)
            api = json.loads(raw)
            result.update(revision=api['sha'], gated=api.get('gated'),
                total_repo_file_bytes=sum(s.get('size',0) for s in api.get('siblings',[])),
                files=[dict(name=s['rfilename'], bytes=s.get('size'),
                            lfs_sha256=s.get('lfs',{}).get('sha256')) for s in api.get('siblings',[])])
            prefix = 'datasets/' if kind=='datasets' else ''
            paths = ['README.md','meta/info.json'] if kind=='datasets' else [
                'README.md','config.json','dataset_statistics.json','policy_preprocessor.json','policy_postprocessor.json']
            available = {s['rfilename'] for s in api.get('siblings',[])}
            result['metadata'] = {}
            for name in paths:
                if name not in available:
                    continue
                link = f'https://huggingface.co/{prefix}{repo}/resolve/{api["sha"]}/{name}'
                try:
                    raw = get(link)
                    (dest/name.replace('/','__')).write_bytes(raw)
                    result['metadata'][name] = dict(url=link, bytes=len(raw),
                        sha256=hashlib.sha256(raw).hexdigest())
                    if name in ('config.json','meta/info.json'):
                        cfg = json.loads(raw)
                        result['schema'] = {k:cfg[k] for k in (
                            'type','model_type','architectures','input_features','output_features',
                            'normalization_mapping','chunk_size','n_action_steps','total_episodes',
                            'total_frames','total_tasks','robot_type','fps','splits') if k in cfg}
                    if name=='dataset_statistics.json':
                        result['normalization'] = json.loads(raw)
                except Exception as exc:
                    result['metadata'][name] = dict(error=repr(exc),url=link)
        except Exception as exc:
            result['error'] = repr(exc)
        return result
    with ThreadPoolExecutor(max_workers=5) as pool:
        repos = list(pool.map(audit, REPOS))
    task_file = Path('/home/ami/scratch/LIBERO/libero/libero/benchmark/libero_suite_task_map.py')
    tree = ast.parse(task_file.read_text())
    mapping = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and isinstance(n.value, ast.Dict))
    tasks = mapping['libero_90']
    (out/'libero_suite_task_map.py.txt').write_bytes(task_file.read_bytes())
    result = dict(created=time.time(), repos=repos, local_libero90=dict(count=len(tasks), tasks=tasks,
        fridge_matches=[s for s in tasks if 'fridge' in s.lower() or 'refrigerator' in s.lower()],
        related_primitives=[s for s in tasks if any(word in s for word in ('microwave','cream_cheese'))],
        source=str(task_file), sha256=hashlib.sha256(task_file.read_bytes()).hexdigest()),
        local_runtime_available={p:importlib.util.find_spec(p) is not None for p in ('lerobot','robocasa')},
        weights_downloaded=False, demonstrations_downloaded=False, remote_code_executed=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(out), repos=[{k:r.get(k) for k in
        ('repo','revision','total_repo_file_bytes','error','schema')} for r in repos]),indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('out',type=Path)
    main(p.parse_args().out.resolve())
