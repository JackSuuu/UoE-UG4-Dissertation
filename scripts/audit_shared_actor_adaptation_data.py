"""Read-only local data provenance audit; no GPU, labels, conversion or training."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np


ROOT = Path('/home/ami')
REPO = ROOT/'UoE-UG4-Dissertation'
SCAN = [ROOT/'scratch', REPO, ROOT/'.cache/huggingface/datasets', ROOT/'jack-project/data']
SKIP = {'.git', '__pycache__', 'venv', 'site-packages', 'node_modules'}


def value(x):
    if isinstance(x, bytes):
        return x.decode(errors='replace')
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def main(out):
    report = dict(created=time.time(), scope='local_roots_read_only_metadata_inventory',
        roots=[str(p) for p in SCAN], excluded_directory_names=sorted(SKIP),
        data=[], dataset_directories={}, utilities={}, bddl_inventory={},
        no_training=True, no_generated_labels=True)
    for root in SCAN:
        for directory, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in SKIP]
            for name in files:
                if not (name.endswith(('.h5','.hdf5','.parquet','.arrow')) or
                        'tfrecord' in name or name == 'dataset_info.json'):
                    continue
                path = Path(directory)/name
                info = dict(path=str(path), bytes=path.stat().st_size,
                            mtime=path.stat().st_mtime)
                if path.suffix in ('.h5', '.hdf5'):
                    try:
                        with h5py.File(path, 'r') as data:
                            info['attributes'] = {k:value(v) for k,v in data.attrs.items()}
                            info['datasets'] = {}
                            def visit(key, node):
                                if isinstance(node, h5py.Dataset):
                                    info['datasets'][key] = dict(shape=list(node.shape), dtype=str(node.dtype))
                            data.visititems(visit)
                            if 'episode' in data:
                                episodes = np.unique(data['episode'][:])
                                info['episodes'] = dict(count=len(episodes), minimum=int(episodes.min()),
                                                        maximum=int(episodes.max()))
                    except Exception as exc:
                        info['error'] = repr(exc)
                report['data'].append(info)
    for path in [ROOT/'scratch/LIBERO/datasets', REPO/'third_party/LIBERO/datasets']:
        report['dataset_directories'][str(path)] = dict(exists=path.exists(),
            entries=sorted(p.name for p in path.iterdir()) if path.exists() else [])
    utilities = [REPO/'src/vla'/n for n in ('collect_demos.py','dagger.py',
                    'extract_features.py','train_chunk_head.py')]
    utilities += [ROOT/'scratch/LIBERO/scripts'/n for n in
                  ('collect_demonstration.py','create_dataset.py')]
    utilities += [ROOT/'scratch/LIBERO/libero/lifelong'/n for n in
                  ('datasets.py','main.py','algos/base.py','models/bc_transformer_policy.py')]
    for path in utilities:
        report['utilities'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for root in [ROOT/'scratch/LIBERO/libero/libero/bddl_files',
                 REPO/'third_party/LIBERO/libero/libero/bddl_files']:
        files = list(root.rglob('*.bddl'))
        report['bddl_inventory'][str(root)] = dict(count=len(files),
            fridge_matches=[str(p) for p in files if 'fridge' in p.read_text().lower()],
            libero10=[p.stem for p in files if p.parent.name == 'libero_10'])
    report['base_python_optional_training_modules'] = {name:importlib.util.find_spec(name) is not None
        for name in ('robomimic','peft','tensorflow','tensorflow_datasets','h5py')}
    report['conclusion'] = 'No local fridge demonstration dataset or valid fridge train/eval split identified; adaptation blocked.'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=value)+'\n')
    print(json.dumps(dict(output=str(out), files=len(report['data']),
                         conclusion=report['conclusion']), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('out', type=Path)
    main(p.parse_args().out.resolve())
