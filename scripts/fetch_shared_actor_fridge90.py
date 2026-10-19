"""Bounded supervisor calls this to fetch one revision and verify weight SHA256."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from huggingface_hub import snapshot_download


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('out',type=Path)
    a=p.parse_args()
    start=time.time()
    repo='VQ-VLA/openvla-7b-finetuned-libero-90'
    revision='794ef81b7be928ea9270e81ca1ef5b60ffa9420f'
    metadata=Path('/home/ami/scratch/shared_actor_fridge_pretrained_audit_20261019/models__VQ-VLA__openvla-7b-finetuned-libero-90/api.json')
    api=json.loads(metadata.read_text())
    assert api['sha']==revision
    path=snapshot_download(repo,revision=revision,local_dir=str(a.out),
        allow_patterns=['*.json','*.safetensors','tokenizer.model'],max_workers=4)
    hashes={}
    for entry in api['siblings']:
        file=a.out/entry['rfilename']
        if not file.is_file(): continue
        with file.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
        expected=entry.get('lfs',{}).get('sha256')
        if expected: assert digest==expected,file
        hashes[entry['rfilename']]=dict(sha256=digest,bytes=file.stat().st_size,lfs_verified=expected is not None)
    assert len([s for s in hashes if s.endswith('.safetensors')])==4
    report=dict(repo=repo,revision=revision,path=path,started=start,ended=time.time(),files=hashes,
                downloaded_bytes=sum(f['bytes'] for f in hashes.values()),status='hash_verified')
    (a.out/'download_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
