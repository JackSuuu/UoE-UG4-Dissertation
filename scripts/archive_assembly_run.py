"""Archive completed assembly evidence, including learned-model identity."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main(root, out):
    launch = json.loads((root/'launch.json').read_text())
    assert launch['status'] in ('complete_verified', 'completed', 'failed')
    out.mkdir(parents=True, exist_ok=False)
    (out/'archives').mkdir()
    archive = out/'archives'/(root.name+'.tar.gz')
    members = []
    with tarfile.open(archive, 'w:gz', compresslevel=4, dereference=True) as tar:
        for path in sorted(root.rglob('*')):
            if not path.is_file():
                continue
            before = digest(path)
            name = root.name+'/'+path.relative_to(root).as_posix()
            tar.add(path, arcname=name, recursive=False)
            assert digest(path) == before, f'File changed while archiving: {path}'
            members.append(dict(path=name, sha256=before, bytes=path.stat().st_size))
    with tarfile.open(archive) as tar:
        for item in members:
            data = tar.extractfile(item['path']).read()
            assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256']
    videos = []
    for path in sorted((root/'review').glob('*.mp4')):
        dest = out/'videos'/path.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(path, dest)
        videos.append(dict(file=dest.relative_to(out).as_posix(), sha256=digest(dest)))
    manifest = dict(status='local_archive_verified', source=str(root), launch=launch,
        learned_predictor=True, implementation='self-developed OrbiSim-role/RefCheckVLA; not official',
        privileged_state=True, archive=archive.relative_to(out).as_posix(),
        archive_sha256=digest(archive), members=members, videos=videos)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(status=manifest['status'], files=len(members), bytes=archive.stat().st_size, archive=str(archive))))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    main(a.root, a.out)
