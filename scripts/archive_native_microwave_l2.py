"""Archive a stopped L2 diagnostic bank, retaining reconstruction and replay."""
import argparse
import json
from pathlib import Path
import tarfile

from backup_research_progress import digest, verify


def main(run, out, extras):
    launch = json.loads((run / 'launch.json').read_text())
    if launch['status'] not in ('completed', 'aborted'):
        raise ValueError('Only stopped diagnostic banks can be archived')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'archives').mkdir()
    grouped = {}
    for cell in launch['cells']:
        for name in (cell['name'], 'replay_' + cell['name']):
            grouped[name] = f'root{cell["root"]}'
    groups = {'common': []}
    for root in [run, *extras]:
        files = sorted(root.rglob('*')) if root.is_dir() else [root]
        for path in files:
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            if path.is_symlink():
                raise ValueError(f'Unexpected symlink: {path}')
            relative = path.relative_to(root) if root.is_dir() else Path()
            target = f'scratch/{root.name}' + ('/' + relative.as_posix() if root.is_dir() else '')
            group = grouped.get(relative.parts[0], 'common') if root == run else 'common'
            groups.setdefault(group, []).append((path, target))
    manifest = dict(schema=1, source_run=str(run), launch=launch,
                    scope='offline_candidate_reachability_not_framework_efficacy', archives=[], videos=[])
    targets = set()
    for group, files in groups.items():
        chunks, chunk, size = [], [], 0
        for path, target in files:
            if target in targets or path.stat().st_size > 1024**3:
                raise ValueError(f'Duplicate target or oversized single file: {target}')
            targets.add(target)
            if chunk and size + path.stat().st_size > 1024**3:
                chunks.append(chunk)
                chunk, size = [], 0
            chunk.append((path, target))
            size += path.stat().st_size
        if chunk:
            chunks.append(chunk)
        for index, chunk in enumerate(chunks):
            archive = out / 'archives' / f'{group}_{index:02d}.tar.gz'
            members = []
            with tarfile.open(archive, 'w:gz', compresslevel=4) as tar:
                for path, target in chunk:
                    before = digest(path)
                    tar.add(path, arcname=target, recursive=False)
                    if digest(path) != before:
                        raise ValueError(f'Artifact changed during archive: {path}')
                    members.append(dict(path=target, bytes=path.stat().st_size, sha256=before))
            manifest['archives'].append(dict(file=archive.relative_to(out).as_posix(),
                bytes=archive.stat().st_size, sha256=digest(archive), members=members))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    verify(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--extra', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    main(args.run.resolve(), args.out.resolve(), [p.resolve() for p in args.extra])
