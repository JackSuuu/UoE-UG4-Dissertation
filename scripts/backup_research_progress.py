"""Snapshot research artifacts, then verify/restore without trusting local originals.

Archive members have workspace-relative names (scratch/ and repository/src/results/).
Git LFS transports archives/videos; manifest hashes also cover every source file.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def create(output, scratch, repository):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'archives').mkdir()
    (output / 'videos').mkdir()
    groups = {'libero': [], 'legacy-scratch': [], 'legacy-results': []}
    for p in sorted(scratch.rglob('*')):
        relative = p.relative_to(scratch)
        if relative.parts[0] in {'LIBERO', 'venv_probe'} or '__pycache__' in relative.parts:
            continue
        if not p.is_file():
            continue
        if p.is_symlink():
            raise ValueError(f'Unexpected symlink: {p}')
        name = 'libero' if relative.parts[0].startswith(('libero', 'openvla_libero')) else 'legacy-scratch'
        if p.suffix == '.h5':
            name = p.stem
        groups.setdefault(name, []).append((p, 'scratch/' + relative.as_posix()))
    for p in sorted((repository / 'src/results').rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:
            groups['legacy-results'].append((p, 'repository/' + p.relative_to(repository).as_posix()))
    manifest = dict(schema=1, date='2026-10-14', source_commit='b97d58d',
                    actor='Frozen task-finetuned OpenVLA; no robot physics repair enabled',
                    excluded=['scratch/LIBERO: pinned submodule instead',
                              'scratch/venv_probe: disposable environment', '__pycache__',
                              'HF full7B weights: pinned upstream downloads, see README',
                              'uncommitted working notes: left in original working tree'],
                    archives=[], videos=[])
    for group, files in groups.items():
        archive = output / 'archives' / (group + '.tar.gz')
        entries = []
        with tarfile.open(archive, 'w:gz', compresslevel=4) as tar:
            for source, target in files:
                before = digest(source)
                tar.add(source, arcname=target, recursive=False)
                if digest(source) != before:
                    raise ValueError(f'Source changed during backup: {source}')
                entries.append(dict(path=target, bytes=source.stat().st_size, sha256=before))
        manifest['archives'].append(dict(file=archive.relative_to(output).as_posix(),
                                        bytes=archive.stat().st_size, sha256=digest(archive), members=entries))
        print(f'{group}: {len(entries)} files, {archive.stat().st_size / 2**20:.1f} MiB', flush=True)
    review = scratch / 'libero_continuation_20261013_a/review'
    for source in sorted(review.glob('*.mp4')):
        target = output / 'videos' / source.name
        shutil.copy2(source, target)
        manifest['videos'].append(dict(file=target.relative_to(output).as_posix(),
                                      bytes=target.stat().st_size, sha256=digest(target)))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    verify(output)


def verify(output, destination=None):
    manifest = json.loads((output / 'manifest.json').read_text())
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=False)
    count = 0
    for item in manifest['archives'] + manifest['videos']:
        source = output / item['file']
        assert source.stat().st_size == item['bytes'], source
        assert digest(source) == item['sha256'], source
        if 'members' not in item:
            continue
        expected = {m['path']: m for m in item['members']}
        seen = set()
        with tarfile.open(source, 'r:gz') as tar:
            for member in tar:
                path = Path(member.name)
                assert member.isfile() and not path.is_absolute() and '..' not in path.parts, member.name
                assert member.name in expected and member.name not in seen, member.name
                seen.add(member.name)
                h = hashlib.sha256()
                size = 0
                out = None
                if destination is not None:
                    target = destination / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    out = target.open('xb')
                try:
                    with tar.extractfile(member) as f:
                        for block in iter(lambda: f.read(1024 * 1024), b''):
                            h.update(block)
                            size += len(block)
                            if out is not None:
                                out.write(block)
                finally:
                    if out is not None:
                        out.close()
                assert h.hexdigest() == expected[member.name]['sha256'], member.name
                assert size == expected[member.name]['bytes'], member.name
                count += 1
        assert seen == set(expected), source
    print(json.dumps(dict(verified_source_files=count, archives=len(manifest['archives']),
                          videos=len(manifest['videos']), restored_to=str(destination) if destination else None)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['create', 'verify', 'restore'])
    parser.add_argument('backup', type=Path)
    parser.add_argument('--scratch', type=Path, default=Path('/home/ami/scratch'))
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    if args.operation == 'create':
        create(args.backup, args.scratch, args.repository)
    elif args.operation == 'restore':
        if args.destination is None:
            parser.error('restore requires a NEW --destination directory')
        verify(args.backup, args.destination)
    else:
        verify(args.backup)
