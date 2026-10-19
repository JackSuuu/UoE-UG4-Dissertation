"""Archive a stopped L1 cohort in bounded files, preserving every attempted cell."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile

from backup_research_progress import digest, verify


def main(run, out, extras):
    launch = json.loads((run / 'launch.json').read_text())
    if launch['status'] not in ('complete_verified', 'complete_with_issues', 'interrupted'):
        raise ValueError('Only stopped cohorts can be archived')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'archives').mkdir()
    (out / 'videos').mkdir()
    grouped = {}
    for job in launch['jobs']:
        for key in ('episode_dir', 'replay_dir'):
            grouped[Path(job[key]).name] = job['task']
    groups = {'common': []}
    for path in sorted(run.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        if path.is_symlink():
            raise ValueError(f'Unexpected symlink {path}')
        relative = path.relative_to(run)
        group = grouped.get(relative.parts[0], 'common')
        groups.setdefault(group, []).append((path, f'scratch/{run.name}/{relative.as_posix()}'))
    for extra in extras:
        if extra.is_dir():
            for path in sorted(extra.rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts:
                    groups['common'].append((path, f'scratch/{extra.name}/{path.relative_to(extra).as_posix()}'))
        else:
            groups['common'].append((extra, f'scratch/{extra.name}'))
    manifest = dict(schema=1, source_run=str(run), launch=launch,
                    scope='native_actor_screening_no_framework_gain_claim',
                    actor_observations='native_RGB_and_original_language_only',
                    diagnostic_observations='privileged_simulator_evidence_offline_only',
                    learned_predictor=False, archives=[], videos=[])
    # Bound each uncompressed group to1GiB; avoids a single oversized LFS object.
    for group, files in groups.items():
        chunks, chunk, size = [], [], 0
        for path, target in files:
            if path.is_symlink():
                raise ValueError(f'Unexpected symlink {path}')
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
    for path in sorted((run / 'review').glob('*.mp4')):
        target = out / 'videos' / path.name
        shutil.copy2(path, target)
        manifest['videos'].append(dict(file=target.relative_to(out).as_posix(),
                                      bytes=target.stat().st_size, sha256=digest(target)))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    verify(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--extra', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    main(args.run.resolve(), args.out.resolve(), [path.resolve() for path in args.extra])
