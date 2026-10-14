"""Archive one stopped/completed LIBERO run using the existing recovery schema."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile

from backup_research_progress import digest, verify


def archive(run, output, extra):
    output.mkdir(parents=True, exist_ok=False)
    (output/'archives').mkdir()
    (output/'videos').mkdir()
    files = [(p, f'scratch/{run.name}/{p.relative_to(run).as_posix()}')
             for p in sorted(run.rglob('*')) if p.is_file()]
    files += [(p, f'scratch/{p.name}') for p in extra]
    path = output/'archives'/f'{run.name}.tar.gz'
    members = []
    with tarfile.open(path, 'w:gz', compresslevel=4) as tar:
        for source, target in files:
            assert not source.is_symlink()
            before = digest(source)
            tar.add(source, arcname=target, recursive=False)
            assert digest(source) == before, f'Source changed during backup: {source}'
            members.append(dict(path=target, bytes=source.stat().st_size, sha256=before))
    videos = []
    for source in sorted((run/'review').glob('*.mp4')):
        target = output/'videos'/source.name
        shutil.copy2(source, target)
        videos.append(dict(file=target.relative_to(output).as_posix(), bytes=target.stat().st_size,
                           sha256=digest(target)))
    manifest = dict(schema=1, source_run=str(run),
                    launch=json.loads((run/'launch.json').read_text()),
                    privileged=True, learned_predictor=False,
                    archives=[dict(file=path.relative_to(output).as_posix(), bytes=path.stat().st_size,
                                   sha256=digest(path), members=members)], videos=videos)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    verify(output)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--extra', type=Path, nargs='*', default=[])
    args = p.parse_args()
    archive(args.run, args.output, args.extra)
