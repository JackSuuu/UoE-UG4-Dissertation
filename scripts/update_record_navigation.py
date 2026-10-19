"""Repair Markdown navigation after the user-requested agent_notes relocation.

Historical launch manifests retain their original paths/hashes. Only Markdown
links change, never protocol prose, controller code, archives or user notes/ files.
Run from the repository root after moving the records with a file patch.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    repo = Path(__file__).resolve().parents[1]
    folder = repo/'exp_record/agent_notes'
    mapping = {}
    records = []
    for new in sorted(folder.glob('*.md')):
        old = repo/'exp_record'/new.name
        result = subprocess.run(['git', 'show', f'HEAD:{old.relative_to(repo)}'], cwd=repo, capture_output=True)
        if result.returncode:
            continue  # New Week5 notes have no old location.
        assert not old.exists()
        assert new.read_bytes() == result.stdout, f'Unexpected content change before relocation: {new}'
        mapping[old] = new
        records.append(dict(old=str(old.relative_to(repo)), new=str(new.relative_to(repo)),
            original_sha256=hashlib.sha256(result.stdout).hexdigest()))
    reverse = {v: k for k, v in mapping.items()}
    paths = [repo/'README.md', repo/'core_research_plan.md', *sorted((repo/'exp_record').glob('week*.md')), *mapping.values()]
    for path in paths:
        original = path.read_text()
        previous = reverse.get(path, path)
        def replace(match):
            target = match.group(1)
            if ':' in target or target.startswith(('#', '/')): return match.group(0)
            name, mark, anchor = target.partition('#')
            resolved = (previous.parent/name).resolve()
            relocated = mapping.get(resolved, resolved)
            if path == previous and resolved not in mapping: return match.group(0)
            new_target = os.path.relpath(relocated, path.parent)+(mark+anchor if mark else '')
            return ']('+new_target+')'
        updated = re.sub(r'\]\(([^)\s]+)\)', replace, original)
        if updated != original:
            assert 'protocol' not in path.name, f'Frozen protocol link edit requires explicit review: {path}'
            path.write_text(updated)
    for item in records:
        item['relocated_sha256'] = hashlib.sha256((repo/item['new']).read_bytes()).hexdigest()
        item['content_identical'] = item['original_sha256'] == item['relocated_sha256']
    result = dict(reason='User requested weekly-only outer exp_record, agent_notes for detailed records',
        original_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        historical_replay='Use the frozen commit/source archive for launchers that require original paths.', records=records)
    (folder/'relocation_manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(relocated=len(records), byte_identical=sum(x['content_identical'] for x in records))))


if __name__ == '__main__': main()
