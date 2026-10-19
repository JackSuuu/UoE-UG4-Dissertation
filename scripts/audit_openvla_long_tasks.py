"""Read-only LIBERO-Long task/source inventory; no environment or actor execution."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import urllib.request


CHECKPOINT = 'openvla/openvla-7b-finetuned-libero-10'
REVISION = '80970322773f81baa2e22fe495d0487b93a05cfa'
PRIORITY = [
    'KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it',
    'KITCHEN_SCENE8_put_both_moka_pots_on_the_stove',
    'LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate',
    'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it',
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def expressions(text):
    tokens = iter(re.findall(r'\(|\)|[^\s()]+', re.sub(r';[^\n]*', '', text)))

    def parse():
        items = []
        for token in tokens:
            if token == ')':
                return items
            items.append(parse() if token == '(' else token)
        return items

    return parse()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--runtime-libero', type=Path, default=Path('/home/ami/scratch/LIBERO'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    base = args.repo / 'third_party/LIBERO/libero/libero'
    task_map_path = base / 'benchmark/libero_suite_task_map.py'
    tree = ast.parse(task_map_path.read_text())
    task_map = next(ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == 'libero_task_map' for t in node.targets))
    rows = []
    for index, name in enumerate(task_map['libero_10']):
        path = base / 'bddl_files/libero_10' / (name + '.bddl')
        raw = path.read_bytes()
        parsed = expressions(raw.decode())[0]
        clauses = {part[0]: part[1:] for part in parsed if isinstance(part, list) and part}
        runtime_bddl = args.runtime_libero / 'libero/libero/bddl_files/libero_10' / path.name
        initial = args.runtime_libero / 'libero/libero/init_files/libero_10' / (name + '.pruned_init')
        same = runtime_bddl.exists() and runtime_bddl.read_bytes() == raw
        assert same, runtime_bddl
        assert initial.is_file(), initial
        rows.append(dict(source_list_index=index, task_name=name,
                         language=' '.join(clauses[':language']), init=clauses[':init'],
                         goal=clauses[':goal'], bddl_sha256=digest(raw),
                         runtime_bddl_identical=same, initial_file=str(initial),
                         initial_file_sha256=digest(initial.read_bytes()),
                         shortlist_rank=PRIORITY.index(name)+1 if name in PRIORITY else None))
    urls = {
        'model_card.md': f'https://huggingface.co/{CHECKPOINT}/raw/{REVISION}/README.md',
        'paper.html': 'https://arxiv.org/html/2406.09246v3',
        'official_eval.py.txt': 'https://raw.githubusercontent.com/openvla/openvla/main/experiments/robot/libero/run_libero_eval.py',
    }
    sources = {}
    for name, url in urls.items():
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        (args.out / name).write_bytes(data)
        sources[name] = dict(url=url, bytes=len(data), sha256=digest(data))
    assert 'LIBERO-Long' in (args.out/'model_card.md').read_text()
    official_eval = (args.out/'official_eval.py.txt').read_text()
    assert 'max_steps = 520' in official_eval and 'num_steps_wait: int = 10' in official_eval
    result = dict(scope='task_inventory_and_protocol_research_no_rollouts',
                  actor=CHECKPOINT, revision=REVISION, suite='libero_10 (LIBERO-Long)',
                  source_map_sha256=digest(task_map_path.read_bytes()),
                  tasks=rows, sources=sources, actor_queries=0, robot_actions=0,
                  published_suite_success=dict(mean_percent=53.7,standard_error_percent=1.3,
                      source='OpenVLA arXiv:2406.09246v3 Appendix E Table 12',
                      interpretation='suite-level published result, not per-task or local reproduction'),
                  official_budget=dict(model_actions=520,initial_settle_actions=10),
                  task_specific_failure_rates='not established by this audit')
    (args.out/'inventory.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(tasks=len(rows), shortlist=PRIORITY, output=str(args.out/'inventory.json')),indent=2))


if __name__ == '__main__':
    main()
