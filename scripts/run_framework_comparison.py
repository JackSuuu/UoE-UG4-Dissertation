"""Historical microwave factorial runner. New task-agnostic runs use run_task.py.

The five-layer controller uses privileged measured verification, not a learned
predictor. Every post-initialization action is one frozen official VLA proposal.

--fusion-mode log/gate uses a pinned trained fusion artifact
to record scores or additionally veto verified exits (budget exits stay bounded).

This entry point is foreground/setsid compatible; only --launch starts models.
Explicit --recovery-options is a distinct oracle-skill action protocol, not an
unchanged VLA or bounded-translation experiment. Skill proposals are recomputed
from live measurements in tape replay; unused VLA proposals remain recorded.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parent.parent
PYTHON = '/home/jack/miniconda3/bin/python'

# Lazy-loaded scorer components (only imported when --recovery-scorer or
# --handoff-scorer is set).
_SCORER_MODULES = None


def _load_scorer_modules():
    global _SCORER_MODULES
    if _SCORER_MODULES is not None:
        return _SCORER_MODULES
    sys.path.insert(0, str(REPO))
    from src.perception.frozen_visual_encoder import FrozenVisualEncoder
    from src.verification.fusion_mlp import extract_physics_features
    from src.verification.physical_verifier import PhysicalVerifier
    from src.repair.physics_evidence import extract_physics_evidence
    from src.repair.recovery_scorer import RecoveryScorer
    from src.repair.handoff_scorer import HandoffScorer
    _SCORER_MODULES = dict(
        FrozenVisualEncoder=FrozenVisualEncoder,
        extract_physics_features=extract_physics_features,
        PhysicalVerifier=PhysicalVerifier,
        extract_physics_evidence=extract_physics_evidence,
        RecoveryScorer=RecoveryScorer,
        HandoffScorer=HandoffScorer,
    )
    return _SCORER_MODULES
LANES = [dict(gpu=2, egl_device=1, physical_uuid='GPU-99b39a1b-5fdd-1acb-398a-426672ebc1a8'),
         dict(gpu=3, egl_device=0, physical_uuid='GPU-c164f9d9-f157-daeb-d7be-5c98029c2a2b')]
COHORTS = ('baseline', 'v2', 'prompt_only', 'prompt_place', 'prompt_deepen', 'prompt_settle')
LIMITS = dict(roots=6, cells=36, seed=0, initialization_noops=10, actions=520,
              queries=520, total_queries=18720, batch=1, parallel=2,
              readiness_seconds=180, query_seconds=30, rollout_seconds=1200,
              outer_seconds=1800, replay_seconds=300, phase_seconds=14400, retries=0)
SOURCES = ('scripts/run_framework_comparison.py', 'src/legacy/microwave_controller.py', 'src/legacy/__init__.py', 'src/__init__.py',
           'scripts/run_native_long_branch.py', 'scripts/native_long_candidate_worker.py',
            'scripts/test_native_framework.py', 'scripts/test_action_correction.py',
           'scripts/test_fusion_handoff.py',
           'scripts/run_fusion_handoff_ablation.py',
             'scripts/test_joint_framework.py', 'scripts/run_joint_framework.py',
             'scripts/run_framework_diagnostic_study.py', 'scripts/check_framework_trace_compatibility.py',
              'src/repair/experimental/contract_progress.py', 'scripts/test_repair_contract_progress.py',
              'scripts/test_repair_candidates.py', 'scripts/run_repair_candidate_study.py',
              'scripts/analyze_repair_candidates.py',
               'scripts/run_repair_data_expansion.py', 'scripts/test_repair_data_expansion.py',
               'scripts/run_fridge_repair_probe.py', 'scripts/test_fridge_repair_probe.py',
            'scripts/test_recovery_options.py', 'scripts/run_recovery_comparison.py',
           *tuple(str(path.relative_to(REPO)) for folder in (
               'adapters', 'perception', 'verification', 'prediction', 'repair', 'memory', 'training')
               for path in sorted((REPO/'src'/folder).glob('*.py'))))
DESCRIPTION = ('Five-layer deterministic framework with privileged measured verification; '
                'no learned predictor. Native In & Close only. Frozen official greedy H1 VLA, '
                'batch one; scoped language during bounded repair; no numeric action edits. '
                'Full-task prompt-refinement study focused on contain_release direction; no gate changes.')

def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def rows(path):
    path = Path(path)
    if not path.exists():
        return []
    # Preserve complete durable records when a process was killed mid-write.
    return [json.loads(line) for line in path.read_text().splitlines(keepends=True)
            if line.endswith('\n') and line.strip()]


def environment(lane):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=lane['physical_uuid'],
        NVIDIA_VISIBLE_DEVICES=lane['physical_uuid'], MUJOCO_EGL_DEVICE_ID=str(lane['egl_device']),
        MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONOPTIMIZE='0',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', TF_NUM_INTRAOP_THREADS='2',
        TF_NUM_INTEROP_THREADS='2', TOKENIZERS_PARALLELISM='false')


def framework_class():
    # The archived adapters package must remain selected for every L1 primitive.
    # A directly executed scripts/ worker does not put REPO on sys.path. Keep
    # archived adapter precedence while enabling explicit src.* skill imports.
    if str(REPO) not in sys.path:
        sys.path.append(str(REPO))
    spec = importlib.util.spec_from_file_location('comparison_controller',
                                                 REPO/'src/legacy/microwave_controller.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.NativeFramework


def cohort_cells(references):
    return [dict(name=f'{cohort}_root{root}', root=root, cohort=cohort,
        arm='baseline' if cohort == 'baseline' else 'framework',
        variant='v2' if cohort == 'baseline' else cohort,
        baseline=references[root]['baseline'], lane=deepcopy(LANES[(index+root)%2]))
        for root in range(6) for index, cohort in enumerate(COHORTS)]


def launch_waves(jobs):
    return [jobs[r*len(COHORTS)+start:r*len(COHORTS)+min(start+2, len(COHORTS))]
            for r in range(6) for start in range(0, len(COHORTS), 2)]


TRAINING_ROOTS = tuple(range(10, 20))
TRAINING_LIMITS = dict(LIMITS, roots=10, cells=20, queries=523, total_queries=26150,
                      maximum_collection_episodes=50)


def training_cells(references):
    """New initial states; root0 supplies runtime/model provenance, not outcomes."""
    return [dict(name=f'{cohort}_root{root}', root=root, cohort=cohort,
        arm='baseline' if cohort == 'baseline' else 'framework', variant='v2',
        baseline=references[0]['baseline'], runtime_reference_root=0,
        data_split='training', lane=deepcopy(LANES[root % 2]))
        for root in TRAINING_ROOTS for cohort in ('baseline', 'v2')]


def prepare(directory, mlp_path=None, joint_path=None, repair_candidates=False, training_data=False):
    """CPU-only plan; training expansion additionally reads selected official states."""
    directory = directory.resolve()
    require(not directory.exists(), f'Output exists: {directory}')
    pins, weights, references = {}, {}, []

    def pin(path, expected=None):
        path = Path(path).absolute()
        actual = pins.get(str(path)) or sha(path)
        require(expected is None or actual == expected, f'Source drift: {path}')
        pins[str(path)] = actual

    for name in SOURCES:
        pin(REPO/name)
    fusion = None
    if mlp_path is not None:
        mlp_path = mlp_path.resolve()
        report = mlp_path.parent/'report.json'
        metadata = load(report)['metadata']
        require(metadata.get('control_ready') and metadata.get('threshold') is not None,
                'Fusion artifact has no validation-selected experimental threshold')
        for path in (mlp_path, report, mlp_path.parent/'dataset_manifest.json',
                     Path(metadata['encoder']['checkpoint'])):
            pin(path)
        for path in sorted((Path(metadata['encoder']['source'])/'dinov2').rglob('*.py')):
            pin(path)
        fusion = dict(model=str(mlp_path), sha256=sha(mlp_path), metadata=metadata,
                      allowed_modes=['off', 'log', 'gate'])
    joint = None
    if joint_path is not None:
        require(fusion is not None, 'Joint runtime requires pinned real fusion artifact')
        joint_path = joint_path.resolve()
        pin(joint_path)
        pin(joint_path.parent/'report.json')
        joint = dict(artifact=str(joint_path), sha256=sha(joint_path),
            report=load(joint_path.parent/'report.json'), schema='joint_visual_physical_prediction_memory_risk_h1_v1')
    for root in range(6):
        baseline = Path(f'/home/jack/scratch/native_long_l1_20261019_{"a" if root < 3 else "b"}')/f'microwave_root{root}'
        status, manifest = load(baseline/'status.json'), load(baseline/'source_manifest.json')
        original = status['args']
        require(original['init_index'] == root and original['seed'] == 0 and
                original['steps'] == 520 and 'microwave' in original['task'], 'Wrong L1 reference')
        for name in ('status.json', 'source_manifest.json', 'initial_source.json',
                     'selected_initial_state.npy', 'scene.bddl', 'model.xml',
                     'steps.jsonl', 'frames.jsonl'):
            pin(baseline/name)
        for name, entry in manifest['repository'].items():
            pin(baseline/'source'/name, entry['sha256'])
        for section in ('runtime', 'assets', 'actor_code'):
            for path, expected in manifest.get(section, {}).items():
                pin(path, expected)
        initial = load(baseline/'initial_source.json')
        pin(initial['path'], initial['sha256'])
        pin(initial['bddl_path'], initial['bddl_sha256'])
        for entry in manifest['checkpoint']['files'].values():
            if 'sha256' in entry:
                pin(entry['path'], entry['sha256'])
            else:
                weights[entry['path']] = {k: entry[k] for k in
                    ('resolved_path', 'bytes', 'cached_blob_id_not_rehashed')}
        references.append(dict(root=root, baseline=str(baseline), original_args=original,
                               checkpoint=manifest['checkpoint']))
    cls = framework_class()
    # Freeze public numeric/string controller constants without inventing a second protocol.
    controller = {variant: cls(mode='framework', variant=variant).config for variant in COHORTS[1:]}
    cells = cohort_cells(references)
    plan = dict(schema=3, description=DESCRIPTION, limits=LIMITS, cells=cells,
                 references=references, source_sha256=pins, weights=weights, fusion=fusion, joint=joint,
                 controller_constants=controller, python=PYTHON, created=time.time(),
                 study=dict(name='native_prompt_refine_20261021', development_only=True,
                     primary='native full-task success for root4 and root0; regression checks on others',
                     factors=['contain_release_prompt_refinement'],
                     secondary=['paired rescues and harms', 'local recovery vs verified exit',
                         'actual nominal handoff', 'post-handoff regression and full completion',
                         'repair/query/action/time costs'],
                     sentinel_roots=dict(root4='prompt_only rescued it', root0='failed in all previous cohorts', root3='must not regress'),
                     analysis='compare against prompt_only baseline; six reused development roots, no held-out or population claim',
                     stopping='complete all thirty-six cells regardless of root4 outcome; no efficacy early stopping',
                     recovery='retain all cases; exact replay; 30 baseline/variant full videos; publish and fresh remote recovery'),
                 success='native env.check_success(): original In & Close',
                wall_scope='scene setup + ten noops + rollout, excluding actor load',
                 schedule='per root: baseline/v2, prompt_only/gate_only, combined; each wave then fresh replays; lanes alternate by root',
                invalid_policy='retain all cells; replay image mismatches remain failures')
    if repair_candidates:
        require(joint is not None, 'Repair candidates require pinned joint verification')
        from src.repair.candidate_pool import CONFIG
        plan['repair_candidates'] = deepcopy(CONFIG)
    if training_data:
        require(repair_candidates, 'Training collection requires the pinned candidate protocol')
        # CPU-only official state selection, no policy/model construction. Full
        # file is hash-pinned; only designated training rows are inspected.
        import numpy as np
        import torch
        reference = load(Path(references[0]['baseline'])/'initial_source.json')
        states = torch.load(reference['path'], weights_only=False, map_location='cpu')
        initial_sources = {}
        for root in TRAINING_ROOTS:
            chosen = np.asarray(states[root])
            initial_sources[str(root)] = dict(reference, index=root,
                selected_state_sha256=hashlib.sha256(np.ascontiguousarray(chosen).tobytes()).hexdigest(),
                selected_state_shape=list(chosen.shape), selected_state_dtype=str(chosen.dtype))
        plan.update(schema=4, training_data=True, cells=training_cells(references), limits=TRAINING_LIMITS,
            training_initial_sources=initial_sources,
            description='Training-only independent initial-state expansion of verification/repair candidate data',
            study=dict(name='repair_training_roots10_19', data_split='training',
                roots=list(TRAINING_ROOTS), calibration_roots=list(range(6,10)),
                final_confirmation_roots=list(range(20,50)),
                primary='Complete native task outcome; collection is not held-out evaluation',
                stopping='All ten baseline/greedy-framework pairs; fixed candidate siblings only when first repair exists'))
    directory.mkdir(parents=True)
    for name in SOURCES:
        target = directory/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO/name).read_bytes())
    save(directory/'plan.json', plan)
    print(json.dumps(dict(plan=str(directory/'plan.json'), sha256=sha(directory/'plan.json'), cells=len(plan['cells']))))
    return 0


def check_plan(plan, check_files=True):
    require(plan['limits'] == (TRAINING_LIMITS if plan.get('training_data') else LIMITS), 'Frozen limits changed')
    if plan.get('repair_candidates') is not None:
        if str(REPO) not in sys.path:
            sys.path.append(str(REPO))
        from src.repair.candidate_pool import CONFIG
        require(plan['repair_candidates'] == CONFIG, 'Candidate protocol drift')
    if plan.get('training_data'):
        require(plan.get('schema') == 4 and plan.get('repair_candidates') is not None, 'Invalid training plan')
        require(plan['cells'] == training_cells(plan['references']), 'Training cells/splits/lanes changed')
        require(set(plan['training_initial_sources']) == {str(r) for r in TRAINING_ROOTS}, 'Training split drift')
        for root, initial in plan['training_initial_sources'].items():
            require(initial['index'] == int(root) and
                plan['source_sha256'].get(initial['path']) == initial['sha256'], 'Initial-state provenance drift')
    else:
        require([(c['root'], c['cohort']) for c in plan['cells']] ==
                [(r, a) for r in range(6) for a in COHORTS], 'Expected thirty-six ordered cells')
        for cell in plan['cells']:
            require(cell['name'] == f"{cell['cohort']}_root{cell['root']}" and
                    cell['lane'] == LANES[(COHORTS.index(cell['cohort'])+cell['root'])%2], 'Cell identity drift')
            require(cell['arm'] == ('baseline' if cell['cohort'] == 'baseline' else 'framework')
                and cell['variant'] == ('v2' if cell['cohort'] == 'baseline' else cell['cohort']), 'Treatment drift')
            require(cell['baseline'] == plan['references'][cell['root']]['baseline'], 'Reference drift')
    cls = framework_class()
    require(plan['controller_constants'] == {v: cls(mode='framework', variant=v).config for v in COHORTS[1:]},
            'Controller constants changed')
    if check_files:
        for path, expected in plan['source_sha256'].items():
            require(sha(path) == expected, f'Frozen file changed: {path}')
        for path, entry in plan['weights'].items():
            require(str(Path(path).resolve()) == entry['resolved_path'] and
                    Path(path).stat().st_size == entry['bytes'], f'Weight blob changed: {path}')


def validate_decision(decision, arm, language):
    require(isinstance(decision, dict) and decision['source'] == 'greedy' and
            decision['seed'] == 0, 'Only original greedy seed-zero proposals allowed')
    require(all(k in decision for k in ('language', 'owner', 'mode', 'subgoal',
                                        'trigger', 'evidence', 'repair_id')), 'Incomplete decision')
    require(isinstance(decision['language'], str) and decision['language'].strip(), 'Empty language')
    if arm == 'baseline':
        require(decision['language'] == language, 'Baseline instruction changed')


def is_repair(decision):
    return 'repair' in str(decision.get('owner', '')).lower() or decision.get('mode') == 'repair'


def controllers(arm, language, variant='v2', fusion_mode='off', joint_mode=False, recovery_options=False,
                joint_admission='all', repair_control='existing'):
    cls = framework_class()
    return cls(mode=arm, language=language, variant=variant, fusion_mode=fusion_mode, joint_mode=joint_mode,
               recovery_options=recovery_options, joint_admission=joint_admission, repair_control=repair_control), cls(
        mode='framework' if arm == 'baseline' else 'baseline', language=language, variant=variant)


def runtime(cell):
    from run_native_long_branch import frozen_runner
    return frozen_runner(Path(cell['baseline']))


def live_region(env, config):
    """Official region box and live orientation; no invented receptacle bounds."""
    import numpy as np
    site = env.sim.model.site_name2id(config['region'])
    require(site >= 0, 'Declared task region site is absent')
    size = np.asarray(env.sim.model.site_size[site], dtype=float)
    require(size.shape == (3,) and np.all(size > 0), 'Region must be a nondegenerate box')
    return dict(name=config['region'], source='official_simulator_region_site_box',
        center=np.asarray(env.sim.data.site_xpos[site]).tolist(),
        rotation=np.asarray(env.sim.data.site_xmat[site]).reshape(3, 3).tolist(), half_size=size.tolist())


def episode(args):
    import numpy as np
    from run_native_long_branch import (CandidateActor, validate_decode, validate_action,
        validate_model_sources, controller_fields, WorkerTimeout, InvalidGeneration)
    plan = load(args.plan)
    check_plan(plan)
    if args.fusion_mode != 'off':
        require(args.mlp_path is not None and plan.get('fusion') is not None and
                str(args.mlp_path.resolve()) == plan['fusion']['model'], 'Fusion model must be pinned in plan')
    if args.joint_components is not None:
        require(args.fusion_mode == 'gate' and plan.get('joint') is not None and
            str(args.joint_components.resolve()) == plan['joint']['artifact'],
            'Joint mode needs pinned components and gate mode')
    require(not args.recovery_options or args.joint_components is not None,
            'Recovery options require pinned joint components')
    require(args.joint_admission == 'all' or (args.joint_components is not None and not args.recovery_options),
            'Admission ablation requires joint unchanged-policy execution')
    require(args.repair_control == 'existing' or (args.joint_components is not None and not args.recovery_options),
            'Contract repair control requires joint unchanged-policy execution')
    candidate_enabled = args.candidate_slot is not None
    require(not candidate_enabled or (plan.get('repair_candidates') is not None and
        args.joint_components is not None and args.joint_admission == 'measured_only' and
        args.repair_control == 'existing' and not args.recovery_options),
        'Candidate collection requires pinned measured-admission existing repair with no action edits')
    query_cap = plan['repair_candidates']['total_query_cap'] if candidate_enabled or plan.get('training_data') else 520
    cell = next(c for c in plan['cells'] if c['name'] == args.cell)
    runner, baseline_manifest = runtime(cell)
    from adapters.libero_openvla import policy_image
    original = argparse.Namespace(**load(Path(cell['baseline'])/'status.json')['args'])
    if plan.get('training_data'):
        original.init_index = cell['root']
    for key in ('gpu', 'egl_device'):
        setattr(original, key, cell['lane'][key])
    original.out = str(args.out.resolve())
    original.readiness_seconds, original.query_seconds = 180, 30
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out/'plan.json', plan)
    status = dict(status='started', cell=cell, args=vars(original), started=time.time(),
                  actor_queries=0, steps_executed=0, initialization_actions=0,
                  description=DESCRIPTION,
                  action_protocol='unchanged_vla',
                  fusion_mode=args.fusion_mode,
                  joint_mode=args.joint_components is not None,
                  recovery_options=args.recovery_options,
                  joint_admission=args.joint_admission,
                  repair_control=args.repair_control,
                   actor_query_cap=query_cap,
                  repair_protocol='frozen_policy_h1_remaining_execution_v1' if args.joint_components else None,
                  plan_sha256=sha(args.plan), pid=os.getpid())
    save(out/'status.json', status)
    if candidate_enabled:
        status.update(candidate_protocol=plan['repair_candidates'], candidate_slot=args.candidate_slot,
            candidate_pools=0, candidate_extra_queries=0, actor_query_cap=query_cap,
            query_accounting='controller counts executed H1 requests; actor_queries includes every candidate',
            action_protocol='selected_unchanged_vla_candidate_v1')
    if args.joint_components is not None:
        status['description'] = ('Joint real-vision/physical/prediction/trained-risk/episodic-memory controller; '
            'frozen-policy H1 repair and fusion/risk/prediction handoff veto; no numeric offsets.')
    actor = env = recorder = tape = settle = None
    handles, executed, control_rows = [], [], []
    initial = episode_start = active = shadow = None
    last_decision = None
    try:
        base = runner.local_environment(original, out)
        identity = load(out/'device_identity.json')
        require(identity['physical_uuid'] == cell['lane']['physical_uuid'] and
                identity['cuda_visible_devices'] == identity['physical_uuid'], 'UUID mask required')
        manifest = runner.freeze_sources(out, base)
        validate_model_sources(runner, original, out, manifest, baseline_manifest)
        for name in SOURCES:
            target = out/'source'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((REPO/name).read_bytes())
        status['phase'] = 'model_load'
        save(out/'status.json', status)
        start = time.monotonic()
        actor = CandidateActor.__new__(CandidateActor)
        CandidateActor.__init__(actor, runner, original, original, out, identity)
        status.update(actor_build=actor.build, model_load_s=time.monotonic()-start)
        runner.seed_all(0)
        episode_start = time.monotonic()
        with runner.deadline(LIMITS['rollout_seconds']):
            status['phase'] = 'scene_setup'
            save(out/'status.json', status)
            env, obs = runner.make_scene(original, out, base)
            runner.record_scene_sources(env, out, manifest)
            expected_initial = (plan['training_initial_sources'][str(cell['root'])] if plan.get('training_data')
                else load(Path(cell['baseline'])/'initial_source.json'))
            require(load(out/'initial_source.json') == expected_initial,
                    'Official initial state or scene changed')
            require(sha(out/'model.xml') == baseline_manifest['model_xml_sha256'], 'Compiled scene drift')
            diag = runner.Diagnostics(env)
            language = env.language_instruction
            active, shadow = controllers(cell['arm'], language, cell['variant'], args.fusion_mode,
                                         args.joint_components is not None, args.recovery_options, args.joint_admission,
                                         args.repair_control)
            status.update(language=language, goal_clauses=diag.clauses)

            fusion_verifier = None
            joint_runtime = joint_evidence = suffix = None
            recovery = None
            if args.joint_components is not None:
                sys.path.insert(0, str(REPO))
                from src.verification.joint_runtime import JointRuntime
                from src.repair.receding_horizon_repair import RecedingHorizonRepair
                joint_runtime = JointRuntime(args.mlp_path, args.joint_components, device='cuda:0')
                fusion_verifier = joint_runtime.handoff
                suffix = RecedingHorizonRepair()
                status['joint_components'] = plan['joint']
                if args.recovery_options:
                    from src.repair.recovery_options import RecoveryOptions, SCHEMA
                    recovery = RecoveryOptions(active.config, controller_fields(env))
                    status.update(action_protocol=SCHEMA,
                        description='Joint runtime with oracle-state recovery skills, progress early stopping and cross-trigger option failure memory; matched unused VLA query every step.')
            elif args.fusion_mode != 'off':
                sys.path.insert(0, str(REPO))
                from src.verification.handoff_verifier import HandoffVerifier
                fusion_verifier = HandoffVerifier(args.mlp_path, device='cuda:0')
            if fusion_verifier is not None:
                status['fusion_model'] = dict(path=str(args.mlp_path.resolve()),
                    sha256=fusion_verifier.model_sha256, threshold=fusion_verifier.threshold,
                    encoder=fusion_verifier.encoder.provenance)
            recorder = runner.ContactRecorder(env.sim, keywords=(*diag.relevant, 'gripper'))
            settle = runner.Tape(out/'initialization')
            settle.capture(env, obs, diag.measure(obs))
            stream = (out/'initialization/steps.jsonl').open('w')
            handles.append(stream)
            status['phase'] = 'initialization'
            for step in range(10):
                obs, _ = runner.step_once(env, obs, runner.NOOP, diag, recorder, settle,
                                          stream, step, 'initialization')
                status['initialization_actions'] += 1
            initial = diag.measure(obs)
            tape = runner.Tape(out)
            tape.capture(env, obs, initial)
            save(out/'initial_controller.json', controller_fields(env))
            stream, queries, controls = [(out/name).open('w') for name in
                ('steps.jsonl', 'queries.jsonl', 'controller.jsonl')]
            handles.extend((stream, queries, controls))
            (out/'model_inputs').mkdir()
            if joint_runtime is not None:
                (out/'joint_inputs').mkdir()
            status['phase'] = 'actor'
            vision_encoder = recovery_scorer = handoff_scorer = scorer_modules = None
            physics_verifier = None
            if args.recovery_scorer is not None or args.handoff_scorer is not None:
                scorer_modules = _load_scorer_modules()
                vision_encoder = scorer_modules['FrozenVisualEncoder'](
                    args.encoder_checkpoint, args.encoder_source, device='cuda:0',
                    expected_sha=args.encoder_sha)
                physics_verifier = scorer_modules['PhysicalVerifier']()
                if args.recovery_scorer is not None:
                    recovery_scorer = scorer_modules['RecoveryScorer'](args.recovery_scorer, device='cuda:0')
                if args.handoff_scorer is not None:
                    handoff_scorer = scorer_modules['HandoffScorer'](args.handoff_scorer, device='cuda:0')
                status['scorer_modules'] = dict(
                    encoder=vision_encoder.provenance,
                    recovery_scorer=recovery_scorer.metadata if recovery_scorer else None,
                    handoff_scorer=handoff_scorer.metadata if handoff_scorer else None,
                    physics_verifier=dict(
                        contact_threshold_n=physics_verifier.contact_threshold_n,
                        grasp_force_threshold_n=physics_verifier.grasp_force_threshold_n,
                        support_margin_m=physics_verifier.support_margin_m,
                    ))
            if initial['native_success']:
                status['termination'] = 'native_success'
            else:
                for step in range(520):
                    measured = tape.last['diagnostics']

                    if joint_runtime is not None and joint_evidence is None:
                        joint_image = policy_image(obs['agentview_image'])
                        joint_image.save(out/'joint_inputs'/f'{step:06d}.png')
                        joint_evidence = joint_runtime.observe(np.asarray(joint_image),
                            active.preview_fusion_state(measured), step, live_region(env, active.config))
                    if recovery is not None:
                        joint_evidence['recovery_admission'] = recovery.admission(
                            active.preview_fusion_state(measured), step)
                    decision = active.before(step, deepcopy(measured), joint_evidence=joint_evidence)
                    shadow_decision = shadow.before(step, deepcopy(measured))
                    validate_decision(decision, cell['arm'], language)
                    validate_decision(shadow_decision,
                        'framework' if cell['arm'] == 'baseline' else 'baseline', language)
                    buffer = io.BytesIO()
                    image = policy_image(obs['agentview_image'])
                    image.save(buffer, format='PNG')
                    png = buffer.getvalue()
                    (out/'model_inputs'/f'{step:06d}.png').write_bytes(png)
                    metadata = dict(step=step, query_index=step, language=decision['language'],
                        source='greedy', seed=0, before=tape.last['fingerprint'],
                        policy_png_sha256=hashlib.sha256(png).hexdigest(),
                        policy_rgb_sha256=hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest())
                    query_request = dict(language=decision['language'], source='greedy', seed=0)
                    repair_replanning = None
                    if suffix is not None and recovery is None:
                        query_request, repair_replanning = suffix.request(decision)
                        metadata['repair_replanning'] = repair_replanning
                    before_controller = controller_fields(env)
                    runner.write_row(controls, dict(event='before', step=step, decision=decision,
                        shadow_decision=shadow_decision, diagnostics=measured,
                        query_budget=dict(used=status['actor_queries'], remaining=query_cap-status['actor_queries'])))
                    probe = candidate_enabled and decision['owner'] == 'repair' and status['candidate_pools'] == 0
                    requests = [query_request]
                    scoring_context = selection = None
                    vision_feat = physics_feat = None
                    evidence_list = None
                    if probe:
                        from src.repair import candidate_pool
                        requests = candidate_pool.requests(decision, cell['root'], step)
                        post_state = active.preview_fusion_state(measured)
                        scoring_context = candidate_pool.context(decision,
                            post_state, metadata['policy_png_sha256'],
                            [r['action'] for r in executed[-6:]])
                        if recovery_scorer is not None:
                            img_arr = np.ascontiguousarray(image)
                            vision_feat = vision_encoder.encode(img_arr)
                            physics_feat = scorer_modules['extract_physics_features'](post_state)
                        if repair_replanning is not None:
                            repair_replanning = dict(repair_replanning,
                                additional_queries=len(requests)-1,
                                candidate_protocol=plan['repair_candidates']['schema'])
                            metadata['repair_replanning'] = repair_replanning
                    require(status['actor_queries']+len(requests) <= query_cap, 'Actor query budget exhausted')
                    candidates, inference_s = [], 0.
                    for request in requests:
                        query_index = status['actor_queries']
                        query_metadata = dict(metadata, query_index=query_index,
                            source=request['source'], seed=request['seed'])
                        status['actor_queries'] += 1
                        runner.write_row(queries, dict(event='requested', **query_metadata))
                        save(out/'status.json', status)
                        start = time.monotonic()
                        decode = actor.query(png, request['language'], request['source'], request['seed'], query_index)
                        elapsed = time.monotonic()-start
                        inference_s += elapsed
                        runner.write_row(queries, dict(event='returned', **query_metadata,
                            decode=decode, inference_s=elapsed))
                        validate_decode(decode, query_metadata)
                        if not decode.get('valid'):
                            raise InvalidGeneration(str(decode.get('invalid_reason')))
                        candidates.append(dict(candidate_id=len(candidates), metadata=query_metadata, decode=decode))
                    if probe:
                        if recovery_scorer is not None:
                            # Measured evidence is common to all unexecuted candidates.
                            evidence_list = []
                            for c in candidates:
                                ev = scorer_modules['extract_physics_evidence'](
                                    measured, active.config['target'],
                                    verifier=physics_verifier)
                                evidence_list.append(ev)
                            decode, selection = candidate_pool.select(
                                candidates, scoring_context,
                                scorer=recovery_scorer,
                                vision_feat=vision_feat, physics_feat=physics_feat,
                                evidence_list=evidence_list)
                        else:
                            decode, selection = candidate_pool.select(
                                candidates, scoring_context, collection_slot=args.candidate_slot)
                        status['candidate_pools'] += 1
                        status['candidate_extra_queries'] += len(requests)-1
                    extra = dict(proposal=decode['action'], native_proposal=decode['native_action'],
                        tokens=decode['tokens'], decode=decode, inference_s=inference_s,
                        decision=decision, shadow_decision=shadow_decision,
                        action_owner=decision['owner'], controller_before=before_controller,
                        episode_elapsed_s=time.monotonic()-episode_start)
                    if selection is not None:
                        extra['candidate_selection'] = selection
                    if repair_replanning is not None:
                        extra['repair_replanning'] = repair_replanning
                    executed_action = list(decode['action'])
                    action_modified = False
                    correction_reason = None
                    if recovery is not None:
                        executed_action, skill_record = recovery.command(decision, measured, decode['action'])
                        if skill_record is not None:
                            extra['recovery_action'] = skill_record
                            extra['original_action'] = list(decode['action'])
                            extra['executed_action_source'] = 'oracle_closed_loop_recovery_skill'
                            action_modified = executed_action != list(decode['action'])
                        else:
                            extra['executed_action_source'] = 'frozen_vla'
                    extra['action_modified'] = action_modified
                    extra['correction_reason'] = correction_reason
                    extra['action_protocol'] = status['action_protocol']
                    extra['correction_delta'] = (np.asarray(executed_action) -
                                                 np.asarray(decode['action'])).tolist()
                    status['numeric_correction_actions'] = 0
                    if recovery is not None:
                        status['oracle_skill_actions'] = status.get('oracle_skill_actions', 0)+int(extra.get('recovery_action') is not None)

                    validate_action(dict(step=step, action=executed_action,
                        executed_mapped_action=executed_action, **extra), step,
                         allow_correction=False, allow_recovery=args.recovery_options)
                    obs, row = runner.step_once(env, obs, executed_action, diag, recorder,
                                                tape, stream, step, 'actor', extra)
                    executed.append(row)
                    status['steps_executed'] = len(executed)

                    fusion_evidence = None
                    if joint_runtime is not None:
                        post_state = active.preview_fusion_state(row['diagnostics'])
                        joint_image = policy_image(obs['agentview_image'])
                        joint_image.save(out/'joint_inputs'/f'{step+1:06d}.png')
                        joint_evidence = joint_runtime.observe(np.asarray(joint_image),
                            post_state, step+1, live_region(env, active.config))
                        if decision['owner'] == 'repair':
                            fusion_evidence = joint_runtime.score_handoff(post_state, decision['recovery_goal'])
                        status['joint_counters'] = dict(joint_runtime.counters)
                    elif fusion_verifier is not None and decision['owner'] == 'repair':
                        fusion_evidence = fusion_verifier.score(policy_image(obs['agentview_image']),
                            active.preview_fusion_state(row['diagnostics']), decision['recovery_goal'])
                    handoff_evidence = dict(fusion_evidence or {})
                    if handoff_scorer is not None and decision['owner'] == 'repair':
                        post_state = active.preview_fusion_state(row['diagnostics'])
                        img_arr = np.ascontiguousarray(policy_image(obs['agentview_image']))
                        vf = vision_encoder.encode(img_arr)
                        pf = scorer_modules['extract_physics_features'](post_state)
                        # Physics evidence for handoff (after execution).
                        curr_pos = post_state.get('position')
                        prev_pos = active.preview_fusion_state(measured).get('position')
                        ev = scorer_modules['extract_physics_evidence'](
                            row['diagnostics'], active.config['target'],
                            executed_action=list(row['action']),
                            prev_pos=prev_pos, curr_pos=curr_pos,
                            verifier=physics_verifier)
                        handoff_result = handoff_scorer.score(vf, pf, ev)
                        handoff_evidence['handoff_scorer'] = handoff_result
                    recovery_feedback = None if recovery is None else recovery.observe(
                        row['diagnostics'], active.preview_fusion_state(row['diagnostics']))
                    if recovery is not None:
                        status['recovery_events'] = deepcopy(recovery.events)
                    verification = active.after(step, deepcopy(row['diagnostics']),
                                                list(row['action']), deepcopy(decision),
                                                 handoff_evidence=handoff_evidence, recovery_feedback=recovery_feedback)
                    if joint_runtime is not None:
                        verification['joint_after'] = deepcopy(joint_evidence)
                    shadow_verification = shadow.after(step, deepcopy(row['diagnostics']),
                        list(row['action']), deepcopy(shadow_decision))
                    handoff = bool(last_decision and is_repair(last_decision) and not is_repair(decision))
                    control = dict(event='after', step=step, decision=decision,
                        verification=verification, shadow_verification=shadow_verification,
                        controller_after=controller_fields(env), actual_handoff=handoff,
                        actual_handoff_definition='first executed nominal action following repair action',
                        query_budget=dict(used=status['actor_queries'], remaining=query_cap-status['actor_queries']))
                    if candidate_enabled:
                        control['recovery_dataset_state'] = active.preview_fusion_state(row['diagnostics'])
                    runner.write_row(controls, control)
                    control_rows.append(control)
                    last_decision = decision
                    save(out/'status.json', status)
                    if row['diagnostics']['native_success']:
                        status['termination'] = 'native_success'
                        break
                    require(not row['done'], 'Environment ended before native success/action cap')
            status.update(status='completed', termination=status.get('termination', 'action_budget'))
    except runner.WallBudget:
        status.update(status='budget_stopped', termination='rollout_budget',
                      interrupted_phase=status.get('phase'))
    except (WorkerTimeout, InvalidGeneration) as error:
        status.update(status='actor_failed', termination=type(error).__name__, error=str(error))
    except BaseException:
        status.update(status='infrastructure_failed', termination='infrastructure_failure',
                      error=traceback.format_exc())
    finally:
        if episode_start is not None:
            status['episode_wall_s'] = time.monotonic()-episode_start
        if initial is not None:
            status.update(runner.summarize(initial, executed))
        if active is not None:
            status.update(controller_summary=active.summary, shadow_summary=shadow.summary)
        exit_record = dict(event='episode_exit', termination=status.get('termination'),
            actions=len(executed), queries=status['actor_queries'], remaining=query_cap-status['actor_queries'],
            last_decision=last_decision, repair_active=bool(last_decision and is_repair(last_decision)),
            verification=control_rows[-1]['verification'] if control_rows else None,
            actual_handoffs=[r['step'] for r in control_rows if r['actual_handoff']])
        save(out/'exit.json', exit_record)
        operations = [('actor', lambda: runner.cleanup_actor(actor))]
        operations += [('stream', h.close) for h in handles]
        operations += [(name, obj.close) for name, obj in
            (('tape', tape), ('initialization', settle), ('contacts', recorder), ('scene', env))
            if obj is not None]
        for name, operation in operations:
            try:
                operation()
            except BaseException:
                status.setdefault('cleanup_errors', []).append(dict(resource=name, error=traceback.format_exc()))
                status['status'] = 'infrastructure_failed'
        np.savez_compressed(out/'actions.npz',
            mapped=np.asarray([r['action'] for r in executed], dtype=float).reshape(-1, 7),
            native=np.asarray([r['native_proposal'] for r in executed], dtype=float).reshape(-1, 7),
            tokens=np.asarray([r['tokens'] for r in executed], dtype=np.int64).reshape(-1, 7))
        status.update(ended=time.time(), total_wall_s=time.time()-status['started'])
        status['valid_episode'] = (status['status'] == 'completed' and
            status['actor_queries'] == status['steps_executed']+status.get('candidate_extra_queries', 0) and
            status['actor_queries'] <= query_cap and
            (status.get('termination') == 'native_success' or status['steps_executed'] == 520))
        save(out/'status.json', status)
    return 0 if status['valid_episode'] else 1


def replay(args):
    """Fresh official reset/noops; saved proposals only, strict images and controllers."""
    import numpy as np
    from run_native_long_branch import (zero_query_guard, evidence_tape, validate_action,
                                       validate_decode, controller_fields)
    source, out = args.replay.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status='started', source=str(source), started=time.time(), actor_queries=0,
        mismatches=[], frames_checked=0, max_state_gap=0., max_integration_gap=0.,
        actor_constructor_attempts=0, actor_query_attempts=0, subprocesses=[],
        worker_spawn_attempts=0, model_weight_open_attempts=0)
    env = recorder = None
    try:
        plan, status = load(source/'plan.json'), load(source/'status.json')
        check_plan(plan)
        runner, _ = runtime(status['cell'])
        zero_query_guard(runner, report)
        from adapters.libero_openvla import policy_image
        original = argparse.Namespace(**status['args'])
        if plan.get('training_data'):
            require(original.init_index == status['cell']['root'] and
                load(source/'initial_source.json') == plan['training_initial_sources'][str(original.init_index)],
                'Training episode differs from registered initial condition')
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == status['cell']['lane']['physical_uuid'],
                'Replay must use recorded UUID lane')
        with runner.deadline(300):
            base = runner.local_environment(original, out)
            manifest = load(source/'source_manifest.json')
            for name, entry in manifest['repository'].items():
                require(sha(source/'source'/name) == entry['sha256'], f'Replay source drift: {name}')
            for name in SOURCES:
                require(sha(source/'source'/name) == plan['source_sha256'][str(REPO/name)],
                        f'Controller/actor source drift: {name}')
            for section in ('runtime', 'assets', 'actor_code'):
                for path, expected in manifest.get(section, {}).items():
                    require(sha(path) == expected, f'Replay {section} drift: {path}')
            runner.seed_all(0)
            env, obs = runner.make_scene(original, out, base)
            require(sha(out/'model.xml') == manifest['model_xml_sha256'], 'Replay scene drift')
            require(load(out/'initial_source.json') == load(source/'initial_source.json'), 'Initial state drift')
            diag = runner.Diagnostics(env)
            active, shadow = controllers(status['cell']['arm'], env.language_instruction,
                                          status['cell']['variant'], status.get('fusion_mode', 'off'),
                                            status.get('joint_mode', False), status.get('recovery_options', False),
                                            status.get('joint_admission', 'all'), status.get('repair_control', 'existing'))
            recovery = None
            recorder = runner.ContactRecorder(env.sim, keywords=(*diag.relevant, 'gripper'))
            before_controls = {r['step']: r for r in rows(source/'controller.jsonl') if r['event'] == 'before'}
            after_controls = {r['step']: r for r in rows(source/'controller.jsonl') if r['event'] == 'after'}
            query_rows = rows(source/'queries.jsonl')
            returned_rows = [r for r in query_rows if r['event'] == 'returned']
            requested_rows = [r for r in query_rows if r['event'] == 'requested']
            require(len(returned_rows) == len(requested_rows) == status['actor_queries'], 'Missing query records')
            require([r['query_index'] for r in returned_rows] == list(range(len(returned_rows))),
                    'Nonsequential actor queries')
            for request, response in zip(requested_rows, returned_rows):
                require(all(response[k] == v for k, v in request.items() if k != 'event'), 'Query request drift')
                validate_decode(response['decode'], response)
            returned = {r['query_index']: r for r in returned_rows}
            query_cursor, candidate_pools = 0, 0
            candidate_enabled = status.get('candidate_protocol') is not None
            if candidate_enabled:
                require(status['candidate_protocol'] == plan.get('repair_candidates'), 'Candidate plan drift')
            prior_decision = None
            for phase, directory in (('initialization', source/'initialization'), ('actor', source)):
                tape = evidence_tape(runner, directory, out)
                recorded = rows(directory/'steps.jsonl')
                require(len(tape.frames) == len(recorded)+1, 'Uncommitted frame/action')
                require(len(recorded) == 10 if phase == 'initialization' else len(recorded) <= 520,
                        'Action count violation')
                tape.compare(0, env, obs, diag.measure(obs), report, phase)
                if phase == 'actor':
                    require(controller_fields(env) == load(source/'initial_controller.json'), 'Initial controller drift')
                    if status.get('recovery_options'):
                        from src.repair.recovery_options import RecoveryOptions
                        recovery = RecoveryOptions(active.config, controller_fields(env))
                for index, row in enumerate(recorded):
                    validate_action(row, index, phase == 'initialization',
                         allow_correction=status.get('action_protocol') == 'bounded_translation_v1',
                         allow_recovery=status.get('recovery_options', False))
                    measured = diag.measure(obs)
                    before = runner.fingerprint(env, obs)
                    fields = []
                    if phase == 'actor':
                        require(not measured['native_success'], 'Action after native success')
                        joint_saved = before_controls[index]['decision']['evidence'].get('joint')
                        if recovery is not None:
                            joint_saved = deepcopy(joint_saved)
                            joint_saved['recovery_admission'] = recovery.admission(active.preview_fusion_state(measured), index)
                        decision = active.before(index, deepcopy(measured), joint_evidence=joint_saved)
                        shadow_decision = shadow.before(index, deepcopy(measured))
                        validate_decision(decision, status['cell']['arm'], env.language_instruction)
                        saved = before_controls[index]
                        # Replay the executed tape, not the untrained advisory models.
                        # This checks the native controller and dynamics, not monitor reproducibility.
                        if status.get('use_v5'):
                            report['monitor_evidence_recomputed'] = False
                            decision['v5_evidence'] = saved['decision']['v5_evidence']
                        for key, value in dict(decision=decision, shadow_decision=shadow_decision,
                                               diagnostics=measured).items():
                            if saved[key] != value:
                                fields.append(key)
                        if row['decision'] != decision or row['shadow_decision'] != shadow_decision:
                            fields.append('action_decisions')
                        if controller_fields(env) != row['controller_before']:
                            fields.append('controller_before')
                        probe = candidate_enabled and decision['owner'] == 'repair' and candidate_pools == 0
                        count = 1
                        query = returned[query_cursor]
                        if probe:
                            from src.repair import candidate_pool
                            requests = candidate_pool.requests(decision, status['cell']['root'], index)
                            candidates = []
                            for offset, request in enumerate(requests):
                                q = returned[query_cursor+offset]
                                require(q['step'] == index and all(q[k] == request[k]
                                    for k in ('source', 'seed', 'language')), 'Candidate request drift')
                                require(q['before'] == before and
                                    q['policy_png_sha256'] == query['policy_png_sha256'] and
                                    q['policy_rgb_sha256'] == query['policy_rgb_sha256'], 'Candidate context drift')
                                candidates.append(dict(candidate_id=offset,
                                    metadata={k: v for k, v in q.items() if k not in ('event', 'decode', 'inference_s')},
                                    decode=q['decode']))
                            scoring_context = candidate_pool.context(decision,
                                active.preview_fusion_state(measured), query['policy_png_sha256'],
                                [r['action'] for r in recorded[max(0,index-6):index]])
                            selected, selection = candidate_pool.select(candidates, scoring_context,
                                collection_slot=status['candidate_slot'])
                            require(selection == row.get('candidate_selection') and selected == row['decode'],
                                    'Candidate selection/provenance mismatch')
                            query = returned[selected['query_index']]
                            count = len(requests)
                            candidate_pools += 1
                        else:
                            require('candidate_selection' not in row, 'Candidate outside first repair')
                            require(query['source'] == 'greedy' and query['seed'] == 0,
                                    'Non-probe action must remain greedy')
                        require(query['language'] == decision['language'] and query['step'] == index,
                                'Query differs from causal decision')
                        query_cursor += count
                        validate_decode(row['decode'], query)
                        if recovery is not None:
                            skill_action, skill_record = recovery.command(decision, measured, row['proposal'])
                            require(skill_action == row['action'] and skill_record == row.get('recovery_action'),
                                    'Recomputed recovery skill differs from execution')
                        require(row['decode'] == query['decode'] and row['decode']['action'] == row['proposal'] and
                            row['decode']['native_action'] == row['native_proposal'] and
                            row['decode']['tokens'] == row['tokens'], 'Saved proposal/query mismatch')
                        require(sha(source/'model_inputs'/f'{index:06d}.png') == query['policy_png_sha256'],
                                'Model input PNG drift')
                        buffer = io.BytesIO()
                        policy_image(obs['agentview_image']).save(buffer, format='PNG')
                        if hashlib.sha256(buffer.getvalue()).hexdigest() != query['policy_png_sha256']:
                            fields.append('model_input_png')
                    recorder.reset()
                    obs, reward, done, _ = env.step(row['action'])
                    after = diag.measure(obs)
                    tape.compare(index+1, env, obs, after, report, phase)
                    for key, value in dict(before=before, after=runner.fingerprint(env, obs),
                        diagnostics=after, events=runner.transitions(measured, after), reward=float(reward),
                        done=bool(done), contacts_substeps=recorder.samples).items():
                        if row[key] != value:
                            fields.append(key)
                    if phase == 'actor':
                        fusion_saved = after_controls[index]['verification']['verification'].get('fusion', {})
                        verification = active.after(index, deepcopy(after), list(row['action']), deepcopy(decision),
                            handoff_evidence=fusion_saved.get('evidence'),
                            recovery_feedback=None if recovery is None else recovery.observe(after, active.preview_fusion_state(after)))
                        if status.get('fusion_mode', 'off') != 'off':
                            report['fusion_scores_recomputed'] = False
                        if status.get('joint_mode'):
                            verification['joint_after'] = after_controls[index]['verification']['joint_after']
                            report['joint_models_recomputed'] = False
                        shadow_verification = shadow.after(index, deepcopy(after), list(row['action']), deepcopy(shadow_decision))
                        expected = after_controls[index]
                        if candidate_enabled:
                            require(expected.get('recovery_dataset_state') == active.preview_fusion_state(after),
                                    'Recovery label state drift')
                        if status.get('use_v5'):
                            verification['v5'] = expected['verification']['v5']
                        for key, value in dict(verification=verification, shadow_verification=shadow_verification,
                            controller_after=controller_fields(env), actual_handoff=bool(prior_decision and
                            is_repair(prior_decision) and not is_repair(decision))).items():
                            if expected[key] != value:
                                fields.append(key)
                        prior_decision = decision
                    if fields:
                        report['mismatches'].append(dict(phase=phase, step=index, fields=fields))
                    save(out/'verification.json', report)
            with np.load(source/'actions.npz', allow_pickle=False) as actions:
                for name, key in (('mapped', 'action'), ('native', 'native_proposal'), ('tokens', 'tokens')):
                    require(np.array_equal(actions[name], np.asarray([r[key] for r in recorded]).reshape(-1, 7)),
                            f'Action archive drift: {name}')
            require(query_cursor == status['actor_queries'], 'Unaccounted model queries')
            require(candidate_pools == status.get('candidate_pools', 0), 'Candidate pool count drift')
            if candidate_enabled:
                require(status['candidate_extra_queries'] == candidate_pools*(plan['repair_candidates']['size']-1)
                    and status['actor_queries'] <= plan['repair_candidates']['total_query_cap'], 'Candidate budget drift')
            if active.summary != status.get('controller_summary') or shadow.summary != status.get('shadow_summary'):
                report['mismatches'].append(dict(phase='actor', fields=['controller_summary']))
            report.update(status='replay_mismatch' if report['mismatches'] else 'exact_replay_passed',
                          steps=len(recorded), original_status=status['status'])
    except BaseException:
        report.update(status='infrastructure_failed', error=traceback.format_exc())
    finally:
        for obj in (recorder, env):
            if obj is not None:
                try:
                    obj.close()
                except BaseException:
                    report.update(status='infrastructure_failed', cleanup_error=traceback.format_exc())
        report.update(ended=time.time(), total_wall_s=time.time()-report['started'])
        save(out/'verification.json', report)
    return 0 if report['status'] == 'exact_replay_passed' else 1


def terminate_group(process):
    # The actor inherits the episode's group, including when its parent has exited.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    end = time.monotonic()+5
    while time.monotonic() < end:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def launch(args):
    plan = load(args.plan)
    check_plan(plan)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out/'plan.json', plan)
    jobs = [dict(c, status='unstarted', replay_status='unstarted') for c in plan['cells']]
    state = dict(status='running', started=time.time(), pid=os.getpid(), plan_sha256=sha(args.plan), jobs=jobs)
    save(out/'launch.json', state)
    end = time.monotonic()+LIMITS['phase_seconds']
    running = []
    try:
        for pair in launch_waves(jobs):
            for replaying in (False, True):
                if time.monotonic() >= end:
                    break
                running = []
                for job in pair:
                    if replaying and not (out/job['name']/'status.json').exists():
                        job['replay_status'] = 'unavailable_episode'
                        continue
                    name = ('replay_' if replaying else '')+job['name']
                    command = [plan['python'], '-B', '-u', str(SCRIPT)]
                    if replaying:
                        command += ['--replay', str(out/job['name'])]
                    else:
                        command += ['--worker', '--plan', str(out/'plan.json'), '--cell', job['name']]
                    command += ['--out', str(out/name)]
                    log = (out/(name+'.log')).open('w')
                    try:
                        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, env=environment(job['lane']), start_new_session=True)
                    except BaseException:
                        log.close()
                        raise
                    key = 'replay_status' if replaying else 'status'
                    job.update({key: 'running', name+'_pid': process.pid})
                    running.append((process, log, job, key, time.monotonic()))
                save(out/'launch.json', state)
                cap = LIMITS['replay_seconds' if replaying else 'outer_seconds']
                while running:
                    for item in running[:]:
                        process, log, job, key, start = item
                        timed_out = time.monotonic()-start >= cap or time.monotonic() >= end
                        if process.poll() is None and not timed_out:
                            continue
                        terminate_group(process)
                        job[key] = 'budget_stopped' if timed_out else 'finished'
                        job[key+'_returncode'] = process.wait()
                        job[key+'_wall_s'] = time.monotonic()-start
                        log.close()
                        running.remove(item)
                        save(out/'launch.json', state)
                    if running:
                        time.sleep(.2)
            if time.monotonic() >= end:
                break
        state['status'] = 'finished' if time.monotonic() < end else 'phase_budget'
    except BaseException:
        state.update(status='interrupted', error=traceback.format_exc())
    finally:
        for process, log, job, key, start in running:
            terminate_group(process)
            log.close()
            job[key] = 'interrupted'
        for job in jobs:
            if job['status'] == 'unstarted':
                job['status'] = 'not_run_phase_stopped'
        state['ended'] = time.time()
        save(out/'launch.json', state)
    summarize(out)
    return 0 if state['status'] == 'finished' else 1


def cell_summary(run, cell):
    directory = run/cell['name']
    status = load(directory/'status.json') if (directory/'status.json').exists() else {}
    proof_path = run/('replay_'+cell['name'])/'verification.json'
    proof = load(proof_path) if proof_path.exists() else {}
    controls = rows(directory/'controller.jsonl')
    before = [r for r in controls if r['event'] == 'before']
    after = [r for r in controls if r['event'] == 'after']
    steps = rows(directory/'steps.jsonl')
    valid = bool(status.get('valid_episode') and proof.get('status') == 'exact_replay_passed')
    handoffs = [r['step'] for r in after if r['actual_handoff']]
    triggers = [dict(step=r['step'], trigger=r['decision']['trigger'], evidence=r['decision']['evidence'])
                for r in before if r['decision'].get('trigger')]
    repairs = status.get('controller_summary', {}).get('repairs', [])
    return dict(name=cell['name'], root=cell['root'], arm=cell['arm'],
        cohort=cell['cohort'], variant=cell['variant'], valid=valid,
        status=status.get('status', 'not_run'), replay=proof.get('status', 'not_run'),
        termination=status.get('termination'), native_success=bool(status.get('any_native_success')),
        scored_native_success=bool(valid and status.get('any_native_success')),
        queries=status.get('actor_queries', 0), actions=len(steps),
        model_load_s=status.get('model_load_s'), rollout_s=status.get('episode_wall_s'),
        total_s=status.get('total_wall_s'), replay_s=proof.get('total_wall_s'),
        triggers=triggers, repair_actions=sum(is_repair(r['decision']) for r in after),
        repair_transactions=repairs, repair_verified=sum(r['verified'] for r in repairs),
        local_recovery_transactions=sum(r.get('local_recovery_step') is not None for r in repairs),
        post_handoff_regressions=sum(r.get('post_handoff_goal_loss_step') is not None for r in repairs),
        verified_handoffs=sum(r['verified'] and r['handoff_step'] is not None for r in repairs),
        verified_handoff_completions=sum(r['verified'] and r['handoff_step'] is not None
            and r['completed_after_handoff'] for r in repairs),
        completed_during_repair=sum(r.get('completed_during_repair', False) for r in repairs),
        unverified_exits=sum(r['exit_step'] is not None and not r['verified'] for r in repairs),
        verification_records=[dict(step=r['step'], verification=r['verification']) for r in after
                              if is_repair(r['decision'])],
        actual_handoffs=handoffs, post_handoff_success=any(r['diagnostics']['native_success']
            and any(h <= r['step'] for h in handoffs) for r in steps))


def render_pair(run, root, summaries):
    """Full endpoint tapes at native 20 Hz, with online-only causal overlays."""
    import numpy as np
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    import textwrap
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    panels = []
    for summary in summaries:
        directory = run/summary['name']
        frames = sorted((directory/'frames').glob('*.npz'))
        controls = {r['step']: r for r in rows(directory/'controller.jsonl') if r['event'] == 'after'}
        panels.append((directory, frames, controls, summary))
    count = max(1, *(len(p[1]) for p in panels))
    path = run/'report'/f"root{root}_{summaries[0]['cohort']}_left_{summaries[1]['cohort']}_right.mp4"
    with imageio.get_writer(path, fps=20, macro_block_size=16) as video:
        for tick in range(count):
            canvas = Image.new('RGB', (1280, 960), '#111827')
            draw = ImageDraw.Draw(canvas)
            for column, (directory, frames, controls, summary) in enumerate(panels):
                x = column*640
                endpoint = min(tick, len(frames)-1)
                if frames:
                    with np.load(frames[endpoint], allow_pickle=False) as frame:
                        camera = Image.fromarray(frame['agentview_image'][::-1, ::-1]).resize((512, 512))
                    canvas.paste(camera, (x+64, 48))
                held = tick >= len(frames)
                draw.text((x+16, 10), f"{summary['cohort'].upper()} | root {root} | sim {tick/20:.2f}s | frame {max(endpoint, 0)}", font=font)
                record = controls.get(endpoint-1, {})
                decision = record.get('decision', {})
                result = record.get('verification', {})
                verification = result.get('verification', {})
                events = result.get('events', [])
                evidence = decision.get('evidence', {})
                evidence_brief = {k: evidence.get(k) for k in ('evidence_through_step', 'first_grasp_step',
                    'in_streak', 'release_transition', 'short_motion_diameter_m')}
                text = [f"{'HOLD: '+str(summary['termination']) if held else 'native endpoint'} | queries {max(endpoint, 0)}/520",
                    f"stage={decision.get('mode', 'initial')} subgoal={decision.get('subgoal')} owner={decision.get('owner')}",
                    'trigger='+str(decision.get('trigger')),
                    'evidence='+json.dumps(evidence_brief, sort_keys=True),
                    'recovery='+str(decision.get('recovery_goal'))+' accepts='+str(decision.get('recovery_acceptance', [])),
                    'blocked='+str(decision.get('intervention_blocked')),
                    f"local {decision.get('local_used', 0)}/32 | repair total {result.get('budget', {}).get('repair_actions', 0)}/160",
                    f"repair_verified={verification.get('repair_verified', False)} rechecks={verification.get('consecutive', 0)}/3",
                    'resumption_required='+str(verification.get('resumption_required'))+' ready='+str(verification.get('exit_condition_met')),
                    'exit='+str(next((e.get('reason') for e in events if e['type'] == 'local_exit'), None)),
                    'actual handoff (executed nominal action)='+str(record.get('actual_handoff', False))]
                y = 574
                for line in text:
                    # Full details remain in the per-step logs; overlays never use later decisions.
                    wrapped = textwrap.wrap(line, 76)[:3]
                    for part in wrapped:
                        draw.text((x+12, y), part, font=font, fill='#e5e7eb')
                        y += 18
                    if y > 920:
                        break
            draw.text((18, 938), '20 fps = native simulator time | privileged measured verification, no learned predictor | all numeric actions: frozen VLA', font=font)
            video.append_data(np.asarray(canvas))
    return str(path)


def summarize(run):
    run = Path(run).resolve()
    plan = load(run/'plan.json')
    check_plan(plan, check_files=False)
    (run/'report').mkdir(exist_ok=True)
    cells = [cell_summary(run, cell) for cell in plan['cells']]
    pairs = []
    for root in range(6):
        selected = [c for c in cells if c['root'] == root]
        left = selected[0]
        for right in selected[1:]:
            valid = left['valid'] and right['valid']
            pairs.append(dict(root=root, cohort=right['cohort'], valid=valid,
                rescue=bool(valid and not left['native_success'] and right['native_success']),
                harm=bool(valid and left['native_success'] and not right['native_success']),
                video=render_pair(run, root, [left, right])))
    aggregates = {}
    for arm in COHORTS:
        selected = [c for c in cells if c['cohort'] == arm]
        valid = [c for c in selected if c['valid']]
        successes = sum(c['scored_native_success'] for c in selected)
        aggregates[arm] = dict(planned=6, valid=len(valid), invalid=6-len(valid),
            native_successes=successes, native_failures=sum(not c['native_success'] for c in valid),
            native_SR_valid=successes/len(valid) if valid else None,
            native_SR_full_six=successes/6 if len(valid) == 6 else None,
            observed_native_successes=sum(c['native_success'] for c in selected),
            trigger_coverage_roots=sum(bool(c['triggers']) for c in selected),
            repair_verified=sum(c['repair_verified'] for c in selected),
            verified_handoffs=sum(c['verified_handoffs'] for c in selected),
            verified_handoff_completion_roots=sum(c['verified_handoff_completions'] > 0 for c in valid),
            completed_during_repair_roots=sum(c['completed_during_repair'] > 0 for c in valid),
            unverified_exits=sum(c['unverified_exits'] for c in selected),
            actual_handoff_roots=sum(bool(c['actual_handoffs']) for c in selected),
            post_handoff_success_roots=sum(c['post_handoff_success'] for c in selected),
            local_recovery_transactions=sum(c['local_recovery_transactions'] for c in selected),
            post_handoff_regressions=sum(c['post_handoff_regressions'] for c in selected),
            rescues=sum(p['rescue'] for p in pairs if p['cohort'] == arm),
            harms=sum(p['harm'] for p in pairs if p['cohort'] == arm),
            queries=sum(c['queries'] for c in selected), actions=sum(c['actions'] for c in selected))
    contrasts = []
    for root in range(6):
        selected = {c['cohort']: c for c in cells if c['root'] == root}
        valid = all(selected[v]['valid'] for v in COHORTS[1:])
        outcomes = {v: int(selected[v]['native_success']) for v in COHORTS[1:]}
        contrasts.append(dict(root=root, valid=valid, outcomes=outcomes,
            prompt_effect_gate_off=outcomes['prompt_only']-outcomes['v2'] if valid else None,
            prompt_effect_gate_on=outcomes['combined']-outcomes['gate_only'] if valid else None,
            gate_effect_prompt_off=outcomes['gate_only']-outcomes['v2'] if valid else None,
            gate_effect_prompt_on=outcomes['combined']-outcomes['prompt_only'] if valid else None,
            interaction=outcomes['combined']-outcomes['prompt_only']-outcomes['gate_only']+outcomes['v2'] if valid else None))
    report = dict(description=DESCRIPTION, cells=cells, pairs=pairs, arms=aggregates,
        factorial_contrasts=contrasts,
        rescues=sum(p['rescue'] for p in pairs), harms=sum(p['harm'] for p in pairs),
        note='Invalid/incomplete/replay-mismatched cells are retained, never counted as native failures.',
        video_semantics='baseline left; named framework variant right; all 24 comparisons; full tapes; 20fps native sim time')
    save(run/'report/report.json', report)
    lines = ['# Native microwave: full-task resumption factorial', '', DESCRIPTION, '',
             '| Root | Arm | Valid | Native success observed | Actions / queries | Exit | Replay |',
             '|---|---|---|---|---|---|---|']
    for c in cells:
        lines.append(f"| {c['root']} | {c['cohort']} | {c['valid']} | {c['native_success']} | {c['actions']} / {c['queries']} | {c['termination']} | {c['replay']} |")
    lines += ['', f"Across four baseline/variant contrasts (not independent roots): rescues {report['rescues']}; harms {report['harms']}.", '', report['note'], '',
              'Full SR, trigger evidence, measured verification, actual handoffs, post-handoff success and all timing are in `report.json`.']
    (run/'report/report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(run/'report/report.json'), arms=aggregates)))
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', type=Path, metavar='DIR')
    mode.add_argument('--launch', action='store_true')
    mode.add_argument('--replay', type=Path, metavar='EPISODE')
    mode.add_argument('--summarize', type=Path, metavar='RUN')
    mode.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--out', type=Path, help='New output directory')
    parser.add_argument('--cell', help=argparse.SUPPRESS)
    parser.add_argument('--recovery-options', action='store_true',
                        help='Oracle-state closed-loop recovery skills; NOT unchanged VLA actions')
    parser.add_argument('--mlp-path', type=Path, metavar='PT',
                        help='Versioned MLP artifact; pass at preparation and worker time')
    parser.add_argument('--fusion-mode', choices=('off', 'log', 'gate'), default='off',
                        help='Matched ablation: off, score-only, or conservative verified-exit gate')
    parser.add_argument('--joint-components', type=Path,
                        help='Pinned trained risk + episodic-stage artifact; activates the joint causal runtime')
    parser.add_argument('--joint-admission', choices=('all', 'measured_only'), default='all',
                        help='Only changes repair entry: joint-only anomalies stay advisory in measured_only mode')
    parser.add_argument('--repair-control', choices=('existing', 'measured_stage', 'measured_stage_progress'),
                         default='existing', help='Isolate measured-stage requests and progress-based unverified exits')
    parser.add_argument('--repair-candidates', action='store_true',
                        help='Pin repair-only candidate coverage protocol during preparation')
    parser.add_argument('--training-data', action='store_true',
                        help='Prepare only reserved training roots10..19; requires --repair-candidates')
    parser.add_argument('--candidate-slot', type=int, choices=range(4),
                        help='Worker-only first-repair fixed slot for executed recovery-data collection')
    parser.add_argument('--recovery-scorer', type=Path, metavar='PT',
                        help='Trained recovery action scorer checkpoint (module A)')
    parser.add_argument('--handoff-scorer', type=Path, metavar='PT',
                        help='Trained handoff-readiness scorer checkpoint (module B)')
    parser.add_argument('--encoder-checkpoint', type=Path, metavar='PT',
                        help='DINOv2 checkpoint for scorer vision encoding')
    parser.add_argument('--encoder-source', type=Path, metavar='DIR',
                        help='DINOv2 source directory for scorer vision encoding')
    parser.add_argument('--encoder-sha', type=str, metavar='SHA',
                        help='Expected SHA256 of encoder checkpoint')
    args = parser.parse_args(argv)
    if (args.launch or args.worker) and not args.plan:
        parser.error('--plan is required')
    if (args.launch or args.worker or args.replay) and not args.out:
        parser.error('--out is required')
    if args.worker and not args.cell:
        parser.error('private worker requires --cell')
    if args.cell and not args.worker:
        parser.error('--cell is private to --worker')
    if args.joint_admission != 'all' and not args.worker:
        parser.error('--joint-admission is a worker-only contrast; replay uses the recorded mode')
    if args.repair_control != 'existing' and not args.worker:
        parser.error('--repair-control is a worker-only contrast; replay uses the recorded mode')
    if args.repair_candidates and not args.prepare:
        parser.error('--repair-candidates is preparation-only')
    if args.training_data and (not args.prepare or not args.repair_candidates):
        parser.error('--training-data requires --prepare and --repair-candidates')
    if args.candidate_slot is not None and not args.worker:
        parser.error('--candidate-slot is worker-only')
    if (args.recovery_scorer is not None or args.handoff_scorer is not None) and not args.worker:
        parser.error('--recovery-scorer/--handoff-scorer are worker-only')
    if (args.recovery_scorer is not None or args.handoff_scorer is not None) and (
            args.encoder_checkpoint is None or args.encoder_source is None):
        parser.error('--recovery-scorer/--handoff-scorer require --encoder-checkpoint and --encoder-source')
    return args


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == '--fridge-probe':
        from run_fridge_repair_probe import main as fridge_main
        return fridge_main(argv[1:])
    args = parse_args(argv)
    if args.prepare:
        return prepare(args.prepare, args.mlp_path, args.joint_components, args.repair_candidates, args.training_data)
    if args.summarize:
        return summarize(args.summarize)
    if args.launch:
        return launch(args)
    if args.worker:
        return episode(args)
    # Public replay is also a detached-compatible bounded fresh child, on its recorded lane.
    status = load(args.replay/'status.json')
    os.environ.update(environment(status['cell']['lane']))
    return replay(args)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
