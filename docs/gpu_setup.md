# GPU-machine setup and stop gates

**Candidate recipe, not a validated environment lock.** Real-model GPU runs have
not been validated during this refactor. Use Python 3.10; do not install latest robosuite or overwrite
OpenVLA's transformers pin with LIBERO's old requirements. Core CPU tests need
none of these packages. Budget at least the BF16 7B model's memory plus inference
overhead; preferably put the planner on another GPU/server.

## 1. Clone and install

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules \
  https://github.com/JackSuuu/UoE-UG4-Dissertation.git
cd UoE-UG4-Dissertation
git submodule update --init --recursive
python -B -m unittest discover -s src/experiment/tests
conda create -n uoe-runtime python=3.10 -y
conda activate uoe-runtime
python -m pip install --upgrade 'pip<25' 'setuptools<70' wheel
python -m pip install torch==2.2.2 torchvision==0.17.2 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-runtime.txt
python -m pip install --no-deps -e third_party/LIBERO
python -m pip check
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__)'
```

Requires a compatible NVIDIA driver and EGL/OpenGL system libraries. Missing
system libraries need installation by the machine administrator. Repository
LFS archives/old scorer weights are **not** needed by the new runtime. LIBERO's
scene assets and initial states **are** needed; do not download demonstration
datasets just to evaluate the official scenes.

Set noninteractive, machine-local LIBERO paths **before its first import**:

```bash
export LIBERO_CONFIG_PATH="$PWD/.runtime/libero"
python - <<'PY'
import os
from pathlib import Path
import yaml
root = (Path.cwd()/'third_party/LIBERO/libero/libero').resolve()
config = Path(os.environ['LIBERO_CONFIG_PATH'])
config.mkdir(parents=True, exist_ok=True)
paths = dict(benchmark_root=str(root), bddl_files=str(root/'bddl_files'),
             init_states=str(root/'init_files'), assets=str(root/'assets'),
             datasets=str(root.parent/'datasets'))
for key in ('bddl_files', 'init_states', 'assets'):
    assert Path(paths[key]).is_dir(), paths[key]
# Never overwrite an existing machine-specific configuration silently.
with (config/'config.yaml').open('x') as stream:
    yaml.safe_dump(paths, stream)
PY
```

If that config already exists, inspect/reuse it rather than rerunning the writer.
Persist `LIBERO_CONFIG_PATH` in subsequent shells. The scene preflight below is
the actual check that meshes, selected initial states and rendering work; an
existing directory alone is not proof that all assets are installed.

## 2. Download OpenVLA to a normal explicit directory

```bash
export POLICY_DIR="$HOME/models/openvla-libero-10"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download('openvla/openvla-7b-finetuned-libero-10',
                  local_dir=os.environ['POLICY_DIR'], local_dir_use_symlinks=False)
PY
```

Do not invent an HF cache `snapshots/main` path. Pin a checkpoint commit using
`revision=` for a frozen study. Model loading uses trusted upstream remote code;
review/pin it as appropriate. The manifest records the resolved model revision
when available, but local-folder weights should also have a download revision
or checksums recorded in the experiment's environment notes.

## 3. Pretrained visual encoder (context/full arms)

```bash
git clone https://github.com/facebookresearch/dinov2.git "$HOME/models/dinov2-source"
curl --fail --location \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth \
  --output "$HOME/models/dinov2_vits14_pretrain.pth"
git -C "$HOME/models/dinov2-source" rev-parse HEAD
sha256sum "$HOME/models/dinov2_vits14_pretrain.pth"
```

Pin/reuse that source revision and checksum for every arm using vision. The
loader hashes both source and weights, freezes DINOv2, and rejects random or
dimension-padded features. Basic ViT-S/14 inference does not require xformers.
Use `--visual-source`, `--visual-checkpoint`, optional `--visual-sha256` and
`--feature-device cpu` (or an explicitly allocated GPU). Newly trained risk/A/B
heads and memory must match the encoder signature. Old stage weights are not
valid replacements. New head loaders/training have CPU synthetic-fixture tests;
no real-data trained artifact or demonstrated efficacy exists yet.

## 4. Vision planner

Use a separately managed vision-capable OpenAI-compatible server accepting image
data URLs and `response_format={"type":"json_object"}` at
`/v1/chat/completions`. Keep its serving dependencies out of the OpenVLA env.
Set `PLANNER_URL` to the base ending in `/v1`, `PLANNER_MODEL` to the exact served
ID, and optionally `PLANNER_API_KEY` for authentication. Do not put secrets in CLI
arguments/output manifests. This repository has not validated or selected a
planner checkpoint/server version; provisioning that service is a prerequisite,
not something a placeholder server command can guarantee. Record its checkpoint,
revision, serving version and GPU allocation with the run.

`GET /models` alone is insufficient. Run the **planner preflight** against the
actual LIBERO image. Inspect generated goals/bindings and measured true/false/
unknown values. Structural validation cannot prove the instruction was understood.

## 5. Run staged checks, then episodes

Use the copyable commands at the top of [Week 5](../exp_record/week5.md): scene →
planner → policy → components → one complete episode → paired development runs. Each output
directory must be new. Preflights execute **zero policy actions**, but all stages
perform the ten recorded simulator settling noops; policy preflight queries once
without executing its proposal. A wall deadline covers imports/setup/model load
and execution; a SIGALRM is not a hard kill for a wedged native/GPU call, so use
an external process timeout as an additional guard for unattended runs.

Save `python -m pip freeze`, `nvidia-smi`, repository/submodule commits and planner
provenance next to the output before treating any run as reproducible. Stop on
infrastructure errors rather than labeling them task failures. Independent
simulator replay and training/calibration on real generic recovery data are still pending.
