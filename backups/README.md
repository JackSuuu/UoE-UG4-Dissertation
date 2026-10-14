# Research recovery and videos

**Snapshot: 14 October 2026.** These robot videos show the **frozen full OpenVLA baseline**, with diagnostic overlays. There is **no physics-prediction action selector or repair enabled**. Worker 2 briefly places the book, then re-contacts it and fails the one-second released-placement criterion. Apparent visual corrections are the VLA's own actions.

**[Remote recovery verified](remote_recovery_verified_20261014.json):** fresh GitHub clone + independent LFS download + complete extraction recovered all 1,674 source files with matching SHA256. All 14 LFS objects (2.45 GB) and the pinned LIBERO submodule were retrieved successfully.

## View/download

- [All three roots, 2x simulated-time playback](20261014/videos/all_roots_2x.mp4)
- [Worker 0](20261014/videos/root_0_2x.mp4) · [Worker 1](20261014/videos/root_1_2x.mp4) · [Worker 2](20261014/videos/root_2_2x.mp4)
- [Pinned LIBERO source, assets and initial states](../third_party/LIBERO) — git submodule at `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
- [Artifact manifest](20261014/manifest.json) — archive and original-file SHA256, byte counts and exact recovery paths.
- [Environment/package inventory and checkpoint revisions](recovery_environment_20261014.json).

GitHub may present an LFS download link instead of an inline video player. Download the actual MP4; a small text file beginning `version https://git-lfs.github.com/spec/v1` is only a pointer and means LFS content has not been fetched.

## What is backed up

`20261014/archives/` stores Git LFS archives, not just server-local paths:

- `libero.tar.gz`: pilot, continuation, all raw videos/states/actions/substep contacts, compiled model XML, manifests, verified summaries, failure audits, annotated review, installation reports, replay/positive-fixture evidence and local upstream reference helpers.
- `legacy-results.tar.gz`: the complete existing `src/results/` experiment tree, including historical confirmation, R5, repair-transition, diagnostics, learned checkpoints and figures.
- `legacy-scratch.tar.gz`: other scratch scripts/logs, trained OpenVLA heads and intermediate artifacts.
- Separate archives for each HDF5 demonstration/feature/DAgger dataset, retaining original bytes.

Disposable `venv_probe`, Python bytecode and the duplicate scratch LIBERO clone are excluded. The pre-existing uncommitted planning-note edit is not part of this snapshot. The Git repository preserves experiment code/protocol history. LIBERO is recovered through its pinned submodule. Full upstream 7B model weights are **not mirrored here**: restore the pinned Hugging Face snapshots below. Thus model re-download still depends on upstream availability; this is an artifact/code backup, not a VM or offline full-model image.

## Restore on a replacement machine

Install Git and Git LFS, then:

```bash
git lfs install
git clone --recurse-submodules https://github.com/JackSuuu/UoE-UG4-Dissertation.git
cd UoE-UG4-Dissertation
git submodule update --init --recursive
git lfs pull
python scripts/backup_research_progress.py verify backups/20261014
python scripts/backup_research_progress.py restore backups/20261014 --destination ../recovered-20261014
```

The destination must not already exist. Restore verifies every archive/video and streams/checks every original file, reconstructing:

```text
recovered-20261014/
  scratch/                         # original research artifacts and data
  repository/src/results/          # historical result/checkpoint tree
```

No GPU or simulator is needed to recover results or watch videos. To resume experiments, copy the recovered results into your working repository and point scripts at the recovered scratch paths. Historical manifests contain original absolute paths; preserve them as evidence rather than rewriting them. The simulator adapter accepts `--root /path/to/repository/third_party/LIBERO` and generates its machine-local config. The original scripts' default `/home/ami/scratch/LIBERO` can also be recreated as a symlink to the submodule on a compatible layout.

## Restore model/runtime separately

Use a fresh environment matching the saved inventory. The active robot run used Python 3.13.5, torch 2.14.0+cu130, transformers 5.17.0, NumPy 2.3.4, MuJoCo 3.14.0, robosuite 1.4.1 and tensorflow-cpu 2.21.0. The inventory records installed versions; it is not a portable binary environment lock. Project-local ABI/decoding adapters are required; the official older transformers parity remains unverified.

With `huggingface_hub` installed, restore exactly these revisions:

```python
from huggingface_hub import snapshot_download
snapshot_download('openvla/openvla-7b',
                  revision='47a0ec7fc4ec123775a391911046cf33cf9ed83f')
snapshot_download('openvla/openvla-7b-finetuned-libero-10',
                  revision='80970322773f81baa2e22fe495d0487b93a05cfa')
```

The base snapshot supplies cached architecture/processor code for the task checkpoint. Keep the expected cache layout; `tf5_compat.snapshot_dir()` requires one base snapshot in its cache directory. Supply the downloaded task snapshot with `--snapshot`. Select a healthy GPU on the new machine; the historical GPU indices are not portable. EGL device selection must match the visible physical device. Re-run environment/renderer, strict-load, preprocessing and deterministic-prefix checks before treating a new rollout as comparable.

Saved `states.npz` contains simulator state, **not a complete branchable controller snapshot**. Exact branch restoration still needs deterministic action-prefix replay or a validated full internal-state restoration method.

## Interpretation and next experiment

The three development roots establish 1/3 released-placement completion, not framework efficacy or generalization. Native-ever success is separately 3/3. CVR/safe success remain unmeasured. See [the task-grounded comparison design](../exp_record/libero_framework_design_20261014.md).

First finish a matched VLA / simple-control / privileged-physics diagnostic on this task. Then learn only the useful correction mechanism, evaluate untouched roots and held-out disturbances, and freeze the method before testing a second task. A second task with extra training demonstrates adaptation, not zero-shot task generalization; report those separately.
