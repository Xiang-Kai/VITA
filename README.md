# VITA Reference

Independent research implementation of [VITA (arXiv:2511.19859v2)](https://arxiv.org/abs/2511.19859).
**Not official author code. No paper-level performance claim.**

This release provides a LIBERO-only training and evaluation recipe with the
paper's optimizer-update budgets, exact stage resume, full held-out offline
evaluation, action-policy export and real LIBERO/MuJoCo rollout integration.
No datasets, pretrained weights, historical logs or private deployment settings
are distributed.

## Training Budget

| Stage | Optimizer updates | Effective batch | Default microbatch | Accumulation |
| --- | ---: | ---: | ---: | ---: |
| Warmup | 200,000 | 1,024 | 16 | 64 |
| Co-training | 100,000 | 1,024 | 16 | 64 |
| LIBERO fine-tuning | 60,000 | 192 | 16 | 12 |
| Total | **360,000** | | | |

Sources: Table 11 (pretraining) and Section 7.3 (LIBERO fine-tuning).
These are **optimizer steps, not gradient-accumulation microsteps or epochs**.
Pretraining uses AdamW, LR `1e-4`, betas `(0.9, 0.95)`, weight decay `0.01`,
BF16. The fine-tuning LR and constant schedule are explicit implementation
choices. Fine-tuning updates only the action decoder.

**Scope:** all stages here reuse LIBERO training episodes, rather than the
paper's mixed video/action pretraining corpus. The default model is compact
and initialized from scratch. Matching budgets does not reproduce the paper's
model, compute cost or success rates. See [methodology](docs/methodology.md).

## Install

Python 3.10+; the reference CPU test stack is Python 3.11, PyTorch 2.5.1 and
Transformers 4.51.3. Install a PyTorch build appropriate for your platform first.

```bash
python -m pip install -c constraints/training.txt -e '.[training,test,pretrained]'
CUDA_VISIBLE_DEVICES='' python -m pytest -q
```

Direct dependency constraints for training and the separate LIBERO simulator
are in `constraints/`. These are version profiles, not complete transitive
lockfiles. The base package permits PyTorch 2.2 for the simulator; the
`training` extra requires PyTorch 2.5 or newer. Do not install both constraint
profiles into the same environment.

LIBERO simulation additionally needs an existing compatible
[LIBERO checkout](https://github.com/Lifelong-Robot-Learning/LIBERO), its BDDL
assets/initial states, MuJoCo, robosuite and OSMesa. Use a separate simulator
Python environment if necessary, and install this project into that environment
as well. See [training and evaluation](docs/training.md) for environment details.
The recipe never downloads datasets or models and never connects to a server.

## Prepare The 360,000-Step Recipe

Place the existing HDF5 data under `data/libero/`, with the four suite folders
`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`. Data are not included.

```bash
export LIBERO_DATASET_ROOT="$PWD/data/libero"
export LIBERO_ROOT="$PWD/third_party/LIBERO"
export LIBERO_SIM_PYTHON="$(command -v python)"

# Writes configuration only; does NOT launch training or allocate a GPU.
python -m vita.libero_recipe prepare --output runs/libero-paper-budget \
  --microbatch 16 --memory-gib 16

python -m vita.libero_recipe run --run-dir runs/libero-paper-budget \
  --stage manifest --dataset-root "$LIBERO_DATASET_ROOT"
```

The recipe uses two cameras, episode-disjoint splits and train-only action
normalization. Default future frame stride is `1`, deliberately adapted for
short LIBERO demonstrations; it is not the paper's one-second keyframe interval.
Every excluded short episode is listed in the generated manifest.

## Train, Resume And Evaluate

Only select a GPU already authorized for your job. Check its free memory and
other processes first; leave the configured cap plus at least 2 GiB free.
The single-device trainer uses gradient accumulation, not DDP.

```bash
# Example only: replace with your authorized physical GPU allocation.
export CUDA_VISIBLE_DEVICES=0

python -m vita.libero_recipe run --run-dir runs/libero-paper-budget \
  --stage pretrain --device cuda:0
python -m vita.libero_recipe run --run-dir runs/libero-paper-budget \
  --stage finetune --device cuda:0

# If a stage was interrupted, resume that stage's latest.pt, without changing source/config:
# python -m vita.libero_recipe run --run-dir runs/libero-paper-budget \
#   --stage pretrain --device cuda:0 --resume

python -m vita.libero_recipe run --run-dir runs/libero-paper-budget \
  --stage offline --device cuda:0
python -m vita.libero_recipe run --run-dir runs/libero-paper-budget --stage export

CUDA_VISIBLE_DEVICES='' python -m vita.libero_recipe run \
  --run-dir runs/libero-paper-budget --stage simulate \
  --dataset-root "$LIBERO_DATASET_ROOT" --libero-root "$LIBERO_ROOT" \
  --sim-python "$LIBERO_SIM_PYTHON"
python -m vita.libero_recipe run --run-dir runs/libero-paper-budget --stage verify
```

The standard rollout request is all 40 tasks, 50 initial states per task,
full suite horizons and action chunk `1`. Fifty trials is an explicit benchmark
convention, **not inferred from the paper's 50 demonstrations per task**.
Simulation defaults to CPU OSMesa and can be slow for a large model.
Offline MSE and simulator task success are separate metrics. A completed run
with zero successes remains zero successes, not evidence of reproduction.

Every command stays in the foreground. No retries, scheduler, SSH command or
background automation is included. `--stage all` runs the sequence on a fresh
prepared run; `--timeout 0` (default) imposes no subprocess time limit.

## Minimal Smoke Run

```bash
python -m vita.libero_recipe prepare --output runs/libero-smoke --smoke \
  --microbatch 2 --episodes-per-task 1 --max-tasks 1
CUDA_VISIBLE_DEVICES='' python -m vita.libero_recipe run \
  --run-dir runs/libero-smoke --stage all --device cpu \
  --dataset-root "$LIBERO_DATASET_ROOT" --libero-root "$LIBERO_ROOT" \
  --sim-python "$LIBERO_SIM_PYTHON"
```

This is **2 + 2 + 2 optimizer updates**, not a paper-budget run. It still requires
real data and the simulator. Unit tests instead generate synthetic local HDF5
fixtures and do not call a robot, provider API or real simulator.

## Larger Independent Model

`configs/libero_paper_model.json` provides paper-scale dimensions with DINOv2
and PaliGemma local checkpoint paths. It is not an official architecture dump
and requires separately licensed foundation weights and substantial memory.
Its full-scale training has not been validated by this release's CPU tests.

```bash
python -m vita.libero_recipe prepare --output runs/libero-large \
  --config configs/libero_paper_model.json --microbatch 1 --memory-gib 64
```

Batch accumulation automatically becomes `1024` and `192`. Edit a copy of the
model configuration **before** preparing the run if the asset paths differ.
Do not edit prepared configs mid-run. Compare assumptions in
[methodology](docs/methodology.md) before treating this as a scientific experiment.

## Publish A Clean Source Release

```bash
python scripts/build_release.py
python scripts/build_release.py --output dist/source
```

The release builder audits the explicit `.release-files` allowlist, rejects
symlinks and detected secrets/private paths, verifies archive members, and
creates a deterministic source archive plus `SHA256SUMS`. Archive owner names,
IDs and timestamps are cleared. Build any wheel/sdist from this clean archive,
not a research working directory. Read [release policy](docs/releasing.md).

## Code Map

- `vita/libero_recipe.py`: paper-budget plan and staged foreground execution.
- `vita/dataset.py`, `vita/train.py`: split-aware data, three training stages and resume.
- `vita/model.py`, `vita/modules.py`, `vita/policy.py`, `vita/quantizer.py`: independent model.
- `vita/evaluate.py`, `vita/deploy.py`, `vita/libero_eval.py`: offline metrics, export and simulation.
- `vita/prepare.py`, `vita/sources.py`, `vita/mixture.py`: optional multimodal data adapters.
- `vita/experiments.py`, `vita/latency.py`: ablation generation and profiling utilities.
- `vita/calvin_eval.py`, `vita/simpler_eval.py`: optional simulator adapters, outside this LIBERO recipe.
- `vita/realworld.py`: offline data contracts only; no physical robot driver.

Original source: MIT. Dependencies, data and weights retain their own licenses;
see [NOTICE](NOTICE). No pretrained or trained checkpoint is part of this release.
