# Training And Evaluation

## Environment And Data

Install the package into both the training interpreter and, if different, the
simulator interpreter. Keep the same source version in both. The simulator needs
LIBERO's compatible dependencies (including its robosuite version), task BDDL
files and initial-state files. A typical Linux CPU renderer setup uses OSMesa
(`libosmesa6`), `MUJOCO_GL=osmesa`, `PYOPENGL_PLATFORM=osmesa`. Consult LIBERO's
own installation guide for platform-specific native libraries. This repository
does not change a shared environment or silently install simulator dependencies.

Two direct-dependency version profiles are provided:

| Role | Python | PyTorch | NumPy | HDF5 Python | Transformers |
| --- | --- | --- | --- | --- | --- |
| Training/tests | 3.11 | 2.5.1 | 1.26.4 | 3.12.1 | 4.51.3 |
| CPU simulator | 3.10 | 2.2.0 | 1.26.4 | 3.15.1 | 4.53.3 |

Use `constraints/training.txt` with `.[training,test,pretrained]` in the
training environment. In a separate compatible LIBERO environment, use
`python -m pip install -c constraints/simulator.txt -e .`. The latter profile
records MuJoCo 3.4.0, robosuite 1.4.1, BDDL 3.6.0, Gym 0.26.2 and PyOpenGL
3.1.10. LIBERO itself, its other dependencies and native OSMesa libraries must
already be installed. Constraints do not install optional dependencies by
themselves and are not a complete transitive lock or a guarantee on every OS.
Install the `pretrained` extra in the simulator environment only if using a
foundation-backed model; the compact smoke model does not need it.

Offline evaluation and LIBERO simulation check the checkpoint's source hashes
against the actual loaded evaluator code before constructing the policy.
Runtime source hashes and dependency versions are recorded separately from
checkpoint provenance. The final verifier checks the runtime source and
the configured base/per-episode seeds and ten settling steps, not just exit
codes. Use the exact matching release in both interpreters; do not rewrite
checkpoint hashes to make a different source revision appear compatible.

Supported data layout:

```text
data/libero/
  libero_spatial/*_demo.hdf5
  libero_object/*_demo.hdf5
  libero_goal/*_demo.hdf5
  libero_10/*_demo.hdf5
```

Each demonstration needs synchronized actions, `agentview_rgb`,
`eye_in_hand_rgb`, `ee_states` and `gripper_states`, with LIBERO metadata.
The adapter requires seven action dimensions, OpenGL image convention and
20 Hz controls, matching the simulator's fixed control frequency. Camera arrays
must be uint8 RGB `[T,H,W,3]`; float images are rejected rather than silently
divided by 255 again. Images are flipped vertically. State arrays must contain
six end-effector values and two gripper values.
Missing or too-short tasks fail rather than silently disappear from evaluation.
Paper-budget runs require all ten distinct task files in each requested suite
before training starts; smoke mode permits smaller synthetic task sets.
At least three eligible demonstrations/task are required to form disjoint
train, validation and test episode sets. Small fractions are rounded with at
least one validation and one test demonstration. All eligible training-episode
actions, not just sampled windows, determine normalization.

## Configurations

`prepare` writes `model.json`, `pretrain.json`, `finetune.json` and `recipe.json`.
It does not load weights, read demonstrations or launch a job. The plan pins
source/config hashes. Recreate a plan after editing code; exact resume rejects
source changes, different manifests and changed hyperparameters.

Model and training configuration counts must be integers, not floating-point
numbers, strings or booleans. Numeric hyperparameters must be finite; boolean
model options require JSON `true`/`false`, not quoted strings. Invalid values
are rejected while loading configuration, before constructing a model.

Paper-budget microbatches may be `1, 2, 4, 8, 16, 32, 64`. The effective batch is
`microbatch * accumulation_steps` on **one** device. There is no DDP/world-size
factor. Smaller microbatches reduce memory but do not reduce the optimizer-step
budget. `--memory-gib` caps the CUDA allocator, not host RAM or all process GPU
allocations. The guard also requires at least 2 GiB beyond the cap to be free;
it is not a reservation or a substitute for a scheduler allocation.

The default compact model is suitable for integration work, not a replacement
for a pretrained VLM. Optional large-model training can be expensive even with
microbatch 1. Do not assume the paper's 16-A100 wall time on this single-device
trainer. Foundation checkpoint paths must already exist locally; adapters use
local-only loading.

## Stages And Dependencies

| Stage | Required input | New output |
| --- | --- | --- |
| `manifest` | Existing HDF5 data, prepared model | `manifest.json` |
| `pretrain` | Manifest | `pretrain/warmup_final.pt`, `pretrain/cotrain_final.pt` |
| `finetune` | Verified completed pretraining | `train/finetune_final.pt` |
| `offline` | Completed fine-tuning | `validation.json`, `test.json` |
| `export` | Completed fine-tuning | `actions.pt` |
| `simulate` | Action export, LIBERO/data/interpreter | `rollout/rollouts.json` |
| `verify` | All outputs | `results.json` |

Pretraining contains 200,000 warmup and 100,000 co-training updates. Fine-tuning
is a separate 60,000-update schedule initialized with **weights only** from the
completed co-training checkpoint. It starts a new optimizer. Exact resume instead
restores optimizer, RNG and sampler state from the current stage's `latest.pt`.
After restoration, unused checkpoint weight copies are released before the
next training forward pass. Offline evaluation retains only checkpoint
metadata after loading weights, not the optimizer, sampler or duplicate model
state. These changes reduce retained host memory, not the transient memory
needed to deserialize a checkpoint or the CUDA allocator requirement.
The transfer records the parent checkpoint SHA-256; final verification checks it.
Fine-tuning resume and downstream evaluation also check that parent before
launching more computation.

Example bounded development run and continuation:

```bash
python -m vita.libero_recipe run --run-dir runs/libero-smoke \
  --stage pretrain --max-updates 1
python -m vita.libero_recipe run --run-dir runs/libero-smoke \
  --stage pretrain --resume
```

A bounded stop is marked `bounded_run_stopped`, not training completion. At
configured intervals and boundaries, checkpoints are written atomically.
Failure receipts and separate invocation logs are preserved; no retry is
performed. If a process crashes between saves, resume replays from the last
saved update. Training history may retain records after that checkpoint; use
checkpoint `next_step`, not log line counts, as the authoritative update count.

Existing nonempty outputs are not replaced automatically. `all` is for fresh
prepared runs, not automatic recovery. After interruption, resume the relevant
training stage, then run the remaining stages individually. For a failed
offline/simulator stage, preserve or rename its partial output before rerunning.
A per-run lock rejects concurrent recipe invocations. If abrupt process death
leaves `.recipe.lock`, verify no process still uses the run before removing it.
A source change invalidates exact resume; weight-only initialization is a new
experiment, not a continuation with identical dynamics.

## Evaluation Protocol

Training uses only the training split and periodic validation. Final offline
evaluation covers every window in both held-out splits and never reports MSE
as task success. Avoid using final test results to select checkpoints.

The standard simulator request evaluates each official initial state once,
50 per task by default. Horizons are 220/280/300/520 for spatial/object/goal/10,
plus ten settling steps; policy action chunk defaults to 1. Success is read
from LIBERO's task check, not inferred from loss. Reports include per-suite
success and Wilson intervals. Seeds and episode IDs are explicit. The runner
uses a run-local LIBERO configuration, without editing a shared checkout.
`--max-tasks` restricts simulation only, not the training task set.

CPU OSMesa simulation is portable but may be impractically slow for a full-size
VLM. GPU simulation/performance optimization is not part of this reference
recipe. The final verifier deliberately enforces the documented CPU protocol.

## Privacy Of New Runs

Runs contain absolute dataset/weight paths, commands and checkpoint provenance.
These are necessary local artifacts, **not public-ready files**. Machine names
and PIDs are omitted from metrics unless `VITA_RECORD_MACHINE_METADATA=1`.
`VITA_EXPECTED_HOST` optionally enforces an operator-provided CUDA host check.
Neither setting assigns a host or grants access to resources. Keep runtime
directories out of source releases and review any separately shared artifacts.
