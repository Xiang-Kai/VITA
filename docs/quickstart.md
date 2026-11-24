# Quickstart

[Documentation](index.md) / Quickstart

Choose the smallest useful step. Installation and CPU tests need no dataset,
weights, simulator or GPU. A real LIBERO smoke run needs existing data and a
compatible simulator. Full training is a separate, expensive experiment.

## 1. Install

The training/test profile uses Python 3.11, PyTorch 2.5.1, NumPy 1.26.4,
h5py 3.12.1 and Transformers 4.51.3. These direct-dependency constraints are
not a full transitive environment lock.

```bash
git clone https://github.com/Xiang-Kai/VITA.git
cd VITA
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the appropriate PyTorch 2.5.1 build for your OS/CUDA first.
python -m pip install -c constraints/training.txt -e '.[training,test,pretrained]'
CUDA_VISIBLE_DEVICES='' python -m pytest -q
python scripts/check_docs.py
python scripts/build_release.py
```

For Linux CPU-only development, you can install PyTorch first with:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

Use the platform-appropriate PyTorch distribution on macOS or for CUDA.
The CPU tests create synthetic fixtures, not real robot or simulator runs.
No model download is needed. The `pretrained` extra tests tiny local adapter
fixtures; it does not provide foundation weights.

## 2. Prepare Without Training

Run from the repository root in the environment containing this package:

```bash
python -m vita.libero_recipe prepare --output runs/libero-smoke --smoke \
  --microbatch 2 --episodes-per-task 1 --max-tasks 1
```

Expected files: `model.json`, `pretrain.json`, `finetune.json`, `recipe.json`.
The schedule contains six optimizer updates. This command does not read data,
load weights, allocate a GPU or launch training. Use a fresh directory; prepared
plans pin configuration and source hashes.

## 3. Point To Existing Assets

The repository does not download or redistribute LIBERO data or foundation
weights. Follow the [LIBERO project](https://github.com/Lifelong-Robot-Learning/LIBERO)
for installation, dataset licensing and simulator dependencies.

```text
data/libero/
  libero_spatial/*_demo.hdf5
  libero_object/*_demo.hdf5
  libero_goal/*_demo.hdf5
  libero_10/*_demo.hdf5
```

Set paths to your existing resources. The following are example locations,
not folders populated by the installation:

```bash
export LIBERO_DATASET_ROOT="$PWD/data/libero"
export LIBERO_ROOT="$PWD/third_party/LIBERO"
# Set this to the Python executable in your compatible LIBERO environment.
export LIBERO_SIM_PYTHON="$(command -v python)"
```

If training and simulation use separate environments, install this same source
revision into both. In the simulator environment, use its own interpreter:

```bash
"$LIBERO_SIM_PYTHON" -m pip install -c constraints/simulator.txt -e .
```

Do not install training and simulator constraints into one shared environment.
The simulator profile is Python 3.10 / PyTorch 2.2 with an existing LIBERO,
robosuite, MuJoCo and native OSMesa installation. See the full
[environment guide](training.md#environment-and-data). Installing this package
alone does not install LIBERO or native OSMesa libraries.

Data must contain 20 Hz controls, two uint8 RGB cameras, seven action values,
six end-effector state values and two gripper values. At least three eligible
demonstrations/task are needed for episode-disjoint splits. Paper-budget runs
require all ten tasks in each requested suite.

## 4. Run A Small Experiment

After prerequisites are satisfied, run the prepared smoke plan:

```bash
CUDA_VISIBLE_DEVICES='' python -m vita.libero_recipe run \
  --run-dir runs/libero-smoke --stage all --device cpu \
  --dataset-root "$LIBERO_DATASET_ROOT" --libero-root "$LIBERO_ROOT" \
  --sim-python "$LIBERO_SIM_PYTHON"
```

The run stays in the foreground and performs:

1. Episode-disjoint manifest generation and train-only normalization.
2. Two warmup updates and two co-training updates.
3. Weight-only transfer followed by two action-decoder fine-tuning updates.
4. Full validation/test offline evaluation and action-policy export.
5. One real simulator initial state for the first task in each requested suite.
6. Final artifact reconciliation in `results.json`.

With the default four suites this means four simulator trials, not one trial
total. `--max-tasks` limits simulation, not the training task set. To smoke-test
only `libero_object`, add `--suites libero_object` when **preparing a new plan**.
The compact random policy is not expected to reproduce paper performance.

| Output | Meaning |
| --- | --- |
| `manifest.json` | Splits, data identity and normalization |
| `pretrain/cotrain_final.pt` | Completed configured pretraining |
| `train/finetune_final.pt` | Fine-tuned training checkpoint |
| `validation.json`, `test.json` | Offline metrics, not task success |
| `actions.pt` | Exported action-policy weights |
| `rollout/rollouts.json` | Real simulator trials, failures and successes |
| `results.json` | Reconciled final report |

Only a successful `verify` stage writes the final completed result. A bounded
stop or failed process is not completion. Zero simulator successes stays zero.

## 5. Use The Paper Update Budget

Prepare a different plan; do not edit the smoke configuration in place:

```bash
python -m vita.libero_recipe prepare --output runs/libero-paper-budget \
  --microbatch 16 --memory-gib 16
python -m vita.libero_recipe run --run-dir runs/libero-paper-budget \
  --stage manifest --dataset-root "$LIBERO_DATASET_ROOT"
```

This configures 200,000 warmup + 100,000 co-training + 60,000 fine-tuning
optimizer updates. Effective batches are 1,024 and 192. On one device,
microbatch 16 uses accumulation 64 and 12. Supported paper-plan microbatches
are 1, 2, 4, 8, 16, 32 and 64.

Use only a GPU already authorized for your job, after inspecting utilization
and free memory. The allocator cap requires another 2 GiB free as a safety
margin; it does not reserve memory. This example assumes physical GPU 0 is
authorized and safe. Replace that assignment with your actual allocation:

```bash
export CUDA_VISIBLE_DEVICES=0
python -m vita.libero_recipe run --run-dir runs/libero-paper-budget \
  --stage pretrain --device cuda:0
python -m vita.libero_recipe run --run-dir runs/libero-paper-budget \
  --stage finetune --device cuda:0
python -m vita.libero_recipe run --run-dir runs/libero-paper-budget \
  --stage offline --device cuda:0
python -m vita.libero_recipe run --run-dir runs/libero-paper-budget --stage export
CUDA_VISIBLE_DEVICES='' python -m vita.libero_recipe run \
  --run-dir runs/libero-paper-budget --stage simulate \
  --dataset-root "$LIBERO_DATASET_ROOT" --libero-root "$LIBERO_ROOT" \
  --sim-python "$LIBERO_SIM_PYTHON"
python -m vita.libero_recipe run --run-dir runs/libero-paper-budget --stage verify
```

There is no DDP, scheduler, automatic retry or background automation. Simulator
evaluation uses CPU OSMesa and may be slow for large models. `--timeout 0`
means no per-stage subprocess timeout, not a runtime estimate.

Matching update budgets does not match the paper's data, model or results.
Review [methodology and limits](methodology.md) before running a long experiment.

## 6. Resume And Diagnose

Exact resume is stage-specific and preserves its original configuration:

```bash
python -m vita.libero_recipe run --run-dir runs/libero-paper-budget \
  --stage pretrain --device cuda:0 --resume
```

Use `--stage finetune --resume` for interrupted fine-tuning. For development,
`--max-updates 1` bounds a single training-stage invocation without modifying
the schedule. Checkpoint `next_step`, not log-line count, is authoritative.

| Symptom | What to check |
| --- | --- |
| Source/config mismatch | Use the exact revision and configs that prepared the run. Do not rewrite hashes. |
| Missing dataset/task | Check the root and suite folders; full runs need ten distinct tasks/suite. |
| Dataset changed | Preserve the old experiment and prepare a new manifest/run for changed data. |
| Missing simulator import/renderer | Check the simulator interpreter, LIBERO assets and native OSMesa installation. |
| CUDA memory guard | Verify your allocation and free memory, then use a smaller microbatch in a new plan. |
| Existing output refused | Preserve or rename previous partial output; do not overwrite evidence. |
| Stale `.recipe.lock` | Verify no process uses the run before removing the lock. Never remove a live lock. |
| Offline MSE improved but success is zero | These are different metrics. Do not relabel either result. |

`--stage all` is for a fresh prepared run, not automated crash recovery. After
resuming training, execute remaining stages individually. Preserve failure
receipts; a successful later invocation does not make a failed invocation pass.

All runtime directories contain local paths and provenance and are **not
public-ready**. Follow the [release policy](releasing.md) before sharing artifacts.
