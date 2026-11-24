<div align="center">

# VITA

### Unifying Perception and Action
A Hybrid-Modality Pipeline with Implicit Visual Chain-of-Thought<br>
for Robotic Action Generation

**Public Reference Implementation**

[Project Page](https://xiang-kai.github.io/vita-cvpr26.github.io/) &middot;
[Paper](https://arxiv.org/abs/2511.19859) &middot;
[Quickstart](docs/quickstart.md) &middot;
[Training Guide](docs/training.md) &middot;
[Documentation](docs/index.md)

[![CPU Tests And Source Audit](https://github.com/Xiang-Kai/VITA/actions/workflows/cpu.yml/badge.svg)](https://github.com/Xiang-Kai/VITA/actions/workflows/cpu.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-e8c886.svg)](LICENSE)

</div>

> **One latent space. Two modalities.**
> Learn a shared motion vocabulary for visual dynamics and robot actions.

**Independent implementation, not official author code. No paper-level performance claim.**
This repository provides a LIBERO-only recipe with paper-matched update budgets,
exact stage resume, full held-out offline evaluation, action-policy export and
real LIBERO/MuJoCo integration. The default model is compact and initialized
from scratch. Datasets, foundation weights, trained checkpoints and historical
experiment logs are **not** distributed.

## Start Here

| Your goal | Start with | What you need |
| --- | --- | --- |
| Understand VITA | [Method and implementation scope](docs/methodology.md) | No installation |
| Test the code | [Install and run CPU tests](docs/quickstart.md#1-install) | Python and package dependencies |
| Inspect a training plan | [Read-only budget inspector](docs/quickstart.md#2-prepare-without-training) | No dataset, weights or GPU |
| Run a small end-to-end experiment | [LIBERO smoke workflow](docs/quickstart.md#4-run-a-small-experiment) | Existing LIBERO data and simulator |
| Train with paper-matched budgets | [Full schedule](docs/quickstart.md#5-use-the-paper-update-budget) | Data, authorized compute and time |
| Contribute or publish | [Contributing](docs/contributing.md) / [Release policy](docs/releasing.md) | Synthetic tests and a clean source tree |

## Release Scope

- [x] **Model:** visual/action quantization, internal motion tokens and action decoding.
- [x] **Training:** three stages, gradient accumulation, exact optimizer/RNG/sampler resume.
- [x] **Data:** episode-disjoint splits, train-only normalization and verified LIBERO conventions.
- [x] **Evaluation:** full offline splits, action export and a real LIBERO simulator adapter.
- [x] **Engineering checks:** CPU regressions, artifact reconciliation and source privacy audit in CI.
- [ ] **Not included:** the paper's mixed pretraining corpus or pretrained/trained checkpoints.
- [ ] **Not established:** full paper-scale training or paper-level benchmark performance.

[Current CI status](https://github.com/Xiang-Kai/VITA/actions/workflows/cpu.yml)
is software validation, not a simulator success-rate measurement. Research
videos on the project page are original paper demonstrations, not outputs of
this independent release.

## Method

1. **Align visual and action representations.** Two autoencoders use a shared
   discrete codebook to connect visual dynamics with action trajectories.
2. **Generate an internal visual chain of thought.** The policy predicts motion
   tokens from observations and instructions. Visual reconstruction provides
   a training signal alongside action prediction.
3. **Decode actions.** The action decoder turns the latent stream into controls.
   The default internal-CoT deployment omits future-image reconstruction.

The code includes optional ablations and local foundation-model adapters.
[Methodology](docs/methodology.md) separates reported paper settings from
implementation assumptions. [Code navigation](docs/index.md#code-navigation)
connects each component to its source.

## Training Budget

| Stage | Optimizer updates | Effective batch | Microbatch | Accumulation |
| --- | ---: | ---: | ---: | ---: |
| Warmup | 200,000 | 1,024 | 16 | 64 |
| Co-training | 100,000 | 1,024 | 16 | 64 |
| LIBERO fine-tuning | 60,000 | 192 | 16 | 12 |
| **Total** | **360,000** | | | |

Table 11 and Section 7.3 of the paper specify the update/batch budgets.
These are **optimizer updates, not microsteps or epochs**, and are provided as
configuration, not a claim that full training was completed.

Inspect any supported microbatch/suite combination before creating a run:

```bash
python scripts/recipe_budget.py --microbatch 16 --suites libero_object
```

At microbatch 16, the full schedule is 19,920,000 training microsteps.
The tool also distinguishes requested simulator trials from offline metrics
and prints a configuration-only prepare command. Add `--smoke` for six updates
or `--json` for machine-readable output. Nothing is trained or written.

Pretraining uses AdamW, LR `1e-4`, betas `(0.9, 0.95)`, weight decay `0.01` and
BF16. Fine-tuning updates only the action decoder. Fine-tuning LR `1e-4` and a
constant schedule are explicit independent choices. All stages reuse LIBERO
training episodes instead of the paper's mixed human/robot pretraining data.

## Quickstart

Use Python 3.11 for the training/test profile. Install the appropriate PyTorch
build for your platform first. [Full installation instructions](docs/quickstart.md#1-install)
cover CPU and separate simulator environments.

```bash
git clone https://github.com/Xiang-Kai/VITA.git
cd VITA
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c constraints/training.txt -e '.[training,test,pretrained]'
CUDA_VISIBLE_DEVICES='' python -m pytest -q
```

Prepare a small plan without reading data or starting training:

```bash
python -m vita.libero_recipe prepare --output runs/libero-smoke --smoke \
  --microbatch 2 --episodes-per-task 1 --max-tasks 1
```

Once the existing LIBERO data and simulator are configured:

```bash
CUDA_VISIBLE_DEVICES='' python -m vita.libero_recipe run \
  --run-dir runs/libero-smoke --stage all --device cpu \
  --dataset-root "$LIBERO_DATASET_ROOT" --libero-root "$LIBERO_ROOT" \
  --sim-python "$LIBERO_SIM_PYTHON"
```

The smoke schedule is **2 + 2 + 2 updates**. It still needs real LIBERO data and
a compatible simulator; CPU unit tests use synthetic fixtures instead. It does
not establish learned task performance. See the [step-by-step guide](docs/quickstart.md)
for asset setup, full-budget training and recovery.

## Evaluate And Resume

The recipe follows explicit foreground stages:

```text
manifest -> pretrain -> finetune -> offline -> export -> simulate -> verify
```

- `offline` evaluates all validation/test windows. MSE is not task success.
- `simulate` runs CPU OSMesa with LIBERO's task-success predicate.
- `verify` reconciles checkpoint, manifest, source, export and rollout identities.
- `--resume` restores the current training stage's `latest.pt` exactly; it is
  not weight-only initialization or automatic recovery of failed evaluation.

The full protocol requests 40 tasks and 50 fixed initial states per task. Fifty
trials is a disclosed benchmark convention, not inferred from 50 demonstrations.
Use the same source/configuration for a prepared run. New source revisions can
invalidate exact resume; never rewrite checkpoint hashes to force compatibility.

[Training and evaluation reference](docs/training.md) &middot;
[Recovery and troubleshooting](docs/quickstart.md#6-resume-and-diagnose)

## Larger Models

`configs/libero_paper_model.json` supplies optional paper-scale dimensions with
local DINOv2 and PaliGemma paths. It is **not** an official architecture dump and
requires separately licensed foundation weights and substantial memory.

```bash
python -m vita.libero_recipe prepare --output runs/libero-large \
  --config configs/libero_paper_model.json --microbatch 1 --memory-gib 64
```

This only writes configuration. The single-device trainer uses accumulation,
not DDP. Full-size training is not validated by compact CPU tests. Check the
[environment, memory and model notes](docs/training.md#configurations) first.

## Contribute And Cite

Read the [contribution guide](docs/contributing.md), run the CPU tests and check
the [source-release policy](docs/releasing.md). Runtime data, logs and deployment
credentials must not enter source control.

If the research informs your work, [cite the VITA paper](docs/citation.md).
Original project code is MIT; dependencies, data and weights retain their own
licenses. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
