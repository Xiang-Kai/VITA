# Documentation

[Project page](https://xiang-kai.github.io/vita-cvpr26.github.io/) /
[Repository](../README.md) / Documentation

## Choose A Path

| Guide | What it covers |
| --- | --- |
| [Quickstart](quickstart.md) | Clone, install, test, prepare, train and evaluate |
| [Training and evaluation](training.md) | Data contracts, environments, exact resume and simulator protocol |
| [Methodology](methodology.md) | Paper settings versus independent choices; reproduction limits |
| [Contributing](contributing.md) | Synthetic testing, source identity and clean changes |
| [Source releases](releasing.md) | Allowlist, privacy audit and reproducible packaging |
| [Citation](citation.md) | Paper attribution and public code scope |

## Code Navigation

| Component | Entry point |
| --- | --- |
| Full LIBERO recipe | [`vita/libero_recipe.py`](../vita/libero_recipe.py) |
| Read-only budget inspector | [`scripts/recipe_budget.py`](../scripts/recipe_budget.py) |
| Split-aware HDF5 data | [`vita/dataset.py`](../vita/dataset.py) |
| Three-stage training and exact resume | [`vita/train.py`](../vita/train.py) |
| Model and freezing rules | [`vita/model.py`](../vita/model.py) |
| Progressive policy / foundation adapter | [`vita/policy.py`](../vita/policy.py) |
| Visual/action encoders and decoders | [`vita/modules.py`](../vita/modules.py) |
| Shared discrete codebook | [`vita/quantizer.py`](../vita/quantizer.py) |
| Offline split metrics | [`vita/evaluate.py`](../vita/evaluate.py) |
| Action-only export | [`vita/deploy.py`](../vita/deploy.py) |
| Real LIBERO simulation | [`vita/libero_eval.py`](../vita/libero_eval.py) |
| Artifact reconciliation | [`vita/libero_pipeline.py`](../vita/libero_pipeline.py) |
| Optional multimodal data | [`vita/mixture.py`](../vita/mixture.py), [`vita/prepare.py`](../vita/prepare.py) |
| Ablations and profiling | [`vita/experiments.py`](../vita/experiments.py), [`vita/latency.py`](../vita/latency.py) |

The optional CALVIN and Simpler adapters are outside the validated LIBERO recipe.
The real-world module contains offline data contracts, not a physical robot driver.

## Configuration Navigation

- [Paper-budget pretraining](../configs/train_paper_pretrain.json) and
  [fine-tuning](../configs/train_paper_finetune.json) use microbatch 1.
- [Optional paper-scale model](../configs/libero_paper_model.json) requires local
  foundation weights. Default recipe preparation uses a compact random model.
- [Training constraints](../constraints/training.txt) and
  [simulator constraints](../constraints/simulator.txt) describe separate environments.

Prefer `python -m vita.libero_recipe prepare` for a new experiment rather than
combining legacy pilot/ablation configurations by filename. The generated
recipe records its exact budgets, hashes and assumptions.

## Evidence Boundaries

The project page presents the research paper and its original demonstrations.
This repository is an independent reference implementation. CPU tests establish
software properties; offline MSE evaluates prediction; LIBERO rollouts measure
task success. None of those should be silently substituted for another.

Historical logs, checkpoints and deployment information are not part of the
source release. See [what validation means](methodology.md#what-validation-means).
