"""Portable, foreground LIBERO-only recipe with paper-matched update budgets."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import uuid

import torch

from .config import VITAConfig
from .dataset import SUITES, atomic_json, load_manifest
from .deploy import file_sha256
from .libero_pipeline import StageRunner, read_json, require, validate_results
from .runtime import check_device_scope
from .train import TrainConfig, core_hashes


STAGES = ("manifest", "pretrain", "finetune", "offline", "export", "simulate", "verify")
PAPER_UPDATES = {"warmup": 200_000, "cotrain": 100_000, "finetune": 60_000}


def training_configs(microbatch=16, *, smoke=False, memory_gib=16.0, seed=17):
    if type(microbatch) is not int or microbatch < 1:
        raise ValueError("microbatch must be a positive integer")
    if not smoke and (1024 % microbatch or 192 % microbatch):
        raise ValueError("microbatch must divide both 1024 and 192 (for example 1, 2, 4, 8, 16, 32, 64)")
    common = dict(batch_size=microbatch, learning_rate=1e-4, weight_decay=0.01,
                  lr_schedule="constant", seed=seed, memory_gib=memory_gib,
                  precision="float32" if smoke else "bfloat16",
                  eval_every=2 if smoke else 1000, eval_samples=8 if smoke else 64,
                  checkpoint_every=2 if smoke else 1000)
    pretrain = TrainConfig(warmup_steps=2 if smoke else PAPER_UPDATES["warmup"],
                          cotrain_steps=2 if smoke else PAPER_UPDATES["cotrain"], finetune_steps=0,
                          accumulation_steps=1 if smoke else 1024 // microbatch, **common)
    finetune = replace(pretrain, warmup_steps=0, cotrain_steps=0,
                       finetune_steps=2 if smoke else PAPER_UPDATES["finetune"],
                       accumulation_steps=1 if smoke else 192 // microbatch)
    return pretrain, finetune


def prepare_recipe(output, *, config=None, microbatch=16, smoke=False, memory_gib=16.0,
                   seed=17, suites=SUITES, frame_stride=1, window_stride=8,
                   episodes_per_task=50, max_tasks=0, action_chunk=1):
    model = VITAConfig.load(config) if config else VITAConfig(num_views=2, text_length=160)
    pretrain, finetune = training_configs(microbatch, smoke=smoke, memory_gib=memory_gib, seed=seed)
    require(model.num_views == 2 and model.state_dim == 8 and model.action_dim == 7,
            "LIBERO requires two views, state_dim=8 and action_dim=7")
    require(model.finetune_scope == "action_only", "This recipe follows the action-only scope in Section 7.3")
    require(suites and set(suites) <= set(SUITES) and len(set(suites)) == len(suites), "Invalid suites")
    require(min(frame_stride, window_stride, episodes_per_task, action_chunk) > 0 and
            max_tasks >= 0 and action_chunk <= model.action_horizon, "Invalid data/rollout lengths")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name, value in (("model.json", model.to_dict()), ("pretrain.json", asdict(pretrain)),
                        ("finetune.json", asdict(finetune))):
        atomic_json(root / name, value)
    recipe = {
        "schema_version": 1, "kind": "libero_only_paper_budget", "smoke": smoke,
        "total_optimizer_updates": pretrain.total_steps + finetune.total_steps,
        "effective_batch": {"pretrain": pretrain.batch_size * pretrain.accumulation_steps,
                            "finetune": finetune.batch_size * finetune.accumulation_steps},
        "suites": list(suites), "frame_stride": frame_stride, "window_stride": window_stride,
        "episodes_per_task": episodes_per_task, "max_tasks": max_tasks, "action_chunk": action_chunk,
        "config_sha256": {name: file_sha256(root / name)
                          for name in ("model.json", "pretrain.json", "finetune.json")},
        "source_sha256": core_hashes(), "benchmark_reproduced": False,
        "scope": "LIBERO-only adaptation; step/batch matching is not paper-data/model/result equivalence",
        "references": {"pretraining": "Table 11", "libero_finetuning": "Section 7.3"},
        "independent_choices": {
            "finetune_lr": 1e-4, "lr_schedule": "constant",
            "temporal_sampling": "frame stride is explicit; default 1 does not reproduce 1-second keyframes",
            "evaluation": "50 fixed simulator initial states/task is a benchmark convention, not a verified paper setting",
            "data_split": "episode-disjoint 80/10/10 train/validation/test; normalization fitted on train only",
        },
    }
    atomic_json(root / "recipe.json", recipe)
    return recipe


def load_recipe(root):
    recipe = read_json(root / "recipe.json")
    require(recipe["schema_version"] == 1 and recipe["kind"] == "libero_only_paper_budget",
            "Unsupported recipe")
    require(set(recipe["config_sha256"]) == {"model.json", "pretrain.json", "finetune.json"},
            "Recipe config inventory changed")
    for name, expected in recipe["config_sha256"].items():
        require(name in {"model.json", "pretrain.json", "finetune.json"} and
                file_sha256(root / name) == expected, "Recipe config changed; prepare a new run")
    require(recipe["source_sha256"] == core_hashes(), "Source snapshot changed; use the prepared source version")
    pretrain, finetune = TrainConfig.load(root / "pretrain.json"), TrainConfig.load(root / "finetune.json")
    require(recipe["total_optimizer_updates"] == pretrain.total_steps + finetune.total_steps and
            recipe["effective_batch"] == {
                "pretrain": pretrain.batch_size * pretrain.accumulation_steps,
                "finetune": finetune.batch_size * finetune.accumulation_steps}, "Recipe budget mismatch")
    return recipe, pretrain, finetune


def completed_training(root, directory, training, stage):
    checkpoint = root / directory / f"{stage}_final.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    summary = read_json(root / directory / "summary.json")
    require(saved["schema_version"] == 2 and saved["stage"] == stage and
            saved["next_step"] == training.total_steps and saved["train_config"] == asdict(training) and
            summary["status"] == "training_schedule_completed" and
            summary["next_step"] == summary["total_steps"] == training.total_steps and
            summary["last_stage"] == stage and summary["test_split_evaluated"] is False,
            f"Incomplete or incompatible {directory} schedule")
    require(saved["source_sha256"] == summary["source_sha256"] == core_hashes() and
            saved["model_config"] == read_json(root / "model.json"), "Training source/model mismatch")
    require(saved["manifest_sha256"] == summary["manifest_sha256"] ==
            load_manifest(root / "manifest.json")["manifest_sha256"], "Training manifest mismatch")
    if stage == "finetune":
        validate_finetune_parent(root, saved)
    return saved


def validate_finetune_parent(root, saved):
    initialization = saved.get("initialization", {})
    require(initialization.get("method") == "weight_only_stage_transfer" and
            initialization.get("selection") == "all" and initialization.get("source_stage") == "cotrain" and
            initialization.get("source_manifest_sha256") == saved["manifest_sha256"] and
            initialization.get("source_checkpoint_sha256") == file_sha256(root / "pretrain/cotrain_final.pt"),
            "Fine-tuning parent checkpoint mismatch")


def validate_data(root, recipe):
    manifest = load_manifest(root / "manifest.json")
    require(set(row["suite"] for row in manifest["files"]) == set(recipe["suites"]),
            "Manifest suite coverage differs from recipe")
    require(manifest["frame_stride"] == recipe["frame_stride"] and
            manifest["window_stride"] == recipe["window_stride"] and
            manifest["seed"] == TrainConfig.load(root / "pretrain.json").seed,
            "Manifest sampling differs from recipe")
    require(all(manifest["counts"][split] > 0 for split in ("train", "validation", "test")),
            "An offline split is empty")
    if not recipe["smoke"]:
        for suite in recipe["suites"]:
            rows = [row for row in manifest["files"] if row["suite"] == suite]
            require(len(rows) == len({row["relative_path"] for row in rows}) == 10,
                    "Paper-budget LIBERO runs require all 10 distinct tasks in every requested suite")
    return manifest


def validate_recipe_results(root):
    recipe, pretrain, finetune = load_recipe(root)
    validate_data(root, recipe)
    completed_training(root, "pretrain", pretrain, "cotrain")
    completed_training(root, "train", finetune, "finetune")
    results = validate_results(root, finetune, recipe["suites"], recipe["max_tasks"],
                               recipe["episodes_per_task"], recipe["action_chunk"])
    results.update(status="libero_recipe_completed", scope=recipe["scope"],
                   total_optimizer_updates=pretrain.total_steps + finetune.total_steps,
                   effective_batch=recipe["effective_batch"], smoke=recipe["smoke"],
                   pretraining=read_json(root / "pretrain/summary.json"))
    return results


@contextmanager
def exclusive_run(root):
    lock = root / ".recipe.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode())
        os.close(descriptor)
        yield
    finally:
        lock.unlink()


def run_recipe(args):
    root = Path(args.run_dir).resolve()
    require(not args.resume or args.stage in {"pretrain", "finetune"},
            "--resume is only valid for a single training stage")
    require(args.max_updates is None or (args.max_updates > 0 and args.stage in {"pretrain", "finetune"}),
            "--max-updates requires a single training stage and a positive count")
    require(args.timeout >= 0, "timeout must be nonnegative; 0 means no timeout")
    with exclusive_run(root):
        invocation = root / "invocations" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
        (invocation / "logs").mkdir(parents=True)
        report = {"status": "running", "stage": args.stage, "stages": [], "benchmark_reproduced": False}
        runner = StageRunner(invocation, report)
        runner.save()
        try:
            recipe, pretrain, finetune = load_recipe(root)
            check_device_scope(args.device)
            stages = STAGES if args.stage == "all" else (args.stage,)
            prefix = [sys.executable, "-m"]
            timeout = args.timeout or None
            cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "") if args.device.startswith("cuda") else ""
            manifest, model = root / "manifest.json", root / "model.json"
            checkpoint = root / "train/finetune_final.pt"
            for stage in stages:
                if stage != "manifest":
                    validate_data(root, recipe)
                if stage == "manifest":
                    require(args.dataset_root and Path(args.dataset_root).is_dir(), "Existing dataset root required")
                    runner.run(stage, [*prefix, "vita.dataset", "--root", args.dataset_root,
                               "--config", model, "--output", manifest, "--suites", *recipe["suites"],
                               "--frame-stride", recipe["frame_stride"], "--window-stride", recipe["window_stride"],
                               "--seed", pretrain.seed], timeout=timeout)
                    validate_data(root, recipe)
                elif stage in {"pretrain", "finetune"}:
                    command = [*prefix, "vita.train", "--config", model, "--manifest", manifest,
                               "--training", root / f"{stage}.json", "--device", args.device,
                               "--output", root / ("train" if stage == "finetune" else "pretrain")]
                    if stage == "finetune":
                        completed_training(root, "pretrain", pretrain, "cotrain")
                        if not args.resume:
                            command.extend(["--initialize-from", root / "pretrain/cotrain_final.pt"])
                    if args.resume:
                        resume_path = root / ("train" if stage == "finetune" else "pretrain") / "latest.pt"
                        if stage == "finetune":
                            saved = torch.load(resume_path, map_location="cpu", weights_only=True)
                            validate_finetune_parent(root, saved)
                            del saved
                        command.extend(["--resume", resume_path])
                    if args.max_updates:
                        command.extend(["--max-updates", args.max_updates])
                    runner.run(stage, command, timeout=timeout, cuda=cuda)
                    directory = "train" if stage == "finetune" else "pretrain"
                    summary = read_json(root / directory / "summary.json")
                    if summary["status"] == "bounded_run_stopped":
                        report["status"] = "bounded_run_stopped"
                        return report
                    completed_training(root, directory, finetune if stage == "finetune" else pretrain,
                                       "finetune" if stage == "finetune" else "cotrain")
                elif stage == "offline":
                    completed_training(root, "train", finetune, "finetune")
                    for split in ("validation", "test"):
                        runner.run(split, [*prefix, "vita.evaluate", "--checkpoint", checkpoint,
                                   "--manifest", manifest, "--split", split, "--output", root / f"{split}.json",
                                   "--device", args.device, "--memory-gib", finetune.memory_gib], timeout=timeout, cuda=cuda)
                elif stage == "export":
                    completed_training(root, "train", finetune, "finetune")
                    runner.run(stage, [*prefix, "vita.deploy", "--checkpoint", checkpoint,
                               "--output", root / "actions.pt"], timeout=timeout)
                elif stage == "simulate":
                    completed_training(root, "train", finetune, "finetune")
                    artifact = torch.load(root / "actions.pt", map_location="cpu", weights_only=True)
                    require(artifact["source_checkpoint_sha256"] == file_sha256(checkpoint),
                            "Simulation export has a different parent")
                    require(args.libero_root and Path(args.libero_root).is_dir() and
                            args.dataset_root and Path(args.dataset_root).is_dir(), "Existing LIBERO/data roots required")
                    command = [args.sim_python or sys.executable, "-m", "vita.libero_eval",
                               "--checkpoint", root / "actions.pt", "--libero-root", args.libero_root,
                               "--dataset-root", args.dataset_root, "--output", root / "rollout", "--device", "cpu",
                               "--suites", *recipe["suites"], "--episodes-per-task", recipe["episodes_per_task"],
                               "--action-chunk", recipe["action_chunk"], "--seed", finetune.seed]
                    if recipe["max_tasks"]:
                        command.extend(["--max-tasks", recipe["max_tasks"]])
                    runner.run(stage, command, timeout=timeout)
                elif stage == "verify":
                    report["results"] = validate_recipe_results(root)
                    atomic_json(root / "results.json", report["results"])
            report["status"] = "libero_recipe_completed" if "verify" in stages else "requested_stages_completed"
            return report
        except Exception as error:
            report.update(status="failed", error=type(error).__name__, message=str(error))
            raise
        finally:
            runner.save()


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Write a plan only; no training, downloads or GPU allocation")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--config", help="Default: compact independent model, NOT paper-size")
    prepare.add_argument("--microbatch", type=int, default=16)
    prepare.add_argument("--smoke", action="store_true", help="2+2+2 updates and microbatch-only effective batch")
    prepare.add_argument("--memory-gib", type=float, default=16)
    prepare.add_argument("--seed", type=int, default=17)
    prepare.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    prepare.add_argument("--frame-stride", type=int, default=1)
    prepare.add_argument("--window-stride", type=int, default=8)
    prepare.add_argument("--episodes-per-task", type=int, default=50)
    prepare.add_argument("--max-tasks", type=int, default=0, help="Simulation only; 0 means all tasks")
    prepare.add_argument("--action-chunk", type=int, default=1)
    run = commands.add_parser("run", help="Run synchronously in this foreground process")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--stage", choices=(*STAGES, "all"), required=True)
    run.add_argument("--dataset-root")
    run.add_argument("--libero-root")
    run.add_argument("--sim-python", help="Interpreter with this package and LIBERO simulator installed")
    run.add_argument("--device", default="cpu")
    run.add_argument("--resume", action="store_true", help="Exact resume of the selected stage's latest.pt")
    run.add_argument("--max-updates", type=int, help="Bounded single-stage run; does not change the schedule")
    run.add_argument("--timeout", type=int, default=0, help="Per-subprocess seconds; 0 means no timeout")
    return parser


def main():
    args = make_parser().parse_args()
    if args.command == "prepare":
        result = prepare_recipe(**{key: value for key, value in vars(args).items() if key != "command"})
    else:
        result = run_recipe(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
