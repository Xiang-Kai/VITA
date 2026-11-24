"""Foreground LIBERO-only training, offline evaluation, export and real simulation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from .config import VITAConfig
from .dataset import SUITES, atomic_json, load_manifest
from .deploy import action_prefixes, file_sha256
from .libero_eval import MAX_STEPS, wilson_interval
from .runtime import check_device_scope, machine_metadata
from .train import TrainConfig, core_hashes


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_results(root, training, suites, max_tasks, episodes_per_task, action_chunk):
    """Reconcile artifacts, not just process exit codes. Zero successes is a valid result."""
    root = Path(root)
    manifest = load_manifest(root / "manifest.json")
    identity = manifest["manifest_sha256"]
    summary = read_json(root / "train/summary.json")
    checkpoint_path = root / "train/finetune_final.pt"
    checkpoint_hash = file_sha256(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(summary["status"] == "training_schedule_completed" and
            summary["next_step"] == summary["total_steps"] == training.total_steps and
            summary["last_stage"] == "finetune" and summary["test_split_evaluated"] is False,
            "Training schedule is incomplete or used the test split")
    require(saved["schema_version"] == 2 and saved["stage"] == "finetune" and
            saved["next_step"] == training.total_steps and saved["train_config"] == asdict(training),
            "Fine-tuned checkpoint disagrees with the training schedule")
    require(saved["manifest_sha256"] == summary["manifest_sha256"] == identity,
            "Training manifest identity mismatch")
    require(saved["source_sha256"] == summary["source_sha256"] == core_hashes(),
            "Source snapshot changed")

    offline = {}
    for split in ("validation", "test"):
        result = read_json(root / f"{split}.json")
        require(result["status"] == "offline_evaluation_completed" and result["split"] == split and
                result["action_source"] == "observation_only_policy", "Offline evaluation did not complete")
        require(result["manifest_sha256"] == identity and result["checkpoint_sha256"] == checkpoint_hash,
                "Offline evaluation checkpoint/manifest mismatch")
        require(result.get("runtime_source_sha256") == saved["source_sha256"],
                "Offline evaluator source snapshot mismatch")
        require(result["samples"] == result["total_available"] == manifest["counts"][split],
                "Offline evaluation omitted held-out windows")
        require(all(math.isfinite(value) for value in result["metrics"].values()) and
                all(result["metrics"][key] >= 0 for key in
                    ("action_mse", "raw_action_mse", "train_mean_baseline_mse")),
                "Invalid offline metrics")
        require(result["benchmark_success_rate"] is None, "Offline MSE is not simulator success")
        offline[split] = result

    artifact_path = root / "actions.pt"
    artifact_hash = file_sha256(artifact_path)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    require(artifact["schema_version"] == 3 and artifact["kind"] == "vita_action_policy" and
            artifact["stage"] == "finetune" and artifact["source_checkpoint_sha256"] == checkpoint_hash,
            "Action export has the wrong training parent")
    for key in ("manifest_sha256", "model_config", "normalization", "cameras", "source_sha256"):
        require(artifact[key] == saved[key], f"Action export changed {key}")
    prefixes = action_prefixes(VITAConfig(**saved["model_config"]))
    expected_weights = {key: value for key, value in saved["model"].items() if key.startswith(prefixes)}
    require(set(artifact["model"]) == set(expected_weights) and
            all(torch.equal(artifact["model"][key], value) for key, value in expected_weights.items()),
            "Exported action weights differ from the training checkpoint")

    rollout = read_json(root / "rollout/rollouts.json")
    require(rollout["status"] == "simulation_evaluation_completed" and
            rollout["backend"] == "real_LIBERO_MuJoCo_simulation" and rollout["physical_robot"] is False,
            "Real simulator evaluation did not complete")
    require(rollout["checkpoint_sha256"] == artifact_hash and
            rollout["source_checkpoint_sha256"] == checkpoint_hash and rollout["manifest_sha256"] == identity,
            "Simulator used a different checkpoint/manifest")
    require(rollout.get("runtime_source_sha256") == saved["source_sha256"],
            "Simulator evaluator source snapshot mismatch")
    require(rollout["suites"] == list(suites) and rollout["episodes_per_task"] == episodes_per_task and
            rollout["max_tasks"] == (max_tasks or None) and rollout["max_steps_override"] is None and
            rollout["default_horizons"] == MAX_STEPS and rollout["action_chunk"] == action_chunk and
            rollout.get("settle_steps") == 10 and rollout.get("seed") == training.seed,
            "Simulator evaluation protocol changed")
    require(rollout["render_backend"] == "osmesa" and rollout["cuda_visible_devices"] == "",
            "Simulator was not CPU OSMesa")
    task_counts = {suite: sum(row["suite"] == suite for row in manifest["files"]) for suite in suites}
    expected = {(suite, task, episode) for suite in suites
                for task in range(min(task_counts[suite], max_tasks) if max_tasks else task_counts[suite])
                for episode in range(episodes_per_task)}
    rows = rollout["episodes"]
    actual = {(row["suite"], row["task_index"], row["episode_id"]) for row in rows}
    require(actual == expected and len(rows) == len(expected) and rollout["trials"] == len(expected),
            "Simulator omitted or duplicated a requested trial")
    for row in rows:
        require(isinstance(row["success"], bool) and
                row.get("seed") == training.seed + row["episode_id"] and
                0 <= row["environment_steps"] <= MAX_STEPS[row["suite"]] and
                math.isfinite(row["clipped_action_fraction"]) and 0 <= row["clipped_action_fraction"] <= 1,
                "Invalid simulator episode metrics")
    successes = sum(row["success"] for row in rows)
    require(rollout["successes"] == successes and rollout["success_rate"] == successes / len(rows),
            "Simulator success aggregation mismatch")
    for suite in suites:
        selected = [row for row in rows if row["suite"] == suite]
        count = sum(row["success"] for row in selected)
        require(rollout["per_suite"][suite]["trials"] == len(selected) and
                rollout["per_suite"][suite]["successes"] == count and
                rollout["per_suite"][suite]["success_rate"] == count / len(selected),
                "Per-suite success aggregation mismatch")
    return {
        "status": "libero_pipeline_completed", "benchmark_reproduced": False,
        "synthetic_fixture": bool(rollout.get("synthetic_fixture") or
                                  any(result.get("synthetic_fixture") for result in offline.values())),
        "scope": "LIBERO-only engineering verification; not paper-scale training or benchmark replication",
        "manifest_sha256": identity, "training_checkpoint_sha256": checkpoint_hash,
        "action_checkpoint_sha256": artifact_hash, "source_sha256": saved["source_sha256"],
        "tasks": task_counts, "windows": manifest["counts"],
        "training": summary, "model_config": saved["model_config"],
        "offline": offline, "exported_weights_exact": True,
        "simulation": {"trials": len(rows), "successes": successes,
                       "success_rate": successes / len(rows), "wilson_95": wilson_interval(successes, len(rows)),
                       "per_suite": rollout["per_suite"], "scope": rollout["scope"],
                       "full_configured_horizons": True},
        "test_usage": "One final all-window evaluation of a fixed checkpoint; no test-driven model selection",
    }


class StageRunner:
    def __init__(self, root, report):
        self.root, self.report = root, report

    def save(self):
        atomic_json(self.root / "pipeline.json", self.report)

    def run(self, name, command, *, timeout, cuda=""):
        record = {"name": name, "status": "running", "command": [str(item) for item in command],
                  "timeout_seconds": timeout, "cuda_visible_devices": cuda,
                  "log": str(self.root / "logs" / f"{name}.log")}
        self.report["stages"].append(record)
        self.save()
        print(f"Starting {name}; log={record['log']}", flush=True)
        started = time.monotonic()
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": cuda, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
               "OPENBLAS_NUM_THREADS": "2", "USE_TF": "0", "USE_FLAX": "0", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
               "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"}
        try:
            with Path(record["log"]).open("w") as stream:
                completed = subprocess.run(record["command"], stdout=stream, stderr=subprocess.STDOUT,
                                           env=env, timeout=timeout, check=False)
            record["exit_code"] = completed.returncode
            if completed.returncode:
                raise subprocess.CalledProcessError(completed.returncode, record["command"])
            record["status"] = "process_completed"
        except Exception as error:
            record.update(status="failed", error=type(error).__name__, message=str(error))
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            self.save()
        print(f"Finished {name} in {record['elapsed_seconds']:.1f}s", flush=True)


def run_pipeline(args):
    root = Path(args.output).resolve()
    if root.exists():
        raise FileExistsError("Use a new pipeline directory; previous receipts must be preserved")
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    report = {"schema_version": 1, "status": "running", "stages": [], **machine_metadata(),
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "benchmark_reproduced": False, "arguments": vars(args)}
    runner = StageRunner(root, report)
    runner.save()
    try:
        config, training = VITAConfig.load(args.config), TrainConfig.load(args.training)
        require(all(getattr(training, f"{stage}_steps") > 0 for stage in ("warmup", "cotrain", "finetune")),
                "The LIBERO pipeline requires all three stages")
        require(config.num_views == 2 and config.state_dim == 8 and config.action_dim == 7,
                "The pipeline requires the verified two-camera LIBERO contract")
        require(0 < args.action_chunk <= config.action_horizon and args.episodes_per_task > 0 and
                args.max_tasks >= 0 and args.frame_stride > 0 and args.window_stride > 0,
                "Invalid data/rollout lengths")
        require(len(set(args.suites)) == len(args.suites), "Duplicate suites")
        require(all(value > 0 for value in (args.prepare_timeout, args.train_timeout,
                                            args.eval_timeout, args.rollout_timeout)), "Invalid timeout")
        require(Path(args.sim_python).is_file() and Path(args.libero_root).is_dir() and
                Path(args.dataset_root).is_dir(), "Missing existing LIBERO environment/data")
        check_device_scope(args.device)
        atomic_json(root / "model.json", config.to_dict())
        atomic_json(root / "training.json", asdict(training))
        prefix = [sys.executable, "-m"]
        manifest = root / "manifest.json"
        checkpoint = root / "train/finetune_final.pt"
        cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "") if args.device.startswith("cuda") else ""
        runner.run("manifest", [*prefix, "vita.dataset", "--root", args.dataset_root,
                   "--config", root / "model.json", "--output", manifest, "--suites", *args.suites,
                   "--frame-stride", args.frame_stride, "--window-stride", args.window_stride,
                   "--seed", training.seed], timeout=args.prepare_timeout)
        runner.run("train", [*prefix, "vita.train", "--config", root / "model.json",
                   "--training", root / "training.json", "--manifest", manifest, "--output", root / "train",
                   "--device", args.device], timeout=args.train_timeout, cuda=cuda)
        for split in ("validation", "test"):
            runner.run(split, [*prefix, "vita.evaluate", "--checkpoint", checkpoint, "--manifest", manifest,
                       "--split", split, "--output", root / f"{split}.json", "--device", args.device],
                       timeout=args.eval_timeout, cuda=cuda)
        runner.run("export", [*prefix, "vita.deploy", "--checkpoint", checkpoint,
                   "--output", root / "actions.pt"], timeout=args.prepare_timeout)
        rollout_command = [args.sim_python, "-m", "vita.libero_eval", "--checkpoint", root / "actions.pt",
                           "--libero-root", args.libero_root, "--dataset-root", args.dataset_root,
                           "--output", root / "rollout", "--device", "cpu", "--suites", *args.suites,
                           "--episodes-per-task", args.episodes_per_task, "--action-chunk", args.action_chunk,
                           "--seed", training.seed]
        if args.max_tasks:
            rollout_command.extend(["--max-tasks", args.max_tasks])
        runner.run("simulation", rollout_command, timeout=args.rollout_timeout)
        report["results"] = validate_results(root, training, args.suites, args.max_tasks,
                                              args.episodes_per_task, args.action_chunk)
        report["status"] = "libero_pipeline_completed"
        return report
    except Exception as error:
        report.update(status="failed", error=type(error).__name__, message=str(error))
        raise
    finally:
        report["ended_utc"] = datetime.now(timezone.utc).isoformat()
        runner.save()


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_balanced.json")
    parser.add_argument("--training", default="configs/train_libero_balanced.json")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--libero-root", required=True)
    parser.add_argument("--sim-python", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    parser.add_argument("--max-tasks", type=int, default=0, help="Simulator tasks per suite; 0 means all")
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--action-chunk", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=20)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--prepare-timeout", type=int, default=300)
    parser.add_argument("--train-timeout", type=int, default=900)
    parser.add_argument("--eval-timeout", type=int, default=300)
    parser.add_argument("--rollout-timeout", type=int, default=3600)
    return parser


def main():
    result = run_pipeline(make_parser().parse_args())
    print(json.dumps(result["results"]), flush=True)


if __name__ == "__main__":
    main()
