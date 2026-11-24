"""Offline synthetic handoff tests; no remote process, GPU, renderer or robot."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from vita.dataset import atomic_json
from vita.deploy import export_action_policy, file_sha256
from vita.libero_eval import MAX_STEPS
from vita.libero_pipeline import StageRunner, make_parser, run_pipeline, validate_results
from vita.train import TrainConfig, core_hashes, run_training
from test_dataset import small_manifest


@pytest.fixture
def pipeline_artifacts(config, tmp_path):
    config = replace(config, num_views=2)
    manifest = small_manifest(tmp_path / "data", config)
    atomic_json(tmp_path / "manifest.json", manifest)
    training = TrainConfig(warmup_steps=1, cotrain_steps=1, finetune_steps=1,
                           batch_size=2, eval_samples=2)
    run_training(config, training, tmp_path / "manifest.json", tmp_path / "train")
    checkpoint = tmp_path / "train/finetune_final.pt"
    export_action_policy(checkpoint, tmp_path / "actions.pt")
    # Explicit synthetic receipts test validation only; these are not scientific measurements.
    for split in ("validation", "test"):
        atomic_json(tmp_path / f"{split}.json", {
            "synthetic_fixture": True, "status": "offline_evaluation_completed", "split": split,
            "action_source": "observation_only_policy", "checkpoint_sha256": file_sha256(checkpoint),
            "runtime_source_sha256": core_hashes(),
            "manifest_sha256": manifest["manifest_sha256"], "samples": manifest["counts"][split],
            "total_available": manifest["counts"][split], "benchmark_success_rate": None,
            "metrics": {"action_mse": 1., "raw_action_mse": 0.2, "train_mean_baseline_mse": 1.1},
        })
    (tmp_path / "rollout").mkdir()
    atomic_json(tmp_path / "rollout/rollouts.json", {
        "synthetic_fixture": True, "status": "simulation_evaluation_completed",
        "backend": "real_LIBERO_MuJoCo_simulation", "physical_robot": False,
        "checkpoint_sha256": file_sha256(tmp_path / "actions.pt"),
        "runtime_source_sha256": core_hashes(), "seed": training.seed, "settle_steps": 10,
        "source_checkpoint_sha256": file_sha256(checkpoint), "manifest_sha256": manifest["manifest_sha256"],
        "suites": ["libero_object"], "episodes_per_task": 1, "max_tasks": None, "max_steps_override": None,
        "default_horizons": MAX_STEPS, "action_chunk": 4, "render_backend": "osmesa", "cuda_visible_devices": "",
        "episodes": [{"suite": "libero_object", "task_index": 0, "episode_id": 0, "success": False,
                      "seed": training.seed,
                      "environment_steps": 280, "clipped_action_fraction": 0.}],
        "trials": 1, "successes": 0, "success_rate": 0., "scope": "synthetic_fixture_only",
        "per_suite": {"libero_object": {"trials": 1, "successes": 0, "success_rate": 0.}},
    })
    return tmp_path, training


def validate_fixture(artifacts):
    root, training = artifacts
    return validate_results(root, training, ["libero_object"], 0, 1, 4)


def test_pipeline_reconciles_zero_success_without_benchmark_claim(pipeline_artifacts):
    result = validate_fixture(pipeline_artifacts)
    assert result["status"] == "libero_pipeline_completed"
    assert result["benchmark_reproduced"] is False
    assert result["synthetic_fixture"] is True
    assert result["exported_weights_exact"]
    assert result["simulation"]["successes"] == 0


@pytest.mark.parametrize("path,field,value,message", [
    ("validation.json", "checkpoint_sha256", "wrong", "checkpoint/manifest"),
    ("validation.json", "runtime_source_sha256", {}, "source snapshot"),
    ("test.json", "runtime_source_sha256", {}, "source snapshot"),
    ("test.json", "samples", 1, "omitted held-out"),
    ("train/summary.json", "status", "bounded_run_stopped", "incomplete"),
    ("rollout/rollouts.json", "status", "failed", "did not complete"),
    ("rollout/rollouts.json", "source_checkpoint_sha256", "wrong", "different checkpoint"),
    ("rollout/rollouts.json", "runtime_source_sha256", {}, "source snapshot"),
    ("rollout/rollouts.json", "seed", 999, "protocol changed"),
    ("rollout/rollouts.json", "settle_steps", 0, "protocol changed"),
    ("rollout/rollouts.json", "max_steps_override", 20, "protocol changed"),
    ("rollout/rollouts.json", "episodes", [], "omitted or duplicated"),
    ("rollout/rollouts.json", "success_rate", 1., "aggregation"),
])
def test_pipeline_rejects_incomplete_or_mixed_receipts(pipeline_artifacts, path, field, value, message):
    root, _ = pipeline_artifacts
    receipt = root / path
    result = json.loads(receipt.read_text())
    result[field] = value
    atomic_json(receipt, result)
    with pytest.raises(ValueError, match=message):
        validate_fixture(pipeline_artifacts)


def test_pipeline_rejects_export_weight_mutation(pipeline_artifacts):
    root, _ = pipeline_artifacts
    path = root / "actions.pt"
    saved = torch.load(path, weights_only=True)
    next(iter(saved["model"].values())).add_(1)
    torch.save(saved, path)
    with pytest.raises(ValueError, match="weights differ"):
        validate_fixture(pipeline_artifacts)


def test_pipeline_rejects_wrong_episode_seed(pipeline_artifacts):
    root, _ = pipeline_artifacts
    path = root / "rollout/rollouts.json"
    result = json.loads(path.read_text())
    result["episodes"][0]["seed"] += 1
    atomic_json(path, result)
    with pytest.raises(ValueError, match="episode metrics"):
        validate_fixture(pipeline_artifacts)


@pytest.mark.parametrize("failure", ["nonzero", "timeout"])
def test_stage_failure_preserves_receipt_and_does_not_retry(tmp_path, monkeypatch, failure):
    (tmp_path / "logs").mkdir()
    report = {"stages": []}
    runner = StageRunner(tmp_path, report)
    calls = []

    def offline_subprocess(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 1)
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(subprocess, "run", offline_subprocess)
    with pytest.raises((subprocess.CalledProcessError, subprocess.TimeoutExpired)):
        runner.run("synthetic", ["not-a-real-command"], timeout=1)
    assert len(calls) == 1
    receipt = json.loads((tmp_path / "pipeline.json").read_text())["stages"][0]
    assert receipt["status"] == "failed"
    assert receipt["elapsed_seconds"] >= 0


def test_pipeline_setup_failure_is_recorded_and_never_overwritten(tmp_path, monkeypatch):
    args = make_parser().parse_args([
        "--config", str(tmp_path / "missing.json"), "--dataset-root", str(tmp_path),
        "--libero-root", str(tmp_path), "--sim-python", sys.executable, "--output", str(tmp_path / "run"),
    ])
    with pytest.raises(FileNotFoundError):
        run_pipeline(args)
    receipt = Path(args.output) / "pipeline.json"
    before = receipt.read_bytes()
    assert json.loads(before)["status"] == "failed"
    with pytest.raises(FileExistsError):
        run_pipeline(args)
    assert receipt.read_bytes() == before


def test_pipeline_stage_order_and_cpu_simulator_scope(config, tmp_path, monkeypatch):
    model = tmp_path / "model.json"
    train = tmp_path / "train.json"
    atomic_json(model, replace(config, num_views=2).to_dict())
    atomic_json(train, asdict(TrainConfig(warmup_steps=1, cotrain_steps=1, finetune_steps=1)))
    args = make_parser().parse_args([
        "--config", str(model), "--training", str(train), "--dataset-root", str(tmp_path),
        "--libero-root", str(tmp_path), "--sim-python", sys.executable, "--output", str(tmp_path / "run"),
    ])
    commands = []

    def synthetic_stage(self, name, command, **kwargs):
        commands.append((name, [str(item) for item in command], kwargs))

    monkeypatch.setattr(StageRunner, "run", synthetic_stage)
    monkeypatch.setattr("vita.libero_pipeline.validate_results", lambda *a: {"synthetic_fixture": True})
    result = run_pipeline(args)
    assert [row[0] for row in commands] == ["manifest", "train", "validation", "test", "export", "simulation"]
    sim = commands[-1]
    assert sim[1][sim[1].index("--device") + 1] == "cpu"
    assert sim[2].get("cuda", "") == ""
    assert "--max-steps" not in sim[1] and "--max-tasks" not in sim[1]
    assert result["status"] == "libero_pipeline_completed"
