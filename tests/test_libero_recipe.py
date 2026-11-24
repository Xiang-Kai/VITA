"""CPU-only synthetic training integration; no provider, robot or simulator calls."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from vita.config import VITAConfig
from vita.dataset import atomic_json, build_manifest
from vita.deploy import file_sha256
from vita.libero_eval import MAX_STEPS
from vita.libero_pipeline import StageRunner
from vita.libero_recipe import (
    completed_training, load_recipe, make_parser, prepare_recipe, run_recipe, training_configs,
    validate_data,
)
from vita.train import TrainConfig, core_hashes
from test_dataset import create_task


@pytest.mark.parametrize("microbatch", [1, 2, 4, 8, 16, 32, 64])
def test_paper_update_and_batch_budgets(microbatch):
    pre, fine = training_configs(microbatch)
    assert (pre.warmup_steps, pre.cotrain_steps, fine.finetune_steps) == (200000, 100000, 60000)
    assert pre.total_steps + fine.total_steps == 360000
    assert pre.batch_size * pre.accumulation_steps == 1024
    assert fine.batch_size * fine.accumulation_steps == 192
    assert fine.warmup_steps == fine.cotrain_steps == pre.finetune_steps == 0
    assert pre.precision == fine.precision == "bfloat16"


@pytest.mark.parametrize("microbatch", [0, -1, 3, 128, 1.5])
def test_invalid_microbatch(microbatch):
    with pytest.raises(ValueError, match="microbatch"):
        training_configs(microbatch)


def test_prepare_does_not_start_training_and_detects_config_edits(tmp_path):
    root = tmp_path / "plan"
    recipe = prepare_recipe(root)
    assert recipe["total_optimizer_updates"] == 360000
    assert set(p.name for p in root.iterdir()) == {"recipe.json", "model.json", "pretrain.json", "finetune.json"}
    with pytest.raises(FileExistsError):
        prepare_recipe(root)
    config = json.loads((root / "pretrain.json").read_text())
    config["warmup_steps"] = 1
    atomic_json(root / "pretrain.json", config)
    with pytest.raises(ValueError, match="config changed"):
        load_recipe(root)


def test_sample_training_configs_agree_with_generated_budget():
    root = Path(__file__).resolve().parents[1]
    expected = training_configs(1, memory_gib=64)
    for name, training in zip(("pretrain", "finetune"), expected):
        assert asdict(TrainConfig.load(root / f"configs/train_paper_{name}.json")) == asdict(training)
    model = VITAConfig.load(root / "configs/libero_paper_model.json")
    assert model.state_dim == 8 and model.num_views == 2 and model.future_frames == 12


def test_paper_budget_rejects_partial_task_data_before_training(tmp_path):
    root, data = tmp_path / "run", tmp_path / "data"
    recipe = prepare_recipe(root, suites=["libero_object"])
    create_task(data, episodes=4)
    manifest = build_manifest(data, VITAConfig.load(root / "model.json"),
                              suites=["libero_object"], frame_stride=1, window_stride=8)
    atomic_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="all 10 distinct tasks"):
        validate_data(root, recipe)
    with pytest.raises(ValueError, match="all 10 distinct tasks"):
        run_recipe(arguments(root, "pretrain"))
    assert not (root / "pretrain").exists()


def arguments(root, stage, *extra):
    return make_parser().parse_args(["run", "--run-dir", str(root), "--stage", stage, *map(str, extra)])


def test_synthetic_staged_train_resume_offline_export(tmp_path, monkeypatch):
    root, data = tmp_path / "run", tmp_path / "data"
    create_task(data, episodes=4)
    prepare_recipe(root, smoke=True, microbatch=2, suites=["libero_object"], window_stride=12,
                   episodes_per_task=1)
    run_recipe(arguments(root, "manifest", "--dataset-root", data))
    partial = run_recipe(arguments(root, "pretrain", "--max-updates", 1))
    assert partial["status"] == "bounded_run_stopped"
    with pytest.raises(FileNotFoundError):
        run_recipe(arguments(root, "finetune"))
    run_recipe(arguments(root, "pretrain", "--resume"))
    _, pre, fine = load_recipe(root)
    completed_training(root, "pretrain", pre, "cotrain")
    run_recipe(arguments(root, "finetune", "--max-updates", 1))
    latest = root / "train/latest.pt"
    prior_bytes = latest.read_bytes()
    bad = torch.load(latest, weights_only=True)
    bad["initialization"]["source_checkpoint_sha256"] = "synthetic-wrong-parent"
    torch.save(bad, latest)
    with pytest.raises(ValueError, match="parent checkpoint"):
        run_recipe(arguments(root, "finetune", "--resume"))
    assert json.loads((root / "train/summary.json").read_text())["next_step"] == 1
    latest.write_bytes(prior_bytes)
    run_recipe(arguments(root, "finetune", "--resume"))
    saved = completed_training(root, "train", fine, "finetune")
    assert saved["initialization"]["optimizer_restored"] is False
    assert len(saved["initialization"]["source_checkpoint_sha256"]) == 64
    run_recipe(arguments(root, "offline"))
    run_recipe(arguments(root, "export"))
    for split in ("validation", "test"):
        result = json.loads((root / f"{split}.json").read_text())
        assert result["samples"] == result["total_available"] > 0
        assert result["benchmark_success_rate"] is None
    exported = torch.load(root / "actions.pt", weights_only=True)
    assert exported["kind"] == "vita_action_policy"
    assert not (root / ".recipe.lock").exists()
    with pytest.raises(FileNotFoundError):
        run_recipe(arguments(root, "verify"))  # No real/synthetic simulator result was produced.
    receipts = [json.loads(path.read_text()) for path in (root / "invocations").glob("*/pipeline.json")]
    assert any(row["status"] == "failed" for row in receipts)
    assert not (root / "results.json").exists()
    calls = []

    def synthetic_simulator(self, name, command, **kwargs):
        # No simulator is launched: this receipt tests artifact reconciliation only.
        calls.append((name, list(map(str, command)), kwargs))
        (root / "rollout").mkdir()
        atomic_json(root / "rollout/rollouts.json", {
            "synthetic_fixture": True, "status": "simulation_evaluation_completed",
            "backend": "real_LIBERO_MuJoCo_simulation", "physical_robot": False,
            "checkpoint_sha256": file_sha256(root / "actions.pt"),
            "runtime_source_sha256": core_hashes(), "seed": fine.seed, "settle_steps": 10,
            "source_checkpoint_sha256": file_sha256(root / "train/finetune_final.pt"),
            "manifest_sha256": saved["manifest_sha256"], "suites": ["libero_object"],
            "episodes_per_task": 1, "max_tasks": None, "max_steps_override": None,
            "default_horizons": MAX_STEPS, "action_chunk": 1,
            "render_backend": "osmesa", "cuda_visible_devices": "",
            "episodes": [{"suite": "libero_object", "task_index": 0, "episode_id": 0,
                          "seed": fine.seed,
                          "success": False, "environment_steps": 280, "clipped_action_fraction": 0.}],
            "trials": 1, "successes": 0, "success_rate": 0., "scope": "synthetic_fixture_only",
            "per_suite": {"libero_object": {"trials": 1, "successes": 0, "success_rate": 0.}},
        })

    monkeypatch.setattr(StageRunner, "run", synthetic_simulator)
    run_recipe(arguments(root, "simulate", "--dataset-root", data, "--libero-root", data))
    assert len(calls) == 1 and calls[0][0] == "simulate"
    assert calls[0][1][calls[0][1].index("--device") + 1] == "cpu"
    assert calls[0][2].get("cuda", "") == "" and calls[0][2]["timeout"] is None
    result = run_recipe(arguments(root, "verify"))
    assert result["results"]["total_optimizer_updates"] == 6
    assert result["results"]["benchmark_reproduced"] is False
    assert result["results"]["synthetic_fixture"] is True
    saved["initialization"]["source_checkpoint_sha256"] = "synthetic-wrong-parent"
    torch.save(saved, root / "train/finetune_final.pt")
    with pytest.raises(ValueError, match="parent checkpoint"):
        run_recipe(arguments(root, "verify"))


def test_recipe_rejects_concurrent_invocation_and_invalid_resume(tmp_path):
    root = tmp_path / "run"
    prepare_recipe(root, smoke=True)
    (root / ".recipe.lock").write_text("Synthetic lock; no live process\n")
    with pytest.raises(FileExistsError):
        run_recipe(arguments(root, "manifest"))
    with pytest.raises(ValueError, match="single training"):
        run_recipe(arguments(root, "all", "--resume"))


def test_recipe_plan_and_source_scope(config, tmp_path, monkeypatch):
    root = tmp_path / "run"
    prepare_recipe(root, smoke=True)
    monkeypatch.setattr("vita.libero_recipe.core_hashes", lambda: {"synthetic.py": "changed"})
    with pytest.raises(ValueError, match="Source snapshot"):
        load_recipe(root)
