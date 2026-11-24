from dataclasses import replace
import weakref

import pytest
import torch

from vita.dataset import atomic_json
from vita.evaluate import load_policy
from vita.train import TrainConfig, run_training
from test_dataset import small_manifest


@pytest.mark.parametrize("field,value", [
    ("warmup_steps", 1.5),
    ("cotrain_steps", True),
    ("finetune_steps", "60000"),
    ("batch_size", 2.0),
    ("accumulation_steps", False),
    ("eval_every", float("nan")),
    ("checkpoint_every", float("inf")),
    ("seed", 17.5),
    ("learning_rate", float("nan")),
    ("weight_decay", float("inf")),
    ("memory_gib", True),
])
def test_training_config_rejects_invalid_numeric_fields(field, value):
    with pytest.raises(ValueError, match=field):
        TrainConfig(**{field: value})


def test_training_config_accepts_zero_stages_and_integer_real_values():
    training = TrainConfig(warmup_steps=0, cotrain_steps=0, finetune_steps=1,
                           seed=0, weight_decay=0, memory_gib=4)
    assert training.total_steps == 1


def test_offline_policy_discards_training_tensors_and_preserves_predictions(config, tmp_path):
    from vita.data import synthetic_batch
    from vita.model import VITA
    from vita.train import core_hashes

    model = VITA(config).eval()
    model.set_stage("finetune")
    checkpoint = tmp_path / "synthetic-checkpoint.pt"
    torch.save({
        "schema_version": 2, "model_config": config.to_dict(), "stage": "finetune",
        "source_sha256": core_hashes(), "model": model.state_dict(),
        "optimizer": {"synthetic_unused_tensor": torch.ones(4)},
        "sampler": {"synthetic_unused_tensor": torch.ones(4)},
        "rng_cpu": torch.get_rng_state(), "rng_cuda": None, "rng_python": None,
        "train_config": {"synthetic_fixture": True}, "manifest_sha256": "synthetic-only",
    }, checkpoint)
    restored, metadata = load_policy(checkpoint, "cpu")
    assert not {"model", "optimizer", "sampler", "rng_cpu", "rng_cuda", "rng_python"} & metadata.keys()
    assert metadata["train_config"] == {"synthetic_fixture": True}
    assert metadata["manifest_sha256"] == "synthetic-only"
    batch = synthetic_batch(config)
    arguments = (batch["instructions"], batch["current"], batch["states"])
    torch.testing.assert_close(model.act(*arguments), restored.act(*arguments), rtol=0, atol=0)


def test_resume_releases_loaded_weight_copies_before_next_forward(config, tmp_path, monkeypatch):
    import vita.train as trainer

    config = replace(config, num_views=2)
    path = tmp_path / "manifest.json"
    atomic_json(path, small_manifest(tmp_path / "data", config))
    training = TrainConfig(warmup_steps=2, cotrain_steps=0, finetune_steps=0,
                           batch_size=2, eval_samples=2)
    run_training(config, training, path, tmp_path / "partial", max_updates=1)
    original_load, original_objective = torch.load, trainer.training_objective
    references, checked = [], []

    def tracked_load(*args, **kwargs):
        state = original_load(*args, **kwargs)
        references.extend(weakref.ref(value) for value in state["model"].values())
        return state

    def checked_objective(*args, **kwargs):
        assert references and all(ref() is None for ref in references)
        checked.append(True)
        return original_objective(*args, **kwargs)

    monkeypatch.setattr(torch, "load", tracked_load)
    monkeypatch.setattr(trainer, "training_objective", checked_objective)
    result = run_training(config, training, path, tmp_path / "resumed",
                          resume=tmp_path / "partial/latest.pt")
    assert checked and result["status"] == "training_schedule_completed"


@pytest.mark.parametrize("cut", [1, 3, 5])
def test_training_exact_resume_across_stages(config, tmp_path, cut):
    config = replace(config, num_views=2)
    manifest = small_manifest(tmp_path / "data", config)
    path = tmp_path / "manifest.json"
    atomic_json(path, manifest)
    training = TrainConfig(warmup_steps=2, cotrain_steps=2, finetune_steps=2,
                           batch_size=2, eval_samples=2, eval_every=2, checkpoint_every=2)
    full = run_training(config, training, path, tmp_path / "full")
    part = run_training(config, training, path, tmp_path / "partial", max_updates=cut)
    assert part["status"] == "bounded_run_stopped"
    resumed = run_training(config, training, path, tmp_path / "resumed",
                           resume=tmp_path / "partial/latest.pt")
    assert full["final_validation"] == resumed["final_validation"]
    original = torch.load(tmp_path / "full/latest.pt", weights_only=True)
    restored = torch.load(tmp_path / "resumed/latest.pt", weights_only=True)
    for name, value in original["model"].items():
        torch.testing.assert_close(value, restored["model"][name], rtol=0, atol=0)
    assert original["sampler"]["cursor"] == restored["sampler"]["cursor"]
    assert resumed["benchmark_reproduced"] is False
    assert resumed["test_split_evaluated"] is False


def test_resume_rejects_hyperparameter_change(config, tmp_path):
    config = replace(config, num_views=2)
    manifest = small_manifest(tmp_path / "data", config)
    path = tmp_path / "manifest.json"
    atomic_json(path, manifest)
    training = TrainConfig(warmup_steps=2, cotrain_steps=2, finetune_steps=2, batch_size=2, eval_samples=2)
    run_training(config, training, path, tmp_path / "partial", max_updates=1)
    with pytest.raises(ValueError, match="train_config"):
        run_training(config, replace(training, learning_rate=0.1), path, tmp_path / "bad",
                     resume=tmp_path / "partial/latest.pt")


def test_resume_with_data_initialized_codebook_and_accumulation(config, tmp_path):
    config = replace(config, num_views=2, codebook_size=8, action_latent_norm=True,
                     codebook_update="both", auxiliary_token_weight=0.5)
    manifest = small_manifest(tmp_path / "data", config)
    path = tmp_path / "manifest.json"
    atomic_json(path, manifest)
    training = TrainConfig(warmup_steps=2, cotrain_steps=2, finetune_steps=2, batch_size=4,
                           accumulation_steps=2, eval_samples=2, codebook_init_batches=1)
    run_training(config, training, path, tmp_path / "full")
    run_training(config, training, path, tmp_path / "partial", max_updates=3)
    run_training(config, training, path, tmp_path / "resumed", resume=tmp_path / "partial/latest.pt")
    full = torch.load(tmp_path / "full/latest.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/latest.pt", weights_only=True)
    for name in full["model"]:
        torch.testing.assert_close(full["model"][name], resumed["model"][name], atol=0, rtol=0)


def test_weight_only_finetuning_starts_new_optimizer_and_preserves_frozen_modules(config, tmp_path):
    config = replace(config, num_views=2)
    manifest = small_manifest(tmp_path / "data", config)
    path = tmp_path / "manifest.json"
    atomic_json(path, manifest)
    pretrain = TrainConfig(warmup_steps=1, cotrain_steps=1, finetune_steps=0, batch_size=2, eval_samples=2)
    run_training(config, pretrain, path, tmp_path / "pretrain")
    finetune = replace(pretrain, warmup_steps=0, cotrain_steps=0, finetune_steps=2)
    result = run_training(config, finetune, path, tmp_path / "finetune",
                          initialize_from=tmp_path / "pretrain/latest.pt")
    assert result["initialization"]["optimizer_restored"] is False
    before = torch.load(tmp_path / "pretrain/latest.pt", weights_only=True)["model"]
    after = torch.load(tmp_path / "finetune/latest.pt", weights_only=True)["model"]
    for name in before:
        if not name.startswith("action_decoder."):
            torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
    assert any(not torch.equal(before[name], after[name]) for name in before
               if name.startswith("action_decoder."))
