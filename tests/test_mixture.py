"""Synthetic CPU fixtures only; no real benchmark successes or model calls."""
from dataclasses import replace
import json

import h5py
import numpy as np
import pytest
import torch

from vita import VITA
from vita.dataset import atomic_json
from vita.mixture import (EpisodeDataset, MixtureBatchSampler, MixtureDataset,
                          build_episode_manifest, build_mixture, load_data_manifest)
from vita.model import masked_mse
from vita.train import TrainConfig, batch_from, run_training, training_objective


def source_fixture(root, config, mode):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{mode}.hdf5"
    rng = np.random.default_rng(22)
    with h5py.File(path, "w") as handle:
        handle.attrs.update(vita_episode_schema=1, control_hz=20,
                            action_convention="synthetic_controls_not_a_robot",
                            provenance="synthetic_unit_test_fixture")
        for index, split in enumerate(("train", "train", "validation", "test")):
            episode = handle.create_group(f"episodes/{index}")
            episode.attrs.update(split=split, group_id=f"synthetic_recording_{index}",
                                 instruction="move synthetic fixture")
            if mode != "action":
                episode["images"] = rng.integers(0, 256, (12, 1, 16, 16, 3), dtype=np.uint8)
            if mode != "video":
                episode["actions"] = rng.normal(size=(12, 3)).astype(np.float32) + index * 10
    return build_episode_manifest(path, config, mode=mode, frame_stride=2,
                                  action_columns=[0, 2, 6] if mode != "video" else None)


def mixture_fixture(root, config):
    rows = []
    weights = {"video": {"warmup": 1, "cotrain": 1}, "action": {"warmup": 1},
               "paired": {"warmup": 1, "cotrain": 1, "finetune": 1}}
    for mode in ("video", "action", "paired"):
        manifest = source_fixture(root, config, mode)
        atomic_json(root / f"{mode}.json", manifest)
        rows.append({"name": mode, "mode": mode, "manifest": f"{mode}.json", "weights": weights[mode]})
    spec = root / "spec.json"
    atomic_json(spec, {"sources": rows})
    return build_mixture(spec, config)


@pytest.mark.parametrize("mode", ["video", "action", "paired"])
def test_canonical_shapes_masks_and_no_fabricated_labels(config, tmp_path, mode):
    config = replace(config, num_views=2)
    manifest = source_fixture(tmp_path, config, mode)
    dataset = EpisodeDataset(manifest, "train")
    sample = dataset[0]
    assert ("actions" in sample) == (mode != "video")
    assert ("current" in sample) == (mode != "action")
    if mode != "action":
        assert sample["current"].shape == (2, 3, 32, 32)
        assert sample["view_mask"].tolist() == [True, False]
        assert not sample["state_mask"].any()
    if mode != "video":
        assert sample["action_mask"][0].tolist() == [True, False, True, False, False, False, True]
        assert (sample["actions"][~sample["action_mask"]] == 0).all()
        with h5py.File(manifest["path"], "r") as handle:
            train = np.concatenate([handle[f"episodes/{i}/actions"][:] for i in (0, 1)])
        np.testing.assert_allclose(np.array(manifest["normalization"]["mean"])[[0, 2, 6]],
                                   train.mean(0), atol=1e-5)
    dataset.close()


def test_group_leakage_is_rejected(config, tmp_path):
    manifest = source_fixture(tmp_path, config, "paired")
    with h5py.File(manifest["path"], "r+") as handle:
        handle["episodes/2"].attrs["group_id"] = "synthetic_recording_0"
    with pytest.raises(ValueError, match="crosses split"):
        build_episode_manifest(manifest["path"], config, mode="paired", action_columns=[0, 2, 6])


def test_mixture_sampler_stage_filtering_and_exact_resume(config, tmp_path):
    manifest = mixture_fixture(tmp_path, config)
    dataset = MixtureDataset(manifest, "train")
    first = MixtureBatchSampler(dataset, 3, 9)
    first.next("warmup")
    saved = first.state_dict()
    second = MixtureBatchSampler(dataset, 3, 9)
    second.load_state_dict(saved)
    seen = set()
    for stage in ["warmup"] * 30 + ["cotrain"] * 30 + ["finetune"] * 10:
        indices = first.next(stage)
        assert indices == second.next(stage)
        modes = {dataset[i]["modality"] for i in indices}
        assert len(modes) == 1
        seen |= modes
        if stage == "cotrain":
            assert modes != {"action"}
        if stage == "finetune":
            assert modes == {"paired"}
    assert seen == {"video", "action", "paired"}
    dataset.close()


def test_all_objectives_with_truly_missing_modalities(config, tmp_path):
    model = VITA(config)
    for mode in ("video", "action", "paired"):
        dataset = EpisodeDataset(source_fixture(tmp_path, config, mode), "train")
        batch = batch_from(dataset, [0, 1], "cpu")
        for step in (0, 1):
            model.set_stage("warmup")
            result = training_objective(model, batch, step)
            assert torch.isfinite(result["loss"])
            result["loss"].backward()
        if mode != "action":
            model.set_stage("cotrain")
            result = training_objective(model, batch)
            result["loss"].backward()
            assert model.policy.code_head.weight.grad.abs().sum() > 0
        dataset.close()


@pytest.mark.parametrize("cut", [1, 4, 7])
def test_mixture_training_exact_resume(config, tmp_path, cut):
    manifest = mixture_fixture(tmp_path / "data", config)
    path = tmp_path / "manifest.json"
    atomic_json(path, manifest)
    assert load_data_manifest(path)["manifest_sha256"] == manifest["manifest_sha256"]
    training = TrainConfig(warmup_steps=3, cotrain_steps=3, finetune_steps=2,
                           accumulation_steps=2, batch_size=2, eval_samples=2, eval_every=2)
    run_training(config, training, path, tmp_path / "full")
    run_training(config, training, path, tmp_path / "part", max_updates=cut)
    result = run_training(config, training, path, tmp_path / "resume", resume=tmp_path / "part/latest.pt")
    first = torch.load(tmp_path / "full/latest.pt", weights_only=True)
    second = torch.load(tmp_path / "resume/latest.pt", weights_only=True)
    for name in first["model"]:
        torch.testing.assert_close(first["model"][name], second["model"][name], rtol=0, atol=0)
    assert set(result["final_validation"]["per_source"]) == {"paired"}


def test_masks_block_invalid_control_gradients_and_state_inputs(config, tmp_path):
    prediction = torch.randn(2, 4, 7, requires_grad=True)
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    mask[..., 0] = True
    masked_mse(prediction, torch.zeros_like(prediction), mask).backward()
    assert (prediction.grad[~mask] == 0).all()
    with pytest.raises(ValueError):
        masked_mse(prediction, prediction, torch.zeros_like(mask))
    dataset = EpisodeDataset(source_fixture(tmp_path, config, "paired"), "train")
    batch = batch_from(dataset, [0, 1], "cpu")
    model = VITA(config).eval()
    before = model.generate(batch)["token_logits"]
    batch["states"][:] = 1000
    torch.testing.assert_close(before, model.generate(batch)["token_logits"], rtol=0, atol=0)
    dataset.close()


def test_empty_training_schedule_is_rejected():
    with pytest.raises(ValueError, match="At least"):
        TrainConfig(warmup_steps=0, cotrain_steps=0, finetune_steps=0)
    assert TrainConfig(warmup_steps=0, cotrain_steps=0, finetune_steps=2).stage_at(0)[0] == "finetune"
