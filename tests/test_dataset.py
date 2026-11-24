from dataclasses import replace
import json

import h5py
import numpy as np
import pytest
import torch

from vita.dataset import LiberoDataset, StatefulBatchSampler, atomic_json, build_manifest, load_manifest


def create_task(root, name="synthetic_demo", episodes=10):
    folder = root / "libero_object"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.hdf5"
    rng = np.random.default_rng(19)
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["problem_info"] = json.dumps({"language_instruction": "synthetic fixture task"})
        data.attrs["env_args"] = json.dumps({"env_kwargs": {"control_freq": 20}})
        data.attrs["macros_image_convention"] = "opengl"
        for index in range(episodes):
            demo = data.create_group(f"demo_{index}")
            demo["actions"] = rng.normal(size=(24, 7)).astype(np.float32) + index
            obs = demo.create_group("obs")
            for key in ("agentview_rgb", "eye_in_hand_rgb"):
                obs[key] = rng.integers(0, 256, (24, 16, 16, 3), dtype=np.uint8)
            obs["ee_states"] = np.zeros((24, 6), dtype=np.float32)
            obs["gripper_states"] = np.zeros((24, 2), dtype=np.float32)
    return path


def small_manifest(root, config):
    create_task(root)
    return build_manifest(root, replace(config, num_views=2), suites=["libero_object"],
                          frame_stride=2, window_stride=4)


def test_manifest_episode_split_and_train_only_statistics(config, tmp_path):
    manifest = small_manifest(tmp_path, config)
    episode_splits = {}
    for row in manifest["windows"]:
        episode_splits.setdefault(row["episode"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in episode_splits.values())
    with h5py.File(manifest["files"][0]["path"], "r") as handle:
        values = np.concatenate([handle["data"][name]["actions"][:]
                                 for name, splits in episode_splits.items() if splits == {"train"}])
    np.testing.assert_allclose(manifest["normalization"]["mean"], values.mean(0), rtol=1e-5)
    assert manifest["normalization"]["fit_split"] == "train"
    assert all(manifest["counts"][key] > 0 for key in ("train", "validation", "test"))


def test_manifest_integrity_and_dataset_mutation(config, tmp_path):
    manifest = small_manifest(tmp_path, config)
    path = tmp_path / "manifest.json"
    atomic_json(path, manifest)
    loaded = load_manifest(path)
    dataset = LiberoDataset(loaded, "validation", image_cache_mib=0)
    sample = dataset[0]
    assert sample["current"].shape == (2, 3, 32, 32)
    assert sample["future"].shape == (2, 2, 3, 32, 32)
    dataset.close()
    source = loaded["files"][0]["path"]
    with h5py.File(source, "r+") as handle:
        handle.attrs["fixture_mutation"] = True
    with pytest.raises(ValueError, match="changed"):
        dataset[0]
    loaded["seed"] += 1
    atomic_json(path, loaded)
    with pytest.raises(ValueError, match="hash"):
        load_manifest(path)


def test_sampler_exact_mid_epoch_resume():
    first = StatefulBatchSampler(17, 6, 19)
    first.next()
    state = first.state_dict()
    second = StatefulBatchSampler(17, 6, 19)
    second.load_state_dict(state)
    for _ in range(12):
        assert first.next() == second.next()
    assert first.epoch == second.epoch


def test_manifest_excludes_short_demos_and_rejects_missing_camera(config, tmp_path):
    path = create_task(tmp_path)
    with h5py.File(path, "r+") as handle:
        del handle["data/demo_0/obs/eye_in_hand_rgb"]
    with pytest.raises(ValueError, match="Missing"):
        build_manifest(tmp_path, replace(config, num_views=2), suites=["libero_object"], frame_stride=2)


@pytest.mark.parametrize("frequency", [10, 30, float("nan"), True, "20"])
def test_libero_manifest_rejects_unverified_control_frequency(config, tmp_path, frequency):
    path = create_task(tmp_path)
    with h5py.File(path, "r+") as handle:
        handle["data"].attrs["env_args"] = json.dumps({"env_kwargs": {"control_freq": frequency}})
    with pytest.raises(ValueError, match="20 Hz"):
        build_manifest(tmp_path, replace(config, num_views=2), suites=["libero_object"], frame_stride=2)


@pytest.mark.parametrize("key,shape,dtype,message", [
    ("agentview_rgb", (24, 16, 16, 3), np.float32, "uint8 RGB"),
    ("eye_in_hand_rgb", (24, 16, 16, 4), np.uint8, "uint8 RGB"),
    ("ee_states", (24, 5), np.float32, "state"),
    ("gripper_states", (24, 3), np.float32, "state"),
])
def test_libero_manifest_rejects_incompatible_observation_schema(config, tmp_path, key, shape, dtype, message):
    path = create_task(tmp_path)
    with h5py.File(path, "r+") as handle:
        observations = handle["data/demo_0/obs"]
        del observations[key]
        observations[key] = np.zeros(shape, dtype=dtype)
    with pytest.raises(ValueError, match=message):
        build_manifest(tmp_path, replace(config, num_views=2), suites=["libero_object"], frame_stride=2)


@pytest.mark.parametrize("dimensions", [{"action_dim": 6}, {"state_dim": 7}])
def test_libero_manifest_rejects_non_libero_dimensions_before_reading_data(config, tmp_path, dimensions):
    with pytest.raises(ValueError, match="action_dim=7 and state_dim=8"):
        build_manifest(tmp_path, replace(config, num_views=2, **dimensions), suites=["libero_object"])
