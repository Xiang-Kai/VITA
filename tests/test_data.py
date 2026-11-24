import json
from dataclasses import replace

import h5py
import numpy as np
import pytest
import torch

from vita.data import ActionNormalizer, load_libero_windows, synthetic_batch, tensor_fingerprint


def fixture_hdf5(path, frames=50):
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["problem_info"] = json.dumps({"language_instruction": "synthetic test instruction"})
        data.attrs["env_args"] = json.dumps({"env_kwargs": {"control_freq": 20}})
        for i in range(3):
            demo = data.create_group(f"demo_{i}")
            demo["actions"] = np.arange(frames * 7, dtype=np.float32).reshape(frames, 7) / 100
            obs = demo.create_group("obs")
            obs["agentview_rgb"] = np.full((frames, 16, 16, 3), i * 50, dtype=np.uint8)
            obs["ee_states"] = np.zeros((frames, 6), dtype=np.float32)
            obs["gripper_states"] = np.zeros((frames, 2), dtype=np.float32)


def test_action_normalizer_roundtrip_and_constant_dimension():
    actions = torch.randn(4, 12, 7)
    actions[..., -1] = 1
    normalizer = ActionNormalizer.fit(actions)
    normalized = normalizer.normalize(actions)
    assert torch.isfinite(normalized).all()
    torch.testing.assert_close(normalizer.inverse(normalized), actions)
    assert normalized[..., -1].abs().sum() == 0


def test_libero_windows_are_episode_bounded_and_read_only(config, tmp_path):
    path = tmp_path / "synthetic_fixture.hdf5"
    fixture_hdf5(path)
    initial = path.read_bytes()
    batch, normalizer, metadata = load_libero_windows(path, config, batch_size=2, frame_stride=20)
    assert path.read_bytes() == initial
    assert batch["current"].shape == (2, 3, 32, 32)
    assert batch["future"].shape == (2, 2, 3, 32, 32)
    assert metadata["future_frame_interval_seconds"] == 1.0
    for window in metadata["windows"]:
        assert max(window["future_indices"]) < 50
        assert window["action_stop_exclusive"] <= 50
    assert torch.isfinite(normalizer.inverse(batch["actions"])).all()


def test_short_episodes_are_rejected(config, tmp_path):
    path = tmp_path / "synthetic_short.hdf5"
    fixture_hdf5(path, frames=10)
    with pytest.raises(ValueError, match="complete windows"):
        load_libero_windows(path, config, batch_size=2, frame_stride=20)


def test_bad_state_dimension_fails(config, tmp_path):
    path = tmp_path / "synthetic_fixture.hdf5"
    fixture_hdf5(path)
    with pytest.raises(ValueError, match="state_dim"):
        load_libero_windows(path, replace(config, state_dim=7), batch_size=2, frame_stride=20)


def test_synthetic_fixture_is_reproducible(config):
    first = synthetic_batch(config, seed=4)
    assert tensor_fingerprint(first) == tensor_fingerprint(synthetic_batch(config, seed=4))
    assert tensor_fingerprint(first) != tensor_fingerprint(synthetic_batch(config, seed=5))
