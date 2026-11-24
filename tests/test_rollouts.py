"""Synthetic offline environment fixture; no robot, renderer or model service."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from vita.libero_eval import execute_episode, observation_batch, run, wilson_interval


class SyntheticEnvironment:
    def __init__(self, succeeds_at=None, raises=False, terminal_at=None):
        self.steps = 0
        self.succeeds_at = succeeds_at
        self.raises = raises
        self.terminal_at = terminal_at
        self.actions = []

    def reset(self):
        self.steps = 0

    def set_init_state(self, _):
        return {}

    def check_success(self):
        return self.succeeds_at is not None and self.steps >= self.succeeds_at

    def step(self, action):
        if self.raises:
            raise RuntimeError("synthetic environment step failure")
        self.steps += 1
        self.actions.append(action)
        return {}, 0, self.steps == self.terminal_at, {}


def test_success_uses_environment_predicate_not_done():
    env = SyntheticEnvironment(terminal_at=1)
    result = execute_episode(env, None, lambda _: np.zeros((4, 7)), max_steps=10, settle_steps=0, action_chunk=4)
    assert result["success"] is False
    assert result["environment_steps"] == 1


def test_chunking_and_action_clipping_are_recorded():
    env = SyntheticEnvironment(succeeds_at=5)
    result = execute_episode(env, None, lambda _: np.ones((4, 7)) * 2, max_steps=10, settle_steps=0, action_chunk=4)
    assert result["success"] is True
    assert result["policy_calls"] == 2
    assert result["clipped_action_fraction"] == 1
    assert np.max(env.actions) == 1


def test_failure_and_nonfinite_controls_are_not_successes():
    with pytest.raises(RuntimeError):
        execute_episode(SyntheticEnvironment(raises=True), None, lambda _: np.zeros((4, 7)),
                        max_steps=10, settle_steps=0, action_chunk=4)
    with pytest.raises(ValueError):
        execute_episode(SyntheticEnvironment(), None, lambda _: np.full((4, 7), np.nan),
                        max_steps=10, settle_steps=0, action_chunk=4)


def test_simulator_state_camera_convention():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[0] = 255
    obs = {"agentview_image": image, "robot0_eef_pos": [1, 2, 3],
           "robot0_eef_quat": [0, 0, 0, 1], "robot0_gripper_qpos": [0.1, -0.1]}
    batch = observation_batch(obs, ["agentview_rgb"], 8, "fixture", lambda _: np.array([4, 5, 6]), "cpu")
    np.testing.assert_allclose(batch["states"].numpy()[0], [1, 2, 3, 4, 5, 6, 0.1, -0.1])
    assert batch["current"][0, 0, :, -1].sum() == 24
    assert batch["current"][0, 0, :, 0].sum() == 0


def test_wilson_interval_and_invalid_count():
    low, high = wilson_interval(0, 10)
    assert low == 0 and 0 < high < 1
    low, high = wilson_interval(10, 10)
    assert 0 < low < 1 and high == 1
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_simulator_setup_failure_is_durable_and_never_overwritten(tmp_path, monkeypatch):
    args = SimpleNamespace(output=str(tmp_path / "rollout"), device="cpu", episodes_per_task=1,
                           action_chunk=1, settle_steps=10, max_steps=None)
    failure = RuntimeError("synthetic initialization failure; no renderer was created")

    def synthetic_setup(*_):
        raise failure

    monkeypatch.setattr("vita.libero_eval.prepare_device", synthetic_setup)
    with pytest.raises(RuntimeError) as caught:
        run(args)
    assert caught.value is failure
    path = tmp_path / "rollout/rollouts.json"
    before = path.read_bytes()
    result = json.loads(before)
    assert result["status"] == "failed" and result["phase"] == "initialization"
    assert result["episodes"] == [] and result["success_rate"] is None
    with pytest.raises(FileExistsError):
        run(args)
    assert path.read_bytes() == before
