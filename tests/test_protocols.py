"""Entirely synthetic peers/recordings, offline CPU, no real robot or provider."""
from dataclasses import replace
from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest
import torch

from vita import VITA
from vita.realworld import EVALUATION_TASKS, TRAIN_TASKS, aggregate_trials, recorded_inference
from vita.simpler_eval import (ControlAdapter, ControlSpec, euler_xyz_to_axis_angle,
                              execute_episode, observation_batch, observed_success, run, TASKS)


def spec(**kwargs):
    values = {"robot": "google_robot", "source_action_convention": "synthetic_fixture",
              "translation_scale": (1, 1, 1), "rotation_scale": (1, 1, 1),
              "rotation_representation": "axis_angle", "gripper_mode": "signed_open"}
    return ControlSpec(**{**values, **kwargs})


@pytest.mark.parametrize("mode,input_value,expected", [
    ("open_fraction", 1, 1), ("open_fraction", 0, -1),
    ("closed_fraction", 1, -1), ("closed_fraction", 0, 1), ("signed_open", -1, -1),
])
def test_explicit_gripper_semantics(mode, input_value, expected):
    raw = np.zeros(7)
    raw[-1] = input_value
    assert ControlAdapter(spec(gripper_mode=mode))(raw)[-1] == expected


def test_sticky_gripper_reset_and_rotation():
    adapter = ControlAdapter(spec(gripper_mode="relative_positive_close", sticky_steps=3))
    value = np.zeros(7)
    value[-1] = 0.9
    assert adapter(value)[-1] == -1
    value[-1] = 0
    assert [adapter(value)[-1] for _ in range(3)] == [-1, -1, 0]
    adapter.reset()
    assert adapter(value)[-1] == 0
    np.testing.assert_allclose(euler_xyz_to_axis_angle([0, 0, np.pi]), [0, 0, np.pi], atol=1e-7)
    np.testing.assert_array_equal(euler_xyz_to_axis_angle([0, 0, 0]), [0, 0, 0])


class FakeEnv:
    def __init__(self, fail=False):
        self.unwrapped = self
        self.action_space = SimpleNamespace(low=np.full(7, -1.0), high=np.ones(7))
        self.fail = fail
        self.n = 0
    def reset(self, **kwargs):
        self.n = 0
        return {}, {}
    def get_language_instruction(self):
        return "synthetic task one" if self.n < 1 else "synthetic task two"
    def step(self, action):
        self.n += 1
        return {}, 100, self.n == 3, False, {"success": self.n == 3 and not self.fail}


def test_simpler_success_oracle_chunk_invalidation_and_failure():
    calls = []
    def policy(observation, instruction):
        calls.append(instruction)
        return np.zeros((5, 7))
    result = execute_episode(FakeEnv(), policy, ControlAdapter(spec()), seed=0)
    assert result["success"] and result["policy_calls"] == 2 and result["environment_steps"] == 3
    assert calls == ["synthetic task one", "synthetic task two"]
    assert not execute_episode(FakeEnv(fail=True), policy, ControlAdapter(spec()), seed=0)["success"]
    with pytest.raises(ValueError, match="success"):
        observed_success({"reward": 100, "done": True})


def test_invalid_simulator_bounds_fail_before_any_step():
    env = FakeEnv()
    env.action_space.low[0] = np.nan
    with pytest.raises(ValueError, match="bounds"):
        execute_episode(env, lambda *_: np.zeros((5, 7)), ControlAdapter(spec()), seed=0)
    assert env.n == 0


def test_simpler_observation_requires_explicit_mapping(config):
    images, states = observation_batch(
        {"state": np.arange(8)}, np.zeros((8, 8, 3), np.uint8), config,
        [{"path": "state", "indices": list(range(8))}], "cpu")
    assert images.shape == (1, 1, 3, 32, 32)
    torch.testing.assert_close(states[0], torch.arange(8).float())
    with pytest.raises(ValueError, match="state mapping"):
        observation_batch({}, np.zeros((8, 8, 3), np.uint8), config, [{"path": "bad"}], "cpu")


def test_realworld_metrics_never_turn_missing_or_failed_trials_into_success():
    assert len(TRAIN_TASKS) == 12 and len(EVALUATION_TASKS) == 6
    rows = [{"task": "color_match", "trial_id": str(i), "success": i == 0, "status": "completed",
             "evidence": "synthetic://fixture", "reviewer": "synthetic_unit_test", "evidence_kind": "synthetic"}
            for i in range(2)]
    result = aggregate_trials(rows, expected_per_task=2, tasks=["color_match"])
    assert result["macro_success_rate"] == 0.5 and result["evidence_kind"] == "synthetic"
    assert aggregate_trials(rows, expected_per_task=100, tasks=["color_match"])["macro_success_rate"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_trials(rows + rows, tasks=["color_match"])
    rows[0]["status"] = "error"
    with pytest.raises(ValueError, match="failed trial"):
        aggregate_trials(rows, tasks=["color_match"])
    rows[0].update(status="completed", evidence_kind="recorded")
    with pytest.raises(ValueError, match="mix"):
        aggregate_trials(rows, tasks=["color_match"])


def test_recorded_inference_is_only_array_output(config, tmp_path):
    config = replace(config, num_views=2, action_horizon=15)
    model = VITA(config).eval()
    checkpoint = tmp_path / "training.pt"
    torch.save({"schema_version": 2, "stage": "finetune", "model": model.state_dict(),
                "model_config": config.to_dict(), "data_manifest": {
                    "kind": "canonical_episodes", "action_convention": "UR5e:synthetic_fixture_NOT_hardware",
                    "action_columns": list(range(7)), "state_columns": list(range(config.state_dim)),
                    "normalization": {"mean": [0] * 7, "std": [1] * 7}}}, checkpoint)
    recording = tmp_path / "recording.npz"
    np.savez(recording, images=np.zeros((2, 2, 16, 16, 3), np.uint8),
             states=np.zeros((2, config.state_dim), np.float32),
             instructions=np.array(["synthetic", "synthetic"]), timestamps=np.array([0, 1.]))
    result = recorded_inference(checkpoint, recording, tmp_path / "predictions.npz")
    assert result["hardware_execution"] is False and result["success_metric"] is None
    with np.load(tmp_path / "predictions.npz", allow_pickle=False) as data:
        assert data["actions"].shape == (2, 15, 7)


@pytest.mark.parametrize("robot", ["google_robot", "widowx"])
def test_full_simpler_entrypoint_with_explicitly_mocked_library(config, tmp_path, monkeypatch, robot):
    model = VITA(config).eval()
    checkpoint = tmp_path / "training.pt"
    torch.save({"schema_version": 2, "stage": "finetune", "model": model.state_dict(),
                "model_config": config.to_dict(), "data_manifest": {
                    "kind": "canonical_episodes", "action_convention": "synthetic_fixture",
                    "action_columns": list(range(7)), "state_columns": [],
                    "normalization": {"mean": [0] * 7, "std": [1] * 7}}}, checkpoint)
    specification = tmp_path / "controls.json"
    specification.write_text(json.dumps({"control": spec(robot=robot).__dict__, "state_fields": []}))
    closed = []
    env = FakeEnv()
    env.close = lambda: closed.append(True)
    monkeypatch.setitem(sys.modules, "simpler_env", SimpleNamespace(make=lambda task: env))
    monkeypatch.setitem(sys.modules, "simpler_env.utils.env.observation_utils",
                        SimpleNamespace(get_image_from_maniskill2_obs_dict=lambda env, obs:
                                        np.zeros((16, 16, 3), np.uint8)))
    args = SimpleNamespace(output=str(tmp_path / "rollout"), device="cpu", memory_gib=4,
                           control_spec=str(specification), tasks=[TASKS[robot][0]], checkpoint=str(checkpoint),
                           data_source=None, episodes_per_task=2, seed=4, max_steps=5, action_chunk=2)
    result = run(args, synthetic_fixture=True)
    assert result["macro_success_rate"] == 1 and len(closed) == 1
    # This run uses a mocked simulator; retain the fixture provenance explicitly.
    assert result["benchmark_reproduced"] is False
    assert result["backend"] == "synthetic_protocol_fixture"
    args.output = str(tmp_path / "failed")
    env.step = lambda action: (_ for _ in ()).throw(RuntimeError("synthetic step failure"))
    with pytest.raises(RuntimeError, match="synthetic"):
        run(args, synthetic_fixture=True)
    receipt = json.loads((tmp_path / "failed/rollouts.json").read_text())
    assert receipt["status"] == "failed"
    assert "macro_success_rate" not in receipt
    assert len(closed) == 2
