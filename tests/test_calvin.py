"""Entirely synthetic CPU environment/oracle. No CALVIN, network, GPU or robot."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from vita.calvin_eval import chain_metrics, execute_sequence, observation_batch


class SyntheticEnv:
    def __init__(self):
        self.steps, self.resets, self.actions = 0, 0, []

    def reset(self, **kwargs):
        self.resets += 1
        self.steps = 0

    def get_info(self):
        return {"steps": self.steps}

    def get_obs(self):
        return {"fixture": True}

    def step(self, action):
        self.steps += 1
        self.actions.append(action)
        # done=True does NOT establish task success and does not reset a CALVIN chain.
        return self.get_obs(), 0, True, self.get_info()


class SyntheticOracle:
    def get_task_info_for_set(self, start, current, tasks):
        return tasks if tasks != {"failure"} and current["steps"] - start["steps"] >= 2 else set()


def test_chain_metrics_match_survival_identity():
    result = chain_metrics([0, 1, 3, 5])
    assert result["chain_success_rates"] == [0.75, 0.5, 0.5, 0.25, 0.25]
    assert result["average_length"] == 2.25
    with pytest.raises(ValueError):
        chain_metrics([6])


def test_sequence_preserves_scene_clears_chunks_and_stops_on_failure():
    env = SyntheticEnv()
    instructions = []

    def policy(observation, instruction):
        instructions.append(instruction)
        return np.ones((4, 7)) * 2

    tasks = ["first", "second", "failure", "unreached"]
    result = execute_sequence(env, SyntheticOracle(), (None, None), tasks,
                              {task: [task] for task in tasks}, policy,
                              max_steps=3, action_chunk=4)
    assert env.resets == 1
    assert result["completed_tasks"] == 2
    assert instructions == ["first", "second", "failure"]
    assert len(env.actions) == 7
    assert all((np.abs(action) <= 1).all() for action in env.actions)
    assert result["tasks"][-1]["success"] is False


def test_bad_actions_raise_not_count_as_success():
    with pytest.raises(ValueError, match="Invalid CALVIN"):
        execute_sequence(SyntheticEnv(), SyntheticOracle(), (None, None), ["task"], {"task": ["task"]},
                         lambda *_: np.full((1, 7), np.nan))


def test_calvin_observation_rgb_orientation_and_state(config):
    config = replace(config, state_dim=15, num_views=2)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:16] = 255
    batch = observation_batch({"rgb_obs": {"rgb_static": image, "rgb_gripper": image},
                               "robot_obs": np.arange(15)}, config, "fixture", "cpu")
    assert batch["current"].shape == (1, 2, 3, 32, 32)
    assert batch["current"][0, 0, :, :16].mean() == 1
    torch.testing.assert_close(batch["states"][0], torch.arange(15).float())
