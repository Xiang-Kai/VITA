"""Google Robot / WidowX SimplerEnv adapter. Simulation only, no hardware drivers."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import ActionNormalizer
from .dataset import atomic_json
from .deploy import data_contract, load_action_policy
from .libero_eval import wilson_interval
from .sources import vector
from .train import prepare_device
from .runtime import machine_metadata


TASKS = {
    "google_robot": ("google_robot_pick_coke_can", "google_robot_move_near", "google_robot_open_drawer"),
    "widowx": ("widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube",
               "widowx_put_eggplant_in_basket"),
}


def euler_xyz_to_axis_angle(angles):
    """XYZ roll/pitch/yaw with R=Rz(yaw) Ry(pitch) Rx(roll), in radians."""
    r, p, y = np.asarray(angles, dtype=np.float64) / 2
    cr, cp, cy, sr, sp, sy = np.cos(r), np.cos(p), np.cos(y), np.sin(r), np.sin(p), np.sin(y)
    q = np.array([sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                  cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy])
    if q[3] < 0:
        q = -q
    norm = np.linalg.norm(q[:3])
    return q[:3] * (2 * np.arctan2(norm, q[3]) / norm if norm > 1e-12 else 2)


@dataclass(frozen=True)
class ControlSpec:
    robot: str
    source_action_convention: str
    translation_scale: tuple
    rotation_scale: tuple
    rotation_representation: str
    gripper_mode: str
    sticky_steps: int = 1
    sticky_threshold: float = 0.5

    def __post_init__(self):
        if self.robot not in TASKS or not self.source_action_convention.strip():
            raise ValueError("Explicit robot and training action convention required")
        for scale in (self.translation_scale, self.rotation_scale):
            if np.asarray(scale).shape != (3,) or not np.isfinite(scale).all() or np.any(np.asarray(scale) <= 0):
                raise ValueError("Explicit positive XYZ scales required")
        if self.rotation_representation not in {"euler_xyz", "axis_angle"}:
            raise ValueError("Unknown rotation convention")
        if self.gripper_mode not in {"open_fraction", "closed_fraction", "signed_open", "relative_positive_close"}:
            raise ValueError("Unknown gripper convention")
        if self.sticky_steps < 1 or not np.isfinite(self.sticky_threshold) or self.sticky_threshold <= 0:
            raise ValueError("Invalid sticky gripper parameters")


class ControlAdapter:
    def __init__(self, spec):
        self.spec = spec
        self.reset()

    def reset(self):
        self.remaining, self.sticky = 0, 0.0

    def __call__(self, raw):
        raw = np.asarray(raw, dtype=np.float64)
        if raw.shape != (7,) or not np.isfinite(raw).all():
            raise ValueError("Expected finite 7D denormalized source action")
        spec = self.spec
        rotation = raw[3:6] * spec.rotation_scale
        if spec.rotation_representation == "euler_xyz":
            rotation = euler_xyz_to_axis_angle(rotation)
        gripper = raw[6]
        if spec.gripper_mode in {"open_fraction", "closed_fraction"}:
            gripper = 2 * gripper - 1 if spec.gripper_mode == "open_fraction" else 1 - 2 * gripper
        elif spec.gripper_mode == "relative_positive_close":
            if self.remaining == 0 and abs(gripper) > spec.sticky_threshold:
                self.sticky, self.remaining = -float(np.sign(gripper)), spec.sticky_steps
            gripper = self.sticky if self.remaining else 0.0
            self.remaining = max(0, self.remaining - 1)
        return np.concatenate([raw[:3] * spec.translation_scale, rotation, [gripper]]).astype(np.float32)


def observed_success(info):
    if "success" not in info or not isinstance(info["success"], (bool, np.bool_)):
        raise ValueError("Simulator must report an explicit boolean success; reward/done is not success")
    return bool(info["success"])


def execute_episode(env, policy, adapter, *, seed, max_steps=300, action_chunk=5, reset_options=None):
    if min(max_steps, action_chunk) < 1:
        raise ValueError("Invalid rollout budget")
    observation, _ = env.reset(seed=seed, options=reset_options or {})
    adapter.reset()
    pending, previous_instruction = deque(), None
    calls, clipped, steps, success = 0, 0, 0, False
    for _ in range(max_steps):
        instruction = env.unwrapped.get_language_instruction()
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Missing simulator language instruction")
        if instruction != previous_instruction:
            pending.clear()
            previous_instruction = instruction
        if not pending:
            chunk = np.asarray(policy(observation, instruction), dtype=np.float32)
            if chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) < action_chunk or not np.isfinite(chunk).all():
                raise ValueError("Invalid predicted action chunk")
            pending.extend(chunk[:action_chunk])
            calls += 1
        action = adapter(pending.popleft())
        low, high = np.asarray(env.action_space.low), np.asarray(env.action_space.high)
        if (low.shape != (7,) or high.shape != (7,) or not np.isfinite(low).all()
                or not np.isfinite(high).all() or np.any(low > high)):
            raise ValueError("SimplerEnv controller must expose its seven-dimensional bounds")
        clipped += int(((action < low) | (action > high)).sum())
        observation, _, terminated, truncated, info = env.step(np.clip(action, low, high))
        steps += 1
        success = observed_success(info)
        if success or terminated or truncated:
            break
    return {"success": success, "environment_steps": steps, "policy_calls": calls,
            "clipped_action_fraction": clipped / (steps * 7), "seed": seed}


def observation_batch(observation, image, config, state_fields, device):
    if config.num_views != 1 or config.action_dim != 7:
        raise ValueError("The paper's SimplerEnv recipe requires one camera and seven action dimensions")
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected RGB HWC uint8 image from the SimplerEnv helper")
    values = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float() / 255
    values = F.interpolate(values, config.image_size, mode="bilinear", align_corners=False, antialias=True)
    try:
        state = vector(observation, state_fields)
    except KeyError as error:
        raise ValueError("Observation state mapping refers to a missing field") from error
    if state is None:
        state = np.zeros(config.state_dim, dtype=np.float32)
    if state.shape != (config.state_dim,) or not np.isfinite(state).all():
        raise ValueError("Observation state mapping differs from training")
    return values[:, None].to(device), torch.tensor(state, dtype=torch.float32, device=device)[None]


def run(args, *, synthetic_fixture=False):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if min(args.episodes_per_task, args.max_steps, args.action_chunk) < 1:
        raise ValueError("Invalid evaluation budget")
    prepare_device(args.device, args.memory_gib)
    contract = json.loads(Path(args.control_spec).read_text())
    spec = ControlSpec(**contract["control"])
    state_fields = contract["state_fields"]
    tasks = args.tasks or list(TASKS[spec.robot])
    if not tasks or any(task not in TASKS[spec.robot] for task in tasks) or len(set(tasks)) != len(tasks):
        raise ValueError("Task list does not match the selected paper benchmark branch")
    model, checkpoint = load_action_policy(args.checkpoint, args.device)
    data = data_contract(checkpoint, args.data_source)
    if checkpoint["stage"] != "finetune" or args.action_chunk > model.config.action_horizon:
        raise ValueError("Need a fine-tuned checkpoint with sufficient action horizon")
    if (data.get("action_convention") != spec.source_action_convention
            or data.get("action_columns") != list(range(7))
            or data.get("state_columns") not in ([], list(range(model.config.state_dim)))):
        raise ValueError("Checkpoint and simulator control/state contracts disagree")
    if bool(data["state_columns"]) != bool(state_fields):
        raise ValueError("Missing state may only be zero-filled when training also masked all states")
    stats = data["normalization"]
    normalizer = ActionNormalizer(torch.tensor(stats["mean"]), torch.tensor(stats["std"]))
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    output.mkdir(parents=True)
    result = {"status": "running", "backend": (
        "synthetic_protocol_fixture" if synthetic_fixture else "real_SimplerEnv_simulation"), "physical_robot": False,
              "synthetic_fixture": synthetic_fixture,
              "robot": spec.robot, "control_spec": contract, **machine_metadata(),
              "checkpoint": str(Path(args.checkpoint).resolve()), "benchmark_reproduced": False,
              "action_chunk": args.action_chunk, "max_steps": args.max_steps, "episodes": []}
    atomic_json(output / "rollouts.json", result)
    try:
        for task in tasks:
            env = simpler_env.make(task)
            try:
                def policy(observation, instruction):
                    images, states = observation_batch(
                        observation, get_image_from_maniskill2_obs_dict(env, observation),
                        model.config, state_fields, args.device)
                    return normalizer.inverse(model.act([instruction], images, states))[0].float().cpu().numpy()
                for episode in range(args.episodes_per_task):
                    record = execute_episode(env, policy, ControlAdapter(spec), seed=args.seed + episode,
                                             max_steps=args.max_steps, action_chunk=args.action_chunk)
                    result["episodes"].append({"task": task, "episode": episode, **record})
                    atomic_json(output / "rollouts.json", result)
            finally:
                env.close()
        metrics = {}
        for task in tasks:
            records = [r for r in result["episodes"] if r["task"] == task]
            successes = sum(r["success"] for r in records)
            metrics[task] = {"episodes": len(records), "success_rate": successes / len(records),
                             "wilson_95": wilson_interval(successes, len(records))}
        result.update(status="synthetic_evaluation_completed" if synthetic_fixture else "simulation_evaluation_completed",
                      per_task=metrics,
                      macro_success_rate=float(np.mean([m["success_rate"] for m in metrics.values()])),
                      scope="seeded make(task) protocol; not a recovered author pose/appearance sweep")
        atomic_json(output / "rollouts.json", result)
        return result
    except Exception as error:
        result.update(status="failed", error=type(error).__name__, message=str(error))
        atomic_json(output / "rollouts.json", result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--control-spec", required=True)
    parser.add_argument("--data-source")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--memory-gib", type=float, default=6)
    parser.add_argument("--episodes-per-task", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--action-chunk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    print(json.dumps(run(parser.parse_args())))


if __name__ == "__main__":
    main()
