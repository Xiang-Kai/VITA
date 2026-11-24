"""CALVIN ABC-D rollout integration using the public task oracle and sequences.

No physical controller exists here. The protocol core also accepts explicit
synthetic fixtures for CPU tests; those fixtures are not benchmark evidence.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.nn import functional as F

from .data import ActionNormalizer
from .dataset import atomic_json
from .deploy import data_contract, load_action_policy
from .train import prepare_device
from .runtime import machine_metadata


def chain_metrics(lengths, chain_length=5):
    if not lengths or chain_length < 1 or any(type(n) is not int or not 0 <= n <= chain_length for n in lengths):
        raise ValueError("Invalid completed-chain lengths")
    rates = [sum(n >= k for n in lengths) / len(lengths) for k in range(1, chain_length + 1)]
    average = sum(lengths) / len(lengths)
    if abs(sum(rates) - average) > 1e-12:
        raise AssertionError("CALVIN mean must equal the sum of chain survival rates")
    return {"sequences": len(lengths), "chain_success_rates": rates, "average_length": average}


def execute_sequence(env, oracle, initial_state, tasks, annotations, policy, *, max_steps=360, action_chunk=1):
    if max_steps < 1 or action_chunk < 1 or not tasks:
        raise ValueError("Invalid CALVIN rollout lengths")
    robot_obs, scene_obs = initial_state
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    records = []
    for task in tasks:
        start_info = env.get_info()
        observation = env.get_obs()
        instruction = annotations[task][0]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Missing CALVIN language annotation")
        # New instruction invalidates the remaining open-loop chunk; the scene is not reset.
        actions = deque()
        success, clipped, steps, calls = False, 0, 0, 0
        for _ in range(max_steps):
            if not actions:
                values = np.asarray(policy(observation, instruction), dtype=np.float32)
                if (values.ndim != 2 or values.shape[1] != 7 or len(values) < action_chunk
                        or not np.isfinite(values).all()):
                    raise ValueError("Invalid CALVIN relative-control chunk")
                actions.extend(values[:action_chunk])
                calls += 1
            action = actions.popleft().copy()
            clipped += int((np.abs(action[:6]) > 1).sum())
            action[:6] = np.clip(action[:6], -1, 1)
            action[6] = 1 if action[6] >= 0 else -1
            observation, _, _, current_info = env.step(action)
            steps += 1
            success = bool(oracle.get_task_info_for_set(start_info, current_info, {task}))
            if success:
                break
        records.append({"task": task, "success": success, "environment_steps": steps,
                        "policy_calls": calls, "clipped_motion_fraction": clipped / (steps * 6)})
        if not success:
            break
    return {"completed_tasks": sum(r["success"] for r in records), "tasks": records}


def observation_batch(observation, config, instruction, device):
    if config.state_dim != 15 or config.action_dim != 7 or config.num_views != 2:
        raise ValueError("CALVIN requires the verified 15D-state, 7D-rel_action, two-camera contract")
    images = []
    static_shape = np.asarray(observation["rgb_obs"]["rgb_static"]).shape
    for key in ("rgb_static", "rgb_gripper"):
        pixels = np.asarray(observation["rgb_obs"][key])
        if pixels.ndim != 3 or pixels.shape[-1] != 3 or pixels.dtype != np.uint8:
            raise ValueError("Expected top-left RGB uint8 CALVIN image")
        pixels = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float()
        if key == "rgb_gripper" and tuple(pixels.shape[1:]) != tuple(static_shape[:2]):
            # Match the canonical converter's explicit uint8 wrist-to-static resize.
            pixels = F.interpolate(pixels[None], static_shape[:2], mode="bilinear",
                                   align_corners=False, antialias=True)[0].round()
        pixels = pixels / 255
        images.append(F.interpolate(pixels[None], config.image_size, mode="bilinear",
                                    align_corners=False, antialias=True)[0])
    state = np.asarray(observation["robot_obs"])
    if state.shape != (15,) or not np.isfinite(state).all():
        raise ValueError("Invalid CALVIN robot_obs")
    return {"instructions": [instruction], "current": torch.stack(images)[None].to(device),
            "states": torch.tensor(state, dtype=torch.float32, device=device)[None],
            "view_mask": torch.ones(1, 2, dtype=torch.bool, device=device)}


def run(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.num_sequences < 1 or args.num_sequences > 1000 or min(args.max_steps, args.action_chunk) < 1:
        raise ValueError("Invalid evaluation budget")
    prepare_device(args.device, args.memory_gib)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = Path(args.calvin_root).resolve()
    sys.path.insert(0, str(root / "calvin_models"))
    sys.path.insert(0, str(root / "calvin_env"))
    from calvin_agent.evaluation.multistep_sequences import get_sequences
    from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
    from calvin_env.envs.play_table_env import get_env
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    model, checkpoint = load_action_policy(args.checkpoint, args.device)
    if checkpoint["stage"] != "finetune" or args.action_chunk > model.config.action_horizon:
        raise ValueError("Need a fine-tuned model and a valid executed horizon")
    manifest = data_contract(checkpoint, args.data_source)
    if (not manifest or manifest.get("kind") != "canonical_episodes"
            or not manifest.get("action_convention", "").startswith("CALVIN rel_actions:")
            or manifest["state_columns"] != list(range(15)) or manifest["action_columns"] != list(range(7))):
        raise ValueError("Checkpoint does not certify the CALVIN training data/action contract")
    stats = manifest["normalization"]
    normalizer = ActionNormalizer(torch.tensor(stats["mean"]), torch.tensor(stats["std"]))
    conf = root / "calvin_models/conf"
    oracle = instantiate(OmegaConf.load(conf / "callbacks/rollout/tasks/new_playtable_tasks.yaml"))
    annotations = OmegaConf.load(conf / "annotations/new_playtable_validation.yaml")
    # The public generator fixes its own sequence RNG; seed here also covers simulator/model libraries.
    sequences = get_sequences(args.num_sequences)
    output.mkdir(parents=True)
    result = {"status": "running", "backend": "real_CALVIN_simulation", "physical_robot": False,
              "benchmark_reproduced": False, **machine_metadata(),
              "checkpoint": str(Path(args.checkpoint).resolve()), "seed": args.seed,
              "max_steps_per_task": args.max_steps, "action_chunk": args.action_chunk,
              "calvin_root": str(root), "episodes": []}
    atomic_json(output / "rollouts.json", result)
    env = None
    try:
        env = get_env(Path(args.dataset_root).resolve() / "validation", show_gui=False)
        for index, (initial, tasks) in enumerate(sequences):
            def policy(observation, instruction):
                batch = observation_batch(observation, model.config, instruction, args.device)
                return normalizer.inverse(model.act(**{
                    "instructions": batch["instructions"], "current": batch["current"],
                    "states": batch["states"], "view_mask": batch["view_mask"],
                }))[0].float().cpu().numpy()

            record = execute_sequence(env, oracle, get_env_state_for_initial_condition(initial),
                                      tasks, annotations, policy, max_steps=args.max_steps,
                                      action_chunk=args.action_chunk)
            record["sequence_id"] = index
            record["requested_tasks"] = list(tasks)
            result["episodes"].append(record)
            atomic_json(output / "rollouts.json", result)
            print(json.dumps(record), flush=True)
        result.update(status="simulation_evaluation_completed",
                      metrics=chain_metrics([r["completed_tasks"] for r in result["episodes"]]),
                      scope="full_configured_protocol" if args.num_sequences == 1000 and args.max_steps == 360
                      else "bounded_integration_pilot")
        atomic_json(output / "rollouts.json", result)
        return result
    except Exception as error:
        result.update(status="failed", error=type(error).__name__, message=str(error), metrics=None)
        atomic_json(output / "rollouts.json", result)
        raise
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--memory-gib", type=float, default=6)
    parser.add_argument("--num-sequences", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument("--action-chunk", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    print(json.dumps(run(parser.parse_args())))


if __name__ == "__main__":
    main()
