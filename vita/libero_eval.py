"""Real MuJoCo LIBERO rollouts. No physical robot drivers, services or API peers."""
from __future__ import annotations

import argparse
from collections import deque
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import ActionNormalizer
from .dataset import SUITES, atomic_json
from .deploy import data_contract, file_sha256, load_action_policy
from .train import core_hashes, prepare_device
from .runtime import machine_metadata


MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
CAMERA_KEYS = {"agentview_rgb": "agentview_image", "eye_in_hand_rgb": "robot0_eye_in_hand_image"}


def wilson_interval(successes: int, total: int):
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Invalid episode counts")
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return [0.0 if successes == 0 else max(0.0, center - half),
            1.0 if successes == total else min(1.0, center + half)]


def observation_batch(observation, cameras, image_size, instruction, quat_to_axisangle, device):
    images = []
    for camera in cameras:
        key = CAMERA_KEYS[camera]
        image = np.asarray(observation[key])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Unexpected simulator camera shape")
        pixels = torch.from_numpy(image.copy()).permute(2, 0, 1).flip(-2).float() / 255
        images.append(F.interpolate(pixels[None], image_size, mode="bilinear",
                                    align_corners=False, antialias=True)[0])
    state = np.concatenate([
        observation["robot0_eef_pos"], quat_to_axisangle(observation["robot0_eef_quat"]),
        observation["robot0_gripper_qpos"],
    ])
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("Unexpected/nonfinite simulator state")
    return {
        "instructions": [instruction],
        "current": torch.stack(images)[None].to(device),
        "states": torch.tensor(state, dtype=torch.float32, device=device)[None],
        "view_mask": torch.ones(1, len(images), dtype=torch.bool, device=device),
    }


def execute_episode(env, initial_state, policy, *, max_steps, settle_steps, action_chunk):
    """A failed step/invalid control raises; it is never recorded as a success."""
    if min(max_steps, action_chunk) < 1 or settle_steps < 0:
        raise ValueError("Invalid rollout lengths")
    env.reset()
    observation = env.set_init_state(initial_state)
    for _ in range(settle_steps):
        observation, _, _, _ = env.step(np.array([0., 0., 0., 0., 0., 0., -1.]))
    actions = deque()
    saturated_values, total_values, calls = 0, 0, 0
    success = bool(env.check_success())
    executed = 0
    for _ in range(max_steps):
        if success:
            break
        if not actions:
            chunk = np.asarray(policy(observation), dtype=np.float32)
            if chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) < action_chunk or not np.isfinite(chunk).all():
                raise ValueError("Invalid normalized-controller action chunk")
            actions.extend(chunk[:action_chunk])
            calls += 1
        action = actions.popleft()
        saturated_values += int((np.abs(action) > 1).sum())
        total_values += len(action)
        # Dataset controls are already OSC_POSE normalized [-1,1]; no extra gripper sign flip.
        action = np.clip(action, -1, 1)
        observation, _, done, _ = env.step(action)
        executed += 1
        success = bool(env.check_success())
        if done and not success:
            break
    return {"success": success, "environment_steps": executed, "policy_calls": calls,
            "clipped_action_fraction": saturated_values / max(total_values, 1)}


def configure_local_libero(source: Path, output: Path, dataset_root: Path):
    package = source / "libero/libero"
    paths = {"benchmark_root": package, "bddl_files": package / "bddl_files",
             "init_states": package / "init_files", "datasets": dataset_root, "assets": package / "assets"}
    for key, value in paths.items():
        if not value.exists():
            raise FileNotFoundError(f"LIBERO {key}: {value}")
    config = output / "libero_config"
    config.mkdir()
    # JSON is valid YAML. Isolate LIBERO's paths instead of editing ~/.libero or a shared checkout.
    (config / "config.yaml").write_text(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    os.environ["LIBERO_CONFIG_PATH"] = str(config)
    sys.path.insert(0, str(source))


def run(args):
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new rollout output directory")
    output.mkdir(parents=True, exist_ok=True)
    try:
        return _run(args)
    except Exception as error:
        receipt = output / "rollouts.json"
        result = json.loads(receipt.read_text()) if receipt.exists() else {
            "episodes": [], "benchmark_reproduced": False, "physical_robot": False,
            "runtime_source_sha256": core_hashes(), "phase": "initialization",
        }
        result.update(status="failed", error=type(error).__name__, message=str(error), success_rate=None)
        atomic_json(receipt, result)
        raise


def _run(args):
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new rollout output directory")
    output.mkdir(parents=True, exist_ok=True)
    if min(args.episodes_per_task, args.action_chunk) < 1 or args.settle_steps < 0:
        raise ValueError("Invalid evaluation lengths")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max_steps must be positive")
    prepare_device(args.device, 4.0)
    configure_local_libero(Path(args.libero_root).resolve(), output, Path(args.dataset_root).resolve())
    # Backend/device must be selected before importing the simulator.
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle

    model, checkpoint = load_action_policy(args.checkpoint, args.device, require_source_match=True)
    if checkpoint["stage"] != "finetune":
        raise ValueError("Simulator evaluation requires a fine-tuned checkpoint")
    if args.action_chunk > model.config.action_horizon:
        raise ValueError("Executed chunk exceeds predicted action horizon")
    if model.config.state_dim != 8 or model.config.action_dim != 7:
        raise ValueError("Only verified 8D-state / 7D-control LIBERO convention is supported")
    contract = data_contract(checkpoint, getattr(args, "data_source", None))
    cameras = contract["cameras"]
    if any(camera not in CAMERA_KEYS for camera in cameras):
        raise ValueError("Unsupported camera convention")
    stats = contract["normalization"]
    normalizer = ActionNormalizer(torch.tensor(stats["mean"]), torch.tensor(stats["std"]))
    episodes = []
    latency_ms = []
    result = {
        "status": "running", "backend": "real_LIBERO_MuJoCo_simulation", "physical_robot": False,
        **machine_metadata(),
        "libero_source": str(Path(args.libero_root).resolve()),
        "dependency_versions": {name: importlib.metadata.version(name)
                                for name in ("torch", "robosuite", "mujoco", "numpy")},
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "manifest_sha256": checkpoint["manifest_sha256"],
        "runtime_source_sha256": core_hashes(),
        "source_checkpoint_sha256": checkpoint.get("source_checkpoint_sha256"),
        "benchmark_reproduced": False, "suites": args.suites,
        "episodes_per_task": args.episodes_per_task, "max_tasks": args.max_tasks,
        "max_steps_override": args.max_steps, "default_horizons": MAX_STEPS,
        "settle_steps": args.settle_steps, "action_chunk": args.action_chunk,
        "seed": args.seed, "render_backend": os.environ["MUJOCO_GL"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "protocol": "fixed init states, task order 0; horizons are explicit evaluator conventions, not recovered author settings",
        "episodes": episodes,
    }
    atomic_json(output / "rollouts.json", result)
    try:
        for suite_name in args.suites:
            suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
            task_count = min(suite.n_tasks, args.max_tasks) if args.max_tasks else suite.n_tasks
            for task_index in range(task_count):
                task = suite.get_task(task_index)
                initial_states = suite.get_task_init_states(task_index)
                if args.episodes_per_task > len(initial_states):
                    raise ValueError("Requested more trials than available initial states; refusing duplicates")
                env = None
                try:
                    env = OffScreenRenderEnv(
                        bddl_file_name=suite.get_task_bddl_file_path(task_index),
                        camera_heights=128, camera_widths=128, control_freq=20,
                        render_gpu_device_id=-1,
                    )
                    for episode_id in range(args.episodes_per_task):
                        seed = args.seed + episode_id
                        np.random.seed(seed)
                        env.seed(seed)

                        def policy(observation):
                            batch = observation_batch(observation, cameras, model.config.image_size,
                                                      task.language, quat2axisangle, args.device)
                            if args.device.startswith("cuda"):
                                torch.cuda.synchronize(args.device)
                            started = time.perf_counter()
                            prediction = model.act(batch["instructions"], batch["current"],
                                                   batch["states"], batch["view_mask"])
                            if args.device.startswith("cuda"):
                                torch.cuda.synchronize(args.device)
                            latency_ms.append((time.perf_counter() - started) * 1000)
                            return normalizer.inverse(prediction)[0].float().cpu().numpy()

                        record = execute_episode(
                            env, initial_states[episode_id], policy,
                            max_steps=args.max_steps or MAX_STEPS[suite_name],
                            settle_steps=args.settle_steps, action_chunk=args.action_chunk,
                        )
                        record.update(suite=suite_name, task=task.name, task_index=task_index,
                                      episode_id=episode_id, seed=seed)
                        episodes.append(record)
                        atomic_json(output / "rollouts.json", result)
                        print(json.dumps(record), flush=True)
                finally:
                    if env is not None:
                        env.close()
        successes = sum(record["success"] for record in episodes)
        result.update(status="simulation_evaluation_completed", successes=successes, trials=len(episodes),
                      success_rate=successes / len(episodes), wilson_95=wilson_interval(successes, len(episodes)),
                      latency_ms_mean=float(np.mean(latency_ms)) if latency_ms else None,
                      scope="bounded_smoke" if args.max_tasks or args.max_steps or args.episodes_per_task < 50
                      else "configured_suite_evaluation_not_a_claim_of_paper_reproduction")
        result["per_suite"] = {}
        for suite in args.suites:
            selected = [row for row in episodes if row["suite"] == suite]
            count = sum(row["success"] for row in selected)
            result["per_suite"][suite] = {"trials": len(selected),
                                         "successes": count, "success_rate": count / len(selected),
                                         "wilson_95": wilson_interval(count, len(selected))}
        atomic_json(output / "rollouts.json", result)
        return result
    except Exception as error:
        result.update(status="failed", error=type(error).__name__, message=str(error), success_rate=None)
        atomic_json(output / "rollouts.json", result)
        raise


def main():
    parser = argparse.ArgumentParser(description="Actual LIBERO simulator rollouts; no hardware control")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-source", help="Explicit paired source in a mixture checkpoint")
    parser.add_argument("--libero-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--action-chunk", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.max_tasks is not None and args.max_tasks < 1:
        raise ValueError("max_tasks must be positive")
    print(json.dumps(run(args)), flush=True)


if __name__ == "__main__":
    main()
