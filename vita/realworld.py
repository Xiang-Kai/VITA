"""UR5e recorded-input inference and externally reviewed trial aggregation ONLY.

This module has no robot driver, socket, actuator, control loop or provider call.
Predicted arrays and synthetic trial fixtures cannot establish physical success.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import ActionNormalizer
from .dataset import atomic_json
from .deploy import data_contract, load_action_policy
from .libero_eval import wilson_interval


TRAIN_TASKS = {
    1: "select_celebrity_portrait", 2: "color_match_towel", 3: "color_match_plate",
    4: "fold_towel", 5: "stack_cups_bowls", 6: "error_correction",
    7: "wipe_plate_anti_tip", 8: "multi_step_transfer", 9: "return_original_position",
    10: "imitate_arrangement", 11: "categorize_object_type", 12: "categorize_object_color",
}
EVALUATION_TASKS = {
    "select_object": {"distribution": "ID", "criterion": "Select the named portrait and place a fruit on it."},
    "color_match": {"distribution": "ID", "criterion": "Place red/blue blocks on corresponding towels."},
    "object_map": {"distribution": "ID", "criterion": "Put red objects on towel and yellow objects in plate."},
    "visual_reason": {"distribution": "ID", "criterion": "Put fruit in the plate with most blue blocks."},
    "inverse_execution": {"distribution": "OOD", "criterion": "Transfer blocks to plate, then restore original positions."},
    "conditional_decision": {"distribution": "OOD", "criterion": "Place strawberry on plate when the displayed letter is B."},
}


def aggregate_trials(records, *, expected_per_task=100, tasks=None):
    tasks = list(EVALUATION_TASKS) if tasks is None else list(tasks)
    if (not tasks or len(set(tasks)) != len(tasks) or any(t not in EVALUATION_TASKS for t in tasks)
            or type(expected_per_task) is not int or expected_per_task < 1):
        raise ValueError("Invalid trial protocol")
    if not records:
        raise ValueError("No reviewed trial records")
    seen, kinds = set(), set()
    for row in records:
        identity = (row["task"], row["trial_id"])
        if identity in seen or row["task"] not in tasks:
            raise ValueError("Duplicate or out-of-protocol trial")
        if not row["trial_id"] or not row.get("evidence") or not row.get("reviewer"):
            raise ValueError("Trial needs identity, evidence reference and reviewer")
        if type(row["success"]) is not bool or row["status"] not in {"completed", "error"}:
            raise ValueError("An explicit reviewed outcome is required")
        if row["status"] == "error" and (row["success"] or not row.get("error")):
            raise ValueError("A failed trial cannot be relabeled successful")
        if row["evidence_kind"] not in {"synthetic", "recorded"}:
            raise ValueError("Unknown evidence provenance")
        seen.add(identity)
        kinds.add(row["evidence_kind"])
    if len(kinds) != 1:
        raise ValueError("Never mix synthetic fixtures and recorded trials")
    metrics = {}
    for task in tasks:
        rows = [r for r in records if r["task"] == task]
        successes = sum(row["success"] for row in rows)
        if len(rows) > expected_per_task:
            raise ValueError("Trial count exceeds the declared protocol")
        metrics[task] = {
            "distribution": EVALUATION_TASKS[task]["distribution"],
            "observed": len(rows), "expected": expected_per_task, "successes": successes,
            "errors": sum(row["status"] == "error" for row in rows),
            "success_rate": successes / len(rows) if rows else None,
            "wilson_95": wilson_interval(successes, len(rows)) if rows else None,
        }
    complete = all(row["observed"] == expected_per_task for row in metrics.values())
    return {"per_task": metrics, "complete_protocol": complete, "evidence_kind": next(iter(kinds)),
            "macro_success_rate": float(np.mean([r["success_rate"] for r in metrics.values()])) if complete else None,
            "benchmark_reproduced": False, "physical_execution_performed_by_this_tool": False,
            "scope": "external verdict aggregation, not automatic verification of the recorded evidence",
            "paper_ambiguity": "Use the six enumerated evaluation tasks, not the inconsistent eight-task sentence."}


@torch.no_grad()
def recorded_inference(checkpoint, recording, output, *, data_source=None, batch_size=1):
    """NPZ: RGB uint8 [N,2,H,W,3], states [N,S], instructions [N], timestamps [N]."""
    output = Path(output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    if output.suffix != ".npz" or batch_size < 1:
        raise ValueError("Need .npz output and positive batch size")
    model, saved = load_action_policy(checkpoint, "cpu")
    config = model.config
    data = data_contract(saved, data_source)
    if (saved["stage"] != "finetune" or config.action_dim != 7 or config.num_views != 2
            or config.action_horizon < 15 or not data.get("action_convention", "").startswith("UR5e:")
            or data.get("action_columns") != list(range(7))):
        raise ValueError("Need an explicit UR5e two-camera, 7D, horizon>=15 trained data contract")
    stats = data["normalization"]
    normalizer = ActionNormalizer(torch.tensor(stats["mean"]), torch.tensor(stats["std"]))
    with np.load(recording, allow_pickle=False) as raw:
        images, states = raw["images"], raw["states"]
        instructions, timestamps = raw["instructions"], raw["timestamps"]
        n = len(images)
        if (images.ndim != 5 or images.shape[1] != 2 or images.shape[-1] != 3 or images.dtype != np.uint8
                or states.shape != (n, config.state_dim) or instructions.shape != (n,)
                or timestamps.shape != (n,) or n == 0 or not np.isfinite(states).all()
                or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0)):
            raise ValueError("Invalid or misaligned recorded inputs")
        if instructions.dtype.kind not in {"U", "S"} or any(not str(s).strip() for s in instructions):
            raise ValueError("Nonempty recorded language instructions are required")
        state_columns = data.get("state_columns", [])
        if state_columns != list(range(config.state_dim)):
            raise ValueError("Recorded UR5e inference needs the complete declared state mapping")
        predictions = []
        for start in range(0, n, batch_size):
            pixels = torch.from_numpy(images[start:start + batch_size].copy()).permute(0, 1, 4, 2, 3).float() / 255
            pixels = F.interpolate(pixels.flatten(0, 1), config.image_size, mode="bilinear",
                                   align_corners=False, antialias=True).unflatten(0, (-1, 2))
            text = [s.decode("utf-8") if isinstance(s, bytes) else str(s)
                    for s in instructions[start:start + batch_size]]
            actions = normalizer.inverse(model.act(text, pixels, torch.tensor(
                states[start:start + batch_size], dtype=torch.float32))).float().cpu().numpy()
            if not np.isfinite(actions).all():
                raise ValueError("Nonfinite recorded-input prediction")
            predictions.append(actions[:, :15])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, actions=np.concatenate(predictions), timestamps=timestamps)
    report = {"output": str(output.resolve()), "recording": str(Path(recording).resolve()),
              "checkpoint": str(Path(checkpoint).resolve()), "observations": n, "action_chunk": 15,
              "action_convention": data["action_convention"], "mode": "offline_CPU_prediction_only",
              "hardware_execution": False, "success_metric": None}
    atomic_json(output.with_suffix(".json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    registry = sub.add_parser("tasks")
    registry.add_argument("--output", required=True)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--input", required=True)
    aggregate.add_argument("--output", required=True)
    aggregate.add_argument("--expected-per-task", type=int, default=100)
    aggregate.add_argument("--tasks", nargs="+", choices=EVALUATION_TASKS)
    replay = sub.add_parser("infer")
    replay.add_argument("--checkpoint", required=True)
    replay.add_argument("--recording", required=True)
    replay.add_argument("--data-source")
    replay.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "infer":
        print(json.dumps(recorded_inference(args.checkpoint, args.recording, args.output, data_source=args.data_source)))
        return
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.command == "tasks":
        result = {"training": TRAIN_TASKS, "evaluation": EVALUATION_TASKS}
    else:
        rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
        result = aggregate_trials(rows, expected_per_task=args.expected_per_task, tasks=args.tasks)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)


if __name__ == "__main__":
    main()
