"""Data preparation without author code or modifications to source datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .config import VITAConfig
from .dataset import SUITES, atomic_json, build_manifest
from .mixture import build_episode_manifest, build_mixture


def prepare_libero(root, output, config, *, frame_stride=20, window_stride=8, suites=SUITES):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    cameras = ("agentview_rgb", "eye_in_hand_rgb")[:config.num_views]
    if len(cameras) != config.num_views:
        raise ValueError("LIBERO supplies at most two verified cameras")
    manifest = build_manifest(root, config, cameras=cameras, suites=suites,
                              frame_stride=frame_stride, window_stride=window_stride)
    atomic_json(output / "libero.json", manifest)
    # Views of one source exercise the mixed objectives, not the paper's 13-source distribution.
    atomic_json(output / "sources.json", {
        "provenance": "Three modality projections of real LIBERO; NOT independent human/action datasets",
        "sources": [
            {"name": "libero_video", "manifest": "libero.json", "mode": "video",
             "weights": {"warmup": 1, "cotrain": 1}},
            {"name": "libero_action", "manifest": "libero.json", "mode": "action",
             "weights": {"warmup": 1}},
            {"name": "libero_paired", "manifest": "libero.json", "mode": "paired",
             "weights": {"cotrain": 1, "finetune": 1}},
        ],
    })
    mixture = build_mixture(output / "sources.json", config)
    atomic_json(output / "mixture.json", mixture)
    return mixture


def convert_calvin(root, annotations, output, *, control_hz=30):
    """Stream CALVIN NPZ frames using a safe JSONL language/split index.

    Records contain id, group_id, split, start, end (inclusive), instruction.
    No pickle annotations are executed, and rel_actions remain in CALVIN's
    normalized 7D relative-control coordinates without an inferred conversion.
    """
    root, output, annotations = Path(root).resolve(), Path(output), Path(annotations)
    if output.exists():
        raise FileExistsError(output)
    records = [json.loads(line) for line in annotations.read_text().splitlines() if line.strip()]
    if not records or control_hz <= 0:
        raise ValueError("Need annotations and positive control rate")
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "x") as handle:
        handle.attrs.update(vita_episode_schema=1, control_hz=control_hz,
                            action_convention="CALVIN rel_actions: normalized relative XYZ/Euler/gripper",
                            provenance=f"CALVIN public NPZ source {root}; explicit index {annotations.resolve()}")
        groups, frame_splits = {}, {}
        for record in records:
            if record["split"] not in {"train", "validation", "test"} or not record["instruction"].strip():
                raise ValueError("Invalid split or instruction")
            name, group = str(record["id"]), str(record["group_id"])
            if not name or "/" in name or not group or record["end"] < record["start"] or record["start"] < 0:
                raise ValueError("Invalid episode identity or bounds")
            if group in groups and groups[group] != record["split"]:
                raise ValueError("A recording crosses split boundaries")
            groups[group] = record["split"]
            episode = handle.create_group(f"episodes/{name}")
            episode.attrs.update(split=record["split"], group_id=group, instruction=record["instruction"])
            length = record["end"] - record["start"] + 1
            for position, frame in enumerate(range(record["start"], record["end"] + 1)):
                if frame in frame_splits and frame_splits[frame] != record["split"]:
                    raise ValueError("Overlapping CALVIN frames cross split boundaries")
                frame_splits[frame] = record["split"]
                with np.load(root / f"episode_{frame:07d}.npz", allow_pickle=False) as source:
                    static, gripper = source["rgb_static"], source["rgb_gripper"]
                    actions, states = source["rel_actions"], source["robot_obs"]
                    if actions.shape != (7,) or states.shape != (15,):
                        raise ValueError("Unverified CALVIN action/state shape")
                    if not np.isfinite(actions).all() or not np.isfinite(states).all():
                        raise ValueError("Nonfinite CALVIN controls/state")
                    if any(im.dtype != np.uint8 or im.ndim != 3 or im.shape[-1] != 3
                           for im in (static, gripper)):
                        raise ValueError("Expected RGB uint8 CALVIN images")
                    # CALVIN camera resolutions differ. Preserve static resolution, resize wrist explicitly.
                    if gripper.shape != static.shape:
                        import torch
                        from torch.nn import functional as F
                        gripper = F.interpolate(torch.from_numpy(gripper.copy()).permute(2, 0, 1)[None].float(),
                                                static.shape[:2], mode="bilinear", align_corners=False,
                                                antialias=True)[0].permute(1, 2, 0).round().byte().numpy()
                    images = np.stack([static, gripper])
                    if position == 0:
                        episode.create_dataset("images", (length, *images.shape), dtype="u1",
                                               chunks=(1, *images.shape), compression="lzf")
                        episode.create_dataset("actions", (length, 7), dtype="f4")
                        episode.create_dataset("states", (length, 15), dtype="f4")
                    episode["images"][position] = images
                    episode["actions"][position] = actions
                    episode["states"][position] = states
    return {"episodes": len(records), "path": str(output.resolve()),
            "source_modified": False, "control_hz_assumption": control_hz}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    libero = commands.add_parser("libero")
    libero.add_argument("--root", required=True)
    libero.add_argument("--config", required=True)
    libero.add_argument("--output", required=True)
    libero.add_argument("--frame-stride", type=int, default=20)
    libero.add_argument("--window-stride", type=int, default=8)
    libero.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    canonical = commands.add_parser("episodes")
    canonical.add_argument("--input", required=True)
    canonical.add_argument("--config", required=True)
    canonical.add_argument("--output", required=True)
    canonical.add_argument("--mode", choices=["video", "action", "paired"], required=True)
    canonical.add_argument("--frame-stride", type=int, default=1)
    canonical.add_argument("--window-stride", type=int, default=1)
    canonical.add_argument("--action-columns", type=int, nargs="+")
    canonical.add_argument("--state-columns", type=int, nargs="+")
    calvin = commands.add_parser("calvin")
    calvin.add_argument("--root", required=True)
    calvin.add_argument("--annotations", required=True)
    calvin.add_argument("--output", required=True)
    calvin.add_argument("--control-hz", type=float, default=30)
    args = parser.parse_args()
    if args.command == "libero":
        result = prepare_libero(args.root, args.output, VITAConfig.load(args.config),
                                frame_stride=args.frame_stride, window_stride=args.window_stride,
                                suites=args.suites)
    elif args.command == "episodes":
        path = Path(args.output)
        if path.exists():
            raise FileExistsError(path)
        result = build_episode_manifest(args.input, VITAConfig.load(args.config), mode=args.mode,
                                        frame_stride=args.frame_stride, window_stride=args.window_stride,
                                        action_columns=args.action_columns, state_columns=args.state_columns)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, result)
    else:
        result = convert_calvin(args.root, args.annotations, args.output, control_hz=args.control_hz)
    print(json.dumps({k: result[k] for k in ("path", "episodes", "counts", "manifest_sha256") if k in result}))


if __name__ == "__main__":
    main()
