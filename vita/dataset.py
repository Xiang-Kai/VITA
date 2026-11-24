"""Split-aware, lazy LIBERO data for training; no simulator or network calls."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .config import VITAConfig
from .data import ActionNormalizer


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def content_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def build_manifest(
    root: str | Path, config: VITAConfig, *, suites=SUITES,
    cameras=("agentview_rgb", "eye_in_hand_rgb"), frame_stride=20, window_stride=8,
    seed=17, validation_fraction=0.1, test_fraction=0.1, max_tasks=None,
) -> dict:
    if config.action_dim != 7 or config.state_dim != 8:
        raise ValueError("LIBERO requires action_dim=7 and state_dim=8")
    if len(cameras) != config.num_views or len(set(cameras)) != len(cameras):
        raise ValueError("Camera list must match num_views and contain no duplicates")
    if min(frame_stride, window_stride) < 1:
        raise ValueError("Temporal strides must be positive")
    if not 0 < validation_fraction < 0.5 or not 0 < test_fraction < 0.5:
        raise ValueError("Need positive validation/test fractions below 0.5")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Need a nonempty training fraction")
    root = Path(root).resolve()
    files, windows, excluded = [], [], []
    sum_actions = torch.zeros(config.action_dim, dtype=torch.float64)
    sum_squares = torch.zeros_like(sum_actions)
    count = 0
    for suite in suites:
        if suite not in SUITES:
            raise ValueError(f"Not a standard four-suite name: {suite}")
        paths = sorted((root / suite).glob("*.hdf5"))
        if max_tasks is not None:
            paths = paths[:max_tasks]
        if not paths:
            raise FileNotFoundError(f"No HDF5 tasks found for {suite}")
        for path in paths:
            with h5py.File(path, "r") as handle:
                data = handle["data"]
                instruction = json.loads(data.attrs["problem_info"])["language_instruction"]
                control_hz = json.loads(data.attrs["env_args"])["env_kwargs"]["control_freq"]
                if type(control_hz) not in {int, float} or control_hz != 20:
                    raise ValueError(f"LIBERO requires verified 20 Hz controls: {path}")
                if data.attrs.get("macros_image_convention", "") != "opengl":
                    raise ValueError(f"Unverified image convention in {path}")
                eligible = []
                for name in sorted(data):
                    episode = data[name]
                    if episode["actions"].shape[1:] != (config.action_dim,):
                        raise ValueError(f"Action dimensionality mismatch: {path}/{name}")
                    required = list(cameras) + ["ee_states", "gripper_states"]
                    if any(key not in episode["obs"] for key in required):
                        raise ValueError(f"Missing observations: {path}/{name}")
                    for camera in cameras:
                        images = episode["obs"][camera]
                        if (images.ndim != 4 or images.shape[-1] != 3 or
                                min(images.shape[1:3]) < 1 or images.dtype != np.uint8):
                            raise ValueError(f"Expected uint8 RGB images [T,H,W,3]: {path}/{name}/{camera}")
                    if (episode["obs"]["ee_states"].shape[1:] != (6,) or
                            episode["obs"]["gripper_states"].shape[1:] != (2,)):
                        raise ValueError(f"Expected 6D end-effector + 2D gripper state: {path}/{name}")
                    lengths = [len(episode["actions"])] + [len(episode["obs"][key]) for key in required]
                    if len(set(lengths)) != 1:
                        raise ValueError(f"Misaligned observations/actions: {path}/{name}")
                    length = lengths[0]
                    stop = min(length - config.action_horizon,
                               length - 1 - config.future_frames * frame_stride)
                    if stop < 0:
                        excluded.append({"task": str(path.relative_to(root)), "episode": name,
                                         "reason": "shorter_than_prediction_horizon", "length": length})
                    else:
                        eligible.append((name, length, stop))
                if len(eligible) < 3:
                    raise ValueError(f"Fewer than 3 usable demonstrations for split: {path}")
                # Hash-sort by stable task/demo IDs, independent of directory traversal and Python hash seed.
                ranked = sorted(eligible, key=lambda item: hashlib.sha256(
                    f"{seed}:{path.relative_to(root)}:{item[0]}".encode()
                ).digest())
                n_val = max(1, int(len(ranked) * validation_fraction))
                n_test = max(1, int(len(ranked) * test_fraction))
                file_index = len(files)
                files.append({
                    "path": str(path), "relative_path": str(path.relative_to(root)),
                    "suite": suite, "instruction": instruction, "control_hz": control_hz,
                    "size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
                    "episodes": {},
                })
                for position, (name, length, stop) in enumerate(ranked):
                    split = "validation" if position < n_val else "test" if position < n_val + n_test else "train"
                    files[-1]["episodes"][name] = {"split": split, "length": length}
                    if split == "train":
                        raw = torch.from_numpy(data[name]["actions"][:]).to(torch.float64)
                        if not torch.isfinite(raw).all():
                            raise ValueError(f"Nonfinite training actions in {path}/{name}")
                        sum_actions += raw.sum(0)
                        sum_squares += raw.square().sum(0)
                        count += len(raw)
                    for start in range(0, stop + 1, window_stride):
                        windows.append({"file": file_index, "episode": name, "start": start, "split": split})
    if count == 0:
        raise ValueError("No training actions")
    mean = sum_actions / count
    std = (sum_squares / count - mean.square()).clamp_min(0).sqrt()
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    manifest = {
        "schema_version": 1, "root": str(root), "seed": seed, "files": files, "windows": windows,
        "cameras": list(cameras), "frame_stride": frame_stride, "window_stride": window_stride,
        "future_frames": config.future_frames, "action_horizon": config.action_horizon,
        "action_dim": config.action_dim, "state_dim": config.state_dim,
        "image_size": config.image_size, "flip_vertical": True,
        "normalization": {"mean": mean.tolist(), "std": std.tolist(),
                          "fit_split": "train", "timesteps": count,
                          "method": "all timesteps of eligible training episodes; z-score"},
        "excluded_episodes": excluded,
        "counts": {split: sum(w["split"] == split for w in windows) for split in ("train", "validation", "test")},
        "integrity": "absolute path + size + mtime + schema; not a full-file cryptographic dataset hash",
        "scope": "episode-held-out offline split; not the official simulator initial-state evaluation",
    }
    manifest["manifest_sha256"] = content_hash(manifest)
    return manifest


def load_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text())
    expected = manifest.pop("manifest_sha256")
    actual = content_hash(manifest)
    manifest["manifest_sha256"] = expected
    if expected != actual:
        raise ValueError("Manifest hash mismatch")
    for split in ("train", "validation", "test"):
        if sum(w["split"] == split for w in manifest["windows"]) != manifest["counts"][split]:
            raise ValueError("Manifest count mismatch")
    seen = {}
    for window in manifest["windows"]:
        key = (window["file"], window["episode"])
        if key in seen and seen[key] != window["split"]:
            raise ValueError("An episode crosses split boundaries")
        seen[key] = window["split"]
    return manifest


class LiberoDataset(Dataset):
    def __init__(self, manifest: dict, split: str, *, max_open_files=4, image_cache_mib=128):
        if split not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        self.manifest, self.split = manifest, split
        self.windows = [w for w in manifest["windows"] if w["split"] == split]
        if not self.windows:
            raise ValueError(f"No {split} windows")
        self.normalizer = ActionNormalizer(
            torch.tensor(manifest["normalization"]["mean"], dtype=torch.float32),
            torch.tensor(manifest["normalization"]["std"], dtype=torch.float32),
        )
        self.max_open_files = max_open_files
        self.cache_limit = image_cache_mib * 2**20
        self._handles = OrderedDict()
        self._images = OrderedDict()
        self._cache_bytes = 0

    def __len__(self):
        return len(self.windows)

    def _file(self, index):
        if index not in self._handles:
            info = self.manifest["files"][index]
            path = Path(info["path"])
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (info["size_bytes"], info["mtime_ns"]):
                raise ValueError(f"Dataset file changed since manifest: {path}")
            if len(self._handles) >= self.max_open_files:
                _, old = self._handles.popitem(last=False)
                old.close()
            self._handles[index] = h5py.File(path, "r")
        self._handles.move_to_end(index)
        return self._handles[index]

    def _image(self, file_index, episode, camera, frame, handle):
        key = (file_index, episode, camera, frame)
        if key in self._images:
            self._images.move_to_end(key)
            return self._images[key]
        pixels = torch.from_numpy(handle["data"][episode]["obs"][camera][frame].copy()).permute(2, 0, 1)
        if self.manifest["flip_vertical"]:
            pixels = pixels.flip(-2)
        pixels = F.interpolate(pixels[None].float() / 255, self.manifest["image_size"],
                               mode="bilinear", align_corners=False, antialias=True)[0]
        size = pixels.numel() * pixels.element_size()
        if size <= self.cache_limit:
            while self._images and self._cache_bytes + size > self.cache_limit:
                _, old = self._images.popitem(last=False)
                self._cache_bytes -= old.numel() * old.element_size()
            self._images[key] = pixels
            self._cache_bytes += size
        return pixels

    def __getitem__(self, index):
        row = self.windows[index]
        file_index, name, start = row["file"], row["episode"], row["start"]
        handle = self._file(file_index)
        episode = handle["data"][name]
        current, future = [], []
        for camera in self.manifest["cameras"]:
            current.append(self._image(file_index, name, camera, start, handle))
        for step in range(self.manifest["future_frames"]):
            frame = start + (step + 1) * self.manifest["frame_stride"]
            future.append(torch.stack([
                self._image(file_index, name, camera, frame, handle) for camera in self.manifest["cameras"]
            ]))
        actions = torch.from_numpy(episode["actions"][start:start + self.manifest["action_horizon"]]).float()
        states = torch.from_numpy(np.concatenate([
            episode["obs"]["ee_states"][start], episode["obs"]["gripper_states"][start],
        ])).float()
        if states.shape != (self.manifest["state_dim"],):
            raise ValueError("Unexpected state vector shape")
        result = {"current": torch.stack(current), "future": torch.stack(future),
                  "actions": self.normalizer.normalize(actions), "states": states,
                  "view_mask": torch.ones(len(current), dtype=torch.bool),
                  "instructions": self.manifest["files"][file_index]["instruction"],
                  "sample_id": f"{file_index}:{name}:{start}"}
        for key, value in result.items():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite {key} at {result['sample_id']}")
        return result

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        self._images.clear()
        self._cache_bytes = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_handles=OrderedDict(), _images=OrderedDict(), _cache_bytes=0)
        return state


class StatefulBatchSampler:
    """Deterministic shuffling with an exact, serializable mid-epoch cursor."""
    def __init__(self, size: int, batch_size: int, seed: int):
        if min(size, batch_size) < 1:
            raise ValueError("Need a nonempty dataset and positive batch size")
        self.size, self.batch_size = size, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.permutation = torch.randperm(size, generator=self.generator)
        self.cursor, self.epoch = 0, 0

    def next(self, stage=None) -> list[int]:
        result = []
        while len(result) < self.batch_size:
            take = min(self.batch_size - len(result), self.size - self.cursor)
            result.extend(self.permutation[self.cursor:self.cursor + take].tolist())
            self.cursor += take
            if self.cursor == self.size:
                self.epoch += 1
                self.cursor = 0
                self.permutation = torch.randperm(self.size, generator=self.generator)
        return result

    def state_dict(self):
        return {"size": self.size, "batch_size": self.batch_size, "generator": self.generator.get_state(),
                "permutation": self.permutation, "cursor": self.cursor, "epoch": self.epoch}

    def load_state_dict(self, state):
        if (state["size"], state["batch_size"]) != (self.size, self.batch_size):
            raise ValueError("Sampler shape changed across resume")
        self.generator.set_state(state["generator"].cpu())
        self.permutation = state["permutation"].cpu()
        self.cursor, self.epoch = state["cursor"], state["epoch"]


def main():
    parser = argparse.ArgumentParser(description="Build immutable split-aware LIBERO training manifest")
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    parser.add_argument("--cameras", nargs="+", default=["agentview_rgb", "eye_in_hand_rgb"])
    parser.add_argument("--frame-stride", type=int, default=20)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    result = build_manifest(args.root, VITAConfig.load(args.config), suites=args.suites,
                            cameras=args.cameras, frame_stride=args.frame_stride,
                            window_stride=args.window_stride, seed=args.seed, max_tasks=args.max_tasks)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, result)
    print(json.dumps({"manifest": str(path), "sha256": result["manifest_sha256"], "counts": result["counts"],
                      "tasks": len(result["files"]), "excluded": len(result["excluded_episodes"])}))


if __name__ == "__main__":
    main()
