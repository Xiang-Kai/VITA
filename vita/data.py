from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .config import VITAConfig


@dataclass
class ActionNormalizer:
    mean: Tensor
    std: Tensor

    @classmethod
    def fit(cls, training_actions: Tensor):
        values = training_actions.flatten(0, -2)
        std = values.std(dim=0, unbiased=False)
        return cls(values.mean(dim=0), torch.where(std < 1e-6, torch.ones_like(std), std))

    def normalize(self, values: Tensor) -> Tensor:
        return (values - self.mean.to(values)) / self.std.to(values)

    def inverse(self, values: Tensor) -> Tensor:
        return values * self.std.to(values) + self.mean.to(values)

    def to_dict(self) -> dict:
        return {"method": "training-window z-score; constant dimensions use std=1",
                "mean": self.mean.tolist(), "std": self.std.tolist()}


def synthetic_batch(config: VITAConfig, batch_size: int = 2, seed: int = 0) -> dict:
    """Explicit synthetic tensors for numerical validation, not robot evidence."""
    generator = torch.Generator().manual_seed(seed)
    current = torch.rand(batch_size, 3, config.image_size, config.image_size, generator=generator)
    future = torch.stack([
        current.roll(shifts=step + 1, dims=-1) for step in range(config.future_frames)
    ], dim=1)
    actions = torch.randn(batch_size, config.action_horizon, config.action_dim, generator=generator) * 0.1
    states = torch.randn(batch_size, config.state_dim, generator=generator) * 0.1
    return {"current": current, "future": future, "actions": actions, "states": states,
            "instructions": ["move the synthetic object"] * batch_size}


def tensor_fingerprint(batch: dict) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(batch.items()):
        digest.update(key.encode())
        if isinstance(value, Tensor):
            value = value.detach().cpu().contiguous()
            digest.update(str((value.shape, value.dtype)).encode())
            digest.update(value.numpy().tobytes())
        else:
            digest.update(json.dumps(value, ensure_ascii=True).encode())
    return digest.hexdigest()


def load_libero_windows(
    path: str | Path, config: VITAConfig, *, batch_size: int,
    frame_stride: int, camera: str = "agentview_rgb", flip_vertical: bool = True,
) -> tuple[dict, ActionNormalizer, dict]:
    """Read-only, episode-bounded single-camera windows for a training smoke test."""
    if batch_size < 1 or frame_stride < 1:
        raise ValueError("batch_size and frame_stride must be positive")
    path = Path(path).resolve()
    observations, futures, actions, states, records = [], [], [], [], []
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        problem = data.attrs.get("problem_info")
        if not problem:
            raise ValueError("LIBERO file needs problem_info with a language_instruction")
        instruction = json.loads(problem)["language_instruction"]
        env = json.loads(data.attrs["env_args"])
        fps = env.get("env_kwargs", {}).get("control_freq")
        # One deterministic window per demonstration, with no episode crossing.
        for name in sorted(data, key=lambda name: int(name.split("_")[-1])):
            demo = data[name]
            available = min(len(demo["actions"]), len(demo["obs"][camera]))
            last_start = min(available - config.action_horizon,
                             available - 1 - config.future_frames * frame_stride)
            if last_start < 0:
                continue
            start = last_start // 2
            future_indices = [start + (step + 1) * frame_stride for step in range(config.future_frames)]
            obs = demo["obs"]
            observations.append(obs[camera][start])
            futures.append(obs[camera][future_indices])
            actions.append(demo["actions"][start:start + config.action_horizon])
            states.append(np.concatenate([obs["ee_states"][start], obs["gripper_states"][start]]))
            records.append({"demo": name, "start": start, "future_indices": future_indices,
                            "action_stop_exclusive": start + config.action_horizon})
            if len(records) == batch_size:
                break
    if len(records) != batch_size:
        raise ValueError(f"Only {len(records)} complete windows, requested {batch_size}")

    def image_tensor(array):
        result = torch.from_numpy(np.stack(array)).float() / 255
        lead = result.shape[:-3]
        result = result.reshape(-1, *result.shape[-3:]).permute(0, 3, 1, 2)
        if flip_vertical:
            result = result.flip(-2)
        result = F.interpolate(result, (config.image_size, config.image_size),
                               mode="bilinear", align_corners=False, antialias=True)
        return result.reshape(*lead, 3, config.image_size, config.image_size)

    raw_actions = torch.from_numpy(np.stack(actions)).float()
    state_tensor = torch.from_numpy(np.stack(states)).float()
    if raw_actions.shape[1:] != (config.action_horizon, config.action_dim):
        raise ValueError("LIBERO action dimensions differ from config")
    if state_tensor.shape[-1] != config.state_dim:
        raise ValueError("Use state_dim=8 for LIBERO ee_states + gripper_states")
    normalizer = ActionNormalizer.fit(raw_actions)
    batch = {
        "current": image_tensor(observations), "future": image_tensor(futures),
        "actions": normalizer.normalize(raw_actions), "states": state_tensor,
        "instructions": [instruction] * batch_size,
    }
    for key, value in batch.items():
        if isinstance(value, Tensor) and not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite values in {key}")
    metadata = {
        "kind": "real_libero_training_windows_not_benchmark",
        "path": str(path), "size_bytes": path.stat().st_size, "camera": camera,
        "flip_vertical": flip_vertical, "control_frequency_hz": fps,
        "frame_stride": frame_stride,
        "future_frame_interval_seconds": frame_stride / fps if fps else None,
        "windows": records, "normalization": normalizer.to_dict(),
        "sample_sha256": tensor_fingerprint(batch),
        "scope": "training-sample overfit only; no held-out data or simulator rollout",
    }
    return batch, normalizer, metadata
