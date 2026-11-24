"""Independent multimodal data contract and exactly resumable batch-level mixing.

No coordinate conversion is inferred from a dataset name. Each source declares
its action convention and explicit column mapping before it can join a mixture.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .config import VITAConfig
from .data import ActionNormalizer
from .dataset import LiberoDataset, StatefulBatchSampler, atomic_json, content_hash, load_manifest


MODES = {"video", "action", "paired"}
ALLOWED = {"warmup": MODES, "cotrain": {"video", "paired"}, "finetune": {"paired"}}


def seal(value):
    value["manifest_sha256"] = content_hash(value)
    return value


def _columns(values, width, label):
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{label} needs a nonempty, unique column mapping")
    if any(type(i) is not int or i < 0 or i >= width for i in values):
        raise ValueError(f"{label} mapping exceeds the configured dimension")
    return list(values)


def build_episode_manifest(path, config, *, mode, frame_stride=1, window_stride=1,
                           action_columns=None, state_columns=None, filter_short=False):
    """Index a canonical HDF5 file, with explicit episode/group splits.

    Images are uint8 [T,V,H,W,3], top-left RGB. Actions/states use declared
    columns. Episode `split` and `group_id` attributes are mandatory so clips
    of the same recording cannot accidentally leak between splits.
    """
    if mode not in MODES or min(frame_stride, window_stride) < 1:
        raise ValueError("Invalid modality or temporal stride")
    path = Path(path).resolve()
    windows, episodes, groups, excluded = [], {}, {}, []
    sums = torch.zeros(config.action_dim, dtype=torch.float64)
    squares, counts = torch.zeros_like(sums), torch.zeros_like(sums)
    action_columns = _columns(action_columns, config.action_dim, "action") if mode != "video" else []
    state_columns = _columns(state_columns, config.state_dim, "state") if state_columns else []
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("vita_episode_schema") != 1:
            raise ValueError("Expected vita_episode_schema=1")
        convention = str(handle.attrs.get("action_convention", "")).strip()
        provenance = str(handle.attrs.get("provenance", "")).strip()
        hz = float(handle.attrs.get("control_hz", 0))
        if not provenance or not math.isfinite(hz) or hz <= 0 or (mode != "video" and not convention):
            raise ValueError("Source needs provenance, positive control_hz and an action convention")
        for name, episode in sorted(handle["episodes"].items()):
            split = str(episode.attrs.get("split", ""))
            group = str(episode.attrs.get("group_id", "")).strip()
            instruction = str(episode.attrs.get("instruction", "")).strip()
            if split not in {"train", "validation", "test"} or not group:
                raise ValueError("Every episode needs an explicit split and physical-recording group_id")
            if group in groups and groups[group] != split:
                raise ValueError("A physical recording crosses split boundaries")
            groups[group] = split
            lengths = []
            if mode != "action":
                if not instruction:
                    raise ValueError("Video-containing episodes need nonempty instructions")
                images = episode["images"]
                if (images.ndim != 5 or images.shape[1] > config.num_views
                        or images.shape[1] < 1 or images.shape[-1] != 3 or images.dtype != np.uint8):
                    raise ValueError("Expected uint8 top-left RGB images [T,V,H,W,3]")
                lengths.append(len(images))
            if mode != "video":
                if episode["actions"].shape[1:] != (len(action_columns),):
                    raise ValueError("Action columns do not match source")
                lengths.append(len(episode["actions"]))
            if state_columns:
                if episode["states"].shape[1:] != (len(state_columns),):
                    raise ValueError("State columns do not match source")
                lengths.append(len(episode["states"]))
            if len(set(lengths)) != 1:
                raise ValueError("Episode streams are not time aligned")
            length = lengths[0]
            stop = length - 1
            if mode != "action":
                stop = min(stop, length - 1 - config.future_frames * frame_stride)
            if mode != "video":
                stop = min(stop, length - config.action_horizon)
            if stop < 0:
                if filter_short:
                    excluded.append({"episode": name, "length": length, "reason": "shorter_than_prediction_horizon"})
                    continue
                raise ValueError(f"Episode {name} is shorter than the requested horizon")
            episodes[name] = {"split": split, "group_id": group, "length": length,
                              "instruction": instruction}
            windows.extend({"episode": name, "start": i, "split": split}
                           for i in range(0, stop + 1, window_stride))
            if split == "train" and mode != "video":
                for offset in range(0, length, 8192):
                    values = torch.from_numpy(episode["actions"][offset:offset + 8192]).double()
                    if not torch.isfinite(values).all():
                        raise ValueError("Nonfinite training actions")
                    sums[action_columns] += values.sum(0)
                    squares[action_columns] += values.square().sum(0)
                    counts[action_columns] += len(values)
    if not all(any(w["split"] == s for w in windows) for s in ("train", "validation")):
        raise ValueError("Source requires nonempty training and validation splits")
    mean = sums / counts.clamp_min(1)
    std = (squares / counts.clamp_min(1) - mean.square()).clamp_min(0).sqrt()
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    result = {
        "schema_version": 1, "kind": "canonical_episodes", "mode": mode, "path": str(path),
        "size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
        "episodes": episodes, "windows": windows,
        "excluded_episodes": excluded,
        "counts": {s: sum(w["split"] == s for w in windows) for s in ("train", "validation", "test")},
        "frame_stride": frame_stride, "window_stride": window_stride,
        **{key: getattr(config, key) for key in
           ("image_size", "future_frames", "action_horizon", "action_dim", "state_dim", "num_views")},
        "action_columns": action_columns, "state_columns": state_columns,
        "action_convention": convention, "provenance": provenance, "control_hz": hz,
        "normalization": {"mean": mean.tolist(), "std": std.tolist(), "fit_split": "train",
                          "valid_timesteps_per_dimension": counts.tolist()},
        "integrity": "manifest SHA256 and file path/size/mtime, not full dataset byte hashing",
    }
    return seal(result)


def load_data_manifest(path):
    value = json.loads(Path(path).read_text())
    if value.get("kind") not in {"canonical_episodes", "mixture"}:
        return load_manifest(path)
    expected = value.pop("manifest_sha256")
    if content_hash(value) != expected:
        raise ValueError("Manifest hash mismatch")
    value["manifest_sha256"] = expected
    return value


class EpisodeDataset(Dataset):
    def __init__(self, manifest, split):
        self.manifest, self.split = manifest, split
        self.windows = [w for w in manifest["windows"] if w["split"] == split]
        if not self.windows:
            raise ValueError(f"Empty {split} split")
        self.normalizer = ActionNormalizer(torch.tensor(manifest["normalization"]["mean"]),
                                           torch.tensor(manifest["normalization"]["std"]))
        self.handle = None

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        m = self.manifest
        if self.handle is None:
            path = Path(m["path"])
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (m["size_bytes"], m["mtime_ns"]):
                raise ValueError("Dataset file changed since manifest")
            self.handle = h5py.File(path, "r")
        row = self.windows[index]
        episode, start = self.handle["episodes"][row["episode"]], row["start"]
        out = {"sample_id": f"{row['episode']}:{start}", "modality": m["mode"]}
        if m["mode"] != "action":
            indices = [start + t * m["frame_stride"] for t in range(m["future_frames"] + 1)]
            pixels = torch.from_numpy(episode["images"][indices]).permute(0, 1, 4, 2, 3)
            time, views = pixels.shape[:2]
            pixels = F.interpolate(pixels.flatten(0, 1).float() / 255, m["image_size"],
                                   mode="bilinear", align_corners=False, antialias=True)
            pixels = pixels.unflatten(0, (time, views))
            padded = pixels.new_zeros(time, m["num_views"], 3, m["image_size"], m["image_size"])
            padded[:, :views] = pixels
            out.update(current=padded[0], future=padded[1:],
                       view_mask=torch.arange(m["num_views"]) < views,
                       instructions=m["episodes"][row["episode"]]["instruction"])
            out["states"] = torch.zeros(m["state_dim"])
            out["state_mask"] = torch.zeros(m["state_dim"], dtype=torch.bool)
            if m["state_columns"]:
                out["states"][m["state_columns"]] = torch.from_numpy(episode["states"][start]).float()
                out["state_mask"][m["state_columns"]] = True
        if m["mode"] != "video":
            actions = torch.zeros(m["action_horizon"], m["action_dim"])
            actions[:, m["action_columns"]] = torch.from_numpy(
                episode["actions"][start:start + m["action_horizon"]]).float()
            mask = torch.zeros_like(actions, dtype=torch.bool)
            mask[:, m["action_columns"]] = True
            out["actions"] = self.normalizer.normalize(actions).masked_fill(~mask, 0)
            out["action_mask"] = mask
        if any(not torch.isfinite(x).all() for x in out.values() if isinstance(x, torch.Tensor)):
            raise ValueError("Nonfinite episode sample")
        return out

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def make_dataset(manifest, split):
    if manifest.get("kind") == "mixture":
        return MixtureDataset(manifest, split)
    if manifest.get("kind") == "canonical_episodes":
        return EpisodeDataset(manifest, split)
    return LiberoDataset(manifest, split)


def build_mixture(spec_path, config):
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text())
    sources, names, physical_groups = [], set(), {}
    shape_keys = ("image_size", "future_frames", "action_horizon", "action_dim", "state_dim")
    for source in spec["sources"]:
        name = source["name"]
        if not name or name in names:
            raise ValueError("Source names must be nonempty and unique")
        names.add(name)
        mode = source["mode"]
        if mode not in MODES:
            raise ValueError("Unknown source modality")
        manifest = load_data_manifest(spec_path.parent / source["manifest"])
        if manifest.get("kind") == "mixture":
            raise ValueError("Nested mixtures are not supported")
        if manifest.get("kind") == "canonical_episodes" and manifest["mode"] != mode:
            raise ValueError("Canonical source modality cannot be relabeled")
        for key in shape_keys:
            if manifest[key] != getattr(config, key):
                raise ValueError(f"Source {name} disagrees on {key}")
        if manifest.get("num_views", len(manifest.get("cameras", []))) != config.num_views:
            raise ValueError("Source view count differs from model")
        for stage, weight in source["weights"].items():
            if stage not in ALLOWED or mode not in ALLOWED[stage] or not math.isfinite(weight) or weight <= 0:
                raise ValueError("Invalid stage/modality/weight combination")
        if not source["weights"]:
            raise ValueError("Each source needs at least one training stage")
        if manifest.get("kind") == "canonical_episodes":
            records = [(str(Path(manifest["path"]).resolve()), e["group_id"], e["split"])
                       for e in manifest["episodes"].values()]
        else:
            records = [(str(Path(f["path"]).resolve()), name, e["split"])
                       for f in manifest["files"] for name, e in f["episodes"].items()]
        for path, group, split in records:
            identity = (path, group)
            if identity in physical_groups and physical_groups[identity] != split:
                raise ValueError("A recording crosses splits between mixture sources")
            physical_groups[identity] = split
        sources.append({"name": name, "mode": mode, "weights": source["weights"], "manifest": manifest})
    if not sources:
        raise ValueError("Empty mixture")
    unique = {s["manifest"]["manifest_sha256"]: s["manifest"] for s in sources}
    return seal({"schema_version": 1, "kind": "mixture", "sources": sources,
                 **{key: getattr(config, key) for key in shape_keys},
                 "num_views": config.num_views,
                 "counts": {s: sum(x["manifest"]["counts"][s] for x in sources)
                            for s in ("train", "validation", "test")},
                 "unique_manifest_counts": {s: sum(m["counts"][s] for m in unique.values())
                                            for s in ("train", "validation", "test")},
                 "source_provenance": spec.get("provenance", "Explicit independently assembled mixture"),
                 "weight_semantics": "explicit per-stage probabilities normalized over eligible sources",
                 "original_paper_mixture_recovered": False})


class MixtureDataset(Dataset):
    def __init__(self, manifest, split):
        self.manifest, self.split = manifest, split
        self.sources = [make_dataset(s["manifest"], split) for s in manifest["sources"]]
        self.offsets = [0]
        for dataset in self.sources:
            self.offsets.append(self.offsets[-1] + len(dataset))

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        import bisect
        source_index = bisect.bisect_right(self.offsets, index) - 1
        sample = self.sources[source_index][index - self.offsets[source_index]]
        source = self.manifest["sources"][source_index]
        mode = source["mode"]
        if mode == "video":
            sample = {k: v for k, v in sample.items() if k not in {"actions", "action_mask"}}
        elif mode == "action":
            sample = {k: v for k, v in sample.items() if k in {"actions", "action_mask", "sample_id"}}
        return {**sample, "modality": mode, "source_name": source["name"]}

    def close(self):
        for dataset in self.sources:
            dataset.close()


class MixtureBatchSampler:
    """One source/modality per batch; both source choice and cursors are saved."""
    def __init__(self, dataset, batch_size, seed):
        self.dataset = dataset
        self.generator = torch.Generator().manual_seed(seed)
        self.samplers = [StatefulBatchSampler(len(s), batch_size, seed + i + 1)
                         for i, s in enumerate(dataset.sources)]
        self.draws = {s["name"]: 0 for s in dataset.manifest["sources"]}

    @property
    def epoch(self):
        return {s["name"]: sampler.epoch
                for s, sampler in zip(self.dataset.manifest["sources"], self.samplers)}

    def next(self, stage="warmup"):
        weights = torch.tensor([s["weights"].get(stage, 0) for s in self.dataset.manifest["sources"]],
                               dtype=torch.float64)
        if stage not in ALLOWED or weights.sum() <= 0:
            raise ValueError(f"No eligible data sources for {stage}")
        source = int(torch.multinomial(weights, 1, generator=self.generator))
        self.draws[self.dataset.manifest["sources"][source]["name"]] += 1
        return [i + self.dataset.offsets[source] for i in self.samplers[source].next()]

    def state_dict(self):
        return {"manifest_sha256": self.dataset.manifest["manifest_sha256"],
                "generator": self.generator.get_state(),
                "draws": self.draws.copy(),
                "sources": [s.state_dict() for s in self.samplers]}

    def load_state_dict(self, state):
        if state["manifest_sha256"] != self.dataset.manifest["manifest_sha256"]:
            raise ValueError("Mixture identity changed")
        if len(state["sources"]) != len(self.samplers):
            raise ValueError("Mixture source count changed")
        self.generator.set_state(state["generator"].cpu())
        self.draws = state["draws"].copy()
        for sampler, saved in zip(self.samplers, state["sources"]):
            sampler.load_state_dict(saved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="Explicit source-manifest and stage-weight JSON")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = build_mixture(args.spec, VITAConfig.load(args.config))
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    print(json.dumps({"manifest": str(output), "sha256": result["manifest_sha256"],
                      "sources": len(result["sources"]), "counts": result["counts"]}))


if __name__ == "__main__":
    main()
