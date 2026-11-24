"""Deterministic recording-level data fractions with training-only refitted statistics."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import math
from pathlib import Path

import h5py
import numpy as np

from .dataset import atomic_json
from .mixture import load_data_manifest, seal


def selected_groups(groups, fraction, seed):
    ranked = sorted(set(groups), key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).digest())
    return set(ranked[:max(1, math.ceil(len(ranked) * fraction))])


def training_subset(manifest, fraction=0.1, seed=17):
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Training fraction must be in (0,1]")
    result = deepcopy(manifest)
    result.pop("manifest_sha256", None)
    if manifest.get("kind") == "mixture":
        cache = {}
        for source in result["sources"]:
            identity = source["manifest"]["manifest_sha256"]
            if identity not in cache:
                cache[identity] = training_subset(source["manifest"], fraction, seed)
            source["manifest"] = deepcopy(cache[identity])
        result["counts"] = {s: sum(x["manifest"]["counts"][s] for x in result["sources"])
                            for s in ("train", "validation", "test")}
        result["unique_manifest_counts"] = {
            s: sum(m["counts"][s] for m in cache.values()) for s in ("train", "validation", "test")}
    else:
        canonical = manifest.get("kind") == "canonical_episodes"
        files = [manifest] if canonical else manifest["files"]
        retained, records = set(), []
        total_groups, kept_groups = 0, 0
        for index, info in enumerate(files):
            path = Path(info["path"])
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (info["size_bytes"], info["mtime_ns"]):
                raise ValueError("Dataset file changed since manifest")
            groups = [e.get("group_id", name) for name, e in info["episodes"].items() if e["split"] == "train"]
            keep = selected_groups(groups, fraction, seed)
            total_groups += len(set(groups))
            kept_groups += len(keep)
            for name, episode in info["episodes"].items():
                if episode["split"] == "train" and episode.get("group_id", name) in keep:
                    retained.add((index, name))
                    records.append((info, name))
            destination = result["episodes"] if canonical else result["files"][index]["episodes"]
            for name in list(destination):
                if destination[name]["split"] == "train" and (index, name) not in retained:
                    del destination[name]
        result["windows"] = [w for w in result["windows"]
                             if w["split"] != "train" or (w.get("file", 0), w["episode"]) in retained]
        result["counts"] = {s: sum(w["split"] == s for w in result["windows"])
                            for s in ("train", "validation", "test")}
        if not result["counts"]["train"]:
            raise ValueError("The retained subset has no training windows")
        width = manifest["action_dim"]
        sums, squares, counts = (np.zeros(width, dtype=np.float64) for _ in range(3))
        if manifest.get("mode") != "video":
            columns = manifest["action_columns"] if canonical else list(range(width))
            for info, name in records:
                with h5py.File(info["path"], "r") as handle:
                    actions = handle[f"{'episodes' if canonical else 'data'}/{name}/actions"]
                    for offset in range(0, len(actions), 8192):
                        values = actions[offset:offset + 8192].astype(np.float64)
                        if not np.isfinite(values).all():
                            raise ValueError("Nonfinite retained training actions")
                        sums[columns] += values.sum(0)
                        squares[columns] += (values**2).sum(0)
                        counts[columns] += len(values)
        mean = sums / np.maximum(counts, 1)
        std = np.sqrt(np.maximum(squares / np.maximum(counts, 1) - mean**2, 0))
        std[std < 1e-6] = 1
        result["normalization"] = {
            "mean": mean.tolist(), "std": std.tolist(), "fit_split": "train",
            "valid_timesteps_per_dimension": counts.tolist(),
            "method": "all timesteps of retained training recordings only; z-score",
        }
        result["retained_training_groups"] = kept_groups
        result["original_training_groups"] = total_groups
    result["training_subset"] = {
        "requested_fraction": fraction, "seed": seed,
        "source_manifest_sha256": manifest["manifest_sha256"],
        "rounding": "ceil, minimum one recording per LIBERO task / canonical source",
        "validation_and_test_changed": False,
    }
    return seal(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    result = training_subset(load_data_manifest(args.manifest), args.fraction, args.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, result)


if __name__ == "__main__":
    main()
