"""Local video / RLDS / HDF5 ingestion with explicit schemas, never guessed controls."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

from .config import VITAConfig
from .dataset import atomic_json
from .mixture import build_episode_manifest, build_mixture


# Table 10 percentages are a selectable recipe, not the missing author manifest.
PAPER_SOURCES = {
    "ssv2": ("video", 20.47), "ego4d": ("video", 27.68),
    "bridge": ("rlds", 1.56), "fractal": ("rlds", 7.41),
    "fmb": ("rlds", 2.14), "kuka": ("rlds", 4.87),
    "berkeley_gnm_recon": ("rlds", 1.17), "droid": ("rlds", 5.26),
    "robomind_franka": ("hdf5", 4.48), "robomind_ur5e": ("hdf5", 5.07),
    "robomind_agilex": ("hdf5", 12.67), "calvin": ("calvin", 3.70),
    "libero": ("libero", 3.51),
}


def lookup(record, path):
    value = record
    for part in path.split("/"):
        value = value[part]
    return value


def vector(step, fields):
    values = []
    for field in fields:
        raw = np.asarray(lookup(step, field["path"]), dtype=np.float32).reshape(-1)
        if "indices" in field:
            indices = field["indices"]
            if not indices or any(type(i) is not int or i < 0 or i >= len(raw) for i in indices):
                raise ValueError("Invalid explicit source field indices")
            raw = raw[indices]
        raw = raw * np.asarray(field.get("scale", 1), dtype=np.float32) + np.asarray(field.get("offset", 0), dtype=np.float32)
        if raw.ndim != 1 or not np.isfinite(raw).all():
            raise ValueError("Invalid mapped control/state field")
        values.append(raw)
    return np.concatenate(values) if values else None


def validate_mapping(mapping):
    if mapping["mode"] not in {"video", "action", "paired"}:
        raise ValueError("Unknown source modality")
    hz = mapping["control_hz"]
    if not math.isfinite(hz) or hz <= 0 or not mapping["provenance"].strip():
        raise ValueError("Explicit source provenance and positive sample rate required")
    if mapping["mode"] != "video" and (
            not mapping.get("actions") or not mapping.get("action_convention", "").strip()):
        raise ValueError("Explicit action fields and coordinate convention required")
    if mapping["mode"] != "action" and not mapping.get("images"):
        raise ValueError("Explicit image fields required")


def write_episodes(episodes, output, mapping):
    """Consume (metadata, iterable-of-step-dicts); preserves source values unless mapped."""
    validate_mapping(mapping)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    groups, identities, count = {}, set(), 0
    # An exception retains only .partial; it cannot be mistaken for a completed dataset.
    with h5py.File(temporary, "x") as handle:
        handle.attrs.update(vita_episode_schema=1, control_hz=mapping["control_hz"],
                            action_convention=mapping.get("action_convention", ""),
                            provenance=mapping["provenance"], source_mapping=json.dumps(mapping))
        for metadata, steps in episodes:
            name, group, split = str(metadata["id"]), str(metadata["group_id"]), metadata["split"]
            if not name or "/" in name or not group or name in identities or split not in {"train", "validation", "test"}:
                raise ValueError("Invalid or duplicate episode metadata")
            if group in groups and groups[group] != split:
                raise ValueError("A physical recording crosses split boundaries")
            instruction = metadata.get("instruction", "")
            if mapping["mode"] != "action" and (not isinstance(instruction, str) or not instruction.strip()):
                raise ValueError("Missing source text annotation; no synthetic instruction is substituted")
            if metadata.get("annotation_quality", "accepted") != "accepted":
                raise ValueError("Index contains an annotation not accepted by its curator")
            identities.add(name)
            groups[group] = split
            episode = handle.create_group(f"episodes/{name}")
            episode.attrs.update(split=split, group_id=group, instruction=instruction,
                                 source_record=json.dumps(metadata))
            n, previous_timestamp = 0, None
            for step in steps:
                row = {}
                if mapping["mode"] != "action":
                    images = []
                    for field in mapping["images"]:
                        image = np.asarray(lookup(step, field["path"]))
                        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                            raise ValueError("Source images must be decoded HWC RGB uint8")
                        if field.get("flip_vertical", False):
                            image = image[::-1]
                        images.append(image)
                    if len({im.shape for im in images}) != 1:
                        raise ValueError("Camera sizes differ; explicitly preprocess to a common size")
                    row["images"] = np.stack(images)
                if mapping["mode"] != "video":
                    row["actions"] = vector(step, mapping["actions"])
                if mapping.get("states"):
                    row["states"] = vector(step, mapping["states"])
                if mapping.get("timestamp"):
                    timestamp = float(lookup(step, mapping["timestamp"]))
                    if not math.isfinite(timestamp) or (previous_timestamp is not None and timestamp <= previous_timestamp):
                        raise ValueError("Source timestamps must be finite and strictly increasing")
                    if (previous_timestamp is not None and
                            not math.isclose(timestamp - previous_timestamp, 1 / mapping["control_hz"],
                                             rel_tol=0.2, abs_tol=1e-6)):
                        raise ValueError("Irregular timestamps require explicit resampling before fixed-stride ingestion")
                    row["timestamps"] = np.array(timestamp, dtype=np.float64)
                    previous_timestamp = timestamp
                for key, value in row.items():
                    if n == 0:
                        episode.create_dataset(key, shape=(0, *value.shape), maxshape=(None, *value.shape),
                                               dtype=value.dtype, chunks=True, compression="lzf")
                    dataset = episode[key]
                    if dataset.shape[1:] != value.shape:
                        raise ValueError("Source shape changed within an episode")
                    dataset.resize(n + 1, axis=0)
                    dataset[n] = value
                n += 1
            if n == 0:
                raise ValueError("Empty source episode")
            count += 1
        if count == 0:
            raise ValueError("Empty source index")
        handle.attrs["conversion_complete"] = True
    temporary.rename(output)
    return {"episodes": count, "path": str(output.resolve()), "source_modified": False}


def read_index(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("Empty source index")
    return records


def import_hdf5(index, root, output, mapping):
    records, root = read_index(index), Path(root).resolve()

    def episodes():
        seen = {}
        for record in records:
            path = (root / record["path"]).resolve()
            identity = (str(path), record.get("episode_path", ""))
            if identity in seen and seen[identity] != record["split"]:
                raise ValueError("One source episode crosses split boundaries")
            seen[identity] = record["split"]
            with h5py.File(path, "r") as handle:
                episode = handle[record.get("episode_path", "/")]
                fields = [f["path"] for key in ("images", "actions", "states") for f in mapping.get(key, [])]
                if mapping.get("timestamp"):
                    fields.append(mapping["timestamp"])
                fields = list(dict.fromkeys(fields))
                lengths = {len(episode[key]) for key in fields}
                if len(lengths) != 1:
                    raise ValueError("Source HDF5 streams are not time aligned")
                # A flat-key dict is converted back to a nested mapping for the common reader.
                def steps():
                    for i in range(next(iter(lengths))):
                        step = {}
                        for key in fields:
                            target = step
                            parts = key.split("/")
                            for part in parts[:-1]:
                                target = target.setdefault(part, {})
                            target[parts[-1]] = episode[key][i]
                        yield step
                yield record, steps()
    return write_episodes(episodes(), output, mapping)


def import_rlds(index, builder_dir, output, mapping):
    """Read an already prepared local TFDS builder; never download or infer action keys."""
    import tensorflow_datasets as tfds

    records = read_index(index)
    grouped = {}
    for record in records:
        key = (record["source_split"], record["source_index"])
        if type(key[1]) is not int or key[1] < 0 or key in grouped:
            raise ValueError("RLDS index requires unique nonnegative source indices")
        grouped[key] = record
    builder = tfds.builder_from_directory(str(Path(builder_dir).resolve()))

    def episodes():
        found = set()
        for split in sorted({key[0] for key in grouped}):
            dataset = builder.as_dataset(split=split, shuffle_files=False)
            for index, episode in enumerate(dataset):
                key = (split, index)
                if key not in grouped:
                    continue
                found.add(key)
                yield grouped[key], tfds.as_numpy(episode["steps"])
        if found != set(grouped):
            raise ValueError("RLDS index references episodes not present in the local builder")
    return write_episodes(episodes(), output, mapping)


def import_videos(index, root, output, *, interval=1.0):
    """SSv2/Ego4D clip index -> sampled RGB; text and grouping come from real annotations."""
    import cv2

    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("Positive frame interval required")
    records, root = read_index(index), Path(root).resolve()

    def episodes():
        physical = {}
        for record in records:
            path = (root / record["path"]).resolve()
            if path in physical and physical[path] != record["split"]:
                raise ValueError("Clips of a video cross split boundaries")
            physical[path] = record["split"]
            capture = cv2.VideoCapture(str(path))
            try:
                fps = capture.get(cv2.CAP_PROP_FPS)
                if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
                    raise ValueError(f"Unreadable video or invalid FPS: {path}")
                start, end = float(record["start_seconds"]), float(record["end_seconds"])
                if not all(math.isfinite(x) for x in (start, end)) or not 0 <= start < end:
                    raise ValueError("Invalid clip time bounds")
                def steps():
                    for timestamp in np.arange(start, end, interval):
                        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                        success, frame = capture.read()
                        if not success:
                            raise ValueError(f"Missing indexed video frame: {path} at {timestamp}")
                        yield {"rgb": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), "time": timestamp}
                yield record, steps()
            finally:
                capture.release()
    return write_episodes(episodes(), output, {
        "mode": "video", "control_hz": 1 / interval, "images": [{"path": "rgb"}], "timestamp": "time",
        "provenance": f"Explicit annotated video index {Path(index).resolve()}; sampling interval={interval}s",
    })


def paper_mixture(manifests, output, config):
    """All 13 manifest identities are mandatory; Table 10 weights are normalized by the sampler."""
    if set(manifests) != set(PAPER_SOURCES):
        raise ValueError("Provide exactly the 13 named source manifests")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    sources = [{"name": name, "manifest": str(Path(manifests[name]).resolve()),
                "mode": "video" if family == "video" else "paired",
                "weights": {"warmup": percent, "cotrain": percent}}
               for name, (family, percent) in PAPER_SOURCES.items()]
    atomic_json(output / "sources.json", {
        "sources": sources,
        "provenance": "Independent 13-source recipe using displayed Table 10 percentages; exact author filtering unknown",
    })
    result = build_mixture(output / "sources.json", config)
    atomic_json(output / "mixture.json", result)
    return result


def one_second_stride(control_hz):
    stride = round(control_hz)
    if stride < 1 or not math.isclose(stride, control_hz, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Nonintegral frame rate needs timestamp-based resampling before fixed-stride indexing")
    return stride


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("hdf5", "rlds", "video"):
        command = sub.add_parser(name)
        command.add_argument("--index", required=True)
        command.add_argument("--root", required=True)
        command.add_argument("--output", required=True)
        if name != "video":
            command.add_argument("--mapping", required=True)
        else:
            command.add_argument("--interval", type=float, default=1)
    mixture = sub.add_parser("mixture")
    mixture.add_argument("--manifests", required=True, help="JSON mapping the 13 source names to manifest paths")
    mixture.add_argument("--config", required=True)
    mixture.add_argument("--output", required=True)
    index = sub.add_parser("index")
    index.add_argument("--input", required=True)
    index.add_argument("--config", required=True)
    index.add_argument("--output", required=True)
    index.add_argument("--mode", choices=["video", "paired", "action"], required=True)
    index.add_argument("--action-columns", type=int, nargs="+")
    index.add_argument("--state-columns", type=int, nargs="+")
    index.add_argument("--filter-short", action="store_true", help="Exclude short episodes with a retained reason ledger")
    args = parser.parse_args()
    if args.command in {"hdf5", "rlds"}:
        converter = import_hdf5 if args.command == "hdf5" else import_rlds
        result = converter(args.index, args.root, args.output, json.loads(Path(args.mapping).read_text()))
    elif args.command == "video":
        result = import_videos(args.index, args.root, args.output, interval=args.interval)
    elif args.command == "mixture":
        result = paper_mixture(json.loads(Path(args.manifests).read_text()), args.output, VITAConfig.load(args.config))
    else:
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(output)
        with h5py.File(args.input, "r") as handle:
            stride = one_second_stride(float(handle.attrs["control_hz"]))
        result = build_episode_manifest(args.input, VITAConfig.load(args.config), mode=args.mode,
                                        frame_stride=stride, action_columns=args.action_columns,
                                        state_columns=args.state_columns, filter_short=args.filter_short)
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(output, result)
    print(json.dumps({k: result[k] for k in ("path", "episodes", "counts", "manifest_sha256") if k in result}))


if __name__ == "__main__":
    main()
