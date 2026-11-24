import json
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from vita.dataset import atomic_json
from vita.sources import (PAPER_SOURCES, import_hdf5, import_rlds, import_videos,
                          one_second_stride, paper_mixture, write_episodes)
from vita.mixture import build_episode_manifest
from test_mixture import source_fixture


def mapping():
    return {"mode": "paired", "images": [{"path": "observation/rgb"}],
            "actions": [{"path": "action/xyz", "scale": [1, 2, 3]}, {"path": "action/grip"}],
            "states": [{"path": "observation/state", "indices": [2, 0]}],
            "timestamp": "time", "action_convention": "synthetic_fixture_only",
            "control_hz": 20, "provenance": "offline_synthetic_unit_test"}


def step(index):
    return {"observation": {"rgb": np.full((8, 8, 3), index, np.uint8),
                            "state": np.array([1, 2, 3], np.float32)},
            "action": {"xyz": np.array([1, 2, 3], np.float32), "grip": np.array(0.5)},
            "time": index / 20}


def record():
    return {"id": "test", "group_id": "synthetic_recording", "split": "train", "instruction": "synthetic movement"}


def test_canonical_writer_mapping_and_partial_failure(tmp_path):
    target = tmp_path / "episode.hdf5"
    write_episodes([(record(), [step(0), step(1)])], target, mapping())
    with h5py.File(target, "r") as handle:
        np.testing.assert_array_equal(handle["episodes/test/actions"][0], [1, 4, 9, 0.5])
        np.testing.assert_array_equal(handle["episodes/test/states"][0], [3, 1])
        assert handle.attrs["conversion_complete"]
    with pytest.raises(ValueError, match="timestamps"):
        write_episodes([(record(), [step(1), step(0)])], tmp_path / "bad.hdf5", mapping())
    assert not (tmp_path / "bad.hdf5").exists()
    assert (tmp_path / "bad.hdf5.partial").exists()
    with pytest.raises(ValueError, match="text annotation"):
        write_episodes([({**record(), "instruction": ""}, [step(0)])], tmp_path / "nolabel.hdf5", mapping())


def test_hdf5_import_streams_values_from_explicit_fields(tmp_path):
    source = tmp_path / "original.hdf5"
    with h5py.File(source, "w") as handle:
        handle["demo/observation/rgb"] = np.zeros((2, 8, 8, 3), np.uint8)
        handle["demo/observation/state"] = [[1, 2, 3], [4, 5, 6]]
        handle["demo/action/xyz"] = [[1, 2, 3], [4, 5, 6]]
        handle["demo/action/grip"] = [0.5, 0.5]
        handle["demo/time"] = [0, 0.05]
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps({**record(), "path": source.name, "episode_path": "demo"}) + "\n")
    import_hdf5(index, tmp_path, tmp_path / "converted.hdf5", mapping())
    with h5py.File(tmp_path / "converted.hdf5", "r") as handle:
        np.testing.assert_array_equal(handle["episodes/test/actions"][1], [4, 10, 18, 0.5])


def test_local_rlds_adapter_with_explicit_synthetic_builder(monkeypatch, tmp_path):
    calls = []
    class Builder:
        def as_dataset(self, **kwargs):
            calls.append(kwargs)
            return [{"steps": [step(0), step(1)]}]
    tfds = SimpleNamespace(builder_from_directory=lambda path: Builder(), as_numpy=lambda value: value)
    monkeypatch.setitem(sys.modules, "tensorflow_datasets", tfds)
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps({**record(), "source_split": "train", "source_index": 0}))
    import_rlds(index, tmp_path, tmp_path / "rlds.hdf5", mapping())
    assert calls == [{"split": "train", "shuffle_files": False}]


def test_video_adapter_with_synthetic_decoded_frames(monkeypatch, tmp_path):
    class Capture:
        closed = False
        def __init__(self, path):
            self.time = 0
        def isOpened(self):
            return True
        def get(self, prop):
            return 30
        def set(self, prop, value):
            self.time = value
        def read(self):
            return True, np.full((8, 8, 3), int(self.time / 1000), np.uint8)
        def release(self):
            Capture.closed = True
    cv2 = SimpleNamespace(VideoCapture=Capture, CAP_PROP_FPS=1, CAP_PROP_POS_MSEC=2,
                          COLOR_BGR2RGB=3, cvtColor=lambda image, _: image[..., ::-1])
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps({**record(), "path": "synthetic.mp4", "start_seconds": 0, "end_seconds": 13}))
    import_videos(index, tmp_path, tmp_path / "video.hdf5")
    with h5py.File(tmp_path / "video.hdf5", "r") as handle:
        assert len(handle["episodes/test/images"]) == 13
        assert "actions" not in handle["episodes/test"]
    assert Capture.closed


def test_thirteen_source_recipe_is_complete_and_explicit(config, tmp_path):
    manifests = {}
    for name, (family, _) in PAPER_SOURCES.items():
        mode = "video" if family == "video" else "paired"
        manifest = source_fixture(tmp_path / name, config, mode)
        path = tmp_path / name / "manifest.json"
        atomic_json(path, manifest)
        manifests[name] = str(path)
    result = paper_mixture(manifests, tmp_path / "recipe", config)
    assert len(result["sources"]) == 13
    assert result["original_paper_mixture_recovered"] is False
    for source in result["sources"]:
        assert source["weights"]["cotrain"] == PAPER_SOURCES[source["name"]][1]
    with pytest.raises(ValueError, match="13"):
        paper_mixture({}, tmp_path / "invalid", config)
    assert one_second_stride(30) == 30
    with pytest.raises(ValueError, match="timestamp"):
        one_second_stride(29.97)


def test_short_episode_filter_records_exclusion(config, tmp_path):
    manifest = source_fixture(tmp_path, config, "video")
    with h5py.File(manifest["path"], "r+") as handle:
        short = handle.create_group("episodes/short")
        short.attrs.update(split="train", group_id="short", instruction="synthetic too short")
        short["images"] = np.zeros((2, 1, 16, 16, 3), np.uint8)
    with pytest.raises(ValueError, match="shorter"):
        build_episode_manifest(manifest["path"], config, mode="video")
    filtered = build_episode_manifest(manifest["path"], config, mode="video", filter_short=True)
    assert filtered["excluded_episodes"] == [
        {"episode": "short", "length": 2, "reason": "shorter_than_prediction_horizon"}]
    assert "short" not in filtered["episodes"]
