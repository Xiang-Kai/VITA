from dataclasses import replace
import json

import h5py
import numpy as np
import pytest

from vita.prepare import convert_calvin, prepare_libero
from test_dataset import create_task


def test_prepare_libero_projects_modalities_without_copying_or_relabeling_splits(config, tmp_path):
    create_task(tmp_path / "source")
    result = prepare_libero(tmp_path / "source", tmp_path / "prepared", replace(config, num_views=2),
                            frame_stride=2, suites=["libero_object"])
    assert len(result["sources"]) == 3
    assert result["original_paper_mixture_recovered"] is False
    assert result["sources"][0]["manifest"] == result["sources"][2]["manifest"]
    with pytest.raises(FileExistsError):
        prepare_libero(tmp_path / "source", tmp_path / "prepared", config)


def test_stream_calvin_conversion_preserves_controls_and_labels(tmp_path):
    source = tmp_path / "npz"
    source.mkdir()
    for index in range(4):
        np.savez(source / f"episode_{index:07d}.npz",
                 rgb_static=np.full((16, 16, 3), index, dtype=np.uint8),
                 rgb_gripper=np.full((8, 8, 3), index, dtype=np.uint8),
                 rel_actions=np.ones(7, dtype=np.float32) * index,
                 robot_obs=np.ones(15, dtype=np.float32) * index)
    records = [{"id": "a", "group_id": "a", "split": "train", "start": 0, "end": 1,
                "instruction": "synthetic train"},
               {"id": "b", "group_id": "b", "split": "validation", "start": 2, "end": 3,
                "instruction": "synthetic validation"}]
    index = tmp_path / "annotations.jsonl"
    index.write_text("\n".join(json.dumps(row) for row in records))
    target = tmp_path / "calvin.hdf5"
    result = convert_calvin(source, index, target)
    assert result["episodes"] == 2
    with h5py.File(target, "r") as handle:
        assert handle["episodes/a/images"].shape == (2, 2, 16, 16, 3)
        np.testing.assert_array_equal(handle["episodes/b/actions"][0], np.full(7, 2))
    records[1]["start"] = 1
    index.write_text("\n".join(json.dumps(row) for row in records))
    with pytest.raises(ValueError, match="cross split"):
        convert_calvin(source, index, tmp_path / "invalid.hdf5")
