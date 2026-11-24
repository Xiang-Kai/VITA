from argparse import Namespace
import json

from vita.validate import validate


def test_complete_validation_receipt(tmp_path):
    result = validate(Namespace(
        config=None, device="cpu", steps=1, batch_size=2, seed=17,
        learning_rate=1e-3, libero_hdf5=None, frame_stride=20, output=str(tmp_path),
    ))
    assert result["status"] == "numerical_validation_passed"
    assert result["benchmark_reproduced"] is False
    assert result["data"]["kind"] == "synthetic_only"
    assert result["checkpoint_roundtrip_exact"]
    assert set(result["stages"]) == {"warmup", "cotrain", "finetune"}
    assert result["learning_diagnostics"]["unique_sequences_in_batch"] >= 1
    assert (tmp_path / "checkpoint.pt").is_file()
    assert json.loads((tmp_path / "metrics.json").read_text())["status"] == result["status"]
