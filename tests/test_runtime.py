"""Synthetic environment checks; no GPU access or remote connections."""
import pytest

from vita.runtime import check_device_scope, dependency_versions, machine_metadata, require_checkpoint_source


def test_cuda_requires_explicit_allocation(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    check_device_scope("cpu")
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        check_device_scope("cuda:0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.delenv("VITA_EXPECTED_HOST", raising=False)
    check_device_scope("cuda:0")


def test_optional_host_guard_is_operator_supplied(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("VITA_EXPECTED_HOST", "synthetic-allowed-host")
    monkeypatch.setattr("vita.runtime.socket.gethostname", lambda: "synthetic-different-host")
    with pytest.raises(RuntimeError, match="hostname"):
        check_device_scope("cuda")
    monkeypatch.setattr("vita.runtime.socket.gethostname", lambda: "synthetic-allowed-host")
    check_device_scope("cuda")


def test_machine_identifiers_are_opt_in(monkeypatch):
    monkeypatch.delenv("VITA_RECORD_MACHINE_METADATA", raising=False)
    assert machine_metadata() == {}
    monkeypatch.setenv("VITA_RECORD_MACHINE_METADATA", "1")
    monkeypatch.setattr("vita.runtime.socket.gethostname", lambda: "synthetic-host")
    assert machine_metadata()["hostname"] == "synthetic-host"


@pytest.mark.parametrize("saved", [{}, {"source_sha256": None}, {"source_sha256": {"fixture.py": "old"}}])
def test_evaluator_rejects_missing_or_changed_source(saved):
    with pytest.raises(ValueError, match="current evaluator"):
        require_checkpoint_source(saved, {"fixture.py": "current"})


def test_evaluator_source_and_versions_have_no_machine_paths():
    require_checkpoint_source({"source_sha256": {"fixture.py": "current"}}, {"fixture.py": "current"})
    result = dependency_versions(("torch", "synthetic-package-that-does-not-exist"))
    assert result["packages"]["torch"] is not None
    assert result["packages"]["synthetic-package-that-does-not-exist"] is None
    assert set(result) == {"python", "packages"}
