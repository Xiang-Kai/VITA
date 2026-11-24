"""Portable device guards and opt-in local machine diagnostics."""
import os
import importlib.metadata
import platform
import socket


def check_device_scope(device):
    if not str(device).startswith("cuda"):
        return
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES explicitly to your authorized allocation")
    expected = os.environ.get("VITA_EXPECTED_HOST")
    if expected and socket.gethostname() != expected:
        raise RuntimeError("Current hostname does not match VITA_EXPECTED_HOST")


def machine_metadata():
    """Hostnames and PIDs are unnecessary in public metrics unless explicitly requested."""
    if os.environ.get("VITA_RECORD_MACHINE_METADATA") == "1":
        return {"hostname": socket.gethostname(), "pid": os.getpid()}
    return {}


def require_checkpoint_source(checkpoint, current_hashes):
    """A checkpoint's recorded source is not proof of the evaluator's source."""
    if not current_hashes or checkpoint.get("source_sha256") != current_hashes:
        raise ValueError("Checkpoint source differs from the current evaluator; use the matching source release")


def dependency_versions(names=("torch", "numpy", "h5py", "transformers")):
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "packages": versions}
