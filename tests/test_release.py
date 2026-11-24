"""Synthetic privacy fixtures; strings below do not represent real credentials."""
import hashlib
import importlib.util
from pathlib import Path
import tarfile

import pytest


spec = importlib.util.spec_from_file_location(
    "release_builder", Path(__file__).resolve().parents[1] / "scripts/build_release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


@pytest.fixture
def release_root(tmp_path):
    root = tmp_path / "source"
    names = sorted({".release-files", "README.md", "LICENSE", "NOTICE", "pyproject.toml",
                    "scripts/build_release.py", "vita/__init__.py"})
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic release fixture\n")
    (root / ".release-files").write_text("\n".join(names) + "\n")
    return root


def test_release_deterministic_allowlist_and_metadata(release_root, tmp_path):
    (release_root / "reports").mkdir()
    (release_root / "reports/history.log").write_text("Synthetic excluded run log")
    (release_root / ".env").write_text("SYNTHETIC_ONLY=1")
    first, digest, count = release.build_release(release_root, tmp_path / "one")
    second, other_digest, _ = release.build_release(release_root, tmp_path / "two")
    assert digest == other_digest == hashlib.sha256(first.read_bytes()).hexdigest()
    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first) as archive:
        assert len(archive.getmembers()) == count
        for member in archive:
            assert member.isfile() and member.uid == member.gid == member.mtime == 0
            assert member.uname == member.gname == ""
            assert not member.name.endswith((".env", ".log"))
    with pytest.raises(FileExistsError):
        release.build_release(release_root, tmp_path / "one")


@pytest.mark.parametrize("text,rule", [
    ("/" + "Users" + "/synthetic-person/data", "private-home-path"),
    ("/" + "mnt" + "/synthetic-volume/data", "private-mount-path"),
    ("gh" + "p_" + "x" * 30, "access-token"),
    ("synthetic" + "@" + "example.invalid", "email-address"),
    ("-----BEGIN " + "PRIVATE KEY-----", "private-key"),
])
def test_release_rejects_sensitive_text_without_echo(release_root, text, rule):
    (release_root / "README.md").write_text(text)
    with pytest.raises(ValueError) as error:
        release.audit(release_root)
    assert rule in str(error.value)
    assert text not in str(error.value)


def test_release_rejects_symlinks_and_traversal(release_root, tmp_path):
    target = release_root / "NOTICE"
    target.unlink()
    target.symlink_to(release_root / "LICENSE")
    with pytest.raises(ValueError, match="Symlink"):
        release.audit(release_root)
    target.unlink()
    target.write_text("Synthetic notice")
    with (release_root / ".release-files").open("a") as stream:
        stream.write("../outside.md\n")
    with pytest.raises(ValueError, match="Unsafe"):
        release.audit(release_root)


def test_actual_project_release_audit():
    contents = release.audit(Path(__file__).resolve().parents[1])
    assert "vita/libero_recipe.py" in contents
    assert not any(name.startswith(("reports/", "runs/", ".git/")) for name in contents)
