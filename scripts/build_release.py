"""Allowlist-only source release; no git history, runtime artifacts or downloads.

This is a conservative automated check, not a guarantee that arbitrary prose is
free of personal information. Review the explicit .release-files list manually.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
from pathlib import Path, PurePosixPath
import re
import tarfile


TOP_LEVEL = {".release-files", ".gitignore", ".gitattributes", "README.md",
             "LICENSE", "NOTICE", "pyproject.toml", "MANIFEST.in"}
SOURCE_ROOTS = {"vita", "tests", "configs", "docs", "scripts", ".github", "constraints"}
TEXT_SUFFIXES = {".py", ".json", ".md", ".toml", ".sh", ".yml", ".yaml"}
RULES = {
    "private-home-path": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+|/[r]oot/|[A-Za-z]:\\Users\\", re.I),
    "private-mount-path": re.compile(r"/(?:mnt|media)/[A-Za-z0-9_.-]+"),
    "notebook-host-id": re.compile(r"\bnb-[a-z0-9]{8,}(?:-\d+)?\b", re.I),
    "email-address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "access-token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
                               r"sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})\b"),
    "credential-url": re.compile(r"https?://[^/\s:@]+:[^/\s@]+@"),
    "literal-secret": re.compile(r"""(?i)(?:api_key|password|access_token|client_secret)\s*[:=]\s*["'][^"'\n]{8,}["']"""),
    "public-ip-address": re.compile(r"\b(?!(?:127|0|255)\.)(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def audit(root):
    root = Path(root).resolve()
    manifest = root / ".release-files"
    if manifest.is_symlink():
        raise ValueError("Symlink in release manifest")
    names = [line.strip() for line in manifest.read_text().splitlines() if line.strip() and not line.startswith("#")]
    if len(names) != len(set(names)) or ".release-files" not in names:
        raise ValueError("Release list must be unique and include itself")
    required = {"README.md", "LICENSE", "NOTICE", "pyproject.toml", "scripts/build_release.py", "vita/__init__.py"}
    if not required <= set(names):
        raise ValueError("Release list is missing required project metadata")
    contents, errors = {}, []
    for name in sorted(names):
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or str(relative) != name:
            raise ValueError("Unsafe release member path")
        if name not in TOP_LEVEL and not (
                len(relative.parts) > 1 and relative.parts[0] in SOURCE_ROOTS and
                (relative.suffix in TEXT_SUFFIXES or (relative.parts[0] == "constraints" and relative.suffix == ".txt")) and
                not any(part.startswith(".") for part in relative.parts[1:])):
            raise ValueError(f"Disallowed release member: {name}")
        path = root / name
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root):
            raise ValueError(f"Symlink in release member: {name}")
        payload = path.read_bytes()
        if len(payload) > 2 * 1024**2 or b"\x00" in payload:
            raise ValueError(f"Binary or oversized release member: {name}")
        text = payload.decode("utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            for rule, pattern in RULES.items():
                if pattern.search(line):
                    errors.append(f"{name}:{number}: {rule}")
        contents[name] = payload
    if errors:
        # Never include the matched text, credentials or path values in diagnostics.
        raise ValueError("Release privacy checks failed:\n" + "\n".join(errors))
    return contents


def build_release(root, output):
    contents = audit(root)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "vita-reference-source.tar.gz"
    checksums = destination / "SHA256SUMS"
    if archive.exists() or checksums.exists():
        raise FileExistsError("Use a new output directory; existing release artifacts are preserved")
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for name, payload in contents.items():
                info = tarfile.TarInfo(f"vita-reference/{name}")
                info.size, info.mode, info.mtime = len(payload), 0o644, 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                tar.addfile(info, io.BytesIO(payload))
    data = buffer.getvalue()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = tar.getmembers()
        if [m.name for m in members] != [f"vita-reference/{name}" for name in contents]:
            raise ValueError("Archive member mismatch")
        for member, payload in zip(members, contents.values()):
            if (not member.isfile() or member.uid or member.gid or member.mtime or member.uname or member.gname or
                    tar.extractfile(member).read() != payload):
                raise ValueError("Archive content or metadata mismatch")
    digest = hashlib.sha256(data).hexdigest()
    with archive.open("xb") as stream:
        stream.write(data)
    with checksums.open("x") as stream:
        stream.write(f"{digest}  {archive.name}\n")
    return archive, digest, len(contents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="Omit for an audit without building")
    args = parser.parse_args()
    if args.output:
        archive, digest, count = build_release(args.root, args.output)
        print(f"Built {archive.name}: {count} allowlisted files; SHA256 {digest}")
    else:
        print(f"Audited {len(audit(args.root))} allowlisted text files")


if __name__ == "__main__":
    main()
