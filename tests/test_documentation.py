"""Offline documentation checks; no website, simulator or API requests."""
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "documentation_checker", Path(__file__).resolve().parents[1] / "scripts/check_docs.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def test_public_documentation_links():
    documents, links = checker.check_docs(Path(__file__).resolve().parents[1])
    assert documents >= 8 and links >= 40


def test_documentation_checker_ignores_external_links_and_code(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text(
        '# Home\n[Remote](https://example.invalid/missing)\n[Page](docs/page.md#start)\n'
        '```bash\n[Not a link](missing.md)\n```\n')
    (tmp_path / "docs/page.md").write_text("# Start\n[Home](../README.md#home)\n")
    assert checker.check_docs(tmp_path) == (2, 2)


@pytest.mark.parametrize("link,reason", [
    ("missing.md", "missing local target"),
    ("README.md#absent", "missing heading"),
    ("../outside.md", "escapes"),
])
def test_documentation_checker_rejects_broken_local_links(tmp_path, link, reason):
    (tmp_path / "README.md").write_text(f"# Home\n[Broken]({link})\n")
    with pytest.raises(ValueError, match=reason):
        checker.check_docs(tmp_path)


def test_github_heading_slugs_and_duplicate_ids():
    assert checker.anchors("# 1. Install\n## Code `here`\n## Code `here`\n") == {
        "1-install", "code-here", "code-here-1",
    }
