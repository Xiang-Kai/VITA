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


def test_reference_links_nested_badges_html_images_and_encoded_paths(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/a page.md").write_text("# Start\n", encoding="utf-8")
    (tmp_path / "docs/badge.svg").write_text("<svg/>", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        '# Home\n[Page][guide]\n\n[guide]: docs/a%20page.md#start "A title"\n\n'
        '[![Badge](docs/badge.svg)](docs/a%20page.md#start)\n'
        '<img src="docs/badge.svg">\n', encoding="utf-8")
    assert checker.check_docs(tmp_path) == (2, 4)


@pytest.mark.parametrize("markup", [
    "[Page][missing]\n\n[missing]: absent.md\n",
    '![Image](absent.svg "Example")\n',
    '<img src="absent.svg">\n',
    '<a href="absent.md">Page</a>\n',
    '[![Remote badge](https://example.invalid/badge)](absent.md)\n',
])
def test_all_rendered_local_link_types_are_checked(tmp_path, markup):
    (tmp_path / "README.md").write_text(markup, encoding="utf-8")
    with pytest.raises(ValueError, match="missing local target"):
        checker.check_docs(tmp_path)


def test_fences_inline_code_comments_and_setext_headings(tmp_path):
    (tmp_path / "README.md").write_text(
        "Start\n=====\n[Self](#start)\n\n"
        "````markdown\n```sample\n[Example](absent.md)\n```\n`````\n"
        "`[Example](absent.md)`\n\n"
        "    [Indented code](absent.md)\n\n"
        "<!-- [Comment](absent.md) -->\n", encoding="utf-8")
    assert checker.check_docs(tmp_path) == (1, 1)


def test_heading_collisions_entities_and_explicit_anchors():
    assert checker.anchors(
        "# Code\n## Code\n## Code-1\n## A &amp; B\n"
        '<a name="custom"></a>\n<div id="panel"></div>\n') == {
            "code", "code-1", "code-1-1", "a--b", "custom", "panel",
        }


def test_documentation_symlink_cannot_read_outside_repository(tmp_path):
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    (root / "README.md").write_text("# Home\n")
    outside = tmp_path / "outside.md"
    outside.write_text("# Synthetic external document\n")
    (root / "docs/linked.md").symlink_to(outside)
    with pytest.raises(ValueError, match="document escapes"):
        checker.check_docs(root)


def test_html_targets_and_url_encoded_escape(tmp_path):
    (tmp_path / "README.md").write_text('[Panel](panel.html#panel)\n')
    (tmp_path / "panel.html").write_text('<div id="panel">Content</div>\n')
    assert checker.check_docs(tmp_path) == (1, 1)
    (tmp_path / "README.md").write_text('[Outside](%2e%2e/outside.md)\n')
    with pytest.raises(ValueError, match="escapes"):
        checker.check_docs(tmp_path)
