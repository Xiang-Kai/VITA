"""Check rendered documentation links offline; requires the project's test extra."""
from __future__ import annotations

import argparse
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt


class HTMLReferences(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for key in ("href", "src"):
            if attrs.get(key):
                self.links.append(attrs[key])
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("name"):
            self.ids.add(attrs["name"])


def inline_text(tokens):
    pieces = []
    for token in tokens:
        if token.type in {"text", "code_inline"}:
            pieces.append(token.content)
        elif token.type in {"softbreak", "hardbreak"}:
            pieces.append(" ")
        elif token.type == "image":
            pieces.append(inline_text(token.children or []))
    return "".join(pieces)


def document_references(text):
    """Parse Markdown before checking it, so code examples never become links."""
    tokens = MarkdownIt("commonmark", {"html": True}).parse(text)
    html = HTMLReferences()
    links, ids, heading_ids = [], set(), set()

    def visit(token):
        if token.type == "link_open":
            links.append(token.attrGet("href"))
        elif token.type == "image":
            links.append(token.attrGet("src"))
        elif token.type in {"html_inline", "html_block"}:
            html.feed(token.content)
        for child in token.children or []:
            visit(child)

    for index, token in enumerate(tokens):
        visit(token)
        if token.type == "heading_open":
            plain = unescape(inline_text(tokens[index + 1].children or [])).lower()
            base = re.sub(r"[^\w\- ]", "", plain).replace(" ", "-")
            slug, suffix = base, 0
            while slug in heading_ids:
                suffix += 1
                slug = f"{base}-{suffix}"
            heading_ids.add(slug)
    html.close()
    ids.update(heading_ids)
    ids.update(html.ids)
    return ids, links + html.links


def anchors(text):
    return document_references(text)[0]


def check_docs(root):
    root = Path(root).resolve()
    paths = [root / "README.md", *sorted((root / "docs").rglob("*.md"))]
    cache, errors = {}, []
    checked = 0

    def references(path):
        if path not in cache:
            cache[path] = document_references(path.read_text(encoding="utf-8"))
        return cache[path]

    for source in paths:
        label = source.relative_to(root).as_posix()
        if not source.resolve().is_relative_to(root):
            errors.append(f"{label}: document escapes the repository")
            continue
        if not source.is_file():
            errors.append(f"{label}: missing document")
            continue
        for target in references(source)[1]:
            try:
                parts = urlsplit(target)
            except ValueError:
                errors.append(f"{label}: malformed link")
                continue
            if parts.scheme or parts.netloc:
                continue
            checked += 1
            destination = (source.parent / unquote(parts.path)).resolve() if parts.path else source
            if not destination.is_relative_to(root):
                errors.append(f"{label}: link escapes the repository")
            elif not destination.exists():
                errors.append(f"{label}: missing local target {target}")
            elif parts.fragment and destination.suffix.lower() in {".md", ".html"}:
                if unquote(parts.fragment) not in references(destination)[0]:
                    errors.append(f"{label}: missing heading {target}")
    if errors:
        raise ValueError("\n".join(errors))
    return len(paths), checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    documents, links = check_docs(args.root)
    print(f"Checked {links} local links in {documents} Markdown documents; no network requests")


if __name__ == "__main__":
    main()
