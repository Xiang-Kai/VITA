"""Check local Markdown/HTML documentation links without network requests."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


def without_code(text):
    return re.sub(r"^(`{3,}|~{3,})[^\n]*\n.*?^\1[ \t]*$", "", text, flags=re.M | re.S)


def anchors(text):
    result = set(re.findall(r'\bid=["\']([^"\']+)["\']', without_code(text)))
    counts = {}
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", without_code(text), re.M):
        plain = re.sub(r"<[^>]*>", "", heading).strip().lower()
        slug = re.sub(r"[^\w\- ]", "", plain).replace(" ", "-")
        number = counts.get(slug, 0)
        counts[slug] = number + 1
        result.add(f"{slug}-{number}" if number else slug)
    return result


def check_docs(root):
    root = Path(root).resolve()
    paths = [root / "README.md", *sorted((root / "docs").rglob("*.md"))]
    errors = []
    checked = 0
    for source in paths:
        text = without_code(source.read_text())
        links = re.findall(r"!?\[[^\]\n]*\]\(([^)\n]+)\)", text)
        links += re.findall(r'\bhref=["\']([^"\']+)["\']', text)
        for target in links:
            target = target.split(' "', 1)[0].strip("<>")
            parts = urlsplit(target)
            if parts.scheme or parts.netloc:
                continue
            checked += 1
            destination = (source.parent / unquote(parts.path)).resolve() if parts.path else source
            label = source.relative_to(root).as_posix()
            if not destination.is_relative_to(root):
                errors.append(f"{label}: link escapes the repository")
            elif not destination.exists():
                errors.append(f"{label}: missing local target {target}")
            elif parts.fragment and destination.suffix == ".md":
                if unquote(parts.fragment) not in anchors(destination.read_text()):
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
