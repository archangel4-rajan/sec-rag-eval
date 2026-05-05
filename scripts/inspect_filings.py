"""Inspect downloaded 10-K files to inform the parser design.

Run: uv run python scripts/inspect_filings.py
"""

import re
from pathlib import Path

FILINGS_DIR = Path("data/filings")


def inspect():
    files = sorted(FILINGS_DIR.glob("*.html"))
    print(f"Total files: {len(files)}\n")

    # Per-file stats
    print(f"{'File':<25} {'Size (KB)':>10} {'First 60 chars of content':<60}")
    print("-" * 100)
    for f in files[:5]:
        size_kb = f.stat().st_size // 1024
        with open(f, encoding="utf-8", errors="replace") as fh:
            content = fh.read(2000)
        # Strip whitespace, show first non-blank chunk
        snippet = re.sub(r"\s+", " ", content).strip()[:60]
        print(f"{f.name:<25} {size_kb:>10} {snippet:<60}")

    # Pick one file for deeper inspection
    sample = files[0] if files else None
    if not sample:
        print("No files found!")
        return

    print(f"\n\n=== Deep inspection: {sample.name} ===\n")
    with open(sample, encoding="utf-8", errors="replace") as fh:
        content = fh.read()

    print(f"Total size: {len(content):,} chars")
    print(f"First 500 chars:\n{content[:500]}\n")

    # What format is this? Look for telltale markers.
    has_html = "<html" in content.lower() or "<HTML" in content
    has_xbrl = "<xbrl" in content.lower() or "<XBRL" in content
    has_sec_header = "<SEC-DOCUMENT>" in content or "<SEC-HEADER>" in content
    has_submission = "<SUBMISSION>" in content
    print(f"Contains <html>: {has_html}")
    print(f"Contains XBRL tags: {has_xbrl}")
    print(f"Contains <SEC-DOCUMENT>: {has_sec_header}")
    print(f"Contains <SUBMISSION>: {has_submission}")

    # How many doc sections are inside (filings often have many embedded docs)?
    doc_count = content.count("<DOCUMENT>")
    print(f"Number of <DOCUMENT> sections: {doc_count}")

    # Find Item 1A and Item 7 markers — case-insensitive, allowing varied formatting
    print("\n=== Item markers found ===")
    for pattern in [
        r"item\s*1a",
        r"item\s*7\b",
        r"item\s*7a",
        r"risk\s*factors",
        r"management.{0,3}s\s*discussion",
    ]:
        matches = list(re.finditer(pattern, content, re.IGNORECASE))
        positions = [m.start() for m in matches[:5]]
        print(f"  /{pattern}/  → {len(matches)} hits, first 5 positions: {positions}")

    # Save a 5KB sample around the first Item 1A hit for inspection
    item_1a = re.search(r"item\s*1a", content, re.IGNORECASE)
    if item_1a:
        start = max(0, item_1a.start() - 200)
        end = min(len(content), item_1a.start() + 1500)
        sample_path = Path("scripts/sample_item_1a_context.txt")
        sample_path.write_text(content[start:end])
        print(f"\nSample context around first Item 1A saved to: {sample_path}")


if __name__ == "__main__":
    inspect()
