"""Diagnostic script for the remaining parser issues after Day 2.

Usage:
    uv run python scripts/inspect_failures.py

Investigates:
- Disney 2025 Item 7: spot-check showed weird first 200 chars; need to see more
  context to determine if it's actually wrong or if Disney's MD&A genuinely
  opens with prior-year cross-references as transition boilerplate.
- Disney 2022 Item 7: control case (no prior-year context to reference);
  should look like a clean MD&A start.
- Citigroup: all 4 years failed completely (diagnostic score -1.0 = no
  candidate pair found). What does Citi's document structure actually look like?
- BAC 2022 Item 7: missing across all years; Item 1A works fine. Likely an
  end-marker issue (banks may combine MD&A with quantitative disclosures).
"""

import json
from pathlib import Path

from sec_rag.ingest.parse import (
    _extract_body_documents,
    _find_all_item_candidates,
    _html_to_text,
)


def section(title: str) -> None:
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    # 1. Disney 2025 Item 7 — more context
    section("DISNEY 2025 Item 7 — first 1500 chars (need more than 250)")
    p = Path("data/chunks/DIS_2025_item7.json")
    if p.exists():
        d = json.loads(p.read_text())
        print(f"Length: {d['char_count']:,} chars  Score: {d['extraction_score']:.1f}")
        print(d["text"][:1500])
    else:
        print(f"  MISSING: {p}")
    print()

    # 2. Disney 2022 Item 7 — control case
    section("DISNEY 2022 Item 7 — first 1000 chars (control)")
    p = Path("data/chunks/DIS_2022_item7.json")
    if p.exists():
        d = json.loads(p.read_text())
        print(f"Length: {d['char_count']:,} chars  Score: {d['extraction_score']:.1f}")
        print(d["text"][:1000])
    else:
        print(f"  MISSING: {p}")
    print()

    # 3. Citigroup structure
    section("CITI 2022 — document structure")
    raw_path = Path("data/filings/C_2022.html")
    if not raw_path.exists():
        print(f"  MISSING: {raw_path}")
    else:
        raw = raw_path.read_text(encoding="utf-8", errors="replace")
        body, count = _extract_body_documents(raw)
        plain = _html_to_text(body)
        print(f"Body docs: {count}   Plain text: {len(plain):,} chars")
        candidates = _find_all_item_candidates(plain)
        print(f"Total Item-N candidates: {len(candidates)}")
        items: dict[str, list[int]] = {}
        for pos, item in candidates:
            items.setdefault(item, []).append(pos)
        for item in ["1A", "1B", "2", "7", "7A", "8"]:
            positions = items.get(item, [])
            print(f"  Item {item:3s}: {len(positions)} occurrences  first 5: {positions[:5]}")

        # Show context around the first 3 Item 1A occurrences
        print("\nFirst 3 'Item 1A' occurrences with context:")
        for pos in items.get("1A", [])[:3]:
            ctx = plain[max(0, pos - 80) : pos + 150].replace("\n", " ")
            print(f"  pos {pos:>10}: ...{ctx}...")
    print()

    # 4. BAC 2022 — why is Item 7 missing?
    section("BAC 2022 — why is Item 7 missing?")
    raw_path = Path("data/filings/BAC_2022.html")
    if not raw_path.exists():
        print(f"  MISSING: {raw_path}")
    else:
        raw = raw_path.read_text(encoding="utf-8", errors="replace")
        body, count = _extract_body_documents(raw)
        plain = _html_to_text(body)
        print(f"Body docs: {count}   Plain text: {len(plain):,} chars")
        candidates = _find_all_item_candidates(plain)
        items = {}
        for pos, item in candidates:
            items.setdefault(item, []).append(pos)
        for item in ["7", "7A", "8"]:
            positions = items.get(item, [])
            print(f"  Item {item:3s}: {len(positions)} occurrences  first 5: {positions[:5]}")

        print("\nFirst 3 'Item 7' occurrences with context:")
        for pos in items.get("7", [])[:3]:
            ctx = plain[max(0, pos - 80) : pos + 150].replace("\n", " ")
            print(f"  pos {pos:>10}: ...{ctx}...")
    print()


if __name__ == "__main__":
    main()
