"""Why does Citi's 1.6 MB document have zero Item-N markers?

Theory: inline-XBRL tags or table layout fragments the section headings such
that 'Item' and '1A' end up on separate lines, joined to neighboring words,
or wrapped in tags that survive the HTML strip.

Usage:
    uv run python scripts/inspect_citi.py
"""

import re
from pathlib import Path

from sec_rag.ingest.parse import _extract_body_documents, _html_to_text


def main() -> None:
    raw = Path("data/filings/C_2022.html").read_text(encoding="utf-8", errors="replace")
    body, count = _extract_body_documents(raw)

    print(f"Raw HTML body: {len(body):,} chars from {count} documents")
    plain = _html_to_text(body)
    print(f"Plain text: {len(plain):,} chars")
    print()

    # 1. Search for "Item" anywhere — does the word even appear?
    item_count = len(re.findall(r"\bItem\b", plain, re.IGNORECASE))
    print(f"'Item' (any) appears: {item_count} times")

    # 2. Search for "Risk Factors" specifically
    rf_count = len(re.findall(r"Risk\s+Factors", plain, re.IGNORECASE))
    print(f"'Risk Factors' appears: {rf_count} times")

    # 3. Search for "1A" anywhere
    one_a = len(re.findall(r"\b1A\b", plain))
    print(f"'1A' (standalone) appears: {one_a} times")

    # 4. Search for variants where Item and number are joined or separated
    print("\n=== Looking for fragmented 'Item 1A' patterns ===")
    for label, pattern in [
        ("Item1A (no space)", r"\bItem1A\b"),
        ("Item  1A (double space)", r"\bItem  +1A\b"),
        ("Item\\n1A (newline between)", r"\bItem\s*\n\s*1A\b"),
        ("ITEM (caps)", r"\bITEM\s+1A\b"),
        ("1A. (period)", r"\b1A\."),
        ("Item 1.A (period in middle)", r"\bItem\s+1\.A\b"),
    ]:
        matches = re.findall(pattern, plain, re.IGNORECASE)
        print(f"  {label:35s}: {len(matches)} hits")

    # 5. Show the first appearance of 'Risk Factors' with surrounding context
    if rf_count > 0:
        print("\n=== First 5 'Risk Factors' occurrences with 200 chars context ===")
        for m in list(re.finditer(r"Risk\s+Factors", plain, re.IGNORECASE))[:5]:
            pos = m.start()
            ctx = plain[max(0, pos - 100) : pos + 200].replace("\n", " ")
            print(f"  pos {pos:>10}: ...{ctx}...")

    # 6. Show what the document looks like at 5%, 25%, 50% positions
    print("\n=== Document samples (300 chars each) ===")
    for pct in [5, 15, 25, 35, 50]:
        pos = int(len(plain) * pct / 100)
        sample = plain[pos : pos + 300].replace("\n", " ")
        print(f"\n  {pct}% (pos {pos:>10,}):")
        print(f"  {sample[:280]}")


if __name__ == "__main__":
    main()
