"""Day 2 tests: parser correctness against real filings.

Two layers of testing:

1. Pure unit tests on the scoring algorithm and helpers — no file I/O, always run.
2. Integration tests against real filings in data/filings/ — skipped if absent.

Integration tests cover:
- Apple parses cleanly (Pattern A: bold-styled headings, single 10-K)
- Amazon parses cleanly (Pattern A with table-layout headings)
- Corpus-wide success rate >= 90% across all filings
"""

from __future__ import annotations

import json

import pytest

from sec_rag.config import settings
from sec_rag.ingest.parse import (
    BODY_DOCUMENT_TYPES,
    CROSS_REF_PHRASES,
    ITEM_PATTERN,
    MIN_SECTION_CHARS,
    _extract_body_documents,
    _find_all_item_candidates,
    _html_to_text,
    _score_pair,
    parse_filing,
)

AAPL_2022 = settings.filings_dir / "AAPL_2022.html"
AMZN_2022 = settings.filings_dir / "AMZN_2022.html"
PARSE_REPORT = settings.chunks_dir / "parse_report.json"


# -- Pure unit tests (no file I/O) -------------------------------------------


def test_item_pattern_matches_basic_forms() -> None:
    """ITEM_PATTERN should match 'Item 1A', 'Item 7', 'Item 7A' across casings."""
    cases = {
        "Item 1A. Risk Factors": "1A",
        "see Item 7 of Part II": "7",
        "Item 7A.": "7A",
        "ITEM 1A": "1A",
        "item  1a.": "1A",
    }
    for text, expected in cases.items():
        m = ITEM_PATTERN.search(text)
        assert m is not None, f"Failed to match: {text!r}"
        assert m.group(1).upper() == expected


def test_item_pattern_rejects_non_section_uses() -> None:
    """Item-like strings that aren't section references should not match as 1A/7/7A."""
    for text in ["Items 1-5 are listed", "Item: foo", "menu item description"]:
        m = ITEM_PATTERN.search(text)
        if m:
            assert m.group(1).upper() not in {"1A", "7", "7A"}


def test_score_pair_rewards_long_gaps() -> None:
    """Larger gaps should produce higher scores (real sections have lots of prose)."""
    short = _score_pair(1000, 2000, "Officer, Corning Incorporated", {1000, 2000})
    long_ = _score_pair(1000, 50000, "Officer, Corning Incorporated", {1000, 50000})
    assert long_ > short


def test_score_pair_penalizes_crossref_context() -> None:
    """A cross-reference context should produce a much lower score."""
    clean = _score_pair(1000, 50000, "end of section heading", {1000, 50000})
    crossref = _score_pair(1000, 50000, "as discussed in Item 7 of Part II", {1000, 50000})
    assert clean - crossref >= 100


def test_score_pair_penalizes_dense_predecessors() -> None:
    """Markers densely packed before a candidate (TOC pattern) reduce its score."""
    clean = _score_pair(2000, 50000, "context", {2000, 50000})
    toc = _score_pair(2000, 50000, "context", {2000, 50000, 1700, 1750, 1800, 1850, 1900})
    assert clean > toc + 100


def test_html_to_text_strips_tags_and_decodes() -> None:
    raw = '<div><span style="font-weight:700">Item 1A.</span> <p>Apple&#8217;s risks</p></div>'
    out = _html_to_text(raw)
    assert "Item 1A" in out
    assert "<" not in out
    assert "  " not in out
    assert "&#" not in out


def test_cross_ref_phrases_cover_observed_patterns() -> None:
    """Every cross-ref pattern observed in real filings should be detected."""
    real_contexts = [
        "looking statements. See Item 1A of Part I",
        "as discussed in Item 7 of Part II",
        "described elsewhere in this Item 1A",
        "outlined in Item 1A of Part I",
        "factors discussed elsewhere in this report",
    ]
    for ctx in real_contexts:
        ctx_lower = ctx.lower()
        matched = [p for p in CROSS_REF_PHRASES if p in ctx_lower]
        assert matched, f"No cross-ref phrase matched: {ctx!r}"


def test_find_all_item_candidates_orders_by_position() -> None:
    """Candidates should be returned in document order with uppercase IDs."""
    plain = "Item 1A. Risk Factors. Some text. Item 1B. Comments. Later: Item 7. MD&A."
    candidates = _find_all_item_candidates(plain)
    items = [item for _, item in candidates]
    positions = [pos for pos, _ in candidates]
    assert items == ["1A", "1B", "7"]
    assert positions == sorted(positions)


def test_body_document_types_includes_known_patterns() -> None:
    """Sanity check the set of TYPEs we treat as body content."""
    assert "10-K" in BODY_DOCUMENT_TYPES
    assert "EX-13" in BODY_DOCUMENT_TYPES
    assert "EX-99" in BODY_DOCUMENT_TYPES


def test_extract_body_documents_picks_up_10k_and_ex13() -> None:
    """A submission with both 10-K and EX-13 blocks should yield both, concatenated."""
    raw = """<SEC-DOCUMENT>
<DOCUMENT>
<TYPE>10-K
<FILENAME>main.htm
<TEXT>
SUMMARY DOCUMENT BODY
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-13
<FILENAME>annual-report.htm
<TEXT>
ANNUAL REPORT BODY WITH ACTUAL CONTENT
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-31.1
<FILENAME>certification.htm
<TEXT>
CERTIFICATION (should be skipped)
</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>"""
    body, count = _extract_body_documents(raw)
    assert count == 2
    assert "SUMMARY DOCUMENT BODY" in body
    assert "ANNUAL REPORT BODY WITH ACTUAL CONTENT" in body
    assert "CERTIFICATION" not in body


def test_extract_body_documents_handles_subtype_variants() -> None:
    """EX-13.1, EX-99.1, etc. should all match the body prefix rule."""
    raw = """<DOCUMENT>
<TYPE>EX-13.1
<TEXT>BODY A</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-99.2
<TEXT>BODY B</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-21
<TEXT>SUBSIDIARY LIST</TEXT>
</DOCUMENT>"""
    body, count = _extract_body_documents(raw)
    assert count == 2
    assert "BODY A" in body
    assert "BODY B" in body
    assert "SUBSIDIARY LIST" not in body


def test_extract_body_documents_returns_empty_when_no_body_blocks() -> None:
    """If the submission has no body-typed documents, we return ('', 0)."""
    raw = """<DOCUMENT>
<TYPE>EX-31.1
<TEXT>just a cert</TEXT>
</DOCUMENT>"""
    body, count = _extract_body_documents(raw)
    assert body == ""
    assert count == 0


# -- Integration tests against real filings ----------------------------------


@pytest.fixture(scope="module")
def aapl_2022_available() -> None:
    if not AAPL_2022.exists():
        pytest.skip(f"{AAPL_2022} not found; run `sec-rag ingest download` first")


@pytest.fixture(scope="module")
def amzn_2022_available() -> None:
    if not AMZN_2022.exists():
        pytest.skip(f"{AMZN_2022} not found; run `sec-rag ingest download` first")


def test_aapl_parses_both_sections(aapl_2022_available: None) -> None:
    """Apple uses bold inline headings in a single 10-K document."""
    result, sections = parse_filing(AAPL_2022)
    assert result.status == "ok", f"Failed: {result.error}"
    assert "1A" in result.sections_found
    assert "7" in result.sections_found
    assert result.body_doc_count >= 1

    by_section = {s.section: s for s in sections}

    item_1a = by_section["1A"]
    assert item_1a.char_count >= MIN_SECTION_CHARS
    assert "risk" in item_1a.text.lower()

    item_7 = by_section["7"]
    assert item_7.char_count >= MIN_SECTION_CHARS
    assert any(k in item_7.text.lower() for k in ["operations", "revenue", "fiscal"])


def test_amzn_parses_both_sections(amzn_2022_available: None) -> None:
    """Amazon uses table-layout headings (font-weight:400) — was failing in v1."""
    result, sections = parse_filing(AMZN_2022)
    assert result.status == "ok", f"Failed: {result.error}"
    assert "1A" in result.sections_found
    assert "7" in result.sections_found

    by_section = {s.section: s for s in sections}
    assert by_section["1A"].char_count >= MIN_SECTION_CHARS
    assert "risk" in by_section["1A"].text.lower()
    assert by_section["7"].char_count >= MIN_SECTION_CHARS


def test_amzn_item_1a_is_real_heading_not_toc(amzn_2022_available: None) -> None:
    """Sanity-check the section starts at real prose, not a TOC entry."""
    _, sections = parse_filing(AMZN_2022)
    by_section = {s.section: s for s in sections}
    item_1a = by_section.get("1A")
    assert item_1a is not None, "Item 1A not extracted"

    head = item_1a.text[:300].lower()
    assert "page" not in head[:100], f"Item 1A starts with TOC: {head[:100]!r}"
    assert any(k in head for k in ["risk", "factor", "consider"])


def test_corpus_success_rate() -> None:
    """End-to-end: at least 90% of corpus filings should parse successfully.

    'Successfully' means status == 'ok' (both Item 1A and Item 7 extracted
    above the minimum-length threshold). Status 'partial' is not counted —
    chunking and eval downstream depend on consistent section coverage.

    The 5 known incorporation-by-reference filers (GE, HON, MCD, MS, WFC)
    have been removed from the corpus per KNOWN_ISSUES.md. Remaining
    failures should be rare format outliers.
    """
    if not PARSE_REPORT.exists():
        pytest.skip(f"{PARSE_REPORT} not found; run `sec-rag ingest parse` first")

    report = json.loads(PARSE_REPORT.read_text())
    total = report["total_attempted"]
    ok_count = report["by_status"].get("ok", 0)
    success_rate = ok_count / total if total else 0.0

    assert total > 0, "Empty parse report"
    assert success_rate >= 0.90, (
        f"Corpus success rate {success_rate:.1%} below 90% threshold "
        f"({ok_count}/{total}). Investigate failing tickers and either "
        f"replace them or document in KNOWN_ISSUES.md."
    )
