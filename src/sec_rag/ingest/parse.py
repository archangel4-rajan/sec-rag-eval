"""Parse 10-K filings to extract Item 1A (Risk Factors) and Item 7 (MD&A).

Design principles
=================

1. **Plain-text first.** HTML styling conventions vary by filer (Apple uses inline
   font-weight:700, Amazon uses font-weight:400 with table layout, older filings
   use <b> tags). We strip HTML once and operate on plain text where section
   markers are universal: every 10-K has the literal string "Item 1A" near the
   start of the Risk Factors section.

2. **Multi-document submissions.** Some filers (banks, conglomerates) split their
   10-K across multiple <DOCUMENT> blocks. Wells Fargo files a 2 MB summary as
   the primary 10-K block plus a 15 MB EX-13 (Annual Report to Shareholders)
   that contains the actual body. We concatenate every relevant document block
   before extracting sections.

3. **Candidate scoring, not regex matching.** Multiple occurrences of "Item 1A"
   exist in any filing — table of contents, cross-references, and the actual
   heading. We score each candidate by structural signals (gap length to next
   section marker, cross-reference context, marker density) and pick the best.

4. **Pair-based extraction.** A real section is a (start, end) pair of headings
   bounding substantial prose. We try the natural end-marker first (Item 1B for
   Item 1A, Item 7A for Item 7) and fall back to the next major section if the
   primary end-marker doesn't yield a plausible pair.

This approach is format-agnostic — it works on Apple's bold-styled headings,
Amazon's table-layout headings, and Wells Fargo's split-document filings without
code changes per filer.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from tqdm import tqdm

from sec_rag.config import settings
from sec_rag.logging_setup import get_logger

logger = get_logger(__name__)

# Sections we extract: (start_id, primary_end_id, fallback_end_id, friendly_name)
# Item 1A normally ends at Item 1B; falls back to Item 2 (Properties) when 1B is
# absent (smaller filers sometimes omit "Unresolved Staff Comments").
# Item 7 normally ends at Item 7A; falls back to Item 8 when 7A is absent.
TARGET_SECTIONS = [
    ("1A", "1B", "2", "Risk Factors"),
    ("7", "7A", "8", "Management's Discussion and Analysis"),
]

# Document TYPEs whose content is part of the 10-K body.
# - 10-K: the primary submission document
# - EX-13: Annual Report to Shareholders (used by banks like WFC for the body)
# - EX-99: catch-all "additional exhibits" used by some filers for body content
BODY_DOCUMENT_TYPES = ("10-K", "EX-13", "EX-99")

# Match "Item N" or "Item NA" in plain text.
ITEM_PATTERN = re.compile(
    r"\bItem\s+([0-9]+[A-Za-z]?)\b\.?",
    re.IGNORECASE,
)

# Phrases indicating a cross-reference rather than a section heading. Checked
# in a window AROUND each candidate (before + token + after) so phrases like
# "See Item 1A of Part I" are detected as a unit.
CROSS_REF_PHRASES = [
    "see item ",
    "in item ",
    "of item ",
    "under item ",
    "discussed in item",
    "described in item",
    "outlined in item",
    "set forth in item",
    "referred to in item",
    "elsewhere in item",
    "elsewhere in this item",
    "this item ",
    "factors discussed",
    "of part i,",
    "of part ii,",
    "of part iii",
    "of part iv",
    "of this form",
    "this annual report",
]

MIN_SECTION_CHARS = 5_000
MAX_SECTION_CHARS = 250_000


@dataclass
class ParsedSection:
    ticker: str
    year: int
    section: str
    section_title: str
    text: str
    char_count: int
    source_file: str
    extraction_score: float


@dataclass
class ParseResult:
    ticker: str
    year: int
    status: str  # "ok" | "no_body" | "no_sections" | "partial" | "error"
    sections_found: list[str]
    diagnostics: dict[str, float] | None = None
    body_doc_count: int = 0
    body_total_chars: int = 0
    error: str | None = None


def parse_filing(filing_path: Path) -> tuple[ParseResult, list[ParsedSection]]:
    """Parse one filing, return a result record and any sections extracted."""
    name = filing_path.stem
    try:
        ticker, year_str = name.split("_")
        year = int(year_str)
    except ValueError:
        return (
            ParseResult(
                ticker=name,
                year=0,
                status="error",
                sections_found=[],
                error=f"Could not parse ticker/year: {name}",
            ),
            [],
        )

    raw = filing_path.read_text(encoding="utf-8", errors="replace")

    # Step 1: Concatenate every <DOCUMENT> block whose TYPE indicates body content.
    body_html, body_doc_count = _extract_body_documents(raw)
    if not body_html:
        return (
            ParseResult(
                ticker=ticker,
                year=year,
                status="no_body",
                sections_found=[],
                error=f"No {'/'.join(BODY_DOCUMENT_TYPES)} document blocks found",
            ),
            [],
        )

    # Step 2: Strip HTML to plain text.
    plain = _html_to_text(body_html)
    if len(plain) < 10_000:
        return (
            ParseResult(
                ticker=ticker,
                year=year,
                status="error",
                sections_found=[],
                body_doc_count=body_doc_count,
                body_total_chars=len(plain),
                error=f"Plain text too short: {len(plain)} chars",
            ),
            [],
        )

    # Step 3: Locate every Item-N candidate.
    all_candidates = _find_all_item_candidates(plain)

    # Step 4: For each target section, find the best (start, end) pair, with
    # fallback to the secondary end marker if the primary doesn't yield one.
    sections: list[ParsedSection] = []
    diagnostics: dict[str, float] = {}
    for start_id, primary_end, fallback_end, friendly_name in TARGET_SECTIONS:
        result = _extract_section_with_fallback(
            plain,
            all_candidates,
            start_id,
            primary_end,
            fallback_end,
        )
        if result is None:
            diagnostics[start_id] = -1.0
            continue
        start_pos, end_pos, score = result
        text = plain[start_pos:end_pos].strip()
        if not (MIN_SECTION_CHARS <= len(text) <= MAX_SECTION_CHARS):
            diagnostics[start_id] = score
            continue
        title = _extract_title(text, start_id, friendly_name)
        sections.append(
            ParsedSection(
                ticker=ticker,
                year=year,
                section=start_id,
                section_title=title,
                text=text,
                char_count=len(text),
                source_file=filing_path.name,
                extraction_score=score,
            )
        )
        diagnostics[start_id] = score

    if not sections:
        return (
            ParseResult(
                ticker=ticker,
                year=year,
                status="no_sections",
                sections_found=[],
                diagnostics=diagnostics,
                body_doc_count=body_doc_count,
                body_total_chars=len(plain),
                error="No target sections passed quality threshold",
            ),
            [],
        )

    status = "ok" if len(sections) == len(TARGET_SECTIONS) else "partial"
    return (
        ParseResult(
            ticker=ticker,
            year=year,
            status=status,
            sections_found=[s.section for s in sections],
            diagnostics=diagnostics,
            body_doc_count=body_doc_count,
            body_total_chars=len(plain),
        ),
        sections,
    )


def _extract_body_documents(raw: str) -> tuple[str, int]:
    """Extract and concatenate every <DOCUMENT> block whose TYPE is body content.

    Returns (concatenated_html, document_count). Returns ('', 0) if none found.
    """
    body_blocks: list[str] = []
    for m in re.finditer(r"<DOCUMENT>(.*?)</DOCUMENT>", raw, re.DOTALL | re.IGNORECASE):
        block_body = m.group(1)
        type_match = re.search(r"<TYPE>([^\n<]+)", block_body, re.IGNORECASE)
        if not type_match:
            continue
        doc_type = type_match.group(1).strip().upper()
        # Match exact "10-K" but allow "EX-13" / "EX-13.1" / "EX-99" / "EX-99.1" etc.
        is_body = doc_type == "10-K" or doc_type.startswith("EX-13") or doc_type.startswith("EX-99")
        if is_body:
            body_blocks.append(block_body)

    return ("\n\n".join(body_blocks), len(body_blocks))


def _html_to_text(html_str: str) -> str:
    """Strip HTML, decode entities, normalize whitespace."""
    soup = BeautifulSoup(html_str, "lxml")
    text = soup.get_text(separator=" ")
    text = html.unescape(text)
    text = re.sub(r"[\xa0\s]+", " ", text)
    return text.strip()


def _find_all_item_candidates(plain: str) -> list[tuple[int, str]]:
    """Find every "Item N" / "Item NA" occurrence."""
    return [(m.start(), m.group(1).upper()) for m in ITEM_PATTERN.finditer(plain)]


def _extract_section_with_fallback(
    plain: str,
    all_candidates: list[tuple[int, str]],
    start_id: str,
    primary_end_id: str,
    fallback_end_id: str,
) -> tuple[int, int, float] | None:
    """Try primary end marker; if no plausible pair, fall back to secondary."""
    primary = _extract_section(plain, all_candidates, start_id, primary_end_id)
    if primary is not None:
        # Accept the primary if it produces a section above minimum length.
        start, end, score = primary
        if (end - start) >= MIN_SECTION_CHARS and score > -100:
            return primary
    # Fall back to secondary end marker.
    return _extract_section(plain, all_candidates, start_id, fallback_end_id)


def _extract_section(
    plain: str,
    all_candidates: list[tuple[int, str]],
    start_id: str,
    end_id: str,
) -> tuple[int, int, float] | None:
    """Find the best (start, end) pair for a section."""
    start_id = start_id.upper()
    end_id = end_id.upper()

    start_candidates = [pos for pos, item in all_candidates if item == start_id]
    end_candidates = [pos for pos, item in all_candidates if item == end_id]

    if not start_candidates or not end_candidates:
        return None

    other_markers = {pos for pos, _ in all_candidates}

    best_score = -1e9
    best_pair: tuple[int, int] | None = None

    for start in start_candidates:
        # Window around start: 120 chars before + 100 chars after for cross-ref detection.
        window = plain[max(0, start - 120) : start + 100]
        for end in end_candidates:
            if end <= start:
                continue
            score = _score_pair(start, end, window, other_markers)
            if score > best_score:
                best_score = score
                best_pair = (start, end)

    if best_pair is None:
        return None
    return (best_pair[0], best_pair[1], best_score)


def _score_pair(
    start: int,
    end: int,
    window_around_start: str,
    all_marker_positions: set[int],
) -> float:
    """Score a (start, end) pair for being a real section span."""
    gap = end - start
    score = 0.0

    # 1. Reward gap length — real sections are tens of thousands of chars.
    score += (gap**0.5) / 5

    # 2. Heavy penalty for cross-reference context.
    window_lower = window_around_start.lower()
    if any(phrase in window_lower for phrase in CROSS_REF_PHRASES):
        score -= 200

    # 3. Density penalty — TOC entries cluster many markers very close together.
    very_close_before = sum(1 for p in all_marker_positions if start - 300 < p < start)
    score -= very_close_before * 30

    # 4. Wider density: softer signal.
    nearby_before = sum(1 for p in all_marker_positions if start - 1500 < p < start - 300)
    score -= nearby_before * 5

    # 5. Penalty for excessive markers between start and end.
    inside = sum(1 for p in all_marker_positions if start < p < end)
    if inside > 20:
        score -= (inside - 20) * 3

    # 6. Hard sanity: ridiculously long sections.
    if gap > MAX_SECTION_CHARS:
        score -= 100

    return score


def _extract_title(section_text: str, section_id: str, fallback: str) -> str:
    """Pull the section title from the first ~400 chars."""
    head = section_text[:400]
    m = re.match(
        rf"\s*Item\s*{re.escape(section_id)}\.?\s+([A-Z][A-Za-z'\u2019\s\-,&]{{2,95}}?)\s+(?:[A-Z][a-z]+\s+){{2,}}",
        head,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    m = re.match(
        rf"\s*Item\s*{re.escape(section_id)}\.?\s+([^\.]{{2,90}})",
        head,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return fallback


def parse_all(
    filings_dir: Path | None = None,
    output_dir: Path | None = None,
) -> list[ParseResult]:
    """Parse every filing, write one JSON per section to output_dir."""
    filings_dir = filings_dir or settings.filings_dir
    output_dir = output_dir or settings.chunks_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(filings_dir.glob("*.html"))
    if not files:
        logger.warning("no_filings_found", dir=str(filings_dir))
        return []

    results: list[ParseResult] = []
    pbar = tqdm(files, desc="Parsing filings", unit="filing")
    for f in pbar:
        pbar.set_postfix_str(f.stem)
        result, sections = parse_filing(f)
        results.append(result)
        for s in sections:
            out_path = output_dir / f"{s.ticker}_{s.year}_item{s.section}.json"
            out_path.write_text(json.dumps(asdict(s), indent=2))

    _write_run_report(results, output_dir)
    _log_summary(results)
    return results


def _write_run_report(results: list[ParseResult], output_dir: Path) -> None:
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    report = {
        "total_attempted": len(results),
        "by_status": by_status,
        "filings": [asdict(r) for r in results],
    }
    (output_dir / "parse_report.json").write_text(json.dumps(report, indent=2))


def _log_summary(results: list[ParseResult]) -> None:
    by_status: dict[str, int] = {}
    section_counts: dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
        for s in r.sections_found:
            section_counts[s] = section_counts.get(s, 0) + 1
    logger.info("parse_complete", status=by_status, sections=section_counts)
    failures = [r for r in results if r.status not in ("ok", "partial")]
    if failures:
        logger.warning(
            "parse_failures",
            count=len(failures),
            sample=[(r.ticker, r.year, r.status) for r in failures[:10]],
        )


if __name__ == "__main__":
    parse_all()
