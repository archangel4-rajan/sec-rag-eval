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

Known limitation: filings that don't use the literal string "Item 1A" as a
section anchor (e.g., Citigroup uses standalone "1A." in cross-reference
indexes plus all-caps topic headings like "RISK FACTORS") fall outside this
parser's scope. See KNOWN_ISSUES.md for details.
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
# friendly_name is documentary — kept for clarity at the call site, not consumed.
TARGET_SECTIONS = [
    ("1A", "1B", "2", "Risk Factors"),
    ("7", "7A", "8", "Management's Discussion and Analysis"),
]

# Document TYPEs whose content is part of the 10-K body.
BODY_DOCUMENT_TYPES = ("10-K", "EX-13", "EX-99")

# Match "Item N" or "Item NA" in plain text.
ITEM_PATTERN = re.compile(
    r"\bItem\s+([0-9]+[A-Za-z]?)\b\.?",
    re.IGNORECASE,
)

# Phrases indicating a cross-reference rather than a section heading.
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

# Plausible section length, in characters of plain text.
# Item 1A: typically 30K-150K chars. Item 7: 20K-100K for tech/consumer
# companies but 200K-350K for banks. Cap at 400K.
MIN_SECTION_CHARS = 5_000
MAX_SECTION_CHARS = 400_000

# SEC-mandated standard titles for each Item number. These are the canonical
# titles every 10-K uses (with minor stylistic variations). We use these both
# as fallbacks when title extraction fails AND as priority anchors when a
# filing's text contains a recognizable variant.
STANDARD_TITLES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities",
    "6": "Selected Financial Data",
    "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
}


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

    all_candidates = _find_all_item_candidates(plain)

    sections: list[ParsedSection] = []
    diagnostics: dict[str, float] = {}
    for start_id, primary_end, fallback_end, _friendly_name in TARGET_SECTIONS:
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
        title = _extract_title(text, start_id)
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
    """Concatenate every <DOCUMENT> block whose TYPE is body content."""
    body_blocks: list[str] = []
    for m in re.finditer(r"<DOCUMENT>(.*?)</DOCUMENT>", raw, re.DOTALL | re.IGNORECASE):
        block_body = m.group(1)
        type_match = re.search(r"<TYPE>([^\n<]+)", block_body, re.IGNORECASE)
        if not type_match:
            continue
        doc_type = type_match.group(1).strip().upper()
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
        start, end, score = primary
        if (end - start) >= MIN_SECTION_CHARS and score > -100:
            return primary
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

    score += (gap**0.5) / 5

    window_lower = window_around_start.lower()
    if any(phrase in window_lower for phrase in CROSS_REF_PHRASES):
        score -= 200

    very_close_before = sum(1 for p in all_marker_positions if start - 300 < p < start)
    score -= very_close_before * 30

    nearby_before = sum(1 for p in all_marker_positions if start - 1500 < p < start - 300)
    score -= nearby_before * 5

    inside = sum(1 for p in all_marker_positions if start < p < end)
    if inside > 20:
        score -= (inside - 20) * 3

    if gap > MAX_SECTION_CHARS:
        score -= 100

    return score


# ---- Title extraction ---------------------------------------------------


def _extract_title(section_text: str, section_id: str) -> str:
    """Extract section title using a layered strategy.

    Priority order:

    Strategy 1: Standard SEC title match. If the section text contains a
    recognized SEC-mandated title (case-insensitive), return the canonical
    form. This handles uppercase variants and company-name-prefixed cases.
    Examples:
       Apple "Item 1A. Risk Factors The Company's..." → "Risk Factors"
       Tesla "ITEM 7. MANAGEMENT'S DISCUSSION..."     → "Management's Discussion..."
       BAC   "Item 7. Bank of America Corporation and Subsidiaries
              Management's Discussion..."             → "Management's Discussion..."

    Strategy 2: Inline extraction. For non-standard but well-formed titles,
    capture text between "Item N." and a clear body-prose marker (a possessive
    's or two consecutive sentence-opener words).

    Fallback: SEC-canonical title from STANDARD_TITLES, or "Item N" if the
    section_id isn't in our table.
    """
    head = section_text[:600]
    section_id = section_id.upper()
    standard = STANDARD_TITLES.get(section_id)

    # Strategy 1: standard title match (highest priority).
    if standard:
        # Try the full canonical title first.
        if re.search(re.escape(standard), head, re.IGNORECASE):
            return standard
        # Try a 40-char prefix of the standard title — handles slight
        # variations in long titles where the filing omits trailing words.
        prefix = standard[:40]
        if len(prefix) >= 15 and re.search(re.escape(prefix), head, re.IGNORECASE):
            return standard

    # Strategy 2: inline extraction with body-prose terminator.
    candidate = _extract_inline_title(head, section_id)
    if candidate is not None:
        return candidate

    # Fallback: canonical SEC title or generic placeholder.
    return standard or f"Item {section_id}"


def _extract_inline_title(head: str, section_id: str) -> str | None:
    """Extract a non-standard title from inline text.

    Matches "Item N. <TITLE>" where <TITLE> is bounded by:
      - a possessive marker ('s) — body prose like "The Company's..."
      - two-word capitalized opener — body prose like "The following discussion..."
    """
    title_chars = r"A-Za-z\u2018\u2019\u201c\u201d\s\-,&"
    m = re.match(
        rf"\s*Item\s*{re.escape(section_id)}\.?\s+"
        rf"([A-Za-z][{title_chars}]{{4,90}}?)"
        rf"(?:\s+(?:[A-Z][a-z]+\s+){{2,}}|['\u2019]s\s+)",
        head,
        re.IGNORECASE,
    )
    if not m:
        return None
    candidate = m.group(1).strip()
    if not _looks_like_title(candidate):
        return None
    return candidate


def _looks_like_title(text: str) -> bool:
    """Sanity check: reject candidates that are clearly cross-references."""
    text_lower = text.lower()
    crossref_markers = (
        " of the company",
        " of part i",
        " of part ii",
        " annual report",
        " of this form",
        " incorporated by reference",
    )
    if any(m in text_lower for m in crossref_markers):
        return False
    words = text.split()
    if not words:
        return False
    first_word = words[0].lower()
    if first_word in {"of", "in", "under", "for", "from", "to", "at", "with"}:
        return False
    return len(words) >= 2


# ---- Pipeline orchestration ---------------------------------------------


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
