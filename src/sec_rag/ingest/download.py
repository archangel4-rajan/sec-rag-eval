"""Download 10-K filings from SEC EDGAR.

Uses the `sec-edgar-downloader` library which respects SEC's 10 req/sec rate limit
and handles retries on 503s. Writes a JSON run report tracking every filing pulled
or skipped, which feeds into the per-day commit message.

Usage:
    python -m sec_rag.ingest.download
    # or via CLI:
    sec-rag ingest download
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from sec_edgar_downloader import Downloader
from tqdm import tqdm

from sec_rag.config import settings
from sec_rag.ingest.tickers import ALL_TICKERS, YEARS
from sec_rag.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class FilingRecord:
    """One row in the ingestion run report."""

    ticker: str
    year: int
    status: str  # "downloaded" | "skipped_existing" | "failed"
    path: str | None = None
    error: str | None = None
    size_bytes: int | None = None


def _parse_user_agent(user_agent: str) -> tuple[str, str]:
    """sec-edgar-downloader requires (company_name, email_address) as a tuple.

    We accept either a single 'Name email@x.com' string or a tuple-style env value.
    """
    parts = user_agent.rsplit(" ", 1)
    if len(parts) == 2 and "@" in parts[1]:
        return parts[0].strip(), parts[1].strip()
    # Fallback — library will warn but accept anything.
    return user_agent, "noreply@example.com"


def download_all(
    tickers: list[str] | None = None,
    years: list[int] | None = None,
    output_dir: Path | None = None,
) -> list[FilingRecord]:
    """Download 10-Ks for every (ticker, year) combination.

    Args:
        tickers: List of tickers; defaults to ALL_TICKERS from tickers.py
        years: List of fiscal years; defaults to YEARS from tickers.py
        output_dir: Where to write filings; defaults to settings.filings_dir

    Returns:
        List of FilingRecord, one per (ticker, year) attempted.
    """
    tickers = tickers or ALL_TICKERS
    years = years or YEARS
    output_dir = output_dir or settings.filings_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    company, email = _parse_user_agent(settings.sec_user_agent)
    # The library writes to <download_folder>/sec-edgar-filings/<ticker>/10-K/...
    # We let it manage that, then reorganize into our flat layout.
    dl = Downloader(company, email, str(output_dir))

    records: list[FilingRecord] = []
    pbar = tqdm(
        [(t, y) for t in tickers for y in years],
        desc="Downloading 10-Ks",
        unit="filing",
    )

    for ticker, year in pbar:
        pbar.set_postfix_str(f"{ticker} {year}")
        target_path = output_dir / f"{ticker}_{year}.html"

        if target_path.exists():
            records.append(
                FilingRecord(
                    ticker=ticker,
                    year=year,
                    status="skipped_existing",
                    path=str(target_path),
                    size_bytes=target_path.stat().st_size,
                )
            )
            continue

        try:
            # Filings for fiscal year N are typically filed in early N+1.
            # We use a 14-month window starting at the start of fiscal year N
            # to catch the matching 10-K regardless of fiscal-year-end variation.
            after_date = f"{year}-01-01"
            before_date = f"{year + 1}-06-30"

            n = dl.get(
                "10-K",
                ticker,
                after=after_date,
                before=before_date,
                limit=1,
                download_details=False,
            )

            if n == 0:
                records.append(
                    FilingRecord(
                        ticker=ticker,
                        year=year,
                        status="failed",
                        error="No 10-K returned by SEC for this window",
                    )
                )
                continue

            # Reorganize: find the downloaded file and copy to our flat layout.
            source = _locate_primary_filing(output_dir, ticker)
            if source is None:
                records.append(
                    FilingRecord(
                        ticker=ticker,
                        year=year,
                        status="failed",
                        error="Could not locate downloaded file",
                    )
                )
                continue

            shutil.copy(source, target_path)
            records.append(
                FilingRecord(
                    ticker=ticker,
                    year=year,
                    status="downloaded",
                    path=str(target_path),
                    size_bytes=target_path.stat().st_size,
                )
            )

        except Exception as e:  # noqa: BLE001 — we want to capture every failure
            logger.exception("download_failed", ticker=ticker, year=year)
            records.append(
                FilingRecord(
                    ticker=ticker,
                    year=year,
                    status="failed",
                    error=str(e),
                )
            )

    _write_run_report(records, output_dir)
    _log_summary(records)
    return records


def _locate_primary_filing(base_dir: Path, ticker: str) -> Path | None:
    """Find the most recent 10-K file the library just dropped for this ticker.

    The library writes to `<base_dir>/sec-edgar-filings/<ticker>/10-K/<accession>/full-submission.txt`.
    We grab the newest accession folder.
    """
    ticker_dir = base_dir / "sec-edgar-filings" / ticker / "10-K"
    if not ticker_dir.exists():
        return None
    accession_dirs = sorted(
        [p for p in ticker_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not accession_dirs:
        return None
    primary = accession_dirs[0] / "full-submission.txt"
    return primary if primary.exists() else None


def _write_run_report(records: list[FilingRecord], output_dir: Path) -> None:
    """Write a JSON run report with every filing's status. Used in commit messages."""
    report = {
        "run_at": datetime.now(UTC).isoformat(),
        "total_attempted": len(records),
        "downloaded": sum(1 for r in records if r.status == "downloaded"),
        "skipped_existing": sum(1 for r in records if r.status == "skipped_existing"),
        "failed": sum(1 for r in records if r.status == "failed"),
        "filings": [asdict(r) for r in records],
    }
    report_path = output_dir / "ingestion_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("run_report_written", path=str(report_path))


def _log_summary(records: list[FilingRecord]) -> None:
    """One-line summary at the end."""
    by_status: dict[str, int] = {}
    for r in records:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    logger.info("ingestion_complete", **by_status)
    failures = [r for r in records if r.status == "failed"]
    if failures:
        logger.warning(
            "failures_detail",
            count=len(failures),
            tickers=[f"{r.ticker}_{r.year}" for r in failures[:10]],
        )


if __name__ == "__main__":
    download_all()
