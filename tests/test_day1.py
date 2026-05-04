"""Day 1 tests: configuration + ticker list sanity. No network calls."""

from sec_rag.config import settings
from sec_rag.ingest.download import FilingRecord, _parse_user_agent
from sec_rag.ingest.tickers import ALL_TICKERS, FINANCIALS_HEALTH, INDUSTRIALS_CONSUMER, TECH, YEARS


def test_ticker_list_size() -> None:
    """The 30-ticker invariant is load-bearing for downstream eval design."""
    assert len(ALL_TICKERS) == 30
    assert len(set(ALL_TICKERS)) == 30, "Duplicates in ticker list"


def test_ticker_sectors_balanced() -> None:
    """Three sectors of 10 each."""
    assert len(TECH) == 10
    assert len(INDUSTRIALS_CONSUMER) == 10
    assert len(FINANCIALS_HEALTH) == 10


def test_years_in_range() -> None:
    """Years should be reasonable; SEC filings before ~2000 have inconsistent format."""
    assert all(y >= 2000 for y in YEARS)
    assert len(YEARS) == 4


def test_settings_paths_resolvable() -> None:
    """Path properties should compose without error."""
    assert settings.filings_dir.name == "filings"
    assert settings.chunks_dir.name == "chunks"
    assert settings.eval_dir.name == "eval"


def test_parse_user_agent_with_email() -> None:
    name, email = _parse_user_agent("Rajan Anand rajan@example.com")
    assert name == "Rajan Anand"
    assert email == "rajan@example.com"


def test_parse_user_agent_fallback() -> None:
    """Strings without an email get a fallback email."""
    name, email = _parse_user_agent("just a name")
    assert "@" in email


def test_filing_record_serializable() -> None:
    """The run-report JSON depends on dataclass-asdict; verify the shape."""
    from dataclasses import asdict

    rec = FilingRecord(ticker="AAPL", year=2023, status="downloaded", path="/x", size_bytes=1024)
    d = asdict(rec)
    assert d["ticker"] == "AAPL"
    assert d["status"] == "downloaded"
    assert d["size_bytes"] == 1024
