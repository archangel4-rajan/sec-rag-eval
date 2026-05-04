"""The 30 tickers we ingest. Mix of sectors to test retrieval across domains.

Why these:
- Tech (10): models likely have strong parametric knowledge — useful for catching
  cases where the LLM ignores retrieved context and answers from priors.
- Industrials/Consumer (10): less parametric coverage, more pure-retrieval.
- Financials/Health (10): dense risk-factor sections, complex regulatory language.
"""

TECH = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "NVDA",  # Nvidia
    "GOOGL",  # Alphabet
    "META",  # Meta
    "AMZN",  # Amazon
    "TSLA",  # Tesla
    "ORCL",  # Oracle
    "CRM",  # Salesforce
    "ADBE",  # Adobe
]

INDUSTRIALS_CONSUMER = [
    "CAT",  # Caterpillar
    "GE",  # GE Aerospace
    "HON",  # Honeywell
    "PG",  # Procter & Gamble
    "KO",  # Coca-Cola
    "PEP",  # PepsiCo
    "NKE",  # Nike
    "MCD",  # McDonald's
    "WMT",  # Walmart
    "COST",  # Costco
]

FINANCIALS_HEALTH = [
    "JPM",  # JPMorgan
    "BAC",  # Bank of America
    "GS",  # Goldman Sachs
    "MS",  # Morgan Stanley
    "WFC",  # Wells Fargo
    "UNH",  # UnitedHealth
    "JNJ",  # Johnson & Johnson
    "PFE",  # Pfizer
    "CVS",  # CVS Health
    "ABBV",  # AbbVie
]

ALL_TICKERS: list[str] = TECH + INDUSTRIALS_CONSUMER + FINANCIALS_HEALTH

# Years to fetch. SEC filings for the most recent fiscal year typically appear
# 2–3 months after fiscal year-end, so we go back 3 full years.
YEARS = [2022, 2023, 2024, 2025]

assert len(ALL_TICKERS) == 30, f"Expected 30 tickers, got {len(ALL_TICKERS)}"
