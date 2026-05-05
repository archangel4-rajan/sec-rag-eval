"""The 30 tickers we ingest. Mix of sectors to test retrieval across domains.

Why these:
- Tech (10): models likely have strong parametric knowledge — useful for catching
  cases where the LLM ignores retrieved context and answers from priors.
- Industrials/Consumer (10): less parametric coverage, more pure-retrieval.
- Financials/Health (10): dense risk-factor sections, complex regulatory language.

Originally the corpus included GE, HON, MCD, MS, and WFC. These were dropped
because they file 10-Ks under SEC Rule 12b-23 (incorporation by reference): the
primary 10-K document contains stub Item sections that point to a separately
filed EX-13 (Annual Report to Shareholders), where the actual prose lives under
topic headings ("Credit Risk", "Operational Risk") rather than Item-N anchors.
Reliable extraction requires NLP-grade section detection that's out of scope
for this project. See KNOWN_ISSUES.md for details.

Replacements: DIS, TGT, LOW, C, AXP — all verified to use single-document
filings with standard Item-N section headers.
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
    "DIS",  # Disney        (replaced GE)
    "LOW",  # Lowe's        (replaced HON)
    "PG",  # Procter & Gamble
    "KO",  # Coca-Cola
    "PEP",  # PepsiCo
    "NKE",  # Nike
    "TGT",  # Target        (replaced MCD)
    "WMT",  # Walmart
    "COST",  # Costco
]

FINANCIALS_HEALTH = [
    "JPM",  # JPMorgan
    "BAC",  # Bank of America
    "GS",  # Goldman Sachs
    "C",  # Citigroup     (replaced MS)
    "AXP",  # American Express  (replaced WFC)
    "UNH",  # UnitedHealth
    "JNJ",  # Johnson & Johnson
    "PFE",  # Pfizer
    "CVS",  # CVS Health
    "ABBV",  # AbbVie
]

ALL_TICKERS: list[str] = TECH + INDUSTRIALS_CONSUMER + FINANCIALS_HEALTH

# Years to fetch. SEC filings for the most recent fiscal year typically appear
# 2-3 months after fiscal year-end, so we go back 4 full years.
YEARS = [2022, 2023, 2024, 2025]

assert len(ALL_TICKERS) == 30, f"Expected 30 tickers, got {len(ALL_TICKERS)}"
assert len(set(ALL_TICKERS)) == 30, "Duplicate ticker in ALL_TICKERS"
