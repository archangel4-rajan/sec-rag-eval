# Known Issues

A running list of edge cases observed during development. Documented but not
fixed, so day-by-day momentum doesn't stall on tangents. Items here are
candidates for "what I'd do differently in production" in the final README.

## Filing pattern: incorporation by reference (corpus-shaping decision)

Some SEC filers — primarily large banks and conglomerates — file 10-Ks under
SEC Rule 12b-23, which permits incorporation by reference. Under this pattern:

- The primary `<DOCUMENT TYPE="10-K">` block contains *stub sections*: each
  Item heading (1A Risk Factors, 7 MD&A, 7A Quantitative Disclosures, 8
  Financial Statements) is followed by 100-200 characters of text along the
  lines of *"Information in response to this Item 1A can be found in the 2022
  Annual Report to Shareholders under 'Financial Review – Risk Factors.' That
  information is incorporated into this item by reference."*
- The actual prose lives in a separately filed `<DOCUMENT TYPE="EX-13">`
  attachment ("Annual Report to Shareholders").
- Within the EX-13, content is organized under topic headings ("Credit Risk",
  "Operational Risk Management", "Market Risk", "Financial Review") rather
  than Item-N anchors. Risk-related prose is also distributed throughout the
  document interleaved with financial statements, capital tables, and
  governance discussion — there is no contiguous "Item 1A" block to extract.

### Examples observed

GE, Honeywell, McDonald's, Morgan Stanley, Wells Fargo. Diagnostic for WFC
2022 found 34 occurrences of the literal string "Risk Factors", but every
one was either (a) a forward-reference inside the summary stub or (b) inline
prose referring back to risk factors discussed elsewhere. None marked the
start of a contiguous section.

### Why we excluded these tickers

Reliable extraction would require either:

1. **NLP-based section classification** — train or prompt a model to identify
   risk-related paragraphs and bound them as a virtual "Item 1A". This is a
   real engineering project, not a parser tweak.
2. **Per-ticker manual mapping** — hand-encode where each filer's risk content
   lives. Doesn't scale and is brittle to filing-format changes.
3. **Fixed-window extraction** — take a heuristic byte range as "Item 1A".
   Cheap but pollutes the eval corpus with non-risk content (~30% of the
   captured window in the WFC case).

For this project, where the eval system is the centerpiece deliverable,
corpus quality matters more than corpus breadth. We swapped the 5 problem
tickers for clean alternatives (DIS, TGT, LOW, C, AXP) that use single-
document filings with standard Item-N anchors, preserving the 30-ticker
size while ensuring every filing in the corpus is reliably parsed.

### Detecting this pattern programmatically

If you wanted to detect IBR filings in a future ingestion run:
- An Item 1A → Item 1B span under 2,000 characters is a strong signal
- Multiple `<DOCUMENT TYPE="EX-13">` or `<DOCUMENT TYPE="EX-99">` blocks
  with size > 1 MB suggests body content lives outside the primary 10-K
- The phrase *"incorporated into this item by reference"* appearing 3+
  times in the primary document is nearly definitive

## Smaller issues

### Older filings (pre-2010) — not currently in scope
Pre-2010 10-Ks sometimes use `<b>` or `<font weight=...>` styling instead of
inline CSS `font-weight:700`. The current parser handles modern (post-2012)
filings well; older filings are out of scope (the corpus uses 2022-2025).

### Chunk boundaries split tables (Day 3 issue)
`RecursiveCharacterTextSplitter` doesn't respect HTML table structure;
financial tables get fragmented across chunks. Mitigation possible at chunk
time but not addressed in v1.

### Golden dataset is self-curated (Day 5-6 issue)
No inter-annotator agreement; production datasets should have at least 2
annotators per question with a measured kappa score.
