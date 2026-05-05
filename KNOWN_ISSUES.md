# Known Issues

A running list of edge cases observed during development. Documented but not
fixed, so day-by-day momentum doesn't stall on tangents. Items here are
candidates for "what I'd do differently in production" in the final README.

## Filing pattern: incorporation by reference

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
  than Item-N anchors.

Examples observed: GE, Honeywell, McDonald's, Morgan Stanley, Wells Fargo.
Excluded from the corpus; replaced with DIS, LOW, TGT, C, AXP.

## Filing pattern: non-Item section anchors (Citigroup)

Citigroup's 10-K filings (2022-2025) don't use the literal string "Item 1A"
or "Item 7" anywhere in the body. Their cross-reference index lists sections
as standalone tokens — `"1A. Risk Factors"`, `"7. Management's Discussion"` —
without the word "Item" in front. Actual section headings are all-caps topic
labels: `"RISK FACTORS"`, `"MANAGING GLOBAL RISK"`, etc.

Diagnostic for Citi 2022:
- 1.6 MB of plain text, only 1 occurrence of the standalone token `"1A."`
- 56 occurrences of `"Risk Factors"` — these are the real anchors
- Zero occurrences of `"Item 1A"`, `"Item 7"`, etc.

Reliable extraction would require a third detection strategy that scores
topic-name candidates ("Risk Factors", "Management's Discussion") with the
same gap/density/cross-reference logic, but anchored on different tokens.
This is feasible but out of scope for the current iteration. Citi (4 filings)
is the only ticker in our corpus affected; remaining 116/120 filings parse
above the quality threshold.

## Smaller issues

### Older filings (pre-2010) — not in scope
Pre-2010 10-Ks sometimes use `<b>` or `<font weight=...>` styling instead of
inline CSS `font-weight:700`. Out of scope (corpus uses 2022-2025).

### Chunk boundaries split tables (Day 3 issue)
`RecursiveCharacterTextSplitter` doesn't respect HTML table structure;
financial tables get fragmented across chunks. Mitigation possible at chunk
time but not addressed in v1.

### Golden dataset is self-curated (Day 5-6 issue)
No inter-annotator agreement; production datasets should have at least 2
annotators per question with a measured kappa score.
