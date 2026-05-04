# Known Issues

A running list of edge cases observed during development. Documented but not fixed,
so day-by-day momentum doesn't stall on tangents. Items here are candidates for
"what I'd do differently in production" in the final README.

## Ingestion (Day 1)

- *(populate as you discover edge cases)*

## Parsing (Day 2)

- _10-K/A amendments_ — currently treated the same as 10-K; some tickers have both
- _Inline XBRL formatting_ — newer filings interleave tags that complicate section detection

## Retrieval (Day 3)

- _Chunk boundaries split tables_ — `RecursiveCharacterTextSplitter` doesn't respect
  HTML table structure; financial tables get fragmented

## Eval (Day 5–7)

- _Golden dataset is self-curated_ — no inter-annotator agreement; production datasets
  should have at least 2 annotators per question
