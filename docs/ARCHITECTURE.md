# Architecture

## Pipeline

1. Download SEC 10-K filings.
2. Parse Item 1A and Item 7 into structured section records.
3. Chunk sections into embedding-ready passages.
4. Embed chunks into Chroma.
5. Retrieve chunks with optional ticker, year, and section filters.
6. Apply refusal policy before retrieval when the question violates corpus scope.
7. Generate citation-grounded extractive answers.
8. Evaluate answers against the golden set.

## Design Choices

## Parser

The parser uses plain-text candidate scoring instead of brittle HTML assumptions. That handles the dominant SEC filing formats in the corpus and documents unsupported incorporation-by-reference patterns separately.

## Embeddings

The project supports production embeddings and a deterministic hash provider. The hash provider exists for offline tests and CI; it is not a semantic quality baseline.

## Answering

The default answerer is extractive and citation-first. This lowers hallucination risk and creates a clean baseline before provider-backed generation is introduced.

## Evaluation

Evaluation is part of the product surface, not an afterthought. Reports include per-category pass rates, evidence recall, citation precision, numeric correctness, refusal accuracy, latency, and confidence intervals.

## API

FastAPI exposes health and query endpoints. Query responses include answer text, citations, retrieved count, refusal status, and refusal reason.
