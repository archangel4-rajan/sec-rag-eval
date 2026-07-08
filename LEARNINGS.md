# Learnings

The project is now a working evaluation-first SEC RAG baseline. These are the decisions worth preserving.

## Day 1 - Foundation

- Keep the stack boring: Typer, FastAPI, Chroma, pytest, ruff. The interesting part is the eval loop.
- Put corpus shape and ticker selection in config so experiments can be repeated.

## Day 2 - Parsing

- SEC HTML is not stable enough for pure DOM extraction.
- Plain-text candidate scoring reached 116/120 filings for Item 1A and Item 7.
- Incorporation-by-reference filings are a separate extraction problem, not parser noise.

## Day 3 - Retrieval

- Chunking and embedding need deterministic offline paths; otherwise CI depends on API keys and network.
- The hash provider is a plumbing tool, not a quality benchmark.

## Day 4 - Query Path

- Citation-grounded extractive answers are a useful first baseline because unsupported claims are easier to detect.
- The API should expose refusal state explicitly, not bury it in answer text.

## Day 5-6 - Golden Dataset

- A small balanced seed set is better than a large unverified set.
- Evidence references belong in the dataset; evals without source anchors are too easy to fool.

## Day 7 - Eval Harness

- Deterministic scoring gives the project a stable CI floor before LLM judges are introduced.
- Refusal accuracy must be scored separately from answer quality.

## Day 8 - Experiments

- Retrieval sweeps should produce artifacts, not notebook-only impressions.
- The current hash baseline is intentionally weak and shows the right next bottleneck: semantic retrieval.

## Day 9 - CI And Benchmarking

- CI can validate code, formatting, tests, and golden dataset integrity without requiring paid provider calls.
- Model comparison is scaffolded but not claimed until real provider runs exist.

## Day 10 - Polish

- Documentation should describe measured reality, not the aspirational roadmap.
- The honest story is stronger: a working baseline, clear eval discipline, and explicit next steps.
