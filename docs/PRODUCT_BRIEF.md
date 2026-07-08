# Product Brief

## Positioning

`sec-rag-eval` is a credibility layer for financial-document AI. It asks a simple product question: before a user trusts an answer from a 10-K corpus, can the system prove where the answer came from and whether similar questions pass a repeatable eval?

## User

The natural user is an AI engineer, data scientist, or technical product lead evaluating whether a RAG system is reliable enough for financial research workflows.

## Product Thesis

The valuable unit is not the chatbot. The valuable unit is the measured answer: question, retrieval path, citation, refusal decision, score, and failure mode.

## What Works Now

- Local SEC 10-K ingestion and retrieval pipeline.
- Citation-grounded extractive answer baseline.
- Evidence-grounded eval seed set.
- Deterministic scoring and experiment artifacts.
- Honest reporting of parser, retrieval, and dataset limits.

## What Is Not Claimed

- Production-grade financial advice.
- Provider-backed semantic benchmark results.
- Complete 130-example eval set.
- Fully solved multi-hop reasoning.
- Observability dashboards.

## Product Direction

The next product milestone is a benchmarkable SEC RAG workbench: run a provider-backed retrieval/generation configuration, get a category-level scorecard, inspect failures, and decide whether the change improved reliability enough to ship.
