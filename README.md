# sec-rag-eval

A production-style RAG system over SEC 10-K filings, built around an evaluation harness that catches quality regressions before they ship.

[![CI](https://github.com/archangel4-rajan/sec-rag-eval/actions/workflows/eval.yml/badge.svg)](https://github.com/archangel4-rajan/sec-rag-eval/actions/workflows/eval.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Most RAG demos answer one question well: *can you build one?* Production systems live or die on a harder one: *can you tell when it's wrong, before users do?*

This project answers the second question for a financial-domain RAG over SEC 10-K filings. The retrieval and generation stack is intentionally conventional. The evaluation infrastructure is the point:

- A hand-curated golden dataset spanning factual recall, multi-hop reasoning, numerical extraction, and adversarial cases the system must refuse
- Per-category scoring across faithfulness, answer relevance, context precision/recall, hallucination rate, and refusal accuracy
- A CI gate that blocks pull requests which regress quality on the eval suite
- End-to-end tracing via Langfuse, capturing retrieval, prompts, latency, and token cost per query
- A documented experiment log showing what was tried, what improved scores, and what was reverted

## Architecture

```
            ┌──────────────┐
   query ──▶│   FastAPI    │
            └──────┬───────┘
                   │
            ┌──────▼───────┐    ┌──────────────┐
            │  Retriever   │───▶│   ChromaDB   │
            └──────┬───────┘    └──────────────┘
                   │
            ┌──────▼───────┐    ┌──────────────┐
            │   Generator  │───▶│  OpenAI /    │
            │              │    │  Anthropic / │
            │              │    │   Llama      │
            └──────┬───────┘    └──────────────┘
                   │
            ┌──────▼───────┐
            │   Langfuse   │  (traces, cost, latency)
            └──────────────┘

   Eval suite ──▶ runs against any commit, gates CI on regression
```

## Corpus

90 10-K filings spanning 30 companies across three sectors (technology, industrials/consumer, financials/health) and three fiscal years (2022–2024). Sectors are mixed deliberately so the eval can distinguish retrieval quality from parametric leakage — the model has strong priors on Apple but weak priors on Caterpillar.

Filings are sourced from SEC EDGAR via [`sec-edgar-downloader`](https://pypi.org/project/sec-edgar-downloader/). Item 1A (Risk Factors) and Item 7 (MD&A) are extracted and chunked at 512 tokens with 50-token overlap.

## Eval methodology

The golden dataset contains 130 question/answer pairs, each manually verified against source filings, distributed across five categories:

| Category | Count | What it tests |
|---|---|---|
| Single-hop factual | 30 | Basic retrieval — can the system find one passage and quote it correctly |
| Multi-hop | 25 | Cross-filing synthesis — comparing risks or claims across years or companies |
| Numerical | 25 | Quantitative extraction with tolerance-based scoring |
| Should-refuse | 25 | Questions outside corpus scope or asking for prediction; system must decline |
| Adversarial | 25 | Date-qualified queries, needle-in-haystack retrieval, near-miss distractors |

Each query is scored on multiple dimensions:

- **Faithfulness** (Ragas) — proportion of answer claims supported by retrieved context
- **Answer relevance** (Ragas) — does the answer address the question asked
- **Context precision / recall** (Ragas) — quality of the retrieved chunks
- **Hallucination rate** — LLM-judge verdict on unsupported claims, using a different model family as judge
- **Refusal accuracy** — binary scoring on should-refuse questions
- **Numerical correctness** — relative-tolerance scoring for quantitative answers

Scores are reported per-category. Aggregated averages hide the failure modes that matter — a system at 94% overall might be at 99% on factual questions and 60% on multi-hop, and the latter is what determines whether the system can ship.

## Results

*Populated after Phase 4 (model comparison study). Reserved structure shown below.*

| Category | GPT-4o-mini | Claude Sonnet | Llama-3.1-70B |
|---|---|---|---|
| Single-hop factual | — | — | — |
| Multi-hop | — | — | — |
| Numerical | — | — | — |
| Should-refuse | — | — | — |
| Adversarial | — | — | — |
| **p95 latency (ms)** | — | — | — |
| **Cost per 1k queries** | — | — | — |

## Quickstart

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/archangel4-rajan/sec-rag-eval.git
cd sec-rag-eval
uv sync --extra dev

cp .env.example .env
# Set OPENAI_API_KEY, ANTHROPIC_API_KEY, and SEC_USER_AGENT

# Build the corpus (~15 min, ~$1 in embedding costs)
uv run sec-rag ingest download
uv run sec-rag ingest parse
uv run sec-rag ingest chunk
uv run sec-rag ingest embed

# Single query
uv run sec-rag query "What were Apple's primary supply chain risks in 2023?"

# Full eval run
uv run sec-rag eval run --dataset data/eval/golden.jsonl
```

## Repository layout

```
src/sec_rag/
  ingest/      Download, parse, chunk, embed
  retrieve/    Vector store, retriever with metadata filtering
  generate/    Model-agnostic LLM client and RAG chain
  eval/        Scorers, golden dataset loader, eval runner
  api/         FastAPI service exposing /query and /health
  observe/     Langfuse instrumentation

data/
  filings/     Raw 10-Ks (gitignored)
  chunks/      Parsed sections (gitignored)
  eval/        Golden dataset (committed)

experiments/   One subdirectory per documented experiment, with eval results
.github/workflows/  CI pipeline including the eval gate
```

## Design decisions

A few choices worth calling out, with the reasoning:

**ChromaDB over Pinecone or Weaviate.** Local-first, no managed-service dependency, and fast enough for 15K chunks. Migrating to a hosted vector DB is a configuration change, not an architecture change.

**Per-category scoring instead of aggregate.** Aggregate scores reward the easy questions and hide systematic failure on the hard ones. Multi-hop and adversarial categories are the actual quality signal; factual recall is table stakes.

**Cross-model LLM-as-judge.** The hallucination scorer uses Claude to judge GPT outputs and vice-versa. Same-model self-evaluation has well-documented agreement bias; switching families costs more but produces verdicts that correlate better with human review.

**Eval gate in CI rather than post-deploy.** The faster a regression is caught, the cheaper it is to fix. A prompt change that drops faithfulness by 3 points should fail the PR, not require a rollback.

## Limitations

What this project deliberately does not address:

- **Prompt-injection guardrails.** No defenses against adversarial input; production deployment would require input validation and output filtering.
- **PII filtering.** 10-Ks are public, so this isn't a concern for the demo corpus, but private-document RAG would require a redaction layer.
- **XBRL structured data.** Quantitative questions are answered from prose in MD&A and Risk Factors. Structured financial data from XBRL would improve numerical accuracy but is out of scope.
- **Eval-set size.** 130 questions catches obvious failure modes; production systems need 10×+ that, ideally with multiple annotators per question for inter-rater agreement.
- **Operational maturity.** Runbooks, on-call procedures, and cost-anomaly alerts are not implemented.

## License

MIT — see [LICENSE](LICENSE).
