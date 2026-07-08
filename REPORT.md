# sec-rag-eval Technical Report

## Abstract

`sec-rag-eval` is an evaluation-first retrieval-augmented generation system over SEC 10-K filings. The project prioritizes measurement over model novelty: every answer is evaluated against evidence-grounded examples, category-level metrics, refusal behavior, and reproducible experiment artifacts.

## Current Claim

The repository currently demonstrates a complete local RAG evaluation loop:

- SEC filing ingestion, parsing, chunking, embedding, retrieval, query answering, and API serving.
- A 25-example evidence-grounded golden seed set across five categories.
- Deterministic scoring for evidence recall, citation precision, answer overlap, numeric correctness, and refusal accuracy.
- Experiment artifacts for baseline evaluation and top-k retrieval sweeps.
- CI checks for linting, formatting, tests, and dataset validity.

It does not yet claim production semantic quality, a 130-example benchmark, provider-backed model comparison, or observability traces.

## Dataset

The corpus target is 120 10-K filings across 30 tickers and fiscal years 2022-2025. The parser extracts Item 1A and Item 7 from 116/120 filings. Known parser exclusions are documented in `KNOWN_ISSUES.md`.

The committed golden set is a balanced 25-example seed:

| Category | Count | Purpose |
|---|---:|---|
| single_hop | 5 | Basic evidence retrieval and citation. |
| multi_hop | 5 | Cross-evidence synthesis pressure. |
| numerical | 5 | Quantitative extraction with tolerance. |
| should_refuse | 5 | Out-of-scope and unsafe request refusal. |
| adversarial | 5 | Near-miss ticker/year/section behavior. |

The original A+ target remains 130 examples. That expansion should be curated, not generated blindly.

## Evaluation Protocol

An answer passes only when it satisfies the category-specific contract:

- Refusal examples must refuse.
- Numeric examples must include the correct value and cite the expected evidence.
- Extractive examples must cite expected evidence and overlap materially with the reference answer.

Reports include Wilson 95% confidence intervals. With 25 examples, intervals are intentionally wide; the project reports them to avoid overstating precision.

## Current Results

The current checked baseline uses deterministic hash embeddings and extractive generation. This is a plumbing baseline, not a semantic retrieval benchmark.

| Run | Pass rate | Interpretation |
|---|---:|---|
| Hash baseline, top_k=5 | 40.0% (10/25), 95% CI 23.4%-59.3% | Offline deterministic baseline; see `experiments/baseline_eval/report.md`. |
| Retrieval sweep, top_k=1 | 28.0% (7/25), 95% CI 14.3%-47.6% | Too little context. |
| Retrieval sweep, top_k=3 | 40.0% (10/25), 95% CI 23.4%-59.3% | Best observed offline setting; see `experiments/retrieval_sweep/report.md`. |
| Retrieval sweep, top_k=5 | 40.0% (10/25), 95% CI 23.4%-59.3% | No gain over top_k=3. |

The most important finding is not the aggregate score; it is the failure shape. Refusal behavior is tractable, while multi-hop and semantic retrieval require provider-backed embeddings and richer retrieval strategies.

## Failure Analysis

Primary current failure modes:

- Hash retrieval returns lexically plausible but semantically wrong chunks.
- Multi-hop examples require retrieval across multiple evidence items, which the current baseline does not optimize.
- Numeric scoring now requires evidence support, preventing correct-but-unsupported numbers from passing.
- The seed set is too small to support strong statistical claims.

## Next Work

1. Expand the golden set to 130 evidence-grounded examples.
2. Rebuild embeddings with a production provider and rerun the full eval suite.
3. Add provider-backed answer generation with citation-aware prompts.
4. Add cross-model LLM judging as a secondary signal, not a replacement for deterministic checks.
5. Report cost, latency, and confidence intervals for every real model run.

## Bottom Line

This is now a serious baseline for an evaluation-first SEC RAG system. The next milestone is empirical: replace the hash baseline with real semantic runs and expand the benchmark without weakening evidence quality.
