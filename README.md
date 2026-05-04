# sec-rag-eval

> An evaluation-first RAG system over SEC 10-K filings. The retrieval and generation are deliberately simple. The eval harness, observability, and CI gate are the point.

[![CI](https://github.com/USERNAME/sec-rag-eval/actions/workflows/eval.yml/badge.svg)](https://github.com/USERNAME/sec-rag-eval/actions/workflows/eval.yml)

## Why this project exists

Most RAG demos answer the easy half of the question: *can you build one?* Production AI systems live or die on the harder half: *can you tell when it's wrong, before your users do?*

This project builds a financial-domain RAG system over a curated corpus of SEC 10-K filings, then wraps it in:

- A hand-curated golden dataset of 130 question/answer pairs across 5 categories (factual, multi-hop, numerical, should-refuse, adversarial)
- An automated eval harness that scores faithfulness, answer relevance, context precision/recall, hallucination rate, refusal accuracy, and numerical correctness — broken out per category
- A regression CI gate that fails PRs which regress quality
- Full observability via Langfuse traces showing retrieval, prompts, latency, and token cost per query
- A documented experiment log showing what was tried, what worked, and what was reverted

## Architecture

*(diagram added Day 10)*

## Quickstart

```bash
# 1. Install
git clone https://github.com/USERNAME/sec-rag-eval.git
cd sec-rag-eval
uv sync --extra dev

# 2. Configure
cp .env.example .env
# Add OPENAI_API_KEY, ANTHROPIC_API_KEY

# 3. Ingest filings (~15 min, ~$1 in embeddings)
sec-rag ingest download
sec-rag ingest parse
sec-rag ingest chunk
sec-rag ingest embed

# 4. Run a query
sec-rag query "What were Apple's primary supply chain risks in 2023?"

# 5. Run the eval suite
sec-rag eval run --dataset data/eval/golden.jsonl
```

## Eval results

*(populated Day 7 onward)*

| Category       | GPT-4o-mini | Claude Sonnet | Llama-3.1-70B |
|----------------|-------------|---------------|---------------|
| Single-hop     | TBD         | TBD           | TBD           |
| Multi-hop      | TBD         | TBD           | TBD           |
| Numerical      | TBD         | TBD           | TBD           |
| Should-refuse  | TBD         | TBD           | TBD           |
| Adversarial    | TBD         | TBD           | TBD           |

## Findings

*(populated Day 10)*

Three things I learned that surprised me:

1. *(TBD)*
2. *(TBD)*
3. *(TBD)*

## Limitations and next steps

What this project does *not* do, and what would be required for true production deployment:

- **Prompt-injection guardrails** — no defenses against malicious queries; would add input filtering and output validation
- **PII filtering** — 10-Ks are public, but a production deployment over private docs would need PII detection
- **Larger eval corpus** — 130 questions catches the obvious failures; a production system would need 10x more
- **Financial statement retrieval** — this version focuses on Item 1A (Risk Factors) and Item 7 (MD&A); structured XBRL data is not retrieved
- **On-call runbooks** — production systems need playbooks for cost spikes, quality regressions, and model-provider outages

## Repo structure

```
sec-rag-eval/
├── src/sec_rag/
│   ├── ingest/      # download, parse, chunk, embed
│   ├── retrieve/    # vector store and retriever
│   ├── generate/    # LLM abstraction and RAG chain
│   ├── eval/        # scorers, golden dataset, runner
│   ├── api/         # FastAPI service
│   └── observe/     # Langfuse instrumentation
├── data/
│   ├── filings/     # raw 10-Ks (gitignored)
│   ├── chunks/      # parsed sections (gitignored)
│   └── eval/        # golden dataset (committed!)
├── experiments/     # one folder per documented experiment
├── tests/
└── .github/workflows/
```

## License

MIT
