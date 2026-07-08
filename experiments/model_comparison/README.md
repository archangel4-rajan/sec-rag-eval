# Model Comparison

Benchmark target:

| Run | Embeddings | Answering | Judge |
|---|---|---|---|
| baseline | hash | extractive | deterministic |
| openai | text-embedding-3-small | GPT-4o-mini | deterministic + LLM judge |
| anthropic | text-embedding-3-small | Claude Sonnet | deterministic + cross-model judge |

Only the baseline is runnable without provider keys. Do not publish GPT/Claude
numbers until the full corpus is embedded with production embeddings and the
same golden set is run with fixed prompts, fixed top-k, recorded costs, and
saved raw outputs.
