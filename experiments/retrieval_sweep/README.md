# Retrieval Sweep

Top-k sweep over the golden seed set.

```bash
uv run sec-rag experiment retrieval \
  --provider hash \
  --top-k-values 1,3,5 \
  --output experiments/retrieval_sweep/results.json

uv run sec-rag eval summarize \
  --results experiments/retrieval_sweep/results.json \
  --output experiments/retrieval_sweep/report.md \
  --title "Hash Retrieval Sweep"
```

Interpretation: this sweep verifies repeatable experiment mechanics. It should not be used as evidence of semantic retrieval quality until the same protocol is run with production embeddings.
