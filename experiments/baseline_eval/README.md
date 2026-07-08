# Baseline Eval

Deterministic eval over the 25-row golden seed set.

```bash
uv run sec-rag eval run \
  --provider hash \
  --top-k 5 \
  --output experiments/baseline_eval/results.json

uv run sec-rag eval summarize \
  --results experiments/baseline_eval/results.json \
  --output experiments/baseline_eval/report.md \
  --title "Hash Baseline Eval"
```

Interpretation: the hash provider is an offline plumbing baseline, not a semantic retrieval model. The report is useful because it validates the scoring harness, failure taxonomy, and artifact workflow.
