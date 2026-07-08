# Hash Retrieval Sweep

Best observed setting: top_k=3 with pass rate 40.0%.

| top_k | Passed | Pass rate | 95% CI | Evidence recall | Citation precision |
|---:|---:|---:|---:|---:|---:|
| 1 | 7/25 | 28.0% | 14.3%-47.6% | 0.100 | 0.100 |
| 3 | 10/25 | 40.0% | 23.4%-59.3% | 0.250 | 0.083 |
| 5 | 10/25 | 40.0% | 23.4%-59.3% | 0.250 | 0.083 |

Interpretation: this is an offline hash-embedding plumbing run unless the metadata states otherwise. Treat semantic retrieval quality as unknown until provider-backed embeddings are used.
