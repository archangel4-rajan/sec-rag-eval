# Hash Baseline Eval

Overall pass rate: 40.0% (10/25), 95% CI 23.4%-59.3%.

## Metrics

| Metric | Value |
|---|---:|
| Avg Answer Overlap | 0.457 |
| Avg Citation Precision | 0.083 |
| Avg Evidence Recall | 0.250 |
| Avg Latency Ms | 36.579 |
| Numeric Accuracy | 0.500 |
| Refusal Accuracy | 1.000 |

## Category Results

| Category | Passed | Pass rate | 95% CI | Evidence recall | Citation precision |
|---|---:|---:|---:|---:|---:|
| adversarial | 1/5 | 20.0% | 3.6%-62.4% | 0.200 | 0.067 |
| multi_hop | 0/5 | 0.0% | 0.0%-43.4% | 0.000 | 0.000 |
| numerical | 2/5 | 40.0% | 11.8%-76.9% | 0.400 | 0.133 |
| should_refuse | 5/5 | 100.0% | 56.6%-100.0% | 0.000 | 0.000 |
| single_hop | 2/5 | 40.0% | 11.8%-76.9% | 0.400 | 0.133 |

## Failure Sample

- single_aapl_2022_trade_policy_risk (single_hop): evidence_recall=0.000, answer_overlap=0.167, numeric_correct=None
- single_aapl_2022_competition_risk (single_hop): evidence_recall=0.000, answer_overlap=0.462, numeric_correct=None
- single_aapl_2024_ai_safety_risk (single_hop): evidence_recall=0.000, answer_overlap=0.056, numeric_correct=None
- numerical_adbe_2022_digital_experience_revenue (numerical): evidence_recall=0.000, answer_overlap=0.833, numeric_correct=True
- numerical_adbe_2022_cash_equivalents (numerical): evidence_recall=0.000, answer_overlap=0.625, numeric_correct=False
- numerical_adbe_2022_operating_cash_flow (numerical): evidence_recall=0.000, answer_overlap=0.727, numeric_correct=False
- multi_adbe_document_cloud_2022_vs_2023 (multi_hop): evidence_recall=0.000, answer_overlap=0.778, numeric_correct=None
- multi_adbe_digital_experience_2022_vs_2023 (multi_hop): evidence_recall=0.000, answer_overlap=0.778, numeric_correct=None
- multi_aapl_ai_regulatory_to_safety_risk (multi_hop): evidence_recall=0.000, answer_overlap=0.074, numeric_correct=None
- multi_aapl_supply_chain_and_trade_risk (multi_hop): evidence_recall=0.000, answer_overlap=0.136, numeric_correct=None
