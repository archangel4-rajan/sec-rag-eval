# Evaluation Protocol

## Objective

Measure whether the system answers SEC 10-K questions with the right evidence, the right refusal behavior, and enough category detail to expose failure modes.

## Categories

| Category | Pass condition |
|---|---|
| single_hop | Cites expected evidence and overlaps with the reference answer. |
| multi_hop | Cites all required evidence items and produces a supported synthesis. |
| numerical | Returns the target value within tolerance and cites expected evidence. |
| should_refuse | Refuses unsupported, private, predictive, or out-of-corpus questions. |
| adversarial | Handles near-miss ticker, year, section, and distractor cases. |

## Metrics

- Pass rate: category-specific binary success.
- Evidence recall: expected source chunks cited by the answer.
- Citation precision: cited chunks that match expected evidence.
- Answer overlap: lightweight deterministic lexical overlap with the reference answer.
- Numeric correctness: value within configured tolerance.
- Refusal accuracy: correct refusal behavior.
- Latency: wall-clock query time per example.

## Confidence

Reports use a Wilson 95% interval for pass rates. This is intentionally conservative for the current 25-example seed set.

## Judge Policy

The current committed eval is deterministic. LLM judges should be added only as a secondary signal after provider-backed runs exist. When added, the judge model family should differ from the answer model family to reduce agreement bias.

## CI Policy

CI should run deterministic checks only:

- lint
- format
- tests
- golden dataset validation

Paid provider runs should produce versioned experiment artifacts, not block every pull request.
