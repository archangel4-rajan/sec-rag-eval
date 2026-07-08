# Dataset Card

## Corpus

The target corpus is 120 SEC 10-K filings across 30 companies and fiscal years 2022-2025. The parser extracts Item 1A and Item 7. Current parser coverage is 116/120 filings.

## Golden Seed Set

The committed golden set contains 25 examples:

- 5 single-hop
- 5 multi-hop
- 5 numerical
- 5 should-refuse
- 5 adversarial

Every non-refusal example includes evidence text, chunk id, ticker, year, and section.

## Validation

Dataset validation checks:

- schema validity
- duplicate ids
- duplicate questions
- category coverage
- answer type consistency
- numeric tolerance presence
- multi-hop evidence count
- evidence metadata alignment
- optional evidence-text match against local chunks

## Known Limits

- The seed set is balanced but small.
- There is no inter-annotator agreement yet.
- The current examples are source-grounded but not large enough for strong statistical claims.
- Incorporation-by-reference filings remain documented exclusions.

## Expansion Target

The A+ target is 130 examples:

| Category | Target |
|---|---:|
| single_hop | 30 |
| multi_hop | 25 |
| numerical | 25 |
| should_refuse | 25 |
| adversarial | 25 |

Expansion should be curated from source chunks and reviewed manually. Automatically generated questions may be used as candidates, not as final labels.
