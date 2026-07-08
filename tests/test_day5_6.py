"""Day 5-6 tests: golden dataset schema and validation."""

from __future__ import annotations

import pytest

from sec_rag.config import settings
from sec_rag.eval.dataset import (
    Evidence,
    GoldenExample,
    NumericAnswer,
    load_golden_dataset,
    summarize_by_category,
    validate_evidence_against_chunks,
    validate_golden_dataset,
)


def test_committed_golden_dataset_is_balanced_and_valid() -> None:
    examples = load_golden_dataset(settings.eval_dir / "golden.jsonl")
    issues = validate_golden_dataset(examples)
    summary = summarize_by_category(examples)

    assert issues == []
    assert len(examples) >= 25
    assert summary == {
        "adversarial": 5,
        "multi_hop": 5,
        "numerical": 5,
        "should_refuse": 5,
        "single_hop": 5,
    }


def test_committed_evidence_matches_local_chunks_when_available() -> None:
    chunks_path = settings.chunks_dir / "chunks.jsonl"
    if not chunks_path.exists():
        pytest.skip(f"{chunks_path} not found; run `sec-rag ingest chunk` first")

    examples = load_golden_dataset(settings.eval_dir / "golden.jsonl")
    issues = validate_evidence_against_chunks(examples, chunks_path)

    assert issues == []


def test_validation_rejects_missing_non_refusal_evidence() -> None:
    example = GoldenExample(
        id="bad",
        category="single_hop",
        question="Question?",
        expected_answer="Answer.",
        answer_type="extractive",
        evidence=[],
    )

    assert "bad: non-refusal example must include evidence" in validate_golden_dataset([example])


def test_validation_rejects_numerical_without_numeric_answer() -> None:
    example = GoldenExample(
        id="bad_numeric",
        category="numerical",
        question="How much?",
        expected_answer="$1",
        answer_type="numeric",
        evidence=[
            Evidence(
                chunk_id="AAPL_2025_item7_chunk0000",
                ticker="AAPL",
                year=2025,
                section="7",
                text="$1",
            )
        ],
    )

    assert "bad_numeric: numerical example must include numeric_answer" in validate_golden_dataset(
        [example]
    )


def test_validation_accepts_refusal_without_evidence() -> None:
    example = GoldenExample(
        id="refusal",
        category="should_refuse",
        question="Predict a stock price.",
        expected_answer="Refuse.",
        answer_type="refusal",
        refusal_expected=True,
    )

    assert validate_golden_dataset([example]) == [
        "missing category: adversarial",
        "missing category: multi_hop",
        "missing category: numerical",
        "missing category: single_hop",
    ]


def test_numeric_answer_allows_zero_tolerance() -> None:
    numeric = NumericAnswer(value=42, unit="million USD", tolerance=0)

    assert numeric.value == 42
