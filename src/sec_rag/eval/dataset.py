"""Golden dataset schema and validation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from sec_rag.config import settings
from sec_rag.ingest.embed import load_chunk_records

Category = Literal["single_hop", "multi_hop", "numerical", "should_refuse", "adversarial"]
AnswerType = Literal["extractive", "numeric", "refusal"]


class Evidence(BaseModel):
    """One source evidence string for a golden example."""

    chunk_id: str = Field(min_length=1)
    ticker: str = Field(min_length=1)
    year: int
    section: str = Field(min_length=1)
    text: str = Field(min_length=1)


class NumericAnswer(BaseModel):
    """Expected numeric value with tolerance."""

    value: float
    unit: str = Field(min_length=1)
    tolerance: float = Field(ge=0)


class GoldenExample(BaseModel):
    """One curated question/answer pair."""

    id: str = Field(min_length=1)
    category: Category
    question: str = Field(min_length=1)
    expected_answer: str = Field(min_length=1)
    answer_type: AnswerType
    tickers: list[str] = Field(default_factory=list)
    years: list[int] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    numeric_answer: NumericAnswer | None = None
    refusal_expected: bool = False


def load_golden_dataset(path: Path | None = None) -> list[GoldenExample]:
    """Load golden examples from JSONL."""
    path = path or settings.eval_dir / "golden.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {path}")

    examples: list[GoldenExample] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            examples.append(GoldenExample.model_validate(payload))
        except Exception as exc:  # noqa: BLE001 - include line number in validation error
            raise ValueError(f"Invalid golden dataset row {line_number}: {exc}") from exc
    return examples


def validate_golden_dataset(examples: list[GoldenExample]) -> list[str]:
    """Return human-readable validation issues."""
    issues: list[str] = []
    if not examples:
        return ["golden dataset is empty"]

    ids = [example.id for example in examples]
    duplicate_ids = sorted(item for item, count in Counter(ids).items() if count > 1)
    for duplicate_id in duplicate_ids:
        issues.append(f"duplicate id: {duplicate_id}")

    normalized_questions = [_normalize(example.question) for example in examples]
    duplicate_questions = sorted(
        question for question, count in Counter(normalized_questions).items() if count > 1
    )
    for duplicate_question in duplicate_questions:
        issues.append(f"duplicate question: {duplicate_question}")

    categories = set(summarize_by_category(examples))
    expected_categories = {"single_hop", "multi_hop", "numerical", "should_refuse", "adversarial"}
    missing = sorted(expected_categories - categories)
    for category in missing:
        issues.append(f"missing category: {category}")

    for example in examples:
        issues.extend(_validate_example(example))
    return issues


def validate_evidence_against_chunks(
    examples: list[GoldenExample],
    chunks_path: Path | None = None,
) -> list[str]:
    """Check evidence snippets against local chunks.jsonl when available."""
    chunks_path = chunks_path or settings.chunks_dir / "chunks.jsonl"
    if not chunks_path.exists():
        return [f"chunk file not found for evidence validation: {chunks_path}"]

    chunks = {str(chunk["id"]): str(chunk["text"]) for chunk in load_chunk_records(chunks_path)}
    issues: list[str] = []
    for example in examples:
        for evidence in example.evidence:
            chunk_text = chunks.get(evidence.chunk_id)
            if chunk_text is None:
                issues.append(f"{example.id}: evidence chunk not found: {evidence.chunk_id}")
                continue
            if evidence.text not in chunk_text:
                issues.append(f"{example.id}: evidence text not found in chunk {evidence.chunk_id}")
    return issues


def summarize_by_category(examples: list[GoldenExample]) -> dict[str, int]:
    """Count examples by category."""
    counts = Counter(example.category for example in examples)
    return dict(sorted(counts.items()))


def _validate_example(example: GoldenExample) -> list[str]:
    issues: list[str] = []
    if example.category == "should_refuse":
        if not example.refusal_expected:
            issues.append(f"{example.id}: should_refuse example must set refusal_expected=true")
        if example.evidence:
            issues.append(f"{example.id}: should_refuse example should not include evidence")
        if example.answer_type != "refusal":
            issues.append(f"{example.id}: should_refuse example must use answer_type=refusal")
        return issues

    if example.refusal_expected:
        issues.append(f"{example.id}: non-refusal example cannot set refusal_expected=true")
    if not example.evidence:
        issues.append(f"{example.id}: non-refusal example must include evidence")
    if len(example.expected_answer.split()) < 5:
        issues.append(f"{example.id}: expected answer is too short to grade reliably")
    if example.category == "multi_hop" and len(example.evidence) < 2:
        issues.append(f"{example.id}: multi_hop example must include at least two evidence items")
    if example.category == "numerical" and example.numeric_answer is None:
        issues.append(f"{example.id}: numerical example must include numeric_answer")
    if example.category == "numerical" and example.answer_type != "numeric":
        issues.append(f"{example.id}: numerical example must use answer_type=numeric")
    if example.answer_type == "numeric" and example.numeric_answer is None:
        issues.append(f"{example.id}: numeric answer_type must include numeric_answer")
    for evidence in example.evidence:
        issues.extend(_validate_evidence_metadata(example, evidence))
    return issues


def _validate_evidence_metadata(example: GoldenExample, evidence: Evidence) -> list[str]:
    issues: list[str] = []
    if example.tickers and evidence.ticker not in example.tickers:
        issues.append(f"{example.id}: evidence ticker {evidence.ticker} not in example tickers")
    if example.years and evidence.year not in example.years:
        issues.append(f"{example.id}: evidence year {evidence.year} not in example years")
    if example.sections and evidence.section not in example.sections:
        issues.append(f"{example.id}: evidence section {evidence.section} not in example sections")
    if len(evidence.text.split()) < 3:
        issues.append(f"{example.id}: evidence text is too short")
    return issues


def _normalize(value: str) -> str:
    return " ".join(value.lower().split())
