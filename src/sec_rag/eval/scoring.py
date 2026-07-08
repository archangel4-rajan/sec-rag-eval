"""Deterministic eval harness for the golden dataset."""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from sec_rag.eval.dataset import GoldenExample
from sec_rag.generate import RagResponse

NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
WORD_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-']*")


class EvalEngine(Protocol):
    """Small protocol implemented by QueryEngine and test doubles."""

    def answer(
        self,
        question: str,
        *,
        top_k: int = 5,
        ticker: str | None = None,
        year: int | None = None,
        section: str | None = None,
    ) -> RagResponse:
        """Answer one query."""
        ...


@dataclass(frozen=True)
class EvalExampleResult:
    """Evaluation result for one golden example."""

    id: str
    category: str
    passed: bool
    refused_correctly: bool | None
    evidence_recall: float | None
    citation_precision: float | None
    answer_overlap: float
    numeric_correct: bool | None
    retrieved_count: int
    latency_ms: float
    cited_ids: list[str]
    expected_evidence_ids: list[str]
    answer: str


@dataclass(frozen=True)
class EvalReport:
    """Full eval report."""

    total_examples: int
    passed_examples: int
    pass_rate: float
    pass_rate_ci_95: dict[str, float | str]
    metrics: dict[str, float]
    failure_summary: dict[str, int]
    by_category: dict[str, dict[str, float | int | dict[str, float | str]]]
    results: list[EvalExampleResult]


def run_eval(
    examples: list[GoldenExample],
    *,
    engine: EvalEngine,
    top_k: int = 5,
) -> EvalReport:
    """Run eval examples through a query engine."""
    results: list[EvalExampleResult] = []
    for example in examples:
        started = time.perf_counter()
        response = engine.answer(
            example.question,
            top_k=top_k,
            ticker=_single_str(example.tickers),
            year=_single_int(example.years),
            section=_single_str(example.sections),
        )
        latency_ms = (time.perf_counter() - started) * 1000
        results.append(evaluate_response(example, response, latency_ms=latency_ms))
    return build_report(results)


def evaluate_response(
    example: GoldenExample,
    response: RagResponse,
    *,
    latency_ms: float = 0.0,
) -> EvalExampleResult:
    """Score one response against one golden example."""
    cited_ids = [citation.id for citation in response.citations]
    expected_ids = [evidence.chunk_id for evidence in example.evidence]
    refused_correctly = _refused_correctly(example, response)
    evidence_recall = _evidence_recall(expected_ids, cited_ids)
    citation_precision = _citation_precision(expected_ids, cited_ids)
    answer_overlap = _answer_overlap(example.expected_answer, response.answer)
    numeric_correct = _numeric_correct(example, response.answer)
    passed = _passed(
        example=example,
        response=response,
        refused_correctly=refused_correctly,
        evidence_recall=evidence_recall,
        answer_overlap=answer_overlap,
        numeric_correct=numeric_correct,
    )
    return EvalExampleResult(
        id=example.id,
        category=example.category,
        passed=passed,
        refused_correctly=refused_correctly,
        evidence_recall=evidence_recall,
        citation_precision=citation_precision,
        answer_overlap=answer_overlap,
        numeric_correct=numeric_correct,
        retrieved_count=response.retrieved_count,
        latency_ms=latency_ms,
        cited_ids=cited_ids,
        expected_evidence_ids=expected_ids,
        answer=response.answer,
    )


def build_report(results: list[EvalExampleResult]) -> EvalReport:
    """Aggregate example-level results."""
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    by_category: dict[str, dict[str, float | int | dict[str, float | str]]] = {}
    for category in sorted({result.category for result in results}):
        category_results = [result for result in results if result.category == category]
        count = len(category_results)
        passed_count = sum(1 for result in category_results if result.passed)
        by_category[category] = {
            "count": count,
            "passed": passed_count,
            "pass_rate": _ratio(passed_count, count),
            "pass_rate_ci_95": wilson_interval(passed_count, count),
            "avg_answer_overlap": _avg(result.answer_overlap for result in category_results),
            "avg_evidence_recall": _avg(
                result.evidence_recall
                for result in category_results
                if result.evidence_recall is not None
            ),
            "avg_citation_precision": _avg(
                result.citation_precision
                for result in category_results
                if result.citation_precision is not None
            ),
            "refusal_accuracy": _avg(
                float(result.refused_correctly)
                for result in category_results
                if result.refused_correctly is not None
            ),
            "numeric_accuracy": _avg(
                float(result.numeric_correct)
                for result in category_results
                if result.numeric_correct is not None
            ),
            "avg_latency_ms": _avg(result.latency_ms for result in category_results),
        }
    return EvalReport(
        total_examples=total,
        passed_examples=passed,
        pass_rate=_ratio(passed, total),
        pass_rate_ci_95=wilson_interval(passed, total),
        metrics=metric_summary(results),
        failure_summary=failure_summary(results),
        by_category=by_category,
        results=results,
    )


def metric_summary(results: list[EvalExampleResult]) -> dict[str, float]:
    """Summarize metrics across all examples."""
    return {
        "avg_answer_overlap": _avg(result.answer_overlap for result in results),
        "avg_evidence_recall": _avg(
            result.evidence_recall for result in results if result.evidence_recall is not None
        ),
        "avg_citation_precision": _avg(
            result.citation_precision for result in results if result.citation_precision is not None
        ),
        "refusal_accuracy": _avg(
            float(result.refused_correctly)
            for result in results
            if result.refused_correctly is not None
        ),
        "numeric_accuracy": _avg(
            float(result.numeric_correct)
            for result in results
            if result.numeric_correct is not None
        ),
        "avg_latency_ms": _avg(result.latency_ms for result in results),
    }


def failure_summary(results: list[EvalExampleResult]) -> dict[str, int]:
    """Count high-level failure modes."""
    unsupported = sum(
        1
        for result in results
        if not result.passed and result.evidence_recall is not None and result.evidence_recall == 0
    )
    wrong_number = sum(1 for result in results if result.numeric_correct is False)
    wrong_refusal = sum(1 for result in results if result.refused_correctly is False)
    weak_overlap = sum(
        1
        for result in results
        if not result.passed and result.answer_overlap < 0.12 and result.numeric_correct is None
    )
    return {
        "failed_examples": sum(1 for result in results if not result.passed),
        "zero_evidence_recall": unsupported,
        "numeric_incorrect": wrong_number,
        "refusal_incorrect": wrong_refusal,
        "weak_answer_overlap": weak_overlap,
    }


def wilson_interval(successes: int, total: int, *, z: float = 1.96) -> dict[str, float | str]:
    """Return a 95% Wilson confidence interval for a binomial rate."""
    if total <= 0:
        return {"low": 0.0, "high": 0.0, "method": "wilson_95"}
    phat = successes / total
    denominator = 1 + z**2 / total
    center = (phat + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total) / denominator
    return {
        "low": max(0.0, center - margin),
        "high": min(1.0, center + margin),
        "method": "wilson_95",
    }


def write_eval_report(report: EvalReport, path: Path) -> None:
    """Write an eval report as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2))


def category_counts(examples: list[GoldenExample]) -> dict[str, int]:
    """Count examples by category for reporting."""
    return dict(sorted(Counter(example.category for example in examples).items()))


def _passed(
    *,
    example: GoldenExample,
    response: RagResponse,
    refused_correctly: bool | None,
    evidence_recall: float | None,
    answer_overlap: float,
    numeric_correct: bool | None,
) -> bool:
    if example.refusal_expected:
        return refused_correctly is True
    if response.refused:
        return False
    if example.answer_type == "numeric":
        return numeric_correct is True and (evidence_recall or 0.0) > 0
    return (evidence_recall or 0.0) > 0 and answer_overlap >= 0.12


def _refused_correctly(example: GoldenExample, response: RagResponse) -> bool | None:
    if not example.refusal_expected and not response.refused:
        return None
    return example.refusal_expected == response.refused


def _evidence_recall(expected_ids: list[str], cited_ids: list[str]) -> float | None:
    if not expected_ids:
        return None
    return len(set(expected_ids) & set(cited_ids)) / len(set(expected_ids))


def _citation_precision(expected_ids: list[str], cited_ids: list[str]) -> float | None:
    if not cited_ids:
        return None
    if not expected_ids:
        return 0.0
    return len(set(expected_ids) & set(cited_ids)) / len(set(cited_ids))


def _answer_overlap(expected_answer: str, answer: str) -> float:
    expected_terms = _content_terms(expected_answer)
    if not expected_terms:
        return 0.0
    answer_terms = _content_terms(answer)
    return len(expected_terms & answer_terms) / len(expected_terms)


def _numeric_correct(example: GoldenExample, answer: str) -> bool | None:
    if example.numeric_answer is None:
        return None
    values = _numbers(answer)
    target = example.numeric_answer.value
    tolerance = example.numeric_answer.tolerance
    return any(abs(value - target) <= tolerance for value in values)


def _numbers(text: str) -> list[float]:
    values: list[float] = []
    for match in NUMBER_PATTERN.findall(text):
        try:
            values.append(float(match.replace(",", "")))
        except ValueError:
            continue
    return values


def _content_terms(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "as",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "with",
    }
    return {
        token
        for token in WORD_PATTERN.findall(text.lower())
        if len(token) > 2 and token not in stopwords
    }


def _single_str(values: list[str]) -> str | None:
    return values[0] if len(values) == 1 else None


def _single_int(values: list[int]) -> int | None:
    return values[0] if len(values) == 1 else None


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _avg(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)
