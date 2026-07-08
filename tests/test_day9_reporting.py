"""Day 9-10 tests: stronger reports and Markdown summaries."""

from __future__ import annotations

import json
from pathlib import Path

from sec_rag.eval.dataset import Evidence, GoldenExample
from sec_rag.eval.reporting import render_json_report_markdown
from sec_rag.eval.scoring import build_report, evaluate_response, wilson_interval
from sec_rag.generate import Citation, RagResponse


def test_wilson_interval_bounds_rate() -> None:
    interval = wilson_interval(11, 25)

    assert 0 < interval["low"] < 0.44
    assert 0.44 < interval["high"] < 1
    assert interval["method"] == "wilson_95"


def test_numeric_result_requires_supporting_evidence() -> None:
    example = _example(category="numerical", answer_type="numeric")
    response = RagResponse(
        question=example.question,
        answer="The answer was $42 million.",
        citations=[_citation("wrong")],
        retrieved_count=1,
    )

    result = evaluate_response(example, response)

    assert result.numeric_correct is True
    assert result.evidence_recall == 0
    assert result.passed is False


def test_markdown_renderer_summarizes_eval_report(tmp_path: Path) -> None:
    result = evaluate_response(
        _example(),
        RagResponse(
            question="Question?",
            answer="Apple reported the relevant risk.",
            citations=[_citation("chunk")],
            retrieved_count=1,
        ),
    )
    report = build_report([result])
    payload = json.loads(json.dumps(report, default=lambda item: item.__dict__))

    markdown = render_json_report_markdown(payload, title="Smoke Report")

    assert "# Smoke Report" in markdown
    assert "Category Results" in markdown
    assert "100.0%" in markdown


def _example(category: str = "single_hop", answer_type: str = "extractive") -> GoldenExample:
    payload: dict[str, object] = {
        "id": "example",
        "category": category,
        "question": "Question?",
        "expected_answer": "Apple reported the relevant risk.",
        "answer_type": answer_type,
        "tickers": ["AAPL"],
        "years": [2022],
        "sections": ["1A"],
        "evidence": [
            Evidence(
                chunk_id="chunk",
                ticker="AAPL",
                year=2022,
                section="1A",
                text="Apple reported the relevant risk.",
            )
        ],
    }
    if answer_type == "numeric":
        payload["expected_answer"] = "Apple reported $42 million."
        payload["numeric_answer"] = {"value": 42, "unit": "million USD", "tolerance": 0}
    return GoldenExample.model_validate(payload)


def _citation(chunk_id: str) -> Citation:
    return Citation(
        id=chunk_id,
        label="AAPL 2022 Item 1A chunk 0",
        ticker="AAPL",
        year=2022,
        section="1A",
        chunk_index=0,
        distance=0.1,
        excerpt="Apple reported the relevant risk.",
    )
