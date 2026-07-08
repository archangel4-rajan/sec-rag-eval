"""Day 7 tests: refusal policy and deterministic eval scoring."""

from __future__ import annotations

import json
from pathlib import Path

from sec_rag.eval.dataset import Evidence, GoldenExample, NumericAnswer
from sec_rag.eval.scoring import evaluate_response, run_eval, write_eval_report
from sec_rag.generate import Citation, RagResponse, RefusalPolicy


def test_refusal_policy_rejects_predictions_private_data_and_future_years() -> None:
    policy = RefusalPolicy()

    assert policy.should_refuse("Predict Apple's stock price next quarter")
    assert policy.should_refuse("What is Rajan's Social Security number?")
    assert policy.should_refuse("What does Tesla's 2027 10-K say?")
    assert policy.should_refuse("What did Apple say about supply chain risk in 2022?") is None


def test_eval_scores_refusal_correctly() -> None:
    example = GoldenExample(
        id="refusal",
        category="should_refuse",
        question="Predict a stock price.",
        expected_answer="Refuse.",
        answer_type="refusal",
        refusal_expected=True,
    )
    response = RefusalPolicy().refusal_response(example.question, "future prediction")

    result = evaluate_response(example, response)

    assert result.passed is True
    assert result.refused_correctly is True
    assert result.evidence_recall is None


def test_eval_scores_evidence_and_overlap() -> None:
    example = _example()
    response = RagResponse(
        question=example.question,
        answer="Apple said supplier facilities were located outside the U.S.",
        citations=[_citation("AAPL_2022_item1A_chunk0000")],
        retrieved_count=1,
    )

    result = evaluate_response(example, response)

    assert result.passed is True
    assert result.evidence_recall == 1.0
    assert result.citation_precision == 1.0
    assert result.answer_overlap > 0


def test_eval_scores_numeric_answers() -> None:
    example = GoldenExample(
        id="numeric",
        category="numerical",
        question="How much?",
        expected_answer="Adobe reported $4,236 million.",
        answer_type="numeric",
        tickers=["ADBE"],
        years=[2022],
        sections=["7"],
        evidence=[
            Evidence(
                chunk_id="ADBE_2022_item7_chunk0075",
                ticker="ADBE",
                year=2022,
                section="7",
                text="Cash and cash equivalents $ 4,236",
            )
        ],
        numeric_answer=NumericAnswer(value=4236, unit="million USD", tolerance=1),
    )
    response = RagResponse(
        question=example.question,
        answer="Adobe reported cash and cash equivalents of $4,236 million.",
        citations=[_citation("ADBE_2022_item7_chunk0075")],
        retrieved_count=1,
    )

    result = evaluate_response(example, response)

    assert result.passed is True
    assert result.numeric_correct is True


def test_run_eval_uses_example_filters_and_writes_report(tmp_path: Path) -> None:
    class FakeEngine:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def answer(self, question: str, **kwargs: object) -> RagResponse:
            self.calls.append({"question": question, **kwargs})
            return RagResponse(
                question=question,
                answer="Apple said supplier facilities were located outside the U.S.",
                citations=[_citation("AAPL_2022_item1A_chunk0000")],
                retrieved_count=1,
            )

    engine = FakeEngine()
    report = run_eval([_example()], engine=engine, top_k=3)
    out = tmp_path / "eval.json"
    write_eval_report(report, out)

    assert report.total_examples == 1
    assert report.pass_rate == 1.0
    assert engine.calls[0]["ticker"] == "AAPL"
    assert engine.calls[0]["year"] == 2022
    assert engine.calls[0]["section"] == "1A"
    assert json.loads(out.read_text())["pass_rate"] == 1.0


def _example() -> GoldenExample:
    return GoldenExample(
        id="single",
        category="single_hop",
        question="Where were supplier facilities located?",
        expected_answer="Apple said supplier facilities were located outside the U.S.",
        answer_type="extractive",
        tickers=["AAPL"],
        years=[2022],
        sections=["1A"],
        evidence=[
            Evidence(
                chunk_id="AAPL_2022_item1A_chunk0000",
                ticker="AAPL",
                year=2022,
                section="1A",
                text="supplier facilities, including manufacturing and assembly sites, are located outside the U.S.",
            )
        ],
    )


def _citation(chunk_id: str) -> Citation:
    return Citation(
        id=chunk_id,
        label="AAPL 2022 Item 1A chunk 0",
        ticker="AAPL",
        year=2022,
        section="1A",
        chunk_index=0,
        distance=0.1,
        excerpt="supplier facilities are located outside the U.S.",
    )
