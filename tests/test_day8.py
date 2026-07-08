"""Day 8 tests: repeatable experiment runners."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sec_rag.eval.dataset import Evidence, GoldenExample
from sec_rag.eval.experiments import parse_top_k_values, run_retrieval_sweep
from sec_rag.generate import Citation, RagResponse


def test_parse_top_k_values() -> None:
    assert parse_top_k_values("1, 3,5") == [1, 3, 5]
    with pytest.raises(ValueError):
        parse_top_k_values("0")


def test_run_retrieval_sweep_writes_summary(tmp_path: Path) -> None:
    class FakeEngine:
        def answer(self, question: str, **_kwargs: object) -> RagResponse:
            return RagResponse(
                question=question,
                answer="Apple said supplier facilities were outside the U.S.",
                citations=[
                    Citation(
                        id="AAPL_2022_item1A_chunk0000",
                        label="AAPL 2022 Item 1A chunk 0",
                        ticker="AAPL",
                        year=2022,
                        section="1A",
                        chunk_index=0,
                        distance=0.1,
                        excerpt="supplier facilities were outside the U.S.",
                    )
                ],
                retrieved_count=1,
            )

    output = tmp_path / "results.json"
    result = run_retrieval_sweep(
        [_example()],
        engine=FakeEngine(),
        top_k_values=[1, 3],
        output_path=output,
    )

    assert result["best"]["top_k"] == 1
    assert len(result["runs"]) == 2
    assert json.loads(output.read_text())["experiment"] == "retrieval_top_k_sweep"


def _example() -> GoldenExample:
    return GoldenExample(
        id="single",
        category="single_hop",
        question="Where were supplier facilities located?",
        expected_answer="Apple said supplier facilities were outside the U.S.",
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
                text="supplier facilities were outside the U.S.",
            )
        ],
    )
