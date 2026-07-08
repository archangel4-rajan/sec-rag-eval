"""Repeatable experiment runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sec_rag.eval.dataset import GoldenExample
from sec_rag.eval.scoring import EvalEngine, run_eval


def parse_top_k_values(raw: str) -> list[int]:
    """Parse and validate comma-separated top-k values."""
    values: list[int] = []
    for part in raw.split(","):
        value = int(part.strip())
        if value <= 0:
            raise ValueError("top-k values must be positive")
        values.append(value)
    if not values:
        raise ValueError("at least one top-k value is required")
    return values


def run_retrieval_sweep(
    examples: list[GoldenExample],
    *,
    engine: EvalEngine,
    top_k_values: list[int],
    output_path: Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run eval across top-k values and persist a compact summary."""
    runs: list[dict[str, Any]] = []
    for top_k in top_k_values:
        report = run_eval(examples, engine=engine, top_k=top_k)
        runs.append(
            {
                "top_k": top_k,
                "pass_rate": report.pass_rate,
                "pass_rate_ci_95": report.pass_rate_ci_95,
                "passed_examples": report.passed_examples,
                "total_examples": report.total_examples,
                "metrics": report.metrics,
                "failure_summary": report.failure_summary,
                "by_category": report.by_category,
            }
        )

    best = max(runs, key=lambda run: (run["pass_rate"], -run["top_k"])) if runs else None
    payload = {
        "experiment": "retrieval_top_k_sweep",
        "metadata": metadata or {},
        "runs": runs,
        "best": best,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return payload
