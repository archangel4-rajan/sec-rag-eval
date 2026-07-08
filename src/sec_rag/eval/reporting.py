"""Markdown rendering for eval and experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON report artifact."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def render_json_report_markdown(payload: dict[str, Any], *, title: str | None = None) -> str:
    """Render a supported JSON eval artifact as compact Markdown."""
    if "runs" in payload and payload.get("experiment") == "retrieval_top_k_sweep":
        return render_retrieval_sweep_markdown(payload, title=title)
    if "results" in payload and "by_category" in payload:
        return render_eval_report_markdown(payload, title=title)
    raise ValueError("unsupported report payload")


def write_markdown_report(markdown: str, path: Path) -> None:
    """Persist rendered Markdown."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown)


def render_eval_report_markdown(payload: dict[str, Any], *, title: str | None = None) -> str:
    """Render an eval report JSON payload."""
    lines = [f"# {title or 'Eval Report'}", ""]
    ci = _ci(payload.get("pass_rate_ci_95", {}))
    lines.append(
        f"Overall pass rate: {_pct(payload.get('pass_rate', 0))} "
        f"({payload.get('passed_examples', 0)}/{payload.get('total_examples', 0)}), "
        f"95% CI {ci}."
    )
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for key, value in sorted(payload.get("metrics", {}).items()):
        lines.append(f"| {_label(key)} | {_num(value)} |")
    lines.append("")
    lines.append("## Category Results")
    lines.append("")
    lines.append(
        "| Category | Passed | Pass rate | 95% CI | Evidence recall | Citation precision |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for category, row in sorted(payload.get("by_category", {}).items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    category,
                    f"{row.get('passed', 0)}/{row.get('count', 0)}",
                    _pct(row.get("pass_rate", 0)),
                    _ci(row.get("pass_rate_ci_95", {})),
                    _num(row.get("avg_evidence_recall", 0)),
                    _num(row.get("avg_citation_precision", 0)),
                ]
            )
            + " |"
        )
    failures = [result for result in payload.get("results", []) if not result.get("passed")]
    if failures:
        lines.append("")
        lines.append("## Failure Sample")
        lines.append("")
        for result in failures[:10]:
            lines.append(
                "- "
                f"{result.get('id')} ({result.get('category')}): "
                f"evidence_recall={_num(result.get('evidence_recall'))}, "
                f"answer_overlap={_num(result.get('answer_overlap'))}, "
                f"numeric_correct={result.get('numeric_correct')}"
            )
    lines.append("")
    return "\n".join(lines)


def render_retrieval_sweep_markdown(payload: dict[str, Any], *, title: str | None = None) -> str:
    """Render a retrieval sweep artifact."""
    lines = [f"# {title or 'Retrieval Sweep Report'}", ""]
    best = payload.get("best") or {}
    if best:
        lines.append(
            f"Best observed setting: top_k={best.get('top_k')} with pass rate "
            f"{_pct(best.get('pass_rate', 0))}."
        )
        lines.append("")
    lines.append("| top_k | Passed | Pass rate | 95% CI | Evidence recall | Citation precision |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for run in payload.get("runs", []):
        metrics = run.get("metrics", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(run.get("top_k")),
                    f"{run.get('passed_examples', 0)}/{run.get('total_examples', 0)}",
                    _pct(run.get("pass_rate", 0)),
                    _ci(run.get("pass_rate_ci_95", {})),
                    _num(metrics.get("avg_evidence_recall", 0)),
                    _num(metrics.get("avg_citation_precision", 0)),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "Interpretation: this is an offline hash-embedding plumbing run unless the metadata "
        "states otherwise. Treat semantic retrieval quality as unknown until provider-backed "
        "embeddings are used."
    )
    lines.append("")
    return "\n".join(lines)


def _pct(value: object) -> str:
    return f"{_as_float(value):.1%}"


def _num(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{_as_float(value):.3f}"


def _ci(value: object) -> str:
    if not isinstance(value, dict):
        return "n/a"
    return f"{_pct(value.get('low', 0))}-{_pct(value.get('high', 0))}"


def _as_float(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"expected numeric value, got {type(value).__name__}")


def _label(value: str) -> str:
    return value.replace("_", " ").title()
