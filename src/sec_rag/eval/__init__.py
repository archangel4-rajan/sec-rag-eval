"""Evaluation dataset utilities."""

from sec_rag.eval.dataset import (
    Evidence,
    GoldenExample,
    NumericAnswer,
    load_golden_dataset,
    summarize_by_category,
    validate_golden_dataset,
)
from sec_rag.eval.scoring import (
    EvalExampleResult,
    EvalReport,
    evaluate_response,
    run_eval,
)

__all__ = [
    "Evidence",
    "GoldenExample",
    "NumericAnswer",
    "load_golden_dataset",
    "summarize_by_category",
    "validate_golden_dataset",
    "EvalExampleResult",
    "EvalReport",
    "evaluate_response",
    "run_eval",
]
