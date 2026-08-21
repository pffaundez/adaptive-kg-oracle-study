from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "dataset",
    "question_id",
    "workflow_id",
    "run_id",
    "quality",
    "cost",
    "latency_ms",
    "llm_calls",
    "sparql_calls",
    "tokens_in",
    "tokens_out",
    "status",
    "prediction",
}


def validate_interactions(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if frame.empty:
        raise ValueError("Interaction log is empty")
    if frame[list(REQUIRED_COLUMNS - {"prediction"})].isnull().any().any():
        raise ValueError("Required fields other than prediction cannot be null")
    if not frame["quality"].between(0.0, 1.0).all():
        raise ValueError("quality must be between 0 and 1")
    if (frame["cost"] < 0).any():
        raise ValueError("cost must be non-negative")

    keys = ["dataset", "question_id", "workflow_id", "run_id"]
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicate rows found for key {keys}")

    coverage = frame.groupby(["dataset", "question_id"])["workflow_id"].nunique()
    if coverage.nunique() != 1:
        raise ValueError("Every question must have the same workflow coverage")

