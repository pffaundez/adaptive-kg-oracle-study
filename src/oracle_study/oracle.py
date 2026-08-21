from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .schema import validate_interactions


@dataclass(frozen=True)
class OracleSummary:
    lambda_cost: float
    best_fixed_workflow: str
    best_fixed_quality: float
    best_fixed_cost: float
    best_fixed_reward: float
    oracle_quality: float
    oracle_cost: float
    oracle_reward: float
    oracle_gap: float

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def analyze(frame: pd.DataFrame, lambda_cost: float) -> tuple[
    OracleSummary, pd.DataFrame, pd.DataFrame
]:
    validate_interactions(frame)
    if lambda_cost < 0:
        raise ValueError("lambda_cost must be non-negative")

    averaged = (
        frame.groupby(["dataset", "question_id", "workflow_id"], as_index=False)
        .agg(quality=("quality", "mean"), cost=("cost", "mean"))
    )
    averaged["reward"] = averaged["quality"] - lambda_cost * averaged["cost"]

    per_workflow = (
        averaged.groupby("workflow_id", as_index=False)
        .agg(quality=("quality", "mean"), cost=("cost", "mean"), reward=("reward", "mean"))
        .sort_values(["reward", "cost", "workflow_id"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    best = per_workflow.iloc[0]

    oracle_choices = (
        averaged.sort_values(
            ["dataset", "question_id", "reward", "cost", "workflow_id"],
            ascending=[True, True, False, True, True],
        )
        .drop_duplicates(["dataset", "question_id"], keep="first")
        .reset_index(drop=True)
    )

    summary = OracleSummary(
        lambda_cost=lambda_cost,
        best_fixed_workflow=str(best["workflow_id"]),
        best_fixed_quality=float(best["quality"]),
        best_fixed_cost=float(best["cost"]),
        best_fixed_reward=float(best["reward"]),
        oracle_quality=float(oracle_choices["quality"].mean()),
        oracle_cost=float(oracle_choices["cost"].mean()),
        oracle_reward=float(oracle_choices["reward"].mean()),
        oracle_gap=float(oracle_choices["reward"].mean() - best["reward"]),
    )
    return summary, per_workflow, oracle_choices

