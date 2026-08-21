import pandas as pd
import pytest

from oracle_study.oracle import analyze
from oracle_study.schema import validate_interactions


def _frame() -> pd.DataFrame:
    rows = []
    for question, values in {
        "q1": {"cheap": (1.0, 1.0), "robust": (1.0, 2.0)},
        "q2": {"cheap": (0.0, 1.0), "robust": (1.0, 2.0)},
    }.items():
        for workflow, (quality, cost) in values.items():
            rows.append(
                {
                    "dataset": "toy",
                    "question_id": question,
                    "workflow_id": workflow,
                    "run_id": 0,
                    "quality": quality,
                    "cost": cost,
                    "latency_ms": 1,
                    "llm_calls": 1,
                    "sparql_calls": 1,
                    "tokens_in": 1,
                    "tokens_out": 1,
                    "status": "success",
                    "prediction": "query",
                }
            )
    return pd.DataFrame(rows)


def test_oracle_uses_cheaper_tie_break() -> None:
    summary, _, choices = analyze(_frame(), lambda_cost=0.1)
    selected = dict(zip(choices["question_id"], choices["workflow_id"]))
    assert selected == {"q1": "cheap", "q2": "robust"}
    assert summary.oracle_gap > 0


def test_rejects_incomplete_coverage() -> None:
    frame = _frame().query("not (question_id == 'q2' and workflow_id == 'cheap')")
    with pytest.raises(ValueError, match="workflow coverage"):
        validate_interactions(frame)

