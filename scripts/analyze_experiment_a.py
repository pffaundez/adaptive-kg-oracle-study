#!/usr/bin/env python3
"""Analyze fixed workflows and retrospective oracles for Experiment A."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


DEFAULT_WORKFLOWS = ("W1", "W2", "W3", "W4", "W5", "W6")


class AnalysisError(ValueError):
    """Raised when result files cannot be compared safely."""


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute fixed-workflow and oracle results for Experiment A."
    )
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=Path("results/experiment_a"),
    )
    parser.add_argument(
        "--workflows",
        nargs="+",
        default=list(DEFAULT_WORKFLOWS),
        help="Workflow IDs in deterministic tie-breaking order.",
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[0.0, 0.01, 0.05, 0.1],
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _record_key(record: Mapping[str, Any]) -> tuple[str, int]:
    question_id = record.get("question_id")
    run_id = record.get("run_id")
    if question_id is None or not isinstance(run_id, int):
        raise AnalysisError("Every record requires question_id and integer run_id.")
    return str(question_id), run_id


def _load_workflow(
    path: Path,
    workflow_id: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing result file for {workflow_id}: {path}")
    records: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnalysisError(
                    f"Invalid JSON at {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise AnalysisError(
                    f"Record at {path}:{line_number} must be an object."
                )
            if record.get("workflow_id") != workflow_id:
                raise AnalysisError(
                    f"Expected workflow_id={workflow_id!r} at "
                    f"{path}:{line_number}."
                )
            key = _record_key(record)
            if key in records:
                raise AnalysisError(f"Duplicate key {key!r} in {path}.")
            records[key] = record
    if not records:
        raise AnalysisError(f"Result file is empty: {path}")
    return records


def _quality(record: Mapping[str, Any]) -> float:
    value = float(record.get("quality", 0.0))
    if not 0.0 <= value <= 1.0:
        raise AnalysisError(f"quality must be in [0, 1], got {value}.")
    return value


def _cost(record: Mapping[str, Any]) -> float:
    cost = record.get("cost")
    if not isinstance(cost, Mapping):
        raise AnalysisError("Every result requires a cost mapping.")
    # Evaluation-only endpoint calls are excluded from deployment action cost.
    return float(cost.get("llm_calls", 0)) + float(
        cost.get("workflow_sparql_executions", 0)
    )


def _metrics(record: Mapping[str, Any]) -> tuple[float, float]:
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        return 0.0, 0.0
    return float(metrics.get("precision", 0.0)), float(metrics.get("recall", 0.0))


def _summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    qualities = [_quality(record) for record in records]
    precisions, recalls = zip(*(_metrics(record) for record in records))
    macro_precision = sum(precisions) / len(records)
    macro_recall = sum(recalls) / len(records)
    denominator = macro_precision + macro_recall
    return {
        "records": len(records),
        "mean_question_f1": sum(qualities) / len(records),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "qald_macro_f1": (
            2.0 * macro_precision * macro_recall / denominator
            if denominator
            else 0.0
        ),
        "mean_action_cost": sum(_cost(record) for record in records) / len(records),
        "statuses": dict(sorted(Counter(str(r.get("status")) for r in records).items())),
    }


def _choose(
    candidates: list[tuple[str, Mapping[str, Any]]],
    *,
    penalty: float,
) -> tuple[str, Mapping[str, Any], float]:
    # Workflow list order is the final deterministic tie-breaker.
    scored = [
        (_quality(record) - penalty * _cost(record), _cost(record), index, workflow, record)
        for index, (workflow, record) in enumerate(candidates)
    ]
    reward, _, _, workflow, record = max(
        scored,
        key=lambda item: (item[0], -item[1], -item[2]),
    )
    return workflow, record, reward


def main() -> int:
    args = _arguments()
    if len(set(args.workflows)) != len(args.workflows):
        raise AnalysisError("Workflow IDs must not be duplicated.")
    if any(value < 0 for value in args.lambdas):
        raise AnalysisError("Lambdas must be non-negative.")

    directory = args.results_directory.resolve()
    by_workflow = {
        workflow: _load_workflow(directory / f"{workflow}-results.jsonl", workflow)
        for workflow in args.workflows
    }
    reference_workflow = args.workflows[0]
    reference_keys = set(by_workflow[reference_workflow])
    for workflow, records in by_workflow.items():
        if set(records) != reference_keys:
            missing = sorted(reference_keys - set(records))
            extra = sorted(set(records) - reference_keys)
            raise AnalysisError(
                f"{workflow} has a different question/run set; "
                f"missing={missing[:10]}, extra={extra[:10]}."
            )

    ordered_keys = sorted(reference_keys, key=lambda key: (int(key[0]), key[1]))
    fixed = {
        workflow: _summary([by_workflow[workflow][key] for key in ordered_keys])
        for workflow in args.workflows
    }
    best_fixed = max(
        args.workflows,
        key=lambda workflow: (
            fixed[workflow]["mean_question_f1"],
            -fixed[workflow]["mean_action_cost"],
            -args.workflows.index(workflow),
        ),
    )

    oracle_rows: list[Mapping[str, Any]] = []
    quality_selections: Counter[str] = Counter()
    per_question: list[dict[str, Any]] = []
    for key in ordered_keys:
        candidates = [(workflow, by_workflow[workflow][key]) for workflow in args.workflows]
        workflow, record, _ = _choose(candidates, penalty=0.0)
        quality_selections[workflow] += 1
        oracle_rows.append(record)
        per_question.append(
            {
                "question_id": key[0],
                "run_id": key[1],
                "quality_workflow": workflow,
                "quality": _quality(record),
                "action_cost": _cost(record),
            }
        )

    quality_oracle = _summary(oracle_rows)
    quality_oracle["selection_frequency"] = dict(sorted(quality_selections.items()))
    quality_oracle["gain_over_best_fixed_mean_f1"] = (
        quality_oracle["mean_question_f1"]
        - fixed[best_fixed]["mean_question_f1"]
    )

    quality_cost_oracles: dict[str, Any] = {}
    for penalty in args.lambdas:
        selections: Counter[str] = Counter()
        rewards: list[float] = []
        qualities: list[float] = []
        costs: list[float] = []
        for key in ordered_keys:
            candidates = [
                (workflow, by_workflow[workflow][key]) for workflow in args.workflows
            ]
            workflow, record, reward = _choose(candidates, penalty=penalty)
            selections[workflow] += 1
            rewards.append(reward)
            qualities.append(_quality(record))
            costs.append(_cost(record))
        quality_cost_oracles[str(penalty)] = {
            "mean_reward": sum(rewards) / len(rewards),
            "mean_quality": sum(qualities) / len(qualities),
            "mean_action_cost": sum(costs) / len(costs),
            "selection_frequency": dict(sorted(selections.items())),
        }

    result = {
        "schema_version": 1,
        "quality_field": "quality (per-question answer F1)",
        "action_cost_definition": "llm_calls + workflow_sparql_executions",
        "evaluation_sparql_executions_in_cost": False,
        "tie_breaker": "highest reward, then lowest cost, then workflow order",
        "workflows": args.workflows,
        "fixed_workflows": fixed,
        "best_fixed_workflow": best_fixed,
        "quality_oracle": quality_oracle,
        "quality_cost_oracles": quality_cost_oracles,
        "per_question": per_question,
    }

    output = (
        args.output.resolve()
        if args.output is not None
        else directory / "experiment_a-analysis.json"
    )
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}; pass --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Analysis written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
