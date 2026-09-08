#!/usr/bin/env python3
"""Rescore existing Experiment A runs against live executions of gold queries.

This script never calls an LLM or a SPARQL endpoint. It joins existing workflow
results with ``gold-query-validation.jsonl`` and reuses the project's answer
metric implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from oracle_study.evaluation.answer_metrics import evaluate_answers  # noqa: E402
from oracle_study.evaluation.sparql_executor import ExecutionStatus  # noqa: E402


WORKFLOWS = ("W1", "W2", "W3", "W4", "W5", "W6")


class RescoreError(ValueError):
    """Raised when input records cannot be joined safely."""


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rescore W1-W6 against gold queries executed on the current endpoint."
    )
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=Path("results/experiment_a"),
    )
    parser.add_argument(
        "--gold-validation",
        type=Path,
        default=Path("results/experiment_a/gold-query-validation.jsonl"),
    )
    parser.add_argument("--workflows", nargs="+", default=list(WORKFLOWS))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/experiment_a/experiment_a-current-endpoint-analysis.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RescoreError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise RescoreError(f"Expected object at {path}:{line_number}")
            records.append(value)
    return records


def _to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, dict):
        raise RescoreError("Answer metric result must be a mapping.")
    return value


def _execution_object(value: Mapping[str, Any]) -> SimpleNamespace:
    """Rebuild the attribute interface expected by ``evaluate_answers``."""
    status_value = str(value.get("status", "execution_error"))
    try:
        status = ExecutionStatus(status_value)
    except ValueError as exc:
        raise RescoreError(f"Unsupported execution status: {status_value}") from exc

    error_type = value.get("error_type")
    return SimpleNamespace(
        status=status,
        rows=value.get("rows") or [],
        boolean=value.get("boolean"),
        variables=value.get("variables") or [],
        error_type=(
            SimpleNamespace(value=str(error_type)) if error_type is not None else None
        ),
        error_message=value.get("error_message"),
        query_form=value.get("query_form"),
    )


def _gold_eligibility(record: Mapping[str, Any]) -> tuple[bool, str | None]:
    """Return whether a live gold result is informative and comparable."""
    execution = record.get("execution")
    if not isinstance(execution, Mapping):
        return False, "missing_gold_execution"
    status = execution.get("status")
    if status == "execution_error":
        return False, "gold_execution_error"
    if status not in {"success", "empty_result"}:
        return False, f"unsupported_gold_status:{status}"

    query_form = str(execution.get("query_form", "")).upper()
    if query_form == "SELECT" and status == "empty_result":
        return False, "empty_current_gold_answer"
    if query_form not in {"SELECT", "ASK"}:
        return False, f"unsupported_gold_form:{query_form or 'missing'}"
    return True, None


def _load_gold(path: Path) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    indexed: dict[str, dict[str, Any]] = {}
    exclusions: Counter[str] = Counter()
    for record in _read_jsonl(path):
        question_id = str(record.get("question_id"))
        if question_id == "None":
            raise RescoreError("Gold validation record missing question_id.")
        if question_id in indexed:
            raise RescoreError(f"Duplicate gold question_id: {question_id}")
        eligible, reason = _gold_eligibility(record)
        record["_eligible_for_rescoring"] = eligible
        record["_exclusion_reason"] = reason
        indexed[question_id] = record
        if reason:
            exclusions[reason] += 1
    return indexed, exclusions


def _rescore_workflow(
    workflow: str,
    path: Path,
    gold: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    exact: list[float] = []
    executable: list[float] = []

    seen: set[tuple[str, int]] = set()
    for record in _read_jsonl(path):
        if record.get("workflow_id") != workflow:
            raise RescoreError(f"Unexpected workflow_id in {path}")
        question_id = str(record.get("question_id"))
        run_id = record.get("run_id")
        if not isinstance(run_id, int):
            raise RescoreError(f"Invalid run_id for {workflow}/{question_id}")
        key = (question_id, run_id)
        if key in seen:
            raise RescoreError(f"Duplicate workflow key: {workflow}/{key}")
        seen.add(key)

        gold_record = gold.get(question_id)
        if gold_record is None:
            raise RescoreError(f"No gold validation for question {question_id}")
        if not gold_record["_eligible_for_rescoring"]:
            continue

        gold_execution = gold_record["execution"]
        prediction = record.get("execution")
        if not isinstance(prediction, Mapping):
            # evaluate_answers expects an execution-like object. Workflow errors
            # have no result and therefore receive zero on all answer metrics.
            metrics = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "execution_accuracy": 0.0,
                "executed": False,
            }
        else:
            metrics = _to_dict(
                evaluate_answers(
                    gold_rows=gold_execution.get("rows"),
                    gold_boolean=gold_execution.get("boolean"),
                    execution=_execution_object(prediction),
                )
            )

        precision = float(metrics.get("precision", 0.0))
        recall = float(metrics.get("recall", 0.0))
        f1 = float(metrics.get("f1", 0.0))
        execution_accuracy = float(metrics.get("execution_accuracy", 0.0))
        executed = float(bool(metrics.get("executed", False)))
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        exact.append(execution_accuracy)
        executable.append(executed)
        rows.append(
            {
                "workflow_id": workflow,
                "question_id": question_id,
                "run_id": run_id,
                "historical_f1": float(record.get("quality", 0.0)),
                "current_endpoint_metrics": metrics,
            }
        )

    if not f1s:
        raise RescoreError(f"No eligible records for {workflow}")
    macro_precision = sum(precisions) / len(precisions)
    macro_recall = sum(recalls) / len(recalls)
    denominator = macro_precision + macro_recall
    summary = {
        "eligible_records": len(f1s),
        "mean_question_f1": sum(f1s) / len(f1s),
        "macro_f1": sum(f1s) / len(f1s),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "f1_from_macro_precision_recall": (
            2.0 * macro_precision * macro_recall / denominator
            if denominator
            else 0.0
        ),
        "exact_answer_accuracy": sum(exact) / len(exact),
        "executability": sum(executable) / len(executable),
    }
    return summary, rows


def _quality_oracle(
    workflows: list[str],
    fixed: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select the highest-current-F1 workflow independently per question."""
    by_key: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["question_id"]), int(row["run_id"]))
        by_key.setdefault(key, {})[str(row["workflow_id"])] = row

    selections: Counter[str] = Counter()
    selected_f1s: list[float] = []
    per_question: list[dict[str, Any]] = []
    for key in sorted(by_key, key=lambda item: (int(item[0]), item[1])):
        candidates = by_key[key]
        missing = [workflow for workflow in workflows if workflow not in candidates]
        if missing:
            raise RescoreError(f"Missing oracle candidates for {key}: {missing}")

        # max() preserves the first workflow on equal F1 because candidates are
        # visited in the explicit workflow order.
        selected_workflow = max(
            workflows,
            key=lambda workflow: float(
                candidates[workflow]["current_endpoint_metrics"].get("f1", 0.0)
            ),
        )
        selected_f1 = float(
            candidates[selected_workflow]["current_endpoint_metrics"].get("f1", 0.0)
        )
        selections[selected_workflow] += 1
        selected_f1s.append(selected_f1)
        per_question.append(
            {
                "question_id": key[0],
                "run_id": key[1],
                "selected_workflow": selected_workflow,
                "f1": selected_f1,
            }
        )

    best_fixed = max(
        workflows,
        key=lambda workflow: float(fixed[workflow]["mean_question_f1"]),
    )
    oracle_f1 = sum(selected_f1s) / len(selected_f1s)
    best_fixed_f1 = float(fixed[best_fixed]["mean_question_f1"])
    return {
        "eligible_records": len(selected_f1s),
        "mean_question_f1": oracle_f1,
        "best_fixed_workflow": best_fixed,
        "best_fixed_mean_question_f1": best_fixed_f1,
        "gain_over_best_fixed_mean_f1": oracle_f1 - best_fixed_f1,
        "selection_frequency": dict(sorted(selections.items())),
        "questions_with_positive_f1": sum(value > 0.0 for value in selected_f1s),
        "per_question": per_question,
    }


def main() -> int:
    args = _arguments()
    results_directory = args.results_directory.resolve()
    gold_path = args.gold_validation.resolve()
    output_path = args.output.resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}; pass --overwrite")

    gold, exclusions = _load_gold(gold_path)
    fixed: dict[str, Any] = {}
    per_question: list[dict[str, Any]] = []
    for workflow in args.workflows:
        summary, rows = _rescore_workflow(
            workflow,
            results_directory / f"{workflow}-results.jsonl",
            gold,
        )
        fixed[workflow] = summary
        per_question.extend(rows)

    oracle = _quality_oracle(args.workflows, fixed, per_question)

    payload = {
        "protocol": "current_endpoint_gold_execution",
        "gold_validation": str(gold_path),
        "gold_records": len(gold),
        "eligible_gold_records": sum(
            bool(record["_eligible_for_rescoring"]) for record in gold.values()
        ),
        "excluded_gold_records": sum(exclusions.values()),
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "fixed_workflows": fixed,
        "quality_oracle": oracle,
        "per_question": per_question,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    printable = {key: value for key, value in payload.items() if key != "per_question"}
    printable["quality_oracle"] = {
        key: value
        for key, value in oracle.items()
        if key != "per_question"
    }
    print(json.dumps(printable, indent=2))
    print(json.dumps({"output": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())