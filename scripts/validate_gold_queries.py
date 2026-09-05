#!/usr/bin/env python3
"""Validate stored QALD-7 answers against the configured live SPARQL endpoint.

This is a dataset/endpoint diagnostic, not a workflow run. It executes each gold
query once and compares the live result with the answer stored in QALD-7 using
the same answer metrics as Experiment A.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from oracle_study.datasets.qald7 import load_qald7  # noqa: E402
from oracle_study.evaluation.answer_metrics import evaluate_answers  # noqa: E402
from oracle_study.evaluation.sparql_executor import (  # noqa: E402
    SPARQLExecutor,
    SPARQLExecutorConfig,
)
from oracle_study.parsing.sparql import parse_sparql_output  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute QALD-7 gold queries against the configured endpoint and "
            "compare live results with the answers stored in the dataset."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Experiment YAML, normally configs/experiment_a.yaml.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Override dataset.max_questions; use 10 for the pilot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSONL path. Defaults to "
            "<output.directory>/gold-query-validation.jsonl."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing validation file.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Configuration loading requires PyYAML.") from exc

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)

    if not isinstance(value, dict):
        raise ValueError("The experiment configuration must be a YAML mapping.")
    return value


def _resolve_from_repository(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _to_dict(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_dict(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"Cannot serialize value of type {type(value).__name__}")


def _attribute(example: Any, name: str) -> Any:
    if not hasattr(example, name):
        raise AttributeError(
            f"QALD7Example does not expose {name!r}; update the validator to "
            "match src/oracle_study/datasets/qald7.py."
        )
    return getattr(example, name)


def _validate_one(example: Any, executor: SPARQLExecutor) -> dict[str, Any]:
    question_id = str(_attribute(example, "question_id"))
    question = _attribute(example, "question")
    gold_sparql = _attribute(example, "gold_sparql")

    parsed = parse_sparql_output(gold_sparql)
    if not parsed.succeeded:
        return {
            "question_id": question_id,
            "question": question,
            "gold_sparql": gold_sparql,
            "status": "gold_parse_error",
            "endpoint_consistent": False,
            "parsing": _to_dict(parsed),
            "execution": None,
            "metrics": None,
        }

    execution = executor.execute(parsed.query, query_form=parsed.query_form)
    metrics = evaluate_answers(
        gold_rows=_attribute(example, "gold_answer_rows"),
        gold_boolean=_attribute(example, "gold_boolean"),
        execution=execution,
    )

    execution_dict = _to_dict(execution)
    metrics_dict = _to_dict(metrics)
    execution_status = execution_dict.get("status")
    endpoint_consistent = bool(metrics_dict.get("execution_accuracy", False))

    if execution_status == "execution_error":
        status = "execution_error"
    elif endpoint_consistent:
        status = "consistent"
    else:
        status = "answer_mismatch"

    return {
        "question_id": question_id,
        "question": question,
        "gold_sparql": gold_sparql,
        "status": status,
        "endpoint_consistent": endpoint_consistent,
        "parsing": _to_dict(parsed),
        "execution": execution_dict,
        "metrics": metrics_dict,
    }


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                    )
                )
                handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    config = _load_yaml(config_path)

    dataset_config = config.get("dataset")
    execution_config = config.get("execution")
    output_config = config.get("output")
    if not isinstance(dataset_config, dict):
        raise ValueError("Missing or invalid dataset configuration.")
    if not isinstance(execution_config, dict):
        raise ValueError("Missing or invalid execution configuration.")
    if not isinstance(output_config, dict):
        raise ValueError("Missing or invalid output configuration.")

    dataset_path_value = dataset_config.get("path")
    endpoint = dataset_config.get("endpoint")
    if not dataset_path_value:
        raise ValueError("dataset.path is required.")
    if not endpoint:
        raise ValueError("dataset.endpoint is required.")

    configured_limit = dataset_config.get("max_questions")
    max_questions = args.max_questions if args.max_questions is not None else configured_limit
    if max_questions is not None and (not isinstance(max_questions, int) or max_questions < 1):
        raise ValueError("max_questions must be a positive integer or null.")

    examples = load_qald7(
        _resolve_from_repository(dataset_path_value),
        language=str(dataset_config.get("language", "en")),
        max_questions=max_questions,
        strict=bool(dataset_config.get("strict", True)),
    )

    executor = SPARQLExecutor(
        SPARQLExecutorConfig.from_mapping(execution_config, endpoint=str(endpoint))
    )

    if args.output is not None:
        output_path = args.output.resolve()
    else:
        output_directory = output_config.get("directory", "results/experiment_a")
        output_path = _resolve_from_repository(output_directory) / "gold-query-validation.jsonl"

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Validation output already exists: {output_path}. "
            "Pass --overwrite to replace it explicitly."
        )

    print(
        json.dumps(
            {
                "dataset": str(_resolve_from_repository(dataset_path_value)),
                "endpoint": str(endpoint),
                "questions": len(examples),
                "output": str(output_path),
            },
            indent=2,
        )
    )

    records: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    for index, example in enumerate(examples, start=1):
        try:
            record = _validate_one(example, executor)
        except Exception as exc:  # keep the diagnostic running across examples
            record = {
                "question_id": str(getattr(example, "question_id", "unknown")),
                "question": getattr(example, "question", None),
                "gold_sparql": getattr(example, "gold_sparql", None),
                "status": "validator_error",
                "endpoint_consistent": False,
                "error": {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                "parsing": None,
                "execution": None,
                "metrics": None,
            }

        records.append(record)
        statuses[record["status"]] += 1
        print(
            f"[{index}/{len(examples)}] question={record['question_id']} "
            f"status={record['status']}"
        )

    _write_jsonl(output_path, records)
    consistent = statuses.get("consistent", 0)
    summary = {
        "records": len(records),
        "statuses": dict(statuses),
        "endpoint_consistency_rate": consistent / len(records) if records else 0.0,
        "output": str(output_path),
    }
    print(json.dumps(summary, indent=2))
    return 0 if not statuses.get("validator_error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
