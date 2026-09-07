#!/usr/bin/env python3
"""Run the currently supported workflow from an experiment YAML file."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from oracle_study.capabilities.sparql_generation import (  # noqa: E402
    SPARQLGenerationCapability,
    SPARQLGenerationConfig,
)
from oracle_study.capabilities.entity_relation_linking import (  # noqa: E402
    EntityRelationLinkingCapability,
    EntityRelationLinkingConfig,
)
from oracle_study.capabilities.sparql_repair import (  # noqa: E402
    SPARQLRepairCapability,
    SPARQLRepairConfig,
)
from oracle_study.datasets.qald7 import load_qald7  # noqa: E402
from oracle_study.evaluation.answer_metrics import (  # noqa: E402
    AnswerNormalizationConfig,
)
from oracle_study.evaluation.sparql_executor import (  # noqa: E402
    SPARQLExecutor,
    SPARQLExecutorConfig,
)
from oracle_study.logging.result_writer import (  # noqa: E402
    JSONLResultWriter,
    ResultKey,
)
from oracle_study.models.huggingface_model import (  # noqa: E402
    HuggingFaceChatModel,
    HuggingFaceModelConfig,
)
from oracle_study.workflows.runner import (  # noqa: E402
    W1DirectRunner,
    W2GroundedRunner,
    W4ExecuteRunner,
    W5RepairRunner,
)


_CUDA_DEVICES = re.compile(r"\d+(?:,\d+)*")


class ExperimentConfigurationError(ValueError):
    """Raised when experiment_a.yaml is incomplete or inconsistent."""


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentConfigurationError(f"{label} must be a YAML mapping.")
    return value


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Configuration loading requires PyYAML.") from exc

    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            payload = yaml.safe_load(stream)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Experiment configuration not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ExperimentConfigurationError(f"Invalid YAML in {path}: {exc}") from exc
    return _mapping(payload, label="experiment configuration")


def _path_from_root(project_root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentConfigurationError(f"{label} must be a non-empty path.")
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _configure_cuda(value: Any) -> str:
    if not isinstance(value, str) or not _CUDA_DEVICES.fullmatch(value.strip()):
        raise ExperimentConfigurationError(
            "model.loading.cuda_visible_devices must look like '0,1'."
        )
    normalized = value.strip()
    if len(set(normalized.split(","))) != len(normalized.split(",")):
        raise ExperimentConfigurationError("CUDA device IDs must not be duplicated.")
    # Must happen before HuggingFaceChatModel imports torch in its lazy load().
    os.environ["CUDA_VISIBLE_DEVICES"] = normalized
    return normalized


def _normalization_config(data: Mapping[str, Any]) -> AnswerNormalizationConfig:
    raw = data.get("answer_normalization") or {}
    raw = _mapping(raw, label="evaluation.answer_normalization")
    return AnswerNormalizationConfig(
        unicode_form=str(raw.get("unicode_form", "NFC")),
        lowercase_language_tags=bool(raw.get("lowercase_language_tags", True)),
        canonicalize_numeric_literals=bool(
            raw.get("canonicalize_numeric_literals", False)
        ),
        canonicalize_boolean_literals=bool(
            raw.get("canonicalize_boolean_literals", False)
        ),
        normalize_dbpedia_https=bool(raw.get("normalize_dbpedia_https", False)),
    )


def _result_path(
    project_root: Path,
    output: Mapping[str, Any],
    *,
    workflow_id: str,
) -> Path:
    directory = _path_from_root(
        project_root,
        output.get("directory"),
        label="output.directory",
    )
    template = output.get("result_file_template", "{workflow_id}-results.jsonl")
    if not isinstance(template, str) or not template.strip():
        raise ExperimentConfigurationError(
            "output.result_file_template must be a non-empty string."
        )
    try:
        filename = template.format(workflow_id=workflow_id)
    except (KeyError, ValueError) as exc:
        raise ExperimentConfigurationError(
            "result_file_template may only require the {workflow_id} field."
        ) from exc
    if Path(filename).name != filename:
        raise ExperimentConfigurationError(
            "result_file_template must produce a filename, not a nested path."
        )
    return directory / filename


def _select_workflow(workflows: Mapping[str, Any]) -> tuple[str, type]:
    enabled = workflows.get("enabled")
    if not isinstance(enabled, list) or len(enabled) != 1:
        raise ExperimentConfigurationError(
            "Enable exactly one workflow while implementations are tested."
        )
    supported = {
        "W1-direct.yaml": ("W1", W1DirectRunner),
        "W2-grounded.yaml": ("W2", W2GroundedRunner),
        "W4-execute.yaml": ("W4", W4ExecuteRunner),
        "W5-repair.yaml": ("W5", W5RepairRunner),
    }
    filename = enabled[0]
    if filename not in supported:
        raise ExperimentConfigurationError(
            "Supported workflows are W1-direct.yaml, W2-grounded.yaml, "
            "W4-execute.yaml, and W5-repair.yaml."
        )
    return supported[filename]


def _read_summary(path: Path) -> dict[str, Any]:
    statuses: Counter[str] = Counter()
    qualities: list[float] = []
    if not path.exists():
        return {"records": 0, "statuses": {}, "mean_question_f1": 0.0}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            statuses[str(record.get("status", "unknown"))] += 1
            qualities.append(float(record.get("quality", 0.0)))
    return {
        "records": len(qualities),
        "statuses": dict(sorted(statuses.items())),
        "mean_question_f1": (
            sum(qualities) / len(qualities) if qualities else 0.0
        ),
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a supported Experiment A workflow.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "experiment_a.yaml",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="Base directory for relative paths in the YAML file.",
    )
    parser.add_argument(
        "--cuda-devices",
        default=None,
        help="Override model.loading.cuda_visible_devices, for example 0,1.",
    )
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _argument_parser().parse_args()
    project_root = args.project_root.resolve()
    config = _load_yaml(args.config.resolve())

    if config.get("schema_version") != 1:
        raise ExperimentConfigurationError("schema_version must be 1.")

    experiment = _mapping(config.get("experiment"), label="experiment")
    dataset = _mapping(config.get("dataset"), label="dataset")
    model_data = _mapping(config.get("model"), label="model")
    loading = _mapping(model_data.get("loading"), label="model.loading")
    capabilities = _mapping(config.get("capabilities"), label="capabilities")
    generation_data = _mapping(
        capabilities.get("sparql_generation"),
        label="capabilities.sparql_generation",
    )
    workflows = _mapping(config.get("workflows"), label="workflows")
    execution_data = _mapping(config.get("execution"), label="execution")
    evaluation_data = _mapping(config.get("evaluation"), label="evaluation")
    output_data = _mapping(config.get("output"), label="output")
    workflow_id, runner_class = _select_workflow(workflows)
    linking_data = None
    if workflow_id == "W2":
        linking_data = _mapping(
            capabilities.get("entity_relation_linking"),
            label="capabilities.entity_relation_linking",
        )
    repair_data = None
    if workflow_id == "W5":
        repair_data = _mapping(
            capabilities.get("sparql_repair"),
            label="capabilities.sparql_repair",
        )

    experiment_id = experiment.get("id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ExperimentConfigurationError("experiment.id must be non-empty.")

    configured_devices = args.cuda_devices or loading.get("cuda_visible_devices")
    visible_devices = _configure_cuda(configured_devices)

    dataset_path = _path_from_root(
        project_root,
        dataset.get("path"),
        label="dataset.path",
    )
    max_questions = (
        args.max_questions
        if args.max_questions is not None
        else dataset.get("max_questions")
    )
    if max_questions is not None:
        max_questions = int(max_questions)

    examples = load_qald7(
        dataset_path,
        language=str(dataset.get("language", "en")),
        max_questions=max_questions,
        strict=bool(dataset.get("strict", True)),
    )
    if not examples:
        raise ExperimentConfigurationError("The configured dataset selected no questions.")

    model_config = HuggingFaceModelConfig.from_mapping(model_data)
    generation_config = SPARQLGenerationConfig.from_mapping(
        generation_data,
        project_root=project_root,
    )
    endpoint = dataset.get("endpoint")
    executor_config = SPARQLExecutorConfig.from_mapping(
        execution_data,
        endpoint=str(endpoint) if endpoint is not None else None,
    )
    normalization = _normalization_config(evaluation_data)
    result_path = _result_path(
        project_root,
        output_data,
        workflow_id=workflow_id,
    )

    model = HuggingFaceChatModel(model_config)
    generation = SPARQLGenerationCapability(model, generation_config)
    # Validate all prompt placeholders without loading model weights.
    generation.build_messages(
        examples[0].question,
        linking_evidence=None,
        schema_evidence=None,
    )
    linking = None
    if linking_data is not None:
        linking_config = EntityRelationLinkingConfig.from_mapping(
            linking_data,
            project_root=project_root,
        )
        linking = EntityRelationLinkingCapability(model, linking_config)
        linking.build_messages(examples[0].question)
    repair = None
    if repair_data is not None:
        repair_config = SPARQLRepairConfig.from_mapping(
            repair_data,
            project_root=project_root,
        )
        repair = SPARQLRepairCapability(model, repair_config)
        repair.build_messages(
            examples[0].question,
            failed_query="SELECT * WHERE { ?s ?p ?o }",
            execution_feedback={
                "status": "empty_result",
                "query_form": "SELECT",
                "rows": [],
                "variables": ["s", "p", "o"],
            },
        )

    preview = {
        "experiment_id": experiment_id,
        "workflow_id": workflow_id,
        "dataset": str(dataset_path),
        "questions": len(examples),
        "model_id": model_config.model_id,
        "requested_revision": model_config.revision,
        "cuda_visible_devices": visible_devices,
        "result_path": str(result_path),
    }
    print(json.dumps(preview, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("Dry run completed; model weights were not loaded.")
        return 0

    runner_kwargs = {
        "experiment_id": experiment_id,
        "generation": generation,
        "executor": SPARQLExecutor(executor_config),
        "answer_normalization": normalization,
    }
    if repair is not None:
        runner_kwargs["repair"] = repair
    if linking is not None:
        runner_kwargs["linking"] = linking
    runner = runner_class(**runner_kwargs)

    repetitions = int(model_data.get("repetitions", 1))
    if repetitions <= 0:
        raise ExperimentConfigurationError("model.repetitions must be positive.")

    output_data_duplicate_policy = str(
        output_data.get("duplicate_policy", "skip")
    )
    processed = skipped = 0
    try:
        with JSONLResultWriter(
            result_path,
            duplicate_policy=output_data_duplicate_policy,
            flush_every=int(output_data.get("flush_every", 1)),
            fsync=bool(output_data.get("fsync", False)),
        ) as writer:
            total = len(examples) * repetitions
            for example in examples:
                for run_id in range(repetitions):
                    key = ResultKey(
                        experiment_id,
                        workflow_id,
                        example.question_id,
                        run_id,
                    )
                    if writer.contains(key) and output_data_duplicate_policy == "skip":
                        skipped += 1
                        continue
                    result = runner.run(example, run_id=run_id)
                    writer.write(result)
                    processed += 1
                    print(
                        f"[{processed + skipped}/{total}] "
                        f"question={example.question_id} run={run_id} "
                        f"status={result.status.value} f1={result.quality:.4f}",
                        flush=True,
                    )
    finally:
        model.unload()

    summary = _read_summary(result_path)
    summary["processed_this_invocation"] = processed
    summary["skipped_existing"] = skipped
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user; completed JSONL records remain resumable.", file=sys.stderr)
        raise SystemExit(130)
