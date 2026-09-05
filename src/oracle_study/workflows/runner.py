"""Workflow runners and structured per-question traces for Experiment A."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Iterator

from oracle_study.capabilities.sparql_generation import (
    SPARQLGenerationCapability,
    SPARQLGenerationOutput,
)
from oracle_study.datasets.qald7 import QALD7Example
from oracle_study.evaluation.answer_metrics import (
    AnswerMetrics,
    AnswerNormalizationConfig,
    evaluate_answers,
)
from oracle_study.evaluation.sparql_executor import (
    SPARQLExecutionResult,
    SPARQLExecutor,
)
from oracle_study.parsing.sparql import SPARQLParseResult, parse_sparql_output


class WorkflowRunStatus(str, Enum):
    COMPLETED = "completed"
    GENERATION_ERROR = "generation_error"
    PARSE_ERROR = "parse_error"
    EXECUTION_ERROR = "execution_error"


@dataclass(frozen=True)
class WorkflowRunError:
    stage: str
    error_type: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class WorkflowCost:
    input_tokens: int
    output_tokens: int
    llm_calls: int
    workflow_sparql_executions: int
    evaluation_sparql_executions: int
    generation_latency_seconds: float
    evaluation_latency_seconds: float
    total_latency_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkflowRunResult:
    experiment_id: str
    workflow_id: str
    question_id: str
    run_id: int
    timestamp_utc: str
    status: WorkflowRunStatus
    question: str
    gold_sparql: str
    generation: SPARQLGenerationOutput | None
    parsing: SPARQLParseResult | None
    execution: SPARQLExecutionResult | None
    metrics: AnswerMetrics | None
    cost: WorkflowCost
    error: WorkflowRunError | None = None

    @property
    def quality(self) -> float:
        """Per-question reward quality; failed stages receive zero."""

        return self.metrics.f1 if self.metrics is not None else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "workflow_id": self.workflow_id,
            "question_id": self.question_id,
            "run_id": self.run_id,
            "timestamp_utc": self.timestamp_utc,
            "status": self.status.value,
            "question": self.question,
            "gold_sparql": self.gold_sparql,
            "quality": self.quality,
            "generation": (
                self.generation.to_dict() if self.generation is not None else None
            ),
            "parsing": self.parsing.to_dict() if self.parsing is not None else None,
            "execution": (
                self.execution.to_dict() if self.execution is not None else None
            ),
            "metrics": self.metrics.to_dict() if self.metrics is not None else None,
            "cost": self.cost.to_dict(),
            "error": self.error.to_dict() if self.error is not None else None,
        }


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_error(exc: Exception, *, stage: str) -> WorkflowRunError:
    message = " ".join(str(exc).split())[:1000] or exc.__class__.__name__
    return WorkflowRunError(
        stage=stage,
        error_type=exc.__class__.__name__,
        message=message,
    )


class W1DirectRunner:
    """Run W1: question -> generation -> parse -> evaluation execution."""

    workflow_id = "W1"
    execution_stage = "evaluation_execution"

    def __init__(
        self,
        *,
        experiment_id: str,
        generation: SPARQLGenerationCapability,
        executor: SPARQLExecutor,
        answer_normalization: AnswerNormalizationConfig | None = None,
    ) -> None:
        if not experiment_id.strip():
            raise ValueError("experiment_id must be non-empty.")
        self.experiment_id = experiment_id
        self.generation = generation
        self.executor = executor
        self.answer_normalization = (
            answer_normalization or AnswerNormalizationConfig()
        )

    @staticmethod
    def _cost(
        *,
        started_at: float,
        generation: SPARQLGenerationOutput | None = None,
        execution: SPARQLExecutionResult | None = None,
    ) -> WorkflowCost:
        return WorkflowCost(
            input_tokens=(generation.generation.input_tokens if generation else 0),
            output_tokens=(generation.generation.output_tokens if generation else 0),
            llm_calls=int(generation is not None),
            workflow_sparql_executions=0,
            evaluation_sparql_executions=int(execution is not None),
            generation_latency_seconds=(
                generation.generation.latency_seconds if generation else 0.0
            ),
            evaluation_latency_seconds=(
                execution.latency_seconds if execution else 0.0
            ),
            total_latency_seconds=time.perf_counter() - started_at,
        )

    def run(self, example: QALD7Example, *, run_id: int = 0) -> WorkflowRunResult:
        if run_id < 0:
            raise ValueError("run_id must be non-negative.")
        started_at = time.perf_counter()
        timestamp = _timestamp_utc()

        try:
            # W1 deliberately supplies no linking or schema evidence.
            generation = self.generation.generate(
                question=example.question,
                linking_evidence=None,
                schema_evidence=None,
            )
        except Exception as exc:
            return WorkflowRunResult(
                experiment_id=self.experiment_id,
                workflow_id=self.workflow_id,
                question_id=example.question_id,
                run_id=run_id,
                timestamp_utc=timestamp,
                status=WorkflowRunStatus.GENERATION_ERROR,
                question=example.question,
                gold_sparql=example.gold_sparql,
                generation=None,
                parsing=None,
                execution=None,
                metrics=None,
                cost=self._cost(started_at=started_at),
                error=_safe_error(exc, stage="sparql_generation"),
            )

        parsing = parse_sparql_output(generation.raw_output)
        if not parsing.succeeded or parsing.query is None or parsing.query_form is None:
            return WorkflowRunResult(
                experiment_id=self.experiment_id,
                workflow_id=self.workflow_id,
                question_id=example.question_id,
                run_id=run_id,
                timestamp_utc=timestamp,
                status=WorkflowRunStatus.PARSE_ERROR,
                question=example.question,
                gold_sparql=example.gold_sparql,
                generation=generation,
                parsing=parsing,
                execution=None,
                metrics=None,
                cost=self._cost(started_at=started_at, generation=generation),
                error=WorkflowRunError(
                    stage="sparql_parsing",
                    error_type=parsing.status.value,
                    message=parsing.message or "SPARQL parsing failed.",
                ),
            )

        try:
            execution = self.executor.execute(
                parsing.query,
                query_form=parsing.query_form,
            )
        except Exception as exc:
            return WorkflowRunResult(
                experiment_id=self.experiment_id,
                workflow_id=self.workflow_id,
                question_id=example.question_id,
                run_id=run_id,
                timestamp_utc=timestamp,
                status=WorkflowRunStatus.EXECUTION_ERROR,
                question=example.question,
                gold_sparql=example.gold_sparql,
                generation=generation,
                parsing=parsing,
                execution=None,
                metrics=None,
                cost=self._cost(started_at=started_at, generation=generation),
                error=_safe_error(exc, stage=self.execution_stage),
            )

        metrics = evaluate_answers(
            gold_rows=example.gold_answer_rows,
            gold_boolean=example.gold_boolean,
            execution=execution,
            normalization=self.answer_normalization,
        )
        status = (
            WorkflowRunStatus.COMPLETED
            if execution.succeeded
            else WorkflowRunStatus.EXECUTION_ERROR
        )
        error = None
        if not execution.succeeded:
            error = WorkflowRunError(
                stage=self.execution_stage,
                error_type=(
                    execution.error_type.value
                    if execution.error_type is not None
                    else "unknown_error"
                ),
                message=execution.error_message or "SPARQL execution failed.",
            )

        return WorkflowRunResult(
            experiment_id=self.experiment_id,
            workflow_id=self.workflow_id,
            question_id=example.question_id,
            run_id=run_id,
            timestamp_utc=timestamp,
            status=status,
            question=example.question,
            gold_sparql=example.gold_sparql,
            generation=generation,
            parsing=parsing,
            execution=execution,
            metrics=metrics,
            cost=self._cost(
                started_at=started_at,
                generation=generation,
                execution=execution,
            ),
            error=error,
        )

    def run_many(
        self,
        examples: Iterable[QALD7Example],
        *,
        repetitions: int = 1,
    ) -> Iterator[WorkflowRunResult]:
        """Run every example and continue after recorded per-question failures."""

        if repetitions <= 0:
            raise ValueError("repetitions must be positive.")
        for example in examples:
            for run_id in range(repetitions):
                yield self.run(example, run_id=run_id)


class W4ExecuteRunner(W1DirectRunner):
    """Run W4: generate, parse, and execute once inside the workflow.

    The internal execution result is reused directly for answer evaluation;
    the evaluator must not issue a second SPARQL request.
    """

    workflow_id = "W4"
    execution_stage = "workflow_execution"

    @staticmethod
    def _cost(
        *,
        started_at: float,
        generation: SPARQLGenerationOutput | None = None,
        execution: SPARQLExecutionResult | None = None,
    ) -> WorkflowCost:
        return WorkflowCost(
            input_tokens=(generation.generation.input_tokens if generation else 0),
            output_tokens=(generation.generation.output_tokens if generation else 0),
            llm_calls=int(generation is not None),
            workflow_sparql_executions=int(execution is not None),
            evaluation_sparql_executions=0,
            generation_latency_seconds=(
                generation.generation.latency_seconds if generation else 0.0
            ),
            # Evaluation reuses W4's internal result and adds no endpoint latency.
            evaluation_latency_seconds=0.0,
            total_latency_seconds=time.perf_counter() - started_at,
        )
