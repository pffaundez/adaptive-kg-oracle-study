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
from oracle_study.capabilities.entity_relation_linking import (
    EntityRelationLinkingCapability,
    EntityRelationLinkingOutput,
)
from oracle_study.capabilities.sparql_repair import (
    SPARQLRepairCapability,
    SPARQLRepairOutput,
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
    REPAIR_ERROR = "repair_error"
    LINKING_ERROR = "linking_error"


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
    linking_calls: int
    repair_calls: int
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
    repair: SPARQLRepairOutput | None = None
    initial_execution: SPARQLExecutionResult | None = None
    linking: EntityRelationLinkingOutput | None = None

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
            "repair": self.repair.to_dict() if self.repair is not None else None,
            "initial_execution": (
                self.initial_execution.to_dict()
                if self.initial_execution is not None
                else None
            ),
            "linking": self.linking.to_dict() if self.linking is not None else None,
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
        linking: EntityRelationLinkingOutput | None = None,
    ) -> WorkflowCost:
        generations = [
            output
            for output in (
                linking.generation if linking is not None else None,
                generation.generation if generation is not None else None,
            )
            if output is not None
        ]
        return WorkflowCost(
            input_tokens=sum(output.input_tokens for output in generations),
            output_tokens=sum(output.output_tokens for output in generations),
            llm_calls=len(generations),
            linking_calls=int(linking is not None),
            repair_calls=0,
            workflow_sparql_executions=0,
            evaluation_sparql_executions=int(execution is not None),
            generation_latency_seconds=(
                sum(output.latency_seconds for output in generations)
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
            linking = self._link(example)
        except Exception as exc:
            return WorkflowRunResult(
                experiment_id=self.experiment_id,
                workflow_id=self.workflow_id,
                question_id=example.question_id,
                run_id=run_id,
                timestamp_utc=timestamp,
                status=WorkflowRunStatus.LINKING_ERROR,
                question=example.question,
                gold_sparql=example.gold_sparql,
                generation=None,
                parsing=None,
                execution=None,
                metrics=None,
                cost=self._cost(started_at=started_at),
                error=_safe_error(exc, stage="entity_relation_linking"),
            )

        try:
            generation = self.generation.generate(
                question=example.question,
                linking_evidence=(linking.evidence if linking is not None else None),
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
                cost=self._cost(started_at=started_at, linking=linking),
                error=_safe_error(exc, stage="sparql_generation"),
                linking=linking,
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
                cost=self._cost(
                    started_at=started_at,
                    generation=generation,
                    linking=linking,
                ),
                error=WorkflowRunError(
                    stage="sparql_parsing",
                    error_type=parsing.status.value,
                    message=parsing.message or "SPARQL parsing failed.",
                ),
                linking=linking,
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
                cost=self._cost(
                    started_at=started_at,
                    generation=generation,
                    linking=linking,
                ),
                error=_safe_error(exc, stage=self.execution_stage),
                linking=linking,
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
                linking=linking,
            ),
            error=error,
            linking=linking,
        )

    def _link(self, example: QALD7Example) -> EntityRelationLinkingOutput | None:
        """Return optional grounding evidence before SPARQL generation."""

        return None

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


class W2GroundedRunner(W1DirectRunner):
    """Run W2 with one entity/relation linking call before generation."""

    workflow_id = "W2"

    def __init__(
        self,
        *,
        experiment_id: str,
        linking: EntityRelationLinkingCapability,
        generation: SPARQLGenerationCapability,
        executor: SPARQLExecutor,
        answer_normalization: AnswerNormalizationConfig | None = None,
    ) -> None:
        super().__init__(
            experiment_id=experiment_id,
            generation=generation,
            executor=executor,
            answer_normalization=answer_normalization,
        )
        self.entity_relation_linking = linking

    def _link(self, example: QALD7Example) -> EntityRelationLinkingOutput:
        return self.entity_relation_linking.link(example.question)


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
        linking: EntityRelationLinkingOutput | None = None,
    ) -> WorkflowCost:
        generations = [
            output
            for output in (
                linking.generation if linking is not None else None,
                generation.generation if generation is not None else None,
            )
            if output is not None
        ]
        return WorkflowCost(
            input_tokens=sum(output.input_tokens for output in generations),
            output_tokens=sum(output.output_tokens for output in generations),
            llm_calls=len(generations),
            linking_calls=int(linking is not None),
            repair_calls=0,
            workflow_sparql_executions=int(execution is not None),
            evaluation_sparql_executions=0,
            generation_latency_seconds=(
                sum(output.latency_seconds for output in generations)
            ),
            # Evaluation reuses W4's internal result and adds no endpoint latency.
            evaluation_latency_seconds=0.0,
            total_latency_seconds=time.perf_counter() - started_at,
        )


class W5RepairRunner(W4ExecuteRunner):
    """Run W5 with one conditional execution-guided repair."""

    workflow_id = "W5"

    def __init__(
        self,
        *,
        experiment_id: str,
        generation: SPARQLGenerationCapability,
        repair: SPARQLRepairCapability,
        executor: SPARQLExecutor,
        answer_normalization: AnswerNormalizationConfig | None = None,
    ) -> None:
        super().__init__(
            experiment_id=experiment_id,
            generation=generation,
            executor=executor,
            answer_normalization=answer_normalization,
        )
        self.repair = repair

    @staticmethod
    def _combined_cost(
        *,
        started_at: float,
        initial_generation: SPARQLGenerationOutput | None,
        repair: SPARQLRepairOutput | None,
        executions: int,
    ) -> WorkflowCost:
        generations = [
            output.generation
            for output in (initial_generation, repair)
            if output is not None
        ]
        return WorkflowCost(
            input_tokens=sum(output.input_tokens for output in generations),
            output_tokens=sum(output.output_tokens for output in generations),
            llm_calls=len(generations),
            linking_calls=0,
            repair_calls=int(repair is not None),
            workflow_sparql_executions=executions,
            evaluation_sparql_executions=0,
            generation_latency_seconds=sum(
                output.latency_seconds for output in generations
            ),
            evaluation_latency_seconds=0.0,
            total_latency_seconds=time.perf_counter() - started_at,
        )

    def run(self, example: QALD7Example, *, run_id: int = 0) -> WorkflowRunResult:
        started_at = time.perf_counter()
        initial = super().run(example, run_id=run_id)

        # Generation and parsing failures occur before W5 can execute or repair.
        if initial.execution is None:
            return WorkflowRunResult(
                **{
                    **initial.__dict__,
                    "cost": self._combined_cost(
                        started_at=started_at,
                        initial_generation=initial.generation,
                        repair=None,
                        executions=0,
                    ),
                }
            )

        initial_execution = initial.execution
        execution_status = initial_execution.status.value
        should_repair = execution_status in {"execution_error", "empty_result"}
        if not should_repair:
            return WorkflowRunResult(
                **{
                    **initial.__dict__,
                    "cost": self._combined_cost(
                        started_at=started_at,
                        initial_generation=initial.generation,
                        repair=None,
                        executions=1,
                    ),
                    "initial_execution": initial_execution,
                }
            )

        assert initial.generation is not None
        assert initial.parsing is not None
        assert initial.parsing.query is not None
        try:
            repair = self.repair.repair(
                example.question,
                failed_query=initial.parsing.query,
                execution_feedback=initial_execution,
            )
        except Exception as exc:
            return WorkflowRunResult(
                **{
                    **initial.__dict__,
                    "status": WorkflowRunStatus.REPAIR_ERROR,
                    "metrics": None,
                    "cost": self._combined_cost(
                        started_at=started_at,
                        initial_generation=initial.generation,
                        repair=None,
                        executions=1,
                    ),
                    "error": _safe_error(exc, stage="sparql_repair"),
                    "initial_execution": initial_execution,
                }
            )

        repaired_parsing = parse_sparql_output(repair.raw_output)
        if (
            not repaired_parsing.succeeded
            or repaired_parsing.query is None
            or repaired_parsing.query_form is None
        ):
            return WorkflowRunResult(
                **{
                    **initial.__dict__,
                    "status": WorkflowRunStatus.PARSE_ERROR,
                    "parsing": repaired_parsing,
                    "execution": None,
                    "metrics": None,
                    "cost": self._combined_cost(
                        started_at=started_at,
                        initial_generation=initial.generation,
                        repair=repair,
                        executions=1,
                    ),
                    "error": WorkflowRunError(
                        stage="repair_parsing",
                        error_type=repaired_parsing.status.value,
                        message=(
                            repaired_parsing.message
                            or "Repaired SPARQL parsing failed."
                        ),
                    ),
                    "repair": repair,
                    "initial_execution": initial_execution,
                }
            )

        try:
            final_execution = self.executor.execute(
                repaired_parsing.query,
                query_form=repaired_parsing.query_form,
            )
        except Exception as exc:
            return WorkflowRunResult(
                **{
                    **initial.__dict__,
                    "status": WorkflowRunStatus.EXECUTION_ERROR,
                    "parsing": repaired_parsing,
                    "execution": None,
                    "metrics": None,
                    "cost": self._combined_cost(
                        started_at=started_at,
                        initial_generation=initial.generation,
                        repair=repair,
                        executions=1,
                    ),
                    "error": _safe_error(exc, stage="repair_execution"),
                    "repair": repair,
                    "initial_execution": initial_execution,
                }
            )

        metrics = evaluate_answers(
            gold_rows=example.gold_answer_rows,
            gold_boolean=example.gold_boolean,
            execution=final_execution,
            normalization=self.answer_normalization,
        )
        succeeded = final_execution.succeeded
        error = None
        if not succeeded:
            error = WorkflowRunError(
                stage="repair_execution",
                error_type=(
                    final_execution.error_type.value
                    if final_execution.error_type is not None
                    else "unknown_error"
                ),
                message=(
                    final_execution.error_message
                    or "Repaired SPARQL execution failed."
                ),
            )
        return WorkflowRunResult(
            experiment_id=self.experiment_id,
            workflow_id=self.workflow_id,
            question_id=example.question_id,
            run_id=run_id,
            timestamp_utc=initial.timestamp_utc,
            status=(
                WorkflowRunStatus.COMPLETED
                if succeeded
                else WorkflowRunStatus.EXECUTION_ERROR
            ),
            question=example.question,
            gold_sparql=example.gold_sparql,
            generation=initial.generation,
            parsing=repaired_parsing,
            execution=final_execution,
            metrics=metrics,
            cost=self._combined_cost(
                started_at=started_at,
                initial_generation=initial.generation,
                repair=repair,
                executions=2,
            ),
            error=error,
            repair=repair,
            initial_execution=initial_execution,
        )