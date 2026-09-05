"""Per-question answer metrics for QALD-style SPARQL evaluation.

Metrics compare denotations (executed answer sets), not SPARQL strings. Dataset-
level macro/micro aggregation belongs to a separate analysis module.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Iterable, Mapping

from oracle_study.evaluation.sparql_executor import (
    ExecutionStatus,
    SPARQLExecutionResult,
)


_NUMERIC_DATATYPES = {
    "http://www.w3.org/2001/XMLSchema#byte",
    "http://www.w3.org/2001/XMLSchema#decimal",
    "http://www.w3.org/2001/XMLSchema#double",
    "http://www.w3.org/2001/XMLSchema#float",
    "http://www.w3.org/2001/XMLSchema#int",
    "http://www.w3.org/2001/XMLSchema#integer",
    "http://www.w3.org/2001/XMLSchema#long",
    "http://www.w3.org/2001/XMLSchema#negativeInteger",
    "http://www.w3.org/2001/XMLSchema#nonNegativeInteger",
    "http://www.w3.org/2001/XMLSchema#nonPositiveInteger",
    "http://www.w3.org/2001/XMLSchema#positiveInteger",
    "http://www.w3.org/2001/XMLSchema#short",
    "http://www.w3.org/2001/XMLSchema#unsignedByte",
    "http://www.w3.org/2001/XMLSchema#unsignedInt",
    "http://www.w3.org/2001/XMLSchema#unsignedLong",
    "http://www.w3.org/2001/XMLSchema#unsignedShort",
}
_BOOLEAN_DATATYPE = "http://www.w3.org/2001/XMLSchema#boolean"


class AnswerKind(str, Enum):
    SELECT = "select"
    ASK = "ask"


@dataclass(frozen=True)
class AnswerNormalizationConfig:
    """Conservative RDF-term normalization applied to gold and predictions."""

    unicode_form: str = "NFC"
    lowercase_language_tags: bool = True
    canonicalize_numeric_literals: bool = False
    canonicalize_boolean_literals: bool = False
    normalize_dbpedia_https: bool = False

    def __post_init__(self) -> None:
        if self.unicode_form not in {"NFC", "NFD", "NFKC", "NFKD"}:
            raise ValueError("unicode_form must be NFC, NFD, NFKC, or NFKD.")


@dataclass(frozen=True)
class AnswerMetrics:
    """Metrics for one question."""

    answer_kind: AnswerKind
    executed: bool
    precision: float
    recall: float
    f1: float
    execution_accuracy: float
    predicted_count: int
    gold_count: int
    true_positive_count: int
    execution_status: str
    error_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["answer_kind"] = self.answer_kind.value
        return payload


CanonicalTerm = tuple[str, str, str, str]
CanonicalRow = tuple[CanonicalTerm, ...]


def _canonical_decimal(value: str) -> str:
    number = Decimal(value)
    if not number.is_finite():
        return value
    normalized = number.normalize()
    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal(1)))
    return format(normalized, "f")


def _canonical_term(
    term: Mapping[str, Any],
    config: AnswerNormalizationConfig,
) -> CanonicalTerm:
    if "value" not in term:
        raise ValueError("Every answer term must contain 'value'.")

    term_type = str(term.get("type", "literal")).lower()
    value = unicodedata.normalize(config.unicode_form, str(term["value"]))
    datatype = str(term.get("datatype", ""))
    language = str(term.get("xml:lang", term.get("lang", "")))
    if config.lowercase_language_tags:
        language = language.lower()

    if config.normalize_dbpedia_https and term_type == "uri":
        value = value.replace("https://dbpedia.org/", "http://dbpedia.org/", 1)

    if config.canonicalize_numeric_literals and datatype in _NUMERIC_DATATYPES:
        try:
            value = _canonical_decimal(value)
        except InvalidOperation:
            pass

    if config.canonicalize_boolean_literals and datatype == _BOOLEAN_DATATYPE:
        lowered = value.lower()
        if lowered in {"true", "1"}:
            value = "true"
        elif lowered in {"false", "0"}:
            value = "false"

    return term_type, value, datatype, language


def _canonical_row(
    row: Mapping[str, Mapping[str, Any]],
    config: AnswerNormalizationConfig,
) -> CanonicalRow:
    """Canonicalize a row without depending on projected variable names."""

    if not isinstance(row, Mapping):
        raise TypeError("Every answer row must be a mapping.")
    terms = [_canonical_term(term, config) for term in row.values()]
    return tuple(sorted(terms))


def canonical_answer_set(
    rows: Iterable[Mapping[str, Mapping[str, Any]]],
    *,
    normalization: AnswerNormalizationConfig | None = None,
) -> set[CanonicalRow]:
    """Convert SPARQL bindings into a duplicate-free set of answer rows."""

    config = normalization or AnswerNormalizationConfig()
    return {_canonical_row(row, config) for row in rows}


def _select_metrics(
    predicted: set[CanonicalRow],
    gold: set[CanonicalRow],
) -> tuple[float, float, float, float, int]:
    true_positives = len(predicted & gold)

    if not predicted and not gold:
        return 1.0, 1.0, 1.0, 1.0, 0
    if not predicted and gold:
        # QALD convention: an empty system answer has precision 1, but recall
        # and F1 are 0 when the gold answer is non-empty.
        return 1.0, 0.0, 0.0, 0.0, 0
    if predicted and not gold:
        return 0.0, 0.0, 0.0, 0.0, 0

    precision = true_positives / len(predicted)
    recall = true_positives / len(gold)
    f1 = (
        0.0
        if precision + recall == 0
        else 2.0 * precision * recall / (precision + recall)
    )
    exact = float(predicted == gold)
    return precision, recall, f1, exact, true_positives


def evaluate_select_answers(
    *,
    gold_rows: Iterable[Mapping[str, Mapping[str, Any]]],
    execution: SPARQLExecutionResult,
    normalization: AnswerNormalizationConfig | None = None,
) -> AnswerMetrics:
    """Evaluate one SELECT denotation against the stored QALD answer rows."""

    gold = canonical_answer_set(gold_rows, normalization=normalization)
    executed = execution.status in {
        ExecutionStatus.SUCCESS,
        ExecutionStatus.EMPTY_RESULT,
    }
    predicted = (
        canonical_answer_set(execution.rows, normalization=normalization)
        if executed
        else set()
    )

    if executed:
        precision, recall, f1, exact, true_positives = _select_metrics(
            predicted, gold
        )
    else:
        precision = recall = f1 = exact = 0.0
        true_positives = 0

    return AnswerMetrics(
        answer_kind=AnswerKind.SELECT,
        executed=executed,
        precision=precision,
        recall=recall,
        f1=f1,
        execution_accuracy=exact,
        predicted_count=len(predicted),
        gold_count=len(gold),
        true_positive_count=true_positives,
        execution_status=execution.status.value,
        error_type=(
            execution.error_type.value if execution.error_type is not None else None
        ),
    )


def evaluate_ask_answer(
    *,
    gold_boolean: bool,
    execution: SPARQLExecutionResult,
) -> AnswerMetrics:
    """Evaluate one ASK result using exact boolean agreement."""

    if not isinstance(gold_boolean, bool):
        raise TypeError("gold_boolean must be bool.")
    executed = execution.status is ExecutionStatus.SUCCESS
    correct = executed and execution.boolean is gold_boolean
    score = float(correct)
    return AnswerMetrics(
        answer_kind=AnswerKind.ASK,
        executed=executed,
        precision=score,
        recall=score,
        f1=score,
        execution_accuracy=score,
        predicted_count=int(executed and execution.boolean is not None),
        gold_count=1,
        true_positive_count=int(correct),
        execution_status=execution.status.value,
        error_type=(
            execution.error_type.value if execution.error_type is not None else None
        ),
    )


def evaluate_answers(
    *,
    gold_rows: Iterable[Mapping[str, Mapping[str, Any]]] = (),
    gold_boolean: bool | None = None,
    execution: SPARQLExecutionResult,
    normalization: AnswerNormalizationConfig | None = None,
) -> AnswerMetrics:
    """Dispatch to SELECT or ASK metrics from the gold-answer representation."""

    if gold_boolean is not None:
        return evaluate_ask_answer(
            gold_boolean=gold_boolean,
            execution=execution,
        )
    return evaluate_select_answers(
        gold_rows=gold_rows,
        execution=execution,
        normalization=normalization,
    )


def aggregate_macro(metrics: Iterable[AnswerMetrics]) -> dict[str, float | int]:
    """Aggregate per-question scores, reporting both common macro-F1 variants."""

    items = list(metrics)
    if not items:
        raise ValueError("At least one AnswerMetrics item is required.")

    macro_precision = math.fsum(item.precision for item in items) / len(items)
    macro_recall = math.fsum(item.recall for item in items) / len(items)
    mean_question_f1 = math.fsum(item.f1 for item in items) / len(items)
    qald_macro_f1 = (
        0.0
        if macro_precision + macro_recall == 0
        else 2.0
        * macro_precision
        * macro_recall
        / (macro_precision + macro_recall)
    )
    return {
        "questions": len(items),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "qald_macro_f1": qald_macro_f1,
        "mean_question_f1": mean_question_f1,
        "execution_accuracy": (
            math.fsum(item.execution_accuracy for item in items) / len(items)
        ),
        "executability": sum(item.executed for item in items) / len(items),
    }