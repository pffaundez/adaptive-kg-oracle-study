"""Loader and normalizer for the QALD-7 JSON datasets.

The loader intentionally does not execute gold SPARQL queries. It preserves the
answers shipped with QALD-7 so that dataset loading and endpoint-drift checks
remain separate operations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


class QALD7FormatError(ValueError):
    """Raised when a QALD-7 file is missing required information."""


@dataclass(frozen=True)
class QALD7Example:
    """Normalized representation of one QALD-7 question."""

    question_id: str
    question: str
    language: str
    gold_sparql: str
    answer_type: str | None
    gold_answer_rows: tuple[dict[str, dict[str, Any]], ...] = field(
        default_factory=tuple
    )
    gold_boolean: bool | None = None
    variables: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            payload = json.load(stream)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"QALD-7 file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise QALD7FormatError(
            f"Invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise QALD7FormatError("The QALD-7 root value must be a JSON object.")
    return payload


def _select_question_text(
    translations: Any,
    *,
    language: str,
    question_id: str,
    strict: bool,
) -> str | None:
    if not isinstance(translations, Sequence) or isinstance(translations, str):
        if strict:
            raise QALD7FormatError(
                f"Question {question_id!r} has no valid 'question' list."
            )
        return None

    for translation in translations:
        if not isinstance(translation, Mapping):
            continue
        if translation.get("language") != language:
            continue
        text = translation.get("string")
        if isinstance(text, str) and text.strip():
            return text.strip()

    if strict:
        raise QALD7FormatError(
            f"Question {question_id!r} has no non-empty {language!r} translation."
        )
    return None


def _extract_sparql(
    query: Any,
    *,
    question_id: str,
    strict: bool,
) -> str | None:
    sparql = query.get("sparql") if isinstance(query, Mapping) else None
    if isinstance(sparql, str) and sparql.strip():
        return sparql.strip()
    if strict:
        raise QALD7FormatError(
            f"Question {question_id!r} has no non-empty 'query.sparql'."
        )
    return None


def _normalize_term(term: Any) -> dict[str, Any]:
    """Preserve a SPARQL JSON result term without adding inferred fields."""

    if not isinstance(term, Mapping):
        raise QALD7FormatError("A gold answer binding must be a JSON object.")

    normalized: dict[str, Any] = {}
    for key in ("type", "value", "datatype", "xml:lang", "lang"):
        if key in term:
            normalized[key] = term[key]

    if "value" not in normalized:
        raise QALD7FormatError("A gold answer binding is missing 'value'.")
    return normalized


def _extract_answers(
    answers: Any,
    *,
    question_id: str,
    strict: bool,
) -> tuple[tuple[dict[str, dict[str, Any]], ...], bool | None, tuple[str, ...]]:
    if answers is None:
        return (), None, ()
    if not isinstance(answers, Sequence) or isinstance(answers, (str, bytes)):
        if strict:
            raise QALD7FormatError(
                f"Question {question_id!r} has an invalid 'answers' field."
            )
        return (), None, ()

    rows: list[dict[str, dict[str, Any]]] = []
    variables: list[str] = []
    boolean: bool | None = None

    for answer_block in answers:
        if not isinstance(answer_block, Mapping):
            if strict:
                raise QALD7FormatError(
                    f"Question {question_id!r} contains an invalid answer block."
                )
            continue

        if isinstance(answer_block.get("boolean"), bool):
            block_boolean = answer_block["boolean"]
            if boolean is not None and boolean != block_boolean and strict:
                raise QALD7FormatError(
                    f"Question {question_id!r} has conflicting boolean answers."
                )
            boolean = block_boolean

        head = answer_block.get("head")
        if isinstance(head, Mapping):
            head_vars = head.get("vars", [])
            if isinstance(head_vars, Sequence) and not isinstance(head_vars, str):
                for variable in head_vars:
                    if isinstance(variable, str) and variable not in variables:
                        variables.append(variable)

        results = answer_block.get("results")
        bindings = results.get("bindings", []) if isinstance(results, Mapping) else []
        if not isinstance(bindings, Sequence) or isinstance(bindings, (str, bytes)):
            if strict:
                raise QALD7FormatError(
                    f"Question {question_id!r} has invalid result bindings."
                )
            continue

        for binding in bindings:
            if not isinstance(binding, Mapping):
                if strict:
                    raise QALD7FormatError(
                        f"Question {question_id!r} contains an invalid result row."
                    )
                continue
            row: dict[str, dict[str, Any]] = {}
            for variable, term in binding.items():
                try:
                    row[str(variable)] = _normalize_term(term)
                except QALD7FormatError:
                    if strict:
                        raise
            rows.append(row)

    return tuple(rows), boolean, tuple(variables)


def _normalize_question(
    raw: Mapping[str, Any],
    *,
    language: str,
    strict: bool,
) -> QALD7Example | None:
    raw_id = raw.get("id")
    if raw_id is None or str(raw_id).strip() == "":
        if strict:
            raise QALD7FormatError("A QALD-7 question is missing its 'id'.")
        return None
    question_id = str(raw_id)

    text = _select_question_text(
        raw.get("question"),
        language=language,
        question_id=question_id,
        strict=strict,
    )
    sparql = _extract_sparql(
        raw.get("query"), question_id=question_id, strict=strict
    )
    if text is None or sparql is None:
        return None

    rows, boolean, variables = _extract_answers(
        raw.get("answers"), question_id=question_id, strict=strict
    )

    known_fields = {"id", "question", "query", "answers", "answertype"}
    metadata = {key: value for key, value in raw.items() if key not in known_fields}

    return QALD7Example(
        question_id=question_id,
        question=text,
        language=language,
        gold_sparql=sparql,
        answer_type=(
            str(raw["answertype"]) if raw.get("answertype") is not None else None
        ),
        gold_answer_rows=rows,
        gold_boolean=boolean,
        variables=variables,
        metadata=metadata,
    )


def _load_qald7_original(
    path: str | Path,
    *,
    language: str = "en",
    max_questions: int | None = None,
    strict: bool = True,
) -> list[QALD7Example]:
    """Load and normalize QALD-7 examples from ``path``.

    Args:
        path: QALD-7 JSON file.
        language: Translation selected from each question.
        max_questions: Optional deterministic prefix used for smoke tests.
        strict: Raise on malformed or missing required fields. When false,
            malformed questions are skipped where possible.
    """

    if max_questions is not None and max_questions < 0:
        raise ValueError("max_questions must be non-negative or None.")
    if not language.strip():
        raise ValueError("language must be a non-empty language code.")

    source = Path(path)
    payload = _read_json(source)
    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, Sequence) or isinstance(
        raw_questions, (str, bytes)
    ):
        raise QALD7FormatError("The QALD-7 root must contain a 'questions' list.")

    examples: list[QALD7Example] = []
    seen_ids: set[str] = set()
    for raw in raw_questions:
        if max_questions is not None and len(examples) >= max_questions:
            break
        if not isinstance(raw, Mapping):
            if strict:
                raise QALD7FormatError("Every item in 'questions' must be an object.")
            continue

        example = _normalize_question(raw, language=language, strict=strict)
        if example is None:
            continue
        if example.question_id in seen_ids:
            raise QALD7FormatError(
                f"Duplicate QALD-7 question id: {example.question_id!r}."
            )
        seen_ids.add(example.question_id)
        examples.append(example)

    return examples


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rebased_gold(
    path: Path,
    *,
    expected_sha256: str | None,
    expected_endpoint: str | None,
) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Rebased gold file not found: {path}")
    if expected_sha256 is not None:
        actual = _sha256(path)
        if actual.lower() != expected_sha256.lower():
            raise QALD7FormatError(
                f"Rebased gold SHA-256 mismatch for {path}: "
                f"expected {expected_sha256}, got {actual}."
            )

    records: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise QALD7FormatError(
                    f"Invalid rebased gold JSON at {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, Mapping):
                raise QALD7FormatError(
                    f"Rebased gold record {line_number} must be an object."
                )
            question_id = str(record.get("question_id", "")).strip()
            if not question_id:
                raise QALD7FormatError(
                    f"Rebased gold record {line_number} has no question_id."
                )
            if question_id in seen_ids:
                raise QALD7FormatError(
                    f"Duplicate rebased gold question id: {question_id!r}."
                )
            if expected_endpoint is not None and record.get("endpoint") != expected_endpoint:
                raise QALD7FormatError(
                    f"Endpoint mismatch for rebased question {question_id!r}: "
                    f"expected {expected_endpoint!r}, got {record.get('endpoint')!r}."
                )
            seen_ids.add(question_id)
            records.append(record)
    return records


def _apply_rebased_gold(
    originals: Sequence[QALD7Example],
    records: Sequence[Mapping[str, Any]],
    *,
    eligible_only: bool,
    strict: bool,
) -> list[QALD7Example]:
    by_id = {example.question_id: example for example in originals}
    rebased: list[QALD7Example] = []

    for record in records:
        question_id = str(record["question_id"])
        original = by_id.get(question_id)
        if original is None:
            if strict:
                raise QALD7FormatError(
                    f"Rebased gold references unknown QALD-7 id {question_id!r}."
                )
            continue
        if record.get("question") != original.question:
            raise QALD7FormatError(
                f"Question text mismatch in rebased gold for id {question_id!r}."
            )
        if record.get("gold_sparql") != original.gold_sparql:
            raise QALD7FormatError(
                f"Gold SPARQL mismatch in rebased gold for id {question_id!r}."
            )

        eligible = record.get("eligible") is True
        if not eligible:
            if eligible_only:
                continue
            # Ineligible records retain their historical answers and carry the
            # exclusion metadata; experiment entrypoints should normally filter them.
            metadata = dict(original.metadata)
            metadata["rebased_gold"] = {
                "eligible": False,
                "exclusion_reason": record.get("exclusion_reason"),
                "validated_at_utc": record.get("validated_at_utc"),
            }
            rebased.append(replace(original, metadata=metadata))
            continue

        answer = record.get("current_answer")
        if not isinstance(answer, Mapping):
            raise QALD7FormatError(
                f"Eligible rebased question {question_id!r} has no current_answer."
            )
        answer_kind = str(answer.get("answer_kind", "")).lower()
        raw_rows = answer.get("rows") or []
        raw_variables = answer.get("variables") or []
        if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
            raise QALD7FormatError(
                f"Invalid rebased rows for question {question_id!r}."
            )
        rows: list[dict[str, dict[str, Any]]] = []
        for raw_row in raw_rows:
            if not isinstance(raw_row, Mapping):
                raise QALD7FormatError(
                    f"Invalid rebased result row for question {question_id!r}."
                )
            rows.append(
                {str(variable): _normalize_term(term) for variable, term in raw_row.items()}
            )

        boolean = answer.get("boolean")
        if answer_kind == "ask":
            if not isinstance(boolean, bool):
                raise QALD7FormatError(
                    f"Rebased ASK question {question_id!r} has no boolean answer."
                )
            rows = []
            variables: tuple[str, ...] = ()
        elif answer_kind == "select":
            boolean = None
            if not isinstance(raw_variables, Sequence) or isinstance(
                raw_variables, (str, bytes)
            ):
                raise QALD7FormatError(
                    f"Invalid rebased variables for question {question_id!r}."
                )
            variables = tuple(str(value) for value in raw_variables)
        else:
            raise QALD7FormatError(
                f"Unsupported rebased answer kind {answer_kind!r} "
                f"for question {question_id!r}."
            )

        metadata = dict(original.metadata)
        metadata["rebased_gold"] = {
            "eligible": True,
            "gold_source": record.get("gold_source"),
            "endpoint": record.get("endpoint"),
            "validated_at_utc": record.get("validated_at_utc"),
            "validation_status": record.get("validation_status"),
        }
        rebased.append(
            replace(
                original,
                gold_answer_rows=tuple(rows),
                gold_boolean=boolean,
                variables=variables,
                metadata=metadata,
            )
        )
    return rebased


def load_qald7(
    path: str | Path,
    *,
    language: str = "en",
    max_questions: int | None = None,
    strict: bool = True,
    gold_snapshot_path: str | Path | None = None,
    gold_snapshot_sha256: str | None = None,
    gold_snapshot_endpoint: str | None = None,
    eligible_only: bool = True,
) -> list[QALD7Example]:
    """Load QALD-7, optionally replacing historical answers with frozen live gold.

    When ``gold_snapshot_path`` is provided, the snapshot controls the question
    subset and ordering. ``max_questions`` is applied after eligibility filtering.
    """

    if max_questions is not None and max_questions < 0:
        raise ValueError("max_questions must be non-negative or None.")
    originals = _load_qald7_original(
        path,
        language=language,
        max_questions=None if gold_snapshot_path is not None else max_questions,
        strict=strict,
    )
    if gold_snapshot_path is None:
        return originals

    records = _read_rebased_gold(
        Path(gold_snapshot_path),
        expected_sha256=gold_snapshot_sha256,
        expected_endpoint=gold_snapshot_endpoint,
    )
    examples = _apply_rebased_gold(
        originals,
        records,
        eligible_only=eligible_only,
        strict=strict,
    )
    return examples if max_questions is None else examples[:max_questions]


def iter_qald7(
    path: str | Path,
    *,
    language: str = "en",
    max_questions: int | None = None,
    strict: bool = True,
) -> Iterator[QALD7Example]:
    """Iterate over normalized examples using the same contract as load_qald7."""

    yield from load_qald7(
        path,
        language=language,
        max_questions=max_questions,
        strict=strict,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and inspect QALD-7 JSON.")
    parser.add_argument("path", type=Path, help="Path to a QALD-7 JSON file.")
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="Skip malformed questions where possible instead of failing.",
    )
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()
    examples = load_qald7(
        args.path,
        language=args.language,
        max_questions=args.max_questions,
        strict=not args.lenient,
    )
    summary = {
        "path": str(args.path),
        "language": args.language,
        "questions_loaded": len(examples),
        "select_answers": sum(bool(item.gold_answer_rows) for item in examples),
        "boolean_answers": sum(item.gold_boolean is not None for item in examples),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
