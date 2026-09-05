"""Append-only JSONL persistence for per-question workflow results.

Each record is keyed by ``(experiment_id, workflow_id, question_id, run_id)``.
The writer supports safe sequential resumption, but deliberately does not claim
multi-process safety. Parallel jobs should write separate shard files.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, TextIO


class SerializableResult(Protocol):
    def to_dict(self) -> dict[str, Any]: ...


class DuplicatePolicy(str, Enum):
    ERROR = "error"
    SKIP = "skip"


class ResultFileError(ValueError):
    """Raised when an existing JSONL result file is malformed or inconsistent."""


@dataclass(frozen=True)
class ResultKey:
    experiment_id: str
    workflow_id: str
    question_id: str
    run_id: int

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ResultKey":
        missing = [
            key
            for key in ("experiment_id", "workflow_id", "question_id", "run_id")
            if key not in record
        ]
        if missing:
            raise ResultFileError(
                "Result record is missing key fields: " + ", ".join(missing)
            )

        experiment_id = record["experiment_id"]
        workflow_id = record["workflow_id"]
        question_id = record["question_id"]
        run_id = record["run_id"]
        if not all(
            isinstance(value, str) and value.strip()
            for value in (experiment_id, workflow_id, question_id)
        ):
            raise ResultFileError(
                "experiment_id, workflow_id, and question_id must be non-empty strings."
            )
        if not isinstance(run_id, int) or isinstance(run_id, bool) or run_id < 0:
            raise ResultFileError("run_id must be a non-negative integer.")
        return cls(experiment_id, workflow_id, question_id, run_id)


@dataclass(frozen=True)
class WriteOutcome:
    key: ResultKey
    written: bool
    reason: str


def _record_mapping(result: SerializableResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, Mapping):
        return dict(result)
    to_dict = getattr(result, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("result must be a mapping or expose a callable to_dict().")
    record = to_dict()
    if not isinstance(record, Mapping):
        raise TypeError("result.to_dict() must return a mapping.")
    return dict(record)


def read_result_keys(path: str | Path) -> set[ResultKey]:
    """Validate an existing JSONL file and return all unique result keys."""

    source = Path(path)
    if not source.exists():
        return set()

    keys: set[ResultKey] = set()
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ResultFileError(
                    f"Blank line in result file {source} at line {line_number}."
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ResultFileError(
                    f"Invalid JSON in {source} at line {line_number}: {exc.msg}."
                ) from exc
            if not isinstance(record, Mapping):
                raise ResultFileError(
                    f"Result at {source}:{line_number} must be a JSON object."
                )
            key = ResultKey.from_record(record)
            if key in keys:
                raise ResultFileError(
                    f"Duplicate result key in {source} at line {line_number}: {key}."
                )
            keys.add(key)
    return keys


class JSONLResultWriter:
    """Sequential append-only writer with resume and duplicate protection."""

    def __init__(
        self,
        path: str | Path,
        *,
        duplicate_policy: DuplicatePolicy | str = DuplicatePolicy.ERROR,
        flush_every: int = 1,
        fsync: bool = False,
    ) -> None:
        self.path = Path(path)
        try:
            self.duplicate_policy = DuplicatePolicy(duplicate_policy)
        except ValueError as exc:
            raise ValueError("duplicate_policy must be 'error' or 'skip'.") from exc
        if flush_every <= 0:
            raise ValueError("flush_every must be positive.")
        self.flush_every = flush_every
        self.fsync = fsync
        self._keys: set[ResultKey] = set()
        self._stream: TextIO | None = None
        self._writes_since_flush = 0
        self.written_count = 0
        self.skipped_count = 0

    @property
    def is_open(self) -> bool:
        return self._stream is not None and not self._stream.closed

    def open(self) -> "JSONLResultWriter":
        if self.is_open:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._keys = read_result_keys(self.path)
        self._stream = self.path.open("a", encoding="utf-8", newline="\n")
        return self

    def contains(self, key: ResultKey) -> bool:
        return key in self._keys

    def write(
        self,
        result: SerializableResult | Mapping[str, Any],
    ) -> WriteOutcome:
        if not self.is_open:
            raise RuntimeError("Result writer is not open.")
        assert self._stream is not None

        record = _record_mapping(result)
        key = ResultKey.from_record(record)
        if key in self._keys:
            if self.duplicate_policy is DuplicatePolicy.SKIP:
                self.skipped_count += 1
                return WriteOutcome(key=key, written=False, reason="duplicate_skipped")
            raise ResultFileError(f"Result key already exists: {key}.")

        try:
            serialized = json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ResultFileError(f"Result is not valid strict JSON: {exc}") from exc

        self._stream.write(serialized + "\n")
        self._keys.add(key)
        self.written_count += 1
        self._writes_since_flush += 1
        if self._writes_since_flush >= self.flush_every:
            self.flush()
        return WriteOutcome(key=key, written=True, reason="written")

    def write_many(
        self,
        results: Iterable[SerializableResult | Mapping[str, Any]],
    ) -> list[WriteOutcome]:
        return [self.write(result) for result in results]

    def flush(self) -> None:
        if not self.is_open:
            return
        assert self._stream is not None
        self._stream.flush()
        if self.fsync:
            os.fsync(self._stream.fileno())
        self._writes_since_flush = 0

    def close(self) -> None:
        if not self.is_open:
            return
        assert self._stream is not None
        self.flush()
        self._stream.close()
        self._stream = None

    def __enter__(self) -> "JSONLResultWriter":
        return self.open()

    def __exit__(self, *_: Any) -> None:
        self.close()