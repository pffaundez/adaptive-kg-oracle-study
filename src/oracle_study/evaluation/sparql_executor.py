"""Bounded SPARQL protocol client with normalized execution outcomes.

The executor is intentionally evaluation-agnostic: it sends a read-only query,
normalizes SELECT/ASK results, and records transport metadata. It does not
compare predictions with gold answers or repair failed queries.
"""

from __future__ import annotations

import json
import re
import socket
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


SPARQL_RESULTS_JSON = "application/sparql-results+json"
DEFAULT_USER_AGENT = "GraphRouter-OracleStudy/0.1 (research client)"
_SYNTAX_ERROR_HINT = re.compile(
    r"syntax|parse error|parser|malformed|sparql compiler|37000",
    flags=re.IGNORECASE,
)


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    EMPTY_RESULT = "empty_result"
    EXECUTION_ERROR = "execution_error"


class ExecutionErrorType(str, Enum):
    SYNTAX_ERROR = "syntax_error"
    TIMEOUT = "timeout"
    ENDPOINT_ERROR = "endpoint_error"
    CLIENT_ERROR = "client_error"
    INVALID_RESPONSE = "invalid_response"
    RESPONSE_TOO_LARGE = "response_too_large"
    UNSUPPORTED_QUERY_FORM = "unsupported_query_form"
    DISALLOWED_SERVICE = "disallowed_service"
    UNKNOWN_ERROR = "unknown_error"


@dataclass(frozen=True)
class SPARQLExecutorConfig:
    endpoint: str
    timeout_seconds: float = 30.0
    max_response_bytes: int = 10_000_000
    user_agent: str = DEFAULT_USER_AGENT
    allow_service: bool = False

    def __post_init__(self) -> None:
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("endpoint must be an absolute HTTP(S) URL.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if self.max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive.")
        if not self.user_agent.strip():
            raise ValueError("user_agent must be non-empty.")

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        endpoint: str | None = None,
    ) -> "SPARQLExecutorConfig":
        resolved_endpoint = endpoint or data.get("endpoint")
        if not isinstance(resolved_endpoint, str) or not resolved_endpoint.strip():
            raise ValueError("A SPARQL endpoint URL is required.")
        return cls(
            endpoint=resolved_endpoint,
            timeout_seconds=float(
                data.get("sparql_timeout_seconds", data.get("timeout_seconds", 30))
            ),
            max_response_bytes=int(data.get("max_response_bytes", 10_000_000)),
            user_agent=str(data.get("user_agent", DEFAULT_USER_AGENT)),
            allow_service=bool(data.get("allow_service", False)),
        )


@dataclass(frozen=True)
class SPARQLExecutionResult:
    status: ExecutionStatus
    query_form: str
    endpoint: str
    latency_seconds: float
    variables: tuple[str, ...] = field(default_factory=tuple)
    rows: tuple[dict[str, dict[str, Any]], ...] = field(default_factory=tuple)
    boolean: bool | None = None
    http_status: int | None = None
    response_bytes: int = 0
    error_type: ExecutionErrorType | None = None
    error_message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status in {ExecutionStatus.SUCCESS, ExecutionStatus.EMPTY_RESULT}

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["error_type"] = (
            self.error_type.value if self.error_type is not None else None
        )
        return payload


def _contains_service_keyword(query: str) -> bool:
    """Find SERVICE outside comments, quoted strings, and IRI references."""

    index = 0
    length = len(query)
    while index < length:
        char = query[index]
        if char == "#":
            newline = query.find("\n", index + 1)
            index = length if newline == -1 else newline + 1
            continue
        if char == "<":
            closing = query.find(">", index + 1)
            index = length if closing == -1 else closing + 1
            continue
        if char in {'"', "'"}:
            delimiter = char * 3 if query.startswith(char * 3, index) else char
            index += len(delimiter)
            while index < length:
                if query[index] == "\\" and index + 1 < length:
                    index += 2
                elif query.startswith(delimiter, index):
                    index += len(delimiter)
                    break
                else:
                    index += 1
            continue
        if char.isalpha() or char == "_":
            start = index
            index += 1
            while index < length and (query[index].isalnum() or query[index] in "_-"):
                index += 1
            if query[start:index].upper() == "SERVICE":
                return True
            continue
        index += 1
    return False


def _normalize_term(term: Any) -> dict[str, Any]:
    if not isinstance(term, Mapping) or "value" not in term:
        raise ValueError("Each SPARQL binding must be an object containing 'value'.")
    normalized: dict[str, Any] = {}
    for key in ("type", "value", "datatype", "xml:lang", "lang"):
        if key in term:
            normalized[key] = term[key]
    return normalized


def _parse_select(payload: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[dict[str, dict[str, Any]], ...]]:
    head = payload.get("head")
    results = payload.get("results")
    if not isinstance(head, Mapping) or not isinstance(results, Mapping):
        raise ValueError("SELECT response must contain 'head' and 'results' objects.")

    raw_variables = head.get("vars", [])
    raw_bindings = results.get("bindings")
    if not isinstance(raw_variables, list) or not all(
        isinstance(variable, str) for variable in raw_variables
    ):
        raise ValueError("SELECT response contains an invalid variable list.")
    if not isinstance(raw_bindings, list):
        raise ValueError("SELECT response contains invalid result bindings.")

    rows: list[dict[str, dict[str, Any]]] = []
    for raw_row in raw_bindings:
        if not isinstance(raw_row, Mapping):
            raise ValueError("Each SELECT result row must be an object.")
        rows.append(
            {str(variable): _normalize_term(term) for variable, term in raw_row.items()}
        )
    return tuple(raw_variables), tuple(rows)


def _sanitized_message(value: Any, *, limit: int = 1000) -> str:
    text = " ".join(str(value).split())
    return text[:limit]


class SPARQLExecutor:
    """Execute bounded SELECT and ASK queries over one configured endpoint."""

    def __init__(
        self,
        config: SPARQLExecutorConfig,
        *,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.config = config
        self._opener = opener

    def _error(
        self,
        *,
        query_form: str,
        started_at: float,
        error_type: ExecutionErrorType,
        message: str,
        http_status: int | None = None,
        response_bytes: int = 0,
    ) -> SPARQLExecutionResult:
        return SPARQLExecutionResult(
            status=ExecutionStatus.EXECUTION_ERROR,
            query_form=query_form,
            endpoint=self.config.endpoint,
            latency_seconds=time.perf_counter() - started_at,
            http_status=http_status,
            response_bytes=response_bytes,
            error_type=error_type,
            error_message=_sanitized_message(message),
        )

    def execute(self, query: str, *, query_form: str) -> SPARQLExecutionResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        normalized_form = query_form.upper().strip()
        started_at = time.perf_counter()

        if normalized_form not in {"SELECT", "ASK"}:
            return self._error(
                query_form=normalized_form,
                started_at=started_at,
                error_type=ExecutionErrorType.UNSUPPORTED_QUERY_FORM,
                message=f"Query form {normalized_form!r} is not supported by QALD evaluation.",
            )
        if not self.config.allow_service and _contains_service_keyword(query):
            return self._error(
                query_form=normalized_form,
                started_at=started_at,
                error_type=ExecutionErrorType.DISALLOWED_SERVICE,
                message="SERVICE clauses are disabled; the experiment targets one KG endpoint.",
            )

        body = urlencode({"query": query}).encode("utf-8")
        request = Request(
            self.config.endpoint,
            data=body,
            method="POST",
            headers={
                "Accept": SPARQL_RESULTS_JSON,
                "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
                "User-Agent": self.config.user_agent,
            },
        )

        try:
            with self._opener(request, timeout=self.config.timeout_seconds) as response:
                http_status = getattr(response, "status", None)
                raw = response.read(self.config.max_response_bytes + 1)
        except HTTPError as exc:
            error_body = exc.read(4096).decode("utf-8", errors="replace")
            error_type = (
                ExecutionErrorType.SYNTAX_ERROR
                if exc.code == 400 and _SYNTAX_ERROR_HINT.search(error_body)
                else ExecutionErrorType.ENDPOINT_ERROR
            )
            return self._error(
                query_form=normalized_form,
                started_at=started_at,
                error_type=error_type,
                message=error_body or str(exc),
                http_status=exc.code,
                response_bytes=len(error_body.encode("utf-8")),
            )
        except (socket.timeout, TimeoutError) as exc:
            return self._error(
                query_form=normalized_form,
                started_at=started_at,
                error_type=ExecutionErrorType.TIMEOUT,
                message=str(exc) or "SPARQL request timed out.",
            )
        except URLError as exc:
            reason = exc.reason
            error_type = (
                ExecutionErrorType.TIMEOUT
                if isinstance(reason, (socket.timeout, TimeoutError))
                else ExecutionErrorType.CLIENT_ERROR
            )
            return self._error(
                query_form=normalized_form,
                started_at=started_at,
                error_type=error_type,
                message=str(reason),
            )
        except OSError as exc:
            return self._error(
                query_form=normalized_form,
                started_at=started_at,
                error_type=ExecutionErrorType.CLIENT_ERROR,
                message=str(exc),
            )

        if len(raw) > self.config.max_response_bytes:
            return self._error(
                query_form=normalized_form,
                started_at=started_at,
                error_type=ExecutionErrorType.RESPONSE_TOO_LARGE,
                message=(
                    "Endpoint response exceeded "
                    f"{self.config.max_response_bytes} bytes."
                ),
                http_status=http_status,
                response_bytes=len(raw),
            )

        try:
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("SPARQL JSON response root must be an object.")

            if normalized_form == "ASK":
                boolean = payload.get("boolean")
                if not isinstance(boolean, bool):
                    raise ValueError("ASK response must contain a boolean value.")
                return SPARQLExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    query_form=normalized_form,
                    endpoint=self.config.endpoint,
                    latency_seconds=time.perf_counter() - started_at,
                    boolean=boolean,
                    http_status=http_status,
                    response_bytes=len(raw),
                )

            variables, rows = _parse_select(payload)
            status = ExecutionStatus.SUCCESS if rows else ExecutionStatus.EMPTY_RESULT
            return SPARQLExecutionResult(
                status=status,
                query_form=normalized_form,
                endpoint=self.config.endpoint,
                latency_seconds=time.perf_counter() - started_at,
                variables=variables,
                rows=rows,
                http_status=http_status,
                response_bytes=len(raw),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            return self._error(
                query_form=normalized_form,
                started_at=started_at,
                error_type=ExecutionErrorType.INVALID_RESPONSE,
                message=str(exc),
                http_status=http_status,
                response_bytes=len(raw),
            )
