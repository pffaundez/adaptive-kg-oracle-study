"""Conservative extraction of one SPARQL query from raw model output.

The parser performs only mechanical extraction. It never repairs syntax,
prefixes, IRIs, variables, or query structure. Full SPARQL validation belongs
to the execution/evaluation layer.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterator


_QUERY_FORMS = {"SELECT", "ASK", "CONSTRUCT", "DESCRIBE"}
_UPDATE_FORMS = {
    "ADD",
    "CLEAR",
    "COPY",
    "CREATE",
    "DELETE",
    "DROP",
    "INSERT",
    "LOAD",
    "MOVE",
    "WITH",
}
_PROLOGUE_FORMS = {"BASE", "PREFIX"}
_FENCE = re.compile(
    r"\A\s*```(?:sparql)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```\s*\Z",
    flags=re.IGNORECASE,
)
_ANY_FENCE = re.compile(r"```")


class SPARQLParseStatus(str, Enum):
    SUCCESS = "success"
    EMPTY_OUTPUT = "empty_output"
    MALFORMED_FENCE = "malformed_fence"
    PROSE_OUTPUT = "prose_output"
    MISSING_QUERY_FORM = "missing_query_form"
    MULTIPLE_QUERIES = "multiple_queries"
    UNSUPPORTED_OPERATION = "unsupported_operation"


@dataclass(frozen=True)
class SPARQLParseResult:
    """Result of mechanically extracting a query from one model response."""

    status: SPARQLParseStatus
    raw_output: str
    query: str | None
    query_form: str | None
    removed_markdown_fence: bool
    message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status is SPARQLParseStatus.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass(frozen=True)
class _Token:
    value: str
    brace_depth: int


def _strip_single_outer_fence(text: str) -> tuple[str, bool, bool]:
    """Return body, removed flag, and malformed/multiple-fence flag."""

    match = _FENCE.fullmatch(text)
    if match:
        body = match.group("body")
        # Nested or additional fences indicate more than one output block.
        return body.strip(), True, bool(_ANY_FENCE.search(body))
    return text.strip(), False, bool(_ANY_FENCE.search(text))


def _iter_word_tokens(text: str) -> Iterator[_Token]:
    """Yield word tokens outside comments, strings, and IRIs with brace depth."""

    index = 0
    depth = 0
    length = len(text)
    while index < length:
        char = text[index]

        if char == "#":
            newline = text.find("\n", index + 1)
            index = length if newline == -1 else newline + 1
            continue

        if char == "<":
            index += 1
            while index < length:
                if text[index] == "\\" and index + 1 < length:
                    index += 2
                elif text[index] == ">":
                    index += 1
                    break
                else:
                    index += 1
            continue

        if char in {'"', "'"}:
            quote = char
            triple = text.startswith(char * 3, index)
            delimiter = char * 3 if triple else char
            index += len(delimiter)
            while index < length:
                if text[index] == "\\" and index + 1 < length:
                    index += 2
                elif text.startswith(delimiter, index):
                    index += len(delimiter)
                    break
                else:
                    index += 1
            continue

        if char == "{":
            depth += 1
            index += 1
            continue
        if char == "}":
            depth = max(0, depth - 1)
            index += 1
            continue

        if char.isalpha() or char == "_":
            start = index
            index += 1
            while index < length and (
                text[index].isalnum() or text[index] in {"_", "-"}
            ):
                index += 1
            yield _Token(text[start:index].upper(), depth)
            continue

        index += 1


def _top_level_words(text: str) -> list[str]:
    return [token.value for token in _iter_word_tokens(text) if token.brace_depth == 0]


def _classify_top_level(words: list[str]) -> tuple[str | None, str | None]:
    """Return query form and an optional classification error."""

    query_forms: list[str] = []
    update_form: str | None = None
    for word in words:
        if word in _UPDATE_FORMS and update_form is None:
            update_form = word
        if word in _QUERY_FORMS:
            query_forms.append(word)

    if update_form is not None:
        return None, f"SPARQL Update operation {update_form!r} is not allowed."
    if not query_forms:
        return None, "No top-level SELECT, ASK, CONSTRUCT, or DESCRIBE form found."
    if len(query_forms) > 1:
        return None, "More than one top-level SPARQL query form was found."
    return query_forms[0], None


def parse_sparql_output(raw_output: str) -> SPARQLParseResult:
    """Extract exactly one read-only SPARQL query from raw model output.

    A single outer `````sparql```` block is tolerated as a defensive measure.
    Text before or after a fenced block, multiple fences, and explanatory prose
    are rejected rather than silently discarded.
    """

    if not isinstance(raw_output, str):
        raise TypeError("raw_output must be a string.")
    if not raw_output.strip():
        return SPARQLParseResult(
            status=SPARQLParseStatus.EMPTY_OUTPUT,
            raw_output=raw_output,
            query=None,
            query_form=None,
            removed_markdown_fence=False,
            message="The model returned an empty response.",
        )

    candidate, removed_fence, bad_fence = _strip_single_outer_fence(raw_output)
    if bad_fence:
        return SPARQLParseResult(
            status=SPARQLParseStatus.MALFORMED_FENCE,
            raw_output=raw_output,
            query=None,
            query_form=None,
            removed_markdown_fence=removed_fence,
            message="Output contains malformed, nested, or multiple Markdown fences.",
        )
    if not candidate:
        return SPARQLParseResult(
            status=SPARQLParseStatus.EMPTY_OUTPUT,
            raw_output=raw_output,
            query=None,
            query_form=None,
            removed_markdown_fence=removed_fence,
            message="The extracted Markdown block is empty.",
        )

    words = _top_level_words(candidate)
    query_form, classification_error = _classify_top_level(words)
    if query_form is None:
        if classification_error and "Update" in classification_error:
            status = SPARQLParseStatus.UNSUPPORTED_OPERATION
        elif classification_error and "More than one" in classification_error:
            status = SPARQLParseStatus.MULTIPLE_QUERIES
        else:
            status = SPARQLParseStatus.MISSING_QUERY_FORM
        return SPARQLParseResult(
            status=status,
            raw_output=raw_output,
            query=None,
            query_form=None,
            removed_markdown_fence=removed_fence,
            message=classification_error,
        )

    first_word = words[0] if words else None
    if first_word != query_form and first_word not in _PROLOGUE_FORMS:
        return SPARQLParseResult(
            status=SPARQLParseStatus.PROSE_OUTPUT,
            raw_output=raw_output,
            query=None,
            query_form=query_form,
            removed_markdown_fence=removed_fence,
            message="Unexpected top-level text appears before the SPARQL query.",
        )

    return SPARQLParseResult(
        status=SPARQLParseStatus.SUCCESS,
        raw_output=raw_output,
        query=candidate,
        query_form=query_form,
        removed_markdown_fence=removed_fence,
        message=None,
    )