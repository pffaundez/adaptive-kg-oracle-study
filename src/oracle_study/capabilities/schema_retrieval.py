"""Structured DBpedia schema-candidate retrieval with a chat model."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from oracle_study.models.huggingface_model import GenerationResult


PLACEHOLDER = "{{ question }}"
_ONTOLOGY_PREFIX = "http://dbpedia.org/ontology/"
_PROPERTY_PREFIXES = (_ONTOLOGY_PREFIX, "http://dbpedia.org/property/")


class SchemaRetrievalError(ValueError):
    """Raised when schema configuration or model output is invalid."""


class ChatModel(Protocol):
    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> GenerationResult: ...


@dataclass(frozen=True)
class SchemaRetrievalConfig:
    prompt_path: Path
    max_class_candidates: int = 10
    max_property_candidates: int = 10
    max_new_tokens: int | None = None

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        project_root: str | Path = ".",
    ) -> "SchemaRetrievalConfig":
        raw_path = data.get("prompt_template") or data.get("prompt_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise SchemaRetrievalError("schema_retrieval requires prompt_template.")
        class_limit = int(data.get("max_class_candidates", 10))
        property_limit = int(data.get("max_property_candidates", 10))
        raw_tokens = data.get("max_new_tokens")
        token_limit = int(raw_tokens) if raw_tokens is not None else None
        if class_limit <= 0 or property_limit <= 0:
            raise SchemaRetrievalError("Schema candidate limits must be positive.")
        if token_limit is not None and token_limit <= 0:
            raise SchemaRetrievalError("max_new_tokens must be positive.")
        path = Path(raw_path)
        return cls(
            prompt_path=path if path.is_absolute() else Path(project_root) / path,
            max_class_candidates=class_limit,
            max_property_candidates=property_limit,
            max_new_tokens=token_limit,
        )


@dataclass(frozen=True)
class SchemaCandidate:
    mention: str
    uri: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class SchemaRetrievalOutput:
    question: str
    class_candidates: tuple[SchemaCandidate, ...]
    property_candidates: tuple[SchemaCandidate, ...]
    raw_output: str
    messages: tuple[dict[str, str], ...]
    prompt_sha256: str
    generation: GenerationResult

    @property
    def evidence(self) -> dict[str, list[dict[str, str]]]:
        return {
            "classes": [item.to_dict() for item in self.class_candidates],
            "properties": [item.to_dict() for item in self.property_candidates],
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["generation"] = self.generation.to_dict()
        payload["evidence"] = self.evidence
        return payload


def _parse_list(
    value: Any,
    *,
    label: str,
    prefixes: tuple[str, ...],
    limit: int,
) -> tuple[SchemaCandidate, ...]:
    if not isinstance(value, list) or len(value) > limit:
        raise SchemaRetrievalError(f"{label} must be a list of at most {limit} items.")
    result: list[SchemaCandidate] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != {"mention", "uri"}:
            raise SchemaRetrievalError(
                f"{label}[{index}] must contain exactly mention and uri."
            )
        mention, uri = item["mention"], item["uri"]
        if not isinstance(mention, str) or not mention.strip():
            raise SchemaRetrievalError(f"{label}[{index}].mention is invalid.")
        if (
            not isinstance(uri, str)
            or not uri.startswith(prefixes)
            or any(character.isspace() for character in uri)
        ):
            raise SchemaRetrievalError(f"{label}[{index}].uri is invalid.")
        if uri in seen:
            # Exact duplicates add no evidence. Keep the first occurrence so a
            # harmless formatting repetition does not fail the whole workflow.
            continue
        seen.add(uri)
        result.append(SchemaCandidate(mention.strip(), uri))
    return tuple(result)


def parse_schema_output(
    raw_output: str,
    *,
    max_class_candidates: int = 10,
    max_property_candidates: int = 10,
) -> tuple[tuple[SchemaCandidate, ...], tuple[SchemaCandidate, ...]]:
    try:
        payload = json.loads(raw_output)
    except (TypeError, json.JSONDecodeError) as exc:
        raise SchemaRetrievalError("Schema output is not valid JSON.") from exc
    if not isinstance(payload, Mapping) or set(payload) != {
        "class_candidates",
        "property_candidates",
    }:
        raise SchemaRetrievalError(
            "Schema output must contain exactly class_candidates and "
            "property_candidates."
        )
    classes = _parse_list(
        payload["class_candidates"],
        label="class_candidates",
        prefixes=(_ONTOLOGY_PREFIX,),
        limit=max_class_candidates,
    )
    properties = _parse_list(
        payload["property_candidates"],
        label="property_candidates",
        prefixes=_PROPERTY_PREFIXES,
        limit=max_property_candidates,
    )
    return classes, properties


def _hash(messages: Sequence[Mapping[str, str]]) -> str:
    digest = hashlib.sha256()
    for message in messages:
        digest.update(message["role"].encode())
        digest.update(b"\0")
        digest.update(message["content"].encode())
        digest.update(b"\0")
    return digest.hexdigest()


class SchemaRetrievalCapability:
    def __init__(self, model: ChatModel, config: SchemaRetrievalConfig) -> None:
        self._model = model
        self.config = config
        self._template: str | None = None

    def build_messages(self, question: str) -> tuple[dict[str, str], ...]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be non-empty.")
        if self._template is None:
            try:
                self._template = self.config.prompt_path.read_text(
                    encoding="utf-8-sig"
                ).strip()
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Schema prompt not found: {self.config.prompt_path}"
                ) from exc
        if not self._template or self._template.count(PLACEHOLDER) != 1:
            raise SchemaRetrievalError(
                f"Schema prompt must contain {PLACEHOLDER!r} exactly once."
            )
        prompt = self._template.replace(PLACEHOLDER, question.strip())
        return ({"role": "user", "content": prompt},)

    def retrieve(self, question: str) -> SchemaRetrievalOutput:
        messages = self.build_messages(question)
        generation = self._model.generate(
            messages, max_new_tokens=self.config.max_new_tokens
        )
        classes, properties = parse_schema_output(
            generation.text,
            max_class_candidates=self.config.max_class_candidates,
            max_property_candidates=self.config.max_property_candidates,
        )
        return SchemaRetrievalOutput(
            question=question.strip(),
            class_candidates=classes,
            property_candidates=properties,
            raw_output=generation.text,
            messages=messages,
            prompt_sha256=_hash(messages),
            generation=generation,
        )