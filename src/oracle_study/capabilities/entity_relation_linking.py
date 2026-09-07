"""Structured entity/relation candidate linking with a chat model."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from oracle_study.models.huggingface_model import GenerationResult


PLACEHOLDER = "{{ question }}"
_UNRESOLVED_PLACEHOLDER = re.compile(r"{{\s*[^{}]+?\s*}}")
_ENTITY_PREFIX = "http://dbpedia.org/resource/"
_RELATION_PREFIXES = (
    "http://dbpedia.org/ontology/",
    "http://dbpedia.org/property/",
)


class EntityRelationLinkingError(ValueError):
    """Raised when linking configuration or model output is invalid."""


class ChatModel(Protocol):
    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> GenerationResult: ...


@dataclass(frozen=True)
class EntityRelationLinkingConfig:
    prompt_path: Path
    max_entity_candidates: int = 10
    max_relation_candidates: int = 10
    max_new_tokens: int | None = None

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        project_root: str | Path = ".",
    ) -> "EntityRelationLinkingConfig":
        raw_path = data.get("prompt_template") or data.get("prompt_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise EntityRelationLinkingError(
                "entity_relation_linking requires prompt_template or prompt_path."
            )
        entity_limit = int(data.get("max_entity_candidates", 10))
        relation_limit = int(data.get("max_relation_candidates", 10))
        if entity_limit <= 0 or relation_limit <= 0:
            raise EntityRelationLinkingError("Candidate limits must be positive.")
        raw_tokens = data.get("max_new_tokens")
        token_limit = int(raw_tokens) if raw_tokens is not None else None
        if token_limit is not None and token_limit <= 0:
            raise EntityRelationLinkingError("max_new_tokens must be positive.")
        path = Path(raw_path)
        root = Path(project_root)
        return cls(
            prompt_path=path if path.is_absolute() else root / path,
            max_entity_candidates=entity_limit,
            max_relation_candidates=relation_limit,
            max_new_tokens=token_limit,
        )


@dataclass(frozen=True)
class LinkingCandidate:
    mention: str
    uri: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class EntityRelationLinkingOutput:
    question: str
    entity_candidates: tuple[LinkingCandidate, ...]
    relation_candidates: tuple[LinkingCandidate, ...]
    raw_output: str
    messages: tuple[dict[str, str], ...]
    prompt_sha256: str
    generation: GenerationResult

    @property
    def evidence(self) -> dict[str, list[dict[str, str]]]:
        return {
            "entities": [candidate.to_dict() for candidate in self.entity_candidates],
            "relations": [candidate.to_dict() for candidate in self.relation_candidates],
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["generation"] = self.generation.to_dict()
        payload["evidence"] = self.evidence
        return payload


def _prompt_hash(messages: Sequence[Mapping[str, str]]) -> str:
    digest = hashlib.sha256()
    for message in messages:
        digest.update(message["role"].encode("utf-8"))
        digest.update(b"\x00")
        digest.update(message["content"].encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _parse_candidates(
    value: Any,
    *,
    label: str,
    prefixes: tuple[str, ...],
    limit: int,
) -> tuple[LinkingCandidate, ...]:
    if not isinstance(value, list):
        raise EntityRelationLinkingError(f"{label} must be a JSON list.")
    if len(value) > limit:
        raise EntityRelationLinkingError(
            f"{label} contains {len(value)} candidates; maximum is {limit}."
        )
    candidates: list[LinkingCandidate] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise EntityRelationLinkingError(f"{label}[{index}] must be an object.")
        if set(item) != {"mention", "uri"}:
            raise EntityRelationLinkingError(
                f"{label}[{index}] must contain exactly mention and uri."
            )
        mention = item["mention"]
        uri = item["uri"]
        if not isinstance(mention, str) or not mention.strip():
            raise EntityRelationLinkingError(f"{label}[{index}].mention is invalid.")
        if not isinstance(uri, str) or not uri.startswith(prefixes):
            raise EntityRelationLinkingError(f"{label}[{index}].uri is invalid.")
        if any(character.isspace() for character in uri):
            raise EntityRelationLinkingError(
                f"{label}[{index}].uri must not contain whitespace."
            )
        if uri in seen:
            # Exact duplicates add no evidence. Keep the first occurrence so a
            # harmless formatting repetition does not fail the whole workflow.
            continue
        seen.add(uri)
        candidates.append(LinkingCandidate(mention=mention.strip(), uri=uri))
    return tuple(candidates)


def parse_linking_output(
    raw_output: str,
    *,
    max_entity_candidates: int = 10,
    max_relation_candidates: int = 10,
) -> tuple[tuple[LinkingCandidate, ...], tuple[LinkingCandidate, ...]]:
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise EntityRelationLinkingError("Linking output is empty.")
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise EntityRelationLinkingError(
            f"Linking output is not valid JSON: {exc.msg}."
        ) from exc
    if not isinstance(payload, Mapping) or set(payload) != {
        "entity_candidates",
        "relation_candidates",
    }:
        raise EntityRelationLinkingError(
            "Linking output must contain exactly entity_candidates and "
            "relation_candidates."
        )
    entities = _parse_candidates(
        payload["entity_candidates"],
        label="entity_candidates",
        prefixes=(_ENTITY_PREFIX,),
        limit=max_entity_candidates,
    )
    relations = _parse_candidates(
        payload["relation_candidates"],
        label="relation_candidates",
        prefixes=_RELATION_PREFIXES,
        limit=max_relation_candidates,
    )
    return entities, relations


class EntityRelationLinkingCapability:
    def __init__(self, model: ChatModel, config: EntityRelationLinkingConfig) -> None:
        self._model = model
        self.config = config
        self._template: str | None = None

    def build_messages(self, question: str) -> tuple[dict[str, str], ...]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be non-empty.")
        if self._template is None:
            try:
                text = self.config.prompt_path.read_text(encoding="utf-8-sig")
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Entity/relation linking prompt not found: {self.config.prompt_path}"
                ) from exc
            if not text.strip():
                raise EntityRelationLinkingError("Linking prompt is empty.")
            self._template = text.strip()
        if self._template.count(PLACEHOLDER) != 1:
            raise EntityRelationLinkingError(
                f"Linking prompt must contain {PLACEHOLDER!r} exactly once."
            )
        prompt = self._template.replace(PLACEHOLDER, question.strip())
        unresolved = _UNRESOLVED_PLACEHOLDER.findall(prompt)
        if unresolved:
            raise EntityRelationLinkingError(
                "Unsupported linking placeholders: " + ", ".join(unresolved)
            )
        return ({"role": "user", "content": prompt},)

    def link(self, question: str) -> EntityRelationLinkingOutput:
        messages = self.build_messages(question)
        generation = self._model.generate(
            messages,
            max_new_tokens=self.config.max_new_tokens,
        )
        entities, relations = parse_linking_output(
            generation.text,
            max_entity_candidates=self.config.max_entity_candidates,
            max_relation_candidates=self.config.max_relation_candidates,
        )
        return EntityRelationLinkingOutput(
            question=question.strip(),
            entity_candidates=entities,
            relation_candidates=relations,
            raw_output=generation.text,
            messages=messages,
            prompt_sha256=_prompt_hash(messages),
            generation=generation,
        )
