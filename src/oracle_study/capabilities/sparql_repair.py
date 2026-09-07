"""Reusable execution-guided SPARQL repair capability."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from oracle_study.models.huggingface_model import GenerationResult


PLACEHOLDERS = {
    "question": "{{ question }}",
    "failed_query": "{{ failed_query }}",
    "execution_feedback": "{{ execution_feedback }}",
}
_UNRESOLVED_PLACEHOLDER = re.compile(r"{{\s*[^{}]+?\s*}}")


class RepairPromptTemplateError(ValueError):
    """Raised when the repair prompt cannot be rendered safely."""


class ChatModel(Protocol):
    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> GenerationResult: ...


@dataclass(frozen=True)
class SPARQLRepairConfig:
    prompt_path: Path
    max_new_tokens: int | None = None

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        project_root: str | Path = ".",
    ) -> "SPARQLRepairConfig":
        raw_path = data.get("prompt_template") or data.get("prompt_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise RepairPromptTemplateError(
                "sparql_repair requires prompt_template or prompt_path."
            )
        raw_limit = data.get("max_new_tokens")
        limit = int(raw_limit) if raw_limit is not None else None
        if limit is not None and limit <= 0:
            raise RepairPromptTemplateError("max_new_tokens must be positive.")
        path = Path(raw_path)
        root = Path(project_root)
        return cls(
            prompt_path=path if path.is_absolute() else root / path,
            max_new_tokens=limit,
        )


@dataclass(frozen=True)
class SPARQLRepairOutput:
    question: str
    failed_query: str
    execution_feedback: dict[str, Any]
    raw_output: str
    messages: tuple[dict[str, str], ...]
    prompt_sha256: str
    generation: GenerationResult

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["generation"] = self.generation.to_dict()
        return payload


def _prompt_hash(messages: Sequence[Mapping[str, str]]) -> str:
    digest = hashlib.sha256()
    for message in messages:
        digest.update(message["role"].encode("utf-8"))
        digest.update(b"\x00")
        digest.update(message["content"].encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _feedback_dict(feedback: Any) -> dict[str, Any]:
    if hasattr(feedback, "to_dict"):
        feedback = feedback.to_dict()
    if not isinstance(feedback, Mapping):
        raise TypeError("execution_feedback must be a mapping or expose to_dict().")
    # Give the model only actionable observations, not endpoint bookkeeping.
    allowed = {
        "status",
        "error_type",
        "error_message",
        "query_form",
        "boolean",
        "variables",
        "rows",
    }
    return {key: feedback[key] for key in sorted(allowed) if key in feedback}


def render_repair_prompt(
    template: str,
    *,
    question: str,
    failed_query: str,
    execution_feedback: Any,
) -> tuple[str, dict[str, Any]]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty.")
    if not isinstance(failed_query, str) or not failed_query.strip():
        raise ValueError("failed_query must be non-empty.")
    feedback = _feedback_dict(execution_feedback)
    values = {
        "question": question.strip(),
        "failed_query": failed_query.strip(),
        "execution_feedback": json.dumps(
            feedback, ensure_ascii=False, indent=2, sort_keys=True
        ),
    }
    rendered = template
    for name, placeholder in PLACEHOLDERS.items():
        count = rendered.count(placeholder)
        if count != 1:
            raise RepairPromptTemplateError(
                f"The repair prompt must contain {placeholder!r} exactly once; "
                f"found {count}."
            )
        rendered = rendered.replace(placeholder, values[name])
    unresolved = _UNRESOLVED_PLACEHOLDER.findall(rendered)
    if unresolved:
        raise RepairPromptTemplateError(
            "Unsupported repair placeholders: " + ", ".join(unresolved)
        )
    return rendered, feedback


class SPARQLRepairCapability:
    def __init__(self, model: ChatModel, config: SPARQLRepairConfig) -> None:
        self._model = model
        self.config = config
        self._template: str | None = None

    def load_prompt(self) -> None:
        if self._template is not None:
            return
        try:
            text = self.config.prompt_path.read_text(encoding="utf-8-sig")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"SPARQL repair prompt not found: {self.config.prompt_path}"
            ) from exc
        if not text.strip():
            raise RepairPromptTemplateError("SPARQL repair prompt is empty.")
        self._template = text.strip()

    def build_messages(
        self,
        question: str,
        *,
        failed_query: str,
        execution_feedback: Any,
        linking_evidence: Any | None = None,
        schema_evidence: Any | None = None,
    ) -> tuple[tuple[dict[str, str], ...], dict[str, Any]]:
        self.load_prompt()
        assert self._template is not None
        prompt, feedback = render_repair_prompt(
            self._template,
            question=question,
            failed_query=failed_query,
            execution_feedback=execution_feedback,
        )
        if linking_evidence is not None:
            prompt += "\n\nEntity/relation evidence:\n" + json.dumps(
                linking_evidence, ensure_ascii=False, indent=2, sort_keys=True
            )
        if schema_evidence is not None:
            prompt += "\n\nSchema evidence:\n" + json.dumps(
                schema_evidence, ensure_ascii=False, indent=2, sort_keys=True
            )
        return ({"role": "user", "content": prompt},), feedback

    def repair(
        self,
        question: str,
        *,
        failed_query: str,
        execution_feedback: Any,
        linking_evidence: Any | None = None,
        schema_evidence: Any | None = None,
    ) -> SPARQLRepairOutput:
        messages, feedback = self.build_messages(
            question,
            failed_query=failed_query,
            execution_feedback=execution_feedback,
            linking_evidence=linking_evidence,
            schema_evidence=schema_evidence,
        )
        result = self._model.generate(
            messages,
            max_new_tokens=self.config.max_new_tokens,
        )
        return SPARQLRepairOutput(
            question=question.strip(),
            failed_query=failed_query.strip(),
            execution_feedback=feedback,
            raw_output=result.text,
            messages=messages,
            prompt_sha256=_prompt_hash(messages),
            generation=result,
        )
