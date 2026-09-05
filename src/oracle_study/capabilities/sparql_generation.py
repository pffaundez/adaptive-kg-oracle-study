"""Reusable SPARQL-generation capability for workflow-based experiments.

This module renders the experiment prompt and invokes a chat model. It does not
extract, validate, execute, or repair SPARQL; those operations belong to later
pipeline stages so the raw model output always remains available for auditing.
The same capability is used by Experiment A workflows and by Experiment B.
"""

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
    "linking_evidence": "{{ linking_evidence }}",
    "schema_evidence": "{{ schema_evidence }}",
}
NO_EVIDENCE = "None provided."
_UNRESOLVED_PLACEHOLDER = re.compile(r"{{\s*[^{}]+?\s*}}")


class PromptTemplateError(ValueError):
    """Raised when a prompt template cannot be rendered safely."""


class ChatModel(Protocol):
    """Minimal interface required by the generation capability."""

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> GenerationResult: ...


@dataclass(frozen=True)
class SPARQLGenerationConfig:
    """Configuration for rendering and submitting the W1 prompt."""

    prompt_path: Path
    system_prompt_path: Path | None = None
    max_new_tokens: int | None = None

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        project_root: str | Path = ".",
    ) -> "SPARQLGenerationConfig":
        root = Path(project_root)
        raw_prompt_path = data.get("prompt_template") or data.get("prompt_path")
        if not isinstance(raw_prompt_path, str) or not raw_prompt_path.strip():
            raise PromptTemplateError(
                "sparql_generation requires 'prompt_template' or 'prompt_path'."
            )

        raw_system_path = data.get("system_prompt_path")
        if raw_system_path is not None and (
            not isinstance(raw_system_path, str) or not raw_system_path.strip()
        ):
            raise PromptTemplateError("system_prompt_path must be a non-empty path.")

        raw_limit = data.get("max_new_tokens")
        limit = int(raw_limit) if raw_limit is not None else None
        if limit is not None and limit <= 0:
            raise PromptTemplateError("max_new_tokens must be positive or omitted.")

        return cls(
            prompt_path=_resolve_path(root, raw_prompt_path),
            system_prompt_path=(
                _resolve_path(root, raw_system_path)
                if raw_system_path is not None
                else None
            ),
            max_new_tokens=limit,
        )


@dataclass(frozen=True)
class SPARQLGenerationOutput:
    """Raw generation plus the exact prompt provenance used to obtain it."""

    question: str
    raw_output: str
    messages: tuple[dict[str, str], ...]
    prompt_sha256: str
    generation: GenerationResult

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["generation"] = self.generation.to_dict()
        return payload


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_prompt(path: Path, *, label: str) -> str:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} file not found: {path}") from exc
    except OSError as exc:
        raise OSError(f"Could not read {label} file {path}: {exc}") from exc

    if not text.strip():
        raise PromptTemplateError(f"{label} file is empty: {path}")
    return text.strip()


def _format_evidence(evidence: Any | None) -> str:
    """Render optional evidence deterministically for prompt provenance."""

    if evidence is None:
        return NO_EVIDENCE
    if isinstance(evidence, str):
        return evidence.strip() or NO_EVIDENCE
    try:
        return json.dumps(
            evidence,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Evidence must be None, a string, or a JSON-serializable value."
        ) from exc


def render_generation_prompt(
    template: str,
    *,
    question: str,
    linking_evidence: Any | None = None,
    schema_evidence: Any | None = None,
) -> str:
    """Render the three-field reusable generation prompt deterministically."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")

    values = {
        "question": question.strip(),
        "linking_evidence": _format_evidence(linking_evidence),
        "schema_evidence": _format_evidence(schema_evidence),
    }
    rendered = template
    for name, placeholder in PLACEHOLDERS.items():
        occurrences = rendered.count(placeholder)
        if occurrences != 1:
            raise PromptTemplateError(
                f"The prompt must contain {placeholder!r} exactly once; "
                f"found {occurrences}."
            )
        rendered = rendered.replace(placeholder, values[name])

    unresolved = _UNRESOLVED_PLACEHOLDER.findall(rendered)
    if unresolved:
        raise PromptTemplateError(
            "Unsupported or unresolved prompt placeholders: " + ", ".join(unresolved)
        )
    return rendered


def _prompt_hash(messages: Sequence[Mapping[str, str]]) -> str:
    """Hash roles and contents with explicit separators for reproducibility."""

    digest = hashlib.sha256()
    for message in messages:
        digest.update(message["role"].encode("utf-8"))
        digest.update(b"\x00")
        digest.update(message["content"].encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


class SPARQLGenerationCapability:
    """Render a question prompt and request one raw model generation."""

    def __init__(self, model: ChatModel, config: SPARQLGenerationConfig) -> None:
        self._model = model
        self.config = config
        self._user_template: str | None = None
        self._system_prompt: str | None = None

    def load_prompts(self) -> None:
        """Read prompt files once; safe to call repeatedly."""

        if self._user_template is None:
            self._user_template = _read_prompt(
                self.config.prompt_path, label="SPARQL generation prompt"
            )
        if self.config.system_prompt_path is not None and self._system_prompt is None:
            self._system_prompt = _read_prompt(
                self.config.system_prompt_path, label="system prompt"
            )

    def build_messages(
        self,
        question: str,
        *,
        linking_evidence: Any | None = None,
        schema_evidence: Any | None = None,
    ) -> tuple[dict[str, str], ...]:
        """Build the exact chat messages sent to the model."""

        self.load_prompts()
        assert self._user_template is not None
        user_prompt = render_generation_prompt(
            self._user_template,
            question=question,
            linking_evidence=linking_evidence,
            schema_evidence=schema_evidence,
        )

        messages: list[dict[str, str]] = []
        if self._system_prompt is not None:
            if _UNRESOLVED_PLACEHOLDER.search(self._system_prompt):
                raise PromptTemplateError(
                    "The system prompt must not contain template placeholders."
                )
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return tuple(messages)

    def generate(
        self,
        question: str,
        *,
        linking_evidence: Any | None = None,
        schema_evidence: Any | None = None,
    ) -> SPARQLGenerationOutput:
        """Generate raw text for one question and retain full provenance."""

        messages = self.build_messages(
            question,
            linking_evidence=linking_evidence,
            schema_evidence=schema_evidence,
        )
        result = self._model.generate(
            messages,
            max_new_tokens=self.config.max_new_tokens,
        )
        return SPARQLGenerationOutput(
            question=question.strip(),
            raw_output=result.text,
            messages=messages,
            prompt_sha256=_prompt_hash(messages),
            generation=result,
        )
