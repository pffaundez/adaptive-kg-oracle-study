# Qwen model with delayed load, BF16 witout quantization, chat template, greedy generation and token/latency measurement.
# Local HF chat-model adapter for Experiment A.

"""
Qwen3-30B
BF16 without quantization
- Delayed load of model weights and tokenizer
- Apply chat template for messages
- Greedy generation (do_sample=False)
- Measure input tokens, output tokens, and latency syncronized with CUDA
- Reproducible generation with fixed seed
- Explicit liberation of memory and accelerator caches after inference
- Message validation and normalization

"""


from __future__ import annotations

import gc
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


DEFAULT_MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
_ALLOWED_ROLES = {"system", "user", "assistant"}


class ModelConfigurationError(ValueError):
    """Raised when the model or generation configuration is invalid."""


class ModelDependencyError(RuntimeError):
    """Raised when optional inference dependencies are unavailable."""


@dataclass(frozen=True)
class HuggingFaceModelConfig:
    """Configuration for an unquantized Hugging Face causal language model."""

    model_id: str = DEFAULT_MODEL_ID
    revision: str | None = None
    dtype: str = "bfloat16"
    device_map: str | Mapping[str, Any] = "auto"
    max_new_tokens: int = 1024
    do_sample: bool = False
    seed: int = 42
    trust_remote_code: bool = False
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ModelConfigurationError("model_id must be non-empty.")
        if self.dtype not in {"bfloat16", "float16", "float32", "auto"}:
            raise ModelConfigurationError(
                "dtype must be bfloat16, float16, float32, or auto."
            )
        if self.max_new_tokens <= 0:
            raise ModelConfigurationError("max_new_tokens must be positive.")
        if self.do_sample:
            raise ModelConfigurationError(
                "Experiment A requires greedy decoding (do_sample=False)."
            )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "HuggingFaceModelConfig":
        """Build the config from the ``model`` block in experiment_a.yaml."""

        loading = data.get("loading") or {}
        generation = data.get("generation") or {}
        if not isinstance(loading, Mapping) or not isinstance(generation, Mapping):
            raise ModelConfigurationError(
                "model.loading and model.generation must be mappings."
            )

        quantization = loading.get("quantization")
        if quantization not in (None, "none"):
            raise ModelConfigurationError(
                "This adapter is configured for unquantized inference; "
                "model.loading.quantization must be null."
            )

        return cls(
            model_id=str(data.get("model_id", DEFAULT_MODEL_ID)),
            revision=data.get("revision"),
            dtype=str(loading.get("dtype", "bfloat16")),
            device_map=loading.get("device_map", "auto"),
            max_new_tokens=int(generation.get("max_new_tokens", 1024)),
            do_sample=bool(generation.get("do_sample", False)),
            seed=int(data.get("seed", 42)),
            trust_remote_code=bool(data.get("trust_remote_code", False)),
            local_files_only=bool(data.get("local_files_only", False)),
        )


@dataclass(frozen=True)
class GenerationResult:
    """Generated continuation and measurements needed by the experiment log."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    model_id: str
    requested_revision: str | None
    resolved_revision: str | None
    dtype: str
    device_map: str | Mapping[str, Any]
    do_sample: bool
    max_new_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_messages(
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    if not messages:
        raise ValueError("messages must contain at least one chat message.")

    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"messages[{index}] must be a mapping.")
        role = message.get("role")
        content = message.get("content")
        if role not in _ALLOWED_ROLES:
            raise ValueError(
                f"messages[{index}].role must be one of {sorted(_ALLOWED_ROLES)}."
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"messages[{index}].content must be non-empty.")
        normalized.append({"role": role, "content": content})
    return normalized


class HuggingFaceChatModel:
    """Thin, lazy-loading wrapper around AutoModelForCausalLM."""

    def __init__(self, config: HuggingFaceModelConfig) -> None:
        self.config = config
        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._resolved_revision: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def resolved_revision(self) -> str | None:
        """Commit hash reported by the loaded model configuration, if present."""

        return self._resolved_revision

    def load(self) -> None:
        """Load tokenizer and unquantized model weights exactly once."""

        if self.is_loaded:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ModelDependencyError(
                "Local inference requires torch, transformers, accelerate, "
                "and safetensors."
            ) from exc

        dtype = self._resolve_dtype(torch)
        common_kwargs = {
            "revision": self.config.revision,
            "trust_remote_code": self.config.trust_remote_code,
            "local_files_only": self.config.local_files_only,
        }
        # Hugging Face accepts revision=None, but omitting it makes logs and
        # debugging clearer when the experiment has not been pinned yet.
        common_kwargs = {key: value for key, value in common_kwargs.items() if value is not None}

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            **common_kwargs,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            trust_remote_code=self.config.trust_remote_code,
            local_files_only=self.config.local_files_only,
            dtype=dtype,
            device_map=self.config.device_map,
            low_cpu_mem_usage=True,
        )
        model.eval()

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._resolved_revision = getattr(model.config, "_commit_hash", None)
        self._set_seed()

    def _resolve_dtype(self, torch: Any) -> Any:
        if self.config.dtype == "auto":
            return "auto"
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.config.dtype]

    def _set_seed(self) -> None:
        random.seed(self.config.seed)
        if self._torch is None:
            return
        self._torch.manual_seed(self.config.seed)
        if self._torch.cuda.is_available():
            self._torch.cuda.manual_seed_all(self.config.seed)

    def _input_device(self) -> Any:
        """Return the device holding input embeddings under device_map='auto'."""

        if self._model is None:
            raise RuntimeError("The model is not loaded.")
        embeddings = self._model.get_input_embeddings()
        return embeddings.weight.device

    def _synchronize_cuda(self) -> None:
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.synchronize()

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        """Generate one deterministic assistant continuation from chat messages."""

        normalized_messages = _validate_messages(messages)
        if not self.is_loaded:
            self.load()
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None

        generation_limit = (
            self.config.max_new_tokens
            if max_new_tokens is None
            else int(max_new_tokens)
        )
        if generation_limit <= 0:
            raise ValueError("max_new_tokens must be positive.")

        encoded = self._tokenizer.apply_chat_template(
            normalized_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        encoded = encoded.to(self._input_device())
        input_length = int(encoded["input_ids"].shape[-1])
        if "attention_mask" in encoded:
            input_tokens = int(encoded["attention_mask"].sum().item())
        else:
            input_tokens = input_length

        self._synchronize_cuda()
        started_at = time.perf_counter()
        with self._torch.inference_mode():
            output = self._model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=generation_limit,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        self._synchronize_cuda()
        latency = time.perf_counter() - started_at

        continuation = output[0, input_length:]
        text = self._tokenizer.decode(
            continuation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=int(continuation.shape[-1]),
            latency_seconds=latency,
            model_id=self.config.model_id,
            requested_revision=self.config.revision,
            resolved_revision=self._resolved_revision,
            dtype=self.config.dtype,
            device_map=self.config.device_map,
            do_sample=False,
            max_new_tokens=generation_limit,
        )

    def unload(self) -> None:
        """Release references to model weights and clear accelerator caches."""

        self._model = None
        self._tokenizer = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def __enter__(self) -> "HuggingFaceChatModel":
        self.load()
        return self

    def __exit__(self, *_: Any) -> None:
        self.unload()