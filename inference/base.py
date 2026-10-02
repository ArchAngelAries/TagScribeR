"""Provider interface shared by every captioning / tagging backend."""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Sequence

from PIL import Image

# Capability flags — the UI enables controls based on these instead of
# assuming every backend supports everything.
CAP_PROMPT = "prompt"            # follows a free-form instruction
CAP_SYSTEM_PROMPT = "system"     # accepts a system prompt
CAP_SAMPLING = "sampling"        # temperature / top-p / top-k
CAP_BATCH = "batch"              # can process several images per call
CAP_TAGS = "tags"                # emits scored booru-style tags
CAP_UNLOADABLE = "unloadable"    # holds local memory that can be freed


class InferenceError(Exception):
    """A user-presentable failure. ``hint`` suggests a remedy."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base}\n→ {self.hint}" if self.hint else base


class Cancelled(Exception):
    pass


@dataclass
class GenerationParams:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0                 # 0 = backend default
    repetition_penalty: float = 1.05
    presence_penalty: float = 0.0  # API path only (transformers has no equivalent)
    seed: int | None = None


@dataclass
class CaptionRequest:
    prompt: str
    system_prompt: str = ""
    params: GenerationParams = field(default_factory=GenerationParams)
    max_image_side: int = 1536     # images are downscaled before inference
    strip_thinking: bool = True


@dataclass(frozen=True)
class ProviderSpec:
    """Everything needed to construct (and cache) a provider.

    ``kind``: 'transformers' | 'openai' | 'wd_tagger'
    ``model``: HF repo id, local folder, or API model name.
    ``options``: backend-specific, hashable (tuple of pairs) so specs are cache keys.
    """
    kind: str
    model: str
    options: tuple[tuple[str, object], ...] = ()

    def opt(self, name: str, default=None):
        return dict(self.options).get(name, default)

    @staticmethod
    def make(kind: str, model: str, **options) -> "ProviderSpec":
        return ProviderSpec(kind, model, tuple(sorted(options.items())))


ProgressFn = Callable[[str], None]


class Provider(ABC):
    capabilities: frozenset[str] = frozenset()
    #: How many images the worker should hand to generate() at once.
    preferred_chunk: int = 1

    def __init__(self, spec: ProviderSpec):
        self.spec = spec

    @property
    def display_name(self) -> str:
        return self.spec.model

    @property
    def loaded(self) -> bool:
        return True

    def load(self, report: ProgressFn, cancel: threading.Event | None = None) -> None:
        """Prepare for inference (download / load weights). Idempotent."""

    def unload(self) -> None:
        """Release memory. Must be safe to call repeatedly."""

    @abstractmethod
    def generate(self, images: Sequence[Image.Image], request: CaptionRequest,
                 cancel: threading.Event | None = None) -> list[str | Exception]:
        """Return one caption (or the Exception that prevented it) per image."""
