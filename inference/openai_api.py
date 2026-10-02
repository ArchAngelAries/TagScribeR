"""OpenAI-compatible chat-completions endpoints.

Covers cloud APIs and every common local server: LM Studio, Ollama (/v1),
llama.cpp ``llama-server``, KoboldCpp, vLLM, text-generation-webui. This is
also the recommended route for GGUF models (e.g. JoyCaption GGUF + mmproj):
serve them with LM Studio or llama-server and point TagScribeR at the URL.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from PIL import Image

from core import captions
from inference.base import (CAP_BATCH, CAP_PROMPT, CAP_SAMPLING, CAP_SYSTEM_PROMPT, Cancelled,
                            CaptionRequest, InferenceError, ProgressFn, Provider, ProviderSpec)
from inference.image_prep import to_jpeg_base64

log = logging.getLogger(__name__)


def is_local_url(url: str) -> bool:
    from urllib.parse import urlparse
    host = (urlparse(url).hostname or "").lower()
    return (host in ("localhost", "127.0.0.1", "::1", "0.0.0.0") or host.endswith(".local")
            or host.startswith(("192.168.", "10.")) or
            (host.startswith("172.") and host.split(".")[1].isdigit() and 16 <= int(host.split(".")[1]) <= 31))


def _friendly_error(e: Exception) -> InferenceError:
    try:
        import openai
    except ImportError:
        return InferenceError(str(e))
    if isinstance(e, openai.AuthenticationError):
        return InferenceError("The API rejected the key (401).", "Check the API key in the profile.")
    if isinstance(e, openai.PermissionDeniedError):
        return InferenceError("Access denied by the API (403).", "Your key may not have access to this model.")
    if isinstance(e, openai.NotFoundError):
        return InferenceError("Model or endpoint not found (404).",
                              "Check the base URL (usually ends in /v1) and the model name.")
    if isinstance(e, openai.RateLimitError):
        return InferenceError("Rate limited by the API (429), even after retries.",
                              "Lower concurrency or wait a moment and resume.")
    if isinstance(e, openai.APITimeoutError):
        return InferenceError("The API timed out.", "Increase the timeout, or check the server is responsive.")
    if isinstance(e, openai.APIConnectionError):
        return InferenceError("Could not connect to the API server.",
                              "Is LM Studio / the server running, and is the URL correct?")
    if isinstance(e, openai.BadRequestError):
        return InferenceError(f"The API refused the request: {getattr(e, 'message', e)}",
                              "The model may not accept images (pick a vision model).")
    if isinstance(e, openai.APIStatusError):
        return InferenceError(f"API error {e.status_code}: {getattr(e, 'message', e)}")
    return InferenceError(f"API call failed: {e}")


class OpenAICompatible(Provider):
    capabilities = frozenset({CAP_PROMPT, CAP_SYSTEM_PROMPT, CAP_SAMPLING, CAP_BATCH})

    def __init__(self, spec: ProviderSpec):
        super().__init__(spec)
        self.client = None
        self.preferred_chunk = max(1, int(spec.opt("concurrency", 1)))

    @property
    def display_name(self) -> str:
        return f"{self.spec.model} @ {self.spec.opt('base_url', '')}"

    def load(self, report: ProgressFn, cancel: threading.Event | None = None) -> None:
        if self.client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as e:
            raise InferenceError("The 'openai' package is missing.", "Run update.bat to install dependencies.") from e
        base_url = (self.spec.opt("base_url") or "").strip()
        if not base_url:
            raise InferenceError("No API base URL set.", "Enter the server URL, e.g. http://127.0.0.1:1234/v1")
        # Local servers accept any key, but the client requires a non-empty one.
        self.client = OpenAI(base_url=base_url, api_key=self.spec.opt("api_key") or "not-needed",
                             timeout=float(self.spec.opt("timeout", 120.0)),
                             max_retries=int(self.spec.opt("max_retries", 3)))
        report(f"Using API: {self.display_name}")

    def unload(self) -> None:
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
        self.client = None

    def _one(self, img: Image.Image, req: CaptionRequest, cancel: threading.Event | None) -> str | Exception:
        if cancel is not None and cancel.is_set():
            return Cancelled()
        messages = []
        if req.system_prompt.strip():
            messages.append({"role": "system", "content": req.system_prompt.strip()})
        messages.append({"role": "user", "content": [
            {"type": "text", "text": req.prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{to_jpeg_base64(img)}"}},
        ]})
        p = req.params
        kwargs = {"model": self.spec.model, "messages": messages, "max_tokens": p.max_new_tokens,
                  "temperature": p.temperature, "top_p": p.top_p}
        if p.seed is not None:
            kwargs["seed"] = p.seed
        if p.presence_penalty:
            kwargs["presence_penalty"] = p.presence_penalty
        thinking = self.spec.opt("disable_thinking", "auto")
        if thinking is True or (thinking == "auto" and is_local_url(self.spec.opt("base_url", ""))):
            # Honoured by llama.cpp / LM Studio / vLLM chat templates (Qwen3.5+); ignored otherwise.
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            log.debug("API call failed", exc_info=True)
            return _friendly_error(e)
        try:
            text = resp.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError):
            return InferenceError("The API returned a malformed response.")
        if req.strip_thinking:
            text = captions.clean_model_output(text)
        if not text.strip():
            finish = getattr(resp.choices[0], "finish_reason", "")
            hint = "Increase Max tokens (reasoning models spend tokens thinking)." if finish == "length" else ""
            return InferenceError("The API returned an empty caption.", hint)
        return text.strip()

    def generate(self, images: Sequence[Image.Image], request: CaptionRequest,
                 cancel: threading.Event | None = None,
                 prompts: Sequence[str] | None = None) -> list[str | Exception]:
        if self.client is None:
            raise InferenceError("API client not initialised.")
        from dataclasses import replace
        reqs = [replace(request, prompt=p) for p in prompts] if prompts else [request] * len(images)
        if len(images) == 1:
            return [self._one(images[0], reqs[0], cancel)]
        with ThreadPoolExecutor(max_workers=len(images), thread_name_prefix="api") as pool:
            return list(pool.map(lambda pair: self._one(pair[0], pair[1], cancel), zip(images, reqs)))
