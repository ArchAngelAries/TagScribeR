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


def is_loopback_url(url: str) -> bool:
    from urllib.parse import urlparse
    host = (urlparse(url).hostname or "").lower()
    return host == "localhost" or host == "::1" or host.startswith("127.")


def url_host(url: str) -> str:
    from urllib.parse import urlparse
    try:
        return (urlparse(url.strip()).hostname or "").lower()
    except ValueError:
        return ""


def key_exposure_warning(url: str, key: str) -> str:
    """Non-empty when an API key would travel unencrypted to a machine other than this one."""
    from urllib.parse import urlparse
    url = url.strip()
    if not key or urlparse(url).scheme.lower() != "http" or not url_host(url) or is_loopback_url(url):
        return ""
    return (f"The key would be sent unencrypted (http) to {url_host(url)}. Use https, or leave the key empty "
            "for a local server.")


# Defaults when a profile does not say: a local server handles one request at a time, so a timed-out
# request must not be sent again (the abandoned one keeps the server busy and the retry queues behind it).
DEFAULT_TIMEOUT = 180.0
CLOUD_RETRIES = 2


def effective_retries(url: str, configured) -> int:
    local = is_local_url(url)
    if configured is None:
        return 0 if local else CLOUD_RETRIES
    return max(0, min(int(configured), 1) if local else int(configured))


def reply_text(content) -> str:
    """Text of a chat-completions ``message.content``: a string, or a list of content parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    out.append(part["text"])
            elif isinstance(getattr(part, "text", None), str):
                out.append(part.text)
        return "".join(out)
    raise TypeError(f"unsupported reply content of type {type(content).__name__}")


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
                             timeout=float(self.spec.opt("timeout") or DEFAULT_TIMEOUT),
                             max_retries=effective_retries(base_url, self.spec.opt("max_retries")))
        report(f"Using API: {self.display_name}")

    def unload(self) -> None:
        client, self.client = self.client, None
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    def _abort_inflight(self, client) -> None:
        """Drop the connection(s) of ``client`` so requests in flight fail now; ``load`` rebuilds it."""
        if self.client is client:
            self.client = None
        try:
            client.close()
        except Exception:
            pass

    def _request(self, client, kwargs: dict, cancel: threading.Event | None):
        """``chat.completions.create`` that returns early (``Cancelled``) when ``cancel`` is set."""
        if cancel is None:
            return client.chat.completions.create(**kwargs)
        box: dict = {}

        def call():
            try:
                box["resp"] = client.chat.completions.create(**kwargs)
            except BaseException as e:   # handed back to the calling thread
                box["err"] = e

        t = threading.Thread(target=call, name="api-request", daemon=True)
        t.start()
        while t.is_alive():
            if cancel.wait(0.25):
                self._abort_inflight(client)
                t.join(2.0)
                raise Cancelled()
        if "err" in box:
            raise box["err"]
        return box["resp"]

    def _one(self, img: Image.Image, req: CaptionRequest, cancel: threading.Event | None) -> str | Exception:
        if cancel is not None and cancel.is_set():
            return Cancelled()
        messages = []
        if req.system_prompt.strip():
            messages.append({"role": "system", "content": req.system_prompt.strip()})
        try:                    # an image that can't be encoded fails alone (not its whole chunk)
            image_url = f"data:image/jpeg;base64,{to_jpeg_base64(img)}"
        except Exception as e:  # noqa: BLE001
            return InferenceError(f"Could not prepare the image for the server: {e}")
        messages.append({"role": "user", "content": [
            {"type": "text", "text": req.prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
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
        client = self.client
        if client is None:
            return Cancelled() if cancel is not None and cancel.is_set() else InferenceError("API client not initialised.")
        try:
            resp = self._request(client, kwargs, cancel)
        except Cancelled:
            return Cancelled()
        except Exception as e:
            if cancel is not None and cancel.is_set():
                return Cancelled()
            log.debug("API call failed", exc_info=True)
            return _friendly_error(e)
        try:
            choice = resp.choices[0]
            text = reply_text(choice.message.content)
            if req.strip_thinking:
                text = captions.clean_model_output(text)
        except Exception:
            log.debug("Malformed API reply", exc_info=True)
            return InferenceError("The API returned a malformed response.",
                                  "The server's reply did not look like an OpenAI chat completion.")
        if not text.strip():
            finish = getattr(choice, "finish_reason", "")
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
