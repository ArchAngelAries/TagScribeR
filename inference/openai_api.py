"""OpenAI-compatible chat-completions endpoints.

Covers cloud APIs and every common local server: LM Studio, Ollama (/v1),
llama.cpp ``llama-server``, KoboldCpp, vLLM, text-generation-webui. This is
also the recommended route for GGUF models (e.g. JoyCaption GGUF + mmproj):
serve them with LM Studio or llama-server and point TagScribeR at the URL.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Sequence

from PIL import Image

from core import captions
from inference.base import (CAP_BATCH, CAP_PROMPT, CAP_SAMPLING, CAP_SYSTEM_PROMPT, Cancelled,
                            CaptionRequest, InferenceError, ProgressFn, Provider, ProviderSpec)
from inference.concurrency import AdaptiveLimiter
from inference.image_prep import to_data_url
from inference.loopguard import LoopWatch, cut_loop
from inference.textsafe import BadApiKey, clean_api_key, plain, url_for_display

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
# The most requests the automatic setting will ever keep in flight when the profile does not say.
LOCAL_MAX_PARALLEL = 8
CLOUD_MAX_PARALLEL = 16
# Words in a "bad request" that mean the request did not fit, which fewer requests at once (or a smaller
# picture) can cure: llama.cpp shares the context between its parallel slots.
_NO_ROOM_WORDS = ("context", "n_ctx", "exceed", "too long", "too large", "too many tokens", "slot")
# No reply to a caption request is anywhere near this (a caption is a few KB; a long hidden reasoning trace a
# few hundred KB). A server that sends more is broken or hostile: the reply is dropped, not held in memory.
MAX_REPLY_BYTES = 8_000_000
MAX_EVENT_BYTES = 1_000_000     # one line of a streamed reply


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


class RedirectRefused(Exception):
    """The server answered with a redirect to another host (or from https to http)."""


class ReplyTooLarge(Exception):
    """The reply is larger than any caption reply could be."""


class ReplyTooSlow(Exception):
    """The reply kept trickling in past the request's time limit."""


class MalformedReply(Exception):
    """The reply is not JSON, or not a chat completion."""


class CompressedReply(Exception):
    """The reply is compressed although the request asked for none (it is not unpacked)."""


class StreamError(Exception):
    """The server reported an error in the middle of a streamed reply (its text is untrusted)."""


_OWN_ERRORS = (RedirectRefused, ReplyTooLarge, ReplyTooSlow, MalformedReply, CompressedReply)


def _guarded_body_class(base):
    """A reply body (a subclass of the HTTP library's own byte-stream type, ``base``) with a size limit and a
    time limit for the whole of it: a server that sends without end, or a byte at a time, can neither fill the
    memory nor hold the worker."""

    class GuardedBody(base):
        def __init__(self, inner, seconds: float):
            self._inner = inner
            self._deadline = time.monotonic() + max(1.0, seconds)
            self._total = 0

        def __iter__(self):
            for chunk in self._inner:
                self._total += len(chunk)
                if self._total > MAX_REPLY_BYTES:
                    raise ReplyTooLarge()
                if time.monotonic() > self._deadline:
                    raise ReplyTooSlow()
                yield chunk

        def close(self) -> None:
            self._inner.close()

    return GuardedBody


def response_guard(body_class, seconds: float):
    """The HTTP client's response hook. It runs for every answer, before the body is read (so it covers
    error replies and streamed replies as well), and

    * follows a redirect only on the same host (same scheme and port, or http to https): the request carries
      the picture, the prompt and possibly a key, and none of it goes to an address the user did not enter;
    * refuses a compressed reply (the client asks for none: a small compressed body can unpack to gigabytes);
    * puts the size and time limits of ``GuardedBody`` on the body.
    """
    def guard(response) -> None:
        if getattr(response, "has_redirect_location", False):
            try:
                old = response.request.url
                new = old.join(response.headers.get("location", ""))
                same = new.host == old.host and ((new.scheme == old.scheme and new.port == old.port)
                                                 or (old.scheme == "http" and new.scheme == "https"))
            except Exception:  # noqa: BLE001 - an address that cannot be read is not followed
                same = False
            if not same:
                raise RedirectRefused()
        if str(response.headers.get("content-encoding", "")).strip().lower() not in ("", "identity"):
            raise CompressedReply()
        response.stream = body_class(response.stream, seconds)
    return guard


def _http_client(openai, seconds: float):
    """The pooled HTTP client the OpenAI client uses, with ``response_guard`` on it; None if this version of
    the package cannot be given one."""
    make = getattr(openai, "DefaultHttpxClient", None)
    if make is None:
        return None
    import sys
    for cls in getattr(make, "__mro__", ()):
        http = sys.modules.get(cls.__module__.split(".")[0])
        base = getattr(http, "SyncByteStream", None)
        if base is not None and hasattr(http, "Client"):
            return make(event_hooks={"response": [response_guard(_guarded_body_class(base), seconds)]})
    return None


def _reply_from_json(data: bytes):
    """A chat completion's JSON as the small object tree the caller reads (``choices[0].message.content``,
    ``finish_reason``, ``usage``). Anything else raises ``MalformedReply``."""
    try:
        obj = json.loads(data)
    except ValueError as e:
        raise MalformedReply() from e
    if not isinstance(obj, dict) or not isinstance(obj.get("choices"), list) or not obj["choices"]:
        raise MalformedReply()
    choices = []
    for c in obj["choices"]:
        c = c if isinstance(c, dict) else {}
        message = c.get("message") if isinstance(c.get("message"), dict) else {}
        choices.append(SimpleNamespace(message=SimpleNamespace(content=message.get("content")),
                                       finish_reason=c.get("finish_reason") or ""))
    usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else {}
    return SimpleNamespace(choices=choices, usage=SimpleNamespace(
        prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens")))


def _friendly_error(e: Exception, key: str = "") -> InferenceError:
    """A user-presentable error. ``busy`` marks failures that fewer requests at once may cure; ``resend``
    those that may be sent once more (never a timeout: the abandoned request still occupies the server).

    Whatever the server or the HTTP library said is untrusted text: it is shown as one short plain line with
    the key blanked (``textsafe.plain``), never as it came."""
    for inner in (e, getattr(e, "__cause__", None), getattr(e, "__context__", None)):
        if isinstance(inner, _OWN_ERRORS):       # raised inside the HTTP client: the package may have wrapped it
            e = inner
            break
    if isinstance(e, RedirectRefused):
        return InferenceError("The server redirected the request to another address, which is not followed.",
                              "Enter the address the server really answers on (check http / https and the port).")
    if isinstance(e, ReplyTooLarge):
        return InferenceError("The server's reply was far too large to be a caption and was dropped.")
    if isinstance(e, CompressedReply):
        return InferenceError("The server sent a compressed reply although none was asked for; it was not unpacked.")
    if isinstance(e, ReplyTooSlow):
        return InferenceError("The API timed out.", "The reply did not finish within the time limit. Increase "
                              "the timeout, or check the server is responsive.", busy=True)
    if isinstance(e, MalformedReply):
        return InferenceError("The API returned a malformed response.",
                              "The server's reply did not look like an OpenAI chat completion.")
    if isinstance(e, StreamError):
        message = plain(e, key)
        full = any(w in message.lower() for w in _NO_ROOM_WORDS)
        return InferenceError(f"The API reported an error while answering: {message}", busy=full, resend=full)
    try:
        import openai
    except ImportError:
        return InferenceError(plain(e, key))
    if isinstance(e, openai.AuthenticationError):
        return InferenceError("The API rejected the key (401).", "Check the API key in the profile.")
    if isinstance(e, openai.PermissionDeniedError):
        return InferenceError("Access denied by the API (403).", "Your key may not have access to this model.")
    if isinstance(e, openai.NotFoundError):
        return InferenceError("Model or endpoint not found (404).",
                              "Check the base URL (usually ends in /v1) and the model name.")
    if isinstance(e, openai.RateLimitError):
        return InferenceError("Rate limited by the API (429), even after retries.",
                              "Lower concurrency or wait a moment and resume.", busy=True, resend=True)
    if isinstance(e, openai.APITimeoutError):
        return InferenceError("The API timed out.", "Increase the timeout, or check the server is responsive.",
                              busy=True)
    if isinstance(e, openai.APIConnectionError):
        return InferenceError("Could not connect to the API server.",
                              "Is LM Studio / the server running, and is the URL correct?", busy=True, resend=True)
    if isinstance(e, openai.BadRequestError):
        message = plain(getattr(e, "message", e), key)
        if any(w in message.lower() for w in _NO_ROOM_WORDS):
            return InferenceError(f"The API refused the request: {message}",
                                  "The request may not fit the model's context: raise the context length in the "
                                  "server, or lower the image size or the number of parallel requests.",
                                  busy=True, resend=True)
        return InferenceError(f"The API refused the request: {message}",
                              "The model may not accept images (pick a vision model).")
    if isinstance(e, openai.APIStatusError):
        overloaded = int(getattr(e, "status_code", 0) or 0) >= 500
        return InferenceError(f"API error {int(getattr(e, 'status_code', 0) or 0)}: "
                              f"{plain(getattr(e, 'message', e), key)}", busy=overloaded, resend=overloaded)
    return InferenceError(f"API call failed: {type(e).__name__}: {plain(e, key)}")


class OpenAICompatible(Provider):
    """Options (``ProviderSpec.options``), all optional; without them the provider asks as it always has:

    ``concurrency``    requests in flight (1).
    ``auto_parallel``  find the best number in flight by measuring, starting at ``concurrency`` (False).
    ``max_parallel``   the most the automatic setting may use (8 for a local server, 16 otherwise).
    ``image_first``    put the picture before the instruction in the message (False: instruction first).
    ``loop_guard``     stream the reply and stop one that repeats itself (False: wait for the whole reply).
    ``jpeg_quality``   quality of the picture sent (92).
    """
    capabilities = frozenset({CAP_PROMPT, CAP_SYSTEM_PROMPT, CAP_SAMPLING, CAP_BATCH})
    pipelined = True

    def __init__(self, spec: ProviderSpec):
        super().__init__(spec)
        self.client = None
        self.preferred_chunk = max(1, int(spec.opt("concurrency", 1)))
        self._totals_lock = threading.Lock()
        self.totals = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "loops_cut": 0}

    @property
    def display_name(self) -> str:
        # shown in the status line and the log: no user name, password or query string of the URL
        return f"{plain(self.spec.model)} @ {url_for_display(self.spec.opt('base_url', '') or '')}"

    def _key(self) -> str:
        """The API key, fit to be an HTTP header value ('' when there is none or it cannot be sent)."""
        try:
            return clean_api_key(self.spec.opt("api_key"))
        except BadApiKey:
            return ""

    def load(self, report: ProgressFn, cancel: threading.Event | None = None) -> None:
        if self.client is not None:
            return
        try:
            import openai
        except ImportError as e:
            raise InferenceError("The 'openai' package is missing.", "Run update.bat to install dependencies.") from e
        base_url = (self.spec.opt("base_url") or "").strip()
        if not base_url:
            raise InferenceError("No API base URL set.", "Enter the server URL, e.g. http://127.0.0.1:1234/v1")
        try:
            # Checked here, before it can reach a header: the HTTP library's own complaint about a key with a
            # space or a line break in it quotes the key.
            key = clean_api_key(self.spec.opt("api_key"))
        except BadApiKey as e:
            raise InferenceError(str(e), "Paste the key again, without spaces or line breaks.") from None
        timeout = float(self.spec.opt("timeout") or DEFAULT_TIMEOUT)
        extra = {}
        http = _http_client(openai, timeout)
        if http is not None:
            # "identity": no compressed replies (see response_guard); a caption reply is a few KB anyway
            extra = {"http_client": http, "default_headers": {"Accept-Encoding": "identity"}}
        # Local servers accept any key, but the client requires a non-empty one. One client per server: its
        # connection pool keeps connections alive between requests (no new TCP / TLS setup per image).
        self.client = openai.OpenAI(base_url=base_url, api_key=key or "not-needed", timeout=timeout,
                                    max_retries=effective_retries(base_url, self.spec.opt("max_retries")), **extra)
        report(f"Using API: {self.display_name}")

    def unload(self) -> None:
        client, self.client = self.client, None
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    # -- scheduling ---------------------------------------------------------------------------------------------
    def make_limiter(self) -> AdaptiveLimiter:
        """How many requests the worker keeps in flight for this job: the number the user set, or, with
        ``auto_parallel``, a number found by measuring (inference/concurrency.py)."""
        start = max(1, int(self.spec.opt("concurrency", 1)))
        if not self.spec.opt("auto_parallel", False):
            return AdaptiveLimiter(start=start, cap=start, adaptive=False)
        base_url = self.spec.opt("base_url", "") or ""
        cap = int(self.spec.opt("max_parallel") or 0) or (LOCAL_MAX_PARALLEL if is_local_url(base_url)
                                                           else CLOUD_MAX_PARALLEL)
        cap = max(1, cap)
        if is_local_url(base_url):
            # A llama.cpp server says how many requests it runs at once: no point in trying more than that.
            try:
                from inference.server_probe import probe
                slots = probe(base_url, self.spec.model, self._key(), timeout=1.5).slots
            except Exception:   # noqa: BLE001 - advice only
                slots = None
            if slots:
                cap = max(1, min(cap, slots))
        return AdaptiveLimiter(start=min(start, cap), cap=cap, adaptive=True,
                               timeout=float(self.spec.opt("timeout") or DEFAULT_TIMEOUT))

    def prepare(self, image, request: CaptionRequest) -> str:
        """The picture as a ``data:`` URL (JPEG + base64). Done on a helper thread, once per image."""
        return to_data_url(image, int(self.spec.opt("jpeg_quality") or 92))

    def caption_prepared(self, prepared, request: CaptionRequest, cancel: threading.Event | None = None,
                         prompt: str | None = None) -> tuple[str | Exception, dict]:
        if prompt is not None and prompt != request.prompt:
            from dataclasses import replace
            request = replace(request, prompt=prompt)
        return self._one_info(prepared, request, cancel)

    # -- one request --------------------------------------------------------------------------------------------
    def _abort_inflight(self, client) -> None:
        """Drop the connection(s) of ``client`` so requests in flight fail now; ``load`` rebuilds it."""
        if self.client is client:
            self.client = None
        try:
            client.close()
        except Exception:
            pass

    def _cancellable(self, client, call, cancel: threading.Event | None):
        """``call()`` that returns early (``Cancelled``) when ``cancel`` is set."""
        if cancel is None:
            return call()
        box: dict = {}

        def run():
            try:
                box["resp"] = call()
            except BaseException as e:   # handed back to the calling thread
                box["err"] = e

        t = threading.Thread(target=run, name="api-request", daemon=True)
        t.start()
        while True:
            t.join(0.25)                 # returns the moment the reply is in: no polling delay per request
            if not t.is_alive():
                break
            if cancel.is_set():
                self._abort_inflight(client)
                t.join(2.0)
                raise Cancelled()
        if "err" in box:
            raise box["err"]
        return box["resp"]

    def _request(self, client, kwargs: dict, cancel: threading.Event | None):
        """``chat.completions.create`` that returns early (``Cancelled``) when ``cancel`` is set."""
        return self._cancellable(client, lambda: self._read_reply(client, kwargs), cancel)

    @staticmethod
    def _read_reply(client, kwargs: dict):
        """The whole reply. Read from the raw bytes, which ``response_guard`` limits in size and time, and
        taken apart here: nothing larger than the limit is ever parsed."""
        raw = getattr(client.chat.completions, "with_streaming_response", None)
        if raw is None:
            return client.chat.completions.create(**kwargs)      # a client without raw access to the reply
        with raw.create(**kwargs) as response:
            data = b"".join(response.iter_bytes())
        return _reply_from_json(data)

    @staticmethod
    def _read_stream(client, kwargs: dict):
        """A streamed reply, closed early if it starts repeating itself. Returns ``(text, finish_reason,
        looped, seconds to the first text)``, or the reply itself when the server answered without streaming.
        The same size and time limits apply, and one event may not exceed ``MAX_EVENT_BYTES``."""
        raw = getattr(client.chat.completions, "with_streaming_response", None)
        if raw is None:
            return client.chat.completions.create(**kwargs)
        t0 = time.monotonic()
        watch, finish, looped, first = LoopWatch(), "", False, None
        with raw.create(stream=True, **kwargs) as response:      # leaving the block closes the connection,
            body = response.iter_bytes()                         # which is what stops the server's generation
            if "text/event-stream" not in str(response.headers.get("content-type", "")).lower():
                return _reply_from_json(b"".join(body))
            pending = b""
            for chunk in body:
                pending += chunk
                *lines, pending = pending.split(b"\n")
                if len(pending) > MAX_EVENT_BYTES:
                    raise ReplyTooLarge()
                for line in lines:
                    line = line.strip()
                    if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                        continue
                    try:
                        event = json.loads(line[5:])
                    except ValueError:
                        continue
                    if isinstance(event, dict) and event.get("error"):
                        error = event["error"]
                        raise StreamError(error.get("message", "") if isinstance(error, dict) else error)
                    try:
                        choice = event["choices"][0]
                    except (LookupError, TypeError):
                        continue                                # a keep-alive, a usage-only or an odd event
                    if not isinstance(choice, dict):
                        continue
                    finish = choice.get("finish_reason") or finish
                    delta = choice.get("delta")
                    content = delta.get("content") if isinstance(delta, dict) else None
                    if not content:
                        continue
                    if first is None:
                        first = time.monotonic() - t0
                    if watch.feed(reply_text(content)):
                        looped = True
                        break
                if looped:
                    break
        return watch.text, finish, looped, first

    def _messages(self, image_url: str, req: CaptionRequest) -> list[dict]:
        messages = []
        if req.system_prompt.strip():
            messages.append({"role": "system", "content": req.system_prompt.strip()})
        text = {"type": "text", "text": req.prompt}
        image = {"type": "image_url", "image_url": {"url": image_url}}
        # Instruction first: what every request shares comes first, which is the part a server's prompt cache
        # can reuse. Picture first is the layout most vision models were trained with; it is an option.
        messages.append({"role": "user",
                         "content": [image, text] if self.spec.opt("image_first", False) else [text, image]})
        return messages

    def _one(self, img, req: CaptionRequest, cancel: threading.Event | None) -> str | Exception:
        return self._one_info(img, req, cancel)[0]

    def _one_info(self, img, req: CaptionRequest, cancel: threading.Event | None) -> tuple[str | Exception, dict]:
        """One caption for ``img`` (a picture, or the ``data:`` URL ``prepare`` made of it), and what the
        scheduler wants to know about the request: ``size`` (output length), token counts, ``cut``."""
        info: dict = {}
        if cancel is not None and cancel.is_set():
            return Cancelled(), info
        if isinstance(img, str):
            image_url = img
        else:
            try:                    # an image that can't be encoded fails alone (not its whole chunk)
                image_url = self.prepare(img, req)
            except Exception as e:  # noqa: BLE001
                return InferenceError(f"Could not prepare the image for the server: {plain(e)}"), info
        p = req.params
        kwargs = {"model": self.spec.model, "messages": self._messages(image_url, req),
                  "max_tokens": p.max_new_tokens, "temperature": p.temperature, "top_p": p.top_p}
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
            return (Cancelled() if cancel is not None and cancel.is_set()
                    else InferenceError("API client not initialised.")), info
        try:
            if self.spec.opt("loop_guard", False):
                resp = self._cancellable(client, lambda: self._read_stream(client, kwargs), cancel)
            else:
                resp = self._request(client, kwargs, cancel)
        except Cancelled:
            return Cancelled(), info
        except Exception as e:
            if cancel is not None and cancel.is_set():
                return Cancelled(), info
            # Never the traceback or the raw text: an HTTP library's exception can quote request headers.
            log.debug("API call failed: %s: %s", type(e).__name__, plain(e, self._key()))
            return _friendly_error(e, self._key()), info
        looped = False
        try:
            if isinstance(resp, tuple):                 # a streamed reply
                text, finish, looped, first = resp
                info["size"] = len(text)
                if first is not None:
                    info["first_text_seconds"] = first
                if looped:
                    text = cut_loop(text) or text
            else:
                choice = resp.choices[0]
                finish = getattr(choice, "finish_reason", "")
                text = reply_text(choice.message.content)
                usage = getattr(resp, "usage", None)
                for name in ("prompt_tokens", "completion_tokens"):
                    value = getattr(usage, name, None)
                    if isinstance(value, int) and not isinstance(value, bool):
                        info[name] = value
                info["size"] = info.get("completion_tokens") or len(text)
            if req.strip_thinking:
                text = captions.clean_model_output(text)
        except Exception as e:
            log.debug("Malformed API reply: %s", type(e).__name__)
            return InferenceError("The API returned a malformed response.",
                                  "The server's reply did not look like an OpenAI chat completion."), info
        info["cut"] = looped
        with self._totals_lock:
            self.totals["requests"] += 1
            self.totals["loops_cut"] += int(looped)
            for name in ("prompt_tokens", "completion_tokens"):
                self.totals[name] += info.get(name, 0)
        if not text.strip():
            if looped:
                return InferenceError("The model repeated itself without giving a caption.",
                                      "Raise the repetition penalty a little, or try another model."), info
            hint = "Increase Max tokens (reasoning models spend tokens thinking)." if finish == "length" else ""
            return InferenceError("The API returned an empty caption.", hint), info
        return text.strip(), info

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
