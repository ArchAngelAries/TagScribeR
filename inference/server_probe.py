"""Ask a local model server what it is and how it is set up. Read-only, and it never loads a model.

Only ``GET`` requests to listing / status endpoints are made (nothing that names a model in a completion
request, which is what makes LM Studio or Ollama load one). Standard library only; no Qt, no torch.

* llama.cpp ``llama-server``: ``/props`` reports ``total_slots`` (parallel requests) and the context size.
* LM Studio: ``/api/v0/models`` reports each model's state (loaded or not), its loaded context length and
  whether it is a vision model. The number of parallel requests is a load setting that these endpoints do not
  reliably report, so it is found by measuring (inference/concurrency.py).
* Anything else: ``/v1/models`` (the OpenAI-compatible list).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from urllib.parse import urlparse

from inference.textsafe import BadApiKey, clean_api_key, plain

MAX_REPLY_BYTES = 2_000_000      # a model list is a few KB; never read an unbounded reply
MAX_MODELS = 500                 # ... and never keep an unbounded list of names from it


@dataclass
class ServerInfo:
    reachable: bool = False
    kind: str = "unknown"                 # llama.cpp | lmstudio | openai-compatible | unknown
    models: list[str] = field(default_factory=list)     # ids exactly as the server sent them: show as plain text
    slots: int | None = None              # parallel requests the server runs at once, when it says
    context: int | None = None            # context length of the asked-for model as loaded, when it says
    max_context: int | None = None        # the most that model supports, when the server says
    loaded: bool | None = None            # is the asked-for model in memory (None: the server does not say)
    vision: bool | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.reachable:
            return "The server did not answer."
        parts = [{"llama.cpp": "llama.cpp server", "lmstudio": "LM Studio"}.get(self.kind, "OpenAI-compatible server")]
        if self.loaded is not None:
            parts.append("model loaded" if self.loaded else "model not loaded yet")
        if self.context:
            parts.append(f"context {self.context}")
        if self.slots:
            parts.append(f"{self.slots} parallel")
        if self.vision is False:
            parts.append("not a vision model")
        return ", ".join(parts)


def server_root(base_url: str) -> str:
    """``http://host:port`` of an API base URL such as ``http://host:port/v1``."""
    u = urlparse(base_url.strip())
    return f"{u.scheme}://{u.netloc}" if u.scheme and u.netloc else ""


class _Unreachable(Exception):
    """Nothing answered at that address (no point in asking it anything else)."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A probe carries the API key: it is never sent on to wherever a redirect points."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _get_json(url: str, api_key: str, timeout: float):
    """The JSON a GET of ``url`` returns, or None. No redirects, at most ``MAX_REPLY_BYTES``, at most about
    twice ``timeout`` seconds in all; a failure says nothing (no URL, no key, no server text in a log)."""
    if urlparse(url).scheme not in ("http", "https"):
        return None
    req = urllib.request.Request(url, method="GET", headers={"Accept": "application/json",
                                                             "Accept-Encoding": "identity"})
    try:
        key = clean_api_key(api_key)
    except BadApiKey:
        key = ""                # a key that cannot be a header value is not sent (the job itself reports it)
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    deadline = time.monotonic() + 2 * timeout
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            if r.status != 200:
                return None
            data = bytearray()
            while True:
                chunk = r.read1(65536)      # what has arrived: a trickle cannot hold the loop past the deadline
                if not chunk:
                    break
                data += chunk
                if len(data) > MAX_REPLY_BYTES or time.monotonic() > deadline:
                    return None
            return json.loads(data.decode("utf-8", "replace"))
    except urllib.error.HTTPError:
        return None             # the server answered (404, a refused redirect ...): just not with this
    except (urllib.error.URLError, OSError) as e:
        raise _Unreachable() from e
    except ValueError:
        return None


def _int(value) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else None


def probe(base_url: str, model: str = "", api_key: str = "", timeout: float = 2.0) -> ServerInfo:
    """What the server at ``base_url`` reports. Safe to call while a job runs; at most three short GETs."""
    info = ServerInfo()
    root = server_root(base_url)
    if not root or urlparse(base_url).scheme not in ("http", "https"):
        return info
    try:
        return _probe(info, root, base_url, model, api_key, timeout)
    except _Unreachable:
        return info


def _probe(info: ServerInfo, root: str, base_url: str, model: str, api_key: str, timeout: float) -> ServerInfo:
    props = _get_json(f"{root}/props", api_key, timeout)
    info.reachable = True                 # something answered at that address (whatever it said)
    if isinstance(props, dict) and ("total_slots" in props or "default_generation_settings" in props):
        info.kind = "llama.cpp"
        info.slots = _int(props.get("total_slots"))
        settings = props.get("default_generation_settings")
        if isinstance(settings, dict):
            info.context = _int(settings.get("n_ctx"))      # per slot
        info.loaded = True                                   # llama-server holds exactly the model it serves
        modalities = props.get("modalities")
        if isinstance(modalities, dict) and "vision" in modalities:
            info.vision = bool(modalities["vision"])
    native = None if info.kind == "llama.cpp" else _get_json(f"{root}/api/v0/models", api_key, timeout)
    rows = native.get("data") if isinstance(native, dict) else None
    if isinstance(rows, list) and any(isinstance(r, dict) and "state" in r for r in rows):
        info.kind = "lmstudio"
        for r in rows:
            if not isinstance(r, dict) or not isinstance(r.get("id"), str):
                continue
            info.models.append(r["id"])
            if model and r["id"] == model:
                info.loaded = r.get("state") == "loaded"
                info.context = _int(r.get("loaded_context_length"))
                info.max_context = _int(r.get("max_context_length"))
                if isinstance(r.get("type"), str):
                    info.vision = r["type"] == "vlm"
    if not info.models:
        listing = _get_json(base_url.rstrip("/") + "/models", api_key, timeout)
        rows = listing.get("data") if isinstance(listing, dict) else None
        if isinstance(rows, list):
            if info.kind == "unknown":
                info.kind = "openai-compatible"
            info.models = [r["id"] for r in rows if isinstance(r, dict) and isinstance(r.get("id"), str)]
    if info.reachable and model and info.models and model not in info.models:
        info.notes.append(f"The server does not list a model named '{plain(model, limit=120)}'.")
    del info.models[MAX_MODELS:]
    return info


def advice(info: ServerInfo, image_tokens: int, prompt_tokens: int, max_new_tokens: int, parallel: int) -> list[str]:
    """Plain-language warnings when the server's reported setup cannot hold the planned requests."""
    out = list(info.notes)
    one = image_tokens + prompt_tokens + max_new_tokens + 64
    if info.vision is False:
        out.append("The server says this is not a vision model: it cannot see the pictures.")
    if info.context:
        if info.kind == "llama.cpp":
            if info.context < one:
                out.append(f"Each request needs about {one} tokens but a slot holds {info.context}: raise the "
                           "context size or send smaller pictures.")
        elif info.context < one * max(1, parallel):
            fit = max(1, info.context // one)
            out.append(f"{parallel} requests of about {one} tokens need a context of {one * parallel}; the model "
                       f"is loaded with {info.context}, which holds {fit}. Raise the context length, send "
                       "smaller pictures, or use fewer parallel requests.")
    if info.slots is not None and parallel > info.slots:
        out.append(f"The server runs {info.slots} request(s) at a time: more than that only wait in its queue.")
    return out
