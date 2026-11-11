"""Keeping secrets and untrusted text out of places they do not belong.

* An API key is checked before it can reach an HTTP header, and is never part of a message or a log line.
* Text that came from a server (an error message, a model name) is untrusted: before it is shown or logged it
  is cut to one short plain line, with nothing a rich-text widget could take for markup.

No Qt, no torch.
"""
from __future__ import annotations

import re

MAX_SERVER_TEXT = 300          # characters of a server's own words kept in a message
_BEARER = re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{6,}")
_KEY_PARAM = re.compile(r"(?i)\b(api[-_]?key|key|token|access_token|authorization)=[^&\s'\"]+")
_SK = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")


class BadApiKey(ValueError):
    """The key cannot be sent as it is. The message never contains the key."""


def clean_api_key(key) -> str:
    """The key with surrounding whitespace removed. Raises ``BadApiKey`` for a key that holds a line break, a
    control character or anything outside printable ASCII: such a key cannot be an HTTP header value, and the
    HTTP library's own complaint about it would quote the key."""
    key = (key or "").strip() if isinstance(key, str) else ""
    if any(not (0x21 <= ord(c) <= 0x7E) for c in key):
        raise BadApiKey("The API key contains a space, a line break or another character that cannot be sent.")
    return key


def redact(text, key: str = "") -> str:
    """``text`` with the key, any ``Bearer ...`` value and any ``key=...`` URL parameter blanked."""
    text = str(text)
    key = (key or "").strip() if isinstance(key, str) else ""
    if len(key) >= 4:
        text = text.replace(key, "[key]")
    text = _BEARER.sub(lambda m: f"{m.group(1)} [key]", text)
    text = _KEY_PARAM.sub(lambda m: f"{m.group(1)}=[key]", text)
    return _SK.sub("[key]", text)


def plain(text, key: str = "", limit: int = MAX_SERVER_TEXT) -> str:
    """Untrusted text as one short, plain line: secrets blanked, control characters and line breaks turned
    into spaces, angle brackets replaced (so it is harmless even where a label renders rich text), cut to
    ``limit`` characters."""
    text = redact(text, key)
    text = "".join(c if c.isprintable() else " " for c in text)
    text = " ".join(text.split()).replace("<", "‹").replace(">", "›")
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def url_for_display(url: str) -> str:
    """``scheme://host:port/path`` of a URL: no user name, password, query or fragment."""
    from urllib.parse import urlparse
    try:
        u = urlparse((url or "").strip())
        host = u.hostname or ""
        if not u.scheme or not host:
            return ""
        port = f":{u.port}" if u.port else ""
    except ValueError:
        return ""
    if ":" in host:
        host = f"[{host}]"
    return plain(f"{u.scheme}://{host}{port}{u.path}")
