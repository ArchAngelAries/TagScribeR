"""Logging setup: concise console output plus a detailed rotating log file.

User-facing code shows short messages; full tracebacks go to
``user_data/logs/tagscriber.log`` so problems can be diagnosed afterwards.

Two rules protect that file (and the console):
* ``RedactionFilter`` blanks out anything that looks like a credential before a record is written, so no
  caller has to be careful;
* ``AppOnlyDebugFilter`` lets only the app's own loggers write DEBUG / INFO lines. Every other library
  (HTTP clients, hub downloads, ...) is limited to WARNING, however it is named or renamed.
"""
from __future__ import annotations

import logging
import re
import sys
from logging.handlers import RotatingFileHandler

from core import paths

_configured = False

REDACTED = "<redacted>"

# Loggers that belong to TagScribeR; everything else is a library.
APP_LOGGER_PREFIXES = ("tagscriber", "core", "tabs", "inference", "training", "tools", "tests", "__main__", "main")

# Libraries that were noisy before the prefix rule existed; kept as an explicit floor (also covers the renamed
# httpx2 / httpcore2 of newer openai releases).
NOISY_LIBRARIES = ("PIL", "urllib3", "httpx", "httpcore", "httpx2", "httpcore2", "openai", "filelock",
                   "huggingface_hub")

_PATTERNS = [
    # Authorization: Bearer abc / {'Authorization': 'Bearer abc'} / authorization=abc
    (re.compile(r"(?i)(authorization\W{1,4})(?:(?:bearer|basic|token)\s+)?[^\s'\",;]+"), r"\1" + REDACTED),
    (re.compile(r"(?i)\b(bearer\s+)[^\s'\",;]+"), r"\1" + REDACTED),
    # api_key=..., api-key: ..., x-api-key, apikey, "api_key": "..."
    (re.compile(r"(?i)(api[_-]?key\W{1,4})[^\s'\",;&]+"), r"\1" + REDACTED),
    # ?key=..., &token=..., access_token=...
    (re.compile(r"(?i)([?&](?:key|token|access_token|auth)=)[^\s&'\"]+"), r"\1" + REDACTED),
    # sk-..., sk-ant-..., hf_..., gsk_..., xai-...
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"), REDACTED),
    (re.compile(r"\b(?:hf|gsk)_[A-Za-z0-9]{16,}"), REDACTED),
    (re.compile(r"\bxai-[A-Za-z0-9]{16,}"), REDACTED),
]


def redact(text: str) -> str:
    """Replace anything credential-shaped, and every key the secrets module knows, with a marker."""
    if not text:
        return text
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    try:
        from core import secrets
        known = secrets.known_secrets()
    except Exception:
        known = []
    for key in known:
        if key in text:
            text = text.replace(key, REDACTED)
    return text


class RedactionFilter(logging.Filter):
    """Redacts the message, its arguments and the formatted exception / stack text of every record."""

    def __init__(self):
        super().__init__()
        self._fmt = logging.Formatter()

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        record.msg = redact(msg)
        record.args = None
        if record.exc_info and not record.exc_text:
            try:
                record.exc_text = self._fmt.formatException(record.exc_info)
            except Exception:
                record.exc_text = "<exception text unavailable>"
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        if record.stack_info:
            record.stack_info = redact(record.stack_info)
        return True


class AppOnlyDebugFilter(logging.Filter):
    """DEBUG and INFO only from the app's own loggers; libraries are limited to WARNING and above."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        top = record.name.split(".", 1)[0]
        return top in APP_LOGGER_PREFIXES


def install_filters(handler: logging.Handler) -> None:
    handler.addFilter(RedactionFilter())
    handler.addFilter(AppOnlyDebugFilter())


def setup_logging(level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return
    _configured = True

    paths.ensure_user_dirs()
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    file_handler = RotatingFileHandler(
        paths.LOG_DIR / "tagscriber.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"
    ))
    install_filters(file_handler)
    root.addHandler(file_handler)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    install_filters(console)
    root.addHandler(console)

    # Third-party libraries are noisy at DEBUG (the handler filter above is the real rule; this also saves the work).
    for noisy in NOISY_LIBRARIES:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    def _excepthook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logging.getLogger("tagscriber").critical("Unhandled exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = _excepthook
