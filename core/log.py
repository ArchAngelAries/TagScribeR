"""Logging setup: concise console output plus a detailed rotating log file.

User-facing code shows short messages; full tracebacks go to
``user_data/logs/tagscriber.log`` so problems can be diagnosed afterwards.
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from core import paths

_configured = False


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
    root.addHandler(file_handler)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(console)

    # Third-party libraries are noisy at DEBUG.
    for noisy in ("PIL", "urllib3", "httpx", "httpcore", "openai", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    def _excepthook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logging.getLogger("tagscriber").critical("Unhandled exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = _excepthook
