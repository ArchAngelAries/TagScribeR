"""Keeps loaded models alive between batches and controls memory.

Model initialisation is the most expensive operation in the app (tens of
seconds and many GB), so providers are cached by their ``ProviderSpec``:

* At most one *heavy* local VLM is resident; requesting a different one
  unloads the previous first (so two 8B models never compete for VRAM).
* Taggers and API clients are light and cached independently.
* ``unload_if_idle`` lets the UI free VRAM after a period of inactivity.
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager

from core import hardware
from inference.base import ProgressFn, Provider, ProviderSpec

log = logging.getLogger(__name__)


def create_provider(spec: ProviderSpec) -> Provider:
    if spec.kind == "transformers":
        from inference.transformers_vlm import TransformersVLM
        return TransformersVLM(spec)
    if spec.kind == "openai":
        from inference.openai_api import OpenAICompatible
        return OpenAICompatible(spec)
    if spec.kind == "wd_tagger":
        from inference.wd_tagger import WDTagger
        return WDTagger(spec)
    raise ValueError(f"Unknown provider kind: {spec.kind}")


def _cache_key(spec: ProviderSpec) -> tuple:
    # Options that only affect per-call behaviour must not force a reload.
    volatile = {"batch_size", "concurrency", "format"}
    return (spec.kind, spec.model, tuple((k, v) for k, v in spec.options if k not in volatile))


class ModelManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._heavy: tuple[tuple, Provider] | None = None
        self._light: dict[tuple, Provider] = {}
        self._last_used = time.monotonic()

    def acquire(self, spec: ProviderSpec, report: ProgressFn, cancel: threading.Event | None = None) -> Provider:
        """Return a loaded provider for ``spec``, reusing a cached one when possible."""
        key = _cache_key(spec)
        with self._lock:
            self._last_used = time.monotonic()
            heavy = spec.kind == "transformers"
            if heavy and self._heavy and self._heavy[0] == key:
                prov = self._heavy[1]
                prov.preferred_chunk = max(1, int(spec.opt("batch_size", 1)))
                if prov.loaded:
                    report(f"Reusing loaded model: {prov.display_name}")
                    return prov
            elif not heavy and key in self._light:
                prov = self._light[key]
                self._apply_volatile(prov, spec)
                prov.load(report, cancel)
                return prov

            prov = create_provider(spec)
            if heavy:
                self._unload_heavy()
                self._heavy = (key, prov)
            else:
                self._light[key] = prov
            try:
                prov.load(report, cancel)
            except BaseException:
                if heavy:
                    self._unload_heavy()
                else:
                    self._light.pop(key, None)
                raise
            return prov

    @staticmethod
    def _apply_volatile(prov: Provider, spec: ProviderSpec) -> None:
        prov.spec = spec
        if spec.kind == "openai":
            prov.preferred_chunk = max(1, int(spec.opt("concurrency", 1)))
        if spec.kind == "wd_tagger":
            from inference.wd_tagger import TagFormat
            prov.fmt = TagFormat(**dict(spec.opt("format", ()) or ()))

    @contextmanager
    def session(self, spec: ProviderSpec, report: ProgressFn, cancel: threading.Event | None = None):
        """Hold the manager for a whole job so models can't be unloaded mid-batch.

        Called from worker threads only; jobs therefore run one at a time, which
        also avoids two jobs fighting over the same GPU.
        """
        with self._lock:
            prov = self.acquire(spec, report, cancel)
            try:
                yield prov
            finally:
                self._last_used = time.monotonic()

    def busy(self) -> bool:
        if self._lock.acquire(blocking=False):
            self._lock.release()
            return False
        return True

    def touch(self) -> None:
        self._last_used = time.monotonic()

    def _unload_heavy(self) -> None:
        if self._heavy:
            _, prov = self._heavy
            self._heavy = None
            log.info("Unloading %s", prov.display_name)
            prov.unload()
            hardware.free_memory()

    def loaded_model_name(self) -> str | None:
        heavy = self._heavy  # lock-free read; safe for status display
        if heavy and heavy[1].loaded:
            return heavy[1].display_name
        return None

    def unload_all(self, blocking: bool = False) -> bool:
        """Free every model. Returns False (and does nothing) while a job is running."""
        if not self._lock.acquire(blocking=blocking):
            return False
        try:
            self._unload_heavy()
            for prov in self._light.values():
                prov.unload()
            self._light.clear()
            hardware.free_memory()
            return True
        finally:
            self._lock.release()

    def unload_if_idle(self, minutes: float) -> bool:
        """Free the heavy model if unused for ``minutes``. Returns True if it unloaded."""
        if minutes <= 0:
            return False
        acquired = self._lock.acquire(blocking=False)  # never block the UI on a running job
        if not acquired:
            return False
        try:
            if self._heavy and time.monotonic() - self._last_used > minutes * 60:
                self._unload_heavy()
                return True
            return False
        finally:
            self._lock.release()


_manager: ModelManager | None = None


def manager() -> ModelManager:
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager
