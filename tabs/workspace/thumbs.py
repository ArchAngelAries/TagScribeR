"""Asynchronous thumbnail loading with memory and disk caches.

* Only thumbnails that are actually painted are requested (the grid is
  virtualized), newest requests first, so scrolling fast through thousands of
  images stays responsive.
* Decoded thumbnails are cached on disk (user_data/cache/thumbnails), keyed by
  path + size + mtime, so reopening a dataset doesn't decode every image again.
* QPixmapCache holds recently shown pixmaps in memory (bounded).
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
from PySide6.QtGui import QImage, QPixmap, QPixmapCache

from core import paths
from core.image_utils import load_qimage

log = logging.getLogger(__name__)

# Stored thumbnail tiers; the grid scales down from the smallest tier that's
# at least as large as the card, so big cards (zoomed in) stay sharp.
TIERS = (384, 768)


def tier_for(display_px: int) -> int:
    return next((t for t in TIERS if t >= display_px), TIERS[-1])


def _disk_path(path: str, mtime: float, size: int, tier: int) -> Path:
    h = hashlib.sha1(f"{os.path.normcase(path)}|{mtime:.3f}|{size}|{tier}".encode("utf-8")).hexdigest()
    return paths.THUMBNAIL_CACHE_DIR / h[:2] / f"{h}.jpg"


class _Emitter(QObject):
    done = Signal(str, int, QImage, int, int)   # key, tier, image, orig width, orig height
    failed = Signal(str, int)


class _Job(QRunnable):
    def __init__(self, key: str, mtime: float, fsize: int, tier: int, emitter: _Emitter, generation: int, loader):
        super().__init__()
        self.key, self.mtime, self.fsize, self.tier = key, mtime, fsize, tier
        self.emitter, self.generation, self.loader = emitter, generation, loader

    def run(self):
        if self.generation != self.loader.generation:
            return  # folder changed while queued
        cache_file = _disk_path(self.key, self.mtime, self.fsize, self.tier)
        try:
            if cache_file.is_file():
                img = QImage(str(cache_file))
                if not img.isNull():
                    self.emitter.done.emit(self.key, self.tier, img, 0, 0)
                    return
            img, (w, h) = load_qimage(self.key, (self.tier, self.tier))
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                img.save(str(cache_file), "JPG", 88)
            except Exception as e:  # cache is an optimisation only
                log.debug("Thumbnail cache write failed: %s", e)
            self.emitter.done.emit(self.key, self.tier, img, w, h)
        except Exception as e:
            log.info("Thumbnail failed for %s: %s", self.key, e)
            self.emitter.failed.emit(self.key, self.tier)


class ThumbnailLoader(QObject):
    ready = Signal(str)                 # key: pixmap now in cache
    dimensions = Signal(str, int, int)  # key, width, height (when learnt while decoding)

    def __init__(self, parent=None):
        super().__init__(parent)
        QPixmapCache.setCacheLimit(400 * 1024)  # KB
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(max(2, (os.cpu_count() or 4) // 2))
        self.generation = 0
        self._pending: set[tuple[str, int]] = set()
        self._failed: set[str] = set()
        self._priority = 0
        self._emitter = _Emitter()
        self._emitter.done.connect(self._on_done)
        self._emitter.failed.connect(self._on_failed)

    @staticmethod
    def _cache_key(key: str, tier: int) -> str:
        return f"ws|{tier}|{key}"

    def reset(self) -> None:
        """Forget queued work (e.g. when another folder is opened)."""
        self.generation += 1
        self.pool.clear()
        self._pending.clear()
        self._failed.clear()

    def invalidate(self, key: str) -> None:
        for t in TIERS:
            QPixmapCache.remove(self._cache_key(key, t))
        self._failed.discard(key)

    def pixmap(self, key: str, mtime: float = 0.0, fsize: int = 0, display_px: int = 0) -> QPixmap | None:
        """Best available pixmap for the display size; requests a sharper tier if needed.

        While a larger tier loads, a smaller cached one is returned so cards
        never go blank when zooming in.
        """
        want = tier_for(display_px or TIERS[0])
        pm = QPixmapCache.find(self._cache_key(key, want))
        if pm is not None and not pm.isNull():
            return pm
        job_id = (key, want)
        if job_id not in self._pending and key not in self._failed:
            self._pending.add(job_id)
            self._priority += 1  # later requests (what's on screen now) run first
            self.pool.start(_Job(key, mtime, fsize, want, self._emitter, self.generation, self), self._priority)
        for t in TIERS:  # fallback: any tier we already have
            if t != want:
                fallback = QPixmapCache.find(self._cache_key(key, t))
                if fallback is not None and not fallback.isNull():
                    return fallback
        return None

    def is_failed(self, key: str) -> bool:
        return key in self._failed

    def _on_done(self, key: str, tier: int, img: QImage, w: int, h: int) -> None:
        self._pending.discard((key, tier))
        QPixmapCache.insert(self._cache_key(key, tier), QPixmap.fromImage(img))
        if w and h:
            self.dimensions.emit(key, w, h)
        self.ready.emit(key)

    def _on_failed(self, key: str, tier: int) -> None:
        self._pending.discard((key, tier))
        self._failed.add(key)
        self.ready.emit(key)
