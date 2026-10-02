"""Background worker for per-file batch operations (image edits, conversions).

Same guarantees as the AI BatchWorker: off the UI thread, cancellable, one
failure never stops the batch, and a summary lists problems at the end.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal

from core.image_ops import SkipImage
from inference.worker import JobSummary

log = logging.getLogger(__name__)


class FileJob(QObject):
    progress = Signal(int, int)
    item_done = Signal(str, str)      # source, result path
    item_failed = Signal(str, str)
    item_skipped = Signal(str, str)
    finished = Signal(object)         # JobSummary

    def __init__(self, items: list[str], fn: Callable[[str], Path | None], title: str):
        super().__init__()
        self.items, self.fn, self.title = items, fn, title
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        s = JobSummary(title=self.title, total=len(self.items))
        t0 = time.monotonic()
        for i, item in enumerate(self.items):
            if self._cancel.is_set():
                s.cancelled = True
                break
            try:
                result = self.fn(item)
                s.done += 1
                self.item_done.emit(item, str(result) if result else "")
            except SkipImage as e:
                s.skipped += 1
                self.item_skipped.emit(item, str(e))
            except Exception as e:  # keep going; report at the end
                log.info("%s failed for %s: %s", self.title, item, e)
                s.failed.append((item, str(e)))
                self.item_failed.emit(item, str(e))
            self.progress.emit(i + 1, len(self.items))
        s.seconds = s.infer_seconds = time.monotonic() - t0
        self.finished.emit(s)


_running: set = set()


def start_file_job(items: list[str], fn, title: str) -> FileJob:
    thread = QThread()
    job = FileJob(items, fn, title)
    job.moveToThread(thread)
    entry = (thread, job)
    _running.add(entry)
    thread.started.connect(job.run)
    job.finished.connect(thread.quit)

    def cleanup():
        _running.discard(entry)
        job.deleteLater()
        thread.deleteLater()

    thread.finished.connect(cleanup)
    thread.start()
    return job


def any_running() -> bool:
    return any(t.isRunning() for t, _ in _running)


def cancel_all(wait_ms: int = 15000) -> None:
    for _, j in list(_running):
        j.cancel()
    for t, _ in list(_running):
        t.wait(wait_ms)
