"""Qt batch worker: runs a captioning / tagging job off the UI thread.

Guarantees
----------
* One failed image never aborts the batch; failures are collected and
  reported at the end with their reasons.
* Each result is saved (atomically, with backup) as soon as it is produced,
  so cancelling or crashing never loses completed work.
* Cancellation is checked between images *and* inside generation.
* The next chunk of images is decoded on a helper thread while the current
  chunk is on the GPU.
* Models are reused across jobs via the ModelManager.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from core import caption_io, captions
from inference.base import Cancelled, CaptionRequest, InferenceError, ProviderSpec
from inference.image_prep import load_rgb
from inference.manager import manager

log = logging.getLogger(__name__)

SAVE_OVERWRITE = "overwrite"
SAVE_SKIP_EXISTING = "skip_existing"
SAVE_APPEND = "append"
SAVE_PREPEND = "prepend"
SAVE_NONE = "none"          # generate only; the UI decides what to keep


def combine(existing: str, generated: str, mode: str, tag_mode: bool,
            prepend: tuple[str, ...] = (), append: tuple[str, ...] = (), sep: str = ",") -> str:
    """Merge a generated caption with the existing one according to the save mode."""
    if tag_mode:
        gen_tags = captions.split_tags(generated, sep)
        if mode == SAVE_PREPEND:
            merged = captions.merge_generated_tags("", gen_tags + captions.split_tags(existing, sep),
                                                   "overwrite", prepend=prepend, append=append, sep=sep)
            return merged
        m = "append" if mode == SAVE_APPEND else "overwrite"
        return captions.merge_generated_tags(existing, gen_tags, m, prepend=prepend, append=append, sep=sep)
    existing = existing.strip()
    if mode == SAVE_APPEND and existing:
        return f"{existing}{sep} {generated}" if captions.looks_like_tags(existing, sep) else f"{existing} {generated}"
    if mode == SAVE_PREPEND and existing:
        return f"{generated}{sep} {existing}" if captions.looks_like_tags(existing, sep) else f"{generated} {existing}"
    return generated


@dataclass
class BatchJob:
    paths: list[Path]
    spec: ProviderSpec
    request: CaptionRequest
    save_mode: str = SAVE_OVERWRITE
    tag_mode: bool = False               # outputs are tag lists (tagger / tag prompts)
    prepend_tags: tuple[str, ...] = ()
    append_tags: tuple[str, ...] = ()
    caption_ext: str = ".txt"
    title: str = "Captioning"


@dataclass
class JobSummary:
    title: str
    total: int = 0
    done: int = 0
    skipped: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    cancelled: bool = False
    fatal: str = ""
    seconds: float = 0.0            # wall time including model load
    infer_seconds: float = 0.0      # time spent after the model was ready

    def text(self) -> str:
        if self.fatal:
            return f"{self.title} could not start: {self.fatal}"
        parts = [f"{self.done} of {self.total} done"]
        if self.skipped:
            parts.append(f"{self.skipped} skipped")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        state = "cancelled" if self.cancelled else "finished"
        rate = f" ({self.infer_seconds / self.done:.1f}s/image)" if self.done else ""
        return f"{self.title} {state}: " + ", ".join(parts) + rate


class BatchWorker(QObject):
    status = Signal(str)
    progress = Signal(int, int)            # processed, total
    item_done = Signal(str, str)           # path, caption text (as saved, or generated if SAVE_NONE)
    item_failed = Signal(str, str)         # path, reason
    item_skipped = Signal(str, str)        # path, reason
    items_started = Signal(list)           # paths now being processed (for live UI badges)
    finished = Signal(object)              # JobSummary

    def __init__(self, job: BatchJob):
        super().__init__()
        self.job = job
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def run(self) -> None:
        job = self.job
        summary = JobSummary(title=job.title, total=len(job.paths))
        t0 = time.monotonic()
        try:
            self._run(job, summary)
        except Cancelled:
            summary.cancelled = True
        except InferenceError as e:
            summary.fatal = str(e)
            log.warning("%s failed: %s", job.title, e)
        except Exception as e:  # last-resort guard: never let a worker thread die silently
            summary.fatal = f"Unexpected error: {e}"
            log.exception("%s crashed", job.title)
        summary.cancelled = summary.cancelled or self.cancelled
        summary.seconds = time.monotonic() - t0
        self.finished.emit(summary)

    def _run(self, job: BatchJob, summary: JobSummary) -> None:
        todo: list[Path] = []
        for p in job.paths:
            if job.save_mode == SAVE_SKIP_EXISTING and caption_io.has_caption(p, job.caption_ext):
                summary.skipped += 1
                self.item_skipped.emit(str(p), "already has a caption")
            else:
                todo.append(p)
        processed = summary.skipped
        self.progress.emit(processed, summary.total)
        if not todo:
            return

        with manager().session(job.spec, self.status.emit, self._cancel) as provider:
            t_ready = time.monotonic()
            try:
                self._process(job, summary, provider, todo, processed)
            finally:
                summary.infer_seconds = time.monotonic() - t_ready

    def _process(self, job: BatchJob, summary: JobSummary, provider, todo: list[Path], processed: int) -> None:
        chunk = max(1, provider.preferred_chunk)
        chunks = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]
        max_side = job.request.max_image_side
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="decode") as pool:
            def decode(paths: list[Path]) -> list:
                out = []
                for p in paths:
                    try:
                        out.append(load_rgb(p, max_side))
                    except Exception as e:
                        out.append(e)
                return out

            pending: Future = pool.submit(decode, chunks[0])
            for idx, paths in enumerate(chunks):
                if self.cancelled:
                    raise Cancelled()
                decoded = pending.result()
                if idx + 1 < len(chunks):
                    pending = pool.submit(decode, chunks[idx + 1])

                ok_paths, ok_images = [], []
                for p, img in zip(paths, decoded):
                    if isinstance(img, Exception):
                        self._fail(summary, p, f"could not open image: {img}")
                    else:
                        ok_paths.append(p)
                        ok_images.append(img)
                results: list = []
                if ok_images:
                    self.status.emit(f"Processing {ok_paths[0].name}"
                                     + (f" (+{len(ok_paths) - 1})" if len(ok_paths) > 1 else "") + "…")
                    try:
                        results = provider.generate(ok_images, job.request, self._cancel)
                    except Cancelled:
                        raise
                    except InferenceError as e:
                        results = [e] * len(ok_images)
                for p, res in zip(ok_paths, results):
                    if isinstance(res, Cancelled):
                        raise Cancelled()
                    if isinstance(res, Exception):
                        self._fail(summary, p, str(res))
                        continue
                    self._save(job, summary, p, res)
                processed += len(paths)
                self.progress.emit(processed, summary.total)

    def _fail(self, summary: JobSummary, path: Path, reason: str) -> None:
        summary.failed.append((str(path), reason))
        log.info("Failed %s: %s", path, reason)
        self.item_failed.emit(str(path), reason)

    def _save(self, job: BatchJob, summary: JobSummary, path: Path, generated: str) -> None:
        if job.save_mode == SAVE_NONE:
            summary.done += 1
            self.item_done.emit(str(path), generated)
            return
        existing = caption_io.read_caption(path, job.caption_ext)
        mode = SAVE_OVERWRITE if job.save_mode == SAVE_SKIP_EXISTING else job.save_mode
        final = combine(existing, generated, mode, job.tag_mode, job.prepend_tags, job.append_tags)
        try:
            caption_io.write_caption(path, final, job.caption_ext)
        except OSError as e:
            self._fail(summary, path, f"could not save caption: {e}")
            return
        summary.done += 1
        self.item_done.emit(str(path), final)


# -- thread lifecycle helper --------------------------------------------------
_running: set[tuple[QThread, BatchWorker]] = set()


def start_job(job: BatchJob) -> BatchWorker:
    """Start ``job`` on its own QThread. Connect to the returned worker's signals
    immediately (they are queued, so nothing is missed). References are kept
    until the thread finishes, so callers can't accidentally destroy a running
    QThread."""
    thread = QThread()
    thread.setObjectName(f"job-{job.title}")
    worker = BatchWorker(job)
    worker.moveToThread(thread)
    entry = (thread, worker)
    _running.add(entry)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)

    def _cleanup():
        _running.discard(entry)
        worker.deleteLater()
        thread.deleteLater()

    thread.finished.connect(_cleanup)
    thread.start()
    return worker


def any_running() -> bool:
    return any(t.isRunning() for t, _ in _running)


def cancel_all(wait_ms: int = 15000) -> None:
    """Cancel every job and wait for threads to finish (used on app exit)."""
    for _, w in list(_running):
        w.cancel()
    for t, _ in list(_running):
        t.wait(wait_ms)
