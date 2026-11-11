"""Qt batch worker: runs a captioning / tagging job off the UI thread.

Guarantees
----------
* One failed image never aborts the batch; failures are collected and
  reported at the end with their reasons.
* Each result is saved (atomically, with backup) as soon as it is produced,
  so cancelling or crashing never loses completed work.
* Cancellation is checked between images *and* inside generation.
* The next chunk of images is decoded (and prepared for the model) on helper
  threads while the current chunk is on the GPU.
* A server (``Provider.pipelined``) gets a steady stream instead of chunks: a
  new request starts the moment one finishes (inference/pipeline.py).
* Models are reused across jobs via the ModelManager.
* ``pause`` holds the job between images; ``stats`` reports the live rate and
  the time left.
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
from inference.concurrency import AdaptiveLimiter, RateMeter, format_eta
from inference.image_prep import load_rgb
from inference.manager import manager
from inference.pipeline import PrepareError, default_prepare_threads, run_pipeline
from inference.textsafe import plain

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
    prompts: dict[str, str] | None = None   # per-image prompt overrides (key: str(path))
    ensure_prefix: str = ""                 # make every caption start with this (e.g. a trigger word)


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
    cut_short: int = 0              # replies stopped because they repeated themselves (kept without the repeats)
    parallel: int = 0               # requests in flight when the job ended (0: not a server job)

    def text(self) -> str:
        if self.fatal:
            return f"{self.title} could not start: {self.fatal}"
        parts = [f"{self.done} of {self.total} done"]
        if self.skipped:
            parts.append(f"{self.skipped} skipped")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        if self.cut_short:
            parts.append(f"{self.cut_short} cut short (repeating)")
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
    stats = Signal(object)                 # dict, after every result: see _report
    finished = Signal(object)              # JobSummary

    def __init__(self, job: BatchJob):
        super().__init__()
        self.job = job
        self._cancel = threading.Event()
        self._pause = threading.Event()
        self._meter: RateMeter | None = None

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def pause(self) -> None:
        """Start nothing new (what is being generated finishes and is saved) until ``resume``."""
        self._pause.set()
        if self._meter is not None:
            self._meter.pause()

    def resume(self) -> None:
        self._pause.clear()
        if self._meter is not None:
            self._meter.resume()

    @property
    def paused(self) -> bool:
        return self._pause.is_set()

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
        self._meter = RateMeter()
        if self.paused:
            self._meter.pause()
        if getattr(provider, "pipelined", False):
            self._process_stream(job, summary, provider, todo, processed)
        else:
            self._process_chunks(job, summary, provider, todo, processed)

    @staticmethod
    def _open(job: BatchJob, provider, path: Path):
        """Decode, shrink to the job's image limits and let the provider do its own per-image CPU work."""
        req = job.request
        return provider.prepare(load_rgb(path, req.max_image_side, max_pixels=req.max_image_pixels), req)

    def _report(self, summary: JobSummary, processed: int, parallel: int = 0, state: str = "") -> None:
        """Progress plus the live numbers: images per minute over the latest results and the time left."""
        self.progress.emit(processed, summary.total)
        meter = self._meter
        eta = meter.eta_seconds(summary.total - processed)
        self.stats.emit({"processed": processed, "total": summary.total, "done": summary.done,
                         "failed": len(summary.failed), "per_minute": meter.per_minute(),
                         "eta_seconds": eta, "eta_text": format_eta(eta), "parallel": parallel,
                         "parallel_state": state, "paused": self.paused})

    def _process_chunks(self, job: BatchJob, summary: JobSummary, provider, todo: list[Path], processed: int) -> None:
        chunk = max(1, provider.preferred_chunk)
        chunks = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]
        pool = ThreadPoolExecutor(max_workers=default_prepare_threads(), thread_name_prefix="decode")
        try:
            def decode(paths: list[Path]) -> list[Future]:
                return [pool.submit(self._open, job, provider, p) for p in paths]

            def opened(fut: Future):
                try:
                    return fut.result()
                except Exception as e:  # noqa: BLE001 - one unreadable image fails alone
                    return e

            pending = decode(chunks[0])
            for idx, paths in enumerate(chunks):
                while self.paused and not self.cancelled:
                    self._cancel.wait(0.25)
                if self.cancelled:
                    raise Cancelled()
                decoded = [opened(f) for f in pending]
                if idx + 1 < len(chunks):
                    pending = decode(chunks[idx + 1])

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
                    per_image = ([job.prompts.get(str(p), job.request.prompt) for p in ok_paths]
                                 if job.prompts else None)
                    self.items_started.emit([str(p) for p in ok_paths])
                    results = self._generate(provider, job, ok_images, per_image)
                for p, res in zip(ok_paths, results):
                    if isinstance(res, Cancelled):
                        raise Cancelled()
                    if isinstance(res, Exception):
                        self._fail(summary, p, str(res))
                        continue
                    self._save(job, summary, p, res)
                processed += len(paths)
                self._meter.tick(len(paths))
                self._report(summary, processed)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    def _process_stream(self, job: BatchJob, summary: JobSummary, provider, todo: list[Path], processed: int) -> None:
        """A server: keep the limiter's number of requests in flight and save each caption as it arrives."""
        make = getattr(provider, "make_limiter", None)
        limiter = make() if make is not None else AdaptiveLimiter(
            start=max(1, provider.preferred_chunk), cap=max(1, provider.preferred_chunk), adaptive=False)
        request = job.request

        def caption(prepared, path, stop):
            prompt = job.prompts.get(str(path), request.prompt) if job.prompts else None
            return provider.caption_prepared(prepared, request, stop, prompt)

        def started(path):
            self.status.emit(f"Processing {path.name}…")
            self.items_started.emit([str(path)])

        shown = limiter.limit
        if limiter.adaptive:
            self.status.emit(f"Finding the best number of parallel requests (starting with {shown}, "
                             f"at most {limiter.cap})…")
        for out in run_pipeline(todo, lambda p: self._open(job, provider, p), caption, limiter,
                                cancel=self._cancel, paused=self._pause, on_start=started):
            path, res = out.key, out.result
            if isinstance(res, PrepareError):
                self._fail(summary, path, f"could not open image: {res}")
            elif isinstance(res, InferenceError):
                self._fail(summary, path, str(res))
            elif isinstance(res, Exception):
                log.error("%s: unexpected error from the provider: %s: %s", job.title, type(res).__name__,
                          plain(res))
                self._fail(summary, path, f"Unexpected error: {type(res).__name__}: {plain(res)}")
            else:
                if out.info.get("cut"):
                    summary.cut_short += 1
                    self.status.emit(f"{path.name}: the reply started repeating itself and was cut short.")
                self._save(job, summary, path, res)
            processed += 1
            self._meter.tick()
            if limiter.limit != shown:
                shown = limiter.limit
                self.status.emit(f"Parallel requests: {limiter.describe()}")
            summary.parallel = limiter.limit
            self._report(summary, processed, limiter.limit, limiter.describe())

    def _generate(self, provider, job: BatchJob, images: list, prompts: list[str] | None) -> list:
        """provider.generate that can only fail the image(s) it was given, never the batch."""
        try:
            return list(provider.generate(images, job.request, self._cancel, prompts))
        except Cancelled:
            raise
        except InferenceError as e:
            return [e] * len(images)
        except Exception as e:
            log.exception("%s: unexpected error from the provider", job.title)
            if len(images) > 1:
                # Find the offending image: run them one at a time, each guarded the same way.
                out: list = []
                for i, img in enumerate(images):
                    out.extend(self._generate(provider, job, [img], [prompts[i]] if prompts else None))
                return out
            return [InferenceError(f"Unexpected error: {type(e).__name__}: {e}")]

    def _fail(self, summary: JobSummary, path: Path, reason: str) -> None:
        summary.failed.append((str(path), reason))
        log.info("Failed %s: %s", path, reason)
        self.item_failed.emit(str(path), reason)

    def _save(self, job: BatchJob, summary: JobSummary, path: Path, generated: str) -> None:
        prefix = job.ensure_prefix.strip()
        if prefix and not generated.lower().startswith(prefix.lower()):
            generated = f"{prefix}, {generated}"
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


class BatchHandle(QObject):
    """UI-thread relay for a BatchWorker running on another thread.

    Qt runs plain Python callables (lambdas, nested functions) connected to a
    worker's signals on the *worker's* thread. Every signal is therefore re-emitted
    from this object, which lives on the thread that started the job, so anything
    connected to the handle — lambdas included — safely runs on the UI thread.
    """
    status = Signal(str)
    progress = Signal(int, int)
    item_done = Signal(str, str)
    item_failed = Signal(str, str)
    item_skipped = Signal(str, str)
    items_started = Signal(list)
    stats = Signal(object)
    finished = Signal(object)

    def __init__(self, worker: BatchWorker):
        super().__init__()
        self._worker = worker
        for name in ("status", "progress", "item_done", "item_failed", "item_skipped", "items_started", "stats",
                     "finished"):
            getattr(worker, name).connect(getattr(self, name))  # cross-thread signal->signal: queued

    def cancel(self) -> None:
        self._worker.cancel()

    def pause(self) -> None:
        self._worker.pause()

    def resume(self) -> None:
        self._worker.resume()

    @property
    def paused(self) -> bool:
        return self._worker.paused

    @property
    def cancelled(self) -> bool:
        return self._worker.cancelled

    @property
    def job(self) -> BatchJob:
        return self._worker.job


def start_job(job: BatchJob) -> BatchHandle:
    """Start ``job`` on its own QThread and return a UI-thread handle with the
    worker's signals. References are kept until the thread finishes, so callers
    can't accidentally destroy a running QThread."""
    thread = QThread()
    thread.setObjectName(f"job-{job.title}")
    worker = BatchWorker(job)
    handle = BatchHandle(worker)   # created on the caller's (UI) thread, before the move
    worker.moveToThread(thread)
    entry = (thread, worker, handle)
    _running.add(entry)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)

    def _cleanup():
        _running.discard(entry)
        worker.deleteLater()
        thread.deleteLater()

    thread.finished.connect(_cleanup)
    thread.start()
    return handle


def any_running() -> bool:
    return any(t.isRunning() for t, *_ in _running)


def cancel_all(wait_ms: int = 15000) -> None:
    """Cancel every job and wait (at most ``wait_ms`` in total) for threads to finish (used on app exit)."""
    for _t, w, *_ in list(_running):
        w.cancel()
    deadline = time.monotonic() + wait_ms / 1000
    for t, *_ in list(_running):
        t.wait(max(1, int((deadline - time.monotonic()) * 1000)))
