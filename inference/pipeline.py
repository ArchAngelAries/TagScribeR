"""A steady stream of requests to a server: prepare ahead, keep N in flight, hand results back as they finish.

Used by the batch worker for providers whose requests are independent (``Provider.pipelined``) and by
``tools/bench_caption.py``. No Qt, no torch.

How it differs from handing the provider chunks of N images
-----------------------------------------------------------
* A chunk waits for its slowest request before the next chunk starts, so slots sit idle; here a new request
  starts the moment one finishes.
* Images are opened, resized and encoded on helper threads ahead of need, so the server never waits for the
  client. The look-ahead is bounded: at most ``limit + 1`` prepared images wait while ``limit`` are in flight,
  so at most ``2 * limit + 1`` encoded pictures (well under 1 MB each at the default size) are held at a time.
* The number in flight comes from an ``AdaptiveLimiter`` and may change while the job runs.
* A request that failed because the server was overloaded while several were in flight is sent once more
  (from the picture already encoded: nothing is read or encoded again) after the limiter has backed off.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

from inference.base import Cancelled
from inference.concurrency import AdaptiveLimiter

POLL_SECONDS = 0.25      # how often a waiting scheduler looks at the cancel and pause flags


def default_prepare_threads() -> int:
    """Threads for decode / resize / encode. Pillow releases the GIL there, so they scale; four keep a server
    that answers ten times a second fed (measured: 25 to 50 images/s for 2 to 4 MP files) without crowding
    the CPU a local server may need for its own image work."""
    return max(1, min(4, (os.cpu_count() or 2) // 2))


class PrepareError(Exception):
    """The image could not be opened or encoded (the request was never sent)."""


@dataclass
class Outcome:
    key: object                    # what the caller passed in (a path)
    result: "str | Exception"
    seconds: float = 0.0           # request time, as the client saw it
    info: dict = field(default_factory=dict)
    attempts: int = 1


@dataclass
class _Flight:
    key: object
    prepared: object
    tag: int
    t0: float
    attempts: int
    limit: int            # how many were allowed in flight when it started


def run_pipeline(keys: Iterable, prepare: Callable[[object], object],
                 caption: Callable[[object, object, threading.Event], tuple],
                 limiter: AdaptiveLimiter, cancel: threading.Event | None = None,
                 paused: threading.Event | None = None, on_start: Callable[[object], None] | None = None,
                 prepare_threads: int | None = None, clock=time.monotonic) -> Iterator[Outcome]:
    """Yield one ``Outcome`` per key, in the order requests finish.

    ``prepare(key)`` runs on helper threads and returns what ``caption(prepared, key, stop)`` needs;
    ``caption`` runs on request threads and returns ``(text or Exception, info)``. ``stop`` is set when
    ``cancel`` is set or the consumer stops iterating, so requests in flight can be dropped. Raises
    ``Cancelled`` when ``cancel`` is set. While ``paused`` is set nothing new starts (requests in flight finish).
    """
    stop = threading.Event()
    source = iter(keys)
    exhausted = False
    ready: deque[tuple[object, Future]] = deque()        # prepared or being prepared, in input order
    again: deque[tuple[object, object, int]] = deque()   # (key, prepared, attempts) to send once more
    flying: dict[Future, _Flight] = {}
    prep_pool = ThreadPoolExecutor(max_workers=prepare_threads or default_prepare_threads(),
                                   thread_name_prefix="prepare")
    req_pool = ThreadPoolExecutor(max_workers=limiter.cap, thread_name_prefix="api")
    try:
        while True:
            if cancel is not None and cancel.is_set():
                stop.set()
                for fut in [f for f in flying if f.done()]:     # already answered: not thrown away
                    flight = flying.pop(fut)
                    result, info = fut.result()
                    if not isinstance(result, Exception):
                        yield Outcome(flight.key, result, clock() - flight.t0, info, flight.attempts)
                raise Cancelled()
            limit = limiter.limit
            while not exhausted and len(ready) < limit + 1:
                try:
                    key = next(source)
                except StopIteration:
                    exhausted = True
                    break
                ready.append((key, prep_pool.submit(prepare, key)))
            holding = paused is not None and paused.is_set()
            while len(flying) < limit and not holding:
                if again:
                    key, prepared, attempts = again.popleft()
                elif ready and ready[0][1].done():
                    key, fut = ready.popleft()
                    attempts = 0
                    try:
                        prepared = fut.result()
                    except Exception as e:  # noqa: BLE001 - one unreadable image fails alone
                        yield Outcome(key, PrepareError(str(e) or type(e).__name__), attempts=0)
                        continue
                else:
                    break
                if on_start is not None:
                    on_start(key)
                fut = req_pool.submit(_guarded, caption, prepared, key, stop)
                flying[fut] = _Flight(key, prepared, limiter.begin(), clock(), attempts + 1, limit)
            if not flying and not ready and not again and exhausted:
                return
            waiting: set[Future] = set(flying)
            if ready and len(flying) < limit and not holding:
                waiting.add(ready[0][1])
            if not waiting:
                (cancel or stop).wait(POLL_SECONDS)       # paused, or the next image is still being prepared
                continue
            finished, _ = wait(waiting, timeout=POLL_SECONDS, return_when=FIRST_COMPLETED)
            for fut in finished:
                flight = flying.pop(fut, None)
                if flight is None:
                    continue                              # a prepared image: picked up at the top of the loop
                result, info = fut.result()
                seconds = clock() - flight.t0
                if isinstance(result, Cancelled):
                    if cancel is None or not cancel.is_set():
                        raise Cancelled()
                    continue          # the top of the loop first keeps what was answered, then raises
                failed = isinstance(result, Exception)
                busy = failed and bool(getattr(result, "busy", False))
                limiter.done(flight.tag, seconds, ok=not failed, size=info.get("size"), busy=busy)
                if failed and getattr(result, "resend", False) and flight.attempts == 1 and flight.limit > 1:
                    again.append((flight.key, flight.prepared, flight.attempts))
                    continue
                yield Outcome(flight.key, result, seconds, info, flight.attempts)
    finally:
        stop.set()
        for _key, fut in ready:
            fut.cancel()
        prep_pool.shutdown(wait=True, cancel_futures=True)
        req_pool.shutdown(wait=True, cancel_futures=True)


def _guarded(caption, prepared, key, stop) -> tuple:
    """``caption`` that hands an unexpected exception back as that image's result."""
    try:
        result, info = caption(prepared, key, stop)
        return result, info or {}
    except Cancelled as e:
        return e, {}
    except Exception as e:  # noqa: BLE001
        return e, {}
