"""How many requests to keep in flight, found by measuring; and a live rate / ETA meter.

No Qt, no torch. ``AdaptiveLimiter`` is fed by one scheduler thread (inference/pipeline.py): it is told when a
request starts and when it ends, and answers with ``limit``, the number allowed in flight right now.

Why latency and not a count of finished images
----------------------------------------------
With N requests in flight all the time, images per second = N / (mean time per request) (Little's law). On a
server with S slots the time per request stays about the same up to N = S, so the rate grows with N; past S the
extra requests only wait in the server's queue, the time per request grows as N / S and the rate stays flat.
The limiter therefore compares ``N / latency`` between neighbouring values of N.

Caption length is the main noise in a request's time, so each window's latency is read at a common output
length (a least-squares slope of time against length inside the window), and what the length does not explain
is the window's measurement error. While requests run side by side that error is about 1 %; once they queue
(a request then waits for the lengths of *other* replies) it is several times larger. A higher level is kept
only when it is better by ``min_gain`` *plus* that error, so a queueing level, which is never truly better,
does not win by luck.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque

MIN_WINDOW = 6            # finished requests before a level is judged (3 x level when that is more)
REPROBE_FIRST = 120       # finished requests at a settled level before its neighbours are tried again
REPROBE_MAX = 2000        # ... doubling after each try that changed nothing, up to this
DRIFT = 1.6               # a settled level whose requests take this many times longer than when it was accepted
DRIFT_WINDOWS = 2         # ... for this many windows in a row is re-examined (one fewer is tried)
ERROR_MARGIN = 2.5        # measurement errors a gain must clear on top of ``min_gain``


class _Window:
    """Requests finished at one level: enough to read their mean time at a given output length."""

    def __init__(self, level: int, started: float = 0.0, area: float = 0.0):
        self.level = level
        self.n = 0
        self.occupancy = float(level)   # mean number in flight while the window was filled (set when it closes)
        self.started, self.area = started, area
        self._t = self._tt = self._s = self._ts = self._ss = 0.0
        self._sized = 0

    def add(self, latency: float, size: float | None) -> None:
        self.n += 1
        self._t += latency
        self._tt += latency * latency
        if size is not None:
            self._sized += 1
            self._s += size
            self._ts += latency * size
            self._ss += size * size

    @property
    def mean_latency(self) -> float:
        return self._t / self.n if self.n else 0.0

    @property
    def mean_size(self) -> float | None:
        return self._s / self.n if self.n and self._sized == self.n else None

    def _slope(self) -> tuple[float, float]:
        """(seconds per unit of output length, variance of the lengths); (0, 0) when it cannot be told."""
        mean_t, mean_s = self.mean_latency, self.mean_size
        if mean_s is None or self.n < 3 or mean_s <= 0:
            return 0.0, 0.0
        var = self._ss / self.n - mean_s * mean_s
        if var <= 1e-9 * max(1.0, mean_s * mean_s):
            return 0.0, 0.0
        slope = (self._ts / self.n - mean_t * mean_s) / var
        return min(max(slope, 0.0), mean_t / mean_s), var     # between "length costs nothing" and "all of it"

    def latency_at(self, size: float | None) -> float:
        """Mean time per request, moved along the window's own time-per-length slope to ``size``."""
        mean_t, mean_s = self.mean_latency, self.mean_size
        if size is None or mean_s is None:
            return mean_t
        return max(1e-6, mean_t + self._slope()[0] * (size - mean_s))

    def error(self) -> float:
        """Standard error of the mean time per request, relative: what output length does not explain."""
        if self.n < 3:
            return 0.5
        mean_t = self.mean_latency
        slope, var_s = self._slope()
        residual = max(0.0, self._tt / self.n - mean_t * mean_t - slope * slope * var_s)
        return math.sqrt(residual / (self.n - 1)) / max(1e-6, mean_t)

    def rate(self, size: float | None = None) -> float:
        """Requests per second at this level: mean number in flight / mean time per request."""
        if not self.n:
            return 0.0
        return self.occupancy / max(1e-6, self.latency_at(size))


class AdaptiveLimiter:
    """The number of requests allowed in flight.

    ``adaptive=False``: always ``start`` (what the user set). ``adaptive=True``: starts at ``start``, never
    leaves ``floor``..``cap``, and

    * tries one more when the current level is measured, and keeps it only if the rate rose by ``min_gain``
      beyond the measurement error (a one-slot server shows no gain, so the limiter goes back and stays);
    * started above ``floor``, first checks that one fewer is really slower (a number chosen too high for the
      server is walked down);
    * looks at both neighbours of a settled level again after ``REPROBE_FIRST`` results, then ever more rarely;
    * halves on a ``busy`` failure (timeout, refused connection, 429, 5xx, context overflow) and does not
      return to a level that failed twice;
    * tries one fewer when requests at a settled level become much slower than when it was accepted;
    * does not try one more when waiting in the server's queue could run a request into ``timeout``.
    """

    def __init__(self, start: int = 1, floor: int = 1, cap: int = 8, adaptive: bool = True,
                 min_gain: float = 0.05, timeout: float | None = None, clock=time.monotonic):
        self.floor = max(1, int(floor))
        self.cap = max(self.floor, int(cap))
        self.limit = min(self.cap, max(self.floor, int(start)))
        self.adaptive = bool(adaptive) and self.cap > self.floor
        self.min_gain = float(min_gain)
        self.timeout = timeout
        self.clock = clock
        self.events: deque[str] = deque(maxlen=50)      # decisions, newest last (for the log and the tests)
        self._lock = threading.Lock()
        self._epoch = 0
        self._flying = 0
        self._area = 0.0                      # requests in flight, summed over time (for the mean number in flight)
        self._seen = clock()
        self._mode = "measure"                # measure | up | down | hold
        self._win = self._window(self.limit)
        self._home: _Window | None = None     # the accepted level's latest window
        self._ceiling = self.cap              # no upward try beyond this until the next look around
        self._hard_cap = self.cap             # levels that failed twice are never tried again in this job
        self._strikes: dict[int, int] = {}
        self._reprobe = REPROBE_FIRST
        self._hold_left = 0
        self._ref_latency: float | None = None
        self._slow = 0
        self._down_steps = 0                  # levels given up in the current walk down
        self._check_down = self.adaptive and self.limit > self.floor

    # -- the scheduler's side ----------------------------------------------------------------------------------
    def _window(self, level: int) -> _Window:
        return _Window(level, self._seen, self._area)

    def _tick(self) -> None:
        now = self.clock()
        self._area += self._flying * (now - self._seen)
        self._seen = now

    def begin(self) -> int:
        """A request starts now; pass the result to ``done`` when it ends."""
        with self._lock:
            self._tick()
            self._flying += 1
            return self._epoch

    def done(self, tag: int, latency: float, ok: bool = True, size: float | None = None,
             busy: bool = False) -> None:
        """The request that got ``tag`` from ``begin`` ended after ``latency`` seconds. ``size``: its output
        length (characters or tokens), when known. ``busy``: it failed because the server was overloaded."""
        with self._lock:
            self._tick()
            self._flying = max(0, self._flying - 1)
            if not self.adaptive or tag != self._epoch:
                return                        # started under another level: says nothing about this one
            if not ok:
                if busy:
                    self._back_off()
                return                        # a failed request's time is not a measurement
            win = self._win
            win.add(latency, size)
            self._hold_left -= 1
            if win.n >= max(MIN_WINDOW, 3 * win.level):
                span = self._seen - win.started
                if span > 0:                  # a client that cannot keep the level filled is not credited with it
                    win.occupancy = min(float(win.level), max(1.0, (self._area - win.area) / span))
                self._decide()

    def describe(self) -> str:
        if not self.adaptive:
            return f"{self.limit} at a time"
        state = {"measure": "measuring", "up": "trying one more", "down": "trying one fewer",
                 "hold": "settled"}[self._mode]
        return f"{self.limit} at a time ({state})"

    @property
    def settled(self) -> bool:
        return not self.adaptive or self._mode == "hold"

    # -- decisions ---------------------------------------------------------------------------------------------
    def _set(self, level: int, mode: str, why: str) -> None:
        level = min(self.cap, max(self.floor, level))
        if level != self.limit:
            self.events.append(f"{self.limit} -> {level}: {why}")
            self._epoch += 1
        self.limit = level
        self._mode = mode
        self._win = self._window(level)

    def _better(self, high: _Window, low: _Window) -> bool:
        """Is the higher level worth it: more images per second by ``min_gain``, beyond the measurement error?"""
        a, b = high.mean_size, low.mean_size
        size = None if a is None or b is None else (a + b) / 2
        margin = ERROR_MARGIN * math.hypot(high.error(), low.error())
        return high.rate(size) >= low.rate(size) * (1.0 + self.min_gain + margin)

    def _room_above(self) -> bool:
        if self.limit >= min(self._ceiling, self._hard_cap):
            return False
        if self.timeout and self._home is not None:
            # worst case for one more: no free slot, so every request waits (limit + 1) / limit as long
            if self._home.mean_latency * (self.limit + 1) / self.limit > 0.5 * self.timeout:
                return False
        return True

    def _settle(self, home: _Window) -> None:
        self._home = home
        self._ref_latency = home.mean_latency
        self._slow = 0

    def _hold(self, rarer: bool = False) -> None:
        self._mode = "hold"
        self._hold_left = self._reprobe
        if rarer:
            self._reprobe = min(REPROBE_MAX, self._reprobe * 2)
        self._win = self._window(self.limit)

    def _up_or_hold(self) -> None:
        if self._room_above():
            self._set(self.limit + 1, "up", "trying one more")
        else:
            self._hold(rarer=True)

    def _decide(self) -> None:
        win = self._win
        if self._mode == "measure":
            self._settle(win)
            if self._check_down and self.limit > self.floor:
                self._down_steps = 0
                self._set(self.limit - 1, "down", "is one fewer just as fast?")
            else:
                self._up_or_hold()
        elif self._mode == "up":
            if self._better(win, self._home):
                self._settle(win)
                self._reprobe = REPROBE_FIRST
                self.events.append(f"{win.level} kept: {win.rate() * 60:.1f} images/min")
                self._up_or_hold()
            else:
                self._ceiling = self._home.level
                self._set(self._home.level, "hold", "one more brought no gain")
                self._hold(rarer=True)
        elif self._mode == "down":
            self._check_down = False
            if self._better(self._home, win):
                # the higher level earns its place: go back to it
                back, gave_up = self._home.level, self._down_steps
                self._down_steps = 0
                self._set(back, "hold", "one fewer was slower")
                if gave_up:
                    self._ceiling = back      # the level above was just found no better
                    self._hold()
                else:
                    self._up_or_hold()        # nothing was given up: look upwards (the old window still stands)
            else:
                self._down_steps += 1
                self._ceiling = win.level
                self._settle(win)
                self._reprobe = REPROBE_FIRST
                self.events.append(f"{win.level} is as fast as {win.level + 1}")
                if self.limit > self.floor:
                    self._set(self.limit - 1, "down", "is one fewer just as fast?")
                else:
                    self._down_steps = 0
                    self._hold()
        else:   # hold: keep watching the accepted level
            ref = self._ref_latency or win.mean_latency
            self._slow = self._slow + 1 if win.mean_latency > DRIFT * ref else 0
            self._home = win
            slower = self._slow >= DRIFT_WINDOWS
            if slower:
                self._slow = 0
                self._ref_latency = win.mean_latency
            if slower or self._hold_left <= 0:
                self._ceiling = self._hard_cap                 # time to look at both neighbours again
                self._down_steps = 0
                if self.limit > self.floor:
                    self._set(self.limit - 1, "down",
                              "requests became slower" if slower else "checking that one fewer is slower")
                    return
                if not slower and self._room_above():
                    self._set(self.limit + 1, "up", "looking again for room")
                    return
                self._hold(rarer=True)
                return
            self._win = self._window(self.limit)

    def _back_off(self) -> None:
        old = self.limit
        if old <= self.floor:
            return
        self._strikes[old] = self._strikes.get(old, 0) + 1
        if self._strikes[old] >= 2:
            self._hard_cap = min(self._hard_cap, old - 1)
        self._ceiling = min(self._ceiling, old - 1)
        self._check_down = False
        self._down_steps = 0
        self._home = None
        self._reprobe = REPROBE_FIRST
        self._set(max(self.floor, old // 2), "measure", "the server was overloaded")


class RateMeter:
    """Images per minute over the most recent results, and the time left at that rate."""

    def __init__(self, window: int = 30, clock=time.monotonic):
        self.clock = clock
        self._lock = threading.Lock()         # ticked by the worker thread, paused from the UI thread
        self._times: deque[float] = deque(maxlen=max(2, window) + 1)
        self._times.append(clock())
        self._paused_at: float | None = None
        self.count = 0

    def tick(self, n: int = 1) -> None:
        with self._lock:
            now = self.clock()
            for _ in range(n):
                self._times.append(now)
            self.count += n

    def pause(self) -> None:
        with self._lock:
            if self._paused_at is None:
                self._paused_at = self.clock()

    def resume(self) -> None:
        with self._lock:
            if self._paused_at is not None:
                gap = self.clock() - self._paused_at
                self._paused_at = None
                for i in range(len(self._times)):
                    self._times[i] += gap          # a pause is not slowness

    def per_minute(self) -> float:
        with self._lock:
            n = len(self._times) - 1
            now = self._paused_at if self._paused_at is not None else self.clock()
            span = now - self._times[0]
            return 60.0 * n / span if n > 0 and span > 0 else 0.0

    def eta_seconds(self, remaining: int) -> float | None:
        rate = self.per_minute()
        return remaining * 60.0 / rate if rate > 0 and remaining >= 0 else None


def format_eta(seconds: float | None) -> str:
    if seconds is None:
        return ""
    s = int(round(seconds))
    if s >= 3600:
        return f"{s // 3600} h {s % 3600 // 60:02d} min"
    if s >= 60:
        return f"{s // 60} min {s % 60:02d} s"
    return f"{s} s"
