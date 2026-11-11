"""A discrete-event model of a caption server for the limiter tests (no threads, no clock, no network).

``simulate`` plays a client that keeps ``limiter.limit`` requests in flight against a server with ``slots``
parallel slots. A request needs ``fixed + per_token * tokens`` seconds of single-stream work; with ``k`` slots
busy every stream runs at ``1 / (1 + penalty * (k - 1))`` of that speed (0 = perfect batching, 1 = none), and a
request beyond the slots waits in a first-in first-out queue. Everything is deterministic for a given seed.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Sim:
    slots: int = 1
    penalty: float = 0.2
    fixed: float = 1.0            # seconds per request that do not depend on the reply (picture, prompt)
    per_token: float = 0.02       # seconds per reply token, one stream alone
    tokens: float = 120.0         # mean reply length
    cv: float = 0.3               # spread of the reply length (uniform)
    fail_above: int = 0           # a request started with more than this many in flight fails at once (0 = never)
    slow_after: int = 0           # after this many finished requests the server ...
    slow_factor: float = 1.0      # ... works this many times slower
    slow_extra_above: int = 0     # ... and, with more than this many busy slots, even slower (thrashing)
    seed: int = 0
    history: list = field(default_factory=list)      # the limit at each finished request
    seconds: float = 0.0          # simulated time so far (the limiter's clock: ``Sim.clock``)
    finished: int = 0
    failed: int = 0

    def clock(self) -> float:
        return self.seconds


def simulate(limiter, sim: Sim, requests: int) -> Sim:
    """Run ``requests`` requests. Build the limiter with ``clock=sim.clock`` so it sees simulated time."""
    rnd = random.Random(sim.seed)
    now = sim.seconds
    active: list[list] = []        # [remaining work, tokens, tag, start]
    waiting: deque = deque()
    started = 0

    def flying() -> int:
        return len(active) + len(waiting)

    while sim.finished + sim.failed < requests:
        while flying() < limiter.limit and started < requests:
            started += 1
            tag = limiter.begin()
            if sim.fail_above and flying() + 1 > sim.fail_above:
                sim.failed += 1
                sim.history.append(limiter.limit)
                limiter.done(tag, 0.05, ok=False, busy=True)
                continue
            tokens = sim.tokens * (1 + sim.cv * 3 ** 0.5 * (2 * rnd.random() - 1))
            job = [sim.fixed + sim.per_token * tokens, tokens, tag, now]
            (active if len(active) < sim.slots else waiting).append(job)
        if not active:
            break
        k = len(active)
        speed = 1.0 / (1.0 + sim.penalty * (k - 1))
        if sim.slow_after and sim.finished >= sim.slow_after:
            speed /= sim.slow_factor
            if sim.slow_extra_above and k > sim.slow_extra_above:
                speed /= 3.0
        step = min(job[0] for job in active) / speed
        now += step
        sim.seconds = now
        for job in active:
            job[0] -= step * speed
        done = [job for job in active if job[0] <= 1e-9]
        active = [job for job in active if job[0] > 1e-9]
        while waiting and len(active) < sim.slots:
            active.append(waiting.popleft())
        for _work, tokens, tag, start in done:
            sim.finished += 1
            sim.history.append(limiter.limit)
            limiter.done(tag, now - start, ok=True, size=tokens)
    sim.seconds = now
    return sim
