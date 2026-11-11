"""Spot a reply that got stuck repeating itself, cheaply, while it is still being generated.

A model that loops ("... a red scarf, a red scarf, a red scarf ...") keeps going until the token limit: several
times the cost of a normal caption, for text nobody wants. With a streamed reply the loop is seen after a few
repeats and the request is closed. No Qt, no torch.
"""
from __future__ import annotations

MAX_PERIOD = 240        # longest repeated unit looked for, in characters (about two sentences)
MIN_REPEATED = 80       # at least this much repeated text ...
MIN_REPEATS = 3         # ... and the unit seen this many times after its first occurrence
CHECK_EVERY = 24        # characters of new text between checks


def find_loop(text: str) -> tuple[int, int] | None:
    """``(start, period)`` when ``text`` ends in the same unit over and over, else None.

    ``start`` is where the repetition begins, so ``text[:start + period]`` is the reply with the unit once.
    A match needs the last ``max(MIN_REPEATS * period, MIN_REPEATED)`` characters to equal the characters
    ``period`` earlier: four occurrences of a phrase, or 81 identical characters.
    """
    n = len(text)
    for period in range(1, min(MAX_PERIOD, n // (MIN_REPEATS + 1)) + 1):
        span = max(MIN_REPEATS * period, MIN_REPEATED)
        if span + period > n:
            continue
        if text[n - span:] == text[n - span - period:n - period]:
            start = n - span - period
            while start > 0 and text[start - 1] == text[start - 1 + period]:
                start -= 1
            return start, period
    return None


def cut_loop(text: str) -> str | None:
    """``text`` without its repeating tail (the repeated unit kept once), or None if it does not loop."""
    hit = find_loop(text)
    if hit is None:
        return None
    start, period = hit
    kept = text[:start + period].rstrip()
    # a unit that starts mid-sentence leaves a ragged end: drop a trailing separator
    return kept.rstrip(",;:- \n\t")


class LoopWatch:
    """Fed the pieces of a streamed reply; ``feed`` returns True once the reply is looping."""

    def __init__(self):
        self.parts: list[str] = []
        self._len = 0
        self._checked = 0

    @property
    def text(self) -> str:
        return "".join(self.parts)

    def feed(self, piece: str) -> bool:
        if not piece:
            return False
        self.parts.append(piece)
        self._len += len(piece)
        if self._len - self._checked < CHECK_EVERY or self._len < MIN_REPEATED:
            return False
        self._checked = self._len
        # only the tail can loop: enough of it to hold the longest unit the needed number of times
        tail = self.text[-(MAX_PERIOD * (MIN_REPEATS + 1) + 8):]
        return find_loop(tail) is not None
