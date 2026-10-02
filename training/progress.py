# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/training/progress.py
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: the RefMod parsers are dropped; parse_epoch_summary / parse_adaptive_line /
# parse_cache_line feed the Train tab's loss chart and status. The tqdm / epoch / preview parsing is Fizgig's.
"""Training-progress parsing kept outside the trainer.

The Train tab receives the trainer's normal console stream (stdout + stderr of the child process). This module
turns those tqdm, epoch, preview and adaptive-LR lines into display state, so the trainer needs no second
progress-reporting channel.
"""
from __future__ import annotations

import math
import re

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TQDM_PROGRESS_RE = re.compile(
    r"(?P<step>\d+)/(?P<total>\d+)\s+\[[^\]<]*<(?P<eta>[^,\]]+),\s*"
    r"(?P<speed>[0-9]+(?:\.[0-9]+)?)(?P<unit>it/s|s/it)"
    r"(?:,\s*avr_loss=(?P<loss>[-+0-9.eE]+))?"
)
_EPOCH_RE = re.compile(r"\bepoch\s+(?P<epoch>\d+)\s*/\s*(?P<total>\d+)", re.I)
_EPOCH_SUMMARY_RE = re.compile(r"\bepoch\s+(?P<epoch>\d+)/(?P<total>\d+)\s+avr_loss=(?P<loss>[-+0-9.eE]+)"
                               r".*?\blr=(?P<lr>[-+0-9.eE]+)")
_ADAPTIVE_RE = re.compile(r"\[adaptive_lr\]\s+epoch\s+(?P<epoch>\d+):.*?lr=(?P<lr>[0-9.eE+-]+(?:->[0-9.eE+-]+)?)"
                          r".*\|\s*(?P<rest>.+)$")
_CACHE_RE = re.compile(r"\[cache\]\s+(?P<stage>latents|text)\s+(?P<done>\d+)/(?P<total>\d+)")


def _clean(line: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", line).replace("\r", "")


def parse_tqdm_progress_line(line: str):
    """The values displayed by the trainer's tqdm bar (`steps: 12/300 [..<.., 1.2s/it, avr_loss=0.12]`), or None.
    Preview sampling bars use another description and are ignored."""
    clean = _clean(line)
    if re.search(r"steps:\s", clean) is None:
        return None
    match = _TQDM_PROGRESS_RE.search(clean)
    if match is None:
        return None
    try:
        result = {
            "step": int(match.group("step")),
            "total_steps": int(match.group("total")),
            "eta_text": match.group("eta").strip(),
            "speed_text": f"{match.group('speed')} {match.group('unit')}",
            "average_loss_text": match.group("loss"),
        }
    except (TypeError, ValueError):
        return None
    if result["total_steps"] <= 0:
        return None
    return result


def parse_epoch_line(line: str):
    """``(epoch, total_epochs)`` from a trainer message."""
    match = _EPOCH_RE.search(_clean(line))
    if match is None:
        return None
    epoch, total = int(match.group("epoch")), int(match.group("total"))
    if epoch <= 0 or total <= 0:
        return None
    return epoch, total


def parse_epoch_summary(line: str):
    """The end-of-epoch line ``epoch 3/30  avr_loss=0.1234 ... lr=2.83e-04`` -> {epoch, total, loss, lr}."""
    m = _EPOCH_SUMMARY_RE.search(_clean(line))
    if m is None:
        return None
    try:
        return {"epoch": int(m.group("epoch")), "total": int(m.group("total")), "loss": float(m.group("loss")),
                "lr": float(m.group("lr"))}
    except ValueError:
        return None


def parse_adaptive_line(line: str):
    """An ``[adaptive_lr] epoch N: ... | ACTION (reason)`` line -> {epoch, lr_text, action}."""
    m = _ADAPTIVE_RE.search(_clean(line))
    if m is None:
        return None
    rest = m.group("rest").strip()
    action = next((a for a in ("HOLD (capped)", "HOLD (floored)") if rest.startswith(a)), rest.split(" (")[0])
    return {"epoch": int(m.group("epoch")), "lr_text": m.group("lr"), "action": action.strip()}


def parse_cache_line(line: str):
    """A ``[cache] latents 12/40`` line from the cache stages -> {stage, done, total}."""
    m = _CACHE_RE.search(_clean(line))
    if m is None:
        return None
    return {"stage": m.group("stage"), "done": int(m.group("done")), "total": int(m.group("total"))}


def parse_preview_phase(line: str):
    """``(phase, epoch)`` for preview messages: start / complete / failed."""
    lower = _clean(line).lower()
    epoch_match = re.search(r"epoch[- ]?(\d+)", lower)
    epoch = int(epoch_match.group(1)) if epoch_match else None
    if "[sample]" in lower and "preview failed" in lower:
        return "failed", epoch
    if "[sample]" in lower and "preview(s) ->" in lower:
        return "complete", epoch
    if "[sample]" in lower and "rendering" in lower:
        return "start", epoch
    return None


class TrainingProgressTracker:
    """Convert the trainer's console output into display state."""

    def __init__(self, total_epochs=1):
        self.reset(total_epochs)

    def reset(self, total_epochs=1):
        try:
            total_epochs = int(total_epochs)
        except (TypeError, ValueError):
            total_epochs = 1
        self.total_epochs = max(1, total_epochs)
        self.current_epoch = 1

    def consume(self, line: str):
        """An update dictionary for a relevant line, otherwise None. kinds: cache, epoch, adaptive, preview,
        training."""
        cache = parse_cache_line(line)
        if cache is not None:
            return {"kind": "cache", **cache}
        summary = parse_epoch_summary(line)
        if summary is not None:
            self.current_epoch, self.total_epochs = summary["epoch"], summary["total"]
            return {"kind": "epoch", **summary}
        adaptive = parse_adaptive_line(line)
        if adaptive is not None:
            return {"kind": "adaptive", **adaptive}
        epoch = parse_epoch_line(line)
        if epoch is not None:
            self.current_epoch, self.total_epochs = epoch
        preview = parse_preview_phase(line)
        if preview is not None:
            phase, preview_epoch = preview
            if preview_epoch is not None:
                self.current_epoch = preview_epoch
            return {"kind": "preview", "phase": phase, "epoch": preview_epoch}
        displayed = parse_tqdm_progress_line(line)
        if displayed is None:
            return None
        # one run-wide bar with a constant number of steps per epoch: derive the epoch from the step
        step, total_steps = displayed["step"], displayed["total_steps"]
        if total_steps >= self.total_epochs:
            inferred = math.ceil(max(1, step) * self.total_epochs / total_steps)
            self.current_epoch = max(1, min(self.total_epochs, inferred))
        return {"kind": "training", "epoch": self.current_epoch, "total_epochs": self.total_epochs, **displayed}
