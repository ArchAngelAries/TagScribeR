# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/training/train_utils.py
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: the accelerate-based state helpers (Klein only) are dropped; the rest is unchanged.
"""Training utilities: loss tracking, checkpoint naming, state-dir pruning, output-name validation."""
import logging
import os
import re
import shutil

logger = logging.getLogger(__name__)

EPOCH_STATE_NAME = "{}-{:06d}-state"
EPOCH_FILE_NAME = "{}-{:06d}"


class LossRecorder:
    """Track per-step losses with a running average over the last epoch's worth of steps.

    Slots are indexed by the in-epoch step. A step that records no loss - e.g. an image the loss watch excluded -
    must be drop()ed so its slot leaves the average; otherwise skipped slots hold stale values that bias avr_loss,
    which the adaptive-LR watcher reads as a real signal."""

    def __init__(self):
        self.loss_list: list[float] = []
        self.loss_total: float = 0.0
        self._empty: set[int] = set()

    def _grow(self, step: int) -> None:
        while len(self.loss_list) <= step:
            self._empty.add(len(self.loss_list))
            self.loss_list.append(0.0)

    def add(self, *, epoch: int, step: int, loss: float) -> None:
        self._grow(step)
        if step not in self._empty:
            self.loss_total -= self.loss_list[step]
        self._empty.discard(step)
        self.loss_list[step] = loss
        self.loss_total += loss

    def drop(self, *, step: int) -> None:
        """Mark an in-epoch step as not-trained (skipped/excluded): its slot leaves the average."""
        self._grow(step)
        if step not in self._empty:
            self.loss_total -= self.loss_list[step]
            self.loss_list[step] = 0.0
            self._empty.add(step)

    @property
    def moving_average(self) -> float:
        n = len(self.loss_list) - len(self._empty)
        if n <= 0:
            return 0.0
        return self.loss_total / n


def list_state_dirs(output_dir: str, output_name: str) -> list[tuple[int, str]]:
    """Every `<output_name>-NNNNNN-state/` in output_dir as (epoch_no, full_path), newest first. Anchored to
    output_name so several LoRAs can share one output folder without one run's pruning deleting another's states."""
    pattern = re.compile(rf"^{re.escape(output_name)}-(\d{{6}})-state$")
    found: list[tuple[int, str]] = []
    try:
        entries = os.listdir(output_dir)
    except OSError:
        return found
    for entry in entries:
        m = pattern.match(entry)
        if not m:
            continue
        full = os.path.join(output_dir, entry)
        if os.path.isdir(full):
            found.append((int(m.group(1)), full))
    found.sort(reverse=True)
    return found


def latest_state_dir(output_dir: str, output_name: str):
    """The newest complete state dir (training_state.json is written last, as the commit marker), or None."""
    for _epoch, path in list_state_dirs(output_dir, output_name):
        if os.path.isfile(os.path.join(path, "training_state.json")):
            return path
    return None


def prune_state_dirs(output_dir: str, output_name: str, keep_n) -> None:
    """Keep the keep_n highest-numbered state dirs, delete the rest. keep_n is clamped to >= 1 (a blank box must
    never mean "delete everything"), and every rmtree is guarded on its own (an AV scanner holding a file must not
    take a multi-hour run down over housekeeping). Always call AFTER the new state is written."""
    try:
        keep_n = max(1, int(keep_n))
    except (TypeError, ValueError):
        keep_n = 1
    for _epoch_no, path in list_state_dirs(output_dir, output_name)[keep_n:]:
        try:
            shutil.rmtree(path)
            logger.info(f"[state] pruned old state: {os.path.basename(path)}")
        except Exception as e:
            logger.warning(f"[state] could not remove {path}: {e}")


_ILLEGAL_NAME_CHARS = '<>:"|?*'


def validate_output_name(output_name: str) -> str:
    """Refuse an output name that cannot become a filename (checked before the run, not an epoch in when the first
    save would fail). Returns it unchanged if it can."""
    name = "" if output_name is None else str(output_name)
    bad = next((c for c in name if c in _ILLEGAL_NAME_CHARS or c < " "), None)
    if bad is not None:
        shown = repr(bad)[1:-1] if bad < " " else bad
        raise ValueError(f"LoRA name {name!r} cannot contain {shown!r} - file names can't include that character. "
                         f"A name pasted from somewhere else often carries a stray line break.")
    if not name.strip() or name != name.strip() or name.endswith("."):
        raise ValueError(f"LoRA name {name!r} cannot be empty, or start or end with a space or a dot.")
    if "/" in name or "\\" in name or os.path.basename(name) != name:
        raise ValueError(f"LoRA name {name!r} must be just a name, not a path - the output folder is set separately.")
    return name


def get_epoch_ckpt_name(model_name: str, epoch_no: int) -> str:
    return EPOCH_FILE_NAME.format(model_name, epoch_no) + ".safetensors"


def get_last_ckpt_name(model_name: str) -> str:
    return model_name + ".safetensors"
