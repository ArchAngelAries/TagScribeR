# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/training/step_log.py
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: the file is `.tagscriber_loss_<output_name>.jsonl` (Fizgig's is `.fizgig_loss_...`); the line
# format, the batching, the resume rule, the request flags and the thinning reader are Fizgig's.
"""The run's per-step loss for the Train tab's loss graph, and the "Sample now" / "Save now" requests it can make.

Loss: `<output_dir>/.tagscriber_loss_<output_name>.jsonl`, one line a step - {"s": step, "e": epoch, "l": loss,
"lr": lr} - written in small batches ("lr" only now and then). A run starting at step N first keeps only the lines up
to N: a fresh run (N = 0) starts the file empty, a resumed one carries on its own curve without the steps it is about
to redo.

Requests: the Train tab drops `.sample_now` / `.save_now` into the output folder, as `.pause_requested` is dropped for
Pause; the trainer takes each once, after an optimizer step.
"""
import json
import os

FLUSH_EVERY = 20
SAMPLE_NOW_FILE = ".sample_now"
SAVE_NOW_FILE = ".save_now"


def loss_path(output_dir, output_name):
    return os.path.join(output_dir, f".tagscriber_loss_{output_name}.jsonl")


class StepLog:
    def __init__(self, output_dir, output_name, start_step=0):
        self.path = loss_path(output_dir, output_name)
        self.buf = []
        kept = []
        if start_step > 0:
            try:
                with open(self.path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            if int(json.loads(line)["s"]) <= start_step:
                                kept.append(line if line.endswith("\n") else line + "\n")
                        except (ValueError, KeyError, TypeError):
                            continue
            except OSError:
                pass
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                f.writelines(kept)
        except OSError:
            self.path = None                 # a read-only folder: the graph is simply not there

    def add(self, step, epoch, loss, lr=None):
        if self.path is None:
            return
        rec = {"s": int(step), "e": int(epoch), "l": round(float(loss), 6)}
        if lr is not None:
            rec["lr"] = float(f"{lr:.4g}")
        self.buf.append(json.dumps(rec) + "\n")
        if len(self.buf) >= FLUSH_EVERY:
            self.flush()

    def flush(self):
        if self.path is None or not self.buf:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.writelines(self.buf)
        except OSError:
            pass
        self.buf = []


def take_request(output_dir, what):
    """True once when `.sample_now` / `.save_now` (what = "sample" / "save") is waiting; the flag is removed."""
    p = os.path.join(output_dir, f".{what}_now")
    if not os.path.exists(p):
        return False
    try:
        os.remove(p)
    except OSError:
        return False                         # already taken (or not ours to remove): don't act twice
    return True


def _parse_row(line):
    try:
        d = json.loads(line)
        return [int(d["s"]), int(d["e"]), float(d["l"]), d.get("lr")]
    except (ValueError, KeyError, TypeError):
        return None


def _summary(rows, epochs, max_points):
    k = max(1, -(-len(rows) // max_points))
    pts = rows[::k]
    if rows and pts[-1] is not rows[-1]:
        pts.append(rows[-1])
    return {"points": pts, "epochs": [list(e) for e in epochs], "steps": len(rows), "stride": k}


def read(path, max_points=4000):
    """The graph's data: {"points": [[step, epoch, loss, lr], ...] (every k-th when longer than max_points, the last
    always kept), "epochs": [[epoch, first step], ...], "steps": lines read, "stride": k} or None when there is no
    file."""
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = _parse_row(line)
                if r is not None:
                    rows.append(r)
    except OSError:
        return None
    epochs, last = [], None
    for r in rows:
        if r[1] != last:
            epochs.append([r[1], r[0]])
            last = r[1]
    return _summary(rows, epochs, max_points)


class StepReader:
    """`read`, but remembering how far it got: each `poll()` parses only the complete lines added since the last one
    (a partial last line waits for the next poll), and starts again from the top when the file shrank or was replaced.
    The result is what `read` returns for the same file."""

    def __init__(self, path, max_points=4000):
        self.path, self.max_points = path, max_points
        self._reset()

    def _reset(self):
        self._offset, self._rows, self._epochs, self._last, self._head = 0, [], [], None, b""
        self.parsed = 0                       # lines parsed over this reader's life (tests watch it)

    def poll(self):
        try:
            with open(self.path, "rb") as f:
                size = os.fstat(f.fileno()).st_size
                head = f.read(64)
                if size < self._offset or (self._offset and not head.startswith(self._head[:len(head)])) \
                        or (self._offset and len(head) < len(self._head)):
                    self._reset()                 # shrank or replaced: from the start
                    f.seek(0)
                    head = f.read(64)
                if not self._offset:
                    self._head = head
                f.seek(self._offset)
                data = f.read()
        except OSError:
            self._reset()
            return None
        cut = data.rfind(b"\n") + 1               # complete lines only
        for raw in data[:cut].split(b"\n"):
            if not raw.strip():
                continue
            self.parsed += 1
            r = _parse_row(raw.decode("utf-8", errors="replace"))
            if r is None:
                continue
            self._rows.append(r)
            if r[1] != self._last:
                self._epochs.append([r[1], r[0]])
                self._last = r[1]
        self._offset += cut
        return _summary(self._rows, self._epochs, self.max_points)
