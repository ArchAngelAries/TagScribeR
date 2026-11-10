# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/krea2/rotation.py (v8.7.3, commit 9162fb20)
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: only what the shared fine-tune (training/ft.py) uses - the schedule, the planner, the
# component helpers and the just-in-time streamer; the original Krea 2 fp8 BlockRotator and its K2FT calibration
# constants (the pre-shared-layer trainer's) are not ported. The streamer's prefetch thread, transfer stream and
# events exist only when the device is a CUDA device (TagScribeR's guard, as training/modules/offloading.py: a HIP
# build reports "cuda" too; a CPU device moves synchronously, which is the path the tests run). Block moves go through
# TagScribeR's weighs_to_device and move_nf4_to_device. Otherwise unchanged.
"""Rotating-window fine-tuning: which slice of the model trains when, how the windows are sized for a card, and the
streamer that keeps the trainable window resident while every other block rings in from CPU.

A naive full fine-tune of a 13B DiT needs ~78 GB (bf16 weights + grads + Adam state). Rotation trains a *window* of
the model at a time (a component - attention, or one MLP matrix - across every block) from a CPU master copy, so
VRAM only ever holds grads and optimizer state for the active window. Over a full rotation every weight trains.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class RotationSchedule:
    """Which block indices (block mode) or component windows (component mode) are trainable at a given epoch.

    `active` blocks train for `rotate_every` epochs, then the window advances. A full cycle (every block trained
    once) takes n_windows * rotate_every epochs - fewer than one full cycle means part of the model never trains.
    """

    def __init__(self, n_blocks: int, active: int = 4, rotate_every: int = 1,
                 order: Optional[Sequence[int]] = None, mode: str = "block",
                 components: Sequence[str] = ("attn", "mlp.gate", "mlp.up", "mlp.down"),
                 start_window: int = 0):
        if n_blocks <= 0:
            raise ValueError("n_blocks must be positive")
        self.n_blocks = int(n_blocks)
        self.active = max(1, min(int(active), self.n_blocks))
        self.rotate_every = max(1, int(rotate_every))
        # A resumed FT run is a fresh process: without an offset it restarts the cycle at window 0, retraining early
        # windows and starving late ones. start_window shifts the whole schedule so "resume for the last epoch" can
        # mean "train the last component".
        self.start_window = int(start_window)
        self.order = list(order) if order is not None else list(range(self.n_blocks))
        if sorted(self.order) != list(range(self.n_blocks)):
            raise ValueError("order must be a permutation of range(n_blocks)")
        # "block": windows are contiguous slices of blocks (depth-split). "component": windows are sub-block
        # components spanning EVERY block - attention across the whole model, then MLP - so a concept is learned by
        # every layer at once instead of by one depth slice at a time.
        self.mode = "component" if str(mode).startswith("comp") else "block"
        self.components = list(components)

    @property
    def n_windows(self) -> int:
        if self.mode == "component":
            return len(self.components)
        return (self.n_blocks + self.active - 1) // self.active

    @property
    def cycle_epochs(self) -> int:
        """Epochs needed for every block to have trained once."""
        return self.n_windows * self.rotate_every

    def window_at(self, epoch: int) -> int:
        """0-based window index for a 0-based epoch (wraps after a full cycle)."""
        return ((int(epoch) // self.rotate_every) + self.start_window) % self.n_windows

    def active_at(self, epoch: int):
        """Block indices (block mode) or component window entries (component mode) - an entry is a bare prefix
        string (full depth), a (prefix, lo, hi) depth-split, or a list of prefixes trained together."""
        w = self.window_at(epoch)
        if self.mode == "component":
            c = self.components[w]
            return list(c) if isinstance(c, list) else [c]
        start = w * self.active
        return sorted(self.order[start:start + self.active])

    def describe(self) -> str:
        if self.mode == "component":
            names = ["+".join(c) if isinstance(c, list) else c if isinstance(c, str) else f"{c[0]}@{c[1]}-{c[2]}"
                     for c in self.components]
            return (f"component windows {names} across all {self.n_blocks} blocks, "
                    f"rotating every {self.rotate_every} epoch(s) - {self.n_windows} windows, "
                    f"{self.cycle_epochs} epochs per full cycle")
        return (f"{self.active} of {self.n_blocks} blocks per window, rotating every "
                f"{self.rotate_every} epoch(s) - {self.n_windows} windows, "
                f"{self.cycle_epochs} epochs per full cycle")


def component_gb_per_block(block: nn.Module, prefixes) -> dict:
    """Exact bf16 GB per block for each component prefix, measured from the model itself: Linear out x in x 2 bytes
    under each prefix (the LOGICAL shape, not weight.numel(): an NF4 Linear's .weight is empty)."""
    out = {p: 0.0 for p in prefixes}
    for name, m in block.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        for p in prefixes:
            if name.startswith(p):
                out[p] += m.out_features * m.in_features * 2 / 1e9
                break
    return out


def snap_ft_epochs(max_epochs, cycle_epochs, start_window=0, rotate_every=1):
    """Snap a fine-tune's epoch count UP so the run ENDS on a rotation-cycle boundary (an off-cycle total leaves the
    final checkpoint with some components trained one more pass than others). Start-aware for pause / resume: a
    resumed leg begins mid-cycle at start_window, so the snap aligns start + length to a boundary."""
    m = int(max_epochs)
    cyc = max(1, int(cycle_epochs))
    if m <= 0:
        return m
    pos = (int(start_window or 0) * max(1, int(rotate_every))) % cyc
    rem = (pos + m) % cyc
    return m if rem == 0 else m + (cyc - rem)


def plan_component_windows(usable_gb, span, n_blocks, comp_gb, overhead_gb, trunk_gb_per_block, slots_gb,
                           allow_stream=True, max_sane_windows=12, spans=None, trunk_credit=True,
                           pack_margin_gb=0.0, max_parts=0):
    """The family-agnostic component-window plan: (windows, stream, reasons).

    comp_gb: ordered (prefix -> bf16 GB per block). windows: RotationSchedule component entries - bare prefixes where
    the full span fits, (prefix, lo, hi) depth-splits where it doesn't, and lists of prefixes where several whole
    parts fit one window (the fewest windows the card holds; one list of every part = the whole model at once).
    stream: True when the frozen out-of-window blocks must stream from CPU to fit. Returns (None, stream, reasons)
    when the budget can't run FT at all. Pure, so the tier tables are pinnable without a card.

    spans: prefix -> the blocks that hold it, where a component lives in some blocks only (Klein: img_attn in the
    double blocks, linear1 in the single ones); comp_gb is then per block OF ITS SPAN, and a depth-split divides that
    span. trunk_credit / pack_margin_gb: how a window of several parts is sized. max_parts: the user's cap on parts
    per window (0 = as many as fit; 1 = one part per window, the most headroom)."""
    span = sorted(int(b) for b in span)
    spans = {p: sorted(int(b) for b in s) for p, s in (spans or {}).items()}

    def span_of(prefix):
        return spans.get(prefix, span)
    usable = float(usable_gb)
    prefixes = list(comp_gb)
    fattest = max(comp_gb.values())
    reasons = []

    def _chunk(prefix, max_gb):
        """Even depth-chunks of the prefix's span whose bf16 window fits max_gb, or None."""
        g = comp_gb[prefix]
        if g <= 0:
            # a component that measures as weightless means the sizer could not see this storage scheme: fail loudly
            # rather than plan a cycle for windows it thinks are free
            raise ValueError(f"component {prefix!r} measured 0 GB per block - the window sizer does not understand "
                             f"this model's weight storage")
        max_len = int(max_gb / g)
        if max_len < 1:
            return None
        sp = span_of(prefix)
        k = (len(sp) + max_len - 1) // max_len
        per = (len(sp) + k - 1) // k
        chunks = [sp[i:i + per] for i in range(0, len(sp), per)]
        if k == 1:
            return [prefix]
        return [(prefix, c[0], c[-1]) for c in chunks]

    # Packed plan: as many whole parts per window as fit, the fewest windows first. A trained Linear drops its
    # quantised copy, so a window of several parts is credited their share of the trunk; a one-part window keeps the
    # calibrated overhead + its weights.
    full = {p: len(span_of(p)) * comp_gb[p] for p in prefixes}
    n_span = len(set().union(*spans.values())) if spans else len(span)
    per_gb = trunk_gb_per_block * n_span / max(1e-9, sum(full.values()))    # trunk GB per bf16 GB trained

    def need(group):
        w = sum(full[p] for p in group)
        if len(group) == 1:
            return overhead_gb + w
        return overhead_gb + w - (per_gb * w if trunk_credit else 0.0) + pack_margin_gb

    groups = []
    for p in sorted(prefixes, key=lambda q: -full[q]):        # first-fit decreasing
        for g in groups:
            if (not max_parts or len(g) < max_parts) and need(g + [p]) <= usable:
                g.append(p)
                break
        else:
            groups.append([p])
    if len(groups) < len(prefixes) and all(need(g) <= usable for g in groups):
        order = {p: i for i, p in enumerate(prefixes)}
        groups = sorted((sorted(g, key=order.get) for g in groups), key=lambda g: order[g[0]])
        reasons.append(f"{len(prefixes)} parts packed into {len(groups)} window(s), each at full depth (largest "
                       f"~{max(need(g) for g in groups):.1f} GB of {usable:.1f} usable)")
        return [g if len(g) > 1 else g[0] for g in groups], False, reasons

    # Full-speed plan: everything resident, split only what the budget forces. Viable only when at least a 1-block
    # window of the fattest component fits the cap.
    cap = usable - overhead_gb
    plain = []
    if cap >= fattest:
        for prefix in prefixes:
            _c = _chunk(prefix, cap)
            if _c is None:
                plain = []
                break
            plain.extend(_c)
    if plain and len(plain) <= max_sane_windows:
        for prefix in prefixes:
            need_gb = overhead_gb + len(span_of(prefix)) * comp_gb[prefix]
            if need_gb > usable:
                reasons.append(f"{prefix} across {len(span_of(prefix))} block(s) would peak ~{need_gb:.1f} GB vs "
                               f"{usable:.1f} usable - depth-split")
        return plain, False, reasons

    if not allow_stream:
        reasons.append(f"{usable:.1f} GB usable cannot hold the frozen trunk + a component window at full "
                       f"residency, and streaming is disabled")
        return (plain or None), False, reasons

    # Streaming plan: only the active chunk's blocks stay resident; every other block's frozen weights ring in from
    # CPU. Per-chunk peak ~ overhead - reclaimed + slots + window.
    base = overhead_gb - trunk_gb_per_block * int(n_blocks) + slots_gb
    stream_windows = []
    for prefix in prefixes:
        g = comp_gb[prefix]
        per_block = g + trunk_gb_per_block         # window bf16 + the block's own frozen share
        max_gb = (usable - base) / per_block * g   # expressed back in window GB
        chunks = _chunk(prefix, max_gb) if max_gb > 0 else None
        if chunks is None:
            reasons.append(f"{usable:.1f} GB usable cannot fit even a 1-block {prefix} window with streaming "
                           f"(~{base + per_block:.1f} GB needed)")
            return None, True, reasons
        stream_windows.extend(chunks)
    reasons.append(f"{usable:.1f} GB usable is below the resident plan - frozen out-of-window blocks stream from "
                   f"CPU (~{trunk_gb_per_block * n_blocks:.1f} GB staged in RAM, steps slower for the PCIe trips)")
    return stream_windows, True, reasons


def is_component_spec(spec) -> bool:
    """True when a rotation spec lists component windows (prefix strings / (prefix, lo, hi) tuples) rather than
    block indices (plain ints)."""
    return bool(spec) and isinstance(spec[0], (str, tuple))


def component_entry_matches(entry, lname: str, block_idx: int) -> bool:
    """Does a component window entry claim this (linear name, block index)? A bare prefix claims the Linear at every
    depth; a (prefix, lo, hi) tuple only for blocks lo..hi inclusive (the depth-split that fits a fat window on a
    smaller card in slices)."""
    if isinstance(entry, str):
        return lname.startswith(entry)
    prefix, lo, hi = entry
    return lname.startswith(prefix) and int(lo) <= int(block_idx) <= int(hi)


# ---------------------------------------------------------------------------
# Rotation-aware block streaming
# ---------------------------------------------------------------------------

def _move_block(block: nn.Module, dev: torch.device):
    """A block's weights to `dev`: the Linears' weights (and fp8 scales), then NF4's packed bytes and quant state,
    which are plain attributes nn.Module.to() cannot see (a block would "move" while its 4-bit weight stayed put, and
    the dequant forward then met mat2 on the wrong device)."""
    from training.modules.nf4 import move_nf4_to_device
    from training.modules.offloading import weighs_to_device
    weighs_to_device(block, dev)
    move_nf4_to_device(block, dev)


class RotationOffloader:
    """Keeps the trainable window pinned on the GPU and streams every other block from CPU.

    The stock offloader keeps a contiguous PREFIX of blocks resident and swaps the suffix, pairing each eviction with
    a matching load. Rotation needs the opposite: an arbitrary resident set (the active window) that MOVES every
    rotation. Rather than rework that pairing, this streams non-resident blocks just-in-time. It implements the same
    small interface the DiT's forward expects, so it drops in as `dit.offloader`.

    Gradient checkpointing recomputes a block's forward *inside* its backward, so blocks are pulled back to the GPU
    on a backward PRE-hook (before the recompute) and evicted on the post-hook.
    """

    def __init__(self, blocks, device: torch.device, resident: Iterable[int], debug: bool = False,
                 prefetch: bool = True):
        self.blocks = blocks
        self.device = torch.device(device)
        self.cpu = torch.device("cpu")
        self.resident = set(int(i) for i in resident)
        self.forward_only = False
        self.debug = debug
        self.moves = 0                      # instrumentation: transfers per step
        self._handles = []
        # Async plumbing: one worker thread issuing copies on a dedicated CUDA stream, so a block's transfer overlaps
        # the previous block's compute. CUDA devices only (TagScribeR's guard): a CPU device moves synchronously.
        self.cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self.prefetch = bool(prefetch and self.cuda)
        self._pool = ThreadPoolExecutor(max_workers=1) if self.prefetch else None
        self._stream = torch.cuda.Stream(device=self.device) if self.prefetch else None
        self._futures = {}                  # block idx -> (device, Future[cuda.Event | None])
        self._register_hooks()

    # -- placement helpers -------------------------------------------------
    def _to(self, idx: int, dev: torch.device):
        _move_block(self.blocks[idx], dev)
        self.moves += 1

    def _move_task(self, idx: int, dev: torch.device, after):
        """Runs on the worker thread, issuing the copy on the transfer stream."""
        torch.cuda.set_device(self.device.index if self.device.index is not None else torch.cuda.current_device())
        with torch.cuda.stream(self._stream):
            if after is not None:
                self._stream.wait_event(after)      # never yank weights out from under queued kernels
            self._to(idx, dev)
            ev = torch.cuda.Event()
            ev.record(self._stream)
        return ev

    def _submit(self, idx: int, dev: torch.device, after=None):
        # Track the DIRECTION, not just that something is in flight: an in-flight eviction must never be mistaken for
        # a completed load.
        pending = self._futures.get(idx)
        if pending is not None:
            if pending[0] == dev:
                return                  # same direction already queued
            self._await(idx)            # opposite direction: settle it first
        self._futures[idx] = (dev, self._pool.submit(self._move_task, idx, dev, after))

    def _await(self, idx: int):
        """Settle any in-flight move for `idx`. Returns the device it moved to, or None."""
        pending = self._futures.pop(idx, None)
        if pending is None:
            return None
        dev, fut = pending
        ev = fut.result()
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)
        return dev

    def _ensure_gpu(self, idx: int):
        if idx in self.resident:
            return
        landed = self._await(idx)
        if landed is not None and landed.type == self.device.type:
            return                      # a prefetch already put it on the GPU
        self._to(idx, self.device)

    def _evict(self, idx: int):
        if idx in self.resident:
            return
        if self.prefetch:
            done = torch.cuda.Event()   # queue the eviction behind the compute that just used this block
            done.record(torch.cuda.current_stream())
            self._submit(idx, self.cpu, after=done)
        else:
            self._to(idx, self.cpu)

    def _prefetch_next(self, idx: int):
        if not self.prefetch or idx < 0 or idx >= len(self.blocks) or idx in self.resident:
            return
        self._submit(idx, self.device)

    # -- rotation ----------------------------------------------------------
    def set_resident(self, resident: Iterable[int]):
        """Re-pin after a rotation: the incoming window goes to the GPU and stays, the outgoing window rejoins the
        streamed pool. EVICT BEFORE LOADING: the reverse order holds both windows on the card at once."""
        for idx in list(self._futures):     # settle everything before re-planning
            self._await(idx)
        new = set(int(i) for i in resident)
        for idx in sorted(self.resident - new):
            self._to(idx, self.cpu)
        for idx in sorted(new - self.resident):
            self._to(idx, self.device)
        self.resident = new
        logger.info("[rotation-swap] resident blocks now %s (others stream from CPU)", sorted(new))

    def view(self, start: int, count: int) -> "OffloaderView":
        """The offloader for one of the model's block lists when `blocks` is several of them end to end (Klein: 8
        double then 24 single blocks, each list swapped by its own offloader counting from 0)."""
        return OffloaderView(self, int(start), int(count))

    def _place(self, idxs):
        for idx in list(self._futures):
            self._await(idx)
        for i in idxs:
            self._to(i, self.device if i in self.resident else self.cpu)
        if self.cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        if idxs:                            # warm the first streamed block so step 0 isn't a stall
            self._prefetch_next(idxs[0] if idxs[0] not in self.resident else idxs[0] + 1)

    # -- interface the DiT forward calls -----------------------------------
    def prepare_block_devices_before_forward(self, blocks):
        self._place(range(len(blocks)))

    def wait_for_block(self, index: int):
        self._ensure_gpu(index)
        self._prefetch_next(index + 1)      # overlap the next load with this block's compute

    def submit_move_blocks_forward(self, blocks, index: int):
        self._evict(index)

    def set_forward_only(self, forward_only: bool):
        self.forward_only = bool(forward_only)

    # -- backward ----------------------------------------------------------
    def _register_hooks(self):
        for i, block in enumerate(self.blocks):
            def pre_hook(module, grad_output, _i=i):
                self._ensure_gpu(_i)
                self._prefetch_next(_i - 1)   # backward walks blocks in reverse
                return None

            def post_hook(module, grad_input, grad_output, _i=i):
                self._evict(_i)
                return None

            self._handles.append(block.register_full_backward_pre_hook(pre_hook))
            self._handles.append(block.register_full_backward_hook(post_hook))

    def remove(self):
        for idx in list(self._futures):
            self._await(idx)
        for h in self._handles:
            h.remove()
        self._handles.clear()
        if self._pool is not None:
            self._pool.shutdown(wait=True)

    remove_hooks = remove               # the name TagScribeR's models call on a replaced offloader


class OffloaderView:
    """One block list inside a RotationOffloader that streams several lists end to end: the same interface the
    forward calls, with the list's own block index shifted to its place in the streamer. Prefetch runs on across the
    boundary (the last double block warms the first single block)."""

    def __init__(self, owner: RotationOffloader, start: int, count: int):
        self.owner, self.start, self.count = owner, start, count

    def prepare_block_devices_before_forward(self, blocks):
        self.owner._place(range(self.start, self.start + len(blocks)))

    def wait_for_block(self, index: int):
        self.owner.wait_for_block(self.start + index)

    def submit_move_blocks_forward(self, blocks, index: int):
        self.owner.submit_move_blocks_forward(None, self.start + index)

    def set_forward_only(self, forward_only: bool):
        self.owner.set_forward_only(forward_only)

    def remove_hooks(self):
        self.owner.remove()

    remove = remove_hooks


__all__ = ["RotationSchedule", "RotationOffloader", "OffloaderView", "component_gb_per_block",
           "component_entry_matches", "is_component_spec", "plan_component_windows", "snap_ft_epochs"]
