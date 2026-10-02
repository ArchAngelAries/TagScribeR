# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/krea2/trainer.py: AdaptiveLR (the class the
# standard layer imports), with the grad-clip-ratio stability signal from Klein's inline implementation
# (src/fizgig/training/trainer.py, ADAPTIVE_CLIP_RATIO_THRESHOLD and the clip event counters).
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: moved into its own module; note_clip() / clip_ratio carry Klein's signal (only counted when
# gradient clipping is on, as in Klein). Decisions, constants, patience rules and log lines are Fizgig's.
"""Adaptive learning rate: a bi-directional, epoch-level plateau tracker.

The user sets Min LR and Max LR; the run starts at their geometric midpoint sqrt(min * max) and the Learning Rate
box is ignored. At each epoch boundary:

* epoch 1 ARMS the baseline (best loss, LoRA weight norm, a CPU snapshot of the weights + optimizer state);
* a stability signal - more than half the steps clipped (grad norm > max_grad_norm), or the LoRA weight norm
  growing more than 30% in one epoch - REDUCEs x0.5 and blends the weights 70/30 back toward the previous snapshot
  with the optimizer state restored (kills bad Adam momentum). The first event acts at once, later ones need two
  red epochs in a row;
* otherwise an improving loss PROBEs UP x1.25 after two improving epochs; a plateau REDUCEs x0.5 after
  `patience_down` epochs (1 in epochs 2-3, otherwise - and after any stability event - 2);
* every rate is clamped to [min, max].

The step scheduler is off while this is on, and an optimizer that sets its own rate (Automagic v3) turns it off.
State is JSON round-trippable for pause / resume; the CPU rollback snapshot is not persisted, so the first epoch
after a resume cannot roll back (as in Fizgig).
"""
import logging

import torch

logger = logging.getLogger(__name__)


class AdaptiveLR:
    """Bi-directional plateau LR tracker - a faithful port of Fizgig's adaptive_lr logic. Call epoch_boundary() at
    each epoch end; call note_clip() after every gradient clip when clipping is on."""

    BLEND = 0.7
    WEIGHT_GROWTH_THRESHOLD = 0.30
    CLIP_RATIO_THRESHOLD = 0.5
    FACTOR_UP = 1.25
    FACTOR_DOWN = 0.5

    def __init__(self, min_lr, max_lr):
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        if self.min_lr <= 0 or self.max_lr < self.min_lr:
            raise ValueError(f"Adaptive LR needs 0 < min <= max, got min={min_lr} max={max_lr}")
        self.best_loss = None
        self.good_streak = 0
        self.bad_streak = 0
        self.stability_streak = 0
        self.stability_triggered = False
        self.prev_weight_norm = None
        self.clip_events = 0
        self.clip_steps = 0
        self.snapshot = None  # {"weights": {...cpu...}, "optim": cpu state} - not persisted
        self.last_action = ""

    @staticmethod
    def start_lr(min_lr, max_lr) -> float:
        """The run's first rate: the geometric midpoint of Min and Max."""
        return (float(min_lr) * float(max_lr)) ** 0.5

    def state_dict(self):
        return {"best_loss": self.best_loss, "good_streak": self.good_streak,
                "bad_streak": self.bad_streak, "stability_streak": self.stability_streak,
                "stability_triggered": self.stability_triggered,
                "prev_weight_norm": self.prev_weight_norm}

    def load_state_dict(self, d):
        if not d:
            return
        self.best_loss = d.get("best_loss")
        self.good_streak = int(d.get("good_streak", 0))
        self.bad_streak = int(d.get("bad_streak", 0))
        self.stability_streak = int(d.get("stability_streak", 0))
        self.stability_triggered = bool(d.get("stability_triggered", False))
        self.prev_weight_norm = d.get("prev_weight_norm")

    # ---- Klein's clip signal ------------------------------------------------------------------------
    def note_clip(self, pre_clip_norm, max_grad_norm) -> None:
        """Count one optimizer step's clip (call only when clipping is on, with clip_grad_norm_'s return value)."""
        self.clip_steps += 1
        try:
            if float(pre_clip_norm) > float(max_grad_norm):
                self.clip_events += 1
        except (TypeError, ValueError):
            pass

    @property
    def clip_ratio(self):
        """Fraction of this epoch's steps whose pre-clip gradient norm exceeded max_grad_norm (None: no clipping)."""
        return self.clip_events / max(self.clip_steps, 1) if self.clip_steps else None

    # ---- helpers ------------------------------------------------------------------------------------
    @staticmethod
    def _weight_norm(network):
        wn = 0.0
        with torch.no_grad():
            for p in network.parameters():
                if p.requires_grad:
                    wn += float(p.detach().float().norm().item()) ** 2
        return wn ** 0.5

    def _snapshot(self, network, optimizer):
        with torch.no_grad():
            weights = {n: p.detach().clone().to("cpu")
                       for n, p in network.named_parameters() if p.requires_grad}

        def _cpu(o):
            if isinstance(o, torch.Tensor):
                return o.detach().clone().to("cpu")
            if isinstance(o, dict):
                return {k: _cpu(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_cpu(v) for v in o]
            return o
        try:
            self.snapshot = {"weights": weights, "optim": _cpu(optimizer.state_dict())}
        except Exception:
            self.snapshot = {"weights": weights, "optim": None}

    def _rollback(self, network, optimizer):
        cur = dict(network.named_parameters())
        with torch.no_grad():
            for name, prev in self.snapshot["weights"].items():
                if name in cur and cur[name].requires_grad:
                    p = cur[name]
                    prev_d = prev.to(device=p.device, dtype=p.dtype)
                    p.copy_(self.BLEND * prev_d + (1.0 - self.BLEND) * p)
        if self.snapshot.get("optim") is not None:
            try:
                optimizer.load_state_dict(self.snapshot["optim"])
            except Exception:
                pass

    def _reset_clip(self):
        self.clip_events = self.clip_steps = 0

    # ---- the decision -------------------------------------------------------------------------------
    def epoch_boundary(self, epoch, current_loss, network, optimizer):
        """epoch is 0-indexed (global). epoch 0 arms the baseline; epoch >= 1 adjusts the LR. Returns the action."""
        clip_ratio = self.clip_ratio
        clip_str = f"{clip_ratio * 100:.0f}%" if clip_ratio is not None else "—"
        if epoch == 0:
            self.best_loss = current_loss
            self.prev_weight_norm = self._weight_norm(network)
            logger.info(f"[adaptive_lr] epoch 1: loss={current_loss:.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} clip={clip_str} | ARMED")
            self._snapshot(network, optimizer)
            self._reset_clip()
            self.last_action = "ARMED"
            return self.last_action

        patience_up = 2
        patience_down = 2 if (self.stability_triggered or epoch == 1 or epoch >= 4) else 1
        cur_lr = optimizer.param_groups[0]["lr"]
        new_lr = cur_lr
        cur_wn = self._weight_norm(network)
        weight_growth = None
        if self.prev_weight_norm and self.prev_weight_norm > 0:
            weight_growth = (cur_wn - self.prev_weight_norm) / self.prev_weight_norm
        stability_reason = None
        if clip_ratio is not None and clip_ratio > self.CLIP_RATIO_THRESHOLD:
            stability_reason = f"grad clip {clip_ratio * 100:.0f}% > {self.CLIP_RATIO_THRESHOLD * 100:.0f}%"
        elif weight_growth is not None and weight_growth > self.WEIGHT_GROWTH_THRESHOLD:
            stability_reason = f"wnorm_Δ {weight_growth*100:+.0f}% > {self.WEIGHT_GROWTH_THRESHOLD*100:.0f}%"

        action, reason = "HOLD", ""
        if stability_reason is not None:
            self.stability_streak += 1
            stability_patience = 1 if not self.stability_triggered else 2
            if self.stability_streak >= stability_patience:
                candidate = max(cur_lr * self.FACTOR_DOWN, self.min_lr)
                note = ""
                if self.snapshot is not None:
                    self._rollback(network, optimizer)
                    note = f"; blended {int(self.BLEND*100)}/{int((1-self.BLEND)*100)} + optim restored"
                if candidate < cur_lr:
                    new_lr = candidate
                    action = "REDUCE+ROLLBACK" if self.snapshot is not None else "REDUCE"
                else:
                    action = "HOLD (floored)"
                reason = f"stability: {stability_reason}{note}"
                self.good_streak = self.bad_streak = self.stability_streak = 0
                self.stability_triggered = True
            else:
                action = "WAIT"
                reason = f"stability: {stability_reason}, streak {self.stability_streak}/{stability_patience}"
        elif self.best_loss is None or current_loss < self.best_loss:
            self.stability_streak = 0
            self.best_loss = current_loss
            self.good_streak += 1
            self.bad_streak = 0
            if self.good_streak >= patience_up:
                candidate = min(cur_lr * self.FACTOR_UP, self.max_lr)
                if candidate > cur_lr:
                    new_lr = candidate
                    action = "PROBE UP"
                    reason = f"loss improving, streak {self.good_streak}"
                else:
                    action = "HOLD (capped)"
                    reason = "loss improving, at max_lr"
                self.good_streak = 0
            else:
                reason = f"loss improving, streak {self.good_streak}/{patience_up}"
        else:
            self.stability_streak = 0
            self.bad_streak += 1
            self.good_streak = 0
            if self.bad_streak >= patience_down:
                candidate = max(cur_lr * self.FACTOR_DOWN, self.min_lr)
                if candidate < cur_lr:
                    new_lr = candidate
                    action = "REDUCE"
                    reason = f"loss plateau, streak {self.bad_streak}"
                else:
                    action = "HOLD (floored)"
                    reason = "loss plateau, at min_lr"
                self.bad_streak = 0
            else:
                reason = f"loss plateau, streak {self.bad_streak}/{patience_down}"

        if new_lr != cur_lr:
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr
        lr_str = f"{cur_lr:.2e}" if new_lr == cur_lr else f"{cur_lr:.2e}->{new_lr:.2e}"
        wn_str = f"{weight_growth*100:+.0f}%" if weight_growth is not None else "—"
        logger.info(f"[adaptive_lr] epoch {epoch + 1}: loss={current_loss:.4f} lr={lr_str} "
                    f"clip={clip_str} wnorm_Δ={wn_str} | {action} ({reason})")
        self.prev_weight_norm = cur_wn
        self._reset_clip()
        self._snapshot(network, optimizer)
        self.last_action = action
        return action
