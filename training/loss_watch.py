# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/families/loss_watch.py
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: auto-recaption uses TagScribeR's own captioners (any provider from inference/ - a local VLM,
# an OpenAI-compatible server or the WD tagger) instead of Fizgig's Krea 2 Qwen3-VL-4B; recaptions are written
# through core.caption_io (atomic, with daily backups); items come from training.dataset.TrainingDataset; caption
# re-encodes also refresh the optional shuffle variants. Verdict logic, files and the caption-fix contract are Fizgig's.
"""The per-image loss watch: detect problem images, per-image LR, look-outlier warm-up and auto-recaption, on the
shared PerImageLossWatch (training/loss_logger.py).

Run-folder files (the Train tab's Problem Images window reads and writes the same ones):
    loss_log/per_image_loss.jsonl         one JSON line per image-step (rotated to .bak on a fresh run)
    loss_log/problem_images.json          verdicts, rewritten at every epoch boundary
    loss_log/caption_updates.json         caption edits queued by the UI; claimed atomically at a boundary
    loss_log/caption_updates_applied.json per-key history of applied fixes (the row badge, the resume replay)
"""
import gc
import json
import logging
import os
import time

import torch

logger = logging.getLogger(__name__)


class Watch:
    """The run's loss watch and the state around it. `watch` is None when every toggle is off."""

    def __init__(self, output_dir, dataset, driver, *, log=False, per_image_lr=False, auto_recaption=False,
                 warmup_look=False, resume=False, start_epoch=0, te_path=None, trigger_word=None,
                 trigger_position="start", captioner=None):
        from training.loss_logger import PerImageLossWatch, is_enabled as env_on
        self.output_dir, self.dataset, self.driver, self.te_path = output_dir, dataset, driver, te_path
        self.trigger_word, self.trigger_position = trigger_word, trigger_position
        self.captioner = captioner or {}
        self.recaptioned = {}
        self.watch = None
        if not resume:
            _cleanup(output_dir)
        if auto_recaption and not self.captioner.get("model"):
            logger.warning("[auto-recaption] needs a captioner (Train tab > Loss watch > Captioner) - off")
            auto_recaption = False
        if not (log or per_image_lr or auto_recaption or warmup_look or env_on()):
            self.auto_recaption = False
            return
        cfg = dataset.config
        dirs = [d["image_directory"] for d in cfg["datasets"] if os.path.isdir(d.get("image_directory") or "")]
        self.image_dir = dirs[0] if dirs else None
        self.caption_ext = cfg.get("caption_extension") or ".txt"
        self.auto_recaption = auto_recaption
        if auto_recaption:
            logger.info(f"[auto-recaption] ON - stuck images re-captioned with {self.captioner.get('model')}"
                        + (f" (trigger: '{trigger_word}')" if trigger_word else ""))
        self.watch = PerImageLossWatch(output_dir, apply_lr=per_image_lr, write_jsonl=True,
                                       dataset_dir=self.image_dir, caption_ext=self.caption_ext)
        keys = set(dataset.items())
        self.watch.preflight(keys)
        logger.info(f"[loss-watch] per-image loss watch ON (per_image_lr={per_image_lr})")
        if warmup_look:
            self._warmup(keys)
        if resume and start_epoch > 0:
            self._resume(start_epoch)

    # ---- setup helpers -------------------------------------------------------------------------------
    def _warmup(self, keys):
        """Look-outlier warm-up, from a look-score file in the dataset folder ({cutoff, scores{stem: score}},
        Fizgig's Look Consistency Filter format). Without the file the warm-up is off for the run."""
        path = None
        for name in ("tagscriber_look_scores.json", "fizgig_look_scores.json"):
            p = os.path.join(self.image_dir or "", name)
            if os.path.isfile(p):
                path = p
                break
        if path is None:
            logger.warning("[look-warmup] no look-score file (tagscriber_look_scores.json / fizgig_look_scores.json) "
                           "in the dataset folder - warm-up off this run")
            return
        try:
            with open(path, encoding="utf-8") as f:
                look = json.load(f)
        except Exception as e:
            logger.warning(f"[look-warmup] could not load look scores ({e}) - warm-up off this run")
            return
        cut = look.get("cutoff")
        if cut is None:
            logger.warning("[look-warmup] no cutoff in the look-score file (too few scored faces) - warm-up off")
            return
        outliers = {k for k, v in (look.get("scores") or {}).items() if isinstance(v, (int, float)) and v < float(cut)}
        gone, use = sorted(outliers - keys), outliers & keys
        if gone:
            logger.info(f"[look-warmup] {len(gone)} scored outlier(s) not in the dataset - skipped")
        if use:
            self.watch.set_warmup_keys(use)
            logger.info(f"[look-warmup] {len(use)} look-outlier image(s) on LR warm-up x0.4 -> x1.0")

    def _resume(self, start_epoch):
        resets = {}
        try:
            with open(os.path.join(self.output_dir, "loss_log", "caption_updates_applied.json"),
                      encoding="utf-8") as f:
                for k, info in json.load(f).items():
                    for e in (info if isinstance(info, list) else [info]):
                        att, auto = int(e.get("attempt", 0) or 0), bool(e.get("auto"))
                        if auto:
                            self.recaptioned[k] = max(self.recaptioned.get(k, 0), att)
                        resets.setdefault(k, []).append((int(e.get("epoch", 0) or 0), att, auto))
        except Exception:
            pass
        self.watch.resume_from_jsonl(up_to_epoch=start_epoch, resets=resets)

    # ---- per step --------------------------------------------------------------------------------------
    def excluded(self, batch):
        return self.watch is not None and self.watch.is_excluded(batch.get("item_keys"))

    def multiplier(self, batch):
        return self.watch.multiplier(batch.get("item_keys")) if self.watch is not None else 1.0

    def observe(self, epoch, step, batch, t, loss):
        if self.watch is not None:
            self.watch.observe(epoch=epoch, step=step, item_keys=batch.get("item_keys"), timestep=t, loss=loss)

    # ---- epoch boundary --------------------------------------------------------------------------------
    def boundary(self, epoch, dit, device, parkable=True):
        """Reclassify, write problem_images.json, then apply queued caption edits and auto-recaptions."""
        if self.watch is not None:
            self.watch.epoch_boundary(epoch)
        try:
            self._captions(epoch, dit, device, parkable)
        except Exception:
            logger.warning("[caption-fix] caption repair failed this boundary - training continues", exc_info=True)

    def _captions(self, epoch, dit, device, parkable):
        path = os.path.join(self.output_dir, "loss_log", "caption_updates.json")
        processing = path + ".processing"
        updates = {}
        if os.path.exists(path):
            try:
                os.replace(path, processing)             # atomic claim: UI edits made meanwhile land in a fresh file
                with open(processing, encoding="utf-8") as f:
                    updates = {str(k): str(v).strip() for k, v in json.load(f).items() if str(v).strip()}
            except Exception:
                logger.warning("[caption-fix] could not read caption_updates.json - skipping", exc_info=True)
                return
        items = self.dataset.items()
        auto = []
        if self.auto_recaption and self.watch is not None:
            for k in sorted(k for k, v in self.watch.verdicts.items() if v == "stuck"):
                if k in updates or self.recaptioned.get(k, 0) >= 2:
                    continue
                img = getattr(items.get(k), "image_path", None)
                if img and os.path.exists(img):
                    auto.append((k, img, self.recaptioned.get(k, 0) + 1))
        if not updates and not auto:
            _remove(processing)
            return
        if not self.te_path:
            logger.warning("[caption-fix] caption work is pending but the run has no text encoder path; left queued")
            _requeue(path, processing, updates)
            return
        todo = [(k, items[k], c, 0) for k, c in updates.items() if k in items]
        for k in updates:
            if k not in items:
                logger.warning(f"[caption-fix] '{k}' not in the training set - skipped")
        auto = [a for a in auto if a[0] in items]
        if not todo and not auto:
            _remove(processing)
            return
        logger.info(f"[caption-fix] epoch boundary {epoch}: {len(todo)} manual edit(s), {len(auto)} stuck image(s) "
                    f"to auto-recaption...")
        from training import quant
        if parkable:
            quant.move(dit, "cpu")
            gc.collect()
            torch.cuda.empty_cache()
        ok = False
        try:
            if auto:
                todo += self._recaption(auto, items)
            if not todo:        # every stuck image failed to caption: nothing to re-encode, they retry next time
                _remove(processing)
                return
            self._reencode(todo, device)
            ok = True
        finally:
            if parkable:
                quant.move(dit, device)
        if not ok:     # already-written AI captions re-queue as if manual: no need to regenerate them
            _requeue(path, processing, {**updates, **{k: c for k, _, c, a in todo if a > 0}})
            return
        for k, _, _, attempt in todo:
            if attempt > 0:
                self.recaptioned[k] = max(self.recaptioned.get(k, 0), attempt)
        if self.watch is not None:
            for k, _, _, attempt in todo:
                self.watch.reset_key(k)
            for k, _, _, attempt in todo:
                if attempt >= 2:
                    self.watch.mark_incorrigible(k)
        _ack(self.output_dir, todo, epoch)
        _remove(processing)
        logger.info(f"[caption-fix] {len(todo)} caption(s) re-encoded - next epoch trains on the fixed text. "
                    f"Loss-watch history reset for: " + ", ".join(os.path.basename(k) for k, _, _, _ in todo))

    def _reencode(self, todo, device):
        """Re-encode the fixed captions into the cache (the loader reads these files every step)."""
        from training import cache
        refs = any(getattr(item, "control_paths", None) for _, item, _, _ in todo)
        te = (self.driver.load_reference_text_encoder(self.te_path, device) if refs
              else self.driver.load_text_encoder(self.te_path, device))
        try:
            fixed = []
            for _, item, cap, _ in todo:
                item.caption = cap
                if refs:
                    self.dataset.load_pixels(item)
                fixed.append(item)
            cache.encode_captions(self.driver, te, self.driver.description, self.dataset, fixed, references=refs)
        finally:
            self.driver.unload_text_encoder(te)
            del te
            gc.collect()
            torch.cuda.empty_cache()

    def _recaption(self, auto, items):
        """Caption the stuck images with TagScribeR's captioner (loaded for the call and freed before the family's
        encoder loads). Writes each caption file and returns (key, item, caption, attempt) rows to re-encode. A
        captioner that does not fit the card turns auto-recaption off for the run."""
        from PIL import Image

        from core.caption_io import write_caption
        from inference.base import CaptionRequest, GenerationParams, ProviderSpec
        from inference.manager import create_provider
        c = self.captioner
        prov = create_provider(ProviderSpec.make(c.get("kind", "transformers"), c["model"],
                                                 **dict(c.get("options") or {})))
        try:
            prov.load(lambda msg: logger.info(f"[auto-recaption] {msg}"))
        except torch.OutOfMemoryError:
            logger.warning("[auto-recaption] off for the rest of this run: the captioner does not fit this card's "
                           "VRAM. Training continues; manual caption edits still apply.")
            self.auto_recaption = False
            gc.collect()
            torch.cuda.empty_cache()
            return []
        except Exception:
            logger.warning("[auto-recaption] the captioner could not load - off for the rest of this run",
                           exc_info=True)
            self.auto_recaption = False
            return []
        rows = []
        try:
            for k, img_path, attempt in auto:
                try:
                    prompt = (c.get("prompt_detailed") if attempt >= 2 else None) or c.get("prompt") or \
                        "Describe this image in detail."
                    req = CaptionRequest(prompt=prompt, system_prompt=c.get("system_prompt", ""),
                                         params=GenerationParams(max_new_tokens=int(c.get("max_tokens", 384))))
                    with Image.open(img_path) as im:
                        result = prov.generate([im.convert("RGB")], req)[0]
                    if isinstance(result, Exception):
                        raise result
                    cap = " ".join(str(result).split())
                    if not cap:
                        raise ValueError("empty caption")
                    if self.trigger_word and self.trigger_word not in cap:
                        cap = (f"{cap}, {self.trigger_word}" if str(self.trigger_position) == "end"
                               else f"{self.trigger_word}, {cap}")
                    try:
                        write_caption(img_path, cap, self.caption_ext)
                    except OSError:
                        logger.warning(f"[auto-recaption] could not write the caption for {k} - this run is fixed, "
                                       "a future re-cache will use the old caption")
                    rows.append((k, items[k], cap, attempt))
                    logger.info(f"[auto-recaption] {os.path.basename(k)} (attempt {attempt}/2"
                                f"{', detailed' if attempt >= 2 else ''}): \"{cap[:110]}\"")
                except Exception:
                    logger.warning(f"[auto-recaption] captioning failed for {os.path.basename(k)} - retry next "
                                   f"boundary", exc_info=True)
        finally:
            try:
                prov.unload()
            except Exception:
                pass
            del prov
            gc.collect()
            torch.cuda.empty_cache()
        return rows


def _cleanup(output_dir):
    """Fresh run: clear the previous run's Problem Images artifacts; rotate (not delete) the research JSONL."""
    ll = os.path.join(output_dir, "loss_log")
    for f in ("problem_images.json", "problem_images.json.tmp", "caption_updates_applied.json",
              "caption_updates_applied.json.tmp", "caption_updates.json", "caption_updates.json.processing"):
        _remove(os.path.join(ll, f))
    jl = os.path.join(ll, "per_image_loss.jsonl")
    if os.path.exists(jl):
        try:
            os.replace(jl, jl + "." + time.strftime("%Y%m%d%H%M%S") + ".bak")
        except OSError:
            pass


def _remove(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass                     # the UI may hold it open; it is consumed next boundary


def _requeue(path, processing, updates):
    """Put a claimed batch back, merged with anything the UI queued meanwhile (newer edits win)."""
    if not updates:
        _remove(processing)
        return
    try:
        newer = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                newer = json.load(f)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump({**updates, **newer}, f, indent=2)
        os.replace(path + ".tmp", path)
        _remove(processing)
    except Exception:
        pass


def _ack(output_dir, todo, epoch):
    """Per-fix history per key (the UI's row badge, and the resume replay's reset timeline)."""
    path = os.path.join(output_dir, "loss_log", "caption_updates_applied.json")
    applied = {}
    try:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                applied = {k: (v if isinstance(v, list) else [v]) for k, v in json.load(f).items()}
    except Exception:
        applied = {}
    for k, _, cap, attempt in todo:
        applied.setdefault(k, []).append({"epoch": epoch, "caption": cap, "attempt": attempt, "auto": attempt > 0})
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(applied, f, indent=2)
        os.replace(path + ".tmp", path)
    except Exception:
        logger.warning("[caption-fix] could not write caption_updates_applied.json", exc_info=True)
