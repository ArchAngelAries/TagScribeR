# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/families/train.py
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR (the gaps docs/FIZGIG_TRAINING_AUDIT.md section 2.5 lists, filled as the plan approved):
#   * gradient accumulation (GRADIENT_ACCUMULATION, already in Fizgig's presets) and batch size > 1 for drivers with
#     supports_batching (the loss watch needs batch 1 and stands down otherwise, as Fizgig's GUI does);
#   * Klein's grad-clip-ratio signal fed to Adaptive LR (only for a family with adaptive_lr_clip_signal, as Fizgig);
#   * previews never end a run: a failure (usually OOM) disables them for the rest of it (Fizgig's Krea 2 policy),
#     and each epoch checkpoint's thumbnail is refreshed with its own preview (Fizgig #122);
#   * TagScribeR's dataset layer and captioners; a CPU device for smoke tests; `--config run.json`;
#   * `driver_options`: family extensions forwarded to driver.configure() (SDXL min-SNR gamma, noise offset, LoCon).
# The pause contract, state dirs, resume, schedulers, EMA, adapters and file naming are Fizgig's.
# Brought level with Fizgig 7.0.1 (commit 1c8ec88): the final save also kept under its epoch number (#176); a failed
# preview undoes exactly what it changed; CFG above 1 honoured with a speed LoRA; preview time left out of the bar's
# s/it; the resumed schedule fast-forwarded by whole optimizer steps with accumulation; an explicit thumbnail kept;
# previews on a preview checkpoint (_CheckpointPreviews, Klein's Distilled).
# Brought level with Fizgig 8.7.3 (the families/train.py changes since 7.0.1): the resume state saved with one retry
# (8.5.9), previews under torch.compiler.set_stance("force_eager") (7.2.2), the per-step loss log and the Sample now /
# Save now flags (8.0.0; training/step_log.py), each preview round's prompts recorded (8.7.3), slider previews ignoring
# the sample override (7.2.0), a prompt slider's prompts changed live at an epoch boundary (8.0.3) and its label band
# sized to the text, and the preview checkpoint taking a typed Steps / CFG (8.6.1 / 8.7.x). Not ported here: Fizgig's
# fine-tune mode, which TagScribeR does not have (so its "a fine-tune previews on the model it trains" guard has no
# counterpart yet).
"""The LoRA trainer for any family, driven through the family's driver.

    python -m training.train --config RUN_FOLDER/train_config.json

The loop is family-agnostic: bucketed cached data, frozen adapters (the family's training adapter, off for
previews; a context LoRA, on for previews; neither in saves), Adaptive LR or a step scheduler, gradient clipping and
accumulation, optional EMA, per-epoch checkpoints in the family's LoRA key format with SAI metadata, resumable state
dirs and the file-based pause contract. Everything model-specific comes from the driver.

Run-folder control files (written by the Train tab):
    .pause_requested         checked at each epoch boundary: save state, exit 0
    .sample_override.json    {prompt, seed, width, height} replaces the preview prompts at the next preview
                             (a slider run ignores it: its previews always show the slider's own prompt)
    .sample_now              a preview round at the next optimizer step; taken once, then removed
    .save_now                a checkpoint <name>-stepNNNNNNN.safetensors at the next optimizer step; taken once
    .slider_prompts.json     {base, pos, neg, guidance}: a prompt slider's new prompts, taken at the next epoch
                             boundary; the trainer answers in .slider_prompts_applied.json
Files the trainer writes for the Train tab:
    .tagscriber_loss_<name>.jsonl     one {"s","e","l","lr"} line a step (training/step_log.py)
    sample/.prompts_<name>.json       {"<epoch>": [prompts]}: what each preview round was rendered from
"""
import argparse
import datetime
import gc
import json
import logging
import math
import os
import random
import shutil
import sys
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from training import quant
from training.adaptive_lr import AdaptiveLR
from training.lora import FamilyLoRA
from training.optimizers import optimizer_lr
from training.step_log import StepLog, take_request
from training.metadata import (build_metadata, latest_sample_image, refresh_checkpoint_thumbnail, resolve_title,
                               sample_for_epoch, thumbnail_data_uri)
from training.registry import get as get_family
from training.train_utils import LossRecorder, prune_state_dirs, validate_output_name

logger = logging.getLogger("training.train")

ADAPTER = "training_adapter"
CONTEXT = "context"
SPEED = "speed_lora"


def _short_ema(ema_decay) -> bool:
    """EMA "Short run" (ema_decay "short"): the decay is sized to the run (Fizgig 7.0.1, H3's)."""
    return str(ema_decay).lower().startswith("short")


# ---- sliders (Fizgig 7.0.1 families/train.py): a signed dial, trained at +1 and -1 ------------------------------
class _SliderBank:
    """Prompt-pair sliders train on no dataset: the base model renders a bank of practice latents from the neutral
    prompt, and each step noises one of them (Fizgig _SliderBank). The TrainingDataset surface the loop uses."""

    def __init__(self, n, res):
        self.latents = [None] * max(1, int(n))
        self.num_items = len(self.latents)
        self.batch_size = 1
        self.buckets = {(res, res): []}
        self.config = {"resolution": [res, res], "datasets": []}
        self.has_control = False
        self._order = list(range(self.num_items))

    def prepare_for_training(self):
        return self.num_items

    def items(self):
        return {}

    def __len__(self):
        return self.num_items

    def shuffle(self, seed):
        self._order = random.Random(seed).sample(range(self.num_items), self.num_items)

    def get_batch(self, idx, rng=None):
        j = self._order[idx]
        return {"latents": self.latents[j], "item_keys": [f"slider_practice_{j:02d}"]}


# The dial is the point, so a slider preview shows it moving: one prompt, one seed, these strengths side by side.
SLIDER_PREVIEW_MULTIPLIERS = (-1.0, 0.0, 1.0)


def _slider_strip(frames, multipliers):
    """The strengths side by side, each labelled (Fizgig _slider_strip)."""
    from PIL import Image, ImageDraw, ImageFont
    w, h = frames[0].size
    gap = 8
    try:
        font = ImageFont.load_default(size=max(14, w // 40))
    except Exception:  # noqa: BLE001 - an older Pillow without sized default fonts
        font = ImageFont.load_default()
    labels = ["strength 0 (slider off)" if m == 0 else f"strength {m:+g}" for m in multipliers]
    # the band fits the text (the font grows with the picture): its full height, descenders too, padded evenly
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    top = min(probe.textbbox((0, 0), t, font=font)[1] for t in labels)
    bottom = max(probe.textbbox((0, 0), t, font=font)[3] for t in labels)
    pad = max(6, (bottom - top) // 3)
    band = (bottom - top) + 2 * pad
    strip = Image.new("RGB", (w * len(frames) + gap * (len(frames) - 1), h + band), (16, 16, 16))
    draw = ImageDraw.Draw(strip)
    for k, (im, label) in enumerate(zip(frames, labels)):
        x = k * (w + gap)
        strip.paste(im, (x, band))
        draw.text((x + 8, pad - top), label, fill=(236, 236, 236), font=font)
    return strip


def _pair_slider_caption(dataset):
    """The caption an image-pair slider's photos share (the most common one if they were edited apart)."""
    from collections import Counter
    from training.dataset import read_caption
    counts = Counter()
    ext = dataset.config.get("caption_extension") or ".txt"
    for it in dataset.items().values():
        text = str(getattr(it, "caption", "") or "").strip()
        if not text and getattr(it, "image_path", None):      # an item without its caption: the file itself
            try:
                text = read_caption(it.image_path, ext).strip()
            except Exception:  # noqa: BLE001 - an unreadable caption file is just not counted
                text = ""
        if text:
            counts[text] += 1
    return counts.most_common(1)[0][0] if counts else None


def _prompt_slider_step(driver, dit, net, latents, enc, gen, *, guidance, min_t, max_t):
    """Concept Sliders, textual form, on flow matching (Fizgig _prompt_slider_step). With the adapter at 0 the frozen
    model predicts the neutral, positive and negative prompts at one noised practice latent; the adapter then trains
    at +1 toward v_neutral + guidance * (v_pos - v_neg) and at -1 toward the mirror. Each pole is backpropagated before
    the flip (checkpointed blocks recompute at the strength of the moment). Returns the detached mean loss and t."""
    state = driver.noise_latents(latents, gen, min_t=min_t, max_t=max_t)
    n_c, p_c, g_c = enc
    try:
        net.set_trainable_multiplier(0.0)
        with torch.no_grad():
            v_n = driver.predict(dit, state, n_c).float()
            delta = float(guidance) * (driver.predict(dit, state, p_c).float() - driver.predict(dit, state, g_c).float())
        total = 0.0
        for m in (1.0, -1.0):
            net.set_trainable_multiplier(m)
            loss = F.mse_loss(driver.predict(dit, state, n_c).float(), v_n + m * delta)
            (0.5 * loss).backward()
            total += 0.5 * loss.item()
    finally:
        net.set_trainable_multiplier(1.0)
    return total, state["t"]


def _render_slider_bank(driver, dit, net, vae, dataset, cond, neg_cond, *, device, res, seed, steps, cfg, speed,
                        swapped):
    """A prompt slider's practice bank: the base model's own renders of the neutral prompt (adapter at 0, training
    adapter off, the speed LoRA on when `speed` - its SamplingSettings - is given), decoded and re-encoded into training
    latents by the driver's practice_latent (Fizgig 8.0.3 render_bank; TagScribeR 7.0.1's inline loop, as a function so a
    changed base prompt can re-render the bank between epochs). Fills dataset.latents in place."""
    n_c = {k: v.to(device) if hasattr(v, "to") else v for k, v in cond.items()}
    g_neg = ({k: v.to(device) if hasattr(v, "to") else v for k, v in neg_cond.items()}
             if neg_cond is not None else None)
    net.set_trainable_multiplier(0.0)
    net.set_enabled(ADAPTER, False)
    use_speed = speed is not None and net.has(SPEED)
    if use_speed:
        net.move_adapter(SPEED, device)
        net.set_enabled(SPEED, True)
    was_training = dit.training
    dit.eval()
    if swapped:
        driver.block_swap_mode(dit, inference=True)
    try:
        # eagerly: a compiled model would build a graph for the eval-mode render (see _eager_stance)
        with torch.no_grad(), driver.still_renders(), _eager_stance():    # stills, whatever the previews are
            for i in tqdm(range(dataset.num_items), desc="[slider] practice images"):
                if use_speed:
                    lat = driver.generate(dit, n_c, res, res, steps=speed.steps, seed=seed + 1000 + i, cfg=speed.cfg,
                                          sigmas=speed.sigmas, options=speed.options)
                else:
                    lat = driver.generate(dit, n_c, res, res, steps=steps, seed=seed + 1000 + i, cfg=cfg,
                                          **({"neg_cond": g_neg} if g_neg is not None else {}))
                dataset.latents[i] = driver.practice_latent(vae, lat, res)
    finally:
        if swapped:
            driver.block_swap_mode(dit, inference=False)
        dit.train(was_training)
        if use_speed:
            net.set_enabled(SPEED, False)
            net.move_adapter(SPEED, "cpu")
        net.set_enabled(ADAPTER, True)
        net.set_trainable_multiplier(1.0)
    logger.info("[slider] %d practice images rendered at %dx%d", dataset.num_items, res, res)


def _batched_slider_cond(raw, device):
    """A prompt slider's three conditionings as training conditioning (batched, as batch_cond gives it)."""
    return [{k: (v[None] if hasattr(v, "dim") else v) for k, v in
             {k: v.to(device) if hasattr(v, "to") else v for k, v in c.items()}.items()} for c in raw]


TRAINABLE_PREVIEW = "preview_lora"     # the epoch's LoRA, frozen on a preview checkpoint
PAUSE_FILE = ".pause_requested"
OVERRIDE_FILE = ".sample_override.json"
SLIDER_LIVE_FILE = ".slider_prompts.json"                  # a prompt slider's new prompts, set while it trains
SLIDER_LIVE_APPLIED_FILE = ".slider_prompts_applied.json"  # what the trainer took, and at which epoch
STATE_RETRY_WAIT_S = 5.0            # the pause before a failed resume-state save is tried once more (tests patch it)


def _step_scheduler(optimizer, kind, warmup, total, cycles=1, power=1.0):
    def f(s):
        if warmup and s < warmup:
            return (s + 1) / warmup
        if kind in ("constant", "constant_with_warmup"):
            return 1.0
        prog = min(1.0, (s - warmup) / max(1, total - warmup))
        if kind == "cosine":
            return 0.5 * (1 + math.cos(math.pi * prog))
        if kind == "cosine_with_restarts":
            return 0.5 * (1 + math.cos(math.pi * ((prog * cycles) % 1.0)))
        if kind == "linear":
            return 1.0 - prog
        if kind == "polynomial":
            return (1.0 - prog) ** power
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def _save_state(output_dir, output_name, net, optimizer, *, epoch, global_step, arch_id, extra=None, ema=None):
    state_dir = os.path.join(output_dir, f"{output_name}-{epoch:06d}-state")
    os.makedirs(state_dir, exist_ok=True)
    net.save(os.path.join(state_dir, "lora.safetensors"), dtype=torch.float32)
    torch.save(optimizer.state_dict(), os.path.join(state_dir, "optimizer.pt"))
    if ema is not None:
        torch.save(ema.state_dict(), os.path.join(state_dir, "ema.pt"))
    rng = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng, os.path.join(state_dir, "rng.pt"))
    with open(os.path.join(state_dir, "training_state.json"), "w", encoding="utf-8") as f:   # commit marker, last
        json.dump({"epoch": epoch, "global_step": global_step, "architecture": arch_id, **(extra or {})}, f)
    logger.info(f"[state] saved -> {state_dir}")
    return state_dir


def _torch_load(path, map_location):
    """torch.load for a resume file, restricted to tensors and plain types (weights_only): a state folder from
    someone else cannot run code. _save_state writes optimizer / EMA / RNG state only, all of which fit."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:  # noqa: BLE001 - pickle raises several types for a refused global
        raise RuntimeError(f"[resume] {os.path.basename(path)} could not be read safely ({type(e).__name__}: {e}). "
                           f"Only state folders written by TagScribeR's trainer can be resumed.") from e


def _load_state(state_dir, net, optimizer, device):
    for need in ("lora.safetensors", "optimizer.pt", "training_state.json"):
        if not os.path.isfile(os.path.join(state_dir, need)):
            raise RuntimeError(f"[resume] {state_dir} is not a saved training state (missing {need}). Pick the "
                               f"folder named like '<lora name>-000012-state'.")
    if net.load_trainable(os.path.join(state_dir, "lora.safetensors")) == 0:
        raise RuntimeError(f"[resume] {state_dir} matched none of this LoRA's modules - different rank or "
                           f"target modules?")
    optimizer.load_state_dict(_torch_load(os.path.join(state_dir, "optimizer.pt"), device))
    rng_path = os.path.join(state_dir, "rng.pt")
    if os.path.exists(rng_path):
        try:
            rng = _torch_load(rng_path, "cpu")
            torch.set_rng_state(rng["torch"])
            if "cuda" in rng and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception:
            logger.warning("[state] RNG restore failed; continuing with fresh RNG", exc_info=True)
    with open(os.path.join(state_dir, "training_state.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return int(meta.get("epoch", 0)), int(meta.get("global_step", 0)), meta


def _empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _preview_vram(tag, reset_peak=False):
    """One line of VRAM state at a preview waypoint."""
    try:
        if not torch.cuda.is_available():
            return
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()
        a = torch.cuda.memory_allocated() / 1024 ** 3
        r = torch.cuda.memory_reserved() / 1024 ** 3
        pk = torch.cuda.max_memory_reserved() / 1024 ** 3
        f = torch.cuda.mem_get_info()[0] / 1024 ** 3
        logger.info(f"[preview-vram] {tag}: allocated {a:.2f} GB, reserved {r:.2f} GB (peak {pk:.2f} GB), "
                    f"free {f:.2f} GB")
    except Exception:
        pass


def _small_card_previews():
    """Cards under 20 GB get the low-memory preview treatment: the training model parks on CPU for the VAE decode and
    the preview canvas caps at 768 px. TAGSCRIBER_PREVIEW_LOWMEM=1/0 forces it; TAGSCRIBER_SIM_VRAM_GB simulates a
    smaller card."""
    ov = os.environ.get("TAGSCRIBER_PREVIEW_LOWMEM", "").strip()
    if ov in ("0", "1"):
        return ov == "1"
    try:
        if not torch.cuda.is_available():
            return False
        sim = os.environ.get("TAGSCRIBER_SIM_VRAM_GB", "").strip()
        total = float(sim) if sim else torch.cuda.get_device_properties(0).total_memory / 1e9
        return total < 20.0
    except Exception:
        return False


def read_sample_override(output_dir):
    """The Train tab's live sample override (<output_dir>/.sample_override.json): {prompt, seed, width, height} while
    a prompt is set, else None."""
    path = os.path.join(output_dir, OVERRIDE_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        prompt = str(d.get("prompt", "")).strip()
        if prompt:
            return {"prompt": prompt, "seed": int(d.get("seed", 1234)), "width": int(d.get("width", 1024)),
                    "height": int(d.get("height", 1024))}
    except (OSError, ValueError, TypeError):
        pass
    return None


def _record_prompts(sample_dir, output_name, epoch, prompts):
    """The prompts a preview round used (<sample_dir>/.prompts_<name>.json, {"<epoch>": [prompts]}), so the gallery shows
    each preview's own prompt after a mid-run change, not the run's first ones (Fizgig 8.7.3). The key is the epoch
    number the round's pictures are named with (0 = before training; a Sample now between epochs uses the epoch in
    progress, as its pictures do, and a later round of the same epoch replaces it). Written whole and atomically; a
    folder that cannot be written costs the record, never the run."""
    path = os.path.join(sample_dir, f".prompts_{output_name}.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    data[str(epoch)] = [str(p) for p in prompts]
    try:
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(path + ".tmp", path)
    except OSError as e:
        logger.warning(f"[sample] could not record the preview prompts ({e})")


def _read_slider_live(output_dir):
    """A prompt slider's new prompts, set while it trains (<output_dir>/.slider_prompts.json): {base, pos, neg,
    guidance} or None - the form's own three (what the picture is, what each end adds), all needed, as at launch
    (Fizgig 8.0.3)."""
    try:
        with open(os.path.join(output_dir, SLIDER_LIVE_FILE), encoding="utf-8") as f:
            d = json.load(f)
        out = {"base": str(d.get("base") or "").strip(), "pos": str(d.get("pos") or "").strip(),
               "neg": str(d.get("neg") or "").strip(), "guidance": float(d.get("guidance"))}
        return out if out["base"] and out["pos"] and out["neg"] and out["guidance"] > 0 else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _eager_stance():
    """torch.compiler.set_stance("force_eager") as a context manager, or a no-op on a torch build without set_stance.
    Previews run the compiled blocks eagerly (Fizgig 7.2.2): eval mode, the preview size and the speed LoRA are each a
    new graph, and building them took minutes of Triton / MSVC work on the first preview after every start or resume;
    a handful of eager steps costs seconds. Training keeps its compiled graphs."""
    import contextlib
    set_stance = getattr(getattr(torch, "compiler", None), "set_stance", None)
    if set_stance is None:
        return contextlib.nullcontext()
    try:
        return set_stance("force_eager")
    except Exception:  # noqa: BLE001 - an older build that lists the call but refuses it
        return contextlib.nullcontext()


def _encode_override(driver, te_path, prompt, dit, device, parkable=True, references=None):
    """Encode one override prompt (or a list of prompts, e.g. a slider's three) mid-run. The text encoder loads beside
    the training model when it fits; otherwise the model waits on CPU for the encode. Returns one conditioning dict per
    prompt."""
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    te_gb = os.path.getsize(te_path) / 1024 ** 3 if te_path and os.path.exists(te_path) else 0.0
    park = parkable and torch.cuda.is_available() and quant.free_vram_gb() < te_gb + 2.0
    if park:
        quant.move(dit, "cpu")
        _empty_cache()
    try:
        te = driver.load_reference_text_encoder(te_path, device) if references else \
            driver.load_text_encoder(te_path, device)
        try:
            if references:
                return driver.encode_text_with_references(te, prompts, [references] * len(prompts))
            return driver.encode_text(te, prompts)
        finally:
            driver.unload_text_encoder(te)
            del te
            _empty_cache()
    finally:
        if park:
            quant.move(dit, device)


def _cap_canvas(width, height, cap=768):
    long = max(width, height)
    if long <= cap:
        return width, height
    s = cap / long
    return max(64, int(width * s) // 32 * 32), max(64, int(height * s) // 32 * 32)


def _reference_size(w, h, area):
    """An edit preview's canvas: the reference's aspect at `area` pixels, in multiples of 64."""
    r = w / h
    return max(64, round((area * r) ** 0.5 / 64) * 64), max(64, round((area / r) ** 0.5 / 64) * 64)


def _load_references(paths, width, height):
    """uint8 (H, W, 3) arrays of the preview's reference images at the first one's aspect and the preview's area,
    and that (width, height)."""
    import numpy as np
    from PIL import Image, ImageOps
    imgs = [Image.open(p).convert("RGB") for p in paths]
    size = _reference_size(*imgs[0].size, width * height)
    return [np.array(ImageOps.fit(im, size, Image.LANCZOS)) for im in imgs], size


def _render_previews(driver, dit, net, vae, encoded, out_dir, epoch, *, output_name, steps, cfg, neg, width,
                     height, seed, ema=None, speed=None, lowmem=False, swapped=False, refs=None, slider=False):
    """Previews on the RESIDENT training model: training adapter OFF (the deployment setup), the family's speed LoRA
    ON if one is loaded (it lives on CPU between previews), EMA weights swapped in. On small cards the model parks on
    CPU for the decode. A slider's preview is its dial at -1 / 0 / +1 side by side. File names:
    <name>_e<epoch:06d>_<idx:02d>_<timestamp>_<seed>.png."""
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    device = next(iter(dit.parameters())).device
    logger.info(f"[sample] epoch {epoch}: rendering {len(encoded)} preview(s)")
    _preview_vram("preview start", reset_peak=True)
    paths = []
    was_training = dit.training
    park = lowmem and not swapped      # a swapped model is already mostly on CPU; moving it would undo the layout
    done = set()           # what this preview has changed, so the finally undoes exactly that, wherever it failed
    try:
        net.set_enabled(ADAPTER, False)
        done.add("adapter")
        if speed is not None:
            done.add("speed")              # before the move: a move that runs out of memory still gets put back
            net.move_adapter(SPEED, device)
            net.set_enabled(SPEED, True)
        if ema is not None:
            ema.swap_in()
            done.add("ema")
        dit.eval()
        if swapped:
            done.add("swap")
            driver.block_swap_mode(dit, inference=True)
        ref_kw = {"refs": [r.to(device) for r in refs]} if refs else {}
        lats = []
        mults = SLIDER_PREVIEW_MULTIPLIERS if slider else (None,)
        eager = _eager_stance()
        eager.__enter__()
        done.add("eager")
        for i, cond in enumerate(encoded):
            cond = {k: v.to(device) if hasattr(v, "to") else v for k, v in cond.items()}
            for m in mults:
                if m is not None:
                    done.add("slider")
                    net.set_trainable_multiplier(m)
                if speed is not None:
                    # the speed LoRA's own CFG, unless the Samples tab asks for more (then the negative applies too)
                    _cfg = cfg if cfg and cfg > 1.0 else speed.cfg
                    lats.append(driver.generate(dit, cond, width, height, steps=steps, seed=seed + i, cfg=_cfg,
                                                neg_cond=neg if _cfg > 1.0 else None,
                                                sigmas=speed.sigmas, options=speed.options, **ref_kw))
                else:
                    lats.append(driver.generate(dit, cond, width, height, steps=steps, seed=seed + i, cfg=cfg,
                                                neg_cond=neg, **ref_kw))
        tok = driver.park_for(dit, device, None, "decode")   # None = the shared rule below
        if tok is not None:
            done.add("decode_park")
        park = park and tok is None
        if park:                            # never hold the training model and the VAE decode together
            done.add("lowmem_park")
            _preview_vram("before decode")
            quant.move(dit, "cpu")
            _empty_cache()
            _preview_vram("model parked for the decode")
        per = len(mults)
        for i in range(len(lats) // per):
            p = os.path.join(out_dir, f"{output_name}_e{epoch:06d}_{i:02d}_{ts}_{seed + i}.png")
            frames = [driver.decode(vae, lat, width, height) for lat in lats[i * per:(i + 1) * per]]
            if slider:
                strip = driver.slider_preview(frames, SLIDER_PREVIEW_MULTIPLIERS)
                paths += driver.save_preview(_slider_strip(frames, SLIDER_PREVIEW_MULTIPLIERS) if strip is None
                                             else strip, p)
            else:
                paths += driver.save_preview(frames[0], p)
    finally:
        if "eager" in done:
            eager.__exit__(None, None, None)
        if "slider" in done:
            net.set_trainable_multiplier(1.0)  # a preview that failed mid-dial must not leave the LoRA scaled
        if "decode_park" in done:
            driver.unpark(dit, device, tok)
        if "lowmem_park" in done:
            quant.move(dit, device)
            _preview_vram("after decode, model restored")
        if "swap" in done:
            driver.block_swap_mode(dit, inference=False)
        if "ema" in done:
            ema.swap_out()
        if "speed" in done:
            net.set_enabled(SPEED, False)
            net.move_adapter(SPEED, "cpu")
        if "adapter" in done:
            net.set_enabled(ADAPTER, True)
        dit.train(was_training)
        _empty_cache()
        _preview_vram("after preview cleanup")
    logger.info(f"[sample] epoch {epoch}: {len(paths)} preview(s) -> {out_dir}")
    return paths


class _CheckpointPreviews:
    """Training previews on the family's preview checkpoint (train_preview_checkpoint: Klein's Distilled), with the old
    Klein trainer's memory handoff (Fizgig 7.0.1 families/train.py): the driver parks the training model, loads the
    checkpoint (its own swap by card, optionally INT8), the epoch's LoRA - the live adapter, EMA weights if EMA is on -
    and the context LoRA ride on it as frozen adapters, and the training model is put back exactly. Between epochs the
    checkpoint stays in system RAM when it isn't block-swapped and the cache mode allows (auto: decided once per run,
    free RAM >= 18 GB); any failure to reuse it falls back to a fresh load."""

    def __init__(self, driver, path, device, cache_mode="auto", int8=False, context=None, scratch_dir=None):
        self.driver, self.path, self.device = driver, path, torch.device(device)
        self.cache_mode, self.int8, self.context = str(cache_mode or "auto").lower(), bool(int8), context
        self.cached = None                 # (model, FamilyLoRA, swapped) on the CPU between epochs
        self._auto = None
        import tempfile
        self._lora_file = os.path.join(scratch_dir or tempfile.gettempdir(),
                                       f"tagscriber_preview_lora_{os.getpid()}.safetensors")

    def _cache_on(self, swapped):
        if self.cache_mode == "off" or swapped:
            return False
        if self.cache_mode == "on":
            return True
        if self._auto is None:
            try:
                import psutil
                self._auto = psutil.virtual_memory().available / 1e9 >= 18.0
            except Exception:
                self._auto = False
        return self._auto

    def render(self, dit, net, ema, render):
        """Park, put the checkpoint up with this epoch's LoRA, call render(model, adapters, swapped), restore."""
        if ema is not None:
            ema.swap_in()
        try:
            net.save(self._lora_file, dtype=torch.bfloat16)
        finally:
            if ema is not None:
                ema.swap_out()
        rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        token = self.driver.park_for_preview(dit, self.device)
        m = fl = None
        swapped = 0
        try:
            gc.collect()
            _empty_cache()
            if self.cached is not None:
                try:
                    m, fl, swapped = self.cached
                    self.cached = None
                    quant.move(m, self.device)
                    fl.swap_file(TRAINABLE_PREVIEW, self._lora_file)
                    logger.info("[sample] preview checkpoint reused from RAM (no disk reload)")
                except Exception:
                    logger.warning("[sample] preview checkpoint reuse failed - reloading from disk", exc_info=True)
                    m = fl = None
            if m is None:
                m, swapped = self.driver.load_preview_checkpoint(self.path, self.device, int8=self.int8)
                fl = FamilyLoRA(m, self.driver, device=self.device)
                fl.add_file(self._lora_file, TRAINABLE_PREVIEW)
                if self.context:
                    fl.add_file(self.context[0], CONTEXT, self.context[1])
                logger.info(f"[sample] previews on {os.path.basename(self.path)}"
                            + (f" ({swapped} blocks streamed)" if swapped else ""))
            return render(m, fl, swapped)
        finally:
            keep = m is not None and self._cache_on(swapped)
            if keep:
                try:
                    quant.move(m, "cpu")
                    self.cached = (m, fl, swapped)
                except Exception:
                    logger.warning("[sample] could not keep the preview checkpoint in RAM", exc_info=True)
                    self.cached = None
            del m, fl
            gc.collect()
            gc.collect()
            _empty_cache()
            self.driver.unpark_after_preview(dit, self.device, token)
            torch.set_rng_state(rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            try:
                os.remove(self._lora_file)
            except OSError:
                pass


def train_family(family, dit_path, dataset_config, output_dir, output_name, *, network_dim=32, network_alpha=32,
                 learning_rate=1e-4, max_train_epochs=16, save_every_n_epochs=1, save_state=False,
                 save_state_on_train_end=False, keep_last_n_states=2, seed=42, precision="bf16",
                 training_adapter=None, training_adapter_strength=1.0,
                 context_lora_path=None, context_lora_strength=1.0, min_timestep=0.0, max_timestep=1.0,
                 speed_lora=None, speed_lora_strength=None,
                 vae_path=None, te_path=None, sample_prompts=None, sample_every_n_epochs=0, sample_width=None,
                 sample_height=None, sample_steps=None, sample_cfg_scale=None, sample_negative=None,
                 sample_at_first=False, sample_seed=42, sample_reference=None,
                 metadata_title=None, metadata_author=None, metadata_description=None, metadata_license=None,
                 metadata_tags=None, metadata_trigger_phrase=None, metadata_thumbnail=None,
                 resume_state_dir=None, adaptive_lr=False, adaptive_lr_min=1e-4, adaptive_lr_max=2e-4,
                 max_grad_norm=1.0, ema_decay=0.0, optimizer_type="adamw", optimizer_args="",
                 lr_scheduler="constant", lr_warmup_steps=0, lr_scheduler_num_cycles=1, lr_scheduler_power=1.0,
                 gradient_checkpointing=True, blocks_to_swap=0, network_type="lora", lokr_factor=8,
                 gradient_accumulation=1,
                 log_per_image_loss=False, per_image_lr=False, auto_recaption=False, warmup_look_outliers=False,
                 trigger_word=None, trigger_position="start", captioner=None, device=None, driver_options=None,
                 train_blocks=None, preview_checkpoint=None, preview_checkpoint_cache="auto", preview_int8=False,
                 slider_pairs=False, slider_diff_weight=1.0, slider_prompts=None, slider_guidance=None,
                 slider_bank=16, slider_bank_res=768):
    desc = get_family(family)
    if desc is None or not desc.training_ready:
        raise RuntimeError(f"unknown or untrainable family {family!r}")
    validate_output_name(output_name)
    driver = desc.load_driver()
    driver_options = dict(driver_options or {})
    compile_mode = driver_options.pop("compile_blocks", None)     # Compile Blocks is the loop's (training/compile.py)
    if driver_options:                  # family extensions (e.g. SDXL min-SNR gamma): the Train tab's DRIVER_OPTIONS
        driver.configure(**driver_options)
    arch = desc.arch_id
    gradient_accumulation = max(1, int(gradient_accumulation or 1))
    speed_desc = desc.preview_speed() if speed_lora else None
    if speed_lora and speed_desc is None:
        logger.warning(f"[sample] {desc.display_name} declares no preview speed LoRA - ignoring the speed LoRA")
        speed_lora = None
    if speed_desc is not None and speed_lora_strength is not None and speed_lora_strength <= 0:
        logger.info(f"[sample] {speed_desc.name} at strength 0 - previews render without it")
        speed_lora = speed_desc = None
    asked_steps, asked_cfg = sample_steps, sample_cfg_scale       # a preview checkpoint takes these when given
    sample_steps = sample_steps or (speed_desc.settings.steps if speed_desc else desc.preview_steps)
    sample_cfg_scale = desc.preview_cfg if sample_cfg_scale is None else sample_cfg_scale
    sample_width = sample_width or desc.preview_width
    sample_height = sample_height or desc.preview_height
    lowmem = _small_card_previews()
    if lowmem and max(sample_width, sample_height) > 768:
        sample_width, sample_height = _cap_canvas(sample_width, sample_height)
        logger.info(f"[sample] card under 20 GB: preview canvas capped to {sample_width}x{sample_height} and the "
                    f"model parks on CPU for the decode. TAGSCRIBER_PREVIEW_LOWMEM=0 turns this off.")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    quant.apply_vram_cap()
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # ---- sliders: a signed dial, trained at +1 and -1 (Fizgig 7.0.1) -------------------------------------
    slider = bool(slider_pairs or slider_prompts)
    slider_guidance = desc.slider_guidance if slider_guidance is None else float(slider_guidance)
    if slider:
        if not desc.slider_training:
            raise RuntimeError(f"{desc.display_name} does not offer slider training")
        if slider_pairs and slider_prompts:
            raise RuntimeError("[slider] image pairs and prompt pairs are two different sliders - pick one")
        if slider_prompts and (len(slider_prompts) != 3 or not all(str(x).strip() for x in slider_prompts[1:])):
            raise RuntimeError("[slider] prompt pairs need NEUTRAL POSITIVE NEGATIVE (the two poles non-empty)")
        if network_type != "lora":
            logger.info("[slider] network type %s -> LoRA: a slider is a plain LoRA whose strength is the dial",
                        network_type)
            network_type = "lora"
        if adaptive_lr or ema_decay or log_per_image_loss or per_image_lr or auto_recaption or warmup_look_outliers:
            logger.info("[slider] adaptive LR, weight averaging and the per-image loss watch are off: they assume "
                        "one target per image, and a slider step has two")
        adaptive_lr, ema_decay = False, 0.0
        log_per_image_loss = per_image_lr = auto_recaption = warmup_look_outliers = False
        if gradient_accumulation > 1:
            logger.info("[grad_accum] off for sliders: each step already backpropagates both ends")
        gradient_accumulation = 1
        if slider_pairs:
            logger.info("[slider] IMAGE-PAIR SLIDER: the adapter trains at +1 toward each image and at -1 toward "
                        "its pair (difference weight %g)", float(slider_diff_weight))
        else:
            logger.info("[slider] PROMPT-PAIR SLIDER: base '%s' | +1 -> '%s' | -1 -> '%s' (guidance %g)",
                        *slider_prompts, float(slider_guidance))
            if not (te_path and vae_path):
                raise RuntimeError("[slider] prompt pairs need the text encoder and VAE paths")
            if str(slider_prompts[0]).strip():     # the dial is shown on the picture it was trained on
                sample_prompts = [str(slider_prompts[0]).strip()]

    # ---- data ------------------------------------------------------------------------------------
    from training.dataset import TrainingDataset
    if slider_prompts:
        dataset = _SliderBank(slider_bank, int(slider_bank_res))       # filled once the model is loaded
    else:
        if not dataset_config:
            raise RuntimeError("a dataset is required (only a prompt-pair slider trains without one)")
        dataset = TrainingDataset(dataset_config, arch, desc.spatial_factor, desc.bucket_step)
    if slider and dataset.batch_size > 1:
        raise RuntimeError("[slider] sliders train at batch size 1 (each step already trains both ends)")
    if dataset.batch_size > 1 and not driver.supports_batching:
        raise RuntimeError(f"{desc.display_name} trains at batch size 1 (conditioning lengths differ per image). "
                           f"Set Batch Size to 1 - Gradient Accumulation gives the same effective batch.")
    if dataset.prepare_for_training() == 0:
        raise RuntimeError("No training items - run the cache stages first (Enable Cache Preparation).")
    if slider_pairs:
        if not dataset.has_control:
            raise RuntimeError("[slider] an image-pair slider needs the -1 end folder (every +1 photo's pair, matched "
                               "by file name)")
        shared = _pair_slider_caption(dataset)
        if shared:                           # the dial is shown on what both ends of a pair have in common
            sample_prompts = [shared]
            logger.info("[slider] previews use the pairs' caption: '%s'", shared)
    if slider:
        driver.slider_setup(dataset)
    steps_per_epoch = len(dataset)
    opt_steps_per_epoch = math.ceil(steps_per_epoch / gradient_accumulation)
    if dataset.batch_size > 1 and (log_per_image_loss or per_image_lr or auto_recaption or warmup_look_outliers):
        logger.info("[loss-watch] off: problem-image detection, per-image LR, auto-recaption and the look warm-up "
                    "need batch size 1")
        log_per_image_loss = per_image_lr = auto_recaption = warmup_look_outliers = False
    logger.info(f"{desc.display_name} training: {dataset.num_items} items, {max_train_epochs} epochs, "
                f"{steps_per_epoch} steps/epoch (batch {dataset.batch_size}"
                + (f", {opt_steps_per_epoch} optimizer steps with accumulation {gradient_accumulation}"
                   if gradient_accumulation > 1 else "") + ")")

    # ---- Auto precision / swap: planned on an empty card -------------------------------------------
    auto_precision = precision == "auto"
    if precision == "auto" or blocks_to_swap < 0:
        req = (precision, blocks_to_swap)
        res = dataset.config["resolution"]
        mp = res[0] * res[1] / 1e6
        own = None if device.type == "cpu" else driver.plan_run(precision, blocks_to_swap, run=dict(
            dit_path=dit_path, network_type=network_type, network_dim=network_dim, lokr_factor=lokr_factor,
            optimizer_type=optimizer_type, training_adapter=training_adapter, context_lora_path=context_lora_path,
            ema_decay=0.98 if _short_ema(ema_decay) else ema_decay, batch_size=dataset.batch_size, megapixels=mp))
        if device.type == "cpu":
            precision, blocks_to_swap, why = (desc.precisions[0], 0, "CPU run: first precision, no swap")
        elif own is not None:
            precision, blocks_to_swap, why = own
        else:
            precision, blocks_to_swap, why = quant.plan(desc, driver, precision, blocks_to_swap, megapixels=mp)
        why += f" at {mp:.2f} MP"
        logger.info(f"[precision] Auto plan: {precision}, block swap {blocks_to_swap} ({why}); asked {req}")

    # ---- torch.compile, decided here on the empty card (the free VRAM its checks read would be the loaded model's
    # leftovers later) and applied after every LoRA has patched the forwards. Auto weighs the warm-up against this
    # run's length; On places the checkpoint where it fits at the largest bucket.
    do_compile = False
    res_max = max((w * h for w, h in dataset.buckets), default=0) / 1e6
    cb = str(compile_mode or "off").strip().lower().split(" ")[0]
    cb = {"1": "on", "true": "on", "yes": "on"}.get(cb, cb)
    if desc.compiles and cb != "off":
        do_compile, why = driver.compile_plan(cb, dataset.num_items * max_train_epochs, precision,
                                              max(0, blocks_to_swap) if precision != "nf4" else 0,
                                              mp=res_max or 0.25, batch=dataset.batch_size)
        if cb == "auto":
            logger.info("[compile] auto: %s - %s", "ENABLED (checkpoint outside)" if do_compile == "outside"
                        else ("ENABLED" if do_compile else "off"), why)
        elif why:
            logger.info("[compile] %s", why)
    if auto_precision and not do_compile and device.type != "cpu":
        alt = driver.auto_uncompiled_precision(dit_path, precision)
        if alt and alt != precision:
            logger.info(f"[precision] Auto: {alt} instead of {precision} - this run is not compiled, and uncompiled "
                        "that is the faster base for this file")
            precision = alt

    # ---- previews: encode prompts once, keep the VAE ---------------------------------------------
    encoded = neg = vae = ref_imgs = ref_latents = None
    slider_enc = slider_raw = slider_neg = None
    if slider_prompts:
        te = driver.load_text_encoder(te_path, device)
        slider_raw = slider_enc = driver.encode_text(te, [str(x) for x in slider_prompts])   # unbatched; batched below
        if sample_cfg_scale > 1.0:
            # the practice pictures render as previews do: the Samples tab's negative (else the family's default)
            text = sample_negative if sample_negative is not None else (desc.preview_negative or "")
            slider_neg = driver.encode_text(te, [text])[0]
        driver.unload_text_encoder(te)
        del te
        _empty_cache()
    sample_dir = os.path.join(output_dir, "sample")
    if sample_reference and not driver.supports_references:
        logger.warning(f"[sample] {desc.display_name} has no edit previews - ignoring the reference photo")
        sample_reference = None
    if sample_prompts and sample_every_n_epochs and te_path and vae_path:
        logger.info("[sample] encoding %d preview prompt(s) with %s", len(sample_prompts), desc.text_encoder_label)
        if sample_reference:
            ref_imgs, (sample_width, sample_height) = _load_references(sample_reference, sample_width, sample_height)
            logger.info(f"[sample] edit previews from {len(ref_imgs)} reference image(s) at "
                        f"{sample_width}x{sample_height}")
            te = driver.load_reference_text_encoder(te_path, device)
            encoded = driver.encode_text_with_references(te, sample_prompts, [ref_imgs] * len(sample_prompts))
            if sample_cfg_scale > 1.0:
                neg = driver.encode_text_with_references(te, [sample_negative or ""], [ref_imgs])[0]
        else:
            te = driver.load_text_encoder(te_path, device)
            encoded = driver.encode_text(te, sample_prompts)
            if sample_cfg_scale > 1.0:
                neg = driver.encode_text(te, [sample_negative or ""])[0]
        driver.unload_text_encoder(te)
        del te
        _empty_cache()
        vae = driver.load_vae(vae_path, device)
        if ref_imgs:
            ref_latents = [z[None].cpu() for z in driver.encode_images(vae, ref_imgs)]
    elif sample_prompts and sample_every_n_epochs:
        logger.warning("[sample] previews need the text encoder and VAE paths - previews are off for this run")

    # ---- model ----------------------------------------------------------------------------------
    logger.info(f"Loading {desc.display_name} model ({precision}) from {dit_path}")
    dit, swapped = quant.load_base(driver, dit_path, device, precision, blocks_to_swap)
    if gradient_checkpointing:
        driver.enable_gradient_checkpointing(dit, True)
    net = FamilyLoRA(dit, driver, device=device)
    if training_adapter:
        n = net.add_file(training_adapter, ADAPTER, training_adapter_strength)
        if n == 0:
            raise RuntimeError(f"Training adapter {training_adapter} matched no {desc.display_name} modules.")
        driver.frozen_file_added(dit, training_adapter, training_adapter_strength, "adapter")
        logger.info(f"[adapter] training adapter ON ({n} modules, strength {training_adapter_strength:g}) - frozen, "
                    f"off in previews, not saved into the LoRA")
    elif desc.training_adapter:
        logger.warning("[adapter] no training adapter for this run")
    if context_lora_path:
        n = net.add_file(context_lora_path, CONTEXT, context_lora_strength)
        driver.frozen_file_added(dit, context_lora_path, context_lora_strength, "context")
        logger.info(f"[context] {os.path.basename(context_lora_path)} frozen + active at {context_lora_strength:g} "
                    f"({n} modules)")
    if speed_lora and encoded is not None:
        _ss = speed_desc.strength if speed_lora_strength is None else speed_lora_strength
        n = net.add_file(speed_lora, SPEED, _ss)
        if n:
            driver.frozen_file_added(dit, speed_lora, _ss, "speed")
        if n == 0:
            net.remove(SPEED)
            logger.warning(f"[sample] {os.path.basename(speed_lora)} matched no {desc.display_name} layers "
                           f"(a turbo LoRA for another model?) - previews render without it, at "
                           f"{desc.preview_steps} steps")
            speed_lora = None
            sample_steps = desc.preview_steps
        else:
            net.set_enabled(SPEED, False)
            net.move_adapter(SPEED, "cpu")
            logger.info(f"[sample] {speed_desc.name}: {n} modules, on CPU between previews, on only while they "
                        f"render ({sample_steps} steps)")
    if network_type == "lokr" and "lokr" not in desc.network_types:
        raise RuntimeError(f"{desc.display_name} does not offer LoKR")
    # a family's own block choice (Klein's Model Area); a slider trains the full model, as in Fizgig 7.0.1
    blocks = None if slider else driver.trainable_blocks()
    if train_blocks:                    # a subset of blocks (Fast Identity Mode)
        known = {b.id for g in driver.block_map(dit) for b in g.blocks}
        unknown = sorted(set(train_blocks) - known)
        if unknown:
            raise RuntimeError(f"train_blocks: {', '.join(unknown)} are not {desc.display_name} blocks")
        blocks = set(train_blocks)
        logger.info(f"[blocks] the LoRA trains {len(blocks)} of {len(known)} blocks only: {', '.join(train_blocks)}")
    net.add_trainable(network_dim, network_alpha, blocks=blocks, kind=network_type, factor=lokr_factor)
    trained_blocks = [b.id for g in driver.block_map(dit) for b in g.blocks if b.id in blocks] if blocks else []
    params = net.parameters()
    if not params:
        raise RuntimeError("the LoRA has no trainable parameters (no target modules matched)")
    logger.info((f"LoKR factor {lokr_factor}" if network_type == "lokr" else
                 f"LoRA rank {network_dim} alpha {network_alpha:g}") +
                f": {len(net.trainable_modules())} modules, {sum(p.numel() for p in params) / 1e6:.2f}M trainable "
                f"params")

    driver.prepare_training(dit, net, precision=precision, blocks_to_swap=int(swapped or 0),
                            total_steps=dataset.num_items * max_train_epochs, megapixels=res_max,
                            batch_size=dataset.batch_size)
    def render_bank():
        """The practice bank for the current neutral prompt (see _render_slider_bank); loads the VAE the first time."""
        nonlocal vae
        if vae is None:
            vae = driver.load_vae(vae_path, device)
        use_speed = speed_lora is not None and speed_desc is not None and net.has(SPEED)
        _render_slider_bank(driver, dit, net, vae, dataset, slider_raw[0], slider_neg, device=device,
                            res=int(slider_bank_res), seed=seed, steps=desc.preview_steps, cfg=sample_cfg_scale,
                            speed=speed_desc.settings if use_speed else None, swapped=bool(swapped))

    if slider_prompts:
        render_bank()
        _empty_cache()
        # the three prompts as training conditioning (batched, as batch_cond gives it)
        slider_enc = _batched_slider_cond(slider_raw, device)
    if do_compile:                      # decided before the load, applied last
        driver.compile_blocks(dit, "outside" if do_compile == "outside" else "inside", int(swapped or 0), precision)

    from training.optimizers import create_optimizer, group_rates, owns_its_rate
    # the family may structure the optimizer's parameters (Krea 2: Automagic v3 per-family groups, Fizgig krea2/trainer.py)
    opt_params, optimizer_args, fam_counts = driver.optimizer_params(net, optimizer_type, learning_rate, optimizer_args)
    if (desc.optimizer_weight_decay is not None and "weight_decay" not in (optimizer_args or "")
            and "adam" in str(optimizer_type or "").lower()):
        optimizer_args = (f"{optimizer_args or ''} weight_decay={desc.optimizer_weight_decay:g}").strip()
    optimizer, opt_label = create_optimizer(optimizer_type, opt_params, learning_rate, optimizer_args,
                                            eps_floor_8bit=desc.optimizer_eps_floor_8bit)
    if fam_counts and len(optimizer.param_groups) > 1:
        logger.info("[optimizer] per-family rates: "
                    + ", ".join(f"{g['family']} ({fam_counts.get(g['family'], 0)} modules)"
                                for g in optimizer.param_groups) + " - each votes its own learning rate")
    if owns_its_rate(optimizer):        # Automagic v3 sets its own rate: the watcher and schedulers stand down
        if adaptive_lr:
            logger.info("[adaptive_lr] ignored - the optimizer sets its own learning rate")
        if per_image_lr or warmup_look_outliers:
            logger.info("[per-image LR] per-image LR and the look warm-up are off - the optimizer sets its own rate")
        adaptive_lr = per_image_lr = warmup_look_outliers = False
        logger.info(f"[optimizer] {opt_label} owns the learning rate from here ({learning_rate:.2e} is its start); "
                    f"the LR scheduler and the adaptive watcher stand down (they set a group rate it does not read); "
                    f"its own trust-region clip bounds each step")
        if lr_scheduler and lr_scheduler != "constant":
            logger.info(f"[lr_scheduler] '{lr_scheduler}' ignored - the optimizer sets its own rate")
    if adaptive_lr:                     # the watcher owns the rate: start at the geometric midpoint of Min/Max
        learning_rate = AdaptiveLR.start_lr(adaptive_lr_min, adaptive_lr_max)
        for g in optimizer.param_groups:
            g["lr"] = learning_rate
        logger.info(f"[adaptive_lr] ENABLED - start_lr={learning_rate:.3e} min_lr={adaptive_lr_min:.3e} "
                    f"max_lr={adaptive_lr_max:.3e} (the Learning Rate box is ignored)")
    adaptive = (AdaptiveLR(adaptive_lr_min, adaptive_lr_max, clip_signal=desc.adaptive_lr_clip_signal)
                if adaptive_lr else None)
    ema = None
    if _short_ema(ema_decay):
        # Fizgig 7.0.1's short-run mode (from H3): the normal ramp never reaches 0.98 on a short run, so the window is
        # sized to the run instead - decay 1 - 4/steps (about the last quarter), fast ramp
        from training.ema import EMAWeights
        _total = max(1, int(dataset.num_items) * max(1, int(max_train_epochs)))
        ema_decay = min(0.995, max(0.5, 1.0 - 4.0 / _total))
        ema = EMAWeights(net.trainable_modules(), ema_decay, ramp=2)
        logger.info(f"[ema] SHORT-RUN mode: {_total} steps -> decay {ema_decay:.3f} (window ~ a quarter of the run), "
                    f"fast ramp - checkpoints and previews use the average")
    elif ema_decay and float(ema_decay) > 0:
        from training.ema import EMAWeights
        ema = EMAWeights(net.trainable_modules(), float(ema_decay))
        logger.info(f"[ema] ON at decay {ema_decay:g} - checkpoints and previews use the running average")

    start_epoch = global_step = 0
    if resume_state_dir:
        start_epoch, global_step, meta = _load_state(resume_state_dir, net, optimizer, device)
        if adaptive:
            adaptive.load_state_dict(meta.get("adaptive_lr_state"))
        if ema is not None and os.path.exists(os.path.join(resume_state_dir, "ema.pt")):
            ema.load_state_dict(_torch_load(os.path.join(resume_state_dir, "ema.pt"), "cpu"))
        logger.info(f"[resume] from {resume_state_dir}: continuing at epoch {start_epoch + 1}/{max_train_epochs}")
    from training.loss_watch import Watch
    watch = Watch(output_dir, dataset, driver, log=log_per_image_loss, per_image_lr=per_image_lr,
                  auto_recaption=auto_recaption, warmup_look=warmup_look_outliers,
                  resume=bool(resume_state_dir), start_epoch=start_epoch, te_path=te_path,
                  trigger_word=trigger_word, trigger_position=trigger_position, captioner=captioner,
                  slider=slider)
    scheduler = None
    if not adaptive and not owns_its_rate(optimizer):
        scheduler = _step_scheduler(optimizer, lr_scheduler, lr_warmup_steps, opt_steps_per_epoch * max_train_epochs,
                                    lr_scheduler_num_cycles, lr_scheduler_power)
        import warnings
        with warnings.catch_warnings():         # fast-forwarding a resumed schedule before the first step
            warnings.simplefilter("ignore", UserWarning)
            # the loop steps ceil(steps / accumulation) times per epoch (a partial group flushes at the epoch end)
            for _ in range(start_epoch * opt_steps_per_epoch if gradient_accumulation > 1 else global_step):
                scheduler.step()

    last_prompt = [None]
    previews_on = [True]
    ckpt_previews = None
    if preview_checkpoint and encoded is not None:
        if not desc.train_preview_checkpoint:
            logger.warning(f"[sample] {desc.display_name} has no checkpoint previews - ignoring the preview checkpoint")
        elif not os.path.isfile(preview_checkpoint):
            logger.warning(f"[sample] preview checkpoint {preview_checkpoint} not found - previews use the training "
                           f"model")
        else:
            ckpt_previews = _CheckpointPreviews(
                driver, preview_checkpoint, device, preview_checkpoint_cache, preview_int8,
                context=(context_lora_path, context_lora_strength) if context_lora_path else None,
                scratch_dir=output_dir)
            ck = desc.preview_checkpoint_sampling
            logger.info(f"[sample] previews render on {os.path.basename(preview_checkpoint)} ({ck.name}: {ck.steps} "
                        f"steps, CFG {ck.cfg:g}); the training model is parked meanwhile")

    slider_history = []                 # [{epoch, prompts, guidance}]: each live change, for the saved metadata

    def metadata(epoch):
        thumb = None if (metadata_thumbnail or "").lower() in ("off", "none") else (
            metadata_thumbnail or sample_for_epoch(output_dir, output_name, epoch) or latest_sample_image(output_dir))
        res = dataset.config["resolution"]
        md = build_metadata(arch, time.time(),
                            title=metadata_title if metadata_title else resolve_title(output_name,
                                                                                      metadata_trigger_phrase),
                            reso=(res[0], res[1]), author=metadata_author or None,
                            description=metadata_description if metadata_description else last_prompt[0],
                            license=metadata_license or None, tags=metadata_tags or None,
                            trigger_phrase=metadata_trigger_phrase or None, thumbnail=thumbnail_data_uri(thumb))
        md.update({"ss_network_module": f"tagscriber.training ({desc.key}, {network_type})",
                   "ss_network_dim": str(network_dim if network_type == "lora" else lokr_factor),
                   "ss_network_alpha": str(network_alpha if network_type == "lora" else 1.0),
                   **({"ss_lokr_factor": str(lokr_factor)} if network_type == "lokr" else {}),
                   "ss_architecture": arch, "ss_epoch": str(epoch),
                   "ss_optimizer": opt_label, "ss_learning_rate": f"{learning_rate:g}",
                   "ss_training_adapter": os.path.basename(training_adapter) if training_adapter else "none"})
        if trained_blocks:
            md["ss_train_blocks"] = ",".join(trained_blocks)
        if slider:
            # the deploy contract: strength is the dial (-1 one pole, +1 the other); tools read this (Fizgig 7.0.1)
            md["ss_slider"] = "prompt_pairs" if slider_prompts else "image_pairs"
            if slider_prompts:
                md.update({"ss_slider_prompts": json.dumps([str(x) for x in slider_prompts]),
                           "ss_slider_guidance": f"{float(slider_guidance):g}"})
                if slider_history:              # the prompts it was steered through, from which epoch (Fizgig 8.0.3)
                    md["ss_slider_prompt_history"] = json.dumps(slider_history)
            else:
                md["ss_slider_diff_weight"] = f"{float(slider_diff_weight):g}"
        md.update(driver.run_metadata())
        if context_lora_path:
            md.update({"ss_context_lora": os.path.basename(context_lora_path),
                       "ss_context_lora_strength": str(context_lora_strength)})
        return {k: v for k, v in md.items() if v is not None}

    def save_lora(path, epoch):
        if ema is not None:
            ema.swap_in()
        try:
            net.save(path, metadata(epoch))
        finally:
            if ema is not None:
                ema.swap_out()
        logger.info(f"[save] {path}")

    def previews(epoch, checkpoint=None):
        if encoded is None or not previews_on[0]:
            return
        conds, w, h, sd, prompts = encoded, sample_width, sample_height, sample_seed, sample_prompts
        ov = read_sample_override(output_dir)
        if ov and slider:       # a slider preview is its own prompt across strengths: an override never applies
            logger.info("[sample override] ignored - slider previews always show the slider's own prompt")
            ov = None
        if ov:
            logger.info(f"[sample override] active - '{ov['prompt'][:60]}' seed={ov['seed']} {ov['width']}x{ov['height']}")
            try:
                conds = _encode_override(driver, te_path, ov["prompt"], dit, device, parkable=not swapped,
                                         references=ref_imgs)
                sd, prompts = ov["seed"], [ov["prompt"]]
                if not ref_imgs:        # an edit keeps the reference's canvas: its latents are already encoded
                    w, h = ov["width"], ov["height"]
                    if lowmem and max(w, h) > 768:
                        w, h = _cap_canvas(w, h)
            except Exception:
                logger.exception("[sample override] could not encode the override prompt - using the configured ones")
                conds = encoded
        if not sd:          # seed 0 = a fresh random seed every preview round
            sd = random.randint(1, 2 ** 31 - 1)
        try:
            if ckpt_previews is not None and not slider:   # the checkpoint's own recipe (Distilled: 4 steps, no CFG)
                ck = desc.preview_checkpoint_sampling
                # the checkpoint's own recipe unless Steps / CFG were typed (an untouched box is not sent: see
                # pipeline.train_kwargs); a typed CFG above 1 brings the negative with it (Fizgig 8.6.1)
                ck_cfg = ck.cfg if asked_cfg is None else asked_cfg
                paths = ckpt_previews.render(dit, net, ema, lambda m, fl, swp: _render_previews(
                    driver, m, fl, vae, conds, sample_dir, epoch, output_name=output_name,
                    steps=asked_steps or ck.steps, cfg=ck_cfg, neg=neg if ck_cfg > 1.0 else None, width=w,
                    height=h, seed=sd, speed=ck, swapped=bool(swp), refs=ref_latents))
            else:
                paths = _render_previews(driver, dit, net, vae, conds, sample_dir, epoch, output_name=output_name,
                                         steps=sample_steps, cfg=sample_cfg_scale, neg=neg, width=w, height=h,
                                         seed=sd, ema=ema,
                                         speed=speed_desc.settings if (speed_lora and speed_desc) else None,
                                         lowmem=lowmem, swapped=bool(swapped), refs=ref_latents, slider=slider)
        except Exception:
            previews_on[0] = False
            logger.exception(f"[sample] epoch {epoch}: preview failed - previews are off for the rest of this run; "
                             f"training and saving continue")
            gc.collect()
            _empty_cache()
            return
        last_prompt[0] = prompts[-1] if prompts else None
        _record_prompts(sample_dir, output_name, epoch, prompts)
        # this epoch's checkpoint was saved before its preview existed, with the previous epoch's as its thumbnail:
        # re-embed its own. An explicit thumbnail (or "off") stays
        if checkpoint and paths and not (metadata_thumbnail or "").strip():
            refresh_checkpoint_thumbnail(checkpoint, paths[0])

    def state(epoch, required=False):
        """Save the resume state. A failed write (a full disk, or a sync / antivirus program holding the file) is
        retried once, then only warned about: the epoch's LoRA is already saved, and the state only serves a resume.
        A pause needs it (required=True), so there the second failure stops the run (Fizgig 8.5.9). Returns whether
        a state was saved; a half-written folder is removed so pruning keeps the last good one."""
        state_dir = os.path.join(output_dir, f"{output_name}-{epoch:06d}-state")
        for attempt in (1, 2):
            try:
                _save_state(output_dir, output_name, net, optimizer, epoch=epoch, global_step=global_step,
                            arch_id=arch, ema=ema,
                            extra={"adaptive_lr_state": adaptive.state_dict()} if adaptive else None)
                return True
            except (OSError, RuntimeError) as e:
                if attempt == 1:
                    logger.warning(f"[state] saving the epoch {epoch} resume state failed ({e}) - trying again")
                    time.sleep(STATE_RETRY_WAIT_S)
                    continue
                shutil.rmtree(state_dir, ignore_errors=True)
                if required:
                    raise
                logger.warning(f"[state] the epoch {epoch} resume state could not be saved ({e}). Training carries "
                               f"on and the epoch's LoRA is saved; only resuming from this epoch is lost. Check free "
                               f"disk space, and keep the output folder out of OneDrive / Dropbox sync: {output_dir}")
        return False

    def slider_live(done):
        """A prompt slider's new prompts (.slider_prompts.json), taken at an epoch boundary: the poles encoded again,
        and a new base renders a new practice bank and becomes the previews' prompt. Training goes on from where it is,
        toward the new ends (Fizgig 8.0.3)."""
        nonlocal slider_prompts, slider_guidance, slider_enc, slider_raw, encoded, sample_prompts
        new = _read_slider_live(output_dir) if slider_prompts else None
        if new is None:
            return
        # each end is the base and its own words, as the launch builds them (training/pipeline.py)
        prompts = [new["base"], f"{new['base']} {new['pos']}", f"{new['base']} {new['neg']}"]
        if prompts == [str(x) for x in slider_prompts] and new["guidance"] == float(slider_guidance):
            return
        base_changed = prompts[0] != str(slider_prompts[0])
        logger.info("[slider] new prompts from epoch %d: base '%s' | +1 -> '%s' | -1 -> '%s' (guidance %g)",
                    done + 1, *prompts, new["guidance"])
        old_raw = slider_raw
        try:
            enc = _encode_override(driver, te_path, prompts, dit, device, parkable=not swapped)
            slider_raw = enc
            if base_changed:
                render_bank()
                _empty_cache()
                if prompts[0] and sample_prompts == [str(slider_prompts[0])]:   # the previews followed the old base
                    sample_prompts = [prompts[0]]
                    if encoded is not None and not ref_imgs:
                        encoded = [enc[0]]              # text-only previews: the new base's own encoding
            slider_enc = _batched_slider_cond(enc, device)
        except Exception as exc:  # noqa: BLE001 - a failed change must not end the run
            slider_raw = old_raw
            logger.warning(f"[slider] could not take the new prompts ({type(exc).__name__}: {str(exc)[:200]}) - "
                           "training goes on with the ones it had", exc_info=True)
            return
        slider_prompts, slider_guidance = prompts, new["guidance"]
        slider_history.append({"epoch": done, "prompts": prompts, "guidance": new["guidance"]})
        try:
            with open(os.path.join(output_dir, SLIDER_LIVE_APPLIED_FILE), "w", encoding="utf-8") as f:
                json.dump({"epoch": done, "base": new["base"], "pos": new["pos"], "neg": new["neg"],
                           "guidance": new["guidance"], "practice_rerendered": base_changed, "at": time.time()}, f)
        except OSError:
            pass

    if sample_at_first and start_epoch == 0:
        previews(0)

    # ---- train ----------------------------------------------------------------------------------
    gen = torch.Generator().manual_seed(seed + start_epoch)
    pause_flag = os.path.join(output_dir, PAUSE_FILE)
    recorder = LossRecorder()
    # the Train tab's loss graph: each step's loss, numbered across a resume (the lines past the resume point are redone)
    steplog = StepLog(output_dir, output_name, start_step=global_step)
    for _w in ("sample", "save"):          # a request left from an earlier run is not this one's
        take_request(output_dir, _w)
    warmup_note_last = 0.0
    progress = tqdm(total=steps_per_epoch * max_train_epochs, initial=global_step, desc="steps", smoothing=0,
                    mininterval=1.0 if sys.stderr.isatty() else 2.0)
    dit.train()

    lr_acc = []                         # the window's per-step LR multipliers (driver.step_policy / info["lr_mult"])

    def optimizer_step():
        if max_grad_norm:
            norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            if adaptive is not None:
                adaptive.note_clip(norm, max_grad_norm)
        m = 1.0 if owns_its_rate(optimizer) else (sum(lr_acc) / len(lr_acc) if lr_acc else 1.0)
        lr_acc.clear()
        if m != 1.0:                    # the window's mean, composed into the LR for this step only (never the loss)
            held = [g["lr"] for g in optimizer.param_groups]
            for g in optimizer.param_groups:
                g["lr"] = g["lr"] * m
        optimizer.step()
        if m != 1.0:
            for g, lr in zip(optimizer.param_groups, held):
                g["lr"] = lr
        driver.after_optimizer_step()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update()
        optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, max_train_epochs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        dataset.shuffle(seed + epoch + 1)
        data_rng = random.Random(seed * 1000003 + epoch)
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        for i in range(steps_per_epoch):
            if epoch < 2 and getattr(driver, "warmup_note", False) and time.time() - warmup_note_last > 30.0:
                warmup_note_last = time.time()
                logger.info("[warm-up] Warm-up phase - the first two epochs start slowly while the GPU plans kernels "
                            "and fills its caches. Nothing is stuck; full speed arrives from epoch 3.")
            batch = dataset.get_batch(i, data_rng)
            if watch.excluded(batch):          # two failed AI recaptions and still stuck: no forward, no loss
                recorder.drop(step=i)
                global_step += 1
                progress.update(1)
                continue
            _skip, _lrm = driver.step_policy(batch, epoch + 1)
            if _skip:                          # the family retires this item for now: no forward, no loss, no record
                recorder.drop(step=i)
                global_step += 1
                progress.update(1)
                continue
            lr_acc.append(_lrm)
            latents = batch["latents"].to(device)
            if slider:
                # both poles backpropagate before the strength flips: checkpointed blocks recompute their forward in
                # backward, at whatever strength is set then (Fizgig 7.0.1)
                if slider_prompts:
                    _l, _t = _prompt_slider_step(driver, dit, net, latents, slider_enc, gen,
                                                 guidance=slider_guidance, min_t=min_timestep, max_t=max_timestep)
                    _info = {"t": _t}
                else:
                    cond = driver.batch_cond(batch, device)
                    _neg = batch.get("latents_control_0")
                    if _neg is None:
                        raise RuntimeError("[slider] this item has no pair image - an image-pair slider needs one in "
                                           "the -1 end folder for every training image, matched by file name")
                    _neg = _neg.to(device)
                    _l = 0.0
                    try:
                        for m, a, b in ((1.0, latents, _neg), (-1.0, _neg, latents)):
                            net.set_trainable_multiplier(m)
                            _pl, _info = driver.training_loss(dit, a, cond, gen, min_t=min_timestep,
                                                              max_t=max_timestep, diff_ref=b,
                                                              diff_weight=slider_diff_weight)
                            (0.5 * _pl).backward()
                            _l += 0.5 * _pl.item()
                    finally:
                        net.set_trainable_multiplier(1.0)
                loss = torch.tensor(_l)
            else:
                cond = driver.batch_cond(batch, device)
                refs = [batch[k].to(device) for k in sorted((k for k in batch if k.startswith("latents_control_")),
                                                            key=lambda k: int(k.rsplit("_", 1)[1]))]
                loss, _info = driver.training_loss(dit, latents, cond, gen, min_t=min_timestep, max_t=max_timestep,
                                                   **({"refs": refs} if refs else {}))
            if "lr_mult" in _info and lr_acc:
                lr_acc[-1] *= _info["lr_mult"]     # where the step landed (Fizgig: the H3 noise-band LR)
            if not slider:
                mult = watch.multiplier(batch)     # per-image LR (batch size 1): the raw loss is still what's recorded
                scaled = loss * mult if mult != 1.0 else loss
                (scaled / gradient_accumulation if gradient_accumulation > 1 else scaled).backward()
            pending += 1
            if pending >= gradient_accumulation or i == steps_per_epoch - 1:
                optimizer_step()
                pending = 0
            global_step += 1
            recorder.add(epoch=epoch, step=i, loss=loss.item())
            steplog.add(global_step, epoch + 1, loss.item(),
                        None if global_step % 20 else optimizer_lr(optimizer))   # the rate now and then
            if pending == 0:                   # the Train tab's Save now / Sample now, between optimizer steps
                if take_request(output_dir, "save"):
                    name = f"{output_name}-step{global_step:07d}.safetensors"
                    save_lora(os.path.join(output_dir, name), epoch + 1)
                    logger.info(f"[save now] step {global_step} saved as {name}")
                if take_request(output_dir, "sample"):
                    logger.info(f"[sample now] previews at step {global_step} (epoch {epoch + 1} in progress)")
                    _tp = time.time()
                    previews(epoch + 1)
                    progress.start_t += time.time() - _tp       # the bar's s/it is training speed, not previews
            watch.observe(epoch + 1, global_step, batch, _info.get("t", 0.5), loss.item())
            progress.set_postfix(avr_loss=f"{recorder.moving_average:.4f}", refresh=False)
            progress.update(1)
        if pending:
            optimizer_step()
        steplog.flush()

        peak = torch.cuda.max_memory_reserved() / 1024 ** 3 if torch.cuda.is_available() else 0.0
        logger.info(f"epoch {epoch + 1}/{max_train_epochs}  avr_loss={recorder.moving_average:.4f}  step={global_step}  "
                    f"{(time.time() - t0) / max(1, steps_per_epoch):.2f}s/step  "
                    f"lr={optimizer.param_groups[0]['lr']:.3e}  peak VRAM {peak:.1f} GB"
                    + (f"  {group_rates(optimizer)}" if owns_its_rate(optimizer) else ""))
        driver.after_epoch(epoch, steps_per_epoch * (max_train_epochs - epoch - 1))
        if adaptive:
            adaptive.epoch_boundary(epoch, recorder.moving_average, net.trainable_modules(), optimizer)
        # problem-image verdicts + queued caption fixes / auto-recaptions, re-encoded before the next epoch
        watch.boundary(epoch + 1, dit, device, parkable=not swapped and device.type != "cpu")

        done = epoch + 1
        cadence = bool(save_every_n_epochs) and done % save_every_n_epochs == 0 and done < max_train_epochs
        ckpt = None
        if cadence:
            ckpt = os.path.join(output_dir, f"{output_name}-{done:06d}.safetensors")
            save_lora(ckpt, done)
        state_saved = False
        if save_state and cadence:
            state_saved = state(done)
            if state_saved:
                prune_state_dirs(output_dir, output_name, keep_last_n_states)
        if sample_every_n_epochs and done % sample_every_n_epochs == 0:
            _tp = time.time()
            previews(done, ckpt)
            progress.start_t += time.time() - _tp       # the bar's s/it is training speed, not previews
        if os.path.exists(pause_flag) and done < max_train_epochs:
            if state_saved:
                logger.info(f"[pause] requested - state for epoch {done} already saved; exiting cleanly")
            else:
                logger.info(f"[pause] requested - saving state at epoch {done} and exiting cleanly")
                state(done, required=True)
            progress.close()
            steplog.flush()
            return None
        if slider_prompts and done < max_train_epochs:
            _tp = time.time()
            slider_live(done)
            progress.start_t += time.time() - _tp       # the bar's s/it is training speed

    progress.close()
    steplog.flush()
    final = os.path.join(output_dir, f"{output_name}.safetensors")
    save_lora(final, max_train_epochs)
    # the last epoch under its number too (Fizgig #176): a resumed run that extends this one ends on the same plain
    # name, and without a numbered copy that epoch would be overwritten and lost
    shutil.copyfile(final, os.path.join(output_dir, f"{output_name}-{max_train_epochs:06d}.safetensors"))
    if save_state_on_train_end:
        state(max_train_epochs)
    logger.info(f"Training complete -> {final}")
    return final


def main(argv=None):
    p = argparse.ArgumentParser(description="LoRA training for a model family")
    p.add_argument("--config", required=True, help="train_config.json written by the Train tab (training.pipeline)")
    p.add_argument("--resume", default=None, help="a <name>-NNNNNN-state folder (overrides the config's)")
    a = p.parse_args(argv)
    if (sys.platform != "win32" and not os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
            and os.environ.get("TAGSCRIBER_NO_EXPANDABLE") != "1"):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    with open(a.config, encoding="utf-8") as f:
        cfg = json.load(f)
    kw = dict(cfg["train"])
    with open(cfg["dataset_config"], encoding="utf-8") as f:
        kw["dataset_config"] = json.load(f)
    if a.resume:
        kw["resume_state_dir"] = a.resume
    if os.environ.get("TAGSCRIBER_TRAINING_DEVICE"):
        kw["device"] = os.environ["TAGSCRIBER_TRAINING_DEVICE"]
    if cfg.get("sample_prompts_file") and os.path.exists(cfg["sample_prompts_file"]):
        with open(cfg["sample_prompts_file"], encoding="utf-8") as f:
            kw["sample_prompts"] = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    result = train_family(**kw)
    if result is None:          # paused: the Train tab sees exit code 0 plus the state dir
        sys.exit(0)


if __name__ == "__main__":
    main()
