# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/families/train.py
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR (the gaps docs/FIZGIG_TRAINING_AUDIT.md section 2.5 lists, filled as the plan approved):
#   * gradient accumulation (GRADIENT_ACCUMULATION, already in Fizgig's presets) and batch size > 1 for drivers with
#     supports_batching (the loss watch needs batch 1 and stands down otherwise, as Fizgig's GUI does);
#   * Klein's grad-clip-ratio signal fed to Adaptive LR;
#   * previews never end a run: a failure (usually OOM) disables them for the rest of it (Fizgig's Krea 2 policy),
#     and each epoch checkpoint's thumbnail is refreshed with its own preview (Fizgig #122);
#   * TagScribeR's dataset layer and captioners; a CPU device for smoke tests; `--config run.json`;
#   * `driver_options`: family extensions forwarded to driver.configure() (SDXL min-SNR gamma, noise offset, LoCon).
# The pause contract, state dirs, resume, schedulers, EMA, adapters and file naming are Fizgig's.
"""The LoRA trainer for any family, driven through the family's driver.

    python -m training.train --config RUN_FOLDER/train_config.json

The loop is family-agnostic: bucketed cached data, frozen adapters (the family's training adapter, off for
previews; a context LoRA, on for previews; neither in saves), Adaptive LR or a step scheduler, gradient clipping and
accumulation, optional EMA, per-epoch checkpoints in the family's LoRA key format with SAI metadata, resumable state
dirs and the file-based pause contract. Everything model-specific comes from the driver.

Run-folder control files (written by the Train tab):
    .pause_requested         checked at each epoch boundary: save state, exit 0
    .sample_override.json    {prompt, seed, width, height} replaces the preview prompts at the next preview
"""
import argparse
import datetime
import json
import logging
import math
import os
import random
import sys
import time

import torch
from tqdm import tqdm

from training import quant
from training.adaptive_lr import AdaptiveLR
from training.lora import FamilyLoRA
from training.metadata import (build_metadata, latest_sample_image, refresh_checkpoint_thumbnail, resolve_title,
                               sample_for_epoch, thumbnail_data_uri)
from training.registry import get as get_family
from training.train_utils import LossRecorder, prune_state_dirs, validate_output_name

logger = logging.getLogger("training.train")

ADAPTER = "training_adapter"
CONTEXT = "context"
SPEED = "speed_lora"
PAUSE_FILE = ".pause_requested"
OVERRIDE_FILE = ".sample_override.json"


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


def _load_state(state_dir, net, optimizer, device):
    for need in ("lora.safetensors", "optimizer.pt", "training_state.json"):
        if not os.path.isfile(os.path.join(state_dir, need)):
            raise RuntimeError(f"[resume] {state_dir} is not a saved training state (missing {need}). Pick the "
                               f"folder named like '<lora name>-000012-state'.")
    if net.load_trainable(os.path.join(state_dir, "lora.safetensors")) == 0:
        raise RuntimeError(f"[resume] {state_dir} matched none of this LoRA's modules - different rank or "
                           f"target modules?")
    optimizer.load_state_dict(torch.load(os.path.join(state_dir, "optimizer.pt"), map_location=device))
    rng_path = os.path.join(state_dir, "rng.pt")
    if os.path.exists(rng_path):
        try:
            rng = torch.load(rng_path)
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


def _encode_override(driver, te_path, prompt, dit, device, parkable=True, references=None):
    """Encode one override prompt mid-run. The text encoder loads beside the training model when it fits; otherwise
    the model waits on CPU for the encode."""
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
                return driver.encode_text_with_references(te, [prompt], [references])
            return driver.encode_text(te, [prompt])
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
                     height, seed, ema=None, speed=None, lowmem=False, swapped=False, refs=None):
    """Previews on the RESIDENT training model: training adapter OFF (the deployment setup), the family's speed LoRA
    ON if one is loaded (it lives on CPU between previews), EMA weights swapped in. On small cards the model parks on
    CPU for the decode. File names: <name>_e<epoch:06d>_<idx:02d>_<timestamp>_<seed>.png."""
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    device = next(iter(dit.parameters())).device
    logger.info(f"[sample] epoch {epoch}: rendering {len(encoded)} preview(s)")
    _preview_vram("preview start", reset_peak=True)
    net.set_enabled(ADAPTER, False)
    if speed is not None:
        net.move_adapter(SPEED, device)
        net.set_enabled(SPEED, True)
    if ema is not None:
        ema.swap_in()
    was_training = dit.training
    dit.eval()
    if swapped:
        driver.block_swap_mode(dit, inference=True)
    paths = []
    ref_kw = {"refs": [r.to(device) for r in refs]} if refs else {}
    park = lowmem and not swapped      # a swapped model is already mostly on CPU; moving it would undo the layout
    try:
        lats = []
        for i, cond in enumerate(encoded):
            cond = {k: v.to(device) if hasattr(v, "to") else v for k, v in cond.items()}
            if speed is not None:
                lats.append(driver.generate(dit, cond, width, height, steps=steps, seed=seed + i, cfg=speed.cfg,
                                            sigmas=speed.sigmas, options=speed.options, **ref_kw))
            else:
                lats.append(driver.generate(dit, cond, width, height, steps=steps, seed=seed + i, cfg=cfg,
                                            neg_cond=neg, **ref_kw))
        if park:                            # never hold the training model and the VAE decode together
            _preview_vram("before decode")
            quant.move(dit, "cpu")
            _empty_cache()
            _preview_vram("model parked for the decode")
        for i, lat in enumerate(lats):
            p = os.path.join(out_dir, f"{output_name}_e{epoch:06d}_{i:02d}_{ts}_{seed + i}.png")
            driver.decode(vae, lat, width, height).save(p)
            paths.append(p)
    finally:
        if park:
            quant.move(dit, device)
            _preview_vram("after decode, model restored")
        if swapped:
            driver.block_swap_mode(dit, inference=False)
        if ema is not None:
            ema.swap_out()
        if speed is not None:
            net.set_enabled(SPEED, False)
            net.move_adapter(SPEED, "cpu")
        net.set_enabled(ADAPTER, True)
        dit.train(was_training)
        _empty_cache()
        _preview_vram("after preview cleanup")
    logger.info(f"[sample] epoch {epoch}: {len(paths)} preview(s) -> {out_dir}")
    return paths


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
                 trigger_word=None, trigger_position="start", captioner=None, device=None, driver_options=None):
    desc = get_family(family)
    if desc is None or not desc.training_ready:
        raise RuntimeError(f"unknown or untrainable family {family!r}")
    validate_output_name(output_name)
    driver = desc.load_driver()
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

    # ---- data ------------------------------------------------------------------------------------
    from training.dataset import TrainingDataset
    dataset = TrainingDataset(dataset_config, arch, desc.spatial_factor, desc.bucket_step)
    if dataset.batch_size > 1 and not driver.supports_batching:
        raise RuntimeError(f"{desc.display_name} trains at batch size 1 (conditioning lengths differ per image). "
                           f"Set Batch Size to 1 - Gradient Accumulation gives the same effective batch.")
    if dataset.prepare_for_training() == 0:
        raise RuntimeError("No training items - run the cache stages first (Enable Cache Preparation).")
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
    if precision == "auto" or blocks_to_swap < 0:
        req = (precision, blocks_to_swap)
        res = dataset.config["resolution"]
        mp = res[0] * res[1] / 1e6
        if device.type == "cpu":
            precision, blocks_to_swap, why = (desc.precisions[0], 0, "CPU run: first precision, no swap")
        else:
            precision, blocks_to_swap, why = quant.plan(desc, driver, precision, blocks_to_swap, megapixels=mp)
        why += f" at {mp:.2f} MP"
        logger.info(f"[precision] Auto plan: {precision}, block swap {blocks_to_swap} ({why}); asked {req}")

    # ---- previews: encode prompts once, keep the VAE ---------------------------------------------
    encoded = neg = vae = ref_imgs = ref_latents = None
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
        logger.info(f"[adapter] training adapter ON ({n} modules, strength {training_adapter_strength:g}) - frozen, "
                    f"off in previews, not saved into the LoRA")
    elif desc.training_adapter:
        logger.warning("[adapter] no training adapter for this run")
    if context_lora_path:
        n = net.add_file(context_lora_path, CONTEXT, context_lora_strength)
        logger.info(f"[context] {os.path.basename(context_lora_path)} frozen + active at {context_lora_strength:g} "
                    f"({n} modules)")
    if speed_lora and encoded is not None:
        n = net.add_file(speed_lora, SPEED, speed_desc.strength if speed_lora_strength is None else speed_lora_strength)
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
    net.add_trainable(network_dim, network_alpha, blocks=driver.trainable_blocks(), kind=network_type,
                      factor=lokr_factor)
    params = net.parameters()
    if not params:
        raise RuntimeError("the LoRA has no trainable parameters (no target modules matched)")
    logger.info((f"LoKR factor {lokr_factor}" if network_type == "lokr" else
                 f"LoRA rank {network_dim} alpha {network_alpha:g}") +
                f": {len(net.trainable_modules())} modules, {sum(p.numel() for p in params) / 1e6:.2f}M trainable "
                f"params")

    res_max = max((w * h for w, h in dataset.buckets), default=0) / 1e6
    driver.prepare_training(dit, net, precision=precision, blocks_to_swap=int(swapped or 0),
                            total_steps=dataset.num_items * max_train_epochs, megapixels=res_max,
                            batch_size=dataset.batch_size)

    from training.optimizers import create_optimizer, group_rates, owns_its_rate
    # the family may structure the optimizer's parameters (Krea 2: Automagic v3 per-family groups, Fizgig krea2/trainer.py)
    opt_params, optimizer_args, fam_counts = driver.optimizer_params(net, optimizer_type, learning_rate, optimizer_args)
    optimizer, opt_label = create_optimizer(optimizer_type, opt_params, learning_rate, optimizer_args)
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
    adaptive = AdaptiveLR(adaptive_lr_min, adaptive_lr_max) if adaptive_lr else None
    ema = None
    if ema_decay and ema_decay > 0:
        from training.ema import EMAWeights
        ema = EMAWeights(net.trainable_modules(), float(ema_decay))
        logger.info(f"[ema] ON at decay {ema_decay:g} - checkpoints and previews use the running average")

    start_epoch = global_step = 0
    if resume_state_dir:
        start_epoch, global_step, meta = _load_state(resume_state_dir, net, optimizer, device)
        if adaptive:
            adaptive.load_state_dict(meta.get("adaptive_lr_state"))
        if ema is not None and os.path.exists(os.path.join(resume_state_dir, "ema.pt")):
            ema.load_state_dict(torch.load(os.path.join(resume_state_dir, "ema.pt"), map_location="cpu"))
        logger.info(f"[resume] from {resume_state_dir}: continuing at epoch {start_epoch + 1}/{max_train_epochs}")
    from training.loss_watch import Watch
    watch = Watch(output_dir, dataset, driver, log=log_per_image_loss, per_image_lr=per_image_lr,
                  auto_recaption=auto_recaption, warmup_look=warmup_look_outliers,
                  resume=bool(resume_state_dir), start_epoch=start_epoch, te_path=te_path,
                  trigger_word=trigger_word, trigger_position=trigger_position, captioner=captioner)
    scheduler = None
    if not adaptive and not owns_its_rate(optimizer):
        scheduler = _step_scheduler(optimizer, lr_scheduler, lr_warmup_steps, opt_steps_per_epoch * max_train_epochs,
                                    lr_scheduler_num_cycles, lr_scheduler_power)
        import warnings
        with warnings.catch_warnings():         # fast-forwarding a resumed schedule before the first step
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(global_step // gradient_accumulation):
                scheduler.step()

    last_prompt = [None]
    previews_on = [True]

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
        if context_lora_path:
            md.update({"ss_context_lora": os.path.basename(context_lora_path),
                       "ss_context_lora_strength": str(context_lora_strength)})
        md.update(driver.extra_metadata())
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
            paths = _render_previews(driver, dit, net, vae, conds, sample_dir, epoch, output_name=output_name,
                                     steps=sample_steps, cfg=sample_cfg_scale, neg=neg, width=w, height=h,
                                     seed=sd, ema=ema,
                                     speed=speed_desc.settings if (speed_lora and speed_desc) else None,
                                     lowmem=lowmem, swapped=bool(swapped), refs=ref_latents)
        except Exception:
            previews_on[0] = False
            logger.exception(f"[sample] epoch {epoch}: preview failed - previews are off for the rest of this run; "
                             f"training and saving continue")
            _empty_cache()
            return
        last_prompt[0] = prompts[-1] if prompts else None
        if checkpoint and paths:
            refresh_checkpoint_thumbnail(checkpoint, paths[0])

    def state(epoch):
        _save_state(output_dir, output_name, net, optimizer, epoch=epoch, global_step=global_step, arch_id=arch,
                    ema=ema, extra={"adaptive_lr_state": adaptive.state_dict()} if adaptive else None)

    if sample_at_first and start_epoch == 0:
        previews(0)

    # ---- train ----------------------------------------------------------------------------------
    gen = torch.Generator().manual_seed(seed + start_epoch)
    pause_flag = os.path.join(output_dir, PAUSE_FILE)
    recorder = LossRecorder()
    warmup_note_last = 0.0
    progress = tqdm(total=steps_per_epoch * max_train_epochs, initial=global_step, desc="steps", smoothing=0,
                    mininterval=1.0 if sys.stderr.isatty() else 2.0)
    dit.train()

    lr_scales = []                      # this window's per-step LR multipliers (driver info["lr_scale"])
    self_rated = owns_its_rate(optimizer)
    lr_scale_noted = []

    def optimizer_step():
        if max_grad_norm:
            norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            if adaptive is not None:
                adaptive.note_clip(norm, max_grad_norm)
        # Fizgig minimax/trainer.py _boundary_step: the window's mean band multiplier scales the optimizer's LR for
        # this one step (never the loss), and is not applied when the optimizer sets its own rate (Automagic v3)
        bm = (sum(lr_scales) / len(lr_scales)) if lr_scales else 1.0
        lr_scales.clear()
        if bm != 1.0 and not self_rated:
            base_lrs = [g["lr"] for g in optimizer.param_groups]
            for g in optimizer.param_groups:
                g["lr"] = g["lr"] * bm
            optimizer.step()
            for g, lr0 in zip(optimizer.param_groups, base_lrs):
                g["lr"] = lr0
        else:
            optimizer.step()
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
            latents = batch["latents"].to(device)
            cond = {k[len("cond__"):]: v.to(device) for k, v in batch.items() if k.startswith("cond__")}
            refs = [batch[k].to(device) for k in sorted((k for k in batch if k.startswith("latents_control_")),
                                                        key=lambda k: int(k.rsplit("_", 1)[1]))]
            loss, _info = driver.training_loss(dit, latents, cond, gen, min_t=min_timestep, max_t=max_timestep,
                                               **({"refs": refs} if refs else {}))
            mult = watch.multiplier(batch)     # per-image LR (batch size 1): the raw loss is still what's recorded
            scaled = loss * mult if mult != 1.0 else loss
            (scaled / gradient_accumulation if gradient_accumulation > 1 else scaled).backward()
            if "lr_scale" in _info:
                if self_rated and not lr_scale_noted:
                    lr_scale_noted.append(True)
                    logger.info("[lr] the per-step LR multiplier is not applied - the optimizer sets its own rate")
                lr_scales.append(float(_info["lr_scale"]))
            pending += 1
            if pending >= gradient_accumulation or i == steps_per_epoch - 1:
                optimizer_step()
                pending = 0
            global_step += 1
            recorder.add(epoch=epoch, step=i, loss=loss.item())
            watch.observe(epoch + 1, global_step, batch, _info.get("t", 0.5), loss.item())
            progress.set_postfix(avr_loss=f"{recorder.moving_average:.4f}", refresh=False)
            progress.update(1)
        if pending:
            optimizer_step()

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
            state(done)
            prune_state_dirs(output_dir, output_name, keep_last_n_states)
            state_saved = True
        if sample_every_n_epochs and done % sample_every_n_epochs == 0:
            previews(done, ckpt)
        if os.path.exists(pause_flag) and done < max_train_epochs:
            if state_saved:
                logger.info(f"[pause] requested - state for epoch {done} already saved; exiting cleanly")
            else:
                logger.info(f"[pause] requested - saving state at epoch {done} and exiting cleanly")
                state(done)
            progress.close()
            return None

    progress.close()
    final = os.path.join(output_dir, f"{output_name}.safetensors")
    save_lora(final, max_train_epochs)
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
