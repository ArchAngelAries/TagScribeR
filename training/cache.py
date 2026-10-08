# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/families/cache.py (with the batching and
# stale-cache cleanup of src/fizgig/scripts/cache_latents.py / cache_text.py).
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: reads the run's dataset.json (training/dataset.py) instead of a TOML; `[cache] stage i/N`
# progress lines for the Train tab; the optional caption-shuffle variants and caption-dropout empty caption.
"""Caching for any family: VAE latents or text conditioning, through the family driver.

    python -m training.cache --family qwen_image21 --stage latents --dataset RUN/dataset.json --model VAE
    python -m training.cache --family qwen_image21 --stage text    --dataset RUN/dataset.json --model TE

Latents are stored as `latent_{h}x{w}` (+ `latent_control_{i}_{h}x{w}` for edit pairs, at the target's bucket);
conditioning as `cond__<driver key>` with the caption in the metadata, so --skip_existing re-encodes exactly the
captions that changed. Cache files whose image is gone are removed afterwards (unless --keep_cache).
"""
import argparse
import logging
import os
import sys

import torch

logger = logging.getLogger("training.cache")
FORMAT_VERSION = "1.0.0"
# Bumped when the way latents are ENCODED changes, so --skip_existing re-encodes caches written the old way.
#   2: the VAEs' wide single-head attention is computed correctly (training/modules/wide_attention.py). Caches
#      without this mark may have been encoded through the broken built-in attention on AMD ROCm.
LATENT_REV = "2"


def _clean(t, what, key):
    t = t.detach().cpu().contiguous()
    if t.is_floating_point() and torch.isnan(t).any():
        logger.warning(f"NaN in {what} for {key} - replaced with 0")
        t[torch.isnan(t)] = 0
    return t


def _dtype_str(dt) -> str:
    return str(dt).replace("torch.", "")


def save_latents(desc, item, latent, controls=()):
    from safetensors.torch import save_file
    _, h, w = latent.shape
    os.makedirs(os.path.dirname(item.latent_cache_path), exist_ok=True)
    sd = {f"latent_{h}x{w}": _clean(latent, "latent", item.item_key)}
    for i, c in enumerate(controls):
        sd[f"latent_control_{i}_{c.shape[-2]}x{c.shape[-1]}"] = _clean(c, "control latent", item.item_key)
    save_file(sd, item.latent_cache_path, metadata={
        "architecture": desc.arch_id, "width": str(item.original_size[0]), "height": str(item.original_size[1]),
        "dtype": _dtype_str(latent.dtype), "format_version": FORMAT_VERSION, "latent_rev": LATENT_REV})


def latent_rev(path) -> str:
    """The encoder revision a latent cache was written with ("" for caches from before the mark). Header only."""
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            return (f.metadata() or {}).get("latent_rev", "")
    except Exception:
        return ""


def _ref_sizes(item):
    """'WxH,WxH' of an item's before-images as cached (the text cache is only valid at these)."""
    return ",".join(f"{c.shape[1]}x{c.shape[0]}" for c in (item.control_content or []))


def save_cond(desc, item, cond, refs="", path=None, caption=None):
    """Write one conditioning file (the item's own text cache unless `path` is given, e.g. a shuffle variant)."""
    from safetensors.torch import save_file
    path = path or item.text_encoder_output_cache_path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    md = {"architecture": desc.arch_id, "caption1": item.caption if caption is None else caption,
          "format_version": FORMAT_VERSION}
    if desc.text_cache_rev:
        md["text_rev"] = desc.text_cache_rev
    if refs:
        md["reference_sizes"] = refs
    tmp = path + ".tmp"
    save_file({f"cond__{k}": _clean(v, k, item.item_key) for k, v in cond.items()}, tmp, metadata=md)
    os.replace(tmp, path)          # the trainer may read this file mid-run (loss-watch caption fixes)


def cached_matches(path, caption, refs="", rev=""):
    """An existing text cache that still fits: same caption, same before-image sizes and the family's current text
    encoding revision (FamilyDescription.text_cache_rev)."""
    from safetensors import safe_open
    try:
        with safe_open(path, framework="pt") as f:
            md = f.metadata() or {}
    except Exception:
        return False
    return (md.get("caption1") == caption and md.get("reference_sizes", "") == refs
            and md.get("text_rev", "") == rev)


def _has_controls(path):
    from safetensors import safe_open
    try:
        with safe_open(path, framework="pt") as f:
            return any(k.startswith("latent_control_") for k in f.keys())
    except Exception:
        return False


def encode_captions(driver, te, desc, dataset, items, *, skip_existing=False, references=False, on_progress=None):
    """Encode each item's caption (and, with the shuffle extension on, its variants) into the text cache. With
    `references`, items carry control_content and the encoder sees the before-images. Returns files written."""
    from training.dataset import shuffle_variants, text_cache_path
    cfg = dataset.config
    jobs = []                      # (item, caption, path)
    for it in items:
        jobs.append((it, it.caption, it.text_encoder_output_cache_path))
        if dataset.variants:
            for i, v in enumerate(shuffle_variants(it.caption, dataset.variants, int(cfg["caption_keep_tokens"]),
                                                   cfg["caption_separator"])):
                jobs.append((it, v, text_cache_path(it.cache_directory, it.item_key, dataset.arch, i)))
            # variants beyond what this caption yields (too few tags, or a smaller count) must not linger
            n = len(shuffle_variants(it.caption, dataset.variants, int(cfg["caption_keep_tokens"]),
                                     cfg["caption_separator"]))
            for i in range(n, 64):
                stale = text_cache_path(it.cache_directory, it.item_key, dataset.arch, i)
                if not os.path.exists(stale):
                    break
                os.remove(stale)
    todo = []
    for it, cap, path in jobs:
        refs = _ref_sizes(it) if references else ""
        if skip_existing and os.path.exists(path) and cached_matches(path, cap, refs, desc.text_cache_rev):
            continue
        todo.append((it, cap, path, refs))
    written = 0
    if references:
        for n, (it, cap, path, refs) in enumerate(todo, 1):
            c = driver.encode_text_with_references(te, [cap], [it.control_content])[0]
            save_cond(desc, it, c, refs, path=path, caption=cap)
            written += 1
            if on_progress:
                on_progress(n, len(todo))
        return written
    for i in range(0, len(todo), 8):
        chunk = todo[i:i + 8]
        for (it, cap, path, refs), c in zip(chunk, driver.encode_text(te, [cap for _, cap, _, _ in chunk])):
            save_cond(desc, it, c, path=path, caption=cap)
            written += 1
        if on_progress:
            on_progress(min(i + 8, len(todo)), len(todo))
    return written


def _remove_stale(dataset, items, keep_cache):
    """Delete this family's cache files whose image is no longer in the dataset (or whose size changed)."""
    import glob

    from training.dataset import EMPTY_KEY
    keep = {os.path.normcase(os.path.abspath(p)) for it in items
            for p in (it.latent_cache_path, it.text_encoder_output_cache_path)}
    keys = {(os.path.normcase(os.path.abspath(it.cache_directory)), it.item_key) for it in items}
    arch = dataset.arch
    for d in dataset.config["datasets"]:
        cdir = d["cache_directory"]
        ncdir = os.path.normcase(os.path.abspath(cdir))
        for f in glob.glob(os.path.join(glob.escape(cdir), f"*_{arch}*.safetensors")):
            name = os.path.basename(f)
            if name.startswith(f"{EMPTY_KEY}_{arch}_te"):
                continue
            nf = os.path.normcase(os.path.abspath(f))
            if nf in keep:
                continue
            if "_te_s" in name:                 # a shuffle variant: kept while its item exists
                stem = name[: name.rindex(f"_{arch}_te_s")]
                if (ncdir, stem) in keys:
                    continue
            if keep_cache:
                logger.info(f"Keeping stale cache: {name}")
                continue
            try:
                os.remove(f)
                logger.info(f"Removed stale cache: {name}")
            except OSError as e:
                logger.warning(f"could not remove stale cache {name}: {e}")


def run_latents(desc, driver, dataset, model_path, device, *, skip_existing=False, keep_cache=False):
    items = dataset.source_items()
    if not items:
        raise SystemExit("No captioned images found in the dataset folder(s).")
    vae = driver.load_vae(model_path, device)
    from training.dataset import latent_cache_matches_reso
    todo = []
    for it in items:
        if skip_existing and os.path.exists(it.latent_cache_path):
            ok = latent_cache_matches_reso(it.latent_cache_path, it.bucket_size, desc.spatial_factor)
            if (ok and _has_controls(it.latent_cache_path) == bool(it.control_paths)
                    and latent_rev(it.latent_cache_path) == LATENT_REV):
                continue
        todo.append(it)
    logger.info(f"[cache] latents: {len(items)} image(s), {len(todo)} to encode")
    done = 0
    print(f"[cache] latents 0/{len(todo)}", flush=True)
    for batch in dataset.latent_batches(todo, batch_size=1):
        for it, z in zip(batch, driver.encode_images(vae, [it.content for it in batch])):
            ctrl = driver.encode_images(vae, it.control_content) if it.control_content else []
            save_latents(desc, it, z, ctrl)
        done += len(batch)
        print(f"[cache] latents {done}/{len(todo)}", flush=True)
    del vae
    _remove_stale(dataset, items, keep_cache)
    return done


def run_text(desc, driver, dataset, model_path, device, *, skip_existing=False, keep_cache=False, slider=False):
    """slider: the control folder is a slider's other pole - its captions are encoded plainly, not as edit pairs."""
    from training.dataset import empty_cache_path
    items = dataset.source_items()
    if not items:
        raise SystemExit("No captioned images found in the dataset folder(s).")
    pairs = any(it.control_paths for it in items) and not slider
    te = driver.load_reference_text_encoder(model_path, device) if pairs else driver.load_text_encoder(model_path,
                                                                                                       device)
    progress = lambda n, total: print(f"[cache] text {n}/{total}", flush=True)  # noqa: E731
    try:
        if pairs:
            for it in items:                 # the encoder must see each before-image at its latent's size
                dataset.load_pixels(it)
        written = encode_captions(driver, te, desc, dataset, items, skip_existing=skip_existing,
                                  references=pairs, on_progress=progress)
        if dataset.caption_dropout and not pairs:
            from types import SimpleNamespace
            for cdir in {it.cache_directory for it in items}:
                path = empty_cache_path(cdir, desc.arch_id)
                if not (skip_existing and os.path.exists(path) and cached_matches(path, "", rev=desc.text_cache_rev)):
                    stub = SimpleNamespace(item_key="(empty caption)", caption="")
                    save_cond(desc, stub, driver.encode_text(te, [""])[0], path=path)
        elif dataset.caption_dropout:
            logger.warning("[cache] caption dropout is off for edit pairs (the empty caption has no before-image)")
    finally:
        driver.unload_text_encoder(te)
    logger.info(f"[cache] text: {written} caption(s) encoded")
    _remove_stale(dataset, items, keep_cache)
    return written


def main(argv=None):
    p = argparse.ArgumentParser(description="Cache latents or text conditioning for a model family")
    p.add_argument("--family", required=True, help="family key, e.g. qwen_image21")
    p.add_argument("--stage", required=True, choices=["latents", "text"])
    p.add_argument("--dataset", required=True, help="the run's dataset.json")
    p.add_argument("--model", required=True, help="the VAE (latents) or text encoder (text) file")
    p.add_argument("--device", default=os.environ.get("TAGSCRIBER_TRAINING_DEVICE") or None)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--keep_cache", action="store_true")
    p.add_argument("--slider", action="store_true",
                   help="the control folder holds a slider's other pole: cache its latents, encode captions plainly "
                        "(not as edit pairs)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    import json

    from training import quant
    from training.dataset import TrainingDataset
    from training.registry import get
    quant.apply_vram_cap()
    desc = get(args.family)
    if desc is None or not desc.training_ready:
        raise SystemExit(f"unknown or untrainable family {args.family!r}")
    driver = desc.load_driver()
    with open(args.dataset, encoding="utf-8") as f:
        cfg = json.load(f)
    dataset = TrainingDataset(cfg, desc.arch_id, desc.spatial_factor, desc.bucket_step)
    if dataset.has_control and args.slider and not desc.slider_training:
        raise SystemExit(f"{desc.display_name} has no slider training")
    if dataset.has_control and not args.slider and not driver.supports_references:
        raise SystemExit(f"{desc.display_name} has no edit training: remove the originals folder from the run")
    if dataset.has_control:           # edit training and sliders are one partner per image
        many = [it.item_key for it in dataset.source_items() if len(it.control_paths) > 1]
        if many:
            raise SystemExit(f"{len(many)} image(s) match more than one partner image (e.g. {', '.join(many[:3])}): "
                             f"keep one per image, named the same")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with torch.no_grad():
        if args.stage == "latents":
            run_latents(desc, driver, dataset, args.model, device, skip_existing=args.skip_existing,
                        keep_cache=args.keep_cache)
        else:
            run_text(desc, driver, dataset, args.model, device, skip_existing=args.skip_existing,
                     keep_cache=args.keep_cache, slider=args.slider)


if __name__ == "__main__":
    main()
