# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/dataset/image_dataset.py (decode_caption,
# glob_images, BucketSelector, resize_image_to_bucket, ItemInfo, cache naming, latent_cache_matches_reso,
# prepare_for_training, BucketBatchManager) and src/fizgig/dataset/config.py (the dataset settings).
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: image-only (no video / audio branches); the dataset config is a JSON dict frozen into the
# run folder instead of a TOML file; one dataset object serves both caching and training; caption shuffle variants
# and caption dropout are an OPTIONAL extension (off by default, and with them off the data path is Fizgig's).
"""Training datasets: images + captions -> buckets -> cached latents / conditioning -> batches.

Dataset config (a dict; the run builder freezes it into the run folder as dataset.json):

    {"datasets": [{"image_directory": "...", "cache_directory": "...", "num_repeats": 1,
                   "control_directory": "..."}],         # control_directory = edit pairs (optional)
     "resolution": [704, 704],                           # target area; buckets keep the aspect ratio
     "caption_extension": ".txt", "batch_size": 1, "enable_bucket": true, "bucket_no_upscale": true,
     "caption_shuffle_variants": 0, "caption_keep_tokens": 0, "caption_dropout": 0.0, "caption_separator": ","}

Cache files (per dataset folder, Fizgig's naming):
    latents  <stem>_<WWWW>x<HHHH>_<arch>.safetensors    W x H = the ORIGINAL image size; tensors latent_{h}x{w}
    text     <stem>_<arch>_te.safetensors                tensors cond__<driver key>, metadata caption1
    (extension) <stem>_<arch>_te_s<i>.safetensors        shuffled-caption variant i
    (extension) _empty_<arch>_te.safetensors             the empty caption, for caption dropout

Training reads caches, never images: the item list is built by globbing the cache folder and cross-checked against
the images actually present, so a deleted image's cache is never trained on.
"""
from __future__ import annotations

import glob
import logging
import math
import os
import random
import zlib
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif")
EMPTY_KEY = "_empty"

DEFAULT_CONFIG = {
    "datasets": [], "resolution": [1024, 1024], "caption_extension": ".txt", "batch_size": 1,
    "enable_bucket": True, "bucket_no_upscale": True,
    "caption_shuffle_variants": 0, "caption_keep_tokens": 0, "caption_dropout": 0.0, "caption_separator": ",",
}


def resolution_for_megapixels(mp: float) -> int:
    """Fizgig's Target Megapixels -> square resolution side: floor(sqrt(MP * 1e6)) // 16 * 16 (0.5 MP -> 704)."""
    return int(math.floor(math.sqrt(float(mp) * 1e6))) // 16 * 16


def normalise_config(cfg: dict) -> dict:
    out = {**DEFAULT_CONFIG, **(cfg or {})}
    res = out["resolution"]
    out["resolution"] = [int(res), int(res)] if isinstance(res, (int, float)) else [int(res[0]), int(res[1])]
    out["datasets"] = [dict(d) for d in out.get("datasets") or []]
    for d in out["datasets"]:
        d.setdefault("num_repeats", 1)
    return out


# ---- captions ----------------------------------------------------------------------------------------------
def decode_caption(raw: bytes, caption_path: str) -> str:
    """A caption file's text, tolerating encodings other than UTF-8: UTF-16 (BOM), UTF-8 (BOM tolerated), cp1252,
    then latin-1 (which cannot fail). Anything past UTF-8 is logged with the path."""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            text = raw.decode("utf-16")
            logger.warning("Caption %s is UTF-16 - decoded, but re-save it as UTF-8.", caption_path)
            return text
        except UnicodeDecodeError:
            pass
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    for encoding in ("cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        logger.warning("Caption %s is not valid UTF-8 - decoded as %s. Re-save it as UTF-8 to silence this "
                       "warning.", caption_path, encoding)
        return text
    return raw.decode("utf-8", errors="replace")


def read_caption(image_path: str, caption_extension: str) -> str:
    path = os.path.splitext(image_path)[0] + caption_extension
    with open(path, "rb") as f:
        return decode_caption(f.read(), path).strip()


def shuffle_variants(caption: str, n: int, keep_tokens: int = 0, separator: str = ",") -> list:
    """Up to `n` distinct tag-order variants of a caption (the optional caption-shuffle extension): the first
    `keep_tokens` tags stay in place, the rest are shuffled. Deterministic per caption (seeded by its CRC32), so a
    re-cache produces the same variants. Captions with too few tags yield fewer (or no) variants."""
    if n <= 0:
        return []
    parts = [p.strip() for p in caption.split(separator)]
    parts = [p for p in parts if p]
    head, tail = parts[:max(0, keep_tokens)], parts[max(0, keep_tokens):]
    if len(tail) < 2:
        return []
    join = (separator.strip() + " ") if separator.strip() else separator
    seen, out = {join.join(parts)}, []
    rng = random.Random(zlib.crc32(caption.encode("utf-8")))
    for _ in range(n * 4):
        t = list(tail)
        rng.shuffle(t)
        v = join.join(head + t)
        if v not in seen:
            seen.add(v)
            out.append(v)
            if len(out) >= n:
                break
    return out


# ---- files -------------------------------------------------------------------------------------------------
def glob_images(directory: str, caption_extension: Optional[str] = None) -> list[str]:
    """Image files in `directory` (not recursive), sorted; with caption_extension, only those with a caption - and
    the ones left out are named in a warning (files renamed after captioning are the usual cause)."""
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    paths = [os.path.join(directory, n) for n in names if os.path.splitext(n)[1].lower() in IMAGE_EXTENSIONS
             and os.path.isfile(os.path.join(directory, n))]
    if caption_extension is not None:
        have = {os.path.splitext(n)[0] for n in names if n.lower().endswith(caption_extension.lower())}
        skipped = sorted(os.path.basename(p) for p in paths if os.path.splitext(os.path.basename(p))[0] not in have)
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in have]
        if skipped:
            shown = ", ".join(skipped[:8]) + (f", ... {len(skipped) - 8} more" if len(skipped) > 8 else "")
            logger.warning(f"[dataset] {len(skipped)} image(s) in {directory} have no {caption_extension} caption and "
                           f"are LEFT OUT of training: {shown}")
    paths.sort()
    return paths


def latent_cache_path(cache_dir: str, item_key: str, original_size: Tuple[int, int], arch: str) -> str:
    w, h = original_size
    return os.path.join(cache_dir, f"{item_key}_{w:04d}x{h:04d}_{arch}.safetensors")


def text_cache_path(cache_dir: str, item_key: str, arch: str, variant: Optional[int] = None) -> str:
    suffix = "" if variant is None else f"_s{variant}"
    return os.path.join(cache_dir, f"{item_key}_{arch}_te{suffix}.safetensors")


def empty_cache_path(cache_dir: str, arch: str) -> str:
    return os.path.join(cache_dir, f"{EMPTY_KEY}_{arch}_te.safetensors")


# ---- buckets -----------------------------------------------------------------------------------------------
def divisible_by(num: int, divisor: int) -> int:
    return num - num % divisor


class BucketSelector:
    """Aspect-ratio bucketing at a resolution step (the family's bucket_step)."""

    def __init__(self, resolution: Tuple[int, int], enable_bucket: bool = True, no_upscale: bool = False,
                 reso_steps: int = 64):
        self.resolution = tuple(resolution)
        self.bucket_area = resolution[0] * resolution[1]
        self.reso_steps = reso_steps
        if not enable_bucket:
            self.bucket_resolutions = [tuple(resolution)]
            self.no_upscale = False
        else:
            self.no_upscale = no_upscale
            sqrt_size = int(math.sqrt(self.bucket_area))
            min_size = divisible_by(sqrt_size // 2, self.reso_steps)
            resolutions = set()
            for w in range(min_size, sqrt_size + self.reso_steps, self.reso_steps):
                h = divisible_by(self.bucket_area // w, self.reso_steps)
                resolutions.add((w, h))
                resolutions.add((h, w))
            self.bucket_resolutions = sorted(resolutions)
        self.aspect_ratios = np.array([w / h for w, h in self.bucket_resolutions])

    def get_bucket_resolution(self, image_size: Tuple[int, int]) -> Tuple[int, int]:
        """The bucket (width, height) for an image of (width, height): the closest aspect ratio, or with no_upscale
        a smaller-than-target image keeps its own size floored to the step."""
        area = image_size[0] * image_size[1]
        if self.no_upscale and area <= self.bucket_area:
            return divisible_by(image_size[0], self.reso_steps), divisible_by(image_size[1], self.reso_steps)
        aspect_ratio = image_size[0] / image_size[1]
        return tuple(self.bucket_resolutions[int(np.abs(self.aspect_ratios - aspect_ratio).argmin())])


def resize_image_to_bucket(image, bucket_reso: Tuple[int, int]) -> np.ndarray:
    """Scale to cover the bucket (Lanczos up, area down) and centre-crop. (width, height) in, (H, W, C) uint8 out."""
    from PIL import Image
    is_pil = isinstance(image, Image.Image)
    image_width, image_height = image.size if is_pil else (image.shape[1], image.shape[0])
    bucket_width, bucket_height = bucket_reso
    if (bucket_width, bucket_height) == (image_width, image_height):
        return np.array(image) if is_pil else image
    scale = max(bucket_width / image_width, bucket_height / image_height)
    new_w, new_h = int(image_width * scale + 0.5), int(image_height * scale + 0.5)
    if scale > 1:
        img = image if is_pil else Image.fromarray(image)
        arr = np.array(img.resize((new_w, new_h), Image.LANCZOS))
    else:
        import cv2
        arr = np.array(image) if is_pil else image
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    left, top = (new_w - bucket_width) // 2, (new_h - bucket_height) // 2
    return arr[top: top + bucket_height, left: left + bucket_width]


class ItemInfo:
    """One training image: its key (file stem), caption, sizes, cache paths and (while caching) its pixels."""

    def __init__(self, item_key: str, caption: str, original_size: Tuple[int, int],
                 bucket_size: Optional[Tuple[int, int]] = None, latent_cache_path: Optional[str] = None):
        self.item_key = item_key
        self.caption = caption
        self.original_size = original_size
        self.bucket_size = bucket_size
        self.latent_cache_path = latent_cache_path
        self.text_encoder_output_cache_path: Optional[str] = None
        self.image_path: Optional[str] = None
        self.control_paths: list = []
        self.content: Optional[np.ndarray] = None
        self.control_content: Optional[list] = None
        self.cache_directory: Optional[str] = None

    def __repr__(self):
        return f"ItemInfo({self.item_key!r}, {self.original_size} -> {self.bucket_size})"


def latent_cache_matches_reso(cache_file: str, bucket_reso: Tuple[int, int], spatial_factor: int) -> Optional[bool]:
    """Does the cached latent match `bucket_reso` (pixel w, h)? The cache FILENAME encodes the original size, so a
    cache written at another Target Megapixels has the same name; training on it would silently run at the old
    resolution. Header-only read; None if unreadable."""
    try:
        from safetensors import safe_open
        with safe_open(cache_file, framework="pt") as f:
            keys = list(f.keys())
    except Exception:
        return None
    expected = {int(bucket_reso[0]) // spatial_factor, int(bucket_reso[1]) // spatial_factor}
    for k in keys:
        if k.startswith("latent_") and not k.startswith("latent_control_"):
            try:
                a, b = k[len("latent_"):].split("x")[-2:]
                return {int(a), int(b)} == expected
            except Exception:
                return None
    return None


def partners(stem: str, names) -> list:
    """The names in `names` that pair with an image whose file stem is `stem` (an edit's before-image, a slider's
    other end): stem.<ext> or stem_<anything>.<ext>, ignoring case (Fizgig 7.0.1 families/launch.py pairs)."""
    st = stem.casefold()
    return [n for n in names if os.path.splitext(n)[1].lower() in IMAGE_EXTENSIONS
            and (n.casefold().startswith(st + ".") or n.casefold().startswith(st + "_"))]


def _control_for(control_dir: str, stem: str) -> list:
    """Edit pairs and image-pair sliders: the partner image(s) named like the image."""
    if not control_dir:
        return []
    try:
        names = sorted(os.listdir(control_dir))
    except OSError:
        return []
    return [os.path.join(control_dir, n) for n in partners(stem, names)]


# ---- the dataset ---------------------------------------------------------------------------------------------
class TrainingDataset:
    """All datasets of a run (one per image folder), for caching (images) and for training (caches)."""

    def __init__(self, config: dict, arch: str, spatial_factor: int, bucket_step: int):
        self.config = normalise_config(config)
        self.arch, self.spatial_factor, self.bucket_step = arch, spatial_factor, bucket_step
        c = self.config
        self.selector = BucketSelector(tuple(c["resolution"]), c["enable_bucket"], c["bucket_no_upscale"],
                                       bucket_step)
        self.batch_size = max(1, int(c["batch_size"]))
        self.buckets: dict = {}            # training: (w, h) -> [ItemInfo] (with repeats)
        self.batches: list = []            # training: [(bucket, start)]
        self.has_control = any(d.get("control_directory") for d in c["datasets"])

    @property
    def variants(self) -> int:
        return max(0, int(self.config.get("caption_shuffle_variants") or 0))

    @property
    def caption_dropout(self) -> float:
        return max(0.0, min(1.0, float(self.config.get("caption_dropout") or 0.0)))

    # ---- caching side ---------------------------------------------------------------------------------
    def source_items(self) -> list:
        """Every captioned image as an ItemInfo (sizes from the header, bucket chosen, cache paths set)."""
        from PIL import Image
        items, seen = [], {}
        ext = self.config["caption_extension"]
        for d in self.config["datasets"]:
            img_dir, cache_dir = d["image_directory"], d["cache_directory"]
            for path in glob_images(img_dir, ext):
                stem = os.path.splitext(os.path.basename(path))[0]
                if (cache_dir, stem) in seen:
                    logger.warning(f"[dataset] {os.path.basename(path)} and {os.path.basename(seen[(cache_dir, stem)])} "
                                   f"share a name - only the first is used. Rename one of them.")
                    continue
                seen[(cache_dir, stem)] = path
                try:
                    with Image.open(path) as im:
                        size = im.size
                    caption = read_caption(path, ext)
                except Exception as e:
                    logger.warning(f"[dataset] skipping {path}: {e}")
                    continue
                it = ItemInfo(stem, caption, size, self.selector.get_bucket_resolution(size),
                              latent_cache_path(cache_dir, stem, size, self.arch))
                it.text_encoder_output_cache_path = text_cache_path(cache_dir, stem, self.arch)
                it.image_path, it.cache_directory = path, cache_dir
                it.control_paths = _control_for(d.get("control_directory") or "", stem)
                items.append(it)
        return items

    @staticmethod
    def load_pixels(item: ItemInfo) -> None:
        """Fill item.content (and control_content) at the item's bucket size: RGB / RGBA uint8 (H, W, C)."""
        from PIL import Image
        img = Image.open(item.image_path)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        item.content = resize_image_to_bucket(img, item.bucket_size)
        if item.control_paths:
            ctrl = []
            for cp in item.control_paths:
                c = Image.open(cp)
                if c.mode not in ("RGB", "RGBA"):
                    c = c.convert("RGB")
                ctrl.append(resize_image_to_bucket(c, item.bucket_size))
            item.control_content = ctrl

    def latent_batches(self, items: list, batch_size: int = 1):
        """Same-bucket groups (the VAE encodes same-size images together), pixels loaded."""
        by_bucket: dict = {}
        for it in items:
            by_bucket.setdefault(it.bucket_size, []).append(it)
        for reso in sorted(by_bucket):
            group = by_bucket[reso]
            for i in range(0, len(group), max(1, batch_size)):
                chunk = group[i:i + batch_size]
                for it in chunk:
                    self.load_pixels(it)
                yield chunk
                for it in chunk:
                    it.content = it.control_content = None

    # ---- training side --------------------------------------------------------------------------------
    def prepare_for_training(self) -> int:
        """Build the buckets from the cache files on disk. Returns the number of training items (with repeats)."""
        self.buckets = {}
        skipped_stale = skipped_reso = missing_te = 0
        for d in self.config["datasets"]:
            cache_dir, img_dir = d["cache_directory"], d["image_directory"]
            valid = None
            if img_dir and os.path.isdir(img_dir):
                valid = {os.path.splitext(f)[0] for f in os.listdir(img_dir)
                         if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS}
            for cache_file in glob.glob(os.path.join(glob.escape(cache_dir), f"*_{self.arch}.safetensors")):
                tokens = os.path.basename(cache_file).split("_")
                try:
                    w, h = map(int, tokens[-2].split("x"))
                except (ValueError, IndexError):
                    continue
                item_key = "_".join(tokens[:-2])
                if valid is not None and item_key not in valid:
                    skipped_stale += 1
                    continue
                te = text_cache_path(cache_dir, item_key, self.arch)
                if not os.path.exists(te):
                    missing_te += 1
                    continue
                bucket = self.selector.get_bucket_resolution((w, h))
                if latent_cache_matches_reso(cache_file, bucket, self.spatial_factor) is False:
                    skipped_reso += 1
                    continue
                it = ItemInfo(item_key, "", (w, h), bucket, cache_file)
                it.text_encoder_output_cache_path = te
                it.cache_directory = cache_dir
                it.image_path = next((os.path.join(img_dir, item_key + e) for e in IMAGE_EXTENSIONS
                                      if os.path.exists(os.path.join(img_dir, item_key + e))), None)
                self.buckets.setdefault(bucket, []).extend([it] * max(1, int(d.get("num_repeats", 1))))
        if skipped_stale:
            logger.info(f"[dataset] ignored {skipped_stale} stale cache file(s) whose image is gone")
        if missing_te:
            logger.warning(f"[dataset] {missing_te} image(s) have latents but no text cache - re-run caching")
        if skipped_reso:
            logger.warning(f"[dataset] {skipped_reso} cache file(s) were written at a different Target Megapixels "
                           f"and are skipped - re-run cache preparation")
        self.shuffle(0)
        for reso in sorted(self.buckets):
            logger.info(f"bucket: {reso}, count: {len(self.buckets[reso])}")
        return self.num_items

    @property
    def num_items(self) -> int:
        return sum(len(b) for b in self.buckets.values())

    def items(self) -> dict:
        """item_key -> ItemInfo over every bucket (what the loss watch addresses)."""
        return {str(it.item_key): it for b in self.buckets.values() for it in b}

    def shuffle(self, seed: int) -> None:
        """New order for an epoch: items within buckets, then the batch order (Fizgig's BucketBatchManager)."""
        rng = random.Random(seed)
        self.batches = []
        for reso in sorted(self.buckets):
            bucket = self.buckets[reso]
            rng.shuffle(bucket)
            self.batches += [(reso, i) for i in range(0, len(bucket), self.batch_size)]
        rng.shuffle(self.batches)

    def __len__(self):
        return len(self.batches)

    def _text_file(self, it: ItemInfo, rng: random.Random) -> str:
        """The conditioning file for this step: the caption, a shuffled variant, or (dropout) the empty caption."""
        if self.caption_dropout and rng.random() < self.caption_dropout:
            p = empty_cache_path(it.cache_directory, self.arch)
            if os.path.exists(p):
                return p
        if self.variants:
            choices = [it.text_encoder_output_cache_path] + [
                p for p in (text_cache_path(it.cache_directory, it.item_key, self.arch, i) for i in range(self.variants))
                if os.path.exists(p)]
            return rng.choice(choices)
        return it.text_encoder_output_cache_path

    def get_batch(self, idx: int, rng: Optional[random.Random] = None) -> dict:
        """Stacked tensors for batch `idx`: latents, latents_control_<i>, cond__<key> (verbatim), item_keys."""
        import torch
        from safetensors.torch import load_file
        rng = rng or random.Random()
        reso, start = self.batches[idx]
        chunk = self.buckets[reso][start:start + self.batch_size]
        data: dict = {}
        for it in chunk:
            sd = {**load_file(it.latent_cache_path), **load_file(self._text_file(it, rng))}
            for key, tensor in sd.items():
                if key.startswith("latent_control_"):
                    k = f"latents_control_{key.split('_')[2]}"
                elif key.startswith("latent_"):
                    k = "latents"
                else:
                    k = key
                data.setdefault(k, []).append(tensor)
        out = {k: torch.stack(v) for k, v in data.items()}
        out["item_keys"] = [it.item_key for it in chunk]
        return out
