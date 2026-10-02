"""Export a curated image set into a training-ready folder.

Sources are never modified. The output gets processed copies (bucket resize,
optional format conversion, metadata stripped), sequential names if wanted, and
caption files beside each image, in an optional kohya-style ``<repeats>_<name>``
subfolder. Existing files in the output are never overwritten.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from core import caption_io, fileops, health, image_ops


UPSCALE_TOLERANCE = 1.10   # up to 10% enlargement is allowed without "allow upscaling"


@dataclass
class ExportOptions:
    out_dir: str = ""
    kohya_folder: bool = False
    repeats: int = 10
    concept: str = "dataset"
    resize: str = "bucket"           # none | bucket | longest
    resolution: int = 1024           # bucket resolution or max longest side
    allow_upscale: bool = False
    fmt: str = ""                    # "" = keep, else a key of image_ops.FORMATS
    quality: int = 95
    strip_metadata: bool = True
    rename: bool = False
    name_prefix: str = "img"
    caption_ext: str = ".txt"
    trigger: str = ""                # prepended to captions if not already present
    skip_uncaptioned: bool = True

    def target_dir(self) -> Path:
        base = Path(self.out_dir)
        if self.kohya_folder:
            safe = "".join(ch for ch in self.concept.strip() if ch not in '<>:"/\\|?*') or "dataset"
            return base / f"{max(1, int(self.repeats))}_{safe}"
        return base


def _bucket_resize(img: Image.Image, resolution: int, allow_upscale: bool) -> Image.Image:
    buckets = health.make_buckets(resolution)
    fit = health.fit_bucket(img.width, img.height, buckets)
    bw, bh = fit.bucket
    scale = max(bw / img.width, bh / img.height)
    # A few percent of upscaling is invisible (e.g. 1024x1021 into a 1024x1024 bucket); only images that
    # would be noticeably enlarged are skipped unless upscaling is allowed.
    if scale > UPSCALE_TOLERANCE and not allow_upscale:
        raise image_ops.SkipImage(f"smaller than its {bw}×{bh} bucket (enable upscaling to include it)")
    nw, nh = max(bw, round(img.width * scale)), max(bh, round(img.height * scale))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS) if (nw, nh) != img.size else img
    left, top = (nw - bw) // 2, (nh - bh) // 2
    out = resized.crop((left, top, left + bw, top + bh))
    out.info.update({k: v for k, v in img.info.items() if k in ("icc_profile", "exif", "dpi")})
    return out


def export_one(src: str | os.PathLike, index: int, opts: ExportOptions) -> Path:
    """Export one image (+ caption). Returns the written image path. Raises SkipImage to skip."""
    src = Path(src)
    caption = caption_io.read_caption(src)
    if opts.skip_uncaptioned and not caption.strip():
        raise image_ops.SkipImage("no caption")
    img = image_ops.open_upright(src)
    if opts.resize == "bucket":
        img = _bucket_resize(img, opts.resolution, opts.allow_upscale)
    elif opts.resize == "longest" and max(img.size) > opts.resolution:
        img = image_ops.apply(img, image_ops.Operation("resize", {"mode": "longest", "size": opts.resolution}))
    if opts.strip_metadata:
        img.info.pop("exif", None)  # keep the ICC profile so colours stay correct
    ext = image_ops.FORMATS[opts.fmt][1] if opts.fmt else src.suffix.lower()
    stem = f"{opts.name_prefix}_{index:04d}" if opts.rename else src.stem
    out_dir = opts.target_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = fileops.unique_stem_for_pair(out_dir, stem, ext)
    dest = image_ops.save(img, out_dir / f"{stem}{ext}", opts.fmt or None, opts.quality)
    trigger = opts.trigger.strip()
    if trigger and caption.strip() and not caption.lower().startswith(trigger.lower()):
        caption = f"{trigger}, {caption.strip()}"
    elif trigger and not caption.strip():
        caption = trigger
    if caption.strip():
        caption_io.atomic_write_text(out_dir / f"{stem}{opts.caption_ext}", caption.strip())
    return dest


def summarize(opts: ExportOptions, n: int) -> str:
    parts = [f"{n} image(s) → {opts.target_dir()}"]
    if opts.resize == "bucket":
        parts.append(f"resized to {opts.resolution}px buckets")
    elif opts.resize == "longest":
        parts.append(f"longest side ≤ {opts.resolution}px")
    if opts.fmt:
        parts.append(f"as {opts.fmt}")
    if opts.strip_metadata:
        parts.append("metadata stripped")
    if opts.trigger.strip():
        parts.append(f"captions start with “{opts.trigger.strip()}”")
    return ", ".join(parts)

