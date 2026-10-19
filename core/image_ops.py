"""Image transforms for dataset preparation (pure Pillow; no Qt).

* Images are opened upright (EXIF orientation applied), so what you see is
  what gets processed.
* Saving keeps the ICC color profile and EXIF metadata when the target format
  supports them (orientation is reset, since pixels are already upright).
* Transparent images saved as JPEG are flattened onto white instead of black.
* ``open_upright(path, keep_extras=True)`` additionally carries PNG text chunks (A1111 / ComfyUI prompts), XMP and
  JPEG comments, and ``save`` writes them back where the target format has a place for them.
* 16-bit greyscale images are scaled to 8 bit (``to_8bit``) instead of being clipped.
* Image files are written atomically and fsynced; an existing file keeps its permission bits.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, PngImagePlugin

from core.caption_io import apply_mode

Image.MAX_IMAGE_PIXELS = 300_000_000

# info keys that always travel with an image through apply(); the "extras" only exist when open_upright was asked
# to keep them (the in-place editor does; export does not, so its "strip metadata" option keeps meaning what it says).
_CORE_INFO = ("icc_profile", "exif", "dpi")
EXTRA_INFO = ("xmp", "comment", "png_text")
CARRY_INFO = _CORE_INFO + EXTRA_INFO
XMP_PNG_KEY = "XML:com.adobe.xmp"

FOCUS_POINTS = {
    "Center": (0.5, 0.5), "Top": (0.5, 0.0), "Bottom": (0.5, 1.0), "Left": (0.0, 0.5), "Right": (1.0, 0.5),
    "Top-Left": (0.0, 0.0), "Top-Right": (1.0, 0.0), "Bottom-Left": (0.0, 1.0), "Bottom-Right": (1.0, 1.0),
}
ASPECTS = {"1:1": (1, 1), "2:3": (2, 3), "3:2": (3, 2), "3:4": (3, 4), "4:3": (4, 3), "4:5": (4, 5),
           "5:4": (5, 4), "9:16": (9, 16), "16:9": (16, 9), "9:21": (9, 21), "21:9": (21, 9)}
FORMATS = {"JPG": ("JPEG", ".jpg"), "PNG": ("PNG", ".png"), "WEBP": ("WEBP", ".webp"),
           "BMP": ("BMP", ".bmp"), "TIFF": ("TIFF", ".tif")}


class SkipImage(Exception):
    """The operation doesn't apply to this image (reported, not treated as a crash)."""


@dataclass
class Operation:
    kind: str                    # rotate | flip | resize | crop | crop_aspect | convert
    params: dict = field(default_factory=dict)

    def describe(self) -> str:
        p = self.params
        if self.kind == "rotate":
            return {"cw": "Rotate right", "ccw": "Rotate left", "180": "Rotate 180°"}[p.get("direction", "cw")]
        if self.kind == "flip":
            return "Flip horizontal" if p.get("axis", "h") == "h" else "Flip vertical"
        if self.kind == "resize":
            m = p.get("mode", "longest")
            if m == "force":
                return f"Resize to {p['w']}×{p['h']}"
            if m == "scale":
                return f"Scale {p['percent']}%"
            return f"Resize {m} side to {p['size']} px"
        if self.kind == "crop":
            return f"Crop {p['w']}×{p['h']} ({p.get('focus', 'Center')})"
        if self.kind == "crop_aspect":
            return f"Crop to {p['aspect']} ({p.get('focus', 'Center')})"
        if self.kind == "convert":
            return f"Convert to {p['format']}"
        return self.kind


def to_8bit(im: Image.Image) -> Image.Image:
    """16-bit greyscale (I;16 / I;16L / I;16B / I) scaled to 8-bit "L"; every other mode is returned unchanged.

    Pillow clips these modes at 255 when converting, which turns a 16-bit gradient into black and white.
    """
    if im.mode not in ("I;16", "I;16L", "I;16B", "I;16N", "I"):
        return im
    arr = np.asarray(im).astype(np.int64)
    top = int(arr.max()) if arr.size else 0
    if im.mode == "I" and top <= 255:
        return im.convert("L")                          # 32-bit data already in 0-255 (a 16-bit mode never is)
    divisor = 257.0 if top <= 65535 else top / 255.0   # 65535 / 257 == 255
    out = np.clip(np.rint(np.clip(arr, 0, None) / divisor), 0, 255).astype(np.uint8)
    return Image.fromarray(out, "L")


def _png_text_of(im: Image.Image) -> dict:
    text = getattr(im, "text", None) or {}
    return {k: v for k, v in text.items() if isinstance(k, str) and isinstance(v, str)}


def open_upright(path: str | os.PathLike, keep_extras: bool = False) -> Image.Image:
    """Open with EXIF orientation applied. ``keep_extras`` also carries PNG text, XMP and JPEG comments."""
    with Image.open(path) as im:
        fmt = im.format
        im.load()
        # XMP rides in out.info (below), patched by exif_transpose; PNG text never holds orientation
        png_text = {k: v for k, v in _png_text_of(im).items() if k != XMP_PNG_KEY} if keep_extras else {}
        out = ImageOps.exif_transpose(im)
        out.load()
    # REPLACE the info, never add to it: exif_transpose's copy carries the source's whole info (XMP, comments ...),
    # and export's "strip metadata" must not leak them. Taken AFTER the transpose, so EXIF and XMP orientation
    # describe the rotated pixels
    post = dict(out.info)
    wanted = CARRY_INFO + (XMP_PNG_KEY,) if keep_extras else _CORE_INFO
    out.info = {k: v for k, v in post.items() if k in wanted}
    if png_text:
        out.info["png_text"] = png_text
    out.format = fmt
    return out


def _focus_box(w: int, h: int, cw: int, ch: int, focus: str) -> tuple[int, int, int, int]:
    fx, fy = FOCUS_POINTS.get(focus, (0.5, 0.5))
    x = round((w - cw) * fx)
    y = round((h - ch) * fy)
    return x, y, x + cw, y + ch


def apply(img: Image.Image, op: Operation) -> Image.Image:
    p = op.params
    w, h = img.size
    keep = {k: v for k, v in img.info.items() if k in CARRY_INFO}
    if op.kind == "rotate":
        out = img.transpose({"cw": Image.Transpose.ROTATE_270, "ccw": Image.Transpose.ROTATE_90,
                             "180": Image.Transpose.ROTATE_180}[p.get("direction", "cw")])
    elif op.kind == "flip":
        out = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT if p.get("axis", "h") == "h"
                            else Image.Transpose.FLIP_TOP_BOTTOM)
    elif op.kind == "resize":
        mode = p.get("mode", "longest")
        if mode == "force":
            nw, nh = int(p["w"]), int(p["h"])
        elif mode == "scale":
            f = float(p["percent"]) / 100
            nw, nh = round(w * f), round(h * f)
        else:
            side = max(w, h) if mode == "longest" else min(w, h)
            f = int(p["size"]) / side
            nw, nh = round(w * f), round(h * f)
        nw, nh = max(1, nw), max(1, nh)
        if (nw > w or nh > h) and not p.get("allow_upscale", False):
            raise SkipImage(f"already smaller than the target ({w}×{h}); upscaling is off")
        out = img if (nw, nh) == (w, h) else img.resize((nw, nh), Image.Resampling.LANCZOS)
    elif op.kind == "crop":
        cw, ch = int(p["w"]), int(p["h"])
        if cw > w or ch > h:
            raise SkipImage(f"too small to crop {cw}×{ch} (image is {w}×{h})")
        out = img.crop(_focus_box(w, h, cw, ch, p.get("focus", "Center")))
    elif op.kind == "crop_aspect":
        aw, ah = ASPECTS.get(p["aspect"], (1, 1))
        target = aw / ah
        if abs(w / h - target) < 1e-3:
            out = img
        elif w / h > target:      # too wide: trim width
            cw, ch = round(h * target), h
            out = img.crop(_focus_box(w, h, cw, ch, p.get("focus", "Center")))
        else:                     # too tall: trim height
            cw, ch = w, round(w / target)
            out = img.crop(_focus_box(w, h, cw, ch, p.get("focus", "Center")))
    elif op.kind == "convert":
        out = img  # format change happens at save time
    else:
        raise ValueError(f"Unknown operation {op.kind}")
    if out is not img:
        out.info.update(keep)
    return out


def _exif_reset_orientation(exif_bytes: bytes | None) -> bytes | None:
    if not exif_bytes:
        return None
    try:
        exif = Image.Exif()
        exif.load(exif_bytes)
        exif[0x0112] = 1
        return exif.tobytes()
    except Exception:
        return None


def _extras_for(img: Image.Image, pil_fmt: str) -> dict:
    """Save keywords that carry PNG text, XMP and the JPEG comment into the target format."""
    info = img.info
    xmp = info.get("xmp") or info.get(XMP_PNG_KEY)       # a PNG source's XMP iTXt, patched by exif_transpose
    if isinstance(xmp, str):
        xmp = xmp.encode("utf-8")
    kw: dict = {}
    if pil_fmt == "PNG":
        text = dict(info.get("png_text") or {})
        if xmp and XMP_PNG_KEY not in text:
            text[XMP_PNG_KEY] = xmp.decode("utf-8", "replace")
        if text:
            meta = PngImagePlugin.PngInfo()
            for k, v in text.items():
                try:
                    meta.add_text(k, v)
                except (UnicodeError, ValueError):
                    meta.add_itxt(k, v)
            kw["pnginfo"] = meta
    elif pil_fmt == "JPEG":
        if xmp:
            kw["xmp"] = xmp
        comment = info.get("comment")
        if comment:
            kw["comment"] = comment
    elif pil_fmt == "WEBP":
        if xmp:
            kw["xmp"] = xmp
        elif info.get("png_text", {}).get(XMP_PNG_KEY):
            kw["xmp"] = info["png_text"][XMP_PNG_KEY].encode("utf-8")
    return kw


def _fsync_path(path) -> None:
    fd = os.open(path, os.O_RDWR if os.name == "nt" else os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save(img: Image.Image, dest: Path, fmt_name: str | None = None, quality: int = 92) -> Path:
    """Save atomically next to ``dest``. ``fmt_name`` is a key of FORMATS or None (keep extension)."""
    dest = Path(dest)
    if fmt_name:
        pil_fmt, ext = FORMATS[fmt_name]
        dest = dest.with_suffix(ext)
    else:
        ext = dest.suffix.lower()
        pil_fmt = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP", ".bmp": "BMP",
                   ".tif": "TIFF", ".tiff": "TIFF"}.get(ext, img.format or "PNG")
    out = to_8bit(img) if pil_fmt in ("JPEG", "WEBP", "BMP") else img   # PNG / TIFF keep 16 bits
    if pil_fmt in ("JPEG", "BMP") and out.mode in ("RGBA", "LA", "P", "PA"):
        rgba = out.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        out = bg
    elif pil_fmt == "JPEG" and out.mode not in ("RGB", "L", "CMYK"):
        out = out.convert("RGB")
    kwargs: dict = {}
    if "icc_profile" in img.info and pil_fmt in ("JPEG", "PNG", "WEBP", "TIFF"):
        kwargs["icc_profile"] = img.info["icc_profile"]
    exif = _exif_reset_orientation(img.info.get("exif"))
    if exif and pil_fmt in ("JPEG", "PNG", "WEBP", "TIFF"):
        kwargs["exif"] = exif
    kwargs.update(_extras_for(img, pil_fmt))
    if pil_fmt in ("JPEG", "WEBP"):
        kwargs["quality"] = int(quality)
    if pil_fmt == "JPEG":
        kwargs["subsampling"] = 0 if quality >= 90 else 2
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".edit-", suffix=dest.suffix, dir=dest.parent)
    os.close(fd)
    try:
        out.save(tmp, format=pil_fmt, **kwargs)
        _fsync_path(tmp)
        apply_mode(dest, tmp)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return dest


def preview(img: Image.Image, op: Operation, max_side: int = 640) -> tuple[Image.Image, tuple[int, int]]:
    """Fast preview: (preview image, full-resolution result size). Raises SkipImage when not applicable."""
    full = apply(img if max(img.size) <= 2048 else _proxy(img, 2048), op)
    w, h = img.size
    # Report the full-size result dimensions by applying the size logic to the original dims.
    result_size = _result_size(img.size, op) or full.size
    shown = full.copy()
    shown.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return shown, result_size


def _proxy(img: Image.Image, side: int) -> Image.Image:
    p = img.copy()
    p.thumbnail((side, side), Image.Resampling.LANCZOS)
    return p


def _result_size(size: tuple[int, int], op: Operation) -> tuple[int, int] | None:
    w, h = size
    p = op.params
    if op.kind == "rotate":
        return (w, h) if p.get("direction") == "180" else (h, w)
    if op.kind in ("flip", "convert"):
        return (w, h)
    if op.kind == "crop":
        return int(p["w"]), int(p["h"])
    if op.kind == "crop_aspect":
        aw, ah = ASPECTS.get(p["aspect"], (1, 1))
        t = aw / ah
        return (round(h * t), h) if w / h > t else (w, round(w / t))
    if op.kind == "resize":
        mode = p.get("mode", "longest")
        if mode == "force":
            return int(p["w"]), int(p["h"])
        if mode == "scale":
            f = float(p["percent"]) / 100
            return round(w * f), round(h * f)
        side = max(w, h) if mode == "longest" else min(w, h)
        f = int(p["size"]) / side
        return round(w * f), round(h * f)
    return None
