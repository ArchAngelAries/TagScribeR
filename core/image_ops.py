"""Image transforms for dataset preparation (pure Pillow; no Qt).

* Images are opened upright (EXIF orientation applied), so what you see is
  what gets processed.
* Saving keeps the ICC color profile and EXIF metadata when the target format
  supports them (orientation is reset, since pixels are already upright).
* Transparent images saved as JPEG are flattened onto white instead of black.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 300_000_000

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


def open_upright(path: str | os.PathLike) -> Image.Image:
    with Image.open(path) as im:
        info = dict(im.info)
        fmt = im.format
        out = ImageOps.exif_transpose(im)
        out.load()
    out.info.update({k: v for k, v in info.items() if k in ("icc_profile", "exif", "dpi")})
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
    keep = {k: v for k, v in img.info.items() if k in ("icc_profile", "exif", "dpi")}
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
    out = img
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
    if pil_fmt in ("JPEG", "WEBP"):
        kwargs["quality"] = int(quality)
    if pil_fmt == "JPEG":
        kwargs["subsampling"] = 0 if quality >= 90 else 2
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".edit-", suffix=dest.suffix, dir=dest.parent)
    os.close(fd)
    try:
        out.save(tmp, format=pil_fmt, **kwargs)
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
