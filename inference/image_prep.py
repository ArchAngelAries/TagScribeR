"""Image preparation shared by all providers."""
from __future__ import annotations

import base64
import io
import math
import os

from PIL import Image, ImageOps

# Large dataset images are common; Pillow's bomb guard would refuse many legitimate photos.
Image.MAX_IMAGE_PIXELS = 300_000_000

# A JPEG that will be shrunk anyway is decoded at a reduced scale (1/2, 1/4, 1/8) when that still leaves
# at least this many times the target size for the final resample. 1.5 halves the decode time of 12 MP+
# photos; the result differs from a full decode by less than the JPEG re-encode does (PSNR 49 dB or better,
# measured on generated pictures; see docs/dev/CAPTION_SPEED.md).
JPEG_DRAFT_GAP = 1.5


def fit_size(width: int, height: int, max_side: int = 0, max_pixels: int = 0) -> tuple[int, int]:
    """The size an image of ``width`` x ``height`` is sent at: the longest side at most ``max_side`` and the
    area at most ``max_pixels`` (0 = no limit for either), aspect ratio kept, never enlarged."""
    width, height = max(1, int(width)), max(1, int(height))
    scale = 1.0
    if max_side and max(width, height) > max_side:
        scale = min(scale, max_side / max(width, height))
    if max_pixels and width * height > max_pixels:
        scale = min(scale, math.sqrt(max_pixels / (width * height)))
    if scale >= 1.0:
        return width, height
    return max(1, round(width * scale)), max(1, round(height * scale))


def load_rgb(path: str | os.PathLike, max_side: int = 0, background=(255, 255, 255),
             max_pixels: int = 0) -> Image.Image:
    """Open an image upright (EXIF), flatten transparency onto ``background``, convert to RGB and optionally
    downscale so the longest side <= ``max_side`` and the area <= ``max_pixels`` (see ``fit_size``)."""
    with Image.open(path) as im:
        target = fit_size(*im.size, max_side, max_pixels)
        if target != im.size and im.format in ("JPEG", "MPO"):
            # Same box in either orientation, so this is right before the EXIF rotation too.
            im.draft("RGB", (math.ceil(target[0] * JPEG_DRAFT_GAP), math.ceil(target[1] * JPEG_DRAFT_GAP)))
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, background)
            bg.paste(im, mask=im.getchannel("A"))
            im = bg
        else:
            im = im.convert("RGB")
        box = fit_size(*im.size, max_side, max_pixels)
        if box != im.size:
            im.thumbnail(box, Image.Resampling.LANCZOS)
        else:
            im.load()
        return im


def to_jpeg_base64(img: Image.Image, quality: int = 92) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def to_data_url(img: Image.Image, quality: int = 92) -> str:
    """The picture as the ``data:`` URL an OpenAI-compatible server takes (made once, reused on a retry)."""
    return f"data:image/jpeg;base64,{to_jpeg_base64(img, quality)}"
