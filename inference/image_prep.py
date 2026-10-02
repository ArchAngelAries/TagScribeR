"""Image preparation shared by all providers."""
from __future__ import annotations

import base64
import io
import os

from PIL import Image, ImageOps

# Large dataset images are common; Pillow's bomb guard would refuse many legitimate photos.
Image.MAX_IMAGE_PIXELS = 300_000_000


def load_rgb(path: str | os.PathLike, max_side: int = 0, background=(255, 255, 255)) -> Image.Image:
    """Open an image upright (EXIF), flatten transparency onto ``background``,
    convert to RGB and optionally downscale so the longest side <= max_side."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, background)
            bg.paste(im, mask=im.getchannel("A"))
            im = bg
        else:
            im = im.convert("RGB")
        if max_side and max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        else:
            im.load()
        return im


def to_jpeg_base64(img: Image.Image, quality: int = 92) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")
