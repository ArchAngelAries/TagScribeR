"""Dataset folder scanning and per-image records."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

_NUM_RE = re.compile(r"(\d+)")


def natural_key(name: str) -> list:
    """Sort key so 'img (2)' comes before 'img (10)'."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in _NUM_RE.split(name)]


def is_image(path: str | os.PathLike) -> bool:
    return str(path).lower().endswith(IMAGE_EXTS)


def scan_images(folder: str | os.PathLike, recursive: bool = False) -> list[Path]:
    """List image files in a folder, naturally sorted. Skips hidden files/folders."""
    root = Path(folder)
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    out: list[Path] = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted((d for d in dirnames if not d.startswith(".")), key=natural_key)
            for f in filenames:
                if not f.startswith(".") and is_image(f):
                    out.append(Path(dirpath) / f)
        out.sort(key=lambda p: natural_key(str(p.relative_to(root))))
    else:
        with os.scandir(root) as it:
            for entry in it:
                if entry.is_file() and not entry.name.startswith(".") and is_image(entry.name):
                    out.append(Path(entry.path))
        out.sort(key=lambda p: natural_key(p.name))
    return out


@dataclass
class ImageInfo:
    width: int = 0
    height: int = 0
    format: str = ""
    file_size: int = 0

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1_000_000


def read_image_info(path: str | os.PathLike) -> ImageInfo:
    """Header-only read of dimensions (honours EXIF rotation). Never decodes pixels."""
    from PIL import Image
    info = ImageInfo()
    try:
        info.file_size = os.path.getsize(path)
        with Image.open(path) as im:
            w, h = im.size
            try:
                orientation = im.getexif().get(0x0112, 1)
            except Exception:
                orientation = 1
            if orientation in (5, 6, 7, 8):
                w, h = h, w
            info.width, info.height, info.format = w, h, im.format or ""
    except Exception:
        pass
    return info
