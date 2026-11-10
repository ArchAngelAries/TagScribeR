"""Build resources/Logo.ico (the window and taskbar icon) from resources/Logo.png.

The logo picture is the mark on a large dark square. Drawn at taskbar size as it is, the mark is a few faint pixels
in a dark tile. This crops to the mark with a small margin, puts it on a rounded tile in the picture's own background
colour, and writes every size Windows asks for (16 to 256 px), each scaled from the full-size picture.

    venv\\Scripts\\python.exe tools/make_icon.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "resources" / "Logo.png"
TARGET = ROOT / "resources" / "Logo.ico"
SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)
MARGIN = 0.10           # empty border around the mark, as a share of the tile
RADIUS = 0.20           # corner radius of the tile, as a share of its side
SUPERSAMPLE = 4         # the tile is drawn this much larger, then scaled down, for smooth corners


def mark_box(img: Image.Image, background: tuple, tolerance: int = 24) -> tuple[int, int, int, int]:
    """The bounding box of everything that is not background."""
    diff = ImageChops.difference(img.convert("RGB"), Image.new("RGB", img.size, background[:3])).convert("L")
    box = diff.point(lambda v: 255 if v > tolerance else 0).getbbox()
    return box or (0, 0, img.width, img.height)


def build_tile(side: int = 1024) -> Image.Image:
    """The icon at full size: the mark centred on a rounded tile, transparent outside the corners."""
    src = Image.open(SOURCE).convert("RGBA")
    background = src.getpixel((2, 2))
    left, top, right, bottom = mark_box(src, background)
    mark = src.crop((left, top, right, bottom))
    inner = int(side * (1 - 2 * MARGIN))
    scale = inner / max(mark.size)
    mark = mark.resize((max(1, round(mark.width * scale)), max(1, round(mark.height * scale))), Image.LANCZOS)
    big = side * SUPERSAMPLE
    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, big - 1, big - 1), radius=int(big * RADIUS), fill=255)
    mask = mask.resize((side, side), Image.LANCZOS)
    tile = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    tile.paste(Image.new("RGBA", (side, side), background[:3] + (255,)), (0, 0), mask)
    tile.alpha_composite(mark, ((side - mark.width) // 2, (side - mark.height) // 2))
    tile.putalpha(ImageChops.multiply(tile.getchannel("A"), mask))
    return tile


def main() -> None:
    tile = build_tile()
    frames = [tile.resize((s, s), Image.LANCZOS) for s in SIZES]
    frames[-1].save(TARGET, format="ICO", sizes=[(s, s) for s in SIZES], append_images=frames[:-1])
    print(f"wrote {TARGET.relative_to(ROOT)} with sizes {', '.join(str(s) for s in SIZES)}")


if __name__ == "__main__":
    main()
