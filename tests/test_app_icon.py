"""The window / taskbar icon (resources/Logo.ico, built by tools/make_icon.py): every size Windows asks for, so the
taskbar never scales one small picture up or shows a dark tile with a few faint pixels."""
from PIL import Image

from core import paths


def test_icon_file_holds_every_size_the_taskbar_asks_for():
    ico = paths.resource("logo.ico")
    assert ico is not None and ico.is_file()
    sizes = set(Image.open(ico).info["sizes"])
    assert {(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (256, 256)} <= sizes, sorted(sizes)


def test_the_mark_fills_the_tile_and_corners_are_clear():
    ico = Image.open(paths.resource("logo.ico"))
    ico.size = (256, 256)
    im = ico.convert("RGBA")
    assert im.getpixel((0, 0))[3] == 0 and im.getpixel((128, 128))[3] == 255      # rounded tile on a clear ground
    background = im.getpixel((128, 10))[:3]
    coloured = [(x, y) for y in range(0, 256, 4) for x in range(0, 256, 4)
                if im.getpixel((x, y))[3] == 255 and max(abs(a - b) for a, b in zip(im.getpixel((x, y))[:3], background)) > 40]
    xs, ys = [p[0] for p in coloured], [p[1] for p in coloured]
    assert max(xs) - min(xs) > 150 or max(ys) - min(ys) > 150, "the mark should span most of the tile"
