"""Render the real TagScribeR UI offscreen and save annotated screenshots for the README.

    venv\\Scripts\\python.exe tools\\make_screenshots.py                    # demo dataset -> docs/screenshots
    venv\\Scripts\\python.exe tools\\make_screenshots.py --dataset D:\\my\\images
    venv\\Scripts\\python.exe tools\\make_screenshots.py --only 01,09 --out some\\dir

Everything runs against a throw-away TAGSCRIBER_USER_DATA folder and a generated (or copied) dataset in a temp
directory, so your own settings, models, collections and images are never read or written. No AI model is loaded and
nothing is trained or captioned. Machine-specific text (paths, GPU info) is replaced by neutral text before each grab,
and the text of every visible widget is checked for leaks.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import re
import shutil
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dataset", help="use a copy of this folder instead of the generated demo dataset")
ap.add_argument("--out", default=str(ROOT / "docs" / "screenshots"), help="output folder (default docs/screenshots)")
ap.add_argument("--only", default="", help="comma separated shot numbers or name parts, e.g. 01,10,health")
ap.add_argument("--keep-temp", action="store_true", help="keep the temp folder (printed at the end)")
ARGS = ap.parse_args()

# ---------------------------------------------------------------------------------------------------- sandbox
TMP = Path(os.path.realpath(tempfile.mkdtemp(prefix="tsr_shots_")))
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["TAGSCRIBER_USER_DATA"] = str(TMP / "user_data")
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.*=false;qt.text.font.*=false")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402

from core import paths  # noqa: E402

# Never read the owner's legacy config / API presets / quick tags from the app folder.
for _name in ("LEGACY_CONFIG_FILE", "LEGACY_API_PRESETS_FILE", "LEGACY_TAGS_FILE"):
    setattr(paths, _name, TMP / "no_such_legacy_file")

NEUTRAL = {  # real (temp) location -> what the screenshot shows
    "demo": r"D:\Datasets\demo_dataset",
    "collections": r"D:\Datasets\Collections",
    "edits": r"D:\Datasets\Image Edits",
    "runs": r"D:\TagScribeR\training_runs",
    "root": r"D:\TagScribeR",
}
DEMO_DIR = TMP / "demo_dataset"
COLL_DIR = TMP / "collections"
EDITS_DIR = TMP / "edits"
RUNS_DIR = TMP / "runs"

# ---------------------------------------------------------------------------------------------------- demo dataset
Rgb = tuple


def _arr(w: int, h: int) -> np.ndarray:
    return np.zeros((h, w, 3), np.float32)


def vgrad(w, h, stops):
    """Vertical gradient through colour stops [(pos 0..1, (r,g,b)), ...]."""
    ys = np.linspace(0, 1, h)
    out = _arr(w, h)
    pos = [p for p, _ in stops]
    for c in range(3):
        out[:, :, c] = np.interp(ys, pos, [col[c] for _, col in stops])[:, None]
    return out


def glow(arr, cx, cy, radius, color, strength=1.0):
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (radius ** 2)
    g = np.exp(-d2 * 2.2)[:, :, None] * strength
    arr += g * (np.array(color, np.float32) - arr * 0.0) * 0.0 + g * np.array(color, np.float32)
    return arr


def ridge_line(w, base, amp, seed, octaves=4):
    rng = random.Random(seed)
    x = np.linspace(0, 1, w)
    y = np.full(w, float(base))
    for i in range(octaves):
        f = 1.2 * (2.1 ** i) + rng.random()
        y += amp / (1.7 ** i) * np.sin(2 * math.pi * f * x + rng.random() * 6.28)
    return y


def fill_below(arr, line, top_color, bottom_color, depth=0.5):
    h, w, _ = arr.shape
    yy = np.arange(h, dtype=np.float32)[:, None]
    mask = yy >= line[None, :]
    t = np.clip((yy - line[None, :]) / (h * depth), 0, 1)[:, :, None]
    col = np.array(top_color, np.float32) * (1 - t) + np.array(bottom_color, np.float32) * t
    arr[:] = np.where(mask[:, :, None], col, arr)
    return arr


def noise_texture(w, h, seed, scale=64, amp=1.0):
    rng = np.random.default_rng(seed)
    small = rng.random((max(2, h // scale), max(2, w // scale)), dtype=np.float32)
    img = Image.fromarray((small * 255).astype("uint8")).resize((w, h), Image.BICUBIC)
    return (np.asarray(img, np.float32) / 255.0 - 0.5) * amp


def to_image(arr) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), "RGB")


def scene_mountains(w, h, seed, sky, sun=None, ridges=None, mist=0.35):
    arr = vgrad(w, h, sky)
    if sun:
        sx, sy, sr, sc = sun
        glow(arr, w * sx, h * sy, w * sr, sc, 0.9)
        yy, xx = np.mgrid[0:h, 0:w]
        disk = ((xx - w * sx) ** 2 + (yy - h * sy) ** 2) < (w * sr * 0.12) ** 2
        arr[disk] = np.array(sc, np.float32) * 0.5 + 135
    layers = ridges or [((190, 150, 170), (150, 110, 140)), ((140, 105, 135), (100, 75, 105)),
                        ((95, 70, 100), (60, 45, 72)), ((50, 38, 62), (28, 22, 40))]
    n = len(layers)
    for i, (c1, c2) in enumerate(layers):
        base = h * (0.42 + 0.1 * i)
        line = ridge_line(w, base, h * (0.16 - 0.015 * i), seed * 7 + i)
        fill_below(arr, line, c1, c2, depth=0.35)
        # mist between layers
        yy = np.arange(h, dtype=np.float32)[:, None]
        m = np.exp(-((yy - (base + h * 0.05)) / (h * 0.07)) ** 2)[:, :, None] * mist * (1 - i / (n + 1))
        arr = arr * (1 - m) + np.array(sky[len(sky) // 2][1], np.float32) * m
    arr += noise_texture(w, h, seed, 48, 6)[:, :, None]
    return to_image(arr)


def scene_sunset_sea(w, h, seed, palette="orange"):
    cols = {"orange": [(0, (40, 30, 80)), (0.35, (200, 80, 90)), (0.55, (255, 170, 80)), (0.6, (255, 215, 130))],
            "rose": [(0, (70, 50, 110)), (0.35, (220, 110, 150)), (0.55, (255, 190, 150)), (0.6, (255, 230, 190))]}[palette]
    horizon = int(h * 0.6)
    arr = vgrad(w, h, [(p * 0.6 / 0.6, c) for p, c in cols])
    arr[horizon:] = 0
    sun_x, sun_y = w * 0.5, horizon - h * 0.06
    glow(arr, sun_x, sun_y, w * 0.25, (255, 190, 110), 0.85)
    yy, xx = np.mgrid[0:h, 0:w]
    arr[(((xx - sun_x) ** 2 + (yy - sun_y) ** 2) < (h * 0.075) ** 2) & (yy < horizon)] = (255, 240, 200)
    sea = vgrad(w, h - horizon, [(0, (255, 190, 120)), (0.15, (190, 90, 100)), (1, (30, 25, 70))])
    rng = np.random.default_rng(seed)
    stripes = np.zeros((h - horizon, w), np.float32)
    for r in range(h - horizon):
        spread = (0.02 + 0.2 * r / (h - horizon)) * w
        for _ in range(3):
            x0 = sun_x + rng.normal(0, spread * 0.6)
            half = rng.uniform(4, 22) * (1 + r / (h - horizon) * 2)
            lo, hi = int(max(0, x0 - half)), int(min(w, x0 + half))
            stripes[r, lo:hi] += rng.uniform(0.25, 0.7) * (1 - 0.8 * r / (h - horizon))
    sea += stripes[:, :, None] * np.array((120, 70, 40), np.float32)
    arr[horizon:] = sea
    # distant islands and a few cloud streaks
    for i in range(3):
        line = ridge_line(w, horizon - h * (0.01 + 0.015 * i), h * 0.012, seed + i, 3)
        band = np.arange(h)[:, None] >= line[None, :]
        band &= (np.arange(h)[:, None] < horizon)
        arr[band] = np.array((60, 40, 80), np.float32) * (0.8 + i * 0.1)
    img = to_image(arr)
    d = ImageDraw.Draw(img, "RGBA")
    for _ in range(9):
        cx, cy = rng.uniform(0, w), rng.uniform(h * 0.08, h * 0.38)
        for k in range(4):
            d.ellipse([cx - 220 + k * 40, cy - 7 + k * 3, cx + 160 - k * 30, cy + 7 + k * 3],
                      fill=(255, 190, 160, 34))
    return img.filter(ImageFilter.GaussianBlur(1.1))


def scene_hills(w, h, seed, tint=0.0):
    arr = vgrad(w, h, [(0, (70 + tint, 140, 215)), (0.55, (170, 215, 245)), (1, (225, 240, 250))])
    img = to_image(arr)
    clouds = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(clouds)
    rng = random.Random(seed)
    for _ in range(7):
        cx, cy = rng.uniform(0.05, 0.95) * w, rng.uniform(0.06, 0.4) * h
        for _k in range(9):
            rx, ry = rng.uniform(0.06, 0.13) * w, rng.uniform(0.025, 0.05) * h
            ox, oy = rng.uniform(-0.1, 0.1) * w, rng.uniform(-0.02, 0.02) * h
            d.ellipse([cx + ox - rx, cy + oy - ry, cx + ox + rx, cy + oy + ry], fill=(255, 255, 255, 190))
    clouds = clouds.filter(ImageFilter.GaussianBlur(h * 0.012))
    img.paste(clouds, (0, 0), clouds)
    arr = np.asarray(img, np.float32).copy()
    hills = [((120, 190, 110), (70, 140, 80)), ((90, 165, 90), (50, 115, 65)), ((60, 130, 70), (35, 90, 50))]
    for i, (c1, c2) in enumerate(hills):
        line = ridge_line(w, h * (0.52 + 0.14 * i), h * 0.07, seed * 3 + i, 3)
        fill_below(arr, line, c1, c2, depth=0.4)
    arr += noise_texture(w, h, seed, 24, 10)[:, :, None]
    return to_image(arr)


def scene_night(w, h, seed, aurora=False):
    arr = vgrad(w, h, [(0, (6, 10, 30)), (0.6, (18, 32, 70)), (1, (40, 60, 100))])
    rng = np.random.default_rng(seed)
    if aurora:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        for k, col in enumerate(((60, 255, 170), (110, 120, 255), (40, 220, 120))):
            wave = h * (0.28 + 0.08 * k) + h * 0.07 * np.sin(xx / w * (5 + k * 2) + k)
            band = np.exp(-((yy - wave) / (h * (0.05 + 0.02 * k))) ** 2)
            band *= 0.5 + 0.5 * np.sin(xx / w * 40 + k * 3) ** 2
            arr += band[:, :, None] * np.array(col, np.float32) * 0.55
    n = 380
    xs, ys = rng.uniform(0, w, n), rng.uniform(0, h * 0.65, n)
    img = to_image(arr)
    d = ImageDraw.Draw(img, "RGBA")
    for x, y in zip(xs, ys):
        r = rng.uniform(0.6, 2.0)
        a = int(rng.uniform(120, 255))
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 250, 235, a))
    if not aurora:
        mx, my, mr = w * 0.72, h * 0.2, h * 0.06
        halo = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(halo).ellipse([mx - mr * 3, my - mr * 3, mx + mr * 3, my + mr * 3], fill=(200, 215, 255, 55))
        halo = halo.filter(ImageFilter.GaussianBlur(mr))
        img.paste(halo, (0, 0), halo)
        ImageDraw.Draw(img).ellipse([mx - mr, my - mr, mx + mr, my + mr], fill=(245, 245, 230))
    arr = np.asarray(img, np.float32).copy()
    layers = [((70, 85, 125), (40, 55, 90)), ((30, 40, 70), (15, 22, 42)), ((12, 16, 30), (6, 8, 16))]
    if aurora:
        layers = [((205, 220, 240), (120, 140, 180)), ((95, 115, 160), (45, 60, 100)), ((20, 26, 44), (8, 10, 20))]
    for i, (c1, c2) in enumerate(layers):
        line = ridge_line(w, h * (0.58 + 0.1 * i), h * (0.12 - 0.015 * i), seed + 11 * i, 4)
        fill_below(arr, line, c1, c2, depth=0.4)
    return to_image(arr)


def scene_desert(w, h, seed):
    arr = vgrad(w, h, [(0, (250, 190, 120)), (0.5, (255, 220, 160)), (1, (255, 235, 200))])
    glow(arr, w * 0.74, h * 0.28, w * 0.2, (255, 235, 180), 0.9)
    yy, xx = np.mgrid[0:h, 0:w]
    arr[((xx - w * 0.74) ** 2 + (yy - h * 0.28) ** 2) < (h * 0.06) ** 2] = (255, 250, 225)
    for i in range(5):
        line = ridge_line(w, h * (0.5 + 0.1 * i), h * 0.08, seed * 5 + i, 3)
        lit = np.array((240 - i * 15, 175 - i * 14, 100 - i * 8), np.float32)
        shade = lit * 0.55
        fill_below(arr, line, lit, shade, depth=0.22)
    arr += noise_texture(w, h, seed, 6, 7)[:, :, None]
    return to_image(arr)


def scene_pines(w, h, seed):
    arr = vgrad(w, h, [(0, (240, 170, 120)), (0.45, (250, 205, 150)), (0.7, (200, 190, 190)), (1, (110, 120, 140))])
    glow(arr, w * 0.3, h * 0.45, w * 0.3, (255, 225, 170), 0.6)
    img = to_image(arr)
    rng = random.Random(seed)
    for layer in range(4):
        shade = (150 - layer * 32, 130 - layer * 28, 150 - layer * 24)
        d = ImageDraw.Draw(img, "RGBA")
        base = h * (0.62 + 0.09 * layer)
        d.rectangle([0, base, w, h], fill=shade + (255,))
        x = -20
        while x < w + 20:
            th = rng.uniform(0.18, 0.3) * h * (1 + layer * 0.35)
            tw = th * 0.26
            for k in range(4):
                top = base - th + k * th * 0.22
                d.polygon([(x, top), (x - tw * (0.5 + k * 0.28), top + th * 0.36), (x + tw * (0.5 + k * 0.28), top + th * 0.36)],
                          fill=shade + (255,))
            x += rng.uniform(0.04, 0.09) * w * (1 + layer * 0.3)
        fog = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(fog).rectangle([0, base - h * 0.04, w, base + h * 0.06], fill=(230, 215, 215, 60))
        fog = fog.filter(ImageFilter.GaussianBlur(h * 0.025))
        img.paste(fog, (0, 0), fog)
    return img


def scene_geometric(w, h, seed):
    arr = vgrad(w, h, [(0, (255, 214, 200)), (0.5, (250, 190, 205)), (1, (190, 175, 235))])
    img = to_image(arr).convert("RGBA")
    rng = random.Random(seed)
    palette = [(255, 255, 255), (116, 185, 255), (0, 184, 148), (253, 203, 110), (108, 92, 231), (255, 118, 117)]
    for _ in range(11):
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        col = rng.choice(palette) + (rng.randint(110, 200),)
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.08, 0.26) * min(w, h)
        if rng.random() < 0.55:
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
        else:
            ang = rng.uniform(0, 6.28)
            d.polygon([(cx + r * math.cos(ang + k * 2.094), cy + r * math.sin(ang + k * 2.094)) for k in range(3)], fill=col)
        img = Image.alpha_composite(img, layer)
    return img.convert("RGB")


def scene_bokeh(w, h, seed):
    arr = vgrad(w, h, [(0, (25, 18, 40)), (1, (60, 28, 36))])
    img = to_image(arr).convert("RGBA")
    rng = random.Random(seed)
    for _ in range(46):
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        r = rng.uniform(0.025, 0.1) * w
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        col = rng.choice([(255, 190, 90), (255, 140, 90), (255, 225, 150), (255, 110, 120), (180, 150, 255)])
        d = ImageDraw.Draw(layer)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col + (rng.randint(55, 130),), outline=col + (170,), width=2)
        img = Image.alpha_composite(img, layer.filter(ImageFilter.GaussianBlur(1.5)))
    return img.convert("RGB")


def scene_city(w, h, seed):
    arr = vgrad(w, h, [(0, (40, 30, 90)), (0.5, (190, 90, 120)), (0.8, (250, 170, 120)), (1, (255, 200, 140))])
    img = to_image(arr)
    rng = random.Random(seed)
    for layer in range(3):
        d = ImageDraw.Draw(img)
        col = (70 - layer * 22, 50 - layer * 16, 100 - layer * 26)
        x = -10
        while x < w:
            bw = rng.uniform(0.04, 0.09) * w
            bh = rng.uniform(0.18, 0.5) * h * (1 - layer * 0.18)
            d.rectangle([x, h - bh, x + bw, h], fill=col)
            if layer == 2:
                for wy in range(int(h - bh + 10), h - 10, 18):
                    for wx in range(int(x + 6), int(x + bw - 8), 14):
                        if rng.random() < 0.45:
                            d.rectangle([wx, wy, wx + 6, wy + 9], fill=(255, 215, 130))
            x += bw + rng.uniform(0, 6)
    return img


def scene_waves(w, h, seed):
    arr = vgrad(w, h, [(0, (225, 245, 250)), (1, (190, 230, 240))])
    cols = [(120, 200, 215), (70, 160, 190), (30, 120, 160), (20, 85, 125), (15, 55, 95)]
    for i, c in enumerate(cols):
        x = np.linspace(0, 1, w)
        line = h * (0.3 + 0.13 * i) + h * 0.05 * np.sin(x * 2 * math.pi * (1.5 + 0.4 * i) + i) + h * 0.02 * np.sin(x * 14 + i)
        fill_below(arr, line, c, tuple(v * 0.8 for v in c), depth=0.3)
    return to_image(arr)


@dataclass
class Demo:
    name: str
    ext: str
    size: tuple
    make: Callable
    caption: str | None
    extra: dict = field(default_factory=dict)


def build_demo_dataset(folder: Path) -> None:
    from PIL import PngImagePlugin
    folder.mkdir(parents=True, exist_ok=True)
    pink = [(0, (60, 50, 110)), (0.4, (225, 130, 160)), (0.62, (255, 200, 170)), (1, (255, 225, 200))]
    blue = [(0, (45, 70, 130)), (0.45, (140, 175, 215)), (0.7, (205, 225, 240)), (1, (225, 238, 248))]
    blue_ridges = [((170, 195, 220), (130, 160, 195)), ((120, 150, 185), (85, 115, 155)),
                   ((80, 105, 145), (55, 75, 112)), ((42, 58, 92), (24, 34, 58))]
    items = [
        Demo("alpine_dawn", ".jpg", (1280, 960), lambda w, h: scene_mountains(w, h, 3, pink, (0.62, 0.38, 0.24, (255, 190, 140))),
             "ohwx, mountain, scenery, dawn, pink sky, mist, no humans, landscape, layered ridges", {"exif": "gps"}),
        Demo("aurora_peaks", ".png", (1024, 1024), lambda w, h: scene_night(w, h, 8, aurora=True),
             "A green aurora glows over snow covered peaks in a clear night sky, with a few stars above the ridge."),
        Demo("bokeh_lights", ".jpg", (1152, 768), lambda w, h: scene_bokeh(w, h, 4), None),
        Demo("city_dusk", ".jpg", (1280, 720), lambda w, h: scene_city(w, h, 5),
             "ohwx, city, skyline, buildings, lit windows, dusk, no humans, cityscape", {"exif": "camera"}),
        Demo("desert_dunes", ".png", (960, 1280), lambda w, h: scene_desert(w, h, 6),
             "Golden sand dunes under a hazy orange sun, with long soft shadows across the ridges."),
        Demo("geometric_shapes", ".png", (1024, 1024), lambda w, h: scene_geometric(w, h, 7), None),
        Demo("green_hills", ".jpg", (1280, 853), lambda w, h: scene_hills(w, h, 2),
             "ohwx, hills, grass, blue sky, clouds, scenery, no humans, countryside, daylight"),
        Demo("lake_sunset", ".png", (1344, 768), lambda w, h: scene_sunset_sea(w, h, 1),
             "ohwx, sunset, ocean, scenery, orange sky, reflection, horizon, no humans, calm water", {"png": "prompt"}),
        Demo("misty_peaks", ".jpg", (832, 1216), lambda w, h: scene_mountains(w, h, 9, blue, (0.3, 0.3, 0.2, (210, 225, 255)), blue_ridges, 0.5),
             "Layers of blue mountains fade into cold mist, with a pale sun low in the sky.", {"exif": "camera"}),
        Demo("pine_forest", ".jpg", (1152, 896), lambda w, h: scene_pines(w, h, 10),
             "ohwx, forest, pine trees, fog, dusk, scenery, no humans, silhouette"),
        Demo("starry_night", ".png", (1024, 1280), lambda w, h: scene_night(w, h, 12),
             "ohwx, night sky, stars, moon, mountain, scenery, no humans, silhouette"),
        Demo("wave_bands", ".png", (1024, 768), lambda w, h: scene_waves(w, h, 13),
             "ohwx, abstract, waves, layered, blue and teal, flat color, minimalism", {"png": "prompt"}),
    ]
    made: dict[str, Image.Image] = {}
    for it in items:
        img = it.make(*it.size)
        made[it.name] = img
        path = folder / f"{it.name}{it.ext}"
        if it.ext == ".jpg":
            exif = Image.Exif()
            kind = it.extra.get("exif")
            if kind:
                exif[0x010F], exif[0x0110] = "ExampleCam", "EC-5 Mark II"
                exif[0x0131] = "Example RAW Editor 12.3"
                exif[0x0132] = "2026:05:14 18:42:11"
                if kind == "gps":
                    exif[0xA431] = "0123456789"          # body serial number
                    exif[0xA430] = "Demo Owner"          # camera owner name
                    gps = exif.get_ifd(0x8825)
                    gps[1], gps[2], gps[3], gps[4] = "N", (46.0, 33.0, 12.0), "E", (8.0, 12.0, 45.0)
            img.save(path, quality=92, exif=exif if kind else Image.Exif())
        else:
            info = None
            if it.extra.get("png") == "prompt":
                info = PngImagePlugin.PngInfo()
                info.add_text("parameters", f"{it.caption}\nNegative prompt: blurry, lowres\nSteps: 28, Sampler: Euler a, "
                                            f"CFG scale: 6, Seed: 482913, Size: {it.size[0]}x{it.size[1]}, Model: demo-model")
            img.save(path, pnginfo=info)
        if it.caption:
            path.with_suffix(".txt").write_text(it.caption, encoding="utf-8")
    # health-scan material: an exact copy, a near-duplicate, a soft image and a tiny one
    shutil.copy2(folder / "lake_sunset.png", folder / "lake_sunset_copy.png")
    shutil.copy2(folder / "lake_sunset.txt", folder / "lake_sunset_copy.txt")
    a = made["alpine_dawn"]
    near = a.crop((24, 18, a.width - 20, a.height - 14)).resize((1216, 912), Image.LANCZOS)
    near = Image.eval(near, lambda v: min(255, int(v * 1.03)))
    near.save(folder / "alpine_dawn_crop.jpg", quality=88)
    (folder / "alpine_dawn_crop.txt").write_text("ohwx, mountain, scenery, dawn", encoding="utf-8")
    made["city_dusk"].filter(ImageFilter.GaussianBlur(14)).save(folder / "soft_focus.jpg", quality=90)
    (folder / "soft_focus.txt").write_text("ohwx, city, lights, dusk, out of focus, blurry", encoding="utf-8")
    made["green_hills"].resize((384, 256), Image.LANCZOS).save(folder / "thumb_small.png")
    (folder / "thumb_small.txt").write_text("ohwx, hills, grass, sky, scenery", encoding="utf-8")


def prepare_dataset() -> None:
    if ARGS.dataset:
        src = Path(ARGS.dataset)
        if not src.is_dir():
            sys.exit(f"--dataset folder not found: {src}")
        shutil.copytree(src, DEMO_DIR)
    else:
        build_demo_dataset(DEMO_DIR)


def build_collections() -> None:
    """Fake collections (copies of demo images) so the Datasets tab doesn't show the owner's real ones."""
    from core import dataset
    imgs = [p for p in dataset.scan_images(DEMO_DIR)]
    plan = {"landscapes_v1": imgs[0:9], "night_scenes": imgs[9:13], "abstract_set": imgs[13:16] or imgs[:3]}
    for name, files in plan.items():
        dest = COLL_DIR / name
        dest.mkdir(parents=True, exist_ok=True)
        for p in files:
            shutil.copy2(p, dest / Path(p).name)
            cap = Path(p).with_suffix(".txt")
            if cap.is_file():
                shutil.copy2(cap, dest / cap.name)
    EDITS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------------------------------- Qt bootstrap
prepare_dataset()
build_collections()

from PySide6.QtCore import QPoint, QRect, Qt, QThreadPool  # noqa: E402
from PySide6.QtGui import QPixmap  # noqa: E402
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QGroupBox, QLabel, QLineEdit,  # noqa: E402
                               QListWidget, QPlainTextEdit, QPushButton, QTableWidget, QTabBar, QTextEdit,
                               QTreeWidget, QWidget)

from core import hardware  # noqa: E402
from core.config import settings  # noqa: E402

hardware.describe_environment = lambda: (
    "Python 3.12 on Windows 11\nPyTorch: 2.9 (ROCm)\n  - GPU 0: Example GPU, 24 GB, bf16\n"
    "Transformers: 4.57\nONNX Runtime: 1.22\nONNX providers: CPUExecutionProvider")

cfg = settings()
cfg.update({"paths.collections_dir": str(COLL_DIR), "paths.edits_dir": str(EDITS_DIR),
            "ui.last_folder": str(DEMO_DIR), "ui.recent_folders": [str(DEMO_DIR)],
            "caption.subject": "ohwx", "caption.subject_first": True, "caption.tag_hints": True,
            "ui.caption_thumbnail_size": 110, "ui.editor_thumbnail_size": 110, "ui.metadata_thumbnail_size": 110,
            "ui.datasets_thumbnail_size": 110, "ui.thumbnail_size": 180})

app = QApplication.instance() or QApplication(sys.argv)
app.setApplicationName("TagScribeR")

from tabs.settings import apply_theme  # noqa: E402

apply_theme(settings().get("ui.theme"))

import main as app_main  # noqa: E402
from tabs import icons  # noqa: E402
from tabs.caption import CaptionTab  # noqa: E402
from tabs.workspace.browser import DatasetBrowser  # noqa: E402
from tabs.workspace.context import workspace  # noqa: E402

CaptionTab._list_devices = staticmethod(lambda: [])              # no GPU probing (would import torch)
app_main.hardware.describe_environment = hardware.describe_environment

_orig_status = DatasetBrowser.update_status
SKIP_PATTERNS = ("archa", "ai-stuff", "\\users\\", "/users/")


# ---------------------------------------------------------------------------------------------------- text sanitising
def _replacements() -> list[tuple[re.Pattern, str]]:
    pairs = [(str(DEMO_DIR), NEUTRAL["demo"]), (str(COLL_DIR), NEUTRAL["collections"]),
             (str(EDITS_DIR), NEUTRAL["edits"]), (str(RUNS_DIR), NEUTRAL["runs"]),
             (str(TMP / "user_data" / "training_runs"), NEUTRAL["runs"]),
             (str(TMP), NEUTRAL["root"]), (str(paths.APP_ROOT), NEUTRAL["root"])]
    out = []
    for real, fake in sorted(pairs, key=lambda p: -len(p[0])):
        for variant in {real, real.replace("\\", "/")}:
            out.append((re.compile(re.escape(variant), re.I), fake))
    return out


REPL = _replacements()


GLYPHS = {"✨ ": "", "◌ ": "", "▫ ": "", "✂ ": "", "⟲ ": "", "⟳ ": "", "⇋ ": "",
          "⇵ ": "", "◀": "<", "▶": ">", "✘ ": ""}  # glyphs the offscreen font renders as empty boxes


def clean_text(text: str) -> str:
    for g, r in GLYPHS.items():
        text = text.replace(g, r)
    for pat, fake in REPL:
        text = pat.sub(lambda _m, f=fake: f, text)
    return text


class Sanitizer:
    """Swap machine-specific strings in visible widgets for neutral ones while grabbing; restore afterwards."""

    def __init__(self, root: QWidget):
        self.root, self.undo = root, []

    def _set(self, getter, setter, obj):
        old = getter()
        new = clean_text(old)
        if new != old:
            blocked = obj.blockSignals(True)
            setter(new)
            obj.blockSignals(blocked)
            self.undo.append((obj, setter, old))

    def __enter__(self):
        for w in [self.root] + self.root.findChildren(QWidget):
            if isinstance(w, QLabel) and not w.pixmap():
                self._set(w.text, w.setText, w)
            elif isinstance(w, QLineEdit):
                self._set(w.text, w.setText, w)
                self._set(w.placeholderText, w.setPlaceholderText, w)
            elif isinstance(w, QPlainTextEdit):
                self._set(w.toPlainText, w.setPlainText, w)
            elif isinstance(w, QGroupBox):
                self._set(w.title, w.setTitle, w)
            elif isinstance(w, QPushButton):
                name, rest = icons.split_emoji(w.text())
                if name:                                  # emoji label set after apply_icons ran: show the icon instead
                    old = w.text()
                    icons.set(w, name, "#ffffff" if "background-color" in w.styleSheet() else icons.DEFAULT_COLOR)
                    w.setText(rest)
                    self.undo.append((w, w.setText, old))
                self._set(w.text, w.setText, w)
            elif isinstance(w, QComboBox):
                for i in range(w.count()):
                    old = w.itemText(i)
                    if clean_text(old) != old:
                        w.setItemText(i, clean_text(old))
                        self.undo.append((w, lambda t, i=i, w=w: w.setItemText(i, t), old))
            elif isinstance(w, QListWidget):
                for i in range(w.count()):
                    it = w.item(i)
                    old = it.text()
                    if clean_text(old) != old:
                        it.setText(clean_text(old))
                        self.undo.append((it, it.setText, old))
            elif isinstance(w, QTableWidget):
                for r in range(w.rowCount()):
                    for c in range(w.columnCount()):
                        it = w.item(r, c)
                        if it is not None and clean_text(it.text()) != it.text():
                            old = it.text()
                            it.setText(clean_text(old))
                            self.undo.append((it, it.setText, old))
            elif isinstance(w, QTreeWidget):
                stack = [w.topLevelItem(i) for i in range(w.topLevelItemCount())]
                while stack:
                    it = stack.pop()
                    old = it.text(0)
                    if clean_text(old) != old:
                        it.setText(0, clean_text(old))
                        self.undo.append((it, lambda t, it=it: it.setText(0, t), old))
                    stack += [it.child(i) for i in range(it.childCount())]
        return self

    def __exit__(self, *_):
        for obj, setter, old in reversed(self.undo):
            try:
                if isinstance(obj, QWidget):
                    blocked = obj.blockSignals(True)
                    setter(old)
                    obj.blockSignals(blocked)
                else:
                    setter(old)
            except RuntimeError:
                pass


def visible_texts(root: QWidget) -> list[str]:
    out = []
    for w in [root] + root.findChildren(QWidget):
        if not w.isVisible():
            continue
        if isinstance(w, QLabel):
            out.append(w.text())
        elif isinstance(w, QLineEdit):
            out += [w.text(), w.placeholderText()]
        elif isinstance(w, (QPlainTextEdit, QTextEdit)):
            out.append(w.toPlainText())
        elif isinstance(w, QPushButton):
            out.append(w.text())
        elif isinstance(w, QComboBox):
            out += [w.itemText(i) for i in range(w.count())] if w.view().isVisible() else [w.currentText()]
        elif isinstance(w, QListWidget):
            out += [w.item(i).text() for i in range(w.count())]
        elif isinstance(w, QTableWidget):
            out += [w.item(r, c).text() for r in range(w.rowCount()) for c in range(w.columnCount())
                    if w.item(r, c) is not None]
        elif isinstance(w, QTreeWidget):
            stack = [w.topLevelItem(i) for i in range(w.topLevelItemCount())]
            while stack:
                it = stack.pop()
                out.append(it.text(0))
                stack += [it.child(i) for i in range(it.childCount())]
    return out


def leaks(root: QWidget) -> list[str]:
    bad = []
    tmp = str(TMP).lower()
    for t in visible_texts(root):
        low = t.lower()
        if any(p in low for p in SKIP_PATTERNS) or tmp in low:
            bad.append(t[:120])
    return bad


# ---------------------------------------------------------------------------------------------------- helpers
class Ctx:
    def __init__(self):
        self.window = app_main.MainWindow()
        icons.apply_icons(self.window)
        self.window.resize(1600, 950)
        self.window.show()
        self.pump(0.3)
        self.ws = workspace()

    # -- event loop
    def pump(self, secs: float = 0.3):
        end = time.time() + secs
        while time.time() < end:
            app.processEvents()
            time.sleep(0.01)

    def until(self, cond: Callable[[], bool], timeout: float = 20.0, step: float = 0.05) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            app.processEvents()
            try:
                if cond():
                    return True
            except RuntimeError:
                pass
            time.sleep(step)
        return False

    def idle(self, secs: float = 0.6):
        """Let thumbnails / previews / background jobs finish."""
        self.pump(0.15)
        for _ in range(200):
            app.processEvents()
            busy = bool(self.ws.loader._pending) or QThreadPool.globalInstance().activeThreadCount() > 0
            if not busy:
                break
            time.sleep(0.03)
        self.pump(secs)

    def go(self, index: int):
        self.window.sidebar.setCurrentRow(index)
        self.pump(0.25)

    def open_dataset(self):
        self.ws.open_folder(str(DEMO_DIR), False, self.window)
        self.pump(0.3)
        self.keys = [e.key for e in self.ws.session.entries]
        self.idle()

    def key(self, name_or_index) -> str:
        if isinstance(name_or_index, int):
            return self.keys[min(name_or_index, len(self.keys) - 1)]
        for k in self.keys:
            if Path(k).stem == name_or_index:
                return k
        return self.keys[0]

    def tab(self, i):
        return (self.window.tab_gallery, self.window.tab_caption, self.window.tab_editor, self.window.tab_datasets,
                self.window.tab_metadata, self.window.tab_train, self.window.tab_settings)[i]


class _DemoMemoryReader:
    latest = ((int(16.4 * 2 ** 30), 24 * 2 ** 30), (int(19.2 * 2 ** 30), 64 * 2 ** 30))
    samples = 1

    def start(self):
        pass

    def stop(self):
        pass


def make_neutral_status(c: Ctx) -> None:
    """Hide the dataset path in every browser's status line (and the Train tab's Folder label)."""
    def patched(self):
        _orig_status(self)
        self.lbl_status.setText(clean_text(self.lbl_status.text()))
    DatasetBrowser.update_status = patched
    for t in (c.window.tab_gallery, c.window.tab_caption, c.window.tab_editor, c.window.tab_datasets,
              c.window.tab_metadata):
        t.browser.update_status()
    c.window.tab_train._update_dataset_label()
    c.window.tab_train.lbl_dataset.setText(clean_text(c.window.tab_train.lbl_dataset.text()))
    # The memory bar would show this machine's real VRAM / RAM: give it fixed demo readings instead.
    bar = c.window.memory_strip.bar
    bar.shutdown()
    bar.reader = _DemoMemoryReader()
    bar.vram.peak, bar.ram.peak = int(17.8 * 2 ** 30), int(21.5 * 2 ** 30)
    bar.poll()


# ---------------------------------------------------------------------------------------------------- scenes and annotation
class Scene:
    """What gets grabbed: a widget, optionally with a dialog/popup composited on top of a dimmed window."""

    def __init__(self, base: QWidget, overlay: QWidget | None = None, dim: float = 0.0):
        self.base, self.overlay, self.dim = base, overlay, dim

    def _grab(self, w: QWidget) -> Image.Image:
        pm = w.grab()
        tmp = TMP / "_grab.png"
        pm.save(str(tmp))
        return Image.open(tmp).convert("RGB").copy()

    def render(self) -> Image.Image:
        img = self._grab(self.base)
        self.scale = img.width / max(1, self.base.width())
        if self.overlay is not None:
            if self.dim:
                img = Image.blend(img, Image.new("RGB", img.size, (0, 0, 0)), self.dim)
            ov = self._grab(self.overlay)
            off = self.overlay.mapToGlobal(QPoint(0, 0)) - self.base.mapToGlobal(QPoint(0, 0))
            self.offset = QPoint(int(off.x() * self.scale), int(off.y() * self.scale))
            img.paste(ov, (self.offset.x(), self.offset.y()))
        return img

    def rect(self, target) -> QRect | None:
        """Pixel rect (in the final image) of a widget / list of widgets / (widget, QRect-in-widget) / QRect."""
        if isinstance(target, QRect):
            return target
        if isinstance(target, (list, tuple)) and target and isinstance(target[0], tuple):
            rs = [self.rect(t) for t in target]
        elif isinstance(target, (list, tuple)) and len(target) == 2 and isinstance(target[1], QRect):
            w, local = target
            return self._rect_in(w, local)
        elif isinstance(target, (list, tuple)):
            rs = [self.rect(t) for t in target]
        else:
            return self._rect_in(target, None)
        rs = [r for r in rs if r is not None]
        if not rs:
            return None
        out = rs[0]
        for r in rs[1:]:
            out = out.united(r)
        return out

    def _rect_in(self, w: QWidget, local: QRect | None) -> QRect | None:
        vis = w.visibleRegion().boundingRect() if local is None else local
        if vis.isEmpty():
            return None
        in_overlay = self.overlay is not None and (w is self.overlay or self.overlay.isAncestorOf(w))
        origin = w.mapTo(self.overlay if in_overlay else self.base, QPoint(0, 0))
        r = QRect(origin + vis.topLeft(), vis.size())
        if in_overlay:
            r.translate(int(self.offset.x() / self.scale), int(self.offset.y() / self.scale))
        s = self.scale
        return QRect(int(r.x() * s), int(r.y() * s), int(r.width() * s), int(r.height() * s))


def item_rect(view, item) -> QRect:
    """Rect of a QTreeWidget/QListWidget item in its viewport, widened to the whole row, as (view, local) target."""
    r = view.visualItemRect(item)
    r = QRect(0, r.y(), view.viewport().width(), r.height())
    return (view.viewport(), r)


def row_rect(table: QTableWidget, row0: int, row1: int | None = None):
    row1 = row0 if row1 is None else row1
    top = table.rowViewportPosition(row0)
    bottom = table.rowViewportPosition(row1) + table.rowHeight(row1)
    return (table.viewport(), QRect(0, top, table.viewport().width(), bottom - top))


@dataclass
class Note:
    target: object                    # widget | list | (widget, QRect) | callable(ctx) -> any of those
    text: str
    pos: tuple                        # (fx, fy): centre of the callout box as a fraction of the image


@dataclass
class Shot:
    filename: str
    setup: Callable[[Ctx], Scene]
    notes: list[Note]
    after: Callable[[Ctx], None] | None = None


RED = (224, 32, 32)
FILL = (30, 30, 30)
FONT_CANDIDATES = [r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def load_font(px: int) -> ImageFont.FreeTypeFont:
    for f in FONT_CANDIDATES:
        if os.path.exists(f):
            return ImageFont.truetype(f, px)
    return ImageFont.load_default()


def nearest_on_rect(rect: QRect, x: float, y: float) -> tuple[float, float]:
    return (min(max(x, rect.left()), rect.right()), min(max(y, rect.top()), rect.bottom()))


def box_exit(box: tuple, tx: float, ty: float) -> tuple[float, float]:
    """Point where the segment from the box centre to (tx, ty) leaves the box."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx, cy
    sx = ((x1 - cx) / abs(dx)) if dx else math.inf
    sy = ((y1 - cy) / abs(dy)) if dy else math.inf
    s = min(sx, sy, 1.0)
    return cx + dx * s, cy + dy * s


def annotate(img: Image.Image, scene: Scene, notes: list[Note], warn: Callable[[str], None]) -> Image.Image:
    d = ImageDraw.Draw(img)
    s = img.width / 1600
    font_px = max(15, round(19 * min(1.0, max(0.85, s))))
    font = load_font(font_px)
    border, line_w, pad = 3, 3, 12
    boxes, targets = [], []
    for n in notes:
        target = n.target(scene) if callable(n.target) else n.target
        r = scene.rect(target)
        if r is None or r.isEmpty():
            warn(f"annotation target not visible: {n.text[:30]!r}")
            continue
        targets.append((n, r))
    placed = []
    for n, r in targets:
        lines = textwrap.wrap(n.text, 40)
        widths = [d.textlength(l, font=font) for l in lines]
        lh = font_px + 5
        bw, bh = max(widths) + 2 * pad, lh * len(lines) + 2 * pad - 5
        cx, cy = n.pos[0] * img.width, n.pos[1] * img.height
        x0 = min(max(10, cx - bw / 2), img.width - bw - 10)
        y0 = min(max(10, cy - bh / 2), img.height - bh - 10)
        placed.append((n, r, lines, (x0, y0, x0 + bw, y0 + bh), lh))
    for n, r, lines, box, lh in placed:                       # outlines first, so callouts sit above them
        pad_r = 4
        d.rectangle([r.left() - pad_r, r.top() - pad_r, r.right() + pad_r, r.bottom() + pad_r], outline=RED, width=3)
    for n, r, lines, box, lh in placed:
        outer = QRect(r.left() - 4, r.top() - 4, r.width() + 8, r.height() + 8)
        bx = QRect(int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1]))
        for other_n, other_r, _l, other_box, _lh in placed:
            ob = QRect(int(other_box[0]), int(other_box[1]), int(other_box[2] - other_box[0]),
                       int(other_box[3] - other_box[1]))
            if other_n is n:
                if bx.intersects(outer):
                    warn(f"callout covers its own target: {n.text[:30]!r}")
            elif bx.intersects(ob):
                warn(f"callouts overlap: {n.text[:25]!r} / {other_n.text[:25]!r}")
        # connector
        bcx, bcy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        tx, ty = nearest_on_rect(outer, bcx, bcy)
        ex, ey = box_exit(box, tx, ty)
        d.line([(ex, ey), (tx, ty)], fill=RED, width=line_w)
        d.ellipse([tx - 5, ty - 5, tx + 5, ty + 5], fill=RED)
        d.rectangle(box, fill=FILL, outline=RED, width=border)
        y = box[1] + pad - 2
        for l in lines:
            d.text((box[0] + pad, y), l, font=font, fill=(255, 255, 255))
            y += lh
    return img


# ---------------------------------------------------------------------------------------------------- shots
def wait_preview(c: Ctx, label: QLabel, timeout=10):
    c.until(lambda: label.pixmap() is not None and not label.pixmap().isNull(), timeout)


def select(c: Ctx, tab, names):
    keys = [c.key(n) for n in names]
    tab.browser.select_keys(keys)
    c.pump(0.3)
    return keys


def shot_gallery(c: Ctx) -> Scene:
    c.go(0)
    t = c.window.tab_gallery
    t.browser.slider.setValue(180)
    t.browser.set_filter("tag:scenery res:>=800")
    c.pump(0.6)
    t.side.setCurrentIndex(0)
    select(c, t, ["green_hills"])
    wait_preview(c, t.inspector.preview)
    c.idle(0.8)
    return Scene(c.window)


def shot_gallery_batch(c: Ctx) -> Scene:
    c.go(0)
    t = c.window.tab_gallery
    t.browser.slider.setValue(120)
    t.browser.set_filter("")
    c.pump(0.4)
    select(c, t, ["alpine_dawn", "city_dusk", "green_hills", "lake_sunset", "pine_forest"])
    t.side.setCurrentIndex(1)
    b = t.batch
    b.inp_add.setText("ohwx style")
    b.inp_remove.setText("blurry")
    b.inp_old.setText("dusk")
    b.inp_new.setText("twilight")
    b.inp_find.setText("mountain")
    b.inp_repl.setText("mountains")
    b.inp_prefix.setText("ohwx, ")
    c.idle(0.6)
    return Scene(c.window)


def shot_tag_stats(c: Ctx) -> Scene:
    c.go(0)
    t = c.window.tab_gallery
    t.browser.slider.setValue(120)
    t.browser.set_filter("")
    t.browser.view.clearSelection()
    t.side.setCurrentIndex(2)
    c.pump(1.0)
    t._refresh_tag_stats()
    c.idle(0.6)
    return Scene(c.window)


def shot_health(c: Ctx) -> Scene:
    c.go(0)
    t = c.window.tab_gallery
    t.browser.slider.setValue(120)
    t.browser.set_filter("")
    t.side.setCurrentIndex(3)
    t.scan_health()
    c.until(lambda: t.health_job is None and t.health.tree.topLevelItemCount() > 0, 40)
    t.health.tree.expandAll()
    c.pump(0.5)
    dup = next((t.health.tree.topLevelItem(i) for i in range(t.health.tree.topLevelItemCount())
                if "duplicate" in t.health.tree.topLevelItem(i).text(0).lower()), None)
    if dup is not None and dup.childCount():
        t.health._clicked(dup.child(0), 0)          # same as clicking a group: selects its images in the grid
    c.idle(0.8)
    return Scene(c.window)


def shot_caption(c: Ctx) -> Scene:
    c.go(1)
    t = c.window.tab_caption
    t.tab_source.setCurrentIndex(0)
    t.log_box.clear()
    t.sec_subject.toggle.setChecked(True)
    t.inp_subject.setText("ohwx")
    t.chk_subject_first.setChecked(True)
    t.chk_tag_hints.setChecked(True)
    t.sec_system.toggle.setChecked(False)
    select(c, t, ["city_dusk", "geometric_shapes", "bokeh_lights", "wave_bands"])
    t.lbl_model_info.setText("Qwen3-VL family. Good all-round captioner; follows long instructions well.")
    t.lbl_loaded.setText("No model loaded")
    c.idle(0.6)
    return Scene(c.window)


def shot_caption_api(c: Ctx) -> Scene:
    c.go(1)
    t = c.window.tab_caption
    t.tab_source.setCurrentIndex(1)
    t.combo_presets.blockSignals(True)
    t.combo_presets.clear()
    t.combo_presets.addItems(["Select profile…", "LM Studio (local)", "Ollama (local)", "Cloud API"])
    t.combo_presets.setCurrentIndex(1)
    t.combo_presets.blockSignals(False)
    t.inp_api_url.setText("http://127.0.0.1:1234/v1")
    t.inp_api_model.setEditText("joycaption-beta-one-gguf")
    t.spin_concurrency.setValue(2)
    t.log_box.clear()
    select(c, t, ["city_dusk", "geometric_shapes", "bokeh_lights", "wave_bands"])
    c.idle(0.5)
    return Scene(c.window)


def shot_editor(c: Ctx) -> Scene:
    c.go(2)
    t = c.window.tab_editor
    select(c, t, ["alpine_dawn", "green_hills", "lake_sunset"])
    t.browser.view.setCurrentIndex(t.browser.proxy.index(0, 0))
    t.combo_aspect.setCurrentIndex(t.combo_aspect.findText("1:1") if t.combo_aspect.findText("1:1") > 0 else 2)
    t.combo_focus.setCurrentIndex(0)
    t._update_output_label()
    t._schedule_preview()
    c.until(lambda: t.lbl_after.pixmap() is not None and not t.lbl_after.pixmap().isNull(), 10)
    c.idle(0.6)
    return Scene(c.window)


def shot_datasets(c: Ctx) -> Scene:
    c.go(3)
    t = c.window.tab_datasets
    t.browser.view.selectAll()
    t.list_datasets.setCurrentRow(0)
    c.pump(0.8)
    from tabs.export_dialog import ExportDialog
    dlg = ExportDialog(t.browser.shown_keys(), "ohwx", c.window)
    dlg.inp_out.setText(r"D:\Datasets\export\demo_training")
    dlg.resize(560, 760)
    icons.apply_icons(dlg)
    dlg.setWindowFlag(Qt.Dialog, True)
    dlg.show()
    dlg.move(c.window.mapToGlobal(QPoint(215, 60)))
    c.keep = dlg
    c.idle(0.6)
    return Scene(c.window, dlg, dim=0.45)


def shot_metadata(c: Ctx) -> Scene:
    c.go(4)
    t = c.window.tab_metadata
    t.browser.set_filter("")
    keys = select(c, t, ["alpine_dawn", "city_dusk", "lake_sunset", "misty_peaks", "wave_bands", "green_hills"])
    t.side.setCurrentIndex(1)
    t.rad_clean_shown.setChecked(True)
    t.audit()
    c.until(lambda: "●" in t.lbl_audit.text(), 15)
    c.idle(0.6)
    return Scene(c.window)


def _fill_model_rows(t):
    """Neutral example paths for the required model files, named after each file's label (works for any family)."""
    for f in t.desc.model_files:
        edit = t.model_edits[f.pref_key]
        if f.required:
            slug = re.sub(r"[^a-z0-9]+", "_", f.label.lower()).strip("_")
            edit.setText(rf"D:\Models\{slug}.safetensors")


def shot_train_setup(c: Ctx) -> Scene:
    c.go(5)
    t = c.window.tab_train
    _fill_model_rows(t)
    t.inp_trigger.setText("ohwx")
    t._set_widget("LORA_NAME", "my_lora")
    t.sections["Training Parameters"][0].toggle.setChecked(True)
    t.sections["Output"][0].toggle.setChecked(False)
    c.pump(0.5)
    from PySide6.QtWidgets import QScrollArea
    sa = next(a for a in t.findChildren(QScrollArea) if a.isAncestorOf(t.sections["Output"][0]))
    sa.verticalScrollBar().setValue(150)
    c.idle(0.4)
    return Scene(c.window)


# A run the real AdaptiveLR would produce for min 2e-4 / max 4e-4: start at the geometric middle, probe up x1.25
# after two improving epochs (capped at max), halve after a two-epoch plateau. Lines use the trainer's own formats
# (training/train.py, training/adaptive_lr.py).
EPOCH_LOSS = [0.2143, 0.1862, 0.1704, 0.1612, 0.1471, 0.1489, 0.1493, 0.1398]
EPOCH_LR = [2.83e-4, 2.83e-4, 2.83e-4, 3.54e-4, 3.54e-4, 4.00e-4, 4.00e-4, 2.00e-4]
ADAPTIVE = {1: (None, "ARMED"), 2: ("+9%", "HOLD (loss improving, streak 1/2)"),
            3: ("+8%", "PROBE UP (loss improving, streak 2)"), 4: ("+7%", "HOLD (loss improving, streak 1/2)"),
            5: ("+6%", "PROBE UP (loss improving, streak 2)"), 6: ("+6%", "HOLD (loss plateau, streak 1/2)"),
            7: ("+5%", "REDUCE (loss plateau, streak 2)"), 8: ("+3%", "HOLD (loss improving, streak 1/2)")}


def feed(t, line: str) -> None:
    """Same path as TrainTab._read for one console line."""
    u = t.tracker.consume(line)
    if u is None or u["kind"] != "training":
        t._console(line)
    t._on_update(u)


def fake_run(c: Ctx, name="my_lora", total_epochs=30, stage=1):
    run_dir = RUNS_DIR / name
    (run_dir / "sample").mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(run_dir=run_dir, output_name=name, total_epochs=total_epochs,
                           stages=[("Cache", ["x"]), ("Training", ["x"])]), stage


def make_samples(c: Ctx, run_dir: Path) -> None:
    src = [Path(k) for k in c.keys[:16]]
    names = ["alpine_dawn", "lake_sunset", "green_hills", "pine_forest", "starry_night", "desert_dunes", "misty_peaks",
             "city_dusk"]
    pics = [next((p for p in src if p.stem == n), src[i % len(src)]) for i, n in enumerate(names)]
    base = time.time() - 3600
    i = 0
    for e_i, epoch in enumerate((2, 4, 6, 8)):
        for idx in (0, 1):
            im = Image.open(pics[(e_i * 2 + idx) % len(pics)]).convert("RGB")
            side = min(im.size)
            im = im.crop(((im.width - side) // 2, (im.height - side) // 2, (im.width + side) // 2, (im.height + side) // 2))
            im = im.resize((640, 640), Image.LANCZOS)
            p = run_dir / "sample" / f"my_lora_e{epoch:06d}_{idx:02d}_2026100312{epoch:02d}00_{1234 + idx}.png"
            im.save(p)
            os.utime(p, (base + i * 60, base + i * 60))
            i += 1


def shot_train_running(c: Ctx) -> Scene:
    c.go(5)
    t = c.window.tab_train
    _fill_model_rows(t)
    t.inp_trigger.setText("ohwx")
    for g in ("Training Parameters", "Output"):
        t.sections[g][0].toggle.setChecked(True)
    run, stage = fake_run(c)
    make_samples(c, run.run_dir)
    t.run, t.stage_index, t._run_ctx = run, stage, (t.desc, t.collect(), str(DEMO_DIR))
    t.tracker.reset(30)
    t.chart.clear()
    t.console.clear()
    t._set_state("running")
    t._console("[check] 16 captioned image(s).")
    t._console(f"=== {t.desc.display_name}: my_lora -> {NEUTRAL['runs']}\\my_lora")
    t._console("--- Caching latents")
    for n in (4, 8, 12, 16):
        feed(t, f"[cache] latents {n}/16")
    t._console("--- Caching text")
    for n in (8, 16):
        feed(t, f"[cache] text {n}/16")
    t._console("--- Training")
    steps_per_epoch = 20
    for epoch in range(1, 9):
        if epoch in (2, 4, 6, 8):
            pass
        step = epoch * steps_per_epoch
        feed(t, f"steps: {step * 100 // 600:3d}%|{'#' * (step // 24):<25}| {step}/600 [{step * 3 // 60:02d}:{step * 3 % 60:02d}"
                f"<{(600 - step) * 3 // 60:02d}:{(600 - step) * 3 % 60:02d},  3.01s/it, avr_loss={EPOCH_LOSS[epoch - 1]:.4f}]")
        feed(t, f"epoch {epoch}/30  avr_loss={EPOCH_LOSS[epoch - 1]:.4f}  step={step}  3.01s/step  "
                f"lr={EPOCH_LR[epoch - 1]:.3e}  peak VRAM 16.1 GB")
        growth, action = ADAPTIVE[epoch]
        cur = EPOCH_LR[epoch - 1]
        nxt = EPOCH_LR[epoch] if epoch < 8 else cur
        lr_str = f"{cur:.2e}" if abs(nxt - cur) < 1e-12 else f"{cur:.2e}->{nxt:.2e}"
        if growth is None:
            feed(t, f"[adaptive_lr] epoch 1: loss={EPOCH_LOSS[0]:.4f} lr={cur:.2e} clip=0% | ARMED")
        else:
            feed(t, f"[adaptive_lr] epoch {epoch}: loss={EPOCH_LOSS[epoch - 1]:.4f} lr={lr_str} clip=0% "
                    f"wnorm_\u0394={growth} | {action}")
        if epoch in (2, 4, 6, 8):
            feed(t, f"[save] {NEUTRAL['runs']}\\my_lora\\my_lora-{epoch:06d}.safetensors")
            feed(t, f"[sample] epoch {epoch}: rendering 2 preview(s)")
            feed(t, f"[sample] epoch {epoch}: 2 preview(s) -> {NEUTRAL['runs']}\\my_lora\\sample")
    step = 8 * steps_per_epoch + 11
    feed(t, f"steps:  {step * 100 // 600}%|{'#' * (step // 24):<25}| {step}/600 [{step * 3 // 60:02d}:{step * 3 % 60:02d}"
            f"<{(600 - step) * 3 // 60:02d}:{(600 - step) * 3 % 60:02d},  3.02s/it, avr_loss=0.1259]")
    _show_run_tab(t, 0)
    c.idle(0.8)
    return Scene(c.window)


def _show_run_tab(t, index: int):
    from PySide6.QtWidgets import QTabWidget
    for tabs in t.findChildren(QTabWidget):
        if tabs.indexOf(t.chart) >= 0:
            tabs.setCurrentIndex(index)
            return tabs


def reset_train(c: Ctx):
    t = c.window.tab_train
    t.run, t.stage_index = None, 0
    t._set_state("idle")
    t.lbl_status.setText("Idle")
    t.progress.setRange(0, 1)
    t.progress.setValue(0)


def clear_health(c: Ctx):
    for e in c.ws.session.entries:
        e.flags = set()
    c.ws.model.refresh([e.key for e in c.ws.session.entries])
    c.ws.health_report = None


def shot_train_samples(c: Ctx) -> Scene:
    c.go(5)
    t = c.window.tab_train
    _fill_model_rows(t)
    t.inp_trigger.setText("ohwx")
    run, stage = fake_run(c)
    make_samples(c, run.run_dir)
    t.run, t.stage_index = run, stage
    t._set_state("idle")
    t.console.clear()
    t._console("[sample] epoch 8: 2 preview(s) -> " + NEUTRAL["runs"] + "\\my_lora\\sample")
    t.samples.clear()
    t._refresh_samples()
    _show_run_tab(t, 1)
    c.idle(1.0)
    t._refresh_samples()
    c.idle(0.6)
    return Scene(c.window)


def shot_problem_images(c: Ctx) -> Scene:
    import json
    c.go(5)
    run, _ = fake_run(c)
    (run.run_dir / "loss_log").mkdir(parents=True, exist_ok=True)
    stems = [Path(k).stem for k in c.keys]
    wanted = ["city_dusk", "misty_peaks", "desert_dunes", "bokeh_lights", "pine_forest", "starry_night", "green_hills",
              "lake_sunset", "alpine_dawn", "aurora_peaks"]
    stems = [s for s in wanted if s in stems] + [s for s in stems if s not in wanted]
    spec = [("stuck", 0.50, 0.0312), ("suspect", 0.70, 0.0187), ("exhausted", 0.80, 0.0091), ("learning", 1.00, 0.0042),
            ("mid", 1.00, 0.0003), ("mid", 1.00, -0.0011), ("easy", 1.00, -0.0054), ("easy", 1.00, -0.0102),
            ("easy", 1.00, -0.0133), ("easy", 1.00, -0.0149)]
    images = {s: {"verdict": v, "multiplier": m, "mean_residual": r} for s, (v, m, r) in zip(stems, spec)}
    (run.run_dir / "loss_log" / "problem_images.json").write_text(
        json.dumps({"epoch": 6, "images": images, "plateaued": False}), encoding="utf-8")
    # one clearly weak caption on the stuck image: that is what the dialog is for
    first = Path(c.key(stems[0]))
    first.with_suffix(".txt").write_text("ohwx, photo", encoding="utf-8")
    from tabs.train_dialogs import ProblemImagesDialog
    dlg = ProblemImagesDialog(str(run.run_dir), ".txt", str(DEMO_DIR), c.window)
    icons.apply_icons(dlg)
    dlg.resize(1100, 680)
    dlg.timer.stop()
    dlg.show()
    c.pump(0.4)
    items = dlg.table.findItems(stems[0], Qt.MatchExactly)
    dlg.stuck_row = items[0].row() if items else 0
    dlg.table.selectRow(dlg.stuck_row)
    c.until(lambda: dlg.preview.pixmap() is not None and not dlg.preview.pixmap().isNull(), 10)
    c.keep = dlg
    c.idle(0.6)
    return Scene(dlg)


def shot_palette(c: Ctx) -> Scene:
    c.go(0)
    c.window.tab_gallery.side.setCurrentIndex(0)
    c.window.tab_gallery.browser.slider.setValue(180)
    c.window.tab_gallery.browser.set_filter("")
    c.window.tab_gallery.browser.select_keys([c.key("green_hills")])
    c.pump(0.5)
    from tabs.palette import CommandPalette
    p = CommandPalette(c.window.palette_commands(), c.window)
    p.show()
    p.input.setText("health")
    c.pump(0.4)
    c.keep = p
    return Scene(c.window, p, dim=0.4)


def shot_help(c: Ctx) -> Scene:
    from tabs.help import HelpDialog
    HelpDialog._instance = None
    dlg = HelpDialog(c.window)
    dlg.resize(1100, 760)
    dlg.show()
    dlg.show_topic("train")
    title, grp, body = dlg.topics["train"]
    dlg.topics["train"] = (title, grp, body.replace("✨ ", ""))     # sparkle glyph renders as a box offscreen
    dlg._render("train")
    c.pump(0.8)
    c.keep = dlg
    return Scene(dlg)


def shot_settings(c: Ctx) -> Scene:
    c.window.resize(1600, 1300)
    c.go(6)
    t = c.window.tab_settings
    t.txt_env.setPlainText(hardware.describe_environment())
    t.list_dirs.clear()
    t.list_dirs.addItems([r"D:\Models\vision"])
    c.pump(0.6)
    return Scene(c.window)


def restore_size(c: Ctx):
    c.window.resize(1600, 950)


def fix_settings_hint(c: Ctx):
    """The Model-folders hint lists real folders (app folder, Stability Matrix); show neutral ones."""
    for lbl in c.window.tab_settings.findChildren(QLabel):
        if lbl.text().startswith("TagScribeR scans these folders"):
            lbl.setText("TagScribeR scans these folders (and subfolders) for Hugging Face vision models. "
                        r"Always included: D:\TagScribeR\models")


# ---- notes --------------------------------------------------------------------------------------------------
def _w(c_attr: str):
    """target helper: attribute path on the main window, e.g. 'tab_gallery.browser.inp_filter'."""
    def get(_scene):
        obj = WINDOW
        for part in c_attr.split("."):
            obj = getattr(obj, part)
        return obj
    return get


WINDOW = None  # set once the window exists

SHOTS: list[Shot] = []


def build_shots() -> None:
    g = lambda p: _w(p)  # noqa: E731

    def first_card(tabname):
        def get(scene):
            b = getattr(WINDOW, tabname).browser
            return (b.view.viewport(), b.view.visualRect(b.proxy.index(0, 0)))
        return get

    def selected_card(tabname):
        def get(scene):
            b = getattr(WINDOW, tabname).browser
            idx = b.view.selectionModel().selectedIndexes()
            return (b.view.viewport(), b.view.visualRect(idx[0]))
        return get

    def bounding_cards(tabname, n):
        def get(scene):
            b = getattr(WINDOW, tabname).browser
            sel = b.view.selectionModel().selectedIndexes()[:n]
            r = b.view.visualRect(sel[0])
            for i in sel[1:]:
                r = r.united(b.view.visualRect(i))
            return (b.view.viewport(), r)
        return get

    S = SHOTS.append
    TG = "tab_gallery"
    S(Shot("01_gallery.png", shot_gallery, [
        Note(g("tab_gallery.browser.inp_filter"),
             "Type a filter to narrow the grid: tags, missing captions, size, aspect ratio. Here: has the scenery tag and a "
             "shorter side of 800 px or more.", (0.17, 0.5)),
        Note(selected_card(TG),
             "Cards show the thumbnail, file name and caption. Images without a caption get a red NO CAPTION badge.",
             (0.5, 0.5)),
        Note(g("tab_gallery.inspector.editor"),
             "Edit the caption of the selected image. Changes stay unsaved and undoable until you press Save.", (0.5, 0.72)),
        Note(g("tab_gallery.inspector.tag_editor"),
             "The same caption as tag bubbles: click one to edit it, drag to reorder, X to remove.", (0.5, 0.9)),
    ]))
    S(Shot("02_gallery_batch.png", shot_gallery_batch, [
        Note(lambda s: [WINDOW.tab_gallery.batch.rad_selected, WINDOW.tab_gallery.batch.rad_shown],
             "Batch tools apply to the selected images or to everything the filter shows.", (0.3, 0.64)),
        Note(lambda s: WINDOW.tab_gallery.batch.findChildren(QGroupBox)[0],
             "Add, remove or replace tags across the whole selection in one step.", (0.3, 0.77)),
        Note(lambda s: WINDOW.tab_gallery.batch.findChildren(QGroupBox)[1],
             "Find and replace text, or add a prefix and suffix such as a trigger word.", (0.3, 0.9)),
    ]))
    S(Shot("02b_tag_stats.png", shot_tag_stats, [
        Note(g("tab_gallery.tags.table"),
             "Every tag with how many images use it. Right-click to rename, merge or delete a tag in all captions at once.",
             (0.3, 0.8)),
        Note(g("tab_gallery.tags.combo_scope"),
             "Count over the whole folder, only the shown images, or the selection.", (0.3, 0.62)),
    ]))
    S(Shot("03_health.png", shot_health, [
        Note(lambda s: WINDOW.tab_gallery.health.tree,
             "The Health scan lists exact duplicates, near-duplicates, blurry and low-resolution images. Click a result to "
             "select those images in the grid.", (0.3, 0.64)),
        Note(lambda s: WINDOW.tab_gallery.health.btn_scan,
             "Runs on your CPU and caches its results, so a second scan is fast.", (0.3, 0.8)),
        Note(lambda s: WINDOW.tab_gallery.health.table,
             "Shows which aspect-ratio buckets your images fall into at the chosen training size.", (0.3, 0.92)),
    ], after=clear_health))
    S(Shot("04_auto_caption.png", shot_caption, [
        Note(g("tab_caption.tab_source"),
             "Pick a local vision model, or switch to the API / Server tab. Models that are not installed can be downloaded "
             "from here.", (0.3, 0.63)),
        Note(g("tab_caption.combo_template"),
             "Instruction presets for tags, short or long captions and more. Edit the text freely and save your own.",
             (0.3, 0.77)),
        Note(g("tab_caption.sec_subject"),
             "Subject and trigger word: the model uses it instead of a generic noun, and it can be forced to the start "
             "of every caption.", (0.3, 0.91)),
    ]))
    S(Shot("05_auto_caption_api.png", shot_caption_api, [
        Note(lambda s: WINDOW.tab_caption.tab_source.tabBar(),
             "Use any OpenAI-compatible server: LM Studio, Ollama, llama.cpp, vLLM, or a cloud API.", (0.3, 0.63)),
        Note(g("tab_caption.inp_api_url"),
             "Server address and key. A key is stored in Windows Credential Manager, not in a file.", (0.3, 0.77)),
        Note(g("tab_caption.btn_fetch_models"),
             "List asks the server which models it offers and tests the connection.", (0.3, 0.91)),
    ]))
    S(Shot("06_image_editor.png", shot_editor, [
        Note(lambda s: [WINDOW.tab_editor.lbl_before, WINDOW.tab_editor.lbl_after],
             "Live before and after preview of the operation you are adjusting.", (0.3, 0.63)),
        Note(g("tab_editor.rad_copies"),
             "Edited copies are saved next to their captions, so originals stay untouched by default.", (0.3, 0.77)),
        Note(g("tab_editor.combo_resize"),
             "Resize, crop, rotate, flip or convert the format for the whole selection at once.", (0.3, 0.91)),
    ]))
    S(Shot("07_datasets_export.png", shot_datasets, [
        Note(lambda s: s.overlay.chk_kohya,
             "Optional kohya-style folder name such as 10_ohwx, where 10 is the repeat count.", (0.65, 0.2)),
        Note(lambda s: s.overlay.findChildren(QGroupBox)[1],
             "Export for Training makes a clean copy: resized to training buckets, metadata stripped, trigger word added.",
             (0.65, 0.48)),
        Note(g("tab_datasets.list_datasets"),
             "Dataset Collections keep finished image and caption sets together. Double-click one to open it.",
             (0.64, 0.8)),
    ]))
    S(Shot("08_metadata.png", shot_metadata, [
        Note(lambda s: WINDOW.tab_metadata.side.tabBar(),
             "Other tabs: Inspect a single file, write your own authorship, or reuse embedded prompts as captions.",
             (0.3, 0.63)),
        Note(g("tab_metadata.lbl_audit"),
             "Audit counts what the chosen images reveal: GPS location, device serial numbers, AI generation data and more.",
             (0.3, 0.77)),
        Note(lambda s: WINDOW.tab_metadata.findChildren(QGroupBox)[0],
             "Choose what to remove. Cleaning is lossless: pixels are never re-encoded.", (0.3, 0.91)),
    ]))
    S(Shot("09_train_setup.png", shot_train_setup, [
        Note(lambda s: WINDOW.tab_train.combo_family.parentWidget(),
             "Pick the model family and a built-in preset. Presets only change the settings they contain.", (0.76, 0.3)),
        Note(g("tab_train.grp_models"),
             "Point to your model files once; they are remembered. Get opens the download page.", (0.76, 0.5)),
        Note(lambda s: WINDOW.tab_train.sections["Training Parameters"][0],
             "Parameter groups are collapsible. Hover a setting for an explanation.", (0.76, 0.75)),
    ]))
    S(Shot("10_train_running.png", shot_train_running, [
        Note(lambda s: WINDOW.tab_train.chart,
             "Loss per epoch (green) and learning rate (dashed). Arrows mark Adaptive LR decisions.", (0.3, 0.5)),
        Note(lambda s: [WINDOW.tab_train.lbl_status, WINDOW.tab_train.progress],
             "Live status: epoch, step, speed and time remaining, read from the trainer's output.", (0.3, 0.3)),
        Note(g("tab_train.console"),
             "The full console output of the cache and training stages, also saved as run.log.", (0.3, 0.75)),
        Note(lambda s: WINDOW.memory_strip.bar,
             "VRAM and RAM in use, on every tab. The white mark is the peak of this run.", (0.3, 0.9)),
    ], after=reset_train))
    S(Shot("11_train_samples.png", shot_train_samples, [
        Note(g("tab_train.samples"),
             "Preview images rendered during training, newest first. Double-click one to open it full size.", (0.3, 0.55)),
        Note(lambda s: WINDOW.tab_train.btn_start.parentWidget().findChildren(QPushButton)[0:3],
             "Start, pause after the current epoch, or stop. Pause saves a state you can resume from.", (0.3, 0.25)),
    ], after=reset_train))
    S(Shot("12_problem_images.png", shot_problem_images, [
        Note(lambda s: row_rect(s.base.table, s.base.stuck_row),
             "Images the loss watch flags. Stuck and suspect images usually have a wrong or too short caption.",
             (0.28, 0.75)),
        Note(lambda s: s.base.edit,
             "Fix the caption here and click Save fix. The running training picks it up at the next epoch.", (0.74, 0.18)),
    ]))
    S(Shot("13_command_palette.png", shot_palette, [
        Note(lambda s: s.overlay.input,
             "Ctrl+K opens the command palette. Type part of a name to find any action, topic or filter.", (0.86, 0.3)),
        Note(lambda s: s.overlay.list,
             "Enter runs the highlighted command; the shortcut, if any, is shown on the right.", (0.86, 0.6)),
    ]))
    S(Shot("14_help_center.png", shot_help, [
        Note(lambda s: s.base.search,
             "Search every guide. F1 opens the topic for the tab you are on.", (0.68, 0.72)),
        Note(lambda s: s.base.list,
             "Guides are grouped by area: basics, gallery, AI, tools, training and troubleshooting.", (0.68, 0.88)),
    ]))
    S(Shot("15_settings.png", shot_settings, [
        Note(lambda s: (WINDOW.tab_settings.findChildren(QGroupBox)[0], QRect(0, 0, 340, 170)),
             "Choose a theme and an interface scale.", (0.7, 0.14)),
        Note(lambda s: (WINDOW.tab_settings.findChildren(QGroupBox)[2], QRect(0, 0, 340, 250)),
             "Extra folders to scan for local vision models.", (0.7, 0.4)),
        Note(lambda s: (WINDOW.tab_settings.txt_env, QRect(0, 0, 340, 170)),
             "Diagnostics: copy this report when you ask for help with a problem.", (0.7, 0.82)),
    ], after=restore_size))


# ---------------------------------------------------------------------------------------------------- main
def wanted(filename: str) -> bool:
    if not ARGS.only:
        return True
    return any(p.strip() and p.strip().lower() in filename.lower() for p in ARGS.only.split(","))


def main() -> int:
    global WINDOW
    out = Path(ARGS.out)
    out.mkdir(parents=True, exist_ok=True)
    c = Ctx()
    WINDOW = c.window
    c.open_dataset()
    make_neutral_status(c)
    fix_settings_hint(c)
    build_shots()
    problems = 0
    for shot in SHOTS:
        if not wanted(shot.filename):
            continue
        warns: list[str] = []
        try:
            scene = shot.setup(c)
            c.pump(0.4)
            fix_settings_hint(c)
            for t in (c.window.tab_gallery, c.window.tab_caption, c.window.tab_editor, c.window.tab_datasets,
                      c.window.tab_metadata):
                t.browser.update_status()
            root = scene.overlay if scene.overlay is not None else scene.base
            with Sanitizer(scene.base), (Sanitizer(scene.overlay) if scene.overlay is not None else _Null()):
                bad = leaks(scene.base) + (leaks(scene.overlay) if scene.overlay is not None else [])
                img = scene.render()
                img = annotate(img, scene, shot.notes, warns.append)
            for b in bad:
                warns.append(f"PRIVATE TEXT visible: {b!r}")
            img.save(out / shot.filename, optimize=True)
            print(f"{shot.filename}: {img.width}x{img.height}" + "".join(f"\n    WARN {w}" for w in warns))
            problems += len(warns)
        except Exception as e:  # keep going: report which shot failed
            import traceback
            traceback.print_exc()
            print(f"{shot.filename}: FAILED ({e})")
            problems += 1
        finally:
            for attr in ("keep",):
                w = getattr(c, attr, None)
                if w is not None:
                    try:
                        w.close()
                        w.deleteLater()
                    except RuntimeError:
                        pass
                    c.keep = None
            if shot.after:
                shot.after(c)
            c.pump(0.2)
    c.window.close()
    QThreadPool.globalInstance().waitForDone(5000)
    if ARGS.keep_temp:
        print(f"temp folder kept: {TMP}")
    else:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f"done, {problems} warning(s)")
    return 0


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)   # skip interpreter teardown (Qt threads / torch-free but Qt objects may still be alive)
