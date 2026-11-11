"""Measure the UI numbers of docs/dev/UI_UX_REDESIGN.md section 5.1 on generated data.

    python tools/bench_ui.py [--count 5000] [--theme dark_teal.xml] [--keep]

Generates --count small pictures with tag captions in a temp folder, points every user-data location at a temp
profile (never the real user_data/), runs Qt offscreen and prints: start to first paint, open folder, return to the
Gallery tab, sort (per mode), filter, clear filter, select one / all, scroll frame time and memory. No model is
loaded, nothing touches the network or the GPU. Other work on the machine changes the figures by about 10 %.
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

T0 = time.perf_counter()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROFILE = Path(tempfile.mkdtemp(prefix="tsr_bench_profile_"))
os.environ["TAGSCRIBER_USER_DATA"] = str(PROFILE / "user_data")      # read by core.paths at import
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HIP_VISIBLE_DEVICES"] = "-1"

TAGS = ("1girl solo smile long_hair blue_eyes looking_at_viewer outdoors sky cloud tree flower dress hat "
        "short_hair red_eyes sitting standing indoors window night city rain hair_ornament ribbon").split()


def generate(folder: Path, count: int) -> None:
    from PIL import Image
    rng = random.Random(7)
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        w, h = rng.choice((320, 384, 448)), rng.choice((256, 320, 448))
        Image.new("RGB", (w, h), (rng.randrange(256), rng.randrange(256), rng.randrange(256))).save(
            folder / f"img_{i:05d}.jpg", quality=70)
        if rng.random() < 0.9:
            tags = rng.sample(TAGS, rng.randint(3, 14))
            (folder / f"img_{i:05d}.txt").write_text(", ".join(tags), encoding="utf-8")


def rss_mb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 1e6


def timed(fn, *a):
    t = time.perf_counter()
    r = fn(*a)
    return time.perf_counter() - t, r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5000)
    ap.add_argument("--theme", default=None, help="ui.theme value (default: the app default, dark_teal.xml)")
    ap.add_argument("--no-defer", action="store_true", help="apply a token theme's stylesheet before the window is "
                    "built (how it was before the deferred start-up), to compare")
    ap.add_argument("--keep", action="store_true", help="keep the generated folder and profile")
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="tsr_bench_"))
    folder = work / "bench_dataset"
    out: dict[str, str] = {}
    try:
        t = time.perf_counter()
        generate(folder, args.count)
        gen_dt = time.perf_counter() - t
        print(f"generated {args.count} pictures in {gen_dt:.1f} s (not part of the numbers)")

        from core import paths
        paths.DEFAULT_COLLECTIONS_DIR = work / "collections"
        paths.DEFAULT_EDITS_DIR = work / "edits"
        paths.LEGACY_CONFIG_FILE = work / "legacy" / "config.json"          # as tests/conftest.py does: the real
        paths.LEGACY_API_PRESETS_FILE = work / "legacy" / "api_presets.json"  # pre-modernization files are not read
        paths.LEGACY_TAGS_FILE = work / "legacy" / "user_tags.txt"
        paths.ensure_user_dirs()
        from core.config import settings
        cfg = settings()
        cfg.set("paths.collections_dir", str(work / "collections"))
        if args.theme:
            cfg.set("ui.theme", args.theme)

        t_import0 = time.perf_counter()
        from PySide6.QtCore import QItemSelectionModel
        from PySide6.QtWidgets import QApplication
        import main as app_main
        from tabs import icons, theme
        from tabs.settings import apply_theme
        t_imports = time.perf_counter() - t_import0

        app = QApplication.instance() or QApplication([])
        t1 = time.perf_counter()
        defer = not args.no_defer and "finish_deferred" in dir(theme)
        apply_theme(settings().get("ui.theme"), **({"defer_stylesheet": True} if defer else {}))
        t_theme = time.perf_counter() - t1
        t1 = time.perf_counter()
        window = app_main.MainWindow()
        if defer:
            theme.finish_deferred(app)
        icons.apply_icons(window)
        theme.refresh_icons(window)
        t_window = time.perf_counter() - t1
        t1 = time.perf_counter()
        window.show()
        app.processEvents()
        window.grab()                                     # force one real paint
        t_show = time.perf_counter() - t1
        out["start (process start to first paint)"] = f"{time.perf_counter() - T0 - gen_dt:.2f} s"
        out["  imports / theme / MainWindow() / show+paint"] = (
            f"{t_imports:.2f} / {t_theme:.2f} / {t_window:.2f} / {t_show:.2f} s")
        out["memory after first paint"] = f"{rss_mb():.0f} MB"

        from tabs.workspace.context import workspace
        ctx = workspace()
        gal = window.tab_gallery
        b = gal.browser

        dt, _ = timed(ctx.open_folder, str(folder), False, None)
        out["open_folder returns after (first read of new files)"] = f"{dt:.2f} s"
        dt, _ = timed(ctx.open_folder, str(folder), False, None)
        out["open_folder again (files cached)"] = f"{dt:.2f} s"
        dt, _ = timed(app.processEvents)
        out["  first event loop pass after it"] = f"{dt:.2f} s"
        # let the thumbnails and the dimension scan settle a little
        t = time.perf_counter()
        while time.perf_counter() - t < 2.0:
            app.processEvents()
            time.sleep(0.01)

        def go(row):
            window.sidebar.setCurrentRow(row)
            app.processEvents()
        go(1)
        dt, _ = timed(go, 0)
        out["return to Gallery tab"] = f"{dt:.3f} s"
        ts = []
        for row in (2, 3, 4, 5, 6, 1, 0):
            d, _ = timed(go, row)
            ts.append(d)
        out["other tab switches (min / max)"] = f"{min(ts) * 1000:.0f} / {max(ts) * 1000:.0f} ms"

        for mode in ("Name", "Megapixels", "Caption length", "Tag count"):
            if mode == "Name":
                b.combo_sort.setCurrentText("Caption length")      # leave Name so the switch is a real sort
                app.processEvents()
            dt, _ = timed(lambda m=mode: (b.combo_sort.setCurrentText(m), app.processEvents()))
            out[f"sort by {mode}"] = f"{dt * 1000:.0f} ms"
        b.combo_sort.setCurrentText("Name")
        app.processEvents()

        for label, q in (("text filter 'smile'", "smile"), ("tag filter", "tag:solo -tag:rain")):
            dt, _ = timed(lambda q=q: (b.set_filter(q), app.processEvents()))
            out[f"filter, {label}"] = f"{dt * 1000:.0f} ms"
        dt, _ = timed(lambda: (b.set_filter(""), app.processEvents()))
        out["clear filter"] = f"{dt * 1000:.0f} ms"

        dt, _ = timed(lambda: (b.view.setCurrentIndex(b.proxy.index(3, 0)),
                               b.view.selectionModel().select(b.proxy.index(3, 0), QItemSelectionModel.ClearAndSelect),
                               app.processEvents()))
        out["select one"] = f"{dt * 1000:.0f} ms"
        dt, _ = timed(lambda: (b.select_all(), app.processEvents()))
        out["select all"] = f"{dt * 1000:.0f} ms"
        b.deselect_all()
        app.processEvents()

        # scroll: paint frames at 220 px (with captions) and 100 px (without); thumbnails are given time to load
        for label, size, caps in (("220 px, captions", 220, True), ("100 px, no captions", 100, False)):
            b.chk_captions.setChecked(caps)
            b.slider.setValue(size)
            app.processEvents()
            bar = b.view.verticalScrollBar()
            t = time.perf_counter()
            while time.perf_counter() - t < 1.5:
                app.processEvents()
                time.sleep(0.01)
            frames = []
            step = max(1, bar.pageStep() // 3)
            for i in range(60):
                bar.setValue(min(bar.maximum(), bar.value() + step))
                d, _ = timed(lambda: (b.view.viewport().repaint(), app.processEvents()))
                frames.append(d * 1000)
                if bar.value() >= bar.maximum():
                    bar.setValue(0)
            frames.sort()
            out[f"scroll frame, {label} (median / p95)"] = (
                f"{statistics.median(frames):.1f} / {frames[int(len(frames) * 0.95) - 1]:.1f} ms")
        out["memory at the end"] = f"{rss_mb():.0f} MB"

        width = max(len(k) for k in out)
        print()
        for k, v in out.items():
            print(f"{k:<{width}}  {v}")
        window.close()
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
            shutil.rmtree(PROFILE, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
