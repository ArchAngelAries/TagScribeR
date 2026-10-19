"""Regression tests for the app-core review fixes (docs/dev/audit/09_app_core_review.md, findings 2-9).

All images are generated here; nothing touches real datasets.
"""
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from core import caption_io, fileops, health, image_ops
from core.image_ops import Operation
from tests.conftest import make_image


def _app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


# ------------------------------------------------------------------ #2 editor: overwrite + same-format convert
class _Sig:
    def __init__(self):
        self.slots = []

    def connect(self, f):
        self.slots.append(f)

    def emit(self, *a):
        for f in list(self.slots):
            f(*a)


class _FakeJob:
    def __init__(self, items, fn, title):
        self.items, self.fn, self.title = items, fn, title
        self.item_skipped, self.progress, self.item_done, self.finished = _Sig(), _Sig(), _Sig(), _Sig()

    def go(self):
        from inference.worker import JobSummary
        s = JobSummary(title=self.title, total=len(self.items))
        for it in self.items:
            r = self.fn(it)
            s.done += 1
            self.item_done.emit(it, str(r))
        self.finished.emit(s)


@pytest.fixture
def editor(monkeypatch):
    _app()
    from tabs import editor as ed
    tab = ed.EditorTab()
    asked, jobs = [], []
    monkeypatch.setattr(ed, "confirm", lambda parent, title, text, **kw: asked.append((title, text)) or True)
    monkeypatch.setattr(ed, "show_job_summary", lambda *a, **k: None)
    monkeypatch.setattr(ed.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(ed, "start_file_job", lambda items, fn, title: jobs.append(_FakeJob(items, fn, title)) or jobs[-1])
    monkeypatch.setattr(tab, "_schedule_preview", lambda: None)
    monkeypatch.setattr(tab.ctx, "images_modified", lambda keys: asked.append(("modified", list(keys))))
    return tab, asked, jobs, monkeypatch


def test_overwrite_convert_same_format_asks_and_says_what_happened(editor, tmp_path):
    tab, asked, jobs, mp = editor
    a = make_image(tmp_path / "a.jpg")
    b = make_image(tmp_path / "b.png")
    mp.setattr(tab.browser, "selected_keys", lambda: [str(a), str(b)])
    tab.rad_overwrite.setChecked(True)
    before = a.read_bytes()
    tab.run(Operation("convert", {"format": "JPG", "quality": 50}))
    title, text = asked[0]
    assert title == "Overwrite originals?" and "1 original" in text and "REPLACE" in text
    jobs[0].go()
    assert a.read_bytes() != before                       # re-encoded in place, after the confirmation
    assert ("modified", [str(a)]) in asked                # thumbnails / mtime refreshed
    assert (tmp_path / "b.jpg").exists() and b.exists()   # the PNG got a new file; original kept
    assert "1 originals replaced, 1 new file(s) created" in tab.lbl_status.text()


def test_overwrite_convert_to_other_format_needs_no_confirmation(editor, tmp_path):
    tab, asked, jobs, mp = editor
    b = make_image(tmp_path / "b.png")
    mp.setattr(tab.browser, "selected_keys", lambda: [str(b)])
    tab.rad_overwrite.setChecked(True)
    tab.run(Operation("convert", {"format": "JPG", "quality": 80}))
    assert not asked
    jobs[0].go()
    assert b.exists() and (tmp_path / "b.jpg").exists()


def test_declining_the_confirmation_changes_nothing(editor, tmp_path):
    tab, asked, jobs, mp = editor
    from tabs import editor as ed
    a = make_image(tmp_path / "a.jpg")
    before = a.read_bytes()
    mp.setattr(ed, "confirm", lambda *a_, **k: False)
    mp.setattr(tab.browser, "selected_keys", lambda: [str(a)])
    tab.rad_overwrite.setChecked(True)
    tab.run(Operation("convert", {"format": "JPG", "quality": 50}))
    assert not jobs and a.read_bytes() == before


def test_in_place_edit_uses_quality_slider_and_keeps_png_text(editor, tmp_path):
    tab, asked, jobs, mp = editor
    j = tmp_path / "a.jpg"
    Image.effect_noise((128, 128), 60).convert("RGB").save(j, quality=95)
    p = tmp_path / "gen.png"
    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", "masterpiece, 1girl\nSteps: 20")
    Image.new("RGB", (32, 16)).save(p, pnginfo=meta)
    mp.setattr(tab.browser, "selected_keys", lambda: [str(j), str(p)])
    tab.rad_overwrite.setChecked(True)
    tab.slider_quality.setValue(50)
    big = j.stat().st_size
    tab.run(Operation("rotate", {"direction": "cw"}))
    jobs[0].go()
    assert j.stat().st_size < big * 0.7                   # q50, not the old fixed q92
    assert Image.open(p).text["parameters"] == "masterpiece, 1girl\nSteps: 20"
    assert Image.open(p).size == (16, 32)


# ------------------------------------------------------------------ #3 collection copy
def test_copy_does_not_overwrite_caption_when_only_extension_differs(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    make_image(src / "a.png")
    (src / "a.txt").write_text("NEW")
    make_image(dst / "a.jpg")
    (dst / "a.txt").write_text("ORIGINAL")
    rep = fileops.CopyReport()
    out = fileops.copy_image_with_caption(src / "a.png", dst, report=rep)
    assert (dst / "a.txt").read_text() == "ORIGINAL"
    assert out.name == "a (1).png" and (dst / "a (1).txt").read_text() == "NEW"
    assert len(rep.renamed) == 1 and "renamed" in rep.summary()


def test_copy_skip_mode_skips_on_sidecar_only_clash(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    make_image(src / "a.png")
    (dst / "a.txt").write_text("keep")
    rep = fileops.CopyReport()
    assert fileops.copy_image_with_caption(src / "a.png", dst, on_conflict="skip", report=rep) is None
    assert not (dst / "a.png").exists() and (dst / "a.txt").read_text() == "keep"


# ------------------------------------------------------------------ #4 metadata survives in-place edits
def test_png_text_xmp_and_comment_survive_edits(tmp_path):
    p = tmp_path / "gen.png"
    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", "masterpiece, 1girl")
    meta.add_text("prompt", '{"1": {}}')
    meta.add_itxt("note", "caf\u00e9 \u2603")
    Image.new("RGB", (40, 20)).save(p, pnginfo=meta, icc_profile=b"x" * 20)
    img = image_ops.open_upright(p, keep_extras=True)
    image_ops.save(image_ops.apply(img, Operation("rotate", {"direction": "cw"})), p)
    out = Image.open(p)
    assert out.size == (20, 40)
    assert out.text["parameters"] == "masterpiece, 1girl" and out.text["prompt"] == '{"1": {}}'
    assert out.text["note"] == "caf\u00e9 \u2603"
    assert out.info.get("icc_profile") == b"x" * 20


def test_jpeg_comment_and_xmp_survive_and_quality_is_honoured(tmp_path):
    p = tmp_path / "a.jpg"
    xmp = b"<x:xmpmeta xmlns:x='adobe:ns:meta/'>hello</x:xmpmeta>"
    Image.effect_noise((96, 96), 50).convert("RGB").save(p, comment=b"my comment", xmp=xmp, quality=95)
    big = p.stat().st_size
    img = image_ops.open_upright(p, keep_extras=True)
    image_ops.save(image_ops.apply(img, Operation("flip", {"axis": "h"})), p, quality=40)
    out = Image.open(p)
    assert out.info.get("comment") == b"my comment"
    assert b"hello" in (out.info.get("xmp") or b"")
    assert p.stat().st_size < big * 0.6


def test_extras_are_not_carried_unless_asked(tmp_path):
    p = tmp_path / "gen.png"
    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", "secret prompt")
    Image.new("RGB", (8, 8)).save(p, pnginfo=meta)
    img = image_ops.open_upright(p)                      # export path: unchanged behaviour
    image_ops.save(img, tmp_path / "o.png")
    assert "parameters" not in Image.open(tmp_path / "o.png").text


_XMP6 = (b'<?xpacket begin=""?><x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf='
         b'"http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description xmlns:tiff="http://ns.adobe.com/tiff/1.0/" '
         b'tiff:Orientation="6"/></rdf:RDF></x:xmpmeta><?xpacket end="w"?>')


def test_export_strip_carries_no_xmp_or_comment(tmp_path):
    """Review of the fix batch: open_upright must REPLACE the info (exif_transpose's copy carries the whole source
    info), so export's strip-metadata path leaks no XMP (GPS, creator, prompts) or comment."""
    from core.export import ExportOptions, export_one
    src = tmp_path / "src"
    src.mkdir()
    ex = Image.Exif()
    ex[0x0112] = 6
    Image.new("RGB", (40, 20), (1, 2, 3)).save(src / "a.jpg", exif=ex, xmp=_XMP6, comment=b"private note")
    meta = PngImagePlugin.PngInfo()
    meta.add_itxt("XML:com.adobe.xmp", _XMP6.decode())
    Image.new("RGB", (40, 20)).save(src / "b.png", pnginfo=meta)
    (src / "a.txt").write_text("a cap", encoding="utf-8")
    (src / "b.txt").write_text("b cap", encoding="utf-8")
    for name, fmt in (("a.jpg", ""), ("a.jpg", "WEBP"), ("b.png", ""), ("b.png", "JPG")):
        opts = ExportOptions(out_dir=str(tmp_path / f"out_{fmt or 'keep'}"), resize="none", fmt=fmt,
                             strip_metadata=True, skip_uncaptioned=False)
        out = export_one(src / name, 0, opts)
        with Image.open(out) as im:
            assert not im.info.get("xmp") and not im.info.get("comment"), (name, fmt, sorted(im.info))
            assert "XML:com.adobe.xmp" not in getattr(im, "text", {}), (name, fmt)


def test_in_place_edit_resets_xmp_orientation_with_the_pixels(tmp_path):
    for ext in ("jpg", "png"):
        p = tmp_path / f"o.{ext}"
        ex = Image.Exif()
        ex[0x0112] = 6
        if ext == "jpg":
            Image.new("RGB", (40, 20)).save(p, exif=ex, xmp=_XMP6)
        else:
            meta = PngImagePlugin.PngInfo()
            meta.add_itxt("XML:com.adobe.xmp", _XMP6.decode())
            Image.new("RGB", (40, 20)).save(p, exif=ex, pnginfo=meta)
        img = image_ops.open_upright(p, keep_extras=True)
        image_ops.save(img, p)
        with Image.open(p) as im:
            xmp = im.info.get("xmp") or im.info.get("XML:com.adobe.xmp") or b""
            xmp = xmp.encode() if isinstance(xmp, str) else xmp
            assert xmp, ext                                       # XMP kept ...
            assert b'tiff:Orientation="6"' not in xmp, ext       # ... but no longer claims the old rotation
            assert im.getexif().get(0x0112, 1) == 1 and im.size == (20, 40), ext


def test_16bit_thumbnail_keeps_exif_orientation(tmp_path):
    from core.image_utils import load_qimage
    p = tmp_path / "g.png"
    ex = Image.Exif()
    ex[0x0112] = 6
    Image.fromarray(np.tile(np.linspace(0, 65535, 20).astype(np.uint16), (40, 1))).save(p, exif=ex)
    img, orig = load_qimage(str(p), size=(400, 400))
    assert orig == (40, 20) and (img.width(), img.height()) == (40, 20)


def test_32bit_mode_i_in_8bit_range_is_not_darkened():
    im = Image.fromarray(np.array([[0, 128, 255]], np.int32), "I")
    assert np.asarray(image_ops.to_8bit(im)).tolist() == [[0, 128, 255]]


# ------------------------------------------------------------------ #5 16-bit
def _gradient16(path, size=(64, 32)):
    a = np.tile(np.linspace(0, 65535, size[0]).astype(np.uint16), (size[1], 1))
    Image.fromarray(a).save(path)
    return path


def test_to_8bit_scales_instead_of_clipping(tmp_path):
    p = _gradient16(tmp_path / "g.png")
    im = Image.open(p)
    assert im.mode == "I;16"
    out = image_ops.to_8bit(im)
    arr = np.asarray(out)
    assert out.mode == "L" and arr[0, 0] == 0 and arr[0, -1] == 255
    assert 100 < int(arr[0, 32]) < 156 and len(set(arr[0].tolist())) > 50
    full = Image.fromarray(np.full((4, 4), 257 * 77, np.uint16))
    assert int(np.asarray(image_ops.to_8bit(full))[0, 0]) == 77


def test_16bit_png_converts_to_jpg_and_thumbnail_without_clipping(tmp_path):
    _app()
    p = _gradient16(tmp_path / "g.png")
    img = image_ops.open_upright(p)
    out = image_ops.save(img, tmp_path / "o", "JPG")
    assert len(set(np.asarray(Image.open(out).convert("L"))[0].tolist())) > 30
    from core.image_utils import load_qimage
    qi, orig = load_qimage(p, (64, 32))
    assert orig == (64, 32)
    assert len({qi.pixelColor(x, 5).red() for x in range(qi.width())}) > 30


def test_16bit_png_stays_16bit_when_saved_as_png(tmp_path):
    p = _gradient16(tmp_path / "g.png")
    out = image_ops.save(image_ops.open_upright(p), tmp_path / "o.png")
    assert Image.open(out).mode == "I;16"


def test_health_blur_metric_uses_scaled_16bit(tmp_path):
    p = _gradient16(tmp_path / "g.png", (128, 128))
    st = health.analyze(p)
    assert not st.error and st.sharpness >= 0
    gray = Image.open(p)
    clipped = np.asarray(gray.convert("L"), dtype=np.float32)
    scaled = np.asarray(image_ops.to_8bit(gray), dtype=np.float32)
    assert len(np.unique(clipped)) <= 2 and len(np.unique(scaled)) > 50


# ------------------------------------------------------------------ #6 pixel budget
def test_oversized_header_is_refused_before_decoding(tmp_path, monkeypatch):
    _app()
    from core import image_utils
    p = make_image(tmp_path / "big.png", size=(200, 100))
    monkeypatch.setattr(image_utils, "MAX_THUMBNAIL_PIXELS", 10_000)    # stands in for a 20000x20000 header
    monkeypatch.setattr(Image.Image, "load", lambda self: pytest.fail("decoded an over-budget image"))
    with pytest.raises(image_utils.ImageTooLarge):
        image_utils.load_qimage(p, (64, 64))
    assert image_utils.load_thumbnail(p, (64, 64)).isNull()


def test_budget_does_not_block_normal_images_or_big_jpeg_drafts(tmp_path, monkeypatch):
    _app()
    from core import image_utils
    j = make_image(tmp_path / "w.jpg", size=(1600, 1200))
    monkeypatch.setattr(image_utils, "MAX_THUMBNAIL_PIXELS", 200_000)
    qi, orig = image_utils.load_qimage(j, (100, 100))     # JPEG draft brings it under the budget
    assert orig == (1600, 1200) and not qi.isNull()


def test_health_refuses_oversized_without_decoding(tmp_path, monkeypatch):
    p = make_image(tmp_path / "big.png", size=(200, 100))
    monkeypatch.setattr(health, "MAX_ANALYZE_PIXELS", 10_000)
    monkeypatch.setattr(Image.Image, "load", lambda self: pytest.fail("decoded an over-budget image"))
    st = health.analyze(p)
    assert "too large" in st.error and (st.width, st.height) == (200, 100) and st.file_hash


def test_editor_open_is_not_limited_by_preview_budget(tmp_path, monkeypatch):
    from core import image_utils
    p = make_image(tmp_path / "a.png", size=(200, 100))
    monkeypatch.setattr(image_utils, "MAX_THUMBNAIL_PIXELS", 10)
    assert image_ops.open_upright(p).size == (200, 100)


# ------------------------------------------------------------------ #7 UTF-16 captions
@pytest.mark.parametrize("enc,bom", [("utf-16-le", True), ("utf-16-be", True), ("utf-16-le", False),
                                     ("utf-16-be", False)])
def test_utf16_caption_decodes(tmp_path, enc, bom):
    text = "1girl, smile, caf\u00e9, long hair, solo"
    data = text.encode(enc)
    if bom:
        data = (b"\xff\xfe" if enc.endswith("le") else b"\xfe\xff") + data
    p = tmp_path / "a.txt"
    p.write_bytes(data)
    assert caption_io.read_text_file(p) == text


def test_utf8_and_cp1252_still_decode(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes("caf\u00e9, ab".encode("utf-8"))
    assert caption_io.read_text_file(p) == "caf\u00e9, ab"
    p.write_bytes("caf\u00e9, ab".encode("cp1252"))
    assert caption_io.read_text_file(p) == "caf\u00e9, ab"
    p.write_bytes(b"\xef\xbb\xbfhello")
    assert caption_io.read_text_file(p) == "hello"


def test_editing_utf16_caption_saves_utf8(tmp_path):
    img = make_image(tmp_path / "a.png")
    cap = tmp_path / "a.txt"
    cap.write_bytes(b"\xff\xfe" + "1girl, smile".encode("utf-16-le"))
    assert caption_io.read_caption(img) == "1girl, smile"
    assert caption_io.write_caption(img, "1girl, smile, solo")
    assert cap.read_bytes() == b"1girl, smile, solo"


# ------------------------------------------------------------------ #8 file mode and fsync
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_caption_and_image_writes_keep_file_mode(tmp_path):
    img = make_image(tmp_path / "a.png")
    cap = tmp_path / "a.txt"
    cap.write_text("old")
    os.chmod(cap, 0o664)
    os.chmod(img, 0o664)
    caption_io.write_caption(img, "new")
    assert stat.S_IMODE(cap.stat().st_mode) == 0o664
    image_ops.save(image_ops.open_upright(img), img)
    assert stat.S_IMODE(img.stat().st_mode) == 0o664
    # a brand-new file gets the umask default, not mkstemp's 0600
    new = image_ops.save(image_ops.open_upright(img), tmp_path / "new.png")
    assert stat.S_IMODE(new.stat().st_mode) == 0o666 & ~caption_io._UMASK
    fresh = make_image(tmp_path / "b.png")
    caption_io.write_caption(fresh, "x")
    assert stat.S_IMODE((tmp_path / "b.txt").stat().st_mode) == 0o666 & ~caption_io._UMASK


def test_image_save_fsyncs_before_replace(tmp_path, monkeypatch):
    calls = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: calls.append("fsync") or real_fsync(fd))
    monkeypatch.setattr(os, "replace", lambda a, b: calls.append("replace") or real_replace(a, b))
    image_ops.save(Image.new("RGB", (8, 8)), tmp_path / "x.png")
    assert calls == ["fsync", "replace"]


# ------------------------------------------------------------------ #9 smaller items
def test_collection_name_validation():
    ok = fileops.valid_collection_name
    assert ok("My set") and ok("set_2")
    for bad in ("", "  ", "..", "../x", "a/b", "a\\b", "/abs", "C:\\x", "a..b"):
        assert not ok(bad), bad


def test_inspector_drops_entries_that_were_removed():
    _app()
    from core.dataset_session import Entry
    from tabs.workspace.panels import InspectorPanel
    panel = InspectorPanel()
    e1, e2 = Entry(Path("a.png")), Entry(Path("b.png"))
    panel._load_preview = lambda e: None
    panel.show_entries([e1])
    assert e2
    assert panel.drop_entries(["zzz"]) is False and panel.entries == [e1]
    assert panel.drop_entries([e1.key]) is True
    assert panel.entries == [] and not panel.tag_editor.isEnabled()


def test_caption_write_never_replaces_an_image(tmp_path):
    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(img)
    before = img.read_bytes()
    with pytest.raises(ValueError):
        caption_io.write_caption(img, "text", ext=".png")
    assert img.read_bytes() == before
    assert caption_io.write_caption(img, "text") and (tmp_path / "a.txt").read_text(encoding="utf-8") == "text"
