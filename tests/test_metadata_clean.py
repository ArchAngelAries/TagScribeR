"""Clean metadata must clean (docs/dev/audit/14, H3), file text must never render as HTML (H4), and the atomic writer
fsyncs and keeps permission bits (L8). Every file here is generated: tiny pictures plus hand-built segments."""
import os
import struct
import sys
import zlib

import pytest
from PIL import Image, PngImagePlugin

from core import metadata as md

# strings that must not survive a clean that covers them
MARKS = [b"MAKERNOTEDATA", b"CamCo", b"SN123456", b"Springfield", b"JaneByline", b"XMPMAINDATA", b"XMPEXTDATA",
         b"MPFINDEX", b"C2PAJUMBF", b"COMMENTMARK", b"TRAILERMARK", b"THUMBMARK", b"+02:00", b"VENDORAPP5"]


def seg(marker: int, payload: bytes) -> bytes:
    return bytes([0xFF, marker]) + struct.pack(">H", len(payload) + 2) + payload


def _iptc() -> bytes:
    ds = b"".join(bytes([0x1C, 2, n]) + struct.pack(">H", len(v)) + v
                  for n, v in ((90, b"Springfield"), (101, b"Freedonia"), (80, b"JaneByline")))
    if len(ds) & 1:
        ds += b"\x00"
    return b"Photoshop 3.0\x00" + b"8BIM" + struct.pack(">H", 0x0404) + b"\x00\x00" + struct.pack(">I", len(ds)) + ds


def _tiff_with_thumbnail(exif: Image.Exif) -> bytes:
    """Pillow writes no IFD1: append one (JPEGInterchangeFormat + Length) holding a tiny JPEG that carries a marker."""
    import io
    thumb = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 10, 10)).save(thumb, "JPEG", comment=b"THUMBMARK")
    tb = thumb.getvalue()
    t = bytearray(exif.tobytes()[6:])
    e = "<" if t[:2] == b"II" else ">"
    off = struct.unpack(e + "I", t[4:8])[0]
    n = struct.unpack(e + "H", t[off:off + 2])[0]
    nxt = off + 2 + 12 * n
    ifd1_at = len(t)
    t[nxt:nxt + 4] = struct.pack(e + "I", ifd1_at)
    data_at = ifd1_at + 2 + 24 + 4
    t += struct.pack(e + "H", 2)
    t += struct.pack(e + "HHII", 0x0201, 4, 1, data_at) + struct.pack(e + "HHII", 0x0202, 4, 1, len(tb))
    t += struct.pack(e + "I", 0) + tb
    return bytes(t)


def make_loaded_jpeg(path, xmp_main: bytes = b"<x:xmpmeta>XMPMAINDATA</x:xmpmeta>"):
    im = Image.new("RGB", (64, 32), (120, 30, 200))
    for x in range(64):
        im.putpixel((x, x % 32), (x * 3, 255 - x * 2, 40))
    ex = im.getexif()
    ex[md.ORIENTATION] = 6
    ex[md.MAKE] = "CamCo"
    ex[md.SOFTWARE] = "Editor 1.0"
    ex[md.DATETIME] = "2026:01:01 10:00:00"
    sub = ex.get_ifd(md.EXIF_IFD)
    sub[md.BODY_SERIAL] = "SN123456"
    sub[md.MAKER_NOTE] = b"MAKERNOTEDATA" + bytes(40)
    sub[md.OFFSET_TIME_ORIGINAL] = "+02:00"
    gps = ex.get_ifd(md.GPS_IFD)
    gps[1] = "N"
    gps[2] = (52.0, 22.0, 1.0)
    im.save(path, exif=ex, icc_profile=b"icc" * 40, quality=95)
    data = path.read_bytes()
    # swap Pillow's EXIF for one with an IFD1 thumbnail
    segs = list(md._jpeg_segments(data))
    out = [md.SOI]
    for m, s in segs:
        if md._is_exif_seg(m, s):
            out.append(seg(0xE1, md.EXIF_HEADER + _tiff_with_thumbnail(ex)))
        elif m == 0xDA:
            out.append(seg(0xED, _iptc()))
            out.append(seg(0xE1, md.XMP_NS + b"\x00" + xmp_main))
            out.append(seg(0xE1, md.XMP_EXT_NS + b"\x00" + b"g" * 32 + struct.pack(">II", 9, 0) + b"XMPEXTDATA"))
            out.append(seg(0xE2, b"MPF\x00MPFINDEX"))
            out.append(seg(0xEB, b"JP\x00\x00C2PAJUMBF"))
            out.append(seg(0xE5, b"VENDORAPP5"))
            out.append(seg(0xFE, b"COMMENTMARK"))
            out.append(s)
        else:
            out.append(s)
    path.write_bytes(b"".join(out) + b"TRAILERMARK" + b"PK\x03\x04")
    return path


def make_loaded_png(path):
    im = Image.new("RGBA", (40, 30), (10, 200, 30, 180))
    for x in range(40):
        im.putpixel((x, x % 30), (x * 6, 5, 250 - x, 255))
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "a prompt\nSteps: 20, Seed: 1")
    info.add_text("Source", "VENDORAPP5 Springfield")
    info.add_text("Software", "Editor 1.0")
    ex = Image.Exif()
    ex[md.ORIENTATION] = 3
    ex[md.MAKE] = "CamCo"
    ex.get_ifd(md.GPS_IFD)[1] = "N"
    ex.get_ifd(md.EXIF_IFD)[md.MAKER_NOTE] = b"MAKERNOTEDATA" + bytes(8)
    im.save(path, pnginfo=info, exif=ex, icc_profile=b"pngicc" * 30)
    data = path.read_bytes()
    chunks, _ = md._png_parse(data)
    out = [md.PNG_SIG]
    for ct, raw, _b in chunks:
        if ct == b"IEND":
            out.append(md._png_chunk(b"caNv", b"VENDORAPP5"))
            out.append(md._png_chunk(b"tIME", struct.pack(">HBBBBB", 2026, 1, 1, 1, 1, 1)))
            out.append(md._png_chunk(b"zTXt", b"Raw profile type exif\x00\x00" + zlib.compress(b"Springfield")))
            out.append(md._png_chunk(b"iTXt", b"XML:com.adobe.xmp\x00\x00\x00\x00\x00<x>XMPMAINDATA</x>"))
        out.append(raw)
    path.write_bytes(b"".join(out) + b"TRAILERMARK")
    return path


def make_loaded_webp(path):
    im = Image.new("RGB", (16, 16), (1, 2, 3))
    for x in range(16):
        im.putpixel((x, x), (x * 10, 9, 200))
    ex = Image.Exif()
    ex[md.ORIENTATION] = 8
    ex[md.MAKE] = "CamCo"
    ex.get_ifd(md.GPS_IFD)[1] = "N"
    ex.get_ifd(md.EXIF_IFD)[md.MAKER_NOTE] = b"MAKERNOTEDATA" + bytes(8)
    im.save(path, exif=ex, xmp=b"<x>XMPMAINDATA</x>", lossless=True, icc_profile=b"webpicc" * 20)
    data = path.read_bytes()
    chunks, _ = md._webp_parse(data)
    chunks.append((b"VNDR", b"VENDORAPP5"))
    body = b"WEBP" + b"".join(c + struct.pack("<I", len(b)) + b + (b"\x00" if len(b) & 1 else b"") for c, b in chunks)
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body + b"TRAILERMARK")
    return path


MAKERS = {"jpg": make_loaded_jpeg, "png": make_loaded_png, "webp": make_loaded_webp}
MODES = {
    "default": md.StripOptions(),
    "everything": md.StripOptions(everything=True),
    "location": md.StripOptions(location=True, device=False, ai_data=False, xmp=False),
    "device": md.StripOptions(location=False, device=True, ai_data=False, xmp=False),
}


def _decoded(path):
    with Image.open(path) as im:
        im.load()
        return im.size, im.tobytes(), im.getexif().get(md.ORIENTATION)


def _pixels_and_orientation(path):
    size, pixels, orient = _decoded(path)
    return size, pixels, orient


@pytest.mark.parametrize("ext", ["jpg", "png", "webp"])
def test_audit_sees_everything_before_a_clean(tmp_path, ext):
    p = MAKERS[ext](tmp_path / f"a.{ext}")
    r = md.read_metadata(p)
    assert not r.error and not r.scan_error
    cats = "+".join(f.category for f in md.privacy_findings(r))
    for c in ("location", "device", "xmp", "ai"):
        assert c in cats
    labels = {f.label for f in md.privacy_findings(r)}
    assert {"GPS location", "Camera-internal data (MakerNote)", "Data after the end of the picture"} <= labels
    if ext == "jpg":
        assert {"IPTC place names", "IPTC by-line / contact", "Embedded preview picture", "Extended XMP",
                "Content credentials (C2PA / JUMBF)", "AI generation data", "Time zone offset",
                "Other embedded data blocks"} <= labels
    if ext == "png":
        assert {"Other embedded text", "Other embedded data blocks"} <= labels


@pytest.mark.parametrize("ext", ["jpg", "png", "webp"])
@pytest.mark.parametrize("mode", ["default", "everything"])
def test_clean_removes_markers_keeps_picture(tmp_path, ext, mode):
    p = MAKERS[ext](tmp_path / f"a.{ext}")
    before = _decoded(p)
    opts = MODES[mode]
    assert md.strip_metadata(p, opts)
    raw = p.read_bytes()
    for m in MARKS:
        assert m not in raw, (ext, mode, m)
    assert b"PK\x03\x04" not in raw
    after = _decoded(p)
    assert after == before                                    # same pixels, same orientation
    r = md.read_metadata(p)
    assert not r.error and not r.scan_error and not r.gps
    assert r.has_icc                                          # colour profile kept
    assert md.leftovers(p, opts) == []
    if mode == "everything":
        assert md.privacy_findings(r) == []
        assert set(r.exif) <= {md.ORIENTATION}
    else:
        assert not md.MAKE in r.exif and md.privacy_findings(r) is not None
    assert os.listdir(tmp_path) == [p.name]                   # no temp file left behind


@pytest.mark.parametrize("ext", ["jpg", "png", "webp"])
def test_everything_with_icc_removed_leaves_only_pixels(tmp_path, ext):
    p = MAKERS[ext](tmp_path / f"a.{ext}")
    before = _decoded(p)
    md.strip_metadata(p, md.StripOptions(everything=True, remove_icc=True))
    r = md.read_metadata(p)
    assert not r.has_icc and not r.text and not r.has_xmp and md.privacy_findings(r) == []
    assert _decoded(p) == before


@pytest.mark.parametrize("ext", ["jpg", "png", "webp"])
@pytest.mark.parametrize("mode", ["location", "device"])
def test_single_category_clean(tmp_path, ext, mode):
    p = MAKERS[ext](tmp_path / f"a.{ext}")
    before = _decoded(p)
    opts = MODES[mode]
    md.strip_metadata(p, opts)
    raw = p.read_bytes()
    # a place or a person category also takes the blocks that cannot be told apart: MakerNote, previews, trailer ...
    for m in (b"MAKERNOTEDATA", b"TRAILERMARK", b"VENDORAPP5", b"Springfield", b"THUMBMARK", b"MPFINDEX", b"C2PAJUMBF",
              b"XMPEXTDATA"):
        assert m not in raw, (ext, mode, m)
    r = md.read_metadata(p)
    assert md.leftovers(p, opts) == []
    assert _decoded(p) == before
    if mode == "location":
        assert not r.gps and b"+02:00" not in raw
        assert md.MAKE in r.exif                      # device data is not asked for
    else:
        assert md.MAKE not in r.exif


def test_main_xmp_naming_a_place_goes_even_when_xmp_is_left_ticked_off(tmp_path):
    p = make_loaded_jpeg(tmp_path / "a.jpg", xmp_main=b"<x exif:GPSLatitude='52,22N'>XMPMAINDATA</x>")
    md.strip_metadata(p, md.StripOptions(location=True, device=False, ai_data=False, xmp=False))
    assert b"XMPMAINDATA" not in p.read_bytes()
    q = make_loaded_jpeg(tmp_path / "b.jpg", xmp_main=b"<x><dc:creator>someone</dc:creator>XMPMAINDATA</x>")
    md.strip_metadata(q, md.StripOptions(location=False, device=False, ai_data=False, xmp=False, timestamps=True))
    assert b"XMPMAINDATA" in q.read_bytes()           # no category covers it: the user's choice stands
    md.strip_metadata(q, md.StripOptions(location=False, device=True, ai_data=False, xmp=False))
    assert b"XMPMAINDATA" not in q.read_bytes()


def test_timestamps_only_keeps_everything_else(tmp_path):
    p = make_loaded_jpeg(tmp_path / "a.jpg")
    md.strip_metadata(p, md.StripOptions(location=False, device=False, ai_data=False, xmp=False, timestamps=True))
    raw = p.read_bytes()
    assert b"2026:01:01" not in raw and b"Springfield" in raw and b"TRAILERMARK" in raw and b"CamCo" in raw


def test_jpeg_scan_with_stuffed_bytes_restart_markers_and_progressive(tmp_path):
    # entropy data contains FF00 stuffing, restart markers and later scans: only the true EOI ends the picture
    im = Image.effect_noise((96, 96), 90).convert("RGB")
    for name, kw in (("p.jpg", {"progressive": True}), ("r.jpg", {"restart_marker_blocks": 1})):
        p = tmp_path / name
        im.save(p, quality=90, **kw)
        p.write_bytes(p.read_bytes() + b"TRAILERMARK")
        before = _decoded(p)
        md.strip_metadata(p, md.StripOptions())
        assert b"TRAILERMARK" not in p.read_bytes() and _decoded(p) == before


def test_authorship_keeps_trailer_and_other_blocks(tmp_path):
    p = make_loaded_jpeg(tmp_path / "a.jpg")
    md.apply_authorship(p, md.Authorship(artist="Demo Artist"))
    raw = p.read_bytes()
    assert b"TRAILERMARK" in raw and b"XMPMAINDATA" in raw and md.read_metadata(p).exif[md.ARTIST] == "Demo Artist"
    q = make_loaded_png(tmp_path / "a.png")
    md.set_text_field(q, "Title", "x")
    assert q.read_bytes().endswith(b"TRAILERMARK")


def test_unparseable_files_are_reported_not_called_clean(tmp_path):
    p = tmp_path / "bad.jpg"
    good = make_loaded_jpeg(tmp_path / "g.jpg").read_bytes()
    p.write_bytes(good[:2] + b"\x00\x00garbage" + good[2:])         # a broken marker stream
    r = md.read_metadata(p)
    f = md.privacy_findings(r)
    assert f and f[0].label == "Could not be checked" and f[0].category == "unchecked"
    before = p.read_bytes()
    with pytest.raises(Exception):
        md.strip_metadata(p, md.StripOptions())
    assert p.read_bytes() == before                                   # untouched on failure
    gif = tmp_path / "a.gif"
    Image.new("P", (4, 4)).save(gif)
    assert md.privacy_findings(md.read_metadata(gif))[0].label == "Could not be checked"
    trunc = tmp_path / "t.png"
    trunc.write_bytes(make_loaded_png(tmp_path / "ok.png").read_bytes()[:-30])
    assert md.privacy_findings(md.read_metadata(trunc))     # never an empty "all clear"


def test_clean_and_verify_raises_when_something_is_left(tmp_path, monkeypatch):
    pytest.importorskip("PySide6.QtWidgets")
    from tabs.metadata import clean_and_verify
    p = make_loaded_jpeg(tmp_path / "a.jpg")
    clean_and_verify(str(p), md.StripOptions())                       # clean: no error
    monkeypatch.setattr(md, "_rewrite", lambda *a, **k: False)        # a cleaner that did nothing
    q = make_loaded_jpeg(tmp_path / "b.jpg")
    with pytest.raises(RuntimeError, match="still contains"):
        clean_and_verify(str(q), md.StripOptions())


# ---------------------------------------------------------------------------- L8
def test_atomic_writer_fsyncs_and_copies_the_mode(tmp_path, monkeypatch):
    synced, moded = [], []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd))[1])
    real_mode = md.apply_mode
    monkeypatch.setattr(md, "apply_mode", lambda target, tmp: (moded.append((str(target), tmp)), real_mode(target, tmp))[1])
    p = make_loaded_jpeg(tmp_path / "a.jpg")
    if sys.platform != "win32":
        os.chmod(p, 0o640)
    md.strip_metadata(p, md.StripOptions())
    assert synced and moded and moded[0][0] == str(p)
    if sys.platform != "win32":
        assert (os.stat(p).st_mode & 0o777) == 0o640


# ---------------------------------------------------------------------------- H4
EVIL = '<img src="file:///C:/x.png"> <a href="http://example.invalid/">link</a>'


def _assert_literal(rich: str, literal_part: str):
    from PySide6.QtGui import QTextDocument
    doc = QTextDocument()
    doc.setHtml(rich)
    assert literal_part in doc.toPlainText()
    block = doc.begin()
    while block.isValid():
        it = block.begin()
        while not it.atEnd():
            fmt = it.fragment().charFormat()
            assert not fmt.isImageFormat() and not fmt.isAnchor()
            it += 1
        block = block.next()


@pytest.fixture
def tab(tmp_path, monkeypatch):
    pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from core import paths, presets
    monkeypatch.setattr(presets, "_camera_store", None)
    monkeypatch.setattr(presets, "_metadata_store", None)
    monkeypatch.setattr(paths, "CAMERA_PRESETS_FILE", tmp_path / "camera_presets.json")
    monkeypatch.setattr(paths, "METADATA_PRESETS_FILE", tmp_path / "metadata_presets.json")
    from tabs.metadata import MetadataTab
    return MetadataTab()


def test_file_text_is_plain_in_every_metadata_sink(tmp_path, tab):
    from PySide6.QtCore import Qt
    from tabs import metadata as tm
    p = make_loaded_jpeg(tmp_path / "a.jpg")
    md.set_exif_text(p, md.HOST_COMPUTER, EVIL)                       # file text lands in findings, rows and tooltips
    md.set_exif_text(p, md.SOFTWARE, "GIMP " + EVIL)
    r = md.read_metadata(p)
    r.error = EVIL
    r.path = str(tmp_path / "x <b>name.jpg")
    tab._show_report(r)
    # label: plain, literal
    assert tab.lbl_file.textFormat() == Qt.PlainText and "<b>name.jpg" in tab.lbl_file.text() \
        and EVIL in tab.lbl_file.text()
    # tooltips: whatever Qt makes of them holds no image or link and shows the characters
    tips = [tab.list_findings.item(i).toolTip() for i in range(tab.list_findings.count())]
    assert any("<img src=" in t or "&lt;img" in t for t in tips)
    for t in tips:
        _assert_literal(t, "") if t else None
    host = [t for t in tips if "file:///C:/x.png" in t]
    assert host
    for t in host:
        _assert_literal(t, '<img src="file:///C:/x.png">')
    # table cells and list items are drawn as plain text
    values = [tab.table.item(i, 1).text() for i in range(tab.table.rowCount())]
    assert any(EVIL in v for v in values)
    items = [tab.list_findings.item(i).text() for i in range(tab.list_findings.count())]
    assert any(EVIL in i for i in items)
    # other labels and message boxes
    assert tab.lbl_prompts.textFormat() == Qt.PlainText and tab.lbl_camera_note.textFormat() == Qt.PlainText
    box = tm._box(tab, tm.QMessageBox.Warning, "t", EVIL)
    assert box.textFormat() == Qt.PlainText and box.text() == EVIL
    assert tm.plain_tip(EVIL) != EVIL
    _assert_literal(tm.plain_tip(EVIL), EVIL)
