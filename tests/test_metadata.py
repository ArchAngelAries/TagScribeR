import json
from fractions import Fraction
import shutil

import pytest
from PIL import Image, PngImagePlugin
from PIL.TiffImagePlugin import IFDRational

from core import metadata as md

A1111 = ("masterpiece, 1girl, red hair\nNegative prompt: lowres, bad hands\n"
         "Steps: 28, Sampler: Euler a, CFG scale: 7, Seed: 1234, Model: ponyXL")
COMFY = json.dumps({
    "3": {"class_type": "KSampler", "inputs": {"positive": ["6", 0], "negative": ["7", 0]}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a castle at dusk, oil painting"}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, watermark, extra long negative prompt text"}},
})


def make_jpeg(path):
    im = Image.new("RGB", (64, 32), (120, 30, 200))
    for x in range(64):  # some detail so pixel comparison is meaningful
        im.putpixel((x, x % 32), (x * 3, 255 - x * 2, 40))
    ex = im.getexif()
    ex[md.ORIENTATION] = 6
    ex[md.MAKE] = "CamCo"
    ex[md.MODEL] = "X100"
    ex[md.SOFTWARE] = "Editor 1.0"
    ex[md.DATETIME] = "2026:01:01 10:00:00"
    sub = ex.get_ifd(md.EXIF_IFD)
    sub[md.BODY_SERIAL] = "SN123456"
    sub[md.USER_COMMENT] = b"ASCII\x00\x00\x00" + A1111.encode()
    gps = ex.get_ifd(md.GPS_IFD)
    gps[1] = "N"
    gps[2] = (52.0, 22.0, 1.0)
    xmp = (b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description '
           b'Iptc4xmpExt:DigitalSourceType="trainedAlgorithmicMedia"/></rdf:RDF></x:xmpmeta>')
    im.save(path, exif=ex, icc_profile=b"icc" * 40, xmp=xmp, comment=b"seed 1234", quality=95)
    return path


def make_png(path, text=None, exif=True):
    im = Image.new("RGBA", (40, 30), (10, 200, 30, 180))
    info = PngImagePlugin.PngInfo()
    for k, v in (text or {"parameters": A1111, "Software": "Forge"}).items():
        info.add_text(k, v)
    kwargs = {"pnginfo": info, "icc_profile": b"pngicc" * 30}
    if exif:
        ex = Image.Exif()
        ex[md.MAKE] = "PhoneCo"
        gps = ex.get_ifd(md.GPS_IFD)
        gps[1] = "S"
        kwargs["exif"] = ex
    im.save(path, **kwargs)
    return path


def test_read_and_findings_jpeg(tmp_path):
    r = md.read_metadata(make_jpeg(tmp_path / "a.jpg"))
    labels = {f.label for f in md.privacy_findings(r)}
    assert {"GPS location", "Device serial numbers", "AI generation data", "XMP metadata",
            "Camera / device model", "Timestamps", "Software"} <= labels
    assert any("AI/provenance" in f.detail for f in md.privacy_findings(r))
    assert md.extract_prompt(r) == "masterpiece, 1girl, red hair"


def test_strip_default_jpeg_is_lossless_and_keeps_orientation_icc(tmp_path):
    src = make_jpeg(tmp_path / "a.jpg")
    ref = shutil.copy(src, tmp_path / "ref.jpg")
    assert md.strip_metadata(src, md.StripOptions())
    r = md.read_metadata(src)
    assert not r.gps and md.BODY_SERIAL not in r.exif and md.MAKE not in r.exif
    assert md.USER_COMMENT not in r.exif and not r.has_xmp and "comment" not in r.text
    assert r.exif.get(md.ORIENTATION) == 6 and r.has_icc
    assert md.DATETIME in r.exif  # timestamps kept unless asked
    assert md.pixels_equal(src, ref)  # no re-encode


def test_strip_everything_jpeg(tmp_path):
    src = make_jpeg(tmp_path / "a.jpg")
    ref = shutil.copy(src, tmp_path / "ref.jpg")
    md.strip_metadata(src, md.StripOptions(everything=True))
    r = md.read_metadata(src)
    assert set(r.exif) == {md.ORIENTATION} and r.has_icc and not r.has_xmp
    assert md.pixels_equal(src, ref)


def test_strip_png_chunks_lossless(tmp_path):
    src = make_png(tmp_path / "a.png")
    ref = shutil.copy(src, tmp_path / "ref.png")
    md.strip_metadata(src, md.StripOptions())
    r = md.read_metadata(src)
    assert "parameters" not in r.text and r.text.get("Software") == "Forge"  # software kept by default
    assert not r.gps and md.MAKE not in r.exif and r.has_icc
    assert md.pixels_equal(src, ref)
    md.strip_metadata(src, md.StripOptions(everything=True, remove_icc=True))
    r = md.read_metadata(src)
    assert not r.text and not r.has_icc and md.pixels_equal(src, ref)


def test_comfy_prompt_follows_positive_link(tmp_path):
    src = make_png(tmp_path / "c.png", {"prompt": COMFY, "workflow": "{}"}, exif=False)
    assert md.extract_prompt(md.read_metadata(src)) == "a castle at dusk, oil painting"


def test_invoke_and_none(tmp_path):
    src = make_png(tmp_path / "i.png", {"invokeai_metadata": json.dumps({"positive_prompt": "a fox"})}, exif=False)
    assert md.extract_prompt(md.read_metadata(src)) == "a fox"
    plain = make_png(tmp_path / "p.png", {"Title": "x"}, exif=False)
    assert md.extract_prompt(md.read_metadata(plain)) is None


def test_authorship_jpeg_png_webp(tmp_path):
    author = md.Authorship(artist="Ari Example", copyright="© 2026 Ari Example, CC BY-NC 4.0",
                           description="Original digital painting", software="Krita")
    j = make_jpeg(tmp_path / "a.jpg")
    ref = shutil.copy(j, tmp_path / "ref.jpg")
    md.apply_authorship(j, author)
    r = md.read_metadata(j)
    assert r.exif[md.ARTIST] == "Ari Example" and r.exif[md.SOFTWARE] == "Krita"
    assert r.exif.get(md.MAKE) == "CamCo"  # authorship alone doesn't remove other data
    assert md.pixels_equal(j, ref)
    p = make_png(tmp_path / "a.png", exif=False)
    md.apply_authorship(p, author)
    r = md.read_metadata(p)
    assert r.text["Author"] == "Ari Example" and r.text["Copyright"].startswith("©")
    assert r.exif[md.COPYRIGHT] == "(c) 2026 Ari Example, CC BY-NC 4.0"  # EXIF is ASCII; PNG text keeps ©
    w = tmp_path / "a.webp"
    im = Image.new("RGB", (16, 16), (1, 2, 3))
    ex = Image.Exif()
    ex[md.MAKE] = "X"
    im.save(w, exif=ex, lossless=True)
    md.apply_authorship(w, author)
    assert md.read_metadata(w).exif[md.ARTIST] == "Ari Example"
    md.strip_metadata(w, md.StripOptions(everything=True))
    assert md.ARTIST not in md.read_metadata(w).exif
    with Image.open(w) as chk:
        chk.load()  # still a valid WebP


def test_set_text_and_exif_fields(tmp_path):
    p = make_png(tmp_path / "a.png", exif=False)
    md.set_text_field(p, "parameters", "edited prompt")
    assert md.read_metadata(p).text["parameters"] == "edited prompt"
    md.set_text_field(p, "parameters", "")
    assert "parameters" not in md.read_metadata(p).text
    j = make_jpeg(tmp_path / "a.jpg")
    md.set_exif_text(j, md.ARTIST, "Me")
    assert md.read_metadata(j).exif[md.ARTIST] == "Me"
    assert md.exif_ascii("Zoë © 2026 — “x”") == 'Zoe (c) 2026 - "x"'


def test_unsupported_format_raises(tmp_path):
    b = tmp_path / "a.bmp"
    Image.new("RGB", (4, 4)).save(b)
    with pytest.raises(ValueError):
        md.strip_metadata(b, md.StripOptions())


def test_rows_marks_editable_fields(tmp_path):
    p = make_png(tmp_path / "a.png", exif=False)
    rows = md.read_metadata(p).rows()
    assert any(edit == "text:parameters" for _k, _v, edit in rows)


# ---------------------------------------------------------------------------- camera data
def _camera_files(tmp_path):
    j = make_jpeg(tmp_path / "a.jpg")
    p = make_png(tmp_path / "a.png", exif=False)
    w = tmp_path / "a.webp"
    im = Image.new("RGB", (16, 16), (1, 2, 3))
    ex = Image.Exif()
    ex[md.ORIENTATION] = 3
    im.save(w, exif=ex, lossless=True)
    return [j, p, w]


def _cam():
    from core.camera_presets import CAMERA_PRESETS, to_camera
    return to_camera(CAMERA_PRESETS["Sony a7 IV (ILCE-7M4) + FE 24-70mm F2.8 GM II"])


def test_apply_camera_all_formats(tmp_path):
    cam = _cam()
    cam.date_time_original = "2026-05-14 16:30:00"
    cam.exposure_bias = "-2/3"
    for f in _camera_files(tmp_path):
        ref = shutil.copy(f, tmp_path / ("ref" + f.suffix))
        before = md.read_metadata(f)
        assert md.apply_camera(f, cam)
        r = md.read_metadata(f)
        assert r.exif[md.MAKE] == "SONY" and r.exif[md.MODEL] == "ILCE-7M4"
        assert r.exif[md.SOFTWARE] == "ILCE-7M4 v1.10"
        assert r.exif[md.LENS_MODEL] == "FE 24-70mm F2.8 GM II"
        assert float(r.exif[md.F_NUMBER]) == 2.8 and float(r.exif[md.EXPOSURE_TIME]) == 1 / 250
        assert float(r.exif[md.FOCAL_LENGTH]) == 35 and float(r.exif[md.EXPOSURE_BIAS]) == pytest.approx(-2 / 3, abs=1e-3)
        assert r.exif[md.ISO_SPEED] == 200 and r.exif[md.FOCAL_35MM] == 35
        assert r.exif[md.EXPOSURE_PROGRAM] == 3 and r.exif[md.METERING_MODE] == 5
        assert r.exif[md.FLASH] == 16 and r.exif[md.WHITE_BALANCE] == 0
        assert r.exif[md.DT_ORIGINAL] == r.exif[md.DT_DIGITIZED] == "2026:05:14 16:30:00"
        # orientation / ICC / pixels as before; nothing trackable invented
        assert r.exif.get(md.ORIENTATION) == before.exif.get(md.ORIENTATION)
        assert r.has_icc == before.has_icc and md.pixels_equal(f, ref)
        ids = {md.BODY_SERIAL, md.LENS_SERIAL, md.OWNER_NAME, md.UNIQUE_ID}
        assert (ids & set(r.exif)) <= (ids & set(before.exif))  # nothing trackable is added
        assert isinstance(r.exif[md.F_NUMBER], IFDRational) and isinstance(r.exif[md.ISO_SPEED], int)
        with Image.open(f) as chk:
            chk.load()


def test_apply_camera_no_trackable_ids_and_keeps_other_data(tmp_path):
    p = make_png(tmp_path / "a.png", exif=False)
    md.apply_camera(p, _cam())
    r = md.read_metadata(p)
    assert not r.gps and not ({md.BODY_SERIAL, md.LENS_SERIAL, md.OWNER_NAME, md.UNIQUE_ID} & set(r.exif))
    j = make_jpeg(tmp_path / "a.jpg")  # a file that already carries a serial keeps it: we only add, never invent
    md.apply_camera(j, md.CameraData(model="Z"))
    r = md.read_metadata(j)
    assert r.exif[md.MODEL] == "Z" and r.exif[md.MAKE] == "CamCo" and r.exif[md.SOFTWARE] == "Editor 1.0"
    assert md.F_NUMBER not in r.exif  # blank fields unchanged
    assert not md.apply_camera(j, md.CameraData())


def test_camera_variation_deterministic_and_in_range(tmp_path):
    cam = _cam()
    var = md.Variation(shutter="1/1000-1/125", iso="100-800", f_number="2.8-5.6")
    seen = set()
    for i in range(40):
        v = md.vary_camera(cam, f"img_{i}.jpg", var)
        assert v == md.vary_camera(cam, f"img_{i}.jpg", var)
        t = Fraction(v.exposure_time)
        assert Fraction(1, 1000) <= t <= Fraction(1, 125)
        assert int(v.iso) in (100, 125, 160, 200, 250, 320, 400, 500, 640, 800)
        assert v.f_number in ("2.8", "3.2", "3.5", "4", "4.5", "5", "5.6")
        seen.add((v.exposure_time, v.iso, v.f_number))
    assert len(seen) > 5
    f = make_jpeg(tmp_path / "x.jpg")
    md.apply_camera(f, cam, var)
    first = md.read_metadata(f).exif[md.ISO_SPEED]
    md.apply_camera(f, cam, var)
    assert md.read_metadata(f).exif[md.ISO_SPEED] == first
    fixed = md.vary_camera(cam, "a.jpg", md.Variation(iso="100-800"))
    assert fixed.f_number == cam.f_number and fixed.exposure_time == cam.exposure_time
    with pytest.raises(ValueError):
        md.validate_camera(cam, md.Variation(iso="fast"))


def test_camera_timestamp_series(tmp_path):
    series = md.TimeSeries(start="2026-05-14 23:59:30", step_seconds=45)
    files = [make_jpeg(tmp_path / f"{i}.jpg") for i in range(3)]
    for i, f in enumerate(files):
        md.apply_camera(f, md.CameraData(), series=series, index=i)
    got = [md.read_metadata(f).exif[md.DT_ORIGINAL] for f in files]
    assert got == ["2026:05:14 23:59:30", "2026:05:15 00:00:15", "2026:05:15 00:01:00"]
    with pytest.raises(ValueError):
        md.validate_camera(md.CameraData(), series=md.TimeSeries(start="yesterday"))


def test_strip_device_removes_camera_data(tmp_path):
    for f in _camera_files(tmp_path):
        md.apply_camera(f, _cam())
        md.strip_metadata(f, md.StripOptions())
        r = md.read_metadata(f)
        assert not ({md.MAKE, md.MODEL, md.LENS_MAKE, md.LENS_MODEL} & set(r.exif))


def test_camera_presets_are_valid():
    from core.camera_presets import CAMERA_PRESETS, to_camera
    from core.presets import camera_presets
    assert len(CAMERA_PRESETS) >= 16
    for name, preset in CAMERA_PRESETS.items():
        cam = to_camera(preset)
        var = md.Variation(preset["shutter_range"], preset["iso_range"], preset["f_range"])
        md.validate_camera(cam, var, md.TimeSeries(start="2026-01-01 10:00:00"))
        assert cam.make and cam.model and preset["note"], name
    assert camera_presets().names()[:len(CAMERA_PRESETS)] == list(CAMERA_PRESETS)


def test_camera_section_builds_and_preset_fills(tmp_path, monkeypatch):
    pytest.importorskip("PySide6.QtWidgets")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from core import presets
    monkeypatch.setattr(presets, "_camera_store", None)
    monkeypatch.setattr(presets, "_metadata_store", None)
    from core import paths
    monkeypatch.setattr(paths, "CAMERA_PRESETS_FILE", tmp_path / "camera_presets.json")
    monkeypatch.setattr(paths, "METADATA_PRESETS_FILE", tmp_path / "metadata_presets.json")
    from tabs.metadata import MetadataTab
    tab = MetadataTab()
    name = "Apple iPhone 15 Pro (main camera)"
    tab.combo_camera.setCurrentIndex(tab.combo_camera.findData(name))
    tab._load_camera(name)
    assert tab.cam_inputs["model"].text() == "iPhone 15 Pro" and tab.cam_inputs["focal_length_35mm"].text() == "24"
    assert tab.cam_combos["exposure_program"].currentData() == "2"
    assert tab._camera().lens_model.startswith("iPhone 15 Pro back triple camera")
    assert tab._variation() is None
    tab.chk_cam_vary.setChecked(True)
    assert tab._variation().iso == "50-1600"
    tab.cam_inputs["model"].setText("My Cam")
    saved = tab.cam_presets.save("Mine", {**tab._camera_values(), "iso_range": "100-400"})
    tab._fill_camera_presets(saved)
    assert tab.combo_camera.currentData() == "Mine" and not tab.cam_presets.is_builtin("Mine")
    app.processEvents()


def test_exif_blocks_carry_exactly_one_header(tmp_path):
    """Regression (docs/dev/audit/09): Pillow's Exif.tobytes() starts with Exif\\0\\0; JPEG APP1 must carry it once,
    PNG eXIf and WebP EXIF not at all, or readers other than Pillow see no EXIF."""
    from PIL import Image
    for ext in ("jpg", "png", "webp"):
        p = tmp_path / f"a.{ext}"
        ex = Image.Exif()
        ex[md.ORIENTATION] = 6
        Image.new("RGB", (16, 12), (9, 99, 199)).save(p, exif=ex)
        md.apply_authorship(p, md.Authorship(artist="Demo"))
        data = p.read_bytes()
        if ext == "jpg":
            assert data.count(b"Exif\x00\x00") == 1
            i = data.find(b"Exif\x00\x00") + 6
        else:
            i = data.find(b"eXIf" if ext == "png" else b"EXIF") + (4 if ext == "png" else 8)
            assert b"Exif\x00\x00" not in data
        assert data[i:i + 2] in (b"II", b"MM")
        r = md.read_metadata(p)
        assert r.exif.get(md.ARTIST) == "Demo" and r.exif.get(md.ORIENTATION) == 6
