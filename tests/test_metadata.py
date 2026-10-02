import json
import shutil

import pytest
from PIL import Image, PngImagePlugin

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
