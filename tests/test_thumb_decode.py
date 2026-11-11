"""The Qt thumbnail decode (core.image_utils._qt_load_qimage) gives the picture the Pillow path gives: same size, same
orientation, pixels within a small tolerance; and it steps aside for everything it is not trusted with."""
import os

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6.QtGui")
from core import image_utils as iu  # noqa: E402

TIER = (384, 384)


def _picture(w, h, mode="RGB"):
    """A smooth, asymmetric picture: four colour regions with a diagonal gradient, so a wrong orientation or a
    wrong scale shows as a large difference."""
    yy, xx = np.mgrid[0:h, 0:w]
    r = (xx * 255 // max(1, w - 1)).astype(np.uint8)
    g = (yy * 255 // max(1, h - 1)).astype(np.uint8)
    b = (((xx + yy) * 255) // max(1, w + h - 2)).astype(np.uint8)
    arr = np.stack([r, g, b], axis=-1)
    arr[: h // 3, : w // 4] = (250, 20, 20)           # a red block top-left: marks the orientation
    arr[h * 2 // 3:, w * 3 // 4:] = (20, 20, 250)     # a blue block bottom-right
    img = Image.fromarray(arr, "RGB")
    if mode == "L":
        return img.convert("L")
    if mode == "RGBA":
        img = img.convert("RGBA")
        img.putalpha(Image.linear_gradient("L").resize((w, h)))
    return img


def _rgba(qimg):
    h, w, bpl = qimg.height(), qimg.width(), qimg.bytesPerLine()
    return np.frombuffer(qimg.constBits(), np.uint8).reshape(h, bpl)[:, : w * 4].reshape(h, w, 4).astype(int)


def _compare(path, size=TIER):
    fast = iu._qt_load_qimage(path, size)
    assert fast is not None, f"{path} should take the Qt path"
    ref_img, ref_orig = iu._pillow_load_qimage(path, size)
    q_img, q_orig = fast
    assert (q_img.width(), q_img.height()) == (ref_img.width(), ref_img.height())
    assert q_orig == ref_orig
    a, b = _rgba(q_img), _rgba(ref_img)
    diff = np.abs(a - b)
    assert diff.mean() < 4.0, f"mean difference {diff.mean():.2f}"
    assert np.percentile(diff, 99) < 40, f"99th percentile {np.percentile(diff, 99):.1f}"
    return q_img


@pytest.mark.parametrize("orientation", range(1, 9))
def test_every_exif_orientation(tmp_path, orientation):
    exif = Image.Exif()
    exif[0x0112] = orientation
    p = tmp_path / "o.jpg"
    _picture(900, 600).save(p, quality=92, exif=exif)
    _compare(p)


@pytest.mark.parametrize("orientation", (1, 6, 8))
def test_small_picture_is_not_enlarged(tmp_path, orientation):
    exif = Image.Exif()
    exif[0x0112] = orientation
    p = tmp_path / "small.jpg"
    _picture(200, 120).save(p, quality=95, exif=exif)
    img = _compare(p)
    assert max(img.width(), img.height()) == 200


@pytest.mark.parametrize("name,mode,kw", [
    ("a.png", "RGB", {}), ("b.png", "RGBA", {}), ("c.png", "L", {}),
    ("d.jpg", "RGB", {"quality": 85}), ("e.jpg", "L", {"quality": 85}),
    ("f.webp", "RGB", {"quality": 90}), ("g.webp", "RGBA", {"quality": 90}),
])
def test_formats_and_modes(tmp_path, name, mode, kw):
    p = tmp_path / name
    _picture(1000, 640, mode).save(p, **kw)
    _compare(p)


@pytest.mark.parametrize("size", [(1000, 7), (7, 1000), (1001, 667), (383, 385), (5000, 3000)])
def test_odd_aspect_ratios_give_pillows_size(tmp_path, size):
    p = tmp_path / "x.png"
    _picture(*size).save(p)
    _compare(p)


def test_other_tiers(tmp_path):
    p = tmp_path / "t.jpg"
    _picture(2000, 1500).save(p, quality=90)
    for tier in (192, 384, 768):
        _compare(p, (tier, tier))


def test_untrusted_pictures_take_the_pillow_path(tmp_path):
    arr16 = (np.arange(64 * 48, dtype=np.uint16).reshape(48, 64) * 40)
    Image.fromarray(arr16).save(tmp_path / "sixteen.png")
    _picture(80, 60).convert("P").save(tmp_path / "palette.png")
    _picture(80, 60).convert("CMYK").save(tmp_path / "cmyk.jpg")
    frames = [_picture(64, 48), _picture(64, 48).rotate(90)]
    frames[0].save(tmp_path / "anim.gif", save_all=True, append_images=frames[1:])
    frames[0].save(tmp_path / "anim.webp", save_all=True, append_images=frames[1:])
    _picture(80, 60).save(tmp_path / "profile.jpg", icc_profile=b"\x00" * 128)
    _picture(80, 60).save(tmp_path / "key.png", transparency=(250, 20, 20))
    _picture(80, 60).save(tmp_path / "plain.bmp")
    for name in ("sixteen.png", "palette.png", "cmyk.jpg", "anim.gif", "anim.webp", "profile.jpg", "key.png",
                 "plain.bmp"):
        assert iu._qt_load_qimage(tmp_path / name, TIER) is None, name
        img, orig = iu.load_qimage(tmp_path / name, TIER)        # and the public call still works for all of them
        assert not img.isNull()
        if name.startswith("anim"):
            assert orig == (64, 48)


def test_the_pixel_guard_still_applies(tmp_path, monkeypatch):
    monkeypatch.setattr(iu, "MAX_THUMBNAIL_PIXELS", 10_000)
    _picture(200, 150).save(tmp_path / "big.png")
    _picture(1200, 900).save(tmp_path / "big.jpg", quality=90)
    with pytest.raises(iu.ImageTooLarge):
        iu.load_qimage(tmp_path / "big.png", TIER)
    with pytest.raises(iu.ImageTooLarge):
        iu.load_qimage(tmp_path / "big.jpg", (100, 100))     # draft-reduced to 600x450 (still above the limit)
    # a JPEG that the draft reduces under the limit is fine, as on the Pillow path
    monkeypatch.setattr(iu, "MAX_THUMBNAIL_PIXELS", 700 * 700)
    img, _orig = iu.load_qimage(tmp_path / "big.jpg", (100, 100))
    assert max(img.width(), img.height()) == 100


def test_a_corrupt_file_falls_back_and_fails_like_before(tmp_path):
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"not a picture" * 10)
    with pytest.raises(Exception):
        iu.load_qimage(p, TIER)
    assert os.path.exists(p)
