import pytest
from PIL import Image

from core import image_ops as ops


def img(w=400, h=200, mode="RGB"):
    return Image.new(mode, (w, h), (10, 20, 30) if mode == "RGB" else (10, 20, 30, 128))


@pytest.mark.parametrize("op,size", [
    (ops.Operation("rotate", {"direction": "cw"}), (200, 400)),
    (ops.Operation("rotate", {"direction": "180"}), (400, 200)),
    (ops.Operation("flip", {"axis": "v"}), (400, 200)),
    (ops.Operation("resize", {"mode": "longest", "size": 200}), (200, 100)),
    (ops.Operation("resize", {"mode": "shortest", "size": 100}), (200, 100)),
    (ops.Operation("resize", {"mode": "force", "w": 64, "h": 64}), (64, 64)),
    (ops.Operation("resize", {"mode": "scale", "percent": 50}), (200, 100)),
    (ops.Operation("crop", {"w": 100, "h": 100, "focus": "Top-Left"}), (100, 100)),
    (ops.Operation("crop_aspect", {"aspect": "1:1"}), (200, 200)),
    (ops.Operation("crop_aspect", {"aspect": "4:1"}) if "4:1" in ops.ASPECTS else
     ops.Operation("crop_aspect", {"aspect": "16:9"}), (356, 200)),
])
def test_apply_sizes_and_predicted_size(op, size):
    out = ops.apply(img(), op)
    assert out.size == size
    assert ops._result_size((400, 200), op) == size


def test_no_accidental_upscale():
    with pytest.raises(ops.SkipImage):
        ops.apply(img(100, 50), ops.Operation("resize", {"mode": "longest", "size": 400}))
    assert ops.apply(img(100, 50), ops.Operation("resize", {"mode": "longest", "size": 400,
                                                              "allow_upscale": True})).size == (400, 200)


def test_crop_too_small_skips():
    with pytest.raises(ops.SkipImage):
        ops.apply(img(50, 50), ops.Operation("crop", {"w": 100, "h": 100}))


def test_crop_focus_positions():
    base = Image.new("RGB", (300, 100), (0, 0, 0))
    base.paste((255, 0, 0), (200, 0, 300, 100))  # right third red
    right = ops.apply(base, ops.Operation("crop_aspect", {"aspect": "1:1", "focus": "Right"}))
    assert right.getpixel((50, 50)) == (255, 0, 0)
    left = ops.apply(base, ops.Operation("crop_aspect", {"aspect": "1:1", "focus": "Left"}))
    assert left.getpixel((50, 50)) == (0, 0, 0)


def test_open_upright_and_save_keeps_metadata(tmp_path):
    src = tmp_path / "a.jpg"
    im = Image.new("RGB", (40, 20), (200, 10, 10))
    exif = im.getexif()
    exif[0x0112] = 6          # rotated 90
    exif[0x010F] = "MyCam"    # Make — must survive
    im.save(src, exif=exif, icc_profile=b"fakeicc" * 10)
    up = ops.open_upright(src)
    assert up.size == (20, 40)
    out = ops.save(ops.apply(up, ops.Operation("flip")), tmp_path / "out.jpg")
    with Image.open(out) as res:
        e = res.getexif()
        assert e.get(0x010F) == "MyCam" and e.get(0x0112) == 1   # orientation reset, data kept
        assert res.info.get("icc_profile") == b"fakeicc" * 10
        assert res.size == (20, 40)


def test_convert_rgba_to_jpeg_flattens_white(tmp_path):
    rgba = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    out = ops.save(rgba, tmp_path / "x.png", fmt_name="JPG")
    assert out.suffix == ".jpg"
    with Image.open(out) as res:
        assert res.mode == "RGB" and res.getpixel((5, 5))[0] > 240


def test_save_is_atomic_no_temp_left(tmp_path):
    ops.save(img(), tmp_path / "z.png")
    assert [p.name for p in tmp_path.iterdir()] == ["z.png"]


def test_preview_reports_full_size():
    shown, size = ops.preview(img(4000, 2000), ops.Operation("resize", {"mode": "longest", "size": 1024}))
    assert size == (1024, 512) and max(shown.size) <= 640
