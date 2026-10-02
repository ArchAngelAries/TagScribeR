import pytest
from PIL import Image

from core import caption_io, image_ops
from core.export import ExportOptions, export_one


def make(tmp, name, size=(1600, 1200), caption="1girl, solo", exif=True):
    p = tmp / "src" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, (100, 150, 200))
    kw = {}
    if exif:
        ex = im.getexif()
        ex[0x010F] = "CamCo"
        kw["exif"] = ex
    im.save(p, **kw)
    if caption:
        caption_io.write_caption(p, caption)
    return p


def test_bucket_export_kohya_trigger_strip(tmp_path):
    src = make(tmp_path, "a.jpg")
    before = src.read_bytes()
    opts = ExportOptions(out_dir=str(tmp_path / "out"), kohya_folder=True, repeats=8, concept="ohwx",
                         resize="bucket", resolution=1024, trigger="ohwx woman", rename=True, fmt="PNG")
    dest = export_one(src, 1, opts)
    assert dest == tmp_path / "out" / "8_ohwx" / "img_0001.png"
    with Image.open(dest) as im:
        assert im.size == (1152, 896) or (im.size[0] % 64 == 0 and im.size[1] % 64 == 0)
        assert 0x010F not in im.getexif()        # metadata stripped
    assert (dest.with_suffix(".txt")).read_text(encoding="utf-8") == "ohwx woman, 1girl, solo"
    assert src.read_bytes() == before            # source untouched


def test_skip_uncaptioned_and_small(tmp_path):
    opts = ExportOptions(out_dir=str(tmp_path / "out"))
    with pytest.raises(image_ops.SkipImage):
        export_one(make(tmp_path, "nocap.png", caption=""), 1, opts)
    with pytest.raises(image_ops.SkipImage):
        export_one(make(tmp_path, "tiny.png", size=(300, 300)), 2, opts)
    opts.allow_upscale = True
    assert export_one(make(tmp_path, "tiny2.png", size=(300, 300)), 3, opts).exists()


def test_never_overwrites(tmp_path):
    src = make(tmp_path, "a.png")
    opts = ExportOptions(out_dir=str(tmp_path / "out"), resize="none")
    first = export_one(src, 1, opts)
    second = export_one(src, 1, opts)
    assert first != second and first.exists() and second.exists()
    assert second.with_suffix(".txt").exists()


def test_longest_side_mode_keeps_aspect(tmp_path):
    src = make(tmp_path, "w.png", size=(3000, 1000))
    opts = ExportOptions(out_dir=str(tmp_path / "out"), resize="longest", resolution=1500, strip_metadata=False)
    with Image.open(export_one(src, 1, opts)) as im:
        assert im.size == (1500, 500)


def test_slightly_small_images_are_not_skipped(tmp_path):
    src = make(tmp_path, "near.png", size=(1024, 1021))       # needs a 0.3% upscale for the 1024 bucket
    with Image.open(export_one(src, 1, ExportOptions(out_dir=str(tmp_path / "out")))) as im:
        assert im.size == (1024, 1024)
