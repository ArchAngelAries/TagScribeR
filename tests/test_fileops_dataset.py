from pathlib import Path

from core import dataset, fileops
from tests.conftest import make_image


def test_natural_sort_and_hidden_skipped(tmp_path):
    for n in ("img (10).png", "img (2).png", "img (1).jpg", ".hidden.png", "notes.txt"):
        (tmp_path / n).write_bytes(b"x")
    names = [p.name for p in dataset.scan_images(tmp_path)]
    assert names == ["img (1).jpg", "img (2).png", "img (10).png"]


def test_recursive_scan(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.png").write_bytes(b"x")
    (tmp_path / "a.png").write_bytes(b"x")
    assert [p.name for p in dataset.scan_images(tmp_path, recursive=True)] == ["a.png", "b.png"]


def test_image_info_respects_exif_rotation(tmp_path):
    from PIL import Image
    p = tmp_path / "r.jpg"
    im = Image.new("RGB", (40, 20))
    exif = im.getexif()
    exif[0x0112] = 6  # rotate 90
    im.save(p, exif=exif)
    info = dataset.read_image_info(p)
    assert (info.width, info.height) == (20, 40)


def test_copy_renames_on_collision_and_keeps_caption_pair(tmp_path):
    src = make_image(tmp_path / "src" / "a.png")
    (tmp_path / "src" / "a.txt").write_text("cap")
    dest = tmp_path / "dest"
    make_image(dest / "a.png")
    (dest / "a.txt").write_text("existing")
    report = fileops.CopyReport()
    out = fileops.copy_image_with_caption(src, dest, report=report)
    assert out.name == "a (1).png"
    assert (dest / "a (1).txt").read_text() == "cap"
    assert (dest / "a.txt").read_text() == "existing"  # never overwritten
    assert len(report.renamed) == 1


def test_copy_skip_mode(tmp_path):
    src = make_image(tmp_path / "src" / "a.png")
    dest = tmp_path / "dest"
    make_image(dest / "a.png")
    report = fileops.CopyReport()
    assert fileops.copy_image_with_caption(src, dest, on_conflict="skip", report=report) is None
    assert report.skipped == [src]


def test_unique_path(tmp_path):
    (tmp_path / "f.png").write_bytes(b"")
    (tmp_path / "f (1).png").write_bytes(b"")
    assert fileops.unique_path(tmp_path / "f.png").name == "f (2).png"


def test_trash_moves_image_and_caption(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fileops, "trash", lambda p: calls.append(Path(p).name))
    img = make_image(tmp_path / "a.png")
    (tmp_path / "a.txt").write_text("c")
    report = fileops.trash_images_with_captions([img])
    assert calls == ["a.png", "a.txt"] and report.trashed == [img]
