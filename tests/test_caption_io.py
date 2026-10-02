from core import caption_io as io_


def test_read_missing_returns_empty(tmp_path):
    assert io_.read_caption(tmp_path / "x.png") == ""


def test_write_empty_does_not_create_file(tmp_path):
    img = tmp_path / "a.png"
    assert io_.write_caption(img, "   ") is False
    assert not (tmp_path / "a.txt").exists()


def test_write_and_read_roundtrip_utf8(tmp_path):
    img = tmp_path / "a.png"
    assert io_.write_caption(img, "café, 1girl") is True
    assert io_.read_caption(img) == "café, 1girl"
    assert io_.write_caption(img, "café, 1girl") is False  # unchanged -> no write


def test_bom_and_cp1252_tolerated(tmp_path):
    (tmp_path / "bom.txt").write_bytes("﻿hello".encode("utf-8"))
    (tmp_path / "old.txt").write_bytes("caf\xe9".encode("latin-1"))
    assert io_.read_caption(tmp_path / "bom.png") == "hello"
    assert io_.read_caption(tmp_path / "old.png") == "café"


def test_overwrite_creates_daily_backup_of_original(tmp_path):
    img = tmp_path / "a.png"
    io_.write_caption(img, "original")
    io_.write_caption(img, "second")
    io_.write_caption(img, "third")
    backup = io_.backup_path_for(tmp_path / "a.txt")
    assert backup.read_text(encoding="utf-8") == "original"  # earliest of the day kept
    assert io_.read_caption(img) == "third"


def test_clearing_existing_caption_writes_empty(tmp_path):
    img = tmp_path / "a.png"
    io_.write_caption(img, "x")
    assert io_.write_caption(img, "") is True
    assert (tmp_path / "a.txt").read_text() == ""


def test_atomic_write_leaves_no_temp_files(tmp_path):
    io_.atomic_write_text(tmp_path / "f.txt", "data")
    assert [p.name for p in tmp_path.iterdir()] == ["f.txt"]
