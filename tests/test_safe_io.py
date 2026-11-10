"""Hostile safetensors headers, atomic LoRA saves and the metadata thumbnail (audit 14, L3 / L4 / L5)."""
from __future__ import annotations

import base64
import json
import os
import struct

import pytest
import torch

from training import pipeline
from training.modules.safetensors_io import MemoryEfficientSafeOpen
from training.st_header import read_header


def _write(path, header, payload=b"", n=None):
    raw = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw) if n is None else n))
        f.write(raw)
        f.write(payload)
    return str(path)


def _ok(tmp_path):
    h = {"__metadata__": {"tagscriber_next_start_window": "3"},
         "w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    return _write(tmp_path / "ok.safetensors", h, b"\0" * 8)


def test_good_file_reads(tmp_path):
    with MemoryEfficientSafeOpen(_ok(tmp_path)) as f:
        assert f.get_tensor("w").shape == (2,)
        assert f.metadata()["tagscriber_next_start_window"] == "3"


@pytest.mark.parametrize("claimed", [2 ** 62, 2 ** 64 - 1, 300 * 1024 * 1024, 10_000])
def test_claimed_length_refused(tmp_path, claimed):
    p = _write(tmp_path / "bad.safetensors", {"a": 1}, n=claimed)
    with open(p, "rb") as f:
        with pytest.raises(ValueError):
            read_header(f)
    with pytest.raises(RuntimeError):
        MemoryEfficientSafeOpen(p)
    assert pipeline.ft_source_unfit(p) == ""
    assert pipeline.ft_continuation(p) == (0, 0, 0)


def test_short_and_garbage_files(tmp_path):
    for body in (b"", b"abc", struct.pack("<Q", 2) + b"[]", struct.pack("<Q", 3) + b"\xff\xfe\xfd"):
        p = tmp_path / "g.safetensors"
        p.write_bytes(body)
        with pytest.raises(RuntimeError):
            MemoryEfficientSafeOpen(str(p))
        assert pipeline.ft_continuation(str(p)) == (0, 0, 0)


@pytest.mark.parametrize("entry", [
    {"dtype": "F32", "shape": [2], "data_offsets": [0, 10 ** 12]},        # past the end
    {"dtype": "F32", "shape": [2], "data_offsets": [-8, 0]},             # negative
    {"dtype": "F32", "shape": [2], "data_offsets": [8, 0]},              # reversed
    {"dtype": "F32", "shape": [4], "data_offsets": [0, 8]},              # shape does not match the span
    {"dtype": "F16", "shape": [2], "data_offsets": [0, 8]},              # dtype does not match the span
    {"dtype": "F32", "shape": [2], "data_offsets": ["0", "8"]},
    {"dtype": "F32", "shape": [-2], "data_offsets": [0, 8]},
    {"dtype": "F32", "shape": [2]},
])
def test_hostile_spans_refused_before_allocating(tmp_path, entry):
    p = _write(tmp_path / "s.safetensors", {"w": entry}, b"\0" * 8)
    with MemoryEfficientSafeOpen(p) as f:
        with pytest.raises(ValueError):
            f.get_tensor("w")


def test_ft_readers_use_bounded_reader(tmp_path):
    p = _write(tmp_path / "m-000004.safetensors",
               {"__metadata__": {"tagscriber_next_start_window": "2", "tagscriber_ft_n_windows": "5"},
                "a.weight_scale": {"dtype": "F32", "shape": [], "data_offsets": [0, 4]}}, b"\0" * 4)
    assert pipeline.ft_continuation(p) == (2, 4, 5)
    assert "pre-quantized" in pipeline.ft_source_unfit(p)


class _Fake:
    def __init__(self, sd):
        self._sd = sd

    def state_dict(self, dtype):
        return self._sd


def test_final_save_is_atomic(tmp_path, monkeypatch):
    import safetensors.torch as st
    from training.lora import FamilyLoRA
    path = str(tmp_path / "net.safetensors")
    FamilyLoRA.save(_Fake({"a": torch.ones(3)}), path, {"k": "v"})
    old = open(path, "rb").read()

    def boom(tensors, filename, metadata=None):
        with open(filename, "wb") as f:
            f.write(b"half")
        raise OSError("disk full")
    monkeypatch.setattr(st, "save_file", boom)
    with pytest.raises(OSError):
        FamilyLoRA.save(_Fake({"a": torch.zeros(3)}), path)
    assert open(path, "rb").read() == old
    assert not [n for n in os.listdir(tmp_path) if n.endswith(".tmp")]


def test_thumbnail_has_no_source_metadata(tmp_path):
    from PIL import Image
    from training.metadata import thumbnail_data_uri
    src = tmp_path / "s.jpg"
    im = Image.new("RGB", (64, 64), (200, 10, 10))
    exif = Image.Exif()
    exif[0x010E] = "SECRET-DESCRIPTION"
    im.save(src, format="JPEG", comment=b"SECRET-COMMENT", exif=exif)
    assert b"SECRET-COMMENT" in src.read_bytes() and b"SECRET-DESCRIPTION" in src.read_bytes()
    uri = thumbnail_data_uri(str(src))
    data = base64.b64decode(uri.split(",", 1)[1])
    assert b"SECRET-COMMENT" not in data and b"SECRET-DESCRIPTION" not in data
    assert b"Exif" not in data
