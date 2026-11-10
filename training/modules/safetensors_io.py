# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/krea2/safetensors_utils.py
# (itself adapted from musubi-tuner, https://github.com/kohya-ss/musubi-tuner, Apache-2.0).
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: only the two pieces the full fine-tune needs (the lazy header reader and the streamed
# writer); mem_eff_save_file and the pinned-memory / memmap GPU paths of get_tensor are left out (every read here lands
# on the CPU, one tensor at a time); otherwise unchanged.
"""Lazy safetensors reading and streamed writing for files far larger than RAM.

MemoryEfficientSafeOpen reads the header once and one tensor at a time on demand (a 26 GB checkpoint is never
loaded whole); stream_save_file writes a file from lazy producers, one tensor in memory at a time.
"""
from __future__ import annotations

import json
import os
import struct
from typing import Any, Dict, Optional

import numpy as np
import torch

from training.st_header import check_span, read_header

_TYPES = {torch.float64: "F64", torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
          torch.int64: "I64", torch.int32: "I32", torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8",
          torch.bool: "BOOL"}
_ELEM = {torch.float64: 8, torch.float32: 4, torch.float16: 2, torch.bfloat16: 2, torch.int64: 8, torch.int32: 4,
         torch.int16: 2, torch.int8: 1, torch.uint8: 1, torch.bool: 1}
for _name, _tag in (("float8_e5m2", "F8_E5M2"), ("float8_e4m3fn", "F8_E4M3")):
    _dt = getattr(torch, _name, None)
    if _dt is not None:
        _TYPES[_dt] = _tag
        _ELEM[_dt] = 1
_DTYPES = {v: k for k, v in _TYPES.items()}
_ALIGN = 256


def torch_dtype(dtype_str: str) -> Optional[torch.dtype]:
    """The torch dtype a safetensors header names, or None."""
    return _DTYPES.get(dtype_str)


def stream_save_file(specs, filename: str, metadata: Dict[str, Any] = None):
    """Write a safetensors file from LAZY tensor producers - one tensor in memory at a time.

    The safetensors format needs the full header (dtype / shape / offsets) before the first payload byte, but shapes
    are known without materialising anything, so the caller declares them up front and each tensor is produced only
    when it is that key's turn to hit the disk, then freed.

    specs: ordered dict {key: (torch_dtype, shape_tuple, producer)} where producer is a zero-argument callable
    returning the tensor (any device). Produced dtype / shape are checked against the declaration: a mismatch would
    silently corrupt every later offset in the file. Atomicity is the caller's job (write to path + '.tmp', then
    os.replace)."""
    header = {}
    offset = 0
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    for k, (dt, shape, _p) in specs.items():
        numel = 1
        for d in shape:
            numel *= int(d)
        size = numel * _ELEM[dt]
        header[k] = {"dtype": _TYPES[dt], "shape": [int(d) for d in shape], "data_offsets": [offset, offset + size]}
        offset += size
    hjson = json.dumps(header).encode("utf-8")
    hjson += b" " * (-(len(hjson) + 8) % _ALIGN)
    with open(filename, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for k, (dt, shape, producer) in specs.items():
            v = producer()
            if v.dtype != dt or tuple(v.shape) != tuple(shape):
                raise RuntimeError(f"stream_save_file: producer for {k!r} returned {v.dtype}/{tuple(v.shape)}, "
                                   f"declared {dt}/{tuple(shape)} - every later offset in the file would be wrong")
            if v.numel() == 0:
                continue
            if v.dim() == 0:
                v = v.unsqueeze(0)
            v = v.contiguous()
            if v.is_cuda:
                v = v.cpu()
            v.view(torch.uint8).numpy().tofile(f)
            del v


class MemoryEfficientSafeOpen:
    """A safetensors reader that parses the header once and reads one tensor at a time on request (`header` is the
    parsed JSON: {key: {dtype, shape, data_offsets}}). Context manager; `file` is the open handle."""

    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, "rb")
        self.header, self.header_size = self._read_header()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def keys(self):
        return [k for k in self.header.keys() if k != "__metadata__"]

    def metadata(self) -> Dict[str, str]:
        return self.header.get("__metadata__", {})

    def _read_header(self):
        # bounded (TagScribeR): a hostile length or a cut-short file is a clear error, not an allocation
        try:
            header, data_start = read_header(self.file)
        except ValueError as e:
            self.file.close()
            raise RuntimeError(f"{self.filename}: not a readable safetensors file ({e})") from e
        return header, data_start - 8

    def get_tensor(self, key: str, device=None, dtype: Optional[torch.dtype] = None):
        """The tensor under `key`, read from the file into memory (CPU), optionally cast / moved."""
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")
        meta = self.header[key]
        if not isinstance(meta, dict):
            raise ValueError(f"Tensor {key!r} has no entry in the header")
        src_dtype = torch_dtype(meta.get("dtype"))
        if src_dtype is None:
            raise ValueError(f"Unsupported safetensors dtype {meta.get('dtype')!r} for {key!r}")
        payload = os.fstat(self.file.fileno()).st_size - self.header_size - 8
        try:
            start, end = check_span(meta, _ELEM[src_dtype], payload)
        except ValueError as e:
            raise ValueError(f"Tensor {key!r}: {e}") from e
        n = end - start
        target = dtype if dtype is not None else src_dtype
        if n == 0:
            return torch.empty(meta["shape"], dtype=target, device=device)
        self.file.seek(self.header_size + 8 + start)
        arr = np.fromfile(self.file, dtype=np.uint8, count=n)
        t = torch.from_numpy(arr).view(src_dtype).reshape(meta["shape"])
        del arr
        return t.to(device=device, dtype=target)
