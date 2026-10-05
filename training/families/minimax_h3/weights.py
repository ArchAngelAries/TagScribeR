# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/minimax/convrot.py (the ConvRot decode).
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: only the decode half of convrot.py is kept (the encode direction, the stochastic-rounding
# save and the Triton kernels belong to Fizgig's rotation fine-tune and are not ported); the safetensors reader below is
# new (a plain seek-and-read per tensor, the way Fizgig's MemoryEfficientSafeOpen avoids the Windows mmap crash on
# 20-60 GB files) - Fizgig's reader is musubi-tuner code (Apache-2.0) and is not copied.
"""Reading the MiniMax H3 checkpoints: a tensor-at-a-time safetensors reader and the ConvRot int8 decode.

`minimax_h3_*_pruned_int8_convrot.safetensors` stores every big block Linear as int8 in a ROTATED basis, marked by a
`<module>.comfy_quant` blob {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}:

    weight        int8   [out, in]   codes of  rotate(W, G)
    weight_scale  fp32   [out, 1]    per-output-row symmetric scale

ConvRot (arXiv:2512.03673) folds a block regular-Hadamard rotation into the weight offline and applies the same
rotation to the activation at runtime; being orthogonal it cancels in the matmul. The regular Hadamard is symmetric and
orthogonal, i.e. its own inverse, so "unrotate" is the same operation as "rotate".
"""
import json
import struct

import torch

_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
    "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
}


class SafeReader:
    """One safetensors file, tensor by tensor, with plain file reads (no memory map). Context manager."""

    def __init__(self, path):
        self.path = str(path)
        self._f = None
        self._header = None
        self._base = 0

    def __enter__(self):
        self._f = open(self.path, "rb")
        (n,) = struct.unpack("<Q", self._f.read(8))
        self._header = json.loads(self._f.read(n).decode("utf-8"))
        self._header.pop("__metadata__", None)
        self._base = 8 + n
        return self

    def __exit__(self, *exc):
        if self._f is not None:
            self._f.close()
            self._f = None

    def keys(self):
        return self._header.keys()

    def shape(self, name):
        return tuple(self._header[name]["shape"])

    def dtype(self, name):
        return _DTYPES[self._header[name]["dtype"]]

    def get_tensor(self, name) -> torch.Tensor:
        e = self._header[name]
        start, end = e["data_offsets"]
        raw = torch.empty(end - start, dtype=torch.uint8)
        if end > start:
            self._f.seek(self._base + start)
            self._f.readinto(memoryview(raw.numpy()))
        return raw.view(_DTYPES[e["dtype"]]).reshape(e["shape"])


# ---- ConvRot -------------------------------------------------------------------------------------------------
_hadamard_cache = {}


def regular_hadamard(rot_size: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Kronecker powers of the 4x4 regular Hadamard, orthonormal - symmetric and orthogonal, so its own inverse.
    Built in fp32 (entries stay exactly +-1 through the krons; rot_size is a power of 4, so the division is exact)."""
    key = (rot_size, str(device), dtype)
    if key not in _hadamard_cache:
        r4 = torch.tensor([[1.0, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
        h = r4.clone()
        while h.shape[0] < rot_size:
            h = torch.kron(h, r4)
        if h.shape[0] != rot_size:
            raise ValueError(f"convrot group size {rot_size} is not a power of 4")
        _hadamard_cache[key] = (h / rot_size ** 0.5).to(device=device, dtype=dtype)
    return _hadamard_cache[key]


def rotate(x: torch.Tensor, rot_size: int) -> torch.Tensor:
    """Block regular-Hadamard rotation along the LAST dim. Self-inverse, so this both applies and undoes it."""
    if rot_size <= 1:
        return x
    if x.shape[-1] % rot_size:
        raise ValueError(f"last dim {x.shape[-1]} is not a multiple of the rotation block {rot_size}")
    h = regular_hadamard(rot_size, x.device, x.dtype)
    return torch.matmul(x.reshape(-1, x.shape[-1] // rot_size, rot_size), h).reshape(x.shape)


def parse_comfy_quant(blob: torch.Tensor) -> dict:
    """The `<module>.comfy_quant` marker: a uint8 tensor holding JSON."""
    return json.loads(bytes(blob.to(torch.uint8).cpu().numpy().tobytes()).decode("utf-8"))


def dequantize_int8_convrot(qweight: torch.Tensor, scale: torch.Tensor, conf: dict,
                            out_dtype=torch.bfloat16) -> torch.Tensor:
    """int8 codes + per-row scale (+ the marker's config) -> the dense weight in the true basis. The rotation is undone
    in fp32 and the result cast once."""
    fmt = conf.get("format")
    if fmt != "int8_tensorwise":
        raise ValueError(f"unsupported comfy quant format {fmt!r} (this decodes int8_tensorwise)")
    group = int(conf.get("convrot_groupsize", 256)) if conf.get("convrot") else 1
    w = qweight.to(torch.float32) * scale.to(torch.float32).reshape(-1, 1)
    return rotate(w, group).to(out_dtype)
