# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/families/base_check.py (commit 9162fb20,
# v8.7.3). Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: torch-free (the Train tab calls it while a run may own the GPU): the safetensors header and
# the one tensor are read with plain file reads and summed in numpy, where Fizgig uses safe_open / torch; a diffusers
# shard set or folder is searched shard by shard for the tensor (Fizgig reads the one file it is given); a save marked
# `tagscriber_finetune` counts as a fine-tune like Fizgig's `fizgig_finetune`; otherwise the same rule.
"""Whether a family's Base DiT file is the model's own released Base (FamilyDescription.standard_base).

A family that previews on a separate checkpoint (MOSS-Image on SFT) does so because its own Base is too soft to judge
a LoRA on. Any other Base - a fine-tune such as Kromoss, a TagScribeR fine-tune, SFT itself - is the model the LoRA
will be used on, so previews belong on it. One tensor's sum identifies the released Base (a norm weight, which
quantised files keep as it was); a fine-tune keeps its norms frozen, so its save's metadata names it instead.
"""
import functools
import glob
import json
import os
import struct

import numpy as np

FINETUNE_MARKS = ("fizgig_finetune", "tagscriber_finetune")
_NP_DTYPES = {"F64": np.float64, "F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32,
              "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}


def _files(path):
    """The safetensors files a Base path names: the file, a folder's files, or every shard of a -NNNNN-of-NNNNN set
    (model.read_state_dict's rule, without torch)."""
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(glob.escape(path), "*.safetensors")))
    if path.endswith(".json"):
        return sorted(glob.glob(os.path.join(glob.escape(os.path.dirname(path)), "*.safetensors")))
    if "-of-" in os.path.basename(path):
        stem = os.path.basename(path).rsplit("-", 3)[0]
        return sorted(glob.glob(os.path.join(glob.escape(os.path.dirname(path)), stem + "-*-of-*.safetensors")))
    return [path]


_MAX_HEADER = 256 * 1024 * 1024       # no real safetensors header comes near this; a larger claim is not read
_MAX_TENSOR = 64 * 1024 * 1024        # the fingerprint is a norm weight (a few KB); never read a model-sized tensor


def _header(f):
    (n,) = struct.unpack("<Q", f.read(8))
    size = os.fstat(f.fileno()).st_size
    if n > _MAX_HEADER or 8 + n > size:
        raise ValueError("not a safetensors header")
    return json.loads(f.read(n).decode("utf-8")), 8 + n


def _tensor_sum(f, base, entry) -> float:
    """The tensor's sum in float64 from its raw bytes (bf16 widened to float32 exactly: the high half of each float32)."""
    start, end = (int(x) for x in entry["data_offsets"])
    if not 0 <= start < end or end - start > _MAX_TENSOR or base + end > os.fstat(f.fileno()).st_size:
        raise ValueError("tensor span outside the file, or too large to be the fingerprint")
    f.seek(base + start)
    raw = f.read(end - start)
    dt = entry["dtype"]
    if dt == "BF16":
        arr = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif dt in _NP_DTYPES:
        arr = np.frombuffer(raw, dtype=_NP_DTYPES[dt])
    else:
        raise ValueError(f"unsupported safetensors dtype {dt!r}")
    return float(np.asarray(arr, dtype=np.float64).sum())


@functools.lru_cache(maxsize=32)
def _check(path, mtime, size, key, want):
    for file in _files(path):
        with open(file, "rb") as f:
            header, base = _header(f)
            md = header.pop("__metadata__", None) or {}
            if any(md.get(m) for m in FINETUNE_MARKS):
                return False
            hit = next((k for k in header if k == key or k.endswith("." + key)), None)
            if hit is None:
                continue                      # a shard set: the tensor lives in one of the files
            return abs(_tensor_sum(f, base, header[hit]) - want) < 1e-6
    return None                               # a layout without that tensor: unknown, treated as the Base


def is_standard_base(desc, path):
    """True = the released Base, False = another model, None = unknown (no fingerprint, no file, unreadable)."""
    if not getattr(desc, "standard_base", ()) or not path or not os.path.exists(path):
        return None
    try:
        key, want = desc.standard_base
        files = _files(path)
        if not files:
            return None
        st = os.stat(files[0])
        return _check(os.path.abspath(path), st.st_mtime, st.st_size, key, float(want))
    except Exception:
        return None
