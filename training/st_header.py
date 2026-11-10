"""One bounded safetensors header reader (torch-free), shared by the resume check, the fine-tune source check and the
lazy reader in training.modules.safetensors_io. A header is untrusted text from a file: this refuses a claimed length
above MAX_HEADER or beyond the end of the file, and turns every failure into a ValueError (never a MemoryError or a
huge allocation)."""
from __future__ import annotations

import json
import os
import struct

MAX_HEADER = 256 * 1024 * 1024      # no real header is this large


def read_header(f, limit: int = MAX_HEADER):
    """(header dict, offset of the first payload byte) from an open binary file at position 0.
    Raises ValueError if the file does not start with a plausible safetensors header."""
    try:
        raw8 = f.read(8)
        if len(raw8) != 8:
            raise ValueError("file too short for a safetensors header")
        (n,) = struct.unpack("<Q", raw8)
        size = os.fstat(f.fileno()).st_size
        if n > limit or 8 + n > size:
            raise ValueError("header length is larger than allowed or than the file")
        raw = f.read(n)
        if len(raw) != n:
            raise ValueError("header is cut short")
        header = json.loads(raw.decode("utf-8"))
    except ValueError:
        raise
    except (MemoryError, OverflowError, struct.error, OSError, RecursionError) as e:
        raise ValueError(f"unreadable safetensors header ({type(e).__name__})") from e
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not an object")
    return header, 8 + n


def check_span(entry, elem_size: int, payload_size: int):
    """(start, end) of a tensor entry after checking that the offsets are ordered, inside the payload and equal to
    dtype size x shape. Raises ValueError otherwise."""
    if not isinstance(entry, dict):
        raise ValueError("tensor entry is not an object")
    try:
        start, end = entry["data_offsets"]
        shape = entry["shape"]
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("offsets are not integers")
        numel = 1
        for d in shape:
            if isinstance(d, bool) or not isinstance(d, int) or d < 0:
                raise ValueError("bad shape")
            numel *= d
    except (KeyError, TypeError) as e:
        raise ValueError("tensor entry lacks valid offsets or shape") from e
    if not 0 <= start <= end:
        raise ValueError("tensor offsets are negative or reversed")
    if end > payload_size:
        raise ValueError("tensor offsets run past the end of the file")
    if end - start != numel * elem_size:
        raise ValueError("tensor span does not match its dtype and shape")
    return start, end
