"""Image metadata: inspection, privacy audit, lossless stripping, authorship, prompt extraction.

All writers work at the container level (JPEG segments, PNG chunks, WebP RIFF
chunks) so image data is never re-encoded: stripping or tagging a JPEG doesn't
cost quality. The EXIF Orientation tag is always preserved so photos keep
displaying upright, and ICC color profiles are kept unless explicitly removed.
"""
from __future__ import annotations

import json
import os
import struct
import tempfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from PIL import ExifTags, Image

# EXIF tag ids
ORIENTATION, MAKE, MODEL, SOFTWARE, DATETIME, ARTIST, COPYRIGHT, DESCRIPTION = (
    0x0112, 0x010F, 0x0110, 0x0131, 0x0132, 0x013B, 0x8298, 0x010E)
USER_COMMENT, DT_ORIGINAL, DT_DIGITIZED = 0x9286, 0x9003, 0x9004
OWNER_NAME, BODY_SERIAL, LENS_SERIAL, LENS_MAKE, LENS_MODEL = 0xA430, 0xA431, 0xA435, 0xA433, 0xA434
GPS_IFD, EXIF_IFD = 0x8825, 0x8769
HOST_COMPUTER, UNIQUE_ID = 0x013C, 0xA420

DEVICE_TAGS = {MAKE, MODEL, OWNER_NAME, BODY_SERIAL, LENS_SERIAL, LENS_MAKE, LENS_MODEL, HOST_COMPUTER, UNIQUE_ID}
TIME_TAGS = {DATETIME, DT_ORIGINAL, DT_DIGITIZED}

# PNG text keys written by generators / tools
AI_TEXT_KEYS = {"parameters", "prompt", "workflow", "invokeai_metadata", "invokeai_graph", "sd-metadata",
                "dream", "comment", "generation_data", "fooocus_scheme", "negative_prompt", "nai"}
AUTHOR_TEXT_KEYS = {"Author": "artist", "Copyright": "copyright", "Description": "description",
                    "Software": "software"}

NUL = bytes([0])
SOI = bytes([0xFF, 0xD8])
EXIF_HEADER = b"Exif" + NUL + NUL
ICC_HEADER = b"ICC_PROFILE" + NUL
XMP_NS = b"http://ns.adobe.com/xap/1.0/"
PNG_SIG = bytes([0x89]) + b"PNG\r\n" + bytes([0x1A]) + b"\n"


# ---------------------------------------------------------------------------- report
@dataclass
class Finding:
    level: str   # "high" | "medium" | "low"
    label: str
    detail: str


@dataclass
class MetaReport:
    path: str
    format: str = ""
    exif: dict[int, object] = field(default_factory=dict)       # 0th + Exif IFD tags
    gps: dict[int, object] = field(default_factory=dict)
    text: dict[str, str] = field(default_factory=dict)          # PNG text chunks / comments
    has_icc: bool = False
    has_xmp: bool = False
    xmp: str = ""
    has_iptc: bool = False
    error: str = ""

    def rows(self) -> list[tuple[str, str, str]]:
        """(display key, value, edit id) rows; edit id is 'exif:<tag>' / 'text:<key>' when editable, else ''."""
        out: list[tuple[str, str, str]] = [("Format", self.format, "")]
        for tag, value in sorted(self.exif.items()):
            editable = isinstance(value, str) and tag in (ARTIST, COPYRIGHT, DESCRIPTION, SOFTWARE)
            out.append((f"EXIF {ExifTags.TAGS.get(tag, hex(tag))}", _short(_val(value)),
                        f"exif:{tag}" if editable else ""))
        if self.gps:
            out.append(("EXIF GPS", ", ".join(ExifTags.GPSTAGS.get(k, hex(k)) for k in self.gps), ""))
        for k, v in self.text.items():
            out.append((f"Text: {k}", v, f"text:{k}" if self.format == "PNG" else ""))
        if self.has_icc:
            out.append(("ICC color profile", "present", ""))
        if self.has_xmp:
            out.append(("XMP", _short(self.xmp, 300), ""))
        if self.has_iptc:
            out.append(("IPTC / Photoshop", "present", ""))
        return out


def _val(v) -> str:
    if isinstance(v, bytes):
        try:
            if v.startswith(b"UNICODE" + NUL):
                return v[8:].decode("utf-16-le", "replace").strip(NUL.decode())
            if v.startswith(b"ASCII" + NUL * 3):
                return v[8:].decode("ascii", "replace").strip(NUL.decode())
            return v.decode("utf-8").strip(NUL.decode())
        except UnicodeDecodeError:
            return f"<{len(v)} bytes>"
    return str(v)


def _short(s: str, n: int = 160) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + "…"


def read_metadata(path: str | os.PathLike) -> MetaReport:
    r = MetaReport(path=str(path))
    try:
        with Image.open(path) as im:
            r.format = im.format or ""
            ex = im.getexif()
            r.exif = {k: v for k, v in ex.items() if k not in (GPS_IFD, EXIF_IFD)}
            try:
                r.exif.update(ex.get_ifd(EXIF_IFD))
            except Exception:
                pass
            try:
                r.gps = dict(ex.get_ifd(GPS_IFD))
            except Exception:
                r.gps = {}
            info = im.info
            r.has_icc = bool(info.get("icc_profile"))
            xmp = info.get("xmp") or info.get("XML:com.adobe.xmp") or b""
            if isinstance(xmp, bytes):
                xmp = xmp.decode("utf-8", "replace")
            r.xmp, r.has_xmp = xmp, bool(xmp)
            applist = getattr(im, "applist", None) or []
            r.has_iptc = "photoshop" in info or any(m == "APP13" for m, _ in applist)
            for k, v in info.items():
                if isinstance(v, str) and k not in ("xmp", "XML:com.adobe.xmp"):
                    r.text[k] = v
            if r.format == "JPEG" and "comment" in info:
                c = info["comment"]
                r.text["comment"] = c.decode("utf-8", "replace") if isinstance(c, bytes) else str(c)
    except Exception as e:
        r.error = str(e)
    return r


def privacy_findings(r: MetaReport) -> list[Finding]:
    f: list[Finding] = []
    if r.gps:
        f.append(Finding("high", "GPS location", "The exact place the picture was taken can be read from this file."))
    serials = [ExifTags.TAGS.get(t, hex(t)) for t in (BODY_SERIAL, LENS_SERIAL, UNIQUE_ID) if t in r.exif]
    if serials:
        f.append(Finding("high", "Device serial numbers", ", ".join(serials) + " can link images to one device."))
    if OWNER_NAME in r.exif or HOST_COMPUTER in r.exif:
        f.append(Finding("high", "Owner / computer name", _val(r.exif.get(OWNER_NAME) or r.exif.get(HOST_COMPUTER))))
    ai_keys = [k for k in r.text if k.lower() in AI_TEXT_KEYS]
    if ai_keys or (USER_COMMENT in r.exif and "Steps:" in _val(r.exif[USER_COMMENT])):
        f.append(Finding("medium", "AI generation data", "Prompts, models, seeds or a full ComfyUI workflow are "
                         "embedded (" + ", ".join(ai_keys or ["UserComment"]) + ")."))
    if r.has_xmp:
        detail = "Editing history and tool information."
        if "trainedAlgorithmicMedia" in r.xmp or "c2pa" in r.xmp.lower():
            detail += " Includes an AI/provenance marker."
        f.append(Finding("medium", "XMP metadata", detail))
    if MAKE in r.exif or MODEL in r.exif:
        f.append(Finding("low", "Camera / device model",
                         f"{_val(r.exif.get(MAKE, ''))} {_val(r.exif.get(MODEL, ''))}".strip()))
    if any(t in r.exif for t in TIME_TAGS):
        f.append(Finding("low", "Timestamps", "When the image was created or edited."))
    if SOFTWARE in r.exif or "Software" in r.text:
        f.append(Finding("low", "Software", _val(r.exif.get(SOFTWARE) or r.text.get("Software"))))
    if ARTIST in r.exif or COPYRIGHT in r.exif or "Author" in r.text:
        f.append(Finding("low", "Author / copyright", "Names in the file (fine if intended)."))
    return f


# ---------------------------------------------------------------------------- prompts
def extract_prompt(r: MetaReport) -> str | None:
    """Positive generation prompt from A1111/Forge, ComfyUI, NovelAI or InvokeAI metadata."""
    t = {k.lower(): v for k, v in r.text.items()}
    if "parameters" in t:
        return _a1111_prompt(t["parameters"])
    if USER_COMMENT in r.exif:
        uc = _val(r.exif[USER_COMMENT])
        if "Steps:" in uc:
            return _a1111_prompt(uc)
    if "prompt" in t:
        p = _comfy_prompt(t["prompt"])
        if p:
            return p
    if "invokeai_metadata" in t:
        try:
            data = json.loads(t["invokeai_metadata"])
            p = data.get("positive_prompt") or data.get("prompt")
            if isinstance(p, str) and p.strip():
                return p.strip()
        except ValueError:
            pass
    if "comment" in t and "description" in t:  # NovelAI: Description holds the prompt
        return t["description"].strip() or None
    return None


def _a1111_prompt(params: str) -> str | None:
    text = params.replace("\r\n", "\n")
    for marker in ("\nNegative prompt:", "\nSteps:"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip() or None


def _comfy_prompt(raw: str) -> str | None:
    try:
        graph = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(graph, dict):
        return None

    def text_of(ref, depth=0):
        if depth > 8:
            return None
        if isinstance(ref, str):
            return ref
        if isinstance(ref, list) and ref and str(ref[0]) in graph:
            inputs = graph[str(ref[0])].get("inputs", {})
            for key in ("text", "text_g", "prompt", "string", "value", "conditioning", "positive"):
                if key in inputs:
                    got = text_of(inputs[key], depth + 1)
                    if got:
                        return got
        return None

    for node in graph.values():
        if isinstance(node, dict) and "sampler" in str(node.get("class_type", "")).lower():
            got = text_of(node.get("inputs", {}).get("positive"))
            if got and got.strip():
                return got.strip()
    texts = [n.get("inputs", {}).get("text") for n in graph.values()
             if isinstance(n, dict) and "cliptextencode" in str(n.get("class_type", "")).lower()]
    texts = [x for x in texts if isinstance(x, str) and x.strip()]
    return max(texts, key=len).strip() if texts else None


# ---------------------------------------------------------------------------- options
@dataclass
class StripOptions:
    everything: bool = False        # all metadata except orientation (and ICC unless remove_icc)
    location: bool = True
    device: bool = True             # make/model, serials, owner, lens, host computer
    ai_data: bool = True            # generation parameters, ComfyUI workflow, UserComment params
    timestamps: bool = False
    software: bool = False
    xmp: bool = True
    remove_icc: bool = False        # off: keep color profile so colors stay correct


@dataclass
class Authorship:
    artist: str = ""
    copyright: str = ""
    description: str = ""
    software: str = ""


_ASCII_SYMBOLS = {"©": "(c)", "®": "(R)", "™": "(TM)", "–": "-", "—": "-", "‘": "'", "’": "'",
                  "“": '"', "”": '"', "…": "..."}


def exif_ascii(value: str) -> str:
    """EXIF text tags are ASCII by spec (Pillow turns other characters into '?').
    Map common symbols and fold accents so the meaning survives; PNG text keeps exact Unicode."""
    import unicodedata
    for sym, rep in _ASCII_SYMBOLS.items():
        value = value.replace(sym, rep)
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".meta-", suffix=path.suffix, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _filter_exif(exif_bytes: bytes | None, opts: StripOptions | None, author: Authorship | None) -> bytes | None:
    """Return new EXIF bytes (or None if nothing should remain)."""
    ex = Image.Exif()
    if exif_bytes:
        try:
            ex.load(exif_bytes)
        except Exception:
            ex = Image.Exif()
    if opts is not None:
        orientation = ex.get(ORIENTATION)
        if opts.everything:
            ex = Image.Exif()
            if orientation:
                ex[ORIENTATION] = orientation
        else:
            drop = set()
            if opts.device:
                drop |= DEVICE_TAGS
            if opts.timestamps:
                drop |= TIME_TAGS
            if opts.software:
                drop.add(SOFTWARE)
            sub = ex.get_ifd(EXIF_IFD) if EXIF_IFD in ex else {}
            if opts.ai_data and USER_COMMENT in sub and "Steps:" in _val(sub[USER_COMMENT]):
                drop.add(USER_COMMENT)
            for t in list(ex.keys()):
                if t in drop:
                    del ex[t]
            if opts.location and GPS_IFD in ex:
                del ex[GPS_IFD]
            if EXIF_IFD in ex:
                ifd = ex.get_ifd(EXIF_IFD)
                for t in list(ifd.keys()):
                    if t in drop:
                        del ifd[t]
    if author is not None:
        for key, tag in (("artist", ARTIST), ("copyright", COPYRIGHT), ("description", DESCRIPTION),
                         ("software", SOFTWARE)):
            value = getattr(author, key).strip()
            if value:
                ex[tag] = exif_ascii(value)
    if not len(ex.keys()):
        return None
    return ex.tobytes()


# ---------------------------------------------------------------------------- JPEG
def _jpeg_segments(data: bytes):
    if data[:2] != SOI:
        raise ValueError("not a JPEG")
    i = 2
    while i < len(data):
        if data[i] != 0xFF:
            raise ValueError("corrupt JPEG marker stream")
        marker = data[i + 1]
        if marker == 0xDA:  # start of scan: the rest is image data
            yield marker, data[i:]
            return
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:
            yield marker, data[i:i + 2]
            i += 2
            continue
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        yield marker, data[i:i + 2 + length]
        i += 2 + length


def _is_exif_seg(marker: int, seg: bytes) -> bool:
    return marker == 0xE1 and seg[4:10] == EXIF_HEADER


def _jpeg_app1(exif: bytes) -> bytes:
    payload = EXIF_HEADER + exif
    return bytes([0xFF, 0xE1]) + struct.pack(">H", len(payload) + 2) + payload


def _assemble_jpeg(segs: list[tuple[int, bytes]], new_exif: bytes | None) -> bytes:
    """SOI, [APP0 JFIF], [new EXIF], remaining segments (old EXIF removed)."""
    body = [(m, s) for m, s in segs if not _is_exif_seg(m, s)]
    out = [SOI]
    if body and body[0][0] == 0xE0:
        out.append(body.pop(0)[1])
    if new_exif:
        out.append(_jpeg_app1(new_exif))
    out.extend(seg for _m, seg in body)
    return b"".join(out)


def _rewrite_jpeg(data: bytes, opts: StripOptions | None, author: Authorship | None) -> bytes:
    segs = list(_jpeg_segments(data))
    exif_src = next((seg[10:] for m, seg in segs if _is_exif_seg(m, seg)), None)
    new_exif = _filter_exif(exif_src, opts, author)
    kept = []
    for marker, seg in segs:
        if opts is not None:
            if marker == 0xE1 and XMP_NS in seg[:40] and (opts.xmp or opts.everything):
                continue
            if marker == 0xED and opts.everything:                       # Photoshop IRB / IPTC
                continue
            if marker == 0xFE and (opts.everything or opts.ai_data):     # comments often hold parameters
                continue
            if marker == 0xE2 and seg[4:16] == ICC_HEADER and opts.remove_icc:
                continue
        kept.append((marker, seg))
    return _assemble_jpeg(kept, new_exif)


# ---------------------------------------------------------------------------- PNG
def _png_chunks(data: bytes):
    if not data.startswith(PNG_SIG):
        raise ValueError("not a PNG")
    i = 8
    while i < len(data):
        length = struct.unpack(">I", data[i:i + 4])[0]
        ctype = data[i + 4:i + 8]
        yield ctype, data[i:i + 12 + length], data[i + 8:i + 8 + length]
        i += 12 + length


def _png_chunk(ctype: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + ctype + body + struct.pack(">I", zlib.crc32(ctype + body) & 0xFFFFFFFF)


def _png_itxt(key: str, value: str) -> bytes:
    # keyword NUL compression-flag(0) compression-method(0) language-tag NUL translated-keyword NUL text
    return _png_chunk(b"iTXt", key.encode("latin-1") + NUL + NUL + NUL + NUL + NUL + value.encode("utf-8"))


def _png_text_key(body: bytes) -> str:
    return body.split(NUL, 1)[0].decode("latin-1", "replace")


def _rewrite_png(data: bytes, opts: StripOptions | None, author: Authorship | None) -> bytes:
    out = [PNG_SIG]
    chunks = list(_png_chunks(data))
    exif_src = next((body for ct, _raw, body in chunks if ct == b"eXIf"), None)
    new_exif = _filter_exif(exif_src, opts, author) if (exif_src or author) else None
    author_keys = {k for k, f in AUTHOR_TEXT_KEYS.items() if author and getattr(author, f, "").strip()}
    for ctype, raw, body in chunks:
        if ctype == b"eXIf":
            continue  # re-added below (filtered)
        if ctype in (b"tEXt", b"zTXt", b"iTXt"):
            key = _png_text_key(body)
            if key in author_keys:
                continue  # replaced by the new authorship value
            if opts is not None:
                if opts.everything:
                    continue
                if opts.ai_data and key.lower() in AI_TEXT_KEYS:
                    continue
                if opts.xmp and key == "XML:com.adobe.xmp":
                    continue
                if opts.software and key == "Software":
                    continue
                if opts.timestamps and key == "Creation Time":
                    continue
        if opts is not None:
            if ctype == b"iCCP" and opts.remove_icc:
                continue
            if ctype == b"tIME" and (opts.everything or opts.timestamps):
                continue
        if ctype == b"IEND":
            if new_exif:
                out.append(_png_chunk(b"eXIf", new_exif))
            for key in sorted(author_keys):
                out.append(_png_itxt(key, getattr(author, AUTHOR_TEXT_KEYS[key]).strip()))
        out.append(raw)
    return b"".join(out)


# ---------------------------------------------------------------------------- WebP
def _rewrite_webp(data: bytes, opts: StripOptions | None, author: Authorship | None) -> bytes:
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise ValueError("not a WebP")
    chunks = []
    i = 12
    while i < len(data):
        ctype = data[i:i + 4]
        size = struct.unpack("<I", data[i + 4:i + 8])[0]
        chunks.append((ctype, data[i + 8:i + 8 + size]))
        i += 8 + size + (size & 1)
    exif_src = next((b for c, b in chunks if c == b"EXIF"), None)
    new_exif = _filter_exif(exif_src, opts, author) if (exif_src or author) else None
    kept = []
    for ctype, body in chunks:
        if ctype == b"EXIF":
            continue
        if opts is not None and ctype == b"XMP " and (opts.xmp or opts.everything):
            continue
        if opts is not None and ctype == b"ICCP" and opts.remove_icc:
            continue
        kept.append([ctype, body])
    if new_exif:
        if not any(c == b"VP8X" for c, _ in kept):
            return b""  # simple WebP can't carry EXIF without a VP8X header; caller reports it
        kept.append([b"EXIF", new_exif])
    names = {c for c, _ in kept}
    for chunk in kept:  # keep VP8X feature flags consistent with the remaining chunks
        if chunk[0] == b"VP8X":
            flags = chunk[1][0] & ~(0x20 | 0x08 | 0x04)
            flags |= (0x20 if b"ICCP" in names else 0) | (0x08 if b"EXIF" in names else 0) | \
                     (0x04 if b"XMP " in names else 0)
            chunk[1] = bytes([flags]) + chunk[1][1:]
    body = b"WEBP" + b"".join(c + struct.pack("<I", len(b)) + b + (NUL if len(b) & 1 else b"") for c, b in kept)
    return b"RIFF" + struct.pack("<I", len(body)) + body


# ---------------------------------------------------------------------------- public writers
def _rewrite(path: Path, opts: StripOptions | None, author: Authorship | None) -> bool:
    data = path.read_bytes()
    if data[:2] == SOI:
        new = _rewrite_jpeg(data, opts, author)
    elif data.startswith(PNG_SIG):
        new = _rewrite_png(data, opts, author)
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        new = _rewrite_webp(data, opts, author)
        if not new:
            raise ValueError("this simple WebP can't store EXIF; convert it to PNG/JPEG first")
    else:
        raise ValueError(f"lossless metadata editing isn't supported for {path.suffix or 'this format'}")
    if new != data:
        _atomic_write_bytes(path, new)
        return True
    return False


def strip_metadata(path: str | os.PathLike, opts: StripOptions) -> bool:
    """Remove the chosen metadata. Returns True if the file changed."""
    return _rewrite(Path(path), opts, None)


def apply_authorship(path: str | os.PathLike, author: Authorship) -> bool:
    """Write artist / copyright / description / software (values entered by the user)."""
    return _rewrite(Path(path), None, author)


def set_text_field(path: str | os.PathLike, key: str, value: str) -> None:
    """Replace (or add) one PNG text field, losslessly. Empty value removes it."""
    p = Path(path)
    data = p.read_bytes()
    if not data.startswith(PNG_SIG):
        raise ValueError("text fields can only be edited on PNG files")
    out = [PNG_SIG]
    for ctype, raw, body in _png_chunks(data):
        if ctype in (b"tEXt", b"zTXt", b"iTXt") and _png_text_key(body) == key:
            continue
        if ctype == b"IEND" and value:
            out.append(_png_itxt(key, value))
        out.append(raw)
    _atomic_write_bytes(p, b"".join(out))


def set_exif_text(path: str | os.PathLike, tag: int, value: str) -> None:
    """Replace one top-level EXIF string tag (JPEG/PNG) losslessly. Empty value removes it."""
    p = Path(path)
    data = p.read_bytes()
    with Image.open(p) as im:
        ex = im.getexif()
    if value:
        ex[tag] = value
    elif tag in ex:
        del ex[tag]
    new_exif = ex.tobytes() if len(ex.keys()) else None
    if data[:2] == SOI:
        _atomic_write_bytes(p, _assemble_jpeg(list(_jpeg_segments(data)), new_exif))
    elif data.startswith(PNG_SIG):
        out = [PNG_SIG]
        for ctype, raw, _body in _png_chunks(data):
            if ctype == b"eXIf":
                continue
            if ctype == b"IEND" and new_exif:
                out.append(_png_chunk(b"eXIf", new_exif))
            out.append(raw)
        _atomic_write_bytes(p, b"".join(out))
    else:
        raise ValueError("EXIF editing is supported for JPEG and PNG")


def pixels_equal(a: str | os.PathLike, b: str | os.PathLike) -> bool:
    """Verification helper: decoded pixels identical."""
    with Image.open(a) as x, Image.open(b) as y:
        return x.size == y.size and x.tobytes() == y.tobytes()
