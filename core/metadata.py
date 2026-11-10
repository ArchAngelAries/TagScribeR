"""Image metadata: inspection, privacy audit, lossless stripping, authorship, prompt extraction.

All writers work at the container level (JPEG segments, PNG chunks, WebP RIFF
chunks) so image data is never re-encoded: stripping or tagging a JPEG doesn't
cost quality. The EXIF Orientation tag is always preserved so photos keep
displaying upright, and ICC color profiles are kept unless explicitly removed.
"""
from __future__ import annotations

import json
import os
import random
import re
import struct
import tempfile
import zlib
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path

from PIL import ExifTags, Image
from PIL.TiffImagePlugin import IFDRational

from core.caption_io import apply_mode

# EXIF tag ids
ORIENTATION, MAKE, MODEL, SOFTWARE, DATETIME, ARTIST, COPYRIGHT, DESCRIPTION = (
    0x0112, 0x010F, 0x0110, 0x0131, 0x0132, 0x013B, 0x8298, 0x010E)
USER_COMMENT, DT_ORIGINAL, DT_DIGITIZED = 0x9286, 0x9003, 0x9004
OWNER_NAME, BODY_SERIAL, LENS_SERIAL, LENS_MAKE, LENS_MODEL = 0xA430, 0xA431, 0xA435, 0xA433, 0xA434
GPS_IFD, EXIF_IFD = 0x8825, 0x8769
HOST_COMPUTER, UNIQUE_ID = 0x013C, 0xA420
MAKER_NOTE = 0x927C
OFFSET_TIME, OFFSET_TIME_ORIGINAL, OFFSET_TIME_DIGITIZED = 0x9010, 0x9011, 0x9012
SUBSEC_TIME, SUBSEC_ORIGINAL, SUBSEC_DIGITIZED = 0x9290, 0x9291, 0x9292
CAMERA_SERIAL, UNIQUE_CAMERA_MODEL = 0xC62F, 0xC614

DEVICE_TAGS = {MAKE, MODEL, OWNER_NAME, BODY_SERIAL, LENS_SERIAL, LENS_MAKE, LENS_MODEL, HOST_COMPUTER, UNIQUE_ID,
               CAMERA_SERIAL, UNIQUE_CAMERA_MODEL}
TIME_TAGS = {DATETIME, DT_ORIGINAL, DT_DIGITIZED, SUBSEC_TIME, SUBSEC_ORIGINAL, SUBSEC_DIGITIZED}
OFFSET_TAGS = {OFFSET_TIME, OFFSET_TIME_ORIGINAL, OFFSET_TIME_DIGITIZED}   # time zone: a coarse location

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
XMP_EXT_NS = b"http://ns.adobe.com/xmp/extension/"
PNG_SIG = bytes([0x89]) + b"PNG\r\n" + bytes([0x1A]) + b"\n"
TRAILER = -1          # pseudo marker: the bytes after a JPEG's end-of-image marker

# What can carry a place or a person inside an XMP packet (used when XMP is otherwise kept).
_XMP_PLACE = re.compile(rb"(?i)(GPS(Latitude|Longitude|Altitude|Position)|photoshop:(City|State|Country)\b|"
                        rb"Iptc4xmp\w+:(Location\w*|CountryCode|CountryName|City|ProvinceState|Sublocation)|"
                        rb"OffsetTime)")
_XMP_PERSON = re.compile(rb"(?i)(tiff:(Make|Model)\b|aux:(SerialNumber|LensSerialNumber|OwnerName)|"
                         rb"exifEX:(BodySerialNumber|CameraOwnerName|LensSerialNumber)|dc:creator|photoshop:Credit|"
                         rb"CreatorContactInfo|xmpRights:Owner|CiEmail\w*|CiAdr\w*|Iptc4xmp\w+:Creator\w*)")


# ---------------------------------------------------------------------------- report
@dataclass
class Finding:
    level: str   # "high" | "medium" | "low"
    label: str
    detail: str
    # Which Remove options clear it, joined with "+": location, device, ai, xmp, time, software, author.
    # "unchecked" means the file could not be examined. Used to verify a clean.
    category: str = ""


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
    scan: dict = field(default_factory=dict)    # container-level facts (see scan_container)
    scan_error: str = ""                        # set when the file's structure could not be read

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
    try:
        r.scan = scan_container(path)
    except Exception as e:
        r.scan_error = str(e) or type(e).__name__
    return r


def privacy_findings(r: MetaReport) -> list[Finding]:
    """Everything the cleaner can remove, as found in the file on disk. When the structure cannot be read, the
    result says so ("could not be checked"); it is never empty for a file that was not understood."""
    f: list[Finding] = []
    if r.error or r.scan_error:
        f.append(Finding("medium", "Could not be checked",
                         "This file could not be read completely (" + (r.scan_error or r.error) + "). It may still "
                         "hold location, device or personal data.", "unchecked"))
    sc = r.scan
    if r.gps:
        f.append(Finding("high", "GPS location", "The exact place the picture was taken can be read from this file.",
                         "location"))
    if any(t in r.exif for t in OFFSET_TAGS):
        f.append(Finding("medium", "Time zone offset", "The UTC offset the picture was taken in hints at the region.",
                         "location"))
    if sc.get("iptc_place"):
        f.append(Finding("high", "IPTC place names", "City, region, country or location names are stored in the "
                         "picture's IPTC block.", "location+device"))
    if sc.get("iptc_person"):
        f.append(Finding("high", "IPTC by-line / contact", "Photographer, credit or contact details are stored in "
                         "the picture's IPTC block.", "location+device"))
    if sc.get("iptc") and not (sc.get("iptc_place") or sc.get("iptc_person")):
        f.append(Finding("medium", "IPTC / Photoshop data", "A Photoshop / IPTC block is present.",
                         "location+device"))
    serials = [ExifTags.TAGS.get(t, hex(t)) for t in (BODY_SERIAL, LENS_SERIAL, UNIQUE_ID, CAMERA_SERIAL)
               if t in r.exif]
    if serials:
        f.append(Finding("high", "Device serial numbers", ", ".join(serials) + " can link images to one device.",
                         "device"))
    if OWNER_NAME in r.exif or HOST_COMPUTER in r.exif:
        f.append(Finding("high", "Owner / computer name", _val(r.exif.get(OWNER_NAME) or r.exif.get(HOST_COMPUTER)),
                         "device"))
    if MAKER_NOTE in r.exif:
        f.append(Finding("medium", "Camera-internal data (MakerNote)", "Vendor data that can hold serial numbers, "
                         "an owner name or settings.", "location+device"))
    if sc.get("thumbnail") or sc.get("mpf"):
        f.append(Finding("medium", "Embedded preview picture", "A smaller copy of the picture is stored inside the "
                         "file; it can show the uncropped original.", "location+device"))
    if sc.get("c2pa"):
        f.append(Finding("medium", "Content credentials (C2PA / JUMBF)", "Signer, tool and edit-history records "
                         "that can name a person or an account.", "location+device+ai"))
    ai_keys = [k for k in r.text if k.lower() in AI_TEXT_KEYS]
    if ai_keys or (USER_COMMENT in r.exif and "Steps:" in _val(r.exif[USER_COMMENT])):
        f.append(Finding("medium", "AI generation data", "Prompts, models, seeds or a full ComfyUI workflow are "
                         "embedded (" + ", ".join(ai_keys or ["UserComment"]) + ").", "ai"))
    if sc.get("comment") and "comment" not in [k.lower() for k in ai_keys]:
        f.append(Finding("medium", "Picture comment", "A free-text comment is embedded.", "ai+location+device"))
    if r.has_xmp or sc.get("xmp"):
        detail = "Editing history and tool information."
        if "trainedAlgorithmicMedia" in r.xmp or "c2pa" in r.xmp.lower():
            detail += " Includes an AI/provenance marker."
        raw = b"".join(sc.get("xmp", [])) or r.xmp.encode("utf-8", "replace")
        cats = ["xmp"] + (["location"] if _XMP_PLACE.search(raw) else []) + (["device"] if _XMP_PERSON.search(raw)
                                                                           else [])
        if len(cats) > 1:
            detail += " Names a place or a person."
        f.append(Finding("medium", "XMP metadata", detail, "+".join(cats)))
    if sc.get("xmp_ext"):
        f.append(Finding("medium", "Extended XMP", "The overflow part of the XMP data (large edit histories, "
                         "previews) is stored in extra segments.", "xmp+location+device"))
    if sc.get("unknown"):
        f.append(Finding("medium", "Other embedded data blocks", "Vendor or unknown blocks: " +
                         ", ".join(sorted(set(sc["unknown"]))[:8]) + ".", "location+device"))
    if sc.get("text_other"):
        f.append(Finding("medium", "Other embedded text", "Text fields (" + ", ".join(sorted(set(sc["text_other"]))[:6])
                         + ") can hold any kind of personal data.", "location+device"))
    if sc.get("trailer"):
        f.append(Finding("high", "Data after the end of the picture", f"{sc['trailer']} extra byte(s) follow the "
                         "picture (another picture, a video or vendor data).", "location+device+xmp+ai"))
    if MAKE in r.exif or MODEL in r.exif:
        f.append(Finding("low", "Camera / device model",
                         f"{_val(r.exif.get(MAKE, ''))} {_val(r.exif.get(MODEL, ''))}".strip(), "device"))
    if any(t in r.exif for t in TIME_TAGS) or sc.get("time_chunk") or "Creation Time" in r.text:
        f.append(Finding("low", "Timestamps", "When the image was created or edited.", "time"))
    if SOFTWARE in r.exif or "Software" in r.text:
        f.append(Finding("low", "Software", _val(r.exif.get(SOFTWARE) or r.text.get("Software")), "software"))
    if ARTIST in r.exif or COPYRIGHT in r.exif or any(k in r.text for k in ("Author", "Copyright")):
        f.append(Finding("low", "Author / copyright", "Names in the file (fine if intended).", "author"))
    return f


def enabled_categories(opts: "StripOptions") -> set[str]:
    """The finding categories a clean with these options is meant to clear."""
    if opts.everything:
        return {"location", "device", "ai", "xmp", "time", "software", "author"}
    return {n for n, on in (("location", opts.location), ("device", opts.device), ("ai", opts.ai_data),
                            ("xmp", opts.xmp), ("time", opts.timestamps), ("software", opts.software)) if on}


def leftovers(path: str | os.PathLike, opts: "StripOptions") -> list[Finding]:
    """Re-read a cleaned file from disk: what is still in it that the chosen options should have removed, plus
    anything that could not be checked."""
    want = enabled_categories(opts)
    return [f for f in privacy_findings(read_metadata(path))
            if f.category == "unchecked" or want & set(f.category.split("+"))]


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
    """Temp file next to the target, fsynced, the target's permission bits copied, then renamed over it."""
    fd, tmp = tempfile.mkstemp(prefix=".meta-", suffix=path.suffix, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        apply_mode(path, tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _filter_exif(exif_bytes: bytes | None, opts: StripOptions | None, author: Authorship | None,
                 camera: "CameraData | None" = None) -> bytes | None:
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
                drop |= DEVICE_TAGS | {MAKER_NOTE}
            if opts.location:
                drop |= {MAKER_NOTE} | OFFSET_TAGS
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
    if camera is not None:
        _apply_camera_tags(ex, camera)
    if not len(ex.keys()):
        return None
    return _tiff_bytes(ex)


# ---------------------------------------------------------------------------- JPEG
def _jpeg_segments(data: bytes):
    """(marker, bytes) pieces that join back to the file. A scan (start-of-scan header plus its entropy-coded data) is
    one piece; the bytes after the end-of-image marker are one final piece with the pseudo marker TRAILER."""
    if data[:2] != SOI:
        raise ValueError("not a JPEG")
    n = len(data)
    i = 2
    while i < n:
        if data[i] != 0xFF:
            raise ValueError("corrupt JPEG marker stream")
        if i + 1 >= n:
            raise ValueError("truncated JPEG")
        marker = data[i + 1]
        if marker == 0xFF:                    # fill byte before a marker
            i += 1
            continue
        if marker == 0xD9:                    # end of image: whatever follows is not part of the picture
            yield marker, data[i:i + 2]
            if i + 2 < n:
                yield TRAILER, data[i + 2:]
            return
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:
            yield marker, data[i:i + 2]
            i += 2
            continue
        if i + 4 > n:
            raise ValueError("truncated JPEG segment")
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        if length < 2 or i + 2 + length > n:
            raise ValueError("corrupt JPEG segment length")
        if marker == 0xDA:                    # start of scan: header, then entropy-coded data up to the next marker
            pos = i + 2 + length
            while True:
                j = data.find(b"\xff", pos)
                if j < 0 or j + 1 >= n:
                    j = n
                    break
                b = data[j + 1]
                if b == 0 or 0xD0 <= b <= 0xD7:
                    pos = j + 2
                elif b == 0xFF:
                    pos = j + 1
                else:
                    break
            yield marker, data[i:j]
            i = j
            continue
        yield marker, data[i:i + 2 + length]
        i += 2 + length


def _is_exif_seg(marker: int, seg: bytes) -> bool:
    return marker == 0xE1 and seg[4:10] == EXIF_HEADER


def _tiff_bytes(ex: Image.Exif) -> bytes:
    """The bare TIFF block of an Exif: what PNG eXIf and WebP EXIF chunks hold, and what follows the Exif\\0\\0 header
    in a JPEG APP1. Pillow's Exif.tobytes() already starts with that header; written as is, JPEGs carried it twice and
    PNG / WebP carried a header their containers forbid (Pillow reads past it, other readers do not)."""
    raw = ex.tobytes()
    return raw[len(EXIF_HEADER):] if raw.startswith(EXIF_HEADER) else raw


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


def _rewrite_jpeg(data: bytes, opts: StripOptions | None, author: Authorship | None,
                  camera: "CameraData | None" = None) -> bytes:
    """With ``opts``: an allow-list. A picture needs only its tables, frame and scans, the JFIF / Adobe headers and the
    ICC profile; when a category that can carry a place or a person is removed, every other APPn block, comment and
    trailing byte goes with it. EXIF is rebuilt from the tags that stay."""
    segs = list(_jpeg_segments(data))
    exif_src = next((seg[10:] for m, seg in segs if _is_exif_seg(m, seg)), None)
    new_exif = _filter_exif(exif_src, opts, author, camera)
    kept = []
    for marker, seg in segs:
        if opts is not None:
            seg = _jpeg_policy(marker, seg, opts)
            if seg is None:
                continue
        kept.append((marker, seg))
    return _assemble_jpeg(kept, new_exif)


def _place_or_person(opts: StripOptions) -> bool:
    return opts.everything or opts.location or opts.device


def _xmp_goes(xmp: bytes, opts: StripOptions) -> bool:
    """Does the cleaner remove this XMP packet? Yes when XMP is chosen, or when it names a place (Location chosen) or a
    person / camera (Device chosen) even though XMP itself was left ticked off."""
    return bool(opts.xmp or opts.everything or (opts.location and _XMP_PLACE.search(xmp))
                or (opts.device and _XMP_PERSON.search(xmp)))


def _jfif_no_thumbnail(seg: bytes) -> bytes:
    if len(seg) > 18 and (seg[16] or seg[17]):
        return seg[:2] + struct.pack(">H", 16) + seg[4:16] + bytes([0, 0])
    return seg


def _jpeg_policy(marker: int, seg: bytes, opts: StripOptions) -> bytes | None:
    """The segment to keep (possibly rewritten) or None to drop it."""
    po = _place_or_person(opts)
    if marker == TRAILER:
        return None if (po or opts.xmp or opts.ai_data) else seg
    if marker == 0xE0:
        if seg[4:9] == b"JFIF" + NUL:
            return _jfif_no_thumbnail(seg) if po else seg
        return None if po else seg
    if marker == 0xE1:
        if _is_exif_seg(marker, seg):
            return seg                                              # replaced by the rebuilt EXIF
        if seg[4:4 + len(XMP_NS)] == XMP_NS:
            return None if _xmp_goes(seg[4 + len(XMP_NS):], opts) else seg
        if seg[4:4 + len(XMP_EXT_NS)] == XMP_EXT_NS:
            return None if (opts.xmp or po) else seg
        return None if po else seg
    if marker == 0xE2:
        if seg[4:16] == ICC_HEADER:
            return None if opts.remove_icc else seg
        return None if po else seg                                  # MPF (embedded previews), FlashPix, vendor
    if marker == 0xEE and seg[4:9] == b"Adobe":
        return seg                                                  # tells decoders the colour transform
    if marker == 0xEB:                                              # APP11: JUMBF / C2PA content credentials
        return None if (po or opts.ai_data) else seg
    if marker == 0xFE:                                              # comments often hold parameters
        return None if (po or opts.ai_data) else seg
    if 0xE0 <= marker <= 0xEF:                                      # Photoshop / IPTC and any unknown APPn
        return None if po else seg
    return seg


# ---------------------------------------------------------------------------- PNG
def _png_parse(data: bytes):
    """([(type, raw chunk, body)], trailer). The list ends with IEND; ``trailer`` is whatever follows it (or the
    unreadable rest of a damaged file)."""
    if not data.startswith(PNG_SIG):
        raise ValueError("not a PNG")
    chunks = []
    i, n = 8, len(data)
    while i < n:
        if i + 12 > n:
            break
        length = struct.unpack(">I", data[i:i + 4])[0]
        if i + 12 + length > n:
            break
        ctype = data[i + 4:i + 8]
        chunks.append((ctype, data[i:i + 12 + length], data[i + 8:i + 8 + length]))
        i += 12 + length
        if ctype == b"IEND":
            break
    return chunks, data[i:]


def _png_chunks(data: bytes):
    return iter(_png_parse(data)[0])


# Chunks a PNG needs to look right (the rest of the ancillary ones are metadata).
_PNG_KEEP = {b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tRNS", b"gAMA", b"cHRM", b"sRGB", b"sBIT", b"bKGD", b"hIST",
             b"pHYs", b"sPLT", b"acTL", b"fcTL", b"fdAT", b"cICP", b"mDCV", b"cLLI", b"mDCv", b"cLLi"}
_PNG_KEEP_TEXT = {"Software", "Creation Time", "Author", "Copyright", "Description", "Title"}
_PNG_XMP_KEY = "XML:com.adobe.xmp"


def _png_text_value(ctype: bytes, body: bytes) -> bytes:
    """The text of a tEXt / zTXt / iTXt chunk as raw bytes (empty if it cannot be decoded)."""
    try:
        key, rest = body.split(NUL, 1)
        if ctype == b"tEXt":
            return rest
        if ctype == b"zTXt":
            return zlib.decompressobj().decompress(rest[1:], 1 << 20)
        flag = rest[0]
        rest = rest[2:].split(NUL, 2)[2] if rest[2:].count(NUL) >= 2 else b""
        return zlib.decompressobj().decompress(rest, 1 << 20) if flag else rest
    except Exception:
        return b""


def _png_chunk(ctype: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + ctype + body + struct.pack(">I", zlib.crc32(ctype + body) & 0xFFFFFFFF)


def _png_itxt(key: str, value: str) -> bytes:
    # keyword NUL compression-flag(0) compression-method(0) language-tag NUL translated-keyword NUL text
    return _png_chunk(b"iTXt", key.encode("latin-1") + NUL + NUL + NUL + NUL + NUL + value.encode("utf-8"))


def _png_text_key(body: bytes) -> str:
    return body.split(NUL, 1)[0].decode("latin-1", "replace")


def _rewrite_png(data: bytes, opts: StripOptions | None, author: Authorship | None,
                 camera: "CameraData | None" = None) -> bytes:
    """With ``opts``: text chunks, time chunks, unknown ancillary chunks and bytes after IEND go when a category that can
    carry a place or a person is removed; picture chunks (and the ICC profile unless asked) always stay."""
    out = [PNG_SIG]
    chunks, trailer = _png_parse(data)
    exif_src = next((body for ct, _raw, body in chunks if ct == b"eXIf"), None)
    new_exif = _filter_exif(exif_src, opts, author, camera) if (exif_src or author or camera) else None
    author_keys = {k for k, f in AUTHOR_TEXT_KEYS.items() if author and getattr(author, f, "").strip()}
    po = opts is not None and _place_or_person(opts)
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
                if key == _PNG_XMP_KEY and _xmp_goes(_png_text_value(ctype, body), opts):
                    continue
                if opts.software and key == "Software":
                    continue
                if opts.timestamps and key == "Creation Time":
                    continue
                if po and key not in _PNG_KEEP_TEXT and key != _PNG_XMP_KEY:
                    continue    # any other text field can hold a name, a place or a raw EXIF / IPTC copy
        elif opts is not None:
            if ctype == b"iCCP":
                if opts.remove_icc:
                    continue
            elif ctype == b"tIME":
                if opts.everything or opts.timestamps:
                    continue
            elif po and ctype not in _PNG_KEEP and ctype[:1].islower():
                continue        # unknown ancillary chunk
        if ctype == b"IEND":
            if new_exif:
                out.append(_png_chunk(b"eXIf", new_exif))
            for key in sorted(author_keys):
                out.append(_png_itxt(key, getattr(author, AUTHOR_TEXT_KEYS[key]).strip()))
        out.append(raw)
    if opts is None or not (po or opts.xmp or opts.ai_data):
        out.append(trailer)
    return b"".join(out)


# ---------------------------------------------------------------------------- WebP
def _rewrite_webp(data: bytes, opts: StripOptions | None, author: Authorship | None,
                  camera: "CameraData | None" = None) -> bytes:
    chunks, trailer = _webp_parse(data)
    exif_src = next((b for c, b in chunks if c == b"EXIF"), None)
    if exif_src and exif_src.startswith(EXIF_HEADER):
        exif_src = exif_src[len(EXIF_HEADER):]
    new_exif = _filter_exif(exif_src, opts, author, camera) if (exif_src or author or camera) else None
    po = opts is not None and _place_or_person(opts)
    kept = []
    for ctype, body in chunks:
        if ctype == b"EXIF":
            continue
        if opts is not None:
            if ctype == b"XMP ":
                if _xmp_goes(body, opts):
                    continue
            elif ctype == b"ICCP":
                if opts.remove_icc:
                    continue
            elif po and ctype not in _WEBP_KEEP:
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
    keep_trailer = opts is None or not (po or opts.xmp or opts.ai_data)
    return b"RIFF" + struct.pack("<I", len(body)) + body + (trailer if keep_trailer else b"")


_WEBP_KEEP = {b"VP8 ", b"VP8L", b"VP8X", b"ALPH", b"ANIM", b"ANMF", b"ICCP"}


def _webp_parse(data: bytes):
    """([(fourcc, body)], trailer): the chunks inside the RIFF size, and any bytes after it."""
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise ValueError("not a WebP")
    end = min(len(data), 8 + struct.unpack("<I", data[4:8])[0])
    chunks = []
    i = 12
    while i < end:
        if i + 8 > end:
            raise ValueError("corrupt WebP")
        size = struct.unpack("<I", data[i + 4:i + 8])[0]
        if i + 8 + size > end:
            raise ValueError("corrupt WebP")
        chunks.append((data[i:i + 4], data[i + 8:i + 8 + size]))
        i += 8 + size + (size & 1)
    return chunks, data[end:]


# ---------------------------------------------------------------------------- container scan (the audit's eyes)
_IPTC_PLACE = {(2, 90), (2, 92), (2, 95), (2, 100), (2, 101), (2, 26)}       # city, sub-location, state, country
_IPTC_PERSON = {(2, 80), (2, 85), (2, 110), (2, 115), (2, 116), (2, 118), (2, 122)}   # by-line, credit, source, ...


def _tiff_has_thumbnail(t: bytes) -> bool:
    """True if the TIFF block has a second IFD (IFD1), which is where cameras put the preview picture."""
    try:
        if len(t) < 8 or t[:2] not in (b"II", b"MM"):
            return False
        e = "<" if t[:2] == b"II" else ">"
        off = struct.unpack(e + "I", t[4:8])[0]
        n = struct.unpack(e + "H", t[off:off + 2])[0]
        nxt = off + 2 + 12 * n
        return struct.unpack(e + "I", t[nxt:nxt + 4])[0] != 0
    except (struct.error, IndexError):
        return False


def _iptc_fields(seg: bytes) -> set[tuple[int, int]] | None:
    """(record, dataset) pairs found in a Photoshop APP13 block; None when it could not be parsed."""
    found: set[tuple[int, int]] = set()
    try:
        p = seg[4:]
        if not p.startswith(b"Photoshop 3.0" + NUL):
            return None
        i = 14
        while i + 12 <= len(p) and p[i:i + 4] == b"8BIM":
            rid = struct.unpack(">H", p[i + 4:i + 6])[0]
            nlen = p[i + 6]
            i += 7 + nlen + ((nlen + 1) & 1)
            size = struct.unpack(">I", p[i:i + 4])[0]
            body = p[i + 4:i + 4 + size]
            i += 4 + size + (size & 1)
            if rid == 0x0404:
                j = 0
                while j + 5 <= len(body) and body[j] == 0x1C:
                    rec, ds, ln = body[j + 1], body[j + 2], struct.unpack(">H", body[j + 3:j + 5])[0]
                    found.add((rec, ds))
                    j += 5 + ln
        return found
    except (struct.error, IndexError):
        return None


def scan_container(path: str | os.PathLike) -> dict:
    """Structural facts the audit needs that Pillow does not report. Raises ValueError for a file whose containers
    cannot be walked (the caller reports "could not be checked")."""
    data = Path(path).read_bytes()
    s: dict = {"xmp": [], "xmp_ext": 0, "unknown": [], "text_other": [], "trailer": 0, "thumbnail": False,
               "mpf": False, "c2pa": False, "comment": False, "iptc": False, "iptc_place": False,
               "iptc_person": False, "time_chunk": False}
    tiff = None
    if data[:2] == SOI:
        for m, seg in _jpeg_segments(data):
            if m == TRAILER:
                s["trailer"] = len(seg)
            elif m == 0xE0:
                if seg[4:9] == b"JFIF" + NUL:
                    s["thumbnail"] |= len(seg) > 18 and bool(seg[16] or seg[17])
                else:
                    s["unknown"].append("APP0")
                    s["thumbnail"] |= seg[4:8] == b"JFXX"
            elif m == 0xE1:
                if _is_exif_seg(m, seg):
                    tiff = tiff or seg[10:]
                elif seg[4:4 + len(XMP_NS)] == XMP_NS:
                    s["xmp"].append(seg[4 + len(XMP_NS):])
                elif seg[4:4 + len(XMP_EXT_NS)] == XMP_EXT_NS:
                    s["xmp_ext"] += 1
                else:
                    s["unknown"].append("APP1")
            elif m == 0xE2:
                if seg[4:16] == ICC_HEADER:
                    continue
                if seg[4:8] == b"MPF" + NUL:
                    s["mpf"] = True
                else:
                    s["unknown"].append("APP2")
            elif m == 0xED:
                s["iptc"] = True
                fields = _iptc_fields(seg)
                if fields is None:
                    s["unknown"].append("APP13")
                else:
                    s["iptc_place"] |= bool(fields & _IPTC_PLACE)
                    s["iptc_person"] |= bool(fields & _IPTC_PERSON)
            elif m == 0xEB:
                s["c2pa"] = True
            elif m == 0xEE and seg[4:9] == b"Adobe":
                continue
            elif m == 0xFE:
                s["comment"] = True
            elif 0xE0 <= m <= 0xEF:
                s["unknown"].append(f"APP{m - 0xE0}")
    elif data.startswith(PNG_SIG):
        chunks, trailer = _png_parse(data)
        s["trailer"] = len(trailer)
        for ctype, _raw, body in chunks:
            if ctype == b"eXIf":
                tiff = tiff or body
            elif ctype == b"tIME":
                s["time_chunk"] = True
            elif ctype in (b"tEXt", b"zTXt", b"iTXt"):
                key = _png_text_key(body)
                if key == _PNG_XMP_KEY:
                    s["xmp"].append(_png_text_value(ctype, body))
                elif key not in _PNG_KEEP_TEXT and key.lower() not in AI_TEXT_KEYS:
                    s["text_other"].append(key)
            elif ctype[:1].islower() and ctype not in _PNG_KEEP and ctype not in (b"iCCP",):
                s["unknown"].append(ctype.decode("latin-1"))
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        chunks, trailer = _webp_parse(data)
        s["trailer"] = len(trailer)
        for ctype, body in chunks:
            if ctype == b"EXIF":
                tiff = tiff or (body[len(EXIF_HEADER):] if body.startswith(EXIF_HEADER) else body)
            elif ctype == b"XMP ":
                s["xmp"].append(body)
            elif ctype not in _WEBP_KEEP:
                s["unknown"].append(ctype.decode("latin-1"))
    else:
        raise ValueError("not a JPEG, PNG or WebP file")
    s["thumbnail"] |= bool(tiff) and _tiff_has_thumbnail(tiff)
    return s


# ---------------------------------------------------------------------------- public writers
def _rewrite(path: Path, opts: StripOptions | None, author: Authorship | None,
             camera: "CameraData | None" = None) -> bool:
    data = path.read_bytes()
    if data[:2] == SOI:
        new = _rewrite_jpeg(data, opts, author, camera)
    elif data.startswith(PNG_SIG):
        new = _rewrite_png(data, opts, author, camera)
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        new = _rewrite_webp(data, opts, author, camera)
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


# ---------------------------------------------------------------------------- camera data
# Sub-IFD (Exif) tag ids used by the camera presets. Serial numbers, owner name, unique image id and GPS are
# deliberately never written: mimicking a camera model must not invent a trackable device.
EXPOSURE_TIME, F_NUMBER, EXPOSURE_PROGRAM, ISO_SPEED = 0x829A, 0x829D, 0x8822, 0x8827
EXPOSURE_BIAS, METERING_MODE, FLASH, FOCAL_LENGTH = 0x9204, 0x9207, 0x9209, 0x920A
WHITE_BALANCE, FOCAL_35MM, EXIF_VERSION = 0xA403, 0xA405, 0x9000

EXPOSURE_PROGRAMS = {0: "Not defined", 1: "Manual", 2: "Program AE", 3: "Aperture priority", 4: "Shutter priority",
                     5: "Creative", 6: "Action", 7: "Portrait", 8: "Landscape"}
METERING_MODES = {0: "Unknown", 1: "Average", 2: "Center-weighted", 3: "Spot", 4: "Multi-spot",
                  5: "Multi-segment (pattern)", 6: "Partial"}
FLASH_MODES = {0: "No flash", 16: "Flash off (compulsory)", 24: "Flash off (auto)", 1: "Flash fired",
               25: "Flash fired (auto)"}
WHITE_BALANCES = {0: "Auto", 1: "Manual"}


@dataclass
class CameraData:
    """EXIF values mimicking a camera. Everything is text; a blank field is left unchanged in the file."""
    make: str = ""
    model: str = ""
    lens_make: str = ""
    lens_model: str = ""
    software: str = ""             # firmware string the camera writes
    f_number: str = ""             # "2.8"
    exposure_time: str = ""        # "1/250" or seconds ("0.5")
    iso: str = ""
    focal_length: str = ""         # mm
    focal_length_35mm: str = ""    # mm
    exposure_program: str = ""     # EXIF code, see EXPOSURE_PROGRAMS
    metering_mode: str = ""        # EXIF code, see METERING_MODES
    flash: str = ""                # EXIF code, see FLASH_MODES
    white_balance: str = ""        # 0 auto / 1 manual
    exposure_bias: str = ""        # EV, "0", "-0.7", "+1", "-2/3"
    date_time_original: str = ""   # "YYYY:MM:DD HH:MM:SS" (also accepts "-" and "T")

    def is_empty(self) -> bool:
        return not any(str(v).strip() for v in vars(self).values())


@dataclass
class Variation:
    """Per-image jitter inside ranges, written as text ('1/2000-1/125', '100-3200', '2.8-5.6')."""
    shutter: str = ""
    iso: str = ""
    f_number: str = ""


@dataclass
class TimeSeries:
    """Timestamps start, start+step, start+2*step ... (one per image, in the order given)."""
    start: str = ""                # "YYYY-MM-DD HH:MM:SS"
    step_seconds: int = 60


def _fraction(text: str) -> Fraction:
    t = str(text).strip().replace("−", "-").replace(",", ".")
    t = t.lstrip("+") if t.startswith("+") else t
    try:
        return Fraction(t)
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"'{text}' is not a number") from None


def _rational(text: str) -> IFDRational:
    f = _fraction(text).limit_denominator(1000)
    return IFDRational(f.numerator, f.denominator)


def _u16(text: str, what: str) -> int:
    try:
        v = int(float(str(text).strip()))
    except ValueError:
        raise ValueError(f"{what} '{text}' is not a whole number") from None
    if not 0 <= v <= 65535:
        raise ValueError(f"{what} {v} is out of range")
    return v


def _exif_datetime(text: str) -> str:
    t = text.strip().replace("T", " ")
    try:
        return datetime.strptime(t.replace("-", ":").replace("/", ":"), "%Y:%m:%d %H:%M:%S").strftime("%Y:%m:%d %H:%M:%S")
    except ValueError:
        raise ValueError(f"date/time '{text}' must look like 2026-01-31 14:05:09") from None


def _apply_camera_tags(ex: "Image.Exif", cam: CameraData) -> None:
    """Write the non-blank camera fields into an Exif object (IFD0 + Exif sub-IFD)."""
    for tag, value in ((MAKE, cam.make), (MODEL, cam.model), (SOFTWARE, cam.software)):
        if value.strip():
            ex[tag] = exif_ascii(value.strip())
    sub: dict = {}
    for tag, value in ((LENS_MAKE, cam.lens_make), (LENS_MODEL, cam.lens_model)):
        if value.strip():
            sub[tag] = exif_ascii(value.strip())
    for tag, value in ((F_NUMBER, cam.f_number), (EXPOSURE_TIME, cam.exposure_time),
                       (FOCAL_LENGTH, cam.focal_length), (EXPOSURE_BIAS, cam.exposure_bias)):
        if value.strip():
            sub[tag] = _rational(value)
    for tag, value, what in ((ISO_SPEED, cam.iso, "ISO"), (FOCAL_35MM, cam.focal_length_35mm, "35mm focal length"),
                             (EXPOSURE_PROGRAM, cam.exposure_program, "exposure program"),
                             (METERING_MODE, cam.metering_mode, "metering mode"), (FLASH, cam.flash, "flash"),
                             (WHITE_BALANCE, cam.white_balance, "white balance")):
        if value.strip():
            sub[tag] = _u16(value, what)
    if cam.date_time_original.strip():
        sub[DT_ORIGINAL] = sub[DT_DIGITIZED] = _exif_datetime(cam.date_time_original)
    if sub:
        ifd = ex.get_ifd(EXIF_IFD)
        ifd.setdefault(EXIF_VERSION, b"0232")
        ifd.update(sub)


# Standard third-stop scales used when jittering.
_SHUTTERS = sorted({Fraction(1, d) for d in (8000, 6400, 5000, 4000, 3200, 2500, 2000, 1600, 1250, 1000, 800, 640,
                                              500, 400, 320, 250, 200, 160, 125, 100, 80, 60, 50, 40, 30, 25, 20,
                                              15, 13, 10, 8, 6, 5, 4, 3)} | {Fraction(2, 5), Fraction(1, 2), Fraction(4, 5),
                                                                           Fraction(1), Fraction(2)})
_ISOS = [50, 64, 80, 100, 125, 160, 200, 250, 320, 400, 500, 640, 800, 1000, 1250, 1600, 2000, 2500, 3200, 4000,
         5000, 6400, 8000, 10000, 12800, 16000, 20000, 25600, 32000, 40000, 51200, 64000, 80000, 102400]
_APERTURES = [1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 2.8, 3.2, 3.5, 4.0, 4.5, 5.0, 5.6, 6.3, 7.1, 8.0,
              9.0, 10.0, 11.0, 13.0, 14.0, 16.0, 18.0, 20.0, 22.0]


def _range(text: str) -> tuple[Fraction, Fraction] | None:
    t = text.strip()
    if not t:
        return None
    parts = t.replace("–", "-").replace(" ", "").split("-")
    if len(parts) != 2 or not all(parts):  # "1/2000-1/125": exactly one dash, no negative numbers
        raise ValueError(f"'{text}' is not a range like 100-3200")
    a, b = _fraction(parts[0]), _fraction(parts[1])
    return (a, b) if a <= b else (b, a)


def vary_camera(cam: CameraData, name: str, var: Variation) -> CameraData:
    """Copy of ``cam`` with exposure time, ISO and f-number picked from standard third-stop values inside the
    ranges. Seeded from the file name, so applying again gives the same values."""
    rng = random.Random(zlib.crc32(Path(name).name.lower().encode("utf-8")))
    out = replace(cam)
    r = _range(var.shutter)
    if r:
        pool = [s for s in _SHUTTERS if r[0] <= s <= r[1]] or [r[0]]
        pick = rng.choice(pool)
        out.exposure_time = f"{pick.numerator}/{pick.denominator}"
    r = _range(var.iso)
    if r:
        pool = [i for i in _ISOS if r[0] <= i <= r[1]] or [int(r[0])]
        out.iso = str(rng.choice(pool))
    r = _range(var.f_number)
    if r:
        pool = [a for a in _APERTURES if r[0] <= Fraction(str(a)) <= r[1]] or [float(r[0])]
        out.f_number = f"{rng.choice(pool):g}"
    return out


def series_time(series: TimeSeries, index: int) -> str:
    start = datetime.strptime(_exif_datetime(series.start), "%Y:%m:%d %H:%M:%S")
    return (start + timedelta(seconds=int(series.step_seconds) * index)).strftime("%Y:%m:%d %H:%M:%S")


def validate_camera(data: CameraData, variation: Variation | None = None, series: TimeSeries | None = None) -> None:
    """Raise ValueError with a readable message if any field can't be written (checked before touching files)."""
    cam = vary_camera(data, "check.jpg", variation) if variation else replace(data)
    if series is not None and series.start.strip():
        cam.date_time_original = series_time(series, 0)
    _apply_camera_tags(Image.Exif(), cam)


def apply_camera(path: str | os.PathLike, data: CameraData, variation: Variation | None = None,
                 series: TimeSeries | None = None, index: int = 0) -> bool:
    """Write camera EXIF (make/model/lens/exposure...) losslessly. Blank fields are left unchanged.

    ``variation`` jitters exposure time / ISO / f-number per file name (deterministic); ``series`` replaces the
    fixed date/time by start + index * step. Never writes serial numbers, owner, image ids or GPS."""
    p = Path(path)
    cam = vary_camera(data, p.name, variation) if variation else replace(data)
    if series is not None and series.start.strip():
        cam.date_time_original = series_time(series, index)
    if cam.is_empty():
        return False
    return _rewrite(p, None, None, cam)


def set_text_field(path: str | os.PathLike, key: str, value: str) -> None:
    """Replace (or add) one PNG text field, losslessly. Empty value removes it."""
    p = Path(path)
    data = p.read_bytes()
    if not data.startswith(PNG_SIG):
        raise ValueError("text fields can only be edited on PNG files")
    out = [PNG_SIG]
    chunks, trailer = _png_parse(data)
    for ctype, raw, body in chunks:
        if ctype in (b"tEXt", b"zTXt", b"iTXt") and _png_text_key(body) == key:
            continue
        if ctype == b"IEND" and value:
            out.append(_png_itxt(key, value))
        out.append(raw)
    out.append(trailer)
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
    new_exif = _tiff_bytes(ex) if len(ex.keys()) else None
    if data[:2] == SOI:
        _atomic_write_bytes(p, _assemble_jpeg(list(_jpeg_segments(data)), new_exif))
    elif data.startswith(PNG_SIG):
        out = [PNG_SIG]
        chunks, trailer = _png_parse(data)
        for ctype, raw, _body in chunks:
            if ctype == b"eXIf":
                continue
            if ctype == b"IEND" and new_exif:
                out.append(_png_chunk(b"eXIf", new_exif))
            out.append(raw)
        out.append(trailer)
        _atomic_write_bytes(p, b"".join(out))
    else:
        raise ValueError("EXIF editing is supported for JPEG and PNG")


def pixels_equal(a: str | os.PathLike, b: str | os.PathLike) -> bool:
    """Verification helper: decoded pixels identical."""
    with Image.open(a) as x, Image.open(b) as y:
        return x.size == y.size and x.tobytes() == y.tobytes()
