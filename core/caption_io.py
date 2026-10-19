"""Reading and writing caption sidecar files safely.

Rules enforced here (previously each tab did its own unguarded file I/O):

* Writes are atomic: content goes to a temp file which then replaces the
  target, so a crash mid-write can never leave a truncated caption.
* Unchanged captions are not rewritten (preserves mtimes, avoids churn).
* An empty caption never *creates* a new file, so "Save All" no longer
  sprinkles empty .txt files next to every uncaptioned image.
* Before an existing caption is overwritten, the first version of the day is
  copied to ``user_data/caption_backups/<date>/...`` for recovery.
* Reads tolerate a UTF-8 BOM, UTF-16 (with or without BOM) and legacy cp1252 files instead of failing silently.
  A UTF-16 caption that gets edited is saved back as UTF-8.
* Written files keep the original's permission bits (new ones get the umask default, not mkstemp's 0600).
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import stat
import tempfile
from datetime import date
from pathlib import Path

from core import paths

log = logging.getLogger(__name__)

DEFAULT_CAPTION_EXT = ".txt"
# a caption is never written under one of these (core.dataset.IMAGE_EXTS and the other picture types)
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".jxl", ".avif", ".heic"}


def caption_path(image_path: str | os.PathLike, ext: str = DEFAULT_CAPTION_EXT) -> Path:
    p = Path(image_path)
    return p.with_name(p.stem + ext)


def _looks_utf16(data: bytes) -> str | None:
    """'utf-16' (BOM), 'utf-16-le' / 'utf-16-be' (BOM-less, from the NUL pattern) or None."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    if len(data) >= 4 and len(data) % 2 == 0 and b"\x00" in data:
        even, odd = data[0::2], data[1::2]
        # ASCII-heavy UTF-16 has a NUL in nearly every second byte; UTF-8 / cp1252 text has none.
        if odd.count(0) >= 0.4 * len(odd) and even.count(0) == 0:
            return "utf-16-le"
        if even.count(0) >= 0.4 * len(even) and odd.count(0) == 0:
            return "utf-16-be"
    return None


def read_text_file(path: Path) -> str:
    data = path.read_bytes()
    enc16 = _looks_utf16(data)
    if enc16:
        try:
            text = data.decode(enc16)
            log.info("%s is UTF-16 encoded; it will be saved as UTF-8 if edited", path)
            return text
        except UnicodeDecodeError:
            pass
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def read_caption(image_path: str | os.PathLike, ext: str = DEFAULT_CAPTION_EXT) -> str:
    """Return the caption text for an image, or '' if there is none or it can't be read."""
    cp = caption_path(image_path, ext)
    try:
        return read_text_file(cp) if cp.is_file() else ""
    except OSError as e:
        log.warning("Could not read caption %s: %s", cp, e)
        return ""


def has_caption(image_path: str | os.PathLike, ext: str = DEFAULT_CAPTION_EXT) -> bool:
    cp = caption_path(image_path, ext)
    try:
        return cp.is_file() and cp.stat().st_size > 0
    except OSError:
        return False


def _read_umask() -> int:
    old = os.umask(0)   # the only way to read it; done once at import, not per write (it is process-wide)
    os.umask(old)
    return old


_UMASK = _read_umask()


def apply_mode(target: Path, tmp: str) -> None:
    """mkstemp creates 0600 files: copy the target's mode, or use the umask default for a new file."""
    try:
        mode = stat.S_IMODE(os.stat(target).st_mode)
    except OSError:
        mode = 0o666 & ~_UMASK
    try:
        os.chmod(tmp, mode)
    except OSError:
        pass


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
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


def backup_path_for(target: Path, day: date | None = None) -> Path:
    day = day or date.today()
    folder = target.parent.resolve()
    digest = hashlib.sha1(str(folder).lower().encode("utf-8")).hexdigest()[:8]
    return paths.CAPTION_BACKUP_DIR / day.isoformat() / f"{folder.name}-{digest}" / target.name


def _backup_once_per_day(target: Path) -> None:
    dest = backup_path_for(target)
    if dest.exists():
        return  # keep the earliest version of the day
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, dest)
    except OSError as e:
        # A failed backup must not block saving, but it should be visible in the log.
        log.warning("Caption backup failed for %s: %s", target, e)


def write_caption(image_path: str | os.PathLike, text: str, ext: str = DEFAULT_CAPTION_EXT,
                  backup: bool = True) -> bool:
    """Write a caption. Returns True if the file changed. Raises OSError on failure, and ValueError when the caption
    path would be an image (a caption extension like .png): a caption must never replace a picture."""
    cp = caption_path(image_path, ext)
    if cp.suffix.lower() in _IMAGE_SUFFIXES or cp.resolve() == Path(image_path).resolve():
        raise ValueError(f"refusing to write a caption to {cp.name}: that is an image file name")
    exists = cp.is_file()
    if not exists and not text.strip():
        return False
    if exists:
        try:
            if read_text_file(cp) == text:
                return False
        except OSError:
            pass
        if backup:
            _backup_once_per_day(cp)
    atomic_write_text(cp, text)
    return True
