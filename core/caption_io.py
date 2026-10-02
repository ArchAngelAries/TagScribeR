"""Reading and writing caption sidecar files safely.

Rules enforced here (previously each tab did its own unguarded file I/O):

* Writes are atomic: content goes to a temp file which then replaces the
  target, so a crash mid-write can never leave a truncated caption.
* Unchanged captions are not rewritten (preserves mtimes, avoids churn).
* An empty caption never *creates* a new file, so "Save All" no longer
  sprinkles empty .txt files next to every uncaptioned image.
* Before an existing caption is overwritten, the first version of the day is
  copied to ``user_data/caption_backups/<date>/...`` for recovery.
* Reads tolerate a UTF-8 BOM and legacy cp1252 files instead of failing silently.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
from datetime import date
from pathlib import Path

from core import paths

log = logging.getLogger(__name__)

DEFAULT_CAPTION_EXT = ".txt"


def caption_path(image_path: str | os.PathLike, ext: str = DEFAULT_CAPTION_EXT) -> Path:
    p = Path(image_path)
    return p.with_name(p.stem + ext)


def read_text_file(path: Path) -> str:
    data = path.read_bytes()
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


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
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
    """Write a caption. Returns True if the file changed. Raises OSError on failure."""
    cp = caption_path(image_path, ext)
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
