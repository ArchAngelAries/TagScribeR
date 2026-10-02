"""Safe file operations for images and their caption sidecars.

* Copies never silently overwrite: name collisions get a numbered suffix
  (or are skipped / overwritten only when explicitly requested).
* Deletes go to the system Recycle Bin / Trash, never a permanent unlink.
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from core.caption_io import caption_path

log = logging.getLogger(__name__)

CAPTION_SIDECAR_EXTS = (".txt", ".caption", ".json")


def unique_path(dest: Path) -> Path:
    """Return ``dest`` or ``name (n).ext`` if it already exists."""
    if not dest.exists():
        return dest
    n = 1
    while True:
        candidate = dest.with_name(f"{dest.stem} ({n}){dest.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def unique_stem_for_pair(dest_dir: Path, stem: str, image_ext: str) -> str:
    """A stem free for both the image and its caption in ``dest_dir``."""
    def taken(s: str) -> bool:
        return (dest_dir / f"{s}{image_ext}").exists() or any(
            (dest_dir / f"{s}{e}").exists() for e in CAPTION_SIDECAR_EXTS)
    if not taken(stem):
        return stem
    n = 1
    while taken(f"{stem} ({n})"):
        n += 1
    return f"{stem} ({n})"


@dataclass
class CopyReport:
    copied: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    renamed: list[tuple[Path, Path]] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{len(self.copied)} copied"]
        if self.renamed:
            parts.append(f"{len(self.renamed)} renamed to avoid overwriting")
        if self.skipped:
            parts.append(f"{len(self.skipped)} skipped (already present)")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        return ", ".join(parts)


def copy_image_with_caption(src: Path, dest_dir: Path, on_conflict: str = "rename",
                            report: CopyReport | None = None) -> Path | None:
    """Copy an image and any same-stem caption sidecars into ``dest_dir``.

    on_conflict: 'rename' (default), 'skip', or 'overwrite'.
    """
    report = report if report is not None else CopyReport()
    src = Path(src)
    dest_dir = Path(dest_dir)
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        if src.parent.resolve() == dest_dir.resolve():
            report.skipped.append(src)
            return None
        stem = src.stem
        if (dest_dir / src.name).exists():
            if on_conflict == "skip":
                report.skipped.append(src)
                return None
            if on_conflict == "rename":
                stem = unique_stem_for_pair(dest_dir, src.stem, src.suffix)
        dest = dest_dir / f"{stem}{src.suffix}"
        shutil.copy2(src, dest)
        for ext in CAPTION_SIDECAR_EXTS:
            side = caption_path(src, ext)
            if side.is_file():
                shutil.copy2(side, dest_dir / f"{stem}{ext}")
        if stem != src.stem:
            report.renamed.append((src, dest))
        report.copied.append(dest)
        return dest
    except OSError as e:
        log.warning("Copy failed %s -> %s: %s", src, dest_dir, e)
        report.failed.append((src, str(e)))
        return None


def trash(path: Path) -> None:
    """Move a file or folder to the Recycle Bin. Raises if that's not possible."""
    from send2trash import send2trash  # small pure-python dependency
    send2trash(os.fspath(path))


@dataclass
class DeleteReport:
    trashed: list[Path] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)


def trash_images_with_captions(images: Iterable[Path]) -> DeleteReport:
    report = DeleteReport()
    for img in images:
        img = Path(img)
        try:
            trash(img)
            report.trashed.append(img)
        except Exception as e:  # send2trash raises OSError subclasses and its own TrashPermissionError
            log.warning("Could not move %s to the Recycle Bin: %s", img, e)
            report.failed.append((img, str(e)))
            continue
        for ext in CAPTION_SIDECAR_EXTS:
            side = caption_path(img, ext)
            if side.is_file():
                try:
                    trash(side)
                except Exception as e:
                    log.warning("Could not move caption %s to the Recycle Bin: %s", side, e)
    return report
