"""The shared workspace: one open dataset used by every tab.

Opening a folder in any tab opens it everywhere. The context owns the
DatasetSession, the Qt list model, the thumbnail cache and the undo stack, so
edits, unsaved state and undo history are consistent across Gallery, Auto
Caption and future tools. Each tab shows its own DatasetBrowser (own
selection, filter, zoom) over the shared model.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
from PySide6.QtGui import QUndoStack
from PySide6.QtWidgets import QInputDialog, QMessageBox, QWidget

from core import caption_io, dataset, fileops, paths
from core.config import settings
from core.dataset_session import Changes, DatasetSession, SaveReport
from tabs.workspace.commands import ChangeCommand
from tabs.workspace.model import DatasetModel
from tabs.workspace.thumbs import ThumbnailLoader

log = logging.getLogger(__name__)

# Per-image job states shown as card badges.
JOB_QUEUED, JOB_WORKING, JOB_FAILED = "queued", "working", "failed"


# A folder with more pictures than this is checked for changed captions on a worker thread.
ASYNC_SCAN_ROWS = 2000
_UNKNOWN = object()             # a caption file whose state was never recorded: always re-read


def _caption_snapshot(dirs, ext: str) -> dict:
    """{directory: {normalised file name: (mtime_ns, size)} or None when unreadable}: one directory pass that lists
    every file ending in the caption extension. Plain data in and out, so it can run on a worker thread."""
    norm_ext = os.path.normcase(ext)
    out: dict = {}
    for d in dirs:
        files: dict | None = {}
        try:
            with os.scandir(d) as it:
                for de in it:
                    name = os.path.normcase(de.name)
                    if not name.endswith(norm_ext):
                        continue
                    try:
                        if de.is_file():
                            st = de.stat()
                            files[name] = (st.st_mtime_ns, st.st_size)
                    except OSError:
                        continue
        except OSError:
            files = None
        out[d] = files
    return out


def _caption_items(session: DatasetSession) -> list[tuple[str, str, str]]:
    """(image key, its directory, normalised caption file name) for every entry."""
    ext = session.caption_ext
    return [(e.key, str(e.path.parent), os.path.normcase(caption_io.caption_path(e.path, ext).name))
            for e in session.entries]


def _changed_captions(items, sigs: dict, ext: str) -> list[tuple[str, object]]:
    """[(key, current signature or None)] for every caption file that differs from the recorded signature
    (changed, appeared or disappeared). A directory that cannot be listed is skipped: nothing is assumed about it."""
    snapshot = _caption_snapshot({d for _k, d, _n in items}, ext)
    changed = []
    for key, d, name in items:
        files = snapshot.get(d)
        if files is None:
            continue
        now = files.get(name)
        if sigs.get(key, _UNKNOWN) != now:
            changed.append((key, now))
    return changed


class _ScanSignals(QObject):
    result = Signal(int, list)  # generation, [(key, signature)]


class _CaptionScan(QRunnable):
    """Looks for caption files that changed on disk (dates and sizes only) without touching the model or any widget."""

    def __init__(self, items, sigs: dict, ext: str, generation: int, signals: _ScanSignals):
        super().__init__()
        self.items, self.sigs, self.ext, self.generation, self.signals = items, sigs, ext, generation, signals

    def run(self):
        try:
            changed = _changed_captions(self.items, self.sigs, self.ext)
        except Exception as e:                      # never leave the owner waiting for a result
            log.warning("Caption check failed: %s", e)
            changed = []
        try:
            self.signals.result.emit(self.generation, changed)
        except RuntimeError:
            pass  # app shutting down


class _InfoSignals(QObject):
    batch = Signal(int, list)   # generation, [(key, w, h)]


class _InfoScan(QRunnable):
    """Reads image dimensions (headers only) for the whole folder in the background."""

    def __init__(self, keys: list[str], generation: int, signals: _InfoSignals, owner):
        super().__init__()
        self.keys, self.generation, self.signals, self.owner = keys, generation, signals, owner

    def run(self):
        out = []
        for key in self.keys:
            if self.owner.generation != self.generation:
                return
            info = dataset.read_image_info(key)
            out.append((key, info.width, info.height))
            if len(out) >= 200:
                self._emit(out)
                out = []
        if out:
            self._emit(out)

    def _emit(self, items):
        try:
            self.signals.batch.emit(self.generation, items)
        except RuntimeError:
            pass  # app shutting down


class WorkspaceContext(QObject):
    session_changed = Signal()          # a different folder was opened (or closed)
    entries_changed = Signal(list)      # keys whose caption / state changed
    dimensions_loaded = Signal(list)    # keys whose width/height became known
    dirty_changed = Signal()
    entries_removed = Signal(list)      # keys whose files left the dataset (trashed, moved)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cfg = settings()
        self.session: DatasetSession | None = None
        self.folder = ""
        self.generation = 0
        self.model = DatasetModel(self)
        self.loader = ThumbnailLoader(self)
        self.undo_stack = QUndoStack(self)
        self.job_state: dict[str, str] = {}
        self.health_report = None
        self.project = None                 # per-folder settings (core.projects.Project)
        self.health_cache: dict[tuple[str, float, int], object] = {}
        self._commit_hooks: list[Callable[[], None]] = []
        self._info_signals = _InfoSignals()
        self._info_signals.batch.connect(self._on_info_batch)
        self._info_pool = QThreadPool(self)
        self._info_pool.setMaxThreadCount(1)
        # What the caption files looked like when each caption was last read, to find the ones that changed later
        # without reading them all again (see pick_up_external_changes).
        self._cap_sigs: dict[str, object] = {}
        self._cap_items: list[tuple[str, str, str]] = []
        self._scan_signals = _ScanSignals()
        self._scan_signals.result.connect(self._on_caption_scan)
        self._scan_pool = QThreadPool(self)
        self._scan_pool.setMaxThreadCount(1)
        self._scan_busy = False
        self.loader.ready.connect(lambda key: self.model.refresh([key]))
        self.loader.dimensions.connect(self._on_dimensions)

    # ------------------------------------------------------------------ hooks
    def add_commit_hook(self, fn: Callable[[], None]) -> None:
        """Called before save/undo/batch ops so in-progress typing is committed first."""
        self._commit_hooks.append(fn)

    def commit_pending_edits(self) -> None:
        for fn in self._commit_hooks:
            fn()

    # ------------------------------------------------------------------ folder
    def has_unsaved(self) -> bool:
        self.commit_pending_edits()
        return bool(self.session and self.session.dirty_entries)

    def confirm_discard(self, parent: QWidget) -> bool:
        """Ask what to do with unsaved edits before they'd be lost. True = proceed."""
        if not self.has_unsaved():
            return True
        n = len(self.session.dirty_entries)
        box = QMessageBox(QMessageBox.Warning, "Unsaved changes",
                          f"{n} caption(s) in this folder have unsaved changes.",
                          QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, parent)
        box.setDefaultButton(QMessageBox.Save)
        r = box.exec()
        if r == QMessageBox.Save:
            return self.save(parent)
        return r == QMessageBox.Discard

    def open_folder(self, folder: str, recursive: bool = False, parent: QWidget | None = None) -> bool:
        ext = self.cfg.get("caption.extension")
        # Dates and sizes of the caption files, taken just before they are read: a file that changes in between is
        # then seen as changed at the next check and read again, never missed.
        try:
            dirs = [str(Path(dp)) for dp, _dn, _fn in os.walk(folder)] if recursive else [str(Path(folder))]
            snapshot = _caption_snapshot(dirs, ext or caption_io.DEFAULT_CAPTION_EXT)
        except Exception:
            snapshot = {}
        try:
            session = DatasetSession.open(folder, recursive=recursive, caption_ext=ext)
        except OSError as e:
            if parent is not None:
                QMessageBox.critical(parent, "Error", f"Failed to read folder:\n{e}")
            return False
        self.generation += 1
        self.loader.reset()
        self.undo_stack.clear()
        self.job_state.clear()
        self.health_report = None
        from core.projects import Project
        self.project = Project(folder)
        self.session = session
        self.folder = folder
        self._cap_items = _caption_items(session)
        self._cap_sigs = {}
        for key, d, name in self._cap_items:
            files = snapshot.get(d)
            self._cap_sigs[key] = _UNKNOWN if files is None else files.get(name)
        self.model.set_session(session)
        self.cfg.update({"ui.last_folder": folder, "ui.recursive_scan": recursive,
                         "ui.recent_folders": list(dict.fromkeys([folder] + self.cfg.get("ui.recent_folders", [])))[:10]})
        self._info_pool.start(_InfoScan([e.key for e in session.entries], self.generation, self._info_signals, self))
        self.session_changed.emit()
        self.dirty_changed.emit()
        return True

    def recent_folders(self) -> list[str]:
        import os
        return [f for f in self.cfg.get("ui.recent_folders", []) if os.path.isdir(f)]

    # ------------------------------------------------------------------ dimensions
    def _on_dimensions(self, key: str, w: int, h: int) -> None:
        if self.session:
            e = self.session.get(key)
            if e is not None and not e.width:
                e.width, e.height, e.info_loaded = w, h, True

    def _on_info_batch(self, generation: int, items: list) -> None:
        if generation != self.generation or not self.session:
            return
        for key, w, h in items:
            e = self.session.get(key)
            if e:
                e.width, e.height, e.info_loaded = w, h, True
        keys = [k for k, _, _ in items]
        self.model.refresh(keys)
        self.dimensions_loaded.emit(keys)

    # ------------------------------------------------------------------ editing
    def _changed(self, keys: list[str]) -> None:
        self.model.refresh(keys)
        self.entries_changed.emit(keys)
        self.dirty_changed.emit()

    def push(self, changes: Changes, description: str, merge_key: str | None = None) -> int:
        if not changes or not self.session:
            return 0
        self.undo_stack.push(ChangeCommand(self.session, changes, description, self._changed, merge_key))
        return len(changes)

    def apply_transform(self, keys: list[str], fn, description: str) -> Changes:
        """Compute (not apply) the change set of ``fn`` over ``keys``."""
        if not self.session:
            return {}
        self.commit_pending_edits()
        return self.session.transform(keys, fn)

    def undo(self) -> None:
        self.commit_pending_edits()
        self.undo_stack.undo()

    def redo(self) -> None:
        self.undo_stack.redo()

    # ------------------------------------------------------------------ disk
    def save(self, parent: QWidget | None = None, keys: list[str] | None = None) -> bool:
        if not self.session:
            return True
        self.commit_pending_edits()
        targets = keys if keys is not None else [e.key for e in self.session.dirty_entries]
        report: SaveReport = self.session.save(targets)
        self._changed(targets)
        if report.failed:
            if parent is not None:
                QMessageBox.critical(parent, "Some captions were not saved",
                                     "\n".join(f"{Path(k).name}: {e}" for k, e in report.failed[:20]))
            return False
        self.last_save_count = report.saved
        return True

    def reload_from_disk(self, keys: list[str]) -> list[str]:
        """Re-read captions written by someone else (an AI job, another app).
        Entries with unsaved edits are skipped so nothing typed is lost."""
        if not self.session:
            return []
        changed = []
        for key in keys:
            e = self.session.get(key)
            if e is None or e.dirty:
                continue
            disk = caption_io.read_caption(e.path, self.session.caption_ext)
            if disk != e.saved_text:
                e.text = e.saved_text = disk
                changed.append(key)
        if changed:
            self._changed(changed)
        return changed

    def pick_up_external_changes(self) -> None:
        """Re-read the captions that were changed, added or removed on disk since they were last read (an AI job, a
        text editor). Finds them by date and size in one directory pass, on a worker thread when the folder is
        large, and reads only those files. Entries with unsaved edits are never touched."""
        if not self.session or any(s == JOB_WORKING for s in self.job_state.values()):
            return
        ext = self.session.caption_ext
        if len(self._cap_items) <= ASYNC_SCAN_ROWS:
            self._apply_caption_changes(self.generation, _changed_captions(self._cap_items, self._cap_sigs, ext))
            return
        if self._scan_busy:
            return
        self._scan_busy = True
        self._scan_pool.start(_CaptionScan(self._cap_items, dict(self._cap_sigs), ext, self.generation,
                                           self._scan_signals))

    def _on_caption_scan(self, generation: int, changed: list) -> None:
        self._scan_busy = False
        self._apply_caption_changes(generation, changed)

    def _apply_caption_changes(self, generation: int, changed: list) -> None:
        if generation != self.generation or not self.session:
            return                                  # another folder was opened while the check ran
        if any(s == JOB_WORKING for s in self.job_state.values()):
            return                                  # a job started meanwhile: it will reload what it wrote
        todo = []
        for key, sig in changed:
            e = self.session.get(key)
            if e is not None and not e.dirty:       # an unsaved edit stays, and the file is looked at again next time
                todo.append((key, sig))
        n = len(self.reload_from_disk([k for k, _s in todo]))
        for key, sig in todo:
            self._cap_sigs[key] = sig
        if n:
            log.info("Picked up %d externally changed caption(s)", n)

    def images_modified(self, keys: list[str]) -> None:
        """Pixels of these files changed on disk (e.g. overwritten by the editor): refresh
        thumbnails, dimensions, size and date without touching captions."""
        if not self.session:
            return
        for key in keys:
            e = self.session.get(key)
            if e is None:
                continue
            self.loader.invalidate(key)
            info = dataset.read_image_info(key)
            e.width, e.height, e.info_loaded = info.width, info.height, True
            try:
                st = e.path.stat()
                e.file_size, e.mtime = st.st_size, st.st_mtime
            except OSError:
                pass
        self.model.refresh(keys)
        self.entries_changed.emit(keys)

    def rescan(self, parent: QWidget | None = None) -> bool:
        """Reload the folder (new files appear), keeping the user's choice about unsaved edits."""
        if not self.folder:
            return False
        if parent is not None and not self.confirm_discard(parent):
            return False
        return self.open_folder(self.folder, self.cfg.get("ui.recursive_scan", False), parent)

    def apply_health(self, report) -> None:
        """Attach health-check flags to entries (used by cards and the flag: filter)."""
        self.health_report = report
        flags = report.flags()
        if not self.session:
            return
        changed = []
        for e in self.session.entries:
            new = flags.get(e.key, set())
            if new != e.flags:
                e.flags = set(new)
                changed.append(e.key)
        self.model.refresh(changed)
        self.dimensions_loaded.emit(changed)  # lets filters using flag: re-evaluate

    def remove_entries(self, keys: list[str]) -> None:
        self.undo_stack.clear()  # history may reference removed images
        self.model.remove_keys(keys)
        self.dirty_changed.emit()
        self.entries_removed.emit(list(keys))

    # ------------------------------------------------------------------ job badges
    def set_job_state(self, keys: list[str], state: str | None) -> None:
        for k in keys:
            if state is None:
                self.job_state.pop(k, None)
            else:
                self.job_state[k] = state
        self.model.refresh(keys)

    # ------------------------------------------------------------------ shared actions
    def copy_to_collection(self, parent: QWidget, keys: list[str]) -> None:
        if not keys:
            QMessageBox.information(parent, "No selection", "Select images first.")
            return
        root = Path(self.cfg.get("paths.collections_dir") or paths.DEFAULT_COLLECTIONS_DIR)
        existing = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
        name, ok = QInputDialog.getItem(parent, "Copy to Collection", "Collection name (new or existing):",
                                        existing, 0, True)
        if not ok or not name.strip():
            return
        if not fileops.valid_collection_name(name.strip()):
            QMessageBox.warning(parent, "Copy to Collection", f"'{name.strip()}' can't be a collection name: use a "
                                "plain folder name (no slashes, no '..').")
            return
        if self.session and any(self.session.get(k) and self.session.get(k).dirty for k in keys):
            if not self.save(parent, keys):
                return  # copy what the user sees; abort if saving failed
        report = fileops.CopyReport()
        for k in keys:
            fileops.copy_image_with_caption(Path(k), root / name.strip(), report=report)
        msg = f"Collection '{name.strip()}': {report.summary()}."
        if report.failed:
            msg += "\n\n" + "\n".join(f"{p.name}: {e}" for p, e in report.failed[:20])
            QMessageBox.warning(parent, "Copy to Collection", msg)
        else:
            QMessageBox.information(parent, "Copy to Collection", msg)

    def trash(self, parent: QWidget, keys: list[str]) -> None:
        from tabs.common import confirm
        if not keys or not confirm(parent, "Move to Recycle Bin",
                                   f"Move {len(keys)} image(s) and their captions to the Recycle Bin?",
                                   destructive=True):
            return
        report = fileops.trash_images_with_captions([Path(k) for k in keys])
        self.remove_entries([str(p) for p in report.trashed])
        if report.failed:
            QMessageBox.warning(parent, "Some files were not moved",
                                "\n".join(f"{p.name}: {e}" for p, e in report.failed[:20]))


_context: WorkspaceContext | None = None


def workspace() -> WorkspaceContext:
    global _context
    if _context is None:
        _context = WorkspaceContext()
    return _context
