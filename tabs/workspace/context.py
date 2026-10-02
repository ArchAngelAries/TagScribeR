"""The shared workspace: one open dataset used by every tab.

Opening a folder in any tab opens it everywhere. The context owns the
DatasetSession, the Qt list model, the thumbnail cache and the undo stack, so
edits, unsaved state and undo history are consistent across Gallery, Auto
Caption and future tools. Each tab shows its own DatasetBrowser (own
selection, filter, zoom) over the shared model.
"""
from __future__ import annotations

import logging
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
        self._commit_hooks: list[Callable[[], None]] = []
        self._info_signals = _InfoSignals()
        self._info_signals.batch.connect(self._on_info_batch)
        self._info_pool = QThreadPool(self)
        self._info_pool.setMaxThreadCount(1)
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
        try:
            session = DatasetSession.open(folder, recursive=recursive, caption_ext=self.cfg.get("caption.extension"))
        except OSError as e:
            if parent is not None:
                QMessageBox.critical(parent, "Error", f"Failed to read folder:\n{e}")
            return False
        self.generation += 1
        self.loader.reset()
        self.undo_stack.clear()
        self.job_state.clear()
        self.session = session
        self.folder = folder
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
        if self.session and not any(s == JOB_WORKING for s in self.job_state.values()):
            n = len(self.reload_from_disk([e.key for e in self.session.entries]))
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

    def remove_entries(self, keys: list[str]) -> None:
        self.undo_stack.clear()  # history may reference removed images
        self.model.remove_keys(keys)
        self.dirty_changed.emit()

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
