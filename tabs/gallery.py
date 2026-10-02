"""Gallery: the dataset workspace.

A virtualized grid (QListView + painted cards) over a DatasetSession, with an
inspector, batch tools and tag statistics. Edits stay in memory (undoable)
until Save; saving uses the safe caption writer.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from PySide6.QtCore import QItemSelectionModel, QObject, QRunnable, QSize, Qt, QThreadPool, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut, QUndoStack
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QInputDialog, QLabel,
                               QLineEdit, QListView, QMenu, QMessageBox, QPushButton, QSlider, QSplitter,
                               QTabWidget, QToolButton, QVBoxLayout, QWidget)

from core import captions, dataset, fileops, paths
from core.config import settings
from core.dataset_session import DatasetSession, op_add_tags, op_remove_tags, op_replace_tag
from core.widgets import AutoTagDialog
from inference.base import CaptionRequest, ProviderSpec
from inference.worker import SAVE_NONE, BatchJob, start_job
from tabs.common import confirm, show_job_summary
from tabs.workspace.commands import ChangeCommand
from tabs.workspace.delegate import CardDelegate
from tabs.workspace.model import ENTRY_ROLE, KEY_ROLE, SORT_MODES, DatasetModel, FilterProxy
from tabs.workspace.panels import BatchPanel, InspectorPanel, TagStatsPanel
from tabs.workspace.thumbs import ThumbnailLoader

log = logging.getLogger(__name__)

QUERY_HELP = (
    "Filter examples:\n"
    "  smile                  caption or file name contains 'smile'\n"
    "  tag:\"long hair\"        has that exact tag      -tag:blurry   lacks it\n"
    "  tag:*hair              wildcard tag\n"
    "  missing:caption        no caption       is:unsaved   edited, not saved\n"
    "  res:<768               shorter side under 768 px   (also w: h: mp:)\n"
    "  ar:>1.5                wide images (width/height)\n"
    "  tags:>40  len:<20      tag count / caption length\n"
    "  name:img_*  ext:png    file name / type\n"
    "Combine terms with spaces; prefix any term with - to negate it."
)


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
                self.signals.batch.emit(self.generation, out)
                out = []
        if out:
            self.signals.batch.emit(self.generation, out)


class GalleryTab(QWidget):
    image_selected = Signal(str)

    def __init__(self):
        super().__init__()
        self.cfg = settings()
        self.session: DatasetSession | None = None
        self.current_folder = ""
        self.generation = 0
        self._tag_generation = 0
        self.tag_worker = None
        self.pending_tag_updates: dict[str, str] = {}
        self.auto_tag_settings: dict = {}
        self._clipboard_caption: str | None = None
        self.undo_stack = QUndoStack(self)

        self.loader = ThumbnailLoader(self)
        self.model = DatasetModel(self)
        self.proxy = FilterProxy(self)
        self.proxy.setSourceModel(self.model)
        self.delegate = CardDelegate(self.loader, self)
        self.loader.ready.connect(self._on_thumb_ready)
        self.loader.dimensions.connect(self._on_dimensions)
        self._info_signals = _InfoSignals()
        self._info_signals.batch.connect(self._on_info_batch)
        self._info_pool = QThreadPool(self)
        self._info_pool.setMaxThreadCount(1)

        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        splitter = QSplitter(Qt.Horizontal)

        # ---------------- left: toolbar + grid ----------------
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self.btn_open = QPushButton("📂 Open Folder")
        self.btn_open.setToolTip("Open a dataset folder (Ctrl+O)")
        self.btn_open.setStyleSheet("background-color: #00b894; color: white; font-weight: bold;")
        self.btn_open.clicked.connect(self.select_folder)
        self.btn_recent = QToolButton()
        self.btn_recent.setText("▾")
        self.btn_recent.setToolTip("Recent folders")
        self.btn_recent.setPopupMode(QToolButton.InstantPopup)
        self.btn_recent.setMenu(QMenu(self.btn_recent))
        self.btn_recent.menu().aboutToShow.connect(self._fill_recent_menu)
        self.chk_recursive = QCheckBox("Subfolders")
        self.chk_recursive.setToolTip("Also load images from subfolders (applies when a folder is opened)")
        self.chk_recursive.setChecked(self.cfg.get("ui.recursive_scan", False))
        self.inp_filter = QLineEdit()
        self.inp_filter.setPlaceholderText("Filter…  e.g.  tag:1girl -tag:blurry missing:caption res:<768   (Ctrl+F)")
        self.inp_filter.setToolTip(QUERY_HELP)
        self.inp_filter.setClearButtonEnabled(True)
        self._filter_timer = QTimer(self, singleShot=True, interval=250)
        self._filter_timer.timeout.connect(self.apply_filter)
        self.inp_filter.textChanged.connect(lambda _: self._filter_timer.start())
        self.combo_sort = QComboBox()
        self.combo_sort.addItems(list(SORT_MODES))
        self.combo_sort.setToolTip("Sort order")
        self.combo_sort.currentTextChanged.connect(self._apply_sort)
        self.btn_desc = QToolButton(text="↓", checkable=True)
        self.btn_desc.setToolTip("Reverse sort order")
        self.btn_desc.toggled.connect(lambda _: self._apply_sort(self.combo_sort.currentText()))
        bar.addWidget(self.btn_open)
        bar.addWidget(self.btn_recent)
        bar.addWidget(self.chk_recursive)
        bar.addWidget(self.inp_filter, 1)
        bar.addWidget(self.combo_sort)
        bar.addWidget(self.btn_desc)
        ll.addLayout(bar)

        bar2 = QHBoxLayout()
        self.btn_undo = QPushButton("↶ Undo")
        self.btn_undo.setEnabled(False)
        self.btn_undo.clicked.connect(self._undo)
        self.btn_redo = QPushButton("↷ Redo")
        self.btn_redo.setEnabled(False)
        self.btn_redo.clicked.connect(self.undo_stack.redo)
        self.undo_stack.canUndoChanged.connect(self.btn_undo.setEnabled)
        self.undo_stack.canRedoChanged.connect(self.btn_redo.setEnabled)
        self.undo_stack.undoTextChanged.connect(lambda t: self.btn_undo.setToolTip(f"Undo {t} (Ctrl+Z)"))
        self.undo_stack.redoTextChanged.connect(lambda t: self.btn_redo.setToolTip(f"Redo {t} (Ctrl+Y)"))
        self.btn_save_all = QPushButton("💾 Save")
        self.btn_save_all.setStyleSheet("background-color: #0984e3; color: white; font-weight: bold;")
        self.btn_save_all.setToolTip("Save changed captions to disk (Ctrl+S). Previous versions are backed up.")
        self.btn_save_all.clicked.connect(self.save_all)
        self.btn_dataset = QPushButton("📦 Copy to Collection")
        self.btn_dataset.setToolTip("Copy the selected images and captions into a Dataset Collection")
        self.btn_dataset.clicked.connect(self.save_to_dataset)
        self.chk_captions = QCheckBox("Captions")
        self.chk_captions.setChecked(True)
        self.chk_captions.setToolTip("Show caption text on cards")
        self.chk_captions.toggled.connect(self._toggle_captions)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(120, 400)
        self.slider.setValue(self.cfg.get("ui.thumbnail_size"))
        self.slider.setFixedWidth(130)
        self.slider.setToolTip("Thumbnail size")
        self.slider.valueChanged.connect(self._set_card_size)
        bar2.addWidget(self.btn_undo)
        bar2.addWidget(self.btn_redo)
        bar2.addWidget(self.btn_save_all)
        bar2.addWidget(self.btn_dataset)
        bar2.addStretch()
        bar2.addWidget(self.chk_captions)
        bar2.addWidget(QLabel("Size"))
        bar2.addWidget(self.slider)
        ll.addLayout(bar2)

        self.view = QListView()
        self.view.setModel(self.proxy)
        self.view.setItemDelegate(self.delegate)
        self.view.setViewMode(QListView.IconMode)
        self.view.setResizeMode(QListView.Adjust)
        self.view.setMovement(QListView.Static)
        self.view.setUniformItemSizes(True)
        self.view.setSelectionMode(QListView.ExtendedSelection)
        self.view.setLayoutMode(QListView.Batched)
        self.view.setBatchSize(200)
        self.view.setMouseTracking(True)
        self.view.setSpacing(0)
        self.view.setVerticalScrollMode(QListView.ScrollPerPixel)
        self.view.verticalScrollBar().setSingleStep(40)
        self.view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._context_menu)
        self.view.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self.view.doubleClicked.connect(lambda _: self.inspector.editor.setFocus())
        self.view.setStyleSheet("QListView { background-color: #1f1f1f; border: none; }")
        self.view.setToolTip("")
        ll.addWidget(self.view, 1)
        self.lbl_status = QLabel("Open a folder to start.")
        self.lbl_status.setStyleSheet("color: #9a9a9a; padding: 2px;")
        ll.addWidget(self.lbl_status)

        # ---------------- right: inspector / batch / tags ----------------
        self.side = QTabWidget()
        self.side.setMinimumWidth(360)
        self.inspector = InspectorPanel()
        self.inspector.text_edited.connect(self._on_text_edited)
        self.inspector.tags_added.connect(lambda tags: self.run_operation(
            f"Add tags: {', '.join(tags)}", op_add_tags(tags), "selected"))
        self.inspector.tags_removed.connect(lambda tags: self.run_operation(
            f"Remove tags: {', '.join(tags)}", op_remove_tags(tags), "selected"))
        self.inspector.navigate.connect(self.navigate)
        self.batch = BatchPanel()
        self.batch.operation.connect(self.run_operation)
        self.batch.auto_tag.connect(self.run_auto_tagger)
        self.batch.quick_tag.connect(lambda tag, pos: self.run_operation(
            f"Add tag: {tag}", op_add_tags([tag], pos), "selected"))
        self.tags = TagStatsPanel()
        self.tags.filter_requested.connect(self._set_filter)
        self.tags.rename_requested.connect(lambda old, new: self.run_operation(
            f"Rename tag '{old}' → '{new}'", op_replace_tag(old, new), "all"))
        self.tags.delete_requested.connect(self._delete_tag_everywhere)
        self.tags.add_to_selection.connect(lambda t: self.run_operation(f"Add tag: {t}", op_add_tags([t]), "selected"))
        self.tags.select_with_tag.connect(lambda t: self._select_matching(f'tag:"{t}"'))
        self.tags.combo_scope.currentIndexChanged.connect(lambda _: self._refresh_tag_stats())
        self.side.addTab(self.inspector, "Inspect")
        self.side.addTab(self.batch, "Batch")
        self.side.addTab(self.tags, "Tags")
        self.side.setTabToolTip(0, "View and edit the selected image's caption")
        self.side.setTabToolTip(1, "Apply tag/text operations to many images at once")
        self.side.setTabToolTip(2, "Tag frequency statistics; rename, merge or delete tags everywhere")
        self.side.currentChanged.connect(lambda i: self._refresh_tag_stats() if i == 2 else None)

        splitter.addWidget(left)
        splitter.addWidget(self.side)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1100, 420])
        root.addWidget(splitter)

        self._stats_timer = QTimer(self, singleShot=True, interval=400)
        self._stats_timer.timeout.connect(self._refresh_tag_stats)
        self._set_card_size(self.slider.value())
        self.setup_hotkeys()

    # ------------------------------------------------------------------ hotkeys
    def setup_hotkeys(self):
        def sc(seq, fn):
            s = QShortcut(QKeySequence(seq), self)
            s.setContext(Qt.WidgetWithChildrenShortcut)
            s.activated.connect(fn)

        sc("Ctrl+S", self.save_all)
        sc("Ctrl+Z", self._undo)
        sc("Ctrl+Y", self.undo_stack.redo)
        sc("Ctrl+Shift+Z", self.undo_stack.redo)
        sc("Ctrl+F", lambda: (self.inp_filter.setFocus(), self.inp_filter.selectAll()))
        sc("Ctrl+O", self.select_folder)
        sc("Ctrl+Down", lambda: self.navigate(1))
        sc("Ctrl+Up", lambda: self.navigate(-1))
        sc("Ctrl+Shift+C", self.copy_caption)
        sc("Ctrl+Shift+V", self.paste_caption)
        sc("F2", lambda: self.inspector.editor.setFocus())
        # Grid-only keys (don't hijack text editing)
        for seq, fn in (("Del", self.delete_text_selection), ("Shift+Del", self.trash_selection)):
            s = QShortcut(QKeySequence(seq), self.view)
            s.setContext(Qt.WidgetShortcut)
            s.activated.connect(fn)

    def _undo(self):
        self.inspector._commit_typing()
        self.undo_stack.undo()

    # ------------------------------------------------------------------ folder
    def has_unsaved_changes(self) -> bool:
        self.inspector._commit_typing()
        return bool(self.session and self.session.dirty_entries)

    def select_folder(self):
        if not self._confirm_discard():
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", self.cfg.get("ui.last_folder"))
        if folder:
            self.open_folder(folder)

    def _confirm_discard(self) -> bool:
        if not self.has_unsaved_changes():
            return True
        n = len(self.session.dirty_entries)
        box = QMessageBox(QMessageBox.Warning, "Unsaved changes",
                          f"{n} caption(s) in this folder have unsaved changes.",
                          QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, self)
        box.setDefaultButton(QMessageBox.Save)
        r = box.exec()
        if r == QMessageBox.Save:
            return self.save_all()
        return r == QMessageBox.Discard

    def _fill_recent_menu(self):
        menu = self.btn_recent.menu()
        menu.clear()
        recent = [f for f in self.cfg.get("ui.recent_folders", []) if os.path.isdir(f)]
        if not recent:
            menu.addAction("(no recent folders)").setEnabled(False)
        for f in recent:
            menu.addAction(f, lambda f=f: self._confirm_discard() and self.open_folder(f))

    def load_grid(self):
        """Compatibility entry point: (re)open ``current_folder``."""
        if self.current_folder:
            self.open_folder(self.current_folder)

    def open_folder(self, folder: str):
        recursive = self.chk_recursive.isChecked()
        try:
            session = DatasetSession.open(folder, recursive=recursive, caption_ext=self.cfg.get("caption.extension"))
        except OSError as e:
            QMessageBox.critical(self, "Error", f"Failed to read folder:\n{e}")
            return
        self.generation += 1
        self.loader.reset()
        self.undo_stack.clear()
        self.session = session
        self.current_folder = folder
        self.model.set_session(session)
        self.inspector.show_entries([])
        self.cfg.update({"ui.last_folder": folder, "ui.recursive_scan": recursive,
                         "ui.recent_folders": list(dict.fromkeys([folder] + self.cfg.get("ui.recent_folders", [])))[:10]})
        self._apply_sort(self.combo_sort.currentText())
        self.apply_filter()
        self._info_pool.start(_InfoScan([e.key for e in session.entries], self.generation, self._info_signals, self))
        self._update_status()
        self._stats_timer.start()

    def showEvent(self, event):
        super().showEvent(event)
        self._pick_up_external_changes()

    def _pick_up_external_changes(self):
        """Captions written elsewhere (Auto Caption tab, other apps) appear here.
        Entries with unsaved edits are left alone so nothing typed is lost."""
        if not self.session or self.tag_worker is not None:
            return
        from core.caption_io import read_caption
        changed = []
        for e in self.session.entries:
            if e.dirty:
                continue
            disk = read_caption(e.path, self.session.caption_ext)
            if disk != e.saved_text:
                e.text = e.saved_text = disk
                changed.append(e.key)
        if changed:
            self._entries_changed(changed)
            log.info("Picked up %d externally changed caption(s)", len(changed))

    # ------------------------------------------------------------------ view helpers
    def _set_card_size(self, w: int):
        self.delegate.set_card_width(w)
        self.view.setGridSize(QSize(w, w + self.delegate.text_height()))
        self.view.doItemsLayout()
        self.cfg.set("ui.thumbnail_size", w, save=False)

    def _toggle_captions(self, on: bool):
        self.delegate.show_captions = on
        self._set_card_size(self.slider.value())

    def _apply_sort(self, name: str):
        self.proxy.set_sort_mode(name, self.btn_desc.isChecked())

    def _set_filter(self, text: str):
        self.inp_filter.setText(text)
        self.apply_filter()

    def apply_filter(self, *_):
        errors = self.proxy.set_query(self.inp_filter.text())
        self.inp_filter.setStyleSheet("QLineEdit { border: 1px solid #d63031; }" if errors else "")
        self.inp_filter.setToolTip(("\n".join(errors) + "\n\n" if errors else "") + QUERY_HELP)
        self._update_status()

    def _on_thumb_ready(self, key: str):
        self.model.refresh([key])

    def _on_dimensions(self, key: str, w: int, h: int):
        if self.session:
            e = self.session.get(key)
            if e is not None and not e.width:
                e.width, e.height, e.info_loaded = w, h, True

    def _on_info_batch(self, generation: int, items: list):
        if generation != self.generation or not self.session:
            return
        for key, w, h in items:
            e = self.session.get(key)
            if e:
                e.width, e.height, e.info_loaded = w, h, True
        self.model.refresh([k for k, _, _ in items])
        if self.proxy.query.needs_info or self.combo_sort.currentText() in ("Width", "Height", "Megapixels", "Aspect ratio"):
            self.proxy.invalidate()
            self._update_status()
        self.inspector.refresh_from_entries()

    # ------------------------------------------------------------------ selection
    def selected_keys(self) -> list[str]:
        rows = sorted(self.view.selectionModel().selectedIndexes(), key=lambda i: i.row())
        return [i.data(KEY_ROLE) for i in rows]

    def shown_keys(self) -> list[str]:
        return [self.proxy.index(r, 0).data(KEY_ROLE) for r in range(self.proxy.rowCount())]

    def _keys_for_scope(self, scope: str) -> list[str]:
        if not self.session:
            return []
        if scope == "all":
            return [e.key for e in self.session.entries]
        if scope == "shown":
            return self.shown_keys()
        return self.selected_keys()

    def _on_selection_changed(self, *_):
        keys = self.selected_keys()
        entries = [self.session.get(k) for k in keys] if self.session else []
        self.inspector.show_entries([e for e in entries if e])
        current = self.view.currentIndex()
        if current.isValid() and current.data(KEY_ROLE) in keys:
            self.image_selected.emit(current.data(KEY_ROLE))
        elif keys:
            self.image_selected.emit(keys[-1])
        self._update_status()
        if self.tags.scope() == "selected" and self.side.currentIndex() == 2:
            self._stats_timer.start()

    def select_all(self):
        self.view.selectAll()

    def _select_matching(self, query: str):
        from core.query import compile_query
        q = compile_query(query)
        sel = self.view.selectionModel()
        sel.clearSelection()
        for r in range(self.proxy.rowCount()):
            idx = self.proxy.index(r, 0)
            if q.predicate(idx.data(ENTRY_ROLE)):
                sel.select(idx, QItemSelectionModel.Select)

    def navigate(self, step: int):
        n = self.proxy.rowCount()
        if not n:
            return
        cur = self.view.currentIndex()
        row = (cur.row() + step) % n if cur.isValid() else 0
        idx = self.proxy.index(row, 0)
        self.view.setCurrentIndex(idx)
        self.view.selectionModel().select(idx, QItemSelectionModel.ClearAndSelect)
        self.view.scrollTo(idx)

    # ------------------------------------------------------------------ editing
    def _entries_changed(self, keys):
        self.model.refresh(keys)
        self.inspector.refresh_from_entries()
        self._update_status()
        self._stats_timer.start()

    def _push(self, changes: dict, description: str, merge_key: str | None = None) -> int:
        if not changes or not self.session:
            return 0
        self.undo_stack.push(ChangeCommand(self.session, changes, description, self._entries_changed, merge_key))
        return len(changes)

    def _on_text_edited(self, key: str, old: str, new: str):
        self._push({key: (old, new)}, f"Edit {Path(key).name}", merge_key=key)

    def run_operation(self, description: str, fn, scope: str = "selected") -> int:
        if not self.session:
            return 0
        self.inspector._commit_typing()
        keys = self._keys_for_scope(scope)
        if not keys:
            self._flash("Nothing to apply to — " + ("select some images first." if scope == "selected"
                                                     else "no images are shown."))
            return 0
        changes = self.session.transform(keys, fn)
        if description.startswith("Clear") and changes and not confirm(
                self, "Clear captions", f"Clear {len(changes)} caption(s)? (You can undo this.)", destructive=True):
            return 0
        n = self._push(changes, f"{description} ({len(changes)})")
        self._flash(f"{description}: {n} caption(s) changed — unsaved." if n else f"{description}: nothing changed.")
        return n

    def _delete_tag_everywhere(self, tag: str):
        if confirm(self, "Delete tag", f"Remove '{tag}' from every caption in this folder? (You can undo this.)"):
            self.run_operation(f"Delete tag '{tag}'", op_remove_tags([tag]), "all")

    def delete_text_selection(self):
        self.run_operation("Clear captions", lambda t: "", "selected")

    def sanitize_selection(self):
        self.run_operation("Convert to ASCII", captions.to_ascii, "selected")

    def copy_caption(self):
        keys = self.selected_keys()
        if keys and self.session:
            self._clipboard_caption = self.session.get(keys[0]).text
            QApplication.clipboard().setText(self._clipboard_caption)
            self._flash("Caption copied. Select images and press Ctrl+Shift+V to paste it onto them.")

    def paste_caption(self):
        text = self._clipboard_caption if self._clipboard_caption is not None else QApplication.clipboard().text()
        if text is not None:
            self.run_operation("Paste caption", lambda _t: text, "selected")

    # ------------------------------------------------------------------ save / files
    def save_all(self) -> bool:
        if not self.session:
            return True
        self.inspector._commit_typing()
        dirty = [e.key for e in self.session.dirty_entries]
        report = self.session.save(dirty)
        self.model.refresh(dirty)
        self.inspector.refresh_from_entries()
        self._update_status()
        if report.failed:
            QMessageBox.critical(self, "Some captions were not saved",
                                 "\n".join(f"{Path(k).name}: {e}" for k, e in report.failed[:20]))
            return False
        self._flash(f"Saved {report.saved} caption(s)." if report.saved else "No unsaved changes.")
        return True

    def save_to_dataset(self):
        keys = self.selected_keys()
        if not keys:
            QMessageBox.information(self, "No selection", "Select images first.")
            return
        root = Path(self.cfg.get("paths.collections_dir") or paths.DEFAULT_COLLECTIONS_DIR)
        existing = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
        name, ok = QInputDialog.getItem(self, "Copy to Collection", "Collection name (new or existing):",
                                        existing, 0, True)
        if not ok or not name.strip():
            return
        if any(self.session.get(k).dirty for k in keys) and not self.save_all():
            return  # copy what the user sees; abort if saving failed
        report = fileops.CopyReport()
        for k in keys:
            fileops.copy_image_with_caption(Path(k), root / name.strip(), report=report)
        msg = f"Collection '{name.strip()}': {report.summary()}."
        if report.failed:
            QMessageBox.warning(self, "Copy to Collection",
                                msg + "\n\n" + "\n".join(f"{p.name}: {e}" for p, e in report.failed[:20]))
        else:
            self._flash(msg)

    def trash_selection(self):
        keys = self.selected_keys()
        if not keys or not confirm(self, "Move to Recycle Bin",
                                   f"Move {len(keys)} image(s) and their captions to the Recycle Bin?",
                                   destructive=True):
            return
        report = fileops.trash_images_with_captions([Path(k) for k in keys])
        self.undo_stack.clear()  # undo history may reference removed images
        self.model.remove_keys([str(p) for p in report.trashed])
        self.inspector.show_entries([])
        self._update_status()
        if report.failed:
            QMessageBox.warning(self, "Some files were not moved",
                                "\n".join(f"{p.name}: {e}" for p, e in report.failed[:20]))

    # ------------------------------------------------------------------ context menu
    def _context_menu(self, pos):
        idx = self.view.indexAt(pos)
        keys = self.selected_keys()
        menu = QMenu(self)
        if idx.isValid():
            key = idx.data(KEY_ROLE)
            menu.addAction("Open image", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(key)))
            menu.addAction("Show in folder", lambda: self._reveal(key))
            menu.addSeparator()
        if keys:
            menu.addAction("Copy caption\tCtrl+Shift+C", self.copy_caption)
            paste = menu.addAction(f"Paste caption to {len(keys)} image(s)\tCtrl+Shift+V", self.paste_caption)
            paste.setEnabled(self._clipboard_caption is not None or bool(QApplication.clipboard().text()))
            menu.addAction("Clear caption(s)\tDel", self.delete_text_selection)
            menu.addSeparator()
            menu.addAction("Copy to collection…", self.save_to_dataset)
            menu.addAction(f"Move {len(keys)} to Recycle Bin…\tShift+Del", self.trash_selection)
            menu.addSeparator()
        menu.addAction("Select all\tCtrl+A", self.select_all)
        menu.addAction("Select images missing captions", lambda: self._select_matching("missing:caption"))
        menu.addAction("Select unsaved", lambda: self._select_matching("is:unsaved"))
        menu.exec(self.view.viewport().mapToGlobal(pos))

    @staticmethod
    def _reveal(path: str):
        if os.name == "nt":
            import subprocess
            subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(path)))

    # ------------------------------------------------------------------ status & stats
    def _update_status(self):
        if not self.session:
            self.lbl_status.setText("Open a folder to start.")
            return
        s = self.session.stats()
        shown, selected = self.proxy.rowCount(), len(self.view.selectionModel().selectedIndexes())
        parts = [f"{shown} of {s['images']} shown" if shown != s["images"] else f"{s['images']} images",
                 f"{selected} selected", f"{s['missing']} missing captions"]
        if s["unsaved"]:
            parts.append(f"● {s['unsaved']} unsaved")
        self.lbl_status.setText("   ·   ".join(parts) + f"      {self.current_folder}")
        self.btn_save_all.setText(f"💾 Save ({s['unsaved']})" if s["unsaved"] else "💾 Save")
        self.batch.set_counts(selected, shown)

    def _refresh_tag_stats(self):
        if not self.session or self.side.currentIndex() != 2:
            return
        keys = self._keys_for_scope(self.tags.scope())
        self.tags.set_counts(self.session.tag_counts(keys), len(keys))

    def _flash(self, msg: str):
        self.lbl_status.setText(msg)
        QTimer.singleShot(4000, self._update_status)

    # ------------------------------------------------------------------ auto tagger
    def run_auto_tagger(self, scope: str = "selected"):
        if self.tag_worker is not None:
            self.tag_worker.cancel()
            self.batch.btn_auto.setText("Stopping…")
            return
        keys = self._keys_for_scope(scope)
        if not keys:
            QMessageBox.information(self, "Auto Tag", "Select images to tag first.")
            return
        dlg = AutoTagDialog(self)
        if not dlg.exec():
            return
        st = self.auto_tag_settings = dlg.get_settings()
        if st["mode"] == "ignore":
            keys = [k for k in keys if not self.session.get(k).has_caption]
            if not keys:
                QMessageBox.information(self, "Nothing to tag", "All chosen images already have captions.")
                return
        fmt = dict(general_threshold=st["threshold"], character_threshold=st["character_threshold"],
                   max_tags=st["max_tags"], underscores_to_spaces=st["underscores_to_spaces"],
                   escape_parentheses=st["escape_parentheses"], include_rating=st["include_rating"],
                   blacklist=tuple(st["blacklist"]))
        spec = ProviderSpec.make("wd_tagger", st["model"], format=tuple(sorted(fmt.items())),
                                 device=self.cfg.get("tagger.device"))
        job = BatchJob(paths=[Path(k) for k in keys], spec=spec, request=CaptionRequest(prompt="", max_image_side=0),
                       save_mode=SAVE_NONE, tag_mode=True, title="Auto tagging")
        self.pending_tag_updates = {}
        self._tag_generation = self.generation
        self.batch.btn_auto.setText(f"⏳ Tagging 0/{len(keys)}…  (click to stop)")
        self.tag_worker = start_job(job)
        self.tag_worker.item_done.connect(self.on_tagger_finished)
        self.tag_worker.progress.connect(lambda d, t: self.batch.btn_auto.setText(f"⏳ Tagging {d}/{t}…  (click to stop)"))
        self.tag_worker.finished.connect(self.finalize_auto_tagging)

    def on_tagger_finished(self, path: str, tag_text: str):
        if not self.session or self.generation != self._tag_generation:
            return  # folder changed while tagging
        e = self.session.get(path)
        if e is None:
            return
        st = self.auto_tag_settings
        new = captions.merge_generated_tags(e.text, captions.split_tags(tag_text), st["mode"],
                                            prepend=st["prepend"], append=st["append"])
        self.pending_tag_updates.setdefault(path, e.text)
        e.text = new
        self.model.refresh([path])

    def finalize_auto_tagging(self, summary=None):
        self.tag_worker = None
        self.batch.btn_auto.setText("✨ Auto Tag (WD tagger)…")
        if self.session and self.pending_tag_updates and self.generation == self._tag_generation:
            changes = {k: (old, self.session.get(k).text) for k, old in self.pending_tag_updates.items()
                       if self.session.get(k) is not None}
            # Entries already hold the new text; the command records it for undo/redo.
            self._push(changes, f"Auto tag ({len(changes)})")
        self.pending_tag_updates = {}
        if summary is not None:
            show_job_summary(self, summary)
            if not summary.fatal:
                self._flash(summary.text() + " — unsaved; press Save (Ctrl+S) to keep.")
