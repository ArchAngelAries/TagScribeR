"""Dataset Collections: gather finished images + captions into training folders.

Left: the shared workspace browser (whatever folder is open). Right: your
collections with image/caption counts. Double-click a collection to open it in
every tab. Copies never overwrite; deletes go to the Recycle Bin.
"""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (QHBoxLayout, QInputDialog, QLabel, QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QSplitter, QVBoxLayout, QWidget)

from core import caption_io, dataset, fileops, paths
from core.config import settings
from tabs.common import confirm, hint_label, run_in_background
from tabs.workspace.browser import DatasetBrowser
from tabs.workspace.context import workspace

INVALID_CHARS = '<>:"/\\|?*'


def collections_root() -> Path:
    return Path(settings().get("paths.collections_dir") or paths.DEFAULT_COLLECTIONS_DIR)


def _collection_stats(root: Path) -> list[tuple[str, int, int]]:
    out = []
    if not root.is_dir():
        return out
    for d in sorted((p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")),
                    key=lambda p: dataset.natural_key(p.name)):
        try:
            imgs = dataset.scan_images(d)
        except OSError:
            continue
        out.append((d.name, len(imgs), sum(1 for i in imgs if caption_io.has_caption(i))))
    return out


class DatasetsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.ctx = workspace()
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(6, 6, 6, 6)
        splitter = QSplitter(Qt.Horizontal)

        self.browser = DatasetBrowser(self.ctx, size_key="ui.datasets_thumbnail_size")
        self.browser.selection_changed.connect(lambda _k: self.update_buttons())
        splitter.addWidget(self.browser)

        right = QWidget()
        right.setMinimumWidth(300)
        rl = QVBoxLayout(right)
        title = QHBoxLayout()
        title.addWidget(QLabel("<b>Dataset Collections</b>"), 1)
        from tabs.help import help_button
        title.addWidget(help_button("datasets", right, "How collections work"))
        rl.addLayout(title)
        self.lbl_root = hint_label("")
        rl.addWidget(self.lbl_root)
        self.list_datasets = QListWidget()
        self.list_datasets.setToolTip("Double-click to open a collection in every tab")
        self.list_datasets.itemSelectionChanged.connect(self.update_buttons)
        self.list_datasets.itemDoubleClicked.connect(self.open_collection)
        rl.addWidget(self.list_datasets, 1)

        row = QHBoxLayout()
        self.btn_new = QPushButton("New")
        self.btn_new.setToolTip("Create a collection (Ctrl+N)")
        self.btn_new.clicked.connect(self.create_collection)
        self.btn_rename = QPushButton("Rename")
        self.btn_rename.clicked.connect(self.rename_collection)
        self.btn_del_col = QPushButton("🗑️ Delete")
        self.btn_del_col.setToolTip("Move the collection folder to the Recycle Bin")
        self.btn_del_col.clicked.connect(self.delete_collection)
        self.btn_refresh = QPushButton("🔄")
        self.btn_refresh.setFixedWidth(34)
        self.btn_refresh.setToolTip("Refresh (F5)")
        self.btn_refresh.clicked.connect(self.refresh_collections)
        for b in (self.btn_new, self.btn_rename, self.btn_del_col, self.btn_refresh):
            row.addWidget(b)
        rl.addLayout(row)
        row2 = QHBoxLayout()
        self.btn_open = QPushButton("📂 Open in workspace")
        self.btn_open.clicked.connect(lambda: self.open_collection(self.list_datasets.currentItem()))
        self.btn_explore = QPushButton("Show in Explorer")
        self.btn_explore.clicked.connect(self._explore)
        row2.addWidget(self.btn_open)
        row2.addWidget(self.btn_explore)
        rl.addLayout(row2)

        self.btn_add = QPushButton("➕ Add Selected to Collection")
        self.btn_add.setToolTip("Copy the selected images and their captions into the chosen collection (Ctrl+Enter)")
        self.btn_add.setStyleSheet("background-color: #00b894; color: white; font-weight: bold; padding: 8px;")
        self.btn_add.clicked.connect(self.add_to_collection)
        rl.addWidget(self.btn_add)
        self.btn_export = QPushButton("🚀 Export for Training…")
        self.btn_export.setToolTip("Training-ready copy of the selected images (or all shown, if none selected): "
                                   "bucket resize, metadata stripped, captions with trigger word, kohya folder layout")
        self.btn_export.setStyleSheet("background-color: #6c5ce7; color: white; font-weight: bold; padding: 8px;")
        self.btn_export.clicked.connect(self.export_for_training)
        rl.addWidget(self.btn_export)
        rl.addWidget(hint_label("Images are copied with their captions. Name clashes are renamed — nothing in a "
                                "collection is ever overwritten. Deleting moves files to the Recycle Bin."))
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1150, 360])
        root_layout.addWidget(splitter)

        for seq, fn in (("Ctrl+N", self.create_collection), ("F5", self.refresh_collections),
                        ("Ctrl+Return", self.add_to_collection), ("Ctrl+Enter", self.add_to_collection)):
            s = QShortcut(QKeySequence(seq), self)
            s.setContext(Qt.WidgetWithChildrenShortcut)
            s.activated.connect(fn)
        collections_root().mkdir(parents=True, exist_ok=True)
        self.refresh_collections()
        self.update_buttons()

    # ------------------------------------------------------------------ list
    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_collections()

    def refresh_collections(self):
        root = collections_root()
        self.lbl_root.setText(str(root))
        current = self._current_name()

        def done(stats):
            self.list_datasets.clear()
            for name, n, captioned in stats:
                item = QListWidgetItem(f"{name}\n    {n} images · {captioned} captioned"
                                       + (f" · {n - captioned} missing" if n - captioned else ""))
                item.setData(Qt.UserRole, name)
                self.list_datasets.addItem(item)
                if name == current:
                    self.list_datasets.setCurrentItem(item)
            self.update_buttons()

        run_in_background(lambda: _collection_stats(root), done)

    def _current_name(self) -> str:
        item = self.list_datasets.currentItem()
        return item.data(Qt.UserRole) if item else ""

    def update_buttons(self):
        has_col = bool(self._current_name())
        n = len(self.browser.selected_keys())
        for b in (self.btn_del_col, self.btn_rename, self.btn_open, self.btn_explore):
            b.setEnabled(has_col)
        self.btn_add.setEnabled(has_col and n > 0)
        self.btn_add.setText(f"➕ Add {n} to '{self._current_name()}'" if has_col and n else
                             "➕ Add Selected to Collection")

    # ------------------------------------------------------------------ actions
    def _valid_name(self, name: str) -> bool:
        if any(c in name for c in INVALID_CHARS):
            QMessageBox.warning(self, "Invalid name", f"Collection names can't contain {INVALID_CHARS}")
            return False
        return True

    def create_collection(self):
        name, ok = QInputDialog.getText(self, "New Collection", "Name:")
        name = name.strip()
        if not ok or not name or not self._valid_name(name):
            return
        path = collections_root() / name
        if path.exists():
            QMessageBox.information(self, "Exists", f"A collection named '{name}' already exists.")
            return
        path.mkdir(parents=True)
        self.refresh_collections()

    def rename_collection(self):
        old = self._current_name()
        new, ok = QInputDialog.getText(self, "Rename collection", "New name:", text=old)
        new = new.strip()
        if not ok or not new or new == old or not self._valid_name(new):
            return
        src, dest = collections_root() / old, collections_root() / new
        if dest.exists():
            QMessageBox.information(self, "Exists", f"A collection named '{new}' already exists.")
            return
        if self.ctx.folder and os.path.normcase(os.path.abspath(self.ctx.folder)) == os.path.normcase(str(src)):
            QMessageBox.information(self, "Collection is open", "Open a different folder before renaming the "
                                    "collection that's currently open.")
            return
        try:
            src.rename(dest)
        except OSError as e:
            QMessageBox.critical(self, "Rename failed", str(e))
            return
        self.refresh_collections()

    def delete_collection(self):
        name = self._current_name()
        path = collections_root() / name
        if not name or not confirm(self, "Delete Collection",
                                   f"Move the collection '{name}' and everything in it to the Recycle Bin?",
                                   destructive=True):
            return
        if self.ctx.folder and os.path.normcase(os.path.abspath(self.ctx.folder)) == os.path.normcase(str(path)):
            QMessageBox.information(self, "Collection is open", "Open a different folder before deleting the "
                                    "collection that's currently open.")
            return
        try:
            fileops.trash(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Couldn't move it to the Recycle Bin:\n{e}\n\nNothing was deleted.")
            return
        self.refresh_collections()

    def open_collection(self, item=None):
        name = item.data(Qt.UserRole) if item else self._current_name()
        if name and self.ctx.confirm_discard(self):
            self.ctx.open_folder(str(collections_root() / name), False, self)

    def _explore(self):
        name = self._current_name()
        if name:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(collections_root() / name)))

    def export_for_training(self):
        keys = self.browser.selected_keys() or self.browser.shown_keys()
        if not keys:
            QMessageBox.information(self, "Export for training", "Open a folder (or a collection) first.")
            return
        s = self.ctx.session
        if s and any(s.get(k) and s.get(k).dirty for k in keys):
            if not confirm(self, "Unsaved captions", "Some captions have unsaved edits. Save them before exporting?"):
                return
            if not self.ctx.save(self, keys):
                return
        from tabs.export_dialog import ExportDialog
        subject = self.ctx.project.get("subject", "") if self.ctx.project else ""
        ExportDialog(keys, subject, self).exec()

    def add_to_collection(self):
        name = self._current_name()
        keys = self.browser.selected_keys()
        if not name or not keys:
            return
        dest = collections_root() / name
        if self.ctx.folder and os.path.normcase(os.path.abspath(self.ctx.folder)) == os.path.normcase(str(dest)):
            QMessageBox.information(self, "Same folder", "These images are already in that collection.")
            return
        s = self.ctx.session
        if s and any(s.get(k) and s.get(k).dirty for k in keys) and not self.ctx.save(self, keys):
            return  # copy what the user sees
        report = fileops.CopyReport()
        for k in keys:
            fileops.copy_image_with_caption(Path(k), dest, report=report)
        msg = f"'{name}': {report.summary()}."
        if report.failed:
            QMessageBox.warning(self, "Add to Collection",
                                msg + "\n\n" + "\n".join(f"{p.name}: {e}" for p, e in report.failed[:20]))
        else:
            self.browser.flash(msg)
        self.refresh_collections()
