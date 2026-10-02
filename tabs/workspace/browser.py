"""DatasetBrowser: toolbar + virtualized grid + status line over the shared workspace.

Embedded by Gallery and Auto Caption (and later other tools). Each browser has
its own selection, filter, sort and zoom; data, undo and saving are shared via
the WorkspaceContext.
"""
from __future__ import annotations

import os

from PySide6.QtCore import QEvent, QItemSelectionModel, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                               QListView, QMenu, QPushButton, QSlider, QToolButton, QVBoxLayout, QWidget)

from core.query import compile_query
from tabs.common import confirm
from tabs.workspace.context import WorkspaceContext
from tabs.workspace.delegate import CardDelegate
from tabs.workspace.model import ENTRY_ROLE, KEY_ROLE, SORT_MODES, FilterProxy

MIN_CARD, MAX_CARD, DEFAULT_CARD, ZOOM_STEP = 100, 720, 220, 40

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
    "  flag:blurry            Health scan results (duplicate, similar, blurry, lowres, crop, any)\n"
    "Combine terms with spaces; prefix any term with - to negate it."
)

_clipboard: dict[str, str | None] = {"caption": None}


class DatasetBrowser(QWidget):
    selection_changed = Signal(list)     # selected keys (grid order)
    current_changed = Signal(str)        # key of the focused image
    menu_about_to_show = Signal(QMenu, list)  # lets the host tab add context actions

    def __init__(self, ctx: WorkspaceContext, size_key: str = "ui.thumbnail_size", parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.size_key = size_key

        self.proxy = FilterProxy(self)
        self.proxy.setSourceModel(ctx.model)
        self.delegate = CardDelegate(ctx.loader, self, job_state=ctx.job_state)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        bar = QHBoxLayout()
        self.btn_open = QPushButton("📂 Open Folder")
        self.btn_open.setToolTip("Open a dataset folder (Ctrl+O). It opens in every tab.")
        self.btn_open.setStyleSheet("background-color: #00b894; color: white; font-weight: bold;")
        self.btn_open.clicked.connect(self.select_folder)
        self.btn_recent = QToolButton(text="▾")
        self.btn_recent.setToolTip("Recent folders")
        self.btn_recent.setPopupMode(QToolButton.InstantPopup)
        self.btn_recent.setMenu(QMenu(self.btn_recent))
        self.btn_recent.menu().aboutToShow.connect(self._fill_recent_menu)
        self.chk_recursive = QCheckBox("Subfolders")
        self.chk_recursive.setToolTip("Also load images from subfolders (applies when a folder is opened)")
        self.chk_recursive.setChecked(ctx.cfg.get("ui.recursive_scan", False))
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
        for w in (self.btn_open, self.btn_recent, self.chk_recursive):
            bar.addWidget(w)
        bar.addWidget(self.inp_filter, 1)
        from tabs.help import help_button
        bar.addWidget(help_button("filter", self, "Filter syntax and examples"))
        bar.addWidget(self.combo_sort)
        bar.addWidget(self.btn_desc)
        lay.addLayout(bar)

        bar2 = QHBoxLayout()
        self.btn_undo = QPushButton("↶ Undo")
        self.btn_undo.setEnabled(ctx.undo_stack.canUndo())
        self.btn_undo.clicked.connect(ctx.undo)
        self.btn_redo = QPushButton("↷ Redo")
        self.btn_redo.setEnabled(ctx.undo_stack.canRedo())
        self.btn_redo.clicked.connect(ctx.redo)
        ctx.undo_stack.canUndoChanged.connect(self.btn_undo.setEnabled)
        ctx.undo_stack.canRedoChanged.connect(self.btn_redo.setEnabled)
        ctx.undo_stack.undoTextChanged.connect(lambda t: self.btn_undo.setToolTip(f"Undo {t} (Ctrl+Z)"))
        ctx.undo_stack.redoTextChanged.connect(lambda t: self.btn_redo.setToolTip(f"Redo {t} (Ctrl+Y)"))
        self.btn_save = QPushButton("💾 Save")
        self.btn_save.setStyleSheet("background-color: #0984e3; color: white; font-weight: bold;")
        self.btn_save.setToolTip("Save changed captions to disk (Ctrl+S). Previous versions are backed up.")
        self.btn_save.clicked.connect(self.save)
        self.btn_collection = QPushButton("📦 Copy to Collection")
        self.btn_collection.setToolTip("Copy the selected images and captions into a Dataset Collection")
        self.btn_collection.clicked.connect(lambda: ctx.copy_to_collection(self, self.selected_keys()))
        for w in (self.btn_undo, self.btn_redo, self.btn_save, self.btn_collection):
            bar2.addWidget(w)
        self.extra_toolbar = QHBoxLayout()  # host tabs add their own quick actions here
        bar2.addLayout(self.extra_toolbar)
        bar2.addStretch()
        self.chk_captions = QCheckBox("Captions")
        self.chk_captions.setChecked(True)
        self.chk_captions.setToolTip("Show caption text on cards")
        self.chk_captions.toggled.connect(self._toggle_captions)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(MIN_CARD, MAX_CARD)
        self.slider.setValue(max(MIN_CARD, min(MAX_CARD, ctx.cfg.get(size_key, DEFAULT_CARD))))
        self.slider.setFixedWidth(150)
        self.slider.setToolTip("Thumbnail size — also Ctrl + mouse wheel over the grid, Ctrl+= / Ctrl+- / Ctrl+0")
        self.slider.valueChanged.connect(self._set_card_size)
        bar2.addWidget(self.chk_captions)
        bar2.addWidget(QLabel("Size"))
        bar2.addWidget(self.slider)
        lay.addLayout(bar2)

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
        self.view.selectionModel().selectionChanged.connect(self._on_selection)
        self.view.selectionModel().currentChanged.connect(
            lambda cur, _prev: cur.isValid() and self.current_changed.emit(cur.data(KEY_ROLE)))
        self.view.setStyleSheet("QListView { background-color: #1f1f1f; border: none; }")
        self.view.viewport().installEventFilter(self)
        lay.addWidget(self.view, 1)
        self.lbl_status = QLabel("Open a folder to start.")
        self.lbl_status.setStyleSheet("color: #9a9a9a; padding: 2px;")
        lay.addWidget(self.lbl_status)

        self._size_save_timer = QTimer(self, singleShot=True, interval=600)
        self._size_save_timer.timeout.connect(lambda: ctx.cfg.set(self.size_key, self.slider.value()))
        self._set_card_size(self.slider.value())

        ctx.session_changed.connect(self._on_session_changed)
        ctx.dirty_changed.connect(self.update_status)
        ctx.dimensions_loaded.connect(self._on_dimensions)
        self._setup_hotkeys()
        if ctx.session is not None:
            self._on_session_changed()

    # ------------------------------------------------------------------ hotkeys
    def _setup_hotkeys(self):
        def sc(seq, fn, widget=None, ctx=Qt.WidgetWithChildrenShortcut):
            s = QShortcut(QKeySequence(seq), widget or self)
            s.setContext(ctx)
            s.activated.connect(fn)

        sc("Ctrl+S", self.save)
        sc("Ctrl+Z", self.ctx.undo)
        sc("Ctrl+Y", self.ctx.redo)
        sc("Ctrl+Shift+Z", self.ctx.redo)
        sc("Ctrl+F", lambda: (self.inp_filter.setFocus(), self.inp_filter.selectAll()))
        sc("Ctrl+O", self.select_folder)
        sc("Ctrl+=", lambda: self.zoom(1))
        sc("Ctrl++", lambda: self.zoom(1))
        sc("Ctrl+-", lambda: self.zoom(-1))
        sc("Ctrl+0", lambda: self.slider.setValue(DEFAULT_CARD))
        sc("Ctrl+Shift+C", self.copy_caption)
        sc("Ctrl+Shift+V", self.paste_caption)
        sc("Del", lambda: self.run_operation("Clear captions", lambda t: ""), self.view, Qt.WidgetShortcut)
        sc("Shift+Del", lambda: self.ctx.trash(self, self.selected_keys()), self.view, Qt.WidgetShortcut)

    # ------------------------------------------------------------------ folder
    def select_folder(self):
        if not self.ctx.confirm_discard(self):
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", self.ctx.cfg.get("ui.last_folder"))
        if folder:
            self.ctx.open_folder(folder, self.chk_recursive.isChecked(), self)

    def _fill_recent_menu(self):
        menu = self.btn_recent.menu()
        menu.clear()
        recent = self.ctx.recent_folders()
        if not recent:
            menu.addAction("(no recent folders)").setEnabled(False)
        for f in recent:
            menu.addAction(f, lambda f=f: self.ctx.confirm_discard(self)
                           and self.ctx.open_folder(f, self.chk_recursive.isChecked(), self))

    def _on_session_changed(self):
        self.chk_recursive.setChecked(self.ctx.cfg.get("ui.recursive_scan", False))
        self._apply_sort(self.combo_sort.currentText())
        self.apply_filter()
        self.selection_changed.emit([])

    def _on_dimensions(self, _keys):
        if self.proxy.query.needs_info or "flag:" in self.proxy.query_text or self.combo_sort.currentText() in ("Width", "Height", "Megapixels",
                                                                             "Aspect ratio"):
            self.proxy.invalidate()
            self.update_status()

    # ------------------------------------------------------------------ view
    def _set_card_size(self, w: int):
        self.delegate.set_card_width(w)
        self.view.setGridSize(QSize(w, w + self.delegate.text_height()))
        self.view.doItemsLayout()
        self._size_save_timer.start()

    def zoom(self, direction: int):
        self.slider.setValue(self.slider.value() + direction * ZOOM_STEP)

    def eventFilter(self, obj, event):
        if obj is self.view.viewport() and event.type() == QEvent.Wheel and event.modifiers() & Qt.ControlModifier:
            self.zoom(1 if event.angleDelta().y() > 0 else -1)
            return True
        return super().eventFilter(obj, event)

    def _toggle_captions(self, on: bool):
        self.delegate.show_captions = on
        self._set_card_size(self.slider.value())

    def _apply_sort(self, name: str):
        self.proxy.set_sort_mode(name, self.btn_desc.isChecked())

    def set_filter(self, text: str):
        self.inp_filter.setText(text)
        self.apply_filter()

    def apply_filter(self, *_):
        errors = self.proxy.set_query(self.inp_filter.text())
        self.inp_filter.setStyleSheet("QLineEdit { border: 1px solid #d63031; }" if errors else "")
        self.inp_filter.setToolTip(("\n".join(errors) + "\n\n" if errors else "") + QUERY_HELP)
        self.update_status()

    # ------------------------------------------------------------------ selection
    def selected_keys(self) -> list[str]:
        rows = sorted(self.view.selectionModel().selectedIndexes(), key=lambda i: i.row())
        return [i.data(KEY_ROLE) for i in rows]

    def shown_keys(self) -> list[str]:
        return [self.proxy.index(r, 0).data(KEY_ROLE) for r in range(self.proxy.rowCount())]

    def keys_for_scope(self, scope: str) -> list[str]:
        if not self.ctx.session:
            return []
        if scope == "all":
            return [e.key for e in self.ctx.session.entries]
        return self.shown_keys() if scope == "shown" else self.selected_keys()

    def _on_selection(self, *_):
        self.selection_changed.emit(self.selected_keys())
        self.update_status()

    def select_all(self):
        self.view.selectAll()

    def select_matching(self, query: str):
        q = compile_query(query)
        sel = self.view.selectionModel()
        sel.clearSelection()
        for r in range(self.proxy.rowCount()):
            idx = self.proxy.index(r, 0)
            if q.predicate(idx.data(ENTRY_ROLE)):
                sel.select(idx, QItemSelectionModel.Select)

    def select_keys(self, keys) -> int:
        """Select exactly these images (those currently shown); returns how many were selected."""
        wanted = set(keys)
        sel = self.view.selectionModel()
        sel.clearSelection()
        first = None
        for r in range(self.proxy.rowCount()):
            idx = self.proxy.index(r, 0)
            if idx.data(KEY_ROLE) in wanted:
                sel.select(idx, QItemSelectionModel.Select)
                first = first or idx
        if first is not None:
            self.view.scrollTo(first)
        return len(self.selected_keys())

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

    # ------------------------------------------------------------------ operations
    def run_operation(self, description: str, fn, scope: str = "selected") -> int:
        keys = self.keys_for_scope(scope)
        if not keys:
            self.flash("Nothing to apply to — " + ("select some images first." if scope == "selected"
                                                    else "no images are shown."))
            return 0
        changes = self.ctx.apply_transform(keys, fn, description)
        if description.startswith("Clear") and changes and not confirm(
                self, "Clear captions", f"Clear {len(changes)} caption(s)? (You can undo this.)", destructive=True):
            return 0
        n = self.ctx.push(changes, f"{description} ({len(changes)})")
        self.flash(f"{description}: {n} caption(s) changed — unsaved." if n else f"{description}: nothing changed.")
        return n

    def save(self) -> bool:
        ok = self.ctx.save(self)
        if ok:
            n = getattr(self.ctx, "last_save_count", 0)
            self.flash(f"Saved {n} caption(s)." if n else "No unsaved changes.")
        return ok

    def copy_caption(self):
        keys = self.selected_keys()
        if keys and self.ctx.session:
            _clipboard["caption"] = self.ctx.session.get(keys[0]).text
            QApplication.clipboard().setText(_clipboard["caption"])
            self.flash("Caption copied. Select images and press Ctrl+Shift+V to paste it onto them.")

    def paste_caption(self):
        text = _clipboard["caption"] if _clipboard["caption"] is not None else QApplication.clipboard().text()
        if text is not None:
            self.run_operation("Paste caption", lambda _t: text)

    # ------------------------------------------------------------------ context menu
    def _context_menu(self, pos):
        idx = self.view.indexAt(pos)
        keys = self.selected_keys()
        menu = QMenu(self)
        if idx.isValid():
            key = idx.data(KEY_ROLE)
            menu.addAction("Open image", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(key)))
            menu.addAction("Show in folder", lambda: _reveal(key))
            menu.addSeparator()
        self.menu_about_to_show.emit(menu, keys)
        if keys:
            menu.addAction("Copy caption\tCtrl+Shift+C", self.copy_caption)
            paste = menu.addAction(f"Paste caption to {len(keys)} image(s)\tCtrl+Shift+V", self.paste_caption)
            paste.setEnabled(_clipboard["caption"] is not None or bool(QApplication.clipboard().text()))
            menu.addAction("Clear caption(s)\tDel", lambda: self.run_operation("Clear captions", lambda t: ""))
            menu.addSeparator()
            menu.addAction("Copy to collection…", lambda: self.ctx.copy_to_collection(self, keys))
            menu.addAction(f"Move {len(keys)} to Recycle Bin…\tShift+Del", lambda: self.ctx.trash(self, keys))
            menu.addSeparator()
        menu.addAction("Select all\tCtrl+A", self.select_all)
        menu.addAction("Select images missing captions", lambda: self.select_matching("missing:caption"))
        menu.addAction("Select unsaved", lambda: self.select_matching("is:unsaved"))
        menu.exec(self.view.viewport().mapToGlobal(pos))

    # ------------------------------------------------------------------ status
    def update_status(self):
        s = self.ctx.session
        if not s:
            self.lbl_status.setText("Open a folder to start.")
            self.btn_save.setText("💾 Save")
            return
        st = s.stats()
        shown, selected = self.proxy.rowCount(), len(self.view.selectionModel().selectedIndexes())
        parts = [f"{shown} of {st['images']} shown" if shown != st["images"] else f"{st['images']} images",
                 f"{selected} selected", f"{st['missing']} missing captions"]
        if st["unsaved"]:
            parts.append(f"● {st['unsaved']} unsaved")
        self.lbl_status.setText("   ·   ".join(parts) + f"      {self.ctx.folder}")
        self.btn_save.setText(f"💾 Save ({st['unsaved']})" if st["unsaved"] else "💾 Save")

    def flash(self, msg: str):
        self.lbl_status.setText(msg)
        QTimer.singleShot(4000, self.update_status)


def _reveal(path: str):
    if os.name == "nt":
        import subprocess
        subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(path)))
