"""Right-hand panels of the dataset workspace."""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QListWidget, QMenu, QPlainTextEdit, QPushButton,
                               QRadioButton, QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from core import captions
from core.config import load_quick_tags, save_quick_tags
from core.dataset_session import Entry
from core.image_utils import load_qimage
from core.widgets import TagEditorWidget
from tabs.common import CollapsibleSection, hint_label, run_in_background


def _scroll(widget: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    sa.setWidget(widget)
    return sa


# --------------------------------------------------------------------------- inspector
class InspectorPanel(QWidget):
    """Single image: preview + raw caption + tag bubbles. Several: shared tags only."""

    text_edited = Signal(str, str, str)       # key, old, new (debounced while typing)
    tags_added = Signal(list)                 # multi-selection: add to all selected
    tags_removed = Signal(list)               # multi-selection: remove from all selected
    navigate = Signal(int)                    # -1 previous / +1 next image

    def __init__(self, parent=None):
        super().__init__(parent)
        self.entries: list[Entry] = []
        self._common: list[str] = []
        self._loading_key = ""
        self._pending_old: str | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.preview = QLabel("Select an image")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(260)
        self.preview.setStyleSheet("background-color: #1a1a1a; border-radius: 6px; color: #777;")
        lay.addWidget(self.preview, 1)

        nav = QHBoxLayout()
        self.btn_prev = QPushButton("◀")
        self.btn_prev.setToolTip("Previous image (Ctrl+Up)")
        self.btn_prev.setFixedWidth(36)
        self.btn_prev.clicked.connect(lambda: self.navigate.emit(-1))
        self.lbl_info = QLabel("")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("color: #9a9a9a; font-size: 11px;")
        self.btn_next = QPushButton("▶")
        self.btn_next.setToolTip("Next image (Ctrl+Down)")
        self.btn_next.setFixedWidth(36)
        self.btn_next.clicked.connect(lambda: self.navigate.emit(1))
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.lbl_info, 1)
        nav.addWidget(self.btn_next)
        lay.addLayout(nav)

        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText("Caption…")
        self.editor.setMinimumHeight(110)
        self.editor.textChanged.connect(self._on_typing)
        lay.addWidget(self.editor)
        self.lbl_counts = QLabel("")
        self.lbl_counts.setStyleSheet("color: #777; font-size: 10px;")
        self.lbl_counts.setAlignment(Qt.AlignRight)
        lay.addWidget(self.lbl_counts)

        self.lbl_tags_title = QLabel("<b>Tags</b>")
        lay.addWidget(self.lbl_tags_title)
        self.tag_editor = TagEditorWidget()
        self.tag_editor.tagsChanged.connect(self._on_bubbles_changed)
        lay.addWidget(self.tag_editor)

        self._typing_timer = QTimer(self)
        self._typing_timer.setSingleShot(True)
        self._typing_timer.setInterval(350)
        self._typing_timer.timeout.connect(self._commit_typing)
        self.show_entries([])

    # -- display ------------------------------------------------------------
    def show_entries(self, entries: list[Entry]) -> None:
        self._commit_typing()
        self.entries = entries
        single = len(entries) == 1
        self.editor.setVisible(single)
        self.lbl_counts.setVisible(single)
        self.btn_prev.setEnabled(bool(entries))
        self.btn_next.setEnabled(bool(entries))
        if not entries:
            self.preview.setPixmap(QPixmap())
            self.preview.setText("Select an image")
            self.lbl_info.setText("")
            self.lbl_tags_title.setText("<b>Tags</b>")
            self.tag_editor.set_tags("")
            self.tag_editor.setEnabled(False)
            return
        self.tag_editor.setEnabled(True)
        if single:
            e = entries[0]
            self._set_editor_text(e.text)
            self.lbl_tags_title.setText("<b>Tags</b>")
            self.tag_editor.set_tags(e.text)
            self._update_info(e)
            self._load_preview(e)
        else:
            self.preview.setPixmap(QPixmap())
            self.preview.setText(f"{len(entries)} images selected")
            self.lbl_info.setText("Bubbles show tags shared by every selected image. "
                                  "Removing or adding one applies to all of them.")
            self._common = self._shared_tags(entries)
            self.lbl_tags_title.setText(f"<b>Shared tags</b> ({len(self._common)})")
            self.tag_editor.set_tags(captions.join_tags(self._common))

    def refresh_from_entries(self) -> None:
        """Re-sync after an external change (undo, batch op) without losing cursor when unchanged."""
        if len(self.entries) == 1:
            e = self.entries[0]
            if self.editor.toPlainText() != e.text and not self._typing_timer.isActive():
                self._set_editor_text(e.text)
            self.tag_editor.set_tags(e.text)
            self._update_info(e)
        elif self.entries:
            self._common = self._shared_tags(self.entries)
            self.lbl_tags_title.setText(f"<b>Shared tags</b> ({len(self._common)})")
            self.tag_editor.set_tags(captions.join_tags(self._common))

    @staticmethod
    def _shared_tags(entries: list[Entry]) -> list[str]:
        def norm(t):
            return " ".join(t.replace("_", " ").split()).lower()
        first = captions.split_tags(entries[0].text)
        common = {norm(t) for t in first}
        for e in entries[1:]:
            common &= {norm(t) for t in captions.split_tags(e.text)}
        return [t for t in first if norm(t) in common]

    def _set_editor_text(self, text: str) -> None:
        self.editor.blockSignals(True)
        self.editor.setPlainText(text)
        self.editor.blockSignals(False)
        self._update_counts()

    def _update_info(self, e: Entry) -> None:
        parts = [e.name]
        if e.width:
            parts.append(f"{e.width} × {e.height}  ({e.width * e.height / 1e6:.2f} MP)")
        if e.file_size:
            parts.append(f"{e.file_size / 1024:.0f} KB")
        if e.dirty:
            parts.append("● unsaved")
        self.lbl_info.setText("   ·   ".join(parts))

    def _update_counts(self) -> None:
        t = self.editor.toPlainText()
        self.lbl_counts.setText(f"{len(captions.split_tags(t))} tags · {len(t.split())} words · {len(t)} chars")

    def _load_preview(self, e: Entry) -> None:
        key = e.key
        self._loading_key = key
        self.preview.setText("Loading…")
        size = max(400, self.preview.width()), max(400, self.preview.height())

        def done(result):
            if self._loading_key != key:
                return  # selection moved on
            img, _ = result
            pm = QPixmap.fromImage(img)
            self.preview.setPixmap(pm.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        run_in_background(lambda: load_qimage(key, size), done,
                          lambda err: self.preview.setText(f"Can't display image:\n{err}"))

    # -- editing ------------------------------------------------------------
    def _on_typing(self) -> None:
        if len(self.entries) != 1:
            return
        if self._pending_old is None:
            self._pending_old = self.entries[0].text
        self._update_counts()
        self._typing_timer.start()

    def _commit_typing(self) -> None:
        self._typing_timer.stop()
        if self._pending_old is None or len(self.entries) != 1:
            self._pending_old = None
            return
        e = self.entries[0]
        old, new = self._pending_old, self.editor.toPlainText()
        self._pending_old = None
        if old != new:
            self.text_edited.emit(e.key, old, new)
            self.tag_editor.set_tags(new)

    def _on_bubbles_changed(self, tag_string: str) -> None:
        new_tags = captions.split_tags(tag_string)
        if len(self.entries) == 1:
            e = self.entries[0]
            # Bubbles edit the tag list; keep any prose as-is by rebuilding from tags only for tag captions.
            new_text = captions.join_tags(new_tags)
            if new_text != e.text:
                self.text_edited.emit(e.key, e.text, new_text)
                self._set_editor_text(new_text)
        elif self.entries:
            def norm(t):
                return " ".join(t.replace("_", " ").split()).lower()
            old_norm = {norm(t) for t in self._common}
            new_norm = {norm(t) for t in new_tags}
            added = [t for t in new_tags if norm(t) not in old_norm]
            removed = [t for t in self._common if norm(t) not in new_norm]
            if removed:
                self.tags_removed.emit(removed)
            if added:
                self.tags_added.emit(added)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        pm = self.preview.pixmap()
        if len(self.entries) == 1 and pm is not None and not pm.isNull():
            self._load_preview(self.entries[0])


# --------------------------------------------------------------------------- batch tools
class BatchPanel(QWidget):
    """Batch caption operations. Emits (description, transform function, scope)."""

    operation = Signal(str, object, str)   # description, fn(text)->text, scope ('selected'|'shown')
    auto_tag = Signal(str)                  # scope
    quick_tag = Signal(str, str)            # tag, position

    def __init__(self, parent=None):
        super().__init__(parent)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(4, 4, 4, 4)

        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Apply to:"))
        self.rad_selected = QRadioButton("Selected")
        self.rad_shown = QRadioButton("All shown")
        self.rad_selected.setChecked(True)
        self.rad_shown.setToolTip("Every image currently visible with the active filter")
        scope_row.addWidget(self.rad_selected)
        scope_row.addWidget(self.rad_shown)
        scope_row.addStretch()
        lay.addLayout(scope_row)
        self.lbl_scope = hint_label("")
        lay.addWidget(self.lbl_scope)

        # Tags
        g = QGroupBox("Tags")
        f = QFormLayout(g)
        self.inp_add = QLineEdit()
        self.inp_add.setPlaceholderText("tag1, tag2")
        self.combo_pos = QComboBox()
        self.combo_pos.addItems(["Append", "Prepend"])
        b_add = QPushButton("Add")
        b_add.clicked.connect(self._add)
        self.inp_add.returnPressed.connect(self._add)
        row = QHBoxLayout()
        row.addWidget(self.inp_add, 1)
        row.addWidget(self.combo_pos)
        row.addWidget(b_add)
        f.addRow("Add:", row)
        self.inp_remove = QLineEdit()
        self.inp_remove.setPlaceholderText("tags to remove")
        b_rm = QPushButton("Remove")
        b_rm.clicked.connect(self._remove)
        self.inp_remove.returnPressed.connect(self._remove)
        row = QHBoxLayout()
        row.addWidget(self.inp_remove, 1)
        row.addWidget(b_rm)
        f.addRow("Remove:", row)
        self.inp_old = QLineEdit()
        self.inp_old.setPlaceholderText("old tag")
        self.inp_new = QLineEdit()
        self.inp_new.setPlaceholderText("new tag (empty = delete)")
        b_rep = QPushButton("Replace")
        b_rep.clicked.connect(self._replace)
        row = QHBoxLayout()
        row.addWidget(self.inp_old, 1)
        row.addWidget(QLabel("→"))
        row.addWidget(self.inp_new, 1)
        row.addWidget(b_rep)
        f.addRow("Replace:", row)
        lay.addWidget(g)

        # Text
        g = QGroupBox("Text")
        f = QFormLayout(g)
        self.inp_find = QLineEdit()
        self.inp_find.setPlaceholderText("find")
        self.inp_repl = QLineEdit()
        self.inp_repl.setPlaceholderText("replace with")
        self.chk_regex = QCheckBox("Regex")
        self.chk_case = QCheckBox("Match case")
        b_fr = QPushButton("Find && Replace")
        b_fr.clicked.connect(self._find_replace)
        f.addRow("Find:", self.inp_find)
        f.addRow("Replace:", self.inp_repl)
        row = QHBoxLayout()
        row.addWidget(self.chk_regex)
        row.addWidget(self.chk_case)
        row.addStretch()
        row.addWidget(b_fr)
        f.addRow(row)
        self.inp_prefix = QLineEdit()
        self.inp_prefix.setPlaceholderText("e.g. ohwx woman, ")
        self.inp_suffix = QLineEdit()
        b_ps = QPushButton("Apply")
        b_ps.clicked.connect(self._prefix_suffix)
        f.addRow("Prefix:", self.inp_prefix)
        row = QHBoxLayout()
        row.addWidget(self.inp_suffix, 1)
        row.addWidget(b_ps)
        f.addRow("Suffix:", row)
        lay.addWidget(g)

        # Cleanup
        sec = CollapsibleSection("Cleanup & normalize", expanded=False)
        self.chk_us = QCheckBox("Underscores → spaces")
        self.chk_lower = QCheckBox("Lowercase")
        self.chk_dedupe = QCheckBox("Remove duplicate tags")
        self.chk_dedupe.setChecked(True)
        self.chk_sort = QCheckBox("Sort tags A→Z")
        self.chk_escape = QCheckBox(r"Escape parentheses ( → \( )")
        for c in (self.chk_us, self.chk_lower, self.chk_dedupe, self.chk_sort, self.chk_escape):
            sec.body_layout.addWidget(c)
        b_norm = QPushButton("Normalize")
        b_norm.clicked.connect(self._normalize)
        b_ascii = QPushButton("Accents → ASCII")
        b_ascii.setToolTip("é → e, ü → u (some trainers' tokenizers dislike accents)")
        b_ascii.clicked.connect(lambda: self._emit("Convert to ASCII", captions.to_ascii))
        b_clear = QPushButton("Clear captions…")
        b_clear.setStyleSheet("color: #ff7675;")
        b_clear.clicked.connect(lambda: self._emit("Clear captions", lambda t: ""))
        row = QHBoxLayout()
        row.addWidget(b_norm)
        row.addWidget(b_ascii)
        row.addWidget(b_clear)
        sec.body_layout.addLayout(row)
        lay.addWidget(sec)

        # AI
        g = QGroupBox("AI tagging")
        v = QVBoxLayout(g)
        self.btn_auto = QPushButton("✨ Auto Tag (WD tagger)…")
        self.btn_auto.setStyleSheet("background-color: #6c5ce7; color: white; font-weight: bold; padding: 6px;")
        self.btn_auto.clicked.connect(lambda: self.auto_tag.emit(self.scope()))
        v.addWidget(self.btn_auto)
        v.addWidget(hint_label("Results stay unsaved (and undoable) until you press Save. "
                               "For natural-language captions use the Auto Caption tab."))
        lay.addWidget(g)

        # Quick tags
        g = QGroupBox("Quick tags")
        v = QVBoxLayout(g)
        row = QHBoxLayout()
        self.inp_quick = QLineEdit()
        self.inp_quick.setPlaceholderText("new quick tag")
        self.inp_quick.returnPressed.connect(self._add_quick)
        self.combo_quick_pos = QComboBox()
        self.combo_quick_pos.addItems(["Append", "Prepend"])
        row.addWidget(self.inp_quick, 1)
        row.addWidget(self.combo_quick_pos)
        v.addLayout(row)
        self.list_quick = QListWidget()
        self.list_quick.setMaximumHeight(170)
        self.list_quick.itemClicked.connect(
            lambda it: self.quick_tag.emit(it.text(), self.combo_quick_pos.currentText().lower()))
        self.list_quick.addItems(load_quick_tags())
        v.addWidget(self.list_quick)
        v.addWidget(hint_label("Click a quick tag to add it to the selection."))
        lay.addWidget(g)
        lay.addStretch()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scroll(inner))

    def scope(self) -> str:
        return "shown" if self.rad_shown.isChecked() else "selected"

    def set_counts(self, selected: int, shown: int) -> None:
        self.rad_selected.setText(f"Selected ({selected})")
        self.rad_shown.setText(f"All shown ({shown})")

    def _emit(self, desc, fn):
        self.operation.emit(desc, fn, self.scope())

    def _add(self):
        tags = captions.split_tags(self.inp_add.text())
        if tags:
            pos = self.combo_pos.currentText().lower()
            self._emit(f"Add tags: {', '.join(tags)}", lambda t: captions.add_tags(t, tags, pos))

    def _remove(self):
        from core.dataset_session import op_remove_tags
        tags = captions.split_tags(self.inp_remove.text())
        if tags:
            self._emit(f"Remove tags: {', '.join(tags)}", op_remove_tags(tags))

    def _replace(self):
        from core.dataset_session import op_replace_tag
        old, new = self.inp_old.text().strip(), self.inp_new.text().strip()
        if old:
            self._emit(f"Replace tag '{old}' → '{new}'", op_replace_tag(old, new))

    def _find_replace(self):
        import re
        find = self.inp_find.text()
        if not find:
            return
        if self.chk_regex.isChecked():
            try:
                re.compile(find)
            except re.error as e:
                self.lbl_scope.setText(f"Invalid regular expression: {e}")
                return
        repl, regex, case = self.inp_repl.text(), self.chk_regex.isChecked(), self.chk_case.isChecked()
        self._emit(f"Find '{find}' → '{repl}'",
                   lambda t: captions.find_replace(t, find, repl, regex=regex, case_sensitive=case))

    def _prefix_suffix(self):
        from core.dataset_session import op_prefix_suffix
        p, s = self.inp_prefix.text(), self.inp_suffix.text()
        if p or s:
            self._emit("Add prefix/suffix", op_prefix_suffix(p, s))

    def _normalize(self):
        from core.dataset_session import op_normalize
        self._emit("Normalize tags", op_normalize(
            underscores_to_spaces=self.chk_us.isChecked(), lowercase=self.chk_lower.isChecked(),
            dedupe=self.chk_dedupe.isChecked(), sort=self.chk_sort.isChecked(),
            escape_parentheses=self.chk_escape.isChecked()))

    def _add_quick(self):
        tag = self.inp_quick.text().strip()
        items = [self.list_quick.item(i).text() for i in range(self.list_quick.count())]
        if tag and tag not in items:
            self.list_quick.addItem(tag)
            try:
                save_quick_tags(items + [tag])
            except OSError:
                pass
        self.inp_quick.clear()


# --------------------------------------------------------------------------- tag statistics
class TagStatsPanel(QWidget):
    filter_requested = Signal(str)          # query text
    rename_requested = Signal(str, str)     # old, new (dataset-wide)
    delete_requested = Signal(str)          # tag (dataset-wide)
    add_to_selection = Signal(str)
    select_with_tag = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        self.combo_scope = QComboBox()
        self.combo_scope.addItems(["Whole folder", "Shown images", "Selected images"])
        self.combo_scope.setToolTip("Which images to count tags over")
        self.inp_filter = QLineEdit()
        self.inp_filter.setPlaceholderText("Search tags…")
        self.inp_filter.textChanged.connect(self._apply_filter)
        row.addWidget(self.combo_scope)
        row.addWidget(self.inp_filter, 1)
        lay.addLayout(row)
        self.lbl_summary = hint_label("")
        lay.addWidget(self.lbl_summary)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Tag", "Images", "%"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSortingEnabled(True)
        self.table.cellDoubleClicked.connect(lambda r, c: self._emit_filter(r))
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._menu)
        lay.addWidget(self.table, 1)
        lay.addWidget(hint_label("Double-click a tag to show only images with it. Right-click to rename, "
                                 "merge or delete it everywhere, or to find rare tags."))

    def scope(self) -> str:
        return ("all", "shown", "selected")[self.combo_scope.currentIndex()]

    def set_counts(self, counts: list[tuple[str, int]], total_images: int) -> None:
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(counts))
        for r, (tag, n) in enumerate(counts):
            self.table.setItem(r, 0, QTableWidgetItem(tag))
            it = QTableWidgetItem()
            it.setData(Qt.DisplayRole, n)
            self.table.setItem(r, 1, it)
            pct = QTableWidgetItem()
            pct.setData(Qt.DisplayRole, round(100 * n / total_images, 1) if total_images else 0.0)
            self.table.setItem(r, 2, pct)
        self.table.setSortingEnabled(True)
        self.table.sortItems(1, Qt.DescendingOrder)
        rare = sum(1 for _, n in counts if n == 1)
        self.lbl_summary.setText(f"{len(counts)} unique tags over {total_images} images · {rare} appear only once")
        self._apply_filter(self.inp_filter.text())

    def _apply_filter(self, text: str) -> None:
        t = text.strip().lower()
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            self.table.setRowHidden(r, bool(t) and t not in item.text().lower())

    def _tag_at(self, row: int) -> str:
        item = self.table.item(row, 0)
        return item.text() if item else ""

    def _emit_filter(self, row: int) -> None:
        tag = self._tag_at(row)
        if tag:
            self.filter_requested.emit(f'tag:"{tag}"')

    def _menu(self, pos) -> None:
        row = self.table.rowAt(pos.y())
        tag = self._tag_at(row) if row >= 0 else ""
        menu = QMenu(self)
        if tag:
            menu.addAction(f"Show images with '{tag}'", lambda: self.filter_requested.emit(f'tag:"{tag}"'))
            menu.addAction(f"Show images without '{tag}'", lambda: self.filter_requested.emit(f'-tag:"{tag}"'))
            menu.addAction("Select images with this tag", lambda: self.select_with_tag.emit(tag))
            menu.addAction("Add to selected images", lambda: self.add_to_selection.emit(tag))
            menu.addSeparator()
            menu.addAction("Rename / merge everywhere…", lambda: self._rename(tag))
            menu.addAction("Delete everywhere…", lambda: self.delete_requested.emit(tag))
            menu.addSeparator()
        menu.addAction("Show only tags used once", lambda: self._show_rare())
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _rename(self, tag: str) -> None:
        from PySide6.QtWidgets import QInputDialog
        new, ok = QInputDialog.getText(self, "Rename / merge tag",
                                       f"Replace '{tag}' everywhere with (an existing tag name merges them):",
                                       text=tag)
        if ok and new.strip() and new.strip() != tag:
            self.rename_requested.emit(tag, new.strip())

    def _show_rare(self) -> None:
        for r in range(self.table.rowCount()):
            self.table.setRowHidden(r, self.table.item(r, 1).data(Qt.DisplayRole) != 1)
