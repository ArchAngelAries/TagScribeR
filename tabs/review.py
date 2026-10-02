"""Review queue for AI-generated captions: compare, edit, accept or reject before anything changes.

Accepted captions go through the shared workspace (undoable, unsaved until
Save); rejected ones are simply dropped. Nothing touches disk here.
"""
from __future__ import annotations

import html
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPlainTextEdit,
                               QPushButton, QSplitter, QTextBrowser, QVBoxLayout, QWidget)

from core import captions
from core.image_utils import load_qimage
from tabs.common import hint_label, run_in_background


def diff_html(old: str, new: str) -> str:
    parts = []
    marked = False
    for op, text in captions.word_diff(old, new):
        t = html.escape(text)
        if op != "same" and not marked:
            parts.append("<a name='first'></a>")  # the view scrolls here so changes are visible at once
            marked = True
        if op == "del":
            parts.append(f"<span style='color:#ff7675;text-decoration:line-through'>{t}</span>")
        elif op == "add":
            parts.append(f"<span style='color:#55efc4;font-weight:bold'>{t}</span>")
        else:
            parts.append(f"<span style='color:#cfcfcf'>{t}</span>")
    return "".join(parts) or "<i style='color:#777'>(empty)</i>"


class ReviewDialog(QDialog):
    """Non-modal. ``items`` is an ordered list of (key, current_text, proposed_text)."""

    accepted_item = Signal(str, str, str)   # key, old text, new text
    rejected_item = Signal(str)

    def __init__(self, items: list[tuple[str, str, str]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Review AI captions")
        self.resize(1150, 720)
        self.items: dict[str, tuple[str, str]] = {k: (cur, new) for k, cur, new in items}
        self.order = [k for k, _c, _n in items]
        self._loading = ""

        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.lbl_left = QLabel()
        ll.addWidget(self.lbl_left)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._show_row)
        ll.addWidget(self.list, 1)
        splitter.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        top = QHBoxLayout()
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setFixedSize(300, 300)
        self.preview.setStyleSheet("background-color: #1a1a1a; border-radius: 6px;")
        top.addWidget(self.preview)
        cur_box = QVBoxLayout()
        cur_box.addWidget(QLabel("<b>Current caption</b>"))
        self.current = QTextBrowser()
        self.current.setStyleSheet("QTextBrowser { color: #bbb; background-color: #1e1e1e; }")
        cur_box.addWidget(self.current)
        top.addLayout(cur_box, 1)
        rl.addLayout(top)
        rl.addWidget(QLabel("<b>Changes</b>  <span style='color:#55efc4'>added</span> · "
                            "<span style='color:#ff7675'>removed</span>"))
        self.changes = QTextBrowser()
        self.changes.setStyleSheet("QTextBrowser { background-color: #1e1e1e; }")
        self.changes.setMaximumHeight(150)
        rl.addWidget(self.changes)
        rl.addWidget(QLabel("<b>Caption after accepting</b> (you can edit it)"))
        self.proposed = QPlainTextEdit()
        self.proposed.textChanged.connect(self._update_changes)
        rl.addWidget(self.proposed, 1)
        self.lbl_tokens = hint_label("")
        rl.addWidget(self.lbl_tokens)

        btns = QHBoxLayout()
        self.btn_accept = QPushButton("✓ Accept  (Ctrl+Enter)")
        self.btn_accept.setStyleSheet("background-color: #00b894; color: white; font-weight: bold; padding: 6px;")
        self.btn_accept.clicked.connect(self.accept_current)
        self.btn_reject = QPushButton("✗ Reject  (Ctrl+Backspace)")
        self.btn_reject.clicked.connect(self.reject_current)
        self.btn_skip = QPushButton("Skip →")
        self.btn_skip.setToolTip("Decide later (Ctrl+Right)")
        self.btn_skip.clicked.connect(lambda: self._move(1))
        self.btn_accept_all = QPushButton("Accept all")
        self.btn_accept_all.setToolTip("Accept every remaining proposal as shown")
        self.btn_accept_all.clicked.connect(self.accept_all)
        self.btn_reject_all = QPushButton("Reject all")
        self.btn_reject_all.clicked.connect(self.reject_all)
        for b in (self.btn_accept, self.btn_reject, self.btn_skip):
            btns.addWidget(b)
        btns.addStretch()
        btns.addWidget(self.btn_accept_all)
        btns.addWidget(self.btn_reject_all)
        rl.addLayout(btns)
        rl.addWidget(hint_label("Accepted captions are applied to the workspace as unsaved, undoable edits — press "
                                "Save in any tab to write them. Rejected proposals are discarded."))
        splitter.addWidget(right)
        splitter.setSizes([300, 850])
        root.addWidget(splitter)

        for seq, fn in (("Ctrl+Return", self.accept_current), ("Ctrl+Enter", self.accept_current),
                        ("Ctrl+Backspace", self.reject_current), ("Ctrl+Right", lambda: self._move(1)),
                        ("Ctrl+Left", lambda: self._move(-1))):
            QShortcut(QKeySequence(seq), self).activated.connect(fn)
        self._refill()

    # ------------------------------------------------------------------ queue
    def add_items(self, items: list[tuple[str, str, str]]) -> None:
        """Add proposals that arrive while the dialog is open."""
        for k, cur, new in items:
            if k not in self.items:
                self.order.append(k)
            self.items[k] = (cur, new)
        self._refill(keep=self.current_key())

    def current_key(self) -> str:
        row = self.list.currentRow()
        return self.order[row] if 0 <= row < len(self.order) else ""

    def _refill(self, keep: str = ""):
        self.list.blockSignals(True)
        self.list.clear()
        for k in self.order:
            self.list.addItem(QListWidgetItem(Path(k).name))
        self.list.blockSignals(False)
        self.lbl_left.setText(f"<b>{len(self.order)}</b> proposal(s) to review")
        if not self.order:
            self.close()
            return
        row = self.order.index(keep) if keep in self.order else 0
        self.list.setCurrentRow(row)
        self._show_row(row)

    def _show_row(self, row: int):
        if not (0 <= row < len(self.order)):
            return
        key = self.order[row]
        cur, new = self.items[key]
        self.current.setPlainText(cur or "(no caption)")
        self.proposed.blockSignals(True)
        self.proposed.setPlainText(new)
        self.proposed.blockSignals(False)
        self._update_changes()
        self._load_preview(key)

    def _update_changes(self):
        key = self.current_key()
        if not key:
            return
        cur = self.items[key][0]
        new = self.proposed.toPlainText()
        self.changes.setHtml(diff_html(cur, new))
        self.changes.scrollToAnchor("first")
        n = captions.estimate_clip_tokens(new)
        self.lbl_tokens.setText(f"≈{n} CLIP tokens" + (" (over SD1.5/SDXL's 75)" if n > 75 else ""))

    def _load_preview(self, key: str):
        self._loading = key

        def done(res):
            if self._loading == key:
                pm = QPixmap.fromImage(res[0])
                self.preview.setPixmap(pm.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        run_in_background(lambda: load_qimage(key, (300, 300)), done)

    def _move(self, step: int):
        if self.order:
            self.list.setCurrentRow((self.list.currentRow() + step) % len(self.order))

    # ------------------------------------------------------------------ decisions
    def _remove(self, key: str):
        idx = self.order.index(key)
        self.order.remove(key)
        self.items.pop(key, None)
        nxt = self.order[min(idx, len(self.order) - 1)] if self.order else ""
        self._refill(keep=nxt)

    def accept_current(self):
        key = self.current_key()
        if key:
            self.accepted_item.emit(key, self.items[key][0], self.proposed.toPlainText())
            self._remove(key)

    def reject_current(self):
        key = self.current_key()
        if key:
            self.rejected_item.emit(key)
            self._remove(key)

    def accept_all(self):
        edited_key, edited_text = self.current_key(), self.proposed.toPlainText()
        for key in list(self.order):
            cur, new = self.items[key]
            self.accepted_item.emit(key, cur, edited_text if key == edited_key else new)
        self.order.clear()
        self.items.clear()
        self._refill()

    def reject_all(self):
        for key in list(self.order):
            self.rejected_item.emit(key)
        self.order.clear()
        self.items.clear()
        self._refill()
