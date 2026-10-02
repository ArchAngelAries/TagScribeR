"""Help Center: searchable, non-modal, context-aware (F1 opens the current tab's topic)."""
from __future__ import annotations

import html
import re

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem, QPushButton,
                               QTextBrowser, QToolButton, QVBoxLayout, QWidget)

from tabs.help_content import HOTKEYS, TOPICS

_STYLE = """
<style>
 body { font-size: 13px; line-height: 1.45; }
 h2 { color: #00b894; margin-bottom: 6px; }
 h3 { color: #d0d0d0; margin-top: 14px; }
 code { background: #2b2b2b; color: #fdcb6e; padding: 1px 4px; }
 td { vertical-align: top; }
</style>
"""


def _hotkeys_html() -> str:
    rows, scope = [], None
    for sc, keys, action in HOTKEYS:
        if sc != scope:
            rows.append(f'<tr><td colspan="2"><h3>{html.escape(sc)}</h3></td></tr>')
            scope = sc
        rows.append(f"<tr><td><code>{html.escape(keys)}</code></td><td>{html.escape(action)}</td></tr>")
    return "<h2>Keyboard shortcuts</h2><table cellpadding='4'>" + "".join(rows) + "</table>"


def _all_topics() -> dict[str, tuple[str, str, str]]:
    topics = dict(TOPICS)
    topics["hotkeys"] = ("Keyboard shortcuts", "Basics", _hotkeys_html())
    return topics


class HelpDialog(QDialog):
    _instance: "HelpDialog | None" = None

    @classmethod
    def open_topic(cls, parent: QWidget | None, topic: str = "start") -> "HelpDialog":
        """Show the (single, non-modal) Help Center at ``topic``."""
        if cls._instance is None:
            cls._instance = HelpDialog(parent)
        cls._instance.show_topic(topic)
        cls._instance.show()
        cls._instance.raise_()
        cls._instance.activateWindow()
        return cls._instance

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TagScribeR — Help Center")
        self.resize(980, 680)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.topics = _all_topics()

        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search help…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        self.list = QListWidget()
        self.list.setMaximumWidth(250)
        self.list.currentItemChanged.connect(self._on_item)
        left.addWidget(self.search)
        left.addWidget(self.list, 1)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        left.addWidget(btn_close)
        lay.addLayout(left)

        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(True)
        self.view.setStyleSheet("QTextBrowser { color: #dcdcdc; background-color: #1e1e1e; padding: 10px; }")
        lay.addWidget(self.view, 1)
        self._populate()

    def _populate(self):
        self.list.clear()
        group = None
        order = ["Basics", "Gallery", "AI", "Tools", "Settings", "Help"]
        for g in order:
            for tid, (title, tgroup, _html) in self.topics.items():
                if tgroup != g:
                    continue
                if g != group:
                    header = QListWidgetItem(g.upper())
                    header.setFlags(Qt.NoItemFlags)
                    header.setForeground(Qt.gray)
                    self.list.addItem(header)
                    group = g
                item = QListWidgetItem("   " + title)
                item.setData(Qt.UserRole, tid)
                self.list.addItem(item)

    def show_topic(self, tid: str):
        if tid not in self.topics:
            tid = "start"
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.UserRole) == tid:
                self.list.setCurrentRow(i)
                break
        self._render(tid)

    def _on_item(self, item, _prev=None):
        if item is not None and item.data(Qt.UserRole):
            self._render(item.data(Qt.UserRole))

    def _render(self, tid: str):
        title, _group, body = self.topics[tid]
        self.view.setHtml(_STYLE + body)

    def _filter(self, text: str):
        q = text.strip().lower()
        for i in range(self.list.count()):
            item = self.list.item(i)
            tid = item.data(Qt.UserRole)
            if not tid:
                item.setHidden(bool(q))
                continue
            title, _g, body = self.topics[tid]
            plain = re.sub(r"<[^>]+>", " ", body).lower()
            item.setHidden(bool(q) and q not in title.lower() and q not in plain)
        if q:
            for i in range(self.list.count()):
                if not self.list.item(i).isHidden() and self.list.item(i).data(Qt.UserRole):
                    self.list.setCurrentRow(i)
                    break


def help_button(topic: str, parent: QWidget | None = None, tooltip: str = "Help for this panel") -> QToolButton:
    """A small '?' button that opens the Help Center at ``topic``."""
    b = QToolButton(parent)
    b.setText("?")
    b.setToolTip(tooltip)
    b.setFixedSize(26, 26)
    b.setStyleSheet("QToolButton { border-radius: 13px; background: #3a3f44; color: #ddd; font-weight: bold;"
                    " padding: 0px; margin: 0px; min-width: 0px; min-height: 0px; }"
                    "QToolButton:hover { background: #00b894; color: white; }")
    b.clicked.connect(lambda: HelpDialog.open_topic(b.window(), topic))
    return b
