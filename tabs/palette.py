"""Command palette (Ctrl+K): type to find and run any action."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout


@dataclass
class Command:
    title: str
    run: Callable[[], object]
    group: str = ""
    shortcut: str = ""
    keywords: str = ""
    enabled: Callable[[], bool] = field(default=lambda: True)


def fuzzy_score(query: str, text: str) -> int | None:
    """Higher is better; None = no match. Every query word must match as a
    substring or as an in-order subsequence; word starts and contiguous runs score more."""
    q_words = query.lower().split()
    t = text.lower()
    if not q_words:
        return 0
    total = 0
    for w in q_words:
        idx = t.find(w)
        if idx >= 0:
            total += 100 + (50 if idx == 0 or not t[idx - 1].isalnum() else 0) - idx // 4
            continue
        pos, score, run = 0, 0, 0
        for ch in w:
            found = t.find(ch, pos)
            if found < 0:
                return None
            run = run + 1 if found == pos else 0
            score += 5 + run * 3 + (8 if found == 0 or not t[found - 1].isalnum() else 0)
            pos = found + 1
        total += score
    return total


def rank(query: str, commands: list[Command]) -> list[Command]:
    scored = []
    for c in commands:
        s = fuzzy_score(query, f"{c.group} {c.title} {c.keywords}")
        if s is not None:
            if c.group == "Help":
                s -= 25  # actions first; help topics still match when they're the best fit
            scored.append((s, c))
    scored.sort(key=lambda sc: -sc[0])
    return [c for _s, c in scored]


class CommandPalette(QDialog):
    def __init__(self, commands: list[Command], parent=None):
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint)
        self.commands = [c for c in commands if c.enabled()]
        self.setMinimumSize(620, 420)
        self.setStyleSheet("QDialog { background-color: #26292c; border: 1px solid #00b894; border-radius: 8px; }")
        lay = QVBoxLayout(self)
        self.input = QLineEdit()
        self.input.setPlaceholderText("Type a command… (e.g. 'missing', 'save', 'health', 'help filter')")
        self.input.textChanged.connect(self._refresh)
        self.input.returnPressed.connect(self._run_current)
        self.input.installEventFilter(self)
        self.list = QListWidget()
        self.list.itemActivated.connect(lambda _i: self._run_current())
        self.hint = QLabel("↑↓ to choose · Enter to run · Esc to close")
        self.hint.setStyleSheet("color: #888; font-size: 11px;")
        lay.addWidget(self.input)
        lay.addWidget(self.list, 1)
        lay.addWidget(self.hint)
        self._refresh("")
        if parent is not None:
            geo = parent.geometry()
            self.move(geo.center().x() - 310, geo.top() + 90)

    def _refresh(self, text: str):
        self.list.clear()
        for c in (rank(text, self.commands) if text.strip() else self.commands)[:60]:
            label = f"{c.group}:  {c.title}" if c.group else c.title
            if c.shortcut:
                label += f"      {c.shortcut}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, c)
            self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(0)

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        if obj is self.input and event.type() == QEvent.KeyPress and event.key() in (Qt.Key_Down, Qt.Key_Up):
            row = self.list.currentRow() + (1 if event.key() == Qt.Key_Down else -1)
            if 0 <= row < self.list.count():
                self.list.setCurrentRow(row)
            return True
        return super().eventFilter(obj, event)

    def _run_current(self):
        item = self.list.currentItem()
        if item is None:
            return
        cmd: Command = item.data(Qt.UserRole)
        self.close()
        cmd.run()
