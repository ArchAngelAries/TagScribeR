"""Shared UI building blocks used across tabs."""
from __future__ import annotations

import logging
import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFrame, QLabel, QMessageBox, QPlainTextEdit,
                               QSizePolicy, QToolButton, QVBoxLayout, QWidget)

from core.image_utils import load_thumbnail

log = logging.getLogger(__name__)


# -- background helpers --------------------------------------------------------
class _TaskSignals(QObject):
    done = Signal(object)
    failed = Signal(str)


class _Task(QRunnable):
    def __init__(self, fn: Callable[[], Any]):
        super().__init__()
        self.fn = fn
        self.signals = _TaskSignals()

    @Slot()
    def run(self):
        try:
            result = self.fn()
        except Exception as e:
            log.debug("Background task failed:\n%s", traceback.format_exc())
            self._emit(self.signals.failed, str(e))
            return
        self._emit(self.signals.done, result)

    @staticmethod
    def _emit(signal, value):
        try:
            signal.emit(value)
        except RuntimeError:
            pass  # receiver/app already shut down (task finished during exit)


def run_in_background(fn: Callable[[], Any], on_done: Callable[[Any], None] | None = None,
                      on_error: Callable[[str], None] | None = None, pool: QThreadPool | None = None) -> None:
    """Run a short blocking function on the global thread pool; callbacks run on the UI thread."""
    task = _Task(fn)
    if on_done:
        task.signals.done.connect(on_done)
    if on_error:
        task.signals.failed.connect(on_error)
    (pool or QThreadPool.globalInstance()).start(task)


class ThumbnailSignals(QObject):
    loaded = Signal(str, QPixmap)


class ThumbnailWorker(QRunnable):
    """Decode a thumbnail off the UI thread (one shared implementation for all tabs)."""

    def __init__(self, path: str, size: tuple[int, int]):
        super().__init__()
        self.path = path
        self.size = size
        self.signals = ThumbnailSignals()

    @Slot()
    def run(self):
        self.signals.loaded.emit(self.path, load_thumbnail(self.path, self.size))


# -- widgets ------------------------------------------------------------------
class CollapsibleSection(QWidget):
    """A titled section whose body can be expanded/collapsed (progressive disclosure)."""

    def __init__(self, title: str, expanded: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.toggle = QToolButton(text=title, checkable=True, checked=expanded)
        self.toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; color: #bbb; }")
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.body = QFrame()
        self.body.setVisible(expanded)
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(8, 2, 0, 4)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self.toggle)
        lay.addWidget(self.body)
        self.toggle.toggled.connect(self._on_toggled)

    def _on_toggled(self, on: bool):
        self.toggle.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self.body.setVisible(on)


def hint_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("color: #8a8a8a; font-size: 11px;")
    return lbl


# -- dialogs ------------------------------------------------------------------
def show_job_summary(parent: QWidget, summary) -> None:
    """Report a finished batch; lists failures with reasons instead of hiding them."""
    if summary.fatal:
        QMessageBox.warning(parent, summary.title, summary.fatal)
        return
    if not summary.failed:
        return  # success is reported unobtrusively in the status area
    dlg = QDialog(parent)
    dlg.setWindowTitle(f"{summary.title} — {len(summary.failed)} problem(s)")
    dlg.resize(720, 420)
    lay = QVBoxLayout(dlg)
    lay.addWidget(QLabel(summary.text() + "\nCompleted images were saved. These need attention:"))
    box = QPlainTextEdit()
    box.setReadOnly(True)
    box.setPlainText("\n\n".join(f"{p}\n    {r}" for p, r in summary.failed))
    lay.addWidget(box)
    btns = QDialogButtonBox(QDialogButtonBox.Close)
    btns.rejected.connect(dlg.reject)
    btns.accepted.connect(dlg.accept)
    lay.addWidget(btns)
    dlg.exec()


def confirm(parent: QWidget, title: str, text: str, destructive: bool = False) -> bool:
    box = QMessageBox(QMessageBox.Warning if destructive else QMessageBox.Question, title, text,
                      QMessageBox.Yes | QMessageBox.Cancel, parent)
    box.setDefaultButton(QMessageBox.Cancel if destructive else QMessageBox.Yes)
    return box.exec() == QMessageBox.Yes
