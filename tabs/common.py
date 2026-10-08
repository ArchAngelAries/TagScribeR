"""Shared UI building blocks used across tabs."""
from __future__ import annotations

import logging
import traceback
from typing import Any, Callable

from PySide6.QtCore import (QEvent, QItemSelection, QItemSelectionModel, QObject, QPersistentModelIndex, QRunnable,
                            Qt, QThreadPool, QTimer, Signal, Slot)
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QFrame, QLabel, QMessageBox, QPlainTextEdit,
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
            pass  # app is shutting down (task finished during exit)


# Tasks are held here until their result has been delivered on the UI thread.
# Without a strong reference Python may garbage-collect a running task (and its
# signal object), silently dropping the result.
_live_tasks: set[_Task] = set()


def run_in_background(fn: Callable[[], Any], on_done: Callable[[Any], None] | None = None,
                      on_error: Callable[[str], None] | None = None, pool: QThreadPool | None = None) -> None:
    """Run a short blocking function on the global thread pool; callbacks run on the UI thread."""
    task = _Task(fn)
    task.setAutoDelete(False)  # Python owns it; released below once the result is delivered
    _live_tasks.add(task)

    def release(*_):
        _live_tasks.discard(task)

    if on_done:
        task.signals.done.connect(on_done)
    if on_error:
        task.signals.failed.connect(on_error)
    task.signals.done.connect(release)
    task.signals.failed.connect(release)
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
        self.toggle = QToolButton(text=title.replace("&", "&&"), checkable=True, checked=expanded)
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


class QuickTagList(QWidget):
    """Persistent quick-tag list: add, remove (right-click / Delete), drag to reorder.

    Every change is written to user_data immediately and all open lists refresh,
    so the Gallery and Settings never disagree. ``tag_clicked`` emits the entry's
    tags (an entry may hold several, e.g. "ohwx, 1girl").
    """

    tag_clicked = Signal(list)

    def __init__(self, parent=None, show_input: bool = True, click_hint: str = ""):
        super().__init__(parent)
        from PySide6.QtWidgets import QAbstractItemView, QHBoxLayout, QLineEdit, QListWidget
        from core import config
        self._config = config
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        if show_input:
            row = QHBoxLayout()
            self.input = QLineEdit()
            self.input.setPlaceholderText("New quick tag (Enter) — e.g. 'ohwx, 1girl' adds both")
            self.input.returnPressed.connect(self._add)
            row.addWidget(self.input, 1)
            lay.addLayout(row)
        self.list = QListWidget()
        self.list.setDragDropMode(QAbstractItemView.InternalMove)
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.setToolTip((click_hint + "\n" if click_hint else "") +
                             "Drag to reorder · right-click or Delete to remove")
        self.list.itemClicked.connect(self._clicked)
        self.list.model().rowsMoved.connect(self._save_order)
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._menu)
        lay.addWidget(self.list)
        from PySide6.QtGui import QKeySequence, QShortcut
        sc = QShortcut(QKeySequence("Del"), self.list)
        sc.setContext(Qt.WidgetShortcut)
        sc.activated.connect(self._remove_selected)
        config.on_quick_tags_changed(self.reload)
        self.reload()

    def reload(self):
        try:
            tags = self._config.load_quick_tags()
            self.list.blockSignals(True)
            self.list.clear()
            self.list.addItems(tags)
            self.list.blockSignals(False)
        except RuntimeError:
            pass  # this widget was already destroyed

    def _clicked(self, item):
        from core import captions
        tags = captions.split_tags(item.text())
        if tags:
            self.tag_clicked.emit(tags)

    def _add(self):
        text = self.input.text().strip()
        if text:
            try:
                self._config.add_quick_tag(text)
            except OSError as e:
                QMessageBox.warning(self, "Quick tags", f"Could not save quick tags: {e}")
        self.input.clear()

    def _remove_selected(self):
        tags = [i.text() for i in self.list.selectedItems()]
        if tags:
            try:
                self._config.remove_quick_tags(tags)
            except OSError as e:
                QMessageBox.warning(self, "Quick tags", f"Could not save quick tags: {e}")

    def _save_order(self, *_):
        order = [self.list.item(i).text() for i in range(self.list.count())]
        try:
            self._config.save_quick_tags(order)
        except OSError as e:
            QMessageBox.warning(self, "Quick tags", f"Could not save quick tags: {e}")

    def _menu(self, pos):
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        n = len(self.list.selectedItems())
        a = menu.addAction(f"Remove {n} quick tag(s)" if n > 1 else "Remove quick tag", self._remove_selected)
        a.setEnabled(n > 0)
        menu.exec(self.list.viewport().mapToGlobal(pos))


class ClickToDeselect(QObject):
    """A plain click on a selected item of an item view deselects that item and keeps the rest of the selection (Qt
    would keep it selected, or select only it). A click on an unselected item, Ctrl / Shift clicks, dragging and
    double-clicking behave as Qt's. Acts on the release, after Qt's own handling."""

    def __init__(self, view):
        super().__init__(view)
        self.view = view
        self._armed = None                  # (the clicked index, press position, the selection at the press)
        view.viewport().installEventFilter(self)

    def eventFilter(self, obj, event):
        t = event.type()
        if t == QEvent.MouseButtonPress:
            self._armed = None
            if event.button() == Qt.LeftButton and not (event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier)):
                pos = event.position().toPoint()
                idx = self.view.indexAt(pos)
                sm = self.view.selectionModel()
                if idx.isValid() and sm is not None and sm.isSelected(idx):
                    self._armed = (QPersistentModelIndex(idx), pos,
                                   [QPersistentModelIndex(i) for i in sm.selectedIndexes()])
        elif t == QEvent.MouseButtonRelease and self._armed is not None:
            idx, pos, before = self._armed
            self._armed = None
            moved = (event.position().toPoint() - pos).manhattanLength() >= QApplication.startDragDistance()
            if not moved and QPersistentModelIndex(self.view.indexAt(event.position().toPoint())) == idx:
                QTimer.singleShot(0, lambda: self._deselect(idx, before))
        elif t == QEvent.MouseButtonDblClick:
            self._armed = None
        return False

    def _deselect(self, idx, before):
        """The selection as it was at the press, without the clicked item."""
        sm = self.view.selectionModel()
        if sm is None:
            return
        keep = QItemSelection()
        for i in before:
            if i.isValid() and i != idx:
                keep.select(sm.model().index(i.row(), i.column(), i.parent()),
                            sm.model().index(i.row(), i.column(), i.parent()))
        sm.select(keep, QItemSelectionModel.ClearAndSelect)


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
