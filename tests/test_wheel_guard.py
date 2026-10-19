"""WheelGuard (tabs/common.py): the mouse wheel scrolls the panel instead of changing a dropdown, number box or slider
the pointer passes over; a focused control still takes the wheel."""
import os

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from tabs.common import install_wheel_guard
    guard = install_wheel_guard(app)
    yield app
    app.removeEventFilter(guard)


def _wheel(app, widget, dy=-120):
    ev = QWheelEvent(QPointF(5, 5), QPointF(widget.mapToGlobal(QPoint(5, 5))), QPoint(0, 0), QPoint(0, dy),
                     Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    app.sendEvent(widget, ev)
    app.processEvents()


def _panel(app):
    area = QtWidgets.QScrollArea()
    inner = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(inner)
    combo = QtWidgets.QComboBox()
    combo.addItems(["a", "b", "c"])
    spin = QtWidgets.QSpinBox()
    spin.setValue(5)
    edit_combo = QtWidgets.QComboBox()
    edit_combo.setEditable(True)
    edit_combo.addItems(["x", "y"])
    for w in (combo, spin, edit_combo):
        lay.addWidget(w)
    lay.addSpacing(3000)                                  # tall enough to scroll
    area.setWidget(inner)
    area.resize(300, 200)
    area.show()
    area.setFocus()                                       # nothing guarded starts focused (an editable combo grabs it)
    app.processEvents()
    return area, combo, spin, edit_combo


def test_unfocused_controls_pass_the_wheel_to_the_panel(app):
    area, combo, spin, edit_combo = _panel(app)
    bar = area.verticalScrollBar()
    assert bar.value() == 0
    _wheel(app, combo)
    assert combo.currentIndex() == 0 and bar.value() > 0          # the panel scrolled, the setting did not move
    v = bar.value()
    _wheel(app, spin)
    assert spin.value() == 5 and bar.value() > v
    _wheel(app, edit_combo.lineEdit())                             # an editable combo's inner line edit too
    assert edit_combo.currentIndex() == 0
    assert combo.focusPolicy() == Qt.StrongFocus                   # a wheel pass no longer focuses it


def test_a_focused_control_still_takes_the_wheel(app):
    area, combo, spin, _e = _panel(app)
    spin.setFocus()
    app.processEvents()
    if not spin.hasFocus():
        pytest.skip("the offscreen platform gave no focus")
    _wheel(app, spin, dy=120)
    assert spin.value() == 6


def test_scroll_bars_are_never_guarded(app):
    area, *_ = _panel(app)
    bar = area.verticalScrollBar()
    _wheel(app, bar)
    assert bar.value() > 0
