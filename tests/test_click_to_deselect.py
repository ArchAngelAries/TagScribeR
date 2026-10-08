"""ClickToDeselect (tabs/common.py): a plain click on a selected image deselects it and keeps the rest selected."""
import os

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _list(app, mode):
    from tabs.common import ClickToDeselect
    w = QtWidgets.QListWidget()
    w.setViewMode(QtWidgets.QListWidget.IconMode)
    w.setSelectionMode(mode)
    for i in range(3):
        w.addItem(f"image {i}")
    w.resize(400, 300)
    w.show()
    app.processEvents()
    w._deselect = ClickToDeselect(w)
    return w


def _click(w, row, mod=Qt.NoModifier):
    QTest.mouseClick(w.viewport(), Qt.LeftButton, mod, w.visualItemRect(w.item(row)).center())
    QtWidgets.QApplication.processEvents()                # the deselect runs after Qt's own release handling


@pytest.mark.parametrize("mode", [QtWidgets.QAbstractItemView.ExtendedSelection,
                                  QtWidgets.QAbstractItemView.SingleSelection])
def test_second_click_deselects(app, mode):
    w = _list(app, mode)
    _click(w, 1)
    assert [i.text() for i in w.selectedItems()] == ["image 1"]
    _click(w, 1)
    assert w.selectedItems() == []
    _click(w, 1)                                          # and a third click selects it again
    assert [i.text() for i in w.selectedItems()] == ["image 1"]


def test_click_deselects_one_image_of_a_multi_selection(app):
    w = _list(app, QtWidgets.QAbstractItemView.ExtendedSelection)
    _click(w, 0)
    _click(w, 1, Qt.ControlModifier)
    _click(w, 2, Qt.ControlModifier)
    assert len(w.selectedItems()) == 3
    _click(w, 1)                                          # a plain click on a selected image: just it goes
    assert sorted(i.text() for i in w.selectedItems()) == ["image 0", "image 2"]
    _click(w, 0)
    assert [i.text() for i in w.selectedItems()] == ["image 2"]
    _click(w, 1)                                          # an unselected image: Qt's plain click, only it
    assert [i.text() for i in w.selectedItems()] == ["image 1"]
    QTest.mouseDClick(w.viewport(), Qt.LeftButton, Qt.NoModifier, w.visualItemRect(w.item(0)).center())
    app.processEvents()
    assert [i.text() for i in w.selectedItems()] == ["image 0"]   # a double-click opens; it never deselects
