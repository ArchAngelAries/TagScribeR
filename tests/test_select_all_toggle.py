"""SelectAllToggle (tabs/common.py): Ctrl+A selects everything shown, Ctrl+A again deselects everything, and
Ctrl+Shift+A always deselects everything."""
import os

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _list(app, n=4):
    from tabs.common import SelectAllToggle
    w = QtWidgets.QListWidget()
    w.setViewMode(QtWidgets.QListWidget.IconMode)
    w.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
    for i in range(n):
        w.addItem(f"image {i}")
    w.resize(400, 300)
    w.show()
    app.processEvents()
    w._toggle = SelectAllToggle(w)
    return w


def _key(w, mod):
    QTest.keyClick(w, Qt.Key_A, mod)
    QtWidgets.QApplication.processEvents()


def test_ctrl_a_selects_all_then_deselects_all(app):
    w = _list(app)
    _key(w, Qt.ControlModifier)
    assert len(w.selectedItems()) == 4
    _key(w, Qt.ControlModifier)
    assert w.selectedItems() == []
    _key(w, Qt.ControlModifier)                           # and a third press selects everything again
    assert len(w.selectedItems()) == 4


def test_ctrl_a_completes_a_partial_selection(app):
    w = _list(app)
    w.item(1).setSelected(True)
    w.item(2).setSelected(True)
    _key(w, Qt.ControlModifier)                           # some selected: the first press selects the rest
    assert len(w.selectedItems()) == 4


def test_ctrl_shift_a_deselects_all(app):
    w = _list(app)
    w.item(0).setSelected(True)
    w.item(3).setSelected(True)
    _key(w, Qt.ControlModifier | Qt.ShiftModifier)
    assert w.selectedItems() == []
    _key(w, Qt.ControlModifier | Qt.ShiftModifier)        # nothing selected: stays empty
    assert w.selectedItems() == []


def test_empty_view_is_left_alone(app):
    w = _list(app, n=0)
    _key(w, Qt.ControlModifier)
    assert w.selectedItems() == []


def test_plain_a_still_reaches_the_view(app):
    w = _list(app)
    _key(w, Qt.NoModifier)                                # keyboard search, not a selection command
    assert len(w.selectedItems()) <= 1


def test_image_grid_ctrl_a_toggles(app, tmp_path, monkeypatch):
    """The real image grid: Ctrl+A twice ends with nothing selected and tells the side panels so."""
    from PIL import Image
    from tabs.workspace.browser import DatasetBrowser
    from tabs.workspace.context import WorkspaceContext
    folder = tmp_path / "demo_dataset"
    folder.mkdir()
    for i in range(3):
        Image.new("RGB", (16, 16), (i * 40, 0, 0)).save(folder / f"img_{i}.png")
    ctx = WorkspaceContext()
    assert ctx.open_folder(str(folder), False, None)
    b = DatasetBrowser(ctx)
    b.resize(600, 400)
    b.show()
    app.processEvents()
    seen = []
    b.selection_changed.connect(seen.append)
    assert b.proxy.rowCount() == 3
    _key(b.view, Qt.ControlModifier)
    assert len(b.selected_keys()) == 3
    _key(b.view, Qt.ControlModifier)
    assert b.selected_keys() == [] and seen[-1] == []
    b.toggle_select_all()
    assert len(b.selected_keys()) == 3
    b.deselect_all()
    assert b.selected_keys() == []
