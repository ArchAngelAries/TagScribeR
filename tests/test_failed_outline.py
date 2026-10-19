"""A failed AND selected card shows both the red error outline and the selection outline."""
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication, QStyle, QStyleOptionViewItem

from tabs.workspace.delegate import ACCENT, MISSING, CardDelegate
from tabs.workspace.model import ENTRY_ROLE


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class _Loader:
    def pixmap(self, *a):
        return None

    def is_failed(self, key):
        return False



def _render(app, width, selected, failed):
    entry = SimpleNamespace(key="k", name="demo.png", mtime=0, file_size=0, dirty=False, has_caption=True,
                            width=0, height=0, flags=None, text="a caption")
    jobs = {"k": "failed"} if failed else {}
    delegate = CardDelegate(_Loader(), job_state=jobs)
    delegate.set_card_width(width)
    size = delegate.sizeHint(None, None)
    img = QImage(size, QImage.Format_ARGB32)
    img.fill(QColor("#101010"))
    opt = QStyleOptionViewItem()
    opt.rect = QRect(0, 0, size.width(), size.height())
    opt.state = QStyle.State_Enabled | (QStyle.State_Selected if selected else QStyle.State_None)

    class Idx:
        def data(self, role=Qt.DisplayRole):
            return entry if role == ENTRY_ROLE else None

    p = QPainter(img)
    delegate.paint(p, opt, Idx())
    p.end()
    return img, size


def _near(img, color, y, x_range):
    """True when a pixel along row y within x_range is close to color."""
    for x in x_range:
        c = QColor(img.pixel(x, y))
        if abs(c.red() - color.red()) + abs(c.green() - color.green()) + abs(c.blue() - color.blue()) < 60:
            return True
    return False


@pytest.mark.parametrize("width", [120, 220, 400])
def test_four_states(app, width):
    for selected in (False, True):
        for failed in (False, True):
            img, size = _render(app, width, selected, failed)
            y = size.height() // 2
            band = range(4, 10)           # card edge (selection) and the strip just inside it (red)
            assert _near(img, ACCENT, y, band) == selected, (width, selected, failed)
            assert _near(img, MISSING, y, band) == failed, (width, selected, failed)


def test_clearing_failed_removes_red(app):
    img, size = _render(app, 220, True, True)
    assert _near(img, MISSING, size.height() // 2, range(4, 10))
    img, size = _render(app, 220, True, False)
    assert not _near(img, MISSING, size.height() // 2, range(4, 10))
    assert _near(img, ACCENT, size.height() // 2, range(4, 10))
