"""The Auto Caption log is plain text with a block limit."""
import pytest
from PySide6.QtWidgets import QApplication, QPlainTextEdit


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def test_log_is_plain_text_and_limited(qapp):
    from tabs.caption import CaptionTab
    tab = CaptionTab.__new__(CaptionTab)          # only the log widget is needed; skip the heavy constructor
    box = QPlainTextEdit()
    box.setMaximumBlockCount(4000)
    tab.log_box = box
    CaptionTab.log(tab, '<img src="\\\\host\\share\\x.png"> hello')
    assert box.toPlainText() == '<img src="\\\\host\\share\\x.png"> hello'
    assert "<img" not in box.document().toHtml().replace("&lt;img", "")      # shown as text, never as an image


def test_log_box_built_by_tab_has_limit(qapp):
    import inspect
    from tabs import caption
    src = inspect.getsource(caption.CaptionTab)
    assert "QPlainTextEdit()" in src and "setMaximumBlockCount(4000)" in src and "appendPlainText" in src
