"""The sidebar's support links (tabs/support_bar.py): two quiet buttons that only ever open an https page."""
import os

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_two_buttons_with_tooltips_that_open_their_pages(app, monkeypatch):
    from core.support import SUPPORT_LINKS
    from tabs import support_bar
    opened = []
    monkeypatch.setattr(support_bar.QDesktopServices, "openUrl", lambda url: opened.append(url.toString()) or True)
    bar = support_bar.SupportBar()
    assert list(bar.buttons) == ["patreon", "kofi"]
    for sid, label, url, _tip in SUPPORT_LINKS:
        b = bar.buttons[sid]
        assert b.text() == label and url in b.toolTip() and "nothing is sent" in b.toolTip()
        assert not b.icon().isNull() and b.property("variant") == "ghost"
        b.click()
    assert opened == [url for _sid, _label, url, _tip in SUPPORT_LINKS]


def test_only_https_pages_are_ever_opened(app, monkeypatch):
    from tabs import support_bar
    opened = []
    monkeypatch.setattr(support_bar.QDesktopServices, "openUrl", lambda url: opened.append(url) or True)
    for bad in ("file:///C:/Windows/System32/calc.exe", "http://example.invalid", "javascript:alert(1)", ""):
        assert support_bar.open_link(bad) is False
    assert opened == []
    assert support_bar.open_link("https://example.invalid/page") is True and len(opened) == 1
