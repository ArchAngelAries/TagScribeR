"""The sidebar's two quiet support links (Patreon, Ko-fi): always there, never in the way.

Plain links from core/support.py. A click opens the page in the system browser; the app itself sends nothing.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from core.support import SUPPORT_LINKS
from tabs import icons, theme


def open_link(url: str) -> bool:
    """Open a support / project page in the system browser. Only https addresses are ever opened."""
    if not str(url).startswith("https://"):
        return False
    return bool(QDesktopServices.openUrl(QUrl(url)))


class SupportBar(QWidget):
    """Two small borderless buttons side by side, in the sidebar's own colours."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("support_bar")
        theme.legacy_style(self, "sidebar_container")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 6)
        lay.setSpacing(2)
        self.buttons: dict[str, QPushButton] = {}
        for sid, label, url, tip in SUPPORT_LINKS:
            b = QPushButton(label)
            b.setObjectName(f"support_{sid}")
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(f"{tip}\n{url}")
            b.setAccessibleName(f"Support TagScribeR on {label}")
            icons.set(b, sid, "nav")
            theme.set_variant(b, "ghost", legacy="support_link")
            b.clicked.connect(lambda _=False, u=url: open_link(u))
            lay.addWidget(b, 1)
            self.buttons[sid] = b
