"""Consistent icons (Material Design via qtawesome), with a graceful no-icon fallback.

Usage:
    icons.icon("save")                 -> QIcon
    icons.set(button, "run")           -> set a button's icon
    icons.apply_icons(main_window)     -> convert legacy emoji-prefixed labels to icon + text
"""
from __future__ import annotations

import logging
import os

os.environ.setdefault("QT_API", "pyside6")  # qtawesome (via qtpy) must bind to PySide6

from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QAbstractButton, QListWidget, QTabWidget, QWidget

log = logging.getLogger(__name__)

NAMES = {
    # navigation
    "gallery": "mdi6.view-grid-outline", "caption": "mdi6.robot-outline", "editor": "mdi6.image-edit-outline",
    "datasets": "mdi6.folder-multiple-image", "metadata": "mdi6.information-outline", "settings": "mdi6.cog-outline",
    "help": "mdi6.help-circle-outline",
    # files / editing
    "open": "mdi6.folder-open-outline", "folder": "mdi6.folder-outline", "save": "mdi6.content-save-outline",
    "undo": "mdi6.undo", "redo": "mdi6.redo", "collection": "mdi6.package-variant-closed", "refresh": "mdi6.refresh",
    "delete": "mdi6.delete-outline", "add": "mdi6.plus", "export": "mdi6.export-variant", "recent": "mdi6.history",
    "filters": "mdi6.star-outline", "download": "mdi6.download", "cloud": "mdi6.cloud-download-outline",
    "installed": "mdi6.check-circle-outline",
    # AI
    "run": "mdi6.rocket-launch-outline", "stop": "mdi6.stop-circle-outline", "review": "mdi6.clipboard-check-outline",
    "tag": "mdi6.tag-multiple-outline", "free": "mdi6.memory", "busy": "mdi6.timer-sand", "magic": "mdi6.auto-fix",
    # review / checks
    "more": "mdi6.dots-horizontal", "accept": "mdi6.check", "reject": "mdi6.close", "warning": "mdi6.alert-outline", "scan": "mdi6.magnify-scan",
    "audit": "mdi6.shield-search", "clean": "mdi6.broom", "sign": "mdi6.draw-pen",
}

# Legacy emoji label prefixes -> icon names (used by apply_icons).
EMOJI = {
    "📂": "open", "📁": "folder", "💾": "save", "📦": "collection", "🔄": "refresh", "🗑️": "delete", "🗑": "delete",
    "➕": "add", "🚀": "run", "🛑": "stop", "📝": "review", "✨": "magic", "🧹": "clean", "🔍": "scan", "✍": "sign",
    "✓": "accept", "✗": "reject", "⬇️": "download", "⬇": "download", "☁️": "cloud", "✅": "installed", "⏳": "busy",
    "❓": "help", "🖼️": "gallery", "🤖": "caption", "✏️": "editor", "ℹ️": "metadata", "⚙️": "settings", "★": "filters",
    "↶": "undo", "↷": "redo", "⚠": "warning",
}

_available: bool | None = None
_cache: dict[tuple[str, str], QIcon] = {}
DEFAULT_COLOR = "#dddddd"


def available() -> bool:
    global _available
    if _available is None:
        import importlib.util
        _available = importlib.util.find_spec("qtawesome") is not None
        if not _available:  # never break the UI over icons
            log.info("qtawesome not installed; using text-only buttons.")
    return _available


def icon(name: str, color: str = DEFAULT_COLOR) -> QIcon:
    key = (name, color)
    if key in _cache:
        return _cache[key]
    ic = QIcon()
    if available() and name in NAMES:
        try:
            import qtawesome as qta
            ic = qta.icon(NAMES[name], color=color, color_disabled="#666666")
        except Exception as e:
            log.debug("icon %s failed: %s", name, e)
    _cache[key] = ic
    return ic


def set(button: QAbstractButton, name: str, color: str = DEFAULT_COLOR) -> None:  # noqa: A001 - short, readable API
    button.setIcon(icon(name, color))
    if button.iconSize().width() < 16:
        button.setIconSize(QSize(18, 18))


def split_emoji(text: str) -> tuple[str | None, str]:
    """('save', 'Save') for '💾 Save'; (None, text) if there's no known emoji prefix."""
    stripped = text.lstrip()
    for emo in sorted(EMOJI, key=len, reverse=True):
        if stripped.startswith(emo):
            rest = stripped[len(emo):].lstrip("️").strip()
            return EMOJI[emo], rest
    return None, text


def apply_icons(root: QWidget) -> int:
    """Replace emoji label prefixes with real icons throughout ``root``. Returns how many were converted."""
    if not available():
        return 0
    n = 0
    for btn in root.findChildren(QAbstractButton):
        name, rest = split_emoji(btn.text())
        if name:
            set(btn, name, "#ffffff" if "background-color" in btn.styleSheet() else DEFAULT_COLOR)
            btn.setText(rest)
            n += 1
    for lst in root.findChildren(QListWidget):
        for i in range(lst.count()):
            item = lst.item(i)
            name, rest = split_emoji(item.text())
            if name:
                item.setIcon(icon(name))
                item.setText(rest)
                n += 1
    for tabs in root.findChildren(QTabWidget):
        for i in range(tabs.count()):
            name, rest = split_emoji(tabs.tabText(i))
            if name:
                tabs.setTabIcon(i, icon(name))
                tabs.setTabText(i, rest)
                n += 1
    return n
