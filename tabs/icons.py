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

from tabs import theme

log = logging.getLogger(__name__)

NAMES = {
    # navigation
    "gallery": "mdi6.view-grid-outline", "caption": "mdi6.robot-outline", "editor": "mdi6.image-edit-outline",
    "datasets": "mdi6.folder-multiple-image", "metadata": "mdi6.information-outline", "camera": "mdi6.camera-outline", "settings": "mdi6.cog-outline", "train": "mdi6.school-outline",
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
    "❓": "help", "🖼️": "gallery", "🤖": "caption", "✏️": "editor", "ℹ️": "metadata", "🎓": "train", "⚙️": "settings", "★": "filters",
    "↶": "undo", "↷": "redo", "⚠": "warning",
}

_available: bool | None = None
_cache: dict[tuple, QIcon] = {}
_records: dict[int, tuple[str, str | None]] = {}     # QIcon.cacheKey() -> (icon name, colour request)
DEFAULT_COLOR = "#dddddd"      # the icon colour under every qt_material theme
DEFAULT_COLOR_NAV = "#ffffff"  # the sidebar's Help button has always had a white icon under them
# A colour request is None (the theme's icon colour), "on_accent" (text on a filled button), "nav" (the sidebar's Help
# button) or a hex colour that never changes. theme.refresh_icons rebuilds an icon from its record, so a request that is
# not a hex follows the theme.


def available() -> bool:
    global _available
    if _available is None:
        import importlib.util
        _available = importlib.util.find_spec("qtawesome") is not None
        if not _available:  # never break the UI over icons
            log.info("qtawesome not installed; using text-only buttons.")
    return _available


def resolve_color(spec: str | None, widget=None, nav: bool = False) -> str:
    """The hex colour a colour request means in the theme in force (qt_material: the colours icons always had)."""
    if spec is None:
        return theme.color("icon_nav" if nav else "icon")
    if spec == "nav":
        return DEFAULT_COLOR_NAV if theme.is_legacy() else theme.color("icon_nav")
    if spec == "on_accent":
        filled = theme.is_legacy() or widget is None or widget.property("variant") in ("primary", "danger")
        return theme.color("text_on_accent") if filled else theme.color("icon")
    return spec


def _make(name: str, spec: str | None, color: str) -> QIcon:
    off = theme.color("text_disabled")
    # a token theme also gives the selected state (a selected sidebar row) the full-strength colour
    selected = color if theme.is_legacy() else theme.color("icon")
    key = (name, spec, color, off, selected)
    if key in _cache:
        return _cache[key]
    ic = QIcon()
    if available() and name in NAMES:
        try:
            import qtawesome as qta
            extra = {} if theme.is_legacy() else {"color_selected": selected, "color_active": color}
            ic = qta.icon(NAMES[name], color=color, color_disabled=off, **extra)
        except Exception as e:
            log.debug("icon %s failed: %s", name, e)
    _cache[key] = ic
    if not ic.isNull():
        _records[ic.cacheKey()] = (name, spec)        # the cache keeps `ic` alive, so its key is never reused
    return ic


def icon(name: str, color: str | None = None) -> QIcon:
    """The icon `name`; `color` None takes the theme's icon colour (see the colour requests above)."""
    return _make(name, color, resolve_color(color))


def lookup(ic: QIcon) -> tuple[str, str | None] | None:
    """(name, colour request) of an icon made here, or None for any other icon (a picture, a thumbnail ...)."""
    return None if ic.isNull() else _records.get(ic.cacheKey())


def rebuild(record: tuple[str, str | None], widget=None, nav: bool = False) -> QIcon:
    """The icon of `record` (from lookup) in the colours of the theme in force."""
    name, spec = record
    return _make(name, spec, resolve_color(spec, widget, nav))


def set(button: QAbstractButton, name: str, color: str | None = None) -> None:  # noqa: A001 - short, readable API
    button.setIcon(_make(name, color, resolve_color(color, button)))
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
            set(btn, name, "on_accent" if "background-color" in btn.styleSheet() else None)
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
