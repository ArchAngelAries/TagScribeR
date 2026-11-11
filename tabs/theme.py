"""Design tokens and the one generated stylesheet (docs/dev/UI_UX_RESEARCH.md section 6).

Everything visual that is not a widget's own painting comes from here: a neutral elevation stack, one accent, the
four semantic colours, a type scale, the 4 px spacing scale and the radii. `build_qss(tokens)` turns a token set
into the application stylesheet; widgets pick a look by a dynamic property instead of an inline stylesheet:

    theme.set_variant(button, "primary")      # filled accent: the one main action of a panel
    theme.set_variant(button, "danger")       # filled red: Stop, Abort, Delete
    theme.set_variant(button, "ghost")        # borderless: quiet actions in toolbars
    theme.set_role(label, "warning")          # text in a semantic colour
    theme.color("danger")                     # a token for QPainter code (charts, bars, item foregrounds)

Two token themes ship ("tagscriber_dark", "tagscriber_light"). The qt_material themes stay selectable. Under them
nothing here styles a widget: every widget that carried an inline style before the token themes gets exactly that
style back (LEGACY_QSS), and QPainter code gets the colours it had (LEGACY_COLORS), so the default look does not
change until the owner picks a token theme in Settings. Colours change at once when the theme changes; a few icons
and sizes made by code that only runs at start-up settle after a restart.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

# -- scales -----------------------------------------------------------------------------------------------------------
SPACE = (4, 8, 12, 16, 24, 32)                  # the 4 px scale, in px
RADIUS = {"input": 4, "card": 8, "pill": 999}   # px
CONTROL_HEIGHT = 28                             # px: comfortable 32, compact 24 are later density options
FONT_PT = {"hint": 9, "body": 10, "label": 10, "section": 11, "panel": 13, "big": 20}
VARIANTS = ("primary", "secondary", "danger", "ghost")
ROLES = ("secondary", "success", "warning", "danger", "info")


@dataclass(frozen=True)
class Tokens:
    name: str
    label: str
    dark: bool
    bg_app: str             # window background, behind panels
    bg_surface: str         # panels, sidebar
    bg_raised: str          # cards, inputs, popovers
    bg_hover: str           # hover rows, secondary buttons
    bg_image: str           # behind images and previews, neutral
    border_subtle: str      # dividers
    border_strong: str      # input borders
    text_primary: str
    text_secondary: str     # labels, hints
    text_disabled: str
    text_on_accent: str
    accent: str             # primary buttons, selection, focus
    accent_hover: str
    accent_pressed: str
    success: str
    warning: str
    danger: str
    info: str
    # Segoe UI first: the static family is the one Qt is known to render well on Windows; the variable one is untried
    font_family: str = '"Segoe UI", "Segoe UI Variable", "Noto Sans", "DejaVu Sans", sans-serif'
    mono_family: str = '"Cascadia Mono", "Consolas", "DejaVu Sans Mono", monospace'
    # the focus ring: 3:1 against the control borders AND the panels, which the accent itself cannot be in both themes
    focus_ring: str = ""
    extra: dict = field(default_factory=dict)   # painter colours that are not a token (the legacy set's exact values)

    @property
    def accent_subtle(self) -> str:
        """The accent at 14 % over the surface: selected rows and cards."""
        return blend(self.accent, self.bg_surface, 0.14)

    @property
    def focus(self) -> str:
        return self.focus_ring or self.accent

    def color(self, key: str) -> str:
        if key == "accent_subtle":
            return self.accent_subtle
        if key == "focus":
            return self.focus
        if key in self.extra:
            return self.extra[key]
        return getattr(self, PAINT_KEYS.get(key, key))

    def colors(self) -> dict[str, str]:
        keys = [k for k, t in self.__dataclass_fields__.items() if t.type == "str" and k not in ("name", "label", "focus_ring")
                and not k.endswith("_family")]
        out = {k: getattr(self, k) for k in keys}
        out["accent_subtle"] = self.accent_subtle
        out["focus"] = self.focus
        return out


# Colours of painted things (charts, list item text, the help page) by use, and the token each one takes in a token
# theme. The legacy set below gives each the exact value it had as a literal in the code.
PAINT_KEYS = {
    "chart_bg": "bg_raised", "chart_frame": "border_strong", "chart_grid": "border_subtle",
    "chart_text": "text_secondary", "chart_muted": "text_secondary", "chart_faint": "text_secondary",
    "chart_lr": "text_secondary", "chart_raw": "accent", "chart_smooth": "info", "chart_avg": "accent",
    "chart_mark": "warning", "chart_cursor": "text_primary",
    "bar_track": "bg_app", "bar_text": "text_primary", "bar_text_on_fill": "bg_app", "bar_muted": "text_secondary",
    "level_high": "danger", "level_medium": "warning", "level_low": "text_secondary", "text_readonly": "text_secondary",
    "diff_add": "success", "diff_del": "danger", "diff_same": "text_primary", "diff_empty": "text_disabled",
    "help_h2": "accent", "help_h3": "text_primary", "help_code": "text_primary", "help_code_bg": "bg_hover",
    "list_header": "text_secondary", "icon": "text_primary", "icon_nav": "text_secondary",
}

DARK = Tokens(
    name="tagscriber_dark", label="TagScribeR dark", dark=True,
    bg_app="#16181b", bg_surface="#1d2024", bg_raised="#252a2f", bg_hover="#2d333a", bg_image="#1a1a1a",
    # border_strong is 3:1 against the panels (control borders), the focus ring 3:1 against it
    border_subtle="#2f353c", border_strong="#6b747e", focus_ring="#5eead4",
    # white on this accent is 2.7:1, so the dark theme writes dark text on accent and danger fills
    text_primary="#e6e8eb", text_secondary="#a7aeb6", text_disabled="#6b727a", text_on_accent="#0e1513",
    accent="#19b394", accent_hover="#2cc9a8", accent_pressed="#12957b",
    success="#3fb950", warning="#e3b341", danger="#f7766f", info="#58a6ff",   # 4.5:1 on every surface and row tint
)

LIGHT = Tokens(
    name="tagscriber_light", label="TagScribeR light", dark=False,
    bg_app="#eef0f3", bg_surface="#f8f9fb", bg_raised="#ffffff", bg_hover="#e4e8ec", bg_image="#d9dcdf",
    border_subtle="#dde1e6", border_strong="#88919b", focus_ring="#06483a",
    text_primary="#1b1f24", text_secondary="#525b65", text_disabled="#8d949c", text_on_accent="#ffffff",
    # white text needs 4.5:1 on the accent and on its hover step, so the light theme's accent steps go darker
    accent="#0b7b65", accent_hover="#0a6f5b", accent_pressed="#085e4d",
    success="#17712f", warning="#8a5a00", danger="#bd1e2a", info="#0860c8",   # 4.5:1 on every surface and row tint
)

# -- the look before the token themes ---------------------------------------------------------------------------------
# What QPainter code and item colours used as literals. These are the same under every qt_material theme, light ones
# included, because they were hard-coded: a qt_material theme has to look exactly as it did.
LEGACY_COLORS = {
    "chart_bg": "#1e1e1e", "chart_frame": "#555555", "chart_grid": "#333333", "chart_text": "#aaaaaa",
    "chart_muted": "#888888", "chart_faint": "#777777", "chart_lr": "#e17055", "chart_raw": "#00b894",
    "chart_smooth": "#81ecec", "chart_avg": "#00b894", "chart_mark": "#fdcb6e", "chart_cursor": "#dfe6e9",
    "bar_track": "#15171c", "bar_text": "#ffffff", "bar_text_on_fill": "#ffffff", "bar_muted": "#8b93a1",
    "level_high": "#ff7675", "level_medium": "#fdcb6e", "level_low": "#9a9a9a", "text_readonly": "#9a9a9a",
    "diff_add": "#55efc4", "diff_del": "#ff7675", "diff_same": "#cfcfcf", "diff_empty": "#777777",
    "help_h2": "#00b894", "help_h3": "#d0d0d0", "help_code": "#fdcb6e", "help_code_bg": "#2b2b2b",
    "list_header": "#a0a0a4", "icon": "#dddddd", "icon_nav": "#dddddd",
}
LEGACY = replace(DARK, name="legacy", label="qt_material", bg_app="#1e1e1e", bg_surface="#232323",
                 bg_raised="#2b2b2b", bg_hover="#333333", border_subtle="#2c2c2c", border_strong="#3c3c3c",
                 text_primary="#dddddd", text_secondary="#8a8a8a", text_disabled="#666666", text_on_accent="#ffffff",
                 accent="#00b894", accent_hover="#1dc9a5", accent_pressed="#00a07f",
                 success="#00b894", warning="#fdcb6e", danger="#d63031", info="#81ecec", focus_ring="",
                 extra=LEGACY_COLORS)

# The inline style each styled widget had. `legacy_style(widget, key)` (or the `legacy=` argument of set_variant,
# set_role and set_surface) puts it on the widget while a qt_material theme is in force and takes it off under a
# token theme, where the object name, variant or role of the same widget does the work.
_ACCENT_BTN = "background-color: #00b894; color: white; font-weight: bold;"
LEGACY_QSS = {
    "hint": "color: #8a8a8a; font-size: 11px;",
    "status": "color: #aaa;",
    "warn": "color: #e17055;",
    "section_toggle": "QToolButton { border: none; font-weight: bold; color: #bbb; }",
    "help_dot": "QToolButton { border-radius: 13px; background: #3a3f44; color: #ddd; font-weight: bold;"
                " padding: 0px; margin: 0px; min-width: 0px; min-height: 0px; }"
                "QToolButton:hover { background: #00b894; color: white; }",
    "help_view": "QTextBrowser { color: #dcdcdc; background-color: #1e1e1e; padding: 10px; }",
    "palette": "QDialog { background-color: #26292c; border: 1px solid #00b894; border-radius: 8px; }",
    "palette_hint": "color: #888; font-size: 11px;",
    "sidebar_container": "background-color: #232323;",
    "sidebar": "QListWidget { border: none; background-color: #232323; font-size: 15px; }"
               "QListWidget::item { padding: 15px; border-bottom: 1px solid #2c2c2c; }"
               "QListWidget::item:selected { background-color: #00b894; color: white; }",
    "sidebar_help": "QPushButton { background-color: #2d3436; color: #aaa; border: none; padding: 15px;"
                    " text-align: left; }"
                    "QPushButton:hover { background-color: #333; color: white; }",
    # the two small support links under Help (new with the buttons; quieter than Help itself)
    "support_link": "QPushButton { background-color: #232323; color: #8a8a8a; border: none; padding: 5px 6px;"
                    " font-size: 11px; }"
                    "QPushButton:hover { background-color: #333; color: white; }",
    "preview": "background-color: #1a1a1a; border-radius: 6px; color: #777;",
    "preview_plain": "background-color: #1a1a1a; border-radius: 6px;",
    "text_current": "QTextBrowser { color: #bbb; background-color: #1e1e1e; }",
    "text_changes": "QTextBrowser { background-color: #1e1e1e; }",
    "accent": f"{_ACCENT_BTN} padding: 6px;",
    "accent_wide": f"{_ACCENT_BTN} padding: 8px;",
    "purple": "background-color: #6c5ce7; color: white; font-weight: bold; padding: 6px;",
    "purple_wide": "background-color: #6c5ce7; color: white; font-weight: bold; padding: 8px;",
    "blue": "background-color: #0984e3; color: white; font-weight: bold; padding: 6px;",
    "teal": "background-color: #00897b; color: white; font-weight: bold; padding: 6px;",
    "run": "background-color: #d63031; font-weight: bold; font-size: 14px;",
    "abort": "background-color: #ff4757; font-weight: bold; font-size: 14px;",
}
# What tabs.settings appends to a qt_material stylesheet (it always did).
LEGACY_APP_QSS = """
    QStackedWidget { background-color: #1e1e1e; }
    QScrollBar:vertical { width: 12px; }
    QLineEdit, QTextEdit, QPlainTextEdit { border-radius: 4px; }
"""

THEMES: dict[str, Tokens] = {DARK.name: DARK, LIGHT.name: LIGHT}
DEFAULT_MATERIAL = "dark_teal.xml"

_current: Tokens = LEGACY
_saved_palette: Any = None
_notifier: Any = None
_pending_qss: str | None = None      # a stylesheet apply(..., defer_stylesheet=True) is holding back


# -- colour helpers ---------------------------------------------------------------------------------------------------
def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex(rgb: tuple[float, float, float]) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, round(c))) for c in rgb)


def blend(fg: str, bg: str, alpha: float) -> str:
    """`fg` at `alpha` over `bg`, as a hex colour."""
    f, b = _rgb(fg), _rgb(bg)
    return _hex(tuple(f[i] * alpha + b[i] * (1 - alpha) for i in range(3)))


def luminance(hex_color: str) -> float:
    def ch(c):
        c /= 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = _rgb(hex_color)
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio between two hex colours (1 to 21)."""
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def rgba(hex_color: str, alpha: float) -> str:
    r, g, b = _rgb(hex_color)
    return f"rgba({r}, {g}, {b}, {alpha:.2f})"


# -- current theme ---------------------------------------------------------------------------------------------------
def is_token_theme(name: str | None) -> bool:
    return bool(name) and name in THEMES


def tokens() -> Tokens:
    """The token set in force (a token theme's, or the overlay of the qt_material theme in use)."""
    return _current


def color(key: str) -> str:
    """A colour of the current token set, for QPainter code: theme.color("danger")."""
    return _current.color(key)


def qcolor(key: str, alpha: int = 255):
    from PySide6.QtGui import QColor
    c = QColor(color(key))
    if alpha != 255:
        c.setAlpha(alpha)
    return c


def level_color(level: str) -> str:
    """Severity ("high", "medium", "low") to a semantic colour."""
    return color(f"level_{level}" if level in ("high", "medium") else "level_low")


def is_legacy() -> bool:
    """True while a qt_material theme is in force (also before any theme was applied)."""
    return _current is LEGACY


def notifier():
    """An object whose `changed` signal fires after every theme change, for views that hold colours they rendered
    (HTML, list item colours): theme.notifier().changed.connect(self._retheme)."""
    global _notifier
    if _notifier is None:
        from PySide6.QtCore import QObject, Signal

        class _Notifier(QObject):
            changed = Signal()
        _notifier = _Notifier()
    return _notifier


# -- widgets -----------------------------------------------------------------------------------------------------------
def _repolish(widget) -> None:
    if is_legacy():                 # no rule reads the properties under a qt_material theme: leave the widget alone
        return
    st = widget.style()
    st.unpolish(widget)
    st.polish(widget)
    widget.update()


def legacy_style(widget, key: str) -> None:
    """Give `widget` the inline style LEGACY_QSS[key] whenever a qt_material theme is in force (see LEGACY_QSS)."""
    if key not in LEGACY_QSS:
        raise ValueError(f"unknown legacy style {key!r}")
    widget.setProperty("legacy_style", key)
    widget.setStyleSheet(LEGACY_QSS[key] if is_legacy() else "")


def set_variant(button, variant: str, legacy: str | None = None) -> None:
    """primary (filled accent) · secondary (outlined, the default) · danger (filled red) · ghost (borderless).
    A variant shows under the token themes only; `legacy` names the inline style the button had before, if any."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown button variant {variant!r}")
    if legacy is not None and button.property("legacy_style") != legacy:
        legacy_style(button, legacy)
    if button.property("variant") == variant:
        return
    button.setProperty("variant", variant)
    _repolish(button)


def set_role(widget, role: str | None, legacy: str | None = None) -> None:
    """A text colour by meaning: secondary, success, warning, danger, info (None = the body colour)."""
    if role is not None and role not in ROLES:
        raise ValueError(f"unknown text role {role!r}")
    if legacy is not None and widget.property("legacy_style") != legacy:
        legacy_style(widget, legacy)
    if widget.property("role") == role:
        return
    widget.setProperty("role", role)
    _repolish(widget)


def set_surface(widget, surface: str, legacy: str | None = None) -> None:
    """A background by use: "image" (neutral surround for previews), "raised" (cards, read-only text)."""
    if legacy is not None:
        legacy_style(widget, legacy)
    widget.setProperty("surface", surface)
    _repolish(widget)


def refresh_icons(root=None) -> None:
    """Rebuild the icons that tabs.icons made (and only those) in the colours of the theme in force. An icon is found
    by the record tabs.icons keeps of every icon it creates (its name and colour request), so a picture thumbnail or
    any other icon is never touched, and going back to a qt_material theme rebuilds the original vector icon. Does
    nothing while the app has only ever shown a qt_material theme, so that look stays as it was."""
    if _saved_palette is None:
        return
    from PySide6.QtWidgets import QAbstractButton, QApplication, QComboBox, QListWidget, QTabBar, QWidget
    from tabs import icons
    widgets = [root] + root.findChildren(QWidget) if root is not None else QApplication.allWidgets()
    for w in widgets:
        if isinstance(w, QAbstractButton):
            rec = icons.lookup(w.icon())
            if rec:
                w.setIcon(icons.rebuild(rec, w))
        elif isinstance(w, QListWidget):
            side = w.objectName() == "sidebar"
            for i in range(w.count()):
                item = w.item(i)
                rec = icons.lookup(item.icon())
                if rec:
                    item.setIcon(icons.rebuild(rec, None, nav=side))
        elif isinstance(w, QTabBar):
            for i in range(w.count()):
                rec = icons.lookup(w.tabIcon(i))
                if rec:
                    w.setTabIcon(i, icons.rebuild(rec))
        elif isinstance(w, QComboBox):
            for i in range(w.count()):
                rec = icons.lookup(w.itemIcon(i))
                if rec:
                    w.setItemIcon(i, icons.rebuild(rec))


def _after_change(app) -> None:
    """Put the legacy inline styles on (or take them off) every tagged widget, recolour icons, tell the views."""
    legacy = is_legacy()
    for w in app.allWidgets():
        key = w.property("legacy_style")
        if key:
            w.setStyleSheet(LEGACY_QSS[key] if legacy else "")
    refresh_icons()
    notifier().changed.emit()


# -- stylesheet --------------------------------------------------------------------------------------------------------
def _indicator_images(t: Tokens) -> dict[str, str]:
    """SVG files for the check mark and the combo chevron in the theme's colours, written once under
    user_data/cache/theme (QSS can only reference image files). Missing files simply leave the indicators as filled
    shapes."""
    try:
        from core import paths
        folder = paths.USER_DATA / "cache" / "theme"
        folder.mkdir(parents=True, exist_ok=True)
        files = {
            "check": ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="M3.5 8.5l3 3 6-7" '
                      'fill="none" stroke="{stroke}" stroke-width="2.2" stroke-linecap="round" '
                      'stroke-linejoin="round"/></svg>', t.text_on_accent),
            "chevron": ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="M4 6.5l4 4 4-4" '
                        'fill="none" stroke="{stroke}" stroke-width="1.8" stroke-linecap="round" '
                        'stroke-linejoin="round"/></svg>', t.text_primary),
        }
        out = {}
        for name, (svg, stroke) in files.items():
            p = folder / f"{name}_{t.name}.svg"
            body = svg.replace("{stroke}", stroke)
            if not p.is_file() or p.read_text(encoding="utf-8") != body:
                p.write_text(body, encoding="utf-8")
            out[name] = p.as_posix()
        return out
    except OSError:
        return {}


def _qss_url(path: str) -> str:
    """A file path for `url("...")` in a stylesheet: forward slashes, with the characters that end a quoted string
    escaped (an unquoted url() stops at an apostrophe or a space and drops the rest of the stylesheet)."""
    return path.replace("\\", "/").replace('"', '\\"')


def build_qss(t: Tokens) -> str:
    """The whole application stylesheet for a token theme."""
    img = _indicator_images(t)
    check = f'image: url("{_qss_url(img["check"])}");' if "check" in img else ""
    chevron = f'image: url("{_qss_url(img["chevron"])}");' if "chevron" in img else ""
    s4, s8, s12, s16 = SPACE[0], SPACE[1], SPACE[2], SPACE[3]
    r_in, r_card = RADIUS["input"], RADIUS["card"]
    disabled_bg = blend(t.bg_raised, t.bg_surface, 0.5)
    sel_text = t.text_primary
    return f"""
/* ---- base ---- */
QWidget {{
    background-color: {t.bg_surface};
    color: {t.text_primary};
    font-family: {t.font_family};
    font-size: {FONT_PT['body']}pt;
    selection-background-color: {t.accent};
    selection-color: {t.text_on_accent};
}}
QMainWindow, QDialog {{ background-color: {t.bg_app}; }}
QStackedWidget {{ background-color: {t.bg_app}; }}
QWidget:disabled {{ color: {t.text_disabled}; }}
QToolTip {{
    background-color: {t.bg_raised}; color: {t.text_primary};
    border: 1px solid {t.border_strong}; border-radius: {r_in}px; padding: {s4}px {s8}px;
}}
QLabel {{ background: transparent; }}
QLabel#hint {{ color: {t.text_secondary}; font-size: {FONT_PT['hint']}pt; }}
QLabel[role="secondary"], QLabel#hint[role="secondary"], QTextBrowser[role="secondary"] {{ color: {t.text_secondary}; }}
QLabel[role="success"], QLabel#hint[role="success"] {{ color: {t.success}; }}
QLabel[role="warning"], QLabel#hint[role="warning"] {{ color: {t.warning}; }}
QLabel[role="danger"], QLabel#hint[role="danger"] {{ color: {t.danger}; }}
QLabel[role="info"], QLabel#hint[role="info"] {{ color: {t.info}; }}
QTextBrowser#help_view {{ padding: {s8 + 2}px; }}
QLabel[surface="image"] {{
    background-color: {t.bg_image}; border-radius: {r_card}px; color: {t.text_secondary};
}}
QFrame[frameShape="4"], QFrame[frameShape="5"] {{ color: {t.border_subtle}; }}

/* ---- panels ---- */
QGroupBox {{
    background-color: {t.bg_surface};
    border: 1px solid {t.border_subtle};
    border-radius: {r_card}px;
    margin-top: {s12 + s4}px;
    padding: {s8}px {s8}px {s8}px {s8}px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left;
    left: {s12}px; padding: 0 {s4}px; color: {t.text_primary};
    font-size: {FONT_PT['section']}pt; font-weight: 600;
}}
QGroupBox::indicator {{ width: 14px; height: 14px; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QSplitter::handle {{ background-color: {t.border_subtle}; }}
QSplitter::handle:horizontal {{ width: 3px; }}
QSplitter::handle:vertical {{ height: 3px; }}
QSplitter::handle:hover {{ background-color: {t.accent}; }}
QToolButton#section_toggle {{
    border: none; background: transparent; color: {t.text_secondary};
    font-size: {FONT_PT['section']}pt; font-weight: 600; padding: {s4}px 0; text-align: left;
}}
QToolButton#section_toggle:hover {{ color: {t.text_primary}; }}
QToolButton#section_toggle:checked {{ color: {t.text_primary}; }}

/* ---- buttons ---- */
QPushButton {{
    background-color: {t.bg_raised};
    color: {t.text_primary};
    border: 1px solid {t.border_strong};
    border-radius: {r_in}px;
    padding: {s4 + 1}px {s12}px;
}}
QPushButton:hover {{ background-color: {t.bg_hover}; border-color: {t.border_strong}; }}
QPushButton:pressed {{ background-color: {t.border_subtle}; }}
QPushButton:focus {{ border: 2px solid {t.focus}; padding: {s4}px {s12 - 1}px; }}
QPushButton:checked {{ background-color: {t.accent_subtle}; border-color: {t.accent}; color: {t.text_primary}; }}
QPushButton:disabled {{ background-color: {disabled_bg}; color: {t.text_disabled}; border-color: {t.border_subtle}; }}
QPushButton:flat {{ border: none; background: transparent; }}
QPushButton:flat:hover {{ background-color: {t.bg_hover}; }}
QPushButton:default {{ border: 1px solid {t.accent}; }}
QPushButton[variant="primary"] {{
    background-color: {t.accent}; color: {t.text_on_accent}; border: 1px solid {t.accent}; font-weight: 600;
    padding: {s4 + 2}px {s16}px;
}}
QPushButton[variant="primary"]:hover {{ background-color: {t.accent_hover}; border-color: {t.accent_hover}; }}
QPushButton[variant="primary"]:pressed {{ background-color: {t.accent_pressed}; border-color: {t.accent_pressed}; }}
QPushButton[variant="primary"]:focus {{ border: 2px solid {t.text_on_accent}; padding: {s4 + 1}px {s16 - 1}px; }}
QPushButton[variant="primary"]:disabled {{
    background-color: {blend(t.accent, t.bg_surface, 0.35)}; color: {blend(t.text_on_accent, t.bg_surface, 0.6)};
    border-color: transparent;
}}
QPushButton[variant="danger"] {{
    background-color: {t.danger}; color: {t.text_on_accent}; border: 1px solid {t.danger}; font-weight: 600;
    padding: {s4 + 2}px {s16}px;
}}
QPushButton[variant="danger"]:hover {{ background-color: {blend(t.danger, '#ffffff', 0.85)}; }}
QPushButton[variant="danger"]:pressed {{ background-color: {blend(t.danger, '#000000', 0.85)}; }}
QPushButton[variant="danger"]:disabled {{
    background-color: {blend(t.danger, t.bg_surface, 0.35)}; color: {blend(t.text_on_accent, t.bg_surface, 0.6)};
    border-color: transparent;
}}
QPushButton[variant="ghost"] {{ background: transparent; border: 1px solid transparent; color: {t.text_secondary}; }}
QPushButton[variant="ghost"]:hover {{ background-color: {t.bg_hover}; color: {t.text_primary}; }}
QPushButton[variant="ghost"]:disabled {{ color: {t.text_disabled}; background: transparent; }}
QPushButton[variant="danger"]:focus {{ border: 2px solid {t.text_on_accent}; padding: {s4 + 1}px {s16 - 1}px; }}
QPushButton[btnSize="large"] {{ font-size: {FONT_PT['section']}pt; }}
QToolButton {{
    background: transparent; border: 1px solid transparent; border-radius: {r_in}px; padding: 3px;
    color: {t.text_secondary};
}}
QToolButton:hover {{ background-color: {t.bg_hover}; color: {t.text_primary}; }}
QToolButton:focus {{ border: 2px solid {t.focus}; padding: 2px; }}
QToolButton:pressed, QToolButton:checked {{ background-color: {t.accent_subtle}; border-color: {t.accent}; }}
QToolButton::menu-indicator {{ image: none; width: 0; }}
QToolButton#help_dot {{
    border-radius: 13px; background-color: {t.bg_raised}; color: {t.text_secondary}; border: 1px solid {t.border_subtle};
    padding: 0; margin: 0; min-width: 0; min-height: 0;
}}
QToolButton#help_dot:hover {{ background-color: {t.accent}; color: {t.text_on_accent}; border-color: {t.accent}; }}
QToolButton#help_dot:focus {{ border: 2px solid {t.focus}; padding: 0; }}
QToolButton#section_toggle:focus {{ border: 2px solid {t.focus}; padding: {s4 - 1}px 0; }}

/* ---- inputs ---- */
QLineEdit, QTextEdit, QPlainTextEdit, QTextBrowser, QSpinBox, QDoubleSpinBox, QDateEdit, QTimeEdit {{
    background-color: {t.bg_raised};
    color: {t.text_primary};
    border: 1px solid {t.border_strong};
    border-radius: {r_in}px;
    padding: 3px {s8}px;
    selection-background-color: {t.accent};
    selection-color: {t.text_on_accent};
    placeholder-text-color: {t.text_secondary};
}}
QLineEdit, QSpinBox, QDoubleSpinBox {{ min-height: {CONTROL_HEIGHT - 8}px; }}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QTextBrowser:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QDateEdit:focus, QTimeEdit:focus {{
    border: 2px solid {t.focus}; padding: 2px {s8 - 1}px;
}}
QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background-color: {disabled_bg}; color: {t.text_disabled}; border-color: {t.border_subtle};
}}
QLineEdit:read-only, QTextEdit[readOnly="true"], QPlainTextEdit[readOnly="true"] {{ border-color: {t.border_subtle}; }}
QLineEdit[error="true"] {{ border: 1px solid {t.danger}; }}
QLineEdit[error="true"]:focus {{ border: 2px solid {t.danger}; }}
QTextEdit[mono="true"], QPlainTextEdit[mono="true"] {{ font-family: {t.mono_family}; }}
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {{
    width: 16px; background: transparent; border: none;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow, QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    width: 8px; height: 8px;
}}
QComboBox {{
    background-color: {t.bg_raised};
    color: {t.text_primary};
    border: 1px solid {t.border_strong};
    border-radius: {r_in}px;
    padding: 3px {s8}px;
    min-height: {CONTROL_HEIGHT - 8}px;
}}
QComboBox:hover {{ border-color: {t.text_secondary}; }}
QComboBox:focus, QComboBox:on {{ border: 2px solid {t.focus}; padding: 2px {s8 - 1}px; }}
QComboBox:disabled {{ background-color: {disabled_bg}; color: {t.text_disabled}; border-color: {t.border_subtle}; }}
QComboBox::drop-down {{ border: none; width: 22px; subcontrol-origin: padding; subcontrol-position: center right; }}
QComboBox::down-arrow {{ {chevron} width: 14px; height: 14px; }}
QComboBox QAbstractItemView {{
    background-color: {t.bg_raised}; color: {t.text_primary};
    border: 1px solid {t.border_strong}; border-radius: {r_in}px;
    selection-background-color: {t.accent_subtle}; selection-color: {sel_text};
    padding: {s4}px;
}}
QComboBox QAbstractItemView::item {{ min-height: {CONTROL_HEIGHT - 6}px; padding: 2px {s8}px; }}
QCheckBox, QRadioButton {{ spacing: {s8}px; background: transparent; }}
QCheckBox::indicator, QRadioButton::indicator, QGroupBox::indicator {{
    width: 16px; height: 16px; border: 1px solid {t.border_strong}; background-color: {t.bg_raised};
}}
QCheckBox::indicator, QGroupBox::indicator {{ border-radius: 3px; }}
QRadioButton::indicator {{ border-radius: 9px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {t.accent}; }}
QCheckBox::indicator:focus, QRadioButton::indicator:focus, QGroupBox::indicator:focus {{
    border: 2px solid {t.focus};
}}
QCheckBox::indicator:checked, QGroupBox::indicator:checked {{
    background-color: {t.accent}; border-color: {t.accent}; {check}
}}
QCheckBox::indicator:indeterminate {{ background-color: {t.accent_subtle}; border-color: {t.accent}; }}
QRadioButton::indicator:checked {{
    width: 8px; height: 8px; background-color: {t.text_on_accent}; border: 5px solid {t.accent};
}}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    border-color: {t.border_subtle}; background-color: {disabled_bg};
}}
QCheckBox::indicator:checked:disabled, QRadioButton::indicator:checked:disabled {{
    background-color: {blend(t.accent, t.bg_surface, 0.4)}; border-color: {blend(t.accent, t.bg_surface, 0.4)};
}}
QSlider::groove:horizontal {{ height: 4px; background: {t.border_strong}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {t.accent}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; background: {t.text_primary};
    border: 1px solid {t.border_strong};
}}
QSlider::handle:horizontal:hover {{ background: {t.accent_hover}; }}
QSlider::handle:horizontal:focus {{ border: 2px solid {t.focus}; }}
QSlider::handle:vertical:focus {{ border: 2px solid {t.focus}; }}
QSlider::groove:vertical {{ width: 4px; background: {t.border_strong}; border-radius: 2px; }}
QSlider::handle:vertical {{
    width: 14px; height: 14px; margin: 0 -5px; border-radius: 7px; background: {t.text_primary};
}}
QProgressBar {{
    background-color: {t.bg_raised}; border: 1px solid {t.border_subtle}; border-radius: {r_in}px;
    text-align: center; color: {t.text_primary}; min-height: 16px;
}}
QProgressBar::chunk {{ background-color: {t.accent}; border-radius: 3px; }}

/* ---- lists, trees, tables ---- */
QListView, QTreeView, QTableView, QListWidget, QTreeWidget, QTableWidget {{
    background-color: {t.bg_raised}; alternate-background-color: {t.bg_surface};
    border: 1px solid {t.border_subtle}; border-radius: {r_in}px; outline: none;    /* the view's border shows focus */
    selection-background-color: {t.accent_subtle}; selection-color: {sel_text};
}}
QListView:focus, QTreeView:focus, QTableView:focus, QListWidget:focus, QTreeWidget:focus, QTableWidget:focus {{
    border: 2px solid {t.focus};
}}
QListView::item, QTreeView::item, QListWidget::item, QTreeWidget::item {{ padding: 3px {s4}px; }}
QListView::item:hover, QTreeView::item:hover, QTableView::item:hover {{ background-color: {t.bg_hover}; }}
QListView::item:selected, QTreeView::item:selected, QTableView::item:selected {{
    background-color: {t.accent_subtle}; color: {sel_text};
}}
QTreeView::branch {{ background: transparent; }}
QHeaderView::section {{
    background-color: {t.bg_surface}; color: {t.text_secondary}; border: none;
    border-bottom: 1px solid {t.border_subtle}; border-right: 1px solid {t.border_subtle};
    padding: {s4}px {s8}px; font-weight: 600;
}}
QTableView QTableCornerButton::section {{ background-color: {t.bg_surface}; border: none; }}
QTableView {{ gridline-color: {t.border_subtle}; }}

/* ---- tabs ---- */
QTabWidget::pane {{
    border: 1px solid {t.border_subtle}; border-radius: {r_card}px; top: -1px; background-color: {t.bg_surface};
}}
QTabBar {{ background: transparent; }}
QTabBar::tab {{
    background: transparent; color: {t.text_secondary}; border: none;
    border-bottom: 2px solid transparent; padding: {s8}px {s12}px; margin-right: {s4}px;
}}
QTabBar::tab:hover {{ color: {t.text_primary}; }}
QTabBar::tab:selected {{ color: {t.text_primary}; border-bottom: 2px solid {t.accent}; }}
QTabBar::tab:disabled {{ color: {t.text_disabled}; }}
QTabBar::tab:selected:focus {{ border: 2px solid {t.focus}; padding: {s8 - 2}px {s12 - 2}px {s8}px {s12 - 2}px; }}

/* ---- scroll bars ---- */
QScrollBar:vertical {{ background: transparent; width: 12px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 0; }}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {t.border_strong}; border-radius: 4px; margin: 2px;
}}
QScrollBar::handle:vertical {{ min-height: 24px; }}
QScrollBar::handle:horizontal {{ min-width: 24px; }}
QScrollBar::handle:hover {{ background: {t.text_secondary}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; background: none; border: none; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

/* ---- menus ---- */
QMenu {{
    background-color: {t.bg_raised}; color: {t.text_primary};
    border: 1px solid {t.border_strong}; border-radius: {r_in}px; padding: {s4}px;
}}
QMenu::item {{ padding: {s4 + 1}px {s16 + s8}px {s4 + 1}px {s12}px; border-radius: 3px; }}
QMenu::item:selected {{ background-color: {t.accent_subtle}; color: {t.text_primary}; }}
QMenu::item:disabled {{ color: {t.text_disabled}; }}
QMenu::separator {{ height: 1px; background: {t.border_subtle}; margin: {s4}px {s8}px; }}
QMenuBar {{ background-color: {t.bg_surface}; }}
QMenuBar::item:selected {{ background-color: {t.bg_hover}; }}
QStatusBar {{ background-color: {t.bg_surface}; color: {t.text_secondary}; }}
QDialogButtonBox QPushButton {{ min-width: 72px; }}
QMessageBox {{ background-color: {t.bg_surface}; }}
QMessageBox QLabel {{ color: {t.text_primary}; }}

/* ---- the window chrome ---- */
{chrome_qss(t)}
"""


def chrome_qss(t: Tokens) -> str:
    """The sidebar, its Help button, the command palette: the selected row is an accent bar plus a tint (research
    Q3), not the solid block of the qt_material look."""
    s8, s12, s16 = SPACE[1], SPACE[2], SPACE[3]
    return f"""
QWidget#sidebar_container {{ background-color: {t.bg_surface}; }}
QListWidget#sidebar {{
    border: none; background-color: {t.bg_surface}; color: {t.text_secondary};
    font-size: {FONT_PT['section']}pt; border-radius: 0; padding: {s8}px 0;
}}
QListWidget#sidebar::item {{
    padding: {s12}px {s12}px; border-left: 3px solid transparent; margin: 1px {s8}px; border-radius: 6px;
}}
QListWidget#sidebar::item:hover {{ background-color: {t.bg_hover}; color: {t.text_primary}; }}
QListWidget#sidebar::item:selected {{
    background-color: {t.accent_subtle}; color: {t.text_primary}; border-left: 3px solid {t.accent};
}}
QPushButton#sidebar_help {{
    background-color: {t.bg_surface}; color: {t.text_secondary}; border: none; border-radius: 0;
    border-top: 1px solid {t.border_subtle}; padding: {s12 + 3}px {s16}px; text-align: left; font-weight: normal;
}}
QPushButton#sidebar_help:hover {{ background-color: {t.bg_hover}; color: {t.text_primary}; }}
QPushButton#sidebar_help:focus {{
    border: 2px solid {t.focus}; padding: {s12 + 2}px {s16 - 1}px;
}}
QDialog#palette {{
    background-color: {t.bg_raised}; border: 1px solid {t.accent}; border-radius: {RADIUS['card']}px;
}}
QDialog#palette QListWidget {{ border: none; background: transparent; }}
"""


def overlay_qss() -> str:
    """What is appended to a qt_material stylesheet: the three tweaks the app always added, nothing else. Variants
    and roles do not show under a qt_material theme; the widgets that were styled keep their old inline style
    (LEGACY_QSS), so the look is exactly the one the app had."""
    return LEGACY_APP_QSS


# -- applying ----------------------------------------------------------------------------------------------------------
def _palette(t: Tokens):
    from PySide6.QtGui import QColor, QPalette
    pal = QPalette()
    roles = {
        QPalette.Window: t.bg_surface, QPalette.WindowText: t.text_primary, QPalette.Base: t.bg_raised,
        QPalette.AlternateBase: t.bg_surface, QPalette.ToolTipBase: t.bg_raised, QPalette.ToolTipText: t.text_primary,
        QPalette.Text: t.text_primary, QPalette.Button: t.bg_raised, QPalette.ButtonText: t.text_primary,
        QPalette.BrightText: t.text_on_accent, QPalette.Highlight: t.accent, QPalette.HighlightedText: t.text_on_accent,
        QPalette.Link: t.info, QPalette.PlaceholderText: t.text_secondary, QPalette.Mid: t.border_strong,
        QPalette.Dark: t.bg_app, QPalette.Light: t.bg_hover, QPalette.Midlight: t.border_subtle,
        QPalette.Shadow: t.bg_app,
    }
    for role, c in roles.items():
        pal.setColor(role, QColor(c))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, QColor(t.text_disabled))
    return pal


def apply(app, name: str, defer_stylesheet: bool = False) -> None:
    """Make `name` (a token theme) the application's look: Fusion style, a palette and the generated stylesheet.

    With `defer_stylesheet` only the palette and the token set are applied; call finish_deferred(app) once the
    windows are built. (Qt re-resolves a stylesheet for every widget each time a finished sub-tree is put into a
    layout, so applying it after construction is cheaper than before; the result is the same.)"""
    global _current, _saved_palette, _pending_qss
    if not is_token_theme(name):                    # an unknown name (a newer or mistyped setting): the default look
        from tabs.settings import apply_theme
        apply_theme(DEFAULT_MATERIAL)
        return
    t = THEMES[name]
    if _saved_palette is None:
        _saved_palette = app.palette()
    app.setStyle("Fusion")
    app.setPalette(_palette(t))
    _current = t
    if defer_stylesheet:
        _pending_qss = build_qss(t)
        return
    _pending_qss = None
    app.setStyleSheet(build_qss(t))
    _after_change(app)


def finish_deferred(app) -> None:
    """Apply the stylesheet that apply(..., defer_stylesheet=True) held back (does nothing when none is pending)."""
    global _pending_qss
    if _pending_qss is None:
        return
    qss, _pending_qss = _pending_qss, None
    app.setStyleSheet(qss)
    _after_change(app)


def normalize_name(name: str | None) -> str:
    """`name` if it is a token theme or a qt_material theme file, else DEFAULT_MATERIAL (an unknown, empty or None
    setting, for example one written by a newer build)."""
    if is_token_theme(name):
        return name
    if isinstance(name, str) and name:
        try:
            from qt_material import list_themes
            if name in list_themes():
                return name
        except ImportError:
            pass
        import os
        if name.lower().endswith(".xml") and os.path.isfile(name):      # qt_material also takes a path
            return name
    return DEFAULT_MATERIAL


def set_legacy(app, theme_name: str | None = "") -> str:
    """Record that a qt_material theme is in force and give QPainter code and the styled widgets the colours they had.
    tabs.settings.apply_theme calls this, then sets the qt_material stylesheet of the name this returns (`theme_name`,
    or DEFAULT_MATERIAL when that is unknown, empty or None)."""
    global _current, _pending_qss
    _pending_qss = None
    theme_name = normalize_name(theme_name)
    if _saved_palette is not None:                 # back from a token theme: qt_material expects the stock palette
        app.setPalette(_saved_palette)
    was_legacy = is_legacy()
    _current = LEGACY
    if not was_legacy:
        _after_change(app)
    else:
        notifier().changed.emit()
    return theme_name


def theme_choices() -> list[tuple[str, str]]:
    """(label, settings value) of every theme the Settings tab offers: the token themes first, then the qt_material
    theme files under their own names (the value stored in ui.theme since the first version)."""
    choices = [(t.label, t.name) for t in THEMES.values()]
    try:
        from qt_material import list_themes
        choices += [(name, name) for name in list_themes() if "dark" in name or "light" in name]
    except ImportError:
        pass
    return choices
