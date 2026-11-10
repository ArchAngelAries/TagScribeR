"""The theme system (tabs/theme.py) and its Settings control: contrast of the token themes, the generated stylesheet,
switching between a qt_material theme and a token theme, and the widget helpers. Offscreen, no GPU."""
import os

import pytest

from tabs import theme

QtWidgets = pytest.importorskip("PySide6.QtWidgets")

TOKEN_THEMES = [theme.DARK, theme.LIGHT]
SURFACES = ("bg_app", "bg_surface", "bg_raised", "bg_hover")


@pytest.fixture
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
    from tabs.settings import apply_theme
    apply_theme("dark_teal.xml")            # leave the shared application in the default look for other tests
    app.setStyleSheet("")


# -- tokens ------------------------------------------------------------------------------------------------------------
def test_contrast_helper_matches_wcag():
    assert theme.contrast("#000000", "#ffffff") == pytest.approx(21.0)
    assert theme.contrast("#777777", "#ffffff") == pytest.approx(4.48, abs=0.01)
    assert theme.contrast("#123456", "#123456") == pytest.approx(1.0)


@pytest.mark.parametrize("t", TOKEN_THEMES, ids=lambda t: t.name)
def test_text_contrast_on_surfaces(t):
    for bg in SURFACES + ("accent_subtle",):
        assert theme.contrast(t.text_primary, t.color(bg)) >= 7.0, bg
        assert theme.contrast(t.text_secondary, t.color(bg)) >= 4.5, bg
    for role in ("success", "warning", "danger", "info"):       # semantic text sits on panels and cards
        assert theme.contrast(t.color(role), t.bg_surface) >= 4.5, role
        assert theme.contrast(t.color(role), t.bg_raised) >= 3.0, role


@pytest.mark.parametrize("t", TOKEN_THEMES, ids=lambda t: t.name)
def test_text_contrast_on_accent_and_danger_fills(t):
    for fill in ("accent", "accent_hover", "accent_pressed", "danger"):
        assert theme.contrast(t.text_on_accent, t.color(fill)) >= 4.5, fill


def test_dark_theme_writes_dark_text_on_accent():
    """White on the dark theme's accent is about 2.7:1: too low, so the text on it is dark."""
    assert theme.contrast("#ffffff", theme.DARK.accent) < 3.0
    assert theme.luminance(theme.DARK.text_on_accent) < 0.05


@pytest.mark.parametrize("t", TOKEN_THEMES + [theme.LEGACY], ids=lambda t: t.name)
def test_every_paint_key_resolves_to_a_colour(t):
    for key in list(theme.PAINT_KEYS) + list(t.colors()):
        value = t.color(key)
        assert value.startswith("#") and len(value) == 7, key
    assert set(theme.LEGACY_COLORS) == set(theme.PAINT_KEYS)    # the legacy set names every painted colour


def test_legacy_set_keeps_the_literals_the_code_had():
    assert theme.LEGACY.color("level_high") == "#ff7675"
    assert theme.LEGACY.color("chart_bg") == "#1e1e1e"
    assert theme.LEGACY.accent == "#00b894" and theme.LEGACY.text_on_accent == "#ffffff"


# -- stylesheet --------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("t", TOKEN_THEMES, ids=lambda t: t.name)
def test_build_qss_builds(t):
    qss = theme.build_qss(t)
    assert qss.count("{") == qss.count("}") > 100
    for needle in ('QPushButton[variant="primary"]', 'QPushButton[variant="danger"]', 'QLabel[role="warning"]',
                   "QListWidget#sidebar::item:selected", "QDialog#palette", "QLabel#hint", t.accent, t.bg_surface):
        assert needle in qss, needle
    assert "None" not in qss and "{t." not in qss


def test_theme_choices_offer_token_and_qt_material_themes():
    values = [value for _label, value in theme.theme_choices()]
    assert values[:2] == ["tagscriber_dark", "tagscriber_light"]
    assert "dark_teal.xml" in values and any(v.startswith("light_") for v in values)
    assert len(values) == len(set(values))
    from qt_material import list_themes
    assert {v for v in list_themes() if "dark" in v or "light" in v} <= set(values)   # none was dropped


# -- applying ----------------------------------------------------------------------------------------------------------
def test_apply_switches_between_qt_material_and_token_themes(app):
    from tabs.settings import apply_theme
    label = QtWidgets.QLabel("hint")
    theme.legacy_style(label, "hint")
    button = QtWidgets.QPushButton("Go")
    theme.set_variant(button, "primary", legacy="accent")

    apply_theme("dark_teal.xml")
    assert theme.is_legacy() and theme.tokens() is theme.LEGACY
    material = app.styleSheet()
    assert "QStackedWidget { background-color: #1e1e1e; }" in material and len(material) > 5000
    assert label.styleSheet() == theme.LEGACY_QSS["hint"] and "#00b894" in button.styleSheet()
    assert theme.color("level_high") == "#ff7675"

    seen = []
    theme.notifier().changed.connect(lambda: seen.append(theme.tokens().name))
    apply_theme("tagscriber_dark")
    assert not theme.is_legacy() and theme.tokens() is theme.DARK
    assert app.styleSheet() == theme.build_qss(theme.DARK)
    assert label.styleSheet() == "" and button.styleSheet() == ""        # the tokens style them now
    assert theme.color("level_high") == theme.DARK.danger
    assert app.palette().color(app.palette().ColorRole.Window).name() == theme.DARK.bg_surface

    apply_theme("tagscriber_light")
    assert theme.tokens() is theme.LIGHT and app.styleSheet() == theme.build_qss(theme.LIGHT)

    apply_theme("light_blue.xml")
    assert theme.is_legacy() and app.styleSheet() != material
    apply_theme("dark_teal.xml")
    assert app.styleSheet() == material                                  # back exactly where it started
    assert label.styleSheet() == theme.LEGACY_QSS["hint"] and "#00b894" in button.styleSheet()
    assert seen[:2] == ["tagscriber_dark", "tagscriber_light"] and seen[-1] == "legacy"


def test_widgets_made_under_a_token_theme_get_no_inline_style(app):
    from tabs.settings import apply_theme
    apply_theme("tagscriber_dark")
    label = QtWidgets.QLabel("hint")
    theme.legacy_style(label, "hint")
    assert label.styleSheet() == ""
    apply_theme("dark_teal.xml")
    assert label.styleSheet() == theme.LEGACY_QSS["hint"]


def test_refresh_icons_recolours_for_a_token_theme(app):
    from tabs import icons
    from tabs.settings import apply_theme
    if not icons.available():
        pytest.skip("qtawesome not installed")
    apply_theme("dark_teal.xml")
    plain, primary = QtWidgets.QPushButton("a"), QtWidgets.QPushButton("b")
    icons.set(plain, "run")
    icons.set(primary, "run", "on_accent")
    theme.set_variant(primary, "primary")
    apply_theme("tagscriber_light")
    theme.refresh_icons(plain)
    theme.refresh_icons(primary)
    assert _opaque_pixel(plain.icon()) == theme.LIGHT.text_primary
    assert _opaque_pixel(primary.icon()) == theme.LIGHT.text_on_accent


# -- widget helpers ----------------------------------------------------------------------------------------------------
def test_variants_and_roles_set_the_dynamic_property(app):
    button, label = QtWidgets.QPushButton("Run"), QtWidgets.QLabel("status")
    for variant in theme.VARIANTS:
        theme.set_variant(button, variant)
        assert button.property("variant") == variant
    for role in theme.ROLES:
        theme.set_role(label, role)
        assert label.property("role") == role
    theme.set_role(label, None)
    assert label.property("role") is None
    theme.set_surface(label, "image")
    assert label.property("surface") == "image"
    with pytest.raises(ValueError):
        theme.set_variant(button, "loud")
    with pytest.raises(ValueError):
        theme.set_role(label, "shouting")
    with pytest.raises(ValueError):
        theme.legacy_style(label, "no_such_style")


def test_level_colour_follows_the_theme(app):
    from tabs.settings import apply_theme
    apply_theme("tagscriber_dark")
    assert theme.level_color("high") == theme.DARK.danger
    assert theme.level_color("medium") == theme.DARK.warning
    assert theme.level_color("low") == theme.level_color("anything else") == theme.DARK.text_secondary
    assert theme.qcolor("danger", 70).alpha() == 70


# -- the Settings control ----------------------------------------------------------------------------------------------
@pytest.fixture
def settings_tab(app, monkeypatch):
    from core import config
    import tabs.settings as settings_mod
    monkeypatch.setattr(config, "_settings", None)                       # a fresh Settings on the isolated user_data
    monkeypatch.setattr(settings_mod, "run_in_background", lambda *a, **k: None)   # no hardware probe
    monkeypatch.setattr(settings_mod, "stability_matrix_llm_dirs", lambda: [])
    tab = settings_mod.SettingsTab()
    yield tab
    tab.deleteLater()


def test_settings_lists_every_theme_and_defaults_to_dark_teal(settings_tab):
    combo = settings_tab.combo_theme
    assert [combo.itemData(i) for i in range(combo.count())] == [v for _l, v in theme.theme_choices()]
    assert combo.currentData() == "dark_teal.xml" == settings_tab.cfg.get("ui.theme")
    assert "TagScribeR dark" in combo.toolTip() and "dark_teal" in combo.toolTip()


def test_settings_applies_a_theme_at_once_and_remembers_it(settings_tab, app):
    combo = settings_tab.combo_theme
    combo.setCurrentIndex(combo.findData("tagscriber_light"))
    assert theme.tokens() is theme.LIGHT and app.styleSheet() == theme.build_qss(theme.LIGHT)
    assert settings_tab.cfg.get("ui.theme") == "tagscriber_light"
    combo.setCurrentIndex(combo.findData("dark_teal.xml"))
    assert theme.is_legacy() and settings_tab.cfg.get("ui.theme") == "dark_teal.xml"


def test_settings_tooltip_makes_no_claim_it_cannot_keep(settings_tab):
    tip = settings_tab.combo_theme.toolTip()
    assert "light everywhere" not in tip and "no restart needed" not in tip
    assert "main panels and controls" in tip and "on panels and on the accent colour" in tip


# -- icons: only tabs.icons icons are ever recoloured -----------------------------------------------------------------
def _opaque_pixel(icon, size=24):
    from PySide6.QtCore import QSize
    img = icon.pixmap(QSize(size, size)).toImage()
    for y in range(img.height()):
        for x in range(img.width()):
            c = img.pixelColor(x, y)
            if c.alpha() == 255:
                return c.name()
    return None


def _picture_icon():
    from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
    pm = QPixmap(48, 48)                                  # generated: a stand-in for a thumbnail, never a real image
    pm.fill(QColor("#c0392b"))
    p = QPainter(pm)
    p.fillRect(8, 8, 20, 20, QColor("#2980b9"))
    p.end()
    return QIcon(pm)


def test_theme_switch_leaves_picture_icons_alone_and_restores_tabs_icons(app):
    from PySide6.QtCore import QSize
    from tabs import icons
    from tabs.settings import apply_theme
    if not icons.available():
        pytest.skip("qtawesome not installed")
    apply_theme("dark_teal.xml")
    lst = QtWidgets.QListWidget()
    pic = QtWidgets.QListWidgetItem(_picture_icon(), "picture")
    vec = QtWidgets.QListWidgetItem(icons.icon("run"), "vector")
    lst.addItem(pic)
    lst.addItem(vec)

    def image(item, px=48):
        return item.icon().pixmap(QSize(px, px)).toImage()
    pic_before, vec_before = image(pic), image(vec)
    assert _opaque_pixel(vec.icon()) == "#dddddd"

    apply_theme("tagscriber_light")
    assert image(pic) == pic_before                       # the picture is untouched
    assert _opaque_pixel(vec.icon()) == theme.LIGHT.text_primary and image(vec) != vec_before
    apply_theme("tagscriber_dark")
    assert image(pic) == pic_before
    apply_theme("dark_teal.xml")
    assert image(pic) == pic_before and image(vec) == vec_before     # the vector icon is back as it was
    assert _opaque_pixel(vec.icon()) == "#dddddd"


def test_icons_made_after_start_up_take_the_theme_colour(app):
    from tabs import icons
    from tabs.settings import apply_theme
    if not icons.available():
        pytest.skip("qtawesome not installed")
    apply_theme("tagscriber_light")
    assert _opaque_pixel(icons.icon("save")) == theme.LIGHT.text_primary         # not #dddddd on white
    assert theme.contrast(_opaque_pixel(icons.icon("save")), theme.LIGHT.bg_surface) >= 4.5
    apply_theme("dark_teal.xml")
    assert _opaque_pixel(icons.icon("save")) == "#dddddd" == icons.DEFAULT_COLOR


def test_filled_button_icon_follows_variant_and_returns(app):
    from tabs import icons
    from tabs.settings import apply_theme
    if not icons.available():
        pytest.skip("qtawesome not installed")
    apply_theme("dark_teal.xml")
    b = QtWidgets.QPushButton("Go")
    icons.set(b, "run", "on_accent")
    assert _opaque_pixel(b.icon()) == "#ffffff"
    theme.set_variant(b, "primary")
    apply_theme("tagscriber_dark")
    assert _opaque_pixel(b.icon()) == theme.DARK.text_on_accent
    apply_theme("dark_teal.xml")
    assert _opaque_pixel(b.icon()) == "#ffffff"


# -- layout: the stylesheet must not leave sizes behind ----------------------------------------------------------------
def test_fixed_height_button_keeps_its_height_over_a_round_trip(app):
    from tabs.settings import apply_theme
    apply_theme("dark_teal.xml")
    host = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(host)
    run = QtWidgets.QPushButton("Run")
    run.setFixedHeight(48)
    run.setProperty("btnSize", "large")
    lay.addWidget(run)
    host.show()
    app.processEvents()
    before = run.height()
    apply_theme("tagscriber_dark")
    app.processEvents()
    assert run.height() == 48
    apply_theme("dark_teal.xml")
    app.processEvents()
    assert before == run.height() == 48 and (run.minimumHeight(), run.maximumHeight()) == (48, 48)
    host.close()


def test_button_size_property_is_not_a_qwidget_property_and_has_a_rule(app):
    run = QtWidgets.QPushButton("Run")
    assert run.metaObject().indexOfProperty("size") >= 0       # QWidget's own: a dynamic "size" is never stored
    assert run.metaObject().indexOfProperty("btnSize") == -1   # a dynamic property, which a QSS selector can read
    run.setProperty("btnSize", "large")
    assert run.property("btnSize") == "large"
    for t in TOKEN_THEMES:
        qss = theme.build_qss(t)
        assert 'QPushButton[btnSize="large"]' in qss and '[size="large"]' not in qss
    from tabs import caption
    assert 'setProperty("btnSize", "large")' in open(caption.__file__, encoding="utf-8").read()


def test_stylesheet_has_no_min_height_on_buttons():
    for t in TOKEN_THEMES:
        qss = theme.build_qss(t)
        start = qss.index("QPushButton {")
        assert "min-height" not in qss[start:qss.index("}", start)]


# -- the stylesheet parses, wherever user_data is -----------------------------------------------------------------------
@pytest.fixture
def qt_warnings():
    from PySide6.QtCore import qInstallMessageHandler
    seen = []
    previous = qInstallMessageHandler(lambda _mode, _ctx, message: seen.append(message))
    yield seen
    qInstallMessageHandler(previous)


@pytest.mark.parametrize("folder", ["plain", "it's a folder"])
@pytest.mark.parametrize("t", TOKEN_THEMES, ids=lambda t: t.name)
def test_token_stylesheet_parses_without_a_warning(app, qt_warnings, monkeypatch, tmp_path, t, folder):
    from core import paths
    root = tmp_path / folder
    root.mkdir()
    monkeypatch.setattr(paths, "USER_DATA", root)
    qss = theme.build_qss(t)
    assert 'image: url("' in qss and (root / "cache" / "theme" / f"check_{t.name}.svg").is_file()
    app.setStyleSheet(qss)
    w = QtWidgets.QCheckBox("x")
    w.show()
    app.processEvents()
    w.close()
    assert not [m for m in qt_warnings if "parse" in m.lower() or "stylesheet" in m.lower()], qt_warnings


# -- contrast of the parts a user must see -------------------------------------------------------------------------------
@pytest.mark.parametrize("t", TOKEN_THEMES, ids=lambda t: t.name)
def test_placeholder_and_empty_state_text_is_readable(app, t):
    from tabs.settings import apply_theme
    apply_theme(t.name)
    placeholder = app.palette().color(app.palette().ColorRole.PlaceholderText).name()
    assert theme.contrast(placeholder, t.bg_raised) >= 4.5
    assert f"placeholder-text-color: {placeholder}" in theme.build_qss(t)
    assert theme.contrast(t.text_secondary, t.bg_image) >= 4.5      # the "empty preview" text of QLabel[surface="image"]
    qss = theme.build_qss(t)
    block = qss[qss.index('QLabel[surface="image"]'):]
    assert f"color: {t.text_secondary}" in block[:block.index("}")]


@pytest.mark.parametrize("t", TOKEN_THEMES, ids=lambda t: t.name)
def test_borders_and_focus_ring_have_3_to_1_contrast(t):
    assert theme.contrast(t.border_strong, t.bg_surface) >= 3.0      # input and checkbox borders on the panel
    assert theme.contrast(t.border_strong, t.bg_raised) >= 3.0       # ... and against what is inside the control
    assert theme.contrast(t.focus, t.border_strong) >= 3.0           # the ring against the unfocused border
    for bg in ("bg_surface", "bg_raised", "bg_app"):
        assert theme.contrast(t.focus, t.color(bg)) >= 3.0, bg


@pytest.mark.parametrize("t", TOKEN_THEMES, ids=lambda t: t.name)
def test_semantic_text_is_readable_on_rows_and_panels(t):
    for role in ("success", "warning", "danger", "info"):
        for bg in SURFACES + ("accent_subtle",):
            assert theme.contrast(t.color(role), t.color(bg)) >= 4.5, (role, bg)


@pytest.mark.parametrize("t", TOKEN_THEMES, ids=lambda t: t.name)
def test_every_focusable_control_has_a_two_pixel_focus_indicator(t):
    qss = theme.build_qss(t)
    ring = f"2px solid {t.focus}"
    for selector in ("QPushButton:focus", "QToolButton:focus", "QLineEdit:focus", "QComboBox:focus",
                     "QCheckBox::indicator:focus", "QRadioButton::indicator:focus", "QSlider::handle:horizontal:focus",
                     "QTabBar::tab:selected:focus", "QListView:focus", "QTableView:focus", "QTreeView:focus"):
        i = qss.index(selector)
        assert ring in qss[i:qss.index("}", i)], selector
    # an item view that hides Qt's dotted rect says what shows focus instead; the sidebar and combo popup keep it
    assert qss.count("outline: none") == 1 and "the view's border shows focus" in qss


@pytest.mark.parametrize("t", TOKEN_THEMES, ids=lambda t: t.name)
def test_memory_bar_text_reads_on_the_fill(app, t):
    from PySide6.QtGui import QColor
    from tabs.memory_bar import _Segment
    from tabs.settings import apply_theme
    apply_theme(t.name)
    for start, end in (("#3FB950", "#E5534B"), ("#3B82F6", "#EAC54F")):
        seg = _Segment("VRAM", start, end, "")
        a, b = QColor(start), QColor(end)
        for frac in (0.1, 0.5, 1.0):
            text = seg._text_on_fill(250, 250 * frac, 150)
            right = max(10.0, min(250 * frac, 160.0))
            for i in range(5):
                f = (10 + (right - 10) * i / 4) / 250
                px = "#%02x%02x%02x" % tuple(round(x + (y - x) * f) for x, y in
                                              ((a.red(), b.red()), (a.green(), b.green()), (a.blue(), b.blue())))
                assert theme.contrast(text, px) >= 4.5, (start, frac, i)


# -- the default look is pinned to the literals the code used to have -------------------------------------------------
def _flat(css: str) -> str:
    return "".join(css.split())          # the rules' own spacing between blocks is not part of the look


OLD_QSS = {
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
               " QListWidget::item { padding: 15px; border-bottom: 1px solid #2c2c2c; }"
               " QListWidget::item:selected { background-color: #00b894; color: white; }",
    "sidebar_help": "QPushButton { background-color: #2d3436; color: #aaa; border: none; padding: 15px;"
                    " text-align: left; } QPushButton:hover { background-color: #333; color: white; }",
    "preview": "background-color: #1a1a1a; border-radius: 6px; color: #777;",
    "preview_plain": "background-color: #1a1a1a; border-radius: 6px;",
    "text_current": "QTextBrowser { color: #bbb; background-color: #1e1e1e; }",
    "text_changes": "QTextBrowser { background-color: #1e1e1e; }",
    "accent": "background-color: #00b894; color: white; font-weight: bold; padding: 6px;",
    "accent_wide": "background-color: #00b894; color: white; font-weight: bold; padding: 8px;",
    "purple": "background-color: #6c5ce7; color: white; font-weight: bold; padding: 6px;",
    "purple_wide": "background-color: #6c5ce7; color: white; font-weight: bold; padding: 8px;",
    "blue": "background-color: #0984e3; color: white; font-weight: bold; padding: 6px;",
    "teal": "background-color: #00897b; color: white; font-weight: bold; padding: 6px;",
    "run": "background-color: #d63031; font-weight: bold; font-size: 14px;",
    "abort": "background-color: #ff4757; font-weight: bold; font-size: 14px;",
}
OLD_COLORS = {      # the literals of the painters, item colours and the help page, by use
    "chart_bg": "#1e1e1e", "chart_frame": "#555555", "chart_grid": "#333333", "chart_text": "#aaaaaa",
    "chart_muted": "#888888", "chart_faint": "#777777", "chart_lr": "#e17055", "chart_raw": "#00b894",
    "chart_smooth": "#81ecec", "chart_avg": "#00b894", "chart_mark": "#fdcb6e", "chart_cursor": "#dfe6e9",
    "bar_track": "#15171c", "bar_text": "#ffffff", "bar_text_on_fill": "#ffffff", "bar_muted": "#8b93a1",
    "level_high": "#ff7675", "level_medium": "#fdcb6e", "level_low": "#9a9a9a", "text_readonly": "#9a9a9a",
    "diff_add": "#55efc4", "diff_del": "#ff7675", "diff_same": "#cfcfcf", "diff_empty": "#777777",
    "help_h2": "#00b894", "help_h3": "#d0d0d0", "help_code": "#fdcb6e", "help_code_bg": "#2b2b2b",
    "list_header": "#a0a0a4", "icon": "#dddddd", "icon_nav": "#dddddd",
}


def test_legacy_qss_is_pinned_to_the_old_inline_styles():
    assert set(theme.LEGACY_QSS) - {"support_link"} == set(OLD_QSS)     # support_link came with the support buttons
    for key, old in OLD_QSS.items():
        assert _flat(theme.LEGACY_QSS[key]) == _flat(old), key


def test_legacy_colours_are_pinned_to_the_old_literals():
    assert theme.LEGACY_COLORS == OLD_COLORS
    for key, old in OLD_COLORS.items():
        assert theme.LEGACY.color(key) == old, key
    assert _flat(theme.LEGACY_APP_QSS) == _flat("QStackedWidget { background-color: #1e1e1e; } "
                                                "QScrollBar:vertical { width: 12px; } "
                                                "QLineEdit, QTextEdit, QPlainTextEdit { border-radius: 4px; }")
    assert theme.overlay_qss() == theme.LEGACY_APP_QSS


# -- startup with a theme name that does not exist ---------------------------------------------------------------------
@pytest.mark.parametrize("name", ["no_such_theme.xml", "", None, "TagScribeR dark"])
def test_unknown_empty_or_none_theme_falls_back_to_dark_teal(app, name):
    from tabs.settings import apply_theme
    apply_theme("dark_teal.xml")
    expected = app.styleSheet()
    apply_theme("tagscriber_dark")
    apply_theme(name)
    assert theme.is_legacy() and app.styleSheet() == expected and len(expected) > 5000
    assert theme.normalize_name(name) == "dark_teal.xml"


def test_theme_apply_with_an_unknown_name_falls_back(app):
    from tabs.settings import apply_theme
    apply_theme("dark_teal.xml")
    expected = app.styleSheet()
    theme.apply(app, "nope")
    assert theme.is_legacy() and app.styleSheet() == expected


# -- the Metadata tab keeps unsaved edits across a theme change ---------------------------------------------------------
def test_metadata_edits_survive_a_theme_change(app, tmp_path, monkeypatch):
    from core import paths, presets
    from PySide6.QtCore import Qt
    from tabs.settings import apply_theme
    monkeypatch.setattr(presets, "_camera_store", None)
    monkeypatch.setattr(presets, "_metadata_store", None)
    monkeypatch.setattr(paths, "CAMERA_PRESETS_FILE", tmp_path / "camera_presets.json")
    monkeypatch.setattr(paths, "METADATA_PRESETS_FILE", tmp_path / "metadata_presets.json")
    from tabs.metadata import FINDING_ROLE, MetadataTab
    apply_theme("dark_teal.xml")
    tab = MetadataTab()
    tab.table.setRowCount(2)
    for i, (editable, text) in enumerate(((True, "original"), (False, "read only"))):
        key = QtWidgets.QTableWidgetItem("Field")
        v = QtWidgets.QTableWidgetItem(text)
        if editable:
            v.setData(Qt.UserRole, "text:Field")
        else:
            v.setFlags(v.flags() & ~Qt.ItemIsEditable)
            v.setForeground(theme.qcolor("text_readonly"))
        tab.table.setItem(i, 0, key)
        tab.table.setItem(i, 1, v)
    ok = QtWidgets.QListWidgetItem("Nothing sensitive found")
    ok.setData(FINDING_ROLE, "ok")
    tab.list_findings.addItem(ok)
    high = QtWidgets.QListWidgetItem("GPS")
    high.setData(FINDING_ROLE, "high")
    tab.list_findings.addItem(high)

    edited = tab.table.item(0, 1)
    edited.setText("edited but not saved")
    apply_theme("tagscriber_light")
    assert tab.table.item(0, 1) is edited and edited.text() == "edited but not saved"
    assert tab.table.item(1, 1).foreground().color().name() == theme.LIGHT.text_secondary
    assert ok.foreground().color().name() == theme.LIGHT.success
    assert high.foreground().color().name() == theme.LIGHT.danger
    apply_theme("dark_teal.xml")
    assert edited.text() == "edited but not saved"
    assert tab.table.item(1, 1).foreground().color().name() == "#9a9a9a"
    assert high.foreground().color().name() == "#ff7675"
    tab.deleteLater()
