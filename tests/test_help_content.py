import warnings
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "tabs" / "help_content.py"


def test_help_content_compiles_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # invalid escapes in HTML strings become test failures
        compile(SRC.read_text(encoding="utf-8"), str(SRC), "exec")


def test_every_tab_has_a_help_topic():
    from tabs.help_content import TAB_TOPICS, TOPICS
    assert all(t in TOPICS for t in TAB_TOPICS)


def test_fizgig_is_named_only_in_credits():
    from tabs.help_content import HOTKEYS, TOPICS
    for tid, (title, _group, html) in TOPICS.items():
        if tid == "credits":
            continue
        assert "fizgig" not in title.lower(), tid
        assert "fizgig" not in html.lower(), tid
    for row in HOTKEYS:
        assert all("fizgig" not in str(cell).lower() for cell in row), row
    credits = TOPICS["credits"][2]
    low = credits.lower()
    assert "fizgig" in low and "apache" in low and "not affiliated" in low


def test_credits_has_support_links():
    from tabs.help_content import TOPICS
    html = TOPICS["credits"][2]
    assert "https://www.patreon.com/c/ArchAngelAries" in html
    assert "https://ko-fi.com/archangelaries" in html
    assert TOPICS["credits"][0] == "Credits and licences"


def test_icon_emoji_mapping():
    from tabs import icons
    assert icons.split_emoji("💾 Save") == ("save", "Save")
    assert icons.split_emoji("🗑️ Delete") == ("delete", "Delete")
    assert icons.split_emoji("Plain") == (None, "Plain")
    assert all(name in icons.NAMES for name in icons.EMOJI.values())
