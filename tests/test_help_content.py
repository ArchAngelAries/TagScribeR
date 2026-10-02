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
