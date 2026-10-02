from dataclasses import dataclass

import pytest

from core import caption_io
from core import dataset_session as ds
from core.query import compile_query
from tests.conftest import make_image


@dataclass
class E:
    name: str = "img.png"
    text: str = ""
    width: int = 0
    height: int = 0
    dirty: bool = False


def match(q, **kw):
    return compile_query(q).predicate(E(**kw))


@pytest.mark.parametrize("q,kw,expected", [
    ("smile", dict(text="1girl, smile"), True),
    ("smile frown", dict(text="1girl, smile"), False),
    ("-smile", dict(text="1girl, smile"), False),
    ("tag:long_hair", dict(text="Long Hair, solo"), True),
    ('tag:"long hair"', dict(text="long_hair"), True),
    ("tag:hair", dict(text="long hair"), False),             # exact tag, not substring
    ("tag:*hair", dict(text="long hair, solo"), True),
    ("-tag:blurry", dict(text="blurry, solo"), False),
    ("missing:caption", dict(text="  "), True),
    ("has:caption", dict(text="x"), True),
    ("is:unsaved", dict(dirty=True), True),
    ("res:<768", dict(width=1024, height=700), True),
    ("res:<768", dict(width=0, height=0), False),            # unknown size never matches
    ("ar:>1.5", dict(width=1600, height=900), True),
    ("mp:>=1", dict(width=1000, height=1000), True),
    ("name:img_*", dict(name="img_001.png"), True),
    ("ext:png", dict(name="a.PNG"), True),
    ("tags:>2", dict(text="a, b, c"), True),
    ("len:<5", dict(text="abc"), True),
    ("img", dict(name="img_1.png"), True),                   # bare words also match filenames
])
def test_query_terms(q, kw, expected):
    assert match(q, **kw) is expected


def test_query_reports_errors_and_dimension_use():
    q = compile_query("res:big bogus:thing res:>10")
    assert len(q.errors) == 2 and q.needs_info
    assert not compile_query("tag:x").needs_info


def test_unbalanced_quotes_dont_crash():
    assert compile_query('tag:"long hair').predicate(E(text="long hair")) in (True, False)


@pytest.fixture
def session(tmp_path):
    for i, cap in enumerate(["1girl, smile, Long_Hair", "1girl, frown", ""]):
        make_image(tmp_path / f"img ({i + 1}).png")
        if cap:
            caption_io.write_caption(tmp_path / f"img ({i + 1}).png", cap)
    return ds.DatasetSession.open(tmp_path)


def test_session_open_and_stats(session):
    assert [e.name for e in session.entries] == ["img (1).png", "img (2).png", "img (3).png"]
    assert session.stats() == {"images": 3, "captioned": 2, "missing": 1, "unsaved": 0}


def test_transform_returns_changes_without_applying(session):
    keys = [e.key for e in session.entries]
    changes = session.transform(keys, ds.op_add_tags(["solo"]))
    assert len(changes) == 3
    assert session.entries[0].text == "1girl, smile, Long_Hair"  # not applied yet
    session.apply(changes)
    assert session.entries[2].text == "solo" and session.entries[0].dirty
    session.apply(changes, undo=True)
    assert not any(e.dirty for e in session.entries)


def test_save_only_writes_dirty(session, tmp_path):
    e = session.entries[1]
    e.text = "1girl, frown, solo"
    report = session.save()
    assert report.saved == 1 and report.unchanged == 2
    assert caption_io.read_caption(e.path) == "1girl, frown, solo" and not e.dirty
    assert not (tmp_path / "img (3).txt").exists()  # empty caption never created


def test_tag_counts_group_spellings(session):
    session.entries[1].text = "1girl, long hair"
    counts = dict(session.tag_counts())
    assert counts["1girl"] == 2
    assert counts.get("Long_Hair", counts.get("long hair")) == 2


@pytest.mark.parametrize("op,before,after", [
    (ds.op_remove_tags(["long hair"]), "a, Long_Hair, b", "a, b"),
    (ds.op_replace_tag("long_hair", "very long hair"), "a, long hair", "a, very long hair"),
    (ds.op_replace_tag("x", "a"), "a, x", "a"),
    (ds.op_find_replace("girl", "woman"), "1girl", "1woman"),
    (ds.op_normalize(underscores_to_spaces=True), "long_hair, long hair, solo", "long hair, solo"),
    (ds.op_normalize(), "", ""),
    (ds.op_prefix_suffix("ohwx, "), "a woman", "ohwx, a woman"),
    (ds.op_prefix_suffix("ohwx, "), "ohwx, a woman", "ohwx, a woman"),
    (ds.op_ascii(), "café", "cafe"),
    (ds.op_clear(), "x", ""),
])
def test_ops(op, before, after):
    assert op(before) == after
