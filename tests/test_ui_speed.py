"""Stage 0 of the UI plan (docs/dev/UI_UX_REDESIGN.md 5.3): no freezes, same results.

Counters stand in for stopwatches: no sort in a hidden browser, no full re-read of captions when the Gallery is shown
again, a hit rate for the delegate's caches, running counts equal to a recount. Everything is generated data."""
import os
import random
from pathlib import Path

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtCore import QRect, QSortFilterProxyModel, Qt  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QStyleOptionViewItem  # noqa: E402

from core.dataset_session import DatasetSession, Entry  # noqa: E402
from tabs.workspace import context as ctxmod  # noqa: E402
from tabs.workspace.delegate import CardDelegate  # noqa: E402
from tabs.workspace.model import ENTRY_ROLE, SORT_MODES, DatasetModel, FilterProxy  # noqa: E402
from tests.conftest import make_image  # noqa: E402


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


_CONTEXTS = []


@pytest.fixture(autouse=True)
def _quiet_contexts():
    """A context that outlives the browsers of its test must not signal their deleted buttons when it is collected."""
    yield
    while _CONTEXTS:
        _CONTEXTS.pop().undo_stack.blockSignals(True)


def _folder(tmp_path, n=12, captioned=0.75, name="demo_dataset"):
    folder = tmp_path / name
    folder.mkdir()
    rng = random.Random(3)
    for i in range(n):
        make_image(folder / f"img_{i:03d}.png", size=(16 + i % 5, 12 + i % 3))
        if rng.random() < captioned:
            (folder / f"img_{i:03d}.txt").write_text(f"tag{i}, solo, smile", encoding="utf-8")
    return folder


def _open(tmp_path, n=12, **kw):
    ctx = ctxmod.WorkspaceContext()
    _CONTEXTS.append(ctx)
    assert ctx.open_folder(str(_folder(tmp_path, n, **kw)), False, None)
    return ctx


def _pump(app, ms=0):
    from PySide6.QtTest import QTest
    app.processEvents()
    if ms:
        QTest.qWait(ms)


# ---------------------------------------------------------------------------------------------- sort: same order
class _OldProxy(QSortFilterProxyModel):
    """The proxy as it was before the key cache: two round trips into the model and the key function per comparison."""

    def __init__(self):
        super().__init__()
        self.sort_key = SORT_MODES["Name"]

    def set_sort_mode(self, name, descending=False):
        self.sort_key = SORT_MODES.get(name, SORT_MODES["Name"])
        self.invalidate()
        self.sort(0, Qt.DescendingOrder if descending else Qt.AscendingOrder)

    def lessThan(self, left, right):
        a, b = left.data(ENTRY_ROLE), right.data(ENTRY_ROLE)
        if a is None or b is None:
            return False
        return self.sort_key(a) < self.sort_key(b)


def _entries(n=240):
    rng = random.Random(11)
    names = ["img (2).png", "img (10).png", "IMG (2).png", "Img_a.PNG", "img_A.png", "é.png", "É.png", "日本.png",
             "zz.png", "a1b2.png", "a1b10.png", "a01b2.png", "", "-.png", "10.png", "9.png"]
    texts = ["", "  ", "solo", "solo, smile", "a, b, c, d", "1girl, solo, long hair, smile", "smile, solo", "é, ü",
             "x\ny", "tag, tag, tag"]
    entries = []
    for i in range(n):
        d = f"dir{rng.randrange(4)}"
        e = Entry(path=Path(d) / f"{rng.choice(names)[:-4] or 'n'}{rng.randrange(6)}.{rng.choice(['png', 'jpg'])}"
                  if rng.random() < 0.5 else Path(d) / rng.choice(names),
                  text=rng.choice(texts), width=rng.choice((0, 64, 64, 512, 1024)),
                  height=rng.choice((0, 64, 480, 768, 768)), file_size=rng.choice((10, 10, 20, 5000)),
                  mtime=float(rng.choice((1, 1, 2, 3.5))))
        e.saved_text = e.text if rng.random() < 0.7 else e.text + "!"
        entries.append(e)
    # distinct paths (the key of an entry is its path): add the row number to a duplicate
    seen = set()
    for i, e in enumerate(entries):
        while e.path in seen:
            e.path = e.path.with_name(f"{e.path.stem}_{i}{e.path.suffix}")
        seen.add(e.path)
    return entries


def _order(proxy):
    return [proxy.index(r, 0).data(ENTRY_ROLE).key for r in range(proxy.rowCount())]


@pytest.mark.parametrize("mode", list(SORT_MODES))
@pytest.mark.parametrize("descending", [False, True])
def test_cached_sort_order_is_identical_to_the_old_comparison(app, mode, descending):
    entries = _entries()
    model = DatasetModel()
    model.set_session(DatasetSession(Path("."), entries))
    new, old = FilterProxy(), _OldProxy()
    new.setSourceModel(model)
    old.setSourceModel(model)
    new.set_sort_mode(mode, descending)
    old.set_sort_mode(mode, descending)
    assert _order(new) == _order(old)
    assert new.lessthan_calls > 0
    # edits drop the cached keys of exactly those rows: the order follows the data, as the old proxy's does
    rng = random.Random(5)
    changed = []
    for e in rng.sample(entries, 40):
        e.text = rng.choice(["", "new, caption", "z" * 30, "a, b"])
        e.width = rng.choice((0, 100, 4000))
        changed.append(e.key)
    model.refresh(changed)
    assert _order(new) == _order(old)
    # and a full re-sort after the edits (cache reused) still agrees
    new.set_sort_mode(mode, descending)
    old.set_sort_mode(mode, descending)
    assert _order(new) == _order(old)


def test_sort_key_is_computed_once_per_row_per_mode(app):
    entries = _entries(60)
    calls = []
    model = DatasetModel()
    model.set_session(DatasetSession(Path("."), entries))
    p = FilterProxy()
    p.setSourceModel(model)
    original = SORT_MODES["Caption length"]
    try:
        SORT_MODES["Caption length"] = lambda e: (calls.append(e.key), original(e))[1]
        p.set_sort_mode("Caption length")
        _order(p)
        first = len(calls)
        assert first == len(entries)                      # one key per row, not one per comparison
        p.set_sort_mode("Caption length", True)
        _order(p)
        assert len(calls) == first                         # cached across sorts
        entries[3].text = "changed"
        model.refresh([entries[3].key])
        _order(p)
        assert len(calls) == first + 1                     # only the changed row was recomputed
    finally:
        SORT_MODES["Caption length"] = original


# ---------------------------------------------------------------------------------- hidden browsers do nothing
def test_a_hidden_browser_does_not_sort_until_it_is_shown_or_read(app, tmp_path):
    from tabs.workspace.browser import DatasetBrowser
    ctx = ctxmod.WorkspaceContext()
    _CONTEXTS.append(ctx)
    shown, hidden = DatasetBrowser(ctx), DatasetBrowser(ctx)
    shown.resize(500, 400)
    shown.show()
    _pump(app)
    assert ctx.open_folder(str(_folder(tmp_path, 30)), False, None)
    _pump(app, 50)
    assert shown._proxy.lessthan_calls > 0 and not shown.is_stale
    assert hidden._proxy.lessthan_calls == 0 and hidden.is_stale
    # data changes while hidden are free too
    keys = [e.key for e in ctx.session.entries][:10]
    ctx.push({k: (ctx.session.get(k).text, "edited") for k in keys}, "edit")
    ctx.set_job_state(keys, "queued")
    _pump(app)
    assert hidden._proxy.lessthan_calls == 0
    # reading the rows from code refreshes first
    hidden.combo_sort.setCurrentText("Caption length")
    hidden.inp_filter.setText("edited")
    hidden.apply_filter()
    assert hidden._proxy.lessthan_calls == 0 and hidden.is_stale
    assert sorted(hidden.shown_keys()) == sorted(keys)
    assert not hidden.is_stale and hidden._proxy.lessthan_calls > 0
    assert hidden.proxy.rowCount() == 10
    assert hidden.keys_for_scope("shown") == hidden.shown_keys()
    # showing a stale browser brings it up to date and writes its status line
    other = DatasetBrowser(ctx)
    assert other.is_stale and "Open a folder" in other.lbl_status.text()
    other.resize(500, 400)
    other.show()
    _pump(app)
    assert not other.is_stale and "30 images" in other.lbl_status.text()


def test_hiding_a_browser_makes_it_lazy_again(app, tmp_path):
    from tabs.workspace.browser import DatasetBrowser
    ctx = _open(tmp_path, 20)
    b = DatasetBrowser(ctx)
    b.resize(500, 400)
    b.show()
    _pump(app)
    assert not b.is_stale
    b.hide()
    _pump(app)
    calls = b._proxy.lessthan_calls
    assert ctx.open_folder(str(_folder(tmp_path, 25, name="second")), False, None)
    ctx.push({ctx.session.entries[0].key: ("", "changed")}, "edit")
    _pump(app)
    assert b._proxy.lessthan_calls == calls and b.is_stale
    b.show()
    _pump(app)
    assert not b.is_stale and "25 images" in b.lbl_status.text()


# ------------------------------------------------------------------- coming back: no full re-read of captions
@pytest.fixture
def counting(monkeypatch):
    reads = []
    real = ctxmod.caption_io.read_caption

    def read(path, ext=".txt"):
        reads.append(str(path))
        return real(path, ext)
    monkeypatch.setattr(ctxmod.caption_io, "read_caption", read)
    return reads


def _touch(path: Path, text: str):
    path.write_text(text, encoding="utf-8")
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))   # the date moves even on a coarse clock


def test_returning_reads_only_the_captions_that_changed(app, tmp_path, counting):
    ctx = _open(tmp_path, 12)
    folder = Path(ctx.folder)
    counting.clear()
    ctx.pick_up_external_changes()
    assert counting == []                                   # nothing changed: nothing read
    # changed, appeared, disappeared
    e1, e2, e3 = (ctx.session.entries[i] for i in (1, 2, 3))
    _touch(folder / "img_001.txt", "rewritten elsewhere, longer than before")
    _touch(folder / "img_002.txt", "appeared")
    (folder / "img_003.txt").unlink(missing_ok=True)
    had3 = bool(e3.saved_text)
    ctx.pick_up_external_changes()
    assert e1.text == "rewritten elsewhere, longer than before" and e2.text == "appeared"
    assert (e3.text == "") if had3 else True
    assert len(counting) <= 3 and all(Path(p).stem in ("img_001", "img_002", "img_003") for p in counting)
    counting.clear()
    ctx.pick_up_external_changes()                           # recorded: a second visit reads nothing
    assert counting == []


def test_unsaved_edits_are_never_overwritten_by_the_check(app, tmp_path, counting):
    ctx = _open(tmp_path, 8)
    folder = Path(ctx.folder)
    e = ctx.session.entries[2]
    ctx.push({e.key: (e.text, "my unsaved edit")}, "edit")
    _touch(folder / "img_002.txt", "someone else wrote this")
    ctx.pick_up_external_changes()
    assert e.text == "my unsaved edit" and e.dirty
    # the file is looked at again once the edit is gone (undo): then the outside change is picked up, as it always was
    ctx.undo_stack.undo()
    ctx.pick_up_external_changes()
    assert e.text == "someone else wrote this" and not e.dirty


def test_check_is_skipped_while_a_job_works(app, tmp_path, counting):
    ctx = _open(tmp_path, 6)
    _touch(Path(ctx.folder) / "img_001.txt", "x" * 50)
    ctx.set_job_state([ctx.session.entries[0].key], ctxmod.JOB_WORKING)
    ctx.pick_up_external_changes()
    assert ctx.session.entries[1].text != "x" * 50
    ctx.set_job_state([ctx.session.entries[0].key], None)
    ctx.pick_up_external_changes()
    assert ctx.session.entries[1].text == "x" * 50


def test_check_runs_on_a_worker_for_a_big_folder_and_a_stale_result_is_ignored(app, tmp_path, monkeypatch, counting):
    monkeypatch.setattr(ctxmod, "ASYNC_SCAN_ROWS", 0)
    ctx = _open(tmp_path, 10)
    folder = Path(ctx.folder)
    _touch(folder / "img_004.txt", "changed in the background check")
    ctx.pick_up_external_changes()
    assert ctx.session.entries[4].text != "changed in the background check"   # not on the calling thread
    for _ in range(100):
        _pump(app, 20)
        if not ctx._scan_busy:
            break
    assert ctx.session.entries[4].text == "changed in the background check"
    # a result that arrives after another folder was opened is dropped
    other = _folder(tmp_path, 6, name="other_dataset")
    _touch(folder / "img_005.txt", "late")
    ctx.pick_up_external_changes()
    assert ctx.open_folder(str(other), False, None)
    for _ in range(100):
        _pump(app, 20)
        if not ctx._scan_busy:
            break
    assert all(e.text != "late" for e in ctx.session.entries)


def test_subfolders_are_checked_too(app, tmp_path, counting):
    folder = tmp_path / "tree"
    (folder / "sub").mkdir(parents=True)
    make_image(folder / "a.png")
    make_image(folder / "sub" / "b.png")
    (folder / "sub" / "b.txt").write_text("deep caption", encoding="utf-8")
    ctx = ctxmod.WorkspaceContext()
    _CONTEXTS.append(ctx)
    assert ctx.open_folder(str(folder), True, None)
    counting.clear()
    ctx.pick_up_external_changes()
    assert counting == []
    _touch(folder / "sub" / "b.txt", "deep caption, rewritten")
    ctx.pick_up_external_changes()
    assert [e.text for e in ctx.session.entries if e.name == "b.png"] == ["deep caption, rewritten"]


# ------------------------------------------------------------------------------------- the delegate's caches
def _render(delegate, model, rows, width=220):
    height = width + delegate.text_height()
    out = []
    for r in rows:
        img = QImage(width, height, QImage.Format_ARGB32)
        img.fill(0xFF202020)
        p = QPainter(img)
        opt = QStyleOptionViewItem()
        opt.rect = QRect(0, 0, width, height)
        delegate.paint(p, opt, model.index(r))
        p.end()
        out.append(img)
    return out


class _NoThumbs:
    def pixmap(self, *a, **k):
        return None

    def is_failed(self, key):
        return False


def _delegate_model():
    texts = ["", "short", "1girl, solo, long hair, blue eyes, smile, looking at viewer, outdoors, sky, cloud, tree",
             "é " * 120, "word " * 200]
    entries = [Entry(path=Path(f"d/pic_{i}.png"), text=t, saved_text=t, width=640, height=480) for i, t in enumerate(texts)]
    entries[1].flags = {"blurry", "duplicate"}
    model = DatasetModel()
    model.set_session(DatasetSession(Path("d"), entries))
    return model, entries


def test_cache_hit_paints_the_same_pixels(app):
    model, entries = _delegate_model()
    d = CardDelegate(_NoThumbs())
    rows = range(len(entries))
    cold = _render(d, model, rows)
    assert d.cache_misses == len(entries) and d.cache_hits == 0
    warm = _render(d, model, rows)
    assert d.cache_hits == len(entries)                       # every caption came from the cache
    assert all(a == b for a, b in zip(cold, warm))
    fresh = _render(CardDelegate(_NoThumbs()), model, rows)   # and equals a delegate that never had a cache
    assert all(a == b for a, b in zip(cold, fresh))


def test_cache_follows_text_width_and_font(app):
    model, entries = _delegate_model()
    d = CardDelegate(_NoThumbs())
    _render(d, model, range(len(entries)))
    entries[2].text = "now a different caption, " * 10
    model.refresh([entries[2].key])
    after_edit = _render(d, model, [2])[0]
    assert after_edit == _render(CardDelegate(_NoThumbs()), model, [2])[0]
    for width in (140, 330, 220):                              # a width change invalidates; going back is right too
        d.set_card_width(width)
        got = _render(d, model, range(len(entries)), width)
        want_d = CardDelegate(_NoThumbs())
        want_d.set_card_width(width)
        assert got == _render(want_d, model, range(len(entries)), width)
    d.show_captions = False
    d.set_card_width(220)
    want_d = CardDelegate(_NoThumbs())
    want_d.show_captions = False
    assert _render(d, model, [0, 1], 220) == _render(want_d, model, [0, 1], 220)
    d._font.setPointSizeF(13)                                   # a font change (not through set_card_width) too
    d.show_captions = True
    want_d = CardDelegate(_NoThumbs())
    want_d._font.setPointSizeF(13)
    assert _render(d, model, range(len(entries))) == _render(want_d, model, range(len(entries)))


def test_cached_elision_equals_the_uncached_function(app):
    model, entries = _delegate_model()
    d = CardDelegate(_NoThumbs())
    fm, _ = d._metrics()
    rng = random.Random(2)
    for _ in range(60):
        e = Entry(path=Path(f"x/{rng.random()}.png"), text=" ".join(rng.choice(["a", "bb", "tag,", "é", "x" * 9])
                                                              for _ in range(rng.randrange(0, 80))))
        width = rng.choice((90, 160, 208, 400))
        want = d._elide_lines(e.text.strip() or "—", fm, width)
        assert d._elided_caption(e, fm, width) == want
        assert d._elided_caption(e, fm, width) == want        # the hit


# ----------------------------------------------------------------------------------------------- running counts
def _expected_status(b, ctx):
    s = ctx.session
    st = s.stats()
    shown = sum(1 for _ in range(b._proxy.rowCount()))
    selected = len(b.view.selectionModel().selectedIndexes())
    parts = [f"{shown} of {st['images']} shown" if shown != st["images"] else f"{st['images']} images",
             f"{selected} selected", f"{st['missing']} missing captions"]
    if st["unsaved"]:
        parts.append(f"● {st['unsaved']} unsaved")
    return "   ·   ".join(parts) + f"      {ctx.folder}", (f"Save ({st['unsaved']})" if st["unsaved"] else "Save")


def test_status_line_equals_a_recount_after_random_edits_and_selections(app, tmp_path):
    from tabs.workspace.browser import DatasetBrowser
    ctx = _open(tmp_path, 30, captioned=0.5)
    b = DatasetBrowser(ctx)
    b.resize(600, 400)
    b.show()
    _pump(app)
    rng = random.Random(8)
    keys = [e.key for e in ctx.session.entries]
    for step in range(120):
        op = rng.choice(["edit", "clear", "undo", "redo", "save", "select", "select_all", "deselect", "filter", "reload",
                         "remove", "jobstate"])
        pick = rng.sample([e.key for e in ctx.session.entries], min(len(ctx.session.entries), rng.randint(1, 6)))
        if op == "edit":
            ctx.push({k: (ctx.session.get(k).text, rng.choice(["x, y", "solo", "new caption"])) for k in pick}, "edit")
        elif op == "clear":
            b.select_keys(pick)
            b.run_operation("Paste caption", lambda t: "")
        elif op == "undo":
            ctx.undo()
        elif op == "redo":
            ctx.redo()
        elif op == "save":
            ctx.save(None)
        elif op == "select":
            b.select_keys(pick)
        elif op == "select_all":
            b.select_all()
        elif op == "deselect":
            b.deselect_all()
        elif op == "filter":
            b.set_filter(rng.choice(["", "missing:caption", "is:unsaved", "solo", "tag:x"]))
        elif op == "reload":
            for k in pick[:2]:
                p = ctx.session.get(k).path.with_suffix(".txt")
                p.write_text(rng.choice(["", "from disk"]), encoding="utf-8")
            ctx.reload_from_disk(pick)
        elif op == "remove" and len(ctx.session.entries) > 10:
            ctx.remove_entries(pick[:2])
        elif op == "jobstate":
            ctx.set_job_state(pick, rng.choice(["working", "failed", None]))
        _pump(app)
        counts, stats = ctx.model.counts(), ctx.session.stats()
        assert counts == stats, f"step {step} {op}"
        text, save = _expected_status(b, ctx)
        assert b.lbl_status.text() == text, f"step {step} {op}"
        assert b.btn_save.text() == save
    assert keys                                               # (the generator ran)


def test_a_burst_of_selection_changes_makes_one_signal(app, tmp_path):
    from PySide6.QtCore import QItemSelectionModel
    from tabs.workspace.browser import DatasetBrowser
    ctx = _open(tmp_path, 20)
    b = DatasetBrowser(ctx)
    b.resize(600, 400)
    b.show()
    _pump(app)
    seen = []
    b.selection_changed.connect(seen.append)
    sel = b.view.selectionModel()
    for r in range(10):
        sel.select(b._proxy.index(r, 0), QItemSelectionModel.Select)
    assert seen == []                                          # coalesced: nothing yet
    _pump(app)
    assert len(seen) == 1 and len(seen[0]) == 10
    assert "10 selected" in b.lbl_status.text()


def test_counts_do_not_walk_the_folder_per_update(app, tmp_path, monkeypatch):
    from PySide6.QtCore import QItemSelectionModel
    from tabs.workspace.browser import DatasetBrowser
    ctx = _open(tmp_path, 15)
    b = DatasetBrowser(ctx)
    b.resize(600, 400)
    b.show()
    _pump(app)
    walked = []
    monkeypatch.setattr(type(ctx.session), "stats", lambda self: walked.append(1) or {})
    for r in range(5):
        b.view.selectionModel().select(b._proxy.index(r, 0), QItemSelectionModel.Select)
        _pump(app)
    b.update_status()
    assert walked == []


# -------------------------------------------------------------------------------- the theme, applied afterwards
def test_deferred_stylesheet_ends_equal_to_applying_it_at_once(app):
    from tabs import theme
    from tabs.settings import apply_theme
    try:
        apply_theme("tagscriber_dark")
        direct_qss, direct_pal = app.styleSheet(), app.palette().color(app.palette().ColorRole.Window).name()
        apply_theme("dark_teal.xml")
        app.setStyleSheet("")                                   # (as at start-up: nothing applied yet)
        apply_theme("tagscriber_dark", defer_stylesheet=True)
        assert app.styleSheet() == "" and theme._pending_qss
        assert theme.is_token_theme("tagscriber_dark") and not theme.is_legacy()
        theme.finish_deferred(app)
        assert app.styleSheet() == direct_qss and theme._pending_qss is None
        assert app.palette().color(app.palette().ColorRole.Window).name() == direct_pal
        theme.finish_deferred(app)                              # a second call does nothing
        assert app.styleSheet() == direct_qss
        # a classic theme is applied at once, whatever the flag says
        apply_theme("dark_teal.xml", defer_stylesheet=True)
        assert app.styleSheet() and theme._pending_qss is None
    finally:
        apply_theme("dark_teal.xml")


def test_start_up_does_not_import_cv2_or_qt_material(tmp_path):
    import subprocess
    import sys
    code = ("import sys; sys.path.insert(0, '.'); import main; "
            "bad = [m for m in ('cv2', 'qt_material') if m in sys.modules]; print(bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                         env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "TAGSCRIBER_USER_DATA": str(tmp_path / "ud")}, cwd=str(Path(__file__).resolve().parent.parent))
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout + out.stderr

