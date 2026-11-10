"""Windows and the Train tab's trust in files from a run folder: windows that are destroyed after use, the loss graph's
incremental reader, text from files shown as text, the Hard images window staying inside the dataset, a paused run's
sidecar, imported presets with paths, and the two "for sharing" actions. Offscreen; no model, no trainer, no GPU."""
import gc
import json
import os
import sys
import types

import pytest

pytest.importorskip("torch")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6 import QtCore  # noqa: E402

from training import params as P  # noqa: E402
from training import pipeline, presets, step_log  # noqa: E402


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def tab(app, tmp_path, monkeypatch):
    import core.config as C
    from core.config import settings
    monkeypatch.setattr(C, "_settings", None)
    assert tmp_path in settings().path.parents
    from tabs.train import TrainTab
    t = TrainTab()
    t._set_widget("LORA_OUTPUT_DIR", str(tmp_path / "runs"))
    t._set_widget("LORA_NAME", "demo_lora")
    yield t
    t.timer.stop()
    t.loss_timer.stop()


def _flush():
    gc.collect()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    gc.collect()


def _no_boxes(monkeypatch):
    for n in ("warning", "information", "question", "critical"):
        monkeypatch.setattr(QtWidgets.QMessageBox, n, staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Ok))


# ---- M2: windows that are destroyed ----------------------------------------------------------------------
def test_command_palette_is_gone_after_each_use(app):
    from main import MainWindow
    from tabs.palette import CommandPalette

    class Host(QtWidgets.QWidget):
        def palette_commands(self):
            return []

    host = Host()
    for _ in range(6):
        MainWindow.open_palette(host)
        assert len(host.findChildren(CommandPalette)) == 1
        host._palette.close()
        _flush()
    MainWindow.open_palette(host)
    MainWindow.open_palette(host)                       # a second Ctrl+K replaces the first
    _flush()
    assert len(host.findChildren(CommandPalette)) == 1
    host._palette.close()
    _flush()
    assert host.findChildren(CommandPalette) == []
    assert getattr(host, "_palette", None) is None


def test_hard_images_window_is_reused_and_destroyed_on_close(tab, monkeypatch):
    from tabs.train_dialogs import ProblemImagesDialog
    for _ in range(6):
        tab.open_problem_images()
        tab.open_problem_images()                        # the same window comes forward, no second one
        dlgs = tab.findChildren(ProblemImagesDialog)
        assert len(dlgs) == 1
        timer = dlgs[0].timer
        assert timer.isActive()
        dlgs[0].close()
        assert not timer.isActive()                      # no poll runs for a closed window
        _flush()
        assert tab.findChildren(ProblemImagesDialog) == []


def test_menus_and_boxes_are_deleted_after_use(app, monkeypatch):
    from tabs import common
    from PySide6.QtWidgets import QMenu
    parent = QtWidgets.QWidget()
    monkeypatch.setattr(QMenu, "exec", lambda self, *a: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda self: QtWidgets.QMessageBox.Yes)
    for _ in range(5):
        assert common.confirm(parent, "t", "x") is True
    _flush()
    assert parent.findChildren(QtWidgets.QMessageBox) == []


# ---- L6: the step log is read incrementally ---------------------------------------------------------------
def _write(path, rows, mode="w"):
    with open(path, mode, encoding="utf-8") as f:
        for s, e, l in rows:
            f.write(json.dumps({"s": s, "e": e, "l": l, "lr": 1e-4}) + "\n")


def test_step_reader_parses_only_new_lines(tmp_path):
    p = tmp_path / "loss.jsonl"
    _write(p, [(i, i // 10, 1.0 / (i + 1)) for i in range(1, 51)])
    rd = step_log.StepReader(str(p), max_points=20)
    assert rd.poll() == step_log.read(str(p), 20)
    assert rd.parsed == 50
    _write(p, [(i, i // 10, 0.5) for i in range(51, 61)], "a")
    out = rd.poll()
    assert rd.parsed == 60                               # ten new lines, not sixty
    assert out == step_log.read(str(p), 20)
    assert rd.poll() == out and rd.parsed == 60          # nothing new: nothing parsed


def test_step_reader_leaves_a_partial_last_line_for_next_time(tmp_path):
    p = tmp_path / "loss.jsonl"
    _write(p, [(1, 0, 1.0), (2, 0, 0.9)])
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"s": 3, "e": 0, "l"')
    rd = step_log.StepReader(str(p))
    assert rd.poll()["steps"] == 2
    with open(p, "a", encoding="utf-8") as f:
        f.write(': 0.8}\n')
    out = rd.poll()
    assert out["steps"] == 3 and out["points"][-1][0] == 3
    assert out == step_log.read(str(p))


def test_step_reader_restarts_when_the_file_shrinks_or_is_replaced(tmp_path):
    p = tmp_path / "loss.jsonl"
    _write(p, [(i, 0, 1.0) for i in range(1, 41)])
    rd = step_log.StepReader(str(p))
    assert rd.poll()["steps"] == 40
    _write(p, [(i, 5, 2.0) for i in range(1, 6)])        # a fresh run: shorter
    assert rd.poll() == step_log.read(str(p))
    _write(p, [(i, 7, 3.0) for i in range(100, 150)])    # replaced by a longer, different file
    out = rd.poll()
    assert out == step_log.read(str(p)) and out["steps"] == 50
    os.remove(p)
    assert rd.poll() is None


def test_step_reader_thins_like_read(tmp_path):
    p = tmp_path / "loss.jsonl"
    rd = step_log.StepReader(str(p), max_points=100)
    for chunk in range(5):
        _write(p, [(chunk * 300 + i, chunk, 1.0 / (i + 1)) for i in range(1, 301)], "a")
        assert rd.poll() == step_log.read(str(p), 100)
    assert rd.poll()["stride"] > 1


def test_train_tab_chart_uses_the_incremental_reader(tab, tmp_path):
    rd = pipeline.run_dir_for(tab.collect())
    rd.mkdir(parents=True, exist_ok=True)
    name = tab._output_name()
    path = step_log.loss_path(str(rd), name)
    _write(path, [(i, 0, 1.0) for i in range(1, 21)])
    tab._refresh_loss(force=True)
    first = tab._step_reader
    assert first is not None and first.parsed == 20
    _write(path, [(i, 0, 1.0) for i in range(21, 31)], "a")
    tab._refresh_loss(force=True)
    assert tab._step_reader is first and first.parsed == 30
    assert tab.chart.steps == step_log.read(path)


# ---- H4 + L11: the Hard images window -----------------------------------------------------------------------
def _report(run_dir, images):
    d = run_dir / "loss_log"
    d.mkdir(parents=True, exist_ok=True)
    (d / "problem_images.json").write_text(json.dumps({"epoch": 3, "images": images}), encoding="utf-8")


def _png(path):
    from PySide6.QtGui import QImage
    path.parent.mkdir(parents=True, exist_ok=True)
    img = QImage(4, 4, QImage.Format_RGB32)
    img.fill(0)
    assert img.save(str(path))


def _select(dlg, key):
    for r in range(dlg.table.rowCount()):
        if dlg.table.item(r, 0).data(QtCore.Qt.UserRole) == key:
            dlg.table.selectRow(r)
            return
    raise AssertionError(key)


def test_hard_images_never_leaves_the_dataset_folder(app, tmp_path):
    from tabs.train_dialogs import ProblemImagesDialog
    data, outside, run = tmp_path / "data", tmp_path / "elsewhere", tmp_path / "run"
    _png(data / "a.png")
    _png(data / "sub" / "c.png")
    _png(outside / "b.png")
    (outside / "b.txt").write_text("secret", encoding="utf-8")
    bad = {"../elsewhere/b": 1, str(outside / "b"): 2, "sub/../../elsewhere/b": 3, "/elsewhere/b": 4}
    _report(run, {k: {"verdict": "stuck"} for k in ["a", "sub/c", *bad]})
    dlg = ProblemImagesDialog(str(run), ".txt", str(data))
    assert dlg._image_path("a") == os.path.join(str(data), "a.png")
    assert dlg._image_path("sub/c") is not None
    for k in bad:
        assert dlg._image_path(k) is None, k
    for k in bad:
        _select(dlg, k)
        assert "Not in this dataset" in dlg.lbl_verdict.text()
        dlg.edit.setPlainText("changed")
        dlg._save()
    assert (outside / "b.txt").read_text(encoding="utf-8") == "secret"
    assert not list(outside.glob("*.bak*")) and not list(tmp_path.glob("**/*.bak*"))
    dlg.close()
    dlg.deleteLater()
    _flush()


def test_hard_images_shows_report_text_as_text(app, tmp_path):
    from tabs.train_dialogs import ProblemImagesDialog
    run, data = tmp_path / "run", tmp_path / "data"
    data.mkdir()
    evil = '<img src="\\\\host\\share\\x.png">'
    _report(run, {"a": {"verdict": evil}})
    dlg = ProblemImagesDialog(str(run), ".txt", str(data))
    _select(dlg, "a")
    text = dlg.lbl_verdict.text()
    assert "<img" not in text and "&lt;img" in text
    assert dlg.lbl_status.textFormat() == QtCore.Qt.PlainText
    dlg.close()
    _flush()


# ---- M3: a paused run's sidecar ---------------------------------------------------------------------------
@pytest.fixture
def paused(tab, tmp_path, monkeypatch):
    _no_boxes(monkeypatch)
    rd = pipeline.run_dir_for(tab.collect())
    rd.mkdir(parents=True, exist_ok=True)
    launched = []
    monkeypatch.setattr(tab, "_launch", lambda desc, values, folder, **kw: launched.append((desc, values, folder, kw))
                        or True)
    notes = []
    monkeypatch.setattr(tab, "_console", notes.append)

    def sidecar(**over):
        st = rd / "demo_lora-state"
        st.mkdir(exist_ok=True)
        info = {"state": str(st), "family": tab.desc.key, "values": dict(tab.collect()), "folder": str(tmp_path),
                "mode": "state"}
        info.update(over)
        pipeline.write_paused(rd, info)
        tab._paused_dir = rd                       # what _restore_paused / a pause sets
        return info

    return types.SimpleNamespace(rd=rd, launched=launched, notes=notes, sidecar=sidecar)


def test_resume_with_a_good_sidecar_launches(tab, paused):
    info = paused.sidecar()
    tab.resume_paused()
    assert len(paused.launched) == 1
    desc, values, folder, kw = paused.launched[0]
    assert kw["resume"] == info["state"] and desc.key == tab.desc.key and values["LORA_NAME"] == "demo_lora"


def test_resume_refuses_a_state_outside_the_run_folder(tab, paused, tmp_path):
    other = tmp_path / "other_state"
    other.mkdir()
    paused.sidecar(state=str(other))
    tab.resume_paused()
    assert paused.launched == []
    assert any("not inside" in n for n in paused.notes)
    for bad in (str(paused.rd / ".." / "other_state"), "", None):
        paused.sidecar(state=bad)
        tab.resume_paused()
        assert paused.launched == []


def test_restore_paused_ignores_a_state_outside_the_run_folder(tab, paused, tmp_path):
    from tabs import train as T
    other = tmp_path / "other_state"
    other.mkdir()
    paused.sidecar(state=str(other))
    tab._restore_paused()
    assert tab.state == T.IDLE and any("not inside" in n for n in paused.notes)
    paused.sidecar()
    tab._restore_paused()
    assert tab.state == T.PAUSED


def test_resume_refuses_an_unknown_family(tab, paused):
    paused.sidecar(family="no_such_family")
    tab.resume_paused()
    assert paused.launched == []
    assert any("does not exist" in n for n in paused.notes)


def test_resume_drops_values_a_preset_would_refuse(tab, paused):
    vals = dict(tab.collect())
    vals["NETWORK_TYPE"] = "not a network type"
    vals["MADE_UP_KEY"] = "x"
    vals["LEARNING_RATE"] = "not a number"
    paused.sidecar(values=vals)
    tab.resume_paused()
    assert len(paused.launched) == 1
    out = paused.launched[0][1]
    assert out["NETWORK_TYPE"] != "not a network type" and "MADE_UP_KEY" not in out
    assert out["LEARNING_RATE"] != "not a number"
    notes = "\n".join(paused.notes)
    assert "NETWORK_TYPE" in notes and "MADE_UP_KEY" in notes


def test_resume_replaces_an_optimizer_the_family_does_not_offer(tab, paused):
    offered = P.options_for(P.BY_KEY["OPTIMIZER_TYPE"], tab.desc)
    vals = dict(tab.collect())
    vals["OPTIMIZER_TYPE"] = "somepackage.evil.Optimizer"
    vals["OPTIMIZER_ARGS"] = "x=1"
    paused.sidecar(values=vals)
    mine = tab.collect()["OPTIMIZER_TYPE"]
    assert mine in offered
    tab.resume_paused()
    out = paused.launched[0][1]
    assert out["OPTIMIZER_TYPE"] == mine and out["OPTIMIZER_ARGS"] == tab.collect()["OPTIMIZER_ARGS"]
    assert any("somepackage.evil.Optimizer" in n for n in paused.notes)


# ---- L10: a preset from Import with paths ------------------------------------------------------------------
def _preset_file(tmp_path, **kv):
    p = tmp_path / "friends_preset.json"
    p.write_text(json.dumps({"LEARNING_RATE": "1e-4", **kv}), encoding="utf-8")
    return p


@pytest.mark.parametrize("keep", [False, True])
def test_import_with_paths_asks_first(tab, tmp_path, monkeypatch, keep):
    assert tmp_path in tab.store.dir.parents
    f = _preset_file(tmp_path, LORA_OUTPUT_DIR="D:/somewhere/out", METADATA_THUMBNAIL="off")
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(f), "")))
    asked = []
    monkeypatch.setattr(tab, "_confirm_import_paths", lambda items: asked.append(items) or keep)
    tab._import_preset()
    assert [k for k, _v in asked[0]] == ["LORA_OUTPUT_DIR"]            # "off" is not a path
    stored = json.loads(tab.store.path(tab.combo_preset.currentText()).read_text(encoding="utf-8"))
    assert ("LORA_OUTPUT_DIR" in stored) is keep and stored["LEARNING_RATE"] == "1e-4"


def test_import_without_paths_does_not_ask(tab, tmp_path, monkeypatch):
    f = _preset_file(tmp_path)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(f), "")))
    monkeypatch.setattr(tab, "_confirm_import_paths", lambda items: pytest.fail("asked"))
    tab._import_preset()
    assert tab.combo_preset.currentText() == "friends_preset"


def test_import_dialog_defaults_to_without_paths(tab, monkeypatch):
    seen = {}

    def fake_exec(self):
        seen["default"] = self.defaultButton().text()
        seen["detail"] = self.informativeText()
        return 0
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", fake_exec)
    assert tab._confirm_import_paths([("LORA_OUTPUT_DIR", "D:/x")]) is False
    assert seen["default"] == "Apply without these paths" and "LORA_OUTPUT_DIR = D:/x" in seen["detail"]


# ---- H5: the two "for sharing" actions ----------------------------------------------------------------------
def test_copy_log_for_sharing_scrubs_with_this_runs_folders(tab, tmp_path, monkeypatch):
    seen = {}

    def fake_scrub(text, extra=()):
        seen["extra"] = list(extra)
        return "SCRUBBED:" + text

    fake = types.ModuleType("core.privacy")
    fake.scrub = fake_scrub
    monkeypatch.setitem(sys.modules, "core.privacy", fake)
    first = next(iter(tab.model_edits))
    model_dir = tmp_path / "models"
    tab.model_edits[first].setText(str(model_dir / "m.safetensors"))
    tab.console.setPlainText("line one\nline two")
    tab.btn_copy_log.click()
    assert QtWidgets.QApplication.clipboard().text() == "SCRUBBED:line one\nline two"
    extra = seen["extra"]
    assert str(pipeline.run_dir_for(tab.collect())) in extra
    assert str(tmp_path / "runs") in extra and str(model_dir) in extra
    assert tab.btn_copy_log.toolTip()


def test_export_for_sharing_writes_what_the_function_returns(tab, tmp_path, monkeypatch):
    got = {}

    def fake_export(values):
        got["values"] = dict(values)
        return {"LEARNING_RATE": "1e-4"}
    monkeypatch.setattr(presets, "export_for_sharing", fake_export, raising=False)
    out = tmp_path / "shared.json"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    tab._export_for_sharing()
    assert json.loads(out.read_text(encoding="utf-8")) == {"LEARNING_RATE": "1e-4"}
    assert got["values"].get("LORA_NAME") == "demo_lora"          # the function does the leaving out, not the tab
    export_btn = [b for b in tab.findChildren(QtWidgets.QPushButton) if b.text().startswith("Export for sharing")]
    assert len(export_btn) == 1 and export_btn[0].toolTip()
