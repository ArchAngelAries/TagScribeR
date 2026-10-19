"""The Train tab's side of Fizgig v8 wave 2: the per-step loss graph, Sample now / Save now, the prompt slider's live
prompts, each preview's prompt, the Distilled tick's Steps / CFG follow, and the multi-reference Edit controls.
Offscreen, on the real family descriptions but no model: nothing here loads weights or starts a trainer."""
import json
import os
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6 import QtCore  # noqa: E402

from training import params as P  # noqa: E402
from training import pipeline  # noqa: E402


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def tab(app, tmp_path, monkeypatch):
    import core.config as C
    from core.config import settings
    monkeypatch.setattr(C, "_settings", None)      # a fresh Settings on the isolated temp user_data, never the real one
    assert tmp_path in settings().path.parents          # the isolated user_data (tests/conftest.py), not the owner's
    from tabs.train import TrainTab
    t = TrainTab()
    t._set_widget("LORA_OUTPUT_DIR", str(tmp_path / "runs"))
    t._set_widget("LORA_NAME", "demo_lora")
    yield t
    t.timer.stop()
    t.loss_timer.stop()


def _fake_run(tab, tmp_path, stage=1, state="running"):
    """A live run without a process: a cache stage then Training, the run folder frozen the way build_run does."""
    from tabs import train as T
    rd = pipeline.run_dir_for(tab.collect())
    rd.mkdir(parents=True, exist_ok=True)
    tab.run = pipeline.Run("krea2", rd, "demo_lora", [("Caching latents", []), ("Training", [])], 4, "")
    tab.stage_index = stage
    tab._set_state(state if state != "running" else T.RUNNING)
    return rd


# ---- params ---------------------------------------------------------------------------------------------
def test_edit_pictures_params_match_the_pipeline():
    assert (P.EDIT_PHOTO_ONLY, P.EDIT_WITH_EXTRAS) == (pipeline.EDIT_PHOTO_ONLY, pipeline.EDIT_WITH_EXTRAS)
    pic, xd = P.BY_KEY["FAMILY_EDIT_PICTURES"], P.BY_KEY["FAMILY_EDIT_EXTRAS_DIR"]
    assert pic.kind == P.CHOICE and pic.default == P.EDIT_PHOTO_ONLY and pic.label == "What the edit uses"
    assert pic.options == (P.EDIT_PHOTO_ONLY, P.EDIT_WITH_EXTRAS) and pic.strict
    assert xd.kind == P.DIR and xd.label == "Extra pictures folder" and xd.default == ""
    assert "FAMILY_EDIT_EXTRAS_DIR" in P.PATH_KEYS
    assert "photo_1.png" in pic.tip and "Originals folder" in pic.tip and "photo.txt" in pic.tip
    assert "beach_1.png" in xd.tip and "test_1.png" in xd.tip


def test_edit_pictures_shown_only_for_multi_reference_families():
    from training.registry import get
    shown = {k: P.family_shows(P.BY_KEY["FAMILY_EDIT_PICTURES"], get(k)) for k in ("klein9b", "qwen_image21", "krea2",
                                                                                 "sdxl", "anima", "minimax_h3")}
    assert shown == {"klein9b": True, "qwen_image21": True, "krea2": False, "sdxl": False, "anima": False,
                     "minimax_h3": False}
    assert P.family_shows(P.BY_KEY["FAMILY_EDIT_EXTRAS_DIR"], get("klein9b"))


def test_edit_extras_rows_follow_the_edit_switch_and_the_choice(tab):
    tab.chk_all.setChecked(True)
    for fam, shows in (("klein9b", True), ("qwen_image21", True), ("krea2", False)):
        tab.combo_family.setCurrentIndex(tab.combo_family.findData(fam))
        pic = tab.widgets["FAMILY_EDIT_PICTURES"][0]
        xd = tab.widgets["FAMILY_EDIT_EXTRAS_DIR"][0].parentWidget()
        assert pic.isHidden() and xd.isHidden()                       # Edit LoRA is off
        if fam == "krea2":
            assert tab.widgets["FAMILY_EDIT"][0].isHidden()
            continue
        tab._set_widget("FAMILY_EDIT", True)
        tab._changed()
        assert pic.isHidden() == (not shows) and xd.isHidden()         # photo only: no folder row
        tab._set_widget("FAMILY_EDIT_PICTURES", P.EDIT_WITH_EXTRAS)
        tab._changed()
        assert not xd.isHidden()
        tab._set_widget("FAMILY_EDIT", False)
        tab._changed()
        assert pic.isHidden() and xd.isHidden()


def test_a_preset_carrying_the_extras_keys_applies(tab):
    from training import presets
    tab.combo_family.setCurrentIndex(tab.combo_family.findData("klein9b"))
    new, rep = presets.apply({"FAMILY_EDIT_PICTURES": "The photo + extra pictures",
                              "FAMILY_EDIT_EXTRAS_DIR": "D:\\Datasets\\demo_extras"}, tab.collect(), tab.desc)
    assert new["FAMILY_EDIT_PICTURES"] == P.EDIT_WITH_EXTRAS and new["FAMILY_EDIT_EXTRAS_DIR"].endswith("demo_extras")
    assert any("FAMILY_EDIT_EXTRAS_DIR" in m for m in rep.messages())      # a folder from a preset is flagged


# ---- the Distilled tick -----------------------------------------------------------------------------------
def test_tick_label_comes_from_the_family_and_keeps_klein_s_text():
    assert P.preview_checkpoint_tick(SimpleNamespace()) == P.SAMPLE_CHECKPOINT_TICK
    assert P.preview_checkpoint_tick(SimpleNamespace(preview_checkpoint_tick="Use the other model")) == \
        "Use the other model"
    assert P.SAMPLE_CHECKPOINT_TICK == "Use Distilled model for samples (4-step, matches ComfyUI)"


def test_tick_label_in_the_tab(tab):
    tab.combo_family.setCurrentIndex(tab.combo_family.findData("klein9b"))
    assert tab.widgets["SAMPLE_USE_DISTILLED"][0].text() == P.SAMPLE_CHECKPOINT_TICK


def _klein_with_file(tab, tmp_path):
    tab.combo_family.setCurrentIndex(tab.combo_family.findData("klein9b"))
    f = tmp_path / "distilled.safetensors"
    f.write_bytes(b"x")
    key = tab.desc.preview_checkpoint()[0].pref_key
    tab.model_edits[key].setText(str(f))
    return tab.desc


def test_distilled_tick_moves_steps_and_cfg_with_the_file_set(tab, tmp_path):
    desc = _klein_with_file(tab, tmp_path)
    ck = desc.preview_checkpoint_sampling
    w = lambda k: tab._get_widget(k)  # noqa: E731
    tab._set_widget("SAMPLE_USE_DISTILLED", True)
    tab._set_widget("SAMPLE_STEPS", desc.preview_steps)
    tab._set_widget("SAMPLE_CFG_SCALE", desc.preview_cfg)
    tab._set_widget("SAMPLE_USE_DISTILLED", False)
    tab._set_widget("SAMPLE_USE_DISTILLED", True)             # ticked: the checkpoint's recipe
    assert (w("SAMPLE_STEPS"), w("SAMPLE_CFG_SCALE")) == (ck.steps, ck.cfg)
    tab._set_widget("SAMPLE_USE_DISTILLED", False)            # unticked: the training model's
    assert (w("SAMPLE_STEPS"), w("SAMPLE_CFG_SCALE")) == (desc.preview_steps, desc.preview_cfg)


def test_distilled_tick_never_replaces_a_typed_value(tab, tmp_path):
    desc = _klein_with_file(tab, tmp_path)
    tab._set_widget("SAMPLE_USE_DISTILLED", False)
    tab._set_widget("SAMPLE_STEPS", 12)                        # typed: not one of the known defaults
    tab._set_widget("SAMPLE_CFG_SCALE", 2.5)
    tab._set_widget("SAMPLE_USE_DISTILLED", True)
    assert tab._get_widget("SAMPLE_STEPS") == 12 and tab._get_widget("SAMPLE_CFG_SCALE") == 2.5
    tab._set_widget("SAMPLE_USE_DISTILLED", False)
    assert tab._get_widget("SAMPLE_STEPS") == 12 and tab._get_widget("SAMPLE_CFG_SCALE") == 2.5
    # one box typed, the other on a default: only the default one moves
    tab._set_widget("SAMPLE_STEPS", 12)
    tab._set_widget("SAMPLE_CFG_SCALE", desc.preview_cfg)
    tab._set_widget("SAMPLE_USE_DISTILLED", True)
    assert tab._get_widget("SAMPLE_STEPS") == 12 and tab._get_widget("SAMPLE_CFG_SCALE") == \
        desc.preview_checkpoint_sampling.cfg


def test_empty_boxes_follow_and_no_file_means_no_move(tab, tmp_path):
    desc = tab.desc
    tab.combo_family.setCurrentIndex(tab.combo_family.findData("klein9b"))
    desc = tab.desc
    tab._set_widget("SAMPLE_USE_DISTILLED", False)
    tab._set_widget("SAMPLE_STEPS", desc.preview_steps)
    tab._set_widget("SAMPLE_USE_DISTILLED", True)              # no file set: the training model renders, no move
    assert tab._get_widget("SAMPLE_STEPS") == desc.preview_steps
    key = desc.preview_checkpoint()[0].pref_key
    f = tmp_path / "d.safetensors"
    f.write_bytes(b"x")
    tab.model_edits[key].setText(str(f))                       # the file arrives: the tick is live now
    assert tab._get_widget("SAMPLE_STEPS") == desc.preview_checkpoint_sampling.steps
    tab._set_widget("SAMPLE_USE_DISTILLED", False)
    tab._set_widget("SAMPLE_STEPS", 0)                         # 0 = empty = the family default
    tab._set_widget("SAMPLE_USE_DISTILLED", True)
    assert tab._get_widget("SAMPLE_STEPS") == desc.preview_checkpoint_sampling.steps


def test_a_form_loading_on_the_other_sides_default_is_put_right(tab, tmp_path):
    desc = _klein_with_file(tab, tmp_path)
    ck = desc.preview_checkpoint_sampling
    tab._apply_values({"SAMPLE_USE_DISTILLED": True, "SAMPLE_STEPS": desc.preview_steps,
                       "SAMPLE_CFG_SCALE": desc.preview_cfg})
    assert (tab._get_widget("SAMPLE_STEPS"), tab._get_widget("SAMPLE_CFG_SCALE")) == (ck.steps, ck.cfg)
    tab._apply_values({"SAMPLE_USE_DISTILLED": False, "SAMPLE_STEPS": ck.steps, "SAMPLE_CFG_SCALE": ck.cfg})
    assert (tab._get_widget("SAMPLE_STEPS"), tab._get_widget("SAMPLE_CFG_SCALE")) == (desc.preview_steps,
                                                                                    desc.preview_cfg)
    tab._apply_values({"SAMPLE_USE_DISTILLED": True, "SAMPLE_STEPS": 9})        # typed: stays on load
    assert tab._get_widget("SAMPLE_STEPS") == 9


def test_other_families_are_untouched_by_the_follow(tab):
    tab.combo_family.setCurrentIndex(tab.combo_family.findData("krea2"))
    before = (tab._get_widget("SAMPLE_STEPS"), tab._get_widget("SAMPLE_CFG_SCALE"))
    tab._follow_update(loading=True)
    assert (tab._get_widget("SAMPLE_STEPS"), tab._get_widget("SAMPLE_CFG_SCALE")) == before


# ---- the loss graph -----------------------------------------------------------------------------------------
def _write_steps(out_dir, name, n=60, per_epoch=20, start=0):
    from training.step_log import StepLog
    log = StepLog(str(out_dir), name, start_step=start)
    for i in range(start + 1, n + 1):
        log.add(i, (i - 1) // per_epoch + 1, 1.0 / (1 + i / 10) + 0.01 * (i % 3), 1e-4 if i % 20 == 0 else None)
    log.flush()


def test_loss_chart_draws_steps_epoch_averages_and_reads_a_step(app, tmp_path):
    from tabs.train_dialogs import LossChart
    _write_steps(tmp_path, "n")
    data = pipeline.read_step_log(tmp_path, "n")
    ch = LossChart()
    ch.resize(500, 220)
    ch.grab()                                                     # nothing yet: the placeholder
    ch.set_steps(data)
    for e, loss in ((1, 0.5), (2, 0.3), (3, 0.2)):
        ch.add(e, loss, 1e-4)
    ch.mark(2, "UP x1.25")
    assert ch._epoch_span() == {1: (1, 20), 2: (21, 40), 3: (41, 60)}
    ch._hover = 29
    assert "step 30 · epoch 2 · loss" in ch.readout(29) and "smoothed" in ch.readout(29)
    assert not ch.grab().isNull()
    changed = []
    ch.update = lambda *a: changed.append(1)                      # an identical curve is not redrawn
    ch.set_steps(pipeline.read_step_log(tmp_path, "n"))
    assert not changed
    ch.set_steps({"points": [], "epochs": [], "steps": 0, "stride": 1})
    assert ch.steps is None
    ch.clear()
    assert ch.points == [] and ch.steps is None


def test_loss_chart_without_a_step_log_is_the_epoch_chart(app):
    from tabs.train_dialogs import LossChart
    ch = LossChart()
    ch.resize(400, 200)
    ch.add(1, 0.5, 1e-4)
    ch.add(2, 0.4, 1e-4)
    assert not ch.grab().isNull() and ch.steps is None


def test_tab_polls_the_step_log_and_skips_an_unchanged_one(tab, tmp_path, monkeypatch):
    from tabs import train as T
    rd = _fake_run(tab, tmp_path)
    tab.show()
    _write_steps(rd, "demo_lora", 40)
    tab._refresh_loss(force=True)
    assert tab.chart.steps and tab.chart.steps["steps"] == 40
    calls = []
    real = pipeline.read_step_log
    monkeypatch.setattr(pipeline, "read_step_log", lambda *a, **k: calls.append(1) or real(*a, **k))
    tab._refresh_loss(force=True)
    assert not calls                                              # same file: not read again
    _write_steps(rd, "demo_lora", 55, start=40)                   # the resumed run carries on its curve
    tab._refresh_loss(force=True)
    assert calls and tab.chart.steps["steps"] == 55
    tab.stage_index = 0                                           # a cache stage: the file is not the run's yet
    tab.proc = QtCore.QProcess()
    calls.clear()
    (rd / ".tagscriber_loss_demo_lora.jsonl").write_text("")
    tab._refresh_loss(force=True)
    assert not calls and tab.state == T.RUNNING
    tab.proc = None


def test_a_finished_run_shows_its_curve_and_another_name_clears_it(tab, tmp_path):
    from tabs import train as T
    tab.show()
    rd = pipeline.run_dir_for(tab.collect())
    rd.mkdir(parents=True)
    _write_steps(rd, "demo_lora", 30)
    tab.state = T.IDLE
    tab._refresh_loss(force=True)
    assert tab.chart.steps["steps"] == 30
    tab._set_widget("LORA_NAME", "other_lora")
    tab._changed()
    tab._refresh_loss(force=True)
    assert tab.chart.steps is None


# ---- Sample now / Save now -------------------------------------------------------------------------------------
def test_now_buttons_follow_the_stage_and_the_previews(tab, tmp_path):
    from tabs import train as T
    rd = _fake_run(tab, tmp_path, stage=0)                         # the cache stage
    assert not tab.btn_sample_now.isEnabled() and not tab.btn_save_now.isEnabled()
    tab.stage_index = 1
    tab._set_state(T.RUNNING)
    assert tab.btn_save_now.isEnabled() and not tab.btn_sample_now.isEnabled()      # no previews in the run folder
    assert "not set up" in tab.btn_sample_now.toolTip()
    (rd / "train_config.json").write_text(json.dumps({"train": {"sample_every_n_epochs": 0}}))
    (rd / "sample_prompts.txt").write_text("a photo\n")
    tab._set_state(T.RUNNING)
    assert not tab.btn_sample_now.isEnabled()                      # Every N epochs = 0
    (rd / "train_config.json").write_text(json.dumps({"train": {"sample_every_n_epochs": 2}}))
    tab._set_state(T.RUNNING)
    assert tab.btn_sample_now.isEnabled()
    (rd / "sample_prompts.txt").write_text("\n")
    tab._set_state(T.RUNNING)
    assert not tab.btn_sample_now.isEnabled()                      # no prompt
    tab._set_state(T.PAUSED)
    assert not tab.btn_save_now.isEnabled()
    assert "Ctrl+Shift+S" in tab.btn_sample_now.toolTip() and "Ctrl+Shift+D" in tab.btn_save_now.toolTip()


def test_sample_now_and_save_now_drop_their_files_and_say_so(tab, tmp_path):
    from tabs import train as T
    rd = _fake_run(tab, tmp_path)
    (rd / "train_config.json").write_text(json.dumps({"train": {"sample_every_n_epochs": 1}}))
    (rd / "sample_prompts.txt").write_text("a photo\n")
    tab._set_state(T.RUNNING)
    tab.sample_now()
    tab.save_now()
    assert (rd / ".sample_now").exists() and (rd / ".save_now").exists()
    text = tab.console.toPlainText()
    assert "[sample now] requested" in text and "[save now] requested" in text and "-stepNNNNNNN" in text
    tab._set_state(T.PAUSED)
    (rd / ".sample_now").unlink()
    tab.sample_now()                                               # disabled: nothing is asked
    assert not (rd / ".sample_now").exists()


def test_save_nows_file_is_in_the_checkpoint_list(tab, tmp_path):
    rd = _fake_run(tab, tmp_path)
    tab.show()
    (rd / "demo_lora-000001.safetensors").write_bytes(b"x" * 10)
    (rd / "demo_lora-step0000123.safetensors").write_bytes(b"x" * 10)
    (rd / "other-step0000123.safetensors").write_bytes(b"x")
    tab._refresh_checkpoints()
    rows = [tab.checkpoints.item(i).text() for i in range(tab.checkpoints.count())]
    assert len(rows) == 2 and any("-step0000123" in r for r in rows)


# ---- the prompt slider's live prompts ---------------------------------------------------------------------------
def _slider_values(tab):
    tab.combo_family.setCurrentIndex(tab.combo_family.findData("krea2"))
    v = tab.collect()
    v.update(FAMILY_SLIDER=True, FAMILY_SLIDER_SOURCE="prompts", FAMILY_SLIDER_BASE="a photo of a woman",
             FAMILY_SLIDER_POS="smiling", FAMILY_SLIDER_NEG="frowning", FAMILY_SLIDER_GUIDANCE=0.0)
    return v


def test_slider_prompts_button_only_for_a_prompt_slider_run(tab, tmp_path):
    from tabs import train as T
    _fake_run(tab, tmp_path)
    tab._run_ctx = (tab.desc, tab.collect(), "")
    tab._set_state(T.RUNNING)
    assert not tab.btn_slider.isVisibleTo(tab)
    tab._run_ctx = (tab.desc, _slider_values(tab), "")
    tab._set_state(T.RUNNING)
    assert tab.btn_slider.isVisibleTo(tab) and tab.btn_slider.isEnabled()
    v = _slider_values(tab)
    v["FAMILY_SLIDER_SOURCE"] = "pairs"                             # an image-pair slider has no live prompts
    tab._run_ctx = (tab.desc, v, "")
    tab._set_state(T.RUNNING)
    assert not tab.btn_slider.isVisibleTo(tab)


def test_slider_dialog_writes_the_prompts_and_shows_what_was_applied(app, tmp_path):
    from tabs.train_dialogs import SliderPromptsDialog
    cur = {"base": "a photo of a woman", "pos": "smiling", "neg": "frowning", "guidance": 2.0}
    d = SliderPromptsDialog(str(tmp_path), cur)
    assert d.prompts() == cur and "run started with" in d.lbl_state.text()
    assert d.apply() and not (tmp_path / ".slider_prompts.json").exists()   # nothing changed: nothing to send
    d.pos.setText("grinning")
    d.guidance.setText("3")
    assert d.apply()
    sent = json.loads((tmp_path / ".slider_prompts.json").read_text())
    assert sent == {"base": "a photo of a woman", "pos": "grinning", "neg": "frowning", "guidance": 3.0}
    assert "Waiting" in d.lbl_state.text()
    (tmp_path / ".slider_prompts_applied.json").write_text(json.dumps({**sent, "epoch": 4,
                                                                       "practice_rerendered": False, "at": 0}))
    d.refresh_state()
    assert "since epoch 5" in d.lbl_state.text() and "Waiting" not in d.lbl_state.text()
    d.base.setText("a photo of a man")
    d.apply()
    (tmp_path / ".slider_prompts_applied.json").write_text(json.dumps({
        **json.loads((tmp_path / ".slider_prompts.json").read_text()), "epoch": 6, "practice_rerendered": True}))
    d.refresh_state()
    assert "since epoch 7" in d.lbl_state.text() and "rendered again" in d.lbl_state.text()
    d.pos.setText("")
    assert d.prompts() is None
    d.timer.stop()


def test_slider_dialog_refuses_a_partial_set(app, tmp_path, monkeypatch):
    from tabs import train_dialogs as D
    shown = []
    monkeypatch.setattr(D.QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
    d = D.SliderPromptsDialog(str(tmp_path), {"base": "b", "pos": "p", "neg": "", "guidance": 2.0})
    assert d.apply() is False and shown and not (tmp_path / ".slider_prompts.json").exists()
    d.timer.stop()


# ---- the Samples view -------------------------------------------------------------------------------------------
def test_each_preview_shows_the_prompt_it_was_made_with(tab, tmp_path):
    from PIL import Image
    rd = pipeline.run_dir_for(tab.collect())
    sd = rd / "sample"
    sd.mkdir(parents=True)
    for epoch, idx in ((0, 0), (2, 0), (2, 1)):
        Image.new("RGB", (16, 16), (10, 20, 30)).save(sd / f"demo_lora_e{epoch:06d}_{idx:02d}_20260101000000_{1234 + idx}.png")
    (sd / ".prompts_demo_lora.json").write_text(json.dumps({"0": ["a photo"], "2": ["new prompt A", "new prompt B"]}))
    tab.show()
    tab._refresh_samples()
    items = {tab.samples.item(i).text().splitlines()[0]: tab.samples.item(i) for i in range(tab.samples.count())}
    assert set(items) == {"epoch 0 #0", "epoch 2 #0", "epoch 2 #1"}
    assert "new prompt B" in items["epoch 2 #1"].text() and "Prompt: new prompt B" in items["epoch 2 #1"].toolTip()
    assert "Prompt: a photo" in items["epoch 0 #0"].toolTip() and "seed 1235" in items["epoch 2 #1"].toolTip()
    (sd / ".prompts_demo_lora.json").write_text(json.dumps({"0": ["a photo"], "2": ["changed", "again"]}))
    tab._refresh_samples()                                           # a prompts record that changed redraws the list
    assert any("changed" in tab.samples.item(i).text() for i in range(tab.samples.count()))


def test_a_preview_without_a_record_shows_no_prompt(tab, tmp_path):
    from PIL import Image
    rd = pipeline.run_dir_for(tab.collect())
    (rd / "sample").mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(rd / "sample" / "demo_lora_e000001_00_20260101000000_1.png")
    tab.show()
    tab._refresh_samples()
    assert tab.samples.count() == 1 and tab.samples.item(0).text() == "epoch 1 #0"
    assert "Prompt:" not in tab.samples.item(0).toolTip()


# ---- help -------------------------------------------------------------------------------------------------------
def test_help_lists_the_new_hotkeys_and_the_section():
    from tabs.help_content import HOTKEYS, TOPICS
    keys = {k for _scope, k, _a in HOTKEYS}
    assert {"Ctrl+Shift+S", "Ctrl+Shift+D", "Ctrl+Shift+L"} <= keys
    assert len([k for _s, k, _a in HOTKEYS if k == "Ctrl+Shift+S"]) == 1
    title, group, html = TOPICS["train_live"]
    assert group == "Train"
    for needle in ("Sample now", "Save now", ".tagscriber_loss_", "sample/.prompts_", "-step", "Extra pictures",
                   "photo_1.png", "test_1.png", "Compile Blocks", "sample override", "resume", "not yet been run"):
        assert needle in html, needle


# ---- review fixes -------------------------------------------------------------------------------------------
def test_a_fresh_run_drops_the_last_runs_slider_files_a_resume_keeps_them(tab, tmp_path, monkeypatch):
    from tabs import train as T
    rd = pipeline.run_dir_for(tab.collect())
    rd.mkdir(parents=True)
    run = pipeline.Run("krea2", rd, "demo_lora", [("Training", [])], 2, "")
    monkeypatch.setattr(pipeline, "preflight", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "build_run", lambda *a, **k: run)
    monkeypatch.setattr(tab, "_start_stage", lambda: None)
    for resume, kept in (("", False), (str(tmp_path / "state"), True)):
        for n in (pipeline.SLIDER_LIVE_FILE, pipeline.SLIDER_LIVE_APPLIED_FILE):
            (rd / n).write_text("{}")
        tab.state = T.IDLE
        assert tab._launch(tab.desc, tab.collect(), str(tmp_path), resume)
        assert all((rd / n).exists() == kept for n in (pipeline.SLIDER_LIVE_FILE, pipeline.SLIDER_LIVE_APPLIED_FILE))
        if tab._log_file:
            tab._log_file.close()
            tab._log_file = None


def test_slider_dialog_refuses_when_no_slider_run_is_going(app, tmp_path, monkeypatch):
    from tabs import train_dialogs as D
    shown = []
    monkeypatch.setattr(D.QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
    live = [True]
    d = D.SliderPromptsDialog(str(tmp_path), {"base": "b", "pos": "p", "neg": "n", "guidance": 2.0},
                              is_live=lambda: live[0])
    d.pos.setText("q")
    live[0] = False
    assert d.apply() is False and shown and not (tmp_path / ".slider_prompts.json").exists()
    d.close()
    assert not d.timer.isActive()


def test_slider_window_is_reused_and_closed_when_the_run_ends(tab, tmp_path):
    from tabs import train as T
    _fake_run(tab, tmp_path)
    tab._run_ctx = (tab.desc, _slider_values(tab), "")
    tab._set_state(T.RUNNING)
    tab.open_slider_prompts()
    first = tab._slider_dlg
    assert first is not None
    tab.open_slider_prompts()
    assert tab._slider_dlg is first
    tab._set_state(T.PAUSED)                                        # the run is over
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    assert tab._slider_dlg is None


def test_edit_rows_are_numbered_like_fizgigs_once_extras_are_chosen(tab):
    from tabs.train import EDIT_STEP_LABELS
    tab.chk_all.setChecked(True)
    tab.combo_family.setCurrentIndex(tab.combo_family.findData("klein9b"))
    keys = [p.key for p in P.PARAMS if p.key in ("FAMILY_EDIT_PICTURES", "FAMILY_EDIT_DIR", "FAMILY_EDIT_EXTRAS_DIR")]
    assert keys == ["FAMILY_EDIT_PICTURES", "FAMILY_EDIT_DIR", "FAMILY_EDIT_EXTRAS_DIR"]
    tab._set_widget("FAMILY_EDIT", True)
    tab._changed()
    assert tab.widgets["FAMILY_EDIT_DIR"][1].text() == "Originals folder:"
    tab._set_widget("FAMILY_EDIT_PICTURES", P.EDIT_WITH_EXTRAS)
    tab._changed()
    for k, text in EDIT_STEP_LABELS.items():
        assert tab.widgets[k][1].text() == text + ":"
    assert EDIT_STEP_LABELS["FAMILY_EDIT_DIR"] == "1. Originals folder (before editing)"
    tab._set_widget("FAMILY_EDIT_PICTURES", P.EDIT_PHOTO_ONLY)
    tab._changed()
    assert tab.widgets["FAMILY_EDIT_DIR"][1].text() == "Originals folder:"
    assert tab.widgets["FAMILY_EDIT_CAPTION"][1].text() == "Edit instruction:"
