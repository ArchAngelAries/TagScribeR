"""Train tab: builds offscreen, applies presets, and runs a tiny CPU training run end to end through QProcess."""
import os

import pytest

pytest.importorskip("torch")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_tab_builds_and_applies_first_preset(app):
    from tabs.train import TrainTab
    tab = TrainTab()
    assert tab.desc.key == "krea2"                                       # listed first: the primary model
    v = tab.collect()
    assert v["NETWORK_DIM"] == 8 and v["ADAPTIVE_LR"] is True and v["FAMILY_EMA"] == "0.98 (recommended)"
    assert v["DATASET_MEGAPIXELS"] == "0.25" and v["SAMPLE_STEPS"] == 28    # ✨ Krea 2 Ultra Fast on first visit
    tab.combo_family.setCurrentIndex(tab.combo_family.findData("qwen_image21"))
    assert tab.desc.key == "qwen_image21"
    v = tab.collect()
    assert v["NETWORK_DIM"] == 8 and v["ADAPTIVE_LR"] is True          # ✨ Qwen 2.1 Fast on first visit
    assert v["ADAPTIVE_LR_MIN"] == "2e-4 - rank 4/8 only" and v["DATASET_MEGAPIXELS"] == "0.5"
    assert tab.widgets["FAMILY_EDIT_DIR"][0].parentWidget().isHidden()
    tab._apply_preset("✨ Qwen 2.1 Style (rank 16, 1.5e-4)", quiet=True)
    v = tab.collect()
    assert v["NETWORK_DIM"] == 16 and v["ADAPTIVE_LR"] is False and v["LEARNING_RATE"] == 1.5e-4
    assert all(p.tip for p in __import__("training.params", fromlist=["PARAMS"]).PARAMS)


def test_every_tab_topic_and_shortcut(app):
    from tabs.help_content import TAB_TOPICS, TOPICS
    assert TAB_TOPICS[5] == "train" and "train_lr" in TOPICS and "train_watch" in TOPICS


def _wait(app, tab, states, timeout=180):
    import time
    t0 = time.time()
    while tab.state not in states or tab.proc is not None:
        app.processEvents()
        time.sleep(0.02)
        assert time.time() - t0 < timeout, f"timed out in state {tab.state}: {tab.lbl_status.text()}"


def test_real_child_processes_pause_and_resume(app, tmp_path, monkeypatch):
    """Start -> child processes (cache latents, cache text, train) on the tiny CPU family -> Pause -> Resume."""
    import json

    from core.config import settings
    from tests.tiny_family import make_dataset, register, write_models
    from training import params as P
    from training import pipeline
    monkeypatch.setenv("TAGSCRIBER_TRAINING_EXTRA_FAMILIES", "tests.tiny_family:register")
    monkeypatch.setenv("TAGSCRIBER_TRAINING_DEVICE", "cpu")
    tiny = register()
    data = make_dataset(str(tmp_path / "scratch"))
    for k, v in write_models(str(tmp_path / "models")).items():
        settings().set(pipeline.model_setting_key(k), v)
    from tabs import train as train_mod
    monkeypatch.setattr(train_mod, "confirm", lambda *a, **k: True)
    tab = train_mod.TrainTab()
    vals = {**P.defaults(), "LORA_OUTPUT_DIR": str(tmp_path / "runs"), "LORA_NAME": "qp", "NETWORK_DIM": 4,
            "MAX_TRAIN_EPOCHS": 2, "OPTIMIZER_TYPE": "adamw", "DATASET_MEGAPIXELS": "0.01",
            "SAMPLE_PROMPT": "tok", "SAMPLE_WIDTH": 64, "SAMPLE_HEIGHT": 64, "SAMPLE_STEPS": 2,
            "SAMPLE_AT_FIRST": False, "ADAPTIVE_LR": True, "ADAPTIVE_LR_MIN": "1e-4"}
    assert tab._launch(tiny, vals, data)
    pipeline.request_pause(tab.run.run_dir)          # honoured at the end of epoch 1
    _wait(app, tab, {"paused", "idle"})
    assert tab.state == "paused", tab.console.toPlainText()[-2000:]
    rd = tab.run.run_dir
    assert (rd / "qp-000001-state" / "training_state.json").exists()
    assert json.loads((rd / pipeline.PAUSED_SIDECAR).read_text())["family"] == "_tiny"
    assert tab.chart.points and tab.chart.points[0][0] == 1
    tab.resume_paused()
    _wait(app, tab, {"idle"})
    assert (rd / "qp.safetensors").exists(), tab.console.toPlainText()[-2000:]
    assert "Done" in tab.lbl_status.text()
    assert (rd / "run.log").read_text(encoding="utf-8").count("=== ") >= 3


def test_console_step_bar_updates_in_place(app):
    """The trainer's step bar is one console line that each update replaces (not a stale `0/1200` for the whole run)."""
    from tabs.train import TrainTab

    class _Proc:
        def __init__(self):
            self.chunks = []

        def readAllStandardOutput(self):
            return self.chunks.pop(0)

    tab = TrainTab()
    tab.proc = _Proc()
    tab.tracker.reset(20)
    tab.console.clear()
    # the trainer's log lines arrive glued to the bar's tail (tqdm ends its line with no newline)
    for chunk in (b"[ema] ON\nsteps:   0%|          | 0/1200 [00:00<?, ?it/s][warm-up] Warm-up phase - slow\n",
                  b"\rsteps:   0%|          | 1/1200 [00:04<1:21:54,  4.10s/it, avr_loss=0.1011]",
                  b"\rsteps:   0%|          | 2/1200 [00:07<1:11:20,  3.57s/it, avr_loss=0.1009]\n",
                  b"epoch 1/20  avr_loss=0.1009  step=60  3.50s/step  lr=2.828e-04  peak VRAM 16.1 GB\n"):
        tab.proc.chunks.append(chunk)
        tab._read()
    lines = tab.console.toPlainText().splitlines()
    assert lines[0] == "[ema] ON" and lines[1].startswith("steps:") and "0/1200" in lines[1]
    assert lines[2] == "[warm-up] Warm-up phase - slow"              # not swallowed by the bar it was glued to
    assert lines[3].startswith("steps:") and "2/1200" in lines[3] and lines[4].startswith("epoch 1/20")
    assert sum(l.startswith("steps:") for l in lines) == 2
    assert "step 2/1200" in tab.lbl_status.text()
    tab.proc = None
