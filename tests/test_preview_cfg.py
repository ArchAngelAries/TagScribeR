"""Previews through a Turbo LoRA render at the Turbo's own CFG unless the user really typed one.

The bug this pins (found on a real run, 2026-11-19): Krea 2's default preview CFG had once been 5.5. A saved form
still held 5.5, the trainer took it for a typed value and rendered the Turbo previews at CFG 5.5: harsh,
over-saturated pictures from the first preview on, before any training."""
import pytest

from training import params as P
from training import pipeline, registry


@pytest.fixture
def krea(tmp_path):
    desc = registry.get("krea2")
    models = {}
    for f in desc.model_files:
        p = tmp_path / f"{f.pref_key}.safetensors"
        p.write_bytes(b"0")
        models[f.pref_key] = str(p)
    vals = {**P.defaults(), "LORA_OUTPUT_DIR": str(tmp_path / "runs"), "LORA_NAME": "k", "SAMPLE_PROMPT": "a cat",
            "SAMPLE_STEPS": desc.preview_steps, "SAMPLE_CFG_SCALE": desc.preview_cfg, "SAMPLE_NEGATIVE": "blurry"}
    return desc, models, vals, tmp_path


def _kw(desc, vals, models, tmp):
    return pipeline.train_kwargs(desc, vals, tmp / "run", models)[0]


def test_a_former_default_cfg_is_not_a_typed_one(krea):
    desc, models, vals, tmp = krea
    turbo = desc.preview_speed()
    assert 5.5 in desc.preview_cfg_retired and turbo.settings.cfg <= 1.0
    for box in (desc.preview_cfg, 5.5, 0, ""):                 # today's default, the old default, empty
        kw = _kw(desc, {**vals, "SAMPLE_CFG_SCALE": box}, models, tmp)
        assert kw["speed_lora"] and kw["sample_cfg_scale"] == turbo.settings.cfg, box
        assert kw["sample_steps"] == desc.preview_speed_defaults()[0] and "sample_negative" not in kw
    # without the Turbo LoRA the old default means today's plain-model default, not 5.5
    kw = _kw(desc, {**vals, "SAMPLE_CFG_SCALE": 5.5, "FAMILY_TURBO_STRENGTH": 0.0}, models, tmp)
    assert "speed_lora" not in kw and kw["sample_cfg_scale"] == desc.preview_cfg


def test_a_typed_cfg_is_kept_and_warned_about(krea):
    desc, models, vals, tmp = krea
    typed = {**vals, "SAMPLE_CFG_SCALE": 3.0}
    kw = _kw(desc, typed, models, tmp)
    assert kw["sample_cfg_scale"] == 3.0 and kw["sample_negative"] == "blurry"      # the user's choice stands
    msg = pipeline.turbo_preview_cfg_problem(desc, typed, models)
    assert "made for CFG 1" in msg and "3" in msg
    assert pipeline.turbo_preview_cfg_problem(desc, vals, models) == ""              # default box: nothing to say
    assert pipeline.turbo_preview_cfg_problem(desc, {**vals, "SAMPLE_CFG_SCALE": 5.5}, models) == ""
    assert pipeline.turbo_preview_cfg_problem(desc, {**typed, "FAMILY_TURBO_STRENGTH": 0.0}, models) == ""
    assert pipeline.turbo_preview_cfg_problem(desc, {**typed, "SAMPLE_ENABLED": False}, models) == ""
    no_turbo = {k: v for k, v in models.items() if k != desc.preview_speed().pref_key}
    assert pipeline.turbo_preview_cfg_problem(desc, typed, no_turbo) == ""


def test_untouched_rule_and_other_families():
    krea, anima = registry.get("krea2"), registry.get("anima")
    assert pipeline.preview_cfg_untouched(krea, 5.5) and pipeline.preview_cfg_untouched(krea, 4.5)
    assert pipeline.preview_cfg_untouched(krea, 0) and pipeline.preview_cfg_untouched(krea, "")
    assert not pipeline.preview_cfg_untouched(krea, 3.0) and not pipeline.preview_cfg_untouched(krea, 1.0)
    assert pipeline.preview_cfg_untouched(anima, 4.0)                     # Anima's default before 4.5
    for d in registry.FAMILIES.values():                                  # a retired value is never the current one
        assert d.preview_cfg not in d.preview_cfg_retired, d.key


def test_train_tab_shows_todays_default_for_a_saved_former_default(monkeypatch):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from core import config
    monkeypatch.setattr(config, "_settings", None)
    config.settings().set("training.values.krea2", {**P.defaults(), "SAMPLE_CFG_SCALE": 5.5, "LORA_NAME": "x"})
    from tabs.train import TrainTab
    tab = TrainTab()
    assert tab.desc.key == "krea2" and float(tab.collect()["SAMPLE_CFG_SCALE"]) == 4.5
