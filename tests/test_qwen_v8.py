"""Qwen Image 2.1 brought level with Fizgig 8.7.1 (commit 5d31b2f): Turbo LoRA recipe, dynamic_shift, Slider preset,
training adapter v2 and ModelFile.replaces / fetch_is_optional. Tiny tensors only; no weights."""
import os

import numpy as np
import torch

from training import registry
from training.description import ModelFile
from training.families.qwen_image21 import sampling as S

SIGMAS = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568)


def _desc():
    return registry.get("qwen_image21")


def test_sigma_schedule_dynamic_shift_off_uses_sigmas_as_given():
    s = S.sigma_schedule(8, 4096, SIGMAS, None, dynamic_shift=False)
    assert s.shape == (9,) and s[-1] == 0
    assert np.allclose(s[:-1].numpy(), SIGMAS, atol=1e-6)
    shifted = S.sigma_schedule(8, 4096, SIGMAS, None)             # default still shifts
    assert not np.allclose(shifted[:-1].numpy(), SIGMAS, atol=1e-3)
    # default behaviour unchanged: linspace, shifted, shift_terminal 0.02
    d = S.sigma_schedule(4, 1024)
    assert d.shape == (5,) and abs(float(d[-2]) - 0.02) < 1e-6 and d[-1] == 0


def test_sample_and_generate_pass_dynamic_shift(monkeypatch):
    from training.families.qwen_image21.driver import QwenImage21Driver
    seen = {}
    monkeypatch.setattr(S, "sample", lambda *a, **k: seen.update(k) or "x")
    d = _desc()
    sp = d.speed_loras[0].settings
    dit = torch.nn.Linear(1, 1)
    cond = {"hidden_states": torch.zeros(3, 4)}
    drv = QwenImage21Driver.__new__(QwenImage21Driver)
    drv.generate(dit, cond, 64, 64, steps=sp.steps, seed=1, cfg=sp.cfg, sigmas=sp.sigmas, options=sp.options)
    assert seen["dynamic_shift"] is False and seen["shift_terminal"] is None and seen["sigmas"] == SIGMAS
    seen.clear()
    drv.generate(dit, cond, 64, 64, steps=25, seed=1)
    assert seen["dynamic_shift"] is True and seen["shift_terminal"] == S.SHIFT_TERMINAL


def test_turbo_lora_recipe():
    d = _desc()
    f = {m.pref_key: m for m in d.model_files}["qwen21_turbo_lora"]
    assert (f.repo, f.path, f.size_gb) == ("Comfy-Org/Qwen-Image-2.1",
                                           "loras/qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors", 0.91)
    assert f.role == "speed_lora" and not f.fetch_is_optional
    sl = d.speed_loras[0]
    assert (sl.file, sl.repo, sl.strength, sl.pref_key) == (f.path, f.repo, 1.0, "qwen21_turbo_lora")
    st = sl.settings
    assert (st.steps, st.cfg, st.sigmas) == (8, 1.0, SIGMAS)
    assert dict(st.options) == {"shift_terminal": None, "dynamic_shift": False}
    assert d.preview_speed_lora == sl.name
    assert (d.preview_speed_steps, d.preview_speed_strength, d.preview_steps, d.preview_cfg) == (8, 1.0, 25, 3.0)
    assert d.preview_negative
    assert not any("iggle" in repr(m) for m in d.model_files) and not any("iggle" in sl.name for sl in d.speed_loras)
    assert not hasattr(d, "retired_preview_defaults") and not hasattr(d, "preview_reset")


def test_presets_slider_rank_8_and_guidance_everywhere():
    d = _desc()
    names = [n for n, _ in d.presets]
    assert "✨ Qwen 2.1 Slider (rank 8, 2e-4)" in names and not any("rank 4" in n for n in names)
    for n, p in d.presets:
        assert p["FAMILY_SLIDER_GUIDANCE"] == "2", n
        assert p["FAMILY_SLIDER"] is (n == "✨ Qwen 2.1 Slider (rank 8, 2e-4)"), n
    slider = dict(d.presets)["✨ Qwen 2.1 Slider (rank 8, 2e-4)"]
    assert (slider["NETWORK_DIM"], slider["NETWORK_ALPHA"], slider["LEARNING_RATE"]) == (8, 8, 2e-4)


def test_training_adapter_v2_and_replaces():
    d = _desc()
    f = {m.pref_key: m for m in d.model_files}["qwen21_training_adapter"]
    assert f.path == "fizgig_qwen_image_2.1_training_adapter_v2.safetensors" and f.size_gb == 0.08
    assert f.replaces == ("fizgig_qwen_image_2.1_training_adapter.safetensors",)
    assert not f.required and not f.fetch_is_optional           # a training adapter is never optional to fetch


def test_migrate_moves_only_the_v1_file(tmp_path):
    f = {m.pref_key: m for m in _desc().model_files}["qwen21_training_adapter"]
    v1, v2, other = (tmp_path / n for n in ("fizgig_qwen_image_2.1_training_adapter.safetensors",
                                           "fizgig_qwen_image_2.1_training_adapter_v2.safetensors", "mine.safetensors"))
    for p in (v1, v2, other):
        p.write_bytes(b"x")
    assert f.migrate(str(v1)) == str(v2)
    assert f.migrate(str(v1), exists=os.path.isfile) == str(v2)
    assert f.migrate(str(other), exists=os.path.isfile) == str(other)     # the user's own file is kept
    assert f.migrate(str(v2)) == str(v2) and f.migrate("") == ""
    v2.unlink()
    assert f.migrate(str(v1), exists=os.path.isfile) == str(v1)           # no v2 on disk yet: keep what works
    assert f.migrate(r"D:\Models\fizgig_qwen_image_2.1_training_adapter.safetensors") == \
        r"D:\Models\fizgig_qwen_image_2.1_training_adapter_v2.safetensors"
    # a family with no `replaces` never rewrites
    dit = {m.pref_key: m for m in _desc().model_files}["qwen21_dit"]
    assert dit.migrate(str(v1)) == str(v1)


def test_fetch_is_optional_rule():
    assert ModelFile("a", "A", False).fetch_is_optional
    assert not ModelFile("a", "A", True).fetch_is_optional
    assert not ModelFile("a", "A", False, role="training_adapter").fetch_is_optional
    assert not ModelFile("a", "A", False, fetch_optional=False).fetch_is_optional
    assert ModelFile("a", "A", False, role="training_adapter", fetch_optional=True).fetch_is_optional
