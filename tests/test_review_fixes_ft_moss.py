"""Findings of the pre-commit reviews of the full fine-tune and the MOSS-Image family (2026-11-09), each pinned by a
test. Tiny made-up files and the tiny CPU family only; no model is loaded."""
import json
import os
import struct
import time

import pytest

torch = pytest.importorskip("torch")

from tests.tiny_family import TINY_FT, TinyFTDiT, make_dataset, register, write_models  # noqa: E402
from training import params as P  # noqa: E402
from training import pipeline, presets, registry  # noqa: E402


@pytest.fixture
def ft(tmp_path, monkeypatch):
    register()
    monkeypatch.setattr(pipeline, "bitsandbytes_available", lambda: True)
    data = make_dataset(str(tmp_path / "scratch_dataset"))
    models = write_models(str(tmp_path / "models"), arch="tinyft", dit_class=TinyFTDiT)
    vals = {**P.defaults(), "LORA_OUTPUT_DIR": str(tmp_path / "runs"), "LORA_NAME": "tiny_ft", "FAMILY_FT": True,
            "DATASET_MEGAPIXELS": "0.01"}
    return TINY_FT, data, models, vals, tmp_path


def _errors(desc, vals, data, models, level="error"):
    return " ".join(c.message for c in pipeline.preflight(desc, vals, data, models) if c.level == level)


def _write_header(path, header, payload=b""):
    raw = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + payload)


# ---- fine-tune ------------------------------------------------------------------------------------------------------
def test_fine_tune_ignores_the_hidden_timestep_range(ft):
    desc, data, models, vals, tmp = ft
    kw, _ = pipeline.train_kwargs(desc, {**vals, "MIN_TIMESTEP": 0.3, "MAX_TIMESTEP": 0.7}, tmp / "run", models,
                                  image_folder=data)
    assert kw["finetune"] and (kw["min_timestep"], kw["max_timestep"]) == (0.0, 1.0)
    kw, _ = pipeline.train_kwargs(desc, {**vals, "FAMILY_FT": False, "MIN_TIMESTEP": 0.3, "MAX_TIMESTEP": 0.7},
                                  tmp / "run", models, image_folder=data)
    assert (kw["min_timestep"], kw["max_timestep"]) == (0.3, 0.7)           # a LoRA keeps its range


def test_fine_tune_never_writes_over_its_own_source(ft):
    from safetensors.torch import save_file

    from training.ft import save_checkpoint
    desc, data, models, vals, tmp = ft
    rd = pipeline.run_dir_for(vals)
    rd.mkdir(parents=True)
    for own in ("tiny_ft.safetensors", "tiny_ft-000004.safetensors"):
        src = rd / own
        save_file({k: v.contiguous() for k, v in TinyFTDiT().state_dict().items()}, str(src))
        assert "own checkpoints" in _errors(desc, vals, data, {**models, "tinyft_dit": str(src)})
    # another LoRA name, or a continuation (which only ever writes later epochs), is fine
    assert "own checkpoints" not in _errors(desc, {**vals, "LORA_NAME": "second_leg"}, data,
                                            {**models, "tinyft_dit": str(src)})
    cont = {"checkpoint": str(src), "start_window": 0, "epochs_done": 4}
    assert "own checkpoints" not in _errors(desc, {**vals, "FAMILY_FT_CONTINUE": cont}, data, models)
    # ... and the writer itself refuses, whatever led there
    before = src.read_bytes()
    with pytest.raises(RuntimeError, match="will not be written over"):
        save_checkpoint({}, str(src), str(src), {})
    assert src.read_bytes() == before


def test_fp8_or_pre_quantised_model_file_is_refused_before_anything_is_cached(ft):
    desc, data, models, vals, tmp = ft
    ok, fp8, scaled = (str(tmp / n) for n in ("ok.safetensors", "fp8.safetensors", "scaled.safetensors"))
    t = {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}
    _write_header(ok, {"a.weight": t}, b"\0\0")
    _write_header(fp8, {"a.weight": {**t, "dtype": "F8_E4M3", "data_offsets": [0, 1]}}, b"\0")
    _write_header(scaled, {"a.weight": t, "a.weight_scale": t}, b"\0\0")
    assert pipeline.ft_source_unfit(ok) == ""
    assert "below 16 bits" in pipeline.ft_source_unfit(fp8) and "pre-quantized" in pipeline.ft_source_unfit(scaled)
    assert pipeline.ft_source_unfit(str(tmp / "missing.safetensors")) == ""
    huge = str(tmp / "huge.safetensors")
    with open(huge, "wb") as f:
        f.write(struct.pack("<Q", 6 * 1024 ** 3) + b"{}")                 # claims a 6 GiB header: never read
    assert pipeline.ft_source_unfit(huge) == ""
    err = _errors(desc, vals, data, {**models, "tinyft_dit": fp8})
    assert "fp8.safetensors" in err and "16-bit" in err
    assert "16-bit" not in _errors(desc, {**vals, "FAMILY_FT": False}, data, {**models, "tinyft_dit": fp8})


def test_fine_tune_warns_about_batch_sizes_above_one(ft):
    desc, data, models, vals, _tmp = ft
    assert "batch sizes above 1" in _errors(desc, {**vals, "DATASET_BATCH_SIZE": 2}, data, models, "warning")
    assert "batch sizes above 1" not in _errors(desc, vals, data, models, "warning")


def test_a_lora_preset_unticks_fine_tune():
    cur = {**P.defaults(), "FAMILY_FT": True, "LEARNING_RATE": 1e-5}
    krea = registry.get("krea2")
    name, lora = next((n, v) for n, v in krea.presets if not v.get("FAMILY_SLIDER"))
    new, rep = presets.apply(lora, cur, krea)                            # names a kind (FAMILY_SLIDER: False)
    assert new["FAMILY_FT"] is False, name
    klein = registry.get("klein9b")
    name, lora = klein.presets[0]
    assert not any(k in lora for k in ("FAMILY_EDIT", "FAMILY_SLIDER", "FAMILY_FT"))
    new, rep = presets.apply(lora, cur, klein)                           # names no kind: still a LoRA's rates
    assert new["FAMILY_FT"] is False and any("LoRA preset" in m for m in rep.messages())
    # a preset that is not a rate recipe leaves the kind alone
    new, rep = presets.apply({"METADATA_AUTHOR": "someone"}, cur, klein)
    assert new["FAMILY_FT"] is True
    # and a Fine-tune preset switches the other kinds off
    sdxl = registry.get("sdxl")
    ftp = next(v for _n, v in sdxl.presets if v.get("FAMILY_FT"))
    new, rep = presets.apply(ftp, {**P.defaults(), "FAMILY_SLIDER": True}, sdxl)
    assert new["FAMILY_FT"] is True and new["FAMILY_SLIDER"] is False


def test_resume_takes_this_legs_checkpoint_not_an_older_runs(tmp_path):
    old, new = tmp_path / "x-000020.safetensors", tmp_path / "x-000004.safetensors"
    old.write_bytes(b"0")
    past = time.time() - 3600
    os.utime(old, (past, past))
    started = time.time() - 60
    new.write_bytes(b"0")
    assert pipeline.latest_ft_checkpoint(str(tmp_path), "x") == str(old)                 # upstream's rule
    assert pipeline.latest_ft_checkpoint(str(tmp_path), "x", since=started) == str(new)
    assert pipeline.latest_ft_checkpoint(str(tmp_path), "x", since=time.time() + 60) == ""


def test_paused_run_is_found_again_after_a_restart(monkeypatch, tmp_path):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from core import config
    monkeypatch.setattr(config, "_settings", None)
    from tabs.train import TrainTab
    tab = TrainTab()
    assert tab.state == "idle" and tab.btn_resume.isHidden()
    rd = pipeline.run_dir_for(tab.collect())
    rd.mkdir(parents=True, exist_ok=True)
    ckpt = rd / "paused-000002.safetensors"
    ckpt.write_bytes(b"0")
    pipeline.write_paused(rd, {"state": str(ckpt), "family": tab.desc.key, "values": tab.collect(), "folder": "",
                               "mode": "ft"})
    tab2 = TrainTab()                                                    # the app started again
    assert tab2.state == "paused" and not tab2.btn_resume.isHidden()
    assert "Paused run found" in tab2.lbl_status.text()
    ckpt.unlink()                                                        # its checkpoint is gone: nothing to resume
    assert TrainTab().state == "idle"


# ---- MOSS-Image -----------------------------------------------------------------------------------------------------
@pytest.fixture
def moss(tmp_path):
    desc = registry.get("moss_image")
    models = {}
    for f in desc.model_files:
        p = tmp_path / f"{f.pref_key}.safetensors"
        p.write_bytes(b"0")
        models[f.pref_key] = str(p)
    vals = {**P.defaults(), "LORA_OUTPUT_DIR": str(tmp_path / "runs"), "LORA_NAME": "moss", "SAMPLE_PROMPT": "a cat",
            "SAMPLE_NEGATIVE": "blurry", "SAMPLE_STEPS": desc.preview_steps, "SAMPLE_CFG_SCALE": desc.preview_cfg,
            "SAMPLE_USE_DISTILLED": True}
    return desc, models, vals, tmp_path


def test_sft_previews_keep_the_samples_negative_and_never_take_the_turbos_steps(moss):
    desc, models, vals, tmp = moss
    ck = desc.preview_checkpoint_sampling
    assert ck.cfg > 1.0                                                   # SFT renders with CFG: a negative applies
    kw, _ = pipeline.train_kwargs(desc, vals, tmp / "run", models)
    assert kw["preview_checkpoint"] and kw.get("sample_negative") == "blurry"
    assert "sample_steps" not in kw and "sample_cfg_scale" not in kw       # untouched boxes: the SFT model's recipe
    # the Turbo LoRA raised with the SFT tick on: its 8 steps and CFG must not reach the SFT render
    kw, _ = pipeline.train_kwargs(desc, {**vals, "FAMILY_TURBO_STRENGTH": 1.0}, tmp / "run", models)
    assert "sample_steps" not in kw and "sample_cfg_scale" not in kw and kw.get("sample_negative") == "blurry"
    # tick off: previews on the training model, where the Turbo's own recipe applies
    kw, _ = pipeline.train_kwargs(desc, {**vals, "SAMPLE_USE_DISTILLED": False, "FAMILY_TURBO_STRENGTH": 1.0},
                                  tmp / "run", models)
    assert "preview_checkpoint" not in kw and kw["sample_steps"] == desc.preview_speed_defaults()[0]


def test_distilled_previews_still_send_no_negative():
    """Klein's Distilled renders without CFG: an untouched CFG box brings no negative (unchanged)."""
    desc = registry.get("klein9b")
    assert desc.preview_checkpoint_sampling.cfg <= 1.0


def test_clean_end_share_is_checked_before_the_run(moss, tmp_path):
    desc, models, vals, _tmp = moss
    data = make_dataset(str(tmp_path / "scratch_dataset"))
    for bad in ("3o", "0", "100", "-5"):
        assert "Low-noise share" in _errors(desc, {**vals, "MOSS_LOWNOISE_PCT": bad}, data, models), bad
    for good in ("", "30", "11.5", "60%"):
        assert "Low-noise share" not in _errors(desc, {**vals, "MOSS_LOWNOISE_PCT": good}, data, models), good


def test_base_check_bounds_what_it_reads(tmp_path):
    import numpy as np

    from training.families.moss_image import base_check
    desc = registry.get("moss_image")
    key, want = desc.standard_base
    good = tmp_path / "base.safetensors"
    vec = np.array([want / 2, want / 2], dtype=np.float64)
    _write_header(good, {key: {"dtype": "F64", "shape": [2], "data_offsets": [0, 16]}}, vec.tobytes())
    assert base_check.is_standard_base(desc, str(good)) is True
    other = tmp_path / "other.safetensors"
    _write_header(other, {key: {"dtype": "F64", "shape": [2], "data_offsets": [0, 16]}},
                  np.array([1.0, 2.0]).tobytes())
    assert base_check.is_standard_base(desc, str(other)) is False
    # hostile or broken files are "unknown", read nothing large and never raise
    huge = tmp_path / "huge.safetensors"
    huge.write_bytes(struct.pack("<Q", 6 * 1024 ** 3) + b"{}")
    span = tmp_path / "span.safetensors"
    _write_header(span, {key: {"dtype": "F64", "shape": [2], "data_offsets": [0, 10 ** 12]}}, b"\0" * 16)
    backwards = tmp_path / "backwards.safetensors"
    _write_header(backwards, {key: {"dtype": "F64", "shape": [2], "data_offsets": [8, 0]}}, b"\0" * 16)
    for bad in (huge, span, backwards):
        assert base_check.is_standard_base(desc, str(bad)) is None, bad.name
