"""The full fine-tune (training/ft.py, Fizgig's families/ft.py): its parameters and presets, the checks before a
run, and a real rotation / save / pause / continue on the tiny CPU family (tests/tiny_family.py TinyFTDriver).

On a CPU there is no 4-bit kernel, so these runs use the rotator's dense-trunk path (TagScribeR's test-only branch):
they prove the plumbing - windows, the per-window optimizer, the master, the streamed full checkpoint, its metadata
and the continuation - not the NF4 swap, which only a GPU run exercises.
"""
import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests.tiny_family import TINY_FT, TinyFTDiT, make_dataset, register, write_models  # noqa: E402
from training import params as P  # noqa: E402
from training import pipeline, presets  # noqa: E402

ONE_PART = "One part per window (most headroom)"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    register()
    # the CPU path never touches bitsandbytes; the check must not depend on the machine the suite runs on
    monkeypatch.setattr(pipeline, "bitsandbytes_available", lambda: True)
    data = make_dataset(str(tmp_path / "scratch_dataset"))
    models = write_models(str(tmp_path / "models"), arch="tinyft", dit_class=TinyFTDiT)
    return TINY_FT, data, models, tmp_path


def _values(tmp_path, **over):
    v = P.defaults()
    v.update({"LORA_OUTPUT_DIR": str(tmp_path / "runs"), "LORA_NAME": "tiny_ft", "LEARNING_RATE": 1e-3,
              "DATASET_MEGAPIXELS": "0.01", "SAMPLE_PROMPT": "tok, a test prompt", "SAMPLE_EVERY_N_EPOCHS": 1,
              "SAMPLE_WIDTH": 64, "SAMPLE_HEIGHT": 64, "SAMPLE_STEPS": 2, "SAMPLE_AT_FIRST": False,
              "FAMILY_FT": True, "FAMILY_FT_ROTATIONS": 2, "FAMILY_FT_MAX_PARTS": ONE_PART})
    v.update(over)
    return v


def _run_stages(run, *, device="cpu"):
    from training import cache, train
    for _label, argv in run.stages:
        mod, args = argv[2], argv[3:]
        if mod == "training.cache":
            cache.main(args + ["--device", device])
        else:
            cfg_path = args[args.index("--config") + 1]
            cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
            cfg["train"]["device"] = device
            Path(cfg_path).write_text(json.dumps(cfg), encoding="utf-8")
            try:
                train.main(args)
            except SystemExit as e:
                assert e.code in (0, None)


def _read(path):
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}, dict(f.metadata() or {})


# ---- parameters and presets ---------------------------------------------------------------------------------------
def test_parameters_are_declared_for_fine_tune_families_only():
    from tests.tiny_family import TINY
    keys = ("FAMILY_FT", "FAMILY_FT_ROTATIONS", "FAMILY_FT_SAVE_EVERY", "FAMILY_FT_ROTATE_EVERY",
            "FAMILY_FT_MAX_PARTS", "FAMILY_FT_FUSED", "FAMILY_FT_REG_DIR", "FAMILY_FT_REG_MULT")
    d = P.defaults()
    assert [d[k] for k in keys] == [False, 10, 1, 1, "Auto - as few windows as fit", True, "", "0.2"]   # Fizgig's
    for k in keys:
        assert P.BY_KEY[k].tip
        assert P.family_shows(P.BY_KEY[k], TINY_FT) and not P.family_shows(P.BY_KEY[k], TINY)
    assert "FAMILY_FT_REG_DIR" in P.PATH_KEYS
    assert tuple(lab for lab, _ in pipeline.FT_WINDOW_SIZES) == P.BY_KEY["FAMILY_FT_MAX_PARTS"].options
    assert pipeline.ft_max_parts({"FAMILY_FT_MAX_PARTS": ONE_PART}) == 1 and pipeline.ft_max_parts({}) == 0


def test_fine_tune_presets_apply_where_the_family_offers_one():
    from tests.tiny_family import TINY
    register()
    preset = dict(TINY_FT.presets[-1][1])
    new, rep = presets.apply(preset, P.defaults(), TINY_FT)
    assert not rep.blocked and not rep.refused and new["FAMILY_FT"] is True and new["FAMILY_FT_ROTATIONS"] == 2
    new, rep = presets.apply(preset, P.defaults(), TINY)                 # no fine-tune there: nothing changes
    assert rep.blocked == [("FAMILY_FT", "full fine-tuning")] and new == P.defaults()
    for legacy in ("KREA2_FINETUNE", "MINIMAX_FINETUNE"):                # deleted trainers' keys stay refused
        assert presets.apply({legacy: True}, P.defaults(), TINY_FT)[1].blocked


def test_real_families_fine_tune_presets_resolve():
    from training import registry
    seen = set()
    for d in registry.FAMILIES.values():
        for name, values in d.presets:
            if not values.get("FAMILY_FT"):
                continue
            assert d.finetune, f"{d.key}: a Fine-tune preset on a family that does not offer one"
            new, rep = presets.apply(values, P.defaults(), d)
            assert not rep.blocked and not rep.refused and not rep.ignored, (d.key, name, rep.refused, rep.ignored)
            assert new["FAMILY_FT"] is True and pipeline.ft_on(d, new)
            seen.add(d.key)
    # Fizgig ships Fine-tune presets for SDXL, Anima and MOSS-Image; Krea 2, Qwen and Klein offer the fine-tune
    # through the tick alone (their descriptions have no preset for it upstream either)
    assert {"sdxl", "anima", "moss_image"} <= seen, seen
    for key in ("krea2", "qwen_image21", "klein9b", "sdxl", "anima", "moss_image"):
        d = registry.get(key)
        assert d.finetune and d.ft_learning_rate > 0, key
    h3 = registry.get("minimax_h3")                                      # its own backend is not ported yet
    assert not h3.finetune and presets.apply({"FAMILY_FT": True}, P.defaults(), h3)[1].blocked


# ---- checks before a run --------------------------------------------------------------------------------------------
def test_preflight_and_train_kwargs(setup):
    from tests.tiny_family import TINY
    desc, data, models, tmp = setup
    vals = _values(tmp, ADAPTIVE_LR=True, CONTEXT_LORA_PATH=str(tmp / "ctx.safetensors"))
    checks = pipeline.preflight(desc, vals, data, models)
    assert not [c for c in checks if c.level == "error"], checks
    notes = " ".join(c.message for c in checks)
    assert "Companion LoRA is not used" in notes and "Adaptive LR is off" in notes and "full checkpoint" in notes
    kw, _ = pipeline.train_kwargs(desc, vals, tmp / "run", models, image_folder=data)
    assert kw["finetune"] and kw["ft_rotations"] == 2 and kw["ft_max_parts"] == 1 and kw["ft_fused_backward"]
    assert kw["resume_state_dir"] is None and "adaptive_lr" not in kw and "context_lora_path" not in kw
    # a continuation trains from the paused checkpoint, which replaces the model file
    cont = {"checkpoint": str(tmp / "x-000002.safetensors"), "start_window": 0, "epochs_done": 2}
    kw, _ = pipeline.train_kwargs(desc, {**vals, "FAMILY_FT_CONTINUE": cont}, tmp / "run", models, image_folder=data)
    assert kw["dit_path"] == cont["checkpoint"] and kw["ft_epochs_done"] == 2 and kw["ft_start_window"] == 0

    def errors(v, d=desc, **k):
        return " ".join(c.message for c in pipeline.preflight(d, v, data, models, **k) if c.level == "error")
    assert "both on" in errors(_values(tmp, FAMILY_SLIDER=True))
    assert "not from a state folder" in errors(vals, resume=str(tmp / "some-state"))
    assert "regularisation folder" in errors(_values(tmp, FAMILY_FT_REG_DIR=str(tmp / "missing")))
    tiny_models = write_models(str(tmp / "tiny_models"))
    assert "does not offer a full fine-tune" in " ".join(
        c.message for c in pipeline.preflight(TINY, vals, data, tiny_models) if c.level == "error")


def test_regularisation_folder_becomes_its_own_dataset_block(setup):
    desc, data, models, tmp = setup
    reg = make_dataset(str(tmp / "reg_photos"), n=2)
    ds = pipeline.dataset_config(desc, _values(tmp, FAMILY_FT_REG_DIR=reg), data)
    assert [bool(b.get("is_reg")) for b in ds["datasets"]] == [False, True]
    assert ds["datasets"][1]["image_directory"] == os.path.abspath(reg)
    kw, _ = pipeline.train_kwargs(desc, _values(tmp, FAMILY_FT_REG_DIR=reg, FAMILY_FT_REG_MULT="0.3"), tmp / "run",
                                  models, image_folder=data)
    assert kw["reg_lr_multiplier"] == 0.3
    # ... and without Fine-tune ticked the folder is not used at all
    ds = pipeline.dataset_config(desc, _values(tmp, FAMILY_FT=False, FAMILY_FT_REG_DIR=reg), data)
    assert len(ds["datasets"]) == 1


# ---- a real run on the tiny family ----------------------------------------------------------------------------------
def test_fine_tune_rotates_saves_whole_models_and_continues(setup, caplog):
    import logging
    desc, data, models, tmp = setup
    src, _ = _read(models["tinyft_dit"])
    vals = _values(tmp)
    run = pipeline.build_run(desc, vals, data, models)
    rd = run.run_dir
    pipeline.request_pause(rd)                       # honoured at the first whole rotation, never inside one
    with caplog.at_level(logging.INFO):
        _run_stages(run)
    text = caplog.text
    assert "FULL FINE-TUNE" in text and "window 1/2" in text and "window 2/2" in text
    assert "2 rotation(s) of 2 epoch(s) = 4 epochs" in text
    assert "rotation complete at epoch 2" in text
    assert not (rd / "tiny_ft-000001.safetensors").exists()             # no save in the middle of a rotation
    ckpt = rd / "tiny_ft-000002.safetensors"
    assert ckpt.exists() and not (rd / "tiny_ft.safetensors").exists()
    assert not list(rd.glob("*-state"))                                  # a fine-tune has no state folders
    assert pipeline.latest_ft_checkpoint(str(rd), "tiny_ft") == str(ckpt)

    mid, md = _read(ckpt)
    assert set(mid) == set(src) and all(mid[k].shape == src[k].shape and mid[k].dtype == src[k].dtype for k in src)
    trained = [k for k in src if not torch.equal(mid[k], src[k])]
    # every block's attn and mlp weights trained (one rotation = both windows); nothing else moved: biases, the
    # convolutions and the embedders are the source's own tensors
    assert sorted(trained) == sorted(f"blocks.{b}.{part}.weight" for b in (0, 1) for part in ("attn", "mlp"))
    assert md["tagscriber_finetune"] == "_tinyft-rotation" and md["tagscriber_ft_epochs_done"] == "2"
    assert md["tagscriber_ft_n_windows"] == "2" and md["tagscriber_source_base"] == "dit.safetensors"
    assert md["tagscriber_next_start_window"] is not None
    assert not any(k.startswith("fizgig_") for k in md)                  # the old names are read, no longer written

    # Resume: one rotation left, from the checkpoint (Fizgig's ft_rotations_left)
    left, done, cont = pipeline.ft_rotations_left(str(ckpt), vals["FAMILY_FT_ROTATIONS"],
                                                  vals["FAMILY_FT_ROTATE_EVERY"])
    assert (left, done) == (1, 1) and cont["checkpoint"] == str(ckpt) and cont["epochs_done"] == 2
    pipeline.clear_pause(rd)
    run2 = pipeline.build_run(desc, {**vals, "FAMILY_FT_CONTINUE": cont, "FAMILY_FT_ROTATIONS": left}, data, models)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        _run_stages(run2)
    assert "Fine-tune complete" in caplog.text and "epoch 3: window" in caplog.text
    final, md = _read(rd / "tiny_ft.safetensors")
    assert md["tagscriber_ft_epochs_done"] == "4"
    assert set(final) == set(src)
    assert all(not torch.equal(final[k], mid[k]) for k in trained)        # the second rotation trained on
    assert all(torch.equal(final[k], src[k]) for k in src if k not in trained)
    assert not (rd / ".tagscriber_ft_master").exists()                   # the scratch master is cleaned up
    assert pipeline.samples(rd)                                          # previews rendered on the trained model


def test_whole_model_in_one_window_when_it_fits(setup, caplog):
    import logging
    desc, data, models, tmp = setup
    vals = _values(tmp, LORA_NAME="tiny_ft_auto", FAMILY_FT_MAX_PARTS="Auto - as few windows as fit",
                   FAMILY_FT_ROTATIONS=1, FAMILY_FT_FUSED=False, SAMPLE_ENABLED=False)
    run = pipeline.build_run(desc, vals, data, models)
    with caplog.at_level(logging.INFO):
        _run_stages(run)
    assert "window 1/1" in caplog.text and "1 rotation(s) of 1 epoch(s) = 1 epochs" in caplog.text
    src, _ = _read(models["tinyft_dit"])
    final, _md = _read(run.run_dir / "tiny_ft_auto.safetensors")
    assert sum(not torch.equal(final[k], src[k]) for k in src) == 4


def test_a_pre_quantised_or_fp8_file_is_refused(tmp_path):
    from safetensors.torch import save_file

    from training.ft import source_unfit_reason
    ok, scaled, fp8 = (str(tmp_path / n) for n in ("ok.safetensors", "scaled.safetensors", "fp8.safetensors"))
    w = torch.zeros(2, 2, dtype=torch.bfloat16)
    save_file({"a.weight": w}, ok)
    save_file({"a.weight": w, "a.weight_scale": torch.ones(1)}, scaled)
    save_file({"a.weight": torch.zeros(2, 2, dtype=torch.int8)}, fp8)
    assert source_unfit_reason(ok) is None
    assert "pre-quantized" in source_unfit_reason(scaled) and "below 16 bits" in source_unfit_reason(fp8)


# ---- the Train tab --------------------------------------------------------------------------------------------------
@pytest.fixture
def app(monkeypatch):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from core import config
    monkeypatch.setattr(config, "_settings", None)        # fresh settings: the family chosen here stays here
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_train_tab_fine_tune_tick(app):
    from tabs.train import FT_HIDDEN, TrainTab
    tab = TrainTab()
    tab.chk_all.setChecked(True)                                         # "Show all settings"
    assert tab.desc.key == "krea2" and tab.desc.finetune

    def hidden(key):
        w = tab.widgets[key][0]
        return (w.parentWidget() if P.BY_KEY[key].kind in (P.PATH, P.DIR) else w).isHidden()
    assert not hidden("FAMILY_FT") and hidden("FAMILY_FT_ROTATIONS") and not hidden("NETWORK_DIM")
    tab._set_widget("FAMILY_SLIDER", True)
    tab._set_widget("ADAPTIVE_LR", True)
    tab.widgets["FAMILY_FT"][0].setChecked(True)                         # ticked by hand
    v = tab.collect()
    assert v["FAMILY_FT"] and not v["FAMILY_SLIDER"] and not v["FAMILY_EDIT"]     # a run is one kind of training
    assert v["LEARNING_RATE"] == tab.desc.ft_learning_rate and v["ADAPTIVE_LR"] is False
    assert not hidden("FAMILY_FT_ROTATIONS") and not hidden("FAMILY_FT_REG_DIR")
    assert all(hidden(k) for k in FT_HIDDEN if k in tab.widgets and P.family_shows(P.BY_KEY[k], tab.desc))
    assert tab.grp_resume.isHidden() and not tab.widgets["MAX_TRAIN_EPOCHS"][0].isEnabled()
    tab.widgets["FAMILY_SLIDER"][0].setChecked(True)                     # another kind: Fine-tune goes off
    v = tab.collect()
    assert v["FAMILY_SLIDER"] and not v["FAMILY_FT"]
    assert hidden("FAMILY_FT_ROTATIONS") and not hidden("NETWORK_DIM") and tab.widgets["MAX_TRAIN_EPOCHS"][0].isEnabled()
    # the simple view shows a fine-tune's length where Epochs was
    tab.widgets["FAMILY_FT"][0].setChecked(True)
    tab.chk_all.setChecked(False)
    assert not hidden("FAMILY_FT_ROTATIONS") and hidden("FAMILY_FT_SAVE_EVERY")
    # a family without the shared fine-tune never shows the tick
    tab.combo_family.setCurrentIndex(tab.combo_family.findData("minimax_h3"))
    tab.chk_all.setChecked(True)
    assert hidden("FAMILY_FT") and hidden("FAMILY_FT_ROTATIONS")


def test_train_tab_blocks_a_fine_tune_preset_only_where_it_cannot_run(app, monkeypatch):
    from tabs import train as train_mod
    tab = train_mod.TrainTab()
    shown = []
    monkeypatch.setattr(train_mod.QMessageBox, "warning", lambda *a, **k: shown.append(a[2]))
    tab.combo_family.setCurrentIndex(tab.combo_family.findData("sdxl"))
    name = next(n for n, v in tab.desc.presets if v.get("FAMILY_FT"))
    tab._apply_preset(name, quiet=True)
    v = tab.collect()
    assert v["FAMILY_FT"] is True and v["FAMILY_FT_ROTATIONS"] == 10 and v["LEARNING_RATE"] == 1e-5 and not shown
    # MiniMax H3's fine-tune needs its own backend (not ported): the same preset there is refused, nothing changes
    tab.combo_family.setCurrentIndex(tab.combo_family.findData("minimax_h3"))
    before = tab.collect()
    new, rep = train_mod.presets.apply({"FAMILY_FT": True, "LEARNING_RATE": 3e-5}, before, tab.desc)
    assert rep.blocked and new == before


def _header_only(path, metadata):
    """A file with a safetensors header (metadata only, no tensors): all ft_continuation reads."""
    import struct
    header = json.dumps({"__metadata__": metadata}).encode()
    Path(path).write_bytes(struct.pack("<Q", len(header)) + header)


def test_a_fine_tune_checkpoint_continues_from_either_metadata_prefix(tmp_path):
    new = tmp_path / "new_ft.safetensors"
    _header_only(new, {"tagscriber_next_start_window": "1", "tagscriber_ft_n_windows": "3",
                       "tagscriber_ft_epochs_done": "4"})
    assert pipeline.ft_continuation(str(new)) == (1, 4, 3)
    # a checkpoint written by an earlier build or another trainer still continues
    old = tmp_path / "old_ft.safetensors"
    _header_only(old, {"fizgig_next_start_window": "2", "fizgig_ft_n_windows": "5", "fizgig_ft_epochs_done": "7"})
    assert pipeline.ft_continuation(str(old)) == (2, 7, 5)
    # both present: TagScribeR's own key wins, a missing one falls back to the old name
    both = tmp_path / "both_ft.safetensors"
    _header_only(both, {"tagscriber_ft_n_windows": "3", "fizgig_ft_n_windows": "9", "fizgig_ft_epochs_done": "6"})
    assert pipeline.ft_continuation(str(both)) == (0, 6, 3)
    # the epoch in a -NNNNNN file name still wins over the metadata
    named = tmp_path / "run-000010.safetensors"
    _header_only(named, {"fizgig_ft_epochs_done": "6"})
    assert pipeline.ft_continuation(str(named))[1] == 10
