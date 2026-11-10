"""Fixes taken from Fizgig 9 (docs/dev/audit/12_fizgig_v9_4_0_training_core.md, section 1): each was a bug TagScribeR
shared. Unit tests on tiny tensors and made-up figures; no model is loaded."""
import os

import pytest

torch = pytest.importorskip("torch")


# ---- F1: a full-matrix LoKR plays at scale 1, whatever alpha its file stores (Fizgig 9.0.7, #304) -------------------
def test_full_matrix_lokr_ignores_the_stored_alpha():
    from training.lora import lycoris_scale_from_keys
    w1, w2 = torch.randn(4, 4), torch.randn(8, 8)
    # LyCORIS writes the network dim into alpha: scaled by it, the LoRA played 16x too strong
    assert lycoris_scale_from_keys({"lokr_w1": w1, "lokr_w2": w2, "alpha": torch.tensor(16.0)}) == 1.0
    assert lycoris_scale_from_keys({"lokr_w1": w1, "lokr_w2": w2}) == 1.0
    # a decomposed LoKR keeps alpha / rank, and LoHa alpha / min(rank)
    dec = {"lokr_w1": w1, "lokr_w2_a": torch.randn(8, 4), "lokr_w2_b": torch.randn(4, 8), "alpha": torch.tensor(2.0)}
    assert lycoris_scale_from_keys(dec) == 0.5
    loha = {"hada_w1_a": torch.randn(8, 2), "hada_w1_b": torch.randn(2, 8), "hada_w2_a": torch.randn(8, 4),
            "hada_w2_b": torch.randn(4, 8), "alpha": torch.tensor(1.0)}
    assert lycoris_scale_from_keys(loha) == 0.5


# ---- F2: Muon's / Prodigy's default Optimizer Args never reach another optimizer (Fizgig 9.2.0) --------------------
def test_optimizer_args_left_from_muon_do_not_turn_the_run_into_fallback_adamw():
    from training.optimizers import MUON_DEFAULT_ARGS, PRODIGY_DEFAULT_ARGS, create_optimizer
    for stale in (MUON_DEFAULT_ARGS, PRODIGY_DEFAULT_ARGS):
        p = [torch.nn.Parameter(torch.zeros(2, 2))]
        opt, label = create_optimizer("adamw", p, 1e-4, stale)
        assert "fallback" not in label and label == "adamw", label
        assert opt.defaults["weight_decay"] == torch.optim.AdamW(p).defaults["weight_decay"]   # not Muon's 0.01 either
    # args typed for the optimizer in use are kept
    opt, label = create_optimizer("adamw", [torch.nn.Parameter(torch.zeros(2, 2))], 1e-4, "weight_decay=0.05")
    assert opt.defaults["weight_decay"] == 0.05 and "weight_decay=0.05" in label


def test_a_fizgig_9_muon_preset_keeps_the_optimizer_and_its_args_as_they_were():
    from training import params as P
    from training import presets, registry
    from training.optimizers import MUON_DEFAULT_ARGS
    desc = registry.get("krea2")
    cur = {**P.defaults(), "OPTIMIZER_TYPE": "adamw8bit", "OPTIMIZER_ARGS": ""}
    preset = {"OPTIMIZER_TYPE": "muon", "OPTIMIZER_ARGS": MUON_DEFAULT_ARGS, "NETWORK_DIM": 32, "LEARNING_RATE": 1e-4}
    new, rep = presets.apply(preset, cur, desc)
    if "muon" in desc.optimizers:                      # once Muon is ported the preset applies whole
        assert new["OPTIMIZER_TYPE"] == "muon" and new["OPTIMIZER_ARGS"] == MUON_DEFAULT_ARGS
        return
    assert new["OPTIMIZER_TYPE"] == "adamw8bit" and new["OPTIMIZER_ARGS"] == ""
    assert new["NETWORK_DIM"] == 32 and "OPTIMIZER_ARGS" not in rep.applied       # the rest of the preset applies
    assert any("Optimizer Args" in m for m in rep.messages())
    # ... while args beside an optimizer that IS offered are applied as before
    new, rep = presets.apply({"OPTIMIZER_TYPE": "adamw", "OPTIMIZER_ARGS": "weight_decay=0.05"}, cur, desc)
    assert new["OPTIMIZER_TYPE"] == "adamw" and new["OPTIMIZER_ARGS"] == "weight_decay=0.05"


# ---- F8: Auto takes NF4 with no swap, closely, before it swaps INT8 blocks (Fizgig 8.9, #216) -----------------------
def test_auto_prefers_a_close_nf4_fit_to_swapping_int8(monkeypatch):
    from training import quant, registry
    desc = registry.get("krea2")
    monkeypatch.setattr(quant, "available", lambda p, device=None: (True, ""))
    drv = desc.load_driver()
    plan = lambda free: quant.plan(desc, drv, "auto", -1, free_gb=free, megapixels=1.0)  # noqa: E731
    assert plan(21.0)[:2] == ("int8", 0)                 # 19.1 + the 1.5 GB margin
    assert plan(17.0)[:2] == ("nf4", 0)                  # 15.0 + 1.5
    p, swap, why = plan(15.6)                            # NF4 fits with 0.5 GB to spare: no swapped INT8 blocks
    assert (p, swap) == ("nf4", 0) and "closely" in why
    p, swap, _ = plan(15.2)                              # not even closely: the swap, as before
    assert p == "int8" and swap > 0


# ---- onnxruntime's telemetry is off (Fizgig 9.2.0) -------------------------------------------------------------------
def test_onnxruntime_telemetry_is_switched_off(monkeypatch):
    from core import hardware
    monkeypatch.delenv("ORT_DISABLE_TELEMETRY", raising=False)
    hardware.apply_runtime_env()
    assert os.environ["ORT_DISABLE_TELEMETRY"] == "1" and os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    monkeypatch.setenv("ORT_DISABLE_TELEMETRY", "0")     # a value the user set wins
    hardware.apply_runtime_env()
    assert os.environ["ORT_DISABLE_TELEMETRY"] == "0"
    hardware.ort_telemetry_off()                         # never raises, with or without onnxruntime


# ---- F4: Resume with another optimizer keeps the weights and the epoch, with a fresh optimizer ---------------------
def _tiny_net():
    from tests.tiny_family import TinyDiT, register
    from training.lora import FamilyLoRA
    torch.manual_seed(0)
    net = FamilyLoRA(TinyDiT().requires_grad_(False), register().load_driver())
    net.add_trainable(4, 4)
    return net


def test_resume_with_another_optimizer_starts_it_fresh(tmp_path, caplog):
    import logging

    from training import train
    net = _tiny_net()
    adamw = torch.optim.AdamW(net.parameters(), lr=1e-3)
    for p in net.parameters():
        p.grad = torch.ones_like(p)
    adamw.step()                                         # some state to carry
    state = train._save_state(str(tmp_path), "run", net, adamw, epoch=2, global_step=8, arch_id="tiny")
    import json
    assert json.loads((tmp_path / "run-000002-state" / "training_state.json").read_text())["optimizer"] == "AdamW"
    # the same optimizer: its state is loaded as before
    same = torch.optim.AdamW(net.parameters(), lr=1e-3)
    epoch, step, meta = train._load_state(state, net, same, "cpu")
    assert (epoch, step) == (2, 8) and not meta.get("fresh_optimizer") and same.state_dict()["state"]
    # another one: no exception, no borrowed moments, and the log says what happened
    other = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    with caplog.at_level(logging.INFO):
        epoch, step, meta = train._load_state(state, net, other, "cpu")
    assert (epoch, step) == (2, 8) and meta["fresh_optimizer"] is True and not other.state_dict()["state"]
    assert "fresh optimizer" in caplog.text and "AdamW" in caplog.text


# ---- F5: previews keep their size on a card under 20 GB, unless the family declares a cap (Krea 2) ------------------
def test_small_card_preview_cap_is_a_familys_own():
    from training import registry
    from training.train import _cap_canvas
    assert registry.get("krea2").preview_small_card_cap == 768
    for key in ("qwen_image21", "klein9b", "moss_image", "sdxl", "anima", "minimax_h3"):
        assert registry.get(key).preview_small_card_cap == 0, key         # their previews are never shrunk
    assert _cap_canvas(1024, 1024, 768) == (768, 768) and _cap_canvas(768, 1024, 768) == (576, 768)
    import inspect

    from training import train
    src = inspect.getsource(train.train_family)
    assert "preview_small_card_cap" in src and "max(sample_width, sample_height) > 768" not in src


# ---- F12: a companion LoRA that matches no layer stops the run -------------------------------------------------------
def test_companion_lora_that_matches_nothing_stops_the_run(tmp_path):
    import json

    from safetensors.torch import save_file

    from tests.tiny_family import make_dataset, register, write_models
    from training import params as P
    from training import cache, pipeline, train
    desc = register()
    data = make_dataset(str(tmp_path / "scratch_dataset"))
    models = write_models(str(tmp_path / "models"))
    wrong = str(tmp_path / "other_model_lora.safetensors")
    save_file({"lora_unet_some_other_block.lora_down.weight": torch.zeros(2, 8),
               "lora_unet_some_other_block.lora_up.weight": torch.zeros(8, 2)}, wrong)
    vals = {**P.defaults(), "LORA_OUTPUT_DIR": str(tmp_path / "runs"), "LORA_NAME": "c", "MAX_TRAIN_EPOCHS": 1,
            "OPTIMIZER_TYPE": "adamw", "DATASET_MEGAPIXELS": "0.01", "SAMPLE_ENABLED": False,
            "CONTEXT_LORA_PATH": wrong}
    run = pipeline.build_run(desc, vals, data, models)
    for _label, argv in run.stages:
        mod, args = argv[2], argv[3:]
        if mod == "training.cache":
            cache.main(args + ["--device", "cpu"])
            continue
        cfg_path = args[args.index("--config") + 1]
        cfg = json.loads(open(cfg_path, encoding="utf-8").read())
        cfg["train"]["device"] = "cpu"
        open(cfg_path, "w", encoding="utf-8").write(json.dumps(cfg))
        with pytest.raises(RuntimeError, match="companion LoRA other_model_lora.safetensors matched no"):
            train.main(args)
    assert not (run.run_dir / "c.safetensors").exists()


def test_frozen_file_hook_reports_what_it_applied():
    from training import registry
    from training.driver import FamilyDriver
    assert FamilyDriver.frozen_file_added(object(), None, "x", 1.0, "context") == 0
    import inspect
    h3 = registry.get("minimax_h3").load_driver()
    assert "return len(pairs)" in inspect.getsource(type(h3).frozen_file_added)
