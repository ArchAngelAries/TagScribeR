"""Training layer unit tests: Adaptive LR decisions, presets, parameters, buckets, captions, LoRA keys, progress.

CPU only, no real models (Fizgig is the reference for the expected behaviour; see docs/FIZGIG_TRAINING_AUDIT.md).
"""
import json
import math
import os

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from training import params as P  # noqa: E402
from training import presets  # noqa: E402
from training.adaptive_lr import AdaptiveLR  # noqa: E402


# ---- Adaptive LR ------------------------------------------------------------------------------------------
class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(10))


def _opt(net, lr):
    return torch.optim.SGD(net.parameters(), lr=lr)


def _run(losses, growth=None, lo=1e-4, hi=4e-4, clip=None):
    """Feed a loss curve through AdaptiveLR; growth[i] scales the weights before epoch i's boundary."""
    net = _Net()
    a = AdaptiveLR(lo, hi)
    opt = _opt(net, AdaptiveLR.start_lr(lo, hi))
    actions, lrs = [], []
    for e, loss in enumerate(losses):
        if growth and growth[e]:
            with torch.no_grad():
                net.w.mul_(growth[e])
        if clip and clip[e] is not None:
            events, steps = clip[e]
            for i in range(steps):
                a.note_clip(2.0 if i < events else 0.5, 1.0)
        actions.append(a.epoch_boundary(e, loss, net, opt))
        lrs.append(opt.param_groups[0]["lr"])
    return actions, lrs, a, net


def test_start_lr_is_geometric_midpoint():
    assert AdaptiveLR.start_lr(2e-4, 4e-4) == pytest.approx(2.828e-4, rel=1e-3)
    assert AdaptiveLR.start_lr(1e-4, 4e-4) == pytest.approx(2e-4)
    assert AdaptiveLR.start_lr(5e-5, 2e-4) == pytest.approx(1e-4)


def test_arms_then_probes_up_after_two_improving_epochs():
    actions, lrs, *_ = _run([1.0, 0.9, 0.8, 0.7, 0.6], lo=1e-4, hi=4e-4)
    assert actions[0] == "ARMED"
    assert actions[1] == "HOLD" and actions[2] == "PROBE UP"
    assert lrs[2] == pytest.approx(2e-4 * 1.25)
    assert actions[3] == "HOLD" and actions[4] == "PROBE UP"


def test_probe_is_capped_at_max():
    actions, lrs, *_ = _run([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2], lo=2e-4, hi=3e-4)
    assert max(lrs) <= 3e-4 + 1e-12
    assert "HOLD (capped)" in actions


def test_plateau_patience_is_one_in_epochs_2_and_3_then_two():
    # epoch index 1 (2nd epoch): patience 2 -> first plateau only counts
    actions, lrs, *_ = _run([1.0, 1.1, 1.2], lo=1e-4, hi=4e-4)
    assert actions[1] == "HOLD"                     # streak 1/2 at epoch index 1
    assert actions[2] == "REDUCE"                   # epoch index 2: patience 1 -> reduce
    assert lrs[2] == pytest.approx(2e-4 * 0.5)
    # from epoch index 4 on, patience is 2 again
    actions, lrs, *_ = _run([1.0, 0.9, 0.8, 0.79, 0.85, 0.86], lo=1e-5, hi=4e-4)
    assert actions[4] == "HOLD" and actions[5] == "REDUCE"


def test_reduce_is_floored_at_min():
    actions, lrs, *_ = _run([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], lo=1.5e-4, hi=2e-4)
    assert min(lrs) >= 1.5e-4 - 1e-12
    assert "HOLD (floored)" in actions


def test_weight_growth_triggers_reduce_and_rollback():
    actions, lrs, a, net = _run([1.0, 0.9, 0.8], growth=[None, None, 1.5], lo=1e-4, hi=4e-4)
    assert actions[2] == "REDUCE+ROLLBACK"
    # weights blended 70/30 toward the epoch-1 snapshot (all ones): 0.7*1 + 0.3*1.5 = 1.15
    assert float(net.w[0].detach()) == pytest.approx(1.15, rel=1e-5)
    assert a.stability_triggered


def test_second_stability_event_needs_two_red_epochs():
    actions, *_ = _run([1.0, 0.9, 0.8, 0.7, 0.6], growth=[None, 1.5, None, 1.5, 1.5])
    assert actions[1] == "REDUCE+ROLLBACK"
    assert actions[3] == "WAIT"
    assert actions[4] in ("REDUCE+ROLLBACK", "HOLD (floored)")


def test_clip_ratio_signal_from_klein():
    actions, *_ = _run([1.0, 0.9, 0.8], clip=[None, (6, 10), None])
    assert actions[1] == "REDUCE+ROLLBACK"           # 60% of steps clipped > 50%
    actions, *_ = _run([1.0, 0.9, 0.8], clip=[None, (4, 10), None])
    assert actions[1] == "HOLD"                      # 40%: no stability signal


def test_state_round_trip():
    _, _, a, _ = _run([1.0, 0.9, 1.0])
    b = AdaptiveLR(1e-4, 4e-4)
    b.load_state_dict(json.loads(json.dumps(a.state_dict())))
    assert b.state_dict() == a.state_dict()
    assert b.snapshot is None                        # never persisted (Fizgig)


# ---- parameters and presets -----------------------------------------------------------------------------
@pytest.fixture
def qwen():
    from training.registry import get
    return get("qwen_image21")


def test_params_cover_every_qwen_preset_key(qwen):
    for _name, values in qwen.presets:
        unknown = [k for k in values if k not in P.BY_KEY]
        assert not unknown, unknown


def test_first_builtin_is_default(qwen):
    tp = presets.TrainingPresets(qwen)
    assert tp.default_name.startswith("✨ Qwen 2.1 Fast")
    assert tp.names()[:5] == [n for n, _ in qwen.presets]


def test_apply_matches_first_token_and_refuses_strict(qwen):
    cur = P.defaults()
    new, rep = presets.apply({"ADAPTIVE_LR_MIN": "2e-4", "OPTIMIZER_TYPE": "lion8bit", "NETWORK_DIM": "16",
                              "SOME_FUTURE_KEY": 1, "FAMILY_PRECISION": "INT8"}, cur, qwen)
    assert new["ADAPTIVE_LR_MIN"] == "2e-4 - rank 4/8 only"
    assert new["NETWORK_DIM"] == 16
    assert new["FAMILY_PRECISION"] == "INT8 (8-bit, fastest)"
    assert new["OPTIMIZER_TYPE"] == cur["OPTIMIZER_TYPE"]           # Qwen offers adamw / adamw8bit only
    assert [r[0] for r in rep.refused] == ["OPTIMIZER_TYPE"]
    assert rep.ignored == ["SOME_FUTURE_KEY"]
    assert rep.messages()[0].startswith("[preset] OPTIMIZER_TYPE")


def test_apply_refuses_bad_numbers():
    new, rep = presets.apply({"NETWORK_DIM": "abc", "MAX_TRAIN_EPOCHS": 0}, P.defaults())
    assert {r[0] for r in rep.refused} == {"NETWORK_DIM", "MAX_TRAIN_EPOCHS"}
    assert new["NETWORK_DIM"] == P.BY_KEY["NETWORK_DIM"].default


def test_user_preset_lifecycle(qwen):
    tp = presets.TrainingPresets(qwen)
    vals = {**P.defaults(), "NETWORK_DIM": 12, "SAMPLE_PROMPT": "not in presets"}
    path = tp.save("My run", vals)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["NETWORK_DIM"] == 12
    assert "SAMPLE_PROMPT" not in data and presets.ARCH_KEY not in data   # no family, no run-only values
    assert "My run" in tp.names()
    assert tp.load("My run")["NETWORK_DIM"] == 12
    with pytest.raises(presets.PresetError):
        tp.save(tp.default_name, vals)
    with pytest.raises(presets.PresetError):
        tp.save("bad/name", vals)
    with pytest.raises(presets.PresetError):
        tp.delete(tp.default_name)
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(presets.PresetError, match="corrupted"):
        tp.load("My run")


def test_import_fizgig_preset_file(qwen, tmp_path):
    src = tmp_path / "From Fizgig.json"
    src.write_text(json.dumps({"NETWORK_DIM": "8", "ADAPTIVE_LR": True, "ADAPTIVE_LR_MIN": "2e-4",
                               "KREA2_LOSS_WATCH": True, "MINIMAX_TREAD": True}), encoding="utf-8")
    tp = presets.TrainingPresets(qwen)
    name = tp.import_file(src)
    assert name == "From Fizgig"
    new, rep = presets.apply(tp.load(name), P.defaults(), qwen)
    assert new["NETWORK_DIM"] == 8 and new["ADAPTIVE_LR"] is True
    assert "MINIMAX_TREAD" in rep.ignored


def test_last_run_snapshot_records_family():
    presets.save_last_run("qwen_image21", {"NETWORK_DIM": 8}, "D:/data")
    snap = presets.load_last_run()
    assert snap[presets.ARCH_KEY] == "qwen_image21" and snap[presets.DATASET_KEY] == "D:/data"


def test_family_options(qwen):
    assert P.options_for(P.BY_KEY["OPTIMIZER_TYPE"], qwen) == ("adamw", "adamw8bit")
    assert P.options_for(P.BY_KEY["FAMILY_PRECISION"], qwen)[0].startswith("Auto")
    assert len(P.options_for(P.BY_KEY["FAMILY_PRECISION"], qwen)) == 4


# ---- dataset ----------------------------------------------------------------------------------------------
def test_megapixel_resolution():
    from training.dataset import resolution_for_megapixels
    assert resolution_for_megapixels(0.25) == 496
    assert resolution_for_megapixels(0.5) == 704


@pytest.mark.parametrize("size,bucket", [((1000, 1000), (704, 704)), ((800, 1000), (640, 768)),
                                         ((1000, 1500), (576, 832)), ((900, 1600), (512, 960)),
                                         ((1600, 900), (960, 512))])
def test_qwen_buckets_at_half_megapixel(size, bucket):
    """Fizgig's BucketSelector at 0.5 MP, step 32 (the code; docs/QWEN_IMAGE.md's table lists 16-px sizes such as
    624x784 that the step-32 selector cannot produce)."""
    from training.dataset import BucketSelector
    sel = BucketSelector((704, 704), True, True, 32)
    assert sel.get_bucket_resolution(size) == bucket
    assert all(w % 32 == 0 and h % 32 == 0 and w * h <= 704 * 704 for w, h in sel.bucket_resolutions)


def test_no_upscale_keeps_small_images():
    from training.dataset import BucketSelector
    assert BucketSelector((704, 704), True, True, 32).get_bucket_resolution((300, 200)) == (288, 192)


def test_decode_caption_fallbacks():
    from training.dataset import decode_caption
    assert decode_caption("café".encode("utf-8"), "x") == "café"
    assert decode_caption("it’s".encode("cp1252"), "x") == "it’s"
    assert decode_caption("hi".encode("utf-16"), "x") == "hi"


def test_shuffle_variants_deterministic_and_keep_tokens():
    from training.dataset import shuffle_variants
    cap = "tok, a, b, c, d, e"
    v1 = shuffle_variants(cap, 3, keep_tokens=1)
    assert v1 == shuffle_variants(cap, 3, keep_tokens=1)
    assert len(v1) == 3 and len(set(v1)) == 3
    assert all(v.startswith("tok, ") for v in v1)
    assert all(sorted(v.split(", ")) == sorted(cap.split(", ")) for v in v1)
    assert shuffle_variants("one tag", 3) == []
    assert shuffle_variants(cap, 0) == []


def test_glob_images_leaves_out_uncaptioned(tmp_path):
    from training.dataset import glob_images
    from tests.conftest import make_image
    make_image(tmp_path / "a.png")
    make_image(tmp_path / "b.jpg")
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    assert [os.path.basename(p) for p in glob_images(str(tmp_path), ".txt")] == ["a.png"]


def test_cache_dir_is_stable_and_unique(tmp_path):
    from training.pipeline import cache_dir_for
    a = cache_dir_for(str(tmp_path / "My Set"), tmp_path)
    assert a == cache_dir_for(str(tmp_path / "My Set"), tmp_path)
    assert a != cache_dir_for(str(tmp_path / "other" / "My Set"), tmp_path)
    assert a.name.startswith("My_Set-")


# ---- LoRA layer -------------------------------------------------------------------------------------------
@pytest.fixture
def tiny():
    from tests.tiny_family import register
    return register()


def _net(tiny, kind="lora"):
    from tests.tiny_family import TinyDiT
    from training.lora import FamilyLoRA
    torch.manual_seed(0)
    dit = TinyDiT().requires_grad_(False)
    drv = tiny.load_driver()
    net = FamilyLoRA(dit, drv)
    net.add_trainable(4, 4, kind=kind, factor=4)
    return dit, net


def test_lora_wraps_linear_and_conv_and_starts_at_zero(tiny):
    from tests.tiny_family import TinyDiT
    from training.lora import LoRAConv2d, LoRALinear
    torch.manual_seed(0)
    ref = TinyDiT()
    dit, net = _net(tiny)
    kinds = {type(w).__name__ for w in net.wrapped.values()}
    assert kinds == {"LoRALinear", "LoRAConv2d"}
    assert isinstance(dit.blocks[0].conv, LoRAConv2d) and isinstance(dit.blocks[0].lin, LoRALinear)
    x, c, t = torch.randn(1, 4, 8, 8), torch.randn(1, 3, 8), torch.tensor([0.5])
    assert torch.allclose(dit(x, c, t), ref(x, c, t), atol=1e-6)        # up weights are zero-initialised


def test_lora_keys_and_round_trip(tiny, tmp_path):
    dit, net = _net(tiny)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.01)
    sd = net.state_dict()
    assert "diffusion_model.blocks.0.conv.lora_down.weight" in sd
    assert sd["diffusion_model.blocks.0.conv.lora_down.weight"].shape == (4, 4, 3, 3)
    assert sd["diffusion_model.blocks.0.conv.lora_up.weight"].shape == (8, 4, 1, 1)
    assert sd["diffusion_model.blocks.1.lin.lora_up.weight"].shape == (8, 4)
    assert float(sd["diffusion_model.blocks.0.lin.alpha"]) == 4.0
    path = tmp_path / "l.safetensors"
    net.save(str(path), {"x": "1"}, dtype=torch.float32)
    _dit2, net2 = _net(tiny)
    assert net2.load_trainable(str(path)) == len(net.wrapped)
    for a, b in zip(net.parameters(), net2.parameters()):
        assert torch.equal(a, b)


def test_frozen_file_in_kohya_layout(tiny, tmp_path):
    from safetensors.torch import save_file
    dit, net = _net(tiny)
    sd = {"lora_unet_blocks_0_conv.lora_down.weight": torch.randn(2, 4, 3, 3),
          "lora_unet_blocks_0_conv.lora_up.weight": torch.randn(8, 2, 1, 1),
          "lora_unet_blocks_0_conv.alpha": torch.tensor(2.0),
          "lora_unet_blocks_1_lin.lora_down.weight": torch.randn(2, 8),
          "lora_unet_blocks_1_lin.lora_up.weight": torch.randn(8, 2)}
    save_file(sd, str(tmp_path / "k.safetensors"))
    assert net.add_file(str(tmp_path / "k.safetensors"), "context") == 2
    net.set_enabled("context", False)
    assert all(w.scales.get("context", 0) == 0 for w in net.wrapped.values())


def test_lokr_is_linear_only(tiny):
    dit, net = _net(tiny, "lokr")
    sd = net.state_dict()
    assert any(k.endswith("lin.lokr_w1") for k in sd)
    assert not any(".conv." in k or ".out." in k for k in sd)


def test_bake_equals_live_sum(tiny):
    dit, net = _net(tiny)
    with torch.no_grad():
        for p in net.parameters():
            p.normal_(0, 0.1)
    for w in net.wrapped.values():
        w.adapters["lora"][0].weight.data = w.adapters["lora"][0].weight.data.float()
    sd, ranks = net.bake(["lora"], dtype=torch.float32)
    w = net.wrapped["blocks.1.lin"]
    a, b = w.adapters["lora"]
    live = (b.weight @ a.weight) * w.scales["lora"]
    baked = sd["diffusion_model.blocks.1.lin.lora_up.weight"] @ sd["diffusion_model.blocks.1.lin.lora_down.weight"]
    alpha = float(sd["diffusion_model.blocks.1.lin.alpha"])
    assert torch.allclose(baked * alpha / ranks["blocks.1.lin"], live, atol=1e-5)


# ---- metadata / progress / utils ---------------------------------------------------------------------------
def test_metadata_for_family(qwen):
    from training.metadata import build_metadata
    md = build_metadata("qwenimage21", 1_700_000_000, title="t", trigger_phrase="ohwx", reso=(704, 704))
    assert md["modelspec.architecture"] == "Qwen-Image-2.1/lora"
    assert md["modelspec.usage_hint"] == "Include 'ohwx' in your prompt."
    assert md["modelspec.resolution"] == "704x704"
    assert all(v is not None for v in md.values())


def test_progress_parsing():
    from training.progress import TrainingProgressTracker
    t = TrainingProgressTracker(10)
    u = t.consume("steps:  12%|#2   | 12/100 [00:30<03:40,  2.50s/it, avr_loss=0.1234]")
    assert u["kind"] == "training" and u["step"] == 12 and u["average_loss_text"] == "0.1234"
    u = t.consume("epoch 3/10  avr_loss=0.1100  step=30  2.10s/step  lr=2.828e-04  peak VRAM 14.1 GB")
    assert u["kind"] == "epoch" and u["epoch"] == 3 and u["loss"] == pytest.approx(0.11)
    u = t.consume("[adaptive_lr] epoch 3: loss=0.1100 lr=2.83e-04->3.54e-04 clip=0% wnorm_Δ=+5% | PROBE UP (loss improving, streak 2)")
    assert u["kind"] == "adaptive" and u["action"] == "PROBE UP"
    u = t.consume("[adaptive_lr] epoch 5: loss=0.1 lr=4.00e-04 clip=0% wnorm_Δ=+1% | HOLD (capped) (loss improving, at max_lr)")
    assert u["action"] == "HOLD (capped)"
    assert t.consume("[cache] latents 3/12")["done"] == 3
    assert t.consume("[sample] epoch 4: 2 preview(s) -> D:/x")["phase"] == "complete"
    assert t.consume("random line") is None


def test_sample_name_parse():
    from training.pipeline import parse_sample_name
    assert parse_sample_name("my_e000012_01_20261002120000_1235.png") == (12, 1, 1235)
    assert parse_sample_name("other.png") is None


def test_output_name_validation():
    from training.train_utils import validate_output_name
    assert validate_output_name("ok_name") == "ok_name"
    for bad in ("", " x", "a/b", "a\nb", "x.", 'a"b'):
        with pytest.raises(ValueError):
            validate_output_name(bad)


def test_prune_keeps_at_least_one(tmp_path):
    from training.train_utils import list_state_dirs, prune_state_dirs
    for e in (1, 2, 3):
        (tmp_path / f"run-{e:06d}-state").mkdir()
    (tmp_path / "other-000001-state").mkdir()
    prune_state_dirs(str(tmp_path), "run", 0)
    assert [e for e, _ in list_state_dirs(str(tmp_path), "run")] == [3]
    assert (tmp_path / "other-000001-state").exists()


def test_step_scheduler_shapes():
    from training.train import _step_scheduler
    net = _Net()
    opt = _opt(net, 1.0)
    s = _step_scheduler(opt, "cosine", 2, 10)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        s.step()
    assert lrs[0] == pytest.approx(0.5) and lrs[1] == pytest.approx(1.0)
    assert lrs[-1] < 0.1
    assert math.isclose(lrs[2], 1.0)


def test_slider_and_finetune_presets_are_refused_not_run_as_loras():
    """A Fizgig Slider or Fine-tune preset changes nothing here (those modes are not ported yet): applied, its
    settings would silently train an ordinary LoRA."""
    from training import params as P
    from training import presets
    cur = P.defaults()
    for key in ("FAMILY_SLIDER", "FAMILY_FT", "KREA2_FINETUNE", "MINIMAX_FINETUNE"):
        new, rep = presets.apply({key: True, "NETWORK_DIM": 4, "LEARNING_RATE": 2e-4}, cur)
        assert new == cur and rep.blocked and not rep.applied, key
        assert "cannot train yet" in rep.messages()[0]
    new, rep = presets.apply({"FAMILY_SLIDER": False, "FAMILY_FT": False, "NETWORK_DIM": 4}, cur)   # Fizgig's LoRAs
    assert not rep.blocked and new["NETWORK_DIM"] == 4


def test_frozen_loha_and_diffusers_dot_spelling(tiny, tmp_path):
    """Fizgig 7.0.0: LoHa files load (the Hadamard delta), and diffusers' own `lora.down` / `lora.up` spelling."""
    from safetensors.torch import save_file
    from training.lora import LoHa, lycoris_scale_from_keys
    dit, net = _net(tiny)
    torch.manual_seed(1)
    loha = {"hada_w1_a": torch.randn(8, 2), "hada_w1_b": torch.randn(2, 8), "hada_w2_a": torch.randn(8, 2),
            "hada_w2_b": torch.randn(2, 8), "alpha": torch.tensor(1.0)}
    sd = {f"lora_unet_blocks_1_lin.{k}": v for k, v in loha.items()}
    sd.update({"transformer.blocks.0.lin.lora.down.weight": torch.randn(2, 8),
               "transformer.blocks.0.lin.lora.up.weight": torch.randn(8, 2)})
    save_file(sd, str(tmp_path / "h.safetensors"))
    assert net.add_file(str(tmp_path / "h.safetensors"), "context") == 2
    w = dit.blocks[1].lin
    assert isinstance(w.adapters["context"], LoHa)
    x = torch.randn(3, 8)
    delta = (loha["hada_w1_a"] @ loha["hada_w1_b"]) * (loha["hada_w2_a"] @ loha["hada_w2_b"])
    want = w.base(x) + lycoris_scale_from_keys(loha) * (x @ delta.T)
    net.set_enabled("lora", False)
    assert torch.allclose(w(x), want, atol=0.05 * float(want.abs().max()))     # bf16 adapter
    baked, ranks = net.bake(["context"])                 # a LoHa bakes through the SVD path like a LoKR
    assert ranks["blocks.1.lin"] > 0 and ranks["blocks.0.lin"] == 2


def test_earlier_exclusions_train_again_and_are_kept_per_family(tmp_path):
    """Fizgig 7.0.0: an image excluded by an earlier run of the same family is not skipped (it trains, marked as past
    its recaptions); another family's exclusion does not touch it; an old flat entry counts for every family."""
    import json as _json
    from training.loss_logger import PerImageLossWatch
    ds = tmp_path / "ds"
    ds.mkdir()
    for n in ("a", "b", "c"):
        (ds / f"{n}.txt").write_text(f"caption {n}", encoding="utf-8")
    keys = [str(ds / n) for n in ("a", "b", "c")]
    (ds / "tagscriber_excluded.json").write_text(_json.dumps({
        keys[0]: {"families": {"krea2": {"epoch": 3}}, "caption": "caption a"},
        keys[1]: {"families": {"klein": {"epoch": 3}}, "caption": "caption b"},
        keys[2]: {"epoch": 2, "reason": "old flat entry", "caption": "caption c"}}), encoding="utf-8")
    w = PerImageLossWatch(str(tmp_path / "out"), dataset_dir=str(ds), family="krea2")
    w.preflight(set(keys))
    assert not any(w.is_excluded([k]) for k in keys)                     # nothing is skipped from step 1
    assert w._known_hard == {keys[0], keys[2]}                            # this family's and the old flat entry
    w._record_exclusion(keys[1], 5)                                       # krea2 excludes b too: both records kept
    saved = _json.loads((ds / "tagscriber_excluded.json").read_text(encoding="utf-8"))
    assert set(saved[keys[1]]["families"]) == {"klein", "krea2"}
