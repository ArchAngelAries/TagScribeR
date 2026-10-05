"""fp8 frozen bases: fp8 / fp8-scaled checkpoints load as fp8, a bf16 file quantises to fp8, either feeds INT8, and
Auto follows Fizgig's Krea 2 ladder (INT8 > NF4 > fp8 > fp8 + swap). CPU, tiny random models."""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from training import quant  # noqa: E402
from training.modules import fp8  # noqa: E402

FP8 = torch.float8_e4m3fn


def test_quantize_weight_block_channel_and_roundtrip():
    torch.manual_seed(0)
    w = torch.randn(8, 128)
    q, s = fp8.quantize_weight(w)
    assert q.dtype == FP8 and s.shape == (8, 2, 1)                 # 64-feature blocks
    back = fp8.dequantize(q, s, torch.float32)
    assert (back - w).abs().max() < 0.08 * w.abs().max()            # e4m3: 3 mantissa bits
    q, s = fp8.quantize_weight(torch.randn(8, 100))
    assert s.shape == (8, 1)                                        # not a multiple of 64: per-channel
    assert fp8.dequantize(q, None, torch.float32).dtype == torch.float32


def test_forward_matches_dequantised_weight_for_every_scale_layout():
    torch.manual_seed(0)
    lin = nn.Linear(128, 8)
    x = torch.randn(3, 128)
    for scale in (None, torch.tensor(0.5), torch.full((8,), 0.25), fp8.quantize_weight(lin.weight.data)[1]):
        m = nn.Linear(128, 8)
        m.load_state_dict(lin.state_dict())
        q = fp8.quantize_weight(lin.weight.data)[0]
        m.weight.data = q
        fp8.attach(m, None if scale is None else fp8._reshape_scale(scale))
        want = nn.functional.linear(x, fp8.dense_weight(m, torch.float32), lin.bias)
        assert torch.allclose(m(x), want, atol=1e-5)
        assert m.weight.dtype == FP8 and not m.weight.requires_grad


def test_krea2_loads_fp8_scaled_and_plain_fp8_files(tmp_path):
    """The owner's case: an fp8 RAW checkpoint. ComfyUI's fp8_scaled layout (weight + .weight_scale + .comfy_quant)
    and a plain fp8 cast both load, stay fp8, and compute what the stored weights say."""
    from safetensors.torch import save_file

    from tests.test_training_krea2 import TINY, _dit
    from training.families.krea2.model import load_krea2_dit
    torch.manual_seed(0)
    ref = _dit(dtype=torch.float32)
    sd = {k: v.to(torch.float32) for k, v in ref.state_dict().items()}
    linear_w = [n + ".weight" for n, m in ref.named_modules() if isinstance(m, nn.Linear) and n.startswith("blocks.")]
    assert linear_w
    scaled = dict(sd)
    for k in linear_w:                                              # per-tensor scale, as ComfyUI writes it
        s = sd[k].abs().max() / torch.finfo(FP8).max
        scaled[k] = (sd[k] / s).to(FP8)
        scaled[k[: -len(".weight")] + ".weight_scale"] = s
        scaled[k[: -len(".weight")] + ".comfy_quant"] = torch.zeros(1, dtype=torch.uint8)
    p = str(tmp_path / "krea2_raw_fp8_scaled.safetensors")
    save_file(scaled, p)
    assert fp8.file_is_fp8(p)
    dit = load_krea2_dit(p, device="cpu", config=TINY)
    mods = dict(dit.named_modules())
    for k in linear_w:
        m = mods[k[: -len(".weight")]]
        assert m.weight.dtype == FP8 and m._is_fp8
        assert torch.allclose(fp8.dense_weight(m, torch.float32), sd[k], atol=0.07 * float(sd[k].abs().max()))
    assert dit.first.weight.dtype == torch.bfloat16                # non-fp8 tensors are cast as before
    plain = {k: (v.to(FP8) if k in linear_w else v) for k, v in sd.items()}
    p2 = str(tmp_path / "plain_fp8.safetensors")
    save_file(plain, p2)
    dit2 = load_krea2_dit(p2, device="cpu", config=TINY)
    assert sum(1 for m in dit2.modules() if getattr(m, "_is_fp8", False)) == len(linear_w)


def _tiny_krea2(tmp_path, as_fp8):
    from safetensors.torch import save_file

    from tests.test_training_krea2 import _dit
    torch.manual_seed(0)
    sd = {k: v.to(torch.float32) for k, v in _dit(dtype=torch.float32).state_dict().items()}
    if as_fp8:
        for k in [k for k in sd if k.startswith("blocks.") and k.endswith(".weight") and sd[k].ndim == 2]:
            s = sd[k].abs().amax(dim=1) / torch.finfo(FP8).max       # per-channel [out], ComfyUI's other layout
            sd[k] = (sd[k] / s[:, None]).to(FP8)
            sd[k[: -len(".weight")] + ".weight_scale"] = s
    p = str(tmp_path / ("fp8.safetensors" if as_fp8 else "bf16.safetensors"))
    save_file(sd, p)
    return p


@pytest.mark.parametrize("as_fp8", [True, False])
def test_fp8_base_trains_a_lora_step(tmp_path, as_fp8):
    """load_base(precision="fp8") from an fp8 file (kept as stored) and from a bf16 file (quantised at load): the
    LoRA trains, the base stays frozen fp8."""
    from tests.test_training_krea2 import TINY, _cond
    from training.families.krea2.driver import Krea2Driver
    from training.families.krea2 import model as km
    from training.lora import FamilyLoRA
    from training.registry import get
    drv = Krea2Driver()
    drv.description = get("krea2")
    path = _tiny_krea2(tmp_path, as_fp8)
    orig = km.load_krea2_dit
    drv.load_dit = lambda p, device: orig(p, device=device, config=TINY).eval().requires_grad_(False)
    dit, swap = quant.load_base(drv, path, torch.device("cpu"), "fp8", 0)
    n_fp8 = sum(1 for m in dit.modules() if getattr(m, "_is_fp8", False))
    assert swap == 0 and n_fp8 == len(quant._targets(dit, drv)) > 0
    net = FamilyLoRA(dit, drv)
    net.add_trainable(4, 4)
    loss, info = drv.training_loss(dit, torch.randn(1, 16, 8, 8), _cond(1), torch.Generator().manual_seed(0))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in net.parameters())
    assert all(not p.requires_grad for n, p in dit.named_parameters() if ".adapters." not in n)


def test_fp8_file_is_a_source_for_int8_and_for_bf16(tmp_path):
    from tests.test_training_krea2 import TINY
    from training.families.krea2 import model as km
    from training.families.krea2.driver import Krea2Driver
    from training.registry import get
    drv = Krea2Driver()
    drv.description = get("krea2")
    drv.load_dit = lambda p, device: km.load_krea2_dit(p, device=device, config=TINY).eval().requires_grad_(False)
    path = _tiny_krea2(tmp_path, True)
    ref = km.load_krea2_dit(path, device="cpu", config=TINY)
    name, _m = quant._targets(ref, drv)[0]
    want = fp8.dense_weight(dict(ref.named_modules())[name], torch.float32)
    dit = km.load_krea2_dit(path, device="cpu", config=TINY)
    quant.quantize(dit, drv, "int8", "cpu")
    m = dict(dit.named_modules())[name]
    assert m.weight.dtype == torch.int8 and not getattr(m, "_is_fp8", False) and "scale_weight" not in m._buffers
    got = m.weight.float() * m._int8_wscale.reshape(-1, 1)
    assert torch.allclose(got, want, atol=0.02 * float(want.abs().max()))
    dit = km.load_krea2_dit(path, device="cpu", config=TINY)
    quant.quantize(dit, drv, "bf16", "cpu")
    m = dict(dit.named_modules())[name]
    assert m.weight.dtype == torch.bfloat16 and "forward" not in m.__dict__
    assert torch.allclose(m.weight.float(), want, atol=0.02 * float(want.abs().max()))


class _Drv:
    def max_blocks_to_swap(self, dit=None):
        return 26


def test_auto_follows_fizgig_krea2_ladder(monkeypatch):
    from training.registry import get
    desc = get("krea2")
    avail = {"int8": True, "nf4": True}
    monkeypatch.setattr(quant, "available", lambda p, device=None: (avail.get(p, True), "" if avail.get(p, True)
                                                                    else f"{p} unavailable"))
    plan = lambda free: quant.plan(desc, _Drv(), "auto", -1, free_gb=free, megapixels=0.25)[:2]  # noqa: E731
    assert plan(24.0) == ("int8", 0)                    # INT8 leads where it fits (16.2 + 1.5 headroom)
    assert plan(16.0) == ("nf4", 0)                     # then NF4 with no swap
    p, n = plan(11.0)                                   # below NF4: fp8 with block swap
    assert p == "fp8" and 0 < n <= 26
    avail.update(int8=False, nf4=False)                 # a card with no _int_mm and no bitsandbytes
    assert plan(24.0) == ("fp8", 0)
    p, n = plan(18.5)                                   # e.g. a 20 GB card: fp8 with a few blocks swapped
    assert p == "fp8" and n == 5                        # ceil((18.7 - 17.0) / 0.42)
    # an explicit INT8 the GPU cannot run degrades to fp8, as Fizgig's strategy does
    assert quant.plan(desc, _Drv(), "int8", 0, free_gb=24.0, megapixels=0.25)[0] == "fp8"


def test_precision_label_and_legacy_key():
    from training import params as P
    from training import pipeline, presets
    from training.registry import get
    desc = get("krea2")
    assert P.PRECISION_LABELS["fp8"] in P.options_for(P.BY_KEY["FAMILY_PRECISION"], desc)
    new, rep = presets.apply({"QUANT_4BIT_MODE": "fp8"}, P.defaults(), desc)
    assert rep.refused == [] and pipeline.precision_key(new, desc) == "fp8"
