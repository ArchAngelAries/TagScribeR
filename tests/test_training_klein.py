"""FLUX.2 Klein Base 9B family: description, the seven presets, the DiT / driver objective on a tiny random config,
every timestep mode against a direct transcription of Fizgig's formulas, Model Area block targeting, kohya LoRA keys,
the loader's fp8 refusal, the AE packing and the text encoder's hidden-state layout. CPU only, no weights."""
import math
import re

import pytest

torch = pytest.importorskip("torch")

from training import params as P  # noqa: E402
from training import pipeline, presets, registry  # noqa: E402
from training.families.klein import sampling as S  # noqa: E402
from training.families.klein.description import KLEIN_9B  # noqa: E402
from training.families.klein.driver import (AREA_PATTERNS, DETAILS_PATTERNS, IDENTITY_PATTERNS,  # noqa: E402
                                            STYLE_COMP_PATTERNS, KleinDriver)
from training.families.klein.model import Flux2Params, KleinDiT, Klein9BParams, load_klein_dit  # noqa: E402
from training.lora import FamilyLoRA  # noqa: E402

registry.register(KLEIN_9B)

TINY = Flux2Params(in_channels=16, context_in_dim=24, hidden_size=64, num_heads=4, depth=2, depth_single_blocks=3,
                   axes_dim=[4, 4, 4, 4], theta=2000, mlp_ratio=3.0, use_guidance_embed=False)


@pytest.fixture(scope="module")
def desc():
    return registry.get("klein9b")


@pytest.fixture
def driver(desc):
    d = KleinDriver()
    d.description = desc
    return d


def _dit(params=TINY, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    return KleinDiT(params).to(dtype).requires_grad_(False)


def _cond(batch, text_len=10, dim=24):
    return {"text_embed": torch.randn(batch, text_len, dim, dtype=torch.bfloat16)}


# ---- description, registry, presets ---------------------------------------------------------------------
def test_description_is_valid_and_registered(desc):
    assert desc.validate() == []
    assert desc is KLEIN_9B and registry.by_arch_id("klein9b") is desc
    assert (desc.key, desc.arch_id, desc.lora_name_suffix) == ("klein9b", "klein9b", "k9b")
    assert "_" not in desc.arch_id
    assert (desc.latent_channels, desc.spatial_factor, desc.bucket_step) == (128, 16, 16)
    assert desc.precisions == ("bf16", "fp8", "int8", "nf4") and set(desc.train_memory) == {"fp8", "int8", "nf4"}
    assert desc.auto_precisions == ("int8", "nf4")                      # Fizgig 7.0.1: INT8, then NF4
    assert desc.network_types == ("lora", "lokr") and desc.ema_default == "Off"        # Fizgig 7.0.1
    assert "automagic3" in desc.optimizers and "lion8bit" in desc.optimizers
    assert desc.speed_loras == () and desc.preview_speed() is None
    assert (desc.preview_steps, desc.preview_cfg, desc.preview_width) == (40, 4.5, 768)
    assert {desc.pref_for(r) for r in ("dit", "vae", "text_encoder", "preview_dit")} == set(desc.pref_keys)
    assert all(k in P.BY_KEY for k in desc.family_options)
    assert {k for k in desc.family_options} <= set(P.DRIVER_OPTIONS)


def test_seven_presets_in_fizgig_order_with_exact_values(desc):
    names = [n for n, _ in desc.presets]
    assert names == ["✨ Old Reliable (rank 16, full model, single subject)",
                     "✨ Old Reliable - Flavour 8 (rank 8, full model, single subject)",
                     "✨ Identity (rank 8, single subject)", "✨ Identity (rank 8, harder dataset)",
                     "✨ Multi-Character (rank 16, multi character or concept)", "✨ Style (late timesteps)",
                     "✨ Style+Composition (all timesteps)"]
    # (rank, lr, epochs, adaptive min, adaptive max, area, min ts, max ts) - Fizgig 7.0.1 families/klein.py:192-202
    want = [(16, 1e-4, 55, "1e-4", "4e-4", "Full Model", "", ""), (8, 1e-4, 55, "1e-4", "4e-4", "Full Model", "", ""),
            (8, 4e-4, 15, "2e-4", "4e-4", "Identity", "", ""), (8, 4e-4, 20, "2e-4", "4e-4", "Identity", "", ""),
            (16, 2e-4, 50, "1e-4", "4e-4", "Identity", "", ""), (4, 4e-4, 15, "1e-5", "4e-4", "Style", "0", "400"),
            (4, 4e-4, 15, "1e-5", "4e-4", "Style+Composition", "", "")]
    for (name, v), w in zip(desc.presets, want):
        assert (v["NETWORK_DIM"], v["LEARNING_RATE"], v["MAX_TRAIN_EPOCHS"], v["ADAPTIVE_LR_MIN"],
                v["ADAPTIVE_LR_MAX"], v["TARGET_LAYERS"], v["MIN_TIMESTEP"], v["MAX_TIMESTEP"]) == w, name
        assert v["NETWORK_ALPHA"] == v["NETWORK_DIM"] and v["ADAPTIVE_LR"] is True
        assert (v["SAVE_EVERY_N_EPOCHS"], v["SEED"], v["OPTIMIZER_TYPE"]) == (1, 42, "adamw8bit")
        assert (v["NETWORK_TYPE"], v["FAMILY_EMA"], v["FAMILY_PRECISION"]) == ("LoRA (standard)", "Off",
                                                                             "Auto (fits your free VRAM)")


def test_every_preset_key_resolves_without_refusals(desc):
    for name, values in desc.presets:
        migrated, notes, ignored = presets.migrate_legacy(values)
        assert [k for k in migrated if k not in P.BY_KEY] == [], name
        assert notes == [] and ignored == []
        new, rep = presets.apply(values, P.defaults(), desc)
        assert rep.refused == [] and rep.ignored == [], (name, rep.refused, rep.ignored)


def test_first_preset_and_style_timestep_conversion(desc):
    tp = presets.TrainingPresets(desc)
    assert tp.default_name == "✨ Old Reliable (rank 16, full model, single subject)"
    new, _ = presets.apply(tp.load(tp.default_name), P.defaults(), desc)
    assert (new["NETWORK_DIM"], new["NETWORK_ALPHA"], new["MAX_TRAIN_EPOCHS"], new["LEARNING_RATE"]) == (16, 16, 55, 1e-4)
    assert new["ADAPTIVE_LR"] is True and new["ADAPTIVE_LR_MIN"] == "1e-4" and new["TARGET_LAYERS"] == "Full Model"
    assert (new["MIN_TIMESTEP"], new["MAX_TIMESTEP"]) == (0.0, 1.0)            # blank boxes = full range
    style, _ = presets.apply(dict(desc.presets)["✨ Style (late timesteps)"], P.defaults(), desc)
    assert (style["MIN_TIMESTEP"], style["MAX_TIMESTEP"]) == (0.0, 0.4)        # 0-400 of 1000
    assert style["TARGET_LAYERS"] == "Style" and P.first_token(style["ADAPTIVE_LR_MIN"]) == "1e-5"


def test_legacy_klein_keys_map_or_are_ignored(desc):
    legacy = {"FP8": True, "SCALED": True, "QUANT_4BIT": False, "QUANT_4BIT_MODE": "auto",
              "NETWORK_DROPOUT": 0, "LORA_LR_RATIO": 1, "FP8_TEXT_ENCODER": True, "LR_DECAY_STEPS": "",
              "GRADIENT_CHECKPOINTING": True, "WEIGHTING_SCHEME": "none", "MODE_SCALE": "1.29",
              "TARGET_LAYERS": "Identity Blocks", "ATTENTION_MECHANISM": "flash3", "MAX_TIMESTEP": "1000", "MIN_TIMESTEP": "250",
              "TRAINING_BLOCKS": {"double_blocks.0": True, "double_blocks.1": False, "single_blocks.5": True},
              "TIMESTEP_SAMPLING": "flux2_shift", "DISCRETE_FLOW_SHIFT": "0", "SIGMOID_SCALE": "1.0"}
    new, rep = presets.apply(legacy, P.defaults(), desc)
    assert rep.refused == [], rep.refused
    assert new["ATTENTION_MECHANISM"] == "flash3"                            # carried, no longer ignored
    assert new["TARGET_LAYERS"] == "Identity" and new["TRAINING_BLOCKS"] == "double_0, single_5"     # 7.0.1 ids
    assert (new["MIN_TIMESTEP"], new["MAX_TIMESTEP"]) == (0.25, 1.0)
    assert new["FAMILY_PRECISION"] == P.PRECISION_LABELS["auto"]
    assert {"FP8", "SCALED", "NETWORK_DROPOUT", "LORA_LR_RATIO", "FP8_TEXT_ENCODER",
            "LR_DECAY_STEPS", "GRADIENT_CHECKPOINTING", "WEIGHTING_SCHEME", "MODE_SCALE"} <= set(rep.ignored)
    bad, rep = presets.apply({"TIMESTEP_SAMPLING": "qwen_shift"}, P.defaults(), desc)
    assert rep.refused and bad["TIMESTEP_SAMPLING"] == "flux2_shift"          # Fizgig: an unsupported mode is refused


# ---- block map and Model Area ------------------------------------------------------------------------------
def test_block_map_is_the_112_block_linears(desc, driver):
    names = driver.lora_target_names(None)
    assert len(names) == len(set(names)) == 8 * 8 + 24 * 2
    assert driver.block_of("double_blocks.3.txt_mlp.2") == "double_3"                 # Fizgig 7.0.1 ids
    assert driver.block_of("single_blocks.23.linear2") == "single_23" and driver.block_of("img_in") is None
    dit = _dit()
    lin = {n for n, m in dit.named_modules() if isinstance(m, torch.nn.Linear)
           and n.startswith(("double_blocks.", "single_blocks."))}
    assert set(driver.lora_target_names(dit)) == lin == set(driver.quant_target_names(dit))
    assert len(lin) == 2 * 8 + 3 * 2


def _expected_modules(patterns):
    """Fizgig create_network: modules of the two block classes whose name fullmatches an include pattern."""
    allm = KleinDriver().lora_target_names(None)
    return {m for m in allm if any(re.fullmatch(p, m) for p in patterns)}


def test_model_area_presets_select_exactly_fizgigs_blocks(driver):
    ids = lambda a: driver.resolve_blocks(a)  # noqa: E731
    assert ids("Full Model") is None
    assert ids("Identity") == {f"single_{i}" for i in range(1, 17)}
    assert ids("Details") == {f"single_{i}" for i in range(12, 24)}
    style = {f"double_{i}" for i in range(8)} | {"single_0", "single_1"}
    assert ids("Style") == ids("Style+Composition") == style
    for area, pats in AREA_PATTERNS.items():
        got = {m for m in driver.lora_target_names(None) if driver.block_of(m) in ids(area)}
        assert got == _expected_modules(pats), area
    assert IDENTITY_PATTERNS == [r".*single_blocks\.(1[0-6]|[1-9])\..*"]
    assert DETAILS_PATTERNS == [r".*single_blocks\.(1[2-9]|2[0-3])\..*"]
    assert STYLE_COMP_PATTERNS == [r".*double_blocks\..*", r".*single_blocks\.[01]\..*"]


def test_custom_blocks_and_configure(driver):
    assert driver.resolve_blocks("Custom", "double_2, single_9") == {"double_2", "single_9"}
    assert driver.resolve_blocks("Custom", "double_blocks.2, single_blocks.9") == {"double_2", "single_9"}   # old ids
    assert driver.resolve_blocks("Custom", {"single_1": True, "single_2": False}) == {"single_1"}
    assert driver.resolve_blocks("Custom", "") is None                         # Fizgig: empty Custom = the full model
    with pytest.raises(ValueError, match="unknown block"):
        driver.resolve_blocks("Custom", "single_24")
    driver.configure(target_layers="Details", timestep_sampling="logsnr", sigmoid_scale=1.5)
    assert driver.trainable_blocks() == {f"single_{i}" for i in range(12, 24)}
    assert (driver.timestep_sampling, driver.sigmoid_scale) == ("logsnr", 1.5)
    with pytest.raises(ValueError):
        driver.configure(timestep_sampling="qwen_shift")
    with pytest.raises(ValueError):
        driver.configure(no_such_option=1)


def test_block_targeting_trains_only_those_modules(desc, driver):
    dit = _dit()
    net = FamilyLoRA(dit, driver)
    blocks = driver.resolve_blocks("Custom", "double_1, single_2")
    net.add_trainable(4, 4, blocks=blocks)
    assert len(net.trainable_modules()) == 8 + 2
    assert {k.rsplit(".", 2)[0] for k in net.state_dict() if k.endswith(".alpha")} == {
        k.rsplit(".", 1)[0] for k in net.state_dict() if k.endswith(".alpha")}


# ---- kohya keys ---------------------------------------------------------------------------------------------
def test_lora_keys_are_fizgigs_kohya_names_and_round_trip(desc, driver, tmp_path):
    dit = _dit()
    # Fizgig networks/lora.py create_network: Linears inside DoubleStreamBlock / SingleStreamBlock, dots -> underscores,
    # exclude patterns fullmatched against the block-relative path (lora_klein.py)
    exclude = [re.compile(p) for p in (r".*(img_mod\.lin|txt_mod\.lin|modulation\.lin).*", r".*(norm).*")]
    expected = set()
    for bname, block in dit.named_modules():
        if type(block).__name__ in ("DoubleStreamBlock", "SingleStreamBlock"):
            for cname, child in block.named_modules():
                if type(child).__name__ == "Linear":
                    full = f"{bname}.{cname}"
                    if not any(p.fullmatch(full) for p in exclude):
                        expected.add("lora_unet_" + full.replace(".", "_"))
    net = FamilyLoRA(dit, driver)
    net.add_trainable(4, 8)
    sd = net.state_dict()
    got = {k.rsplit(".", 2)[0] for k in sd if k.endswith(".lora_down.weight")}
    assert got == expected and len(expected) == 16 + 6
    assert {k.split(".", 1)[1] for k in sd} == {"lora_down.weight", "lora_up.weight", "alpha"}
    assert sd["lora_unet_single_blocks_0_linear1.alpha"].item() == 8.0
    for excluded in ("img_in", "txt_in", "time_in", "final_layer", "double_stream_modulation", "single_stream_mod"):
        assert not any(k.startswith(f"lora_unet_{excluded}") for k in sd), excluded
    assert not any("modulation" in k for k in sd)
    path = tmp_path / "k.safetensors"
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    net.save(str(path))
    net2 = FamilyLoRA(_dit(), driver)
    net2.add_trainable(4, 8)
    assert net2.load_trainable(str(path)) == len(expected)
    for k, v in net.state_dict().items():
        assert torch.equal(net2.state_dict()[k], v), k


# ---- timestep modes against Fizgig's formulas (trainer.py:791-960) --------------------------------------------
def _lin(x, x1=256, y1=0.5, x2=4096, y2=1.15):
    m = (y2 - y1) / (x2 - x1)
    return m * x + (y1 - m * x1)


def _ref_base(mode, n, h, w, g, scale=1.0, dshift=3.0, mean=0.0, std=1.0):
    if mode == "uniform":
        return torch.rand((n,), generator=g)
    if mode == "sigmoid":
        return torch.sigmoid(scale * torch.randn((n,), generator=g))
    if mode in ("shift", "flux_shift", "flux2_shift"):
        shift = dshift if mode == "shift" else math.exp(_lin((h // 2) * (w // 2)) if mode == "flux_shift"
                                                         else _lin(h * w))
        t = (torch.randn((n,), generator=g) * scale).sigmoid()
        return (t * shift) / (1 + (shift - 1) * t)
    if mode == "logsnr":
        return torch.sigmoid(-torch.normal(mean=mean, std=std, size=(n,), generator=g) / 2)
    if mode == "qinglong_flux":
        d = torch.rand((n,), generator=g)
        mid, l1, l2 = d < 0.80, (d >= 0.80) & (d < 0.875), d >= 0.875
        t = torch.zeros((n,))
        if mid.any():
            s = math.exp(_lin((h // 2) * (w // 2)))
            tm = (torch.randn((int(mid.sum()),), generator=g) * scale).sigmoid()
            t[mid] = (tm * s) / (1 + (s - 1) * tm)
        if l1.any():
            t[l1] = torch.sigmoid(-torch.normal(mean=mean, std=std, size=(int(l1.sum()),), generator=g) / 2)
        if l2.any():
            t[l2] = torch.sigmoid(-torch.normal(mean=5.36, std=1.0, size=(int(l2.sum()),), generator=g) / 2)
        return t
    raise AssertionError(mode)


@pytest.mark.parametrize("mode", ["uniform", "sigmoid", "shift", "flux_shift", "flux2_shift", "logsnr", "qinglong_flux"])
def test_timestep_modes_match_fizgigs_formulas(mode):
    kw = dict(sigmoid_scale=1.3, shift=2.5, logit_mean=0.4, logit_std=1.2)
    for (lo, hi) in ((0.0, 1.0), (0.0, 0.4), (0.25, 0.9)):
        t, tm = S.sample_timesteps(mode, 64, 32, 24, torch.Generator().manual_seed(5), min_t=lo, max_t=hi, **kw)
        ref = _ref_base(mode, 64, 32, 24, torch.Generator().manual_seed(5), 1.3, 2.5, 0.4, 1.2) * (hi - lo) + lo
        assert torch.allclose(t, ref, atol=1e-6), (mode, lo, hi)
        assert torch.allclose(tm, (ref * 1000 + 1) / 1000, atol=1e-6)         # Fizgig: timesteps += 1, then / 1000
        assert float(t.min()) >= lo - 1e-6 and float(t.max()) <= hi + 1e-6


def test_flux2_shift_uses_the_packed_token_count():
    h, w = 32, 32                                       # 1024 tokens: mu = 0.5 + 0.65 * (1024 - 256) / 3840
    mu = 0.5 + 0.65 * (1024 - 256) / 3840
    g = torch.Generator().manual_seed(3)
    z = torch.randn((8,), generator=torch.Generator().manual_seed(3))
    s = math.exp(mu)
    want = (z.sigmoid() * s) / (1 + (s - 1) * z.sigmoid())
    t, _ = S.sample_timesteps("flux2_shift", 8, h, w, g)
    assert torch.allclose(t, want, atol=1e-6)


def test_sigma_mode_is_the_discrete_schedule_in_the_window():
    t, tm = S.sample_timesteps("sigma", 200, 16, 16, torch.Generator().manual_seed(1), min_t=0.0, max_t=0.4)
    assert torch.equal(t, tm)                             # no +0.001 in sigma mode
    assert float(t.max()) <= 0.4 + 1e-6 and float(t.min()) >= 0.0
    u = torch.rand((200,), generator=torch.Generator().manual_seed(1))
    idx = (u * (1000 - 600.0) + 600.0).long().clamp(0, 999)       # lo_idx = n - 400, hi_idx = n - 0
    assert torch.allclose(t, torch.linspace(1, 0, 1001)[idx])
    assert torch.allclose(((t * 1000).round() % 1), torch.zeros(200), atol=1e-4)


def test_preserve_distribution_rejects_into_the_window():
    t, _ = S.sample_timesteps("flux2_shift", 32, 16, 16, torch.Generator().manual_seed(2), min_t=0.2, max_t=0.5,
                              preserve=True)
    assert t.shape == (32,) and float(t.min()) >= 0.2 and float(t.max()) <= 0.5
    with pytest.raises(ValueError):
        S.sample_timesteps("nope", 2, 4, 4, torch.Generator())


# ---- the DiT and the objective ----------------------------------------------------------------------------
def test_training_loss_batch_one_and_two_with_grads(desc, driver):
    dit = _dit()
    dit.enable_gradient_checkpointing(True)
    dit.train()
    net = FamilyLoRA(dit, driver)
    net.add_trainable(4, 4)
    assert len(net.trainable_modules()) == 22
    for batch in (1, 2):
        for p in net.parameters():
            p.grad = None
        loss, info = driver.training_loss(dit, torch.randn(batch, 16, 6, 4), _cond(batch),
                                          torch.Generator().manual_seed(1), min_t=0.2, max_t=0.8)
        loss.backward()
        assert torch.isfinite(loss) and 0.2 <= info["t"] <= 0.8
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()), batch
        assert any(p.grad.abs().sum() > 0 for p in net.parameters())


def test_loss_is_mse_to_noise_minus_latents_with_t_plus_a_thousandth(desc, driver):
    seen = {}

    class Probe(torch.nn.Module):
        def forward(self, x, x_ids, timesteps, ctx, ctx_ids, guidance=None):
            seen.update(x=x, ids=x_ids, t=timesteps, ctx=ctx, cids=ctx_ids)
            return torch.zeros_like(x)
    lat = torch.randn(1, 16, 6, 4)
    g = torch.Generator().manual_seed(11)
    loss, info = driver.training_loss(Probe(), lat, _cond(1), g, min_t=0.0, max_t=1.0)
    g = torch.Generator().manual_seed(11)
    noise = torch.randn(lat.shape, generator=g)
    t, tm = S.sample_timesteps("flux2_shift", 1, 6, 4, g)
    assert torch.allclose(loss, ((noise - lat) ** 2).mean(), atol=1e-6)
    assert torch.allclose(seen["t"], tm) and torch.allclose(tm - t, torch.tensor(0.001), atol=1e-6)
    t4 = t.view(1, 1, 1, 1)
    want = ((1 - t4) * lat + t4 * noise).flatten(2).transpose(1, 2).to(torch.bfloat16)
    assert torch.equal(seen["x"], want) and seen["x"].shape == (1, 24, 16)
    assert seen["ids"].shape == (1, 24, 4) and seen["ids"][0, 5].tolist() == [0, 1, 1, 0]       # [t, h, w, l]
    assert seen["cids"][0, 3].tolist() == [0, 0, 0, 3] and seen["ctx"].shape == (1, 10, 24)


def test_block_swap_does_not_change_the_loss(desc, driver):
    params = Flux2Params(**{**TINY.__dict__, "depth": 4, "depth_single_blocks": 6})
    args = (torch.randn(1, 16, 6, 4), _cond(1))

    def loss_of(swap):
        dit = _dit(params)
        if swap:
            driver.enable_block_swap(dit, swap, torch.device("cpu"), True)
        return driver.training_loss(dit, *args, torch.Generator().manual_seed(3))[0].item()
    assert loss_of(1) == pytest.approx(loss_of(0), rel=1e-5)


def test_swap_formula_matches_fizgigs_real_layout():
    with torch.device("meta"):
        real = KleinDiT(Klein9BParams())
    assert (len(real.double_blocks), len(real.single_blocks)) == (8, 24)
    assert real.max_block_swap() == 16 and KleinDriver().max_blocks_to_swap(real) == 16
    assert KleinDiT.swap_split(12, 8, 24) == (5, 15) and KleinDiT.swap_split(16, 8, 24) == (6, 18)
    assert real.img_in.in_features == 128 and real.txt_in.in_features == 12288
    blocks = sum(p.numel() for n, p in real.named_parameters() if n.startswith(("double_blocks.", "single_blocks.")))
    assert blocks == pytest.approx(8.72e9, rel=0.01) and sum(p.numel() for p in real.parameters()) == 9078581248
    n_lin = sum(1 for n, m in real.named_modules() if isinstance(m, torch.nn.Linear)
                and n.startswith(("double_blocks.", "single_blocks.")))
    assert n_lin == 112


def test_generate_shape_and_cfg(desc, driver):
    dit = _dit()
    cond = {"text_embed": torch.randn(10, 24)}                      # unbatched, as the preview loop holds it
    a = driver.generate(dit, cond, 64, 48, steps=2, seed=7, cfg=1.0)
    b = driver.generate(dit, cond, 64, 48, steps=2, seed=7, cfg=3.0, neg_cond={"text_embed": torch.randn(10, 24)})
    assert a.shape == b.shape == (1, 16, 3, 4)
    assert not torch.equal(a, b) and torch.isfinite(a.float()).all() and torch.isfinite(b.float()).all()
    assert torch.equal(a, driver.generate(dit, cond, 64, 48, steps=2, seed=7))
    assert driver.initial_noise(7, 64, 48).shape == (1, 128, 3, 4)
    seen = []
    driver.generate(dit, cond, 64, 48, steps=3, seed=1, on_step=lambda i, n: seen.append((i, n)))
    assert seen == [(0, 3), (1, 3), (2, 3)]
    sched = S.get_schedule(4, 1024)
    assert sched[0] == 1.0 and sched[-1] == 0.0 and sched == sorted(sched, reverse=True) and len(sched) == 5


# ---- autoencoder, text encoder, loader ---------------------------------------------------------------------
def test_vae_packing_and_driver_encode_decode(driver):
    from training.families.klein.vae import AutoEncoder, AutoEncoderParams, _pack, _unpack
    x = torch.randn(2, 32, 8, 12)
    ref = x.reshape(2, 32, 4, 2, 6, 2).permute(0, 1, 3, 5, 2, 4).reshape(2, 128, 4, 6)      # (c pi pj) channel order
    assert torch.equal(_pack(x, [2, 2]), ref) and torch.equal(_unpack(ref, [2, 2]), x)
    ae = AutoEncoder(AutoEncoderParams(ch=32, ch_mult=[1, 2], num_res_blocks=1, z_channels=32)).eval()
    import numpy as np
    imgs = [np.random.randint(0, 255, (32, 48, 3), dtype=np.uint8) for _ in range(2)]
    lats = driver.encode_images(ae, imgs)
    assert len(lats) == 2 and lats[0].shape == (128, 32 // 4, 48 // 4) and lats[0].dtype == torch.float32
    im = driver.decode(ae, lats[0][None], 48, 32)
    assert im.size == (48, 32) and im.mode == "RGB"


def test_text_encoder_layout_on_a_tiny_qwen3():
    transformers = pytest.importorskip("transformers")
    from training.families.klein import embedder as E
    cfg = E.build_config()
    assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.num_key_value_heads) == (36, 4096, 8)
    theta = getattr(cfg, "rope_theta", None) or cfg.rope_parameters["rope_theta"]
    assert theta == 1000000
    assert E.OUTPUT_LAYERS_QWEN3 == (9, 18, 27) and E.MAX_LENGTH == 512
    tiny = E.build_config(hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=8, vocab_size=100)
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(tiny).eval()

    class Tok:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
            assert (tokenize, add_generation_prompt, enable_thinking) == (False, True, False)
            return "<u>" + messages[0]["content"]

        def __call__(self, text, return_tensors, padding, truncation, max_length):
            assert (padding, truncation, max_length) == ("max_length", True, 12)
            ids = [ord(c) % 90 + 1 for c in text][:max_length]
            n = len(ids)
            return {"input_ids": torch.tensor([ids + [0] * (max_length - n)]),
                    "attention_mask": torch.tensor([[1] * n + [0] * (max_length - n)])}
    enc = E.KleinTextEncoder(device="cpu", dtype=torch.float32, model=model, tokenizer=Tok(), max_length=12,
                             layers=(1, 2, 3))
    out = enc.encode(["a cat", "dog"])
    assert out.shape == (2, 12, 3 * 32)
    tok = Tok()("<u>a cat", "pt", "max_length", True, 12)
    hs = model.model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"], output_hidden_states=True,
                     use_cache=False).hidden_states
    want = torch.cat([model.model.norm(hs[k]) for k in (1, 2, 3)], dim=-1)             # (c d) with c major
    assert torch.allclose(out[0:1], want, atol=1e-5)


def test_loader_roundtrip_and_wrong_keys(tmp_path):
    from safetensors.torch import save_file
    dit = _dit()
    path = tmp_path / "klein.safetensors"
    save_file({k: v.contiguous() for k, v in dit.state_dict().items()}, str(path))
    loaded = load_klein_dit(path, device="cpu", params=TINY).eval()
    x = dict(x=torch.randn(1, 24, 16, dtype=torch.bfloat16), x_ids=S.pack_img(torch.zeros(1, 16, 6, 4))[1],
             timesteps=torch.tensor([0.5]), ctx=torch.randn(1, 10, 24, dtype=torch.bfloat16),
             ctx_ids=S.pack_txt(torch.zeros(1, 10, 24))[1])
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.equal(dit.eval()(**x), loaded(**x))
    sd = {k: v for k, v in dit.state_dict().items() if "norm" not in k}
    bad = tmp_path / "bad.safetensors"
    save_file({k: v.contiguous() for k, v in sd.items()}, str(bad))
    with pytest.raises(ValueError, match="mismatch"):
        load_klein_dit(bad, device="cpu", params=TINY)


# ---- the run builder ------------------------------------------------------------------------------------------
def test_train_kwargs_map_timesteps_area_and_preview(desc, tmp_path):
    tp = presets.TrainingPresets(desc)
    vals, _ = presets.apply(tp.load("✨ Style (late timesteps)"), P.defaults(), desc)
    vals.update(LORA_NAME="k9", SAMPLE_STEPS=0)
    models = {"klein_dit": "dit.safetensors", "klein_vae": "ae.safetensors", "klein_text_encoder": "te.safetensors"}
    kw, prompts = pipeline.train_kwargs(desc, vals, tmp_path / "run", models)
    assert kw["family"] == "klein9b" and kw["dit_path"] == "dit.safetensors" and kw["te_path"] == "te.safetensors"
    assert (kw["network_dim"], kw["network_alpha"], kw["precision"], kw["blocks_to_swap"]) == (4, 4.0, "auto", -1)
    assert (kw["min_timestep"], kw["max_timestep"]) == (0.0, 0.4) and kw["adaptive_lr"] is True
    assert (kw["adaptive_lr_min"], kw["adaptive_lr_max"]) == (1e-5, 4e-4) and kw["network_type"] == "lora"
    assert kw["optimizer_type"] == "adamw8bit" and kw["ema_decay"] == 0.0 and "speed_lora" not in kw
    assert kw["driver_options"] == {"compile_blocks": "Auto", "target_layers": "Style", "training_blocks": "", "timestep_sampling": "flux2_shift",
                                    "discrete_flow_shift": 3.0, "sigmoid_scale": 1.0, "logit_mean": 0.0,
                                    "logit_std": 1.0, "preserve_distribution": False,
                                    "attention_mechanism": "sdpa"}
    assert prompts and kw["sample_cfg_scale"] == 4.5 and "sample_steps" not in kw      # 0 = the family's 40
    d = desc.load_driver()
    d.configure(**kw["driver_options"])
    assert d.trainable_blocks() == {f"double_{i}" for i in range(8)} | {"single_0", "single_1"}
    vals.update(TARGET_LAYERS="Custom", TRAINING_BLOCKS="single_4", TIMESTEP_SAMPLING="qinglong_flux")
    kw, _ = pipeline.train_kwargs(desc, vals, tmp_path / "run", models)
    d = desc.load_driver()
    d.configure(**kw["driver_options"])
    assert d.trainable_blocks() == {"single_4"} and d.timestep_sampling == "qinglong_flux"


def test_params_are_family_only(desc):
    other = registry.get("qwen_image21")      # a family whose only option is Compile Blocks
    for key in ("TARGET_LAYERS", "TIMESTEP_SAMPLING", "PRESERVE_DISTRIBUTION"):
        assert P.family_shows(P.BY_KEY[key], desc) and not P.family_shows(P.BY_KEY[key], other)
    assert pipeline.driver_options(other, P.defaults()) == {"compile_blocks": "Auto"}
    assert KleinDriver.supports_batching is True


# ---- Attention Mechanism (Fizgig lora_trainer_gui.py:10778-10788, trainer.py:1954-1963) -------------------------
def test_attention_mechanism_param_and_presets(desc):
    p = P.BY_KEY["ATTENTION_MECHANISM"]
    assert (p.kind, p.default, p.options, p.strict, p.family_only) == (P.CHOICE, "sdpa", ("sdpa", "flash3"), True, "option")
    assert "flash-attn" in p.tip and "Hopper or Blackwell" in p.tip
    assert P.DRIVER_OPTIONS["ATTENTION_MECHANISM"] == "attention_mechanism" and "ATTENTION_MECHANISM" in desc.family_options
    assert not P.family_shows(p, registry.get("krea2"))
    new, rep = presets.apply({"ATTENTION_MECHANISM": "flash3"}, P.defaults(), desc)
    assert new["ATTENTION_MECHANISM"] == "flash3" and rep.refused == [] and "ATTENTION_MECHANISM" not in rep.ignored
    new, rep = presets.apply({"ATTENTION_MECHANISM": "xformers"}, P.defaults(), desc)      # a readonly combobox's rule
    assert rep.refused and new["ATTENTION_MECHANISM"] == "sdpa"


def test_attention_reaches_driver_and_model(desc, tmp_path):
    vals = dict(P.defaults(), LORA_NAME="k9")
    models = {"klein_dit": "d", "klein_vae": "v", "klein_text_encoder": "t"}
    kw, _ = pipeline.train_kwargs(desc, vals, tmp_path / "r", models)
    assert kw["driver_options"]["attention_mechanism"] == "sdpa"
    vals["ATTENTION_MECHANISM"] = "flash3"
    kw, _ = pipeline.train_kwargs(desc, vals, tmp_path / "r", models)
    assert kw["driver_options"]["attention_mechanism"] == "flash3"
    d = desc.load_driver()
    d.configure(**kw["driver_options"])
    assert d.attention_mechanism == "flash3"
    with pytest.raises(ValueError, match="attention mechanism"):
        desc.load_driver().configure(attention_mechanism="xformers")


def test_sdpa_default_and_flash3_selection_reach_every_block(driver, tmp_path):
    from safetensors.torch import save_file
    dit = _dit()
    assert dit.attn_mode == "torch" and {b.attn_mode for b in list(dit.double_blocks) + list(dit.single_blocks)} == {"torch"}
    path = tmp_path / "k.safetensors"
    save_file({k: v.contiguous() for k, v in dit.state_dict().items()}, str(path))
    import training.families.klein.model as M
    orig = M.load_klein_dit
    M.load_klein_dit = lambda p, device="cpu": orig(p, device=device, params=TINY)
    try:
        assert driver.load_dit(str(path), "cpu").attn_mode == "torch"          # default option = sdpa
        driver.configure(attention_mechanism="flash3")
        loaded = driver.load_dit(str(path), "cpu")
    finally:
        M.load_klein_dit = orig
    assert loaded.attn_mode == "flash3"
    assert {b.attn_mode for b in list(loaded.double_blocks) + list(loaded.single_blocks)} == {"flash3"}


def test_flash3_fails_like_fizgig_with_no_flash_attn(driver):
    """Fizgig's dispatcher (modules/attention.py:245) has no flash3 branch: the first forward raises, whether or not
    flash-attn is installed. flash-attn is not installed here either way."""
    import importlib.util
    assert importlib.util.find_spec("flash_attn") is None
    dit = _dit()
    dit.set_attn_mode("flash3")
    with pytest.raises(ValueError, match="Unsupported attention mode: flash3"):
        driver.training_loss(dit, torch.randn(1, 16, 6, 4), _cond(1), torch.Generator().manual_seed(1))
    dit.set_attn_mode("torch")
    loss, _ = driver.training_loss(dit, torch.randn(1, 16, 6, 4), _cond(1), torch.Generator().manual_seed(1))
    assert torch.isfinite(loss)


def test_auto_plan_follows_fizgig_7(desc, monkeypatch, tmp_path):
    """Fizgig 7.0.1 families/klein.py: INT8 if it fits, else NF4, never an Auto block swap; uncompiled, an fp8 Base
    file trains as it is (TagScribeR's fp8 choice)."""
    import torch
    from safetensors.torch import save_file
    from training import quant
    from training.families.klein.driver import is_fp8_file
    monkeypatch.setattr(quant, "available", lambda p, device=None: (True, ""))
    drv = desc.load_driver()
    plan = lambda free, mp=0.25: quant.plan(desc, drv, "auto", -1, free_gb=free, megapixels=mp)[:2]  # noqa: E731
    assert plan(16.0) == ("int8", 0)                     # 12.6 + 1.5
    assert plan(12.0) == ("nf4", 0)
    assert plan(18.0, mp=1.0) == ("nf4", 0)              # 17.1 + 1.5 > 18
    assert plan(5.0) == ("nf4", 0)                       # nothing fits: never a swap (no measured saving)
    assert quant.plan(desc, drv, "fp8", -1, free_gb=10.0, megapixels=0.25)[1] > 0   # a manual fp8 still swaps
    fp8, bf = str(tmp_path / "fp8.safetensors"), str(tmp_path / "bf16.safetensors")
    save_file({"a.weight": torch.zeros(2, 2, dtype=torch.float8_e4m3fn), "b.weight": torch.zeros(2, 2)}, fp8)
    save_file({"a.weight": torch.zeros(2, 2, dtype=torch.bfloat16)}, bf)
    assert is_fp8_file(fp8) and not is_fp8_file(bf)
    assert drv.auto_uncompiled_precision(fp8, "int8") == "fp8"
    assert drv.auto_uncompiled_precision(bf, "int8") is None and drv.auto_uncompiled_precision(fp8, "nf4") is None


def test_fizgig_as_the_file_precision_imports(desc):
    new, rep = presets.apply({"FAMILY_PRECISION": "As the file (bf16 or fp8)"}, P.defaults(), desc)
    assert new["FAMILY_PRECISION"] == P.PRECISION_LABELS["fp8"] and not rep.refused and rep.notes


def test_diffusers_flux_lora_fuses_qkv_into_klein(driver, tmp_path):
    """Fizgig 7.0.1 KleinDriver.convert_lora_state_dict: a diffusers-format Flux LoRA (split to_q / to_k / to_v, single
    blocks' proj_mlp) attaches to Klein's fused qkv / linear1 and adds exactly the split adapters' delta."""
    from safetensors.torch import save_file
    dit = _dit()
    qkv = dit.double_blocks[0].img_attn.qkv
    lin1 = dit.single_blocks[0].linear1
    h = qkv.in_features
    torch.manual_seed(3)
    sd, want_qkv, want_l1 = {}, [], []
    for name in ("to_q", "to_k", "to_v"):
        a, b = torch.randn(2, h), torch.randn(h, 2)
        sd[f"transformer.transformer_blocks.0.attn.{name}.lora_A.weight"] = a
        sd[f"transformer.transformer_blocks.0.attn.{name}.lora_B.weight"] = b
        want_qkv.append(b @ a)
    mlp_out = lin1.out_features - 3 * h
    for name, out in (("attn.to_q", h), ("attn.to_k", h), ("attn.to_v", h), ("proj_mlp", mlp_out)):
        a, b = torch.randn(2, h), torch.randn(out, 2)
        sd[f"transformer.single_transformer_blocks.0.{name}.lora_A.weight"] = a
        sd[f"transformer.single_transformer_blocks.0.{name}.lora_B.weight"] = b
        want_l1.append(b @ a)
    save_file(sd, str(tmp_path / "diffusers.safetensors"))
    net = FamilyLoRA(dit, driver)
    assert net.add_file(str(tmp_path / "diffusers.safetensors"), "context") == 2
    x = torch.randn(1, h)
    for w, want in ((dit.double_blocks[0].img_attn.qkv, torch.cat(want_qkv)),
                    (dit.single_blocks[0].linear1, torch.cat(want_l1))):
        got = w(x.to(torch.bfloat16)).float() - w.base(x.to(torch.bfloat16)).float()
        assert torch.allclose(got, x @ want.T, atol=0.05 * float((x @ want.T).abs().max()))


def test_klein_compile_rules(desc, monkeypatch):
    """Fizgig 7.0.1 families/klein.py: payback 200 steps INT8 / 400 NF4, never fp8 by Auto; the measured compiled peaks
    pick the boundary; an fp8 base compiles inside; a refused list keeps the blocks' own checkpointing."""
    from training import compile as C
    from training import quant
    d = desc.load_driver()
    monkeypatch.setattr(C, "compile_blocker", lambda swap: None)
    monkeypatch.setattr(quant, "free_vram_gb", lambda: 16.0)
    assert d.compile_plan("auto", 199, "int8", 0)[0] is False
    assert d.compile_plan("auto", 200, "int8", 0, mp=0.25)[0] == "outside"      # 11.8 fits, inside's 20.7 not needed
    assert d.compile_plan("auto", 400, "nf4", 0, mp=0.25)[0] == "inside"        # NF4's only measured boundary
    assert d.compile_plan("auto", 10 ** 5, "fp8", 0)[0] is False
    monkeypatch.setattr(quant, "free_vram_gb", lambda: 12.0)
    assert d.compile_plan("auto", 10 ** 5, "int8", 0, mp=1.0)[0] is False      # 15.7 GB at 1 MP does not fit
    dit = _dit()
    dit.enable_gradient_checkpointing(True)
    blocks = list(dit.double_blocks)
    monkeypatch.setattr(C, "_triton_importable", lambda: False)
    assert d.compile_blocks(dit, "outside", 0, "int8") == 0
    assert list(dit.double_blocks) == blocks and all(b.gradient_checkpointing for b in blocks)


def test_compiled_klein_blocks_equal_the_eager_ones(driver):
    """The shared wrapper around both block lists (Klein's double blocks return two tensors) traces with dynamo's
    eager backend and gives the eager loss and gradients, with the blocks' own checkpointing handed to the wrapper."""
    from training.compile import CheckpointedBlock
    dit = _dit()
    dit.enable_gradient_checkpointing(True)
    net = FamilyLoRA(dit, driver)
    net.add_trainable(4, 4)
    for p in net.parameters():
        p.data.normal_(0, 0.05)
    lat, cond = torch.randn(1, 16, 6, 4), _cond(1)

    def run():
        dit.train()
        for p in net.parameters():
            p.grad = None
        loss, _ = driver.training_loss(dit, lat, cond, torch.Generator().manual_seed(0))
        loss.backward()
        return loss.item(), [p.grad.clone() for p in net.parameters()]
    eager_loss, eager_grads = run()
    for blocks in (dit.double_blocks, dit.single_blocks):
        for i, b in enumerate(list(blocks)):
            b.gradient_checkpointing = False
            blocks[i] = CheckpointedBlock(torch.compile(b, fullgraph=False, backend="eager"), True)
    loss, grads = run()
    assert loss == pytest.approx(eager_loss, rel=1e-4)
    assert all(torch.allclose(a, b, atol=1e-3) for a, b in zip(grads, eager_grads))


def test_edit_references_ride_after_the_image_tokens(desc, driver, tmp_path):
    """Fizgig 7.0.1: Klein trains Edit LoRAs - reference latents packed after the image tokens at time offsets 10, 20,
    ...; the loss and previews read only the image tokens; the text encoder never sees the references."""
    from training.families.klein import sampling as Ks
    tok, ids = Ks.pack_refs([torch.zeros(1, 16, 2, 3), torch.zeros(16, 4, 2)])
    assert tok.shape == (1, 2 * 3 + 4 * 2, 16) and ids[0, 0, 0] == 10 and ids[0, -1, 0] == 20
    assert Ks.pack_refs(None) == (None, None)
    assert driver.supports_references and desc.edit_training
    dit = _dit()
    net = FamilyLoRA(dit, driver)
    net.add_trainable(4, 4)
    lat, cond, ref = torch.randn(1, 16, 6, 4), _cond(1), torch.randn(1, 16, 6, 4)
    plain = driver.training_loss(dit, lat, cond, torch.Generator().manual_seed(2))[0]
    loss, _ = driver.training_loss(dit, lat, cond, torch.Generator().manual_seed(2), refs=[ref])
    assert torch.isfinite(loss) and loss.item() != plain.item()           # the reference reaches the DiT
    loss.backward()
    assert all(p.grad is not None for p in net.parameters())
    with torch.no_grad():
        out = driver.generate(dit, {"text_embed": _cond(1)["text_embed"][0]}, 64, 96, steps=2, seed=1, refs=[ref])
    assert out.shape == (1, 16, 6, 4)
    vals = {**P.defaults(), "FAMILY_EDIT": True, "FAMILY_EDIT_DIR": str(tmp_path)}
    assert P.family_shows(P.BY_KEY["FAMILY_EDIT"], desc)
    ds = pipeline.dataset_config(desc, vals, str(tmp_path), None)
    assert ds["datasets"][0]["control_directory"] == str(tmp_path.resolve())


def test_distilled_previews_settings_and_schedule(desc, tmp_path):
    """Fizgig 7.0.1: "Use Distilled model for samples" (on by default) sends the Distilled file as the preview
    checkpoint, with the RAM-cache mode and INT8; its recipe is 4 steps, no CFG, ComfyUI's simple schedule at 2.02."""
    from training.families.klein import sampling as Ks
    ck_file, ck = desc.preview_checkpoint()
    assert ck_file.role == "preview_dit" and not ck_file.required and desc.train_preview_checkpoint
    assert (ck.steps, ck.cfg, dict(ck.options)["schedule"], dict(ck.options)["shift"]) == (4, 1.0, "simple", 2.02)
    assert P.BY_KEY["SAMPLE_USE_DISTILLED"].default is True
    for key in ("SAMPLE_USE_DISTILLED", "CACHE_SAMPLE_MODEL", "PREVIEW_INT8"):
        assert P.family_shows(P.BY_KEY[key], desc) and not P.family_shows(P.BY_KEY[key], registry.get("krea2"))
    sched = Ks.get_simple_euler_schedule(4, 2.02)
    assert len(sched) == 5 and sched[-1] == 0.0 and sched[0] > sched[1] > sched[2] > sched[3] > 0
    distilled = tmp_path / "distilled.safetensors"
    distilled.write_bytes(b"x")
    vals = {**P.defaults(), "SAMPLE_ENABLED": True, "SAMPLE_EVERY_N_EPOCHS": 1, "SAMPLE_PROMPT": "a cat",
            "CACHE_SAMPLE_MODEL": "off", "PREVIEW_INT8": True}
    kw, _ = pipeline.train_kwargs(desc, vals, tmp_path / "run", {"klein_distilled_dit": str(distilled)})
    assert (kw["preview_checkpoint"], kw["preview_checkpoint_cache"], kw["preview_int8"]) == (str(distilled), "off",
                                                                                              True)
    kw, _ = pipeline.train_kwargs(desc, {**vals, "SAMPLE_USE_DISTILLED": False}, tmp_path / "run",
                                  {"klein_distilled_dit": str(distilled)})
    assert "preview_checkpoint" not in kw
    kw, _ = pipeline.train_kwargs(desc, vals, tmp_path / "run", {})
    assert "preview_checkpoint" not in kw                                # no file: previews on the training model


def test_preview_handoff_parks_and_restores_the_training_model(driver, monkeypatch):
    """park_for_preview swaps 6 double + 22 single blocks of the training model (per type), unpark restores the run's
    own layout; the Distilled's swap by card follows Fizgig's thresholds."""
    monkeypatch.setenv("TAGSCRIBER_SIM_VRAM_GB", "24")
    assert driver._preview_swap() == (0, None, None)
    monkeypatch.setenv("TAGSCRIBER_SIM_VRAM_GB", "16")
    assert driver._preview_swap() == (16, None, None)
    monkeypatch.setenv("TAGSCRIBER_SIM_VRAM_GB", "12")
    assert driver._preview_swap() == (28, 6, 22)
    calls = []

    class _Dit:
        blocks_to_swap, num_double_blocks, num_single_blocks, _nf4_quantized = 0, 8, 24, False

        def enable_block_swap(self, n, device, backward, double_blocks_to_swap=None, single_blocks_to_swap=None):
            calls.append(("swap", n, double_blocks_to_swap, single_blocks_to_swap))

        def __getattr__(self, name):
            return lambda *a, **k: calls.append((name,))
    dit = _Dit()
    token = driver.park_for_preview(dit, "cpu")
    assert token == 0 and calls[0] == ("swap", 28, 6, 22)
    monkeypatch.setattr("training.quant.move", lambda m, d: calls.append(("move", str(d))))
    driver.unpark_after_preview(dit, "cpu", token)
    assert ("disable_block_swap",) in calls and ("move", "cpu") in calls
