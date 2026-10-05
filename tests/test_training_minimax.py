"""MiniMax H3 family: description, presets, the DiT / driver objective on a tiny random config, the timestep and shift
mapping, the x0 - noise target, the Fast-mode block window, the kohya LoRA keys, the ConvRot int8 Linear and loader, the
nvfp4 dequant and text encoder plumbing, the VAE on tiny configs, generate and the pipeline mapping. CPU only, no weights."""
import json
import re

import pytest

torch = pytest.importorskip("torch")

from training import params as P  # noqa: E402
from training import pipeline, presets, registry  # noqa: E402
from training.families.minimax_h3 import sampling as S  # noqa: E402
from training.families.minimax_h3.description import MINIMAX_H3  # noqa: E402
from training.families.minimax_h3.driver import MiniMaxH3Driver  # noqa: E402
from training.families.minimax_h3.model import (ConvRotInt8Linear, MiniMaxH3Config, MiniMaxH3DiT,  # noqa: E402
                                                load_minimax_h3_dit)
from training.families.minimax_h3.weights import rotate  # noqa: E402
from training.lora import FamilyLoRA  # noqa: E402

registry.register(MINIMAX_H3)
OFF = "Off - hand-pick the blocks below"
TINY = dict(hidden_size=64, num_layers=4, token_refiner_num_layers=1, num_attention_heads=2, attention_head_dim=96,
            ffn_hidden_size=32, text_dim=24, timestep_input_dim=16, time_embed_hidden_size=32, time_embed_dim=16)


@pytest.fixture(scope="module")
def desc():
    return registry.get("minimax_h3")


@pytest.fixture
def driver(desc):
    d = MiniMaxH3Driver()
    d.description = desc
    d.compute_dtype = torch.float32
    d.configure(likeness_mode=OFF, blocks="all")        # the tiny model has 4 blocks: train them all
    return d


def _dit(pruned=False, seed=0):
    torch.manual_seed(seed)
    cfg = MiniMaxH3Config(**{**TINY, **({"adaln_t_table_size": 9, "time_embed_dim": 8} if pruned else {})})
    dit = MiniMaxH3DiT(cfg)
    if pruned:
        dit.adaln_t_table.copy_(torch.randn_like(dit.adaln_t_table))
    return dit.requires_grad_(False)


def _cond(L=7):
    return {"hidden_states": torch.randn(1, L, 24)}


def test_description_presets_and_registry(desc):
    assert desc.validate() == []
    assert desc.arch_id == "minimaxh3" and "_" not in desc.arch_id
    assert (desc.latent_channels, desc.spatial_factor, desc.bucket_step, desc.n_blocks) == (24, 16, 32, 50)
    assert desc.lora.kohya and desc.training_adapter == "minimax_circlestone_adapter"
    names = [n for n, _ in desc.presets]
    assert names == ["✨ MiniMax H3 Fast (LoRA 8, 50 epochs)", "✨ MiniMax H3 (rank 16, 60 epochs)",
                     "✨ MiniMax H3 Style (LoRA 8)"]
    fast, base, style = (v for _, v in desc.presets)
    assert (fast["NETWORK_DIM"], fast["NETWORK_ALPHA"], fast["MAX_TRAIN_EPOCHS"], fast["LEARNING_RATE"]) == (8, 8, 50, 1e-6)
    assert fast["OPTIMIZER_TYPE"] == "automagic3" and fast["MINIMAX_LOWNOISE_PCT"] == "60"
    assert fast["MINIMAX_CAPTION_DROPOUT"] == "0.05 (default)" and fast["MINIMAX_EMA"] == "0.98 (recommended)"
    assert base["NETWORK_DIM"] == 16 and base["MAX_TRAIN_EPOCHS"] == 60
    assert fast["MINIMAX_CLIP_STILL"] is True and style["MINIMAX_CLIP_STILL"] is False
    assert registry.by_arch_id("minimaxh3") is desc


def test_preset_keys_resolve_without_refusals(desc):
    for name, values in desc.presets:
        migrated, notes, ignored = presets.migrate_legacy(values)
        assert [k for k in migrated if k not in P.BY_KEY] == [], name
        assert all(k.startswith("MINIMAX_") for k in ignored)
        new, rep = presets.apply(values, P.defaults(), desc)
        assert rep.refused == [], (name, rep.refused)
        assert new["FAMILY_EMA"].startswith("0.98") and new["CAPTION_DROPOUT"] == 0.05
        assert new["FAMILY_TRAINING_ADAPTER"] is True and new["OPTIMIZER_TYPE"] == "automagic3"
        assert new["FAMILY_PRECISION"] == P.PRECISION_LABELS["auto"]
        assert float(new["MINIMAX_LOWNOISE_PCT"]) == 60.0 and new["MINIMAX_LIKENESS_MODE"] == "Default"
    new, rep = presets.apply({"MINIMAX_ADAPTER": "Off", "MINIMAX_BASE_QUANT": "int8 · most accurate"}, P.defaults(), desc)
    assert new["FAMILY_TRAINING_ADAPTER"] is False and new["FAMILY_PRECISION"] == P.PRECISION_LABELS["int8"]


def test_shift_mapping_and_sigma_distribution_match_fizgig():
    assert S.lownoise_to_shift(60) == pytest.approx(2 / 3)
    assert S.lownoise_to_shift(50) == pytest.approx(1.0) and S.lownoise_to_shift(100) is None
    assert S.lownoise_to_shift("7.7%") == pytest.approx(12.0, rel=0.01)
    for shift in (2 / 3, 1.0, 12.0):                      # Fizgig trainer.py sample_sigmas, numeric shift
        g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
        base = torch.rand(64, generator=g1)
        want = (shift * base) / (1.0 + (shift - 1.0) * base)
        assert torch.equal(S.sample_sigmas(64, shift, g2), want)
    sig = S.sample_sigmas(200000, S.lownoise_to_shift(60), torch.Generator().manual_seed(0))
    assert float((sig < 0.5).float().mean()) == pytest.approx(0.60, abs=0.01)   # the dial means what it says


def test_target_is_x0_minus_noise_with_t_one_minus_sigma(driver):
    """A fake DiT that returns the CORRECT head output (x0 - noise) gives zero loss; the flipped sign does not."""
    x0 = torch.randn(1, 24, 4, 6)
    seen = {}

    class Fake:
        patch_size = (1, 2, 2)

        class config:
            audio_latents_dim = 32

        def __call__(self, noised, t, text, audio_noise=None):
            g = torch.Generator().manual_seed(11)
            noise = torch.randn(1, 24, 1, 4, 6, generator=g)
            sigma = S.sample_sigmas(1, driver.shift, g)
            seen.update(t=t.item(), sigma=sigma.item())
            assert torch.allclose(noised, (1 - sigma.item()) * x0[:, :, None] + sigma.item() * noise, atol=1e-6)
            return seen.setdefault("sign", 1.0) * (x0[:, :, None] - noise)
    loss, info = driver.training_loss(Fake(), x0, _cond(), torch.Generator().manual_seed(11))
    assert loss.item() == pytest.approx(0.0, abs=1e-10)
    assert seen["t"] == pytest.approx(1.0 - seen["sigma"], abs=1e-6) and info["t"] == pytest.approx(seen["sigma"])
    seen["sign"] = -1.0
    seen.pop("sign")
    seen["sign"] = -1.0
    flipped = driver.training_loss(Fake(), x0, _cond(), torch.Generator().manual_seed(11))[0]
    assert flipped.item() > 0.1


@pytest.mark.parametrize("pruned", [False, True])
def test_training_loss_forward_backward_grads_on_every_param(driver, pruned):
    dit = _dit(pruned)
    net = FamilyLoRA(dit, driver)
    net.add_trainable(4, 4)
    loss, info = driver.training_loss(dit, torch.randn(1, 24, 4, 6), _cond(), torch.Generator().manual_seed(1))
    assert torch.isfinite(loss) and 0 < info["t"] < 1
    loss.backward()
    ps = net.parameters()
    assert len(ps) == 4 * 4 * 2
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ps)
    ups = [m for m in net.trainable_modules()]
    assert all(m[1].weight.grad.abs().sum() > 0 for m in ups)


def test_odd_latent_grid_is_cropped_to_the_patch(driver):
    loss, _ = driver.training_loss(_dit(), torch.randn(1, 24, 5, 7), _cond(), torch.Generator().manual_seed(2))
    assert torch.isfinite(loss)


def test_block_swap_does_not_change_the_loss(driver):
    args = (torch.randn(1, 24, 4, 6), _cond())

    def loss_of(swap):
        dit = _dit()
        if swap:
            driver.enable_block_swap(dit, swap, torch.device("cpu"), True)
        return driver.training_loss(dit, *args, torch.Generator().manual_seed(3))[0].item()
    assert driver.max_blocks_to_swap(_dit()) == 2
    assert loss_of(2) == pytest.approx(loss_of(0), rel=1e-5)


def test_fast_mode_trains_exactly_blocks_20_to_49(desc):
    d = MiniMaxH3Driver()
    d.description = desc
    assert d.trained_blocks() == list(range(20, 50))                  # Fizgig MINIMAX_LIKENESS_BLOCKS "20-49"
    targets = d.lora_target_names(None)
    assert len(targets) == 120 and targets[0] == "blocks.20.attn.qkv_proj" and targets[-1] == "blocks.49.mlp.fc2"
    assert not any(int(re.search(r"blocks\.(\d+)", t).group(1)) < 20 for t in targets)
    assert len(d.quant_target_names(None)) == 200                     # the frozen base is quantised everywhere
    d.configure(likeness_mode="More Blocks")
    assert d.trained_blocks() == list(range(6, 50))
    d.configure(likeness_mode=OFF, blocks="3-5, 9")
    assert d.trained_blocks() == [3, 4, 5, 9]
    with pytest.raises(ValueError):
        d.configure(likeness_mode=OFF, blocks="3-x")
        d.trained_blocks()
    with pytest.raises(ValueError):
        d.configure(lownoise_pct=0)


def test_lora_keys_are_kohya_and_round_trip(desc, driver, tmp_path):
    from safetensors.torch import load_file
    dit = _dit()
    net = FamilyLoRA(dit, driver)
    net.add_trainable(4, 6)
    sd = net.state_dict()
    pat = re.compile(r"lora_unet_blocks_\d+_(attn_qkv_proj|attn_out_proj|mlp_fc1|mlp_fc2)\.(lora_down\.weight|"
                     r"lora_up\.weight|alpha)$")
    assert len(sd) == 4 * 4 * 3 and all(pat.match(k) for k in sd)
    assert sd["lora_unet_blocks_0_attn_qkv_proj.lora_down.weight"].shape == (4, 64)
    assert float(sd["lora_unet_blocks_3_mlp_fc2.alpha"]) == 6.0
    for p in net.parameters():
        p.data.normal_()
    path = str(tmp_path / "h3.safetensors")
    net.save(path, dtype=torch.float32)
    assert set(load_file(path)) == set(sd)
    saved = [p.detach().clone() for p in net.parameters()]
    for p in net.parameters():
        p.data.zero_()
    assert net.load_trainable(path) == 16
    assert all(torch.equal(a, b) for a, b in zip(saved, net.parameters()))
    assert FamilyLoRA(_dit(), driver).add_file(path, "frozen") == 16     # the training adapter's route


def test_generate_shape_noise_and_cfg(desc, driver):
    dit = _dit(pruned=True)
    cond = {"hidden_states": torch.randn(7, 24)}
    a = driver.generate(dit, cond, 64, 96, steps=3, seed=1)
    assert a.shape == (1, 24, 1, 6, 4) and torch.isfinite(a).all()
    assert torch.equal(a, driver.generate(dit, cond, 64, 96, steps=3, seed=1))
    assert not torch.equal(a, driver.generate(dit, cond, 64, 96, steps=3, seed=2))
    n = driver.initial_noise(1, 64, 96)
    assert n.shape == (1, 24, 1, 6, 4) and torch.equal(a, driver.generate(dit, cond, 64, 96, steps=3, seed=1, noise=n))
    c = driver.generate(dit, cond, 64, 96, steps=2, seed=1, cfg=3.0, neg_cond={"hidden_states": torch.randn(3, 24)})
    assert c.shape == a.shape
    sig = S.sample_schedule(20)
    assert len(sig) == 21 and sig[-1] == 0.0 and sig[0] == pytest.approx(1.0, abs=1e-3) \
        and sig[-2] == pytest.approx(0.3871, abs=1e-3)                  # Fizgig: "0.3871 for comfy at 20 steps"


def test_convrot_linear_matches_dense_and_quantiser_leaves_it_alone(driver):
    from training import quant
    torch.manual_seed(0)
    w = torch.randn(32, 64)
    wr = rotate(w, 16)
    s = wr.abs().amax(dim=1, keepdim=True) / 127.0
    lin = ConvRotInt8Linear(64, 32, rot=16)
    lin.weight = torch.nn.Parameter(torch.round(wr / s).to(torch.int8), requires_grad=False)
    lin._convrot_wscale = s
    x = torch.randn(5, 64, requires_grad=True)
    y = lin(x)
    assert (y - x @ w.T).abs().max() < 0.15 and torch.allclose(lin.dense_weight(dtype=torch.float32), w, atol=0.05)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.ones(5, 32) @ lin.dense_weight(dtype=torch.float32), atol=1e-3)
    before = lin.weight.data.clone()

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Module()])
            self.blocks[0].attn = torch.nn.Module()
            self.blocks[0].attn.qkv_proj = lin
    d = MiniMaxH3Driver()
    d.description = driver.description
    d.quant_target_names = lambda dit: ["blocks.0.attn.qkv_proj"]
    quant.quantize(M(), d, "int8", "cpu")
    assert lin.weight.dtype == torch.int8 and torch.equal(lin.weight.data, before)


def _write_pruned(path, dit):
    from safetensors.torch import save_file
    sd = {}
    for name, p in dit.state_dict().items():
        if re.fullmatch(r"blocks\.\d+\.(attn\.qkv_proj|attn\.out_proj|mlp\.fc1|mlp\.fc2)\.weight", name):
            wr = rotate(p.float(), 16)
            s = wr.abs().amax(dim=1, keepdim=True) / 127.0
            sd[name] = torch.round(wr / s).to(torch.int8)
            sd[name[:-len(".weight")] + ".weight_scale"] = s
            blob = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 16}).encode()
            sd[name[:-len(".weight")] + ".comfy_quant"] = torch.tensor(list(blob), dtype=torch.uint8)
        else:
            sd[name] = p.detach().clone()
    save_file(sd, str(path))


def test_loader_keeps_int8_convrot_and_matches_the_decoded_model(tmp_path):
    src = _dit(pruned=True)
    f = tmp_path / "pruned.safetensors"
    _write_pruned(f, src)
    loaded = load_minimax_h3_dit(str(f), dtype=torch.float32, config=src.config)
    assert sum(isinstance(m, ConvRotInt8Linear) for m in loaded.modules()) == 16
    assert loaded.blocks[0].attn.qkv_proj.weight.dtype == torch.int8 and loaded.adaln_fp32
    ref = _dit(pruned=True)
    ref.load_state_dict(src.state_dict())
    for n, m in loaded.named_modules():
        if isinstance(m, ConvRotInt8Linear):
            ref.get_submodule(n).weight.data = m.dense_weight(dtype=torch.float32)
    z, t, txt = torch.randn(1, 24, 1, 4, 6), torch.tensor([0.4]), torch.randn(1, 5, 24)
    an = torch.randn(4, 32)
    with torch.no_grad():
        assert torch.allclose(loaded(z, t, txt, audio_noise=an), ref(z, t, txt, audio_noise=an), atol=1e-3)
    bad = tmp_path / "full.safetensors"
    from safetensors.torch import save_file
    save_file({"x": torch.zeros(1)}, str(bad))
    with pytest.raises(ValueError, match="PRUNED"):
        load_minimax_h3_dit(str(bad))


def test_nvfp4_dequant_and_blocked_scale_layout():
    from training.families.minimax_h3.embedder import _E2M1_MAG, _from_blocked, _nvfp4_dequant
    rows, cols = 130, 8                                     # scales [rows, cols/16... ] use cols=8 blocks of 16
    sc = torch.rand(rows, cols).to(torch.float8_e4m3fn)
    nrb, ncb = -(-rows // 128), -(-cols // 4)               # ComfyUI float.py to_blocked
    pad = torch.zeros(nrb * 128, ncb * 4, dtype=torch.float32)
    pad[:rows, :cols] = sc.float()
    blocked = pad.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3).reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    assert torch.equal(_from_blocked(blocked, rows, cols), sc.float())
    # known codes with unit scales: value = signed e2m1 table
    out, inp = 128, 32
    codes = torch.randint(0, 16, (out, inp), dtype=torch.uint8)
    packed = (codes[:, 0::2] << 4) | codes[:, 1::2]
    ones = torch.ones(out, inp // 16).to(torch.float8_e4m3fn)
    pad = torch.zeros(128, 4)
    pad[:, :2] = ones.float()
    blk = pad.view(1, 128, 1, 4).permute(0, 2, 1, 3).reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    w = _nvfp4_dequant(packed, blk.to(torch.float8_e4m3fn), torch.tensor(2.0))
    tbl = torch.cat([_E2M1_MAG, -_E2M1_MAG])
    assert torch.equal(w.float(), (tbl[codes.long()] * 2.0))


class _Tok:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        ids = [1 + (ord(c) % 20) for c in text]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


@pytest.mark.parametrize("stream", [False, True])
def test_text_encoder_batching_is_exact_and_streaming_matches(stream):
    pytest.importorskip("transformers")
    from training.families.minimax_h3.embedder import H3TextEncoder, build_qwen3_te
    torch.manual_seed(0)
    model = build_qwen3_te(dict(hidden_size=32, num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
                                head_dim=16, intermediate_size=48, vocab_size=64)).eval()
    te = H3TextEncoder(model, _Tok(), device="cpu", compute_dtype=torch.float32, stream=stream)
    caps = ["abc", "hello world", "", "abc"]
    batch = te.encode_batch(caps, batch_size=2)
    te2 = H3TextEncoder(model, _Tok(), device="cpu", compute_dtype=torch.float32)
    for c, b in zip(caps, batch):
        single = te2.encode_batch([c], batch_size=1)[0]
        assert b.shape == single.shape and torch.allclose(b, single, atol=1e-5)
    assert batch[2].shape[0] == 1                           # empty caption = one pad token


def test_vae_encoder_decoder_tiny_and_driver_encode_images(driver):
    from training.families.minimax_h3.vae import MiniMaxH3VideoVAEEncoder, ViT3DDecoder
    import numpy as np
    enc = MiniMaxH3VideoVAEEncoder(ch=32, ch_mult=(1, 1, 1, 1, 1, 1), num_res_blocks=1).eval()
    assert enc.vae_ratio == 16

    class V:
        device = torch.device("cpu")

        def encoder(self):
            return enc
    imgs = [np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8) for _ in range(2)]
    lat = driver.encode_images(V(), imgs)
    assert len(lat) == 2 and lat[0].shape == (24, 4, 6) and lat[0].dtype == torch.float32
    dec = ViT3DDecoder(num_layers=1, heads=2, dim_head=64).eval()
    with torch.no_grad():
        assert dec(torch.randn(1, 24, 5, 2, 2)).shape == (1, 3, 20, 32, 32)


def test_pipeline_driver_options_and_train_kwargs(desc, tmp_path):
    tp = presets.TrainingPresets(desc)
    vals, _ = presets.apply(tp.load(tp.default_name), P.defaults(), desc)
    vals.update(LORA_NAME="h3")
    assert pipeline.driver_options(desc, vals) == {"lownoise_pct": 60.0, "highnoise_lr_pct": 100.0,
                                                   "likeness_mode": "Default", "blocks": "all"}
    models = {"minimax_dit": "d.safetensors", "minimax_vae": "v.safetensors", "minimax_text_encoder": "t.safetensors",
              "minimax_circlestone_adapter": str(tmp_path / "ad.safetensors")}
    (tmp_path / "ad.safetensors").write_bytes(b"x")
    kw, _ = pipeline.train_kwargs(desc, vals, tmp_path / "run", models)
    assert kw["family"] == "minimax_h3" and kw["optimizer_type"] == "automagic3" and kw["ema_decay"] == 0.98
    assert (kw["network_dim"], kw["max_train_epochs"], kw["precision"]) == (8, 50, "auto")
    assert kw["training_adapter"].endswith("ad.safetensors")
    assert kw["driver_options"]["lownoise_pct"] == 60.0
    vals["MINIMAX_LOWNOISE_PCT"] = 25
    assert pipeline.driver_options(desc, vals)["lownoise_pct"] == 25.0
    cfg = pipeline.dataset_config(desc, vals, str(tmp_path), None)
    assert cfg["caption_dropout"] == 0.05 if "caption_dropout" in cfg else True


# ---- Medium to High Noise LR (Fizgig --highnoise_lr_scale) ---------------------------------------------------------
def test_highnoise_lr_scale_is_reported_per_step_and_stamped(desc):
    d = MiniMaxH3Driver()
    d.description = desc
    d.compute_dtype = torch.float32
    d.configure(likeness_mode=OFF, blocks="all")
    dit = _dit()
    _, info = d.training_loss(dit, torch.randn(1, 24, 4, 6), _cond(), torch.Generator().manual_seed(1))
    assert "lr_scale" not in info                                     # 100% = the loop is left alone
    d.configure(highnoise_lr_pct=25)
    assert d.highnoise_lr_scale == 0.25
    seen = set()
    for seed in range(40):
        _, info = d.training_loss(dit, torch.randn(1, 24, 4, 6), _cond(), torch.Generator().manual_seed(seed))
        assert info["lr_scale"] == (0.25 if info["t"] >= 0.5 else 1.0)  # Fizgig: sigma >= 0.5 is the noisy half
        seen.add(info["lr_scale"])
    assert seen == {0.25, 1.0}
    md = d.extra_metadata()
    assert md["ss_highnoise_lr_scale"] == "0.25" and md["ss_timestep_density"] == "shift0.666667"
    for bad in (-1, 101):
        with pytest.raises(ValueError):
            d.configure(highnoise_lr_pct=bad)
    assert P.BY_KEY["MINIMAX_HIGHNOISE_LR_PCT"].default == 100.0 and "MINIMAX_HIGHNOISE_LR_PCT" in desc.family_options


@pytest.mark.parametrize("optimizer,moves", [("adamw", False), ("automagic3", True)])
def test_loop_scales_the_optimizer_lr_not_the_loss(tmp_path, monkeypatch, optimizer, moves):
    """A step multiplier of 0 freezes an AdamW run (the LR is what is scaled) and is ignored by Automagic v3, which
    sets its own rate (Fizgig trainer.py:5266 `and not _automagic`)."""
    from safetensors.torch import load_file
    from tests import tiny_family as T
    from training import cache, train
    tiny = T.register()
    data = T.make_dataset(str(tmp_path / "scratch_dataset"))
    models = T.write_models(str(tmp_path / "models"))
    orig = T.TinyDriver.training_loss

    def loss(self, *a, **k):
        l, info = orig(self, *a, **k)
        return l, {**info, "lr_scale": 0.0}
    monkeypatch.setattr(T.TinyDriver, "training_loss", loss)
    vals = P.defaults()
    vals.update({"LORA_OUTPUT_DIR": str(tmp_path / "runs"), "LORA_NAME": "band", "NETWORK_DIM": 4, "NETWORK_ALPHA": 4,
                 "MAX_TRAIN_EPOCHS": 1, "OPTIMIZER_TYPE": optimizer, "LEARNING_RATE": 1e-3, "FAMILY_EMA": "Off",
                 "DATASET_MEGAPIXELS": "0.01", "SAMPLE_ENABLED": False, "GRADIENT_ACCUMULATION": 2})
    run = pipeline.build_run(tiny, vals, data, models)
    for _label, argv in run.stages:
        if argv[2] == "training.cache":
            cache.main(argv[3:] + ["--device", "cpu"])
        else:
            cfg_path = argv[argv.index("--config") + 1]
            cfg = json.loads(open(cfg_path, encoding="utf-8").read())
            cfg["train"]["device"] = "cpu"
            open(cfg_path, "w", encoding="utf-8").write(json.dumps(cfg))
            train.main(argv[3:])
    sd = load_file(str(run.run_dir / "band.safetensors"))
    up = sum(float(v.abs().sum()) for k, v in sd.items() if "lora_up" in k)
    assert (up > 0) is moves
