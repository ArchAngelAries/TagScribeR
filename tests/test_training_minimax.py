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
OFF = "Off · hand-pick the blocks in Blocks to train"
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
    assert desc.lora.kohya and desc.training_adapter == "" and desc.adapter_choice == "H3_ADAPTER"   # Fizgig 7.0.1
    assert desc.trainable_dtype == "bf16" and desc.ema_short_run and desc.optimizer_weight_decay == 1e-4
    assert desc.optimizers[:3] == ("automagic3", "adamw8bit", "adamw") and desc.optimizer_eps_floor_8bit
    assert (desc.preview_width, desc.preview_height) == (768, 768)
    names = [n for n, _ in desc.presets]
    assert names == ["✨ MiniMax H3 Fast (LoRA 8, 50 epochs)", "✨ MiniMax H3 (rank 16, 60 epochs)",
                     "✨ MiniMax H3 Style (LoRA 8)", "✨ MiniMax H3 Slider (rank 8, 2e-4)"]
    fast, base, style, slider = (v for _, v in desc.presets)
    assert slider["FAMILY_SLIDER"] is True and (slider["OPTIMIZER_TYPE"], slider["LEARNING_RATE"]) == ("adamw8bit", 2e-4)
    assert (fast["NETWORK_DIM"], fast["NETWORK_ALPHA"], fast["MAX_TRAIN_EPOCHS"], fast["LEARNING_RATE"]) == (8, 8, 50, 1e-6)
    assert fast["OPTIMIZER_TYPE"] == "automagic3" and fast["H3_LOWNOISE_PCT"] == "60"
    assert fast["H3_CAPTION_DROPOUT"] == "0.05 (default)" and fast["FAMILY_EMA"] == "0.98 (recommended)"
    assert base["NETWORK_DIM"] == 16 and base["MAX_TRAIN_EPOCHS"] == 60
    assert fast["H3_CLIP_STILL"] == "1" and style["H3_CLIP_STILL"] == ""
    assert registry.by_arch_id("minimaxh3") is desc
    assert P.options_for(P.BY_KEY["FAMILY_EMA"], desc)[-1] == P.EMA_SHORT


def test_preset_keys_resolve_without_refusals(desc):
    for name, values in desc.presets:
        migrated, notes, ignored = presets.migrate_legacy(values)
        assert [k for k in migrated if k not in P.BY_KEY and not k.startswith("FAMILY_SLIDER")] == [], name
        assert set(ignored) <= {"H3_TREAD", "H3_CLIP_STILL", "H3_DISTILL"} and notes == []
        new, rep = presets.apply(values, P.defaults(), desc)
        assert rep.refused == [] and rep.blocked == [], (name, rep.refused)
        assert new["FAMILY_EMA"].startswith("0.98") and new["CAPTION_DROPOUT"] == 0.05
        assert new["H3_ADAPTER"] == P.H3_ADAPTERS[0]
        assert new["OPTIMIZER_TYPE"] == ("adamw8bit" if values["FAMILY_SLIDER"] else "automagic3")
        assert new["FAMILY_PRECISION"] == P.PRECISION_LABELS["auto"] and new["H3_STRUCTURE"] == P.H3_STRUCTURES[0]
        assert float(new["H3_LOWNOISE_PCT"]) == 60.0 and new["H3_LIKENESS_MODE"] == "Default"
        assert float(new["H3_HIGHNOISE_LR_PCT"]) == 100.0 and new["H3_TRAIN_REFINER"] is False
    # Fizgig's older MINIMAX_* keys map onto the H3_* ones (an explicit H3_* key wins)
    old = {"MINIMAX_ADAPTER": "Ostris — best for videos", "MINIMAX_BASE_QUANT": "int8 · most accurate",
           "MINIMAX_LOWNOISE_PCT": "8", "MINIMAX_HIGHNOISE_LR_PCT": "70", "MINIMAX_CAPTION_DROPOUT": "0.10 (strong)",
           "MINIMAX_TRAIN_ADALN": False, "MINIMAX_ADAPTER_RAMP": "0.005 (recommended)", "MINIMAX_EMA": "Off",
           "MINIMAX_LIKENESS_MODE": "More Blocks", "MINIMAX_TRAIN_BASE": "Reference (ref2va)"}
    new, rep = presets.apply(old, P.defaults(), desc)
    assert new["H3_ADAPTER"] == P.H3_ADAPTERS[1] and new["FAMILY_PRECISION"] == P.PRECISION_LABELS["int8"]
    assert new["H3_STRUCTURE"] == P.H3_STRUCTURES[1] and float(new["H3_LOWNOISE_PCT"]) == 8.0
    assert float(new["H3_HIGHNOISE_LR_PCT"]) == 70.0 and new["CAPTION_DROPOUT"] == 0.1
    assert new["H3_ADAPTER_RAMP"] == "0.005 (recommended)" and new["FAMILY_EMA"] == "Off"
    assert new["H3_LIKENESS_MODE"] == "More Blocks" and new["H3_TRAIN_BASE"] == P.H3_BASES[1]
    new, rep = presets.apply({"MINIMAX_LOWNOISE_PCT": "45", "H3_LOWNOISE_PCT": "30"}, P.defaults(), desc)
    assert float(new["H3_LOWNOISE_PCT"]) == 30.0
    new, rep = presets.apply({"MINIMAX_LOWNOISE_PCT": "45"}, P.defaults(), desc)
    assert new["H3_STRUCTURE"] == "Custom" and float(new["H3_LOWNOISE_PCT"]) == 45.0
    new, rep = presets.apply({"FAMILY_PRECISION": "4-bit HQQ · lower error than 4-bit, slower",
                              "H3_DISTILL": "1"}, P.defaults(), desc)
    assert new["FAMILY_PRECISION"] == P.PRECISION_LABELS["nf4"]
    assert any("HQQ" in n for n in rep.notes) and any("H3_DISTILL" in n for n in rep.notes)


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


def test_fizgig_701_options_structure_noise_lr_refiner_ramp_metadata(desc):
    d = MiniMaxH3Driver()
    d.description = desc
    d.configure(structure=P.H3_STRUCTURES[1], lownoise_pct=33)               # a structure owns the share
    assert d.lownoise_pct == 8.0 and d.run_metadata()["ss_timestep_density"] == f"{(1 - 0.08) / 0.08:g}"
    d.configure(structure="Custom", lownoise_pct=33)
    assert d.lownoise_pct == 33.0
    # Medium to High Noise LR: clamped to 0-100 %, a noisy-half step carries the multiplier (Fizgig lr_mult)
    d.configure(highnoise_lr_pct=150)
    assert d.highnoise_lr == 1.0
    d.configure(highnoise_lr_pct="40%")
    assert d.highnoise_lr == pytest.approx(0.4)

    class Fake:
        patch_size = (1, 2, 2)

        class config:
            audio_latents_dim = 32

        def __call__(self, noised, t, text, audio_noise=None):
            return torch.zeros_like(noised)
    seen = set()
    for seed in range(40):
        _l, info = d.training_loss(Fake(), torch.randn(1, 24, 4, 6), _cond(), torch.Generator().manual_seed(seed))
        assert info["lr_mult"] == (pytest.approx(0.4) if info["t"] >= 0.5 else 1.0)
        seen.add(info["lr_mult"] == 1.0)
    assert seen == {True, False}
    # block ids are Fizgig 7.0.1's (h3blk_N, h3_rf_N); the refiner trains only in mode Off over every block
    ids = [b.id for g in d.block_map() for b in g.blocks]
    assert ids[:2] == ["h3blk_0", "h3blk_1"] and ids[-2:] == ["h3_rf_0", "h3_rf_1"] and len(ids) == 52
    d.configure(likeness_mode="Default", train_token_refiner=True)
    assert not d.trains_refiner() and d.run_metadata()["ss_photo_blocks"] == "20-49"
    d.configure(likeness_mode=OFF, blocks="all")
    assert d.trains_refiner() and len(d.lora_target_names(None)) == 208
    md = d.run_metadata()
    assert md["ss_train_token_refiner"] == "1" and md["ss_train_blocks"].endswith("h3blk_49,h3_rf_0,h3_rf_1")
    d.configure(blocks="h3blk_3, 7-8", train_token_refiner=False)
    assert d.trained_blocks() == [3, 7, 8] and d.run_metadata()["ss_train_blocks"] == "h3blk_3,h3blk_7,h3blk_8"
    # the adapter-relative LR ramp: starts at 10% of the LR, follows the adapter's growth, off by default
    assert d.step_policy({}, 1) == (False, 1.0) and d.run_metadata()["ss_adapter_ramp"] == "off"
    d.configure(adapter_ramp="0.005 (recommended)", likeness_mode=OFF, blocks="all")
    dit = _dit()
    net = FamilyLoRA(dit, d)
    net.add_trainable(4, 4)
    assert all(p.dtype == torch.bfloat16 for p in net.parameters())         # H3 trains its LoRA in bf16 (Fizgig)
    d.prepare_training(dit, net, precision="int8", blocks_to_swap=0, total_steps=10, megapixels=0.25, batch_size=1)
    assert d.step_policy({}, 1) == (False, pytest.approx(0.1))
    d.after_optimizer_step()                                                   # the first reading only records
    for p in net.parameters():
        p.data.add_(0.01)
    d.after_optimizer_step()                                                   # a big step: the ramp backs off
    assert d.step_policy({}, 1)[1] == pytest.approx(0.1 * 0.95)
    assert d.run_metadata()["ss_adapter_ramp"] == "0.005" and "[ramp]" in d._ramp.epoch_report()


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
    assert pipeline.driver_options(desc, vals) == {
        "structure": P.H3_STRUCTURES[0], "lownoise_pct": 60.0, "highnoise_lr_pct": 100.0, "likeness_mode": "Default",
        "blocks": "all", "adapter_ramp": "Off", "train_token_refiner": False}
    files = {k: str(tmp_path / f"{k}.safetensors") for k in ("ad", "ost", "ostref", "ref")}
    for f in files.values():
        open(f, "wb").write(b"x")
    models = {"minimax_dit": "d.safetensors", "minimax_vae": "v.safetensors", "minimax_text_encoder": "t.safetensors",
              "minimax_circlestone_adapter": files["ad"], "minimax_training_adapter": files["ost"],
              "minimax_ref_training_adapter": files["ostref"], "minimax_ref_dit": files["ref"]}
    kw, _ = pipeline.train_kwargs(desc, vals, tmp_path / "run", models)
    assert kw["family"] == "minimax_h3" and kw["optimizer_type"] == "automagic3" and kw["ema_decay"] == 0.98
    assert (kw["network_dim"], kw["max_train_epochs"], kw["precision"]) == (8, 50, "auto")
    assert kw["training_adapter"] == files["ad"] and kw["dit_path"] == "d.safetensors"
    assert kw["driver_options"]["lownoise_pct"] == 60.0
    # the adapter choice and the training base pick the files (Fizgig: Ostris per base, Circlestone for both)
    vals.update(H3_ADAPTER=P.H3_ADAPTERS[1])
    assert pipeline.train_kwargs(desc, vals, tmp_path / "run", models)[0]["training_adapter"] == files["ost"]
    vals.update(H3_TRAIN_BASE=P.H3_BASES[1])
    kw, _ = pipeline.train_kwargs(desc, vals, tmp_path / "run", models)
    assert kw["training_adapter"] == files["ostref"] and kw["dit_path"] == files["ref"]
    vals.update(H3_ADAPTER="Off", FAMILY_EMA=P.EMA_SHORT)
    kw, _ = pipeline.train_kwargs(desc, vals, tmp_path / "run", models)
    assert "training_adapter" not in kw and kw["ema_decay"] == "short"
    cfg = pipeline.dataset_config(desc, vals, str(tmp_path), None)
    assert cfg["caption_dropout"] == 0.05 if "caption_dropout" in cfg else True


# ---- Turbo-LoRA previews and frozen AdaLN rows (Fizgig 7.0.1 frozen_file_added / turbo_adaln_patch) ---------------
E_WIDE = 12          # the tiny stand-in for the full model's 2688-wide silu(t_emb) space


def _turbo_file(path, dit, rank=2, dotted=False):
    """A tiny 'Turbo' LoRA: block Linears the base can host, plus AdaLN rows in the full model's (E_WIDE) space."""
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(7)
    sd = {}

    def put(mod, n_in, n_out):
        stem = f"diffusion_model.{mod}" if dotted else "lora_unet_" + mod.replace(".", "_")
        a, b = (".lora_A.weight", ".lora_B.weight") if dotted else (".lora_down.weight", ".lora_up.weight")
        sd[stem + a] = torch.randn(rank, n_in, generator=g) * 0.3
        sd[stem + b] = torch.randn(n_out, rank, generator=g) * 0.3
        if not dotted:
            sd[stem + ".alpha"] = torch.tensor(float(rank))
    for i in range(len(dit.blocks)):
        lin = dit.blocks[i].attn.qkv_proj
        put(f"blocks.{i}.attn.qkv_proj", lin.in_features, lin.out_features)
        put(f"blocks.{i}.adaln_proj.linear", E_WIDE, dit.blocks[i].adaln_proj.linear.out_features)
    put("final_layer.adaln_proj.linear", E_WIDE, dit.final_layer.adaln_proj.linear.out_features)
    save_file(sd, str(path))
    return sd


def test_turbo_grid_asset_is_the_full_models_table():
    from training.families.minimax_h3 import turbo
    grid = turbo.load_h3_egrid()
    assert tuple(grid.shape) == (1025, 2688) and torch.isfinite(grid.float()).all()


@pytest.mark.parametrize("dotted", [False, True])
def test_turbo_adaln_rows_are_injected_and_removed(tmp_path, dotted):
    from training.families.minimax_h3 import turbo
    dit = _dit(pruned=True)
    f = tmp_path / "turbo.safetensors"
    sd = _turbo_file(f, dit, dotted=dotted)
    pairs = turbo.read_adaln_pairs(dit, str(f), 0.75)
    assert len(pairs) == len(dit.blocks) + 1                     # every AdaLN projection, none of the block Linears
    assert torch.equal(turbo.read_adaln_pairs(dit, str(f), 1.5)[0][2], 2 * pairs[0][2])   # read once, rescaled
    egrid = torch.randn(9, E_WIDE, generator=torch.Generator().manual_seed(3))
    ap = dit.blocks[1].adaln_proj
    stem = "diffusion_model.blocks.1.adaln_proj.linear" if dotted else "lora_unet_blocks_1_adaln_proj_linear"
    A = sd[stem + (".lora_A.weight" if dotted else ".lora_down.weight")]
    B = sd[stem + (".lora_B.weight" if dotted else ".lora_up.weight")]
    t_emb = dit.adaln_t_table[[2, 6]].float()                    # exactly on table rows 2 and 6
    before = torch.cat(ap(t_emb), dim=-1)
    assert turbo.adaln_patch(dit, pairs, "cpu", torch.float32, egrid=egrid) == len(pairs)
    after = torch.cat(ap(t_emb), dim=-1)
    want = (0.75 * B @ (A @ egrid[[2, 6]].T)).T.reshape(after.shape)   # x += B @ A @ silu(t_emb), strength in B
    assert torch.allclose(after - before, want, atol=1e-5)
    off = dit.adaln_t_table[[2, 6]].float() + 1e-4               # near a row: the nearest row's grid entry stands in
    assert torch.allclose(torch.cat(ap(off), dim=-1) - torch.cat(turbo._adaln_forward(ap, [], dit.adaln_t_table,
                                                                 egrid)(off), dim=-1), want, atol=1e-5)
    turbo.adaln_unpatch(pairs)
    turbo.adaln_unpatch(pairs)                                   # idempotent
    assert torch.equal(torch.cat(ap(t_emb), dim=-1), before) and "forward" not in vars(ap)
    assert turbo.adaln_patch(dit, pairs, "cpu", torch.float32) == 0        # the real grid has 1025 rows, not 9
    assert turbo.adaln_patch(_dit(), pairs, "cpu", torch.float32, egrid=egrid) == 0   # a full base needs no injection


def test_frozen_rows_training_set_and_preview_set(desc, driver, tmp_path, monkeypatch):
    """Fizgig 7.0.1: the adapter's and the context's rows train; a preview swaps in context + speed, then back."""
    from PIL import Image
    from training import train
    from training.families.minimax_h3 import turbo
    dit = _dit(pruned=True)
    f, g = tmp_path / "turbo.safetensors", tmp_path / "adapter.safetensors"
    _turbo_file(f, dit)
    _turbo_file(g, dit, rank=3)
    egrid = torch.randn(9, E_WIDE, generator=torch.Generator().manual_seed(3))
    monkeypatch.setattr(turbo, "load_h3_egrid", lambda: egrid)
    monkeypatch.setattr(MiniMaxH3Driver, "decode", lambda self, vae, lat, w, h: Image.new("RGB", (w, h)))
    net = FamilyLoRA(dit, driver)
    assert net.add_file(str(g), train.ADAPTER, 1.0) == len(dit.blocks)
    driver.frozen_file_added(dit, str(g), 1.0, "adapter")
    ap = dit.blocks[0].adaln_proj
    patched = vars(ap)["forward"]
    assert "forward" in vars(ap)                                  # the training set is on for every step
    assert net.add_file(str(f), train.SPEED, 0.75) == len(dit.blocks)      # the AdaLN rows do not fit as weights
    net.set_enabled(train.SPEED, False)
    driver.frozen_file_added(dit, str(f), 0.75, "speed")
    assert set(driver._frozen_adaln) == {"adapter", "speed"}
    net.add_trainable(4, 4)
    cond = {"hidden_states": torch.randn(7, 24)}
    speed = desc.preview_speed()
    assert (speed.strength, speed.settings.steps, desc.preview_speed_defaults()) == (0.75, 6, (6, 0.75))
    seen = []
    real = driver._generate

    def spy(d, c, w, h, *a):
        rows = {id(m) for r in ("speed",) for m, _a, _b in driver._frozen_adaln[r]}
        seen.append((id(d.blocks[0].adaln_proj) in rows and "forward" in vars(d.blocks[0].adaln_proj),
                     vars(d.blocks[0].adaln_proj).get("forward") is not patched,
                     d.blocks[0].attn.qkv_proj.scales[train.SPEED]))
        return real(d, c, w, h, *a)
    monkeypatch.setattr(driver, "_generate", spy)
    neg = {"hidden_states": torch.randn(3, 24)}
    paths = train._render_previews(driver, dit, net, None, [cond], str(tmp_path / "sample"), 1, output_name="t",
                                   steps=6, cfg=2.5, neg=neg, width=64, height=64, seed=5, speed=speed.settings)
    assert len(paths) == 1
    assert seen == [(True, True, 0.75)]                          # the preview set, the Turbo's Linears at 0.75
    assert "forward" in vars(ap) and dit.blocks[0].attn.qkv_proj.scales[train.SPEED] == 0.0   # training set back
    monkeypatch.setattr(driver, "_generate", real)
    plain = driver.generate(dit, cond, 64, 64, steps=2, seed=5)
    net.set_enabled(train.SPEED, True)
    assert not torch.allclose(plain, driver.generate(dit, cond, 64, 64, steps=2, seed=5))   # the Turbo changes it


def test_turbo_pipeline_pace_and_visibility(desc, tmp_path):
    from training.families.krea2.description import KREA2
    steps, pace, strength = (P.BY_KEY[k] for k in ("FAMILY_TURBO_STEPS", "FAMILY_TURBO_PACE", "FAMILY_TURBO_STRENGTH"))
    assert P.family_shows(steps, desc) and P.family_shows(pace, desc) and not P.family_shows(strength, desc)
    assert P.family_shows(strength, KREA2) and not P.family_shows(steps, KREA2)
    assert not steps.preset and "FAMILY_TURBO_PACE" not in P.PRESET_KEYS and (steps.default, pace.default) == (6, 75.0)
    turbo = tmp_path / "turbo.safetensors"
    turbo.write_bytes(b"x")
    vals = {**P.defaults(), "LORA_NAME": "h3", "SAMPLE_STEPS": 20, "SAMPLE_PROMPT": "a photo"}
    models = {"minimax_dit": "d", "minimax_vae": "v", "minimax_text_encoder": "t", "minimax_turbo_lora": str(turbo)}
    kw, _ = pipeline.train_kwargs(desc, vals, tmp_path / "run", models)
    assert kw["speed_lora"] == str(turbo) and "speed_lora_strength" not in kw and kw["sample_steps"] == 6
    kw, _ = pipeline.train_kwargs(desc, {**vals, "FAMILY_TURBO_STEPS": 4, "FAMILY_TURBO_PACE": 250}, tmp_path / "run",
                                  models)
    assert (kw["speed_lora_strength"], kw["sample_steps"]) == (2.0, 4)       # Fizgig clamps the strength to 0-2
    kw, _ = pipeline.train_kwargs(desc, {**vals, "FAMILY_TURBO_PACE": 0}, tmp_path / "run", models)
    assert "speed_lora" not in kw and kw["sample_steps"] == 20              # 0 % = previews without it
    kw, _ = pipeline.train_kwargs(desc, vals, tmp_path / "run", {**models, "minimax_turbo_lora": ""})
    assert "speed_lora" not in kw and kw["sample_steps"] == 20              # no Turbo file: the Steps box applies
