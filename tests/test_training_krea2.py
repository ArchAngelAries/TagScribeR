"""Krea 2 family: description, presets, the DiT / driver objective on a tiny random config, the timestep sampler, the
kohya LoRA key format, the Turbo LoRA's bias deltas, the checkpoint loader's fp8 refusal and the text encoder's
hidden-state layout. CPU only, no weights; every model here is a few layers wide."""
import math
import os

import pytest

torch = pytest.importorskip("torch")

from training import params as P  # noqa: E402
from training import pipeline, presets  # noqa: E402
from training import registry  # noqa: E402
from training.families.krea2 import sampling as S  # noqa: E402
from training.families.krea2.driver import Krea2Driver  # noqa: E402
from training.families.krea2.model import SingleMMDiTConfig, SingleStreamDiT  # noqa: E402
from training.lora import FamilyLoRA, LoKR  # noqa: E402

TINY = SingleMMDiTConfig(features=64, tdim=32, txtdim=32, heads=4, kvheads=2, multiplier=2, layers=3, patch=2,
                         channels=16, txtheads=2, txtkvheads=2, txtlayers=3)


@pytest.fixture(scope="module")
def desc():
    return registry.get("krea2")


@pytest.fixture
def driver(desc):
    d = Krea2Driver()
    d.description = desc
    return d


def _dit(config=TINY, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    return SingleStreamDiT(config).to(dtype).requires_grad_(False)


def _cond(batch, text_len=20, layers=3, dim=32, valid=(7, 12)):
    """Conditioning shaped like the cache: a padded stack (L, layers, dim) and a mask [valid prompt, pad, 5 suffix]."""
    hs = torch.randn(batch, text_len, layers, dim)
    mask = torch.zeros(batch, text_len, dtype=torch.bool)
    for i in range(batch):
        n = valid[i % len(valid)]
        mask[i, :n] = True
        mask[i, text_len - 5:] = True
    return {"hidden_states": hs, "attention_mask": mask}


# ---- description, registry, presets ---------------------------------------------------------------------
def test_description_is_valid_and_registered_first(desc):
    assert desc.validate() == []
    assert list(registry.FAMILIES)[0] == "krea2"
    assert registry.training_families()[0].key == "krea2"
    assert registry.by_arch_id("krea2") is desc
    assert "_" not in desc.arch_id                       # cache filenames split on "_"
    assert (desc.latent_channels, desc.spatial_factor, desc.bucket_step) == (16, 8, 16)
    assert desc.precisions == ("bf16", "fp8", "int8", "nf4") and set(desc.train_memory) == {"fp8", "int8", "nf4"}
    assert desc.auto_order == ("int8", "nf4", "fp8") and desc.auto_swap_order == ("fp8",)     # Fizgig's ladder
    assert "automagic3" not in desc.optimizers and "adamw8bit" in desc.optimizers
    assert desc.network_types == ("lora", "lokr") and desc.ema_default == "0.98"
    assert {desc.pref_for(r) for r in ("dit", "vae", "text_encoder", "speed_lora")} == set(desc.pref_keys)
    sp = desc.preview_speed()
    assert (sp.strength, sp.settings.steps, sp.settings.cfg) == (1.0, 8, 1.0) and dict(sp.settings.options)["mu"] == 1.15
    assert desc.preview_speed_defaults() == (8, 1.0)


def test_preset_keys_are_known_or_migrated(desc):
    for name, values in desc.presets:
        migrated, notes, ignored = presets.migrate_legacy(values)
        assert [k for k in migrated if k not in P.BY_KEY] == [], name
        assert set(ignored) <= {"COMPILE_BLOCKS", "TARGET_LAYERS"}
        assert notes == []
        new, rep = presets.apply(values, P.defaults(), desc)
        assert rep.refused == [], (name, rep.refused)


def test_first_preset_is_the_ultra_fast_default(desc):
    tp = presets.TrainingPresets(desc)
    assert tp.default_name == "✨ Krea 2 Ultra Fast (rank 8, adaptive LR)"
    new, rep = presets.apply(tp.load(tp.default_name), P.defaults(), desc)
    assert (new["NETWORK_DIM"], new["NETWORK_ALPHA"], new["MAX_TRAIN_EPOCHS"]) == (8, 8, 30)
    assert new["ADAPTIVE_LR"] is True and P.first_token(new["ADAPTIVE_LR_MIN"]) == "2e-4"
    assert new["ADAPTIVE_LR_MAX"] == "4e-4" and new["FAMILY_EMA"] == "0.98 (recommended)"
    assert new["FAMILY_PRECISION"] == P.PRECISION_LABELS["auto"]
    assert (new["MIN_TIMESTEP"], new["MAX_TIMESTEP"]) == (0.0, 1.0)         # Fizgig's blank boxes
    assert new["DATASET_MEGAPIXELS"] == "0.25" and new["KREA2_PER_IMAGE_LR"] is True
    std = dict(desc.presets)["✨ Krea 2 Standard (rank 32, full model)"]
    assert (std["NETWORK_DIM"], std["MAX_TRAIN_EPOCHS"], std["ADAPTIVE_LR"]) == (32, 64, False)
    sty = dict(desc.presets)["✨ Krea 2 Style (rank 16, gentle LR)"]
    assert (sty["NETWORK_DIM"], sty["ADAPTIVE_LR_MIN"], sty["ADAPTIVE_LR_MAX"]) == (16, "5e-5", "2e-4")


def test_legacy_precision_and_ema_keys_map(desc):
    cur = P.defaults()
    new, rep = presets.apply({"QUANT_4BIT_MODE": "nf4", "KREA2_EMA": "Off", "KREA2_FINETUNE_ROTATION": 3,
                              "COMPILE_BLOCKS": "On"}, cur, desc)
    assert new["FAMILY_PRECISION"] == P.PRECISION_LABELS["nf4"] and new["FAMILY_EMA"] == "Off"
    assert sorted(rep.ignored) == ["COMPILE_BLOCKS", "KREA2_FINETUNE_ROTATION"] and rep.refused == []
    new, rep = presets.apply({"QUANT_4BIT_MODE": "fp8"}, cur, desc)
    assert new["FAMILY_PRECISION"] == P.PRECISION_LABELS["fp8"] and rep.refused == []
    for old, want in (("no_4bit", "int8"), ("Off", "int8"), ("On", "nf4"), ("Auto (recommended)", "auto"),
                      ("INT8 — 8-bit, fastest", "int8"), ("4-bit NF4 — smallest", "nf4")):
        new, _ = presets.apply({"QUANT_4BIT_MODE": old}, cur, desc)
        assert new["FAMILY_PRECISION"] == P.PRECISION_LABELS[want], old
    new, _ = presets.apply({"QUANT_4BIT_MODE": "int8", "FAMILY_PRECISION": P.PRECISION_LABELS["nf4"]}, cur, desc)
    assert new["FAMILY_PRECISION"] == P.PRECISION_LABELS["nf4"]            # a new-style key wins over its legacy twin


def test_train_kwargs_map_precision_ema_adaptive_and_turbo(desc, tmp_path):
    tp = presets.TrainingPresets(desc)
    vals, _ = presets.apply(tp.load(tp.default_name), P.defaults(), desc)
    vals.update(LORA_NAME="k2", SAMPLE_STEPS=desc.preview_steps, SAMPLE_CFG_SCALE=desc.preview_cfg)
    turbo = tmp_path / "turbo.safetensors"
    turbo.write_bytes(b"x")
    models = {"krea2_dit": "dit.safetensors", "krea2_vae": "vae.safetensors", "krea2_text_encoder": "te.safetensors",
              "krea2_turbo_lora": str(turbo)}
    kw, prompts = pipeline.train_kwargs(desc, vals, tmp_path / "run", models)
    assert kw["family"] == "krea2" and kw["dit_path"] == "dit.safetensors" and kw["te_path"] == "te.safetensors"
    assert (kw["network_dim"], kw["network_alpha"], kw["precision"], kw["blocks_to_swap"]) == (8, 8.0, "auto", -1)
    assert kw["ema_decay"] == 0.98 and kw["adaptive_lr"] is True
    assert (kw["adaptive_lr_min"], kw["adaptive_lr_max"]) == (2e-4, 4e-4) and kw["network_type"] == "lora"
    assert kw["log_per_image_loss"] is True and kw["per_image_lr"] is True and kw["optimizer_type"] == "adamw8bit"
    assert prompts and kw["speed_lora"] == str(turbo) and "speed_lora_strength" not in kw
    assert kw["sample_steps"] == 8                                          # the Turbo LoRA's own steps, not 28
    vals["SAMPLE_STEPS"] = 12
    assert pipeline.train_kwargs(desc, vals, tmp_path / "run", models)[0]["sample_steps"] == 12
    new, _ = presets.apply({"QUANT_4BIT_MODE": "int8", "KREA2_EMA": "0.99 (stronger)"}, vals, desc)
    kw, _ = pipeline.train_kwargs(desc, new, tmp_path / "run", {**models, "krea2_turbo_lora": ""})
    assert kw["precision"] == "int8" and kw["ema_decay"] == 0.99 and "speed_lora" not in kw
    assert kw["sample_steps"] == 12


# ---- the DiT and the objective ----------------------------------------------------------------------------
def test_block_map_covers_every_linear_exactly_once(desc, driver):
    dit = _dit()
    names = driver.lora_target_names(dit)
    linears = {n for n, m in dit.named_modules() if isinstance(m, torch.nn.Linear)}
    assert set(names) == linears and len(names) == len(set(names))
    assert len(names) == 3 * 8 + (4 * 8 + 1) + 7
    assert all(driver.block_of(n) is not None for n in names)
    assert driver.block_of("blocks.2.mlp.up") == "block_2" and driver.block_of("last.linear") == "io_out"
    assert driver.block_of("txtfusion.projector") == "txtfusion_projector"
    # quantisation reaches only the main blocks' Linears
    assert set(driver.quant_target_names(dit)) == {n for n in linears if n.startswith("blocks.")}
    assert sum(len(b.modules) for g in driver.block_map() for b in g.blocks) == 28 * 8 + 33 + 7 == 264


def test_training_loss_batch_one_and_two_with_grads(desc, driver):
    dit = _dit()
    dit.enable_gradient_checkpointing(True)
    net = FamilyLoRA(dit, driver)
    net.add_trainable(4, 4)
    assert len(net.trainable_modules()) == 64
    for batch in (1, 2):
        for p in net.parameters():
            p.grad = None
        loss, info = driver.training_loss(dit, torch.randn(batch, 16, 8, 8), _cond(batch),
                                          torch.Generator().manual_seed(1), min_t=0.2, max_t=0.8)
        loss.backward()
        assert torch.isfinite(loss) and 0.2 <= info["t"] <= 0.8
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()), batch
        assert any(p.grad.abs().sum() > 0 for p in net.parameters())


def test_block_swap_does_not_change_the_loss(desc, driver):
    cfg = SingleMMDiTConfig(**{**TINY.__dict__, "layers": 4})
    args = (torch.randn(1, 16, 8, 8), _cond(1))

    def loss_of(swap):
        dit = _dit(cfg)
        if swap:
            driver.enable_block_swap(dit, swap, torch.device("cpu"), True)
        return driver.training_loss(dit, *args, torch.Generator().manual_seed(3))[0].item()
    assert driver.max_blocks_to_swap(_dit(cfg)) == 2
    assert loss_of(2) == pytest.approx(loss_of(0), rel=1e-5)


def test_text_padding_does_not_change_the_prediction():
    """The key-padding mask keeps padded text out of every attention (float32 so the comparison is tight)."""
    dit = _dit(dtype=torch.float32).eval()
    noise = torch.randn(1, 16, 8, 8)
    hs = torch.randn(1, 6, 3, 32)

    def run(pad):
        txt = torch.cat([hs, torch.randn(1, pad, 3, 32)], dim=1)
        mask = torch.cat([torch.ones(1, 6, dtype=torch.bool), torch.zeros(1, pad, dtype=torch.bool)], dim=1)
        tokens, pos, m = S.prepare(noise, txt.shape[1], 2, mask)
        return dit(img=tokens, context=txt, t=torch.tensor([0.4]), pos=pos, mask=m)
    assert torch.allclose(run(0), run(70), atol=1e-4)


def test_gather_valid_text_drops_interior_padding():
    cond = _cond(2)
    txt, mask = S.gather_valid_text(cond["hidden_states"], cond["attention_mask"])
    assert txt.shape[1] % S.TEXT_PAD_MULTIPLE == 0 and mask.sum(1).tolist() == [12, 17]
    assert torch.equal(txt[0, :12], cond["hidden_states"][0][cond["attention_mask"][0]])
    assert not txt[0, 12:].any()


def test_patchify_round_trip_and_positions():
    x = torch.randn(2, 16, 8, 12)
    tokens, pos, mask = S.patchify_block(x, 2)
    assert tokens.shape == (2, 4 * 6, 64) and mask.all()
    assert torch.equal(S.unpatchify(tokens, 4, 6, 2)[:, :, 0], x)
    assert pos[0, 7].tolist() == [0.0, 1.0, 1.0] and pos[1, 23].tolist() == [0.0, 3.0, 5.0]


def test_generate_shapes_cfg_and_noise(desc, driver):
    dit = _dit()
    cond = {k: v[0] for k, v in _cond(1).items()}                       # unbatched, as the preview loop holds it
    neg = {k: v[0] for k, v in _cond(1, valid=(5,)).items()}
    out = driver.generate(dit, cond, 64, 64, steps=2, seed=1)
    assert out.shape == (1, 16, 1, 8, 8)
    again = driver.generate(dit, cond, 64, 64, steps=2, seed=1)
    assert torch.equal(out, again)
    cfg_out = driver.generate(dit, cond, 64, 64, steps=2, seed=1, cfg=3.0, neg_cond=neg)
    assert cfg_out.shape == out.shape and not torch.equal(cfg_out, out)
    turbo = driver.generate(dit, cond, 64, 64, steps=2, seed=1, options=(("mu", 1.15),))
    assert turbo.shape == out.shape
    noise = driver.initial_noise(5, 64, 96)
    assert noise.shape == (1, 16, 12, 8) and noise.dtype == torch.float32
    ds = driver.generate(dit, cond, 64, 96, steps=1, seed=5, noise=noise)
    assert ds.shape == (1, 16, 1, 12, 8)
    assert driver.initial_noise(1, 70, 70).shape == (1, 16, 10, 10)           # rounded up to the 16 px grid


# ---- timesteps ---------------------------------------------------------------------------------------------
def _fizgig_t(u, n_tokens, lo=0.0, hi=1.0):
    """A direct port of Fizgig krea2/trainer.py sample_krea2_timesteps' formula for a given base normal u."""
    m = (1.15 - 0.5) / (6400 - 256)
    mu = m * n_tokens + (0.5 - m * 256)
    shift = math.exp(mu)
    t = 1 / (1 + math.exp(-u))
    t = (t * shift) / (1.0 + (shift - 1.0) * t)
    if lo > 0.0 or hi < 1.0:
        t = lo + t * max(hi - lo, 1e-6)
    return t


def test_timestep_sampler_matches_fizgig_formula():
    for n_tokens in (256, 1024, 4096, 6400):
        g = torch.Generator().manual_seed(11)
        u = torch.randn(64, generator=torch.Generator().manual_seed(11))
        t = S.sample_krea2_timesteps(64, n_tokens, generator=g)
        assert torch.allclose(t, torch.tensor([_fizgig_t(float(x), n_tokens) for x in u]), atol=1e-6)
        assert ((t > 0) & (t < 1)).all()
    g = torch.Generator().manual_seed(11)
    w = S.sample_krea2_timesteps(64, 1024, min_timestep=0.3, max_timestep=0.6, generator=g)
    assert ((w >= 0.3) & (w <= 0.6)).all()
    assert torch.allclose(w, torch.tensor([_fizgig_t(float(x), 1024, 0.3, 0.6) for x in
                                           torch.randn(64, generator=torch.Generator().manual_seed(11))]), atol=1e-6)
    assert w.min() > 0.3 and w.max() < 0.6                                  # rescaled INTO the window, not clamped


def test_resolution_shift_is_monotonic():
    ts = [S.sample_krea2_timesteps(256, n, generator=torch.Generator().manual_seed(5)) for n in (256, 1024, 4096, 6400)]
    for a, b in zip(ts, ts[1:]):
        assert (b >= a).all() and b.mean() > a.mean()                       # more tokens = more noise (larger t)
    sched = [S.timesteps(1024, 8, 256, 6400, mu=m) for m in (0.5, 1.15)]
    assert sched[0][0] == 1.0 and sched[0][-1] == 0.0 and all(b <= a for a, b in zip(sched[0], sched[0][1:]))
    assert all(hi >= lo for hi, lo in zip(sched[1], sched[0]))


# ---- LoRA files -------------------------------------------------------------------------------------------
def test_lora_keys_are_kohya_and_round_trip(desc, driver, tmp_path):
    from safetensors.torch import load_file
    dit = _dit()
    net = FamilyLoRA(dit, driver)
    net.add_trainable(4, 6)
    sd = net.state_dict()
    expect = set()
    for n in driver.lora_target_names(_dit()):          # Fizgig: "lora_unet" + "." + path, dots -> underscores
        stem = "lora_unet_" + n.replace(".", "_")
        expect |= {f"{stem}.lora_down.weight", f"{stem}.lora_up.weight", f"{stem}.alpha"}
    assert set(sd) == expect and len(sd) == 64 * 3
    assert "lora_unet_blocks_0_attn_wq.lora_down.weight" in sd and "lora_unet_last_linear.lora_up.weight" in sd
    assert "lora_unet_txtfusion_layerwise_blocks_1_mlp_down.alpha" in sd and float(sd["lora_unet_first.alpha"]) == 6.0
    assert sd["lora_unet_blocks_0_attn_wq.lora_down.weight"].shape == (4, 64)
    for p in net.parameters():
        p.data.normal_()
    path = str(tmp_path / "k2.safetensors")
    net.save(path, metadata={"ss_network_dim": 4}, dtype=torch.float32)
    assert set(load_file(path)) == expect
    saved = [p.detach().clone() for p in net.parameters()]
    for p in net.parameters():
        p.data.zero_()
    assert net.load_trainable(path) == 64
    assert all(torch.equal(a, b) for a, b in zip(saved, net.parameters()))
    # a kohya file reads back as a frozen adapter on every module (the Turbo LoRA's route)
    dit2 = _dit()
    net2 = FamilyLoRA(dit2, driver)
    assert net2.add_file(path, "frozen") == 64


def test_lokr_keys_are_diffusion_model_dotted_and_round_trip(desc, driver, tmp_path):
    dit = _dit()
    net = FamilyLoRA(dit, driver)
    net.add_trainable(4, 4, kind="lokr", factor=4)
    sd = net.state_dict()
    assert "diffusion_model.blocks.0.attn.wq.lokr_w1" in sd and "diffusion_model.last.linear.lokr_w2" in sd
    assert sd["diffusion_model.first.alpha"].item() == 1.0 and len(sd) == 64 * 3
    assert not any(k.startswith("lora_unet_") for k in sd)
    for p in net.parameters():
        p.data.normal_()
    path = str(tmp_path / "lokr.safetensors")
    net.save(path, dtype=torch.float32)
    saved = [p.detach().clone() for p in net.parameters()]
    for p in net.parameters():
        p.data.zero_()
    assert net.load_trainable(path) == 64
    assert all(torch.equal(a, b) for a, b in zip(saved, net.parameters()))
    assert isinstance(net.wrapped["first"].adapters["lora"], LoKR)
    net2 = FamilyLoRA(_dit(), driver)
    assert net2.add_file(path, "frozen") == 64                              # LyCORIS names read back


def test_turbo_style_file_applies_and_restores_bias_deltas(desc, driver, tmp_path):
    from safetensors.torch import save_file
    dit = _dit()
    net = FamilyLoRA(dit, driver)
    f = {"diffusion_model.blocks.0.attn.wq.lora_down.weight": torch.randn(4, 64),
         "diffusion_model.blocks.0.attn.wq.lora_up.weight": torch.randn(64, 4),
         "diffusion_model.blocks.0.attn.wq.alpha": torch.tensor(4.0),
         "diffusion_model.first.diff_b": torch.full((64,), 0.5), "diffusion_model.last.linear.diff_b": torch.ones(64),
         "diffusion_model.nothing.diff_b": torch.ones(3)}
    path = str(tmp_path / "turbo.safetensors")
    save_file(f, path)
    base = dit.first.base.bias.detach().clone()
    assert net.add_file(path, "speed_lora") == 1
    assert torch.allclose(dit.first.base.bias.float(), (base + 0.5).float(), atol=1e-2)       # on after add_file
    net.set_enabled("speed_lora", False)
    assert torch.equal(dit.first.base.bias, base)                                              # restored exactly
    net.set_enabled("speed_lora", True)
    assert not torch.equal(dit.first.base.bias, base)
    net.set_strength("speed_lora", 0.0)
    assert torch.equal(dit.first.base.bias, base)
    net.set_strength("speed_lora", 1.0)
    net.remove("speed_lora")
    assert torch.equal(dit.first.base.bias, base)


# ---- loading ------------------------------------------------------------------------------------------------
def test_loader_reads_bf16_and_rejects_wrong_keys(tmp_path):
    from safetensors.torch import save_file

    from training.families.krea2.model import load_krea2_dit
    sd = {k: v.to(torch.float32) for k, v in _dit(dtype=torch.float32).state_dict().items()}
    good = str(tmp_path / "raw.safetensors")
    save_file(sd, good)
    dit = load_krea2_dit(good, device="cpu", config=TINY)
    assert dit.first.weight.dtype == torch.bfloat16 and torch.equal(dit.first.weight.float(),
                                                                    sd["first.weight"].bfloat16().float())
    partial = str(tmp_path / "partial.safetensors")
    save_file({k: v for k, v in sd.items() if k != "first.weight"}, partial)
    with pytest.raises(ValueError, match="mismatch"):
        load_krea2_dit(partial, device="cpu", config=TINY)


def test_text_encoder_prequantized_dequantisation():
    from training.families.krea2.embedder import dequantize_prequantized
    w = torch.randn(4, 8)
    q = {"a.weight": (w / 0.1), "a.weight_scale": torch.tensor(0.1), "b.weight": w, "c.weight": w,
         "c.weight_scale": torch.full((4,), 0.5), "a.comfy_quant": torch.zeros(1)}
    out = dequantize_prequantized(q)
    assert set(out) == {"a.weight", "b.weight", "c.weight"}
    assert torch.allclose(out["a.weight"].float(), w, atol=2e-2) and torch.equal(out["b.weight"], w)
    assert torch.allclose(out["c.weight"].float(), w * 0.5, atol=2e-2)
    assert dequantize_prequantized({"x.weight": w}) == {"x.weight": w}


def test_text_encoder_hidden_state_stack_layout():
    """The conditioning stack is hidden_states[i] of the base model for the chosen layers: the INPUT of decoder layer i
    (so it holds on the installed transformers), padded to max_length with the mask [prompt, pad, suffix]."""
    tf = pytest.importorskip("transformers")
    from transformers import BatchEncoding
    cfg = tf.Qwen3VLConfig.from_dict({
        "model_type": "qwen3_vl", "image_token_id": 90, "video_token_id": 91, "vision_start_token_id": 92,
        "vision_end_token_id": 93, "tie_word_embeddings": True,
        "text_config": {"model_type": "qwen3_vl_text", "hidden_size": 32, "intermediate_size": 64,
                        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "num_hidden_layers": 4,
                        "vocab_size": 100, "rms_norm_eps": 1e-6, "rope_theta": 1000, "max_position_embeddings": 256,
                        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [2, 1, 1], "rope_type": "default"},
                        "tie_word_embeddings": True},
        "vision_config": {"model_type": "qwen3_vl", "depth": 2, "hidden_size": 16, "intermediate_size": 32,
                          "num_heads": 2, "in_channels": 3, "patch_size": 4, "spatial_merge_size": 2,
                          "temporal_patch_size": 2, "out_hidden_size": 32, "num_position_embeddings": 16,
                          "deepstack_visual_indexes": [0, 1]}})
    model = tf.Qwen3VLForConditionalGeneration._from_config(cfg).eval()
    model.lm_head = None
    model.model.visual = None
    from training.families.krea2 import embedder as E

    class FakeTokenizer:
        """Word-free stand-in: the template prefix is 34 tokens, the suffix 5, a prompt one token per character."""
        def __call__(self, text, truncation=False, return_length=False, return_overflowing_tokens=False,
                     padding=False, max_length=None, return_tensors=None):
            single = isinstance(text, str)
            texts = [text] if single else text
            ids = []
            for t in texts:
                if t.startswith(E.PREFIX):
                    row = [1] * 34 + [ord(c) % 60 + 5 for c in t[len(E.PREFIX):]]
                elif t == E.SUFFIX:
                    row = [3] * 5
                else:
                    raise AssertionError(t)
                ids.append(row[:max_length] if truncation and max_length else row)
            width = max_length if padding == "max_length" else max(map(len, ids))
            mask = [[1] * len(r) + [0] * (width - len(r)) for r in ids]
            ids = [r + [0] * (width - len(r)) for r in ids]
            if return_tensors is None:
                return BatchEncoding({"input_ids": ids[0] if single else ids,
                                      "attention_mask": mask[0] if single else mask})
            return BatchEncoding({"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)})

    enc = E.Krea2TextEncoder(device="cpu", dtype=torch.float32, model=model, tokenizer=FakeTokenizer(),
                             select_layers=(0, 1, 2, 3))
    hs, mask = enc.encode(["a cat", "a longer prompt"])
    assert hs.shape == (2, 512, 4, 32) and mask.shape == (2, 512)
    assert mask.sum(1).tolist() == [5 + 5, 15 + 5]
    assert mask[0, :5].all() and not mask[0, 5:507].any() and mask[0, 507:].all()       # prompt, padding, suffix
    caps = {}
    handles = [model.model.language_model.layers[i].register_forward_pre_hook(
        lambda m, args, kwargs, i=i: caps.__setitem__(i, (args[0] if args else kwargs["hidden_states"]).detach().clone()),
        with_kwargs=True) for i in range(4)]
    enc.encode(["a cat"])
    for h in handles:
        h.remove()
    hs1, _ = enc.encode(["a cat"])
    for slot, i in enumerate((0, 1, 2, 3)):
        assert torch.allclose(hs1[0, :, slot], caps[i][0, E.PREFIX_START_IDX:], atol=1e-5), i
    # the cached form stacks and gathers to what the DiT is fed
    cond = Krea2Driver().encode_text(enc, ["a cat"])[0]
    assert cond["hidden_states"].dtype == torch.bfloat16 and cond["attention_mask"].dtype == torch.bool
    txt, m = S.gather_valid_text(cond["hidden_states"][None], cond["attention_mask"][None])
    assert m.sum() == 10 and txt.shape[1] == 64


def test_driver_encode_images_shapes_and_normalisation(driver):
    """encode_images squeezes the VAE frame axis to (16, h, w) bf16 latents (a stand-in VAE with the same interface)."""
    class Vae:
        device, dtype = torch.device("cpu"), torch.float32

        def encode_pixels_to_latents(self, x):
            assert x.shape[1:3] == (3, 1) and x.min() >= -1 and x.max() <= 1
            b, _, _, h, w = x.shape
            return torch.zeros(b, 16, 1, h // 8, w // 8)
    import numpy as np
    out = driver.encode_images(Vae(), [np.zeros((32, 48, 3), np.uint8), np.full((32, 48, 3), 255, np.uint8)])
    assert len(out) == 2 and out[0].shape == (16, 4, 6) and out[0].dtype == torch.bfloat16


def test_third_party_notice_and_headers_are_present():
    root = os.path.dirname(os.path.dirname(__file__))
    notices = open(os.path.join(root, "THIRD_PARTY_NOTICES.md"), encoding="utf-8").read()
    assert "training/families/krea2/" in notices and "ai-toolkit" in notices
    for name in ("model", "sampling", "vae", "vae_loader", "embedder", "driver", "description"):
        head = open(os.path.join(root, "training", "families", "krea2", f"{name}.py"), encoding="utf-8").read(900)
        assert head.startswith("# Ported from Fizgig"), name
        assert "Apache License, Version 2.0" in head, name


def test_vae_port_encodes_and_decodes_on_a_tiny_config(driver):
    """The ported Qwen-Image VAE (a few channels wide): the driver's encode_images / decode keep shapes and sizes."""
    import numpy as np

    from training.families.krea2.vae import AutoencoderKLQwenImage
    vae = AutoencoderKLQwenImage(base_dim=8, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=1, attn_scales=[],
                                 temperal_downsample=[False, True, True], input_channels=3).eval()
    lat = driver.encode_images(vae, [(np.random.rand(32, 48, 3) * 255).astype(np.uint8)])
    assert lat[0].shape == (16, 4, 6) and lat[0].dtype == torch.bfloat16
    assert driver.decode(vae, lat[0][None, :, None], 48, 32).size == (48, 32)
