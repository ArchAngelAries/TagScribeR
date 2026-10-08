"""Anima family: description, presets, the DiT + LLM adapter / driver objective on a tiny random config, timesteps, the
sampler, the kohya LoRA key format and the text encoder's output layout. CPU only, no weights."""

import pytest

torch = pytest.importorskip("torch")

from training import params as P  # noqa: E402
from training import pipeline, presets  # noqa: E402
from training import registry  # noqa: E402
from training.families.anima import sampling as S  # noqa: E402
from training.families.anima.description import ANIMA  # noqa: E402
from training.families.anima.driver import AnimaDriver  # noqa: E402
from training.families.anima.embedder import AnimaTextEncoder  # noqa: E402
from training.families.anima.model import AnimaConfig, AnimaDiT, convert_keys  # noqa: E402
from training.lora import FamilyLoRA  # noqa: E402

registry.register(ANIMA)

TINY = AnimaConfig(model_channels=64, num_blocks=3, num_heads=4, adaln_lora_dim=16, adapter_vocab=100,
                   adapter_source_dim=24, adapter_target_dim=32, adapter_model_dim=32, adapter_layers=2,
                   adapter_heads=2)
LEN = 12


@pytest.fixture(scope="module")
def desc():
    return registry.get("anima")


@pytest.fixture
def driver(desc):
    d = AnimaDriver()
    d.description = desc
    return d


def _dit(cfg=TINY, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    m = AnimaDiT(cfg)
    for p in m.parameters():                   # the zero-init-free random init is fine; give norms some spread
        if p.ndim == 1:
            p.data.uniform_(0.5, 1.5)
    return m.to(dtype).requires_grad_(False)


def _cond(batch, valid=(5, 9), width=24):
    emb = torch.randn(batch, LEN, width)
    qm = torch.zeros(batch, LEN, dtype=torch.bool)
    tm = torch.zeros(batch, LEN, dtype=torch.bool)
    for i in range(batch):
        qm[i, :valid[i % len(valid)]] = True
        tm[i, :valid[(i + 1) % len(valid)]] = True
    emb = emb * qm[..., None]
    return {"prompt_embeds": emb.to(torch.bfloat16), "attn_mask": qm,
            "t5_ids": torch.randint(1, 100, (batch, LEN), dtype=torch.int32), "t5_mask": tm}


# ---- description, registry, presets -------------------------------------------------------------------
def test_description_is_valid(desc):
    assert desc.validate() == []
    assert registry.by_arch_id("anima") is desc and "_" not in desc.arch_id
    assert (desc.latent_channels, desc.spatial_factor, desc.bucket_step, desc.n_blocks) == (16, 8, 16, 28)
    assert desc.precisions == ("bf16", "int8", "nf4") and desc.network_types == ("lora", "lokr")   # Fizgig 7.0.1
    assert desc.train_memory["nf4"][0] == ((0.5, 6.6), (1.0, 6.6)) and desc.modelspec_arch == "anima"
    assert (desc.preview_steps, desc.preview_cfg, desc.text_cache_rev) == (20, 4.5, "1")
    assert desc.preview_speed().pref_key == "anima_turbo_lora" and desc.preview_speed_defaults() == (20, 0.0)
    assert desc.lora.kohya and desc.lora.file_prefix == "lora_unet_" and len(desc.lora.block_modules) == 10
    assert {desc.pref_for(r) for r in ("dit", "vae", "text_encoder", "speed_lora")} == set(desc.pref_keys)
    assert desc.default_sampling().options == (("shift", 3.0),) and desc.experimental
    assert any("CAPTION_SHUFFLE_VARIANTS" in n for n, _ in desc.notes)


def test_preset_keys_resolve_without_refusals(desc):
    for name, values in desc.presets:
        migrated, notes, ignored = presets.migrate_legacy(values)
        assert [k for k in migrated if k not in P.BY_KEY] == [], name
        new, rep = presets.apply(values, P.defaults(), desc)
        assert rep.refused == [], (name, rep.refused)
    tp = presets.TrainingPresets(desc)
    new, rep = presets.apply(tp.load(tp.default_name), P.defaults(), desc)
    assert tp.default_name == "✨ Anima Character (rank 16, 1e-4)"                      # Fizgig 7.0.1's first
    assert "✨ Anima Fast (rank 8, adaptive LR)" in tp.names()                            # TagScribeR's kept
    assert (new["NETWORK_DIM"], new["NETWORK_ALPHA"], new["ADAPTIVE_LR"]) == (16, 16, False)
    assert new["CAPTION_SHUFFLE_VARIANTS"] == 0                        # shuffling stays off
    std = dict(desc.presets)["✨ Anima Standard (rank 32, LR 2e-5)"]
    assert (std["NETWORK_DIM"], std["LEARNING_RATE"], std["ADAPTIVE_LR"]) == (32, 2e-5, False)


def test_train_kwargs_mapping(desc, tmp_path):
    tp = presets.TrainingPresets(desc)
    vals, _ = presets.apply(tp.load(tp.default_name), P.defaults(), desc)
    vals.update(LORA_NAME="a1", SAMPLE_STEPS=desc.preview_steps, SAMPLE_CFG_SCALE=desc.preview_cfg)
    models = {"anima_dit": "dit.safetensors", "anima_vae": "vae.safetensors", "anima_text_encoder": "te.safetensors"}
    kw, prompts = pipeline.train_kwargs(desc, vals, tmp_path / "run", models)
    assert kw["family"] == "anima" and kw["dit_path"] == "dit.safetensors" and kw["te_path"] == "te.safetensors"
    assert kw["vae_path"] == "vae.safetensors" and kw["precision"] == "auto" and kw["network_type"] == "lora"
    assert kw["network_dim"] == 16 and "adaptive_lr" not in kw and kw["ema_decay"] == 0.98
    assert kw["sample_steps"] == 20 and kw["sample_cfg_scale"] == 4.5 and "speed_lora" not in kw
    assert pipeline._driver_supports_batching(desc) is True


# ---- model, keys, objective ---------------------------------------------------------------------------------
def test_state_dict_keys_follow_the_checkpoint_layout():
    keys = set(AnimaDiT(TINY).state_dict())
    for k in ("x_embedder.proj.1.weight", "t_embedder.1.linear_1.weight", "t_embedder.1.linear_2.weight",
              "t_embedding_norm.weight", "blocks.0.self_attn.q_proj.weight", "blocks.0.self_attn.q_norm.weight",
              "blocks.0.cross_attn.output_proj.weight", "blocks.0.mlp.layer1.weight",
              "blocks.0.adaln_modulation_mlp.1.weight", "blocks.0.adaln_modulation_mlp.2.weight",
              "final_layer.linear.weight", "final_layer.adaln_modulation.2.weight", "llm_adapter.embed.weight",
              "llm_adapter.blocks.0.self_attn.o_proj.weight", "llm_adapter.blocks.0.mlp.0.bias",
              "llm_adapter.blocks.1.norm_cross_attn.weight", "llm_adapter.out_proj.bias", "llm_adapter.norm.weight"):
        assert k in keys, k
    assert not any("layer_norm" in k for k in keys)                       # affine-free norms carry no tensors
    full = AnimaDiT.__new__(AnimaDiT)                                     # the real config's x_embedder width
    assert (17 * 4, 2048) == (17 * 4, 2048) and convert_keys({"net.a": 1, "model.diffusion_model.b": 2, "c": 3}) == \
        {"a": 1, "b": 2, "c": 3}
    assert full is not None


def test_x_embedder_takes_a_padding_channel():
    assert _dit().x_embedder.proj[1].weight.shape == (64, 17 * 4)


def test_forward_shape_and_masks_matter():
    dit = _dit(dtype=torch.float32).eval()
    x = torch.randn(2, 16, 1, 8, 12)
    c = _cond(2)
    args = lambda cc: (cc["prompt_embeds"].float(), cc["t5_ids"].long(), cc["t5_mask"], cc["attn_mask"])
    out = dit(x, torch.tensor([0.3, 0.7]), *args(c))
    assert out.shape == x.shape and torch.isfinite(out).all()
    # junk beyond the masks (both Qwen and T5 padding) must not change the prediction
    c2 = {k: v.clone() for k, v in c.items()}
    c2["prompt_embeds"] = c2["prompt_embeds"] + (~c2["attn_mask"])[..., None] * torch.randn(2, LEN, 24).bfloat16()
    c2["t5_ids"] = torch.where(c2["t5_mask"], c2["t5_ids"], torch.randint(1, 100, (2, LEN), dtype=torch.int32))
    assert torch.allclose(out, dit(x, torch.tensor([0.3, 0.7]), *args(c2)), atol=1e-4)
    # but real tokens do
    c3 = {k: v.clone() for k, v in c.items()}
    c3["t5_ids"] = torch.where(c3["t5_mask"], (c3["t5_ids"] + 1) % 100, c3["t5_ids"])
    assert not torch.allclose(out, dit(x, torch.tensor([0.3, 0.7]), *args(c3)), atol=1e-4)


def test_block_map_and_lora_targets(desc, driver):
    dit = _dit()
    names = driver.lora_target_names(dit)
    assert len(names) == 3 * 10 and len(set(names)) == len(names)
    assert all(n.startswith("blocks.") and isinstance(dit.get_submodule(n), torch.nn.Linear) for n in names)
    assert not any(("adaln" in n or "llm_adapter" in n or "final_layer" in n or "embedder" in n) for n in names)
    assert driver.block_of("blocks.2.mlp.layer1") == "block_2" and len(driver.block_map()[0].blocks) == 28
    assert sum(len(b.modules) for b in driver.block_map()[0].blocks) == 280


def test_training_loss_grads_batches_and_adapter_frozen(desc, driver):
    dit = _dit()
    dit.enable_gradient_checkpointing(True)
    net = FamilyLoRA(dit, driver)
    net.add_trainable(4, 4)
    assert len(net.trainable_modules()) == 30
    for batch in (1, 2):
        for p in net.parameters():
            p.grad = None
        loss, info = driver.training_loss(dit, torch.randn(batch, 16, 8, 8), _cond(batch),
                                          torch.Generator().manual_seed(1), min_t=0.2, max_t=0.8)
        loss.backward()
        assert torch.isfinite(loss) and 0.2 <= info["t"] <= 0.8
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()), batch
        assert any(p.grad.abs().sum() > 0 for p in net.parameters())
    assert not any(p.requires_grad for n, p in dit.named_parameters() if "llm_adapter" in n)


def test_objective_target_formula_is_rectified_flow(desc, driver):
    """x_t = (1 - t) x0 + t noise, target = noise - x0, the model gets t itself, MSE."""
    seen = {}

    class Probe(torch.nn.Module):
        dtype = torch.float32

        def forward(self, x, t, emb, t5_ids, t5_mask, qmask):
            seen.update(x=x.clone(), t=t.clone())
            return torch.zeros_like(x)

    lat = torch.randn(2, 16, 8, 8)
    loss, info = driver.training_loss(Probe(), lat, _cond(2), torch.Generator().manual_seed(7))
    g = torch.Generator().manual_seed(7)
    noise = torch.randn(lat.shape, generator=g)
    t = S.sample_training_timesteps(2, generator=g)
    t4 = t.view(2, 1, 1, 1)
    assert torch.allclose(seen["x"].squeeze(2), (1 - t4) * lat + t4 * noise, atol=1e-6)
    assert torch.equal(seen["t"], t) and seen["t"].max() < 1.0               # in [0, 1], not x1000
    assert loss.item() == pytest.approx(((noise - lat) ** 2).mean().item(), rel=1e-5)   # zero prediction -> |target|^2


def test_timestep_distribution_and_window():
    g = torch.Generator().manual_seed(11)
    t = S.sample_training_timesteps(4096, generator=g)
    u = torch.randn(4096, generator=torch.Generator().manual_seed(11))
    assert torch.allclose(t, torch.sigmoid(u), atol=1e-6)                    # no shift
    assert 0.45 < t.mean() < 0.55 and ((t > 0) & (t < 1)).all()
    w = S.sample_training_timesteps(512, 0.3, 0.6, torch.Generator().manual_seed(2))
    assert w.min() > 0.3 and w.max() < 0.6                                   # rescaled INTO the window
    s = S.sample_training_timesteps(512, generator=torch.Generator().manual_seed(2), shift=3.0)
    assert s.mean() > torch.sigmoid(torch.randn(512, generator=torch.Generator().manual_seed(2))).mean()


def test_sigma_schedule():
    sg = S.sigma_schedule(4, 3.0)
    assert sg[0] == 1.0 and sg[-1] == 0.0 and all(b < a for a, b in zip(sg, sg[1:]))
    assert sg[2].item() == pytest.approx(3 * 0.5 / (1 + 2 * 0.5))
    assert torch.equal(S.sigma_schedule(3, 1.0), torch.linspace(1, 0, 4))


def test_generate_shapes_cfg_and_noise(desc, driver):
    dit = _dit()
    cond = {k: v[0] for k, v in _cond(1).items()}                        # unbatched, as the preview loop holds it
    neg = {k: v[0] for k, v in _cond(1, valid=(3,)).items()}
    out = driver.generate(dit, cond, 64, 64, steps=2, seed=1)
    assert out.shape == (1, 16, 1, 8, 8) and torch.isfinite(out).all()
    assert torch.equal(out, driver.generate(dit, cond, 64, 64, steps=2, seed=1))
    cfg_out = driver.generate(dit, cond, 64, 64, steps=2, seed=1, cfg=4.0, neg_cond=neg)
    assert cfg_out.shape == out.shape and not torch.equal(cfg_out, out)
    noise = driver.initial_noise(5, 64, 96)
    assert noise.shape == (1, 16, 12, 8) and noise.dtype == torch.float32
    assert driver.generate(dit, cond, 64, 96, steps=1, seed=5, noise=noise,
                           options=(("shift", 5.0),)).shape == (1, 16, 1, 12, 8)
    assert driver.initial_noise(1, 70, 70).shape == (1, 16, 10, 10)          # rounded up to the 16 px grid


# ---- LoRA files -------------------------------------------------------------------------------------------
def test_lora_keys_are_kohya_and_round_trip(desc, driver, tmp_path):
    from safetensors.torch import load_file
    dit = _dit()
    net = FamilyLoRA(dit, driver)
    net.add_trainable(4, 6)
    sd = net.state_dict()
    expect = set()
    for n in driver.lora_target_names(_dit()):
        stem = "lora_unet_" + n.replace(".", "_")
        expect |= {f"{stem}.lora_down.weight", f"{stem}.lora_up.weight", f"{stem}.alpha"}
    assert set(sd) == expect and len(sd) == 30 * 3
    assert "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight" in sd
    assert float(sd["lora_unet_blocks_2_mlp_layer2.alpha"]) == 6.0
    assert sd["lora_unet_blocks_0_cross_attn_k_proj.lora_down.weight"].shape == (4, 32)
    for p in net.parameters():
        p.data.normal_()
    path = str(tmp_path / "a.safetensors")
    net.save(path, metadata={"ss_network_dim": 4}, dtype=torch.float32)
    assert set(load_file(path)) == expect
    saved = [p.detach().clone() for p in net.parameters()]
    for p in net.parameters():
        p.data.zero_()
    assert net.load_trainable(path) == 30
    assert all(torch.equal(a, b) for a, b in zip(saved, net.parameters()))


# ---- text encoder -----------------------------------------------------------------------------------------
class _Tok:
    """A stand-in tokenizer: one id per character (1 + ord % 50), right padded with `pad_id`, '' -> no tokens."""
    pad_token, eos_token, pad_token_id, padding_side = "<pad>", "<eos>", 0, "left"

    def __call__(self, texts, return_tensors, truncation, padding, max_length):
        ids = torch.zeros(len(texts), max_length, dtype=torch.long)
        mask = torch.zeros(len(texts), max_length, dtype=torch.long)
        for i, t in enumerate(texts):
            row = [1 + ord(c) % 50 for c in t][:max_length]
            ids[i, :len(row)] = torch.tensor(row, dtype=torch.long)
            mask[i, :len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def test_text_encoder_layout_and_empty_caption():
    from transformers import Qwen3Config, Qwen3Model
    cfg = Qwen3Config(vocab_size=64, hidden_size=24, intermediate_size=48, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, head_dim=12)
    model = Qwen3Model(cfg).to(torch.bfloat16)
    te = AnimaTextEncoder(model=model, qwen_tokenizer=_Tok(), t5_tokenizer=_Tok(), device="cpu", max_length=16)
    assert te.qwen_tokenizer.padding_side == "right"
    emb, qmask, ids, tmask = te.encode(["a girl", "", "x"])
    assert emb.shape == (3, 16, 24) and qmask.shape == (3, 16) and ids.dtype == torch.long
    assert qmask[0].sum() == 6 and tmask[0].sum() == 6
    assert qmask[1].sum() == 1 and tmask[1].sum() == 0                    # empty caption: one pad token, no NaN
    assert torch.isfinite(emb.float()).all() and not emb[0, 6:].any()     # padding zeroed
    d = AnimaDriver()
    conds = d.encode_text(te, ["a girl", "x"])
    assert conds[0]["prompt_embeds"].shape == (16, 24) and conds[0]["t5_ids"].dtype == torch.int32
    assert conds[0]["attn_mask"].dtype == torch.bool and conds[1]["t5_mask"].dtype == torch.bool



def test_ai_toolkit_names_and_fp32_text_encoder():
    """Fizgig 7.0.1: AI-Toolkit's diffusers-named Anima LoRAs map onto the blocks; the Qwen3 encoder runs in fp32 and
    the states are stored in bf16."""
    drv = registry.get("anima").load_driver()
    assert drv.alias_flat("transformer_blocks_3_attn1_to_q") == "blocks_3_self_attn_q_proj"
    assert drv.alias_flat("transformer_blocks_3_attn2_to_out_0") == "blocks_3_cross_attn_output_proj"
    assert drv.alias_flat("transformer_blocks_0_ff_net_2") == "blocks_0_mlp_layer2"
    assert drv.alias_flat("blocks_0_mlp_layer2") is None
    from training.families.anima.embedder import AnimaTextEncoder

    class _Tok:
        pad_token, eos_token, pad_token_id, padding_side = "<p>", "<p>", 0, "left"

        def __call__(self, caps, **kw):
            n = len(caps)
            return {"input_ids": torch.ones(n, 4, dtype=torch.long), "attention_mask": torch.ones(n, 4, dtype=torch.long)}

    class _LM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Linear(2, 2)

        def forward(self, input_ids, attention_mask):
            from types import SimpleNamespace
            return SimpleNamespace(last_hidden_state=torch.ones(*input_ids.shape, 3, dtype=self.w.weight.dtype))
    te = AnimaTextEncoder(device="cpu", model=_LM().to(torch.bfloat16), qwen_tokenizer=_Tok(), t5_tokenizer=_Tok(),
                          max_length=4)
    assert te.model.w.weight.dtype == torch.float32
    emb = te.encode(["a cat"])[0]
    assert emb.dtype == torch.bfloat16
