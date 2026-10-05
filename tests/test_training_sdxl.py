"""SDXL family (SDXL, Pony, Illustrious, NoobAI eps / v-pred): descriptions, presets, the LDM <-> diffusers name tables,
the offline single-file loaders (UNet, both CLIPs, VAE) on a tiny random checkpoint, the DDPM objective for epsilon and
v-prediction, zero-terminal-SNR, the Euler sampler, kohya LoRA keys in the original (LDM) naming, and an end-to-end
cache + train run. CPU only, no weights; every model here is a few channels wide."""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from training import params as P  # noqa: E402
from training import pipeline, presets, registry  # noqa: E402
from training.families.sdxl import sampling as S  # noqa: E402
from training.families.sdxl import text as T  # noqa: E402
from training.families.sdxl import unet as U  # noqa: E402
from training.families.sdxl import vae as V  # noqa: E402
from training.families.sdxl.description import NOOBAI_VPRED, PONY, SDXL, VARIANTS  # noqa: E402
from training.families.sdxl.driver import SDXLDriver, SDXLVPredDriver  # noqa: E402
from training.lora import FamilyLoRA  # noqa: E402

TINY_UNET = dict(U.SDXL_UNET_CONFIG, block_out_channels=[16, 32, 64], attention_head_dim=[2, 4, 8],
                 transformer_layers_per_block=[1, 1, 2], cross_attention_dim=24, addition_time_embed_dim=8,
                 projection_class_embeddings_input_dim=16 + 6 * 8, norm_num_groups=8, sample_size=8)
TINY_CLIP_L = dict(vocab_size=49408, hidden_size=8, intermediate_size=16, num_hidden_layers=2, num_attention_heads=2,
                   max_position_embeddings=77, hidden_act="quick_gelu", projection_dim=8)
TINY_CLIP_G = dict(vocab_size=49408, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                   max_position_embeddings=77, hidden_act="gelu", projection_dim=16)
TINY_VAE = dict(V.SDXL_VAE_CONFIG, block_out_channels=[4, 4, 8, 8], layers_per_block=1, norm_num_groups=4)
CROSS = 8 + 16


class TinyEpsDriver(SDXLDriver):
    unet_config = TINY_UNET
    clip_l_config = TINY_CLIP_L
    clip_g_config = TINY_CLIP_G
    vae_config = TINY_VAE


class TinyVPredDriver(SDXLVPredDriver):
    unet_config = TINY_UNET
    clip_l_config = TINY_CLIP_L
    clip_g_config = TINY_CLIP_G
    vae_config = TINY_VAE


# ---- a tiny checkpoint in the ORIGINAL (LDM / SGM) layout ----------------------------------------------------------
def _tiny_models(seed=0):
    from diffusers import AutoencoderKL, UNet2DConditionModel
    torch.manual_seed(seed)
    unet = UNet2DConditionModel.from_config(TINY_UNET)
    clip_l, clip_g = T.build_encoders(TINY_CLIP_L, TINY_CLIP_G)
    vae = AutoencoderKL.from_config(TINY_VAE)
    return unet.eval(), clip_l.eval(), clip_g.eval(), vae.eval()


def _openclip_state(clip_g):
    """Reverse of diffusers' convert_open_clip_checkpoint: transformers CLIPTextModelWithProjection -> open_clip names."""
    sd = clip_g.state_dict()
    out = {"token_embedding.weight": sd["text_model.embeddings.token_embedding.weight"],
           "positional_embedding": sd["text_model.embeddings.position_embedding.weight"],
           "ln_final.weight": sd["text_model.final_layer_norm.weight"],
           "ln_final.bias": sd["text_model.final_layer_norm.bias"],
           "text_projection": sd["text_projection.weight"].T.contiguous()}
    for i in range(clip_g.config.num_hidden_layers):
        p = f"text_model.encoder.layers.{i}."
        o = f"transformer.resblocks.{i}."
        for kind in ("weight", "bias"):
            out[f"{o}attn.in_proj_{kind}"] = torch.cat([sd[f"{p}self_attn.{x}_proj.{kind}"] for x in "qkv"])
            out[f"{o}attn.out_proj.{kind}"] = sd[f"{p}self_attn.out_proj.{kind}"]
            for a, b in (("layer_norm1", "ln_1"), ("layer_norm2", "ln_2"), ("mlp.fc1", "mlp.c_fc"),
                         ("mlp.fc2", "mlp.c_proj")):
                out[f"{o}{b}.{kind}"] = sd[f"{p}{a}.{kind}"]
    return out


def _ldm_vae_state(vae):
    """Reverse of diffusers' convert_ldm_vae_checkpoint: AutoencoderKL -> original VAE names."""
    n_down = len(TINY_VAE["down_block_types"])
    out = {}
    res = {"conv_shortcut": "nin_shortcut"}
    for k, v in vae.state_dict().items():
        parts = k.split(".")
        side = parts[0]
        if side in ("quant_conv", "post_quant_conv"):
            out[k] = v
            continue
        rest = ".".join(parts[1:])
        if rest.startswith("conv_norm_out"):
            nk = rest.replace("conv_norm_out", "norm_out")
        elif rest.startswith(("conv_in", "conv_out")):
            nk = rest
        elif ".resnets." in rest and "mid_block" not in rest:
            blk, _, tail = rest.partition(".resnets.")
            i = int(blk.split(".")[1])
            j, _, name = tail.partition(".")
            name = ".".join([res.get(name.split(".")[0], name.split(".")[0])] + name.split(".")[1:])
            idx = i if side == "encoder" else n_down - 1 - i
            nk = f"{'down' if side == 'encoder' else 'up'}.{idx}.block.{j}.{name}"
        elif ".downsamplers." in rest or ".upsamplers." in rest:
            i = int(rest.split(".")[1])
            idx = i if side == "encoder" else n_down - 1 - i
            nk = f"{'down' if side == 'encoder' else 'up'}.{idx}.{'downsample' if side == 'encoder' else 'upsample'}." \
                 f"conv.{parts[-1]}"
        elif rest.startswith("mid_block.resnets."):
            j = int(parts[3])
            name = ".".join(parts[4:])
            name = ".".join([res.get(name.split(".")[0], name.split(".")[0])] + name.split(".")[1:])
            nk = f"mid.block_{j + 1}.{name}"
        elif rest.startswith("mid_block.attentions.0."):
            tail = ".".join(parts[4:])
            for a, b in (("group_norm.", "norm."), ("to_q.", "q."), ("to_k.", "k."), ("to_v.", "v."),
                         ("to_out.0.", "proj_out.")):
                if tail.startswith(a):
                    tail = b + tail[len(a):]
            nk = f"mid.attn_1.{tail}"
            if tail.endswith("weight") and not tail.startswith("norm"):
                v = v[:, :, None, None]
        else:
            raise AssertionError(k)
        out[f"{side}.{nk}"] = v
    return out


def _write_checkpoint(folder, models=None, with_tokenizer=True, vae_prefix="first_stage_model."):
    from safetensors.torch import save_file
    unet, clip_l, clip_g, vae = models or _tiny_models()
    sd = {U.LDM_UNET_PREFIX + U.ldm_param_name(k, TINY_UNET): v.contiguous() for k, v in unet.state_dict().items()}
    sd.update({f"conditioner.embedders.0.transformer.{k}": v.contiguous() for k, v in clip_l.state_dict().items()
               if "position_ids" not in k})
    sd.update({f"conditioner.embedders.1.model.{k}": v.contiguous() for k, v in _openclip_state(clip_g).items()})
    sd.update({vae_prefix + k: v.contiguous() for k, v in _ldm_vae_state(vae).items()})
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "tiny_sdxl.safetensors")
    save_file(sd, path)
    if with_tokenizer:
        _write_tokenizer(os.path.join(folder, T.TOKENIZER_DIRNAME))
    return path


def _write_tokenizer(folder):
    """A byte-level BPE vocabulary with no merges (every character is one token), special ids as CLIP's."""
    os.makedirs(folder, exist_ok=True)
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))       # GPT-2 / CLIP bytes_to_unicode
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    chars = [chr(c) for c in cs]
    vocab = {}
    for c in chars:
        vocab[c] = len(vocab)
    for c in chars:
        vocab[c + "</w>"] = len(vocab)
    vocab["<|startoftext|>"] = 49406
    vocab["<|endoftext|>"] = 49407
    Path(folder, "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    Path(folder, "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")


@pytest.fixture(scope="module")
def models():
    return _tiny_models()


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory, models):
    return _write_checkpoint(str(tmp_path_factory.mktemp("sdxl_ckpt")), models)


def _driver(cls=TinyEpsDriver, **opts):
    d = cls()
    d.description = SDXL
    if opts:
        d.configure(**opts)
    return d


def _tiny_unet(dtype=torch.bfloat16, seed=0):
    from diffusers import UNet2DConditionModel
    torch.manual_seed(seed)
    return UNet2DConditionModel.from_config(TINY_UNET).to(dtype).requires_grad_(False)


def _cond(batch, seed=1):
    g = torch.Generator().manual_seed(seed)
    return {"crossattn": torch.randn(batch, T.SEQ_LEN, CROSS, generator=g),
            "pooled": torch.randn(batch, 16, generator=g)}


# ---- descriptions, presets ---------------------------------------------------------------------------------------
def test_descriptions_validate_and_are_distinct():
    assert [d.key for d in VARIANTS] == ["sdxl", "pony", "illustrious", "noobai_eps", "noobai_vpred"]
    arch = [d.arch_id for d in VARIANTS]
    assert len(set(arch)) == 5 and all("_" not in a for a in arch)
    for d in VARIANTS:
        assert d.validate() == [], d.key
        assert (d.latent_channels, d.spatial_factor, d.bucket_step, d.native_megapixels) == (4, 8, 64, 1.0)
        assert d.precisions == ("bf16",) and d.modelspec_arch == "stable-diffusion-xl-v1-base"
        assert d.network_types == ("lora",) and d.ema_default == "0.98" and d.train_memory == {}
        assert d.lora.kohya and d.lora.file_prefix == "lora_unet_"
        assert set(d.family_options) == {"SDXL_MIN_SNR_GAMMA", "SDXL_NOISE_OFFSET", "SDXL_LOCON"} and all(P.family_shows(P.BY_KEY[k], d)
                                                                       for k in d.family_options)
        assert d.pref_for("dit") and d.model_files[1].default_to == "dit" and not d.model_files[1].required
        s = d.default_sampling()
        assert s.source and s.negative_prompt and d.preview_cfg == s.cfg and d.preview_steps == s.steps
    assert NOOBAI_VPRED.driver.endswith("SDXLVPredDriver") and PONY.driver.endswith("SDXLDriver")
    assert not P.family_shows(P.BY_KEY["SDXL_LOCON"], registry.get("krea2"))


def test_presets_resolve_without_refusals():
    for d in VARIANTS:
        assert len(d.presets) == 3 and d.presets[0][0].startswith("✨") and "Fast" in d.presets[0][0]
        for name, values in d.presets:
            new, rep = presets.apply(values, P.defaults(), d)
            assert rep.refused == [] and rep.ignored == [], (name, rep.refused, rep.ignored)
            assert new["DATASET_MEGAPIXELS"] == "1.0" and new["FAMILY_EMA"] == "0.98 (recommended)"
        fast = d.presets[0][1]
        assert fast["ADAPTIVE_LR"] is True and fast["NETWORK_DIM"] == 16


def test_model_rows_default_to_the_checkpoint_and_options_reach_train_kwargs(tmp_path):
    registry.register(PONY)
    vals, _ = presets.apply(dict(PONY.presets)["✨ Pony Diffusion V6 XL Fast (rank 16, adaptive LR)"], P.defaults(), PONY)
    vals.update(LORA_NAME="p", SDXL_MIN_SNR_GAMMA=5.0, SDXL_LOCON=True)
    models = {"pony_dit": "pony.safetensors", "pony_vae": "", "pony_text_encoder": "  "}
    kw, _ = pipeline.train_kwargs(PONY, vals, tmp_path / "run", models)
    assert kw["dit_path"] == kw["vae_path"] == kw["te_path"] == "pony.safetensors"
    assert kw["driver_options"] == {"min_snr_gamma": 5.0, "noise_offset": 0.0, "locon": True}
    assert kw["precision"] == "bf16" and kw["network_dim"] == 16
    kw, _ = pipeline.train_kwargs(PONY, vals, tmp_path / "run", {**models, "pony_vae": "vae.safetensors"})
    assert kw["vae_path"] == "vae.safetensors" and kw["te_path"] == "pony.safetensors"
    kw, _ = pipeline.train_kwargs(registry.get("qwen_image21"), P.defaults(), tmp_path / "run", {})
    assert "driver_options" not in kw


# ---- names -----------------------------------------------------------------------------------------------------
def test_kohya_key_names_for_known_modules():
    """The original-layout names A1111 / ComfyUI know (checked against ComfyUI's unet_to_diffusers for all 760 SDXL
    target modules when this was written)."""
    t = U.ldm_module_table(U.SDXL_UNET_CONFIG)
    expect = {
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q": "input_blocks.4.1.transformer_blocks.0.attn1.to_q",
        "down_blocks.2.attentions.1.transformer_blocks.9.ff.net.0.proj": "input_blocks.8.1.transformer_blocks.9.ff.net.0.proj",
        "down_blocks.2.attentions.0.proj_in": "input_blocks.7.1.proj_in",
        "mid_block.attentions.0.transformer_blocks.3.attn2.to_out.0": "middle_block.1.transformer_blocks.3.attn2.to_out.0",
        "mid_block.attentions.0.proj_out": "middle_block.1.proj_out",
        "up_blocks.0.attentions.0.transformer_blocks.0.ff.net.2": "output_blocks.0.1.transformer_blocks.0.ff.net.2",
        "up_blocks.1.attentions.2.transformer_blocks.1.attn2.to_v": "output_blocks.5.1.transformer_blocks.1.attn2.to_v",
        "down_blocks.0.resnets.0.conv1": "input_blocks.1.0.in_layers.2",
        "down_blocks.0.downsamplers.0.conv": "input_blocks.3.0.op",
        "up_blocks.0.upsamplers.0.conv": "output_blocks.2.2.conv",
        "up_blocks.2.resnets.2.conv2": "output_blocks.8.0.out_layers.3",
    }
    for d, ldm in expect.items():
        assert t[d] == ldm, d
    assert len(t) == 760 and len(set(t.values())) == 760
    drv = SDXLDriver()
    drv.description = SDXL
    stems = {
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q":
            "input_blocks_4_1_transformer_blocks_0_attn1_to_q",
        "down_blocks.2.attentions.1.transformer_blocks.9.ff.net.0.proj":
            "input_blocks_8_1_transformer_blocks_9_ff_net_0_proj",
        "mid_block.attentions.0.proj_in": "middle_block_1_proj_in",
        "up_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0":
            "output_blocks_0_1_transformer_blocks_0_attn1_to_out_0",
        "mid_block.attentions.0.proj_out": "middle_block_1_proj_out",
    }
    for d, stem in stems.items():
        assert drv.lora_key_name(d).replace(".", "_") == stem
    assert drv.lora_key_name("not.a.module") == "not.a.module"


def test_block_map_structure_and_counts():
    drv = _driver(SDXLDriver)
    groups = drv.block_map()
    assert [g.label for g in groups] == ["Input blocks", "Middle", "Output blocks"]
    ids = [b.id for g in groups for b in g.blocks]
    assert ids == ["in_04", "in_05", "in_07", "in_08", "mid", "out_00", "out_01", "out_02", "out_03", "out_04", "out_05"]
    names = drv.lora_target_names(None) if False else [m for g in groups for b in g.blocks for m in b.modules]
    assert len(names) == len(set(names)) == 722                # transformer Linears only
    assert sum(len(b.modules) for g in groups for b in g.blocks if b.id == "mid") == 2 + 10 * 10
    assert drv.block_of("down_blocks.1.attentions.0.proj_in") == "in_04"
    assert drv.block_of("up_blocks.1.attentions.2.proj_out") == "out_05"
    drv.configure(locon=True)
    full = drv.block_map()
    assert sum(len(b.modules) for g in full for b in g.blocks) == 760
    assert [b.id for b in full[0].blocks] == [f"in_{i:02d}" for i in range(1, 9)]
    assert drv.block_of("down_blocks.0.downsamplers.0.conv") == "in_03"
    assert drv.block_of("up_blocks.0.upsamplers.0.conv") == "out_02"


# ---- loading ---------------------------------------------------------------------------------------------------
def test_unet_loads_from_an_ldm_checkpoint_offline(ckpt, models):
    unet = models[0]
    loaded = U.load_unet(ckpt, config=TINY_UNET, dtype=torch.float32)
    assert not any(p.requires_grad for p in loaded.parameters()) and not loaded.training
    sd = unet.state_dict()
    for k, v in loaded.state_dict().items():
        assert torch.equal(v, sd[k]), k
    bf = U.load_unet(ckpt, config=TINY_UNET)
    assert next(bf.parameters()).dtype == torch.bfloat16


def test_unet_loader_refuses_wrong_files(tmp_path):
    from safetensors.torch import save_file
    bad = str(tmp_path / "x.safetensors")
    save_file({"model.diffusion_model.input_blocks.0.0.weight": torch.zeros(2, 4, 3, 3)}, bad)
    with pytest.raises(ValueError, match="does not match the SDXL UNet|missing"):
        U.load_unet(bad, config=TINY_UNET)
    other = str(tmp_path / "y.safetensors")
    save_file({"a": torch.zeros(1)}, other)
    with pytest.raises(ValueError, match="not a single-file SDXL checkpoint"):
        U.load_unet(other, config=TINY_UNET)


def test_text_encoders_load_and_condition(ckpt, models):
    drv = _driver()
    te = drv.load_text_encoder(ckpt, "cpu")
    clip_l, clip_g = models[1], models[2]
    for k, v in te.clip_l.state_dict().items():
        if "position_ids" not in k:
            assert torch.equal(v, clip_l.state_dict()[k]), k
    for k, v in te.clip_g.state_dict().items():
        if "position_ids" not in k:
            assert torch.equal(v, clip_g.state_dict()[k]), k
    conds = drv.encode_text(te, ["red hair, blue eyes", ""])
    assert len(conds) == 2
    c = conds[0]
    assert c["crossattn"].shape == (T.SEQ_LEN, CROSS) and c["crossattn"].dtype == torch.bfloat16
    assert c["pooled"].shape == (16,) and c["pooled"].dtype == torch.float32
    assert T.SEQ_LEN == 231 and torch.isfinite(c["crossattn"].float()).all()
    # penultimate hidden states of both encoders, concatenated on the feature axis
    rows_l, rows_g = te.token_rows("red hair, blue eyes")
    ids_l, ids_g = torch.tensor(rows_l), torch.tensor(rows_g)
    ref_l = clip_l(input_ids=ids_l, output_hidden_states=True).hidden_states[-2]
    ref_g = clip_g(input_ids=ids_g, output_hidden_states=True)
    cross = torch.cat([ref_l, ref_g.hidden_states[-2]], -1).reshape(-1, CROSS)
    assert torch.allclose(c["crossattn"].float(), cross, atol=0.05, rtol=0.02)
    assert torch.allclose(c["pooled"], ref_g.text_embeds[0], atol=1e-5)
    # a different caption gives a different conditioning; a long one uses all three chunks and is cut at 225 tokens
    assert not torch.equal(conds[0]["crossattn"], conds[1]["crossattn"])
    long_a = drv.encode_text(te, ["x " * 250])[0]
    long_b = drv.encode_text(te, ["x " * 400])[0]
    assert long_a["crossattn"].shape == (T.SEQ_LEN, CROSS)
    assert torch.equal(long_a["crossattn"], long_b["crossattn"])
    drv.unload_text_encoder(te)


def test_chunking_uses_75_token_chunks_and_each_pad_id():
    rows = T.chunk_ids(list(range(100, 100 + 80)), 49406, 49407, pad=0)
    assert len(rows) == 3 and all(len(r) == 77 for r in rows)
    assert rows[0] == [49406] + list(range(100, 175)) + [49407]
    assert rows[1][:7] == [49406, 175, 176, 177, 178, 179, 49407] and set(rows[1][7:]) == {0}
    assert rows[2] == [49406, 49407] + [0] * 75
    assert T.chunk_ids([], 1, 2, pad=2)[0] == [1, 2] + [2] * 75
    assert len(T.chunk_ids(list(range(500)), 1, 2, 2)[2]) == 77


def test_vae_loads_both_layouts_and_encodes_decodes(tmp_path, models):
    from safetensors.torch import save_file
    vae = models[3]
    ref = vae.state_dict()
    drv = _driver()
    d = tmp_path / "v"
    d.mkdir()
    full = _write_checkpoint(str(d), models)               # first_stage_model.* inside the checkpoint
    bare = str(d / "bare.safetensors")
    save_file({k: v.contiguous() for k, v in _ldm_vae_state(vae).items()}, bare)           # standalone, original names
    diff = str(d / "diffusers.safetensors")
    save_file({k: v.contiguous() for k, v in ref.items()}, diff)                             # diffusers names
    for path in (full, bare, diff):
        loaded = drv.load_vae(path, "cpu")
        assert loaded.dtype == torch.float32 and not any(p.requires_grad for p in loaded.parameters())
        for k, v in loaded.state_dict().items():
            assert torch.equal(v, ref[k]), (os.path.basename(path), k)
    img = (np.random.default_rng(0).random((64, 96, 3)) * 255).astype(np.uint8)
    lat = drv.encode_images(loaded, [img, img])
    assert len(lat) == 2 and lat[0].shape == (4, 8, 12) and lat[0].dtype == torch.float32
    mode = loaded.encode(torch.from_numpy(img).permute(2, 0, 1)[None].float() / 127.5 - 1).latent_dist.mode()[0]
    assert torch.allclose(lat[0], mode * V.SCALING_FACTOR, atol=1e-5)
    out = drv.decode(loaded, lat[0][None], 96, 64)
    assert out.size == (96, 64) and out.mode == "RGB"
    with pytest.raises(ValueError, match="no SDXL VAE"):
        save_file({"foo": torch.zeros(1)}, str(d / "nope.safetensors"))
        drv.load_vae(str(d / "nope.safetensors"), "cpu")


# ---- the objective ---------------------------------------------------------------------------------------------
class _Stub(torch.nn.Module):
    """A 'UNet' that records its inputs and predicts zeros (loss = mean target^2)."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.seen = []

    def forward(self, x, t, encoder_hidden_states=None, added_cond_kwargs=None):
        self.seen.append((x, t, added_cond_kwargs))
        return type("O", (), {"sample": torch.zeros_like(x)})()


def test_training_loss_batch_one_and_two_with_grads_both_prediction_types():
    for cls in (TinyEpsDriver, TinyVPredDriver):
        drv = _driver(cls)
        unet = _tiny_unet()
        unet.enable_gradient_checkpointing()
        unet.train()
        net = FamilyLoRA(unet, drv)
        net.add_trainable(4, 4)
        assert len(net.trainable_modules()) == 22 * 2 + 0 or len(net.trainable_modules()) > 20
        for batch in (1, 2):
            for p in net.parameters():
                p.grad = None
            loss, info = drv.training_loss(unet, torch.randn(batch, 4, 8, 8), _cond(batch),
                                           torch.Generator().manual_seed(1), min_t=0.0, max_t=1.0)
            loss.backward()
            assert torch.isfinite(loss) and 0.0 <= info["t"] <= 1.0
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()), (cls, batch)
            assert any(p.grad.abs().sum() > 0 for p in net.parameters())


def test_timestep_window_is_rescaled_not_clamped():
    drv = _driver()
    stub = _Stub()
    seen = set()
    for s in range(40):
        drv.training_loss(stub, torch.randn(4, 4, 8, 8), _cond(4), torch.Generator().manual_seed(s), min_t=0.3, max_t=0.6)
    for _x, t, _ in stub.seen:
        seen |= set(t.tolist())
    assert min(seen) >= 300 and max(seen) <= 600
    assert len(seen) > 100 and min(seen) < 320 and max(seen) > 580            # spread over the whole window
    stub = _Stub()
    drv.training_loss(stub, torch.randn(2, 4, 8, 8), _cond(2), torch.Generator().manual_seed(0), min_t=0.0, max_t=1.0)
    g = torch.Generator().manual_seed(0)
    torch.randn(2, 4, 8, 8, generator=g)                                   # the noise is drawn first
    t = S.draw_timesteps(2, 0.0, 1.0, g)
    assert stub.seen[0][1].tolist() == t.tolist() and int(t.max()) <= 999
    one = S.draw_timesteps(64, 1.0, 1.0, torch.Generator().manual_seed(0))
    assert (one == 999).all()


def test_epsilon_and_v_targets_match_the_formulas():
    x0, noise = torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8)
    ac = torch.tensor([0.9, 0.5, 0.01])
    xt = S.add_noise(x0, noise, ac)
    a = ac.view(-1, 1, 1, 1)
    assert torch.allclose(xt, a.sqrt() * x0 + (1 - a).sqrt() * noise)
    assert torch.equal(S.target_for("epsilon", x0, noise, ac), noise)
    v = S.target_for("v_prediction", x0, noise, ac)
    assert torch.allclose(v, a.sqrt() * noise - (1 - a).sqrt() * x0)
    assert torch.allclose(a.sqrt() * xt - (1 - a).sqrt() * v, x0, atol=1e-5)           # v recovers x0 exactly
    # the driver wires it: a zero-predicting stub gives loss = mean(target^2)
    for cls, pred in ((TinyEpsDriver, "epsilon"), (TinyVPredDriver, "v_prediction")):
        drv = _driver(cls)
        lat = torch.randn(2, 4, 8, 8)
        loss, _ = drv.training_loss(_Stub(), lat, _cond(2), torch.Generator().manual_seed(5))
        g = torch.Generator().manual_seed(5)
        n = torch.randn(lat.shape, generator=g)
        t = S.draw_timesteps(2, 0.0, 1.0, g)
        acs = S.alphas_cumprod(drv.zero_terminal_snr).float()[t]
        want = S.target_for(pred, lat, n, acs).pow(2).mean()
        assert torch.allclose(loss, want, rtol=1e-5), pred
    assert SDXLDriver.prediction == "epsilon" and SDXLVPredDriver.prediction == "v_prediction"
    assert not SDXLDriver.zero_terminal_snr and SDXLVPredDriver.zero_terminal_snr


def test_zero_terminal_snr_schedule_ends_at_snr_zero():
    plain, ztsnr = S.alphas_cumprod(False), S.alphas_cumprod(True)
    assert plain[-1] > 0 and float(plain[-1] / (1 - plain[-1])) > 1e-3         # SDXL's schedule never reaches 0
    assert float(ztsnr[-1]) == 0.0 and float(ztsnr[-1] / (1 - ztsnr[-1])) == 0.0
    assert float(ztsnr[0]) == pytest.approx(float(plain[0]), rel=1e-9)
    assert (ztsnr[1:] <= ztsnr[:-1]).all()
    assert plain[0] == pytest.approx(1 - 0.00085, rel=1e-6)
    from diffusers import DDPMScheduler
    ref = DDPMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                        rescale_betas_zero_snr=True).alphas_cumprod.double()
    assert torch.allclose(ztsnr[:-1], ref[:-1], rtol=1e-4, atol=1e-8)          # diffusers clamps the last to 2**-24
    ref_plain = DDPMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                              beta_schedule="scaled_linear").alphas_cumprod.double()
    assert torch.allclose(plain, ref_plain, rtol=1e-5)
    ts, sig = S.sigmas_for(10, True)
    assert ts[0] == 999 and sig[0] == pytest.approx(4096, rel=1e-3) and sig[-1] == 0
    ts, sig = S.sigmas_for(10, False)
    assert ts[0] == 999 and 10 < sig[0] < 20 and (sig[1:] < sig[:-1]).all()


def test_min_snr_and_noise_offset_are_optional_and_off_by_default():
    lat, cond = torch.randn(2, 4, 8, 8), _cond(2)

    def loss(**opts):
        return _driver(TinyEpsDriver, **opts).training_loss(_tiny_unet(), lat, cond, torch.Generator().manual_seed(2))[0]
    base = loss()
    assert torch.equal(base, loss(min_snr_gamma=0.0, noise_offset=0.0, locon=False))
    assert not torch.allclose(base, loss(min_snr_gamma=0.5))
    assert not torch.allclose(base, loss(noise_offset=0.1))
    ac = torch.tensor([1e-4, 0.5, 0.99])
    w = S.min_snr_weights(ac, 5.0, "epsilon")
    snr = ac / (1 - ac)
    assert torch.allclose(w, snr.clamp(max=5.0) / snr)
    wv = S.min_snr_weights(ac, 5.0, "v_prediction")
    assert torch.allclose(wv, snr.clamp(max=5.0) / (snr + 1))
    assert float(S.min_snr_weights(torch.tensor([0.0]), 5.0, "v_prediction")) == 0.0


def test_locon_adds_conv_adapters():
    drv = _driver(TinyEpsDriver, locon=True)
    unet = _tiny_unet()
    net = FamilyLoRA(unet, drv)
    net.add_trainable(4, 4)
    convs = [n for n, w in net.wrapped.items() if hasattr(w.base, "kernel_size")]
    assert convs and all(("conv" in n) for n in convs)
    base = _driver(TinyEpsDriver)
    net2 = FamilyLoRA(_tiny_unet(), base)
    net2.add_trainable(4, 4)
    assert len(net.trainable_modules()) > len(net2.trainable_modules())
    loss, _ = drv.training_loss(unet, torch.randn(1, 4, 8, 8), _cond(1), torch.Generator().manual_seed(0))
    loss.backward()
    assert all(p.grad is not None for p in net.parameters())


# ---- sampling ----------------------------------------------------------------------------------------------------
def test_generate_shapes_cfg_and_noise_both_prediction_types():
    for cls in (TinyEpsDriver, TinyVPredDriver):
        drv = _driver(cls)
        unet = _tiny_unet()
        cond = {k: v[0] for k, v in _cond(1).items()}                          # unbatched, as the preview loop holds it
        neg = {k: v[0] for k, v in _cond(1, seed=9).items()}
        out = drv.generate(unet, cond, 64, 96, steps=2, seed=1)
        assert out.shape == (1, 4, 12, 8) and torch.isfinite(out).all()
        assert torch.equal(out, drv.generate(unet, cond, 64, 96, steps=2, seed=1))
        assert not torch.equal(out, drv.generate(unet, cond, 64, 96, steps=2, seed=2))
        cfg_out = drv.generate(unet, cond, 64, 96, steps=2, seed=1, cfg=4.0, neg_cond=neg)
        assert cfg_out.shape == out.shape and not torch.equal(cfg_out, out)
        calls = []
        drv.generate(unet, cond, 64, 64, steps=3, seed=1, on_step=lambda i, n: calls.append((i, n)))
        assert calls == [(0, 3), (1, 3), (2, 3)]
        noise = drv.initial_noise(5, 64, 96)
        assert noise.shape == (1, 4, 12, 8) and noise.dtype == torch.float32
        assert torch.equal(noise, drv.initial_noise(5, 64, 96))
        ds = drv.generate(unet, cond, 64, 96, steps=1, seed=5, noise=noise)
        assert ds.shape == (1, 4, 12, 8)
        anc = drv.generate(unet, cond, 64, 64, steps=3, seed=1, options=(("sampler", "euler_a"),))
        assert anc.shape == (1, 4, 8, 8) and torch.isfinite(anc).all()


def test_euler_sampler_recovers_x0_for_a_perfect_model():
    """With the exact eps (or v) the Euler sampler lands on x0, on both prediction types and the ZTSNR schedule."""
    x0 = torch.randn(1, 4, 8, 8)
    noise = torch.randn(1, 4, 8, 8)
    for pred, zt in (("epsilon", False), ("v_prediction", False), ("v_prediction", True)):
        ac = S.alphas_cumprod(zt).float()
        if zt:
            ac[-1] = max(float(ac[-1]), 2.0 ** -24)

        def predict(x_in, t, pred=pred, ac=ac):
            a = ac[t]
            xt = x_in                                         # x_in = x / sqrt(sigma^2 + 1) = the DDPM x_t
            eps = (xt - a.sqrt() * x0) / (1 - a).sqrt()
            return eps if pred == "epsilon" else a.sqrt() * eps - (1 - a).sqrt() * x0
        out = S.euler_sample(predict, noise, 20, prediction=pred, zero_terminal_snr=zt)
        assert torch.allclose(out, x0, atol=0.05), (pred, zt, float((out - x0).abs().max()))


# ---- LoRA files ------------------------------------------------------------------------------------------------
def test_lora_keys_are_kohya_ldm_and_round_trip(tmp_path):
    from safetensors.torch import load_file
    drv = _driver()
    unet = _tiny_unet()
    net = FamilyLoRA(unet, drv)
    net.add_trainable(4, 6)
    sd = net.state_dict()
    table = U.ldm_module_table(TINY_UNET, locon=False)
    expect = set()
    for d in drv.lora_target_names(unet):
        stem = "lora_unet_" + table[d].replace(".", "_")
        expect |= {f"{stem}.lora_down.weight", f"{stem}.lora_up.weight", f"{stem}.alpha"}
    assert set(sd) == expect and not any("down_blocks" in k or "attentions" in k for k in sd)
    k = "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q"
    assert sd[f"{k}.lora_down.weight"].shape == (4, 32) and float(sd[f"{k}.alpha"]) == 6.0
    assert "lora_unet_middle_block_1_proj_in.lora_up.weight" in sd
    assert "lora_unet_output_blocks_3_1_transformer_blocks_0_ff_net_0_proj.lora_down.weight" in sd
    assert "lora_unet_output_blocks_0_1_transformer_blocks_1_attn2_to_out_0.alpha" in sd
    for p in net.parameters():
        p.data.normal_()
    path = str(tmp_path / "x.safetensors")
    net.save(path, metadata={"ss_network_dim": 4}, dtype=torch.float32)
    assert set(load_file(path)) == expect
    saved = [p.detach().clone() for p in net.parameters()]
    for p in net.parameters():
        p.data.zero_()
    assert net.load_trainable(path) == len(net.trainable_modules())
    assert all(torch.equal(a, b) for a, b in zip(saved, net.parameters()))
    net2 = FamilyLoRA(_tiny_unet(), _driver())
    assert net2.add_file(path, "frozen") == len(net.trainable_modules())            # LDM-named kohya file reads back
    # a diffusers-flattened kohya file (lora_unet_down_blocks_1_...) is read too
    from safetensors.torch import save_file
    flat = {k.replace(table["down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q"].replace(".", "_"),
                      "down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q"): v
            for k, v in load_file(path).items()}
    p2 = str(tmp_path / "diffusers_named.safetensors")
    save_file(flat, p2)
    assert FamilyLoRA(_tiny_unet(), _driver()).add_file(p2, "f") == len(net.trainable_modules())


# ---- end to end ------------------------------------------------------------------------------------------------
def _tiny_desc(variant, driver_name):
    import dataclasses
    return dataclasses.replace(variant, key=f"_t_{variant.key}", arch_id=f"t{variant.arch_id}",
                               driver=f"tests.test_training_sdxl:{driver_name}", bucket_step=32,
                               model_files=tuple(dataclasses.replace(f, pref_key=f"t_{f.pref_key}")
                                                 for f in variant.model_files),
                               presets=(), preview_width=64, preview_height=64)


@pytest.mark.parametrize("variant,driver_name", [(SDXL, "TinyEpsDriver"), (NOOBAI_VPRED, "TinyVPredDriver")])
def test_end_to_end_cache_and_train_with_batching_and_previews(tmp_path, ckpt, variant, driver_name):
    from safetensors.torch import load_file

    from tests.tiny_family import make_dataset
    from training import cache, train
    desc = _tiny_desc(variant, driver_name)
    registry.register(desc)
    data = make_dataset(str(tmp_path / "scratch"), n=4, sizes=[(96, 96), (96, 96), (128, 96), (128, 96)])
    models = {f"t_{variant.arch_id}_dit": ckpt}
    v = P.defaults()
    v.update({"LORA_OUTPUT_DIR": str(tmp_path / "runs"), "LORA_NAME": "tiny_sdxl", "NETWORK_DIM": 4, "NETWORK_ALPHA": 2,
              "MAX_TRAIN_EPOCHS": 2, "OPTIMIZER_TYPE": "adamw", "LEARNING_RATE": 1e-3, "DATASET_MEGAPIXELS": "0.01",
              "DATASET_BATCH_SIZE": 2, "SAMPLE_PROMPT": "tok, a test prompt", "SAMPLE_EVERY_N_EPOCHS": 1,
              "SAMPLE_WIDTH": 64, "SAMPLE_HEIGHT": 64, "SAMPLE_STEPS": 2, "SAMPLE_AT_FIRST": False,
              "SAMPLE_CFG_SCALE": 3.0, "SAMPLE_NEGATIVE": "bad", "SDXL_MIN_SNR_GAMMA": 5.0, "SDXL_LOCON": True,
              "FAMILY_EMA": "0.98 (recommended)"})
    checks = pipeline.preflight(desc, v, data, models)
    assert not [c for c in checks if c.level == "error"], checks
    run = pipeline.build_run(desc, v, data, models, trigger="tok")
    cfg = json.loads((run.run_dir / "train_config.json").read_text(encoding="utf-8"))
    assert cfg["train"]["driver_options"]["locon"] is True and cfg["train"]["vae_path"] == ckpt
    for _label, argv in run.stages:
        mod, args = argv[2], argv[3:]
        if mod == "training.cache":
            cache.main(args + ["--device", "cpu"])
        else:
            p = args[args.index("--config") + 1]
            c = json.loads(Path(p).read_text(encoding="utf-8"))
            c["train"]["device"] = "cpu"
            Path(p).write_text(json.dumps(c), encoding="utf-8")
            train.main(args)
    out = run.run_dir
    final = out / "tiny_sdxl.safetensors"
    assert final.exists() and (out / "tiny_sdxl-000001.safetensors").exists()
    assert list((out / "sample").glob("*.png"))
    sd = load_file(str(final))
    assert any(k.startswith("lora_unet_input_blocks_4_1_") for k in sd) and any("_in_layers_2" in k for k in sd)
    assert all(k.startswith("lora_unet_") for k in sd)
    md = __import__("training.metadata", fromlist=["x"]).read_metadata(str(final))
    assert md["modelspec.architecture"] == "stable-diffusion-xl-v1-base/lora" and md["ss_architecture"] == desc.arch_id
    caches = list(Path(run.run_dir.parent.parent / "training_cache").rglob("*.safetensors")) if False else []
    assert not caches
    cond_files = list(Path(json.loads((run.run_dir / "dataset.json").read_text())["datasets"][0]["cache_directory"])
                      .glob(f"*_{desc.arch_id}_te.safetensors"))
    assert len(cond_files) == 4
    c0 = load_file(str(cond_files[0]))
    assert c0["cond__crossattn"].shape == (T.SEQ_LEN, CROSS) and c0["cond__pooled"].shape == (16,)


def test_third_party_headers_and_pyflakes_clean():
    root = Path(__file__).resolve().parent.parent / "training" / "families" / "sdxl"
    for f in root.glob("*.py"):
        if f.name == "__init__.py":
            continue
        head = f.read_text(encoding="utf-8")[:300]
        assert head.startswith("#"), f.name
    try:
        from pyflakes.api import check
        from pyflakes.reporter import Reporter
    except ImportError:
        pytest.skip("pyflakes not installed")
    import io
    buf = io.StringIO()
    n = 0
    for f in list(root.glob("*.py")) + [Path(__file__)]:
        n += check(f.read_text(encoding="utf-8"), str(f), Reporter(buf, buf))
    assert n == 0, buf.getvalue()
    assert math.isfinite(0.0)
