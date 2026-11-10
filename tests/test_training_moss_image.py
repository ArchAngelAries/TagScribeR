"""MOSS-Image family: description and the five presets verbatim, the DiT on a tiny random config (forward, skip
layers, the text mask, checkpointing, block swap), the technical report's timestep law and the Clean-end share, the
objective, Train attention only, kohya / LoKR LoRA keys, every checkpoint layout the loader takes (fused ComfyUI, the
diffusers prefix, int8 ConvRot), the sampler against an exact denoiser, the shared FLUX.2 autoencoder with the ROCm
wide-attention path, the Qwen3.5 text encoder on a tiny config (resident == streamed), the SFT preview handoff, the
Turbo-at-zero rule, the Auto plan over Fizgig's measured figures and the torch-free Base fingerprint. CPU only, no
weights."""
import json
import math
import struct

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training import params as P  # noqa: E402
from training import pipeline, presets, quant, registry  # noqa: E402
from training.families.minimax_h3 import weights as W  # noqa: E402
from training.families.moss_image import base_check, sampling as S  # noqa: E402
from training.families.moss_image.description import MOSS_IMAGE  # noqa: E402
from training.families.moss_image.driver import (ATTN_MODULES, BLOCK_MODULES, N_BLOCKS, T_CUTOFF, T_MU,  # noqa: E402
                                                 MOSSImageDriver, _t_cdf, mu_for_clean_share)
from training.families.moss_image.model import MOSSImageDiT, load_moss_dit, read_state_dict  # noqa: E402
from training.lora import FamilyLoRA  # noqa: E402
from training.modules import flux2_vae as V  # noqa: E402

TINY = dict(dim=32, ffn_dim=48, num_heads=2, num_layers=4, text_dim=24, freq_dim=8, latent_channels=128, rope_base=100.0)


@pytest.fixture(scope="module")
def desc():
    return registry.get("moss_image")


@pytest.fixture
def driver(desc):
    d = MOSSImageDriver()
    d.description = desc
    return d


def _dit(seed=0, dtype=torch.bfloat16):
    torch.manual_seed(seed)
    return MOSSImageDiT(**TINY).to(dtype).eval().requires_grad_(False)


def _cond(length=5, dim=24, seed=1):
    g = torch.Generator().manual_seed(seed)
    return {"hidden_states": torch.randn(length, dim, generator=g).to(torch.bfloat16)}


def _latents(h=2, w=3, seed=2):
    return torch.randn(1, 128, h, w, generator=torch.Generator().manual_seed(seed))


# ---- description, registry, presets ---------------------------------------------------------------------
def test_description_is_valid_registered_and_fizgigs_facts(desc):
    assert desc.validate() == [] and desc is MOSS_IMAGE and registry.by_arch_id("mossimage") is desc
    assert (desc.key, desc.arch_id, desc.lora_name_suffix, desc.aliases) == ("moss_image", "mossimage", "moss",
                                                                              ("moss-image", "mossimage"))
    assert (desc.latent_channels, desc.spatial_factor, desc.bucket_step, desc.native_megapixels) == (128, 16, 16, 1.0)
    assert (desc.n_blocks, desc.block_prefix) == (42, "layers")
    assert desc.precisions == ("int8", "nf4", "bf16") and desc.auto_precisions == ("int8", "nf4")
    assert desc.train_memory == {"bf16": (((0.25, 15.1), (1.0, 15.1)), 0.211), "int8": (((0.25, 10.5), (1.0, 10.5)), 0.105),
                                 "nf4": (((0.25, 8.5), (1.0, 8.5)), 0.0)}
    assert desc.optimizers == ("adamw", "adamw8bit") and desc.network_types == ("lora", "lokr")
    assert desc.ema_default == "0.98" and desc.slider_training and not desc.edit_training
    assert desc.compiles and desc.compile_boundary == "outside"
    assert desc.compile_payback_steps == {"bf16": 200, "int8": 200, "nf4": 400}
    assert (desc.preview_steps, desc.preview_cfg, desc.preview_width, desc.preview_height) == (30, 4.0, 1024, 1024)
    assert desc.preview_negative == "worst quality, bad anatomy, watermark, jpeg artifact"
    assert desc.status_note and desc.modelspec_arch == "MOSS-Image"
    assert desc.helper_files == (("OpenMOSS-Team/MOSS-Image-Base", ("tokenizer/*",)),)
    assert desc.lora.key_template == "lora_unet_layers_{block}_{module}.{ab}.weight" and desc.lora.kohya
    assert desc.lora.block_modules == BLOCK_MODULES and not desc.lora.lokr_kohya_stems
    assert list(registry.FAMILIES).index("moss_image") == list(registry.FAMILIES).index("klein9b") + 1
    assert desc in registry.training_families()


def test_model_files_follow_fizgig(desc):
    by = {f.pref_key: f for f in desc.model_files}
    assert list(by) == ["moss_image_dit", "moss_image_vae", "moss_image_turbo_lora", "moss_image_text_encoder",
                        "moss_image_sft_dit"]
    dit = by["moss_image_dit"]
    assert (dit.repo, dit.path, dit.size_gb, dit.role) == ("Comfy-Org/MOSS-Image",
                                                           "diffusion_models/moss_image_base_bf16.safetensors", 10.16, "dit")
    assert dit.fetch_choice == ("MOSS-Image Base", "Kromoss") and dit.alt_size_gb == 10.16
    vae = by["moss_image_vae"]
    assert (vae.path, vae.size_gb, vae.replaces) == ("vae/flux2-vae_small.safetensors", 0.25,
                                                     ("flux2-vae.safetensors", "ae.safetensors"))
    assert vae.migrate("D:/models/flux2-vae.safetensors", exists=lambda p: True) == "D:/models/flux2-vae_small.safetensors"
    te = by["moss_image_text_encoder"]
    assert (te.repo, te.path, te.size_gb) == ("Comfy-Org/Qwen3.5", "text_encoders/qwen3.5_9b_bf16.safetensors", 19.31)
    sft = by["moss_image_sft_dit"]
    assert (sft.role, sft.required, sft.path, sft.size_gb) == ("preview_dit", True,
                                                               "diffusion_models/moss_image_sft_int8_convrot.safetensors", 5.41)
    turbo = by["moss_image_turbo_lora"]
    assert (turbo.role, turbo.required, turbo.repo, turbo.size_gb) == ("speed_lora", False,
                                                                      "lodestones/MOSS-Image-SFT-8-steps-dmd", 1.19)
    assert turbo.fetch_is_optional


def test_five_presets_verbatim(desc):
    names = [n for n, _ in desc.presets]
    assert names == ["◆ Balanced (rank 32, 1e-4)", "◆ Quick start (rank 16, 0.25 MP)",
                     "◆ Style (rank 16, 5e-5)", "◆ Slider (rank 8, 2e-4)",
                     "◆ Full fine-tune (1e-5)"]
    # (rank, lr, epochs, MP, EMA, slider, ft) - Fizgig 8.7.3 families/moss_image.py:234-245
    want = [(32, 1e-4, 70, "1.0", "0.98 (recommended)", False, False), (16, 1e-4, 70, "0.25", "0.98 (recommended)", False, False),
            (16, 5e-5, 50, "1.0", "Off", False, False), (8, 2e-4, 20, "1.0", "0.98 (recommended)", True, False),
            (16, 1e-5, 30, "1.0", "0.98 (recommended)", False, True)]
    for (name, v), w in zip(desc.presets, want):
        assert (v["NETWORK_DIM"], v["LEARNING_RATE"], v["MAX_TRAIN_EPOCHS"], v["DATASET_MEGAPIXELS"], v["FAMILY_EMA"],
                v["FAMILY_SLIDER"], v["FAMILY_FT"]) == w, name
        assert v["NETWORK_ALPHA"] == v["NETWORK_DIM"] and v["ADAPTIVE_LR"] is False
        assert (v["ADAPTIVE_LR_MIN"], v["ADAPTIVE_LR_MAX"]) == ("1e-4", "4e-4")
        assert (v["SAVE_EVERY_N_EPOCHS"], v["SEED"], v["OPTIMIZER_TYPE"], v["GRADIENT_ACCUMULATION"],
                v["MAX_GRAD_NORM"]) == (1, 42, "adamw", 1, 1.0)
        assert (v["NETWORK_TYPE"], v["FAMILY_PRECISION"], v["BLOCKS_SWAP"]) == (
            "LoRA (standard)", "Auto (fits your free VRAM)", "Auto (detect from GPU)")
        for k in ("KREA2_LOSS_WATCH", "KREA2_PER_IMAGE_LR", "KREA2_AUTO_RECAPTION", "KREA2_WARMUP_LOOK", "FAMILY_EDIT",
                  "FAMILY_FAST_ID"):
            assert v[k] is False, (name, k)
        assert v["FAMILY_SLIDER_GUIDANCE"] == "3"
        assert ("FAMILY_FT_ROTATIONS" in v) == w[6] and v.get("FAMILY_FT_ROTATIONS", "10") == "10"


def test_presets_apply_the_fine_tune_one_included(desc):
    for name, values in desc.presets:
        migrated, notes, ignored = presets.migrate_legacy(values)
        assert notes == [] and ignored == []
        new, rep = presets.apply(values, P.defaults(), desc)
        assert rep.blocked == [] and rep.refused == [] and rep.ignored == [], (name, rep.refused, rep.ignored)
        assert new["DATASET_MEGAPIXELS"] == values["DATASET_MEGAPIXELS"] and new["OPTIMIZER_TYPE"] == "adamw"
    tp = presets.TrainingPresets(desc)
    assert tp.default_name == "◆ Balanced (rank 32, 1e-4)"
    ft, rep = presets.apply(dict(desc.presets)["◆ Full fine-tune (1e-5)"], P.defaults(), desc)
    assert desc.finetune and ft["FAMILY_FT"] is True and ft["FAMILY_FT_ROTATIONS"] == 10 and ft["LEARNING_RATE"] == 1e-5


def test_fine_tune_spec_is_fizgigs(desc):
    """Fizgig 8.7.3 moss_image/driver.py:175-190: six parts per block, attention only with Train attention only, and
    ComfyUI's fused q / k / v written back as thirds of proj.qkv."""
    from training.ft import _file_loc
    drv = desc.load_driver()
    spec = drv.ft_spec(None)
    assert spec.blocks == "layers" and spec.components == ("proj.q", "proj.k", "proj.v", "proj.o", "proj.ffn_up_gate",
                                                           "proj.ffn_down")
    assert (spec.overhead_gb, spec.calib_mp, spec.act_gb_per_mp) == (4.5, 1.0, 2.0)
    have = {"layers.0.proj.qkv.weight", "layers.0.proj.o.weight"}
    assert [_file_loc(spec, f"layers.0.proj.{p}.weight", have) for p in "qkv"] == [
        ("layers.0.proj.qkv.weight", i, 3) for i in range(3)]
    assert _file_loc(spec, "layers.0.proj.o.weight", have) == ("layers.0.proj.o.weight", 0, 1)
    drv.configure(lora_sft=True)
    sft = drv.ft_spec(None)
    assert sft.components == ("proj.q", "proj.k", "proj.v", "proj.o") and sft.overhead_gb == 6.0


def test_family_options_route_to_the_driver(desc):
    assert desc.family_options == ("COMPILE_BLOCKS", "MOSS_LORA_SFT", "MOSS_LOWNOISE_PCT")
    assert "COMPILE_BLOCKS" in P.DRIVER_OPTIONS
    for key in ("MOSS_LORA_SFT", "MOSS_LOWNOISE_PCT"):
        assert P.DRIVER_OPTIONS[key] == {"MOSS_LORA_SFT": "lora_sft", "MOSS_LOWNOISE_PCT": "lownoise_pct"}[key]
        assert P.family_shows(P.BY_KEY[key], desc) and not P.family_shows(P.BY_KEY[key], registry.get("klein9b"))
    # the form's own values: the tick as a bool, the share as typed (empty = the model's own schedule)
    opts = pipeline.driver_options(desc, P.defaults())
    assert opts["lora_sft"] is False and opts["lownoise_pct"] == "30"
    assert pipeline.driver_options(desc, {**P.defaults(), "MOSS_LOWNOISE_PCT": ""})["lownoise_pct"] == ""
    opts = pipeline.driver_options(desc, {**P.defaults(), "MOSS_LORA_SFT": True, "MOSS_LOWNOISE_PCT": 30.0})
    assert set(opts) <= {"compile_blocks", "lora_sft", "lownoise_pct"}


def test_moss_parameters_exist_with_fizgigs_defaults():
    assert P.BY_KEY["MOSS_LORA_SFT"].default is False and P.BY_KEY["MOSS_LORA_SFT"].label == "Train attention only"
    assert float(P.BY_KEY["MOSS_LOWNOISE_PCT"].default) == 30.0


# ---- the DiT on a tiny config -----------------------------------------------------------------------------
def test_dit_forward_shape_skip_layers_mask_and_checkpointing():
    dit = _dit()
    x = _latents().to(torch.bfloat16)
    cond = _cond()["hidden_states"][None]
    t = torch.tensor([500.0])
    out = dit(x, t, cond)
    assert out.shape == (1, 128, 2, 3) and out.dtype == torch.bfloat16
    assert not torch.equal(out, dit(x, t, cond, skip_layers=(1,)))            # the SLG branch leaves a block out
    # a batch with a padding mask equals the per-sample forwards (padding is dropped, as the reference does)
    c2 = _cond(length=3, seed=3)["hidden_states"]
    padded = torch.zeros(2, 5, 24, dtype=torch.bfloat16)
    padded[0] = cond[0]
    padded[1, :3] = c2
    mask = torch.tensor([[True] * 5, [True] * 3 + [False] * 2])
    both = dit(torch.cat([x, x]), torch.tensor([500.0, 500.0]), padded, encoder_attention_mask=mask)
    assert torch.equal(both[0], out[0]) and torch.equal(both[1], dit(x, t, c2[None])[0])
    with pytest.raises(ValueError, match="valid text token"):
        dit(x, t, padded, encoder_attention_mask=torch.zeros(2, 5, dtype=torch.bool))
    dit.enable_gradient_checkpointing(True)
    with torch.enable_grad():
        xg = x.clone().requires_grad_(True)
        ck = dit(xg, t, cond)
        assert torch.allclose(ck.float(), out.float(), atol=1e-2)
        ck.float().sum().backward()
        assert xg.grad is not None
    with pytest.raises(ValueError):
        MOSSImageDiT(dim=30, num_heads=4)


def test_block_swap_on_cpu_keeps_the_output(driver):
    dit = _dit()
    x, cond, t = _latents().to(torch.bfloat16), _cond()["hidden_states"][None], torch.tensor([300.0])
    ref = dit(x, t, cond)
    assert driver.max_blocks_to_swap(dit) == 2 and driver.max_blocks_to_swap() == N_BLOCKS - 2
    driver.enable_block_swap(dit, 2, "cpu", supports_backward=False)
    driver.block_swap_mode(dit, inference=True)
    assert dit.blocks_to_swap == 2 and torch.equal(dit(x, t, cond), ref)
    with pytest.raises(ValueError, match="block swap"):
        dit.enable_block_swap(3, "cpu")
    dit.disable_block_swap()
    assert dit.offloader is None and dit.blocks_to_swap == 0


# ---- the timestep law (report Appendix A.2) and the Clean-end share --------------------------------------
def _share_below_half(mu, n=20000, seed=0):
    g = torch.Generator().manual_seed(seed)
    return sum(MOSSImageDriver._sample_t(g, 0.0, 1.0, mu) < 0.5 for _ in range(n)) / n


def test_reference_mu_gives_about_eleven_percent_clean_end_and_the_share_moves_mu():
    assert abs(_share_below_half(T_MU) - 0.11) < 0.015
    for pct in (30, 60):
        mu = mu_for_clean_share(pct)
        want = (pct / 100 - 0.005) / 0.99
        assert abs(_t_cdf(0.5 * T_CUTOFF, mu) / _t_cdf(T_CUTOFF, mu) - want) < 1e-6
        assert abs(_share_below_half(mu) - pct / 100) < 0.015
    assert mu_for_clean_share(30) > T_MU > -2.0
    assert abs(mu_for_clean_share(11.0) - T_MU) < 0.1


def test_sample_t_is_on_the_grid_truncated_and_scaled_into_the_window():
    g = torch.Generator().manual_seed(1)
    draws = [MOSSImageDriver._sample_t(g, 0.0, 1.0) for _ in range(5000)]
    assert all(0 < d <= 1.0 and abs(d * 1000 - round(d * 1000)) < 1e-6 for d in draws)
    assert max(draws) > 0.95 or sum(d > 0.95 for d in draws) <= 0.02 * len(draws)   # only the 1% uniform exceeds the cutoff
    g = torch.Generator().manual_seed(1)
    windowed = [MOSSImageDriver._sample_t(g, 0.2, 0.6) for _ in range(5000)]
    assert all(abs(w - (0.2 + 0.4 * d)) < 1e-9 for w, d in zip(windowed, draws))


def test_configure_sets_fizgigs_option_strings(driver):
    assert driver._t_mu() == T_MU and not driver.attention_only
    driver.configure(lora_sft=True, lownoise_pct=30.0, compile_blocks="Auto")
    assert driver.options == {"lora_sft": "1", "lownoise_pct": "30"} and driver.attention_only
    assert driver._t_mu() == pytest.approx(mu_for_clean_share(30))
    driver.configure(lownoise_pct="")
    assert driver._t_mu() == T_MU and driver.options["lownoise_pct"] == ""
    driver.configure(lora_sft=False, lownoise_pct="45%")
    assert driver.options == {"lora_sft": "", "lownoise_pct": "45"}
    for bad in (0, 100, -3):
        with pytest.raises(ValueError, match="strictly between"):
            driver.configure(lownoise_pct=bad)
    with pytest.raises(ValueError, match="unknown MOSS-Image option"):
        driver.configure(photo_blocks="1")
    d2 = MOSSImageDriver()
    d2.set_options({"lora_sft": "1", "lownoise_pct": "30"})          # Fizgig's --family_option path
    assert d2.attention_only and d2._t_mu() == pytest.approx(mu_for_clean_share(30))


# ---- the objective ----------------------------------------------------------------------------------------
class _Recording(torch.nn.Module):
    """Stands in for the DiT: records what it was told and returns a fixed velocity."""

    def __init__(self, v):
        super().__init__()
        self.v = torch.nn.Parameter(v, requires_grad=False)
        self.calls = []

    def forward(self, x, t, text, skip_layers=()):
        self.calls.append((x.clone(), float(t[0]), tuple(skip_layers)))
        return self.v.expand_as(x)


def test_training_loss_is_mse_to_noise_minus_x0_with_the_dit_told_a_thousand_sigma(driver):
    lat = _latents()
    dit = _Recording(torch.full((1, 128, 1, 1), 0.3))
    loss, info = driver.training_loss(dit, lat, _cond(), torch.Generator().manual_seed(7))
    g = torch.Generator().manual_seed(7)
    noise = torch.randn(lat.shape, generator=g)
    t = MOSSImageDriver._sample_t(g, 0.0, 1.0, T_MU)
    assert info["t"] == t and dit.calls[0][1] == pytest.approx(t * 1000)
    assert torch.allclose(dit.calls[0][0], (1 - t) * lat + t * noise)
    want = torch.nn.functional.mse_loss(torch.full_like(lat, 0.3), noise - lat)
    assert loss.item() == pytest.approx(want.item(), rel=1e-5)
    with pytest.raises(RuntimeError, match="no edit"):
        driver.training_loss(dit, lat, _cond(), torch.Generator().manual_seed(7), refs=[lat])
    # image-pair sliders: a tilted difference weighting changes the loss, an identical pair does not
    other = lat + torch.randn_like(lat) * torch.linspace(0, 2, 3)[None, None, None, :]
    weighted, _ = driver.training_loss(dit, lat, _cond(), torch.Generator().manual_seed(7), diff_ref=other, diff_weight=0.8)
    same, _ = driver.training_loss(dit, lat, _cond(), torch.Generator().manual_seed(7), diff_ref=lat, diff_weight=0.8)
    assert weighted.item() != pytest.approx(loss.item()) and same.item() == pytest.approx(loss.item(), rel=1e-5)
    state = driver.noise_latents(lat, torch.Generator().manual_seed(7))
    assert torch.allclose(state["xt"], dit.calls[0][0]) and state["t"] == t
    assert driver.predict(dit, state, _cond()).shape == lat.shape


def test_training_loss_runs_through_the_tiny_dit_with_grads(driver):
    dit = _dit()
    net = FamilyLoRA(dit, driver)
    net.add_trainable(4, 4)
    loss, info = driver.training_loss(dit, _latents(), _cond(), torch.Generator().manual_seed(3))
    loss.backward()
    assert math.isfinite(loss.item()) and 0 < info["t"] <= 1
    assert all(p.grad is not None for p in net.parameters())


# ---- LoRA: what is trained, and the keys ------------------------------------------------------------------
def test_lora_targets_are_the_blocks_only_and_attention_only_on_sft(driver):
    dit = _dit()
    full = driver.lora_target_names(dit)
    assert full == [f"layers.{i}.{m}" for i in range(4) for m in BLOCK_MODULES]
    assert not any(n.startswith("text_embedder") for n in full)
    groups = driver.block_map(dit)
    assert [g.label for g in groups] == ["Blocks", "Text adapter"]
    assert [b.id for b in groups[1].blocks] == ["text_0", "text_1"] and len(groups[1].blocks[0].modules) == 5
    assert [b.id for b in driver.block_map()[0].blocks][:3] == ["block_0", "block_1", "block_2"]
    assert len(driver.block_map()[0].blocks) == N_BLOCKS
    assert driver.block_of("layers.22.proj.q") == "block_22" and driver.block_of("text_embedder.blocks.0.qkv") == "text_0"
    driver.configure(lora_sft=True)
    attn = driver.lora_target_names(dit)
    assert attn == [f"layers.{i}.{m}" for i in range(4) for m in ATTN_MODULES]
    assert driver.quant_target_names(dit) == full                      # every block Linear is quantised regardless


def test_lora_keys_are_comfyuis_kohya_names_and_lokr_uses_diffusion_model_stems(driver, tmp_path):
    dit = _dit()
    net = FamilyLoRA(dit, driver)
    net.add_trainable(4, 8)
    sd = net.state_dict()
    stems = {k.rsplit(".", 2)[0] for k in sd if k.endswith(".lora_down.weight")}
    assert stems == {f"lora_unet_layers_{i}_{m.replace('.', '_')}" for i in range(4) for m in BLOCK_MODULES}
    assert sd["lora_unet_layers_0_proj_ffn_up_gate.alpha"].item() == 8.0
    assert sd["lora_unet_layers_1_proj_q.lora_down.weight"].shape == (4, 32)
    assert sd["lora_unet_layers_1_proj_ffn_down.lora_up.weight"].shape == (32, 4)
    assert desc_key_round_trip(driver, net, tmp_path)
    lokr = FamilyLoRA(_dit(), driver)
    lokr.add_trainable(4, 4, kind="lokr", factor=4)
    assert any(k.startswith("diffusion_model.layers.0.proj.q.") for k in lokr.state_dict())
    assert not any(k.startswith("lora_unet_") for k in lokr.state_dict())


def desc_key_round_trip(driver, net, tmp_path):
    path = tmp_path / "moss.safetensors"
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    net.save(str(path))
    net2 = FamilyLoRA(_dit(), driver)
    net2.add_trainable(4, 8)
    assert net2.load_trainable(str(path)) == 24
    return all(torch.equal(net2.state_dict()[k], v) for k, v in net.state_dict().items())


# ---- the loader: ComfyUI's fused file, the diffusers prefix, int8 ConvRot -------------------------------------
def _fused_comfy_state_dict(dit, prefix=""):
    sd = {}
    for k, v in dit.state_dict().items():
        if k.endswith(".proj.q.weight"):
            base = k[: -len("q.weight")]
            sd[prefix + base + "qkv.weight"] = torch.cat([dit.state_dict()[base + f"{n}.weight"] for n in "qkv"])
        elif not (k.endswith(".proj.k.weight") or k.endswith(".proj.v.weight")):
            sd[prefix + k] = v
    return sd


def test_loader_splits_comfyuis_fused_qkv_and_strips_the_prefix(tmp_path):
    from safetensors.torch import save_file
    dit = _dit(seed=5)
    sd = _fused_comfy_state_dict(dit, prefix="model.diffusion_model.")
    sd["model.diffusion_model.rope.bands"] = torch.zeros(2)
    sd["model.diffusion_model.layers.0.comfy_attention.config"] = torch.zeros(1, dtype=torch.uint8)
    path = tmp_path / "moss_image_base_bf16.safetensors"
    save_file({k: v.contiguous() for k, v in sd.items()}, str(path))
    loaded = load_moss_dit(str(path), device="cpu", config=TINY)
    for k, v in dit.state_dict().items():
        assert torch.equal(loaded.state_dict()[k], v), k
    x, c, t = _latents().to(torch.bfloat16), _cond()["hidden_states"][None], torch.tensor([100.0])
    assert torch.equal(loaded(x, t, c), dit(x, t, c))
    bad = dict(sd)
    bad.pop("model.diffusion_model.final_linear_video.weight")
    save_file({k: v.contiguous() for k, v in bad.items()}, str(tmp_path / "bad.safetensors"))
    with pytest.raises(ValueError, match="keys mismatch"):
        load_moss_dit(str(tmp_path / "bad.safetensors"), device="cpu", config=TINY)


def test_loader_takes_diffusers_shards_and_decodes_int8_convrot(tmp_path):
    from safetensors.torch import save_file
    dit = _dit(seed=6)
    sd = dict(dit.state_dict())
    items = list(sd.items())
    save_file({k: v.contiguous() for k, v in items[: len(items) // 2]},
              str(tmp_path / "diffusion_pytorch_model-00001-of-00002.safetensors"))
    save_file({k: v.contiguous() for k, v in items[len(items) // 2:]},
              str(tmp_path / "diffusion_pytorch_model-00002-of-00002.safetensors"))
    got = read_state_dict(str(tmp_path / "diffusion_pytorch_model-00002-of-00002.safetensors"))
    assert set(got) == set(sd) and all(torch.equal(got[k], v) for k, v in sd.items())
    assert set(read_state_dict(str(tmp_path))) == set(sd)                   # a folder of shards
    # ComfyUI's int8 ConvRot repack of one block Linear: int8 codes of rotate(W, G) with a per-row scale and the marker
    name = "layers.1.proj.ffn_down.weight"
    w = sd[name].float()                                                   # (32, 48): 48 = 3 x 16, rot group 16
    rot = W.rotate(w, 16)
    scale = rot.abs().amax(dim=1, keepdim=True) / 127.0
    q = (rot / scale).round().clamp(-127, 127).to(torch.int8)
    conf = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 16}).encode()
    packed = {k: v for k, v in sd.items() if k != name}
    packed[name] = q
    packed["layers.1.proj.ffn_down.weight_scale"] = scale.reshape(-1)
    packed["layers.1.proj.ffn_down.comfy_quant"] = torch.frombuffer(bytearray(conf), dtype=torch.uint8)
    save_file({k: v.contiguous() for k, v in packed.items()}, str(tmp_path / "moss_image_sft_int8_convrot.safetensors"))
    dec = read_state_dict(str(tmp_path / "moss_image_sft_int8_convrot.safetensors"))
    assert dec[name].dtype == torch.bfloat16 and "layers.1.proj.ffn_down.comfy_quant" not in dec
    assert torch.allclose(dec[name].float(), w, atol=0.05, rtol=0.05)
    assert (dec[name].float() - w).abs().mean() < 0.01
    loaded = load_moss_dit(str(tmp_path / "moss_image_sft_int8_convrot.safetensors"), device="cpu", config=TINY)
    assert loaded.layers[1].proj["ffn_down"].weight.shape == (32, 48)


# ---- the sampler ------------------------------------------------------------------------------------------
A = 2.0


class _ExactDenoiser(torch.nn.Module):
    """v = noise - x0, with x0 the exact posterior mean of N(0, A^2) data at this sigma (Fizgig checks/test_moss_sampler.py)."""

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, t, text, skip_layers=()):
        s = t / 1000.0
        x0 = x * (1 - s) * A * A / ((1 - s) ** 2 * A * A + s ** 2)
        return (x - x0) / s


def _run(steps, sampler, noise, scheduler="reference"):
    return S.sample(_ExactDenoiser(), torch.zeros(4, 24), 128, 128, steps=steps, seed=0, cfg=1.0, noise=noise,
                    sampler=sampler, scheduler=scheduler)


@pytest.fixture(scope="module")
def ode():
    noise = torch.randn(1, 128, 8, 8, generator=torch.Generator().manual_seed(0))
    return noise, _run(1000, "euler", noise)


@pytest.mark.parametrize("steps", [10, 20, 30, 50])
def test_res_multistep_beats_euler(ode, steps):
    noise, ref = ode
    err = lambda x: ((x - ref).norm() / ref.norm()).item()                 # noqa: E731
    assert err(_run(steps, "res_multistep", noise)) < err(_run(steps, "euler", noise))
    assert err(_run(steps, "res_multistep", noise, "simple")) < 0.2


def test_schedules_and_the_slg_window():
    s = S.schedule(30)
    assert float(s[-1]) == 0.0 and float(s[0]) <= 1.0 and len(s) <= 31
    assert all(float(a) > float(b) for a, b in zip(s[:-1], s[1:]))
    grid = torch.arange(1, 1001) / 1000.0
    assert all(any(abs(float(v) - float(gv)) < 1e-6 for gv in grid) for v in s[:-1])    # snapped to j / 1000
    c = S.comfy_simple_schedule(30)
    assert len(c) == 31 and float(c[-1]) == 0.0 and float(c[0]) == pytest.approx(1.0)
    assert S.slg_sigma_end(0.5) == pytest.approx(5 * 0.5 / (1 + 4 * 0.5)) and S.slg_sigma_end(1.0) == 0.0
    assert S.slg_sigma_end(0.0) == 1.0
    assert S.latent_hw(1024, 768) == (64, 48)
    with pytest.raises(ValueError):
        S.latent_hw(1000, 1024)
    n = S.initial_noise(3, 64, 32)
    assert n.shape == (1, 128, 4, 2) and n.dtype == torch.float32 and torch.equal(n, S.initial_noise(3, 64, 32))


def test_generate_uses_the_default_recipe_and_skips_block_22_for_the_first_half(driver):
    dit = _Recording(torch.zeros(1, 128, 1, 1))
    out = driver.generate(dit, _cond(), 48, 32, steps=6, seed=1, cfg=4.0, neg_cond=_cond(seed=9))
    assert out.shape == (1, 128, 2, 3)
    pos = [c for i, c in enumerate(dit.calls) if i % 2 == 0]
    neg = [c for i, c in enumerate(dit.calls) if i % 2 == 1]
    assert len(pos) == len(neg) == 6 and all(c[2] == () for c in pos)
    end = S.slg_sigma_end(0.5)
    assert all((c[2] == (22,)) == (c[1] / 1000 >= end) for c in neg)     # SLG for the first half (ComfyUI's end 0.5)
    assert any(c[2] == (22,) for c in neg) and any(c[2] == () for c in neg)
    with pytest.raises(ValueError, match="encoded negative"):
        driver.generate(dit, _cond(), 48, 32, steps=2, seed=1, cfg=4.0)
    dit.calls.clear()
    driver.generate(dit, _cond(), 48, 32, steps=4, seed=1, cfg=1.0, options=(("uncond_skip", ()), ("sampler", "euler")))
    assert len(dit.calls) == 4                                             # no negative branch at CFG 1
    # a Turbo LoRA's own schedule applies at its step count only
    dit.calls.clear()
    driver.generate(dit, _cond(), 48, 32, steps=3, seed=1, sigmas=(0.9, 0.5, 0.2), options=(("uncond_skip", ()),))
    assert [c[1] for c in dit.calls] == pytest.approx([900.0, 500.0, 200.0])
    padded = driver.pad_conditioning([_cond(length=3), _cond(length=5)])
    assert padded[0]["hidden_states"].shape == (5, 24) and padded[0]["mask"].tolist() == [True] * 3 + [False] * 2
    assert driver._real_tokens(padded[0]).shape == (3, 24)
    assert driver.initial_noise(5, 48, 32).shape == (1, 128, 2, 3)


def test_tiny_dit_generates_and_decodes_the_right_shapes(driver):
    dit = _dit()
    lat = driver.generate(dit, _cond(), 32, 32, steps=2, seed=0, cfg=1.0)
    assert lat.shape == (1, 128, 2, 2) and torch.isfinite(lat).all()


# ---- the shared FLUX.2 autoencoder -----------------------------------------------------------------------
@pytest.fixture(scope="module")
def vae():
    torch.manual_seed(0)
    ae = V.AutoEncoder()
    with torch.no_grad():
        ae.bn.running_mean.normal_(0, 0.1)
        ae.bn.running_var.uniform_(0.5, 1.5)
    return ae.eval().requires_grad_(False)


def test_vae_tokens_ids_and_the_driver_encode_decode(driver, vae, monkeypatch):
    calls = []
    real = V.wide_attention
    monkeypatch.setattr(V, "wide_attention", lambda q, k, v: calls.append(q.shape) or real(q, k, v))
    img = (np.random.RandomState(0).rand(32, 48, 3) * 255).astype(np.uint8)
    (z,) = driver.encode_images(vae, [img])
    assert z.shape == (128, 2, 3) and z.dtype == torch.float32
    assert calls and calls[0][-1] == 512 and calls[0][1] == 1         # one 512-wide head through wide_attention
    tokens, ids = vae.encode(torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1)
    assert tokens.shape == (6, 128) and torch.equal(ids, V.latent_ids(2, 3))
    assert torch.equal(tokens.reshape(2, 3, 128).permute(2, 0, 1), z)
    pic = driver.decode(vae, z[None], 48, 32)
    assert pic.size == (48, 32) and pic.mode == "RGB"
    assert V.latent_shape(1024, 768) == (64, 48)
    with pytest.raises(ValueError):
        V.latent_shape(100, 64)


def test_vae_loader_takes_the_reference_names_and_the_small_decoder(vae, tmp_path):
    from safetensors.torch import save_file
    full = tmp_path / "flux2-vae.safetensors"
    save_file({k: v.contiguous() for k, v in vae.state_dict().items()}, str(full))
    ae = V.load_flux2_vae(str(full))
    assert ae.decoder.conv_in.out_channels == 512
    assert all(torch.equal(ae.state_dict()[k], v) for k, v in vae.state_dict().items())
    torch.manual_seed(1)
    small = V.AutoEncoder(decoder_ch=96)
    assert small.decoder.conv_out.in_channels == 96
    sd = {"vae." + k: v.contiguous() for k, v in small.state_dict().items()}
    save_file(sd, str(tmp_path / "flux2-vae_small.safetensors"))
    ae2 = V.load_flux2_vae(str(tmp_path / "flux2-vae_small.safetensors"))
    assert ae2.decoder.conv_out.in_channels == 96
    bad = {k: v for k, v in vae.state_dict().items() if k != "encoder.conv_in.weight"}
    save_file({k: v.contiguous() for k, v in bad.items()}, str(tmp_path / "bad.safetensors"))
    with pytest.raises(KeyError, match="not the FLUX.2"):
        V.load_flux2_vae(str(tmp_path / "bad.safetensors"))


def test_diffusers_vae_names_map_onto_the_reference():
    sd = {"quant_conv.weight": torch.zeros(64, 64, 1, 1), "post_quant_conv.bias": torch.zeros(32),
          "encoder.down_blocks.1.resnets.0.conv_shortcut.weight": torch.zeros(1),
          "encoder.down_blocks.0.downsamplers.0.conv.weight": torch.zeros(1),
          "decoder.up_blocks.0.resnets.2.norm1.weight": torch.zeros(1),
          "decoder.up_blocks.3.upsamplers.0.conv.weight": torch.zeros(1),
          "decoder.mid_block.attentions.0.to_q.weight": torch.zeros(512, 512),
          "decoder.mid_block.attentions.0.group_norm.weight": torch.zeros(512),
          "decoder.mid_block.resnets.1.conv1.weight": torch.zeros(1), "decoder.conv_norm_out.weight": torch.zeros(1)}
    out = V._from_diffusers(sd)
    assert set(out) == {"encoder.quant_conv.weight", "decoder.post_quant_conv.bias",
                        "encoder.down.1.block.0.nin_shortcut.weight", "encoder.down.0.downsample.conv.weight",
                        "decoder.up.3.block.2.norm1.weight", "decoder.up.0.upsample.conv.weight",
                        "decoder.mid.attn_1.q.weight", "decoder.mid.attn_1.norm.weight",
                        "decoder.mid.block_2.conv1.weight", "decoder.norm_out.weight"}
    assert out["decoder.mid.attn_1.q.weight"].shape == (512, 512, 1, 1)


# ---- the Qwen3.5 text encoder on a tiny config -----------------------------------------------------------
TINY_QWEN = dict(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=4, num_attention_heads=2,
                 num_key_value_heads=1, head_dim=16, rms_norm_eps=1e-6, rope_theta=10000.0, partial_rotary_factor=0.25,
                 full_attention_interval=4, linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_num_key_heads=2,
                 linear_num_value_heads=4, linear_value_head_dim=8)


class _Tok:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert add_generation_prompt and enable_thinking and messages[0]["role"] == "user"
        return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n<think>\n"

    def __call__(self, text, return_tensors, truncation, max_length):
        ids = [(ord(ch) % 60) + 1 for ch in text][:max_length]
        return {"input_ids": torch.tensor([ids])}


def _tiny_text_file(tmp_path, prefix="model.language_model.", extra=True):
    from safetensors.torch import save_file
    from training.families.moss_image.embedder import Qwen35TextModel
    torch.manual_seed(0)
    m = Qwen35TextModel(TINY_QWEN).to(torch.bfloat16)
    sd = {prefix + k: v.contiguous() for k, v in m.state_dict().items()}
    if extra:
        sd["model.visual.blocks.0.attn.qkv.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)
        sd["lm_head.weight"] = torch.zeros(64, 32, dtype=torch.bfloat16)
    path = tmp_path / "qwen3.5_9b_bf16.safetensors"
    save_file(sd, str(path))
    return path, m


def test_text_encoder_layout_streamed_equals_resident_and_reads_every_layout(tmp_path):
    from training.families.moss_image.embedder import MOSSTextEncoder, read_text_state_dict
    path, model = _tiny_text_file(tmp_path)
    sd = read_text_state_dict(str(path))
    assert all(k.startswith("language_model.") for k in sd) and not any("visual" in k or "lm_head" in k for k in sd)
    te = MOSSTextEncoder(str(path), device="cpu", tokenizer=_Tok(), cfg=TINY_QWEN)
    (h,) = te.encode(["a red fox"])
    assert h.dtype == torch.bfloat16 and h.shape[1] == 32
    assert h.shape[0] == len("<|im_start|>user\na red fox<|im_end|>\n<|im_start|>assistant\n<think>\n")   # unpadded
    streamed = MOSSTextEncoder(str(path), device="cpu", tokenizer=_Tok(), cfg=TINY_QWEN, stream=True)
    assert streamed.stream
    a, b = streamed.encode(["a red fox", "two cats on a wall"])
    assert torch.equal(a, h) and b.shape[0] > a.shape[0]
    int8 = MOSSTextEncoder(str(path), device="cpu", tokenizer=_Tok(), cfg=TINY_QWEN, int8=True, stream=True)
    assert not int8.stream and torch.allclose(int8.encode(["a red fox"])[0].float(), h.float(), atol=0.3, rtol=0.1)
    te.unload()
    assert te.model is None
    (tmp_path / "shards").mkdir()
    path2, _ = _tiny_text_file(tmp_path / "shards", prefix="language_model.", extra=False)
    assert set(read_text_state_dict(str(path2))) == set(sd)
    with pytest.raises(ValueError, match="not the Qwen3.5-9B"):
        MOSSTextEncoder(str(path), device="cpu", tokenizer=_Tok(), cfg={**TINY_QWEN, "num_hidden_layers": 8})


def test_encode_text_and_the_cache_batch_hint(driver, tmp_path):
    from training.families.moss_image.embedder import MOSSTextEncoder
    path, _ = _tiny_text_file(tmp_path)
    te = MOSSTextEncoder(str(path), device="cpu", tokenizer=_Tok(), cfg=TINY_QWEN)
    conds = driver.encode_text(te, ["one", "two words"])
    assert [set(c) for c in conds] == [{"hidden_states"}] * 2 and conds[0]["hidden_states"].dim() == 2
    assert driver.text_cache_batch_size == 4096 and not driver.supports_batching


# ---- SFT previews: the tick, the handoff, Turbo at zero -------------------------------------------------------
def test_sft_preview_tick_recipe_and_train_kwargs(desc, tmp_path):
    ck_file, ck = desc.preview_checkpoint()
    assert ck_file.pref_key == "moss_image_sft_dit" and desc.train_preview_checkpoint and desc.preview_checkpoint_on
    assert (ck.steps, ck.cfg, ck.sampler, dict(ck.options)["slg_end"], dict(ck.options)["uncond_skip"]) == (
        30, 4.0, "res_multistep", 0.5, (22,))
    assert ck.options == desc.default_sampling().options
    assert "SFT" in P.preview_checkpoint_tick(desc) and "SFT" in desc.preview_checkpoint_steps_note
    assert desc.standard_base == ("layers.41.norm2.weight", 3658.7021484375)
    for key in ("SAMPLE_USE_DISTILLED", "CACHE_SAMPLE_MODEL", "PREVIEW_INT8"):
        assert P.family_shows(P.BY_KEY[key], desc)
    sft = tmp_path / "moss_image_sft_int8_convrot.safetensors"
    sft.write_bytes(b"x")
    vals = {**P.defaults(), "SAMPLE_ENABLED": True, "SAMPLE_EVERY_N_EPOCHS": 1, "SAMPLE_PROMPT": "a cat"}
    kw, _ = pipeline.train_kwargs(desc, vals, tmp_path / "run", {"moss_image_sft_dit": str(sft)})
    assert kw["preview_checkpoint"] == str(sft)
    kw, _ = pipeline.train_kwargs(desc, {**vals, "SAMPLE_USE_DISTILLED": False}, tmp_path / "run",
                                  {"moss_image_sft_dit": str(sft)})
    assert "preview_checkpoint" not in kw


def test_turbo_previews_start_at_zero_and_load_unmerged(desc):
    sp = desc.preview_speed()
    assert sp is not None and sp.name == "MOSS-Image Turbo LoRA" and sp.load_unmerged and sp.strength == 1.0
    assert sp.pref_key == "moss_image_turbo_lora" and (sp.settings.steps, sp.settings.cfg) == (8, 1.0)
    assert dict(sp.settings.options)["uncond_skip"] == ()
    assert desc.preview_speed_defaults() == (8, 0.0)                     # off until the Samples tab raises it
    assert pipeline.turbo_strength(desc, P.defaults()) == 0.0
    assert pipeline.turbo_strength(desc, {**P.defaults(), "FAMILY_TURBO_STRENGTH": 1.0}) == 1.0
    assert P.family_shows(P.BY_KEY["FAMILY_TURBO_STRENGTH"], desc)


def test_preview_handoff_parks_and_restores_the_training_model(driver, monkeypatch, tmp_path):
    calls = []

    class _Dit:
        blocks_to_swap, _nf4_quantized = 0, False
        layers = list(range(42))

        def enable_block_swap(self, n, device, backward):
            calls.append(("swap", n, backward))

        def __getattr__(self, name):
            return lambda *a, **k: calls.append((name,))
    dit = _Dit()
    token = driver.park_for_preview(dit, "cpu")
    assert token == 0 and calls[0] == ("swap", 40, True) and ("prepare_block_swap_before_forward",) in calls
    monkeypatch.setattr("training.quant.move", lambda m, d: calls.append(("move", str(d))))
    driver.unpark_after_preview(dit, "cpu", token)
    assert ("disable_block_swap",) in calls and ("move", "cpu") in calls and ("switch_block_swap_for_training",) in calls
    calls.clear()
    dit.blocks_to_swap = 5
    assert driver.park_for_preview(dit, "cpu") == 5
    driver.unpark_after_preview(dit, "cpu", 5)
    assert ("swap", 5, True) in calls and ("move_to_device_except_swap_blocks",) in calls
    nf4 = _Dit()
    nf4._nf4_quantized = True
    assert driver.park_for_preview(nf4, "cpu") is None
    # the SFT load: bf16 with room (no CUDA = unlimited), INT8 when asked; a convrot file counts double on disk
    loads = []
    monkeypatch.setattr("training.quant.load_base",
                        lambda drv, path, device, precision, swap, supports_backward: loads.append(
                            (precision, swap, supports_backward)) or ("model-stub", swap))
    sft = tmp_path / "moss_image_sft_int8_convrot.safetensors"
    sft.write_bytes(b"0" * 1024)

    class _M(str):
        def eval(self):
            return self

        def requires_grad_(self, flag):
            return self
    monkeypatch.setattr("training.quant.load_base",
                        lambda drv, path, device, precision, swap, supports_backward: loads.append(
                            (precision, swap, supports_backward)) or (_M("m"), swap))
    assert driver.load_preview_checkpoint(str(sft), "cpu") == ("m", 0) and loads[-1] == ("bf16", 0, False)
    driver.load_preview_checkpoint(str(sft), "cpu", int8=True)
    assert loads[-1] == ("int8", 0, False)


# ---- the Auto plan over Fizgig's measured figures ----------------------------------------------------------
def test_auto_plan_int8_then_nf4_then_swapped_int8(desc, driver, monkeypatch):
    monkeypatch.setattr(quant, "available", lambda p, device=None: (True, ""))
    assert quant.plan(desc, driver, "auto", -1, free_gb=24.0, megapixels=1.0)[:2] == ("int8", 0)
    assert quant.plan(desc, driver, "auto", -1, free_gb=12.0, megapixels=1.0)[:2] == ("int8", 0)
    assert quant.plan(desc, driver, "auto", -1, free_gb=10.0, megapixels=0.25)[:2] == ("nf4", 0)
    # NF4 with no swap, closely (0.5 GB to spare), goes before swapped INT8 blocks (Fizgig 8.9, #216)
    assert quant.plan(desc, driver, "auto", -1, free_gb=9.0, megapixels=1.0)[:2] == ("nf4", 0)
    p, n, _ = quant.plan(desc, driver, "auto", -1, free_gb=8.8, megapixels=1.0)
    assert p == "int8" and n == math.ceil((10.5 - 7.3) / 0.105) and n <= driver.max_blocks_to_swap()
    assert quant.plan(desc, driver, "bf16", -1, free_gb=14.0, megapixels=1.0)[:2] == ("bf16", math.ceil((15.1 - 12.5) / 0.211))
    assert quant.plan(desc, driver, "auto", -1, free_gb=40.0, megapixels=1.0)[0] != "bf16"   # bf16 stays a manual choice


# ---- the torch-free Base fingerprint ----------------------------------------------------------------------
def _write_safetensors(path, tensors, metadata=None):
    """A minimal safetensors writer (bf16 / f32) so the check is exercised without torch's writer."""
    header, blobs, offset = {}, [], 0
    for name, (dtype, arr) in tensors.items():
        raw = arr.tobytes()
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    if metadata:
        header["__metadata__"] = metadata
    h = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def _bf16(values):
    return (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def test_base_fingerprint_true_false_none_shards_and_finetune_marks(desc, tmp_path):
    norm = np.linspace(0.5, 1.5, 3072).astype(np.float32)
    norm = (norm.view(np.uint32) >> 16 << 16).view(np.float32)             # bf16-representable
    want = float(np.asarray(norm, dtype=np.float64).sum())

    class _D:
        standard_base = ("layers.41.norm2.weight", want)
    base = tmp_path / "base.safetensors"
    _write_safetensors(base, {"layers.41.norm2.weight": ("BF16", _bf16(norm)), "x": ("F32", np.ones(2, np.float32))})
    assert base_check.is_standard_base(_D, str(base)) is True
    tuned = tmp_path / "tuned.safetensors"
    _write_safetensors(tuned, {"layers.41.norm2.weight": ("BF16", _bf16(norm + 0.0625))})   # a fine-tune's norm
    assert base_check.is_standard_base(_D, str(tuned)) is False
    marked = tmp_path / "ft.safetensors"
    _write_safetensors(marked, {"layers.41.norm2.weight": ("BF16", _bf16(norm))}, {"tagscriber_finetune": "1"})
    assert base_check.is_standard_base(_D, str(marked)) is False
    prefixed = tmp_path / "comfy.safetensors"
    _write_safetensors(prefixed, {"model.diffusion_model.layers.41.norm2.weight": ("BF16", _bf16(norm))})
    assert base_check.is_standard_base(_D, str(prefixed)) is True
    other = tmp_path / "other.safetensors"
    _write_safetensors(other, {"layers.0.norm2.weight": ("BF16", _bf16(norm))})
    assert base_check.is_standard_base(_D, str(other)) is None
    assert base_check.is_standard_base(_D, str(tmp_path / "missing.safetensors")) is None
    assert base_check.is_standard_base(desc, "") is None and base_check.is_standard_base(registry.get("klein9b"), str(base)) is None
    shards = tmp_path / "shards"
    shards.mkdir()
    _write_safetensors(shards / "m-00001-of-00002.safetensors", {"a": ("F32", np.ones(2, np.float32))})
    _write_safetensors(shards / "m-00002-of-00002.safetensors", {"layers.41.norm2.weight": ("BF16", _bf16(norm))})
    assert base_check.is_standard_base(_D, str(shards / "m-00001-of-00002.safetensors")) is True
    assert base_check.is_standard_base(_D, str(shards)) is True
    # the real fingerprint against a bf16 vector built to sum to it: torch's double sum and ours agree
    key, real = desc.standard_base
    vec = torch.full((3072,), real / 3072).to(torch.bfloat16)
    vec[0] = vec[0] + (real - float(vec.double().sum()))
    rebuilt = tmp_path / "rebuilt.safetensors"
    _write_safetensors(rebuilt, {key: ("BF16", _bf16(vec.float().numpy()))})
    assert base_check.is_standard_base(desc, str(rebuilt)) is (abs(float(vec.double().sum()) - real) < 1e-6)
