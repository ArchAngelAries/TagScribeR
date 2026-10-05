"""Krea 2 capabilities restored from Fizgig: Automagic v3 with per-family groups, the fp8_scaled text encoder kept fp8, the
attention trim and the torch.compile rules. CPU only, tiny random models."""
import logging

import pytest

torch = pytest.importorskip("torch")

from training import params as P  # noqa: E402
from training import presets, registry  # noqa: E402
from training.families.krea2 import attention as A  # noqa: E402
from training.families.krea2 import compile as C  # noqa: E402
from training.families.krea2 import sampling as S  # noqa: E402
from training.families.krea2.driver import Krea2Driver  # noqa: E402
from training.families.krea2.model import SingleMMDiTConfig, SingleStreamDiT  # noqa: E402
from training.lora import FamilyLoRA  # noqa: E402

TINY = SingleMMDiTConfig(features=64, tdim=32, txtdim=32, heads=4, kvheads=2, multiplier=2, layers=3, patch=2,
                         channels=16, txtheads=2, txtkvheads=2, txtlayers=3)


@pytest.fixture
def driver():
    d = Krea2Driver()
    d.description = registry.get("krea2")
    return d


def _dit(dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    return SingleStreamDiT(TINY).to(dtype).requires_grad_(False)


def _cond(batch, text_len=20, valid=(7, 12)):
    hs = torch.randn(batch, text_len, 3, 32)
    mask = torch.zeros(batch, text_len, dtype=torch.bool)
    for i in range(batch):
        mask[i, :valid[i % len(valid)]] = True
        mask[i, text_len - 5:] = True
    return {"hidden_states": hs, "attention_mask": mask}


# ---- 1. Automagic v3 ----------------------------------------------------------------------------------------
def _fizgig_family(lora_name):
    """Fizgig's optimizers.family_of (KREA2_LORA_FAMILIES), spelled out: txtfusion wins over attn / mlp."""
    if "txtfusion" in lora_name:
        return "txtfusion"
    if "_attn_" in lora_name:
        return "attn"
    if "_mlp_" in lora_name:
        return "mlp"
    return "io"


def test_automagic_is_offered_and_groups_match_fizgigs_family_split(driver):
    assert "automagic3" in registry.get("krea2").optimizers
    dit = _dit()
    net = FamilyLoRA(dit, driver)
    net.add_trainable(4, 4)
    params, args, counts = driver.optimizer_params(net, "automagic3", 1e-6, "")
    assert args == "polarity_history=16"
    # 3 blocks: attn 3 x 5, mlp 3 x 3, txtfusion 4 blocks x 8 + projector, io 7
    assert counts == {"txtfusion": 33, "attn": 15, "mlp": 9, "io": 7}
    assert [g["family"] for g in params] == ["txtfusion", "attn", "mlp", "io"]
    want = {}
    for name in driver.lora_target_names(_dit()):
        fam = _fizgig_family("lora_unet_" + name.replace(".", "_"))
        want[fam] = want.get(fam, 0) + 1
    assert counts == want
    assert sum(len(g["params"]) for g in params) == len(net.parameters())      # every trainable tensor, once
    assert all(g["lr"] == 1e-6 for g in params)
    # the full-size block map splits 264 modules the same way Fizgig's note says (28 blocks)
    full = {}
    for b in [b for g in driver.block_map() for b in g.blocks]:
        for m in b.modules:
            f = _fizgig_family("lora_unet_" + m.replace(".", "_"))
            full[f] = full.get(f, 0) + 1
    assert full == {"txtfusion": 33, "attn": 140, "mlp": 84, "io": 7} and sum(full.values()) == 264


def test_automagic_args_and_other_optimizers_are_untouched(driver):
    net = FamilyLoRA(_dit(), driver)
    net.add_trainable(4, 4)
    assert driver.optimizer_params(net, "automagic3", 1e-6, "polarity_history=8 fused=False")[1] == \
        "polarity_history=8 fused=False"                                     # an explicit window wins
    assert driver.optimizer_params(net, "automagic3", 1e-6, "weight_decay=0.01")[1] == \
        "weight_decay=0.01 polarity_history=16"
    flat, args, counts = driver.optimizer_params(net, "adamw", 1e-4, "x=1")
    assert args == "x=1" and counts == {} and len(flat) == len(net.parameters())


def test_automagic_short_run_trains_and_owns_the_rate(driver, caplog):
    from training.optimizers import create_optimizer, group_rates, owns_its_rate
    dit = _dit()
    net = FamilyLoRA(dit, driver)
    net.add_trainable(4, 4)
    params, args, counts = driver.optimizer_params(net, "automagic3", 1e-4, "")
    opt, label = create_optimizer("automagic3", params, 1e-4, args)
    assert owns_its_rate(opt) and len(opt.param_groups) == 4 and label.startswith("automagic3")
    assert all(g["polarity_history"] == 16 for g in opt.param_groups)
    before = [p.detach().clone() for p in net.parameters()]
    cond, lat = _cond(1), torch.randn(1, 16, 8, 8)
    for step in range(4):
        loss, _ = driver.training_loss(dit, lat, cond, torch.Generator().manual_seed(step))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        assert torch.isfinite(loss)
    assert any(not torch.equal(a, b) for a, b in zip(before, net.parameters()))
    rates = group_rates(opt)
    assert all(name in rates for name in ("txtfusion", "attn", "mlp", "io"))
    sd = opt.state_dict()                                                    # resumes: state saves and loads
    opt2, _ = create_optimizer("automagic3", driver.optimizer_params(net, "automagic3", 1e-4, "")[0], 1e-4, args)
    opt2.load_state_dict(sd)
    assert group_rates(opt2) == rates


# ---- 2. the fp8_scaled text encoder stays fp8 ------------------------------------------------------------------
QWEN_TINY = {
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
                      "deepstack_visual_indexes": [0, 1]}}


class _Tok:
    """Stand-in tokenizer: template prefix 34 tokens, suffix 5, one token per prompt character."""
    def __call__(self, text, truncation=False, return_length=False, return_overflowing_tokens=False, padding=False,
                 max_length=None, return_tensors=None):
        from transformers import BatchEncoding

        from training.families.krea2 import embedder as E
        single = isinstance(text, str)
        ids = []
        for t in ([text] if single else text):
            if t.startswith(E.PREFIX):
                row = [1] * 34 + [ord(c) % 60 + 5 for c in t[len(E.PREFIX):]]
            else:
                assert t == E.SUFFIX
                row = [3] * 5
            ids.append(row[:max_length] if truncation and max_length else row)
        width = max_length if padding == "max_length" else max(map(len, ids))
        mask = [[1] * len(r) + [0] * (width - len(r)) for r in ids]
        ids = [r + [0] * (width - len(r)) for r in ids]
        if return_tensors is None:
            return BatchEncoding({"input_ids": ids[0] if single else ids, "attention_mask": mask[0] if single else mask})
        return BatchEncoding({"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)})


def _comfy_fp8(sd):
    """The language Linears of a state dict as ComfyUI's fp8_scaled stores them: fp8 weight + a scalar `.weight_scale`
    (+ a `.comfy_quant` marker); everything else (embeddings, norms, the vision tower) untouched."""
    out = {}
    for k, v in sd.items():
        if ".language_model.layers." in k and k.endswith(".weight") and v.ndim == 2 and "norm" not in k:
            scale = v.float().abs().max() / 448.0
            out[k] = (v.float() / scale).to(torch.float8_e4m3fn)
            out[k[:-len(".weight")] + ".weight_scale"] = scale
            out[k[:-len(".weight")] + ".comfy_quant"] = torch.zeros(1, dtype=torch.uint8)
        else:
            out[k] = v
    return out


def test_fp8_text_encoder_stays_fp8_and_matches_the_expanded_model():
    pytest.importorskip("transformers")
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    from training.families.krea2 import embedder as E
    from training.modules import fp8
    torch.manual_seed(0)
    cfg = Qwen3VLConfig.from_dict(QWEN_TINY)
    ref = Qwen3VLForConditionalGeneration._from_config(cfg).to(torch.float32).eval()
    ref.lm_head = None
    sd = {k: v.detach().clone() for k, v in ref.state_dict().items() if not k.startswith("lm_head.")}
    comfy = _comfy_fp8(sd)

    kept = E.build_language_model(dict(comfy), QWEN_TINY, dtype=torch.float32)
    lm = kept.model.language_model
    n_fp8 = sum(1 for m in kept.modules() if isinstance(m, torch.nn.Linear) and fp8.is_fp8(m.weight))
    assert n_fp8 == 4 * 7 and fp8.is_fp8(lm.layers[0].self_attn.q_proj.weight)       # 7 Linears x 4 layers stay fp8
    assert lm.layers[0].self_attn.q_proj.weight.dtype == torch.float8_e4m3fn
    assert lm.embed_tokens.weight.dtype == torch.float32                              # not an fp8 tensor
    assert kept.model.visual is not None                                              # the vision tower loads, bf16 as stored
    assert not fp8.is_fp8(kept.model.visual.blocks[0].attn.qkv.weight)
    assert kept.lm_head is None
    # a device move must not expand it (Fizgig: never cast an fp8 model to a dtype)
    kept = kept.to("cpu")
    assert lm.layers[0].self_attn.q_proj.weight.dtype == torch.float8_e4m3fn

    # the same weights expanded to dense: what the fp8 forward computes
    expanded_sd = {}
    plain, scales = fp8.split_scales(dict(comfy))
    for k, v in plain.items():
        expanded_sd[k] = fp8.dequantize(v, scales.get(k[:-len(".weight")]), torch.float32) if fp8.is_fp8(v) else v
    dense = E.build_language_model(expanded_sd, QWEN_TINY, dtype=torch.float32)
    assert not any(fp8.is_fp8(m.weight) for m in dense.modules() if isinstance(m, torch.nn.Linear))

    enc_kept = E.Krea2TextEncoder(device="cpu", dtype=torch.float32, model=kept, tokenizer=_Tok(),
                                  select_layers=(0, 1, 2, 3))
    enc_dense = E.Krea2TextEncoder(device="cpu", dtype=torch.float32, model=dense, tokenizer=_Tok(),
                                   select_layers=(0, 1, 2, 3))
    enc_ref = E.Krea2TextEncoder(device="cpu", dtype=torch.float32, model=ref, tokenizer=_Tok(),
                                 select_layers=(0, 1, 2, 3))
    h_kept, m1 = enc_kept.encode(["a cat on a mat"])
    h_dense, m2 = enc_dense.encode(["a cat on a mat"])
    h_ref, _ = enc_ref.encode(["a cat on a mat"])
    assert torch.equal(m1, m2)
    assert torch.allclose(h_kept, h_dense, atol=1e-4)                  # fp8 kept == fp8 expanded, exactly the same maths
    valid = m1[0]
    cos = torch.nn.functional.cosine_similarity(h_kept[0][valid].flatten(1), h_ref[0][valid].flatten(1), dim=-1)
    assert cos.min() > 0.95                                             # and within fp8 tolerance of the original
    # a bf16 file (no scales) still loads as plain bf16 weights
    bf = E.build_language_model(dict(sd), QWEN_TINY, dtype=torch.bfloat16)
    assert not any(fp8.is_fp8(m.weight) for m in bf.modules() if isinstance(m, torch.nn.Linear))
    assert bf.model.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16


def test_text_encoder_has_no_int8_fallback_and_driver_loads_it_plain():
    import inspect

    from training.families.krea2 import driver as D
    from training.families.krea2 import embedder as E
    assert "int8" not in inspect.signature(E.Krea2TextEncoder.__init__).parameters
    assert "int8" not in inspect.getsource(D.Krea2Driver.load_text_encoder)


# ---- 3. attention: the uniform-length trim -------------------------------------------------------------------
def _forward(dit, noise, txt, mask, t=0.4):
    tokens, pos, m = S.prepare(noise, txt.shape[1], 2, mask)
    with torch.no_grad():
        return dit(img=tokens, context=txt, t=torch.full((noise.shape[0],), t), pos=pos, mask=m)


def test_trim_decision_follows_fizgigs_rules(monkeypatch):
    mask = torch.zeros(1, 64, dtype=torch.bool)
    mask[0, :40] = True
    p = A.AttentionParams.create_attention_params_from_mask(16, mask)
    assert (p.uniform_seqlen, p.uniform_exact, p.max_seqlen) == (64, False, 80)         # 56 valid -> next multiple of 64
    monkeypatch.setattr(A, "TRIM_MULTIPLE", 1)
    p = A.AttentionParams.create_attention_params_from_mask(16, mask)
    assert (p.uniform_seqlen, p.uniform_exact) == (56, True)                              # exact: no mask needed
    monkeypatch.setattr(A, "TRIM_MULTIPLE", 64)
    monkeypatch.setenv("TAGSCRIBER_ATTN_TRIM", "0")
    assert A.AttentionParams.create_attention_params_from_mask(16, mask).uniform_seqlen is None
    monkeypatch.delenv("TAGSCRIBER_ATTN_TRIM")
    ragged = torch.zeros(2, 64, dtype=torch.bool)
    ragged[0, :40] = True
    ragged[1, :50] = True
    assert A.AttentionParams.create_attention_params_from_mask(16, ragged).uniform_seqlen is None   # unequal: no trim
    equal = torch.zeros(2, 64, dtype=torch.bool)
    equal[:, :40] = True
    assert A.AttentionParams.create_attention_params_from_mask(16, equal).uniform_seqlen == 64
    assert A.AttentionParams.create_attention_params_from_mask(0, None).attention_mask is None
    assert A.AttentionParams.create_attention_params().uniform_seqlen is None


def test_trimmed_and_untrimmed_forward_agree_on_valid_tokens(monkeypatch):
    dit = _dit(torch.float32).eval()
    noise = torch.randn(1, 16, 8, 8)
    txt = torch.randn(1, 64, 3, 32)
    mask = torch.zeros(1, 64, dtype=torch.bool)
    mask[0, :23] = True
    trimmed = _forward(dit, noise, txt, mask)
    monkeypatch.setenv("TAGSCRIBER_ATTN_TRIM", "0")
    untrimmed = _forward(dit, noise, txt, mask)
    assert torch.allclose(trimmed, untrimmed, atol=1e-4)
    monkeypatch.delenv("TAGSCRIBER_ATTN_TRIM")
    monkeypatch.setattr(A, "TRIM_MULTIPLE", 1)                       # exact trim: the window holds no padding at all
    assert torch.allclose(_forward(dit, noise, txt, mask), untrimmed, atol=1e-4)


def test_batch_of_two_with_unequal_lengths_matches_each_alone():
    """Fizgig: unequal valid lengths attend over the padded length with the key mask - each sample must equal its own
    batch-1 (trimmed) forward."""
    dit = _dit(torch.float32).eval()
    noise = torch.randn(2, 16, 8, 8)
    txt = torch.randn(2, 64, 3, 32)
    mask = torch.zeros(2, 64, dtype=torch.bool)
    mask[0, :9] = True
    mask[1, :31] = True
    both = _forward(dit, noise, txt, mask)
    for i in range(2):
        alone = _forward(dit, noise[i:i + 1], txt[i:i + 1], mask[i:i + 1])
        assert torch.allclose(both[i], alone[0], atol=1e-4), i


def test_driver_batch_two_with_unequal_valid_lengths_is_finite(driver):
    dit = _dit()
    net = FamilyLoRA(dit, driver)
    net.add_trainable(4, 4)
    loss, _ = driver.training_loss(dit, torch.randn(2, 16, 8, 8), _cond(2, valid=(5, 14)),
                                   torch.Generator().manual_seed(2))
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_sdpa_backend_selection_mirrors_fizgig(monkeypatch):
    from training.modules import sdpa
    ctx = sdpa.sdpa_backend_ctx("cpu")
    with ctx:                                              # CPU tensors never reach the cuDNN probe
        pass
    monkeypatch.setenv("TAGSCRIBER_SDPA_BACKEND", "default")
    with sdpa.sdpa_backend_ctx("cuda"):
        pass
    seen = set(sdpa._seen_shapes)
    sdpa._seen_shapes.clear()
    try:
        assert sdpa.consider_training_backend(10 ** 6) is None                      # nothing recorded
        sdpa.note_shape(1152)
        assert sdpa.consider_training_backend(10 ** 6) is None                      # explicitly default: not second-guessed
        monkeypatch.setenv("TAGSCRIBER_SDPA_BACKEND", "auto")
        assert sdpa.consider_training_backend(10) is None                           # short run: stays on the default
        sdpa.note_shape(1216)
        assert sdpa.consider_training_backend(10 ** 6) == (2, 140)                  # 2 shapes x 35 x margin 2
        assert sdpa._training_cudnn
    finally:
        sdpa._training_cudnn = False
        sdpa._seen_shapes.clear()
        sdpa._seen_shapes.update(seen)


def test_after_epoch_is_suppressed_when_compile_was_requested(driver, monkeypatch):
    calls = []
    import training.modules.sdpa as sdpa
    monkeypatch.setattr(sdpa, "consider_training_backend", lambda n: calls.append(n))
    driver._compile_requested = True
    driver.after_epoch(0, 100)
    assert calls == []
    driver._compile_requested = False
    driver.after_epoch(0, 100)
    assert calls == [100]


# ---- 4. torch.compile ---------------------------------------------------------------------------------------
@pytest.fixture
def machine(monkeypatch):
    """A machine where compile could win: NVIDIA, triton + a compiler present, a big free card."""
    monkeypatch.setattr(C, "is_rocm", lambda: False)
    monkeypatch.setattr(C, "_triton_importable", lambda: True)
    monkeypatch.setattr(C, "triton_matches_torch", lambda *a, **k: (True, ""))
    monkeypatch.setattr(C, "has_host_c_compiler", lambda platform=None: True)
    return monkeypatch


def test_should_compile_decision_table(machine):
    sc = C.should_compile
    ok, why = sc(700, "int8", 0, vram_gb=24)
    assert ok is True and "pays back within ~600" in why
    assert sc(599, "int8", 0, vram_gb=24)[0] is False and "too short" in sc(599, "int8", 0, vram_gb=24)[1]
    assert sc(1176, "nf4", 0, vram_gb=16)[0] is True
    assert sc(1175, "nf4", 0, vram_gb=16)[0] is False
    for kind in ("fp8", "bf16"):
        ok, why = sc(10 ** 6, kind, 0, vram_gb=32)
        assert ok is False and "only measured for the quantised paths" in why
    ok, why = sc(10 ** 6, "int8", 4, vram_gb=32)
    assert ok is False and "block swap" in why
    # memory: INT8 + compile needs ~21.5 GB free at 0.25 MP; below that the checkpoint-outside boundary, then no compile
    assert sc(700, "int8", 0, vram_gb=22)[0] is True
    ok, why = sc(700, "int8", 0, vram_gb=21)
    assert ok == "outside" and "OUTSIDE" in why
    ok, why = sc(700, "int8", 0, vram_gb=19)
    assert ok is False and "INT8 alone still fits, compile does not" in why
    ok, why = sc(700, "nf4", 0, vram_gb=14, mp=1.0)                       # 13 + 15 x 0.75 + 1.5 GB needed
    assert ok is False and "NF4 + compile peaks" in why and "lower Target Megapixels" in why
    assert sc(2000, "nf4", 0, vram_gb=14, mp=0.25)[0] is True
    assert sc(2000, "nf4", 0, vram_gb=40, mp=1.0)[0] is True


def test_should_compile_environment_gates(machine):
    machine.setattr(C, "is_rocm", lambda: True)
    ok, why = C.should_compile(10 ** 6, "int8", 0, vram_gb=32)
    assert ok is False and "ROCm/HIP" in why and "set Compile Blocks to On to override" in why
    machine.setattr(C, "is_rocm", lambda: False)
    machine.setattr(C, "_triton_importable", lambda: False)
    assert "triton is not installed" in C.should_compile(10 ** 6, "int8", 0, vram_gb=32)[1]
    machine.setattr(C, "_triton_importable", lambda: True)
    machine.setattr(C, "triton_matches_torch", lambda *a, **k: (False, "triton 3.8 does not pair"))
    assert C.should_compile(10 ** 6, "int8", 0, vram_gb=32) == (False, "triton 3.8 does not pair")
    machine.setattr(C, "triton_matches_torch", lambda *a, **k: (True, ""))
    machine.setattr(C, "has_host_c_compiler", lambda platform=None: False)
    assert "no C compiler" in C.should_compile(10 ** 6, "int8", 0, vram_gb=32)[1]


def test_compile_boundary_and_triton_pairing():
    assert C.compile_boundary("int8", vram_gb=24) == "inside"
    assert C.compile_boundary("int8", vram_gb=18) == "outside"
    assert C.compile_boundary("int8", vram_gb=24, mp=1.0) == "outside"       # 20 + 11.25 + 1.5 > 24
    assert C.compile_boundary("int8", vram_gb=32, mp=0.25, batch=2) == "inside"
    assert C.compile_boundary("int8", vram_gb=24, mp=0.25, batch=2) == "outside"      # batch multiplies tokens
    assert C.compile_boundary("nf4", vram_gb=8) == "inside" and C.compile_boundary("fp8", vram_gb=8) == "inside"
    assert C.compile_boundary("int8", vram_gb=0) == "inside"                  # no readable GPU: no basis to pick
    assert C.triton_matches_torch("3.5.1", "2.10.0+cu128") == (True, "")
    assert C.triton_matches_torch("3.6.0", "2.10.0")[0] is True
    ok, why = C.triton_matches_torch("3.8.0", "2.10.0")
    assert ok is False and "does not pair" in why and "triton-windows>=3.5.0,<3.7" in why
    assert C.triton_matches_torch("9.9", "1.0.0") == (True, "")               # unknown torch: not gated
    assert C.has_host_c_compiler("nt") is True


def test_resolve_matches_the_trainers_modes(machine, caplog):
    caplog.set_level(logging.INFO)
    kw = dict(precision="int8", blocks_to_swap=0, total_steps=10 ** 5, mp=0.25, batch=1)
    machine.setattr(C, "_free_vram_gb", lambda: 24.0)
    assert C.resolve("Off", **kw) is False
    assert C.resolve("On", **kw) is True
    assert C.resolve("outside", **kw) == "outside"
    assert C.resolve("Outside", **kw) == "outside"
    assert C.resolve("auto", **kw) is True and "[compile] auto: ENABLED" in caplog.text
    machine.setattr(C, "_free_vram_gb", lambda: 18.0)
    assert C.resolve("On", **kw) == "outside" and "won't fit at this token load" in caplog.text
    machine.setattr(C, "is_rocm", lambda: True)
    assert C.resolve("Auto", **kw) is False and "[compile] auto: off - ROCm/HIP" in caplog.text
    assert C.resolve("On", **kw) in ("outside", True)                         # On overrides the ROCm default
    caplog.clear()
    C.resolve("On", **{**kw, "blocks_to_swap": 8})
    assert "block swap is active (8 blocks)" in caplog.text


def test_compile_blocks_degrades_like_fizgig(monkeypatch, caplog):
    dit = _dit()
    dit.enable_gradient_checkpointing(True)
    blocks = list(dit.blocks)
    assert C.compile_blocks(dit, 2) == 0 and "block swap moves weights" in caplog.text
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 6))
    assert C.compile_blocks(dit, 0, fp8_scaled=True) == 0 and "SM 8.6" in caplog.text
    monkeypatch.undo()
    monkeypatch.setattr(C, "_triton_importable", lambda: False)
    assert C.compile_blocks(dit, 0) == 0 and "triton is not installed" in caplog.text
    monkeypatch.setattr(C, "_triton_importable", lambda: True)
    monkeypatch.setattr(C, "triton_matches_torch", lambda *a, **k: (False, "triton 3.8 does not pair"))
    assert C.compile_blocks(dit, 0) == 0 and "Training continues uncompiled" in caplog.text
    monkeypatch.setattr(C, "triton_matches_torch", lambda *a, **k: (True, ""))
    monkeypatch.setattr(C, "find_host_compiler", lambda: False)
    assert C.compile_blocks(dit, 0) == 0
    assert list(dit.blocks) == blocks                      # nothing was wrapped on any refusal


def test_prepare_training_off_and_on_a_machine_without_triton(driver, monkeypatch):
    dit = _dit()
    net = FamilyLoRA(dit, driver)
    net.add_trainable(4, 4)
    blocks = list(dit.blocks)
    driver.configure(compile_blocks="Off")
    driver.prepare_training(dit, net, precision="int8", blocks_to_swap=0, total_steps=10 ** 5, megapixels=0.25,
                            batch_size=1)
    assert list(dit.blocks) == blocks and driver._compile_requested is False
    monkeypatch.setattr(C, "_triton_importable", lambda: False)
    driver.configure(compile_blocks="On")
    driver.prepare_training(dit, net, precision="int8", blocks_to_swap=0, total_steps=10 ** 5, megapixels=0.25,
                            batch_size=1)
    assert list(dit.blocks) == blocks and driver._compile_requested is True       # asked for, degraded to eager
    loss, _ = driver.training_loss(dit, torch.randn(1, 16, 8, 8), _cond(1), torch.Generator().manual_seed(1))
    assert torch.isfinite(loss)


def test_compiled_block_equals_the_eager_block():
    """The real torch.compile path (inductor) where this machine can run it; the traced-graph path (dynamo's eager
    backend: fullgraph, the checkpoint inside, the LoRA wrapper and AttentionParams all traceable) everywhere."""
    from training.families.krea2.driver import Krea2Driver as D
    drv = D()
    drv.description = registry.get("krea2")
    dit = _dit()
    dit.enable_gradient_checkpointing(True)
    net = FamilyLoRA(dit, drv)
    net.add_trainable(4, 4)
    for p in net.parameters():
        p.data.normal_(0, 0.05)
    lat, cond = torch.randn(1, 16, 8, 8), _cond(1)

    def run():
        dit.train()
        net.parameters()[0].grad = None
        loss, _ = drv.training_loss(dit, lat, cond, torch.Generator().manual_seed(0))
        loss.backward()
        return loss.item(), [None if p.grad is None else p.grad.clone() for p in net.parameters()]
    eager_loss, eager_grads = run()
    for p in net.parameters():
        p.grad = None
    original = list(dit.blocks)
    for i, b in enumerate(original):
        dit.blocks[i] = torch.compile(C.CheckpointedBlock(b, True), fullgraph=True, backend="eager")
    loss, grads = run()
    assert loss == pytest.approx(eager_loss, rel=1e-5)
    assert all(g is not None for g in grads)
    assert torch.allclose(grads[3], eager_grads[3] if eager_grads[3] is not None else grads[3], atol=1e-4)
    for i, b in enumerate(original):                        # the outside boundary traces too
        dit.blocks[i] = C.CheckpointedBlock(torch.compile(b, fullgraph=True, backend="eager"), True)
    assert run()[0] == pytest.approx(eager_loss, rel=1e-5)
    # inductor itself: only where triton and a compiler exist; refused cleanly otherwise (returns 0)
    for i, b in enumerate(original):
        dit.blocks[i] = b
    n = C.compile_blocks(dit, 0, boundary="inside")
    if n == 0:
        pytest.skip("torch.compile (inductor) cannot run on this machine: no matching triton / C compiler")
    assert run()[0] == pytest.approx(eager_loss, rel=1e-3)


def test_compile_blocks_param_and_presets(driver):
    desc = registry.get("krea2")
    assert "COMPILE_BLOCKS" in desc.family_options and P.DRIVER_OPTIONS["COMPILE_BLOCKS"] == "compile_blocks"
    p = P.BY_KEY["COMPILE_BLOCKS"]
    assert p.default == "Auto" and p.options[:3] == ("Auto", "On", "Off") and p.family_only == "option"
    assert "AMD" in p.tip and "NVIDIA" in p.tip and "OFF" in p.tip
    assert P.family_shows(p, desc) and not P.family_shows(p, registry.get("qwen_image21"))
    new, rep = presets.apply(dict(desc.presets[0][1]), P.defaults(), desc)
    assert new["COMPILE_BLOCKS"] == "Auto" and "COMPILE_BLOCKS" not in rep.ignored and rep.refused == []
    for old, want in (("auto", "Auto"), ("on", "On"), ("OFF", "Off"), ("outside", "Outside"), ("Auto", "Auto"),
                      (True, "On")):
        assert presets.apply({"COMPILE_BLOCKS": old}, P.defaults(), desc)[0]["COMPILE_BLOCKS"] == want
    from training import pipeline
    assert pipeline.driver_options(desc, {**P.defaults(), "COMPILE_BLOCKS": "On"}) == {"compile_blocks": "On"}
    d = Krea2Driver()
    d.configure(compile_blocks="Outside")
    assert d.compile_blocks == "Outside"


def test_requirements_carry_fizgigs_triton_line():
    import os
    root = os.path.dirname(os.path.dirname(__file__))
    text = open(os.path.join(root, "requirements.txt"), encoding="utf-8").read()
    assert "triton-windows>=3.5.1,<3.7 ; sys_platform == 'win32'" in text.splitlines()
