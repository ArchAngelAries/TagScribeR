# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/minimax/embedder.py (text-only encoder, the
# nvfp4 Linear and its dequant).
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: ENCODE ONLY and TEXT ONLY - no vision tower / reference images, no bitsandbytes NF4 route
# for the bf16 text-encoder file (the nvfp4-awq file ComfyUI ships is what is read, kept packed as Fizgig does), no
# comfy int8 text-encoder variant; Fizgig's pinned host-to-device layer streaming for small cards (embedderH2D.py) is
# replaced by a plain per-layer move: the packed layers stay on the CPU and each rides to the GPU for its own forward;
# the tokenizer comes from a qwen3vl_tokenizer/ folder next to the checkpoint or the Hugging Face cache (Fizgig bundles
# the files).
# Pre-sm_80 NVIDIA cards (V100, T4): the chunked prefill (_plan_chunk, _text_forward) is ported from Fizgig commit b94ffcaf
# ("MiniMax H3 text encoder: chunked prefill for long captions on pre-sm_80 cards", #172, v8.x; Fizgig diff 1c8ec88..5d31b2f).
# Changes: gated on a real CUDA (not ROCm/HIP) device below compute capability 8.0; otherwise the original call. Never run on
# real pre-sm_80 hardware by TagScribeR: unit-tested on tiny CPU models only.
"""MiniMax H3 text conditioning: Qwen3-VL-32B language model truncated to 50 layers.

The DiT is conditioned on the UNNORMALISED hidden state after language layer 50 (no final norm - comfy applies none).
Captions are tokenised raw with NO special tokens (the H3 convention) and NO truncation; an empty caption is a single
pad token. Output per caption: (L, 5120) bf16.
"""
import logging
import os

import torch
import torch.nn as nn

from training.families.minimax_h3.weights import SafeReader

logger = logging.getLogger(__name__)

_QWEN3_32B_TRUNC50 = dict(
    hidden_size=5120, num_hidden_layers=50, num_attention_heads=64, num_key_value_heads=8,
    head_dim=128, intermediate_size=25600, vocab_size=151936, max_position_embeddings=262144,
    rms_norm_eps=1e-6, rope_theta=5000000.0, attention_bias=False, tie_word_embeddings=False,
)
_QUANT_SUFFIXES = (".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight",
                   ".self_attn.o_proj.weight", ".mlp.gate_proj.weight", ".mlp.up_proj.weight",
                   ".mlp.down_proj.weight")
# H3 extends the stock Qwen3-VL vocabulary with these, IN THIS ORDER (ids 151669..151675): they exist in neither
# vocab.json nor tokenizer.json, so a reordering would silently shift every id.
_H3_SPECIAL_TOKENS = ("<d>", "</d>", "<|cutoff|>", "<|lyrics_start|>", "<|lyrics_end|>", "<|caption_start|>",
                      "<|caption_end|>")
TOKENIZER_REPO = "Qwen/Qwen3-VL-4B-Instruct"       # the 4B / 32B tokenizers share one vocabulary
TOKENIZER_FILES = ("chat_template.json", "generation_config.json", "merges.txt", "preprocessor_config.json",
                   "tokenizer.json", "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json")
# Resident nvfp4 build: 12.8 GB of packed weights + ~1.9 GB forward peak (Fizgig embedder.py). Below this free VRAM the
# layers stay on the CPU and are moved in one at a time.
RESIDENT_NEED_GB = 15.0

_E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
_E2M1_SIGNED = torch.cat([_E2M1_MAG, -_E2M1_MAG])


def _from_blocked(blocked, rows, cols):
    """Invert ComfyUI's to_blocked (comfy/float.py): (-1, 32, 16) tiles -> row-major [rows, cols]."""
    nrb = -(-rows // 128)
    ncb = -(-cols // 4)
    x = blocked.reshape(-1, 32, 4, 4).transpose(1, 2)
    x = x.reshape(nrb, ncb, 128, 4).permute(0, 2, 1, 3).reshape(nrb * 128, ncb * 4)
    return x[:rows, :cols].contiguous()


def _nvfp4_dequant(packed, block_scale_fp8, global_scale):
    """packed U8 [out, in/2] -> bf16 [out, in].  W = e2m1(code) * block_scale * global_scale (blocks of 16 along the
    input dim; HIGH nibble = first element). Row-chunked so the transients stay bounded."""
    out, in2 = packed.shape
    inp = in2 * 2
    dev = packed.device
    bs = _from_blocked(block_scale_fp8.to(torch.float32).reshape(-1, 32, 16), out, inp // 16)
    gs = global_scale.to(torch.float32)
    table = _E2M1_SIGNED.to(dev)
    w_out = torch.empty(out, inp, dtype=torch.bfloat16, device=dev)
    chunk = max(1, (32 << 20) // max(inp, 1))
    for r0 in range(0, out, chunk):
        r1 = min(out, r0 + chunk)
        codes = torch.empty(r1 - r0, inp, dtype=torch.uint8, device=dev)
        codes[:, 0::2] = packed[r0:r1] >> 4
        codes[:, 1::2] = packed[r0:r1] & 0x0F
        vals = table[codes.long()]
        w = (vals.view(r1 - r0, inp // 16, 16) * bs[r0:r1].unsqueeze(-1)).mul_(gs)
        w_out[r0:r1] = w.view(r1 - r0, inp).to(torch.bfloat16)
    return w_out


class Nvfp4Linear(nn.Linear):
    """A frozen Linear that KEEPS the checkpoint's nvfp4 weights (0.5 byte/param) and dequantises per matmul. The
    block scales (fp8) and the global scale (fp32) are held as raw BYTES so no stray `.to(dtype)` can cast them. AWQ's
    `pre_quant_scale` multiplies the INPUT (as ComfyUI does)."""

    def __init__(self, in_features, out_features, bias=False, compute_dtype=torch.bfloat16):
        super().__init__(in_features, out_features, bias=bias)
        del self._parameters["weight"]
        self.compute_dtype = compute_dtype
        self.register_buffer("packed", torch.empty(out_features, in_features // 2, dtype=torch.uint8), persistent=False)
        self.register_buffer("bscale", torch.empty(out_features, in_features // 16, dtype=torch.uint8),
                             persistent=False)
        self.register_buffer("gscale", torch.empty(4, dtype=torch.uint8), persistent=False)
        self.register_buffer("pre_quant_scale", None, persistent=False)

    def _scales(self):
        return self.bscale.view(torch.float8_e4m3fn), self.gscale.view(torch.float32)

    def forward(self, x):
        dt = self.compute_dtype
        if self.pre_quant_scale is not None:
            x = x * self.pre_quant_scale.to(x.dtype)
        bs, gs = self._scales()
        return torch.nn.functional.linear(x.to(dt), _nvfp4_dequant(self.packed, bs, gs).to(dt), self.bias)


def tokenizer_source(model_path: str) -> str:
    """A qwen3vl_tokenizer/ folder next to the checkpoint, else the repo id (Hugging Face cache first)."""
    local = os.path.join(os.path.dirname(os.path.abspath(model_path)), "qwen3vl_tokenizer")
    if os.path.isfile(os.path.join(local, "tokenizer_config.json")):
        return local
    return TOKENIZER_REPO


def build_qwen3_te(config_overrides=None):
    """A Qwen3Model with no final norm (returns the raw layer-50 output). Text only: for a caption without images
    Qwen3-VL's mrope collapses to ordinary 1-D rope, so this equals the full VL stack."""
    from transformers import Qwen3Config, Qwen3Model
    cfg = dict(_QWEN3_32B_TRUNC50)
    if config_overrides:
        cfg.update(config_overrides)
    model = Qwen3Model(Qwen3Config(**cfg))
    model.norm = nn.Identity()
    return model


def _cfg_attr(cfg, name, default):
    """A config attribute at the top level (Qwen3Model) or under text_config (Qwen3VLModel)."""
    for cand in (cfg, getattr(cfg, "text_config", None)):
        v = getattr(cand, name, None)
        if isinstance(v, int) and v > 0:
            return v
    return default


def is_pre_sm80_cuda(device) -> bool:
    """True only for a real NVIDIA CUDA device with compute capability below 8.0 (V100, T4 ...). False on CPU, on
    ROCm/HIP builds (which also answer torch.cuda.*) and on sm_80+ cards, where nothing changes."""
    dev = torch.device(device)
    if dev.type != "cuda" or getattr(torch.version, "hip", None):
        return False
    return torch.cuda.get_device_capability(dev.index)[0] < 8


class H3TextEncoder:
    """Encodes captions to (L, 5120) bf16 on the CPU. Memoised by caption text."""

    def __init__(self, model, tokenizer, device="cuda", compute_dtype=torch.bfloat16, cpu_embed=False, stream=False):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.compute_dtype = compute_dtype
        self.cpu_embed = cpu_embed
        self._cache = {}
        if stream:
            self._stream_layers()

    def _stream_layers(self):
        """Layers stay on the CPU; each is moved to the device for its own forward and back afterwards."""
        dev = self.device

        def pre(mod, _args):
            mod.to(dev)

        def post(mod, _args, _out):
            mod.to("cpu")
        for layer in self.model.layers:
            layer.register_forward_pre_hook(pre)
            layer.register_forward_hook(post)

    def _pad_id(self) -> int:
        pid = getattr(self.tokenizer, "pad_token_id", None)
        return 151643 if pid is None else int(pid)

    def _forward(self, ids):
        """Hidden states [B, L, 5120]. Long sequences on pre-sm_80 NVIDIA cards are prefilled in L-chunks (see
        _plan_chunk): SDPA there falls back to the math backend, which materialises the [B, H, L, L] score tensor per
        layer. Chunk k passes the past_key_values of chunk k-1; causal attention makes this exact (each query sees the
        same keys in the same single softmax) and right padding stays safe (a real token only looks left). Everywhere
        else this is the original single call."""
        chunk = self._plan_chunk(ids)
        if chunk is None or chunk >= ids.shape[1]:
            if self.cpu_embed:
                emb = self.model.embed_tokens(ids.to("cpu")).to(self.device)
                return self.model(inputs_embeds=emb).last_hidden_state
            return self.model(input_ids=ids.to(self.device)).last_hidden_state
        outs, past = [], None
        for s in range(0, ids.shape[1], chunk):
            e = min(s + chunk, ids.shape[1])
            if self.cpu_embed:
                kw = dict(inputs_embeds=self.model.embed_tokens(ids[:, s:e].to("cpu")).to(self.device))
            else:
                kw = dict(input_ids=ids[:, s:e].to(self.device))
            kw["use_cache"] = True
            if past is not None:
                kw["past_key_values"] = past
            r = self.model(**kw)
            past = r.past_key_values
            outs.append(r.last_hidden_state)
        return torch.cat(outs, dim=1)

    def _plan_chunk(self, ids):
        """Prefill chunk size, or None for the single pass. None unless the device is a pre-sm_80 NVIDIA card and
        B*L^2 > 1024^2. The per-layer math-SDPA peak is H*B*L^2 elements at (2*itemsize + 4) bytes (bf16 scores, mask
        temporary, fp32 softmax); the chunk C is the largest keeping the [B, H, C, L] slice under the live free VRAM
        minus the KV cache (needed: it is the exactness mechanism), MLP and hidden overhead and a 10% margin,
        clamped to [128, 2048] (Fizgig b94ffcaf)."""
        if not is_pre_sm80_cuda(self.device):
            return None
        B, L = ids.shape
        if B * L * L <= 1024 * 1024:
            return None
        dev = torch.device(self.device)
        free_b, _ = torch.cuda.mem_get_info(dev.index)
        cfg = self.model.config
        heads = _cfg_attr(cfg, "num_attention_heads", 64)
        itemsize = torch.empty((), dtype=self.compute_dtype).element_size()
        kv_heads = _cfg_attr(cfg, "num_key_value_heads", heads)
        head_dim = _cfg_attr(cfg, "head_dim", _cfg_attr(cfg, "hidden_size", 5120) // heads)
        layers = _cfg_attr(cfg, "num_hidden_layers", 50)
        kv = B * kv_heads * L * head_dim * 2 * itemsize * layers
        mlp = B * L * _cfg_attr(cfg, "intermediate_size", 0) * itemsize
        extra = B * L * _cfg_attr(cfg, "hidden_size", 5120) * itemsize * 4
        budget = free_b - kv - mlp - extra - int(0.10 * free_b)
        if budget <= 0:
            budget = int(0.25 * free_b)
        c = max(128, min(2048, budget // max(1, B * heads * L * (2 * itemsize + 4))))
        if not getattr(self, "_chunk_logged", False):
            self._chunk_logged = True
            logger.info("TE forward: B=%d L=%d single-pass peak ~%.2f GB > %.2f GB free (KV ~%.2f GB) - chunked "
                        "prefill (C=%d, exact: causal attention over past_key_values)",
                        B, L, (B * heads * L * L * (2 * itemsize + 4) + kv + mlp) / 2 ** 30,
                        free_b / 2 ** 30, kv / 2 ** 30, min(c, L))
        return min(c, L)

    def _tokens(self, caption):
        ids = self.tokenizer(caption, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        return ids if ids.numel() else torch.tensor([self._pad_id()])

    @torch.no_grad()
    def encode_batch(self, captions, batch_size: int = 8):
        """-> list of (L_i, 5120) tensors on the CPU. Right-padded batches, no attention mask: the stack is causal, so
        trailing pads cannot influence a real token and slicing each row back recovers the single-caption result."""
        out = [None] * len(captions)
        todo = []
        for i, c in enumerate(captions):
            if c in self._cache:
                out[i] = self._cache[c].clone()
            else:
                todo.append(i)
        pad = self._pad_id()
        for s in range(0, len(todo), batch_size):
            idxs = todo[s:s + batch_size]
            toks = [self._tokens(captions[i]) for i in idxs]
            L = max(t.numel() for t in toks)
            ids = torch.full((len(toks), L), pad, dtype=torch.long)
            for r, t in enumerate(toks):
                ids[r, :t.numel()] = t
            hs = self._forward(ids)
            for r, i in enumerate(idxs):
                emb = hs[r, :toks[r].numel()].to(self.compute_dtype).detach().cpu()
                self._cache[captions[i]] = emb
                out[i] = emb.clone()
        return out

    def unload(self):
        self.model.to("cpu")
        self.model = None
        self._cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_h3_text_encoder(path: str, device="cuda", compute_dtype=torch.bfloat16, free_gb=None) -> H3TextEncoder:
    """The nvfp4-awq Qwen3-VL-32B file (language layers only), packed weights kept. Resident when `free_gb` (VRAM) fits
    it, else layer-streamed from the CPU (needs ~19 GB of RAM)."""
    from transformers import AutoTokenizer
    path = str(path)
    dev = torch.device(device)
    with SafeReader(path) as probe:
        if not any(k.endswith(".comfy_quant") for k in probe.keys()):
            raise ValueError(
                f"{path} is not the nvfp4 MiniMax H3 text encoder. TagScribeR reads 'qwen3vl_32b_minimax_h3_"
                f"nvfp4_awq.safetensors' (15.7 GB, Comfy-Org/MiniMax-H3); the 48 GB bf16 file is not supported.")
    stream = dev.type == "cuda" and free_gb is not None and free_gb < RESIDENT_NEED_GB
    cpu_embed = dev.type == "cuda"
    with torch.device("meta"):
        model = build_qwen3_te()
        for mod_name, module in list(model.named_modules()):
            for child_name, child in list(module.named_children()):
                full = f"{mod_name}.{child_name}" if mod_name else child_name
                if isinstance(child, nn.Linear) and (full + ".weight").endswith(_QUANT_SUFFIXES):
                    setattr(module, child_name, Nvfp4Linear(child.in_features, child.out_features,
                                                            bias=child.bias is not None, compute_dtype=compute_dtype))
    place = torch.device("cpu") if stream else dev
    logger.info("[text encoder] loading Qwen3-VL-32B (nvfp4, packed weights kept) - %s",
                "layers stream from the CPU (low VRAM)" if stream else "resident")
    with SafeReader(path) as f:
        ckpt = set(f.keys())
        for mod_name, module in model.named_modules():
            if not isinstance(module, Nvfp4Linear):
                continue
            fm = "model." + mod_name
            module.packed = f.get_tensor(fm + ".weight").to(torch.uint8).to(place)
            module.bscale = f.get_tensor(fm + ".weight_scale").contiguous().view(torch.uint8).to(place)
            module.gscale = f.get_tensor(fm + ".weight_scale_2").to(torch.float32).reshape(1).contiguous() \
                .view(torch.uint8).to(place)
            pqs = fm + ".pre_quant_scale"
            module.pre_quant_scale = f.get_tensor(pqs).to(compute_dtype).to(place) if pqs in ckpt else None
        missing = []
        for name, _ in list(model.named_parameters()):
            src = "model." + name
            if src not in ckpt:
                missing.append(name)
                continue
            w = f.get_tensor(src)
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            tgt = torch.device("cpu") if (cpu_embed and name == "embed_tokens.weight") else place
            keep = w.to(torch.float32) if w.dtype == torch.float32 else w.to(compute_dtype)
            setattr(parent, name.rsplit(".", 1)[1], nn.Parameter(keep.to(tgt), requires_grad=False))
    if missing:
        raise ValueError(f"text encoder checkpoint is missing {len(missing)} tensors, e.g. {missing[:5]}")
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    model.rotary_emb = Qwen3RotaryEmbedding(model.config).to(dev)   # stays on the device even when layers stream
    for mod in model.modules():
        for bname, buf in list(mod.named_buffers(recurse=False)):
            if buf is not None and buf.is_meta:
                mod.register_buffer(bname, torch.zeros(buf.shape, dtype=buf.dtype, device=place))
    model.requires_grad_(False)
    from training.hf_cache import from_pretrained_cache_first
    tok = from_pretrained_cache_first(AutoTokenizer, tokenizer_source(path))   # offline once cached (Fizgig #174)
    if [t for t in _H3_SPECIAL_TOKENS if tok.convert_tokens_to_ids(t) is None]:
        tok.add_tokens(list(_H3_SPECIAL_TOKENS), special_tokens=True)
    return H3TextEncoder(model, tok, device=dev, compute_dtype=compute_dtype, cpu_embed=cpu_embed, stream=stream)
