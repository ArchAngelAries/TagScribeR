# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/moss_image/embedder.py (commit 9162fb20,
# v8.7.3). Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: the tokenizer loads cache-first through training/hf_cache.py; the ConvRot decode is
# model.decode_convrot (training/families/minimax_h3/weights.py); the optional 8-bit weights come from
# training/families/qwen_image21/embedder.py; the pinned-RAM state-dict cache between workbench prompts
# (_kept_state_dict / kept_gb / drop_kept_state_dict, keep_in_ram) is left out - TagScribeR has no workbench and caches
# captions once per run; otherwise unchanged. Fizgig's own header follows.
#
# The Qwen3.5 text model below is ported from huggingface/transformers `models/qwen3_5/modeling_qwen3_5.py` (main,
# d4804f9d76, 17 Oct 2026). Copyright 2025 The Qwen Team and The HuggingFace Inc. team. Licensed under the Apache
# License, Version 2.0. Changes for Fizgig: text path only (no vision tower, LM head or MTP head), no KV / recurrent
# cache (one prefill per prompt), the pure-torch chunked gated-delta rule and causal conv1d (transformers' own
# fallbacks), SDPA for the full-attention layers. Fizgig pins transformers 4.57, which has no Qwen3.5.
"""MOSS-Image text conditioning: Qwen3.5-9B, the final hidden state (after the final norm), one prompt at a time.

Follows the reference (pipeline_moss_image.py MOSSImagePipeline.encode_prompt, and ComfyUI's comfy/text_encoders/
moss_image.py, which agree):
* the prompt as one user turn through the tokenizer's chat template with add_generation_prompt=True and
  enable_thinking=True: "<|im_start|>user\\n{prompt.strip()}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n"
* truncated to 4096 tokens, no padding ("Unpadded individual forwards also avoid padding effects in Qwen GDN")
* output.last_hidden_state[0] -> (L, 4096); the DiT keeps every token

Weights: Comfy-Org/Qwen3.5's single file (text_encoders/qwen3.5_9b_bf16.safetensors, keys model.language_model.*),
MOSS-Image's own shards (text_encoder/, keys language_model.*) or ComfyUI's int8 ConvRot repack (decoded). Every tensor
is cast to bf16, as the reference's from_pretrained(dtype=bfloat16) does (A_log and the gated norm included).
Tokenizer + chat template: OpenMOSS-Team/MOSS-Image-Base tokenizer/ (the description's helper_files).

The vendored model is kept (TagScribeR runs transformers 5.x, which has qwen3_5) because it brings the layer-major
streamed pass for cards without room for the 15 GB encoder, and because its gated-delta rule is the pure-torch path:
on AMD ROCm there are no flash-linear-attention kernels, so this is the path that runs there anyway.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

TOKENIZER_REPO = "OpenMOSS-Team/MOSS-Image-Base"
MAX_LENGTH = 4096                    # pipeline_moss_image.py encode_prompt max_sequence_length

# OpenMOSS-Team/MOSS-Image-Base text_encoder/config.json text_config, spelled out (a config written for a newer
# transformers can be misread silently)
QWEN35_9B = dict(vocab_size=248320, hidden_size=4096, intermediate_size=12288, num_hidden_layers=32,
                 num_attention_heads=16, num_key_value_heads=4, head_dim=256, rms_norm_eps=1e-6,
                 rope_theta=10000000.0, partial_rotary_factor=0.25, full_attention_interval=4,
                 linear_conv_kernel_dim=4, linear_key_head_dim=128, linear_num_key_heads=16,
                 linear_num_value_heads=32, linear_value_head_dim=128)


def layer_types(cfg):
    return ["full_attention" if (i + 1) % cfg["full_attention_interval"] == 0 else "linear_attention"
            for i in range(cfg["num_hidden_layers"])]


class Qwen35RMSNorm(nn.Module):
    """Zero-centred weight: x_normed * (1 + w), computed in fp32."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * (1.0 + self.weight.float())).type_as(x)


class Qwen35RMSNormGated(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate):
        input_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        h = self.weight * h.to(input_dtype)
        h = h * F.silu(gate.to(torch.float32))
        return h.to(input_dtype)


def _l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64):
    """transformers' torch_chunk_gated_delta_rule (no initial state, no final state), use_qk_l2norm_in_kernel=True."""
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    query, key, value, beta, decay = [x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
                                      for x in (query, key, value, beta, g)]
    query = _l2norm(query)
    key = _l2norm(key)
    query = query * query.shape[-1] ** -0.5
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))
    total = sequence_length + pad_size
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
                                  for x in (query, key, k_beta, v_beta)]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)
    upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    cum_decay = decay.cumsum(dim=3)
    pairwise = (cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)).masked_fill(upper, float("-inf")).exp()
    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise
    intra = (query @ key.transpose(-1, -2)) * pairwise
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)
    new_values = torch.linalg.solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(ut_system, decayed_k_beta, upper=False, unitriangular=True)
    state = torch.zeros(batch_size, num_v_heads, k_head_dim, v_head_dim, dtype=new_values.dtype,
                        device=new_values.device)
    out = torch.zeros_like(new_values)
    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]
    for i in range(total // chunk_size):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ state
        out[:, :, i] = query[:, :, i] @ state + intra[:, :, i] @ v_new
        state = state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new
    out = out.reshape(batch_size, num_v_heads, -1, v_head_dim)[:, :, :sequence_length]
    return out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format)


class Qwen35GatedDeltaNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_v_heads, self.num_k_heads = cfg["linear_num_value_heads"], cfg["linear_num_key_heads"]
        self.head_k_dim, self.head_v_dim = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim
        k = cfg["linear_conv_kernel_dim"]
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, bias=False, kernel_size=k, groups=self.conv_dim,
                                padding=k - 1)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = Qwen35RMSNormGated(self.head_v_dim, eps=cfg["rms_norm_eps"])
        self.out_proj = nn.Linear(self.value_dim, cfg["hidden_size"], bias=False)
        self.in_proj_qkv = nn.Linear(cfg["hidden_size"], self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(cfg["hidden_size"], self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(cfg["hidden_size"], self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(cfg["hidden_size"], self.num_v_heads, bias=False)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        # transformers' causal_conv1d_fn fallback, activation silu
        w = self.conv1d.weight
        mixed_qkv = F.silu(F.conv1d(mixed_qkv.to(w.dtype), w, None, padding=w.shape[-1] - 1,
                                    groups=self.conv_dim)[:, :, :seq_len]).to(hidden_states.dtype)
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        out = chunk_gated_delta_rule(query, key, value, g=g, beta=beta)
        out = self.norm(out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(out.reshape(batch_size, seq_len, -1))


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class Qwen35Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.head_dim = cfg["head_dim"]
        self.num_heads, self.num_kv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        h = cfg["hidden_size"]
        self.q_proj = nn.Linear(h, self.num_heads * self.head_dim * 2, bias=False)     # query and its output gate
        self.k_proj = nn.Linear(h, self.num_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(h, self.num_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, h, bias=False)
        self.q_norm = Qwen35RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])
        self.k_norm = Qwen35RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])

    def forward(self, hidden_states, cos, sin):
        b, s = hidden_states.shape[:2]
        query, gate = torch.chunk(self.q_proj(hidden_states).view(b, s, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(b, s, -1)
        q = self.q_norm(query.reshape(b, s, -1, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(b, s, -1, self.head_dim)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, s, -1, self.head_dim).transpose(1, 2)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        r = cos.shape[-1]                                  # partial rotary: the first 64 of 256 dims
        q = torch.cat([q[..., :r] * cos + _rotate_half(q[..., :r]) * sin, q[..., r:]], dim=-1)
        k = torch.cat([k[..., :r] * cos + _rotate_half(k[..., :r]) * sin, k[..., r:]], dim=-1)
        rep = self.num_heads // self.num_kv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=s > 1)
        out = out.transpose(1, 2).reshape(b, s, -1)
        return self.o_proj(out * torch.sigmoid(gate))


class Qwen35MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"], bias=False)
        self.up_proj = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(cfg["intermediate_size"], cfg["hidden_size"], bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen35DecoderLayer(nn.Module):
    def __init__(self, cfg, kind):
        super().__init__()
        self.kind = kind
        if kind == "linear_attention":
            self.linear_attn = Qwen35GatedDeltaNet(cfg)
        else:
            self.self_attn = Qwen35Attention(cfg)
        self.mlp = Qwen35MLP(cfg)
        self.input_layernorm = Qwen35RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.post_attention_layernorm = Qwen35RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])

    def forward(self, x, cos, sin):
        h = self.input_layernorm(x)
        h = self.linear_attn(h) if self.kind == "linear_attention" else self.self_attn(h, cos, sin)
        x = x + h
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen35TextModel(nn.Module):
    def __init__(self, cfg=QWEN35_9B):
        super().__init__()
        self.cfg = dict(cfg)
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList([Qwen35DecoderLayer(cfg, k) for k in layer_types(cfg)])
        self.norm = Qwen35RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])

    def rope(self, length, device, dtype):
        """Text-only positions 0..L-1 on all three mRoPE axes, so the interleaved sections pick identical
        frequencies: plain RoPE over the rotary dims. inv_freq in fp32 (transformers' default init), cos / sin cast
        to the activation dtype as the reference does."""
        dim = int(self.cfg["head_dim"] * self.cfg["partial_rotary_factor"])
        inv_freq = 1.0 / (self.cfg["rope_theta"] ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        freqs = torch.arange(length, dtype=torch.float32)[:, None] * inv_freq[None]
        emb = torch.cat((freqs, freqs), dim=-1)[None].to(device)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        cos, sin = self.rope(input_ids.shape[1], x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)


class _Wrapper(nn.Module):
    """language_model.* naming, as the checkpoints store it."""

    def __init__(self, cfg=QWEN35_9B):
        super().__init__()
        self.language_model = Qwen35TextModel(cfg)


def read_text_state_dict(path, dtype=torch.bfloat16) -> dict:
    """language_model.* tensors from Comfy-Org's single file (model.language_model.*), MOSS-Image's shards
    (language_model.*) or ComfyUI's int8 ConvRot repack (decoded). The vision tower, LM head and MTP head are left
    out. Cast to `dtype`, as the reference's from_pretrained(dtype=bfloat16)."""
    from safetensors import safe_open
    from training.families.moss_image.model import _files, decode_convrot
    sd = {}
    for f in _files(path):
        with safe_open(f, "pt") as fh:
            for k in fh.keys():
                name = k[len("model."):] if k.startswith("model.language_model.") else k
                if name.startswith("language_model."):
                    sd[name] = fh.get_tensor(k)
    sd = decode_convrot(sd, dtype)
    if not sd:
        raise ValueError(f"{path}: no Qwen3.5 language-model weights (language_model.* / model.language_model.*)")
    return {k: v.to(dtype) if v.is_floating_point() else v for k, v in sd.items()}


STREAM_TOKENS = 1 << 18           # a streamed pass's captions: about 2 GB of bf16 hidden states at width 4096


def load_tokenizer():
    """The MOSS-Image tokenizer and chat template (OpenMOSS-Team/MOSS-Image-Base tokenizer/), from the Hugging Face
    cache first so a warm machine runs offline (training/hf_cache.py; the description's helper_files)."""
    from transformers import AutoTokenizer
    from training.hf_cache import from_pretrained_cache_first
    return from_pretrained_cache_first(AutoTokenizer, TOKENIZER_REPO, subfolder="tokenizer")


class MOSSTextEncoder:
    """stream=True (cards without room for the bf16 encoder, about 15 GB): the weights stay on the CPU and encode()
    runs layer-major - each of the 32 layers goes to the GPU once per group of captions and runs every caption in
    the group, as layer-major text caching did in Fizgig's MOSS pull request. The same bf16 ops on the same device,
    so the output is the resident encoder's, bit for bit; one prompt (previews) streams the 15 GB once. int8=True:
    8-bit language-model weights (weight-only, per-group scales) for a small card that must hold it resident -
    Fizgig's driver never asks for it (it streams instead, keeping the exact bf16 OpenMOSS ask for)."""

    def __init__(self, model_path, device="cuda", dtype=torch.bfloat16, int8=False, tokenizer=None, cfg=QWEN35_9B,
                 stream=False):
        with torch.device("meta"):
            wrapper = _Wrapper(cfg)
        sd = read_text_state_dict(model_path, dtype)
        missing, unexpected = wrapper.load_state_dict(sd, strict=False, assign=True)
        if missing or unexpected:
            raise ValueError(f"{model_path}: not the Qwen3.5-9B text encoder - missing {missing[:5]} "
                             f"({len(missing)}), unexpected {unexpected[:5]} ({len(unexpected)})")
        self.device = torch.device(device)
        if int8:
            from training.families.qwen_image21.embedder import _int8_weights
            n = _int8_weights(wrapper, "language_model.layers.", self.device)
            logger.info(f"[text encoder] 8-bit weights: {n} language-model Linears (low-VRAM card); matmuls stay bf16")
        self.stream = bool(stream) and not int8
        model = wrapper.language_model.eval().requires_grad_(False)
        if self.stream:
            model.norm.to(self.device)      # the embedding and the layers stay on the CPU between passes
        else:
            model.to(self.device)
        self.model = model
        self.tokenizer = load_tokenizer() if tokenizer is None else tokenizer

    def unload(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def template(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=True)

    def token_ids(self, prompt: str) -> torch.Tensor:
        return self.tokenizer(self.template(prompt), return_tensors="pt", truncation=True,
                              max_length=MAX_LENGTH)["input_ids"]

    @torch.no_grad()
    def encode_one(self, prompt: str) -> torch.Tensor:
        """prompt -> (L, 4096) bf16 on CPU, L the formatted prompt's real token count (at most 4096)."""
        ids = self.token_ids(prompt).to(self.device)
        return self.model(ids)[0].to(torch.bfloat16).cpu()

    def encode(self, prompts):
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        if not self.stream:
            return [self.encode_one(p) for p in prompts]
        ids = [self.token_ids(p) for p in prompts]
        out, group, n = [], [], 0
        for i in ids:
            if group and n + i.shape[1] > STREAM_TOKENS:
                out += self._streamed(group)
                group, n = [], 0
            group.append(i)
            n += i.shape[1]
        return out + (self._streamed(group) if group else [])

    @torch.no_grad()
    def _streamed(self, ids):
        """One layer-major pass: every caption through layer 0, then layer 1, ... (Qwen35TextModel.forward per
        caption, the layers moved to the GPU in turn)."""
        m = self.model
        xs = [m.embed_tokens(i).to(self.device) for i in ids]
        ropes = [m.rope(x.shape[1], x.device, x.dtype) for x in xs]
        for layer in m.layers:
            layer.to(self.device)
            xs = [layer(x, c, s) for x, (c, s) in zip(xs, ropes)]
            layer.to("cpu")
        return [m.norm(x)[0].to(torch.bfloat16).cpu() for x in xs]
