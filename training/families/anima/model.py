# Architecture facts from NVIDIA's Cosmos-Predict2 "MiniTrainDIT" (Apache-2.0, Copyright (c) 2025 NVIDIA CORPORATION &
# AFFILIATES; as vendored in diffusion-pipe models/cosmos_predict2_modeling.py and kohya sd-scripts
# library/anima_models.py, Apache-2.0) and the Anima "LLM adapter" (diffusion-pipe models/llm_adapter.py, Apache-2.0
# file header; sd-scripts library/anima_models.py, Apache-2.0). The module and parameter NAMES are the checkpoint's;
# the code below is a compact re-implementation for TagScribeR, not a copy: plain torch reshapes instead of einops /
# torchvision, no Transformer-Engine / context-parallel / fps paths, RoPE computed on the fly (no buffers, so the
# model builds on the meta device), T = 1 only (an image model), SDPA attention, per-block gradient checkpointing.
# See THIRD_PARTY_NOTICES.md.
"""Anima DiT: Cosmos-Predict2-2B (28 AdaLN-LoRA blocks, 2048 wide) + the 6-layer LLM adapter that turns the Qwen3-0.6B
hidden states into the T5-vocabulary-conditioned cross-attention context.

forward(latents (B, 16, 1, h, w), t (B,) in [0, 1], context = Qwen3 hidden states (B, L, 1024), t5_ids, t5_mask,
qwen_mask) -> velocity (B, 16, 1, h, w). The adapter runs INSIDE forward (frozen: the Anima model card says never to
train it), so conditioning is cached as plain Qwen3 states + T5 token ids.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


@dataclass
class AnimaConfig:
    model_channels: int = 2048
    num_blocks: int = 28
    num_heads: int = 16
    in_channels: int = 16
    out_channels: int = 16
    patch_spatial: int = 2
    mlp_ratio: float = 4.0
    adaln_lora_dim: int = 256
    rope_h_extrapolation: float = 4.0
    rope_w_extrapolation: float = 4.0
    rope_t_extrapolation: float = 1.0
    # LLM adapter (crossattn_emb_channels == adapter_target_dim: the DiT's cross-attention context width)
    adapter_vocab: int = 32128
    adapter_source_dim: int = 1024
    adapter_target_dim: int = 1024
    adapter_model_dim: int = 1024
    adapter_layers: int = 6
    adapter_heads: int = 16


# sd-scripts library/anima_utils.py load_anima_model "We currently support fixed DiT config for Anima models"
ANIMA_CONFIG = AnimaConfig()


# ---- shared pieces ---------------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """x * rsqrt(mean(x^2) + eps) * weight, the variance in fp32 (the DiT's norm and the adapter's T5-style norm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return out.to(x.dtype) * self.weight


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x, cos, sin):
    """x (B, S, H, D); cos / sin (S, D) fp32 -> rotated, in x's dtype (cos / sin cast after the trig, as the original)."""
    cos = cos[None, :, None, :].to(x.dtype)
    sin = sin[None, :, None, :].to(x.dtype)
    return x * cos + _rotate_half(x) * sin


def _attend(q, k, v, mask=None):
    """q (B, Sq, H, D), k / v (B, Sk, H, D), mask (B, 1, 1, Sk) bool (True = attend) -> (B, Sq, H * D)."""
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[1], -1)


# ---- the DiT ---------------------------------------------------------------------------------------------------
def rope_3d(t_len, h_len, w_len, head_dim, h_ext, w_ext, t_ext, device):
    """Cosmos 3-axis RoPE angles (t, h, w) for a (T, H, W) token grid, flattened (t h w): (T*H*W, head_dim) fp32. Axis
    split dim_h = dim_w = head_dim // 6 * 2, dim_t = the rest; NTK-style theta = 10000 * ratio ** (dim / (dim - 2))."""
    dim_h = head_dim // 6 * 2
    dim_t = head_dim - 2 * dim_h

    def freqs(dim, ratio):
        rng = torch.arange(0, dim, 2, device=device)[: dim // 2].float() / dim
        return 1.0 / ((10000.0 * ratio ** (dim / (dim - 2))) ** rng)
    th = torch.outer(torch.arange(t_len, device=device).float(), freqs(dim_t, t_ext))
    hh = torch.outer(torch.arange(h_len, device=device).float(), freqs(dim_h, h_ext))
    ww = torch.outer(torch.arange(w_len, device=device).float(), freqs(dim_h, w_ext))
    em = torch.cat([th[:, None, None, :].expand(-1, h_len, w_len, -1),
                    hh[None, :, None, :].expand(t_len, -1, w_len, -1),
                    ww[None, None, :, :].expand(t_len, h_len, -1, -1)] * 2, dim=-1)
    return em.reshape(t_len * h_len * w_len, head_dim)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, query_dim // n_heads
        inner = self.head_dim * n_heads
        self.q_proj = nn.Linear(query_dim, inner, bias=False)
        self.q_norm = RMSNorm(self.head_dim, 1e-6)
        self.k_proj = nn.Linear(context_dim, inner, bias=False)
        self.k_norm = RMSNorm(self.head_dim, 1e-6)
        self.v_proj = nn.Linear(context_dim, inner, bias=False)
        self.output_proj = nn.Linear(inner, query_dim, bias=False)

    def forward(self, x, context=None, rope=None):
        ctx = x if context is None else context
        b, s, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, s, self.n_heads, self.head_dim))
        k = self.k_norm(self.k_proj(ctx).view(b, ctx.shape[1], self.n_heads, self.head_dim))
        v = self.v_proj(ctx).view(b, ctx.shape[1], self.n_heads, self.head_dim)
        if rope is not None:                                   # self-attention only
            cos, sin = rope
            q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        return self.output_proj(_attend(q, k, v))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.layer1 = nn.Linear(dim, hidden, bias=False)
        self.layer2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.layer2(F.gelu(self.layer1(x)))


def _modulation(dim, lora_dim, chunks):
    return nn.Sequential(nn.SiLU(), nn.Linear(dim, lora_dim, bias=False), nn.Linear(lora_dim, chunks * dim, bias=False))


class Block(nn.Module):
    """Self-attention, cross-attention and MLP, each AdaLN-modulated (shift, scale, gate) with AdaLN-LoRA."""

    def __init__(self, dim, context_dim, heads, mlp_ratio, lora_dim):
        super().__init__()
        self.layer_norm_self_attn = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(dim, dim, heads)
        self.layer_norm_cross_attn = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(dim, context_dim, heads)
        self.layer_norm_mlp = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio))
        self.adaln_modulation_self_attn = _modulation(dim, lora_dim, 3)
        self.adaln_modulation_cross_attn = _modulation(dim, lora_dim, 3)
        self.adaln_modulation_mlp = _modulation(dim, lora_dim, 3)

    def forward(self, x, emb, context, rope, adaln_lora):
        """x (B, L, D); emb (B, 1, D); context (B, N, D_ctx); adaln_lora (B, 1, 3D)."""
        sh1, sc1, g1 = (self.adaln_modulation_self_attn(emb) + adaln_lora).chunk(3, dim=-1)
        sh2, sc2, g2 = (self.adaln_modulation_cross_attn(emb) + adaln_lora).chunk(3, dim=-1)
        sh3, sc3, g3 = (self.adaln_modulation_mlp(emb) + adaln_lora).chunk(3, dim=-1)
        x = x + g1 * self.self_attn(self.layer_norm_self_attn(x) * (1 + sc1) + sh1, rope=rope)
        x = x + g2 * self.cross_attn(self.layer_norm_cross_attn(x) * (1 + sc2) + sh2, context=context)
        return x + g3 * self.mlp(self.layer_norm_mlp(x) * (1 + sc3) + sh3)


class PatchEmbed(nn.Module):
    def __init__(self, patch, in_channels, out_channels):
        super().__init__()
        self.patch = patch
        # `proj` is a Sequential in the checkpoint (a Rearrange, then the Linear): the weight key is proj.1.weight
        self.proj = nn.Sequential(nn.Identity(), nn.Linear(in_channels * patch * patch, out_channels, bias=False))

    def forward(self, x):
        """(B, C, 1, H, W) -> (B, H/p, W/p, D); each patch is flattened (c, m, n) as 'b c (h m) (w n) -> b h w (c m n)'."""
        b, c, _, h, w = x.shape
        p = self.patch
        x = x.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5).reshape(b, h // p, w // p, c * p * p)
        return self.proj[1](x)


class TimestepEmbedding(nn.Module):
    """Sinusoidal -> (the sinusoid itself, an AdaLN-LoRA modulation (B, 1, 3D)) - the Cosmos AdaLN-LoRA variant."""

    def __init__(self, dim):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim, bias=False)
        self.linear_2 = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, sample):
        h = sample.to(self.linear_1.weight.dtype)           # the original runs these Linears under bf16 autocast
        return sample, self.linear_2(F.silu(self.linear_1(h)))


class Timesteps(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """t (B,) fp32 -> (B, 1, dim): cos | sin of t * exp(-ln(10000) * i / (dim / 2)); t is already in [0, 1]."""
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        emb = t.float()[:, None] * freqs[None, :]
        return torch.cat([emb.cos(), emb.sin()], dim=-1)[:, None, :]


class FinalLayer(nn.Module):
    def __init__(self, dim, patch, out_channels, lora_dim):
        super().__init__()
        self.dim = dim
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch * patch * out_channels, bias=False)
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, lora_dim, bias=False),
                                              nn.Linear(lora_dim, 2 * dim, bias=False))

    def forward(self, x, emb, adaln_lora):
        shift, scale = (self.adaln_modulation(emb) + adaln_lora[:, :, : 2 * self.dim]).chunk(2, dim=-1)
        return self.linear(self.layer_norm(x) * (1 + scale) + shift)


# ---- the LLM adapter ---------------------------------------------------------------------------------------------
class AdapterAttention(nn.Module):
    """Same names as the DiT attention except the output projection (`o_proj`); RoPE on the query AND key."""

    def __init__(self, query_dim, context_dim, n_heads):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, query_dim // n_heads
        inner = self.head_dim * n_heads
        self.q_proj = nn.Linear(query_dim, inner, bias=False)
        self.q_norm = RMSNorm(self.head_dim, 1e-6)
        self.k_proj = nn.Linear(context_dim, inner, bias=False)
        self.k_norm = RMSNorm(self.head_dim, 1e-6)
        self.v_proj = nn.Linear(context_dim, inner, bias=False)
        self.o_proj = nn.Linear(inner, query_dim, bias=False)

    def forward(self, x, mask, context, rope_q, rope_k):
        b, s, _ = x.shape
        sk = context.shape[1]
        q = self.q_norm(self.q_proj(x).view(b, s, self.n_heads, self.head_dim))
        k = self.k_norm(self.k_proj(context).view(b, sk, self.n_heads, self.head_dim))
        v = self.v_proj(context).view(b, sk, self.n_heads, self.head_dim)
        q, k = _apply_rope(q, *rope_q), _apply_rope(k, *rope_k)
        return self.o_proj(_attend(q, k, v, mask))


class AdapterBlock(nn.Module):
    def __init__(self, source_dim, dim, heads):
        super().__init__()
        self.norm_self_attn = RMSNorm(dim)
        self.self_attn = AdapterAttention(dim, dim, heads)
        self.norm_cross_attn = RMSNorm(dim)
        self.cross_attn = AdapterAttention(dim, source_dim, heads)
        self.norm_mlp = RMSNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, context, target_mask, source_mask, rope_target, rope_source):
        n = self.norm_self_attn(x)
        x = x + self.self_attn(n, target_mask, n, rope_target, rope_target)
        n = self.norm_cross_attn(x)
        x = x + self.cross_attn(n, source_mask, context, rope_target, rope_source)
        return x + self.mlp(self.norm_mlp(x))


def _adapter_rope(length, head_dim, device):
    inv = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(length, device=device).float(), inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


class LLMAdapter(nn.Module):
    """T5-vocabulary token embeddings (the 'target' sequence) that cross-attend to the Qwen3 hidden states (the
    'source'): the model's own text-conditioning bridge. Output (B, T5 length, target_dim)."""

    def __init__(self, cfg: AnimaConfig):
        super().__init__()
        self.head_dim = cfg.adapter_model_dim // cfg.adapter_heads
        self.embed = nn.Embedding(cfg.adapter_vocab, cfg.adapter_target_dim)
        self.in_proj = (nn.Identity() if cfg.adapter_model_dim == cfg.adapter_target_dim
                        else nn.Linear(cfg.adapter_target_dim, cfg.adapter_model_dim))
        self.blocks = nn.ModuleList([AdapterBlock(cfg.adapter_source_dim, cfg.adapter_model_dim, cfg.adapter_heads)
                                     for _ in range(cfg.adapter_layers)])
        self.out_proj = nn.Linear(cfg.adapter_model_dim, cfg.adapter_target_dim)
        self.norm = RMSNorm(cfg.adapter_target_dim)

    def forward(self, source, target_ids, target_mask, source_mask):
        """source (B, Ls, source_dim); target_ids (B, Lt) long; masks (B, L) bool (True = real token)."""
        tmask = target_mask.bool()[:, None, None, :]
        smask = source_mask.bool()[:, None, None, :]
        x = self.in_proj(self.embed(target_ids))
        rope_t = _adapter_rope(x.shape[1], self.head_dim, x.device)
        rope_s = _adapter_rope(source.shape[1], self.head_dim, x.device)
        for block in self.blocks:
            x = block(x, source, tmask, smask, rope_t, rope_s)
        return self.norm(self.out_proj(x))


# ---- the model -------------------------------------------------------------------------------------------------
class AnimaDiT(nn.Module):
    def __init__(self, cfg: AnimaConfig = ANIMA_CONFIG):
        super().__init__()
        self.cfg = cfg
        d = cfg.model_channels
        assert (d // cfg.num_heads) % 2 == 0 and (d // cfg.num_heads) // 6 * 2 >= 4, "head dim >= 12 (RoPE axis split)"
        # + 1 channel: the all-zero padding mask the model concatenates to its input
        self.x_embedder = PatchEmbed(cfg.patch_spatial, cfg.in_channels + 1, d)
        self.t_embedder = nn.Sequential(Timesteps(d), TimestepEmbedding(d))
        self.t_embedding_norm = RMSNorm(d, 1e-6)
        self.blocks = nn.ModuleList([Block(d, cfg.adapter_target_dim, cfg.num_heads, cfg.mlp_ratio, cfg.adaln_lora_dim)
                                     for _ in range(cfg.num_blocks)])
        self.final_layer = FinalLayer(d, cfg.patch_spatial, cfg.out_channels, cfg.adaln_lora_dim)
        self.llm_adapter = LLMAdapter(cfg)
        self.gradient_checkpointing = False
        # the adapter holds knowledge the model card says not to disturb: it is frozen and no LoRA targets it, so its
        # forward needs no graph (saves its activations); set False if a future option trains it
        self.adapter_no_grad = True

    @property
    def dtype(self):
        return self.t_embedding_norm.weight.dtype

    def enable_gradient_checkpointing(self, on: bool = True):
        self.gradient_checkpointing = on

    def encode_context(self, context, t5_ids, t5_mask, qwen_mask):
        """The cross-attention context: adapter output, zeroed where the T5 side is padding, (B, Lt, target_dim)."""
        def run():
            out = self.llm_adapter(context, t5_ids, t5_mask, qwen_mask)
            return out.masked_fill(~t5_mask.bool()[..., None], 0.0)
        if self.adapter_no_grad:
            with torch.no_grad():
                return run()
        return run()

    def forward(self, x, t, context, t5_ids, t5_mask, qwen_mask):
        """x (B, C, 1, h, w) noisy latents; t (B,) in [0, 1] (the flow time = sigma, NOT x1000); context (B, Ls, 1024)
        Qwen3 states; t5_ids / t5_mask (B, Lt); qwen_mask (B, Ls). -> (B, C, 1, h, w)."""
        cfg = self.cfg
        b, c, frames, h, w = x.shape
        assert frames == 1, "Anima trains and samples single images (T = 1)"
        pad = torch.zeros(b, 1, 1, h, w, dtype=x.dtype, device=x.device)       # the padding-mask channel (all zeros)
        tokens = self.x_embedder(torch.cat([x, pad], dim=1))                    # (B, h/p, w/p, D)
        hp, wp = tokens.shape[1], tokens.shape[2]
        tokens = tokens.reshape(b, hp * wp, -1)
        cos_sin = rope_3d(1, hp, wp, cfg.model_channels // cfg.num_heads, cfg.rope_h_extrapolation,
                          cfg.rope_w_extrapolation, cfg.rope_t_extrapolation, x.device)
        rope = (cos_sin.cos(), cos_sin.sin())
        sinus, adaln_lora = self.t_embedder(t.float())
        emb = self.t_embedding_norm(sinus).to(tokens.dtype)                    # the norm runs on the fp32 sinusoid
        adaln_lora = adaln_lora.to(tokens.dtype)
        ctx = self.encode_context(context, t5_ids, t5_mask, qwen_mask).to(tokens.dtype)
        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                tokens = torch.utils.checkpoint.checkpoint(block, tokens, emb, ctx, rope, adaln_lora,
                                                           use_reentrant=False)
            else:
                tokens = block(tokens, emb, ctx, rope, adaln_lora)
        out = self.final_layer(tokens, emb, adaln_lora)                         # (B, hp*wp, p*p*C)
        p = cfg.patch_spatial
        # unpatchify 'B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)' with t = 1
        out = out.reshape(b, hp, wp, p, p, cfg.out_channels).permute(0, 5, 1, 3, 2, 4)
        return out.reshape(b, cfg.out_channels, 1, hp * p, wp * p)


# ---- weights ---------------------------------------------------------------------------------------------------
def refuse_fp8(path: str) -> None:
    """Anima ships bf16; a pre-quantised fp8 file would load with its per-layer scales dropped (silently wrong)."""
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if k.endswith((".weight_scale", ".scale_weight", ".comfy_quant")) or "F8" in str(f.get_slice(k).get_dtype()):
                raise ValueError(f"{path} is a pre-quantised fp8 checkpoint (found {k!r}); Anima training needs a "
                                 f"bf16 checkpoint such as anima-base-v1.0.safetensors from circlestone-labs/Anima.")


def convert_keys(sd: dict) -> dict:
    """Checkpoint keys -> this model's: strip the `net.` prefix of anima-base-v1.0 and earlier, or the
    `model.diffusion_model.` prefix of the aesthetic / turbo releases (sd-scripts library/anima_utils.py rename_hook)."""
    out = {}
    for k, v in sd.items():
        if k.startswith("net."):
            k = k[len("net."):]
        elif k.startswith("model.diffusion_model."):
            k = k[len("model.diffusion_model."):]
        out[k] = v
    return out


def load_anima_dit(path, device="cuda", dtype=torch.bfloat16, config: AnimaConfig = ANIMA_CONFIG):
    """Build on the meta device and load a bf16 checkpoint (assign=True, strict: any missing or unexpected key aborts
    with the offending names - the architecture facts were verified from the reference trainers, not from the file)."""
    from training.families.qwen_image21.embedder import load_split_weights
    path = str(path)
    refuse_fp8(path)
    with torch.device("meta"):
        dit = AnimaDiT(config)
    sd = convert_keys(load_split_weights(path))
    sd = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in sd.items()}
    missing, unexpected = dit.load_state_dict(sd, strict=False, assign=True)
    if missing or unexpected:
        raise ValueError(f"Anima DiT keys mismatch: missing {missing[:8]} ({len(missing)}), unexpected "
                         f"{unexpected[:8]} ({len(unexpected)}) - is this an Anima checkpoint "
                         f"(anima-base-v1.0.safetensors)?")
    return dit.to(device)
