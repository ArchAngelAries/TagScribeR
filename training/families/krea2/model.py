# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/krea2/model.py (+ the config and loader of
# krea2/utils.py, the attention of krea2/attention.py).
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: einops is replaced by plain torch reshapes (not installed here); attention goes through
# training/families/krea2/attention.py (Fizgig's torch-SDPA mode with its uniform-length trim; the other backends are
# unreachable for Krea 2 there); block swap uses training.modules.offloading.ModelOffloader the way the Qwen DiT does;
# the Repair Studio activation cache (forward_cached) is not ported; the loader reads bf16 and pre-quantised fp8 checkpoints (training/modules/fp8.py keeps fp8 weights fp8).
# Fizgig's own header follows.
#
# Upstream: the backbone is ported from ai-toolkit (Ostris, LLC - MIT;
# https://github.com/ostris/ai-toolkit, extensions_built_in/diffusion_models/krea2/src/mmdit.py), plus musubi-tuner
# training hooks (gradient checkpointing, block swap - Apache-2.0). See THIRD_PARTY_NOTICES.md.
"""Krea 2 (K2) single-stream MMDiT, 12.9B: 28 identical blocks over [image tokens | text tokens], a text-fusion
transformer that folds the text encoder's 12-layer hidden-state stack into one stream, and the 3-axis RoPE."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor

from training.families.krea2.attention import AttentionParams, attention as common_attention

logger = logging.getLogger(__name__)


@dataclass
class SingleMMDiTConfig:
    features: int
    tdim: int
    txtdim: int
    heads: int
    multiplier: int
    layers: int
    patch: int
    channels: int
    bias: bool = False
    theta: float = 1e3
    kvheads: int | None = None
    txtlayers: int = 1
    txtheads: int = 20
    txtkvheads: int = 20


# The single config shipped with the OSS checkpoints (single_mmdit_large_wide), Fizgig krea2/utils.py.
KREA2_CONFIG = SingleMMDiTConfig(
    features=6144, tdim=256, txtdim=2560, heads=48, kvheads=12, multiplier=4, layers=28, patch=2, channels=16,
    txtheads=20, txtkvheads=20, txtlayers=12,
)


def rope(pos: Tensor, dim: int, theta: float = 1e4, ntk: float = 1.0) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / ((theta * ntk) ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = out.reshape(*out.shape[:-1], 2, 2)                      # b n d (i j) -> b n d i j
    return out.float()


def ropeapply(xq: Tensor, xk: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    freqs = freqs[:, None, :, :, :]
    xq_ = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    xk_ = freqs[..., 0] * xk_[..., 0] + freqs[..., 1] * xk_[..., 1]
    return xq_.reshape(*xq.shape).to(xq.dtype), xk_.reshape(*xk.shape).to(xk.dtype)


def temb(t: Tensor, dim: int, period: float = 1e4, tfactor: float = 1e3, device: torch.device = None,
         dtype: torch.dtype = None) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(period) * torch.arange(half, dtype=torch.float32, device=device) / half)
    # t: (B,) -> args: (B, 1, half), so the embedding broadcasts as a per-sample vec.
    args = (t.float() * tfactor)[:, None, None] * freqs
    sin, cos = torch.sin(args), torch.cos(args)
    return torch.cat((cos, sin), dim=-1).to(dtype=dtype)


class SimpleModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(2, dim))
        self.multiplier = 2

    def forward(self, vec: Tensor):                                # vec (b, 1, d)
        out = vec + self.lin[None]                                 # "two d -> 1 two d"
        scale, shift = out.chunk(self.multiplier, dim=1)
        return scale, shift


class DoubleSharedModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(6 * dim))

    def forward(self, vec: Tensor):                                # vec (b, 1, 6d)
        out = vec + self.lin
        prescale, preshift, pregate, postscale, postshift, postgate = out.chunk(6, dim=-1)
        return prescale, preshift, pregate, postscale, postshift, postgate


class PositionalEncoding(nn.Module):
    def __init__(self, dim, axdims: list[int], theta: float = 1e2, ntk: float = 1.0):
        super().__init__()
        self.axdims = axdims            # how to split the head dimension across the position axes
        self.theta = theta
        self.ntk = ntk

    def forward(self, pos: Tensor) -> Tensor:
        return torch.cat([rope(pos[..., i], d, self.theta, self.ntk) for i, d in enumerate(self.axdims)], dim=-3)


class RMSNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-05, device: torch.device = None):
        super().__init__()
        self.features = features
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(features, device=device, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        t, dtype = x.float(), x.dtype
        t = F.rms_norm(t, (self.features,), eps=self.eps, weight=(self.scale.float() + 1.0))
        return t.to(dtype)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.qnorm(q), self.knorm(k), v


class SwiGLU(nn.Module):
    def __init__(self, features: int, multiplier: int, bias: bool = False, multiple: int = 128):
        super().__init__()
        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
        self.gate = nn.Linear(features, mlpdim, bias=bias)
        self.up = nn.Linear(features, mlpdim, bias=bias)
        self.down = nn.Linear(mlpdim, features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: int = None, bias: bool = False):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads
        self.wq = nn.Linear(dim, self.headdim * self.heads, bias=bias)
        self.wk = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim)
        self.wo = nn.Linear(dim, dim, bias=bias)

    def forward(self, qkv: Tensor, freqs: Tensor | None = None, attn_params: AttentionParams | None = None) -> Tensor:
        """attn_params: the forward's AttentionParams (key-padding mask, uniform-length trim), or None."""
        q, k, v, gate = self.wq(qkv), self.wk(qkv), self.wv(qkv), self.gate(qkv)
        # QKNorm + RoPE run in [B, H, L, D] (K2-native layout) to preserve the reference numerics.
        q = q.unflatten(-1, (self.heads, -1)).transpose(1, 2)
        k = k.unflatten(-1, (self.kvheads, -1)).transpose(1, 2)
        v = v.unflatten(-1, (self.kvheads, -1)).transpose(1, 2)
        q, k, v = self.qknorm(q, k, v)
        if freqs is not None:
            q, k = ropeapply(q, k, freqs)
        # The shared attention expects [B, L, H, D] and returns [B, L, H*D]; GQA (heads != kvheads) is handled inside it.
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        x = common_attention(q, k, v, attn_params=attn_params)
        return self.wo(x * torch.sigmoid(gate))


class LastLayer(nn.Module):
    def __init__(self, features: int, patch: int, channels: int):
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x: Tensor, tvec: Tensor) -> Tensor:
        scale, shift = self.modulation(tvec)
        x = (1 + scale) * self.norm(x) + shift
        return self.linear(x)


class TextFusionBlock(nn.Module):
    def __init__(self, features: int, heads: int, multiplier: int, bias: bool = False, kvheads: int = None):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: Tensor, attn_params: AttentionParams | None = None) -> Tensor:
        x = x + self.attn(self.prenorm(x), attn_params=attn_params)
        return x + self.mlp(self.postnorm(x))


class TextFusionTransformer(nn.Module):
    # num_txt_layers is the number of selected encoder hidden-state layers fed in (projected down to 1), NOT the
    # transformer depth - that is fixed at 2 + 2 blocks.
    def __init__(self, num_txt_layers: int, txt_dim: int, heads: int, multiplier: int, bias: bool = False,
                 kvheads: int = None):
        super().__init__()
        self.layerwise_blocks = nn.ModuleList(
            [TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)])
        self.projector = nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList(
            [TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)])

    def forward(self, x: Tensor, attn_params_nomask: AttentionParams | None = None,
                attn_params: AttentionParams | None = None) -> Tensor:
        """x (b, l, n, d): b prompts, l tokens, n encoder layers. The per-layer blocks flatten to (b*l, n, d) and
        attend across the layer stack for each token (the token count is their BATCH, so a key-padding mask does not
        apply to them); the refiner attends the token sequence itself and masks the padding."""
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous(), attn_params=attn_params_nomask)
        x = x.reshape(b, l, n, d).permute(0, 1, 3, 2)              # (b l) n d -> b l d n
        x = self.projector(x).squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x, attn_params=attn_params)
        return x


class SingleStreamBlock(nn.Module):
    def __init__(self, features: int, heads: int, multiplier: int, bias: bool = False, kvheads: int = None):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: Tensor, vec: Tensor, freqs: Tensor, attn_params: AttentionParams | None = None) -> Tensor:
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, freqs, attn_params)
        return x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)


class SingleStreamDiT(nn.Module):
    def __init__(self, config: SingleMMDiTConfig):
        super().__init__()
        self.config = config
        headdim = config.features // config.heads
        axes = [headdim - 12 * (headdim // 16), 6 * (headdim // 16), 6 * (headdim // 16)]
        assert sum(axes) == headdim, f"sum(axes) = {sum(axes)}, headdim = {headdim}"
        assert all(a % 2 == 0 for a in axes), f"axes = {axes}"
        self.posemb = PositionalEncoding(config.features, axes, theta=config.theta, ntk=1.0)
        self.first = nn.Linear(config.channels * config.patch ** 2, config.features, bias=True)
        self.blocks = nn.ModuleList([
            SingleStreamBlock(config.features, config.heads, config.multiplier, config.bias, config.kvheads)
            for _ in range(config.layers)])
        self.tmlp = nn.Sequential(nn.Linear(config.tdim, config.features), nn.GELU(approximate="tanh"),
                                  nn.Linear(config.features, config.features))
        self.txtfusion = TextFusionTransformer(config.txtlayers, config.txtdim, config.txtheads, config.multiplier,
                                               config.bias, config.txtkvheads)
        self.txtmlp = nn.Sequential(RMSNorm(config.txtdim), nn.Linear(config.txtdim, config.features),
                                    nn.GELU(approximate="tanh"), nn.Linear(config.features, config.features))
        self.last = LastLayer(config.features, config.patch, config.channels)
        self.tproj = nn.Sequential(nn.GELU(approximate="tanh"), nn.Linear(config.features, config.features * 6))
        self.gradient_checkpointing = False
        self.blocks_to_swap = 0
        self.offloader = None

    def enable_gradient_checkpointing(self, on: bool = True):
        self.gradient_checkpointing = on

    # ---- block swap (Fizgig's shared offloader; the main blocks only) --------------------------------------------
    def enable_block_swap(self, num_blocks: int, device, supports_backward: bool = True):
        from training.modules.offloading import ModelOffloader
        if self.offloader is not None:
            self.offloader.remove_hooks()      # stale backward hooks double-swap blocks ("mat2 is on cpu")
        n = len(self.blocks)
        if not 0 < num_blocks <= n - 2:
            raise ValueError(f"block swap: 1..{n - 2} blocks, got {num_blocks}")
        self.blocks_to_swap = num_blocks
        self.offloader = ModelOffloader("krea2", list(self.blocks), n, num_blocks, supports_backward,
                                        torch.device(device))

    def move_to_device_except_swap_blocks(self, device):
        """Everything to `device` except the swapped blocks' weights (the model is assumed to be on CPU). try/finally:
        an OOM inside .to() must not leave the model holding the empty placeholder ModuleList (a silent lobotomy)."""
        blocks = self.blocks
        if self.blocks_to_swap:
            self.blocks = nn.ModuleList()
        try:
            self.to(device)
        finally:
            self.blocks = blocks
        self.prepare_block_swap_before_forward()

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap:
            self.offloader.prepare_block_devices_before_forward(list(self.blocks))

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()

    def forward(self, img: Tensor, context: Tensor, t: Tensor, pos: Tensor, mask: Tensor | None = None) -> Tensor:
        """img (B, N, c*p*p) patchified latents; context (B, L, layers, txtdim) the text stack; t (B,) in [0, 1];
        pos (B, N + L, 3) RoPE positions; mask (B, N + L) bool, True = a real token. The sequence is image-first:
        [img (all valid), text (valid tokens + padding)]. Returns (B, N, c*p*p)."""
        img = self.first(img)
        t = self.tmlp(temb(t, self.config.tdim, device=img.device, dtype=img.dtype))
        tvec = self.tproj(t)

        imglen = img.shape[1]
        txtmask = mask[:, imglen:]                                  # (B, txt_len) bool
        # Fizgig notes (krea2/model.py): padding the text is NOT numerically inert in the text-fusion stage (its
        # per-layer blocks carry the text length in their batch dimension) - sampling.gather_valid_text documents
        # the accepted perturbation.
        txt_attn_params_nomask = AttentionParams.create_attention_params_from_mask(0, None)
        txt_attn_params = AttentionParams.create_attention_params_from_mask(0, txtmask)
        context = self.txtfusion(context, txt_attn_params_nomask, txt_attn_params)
        context = self.txtmlp(context)

        combined = torch.cat((img, context), dim=1)                 # image first, then text

        # Pad the combined sequence to a multiple of 256 (Fizgig: stable kernel shapes). The pad lands on the text
        # tail and is masked, so it is numerically inert.
        padlen = (-combined.shape[1]) % 256
        if padlen > 0:
            combined = F.pad(combined, (0, 0, 0, padlen))
            pos = F.pad(pos, (0, 0, 0, padlen))
            txtmask = F.pad(txtmask, (0, padlen), value=False)
        # bidirectional attention over [image (all valid) + text (padded)]; image-first keeps each sample's valid tokens a
        # contiguous prefix, which the uniform-length trim relies on
        attn_params = AttentionParams.create_attention_params_from_mask(imglen, txtmask)

        freqs = self.posemb(pos)
        blocks = list(self.blocks) if self.blocks_to_swap else None
        for index, block in enumerate(self.blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(index)
            if getattr(block, "_handles_checkpointing", False):
                # torch.compile wraps blocks in a module that checkpoints itself, so the recompute is captured inside
                # the compiled graph; checkpointing again here would nest it (Fizgig krea2/model.py)
                combined = block(combined, tvec, freqs, attn_params)
            elif torch.is_grad_enabled() and self.gradient_checkpointing:
                combined = torch.utils.checkpoint.checkpoint(block, combined, tvec, freqs, attn_params,
                                                             use_reentrant=False)
            else:
                combined = block(combined, tvec, freqs, attn_params)
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(blocks, index)

        final = self.last(combined, t)
        return final[:, :imglen, :]                                 # image tokens are the leading slice


# ---- weights ---------------------------------------------------------------------------------------------------


def load_krea2_dit(path, device="cuda", dtype=torch.bfloat16, config: SingleMMDiTConfig = KREA2_CONFIG):
    """Build the DiT on meta and load a RAW checkpoint (assign=True, strict): bf16, or a pre-quantised fp8 /
    fp8-scaled file, whose Linear weights stay fp8 with their scales and are dequantised per matmul (Fizgig
    krea2/utils.py load_krea2_dit, pre-quantised branch). A sharded checkpoint (...-00001-of-0000N.safetensors) is
    read whole. The base precision (fp8 / INT8 / NF4 / bf16) is applied afterwards by training/quant.py, which
    accepts either kind of file as its source."""
    from training.families.qwen_image21.embedder import load_split_weights
    from training.modules import fp8
    path = str(path)
    with torch.device("meta"):
        dit = SingleStreamDiT(config)
    sd = load_split_weights(path)
    if sd and all(k.startswith("diffusion_model.") for k in sd):
        sd = {k[len("diffusion_model."):]: v for k, v in sd.items()}
    missing, unexpected, _n = fp8.load_state_dict(dit, sd, dtype)
    if missing or unexpected:
        raise ValueError(f"Krea 2 DiT keys mismatch: missing {missing[:8]} ({len(missing)}), unexpected "
                         f"{unexpected[:8]} ({len(unexpected)}) - is this the Krea 2 RAW checkpoint?")
    return dit.to(device)
