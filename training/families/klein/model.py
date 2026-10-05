# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/klein/model.py (the KleinDiT part) and
# klein/model_utils.py (the DiT loader).
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: einops is replaced by plain torch reshapes (not installed here); attention is torch SDPA
# (Fizgig's flash / sageattn / xformers dispatch, split attention and its AttentionParams are not ported - Klein
# attends over the full sequence, no mask); block swap uses training.modules.offloading.ModelOffloader the way the
# other DiTs here do (Fizgig's two-offloader double / single layout and its swap-count formula are kept); the Repair
# Studio activation cache (forward_cached), activation CPU offloading and the fp8 monkey patch are not ported; the
# loader reads bf16 and pre-quantised fp8 checkpoints (fp8 weights stay fp8: training/modules/fp8.py).
# Fizgig's own header follows.
#
# Fizgig-native Klein DiT model
# Based on FLUX repo: https://github.com/black-forest-labs/flux
# License: Apache-2.0 License
"""FLUX.2 Klein Base 9B DiT: 8 double-stream blocks (separate image / text streams that attend jointly) followed by 24
single-stream blocks over [text | image] tokens, 4-axis RoPE (t, h, w, layer) and shared modulation."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from training.modules.sdpa import sdpa_backend_ctx

logger = logging.getLogger(__name__)


# Fizgig klein/model.py:81-82 (the Linears fp8 / NF4 quantise: only the two block stacks; norms, the embedders and
# the modulation layers stay as shipped). Here INT8 / NF4 come from training/quant.py on the driver's block map.
FP8_OPTIMIZATION_TARGET_KEYS = ["double_blocks", "single_blocks"]


@dataclass
class Flux2Params:
    in_channels: int = 128  # packed latent channels
    context_in_dim: int = 15360
    hidden_size: int = 6144
    num_heads: int = 48
    depth: int = 8
    depth_single_blocks: int = 48
    axes_dim: list = field(default_factory=lambda: [32, 32, 32, 32])
    theta: int = 2000
    mlp_ratio: float = 3.0
    use_guidance_embed: bool = True


@dataclass
class Klein9BParams(Flux2Params):
    context_in_dim: int = 12288
    hidden_size: int = 4096
    num_heads: int = 32
    depth: int = 8
    depth_single_blocks: int = 24
    axes_dim: list = field(default_factory=lambda: [32, 32, 32, 32])
    theta: int = 2000
    mlp_ratio: float = 3.0
    use_guidance_embed: bool = False


class KleinDiT(nn.Module):
    def __init__(self, params: Flux2Params) -> None:
        super().__init__()

        self.in_channels = params.in_channels
        self.out_channels = params.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}")
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads

        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=False)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, disable_bias=True)
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size, bias=False)

        self.use_guidance_embed = params.use_guidance_embed
        if self.use_guidance_embed:
            self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, disable_bias=True)

        self.double_blocks = nn.ModuleList(
            [DoubleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
             for _ in range(params.depth)])
        self.single_blocks = nn.ModuleList(
            [SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
             for _ in range(params.depth_single_blocks)])

        self.double_stream_modulation_img = Modulation(self.hidden_size, double=True, disable_bias=True)
        self.double_stream_modulation_txt = Modulation(self.hidden_size, double=True, disable_bias=True)
        self.single_stream_modulation = Modulation(self.hidden_size, double=False, disable_bias=True)

        self.final_layer = LastLayer(self.hidden_size, self.out_channels)

        self.gradient_checkpointing = False
        self.attn_mode = "torch"
        self.blocks_to_swap = None

        self.offloader_double = None
        self.offloader_single = None
        self.num_double_blocks = len(self.double_blocks)
        self.num_single_blocks = len(self.single_blocks)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def set_attn_mode(self, mode: str):
        """Fizgig passes attn_mode to KleinDiT(...) and on to every block via AttentionParams (klein/model.py:463-485,
        707): "torch" (sdpa) or "flash3". Every block reads it, also on a gradient-checkpoint recompute."""
        self.attn_mode = mode
        for block in list(self.double_blocks) + list(self.single_blocks):
            block.attn_mode = mode

    # ---- gradient checkpointing ------------------------------------------------------------------------------
    def enable_gradient_checkpointing(self, on: bool = True):
        """Fizgig enable_gradient_checkpointing / disable_gradient_checkpointing (activation CPU offloading is not
        ported)."""
        self.gradient_checkpointing = bool(on)
        self.time_in.gradient_checkpointing = bool(on)
        if self.use_guidance_embed:
            self.guidance_in.gradient_checkpointing = bool(on)
        for block in list(self.double_blocks) + list(self.single_blocks):
            block.gradient_checkpointing = bool(on)

    # ---- block swap (Fizgig's two offloaders: double blocks and single blocks) --------------------------------
    @staticmethod
    def swap_split(num_blocks: int, num_double: int, num_single: int) -> tuple:
        """Fizgig klein/model.py enable_block_swap's ratio formula: how many double / single blocks `num_blocks`
        swap. One unit of `num_blocks` is about 0.4 of the blocks' weights: 16 (the maximum, Fizgig's GUI range 0-16)
        streams 6 double + 18 single blocks, leaving 2 + 6 resident."""
        if num_blocks <= 0:
            return 0, 0
        if num_double == 0:
            return 0, num_blocks
        if num_single == 0:
            return num_blocks, 0
        swap_ratio = num_single / num_double
        double = int(round(num_blocks / (1.0 + swap_ratio / 2.0)))
        single = int(round(double * swap_ratio))
        # adjust if we exceed available blocks
        if num_double * 2 < num_single:
            while double >= 1 and double > num_double - 2:
                double -= 1
                single += 2
        else:
            while single >= 2 and single > num_single - 2:
                single -= 2
                double += 1
        if double == 0 and single == 0:
            if num_single >= num_double:
                single = 1
            else:
                double = 1
        return double, single

    def max_block_swap(self) -> int:
        """The largest `num_blocks` swap_split accepts (16 for the 8 + 24 layout)."""
        best = 0
        for n in range(1, 4 * (self.num_double_blocks + self.num_single_blocks)):
            d, s = self.swap_split(n, self.num_double_blocks, self.num_single_blocks)
            if d <= self.num_double_blocks - 2 and s <= self.num_single_blocks - 2:
                best = n
            else:
                break
        return best

    def enable_block_swap(self, num_blocks: int, device, supports_backward: bool = True):
        from training.modules.offloading import ModelOffloader
        # Detach any previous offloaders' backward hooks first: stale hooks fire next to the new ones and double-swap
        # blocks ("mat2 is on cpu", Fizgig's note at klein/model.py enable_block_swap).
        for off in (self.offloader_double, self.offloader_single):
            if off is not None:
                off.remove_hooks()
        double, single = self.swap_split(num_blocks, self.num_double_blocks, self.num_single_blocks)
        if double > self.num_double_blocks - 2 or single > self.num_single_blocks - 2:
            raise ValueError(
                f"Cannot swap more than {self.num_double_blocks - 2} double blocks and {self.num_single_blocks - 2} "
                f"single blocks. Requested {double} double blocks and {single} single blocks (blocks to swap "
                f"{num_blocks}; the maximum is {self.max_block_swap()}).")
        self.blocks_to_swap = num_blocks
        device = torch.device(device)
        self.offloader_double = ModelOffloader("double", list(self.double_blocks), self.num_double_blocks, double,
                                               supports_backward, device)
        self.offloader_single = ModelOffloader("single", list(self.single_blocks), self.num_single_blocks, single,
                                               supports_backward, device)
        logger.info(f"KleinDiT: block swap {num_blocks} -> {double} double + {single} single blocks stream "
                    f"between CPU and GPU")

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader_double.set_forward_only(True)
            self.offloader_single.set_forward_only(True)
            self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader_double.set_forward_only(False)
            self.offloader_single.set_forward_only(False)
            self.prepare_block_swap_before_forward()

    def move_to_device_except_swap_blocks(self, device):
        """The model is assumed to be on CPU; everything goes to `device` except the swapped blocks' weights.
        try/finally: an OOM inside .to() must not leave the model holding the empty placeholder ModuleLists."""
        if self.blocks_to_swap:
            save_double, save_single = self.double_blocks, self.single_blocks
            self.double_blocks = nn.ModuleList()
            self.single_blocks = nn.ModuleList()
        try:
            self.to(device)
        finally:
            if self.blocks_to_swap:
                self.double_blocks, self.single_blocks = save_double, save_single
        self.prepare_block_swap_before_forward()

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader_double.prepare_block_devices_before_forward(list(self.double_blocks))
        self.offloader_single.prepare_block_devices_before_forward(list(self.single_blocks))

    # ---- forward ------------------------------------------------------------------------------------------------
    def forward(self, x: Tensor, x_ids: Tensor, timesteps: Tensor, ctx: Tensor, ctx_ids: Tensor,
                guidance: Tensor | None = None) -> Tensor:
        """x (B, N, in_channels) packed image tokens, x_ids (B, N, 4), timesteps (B,) in [0, 1], ctx (B, T,
        context_in_dim), ctx_ids (B, T, 4). Returns (B, N, in_channels). Run it under bf16 autocast (the family driver
        does): Fizgig's trainer runs the DiT under accelerate's mixed precision."""
        num_txt_tokens = ctx.shape[1]

        timestep_emb = timestep_embedding(timesteps, 256)
        del timesteps
        vec = self.time_in(timestep_emb)
        del timestep_emb
        if self.use_guidance_embed:
            guidance_emb = timestep_embedding(guidance, 256)
            vec = vec + self.guidance_in(guidance_emb)
            del guidance_emb

        double_block_mod_img = self.double_stream_modulation_img(vec)
        double_block_mod_txt = self.double_stream_modulation_txt(vec)
        single_block_mod, _ = self.single_stream_modulation(vec)

        img = self.img_in(x)
        del x
        txt = self.txt_in(ctx)
        del ctx
        pe_x = self.pe_embedder(x_ids)
        del x_ids
        pe_ctx = self.pe_embedder(ctx_ids)
        del ctx_ids

        for block_idx, block in enumerate(self.double_blocks):
            if self.blocks_to_swap:
                self.offloader_double.wait_for_block(block_idx)

            img, txt = block(img, txt, pe_x, pe_ctx, double_block_mod_img, double_block_mod_txt)

            if self.blocks_to_swap:
                self.offloader_double.submit_move_blocks_forward(list(self.double_blocks), block_idx)

        del double_block_mod_img, double_block_mod_txt

        img = torch.cat((txt, img), dim=1)
        del txt
        pe = torch.cat((pe_ctx, pe_x), dim=2)
        del pe_ctx, pe_x

        for block_idx, block in enumerate(self.single_blocks):
            if self.blocks_to_swap:
                self.offloader_single.wait_for_block(block_idx)

            img = block(img, pe, single_block_mod)

            if self.blocks_to_swap:
                self.offloader_single.submit_move_blocks_forward(list(self.single_blocks), block_idx)

        del single_block_mod, pe

        img = img[:, num_txt_tokens:, ...]

        img = self.final_layer(img, vec)
        return img


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)

        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim, bias=False)


class SiLUActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool, disable_bias: bool = False):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=not disable_bias)

    def forward(self, vec: Tensor):
        org_dtype = vec.dtype
        vec = vec.to(torch.float32)  # for numerical stability
        out = self.lin(F.silu(vec))
        if out.ndim == 2:
            out = out[:, None, :]
        out = out.to(org_dtype)
        out = out.chunk(self.multiplier, dim=-1)
        return out[:3], out[3:] if self.is_double else None


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        org_dtype = x.dtype
        vec = vec.to(torch.float32)  # for numerical stability
        mod = self.adaLN_modulation(vec)
        shift, scale = mod.chunk(2, dim=-1)
        if shift.ndim == 2:
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        x = x.to(torch.float32)  # for numerical stability
        x = (1 + scale) * self.norm_final(x) + shift
        x = self.linear(x)
        return x.to(org_dtype)


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()

        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = head_dim ** -0.5
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_mult_factor = 2

        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim * self.mlp_mult_factor, bias=False)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, bias=False)

        self.norm = QKNorm(head_dim)
        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = SiLUActivation()

        self.gradient_checkpointing = False
        self.attn_mode = "torch"

    def _forward(self, x: Tensor, pe: Tensor, mod) -> Tensor:
        mod_shift, mod_scale, mod_gate = mod
        del mod
        x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift
        del mod_scale, mod_shift

        qkv, mlp = torch.split(self.linear1(_ac(x_mod)), [3 * self.hidden_size, self.mlp_hidden_dim * self.mlp_mult_factor],
                               dim=-1)

        q, k, v = _split_qkv(qkv, self.num_heads)
        del qkv
        q, k = self.norm(q, k, v)

        attn = attention(q, k, v, pe, self.attn_mode)
        del q, k, v, pe

        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod_gate * output

    def forward(self, x: Tensor, pe: Tensor, mod) -> Tensor:
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, pe, mod, use_reentrant=False)
        return self._forward(x, pe, mod)


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0, f"{hidden_size=} must be divisible by {num_heads=}"

        self.hidden_size = hidden_size
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_mult_factor = 2

        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim * self.mlp_mult_factor, bias=False),
            SiLUActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
        )

        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim * self.mlp_mult_factor, bias=False),
            SiLUActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
        )
        self.gradient_checkpointing = False
        self.attn_mode = "torch"

    def _forward(self, img: Tensor, txt: Tensor, pe: Tensor, pe_ctx: Tensor, mod_img, mod_txt):
        img_mod1, img_mod2 = mod_img
        txt_mod1, txt_mod2 = mod_txt
        del mod_img, mod_txt

        img_mod1_shift, img_mod1_scale, img_mod1_gate = img_mod1
        img_mod2_shift, img_mod2_scale, img_mod2_gate = img_mod2
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate = txt_mod1
        txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = txt_mod2
        del img_mod1, img_mod2, txt_mod1, txt_mod2

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1_scale) * img_modulated + img_mod1_shift
        del img_mod1_scale, img_mod1_shift

        img_qkv = self.img_attn.qkv(_ac(img_modulated))
        del img_modulated
        img_q, img_k, img_v = _split_qkv(img_qkv, self.num_heads)
        del img_qkv
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1_scale) * txt_modulated + txt_mod1_shift
        del txt_mod1_scale, txt_mod1_shift
        txt_qkv = self.txt_attn.qkv(_ac(txt_modulated))
        del txt_modulated
        txt_q, txt_k, txt_v = _split_qkv(txt_qkv, self.num_heads)
        del txt_qkv
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        txt_len = txt_q.shape[2]
        q = torch.cat((txt_q, img_q), dim=2)
        del txt_q, img_q
        k = torch.cat((txt_k, img_k), dim=2)
        del txt_k, img_k
        v = torch.cat((txt_v, img_v), dim=2)
        del txt_v, img_v

        pe = torch.cat((pe_ctx, pe), dim=2)
        del pe_ctx
        attn = attention(q, k, v, pe, self.attn_mode)
        del q, k, v, pe
        txt_attn, img_attn = attn[:, :txt_len], attn[:, txt_len:]
        del attn

        # calculate the img blocks
        img = img + img_mod1_gate * self.img_attn.proj(img_attn)
        del img_mod1_gate, img_attn
        img = img + img_mod2_gate * self.img_mlp(_ac((1 + img_mod2_scale) * (self.img_norm2(img)) + img_mod2_shift))
        del img_mod2_gate, img_mod2_scale, img_mod2_shift

        # calculate the txt blocks
        txt = txt + txt_mod1_gate * self.txt_attn.proj(txt_attn)
        del txt_mod1_gate, txt_attn
        txt = txt + txt_mod2_gate * self.txt_mlp(_ac((1 + txt_mod2_scale) * (self.txt_norm2(txt)) + txt_mod2_shift))
        del txt_mod2_gate, txt_mod2_scale, txt_mod2_shift
        return img, txt

    def forward(self, img: Tensor, txt: Tensor, pe: Tensor, pe_ctx: Tensor, mod_img, mod_txt):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, img, txt, pe, pe_ctx, mod_img, mod_txt, use_reentrant=False)
        return self._forward(img, txt, pe, pe_ctx, mod_img, mod_txt)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, disable_bias: bool = False):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=not disable_bias)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=not disable_bias)
        self.gradient_checkpointing = False

    def _forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))

    def forward(self, *args, **kwargs):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, *args, use_reentrant=False, **kwargs)
        return self._forward(*args, **kwargs)


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(len(self.axes_dim))], dim=-3)
        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, device=t.device,
                                                           dtype=torch.float32) / half)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


def _ac(x: Tensor) -> Tensor:
    """Under autocast LayerNorm hands back float32; the Linears here may be INT8 / NF4 stand-ins whose patched forwards
    do not autocast, so the modulated activations go in as bf16 (what autocast would cast a plain Linear's input to)."""
    if x.dtype != torch.bfloat16 and torch.is_autocast_enabled(x.device.type):
        return x.to(torch.bfloat16)
    return x


def _split_qkv(qkv: Tensor, num_heads: int):
    """'B L (K H D) -> K B H L D' with K = 3 (einops in Fizgig)."""
    b, l, _ = qkv.shape
    qkv = qkv.reshape(b, l, 3, num_heads, -1).permute(2, 0, 3, 1, 4)
    return qkv[0], qkv[1], qkv[2]


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, attn_mode: str = "torch") -> Tensor:
    """RoPE, then full-sequence scaled-dot-product attention (no mask). q, k, v (B, H, L, D) -> (B, L, H * D).
    Fizgig's attention() transposes to (B, L, H, D) for its dispatcher and its torch path transposes back; the maths is
    the same. attn_mode "torch" is PyTorch SDPA inside the shared backend context (cuDNN for no-grad renders, PyTorch's
    own choice while training: modules/sdpa.py). Any other mode raises exactly as Fizgig's dispatcher does: its
    modules/attention.py has no branch for "flash3" (the GUI's second option), so it ends in the final else."""
    q, k = apply_rope(q, k, pe)
    if attn_mode == "torch":
        with sdpa_backend_ctx(q.device.type):
            x = F.scaled_dot_product_attention(q, k, v)
    else:
        raise ValueError(f"Unsupported attention mode: {attn_mode}")       # Fizgig modules/attention.py:245
    return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(torch.float64), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = out.reshape(*out.shape[:-1], 2, 2)               # b n d (i j) -> b n d i j
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


# ---- weights ---------------------------------------------------------------------------------------------------
def load_klein_dit(path, device="cuda", dtype=torch.bfloat16, params: Flux2Params | None = None):
    """Build the DiT on meta and load a Base checkpoint (assign=True, strict): bf16, or a pre-quantised fp8 file,
    whose Linear weights stay fp8 (with their scales, when the file has them) and are dequantised per matmul
    (Fizgig klein/model_utils.py load_dit). A sharded checkpoint (...-00001-of-0000N.safetensors) is read whole.
    The base precision is applied afterwards by training/quant.py, from either kind of file."""
    from training.families.qwen_image21.embedder import load_split_weights
    from training.modules import fp8
    path = str(path)
    with torch.device("meta"):
        model = KleinDiT(params or Klein9BParams())
    sd = load_split_weights(path)
    for prefix in ("model.diffusion_model.", "diffusion_model."):
        if sd and all(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):]: v for k, v in sd.items()}
            break
    missing, unexpected, _n = fp8.load_state_dict(model, sd, dtype)
    if missing or unexpected:
        raise ValueError(f"Klein DiT keys mismatch: missing {missing[:8]} ({len(missing)}), unexpected "
                         f"{unexpected[:8]} ({len(unexpected)}) - is this the FLUX.2 Klein Base 9B checkpoint?")
    return model.to(device)
