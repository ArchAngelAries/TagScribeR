# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/minimax/model.py (the DiT), minimax/loader.py
# (the checkpoint loader) and the ConvRot int8 Linear of minimax/convrot.py.
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Fizgig's header calls model.py a "faithful, weight-name-compatible port of ComfyUI's comfy/ldm/minimax/model.py" -
# see THIRD_PARTY_NOTICES.md for what that means for the licence of this file.
# Changes for TagScribeR: STILL IMAGES ONLY - the reference / keyframe condition rows, the clip (T > 1) layout, TREAD
# token routing (Fizgig runs it on clip steps only), the Repair Studio activation cache (forward_cached), the int8
# attention kernel and the H2D streaming rings are not ported; block swap uses training.modules.offloading.ModelOffloader
# the way the Krea 2 DiT does (Fizgig parks the last n blocks on the CPU one at a time); the ConvRot int8 Linear keeps
# its codes in `weight` (so the shared offloader can stream them) and runs the eager path only (no Triton kernels); the
# loader reads the pruned ConvRot checkpoint (and a dense pruned one) and refuses the 66 GB bf16 release.
"""MiniMax H3 DiT, pure PyTorch, for image (single latent frame) LoRA training and previews.

A still image is one video frame (T = 1), so the packed sequence is [text | audio | video]. H3 always packs an audio
block - for one still that is 4 rows of silence noised on the audio schedule - so the frozen base runs in the layout it
was trained in; they carry no loss. `final_layer.audio_out` is built for checkpoint compatibility; it is only run by the
sampler's joint audio denoising.

Module and parameter names match the checkpoint (blocks.N.attn.qkv_proj.weight, adaln_proj.linear.weight,
video_patch_proj, condition_proj, adaln_t_table, rope.inv_freq, token_refiner.blocks.N..., final_layer...), so a LoRA
trained here maps straight back onto the base.
"""
import logging
import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.families.minimax_h3.weights import SafeReader, parse_comfy_quant, rotate

logger = logging.getLogger(__name__)


@dataclass
class MiniMaxH3Config:
    """FL2VA/transformer/config.json (MiniMaxH3DiTModel). Defaults are the real model; tests pass a tiny override."""
    hidden_size: int = 5376
    num_layers: int = 50
    token_refiner_num_layers: int = 2
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: tuple = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    rope_inv_freq_len: int = 16
    # PRUNED checkpoints (what ComfyUI ships) replace the timestep MLP with a lookup table sampled by linear
    # interpolation: no `time_embedder`, an `adaln_t_table` of [size, time_embed_dim], time_embed_dim 8, and NO silu in
    # front of the AdaLN projections. None = the full bf16 model.
    adaln_t_table_size: Optional[int] = None
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5


# ---- flow constants and helpers -------------------------------------------------------------------------------
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
AUDIO_CHANNELS = 2
AUDIO_LATENTS_PER_SECOND = 40
FPS = 24
VIDEO_SIGMA_SHIFT = 12.0
AUDIO_SIGMA_SHIFT = 3.0
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2       # AdaLN modality rows - a checkpoint contract
MODALITY_NUM = 3


def patchify_video(latent: torch.Tensor, patch_size=(1, 2, 2)) -> torch.Tensor:
    """[B, C, T, H, W] -> [B*t*h*w, C*pt*ph*pw]  (row-major t,h,w; channel-major within a patch)."""
    b, c, t_full, h_full, w_full = latent.shape
    pt, ph, pw = patch_size
    t, h, w = t_full // pt, h_full // ph, w_full // pw
    x = latent.reshape(b, c, t, pt, h, ph, w, pw)
    x = torch.einsum("nctrhpwq->nthwcrpq", x)
    return x.reshape(b * t * h * w, c * pt * ph * pw)


def unpatchify_video(rows: torch.Tensor, t, h, w, c=24, patch_size=(1, 2, 2)) -> torch.Tensor:
    pt, ph, pw = patch_size
    x = rows.reshape(-1, t, h, w, c, pt, ph, pw)
    x = torch.einsum("nthwcrpq->nctrhpwq", x)
    return x.reshape(-1, c, t * pt, h * ph, w * pw)


def audio_latents_for_frames(num_frames: int = 1) -> int:
    """Audio latents covering `num_frames` PIXEL frames (24 fps video, 40 Hz audio): a still is round(40/24) = 2."""
    return int(round(num_frames / FPS * AUDIO_LATENTS_PER_SECOND))


def shift_sigma(sigma, shift: float):
    """Exponential timeshift: shift*sigma / (1 + (shift-1)*sigma)."""
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def remap_sigma(sigma, from_shift: float = VIDEO_SIGMA_SHIFT, to_shift: float = AUDIO_SIGMA_SHIFT):
    """A sigma on the `from_shift` schedule -> the same schedule POSITION on `to_shift` (video shift 12, audio 3): the
    closed form that keeps the two streams at the same underlying point. Used for the audio rows' timestep."""
    return shift_sigma(sigma / (from_shift + sigma * (1.0 - from_shift)), to_shift)


def _axis_from_sqrt_area(dim, patch, sqrt_area):
    ratio = dim / sqrt_area
    n = dim // patch
    return (torch.arange(n, dtype=torch.float64) * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0


def _frame_grid(h, w):
    """Area-normalized (h, w) coords of one latent frame's 2x2-patch rows: [(h//2)*(w//2), 2]."""
    area = math.sqrt(h * w)
    hh, ww = torch.meshgrid(_axis_from_sqrt_area(h, 2, area), _axis_from_sqrt_area(w, 2, area), indexing="ij")
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)


def image_position_ids(text_len, latent_h, latent_w, num_audio_latents: int = 0) -> torch.Tensor:
    """3-axis (t, h, w) position ids for a [text | audio | video] sequence of ONE latent frame.

    Text rows: t = 0..text_len-1, h=w=0 - so prompt length shifts the whole media clock. Audio rows: t = text_len +
    0..A-1 repeated per channel, h = 0, w pinned to the frame grid's first column for channel 0 and its last for
    channel 1 (the reference's stereo convention). Video rows: t = text_len, h/w the area-normalised grid.
    Returns [S, 3] float64."""
    frame = _frame_grid(latent_h, latent_w)
    frame_rows = frame.shape[0]
    text = torch.zeros(text_len, 3, dtype=torch.float64)
    text[:, 0] = torch.arange(text_len, dtype=torch.float64)
    rows = [text]
    cursor = float(text_len)
    if num_audio_latents:
        w_axis = _axis_from_sqrt_area(latent_w, 2, math.sqrt(latent_h * latent_w))
        aud = torch.zeros(num_audio_latents * AUDIO_CHANNELS, 3, dtype=torch.float64)
        aud[:, 0] = (cursor + torch.arange(num_audio_latents, dtype=torch.float64)).repeat(AUDIO_CHANNELS)
        aud[:, 2] = torch.cat([torch.full((num_audio_latents,), float(w_axis[0]), dtype=torch.float64),
                               torch.full((num_audio_latents,), float(w_axis[-1]), dtype=torch.float64)])
        rows.append(aud)
    vid = torch.empty(frame_rows, 3, dtype=torch.float64)
    vid[:, 0] = cursor
    vid[:, 1:] = frame
    rows.append(vid)
    return torch.cat(rows, dim=0)


def rope_cos_sin(position_ids: torch.Tensor, inv_freq: torch.Tensor):
    """position_ids [S,3] (t,h,w) x inv_freq[16] -> cos, sin each [S, 48] (angles are [t(16) | h(16) | w(16)])."""
    pos = position_ids.to(torch.float32)
    ang = (pos.unsqueeze(-1) * inv_freq.to(torch.float32).view(1, 1, -1)).reshape(pos.shape[0], -1)
    return torch.cos(ang), torch.sin(ang)


def apply_rope_split_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [S, heads, head_dim]; cos/sin [S, rot_half]. Rotates the first 2*rot_half dims split-half style (dim i with
    dim i+rot_half); the tail passes through unrotated."""
    rot_half = cos.shape[-1]
    rot = 2 * rot_half
    xr, xp = x[..., :rot], x[..., rot:]
    x1, x2 = xr[..., :rot_half], xr[..., rot_half:]
    c = cos.unsqueeze(1)
    s = sin.unsqueeze(1)
    rotated = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return torch.cat([rotated, xp], dim=-1)


# ---- blocks ---------------------------------------------------------------------------------------------------
class TimeEmbedder(nn.Module):
    def __init__(self, freq_dim, hidden, out):
        super().__init__()
        self.freq_dim = freq_dim
        self.proj_in = nn.Linear(freq_dim, hidden, bias=True)
        self.proj_out = nn.Linear(hidden, out, bias=True)

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t.to(torch.float32)[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.proj_out(F.silu(self.proj_in(emb.to(self.proj_in.weight.dtype))))


class Attention(nn.Module):
    def __init__(self, hidden, heads, head_dim, eps):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.qkv_proj = nn.Linear(hidden, inner * 3, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps)
        self.out_proj = nn.Linear(inner, hidden, bias=False)

    def forward(self, x, cos=None, sin=None):
        s = x.shape[0]
        q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
        q = self.q_norm(q.view(s, self.heads, self.head_dim))
        k = self.k_norm(k.view(s, self.heads, self.head_dim))
        v = v.view(s, self.heads, self.head_dim)
        if cos is not None:
            q = apply_rope_split_half(q, cos, sin)
            k = apply_rope_split_half(k, cos, sin)
        q = q.transpose(0, 1).unsqueeze(0)                  # [S, H, D] -> [1, H, S, D]
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.squeeze(0).transpose(0, 1).reshape(s, self.heads * self.head_dim)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, hidden, ffn):
        super().__init__()
        self.fc1 = nn.Linear(hidden, ffn * 2, bias=False)
        self.fc2 = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x):
        a, b = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(a) * b)                       # swiglu


class AdalnProj(nn.Module):
    """t_dim -> `expand` modulation tensors, each [M*modalities, hidden]."""

    def __init__(self, t_dim, hidden, expand, modalities, apply_silu=True):
        super().__init__()
        self.expand = expand
        self.modalities = modalities
        self.hidden = hidden
        self.apply_silu = apply_silu
        self.linear = nn.Linear(t_dim, expand * hidden * modalities, bias=True)

    def forward(self, t_emb):
        x = self.linear(F.silu(t_emb) if self.apply_silu else t_emb)
        x = x.view(x.shape[0] * self.modalities, self.expand * self.hidden)
        return x.chunk(self.expand, dim=-1)


def _mod_scale_shift(h, shift, scale, mod_row):
    """h [S,hidden] modulated per row: h*(1+scale[row]) + shift[row]. Fully out-of-place (autograd-safe)."""
    return h * (1.0 + scale[mod_row].to(h.dtype)) + shift[mod_row].to(h.dtype)


def _mod_gate(x, gate, other, mod_row):
    """Gated residual add: x + other*gate[row], per row. Out-of-place."""
    return x + other * gate[mod_row].to(x.dtype)


class RefinerBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, eps, qk_eps):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden, eps=eps)
        self.norm2 = nn.RMSNorm(hidden, eps=eps)
        self.attn = Attention(hidden, heads, head_dim, qk_eps)
        self.mlp = MLP(hidden, ffn)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        return self.mlp(self.norm2(x)) + x


class TokenRefiner(nn.Module):
    def __init__(self, num_layers, hidden, heads, head_dim, ffn, eps, qk_eps, final_eps):
        super().__init__()
        self.blocks = nn.ModuleList([RefinerBlock(hidden, heads, head_dim, ffn, eps, qk_eps)
                                     for _ in range(num_layers)])
        self.final_norm = nn.RMSNorm(hidden, eps=final_eps)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class DiTBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, t_dim, eps, qk_eps, apply_silu=True):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden, eps=eps)
        self.norm2 = nn.RMSNorm(hidden, eps=eps)
        self.attn = Attention(hidden, heads, head_dim, qk_eps)
        self.mlp = MLP(hidden, ffn)
        self.adaln_proj = AdalnProj(t_dim, hidden, 6, 3, apply_silu=apply_silu)

    def forward(self, x, t_emb, mod_row, cos, sin):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_row)
        x = _mod_gate(x, gate_msa, self.attn(h, cos=cos, sin=sin), mod_row)
        h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_row)
        return _mod_gate(x, gate_mlp, self.mlp(h), mod_row)


class FinalLayer(nn.Module):
    def __init__(self, hidden, t_dim, video_dim, audio_dim, eps, apply_silu=True):
        super().__init__()
        self.norm = nn.RMSNorm(hidden, eps=eps)
        self.adaln_proj = AdalnProj(t_dim, hidden, 2, 1, apply_silu=apply_silu)
        self.video_out = nn.Linear(hidden, video_dim, bias=True)      # fp32 output heads (the checkpoint's fp32 island)
        self.audio_out = nn.Linear(hidden, audio_dim, bias=True)

    def forward(self, x_video, t_emb, t_index: int = 0):
        """x_video [n_video, hidden] (just the video segment). modalities=1: one table row per distinct timestep, and
        `t_index` selects the video rows' own."""
        shift, scale = self.adaln_proj(t_emb)
        hv = self.norm(x_video) * (1.0 + scale[t_index]) + shift[t_index]
        return self.video_out(hv.to(self.video_out.weight.dtype))

    def forward_audio(self, x_audio, t_emb, t_index: int = 0):
        """Same shared norm + modulation at the AUDIO rows' timestep, through the audio head (joint audio denoising)."""
        shift, scale = self.adaln_proj(t_emb)
        ha = self.norm(x_audio) * (1.0 + scale[t_index]) + shift[t_index]
        return self.audio_out(ha.to(self.audio_out.weight.dtype))


# ---- ConvRot int8 Linear --------------------------------------------------------------------------------------
class _Int8RotLinearFn(torch.autograd.Function):
    """linear(rotate(x), dequant(q, s)) that does NOT keep the dequantized weight alive (Fizgig convrot.py).

    The base is frozen, so no weight gradient is needed and the input need not be saved either:
    grad_x = rotate((grad_y * s) @ q), recomputed from the int8 codes that are resident anyway. The scale is applied to
    the OUTPUT (s is per output row): the int8 codes survive the bf16 cast bit-for-bit, the fp32 scales do not."""

    @staticmethod
    def forward(ctx, x, qdata, wscale, bias, rot, dt):
        ctx.save_for_backward(qdata, wscale)
        ctx.rot, ctx.dt = rot, dt
        xr = rotate(x.to(dt), rot) if rot > 1 else x.to(dt)
        y = F.linear(xr, qdata.to(dt))
        y = (y.float() * wscale.reshape(-1).float()).to(dt)
        return y if bias is None else y + bias.to(dt)

    @staticmethod
    def backward(ctx, grad_out):
        qdata, wscale = ctx.saved_tensors
        gs = (grad_out.float() * wscale.reshape(-1).float()).to(ctx.dt)
        gx = gs @ qdata.to(ctx.dt)
        if ctx.rot > 1:
            gx = rotate(gx, ctx.rot)                       # H is symmetric: R^T == R
        return gx, None, None, None, None, None


class ConvRotInt8Linear(nn.Linear):
    """A frozen Linear that KEEPS the checkpoint's int8 ConvRot weights - the reference's own storage (~0.17% error in
    the frozen base; decoding to bf16 and re-quantising to NF4 is ~9.5%). The int8 codes live in `weight` (a frozen
    int8 Parameter) so the shared block-swap offloader streams them like any Linear weight; `_convrot_wscale` is the
    per-output-row fp32 scale. The activation is rotated instead of the weight: (x R)(W_rot)^T = x W^T.

    `_prequantized` tells training/quant.py to leave it alone for INT8 and to read `dense_weight()` for NF4."""

    _prequantized = True

    def __init__(self, in_features, out_features, bias=False, rot=256):
        super().__init__(in_features, out_features, bias=bias)
        self.rot = int(rot)
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, dtype=torch.int8), requires_grad=False)
        self.register_buffer("_convrot_wscale", torch.ones(out_features, 1, dtype=torch.float32), persistent=False)

    def dense_weight(self, device=None, dtype=torch.bfloat16) -> torch.Tensor:
        """The dense weight in the TRUE basis (what NF4 quantises); the rotation runs on `device` if given."""
        q = self.weight.data if device is None else self.weight.data.to(device)
        s = self._convrot_wscale if device is None else self._convrot_wscale.to(device)
        return rotate(q.float() * s.float(), self.rot).to(dtype)

    def forward(self, x):
        if self.bias is not None and self.bias.requires_grad:
            raise RuntimeError("ConvRotInt8Linear holds a FROZEN base weight; its bias must not require grad")
        return _Int8RotLinearFn.apply(x, self.weight.data, self._convrot_wscale, self.bias, self.rot, x.dtype)


# ---- the model ------------------------------------------------------------------------------------------------
class MiniMaxH3DiT(nn.Module):
    """Image-only forward for the MiniMax H3 DiT. Names match the checkpoint."""

    def __init__(self, config: Optional[MiniMaxH3Config] = None):
        super().__init__()
        c = config or MiniMaxH3Config()
        self.config = c
        self.hidden_size = c.hidden_size
        self.patch_size = tuple(c.patch_size)
        self.latents_dim = c.latents_dim
        video_patch_dim = c.latents_dim * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]

        self.video_patch_proj = nn.Linear(video_patch_dim, c.hidden_size, bias=True)
        self.audio_patch_proj = nn.Linear(c.audio_latents_dim, c.hidden_size, bias=True)
        self.condition_proj = nn.Linear(c.text_dim, c.hidden_size, bias=True)
        self.pruned_adaln = c.adaln_t_table_size is not None
        # Set by the loader when it keeps the AdaLN projections fp32 (pruned checkpoints, as ComfyUI does): the forward
        # then hands them an fp32 t_emb instead of demoting it.
        self.adaln_fp32 = False
        _silu = not self.pruned_adaln              # the table already absorbs the nonlinearity
        if self.pruned_adaln:
            self.time_embedder = None
            self.register_buffer("adaln_t_table", torch.zeros(c.adaln_t_table_size, c.time_embed_dim))
        else:
            self.time_embedder = TimeEmbedder(c.timestep_input_dim, c.time_embed_hidden_size, c.time_embed_dim)
        self.rope = nn.Module()
        self.rope.register_buffer("inv_freq", rope_inv_freq(c.rope_inv_freq_len))
        self.token_refiner = TokenRefiner(c.token_refiner_num_layers, c.hidden_size, c.num_attention_heads,
                                          c.attention_head_dim, c.ffn_hidden_size, c.norm_eps, c.qk_norm_eps,
                                          c.final_norm_eps)
        self.blocks = nn.ModuleList([
            DiTBlock(c.hidden_size, c.num_attention_heads, c.attention_head_dim, c.ffn_hidden_size,
                     c.time_embed_dim, c.norm_eps, c.qk_norm_eps, apply_silu=_silu)
            for _ in range(c.num_layers)])
        self.final_layer = FinalLayer(c.hidden_size, c.time_embed_dim, video_patch_dim, c.audio_latents_dim,
                                      c.final_norm_eps, apply_silu=_silu)
        self.gradient_checkpointing = False
        self.blocks_to_swap = 0
        self.offloader = None
        # Pack the reference's silence audio block (the base was trained with these rows present; off is an A/B hatch).
        self.pack_audio_rows = True

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """[M] cleanness values in [0,1] -> [M, time_embed_dim] modulation input. Full model: the sinusoid + MLP time
        embedder. Pruned: linear interpolation into the curve table (t=0 is row 0, t=1 the last row)."""
        if not self.pruned_adaln:
            return self.time_embedder(t)
        table = self.adaln_t_table.float()
        pos = t.to(torch.float32).clamp(0.0, 1.0) * (table.shape[0] - 1)
        pos = pos.to(table.device)
        lo = pos.floor().long()
        hi = (lo + 1).clamp(max=table.shape[0] - 1)
        frac = (pos - lo.float()).unsqueeze(1)
        return (table[lo] * (1.0 - frac) + table[hi] * frac).to(t.device)

    def enable_gradient_checkpointing(self, on: bool = True):
        self.gradient_checkpointing = on

    # ---- block swap (the shared offloader; the 50 main blocks) ---------------------------------------------------
    def enable_block_swap(self, num_blocks: int, device, supports_backward: bool = True):
        from training.modules.offloading import ModelOffloader
        if self.offloader is not None:
            self.offloader.remove_hooks()      # stale backward hooks double-swap blocks ("mat2 is on cpu")
        n = len(self.blocks)
        if not 0 < num_blocks <= n - 2:
            raise ValueError(f"block swap: 1..{n - 2} blocks, got {num_blocks}")
        self.blocks_to_swap = num_blocks
        self.offloader = ModelOffloader("minimax_h3", list(self.blocks), n, num_blocks, supports_backward,
                                        torch.device(device))

    def move_to_device_except_swap_blocks(self, device):
        """Everything to `device` except the swapped blocks' weights (the model is assumed to be on CPU). try/finally:
        an OOM inside .to() must not leave the model holding the empty placeholder ModuleList."""
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

    # ---- forward -------------------------------------------------------------------------------------------------
    def forward(self, video_latent: torch.Tensor, t: torch.Tensor, text_embeds: torch.Tensor, *,
                audio_noise: torch.Tensor = None, audio_rows: torch.Tensor = None, return_audio: bool = False):
        """
        video_latent : [1, C=latents_dim, 1, H, W] - one latent frame (H and W even).
        t            : scalar or [1] flow time in [0, 1] - the CLEANNESS fed to the time embedder (1 - sigma).
        text_embeds  : [1, L, text_dim] Qwen3-VL hidden states (or already [1, L, hidden]).
        audio_noise  : optional [A*2, audio_latents_dim] unit noise for the SILENCE rows, scaled by the audio sigma
                       internally. None draws fresh (training: image datasets have no soundtrack, so silence noised at
                       sigma_a rides along without contributing loss).
        audio_rows   : optional [A*2, audio_latents_dim] EXPLICIT row content, used verbatim (the sampler's joint audio
                       denoising). Overrides audio_noise.
        return_audio : also return the audio head's prediction for those rows, so a sampler can step them.
        returns      : [1, C, 1, H, W] video prediction (the head's raw output = x0 - noise), or (video, audio).
        """
        if video_latent.shape[0] != 1:
            raise ValueError("MiniMax H3 image training is batch size 1")
        device = video_latent.device
        dtype = text_embeds.dtype
        _, _, latent_t, lat_h, lat_w = video_latent.shape
        if latent_t != 1:
            raise ValueError("this port trains and samples single-frame images only (latent T must be 1)")
        text_len = text_embeds.shape[1]

        text_states = text_embeds[0]
        if text_states.shape[-1] != self.hidden_size:
            text_states = self.token_refiner(self.condition_proj(text_states))
        video_rows = patchify_video(video_latent.to(torch.float32), self.patch_size)
        video_embed = self.video_patch_proj(video_rows.to(self.video_patch_proj.weight.dtype)).to(dtype)

        # audio: silence (x0 = 0) noised on the audio schedule at the same schedule position as the video rows
        t_val = t.reshape(-1)[:1].to(torch.float32) if torch.is_tensor(t) else torch.tensor([float(t)])
        t_val = t_val.to(device)
        n_audio_latents = audio_latents_for_frames(1) if self.pack_audio_rows else 0
        audio_embed = None
        if n_audio_latents:
            sigma_v = (1.0 - t_val).clamp(0.0, 1.0)
            t_audio = 1.0 - remap_sigma(sigma_v)
            if audio_rows is not None:
                _want = n_audio_latents * AUDIO_CHANNELS
                if audio_rows.shape[0] != _want:
                    raise ValueError(f"audio_rows has {audio_rows.shape[0]} rows but a still packs {_want}")
                _arows = audio_rows.to(device=device, dtype=torch.float32)
            else:
                eps = audio_noise
                if eps is None:
                    eps = torch.randn(n_audio_latents * AUDIO_CHANNELS, self.config.audio_latents_dim,
                                      device=device, dtype=torch.float32)
                _arows = remap_sigma(sigma_v) * eps.to(device=device, dtype=torch.float32)
            audio_embed = self.audio_patch_proj(_arows.to(self.audio_patch_proj.weight.dtype)).to(dtype)

        parts = [text_states.to(dtype)] + ([audio_embed] if audio_embed is not None else []) + [video_embed]
        h = torch.cat(parts, dim=0)
        seq_len = h.shape[0]
        n_audio = 0 if audio_embed is None else audio_embed.shape[0]
        audio_start = text_len
        video_start = audio_start + n_audio

        # One modulation row-set per DISTINCT timestep (video/text at t, audio at t_audio), indexed per row as
        # `timestep_index * 3 + modality_tag`. Tags: video 0, text 1, audio 2.
        t_all = torch.cat([t_val, t_audio]) if audio_embed is not None else t_val
        uniq, inverse = torch.unique(t_all, sorted=True, return_inverse=True)
        t_emb = self._time_embedding(uniq)                                        # [M, t_dim]
        if not self.adaln_fp32:
            t_emb = t_emb.to(dtype)
        tags = torch.full((seq_len,), VIDEO_TAG, dtype=torch.long, device=device)
        tags[:text_len] = TEXT_TAG
        row_t_index = torch.full((seq_len,), int(inverse[0]), dtype=torch.long, device=device)
        if audio_embed is not None:
            tags[audio_start:video_start] = AUDIO_TAG
            row_t_index[audio_start:video_start] = int(inverse[1])
        mod_row = row_t_index * MODALITY_NUM + tags
        video_t_index = int(inverse[0])

        pos = image_position_ids(text_len, lat_h, lat_w, n_audio_latents).to(device)
        cos, sin = rope_cos_sin(pos, self.rope.inv_freq.to(device))
        cos, sin = cos.to(dtype), sin.to(dtype)

        # Gate on grad-enabled only, NOT self.training: the frozen base stays in eval() during LoRA training.
        use_ckpt = self.gradient_checkpointing and torch.is_grad_enabled()
        blocks = list(self.blocks) if self.blocks_to_swap else None
        for i, block in enumerate(self.blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(i)
            if use_ckpt:
                h = torch.utils.checkpoint.checkpoint(block, h, t_emb, mod_row, cos, sin, use_reentrant=False)
            else:
                h = block(h, t_emb, mod_row, cos, sin)
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(blocks, i)

        v = self.final_layer(h[video_start:], t_emb, video_t_index)              # [n_video, video_patch_dim]
        out = unpatchify_video(v, 1, lat_h // self.patch_size[1], lat_w // self.patch_size[2], self.latents_dim,
                               self.patch_size).to(video_latent.dtype)
        if return_audio:
            a = None
            if n_audio:
                a = self.final_layer.forward_audio(h[audio_start:video_start], t_emb, int(inverse[1]))
            return out, a
        return out


def rope_inv_freq(n: int = 16) -> torch.Tensor:
    return 1.0 / (10000.0 ** (torch.arange(0, n, dtype=torch.float32) / n))


# ---- the checkpoint -------------------------------------------------------------------------------------------
def config_from_checkpoint(keys, table_shape=None) -> MiniMaxH3Config:
    """The config the file describes: a pruned file carries `adaln_t_table` and no `time_embedder.*`."""
    if "adaln_t_table" in keys:
        if table_shape is None:
            raise ValueError("pruned checkpoint: adaln_t_table shape is required")
        return MiniMaxH3Config(adaln_t_table_size=int(table_shape[0]), time_embed_dim=int(table_shape[1]))
    return MiniMaxH3Config()


def _owner_and_leaf(model, name: str):
    parent_path, _, leaf = name.rpartition(".")
    return (model.get_submodule(parent_path) if parent_path else model), leaf


def load_minimax_h3_dit(path: str, device="cpu", dtype=torch.bfloat16, adaln_fp32: bool = True,
                        config: Optional[MiniMaxH3Config] = None) -> MiniMaxH3DiT:
    """The H3 DiT from the pruned checkpoint (`minimax_h3_fl2va_pruned_int8_convrot.safetensors`, what ComfyUI ships),
    frozen, on `device`. The block Linears the file stores as int8 ConvRot stay int8 (ConvRotInt8Linear, ~21 GB for the
    whole model); everything else is bf16 - except the fp32 output-head island and the AdaLN projections, which stay
    fp32 as ComfyUI runs them (the file stores them fp16; bf16 would drop 3 mantissa bits on the shift / scale / GATE of
    every block, a deterministic perturbation applied 100 times per forward). Built on `meta` and filled tensor by
    tensor, so peak RAM is the model itself.

    The full 66 GB bf16 release is refused: its full-width AdaLN alone is 26 GB, which needs a streaming base that
    this port does not have - use the pruned file."""
    path = str(path)
    if os.path.isdir(path):
        raise ValueError(f"{path} is a folder: point at the single pruned file (minimax_h3_fl2va_pruned_int8_convrot"
                         f".safetensors), not the sharded Hub download")
    with SafeReader(path) as f:
        keys = set(f.keys())
        if "adaln_t_table" not in keys:
            raise ValueError(
                f"{path} is not the PRUNED MiniMax H3 checkpoint (no adaln_t_table). Training needs "
                f"minimax_h3_fl2va_pruned_int8_convrot.safetensors (21 GB) from Comfy-Org/MiniMax-H3 - the full bf16 "
                f"release is 66 GB and is not supported by this trainer.")
        cfg = config or config_from_checkpoint(keys, tuple(f.shape("adaln_t_table")))
        quant_conf = {k[: -len(".comfy_quant")]: parse_comfy_quant(f.get_tensor(k))
                      for k in keys if k.endswith(".comfy_quant")}
        for mod, conf in quant_conf.items():
            if conf.get("format") != "int8_tensorwise":
                raise ValueError(f"{mod}: unsupported comfy quant format {conf.get('format')!r} - this trainer reads "
                                 f"the int8 ConvRot H3 checkpoint")
        adaln32 = bool(adaln_fp32)

        with torch.device("meta"):
            model = MiniMaxH3DiT(cfg)
            for mod_name, module in list(model.named_modules()):
                for child_name, child in list(module.named_children()):
                    full = f"{mod_name}.{child_name}" if mod_name else child_name
                    conf = quant_conf.get(full)
                    if conf is not None and isinstance(child, nn.Linear):
                        rot = int(conf.get("convrot_groupsize", 256)) if conf.get("convrot") else 1
                        setattr(module, child_name, ConvRotInt8Linear(child.in_features, child.out_features,
                                                                      bias=child.bias is not None, rot=rot))
        logger.info(f"[load] MiniMax H3 base: {len(quant_conf)} int8 ConvRot Linears kept as int8 (the reference's "
                    f"own storage), everything else {str(dtype).replace('torch.', '')}")

        missing = []
        for name, _ in list(model.named_parameters()):
            if name not in keys:
                missing.append(name)
                continue
            parent, leaf = _owner_and_leaf(model, name)
            if isinstance(parent, ConvRotInt8Linear) and leaf == "weight":
                mod_path = name.rpartition(".")[0]
                parent.weight = nn.Parameter(f.get_tensor(name).to(torch.int8), requires_grad=False)
                parent._convrot_wscale = f.get_tensor(f"{mod_path}.weight_scale").to(torch.float32).reshape(-1, 1)
                continue
            w = f.get_tensor(name)
            fp32 = w.dtype == torch.float32 or (adaln32 and ".adaln_proj.linear." in name)
            setattr(parent, leaf, nn.Parameter(w.to(torch.float32 if fp32 else dtype), requires_grad=False))
        for name, buf in list(model.named_buffers()):
            if name in keys:
                parent, leaf = _owner_and_leaf(model, name)
                parent.register_buffer(leaf, f.get_tensor(name).to(torch.float32))
            elif name == "rope.inv_freq":
                model.rope.register_buffer("inv_freq", rope_inv_freq(cfg.rope_inv_freq_len))
            elif buf.is_meta:
                missing.append(name)
    if missing:
        raise ValueError(f"MiniMax H3 checkpoint is missing {len(missing)} tensors, e.g. {missing[:6]} - is this the "
                         f"pruned fl2va file?")
    model.adaln_fp32 = adaln32
    return model.to(device).eval().requires_grad_(False)
