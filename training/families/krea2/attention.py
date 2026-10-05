# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/krea2/attention.py
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: env var prefix TAGSCRIBER_ (ATTN_TRIM, ATTN_TRIM_MULTIPLE); the SDPA backend choice lives in
# training/modules/sdpa.py. Only the "torch" (SDPA) attention mode is ported: it is the only one Krea 2 ever runs in
# Fizgig (krea2/utils.py:92 defaults attn_mode="torch", split_attn=False, and no trainer, script or GUI setting passes
# anything else for Krea 2; flash / sageattn / xformers and split attention are reachable only from Klein-era code
# paths). The logic of that mode - the key-padding mask, the uniform-length trim, GQA expansion, shape bookkeeping for
# the cuDNN switch - is Fizgig's.
# Fizgig's own header follows.
#
# Upstream: Unified attention function supporting various implementations, adapted from musubi-tuner
# (https://github.com/kohya-ss/musubi-tuner), Apache-2.0. Modified for Fizgig. See THIRD_PARTY_NOTICES.md.
"""Krea 2 attention: torch SDPA over [image tokens | text tokens] with variable text lengths.

Fizgig's AttentionParams decides, once per forward, whether every sequence in the batch has the same valid length:
if so, attention trims to that length (rounded UP to a multiple of TRIM_MULTIPLE, 64 by default, so a dataset's
captions land in a handful of shapes) instead of attending over padding, and pads the output back. A batch with
unequal lengths attends over the padded length with the key-padding mask. Rounded windows still contain padding, so
the mask is kept then (`uniform_exact` False).

    TAGSCRIBER_ATTN_TRIM=0              keep the padded length and pass the mask (slower raw compute, but far fewer
                                        distinct shapes - what shape-planning backends such as cuDNN or torch.compile
                                        prefer)
    TAGSCRIBER_ATTN_TRIM_MULTIPLE=N     the rounding (default 64; 1 = exact valid length, the old behaviour)
"""
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch

from training.modules.sdpa import note_shape as _note_shape
from training.modules.sdpa import sdpa_backend_ctx as _sdpa_backend_ctx

logger = logging.getLogger(__name__)

# Sequence lengths are rounded up to a multiple of this before attention (Fizgig attention.py:35-47; guarded parse:
# a typo'd env var names itself instead of a bare ValueError at import).
try:
    TRIM_MULTIPLE = int(os.environ.get("TAGSCRIBER_ATTN_TRIM_MULTIPLE", "64") or 64)
except ValueError:
    logger.warning("TAGSCRIBER_ATTN_TRIM_MULTIPLE=%r is not an integer - using the default 64",
                   os.environ.get("TAGSCRIBER_ATTN_TRIM_MULTIPLE"))
    TRIM_MULTIPLE = 64


@dataclass
class AttentionParams:
    img_len: Optional[int] = None
    attention_mask: Optional[torch.Tensor] = None      # bool [B, 1, 1, img_len + L] (True = attend) or None
    seqlens: Optional[torch.Tensor] = None
    max_seqlen: Optional[int] = None

    def __post_init__(self):
        # Every sequence the same length means attention can trim to it instead of attending over padding. Deciding that
        # requires reading a device tensor on the CPU, so it is resolved here, once per forward, rather than inside each
        # of the 28 blocks (a sync per block, and a graph break under torch.compile).
        self.uniform_seqlen = None
        self.uniform_exact = True          # False -> the trimmed window still contains padding
        if (self.seqlens is not None and self.attention_mask is not None
                and os.environ.get("TAGSCRIBER_ATTN_TRIM", "1") != "0"):
            if bool(torch.all(self.seqlens == self.seqlens[0])):
                valid = int(self.seqlens[0])
                q = TRIM_MULTIPLE
                if q > 1 and self.max_seqlen:
                    rounded = min(((valid + q - 1) // q) * q, int(self.max_seqlen))
                    self.uniform_exact = rounded == valid
                    self.uniform_seqlen = rounded
                else:
                    self.uniform_seqlen = valid
        # Tell the backend chooser which shape this forward will ACTUALLY attend - including when trim is disabled or the
        # batch is ragged: both attend the padded length.
        if self.attention_mask is not None:
            attended = self.uniform_seqlen or (int(self.max_seqlen) if self.max_seqlen else None)
            if attended:
                _note_shape(attended)

    @staticmethod
    def create_attention_params() -> "AttentionParams":
        return AttentionParams()

    @staticmethod
    def create_attention_params_from_mask(img_len: Optional[int],
                                          attention_mask: Optional[torch.Tensor]) -> "AttentionParams":
        """attention_mask is for the TEXT tokens only (True = real); img_len image tokens, all valid, lead them."""
        if attention_mask is None:
            return AttentionParams()          # no mask provided: assume all tokens are valid
        seqlens = attention_mask.sum(dim=1).to(torch.int32) + img_len
        max_seqlen = attention_mask.shape[1] + img_len
        mask = torch.nn.functional.pad(attention_mask, (img_len, 0), value=1)        # [B, img_len + L]
        return AttentionParams(img_len, mask[:, None, None, :].to(torch.bool), seqlens, max_seqlen)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              attn_params: Optional[AttentionParams] = None) -> torch.Tensor:
    """Scaled dot-product attention with variable sequence lengths (the "torch" mode of Fizgig's attention()).

    q, k, v: [B, L, H, D] (k / v may carry fewer heads: Krea 2 = 48 query / 12 kv). Returns [B, L, H*D]."""
    if attn_params is None:
        attn_params = AttentionParams.create_attention_params()
    # GQA: expanding k/v to q's head count stays the right call (Fizgig: enable_gqa forces SDPA onto the slow math
    # kernel, ~7x slower than the fused ones at K2 scale; the repeat is numerically identical).
    enable_gqa = q.shape[-2] != k.shape[-2]

    seqlen_trimmed = False
    seqlen = attn_params.uniform_seqlen
    max_seqlen = attn_params.max_seqlen
    mask = attn_params.attention_mask
    if seqlen is not None:
        q = q[:, :seqlen]
        k = k[:, :seqlen]
        v = v[:, :seqlen]
        # The window is rounded up to a multiple, so unless it landed exactly on the valid length it still contains
        # padding - which must stay masked or those tokens join the attention. The mask is [B, 1, 1, S].
        mask = None if attn_params.uniform_exact else attn_params.attention_mask[..., :seqlen]
        seqlen_trimmed = True

    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)               # [B, H, L, D]
    if enable_gqa:
        g = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)
    with _sdpa_backend_ctx(q.device.type):
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    x = x.transpose(1, 2)                                                           # [B, L, H, D]
    x = x.reshape(x.shape[0], x.shape[1], -1)                                       # [B, L, H*D]
    if seqlen_trimmed:
        x = torch.nn.functional.pad(x, (0, 0, 0, max_seqlen - x.shape[1]), value=0)  # pad back to max_seqlen
    return x
