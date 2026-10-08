# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/modules/int8.py (commit 1c8ec88)
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: import paths; an fp8 source module (training/modules/fp8.py) is detached after its weight is
# replaced, as training/quant.py does for the training bases. Used for the optional INT8 preview checkpoint (Klein's
# Distilled previews).
"""INT8 (W8A8) dynamic quantization for a frozen DiT — experimental inference speedup.

Inference path for the INT8 experiment. Like the fp8 monkey-patch, each `nn.Linear` is kept (so
LoRA targeting by module path is unchanged) and its `forward` is patched to run an int8 matmul.

Design that composes with block swap
------------------------------------
The block-swap offloader streams a block by moving `module.weight.data` (and stages it through a
`torch.empty_like(weight.data)` pinned buffer). So — unlike NF4, whose 4-bit packed weight + non-
tensor QuantState can't live in `.weight` — we keep the int8 weight **inside `module.weight.data`**.
That means:
  * `nn.Module.to()` / `weighs_to_device()` move it (whole-model park + resident-block placement), and
  * the offloader's per-forward staging inherits the int8 dtype (empty_like), so `copy_` is int8->int8.
=> INT8 stacks with inference block swap for free, *provided quantization runs before
`enable_block_swap`* (so the staging buffers are allocated int8-shaped).

Weight layout — measured, and NOT what intuition suggests
---------------------------------------------------------
The weight is stored in the natural `(N, K)` orientation and passed to `_int_mm` as `W.t()`.
That transpose is a **free view**, and the column-major layout it presents is what the int8
tensor cores actually want. Storing it "pre-transposed" as (K, N) to avoid the transpose — which
this module used to do — is 3.4x SLOWER (0.452 ms vs 0.131 ms at 6144x6144 on a 5090), and made
the whole INT8 path only 1.02x faster than plain bf16, i.e. pointless. Activations are quantized
dynamically per-token each forward; the matmul is `torch._int_mm` (int8 x int8 -> int32), then dequant.

Why int8: on Blackwell the int8 tensor cores are ~1.4-1.8x faster than fp8 at the matmul level, and
INT8 cores exist on 30-series too (which have no fast fp8). Inference-only (dynamic activation quant
isn't differentiable) — for previews / the workbench, not the training base (fp8 / NF4 stay there).
"""
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_INT8_TARGET_DEFAULT = ("blocks.",)


def int8_linear_forward_patch(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Per-token dynamic int8 activation quant -> int8 matmul -> dequant. Symmetric (no zero point).
    `self.weight.data` is the (N, K) int8 weight — passed as `.t()`, a free view in the layout the
    int8 tensor cores want; `self._int8_wscale` is (1, N)."""
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])
    a_scale = x2d.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0     # (M, 1)
    a_i8 = (x2d / a_scale).round_().clamp_(-127, 127).to(torch.int8)
    w_i8 = self.weight.data                                                    # (N, K) int8
    try:
        acc = torch._int_mm(a_i8, w_i8.t())                                    # (M, N) int32
    except Exception:
        # Shape-constraint edge case (tiny M / odd alignment): fall back to a bf16 matmul from the
        # dequantized int8 weight so a preview never crashes on an unusual resolution.
        w = w_i8.to(torch.float32) * self._int8_wscale.reshape(-1, 1)         # (N, K) fp32
        out = (x2d.to(torch.float32) @ w.t()).to(x.dtype)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], -1)
    out = (acc.to(torch.float32) * a_scale.to(torch.float32) * self._int8_wscale).to(x.dtype)
    if self.bias is not None:
        out = out + self.bias
    return out.reshape(*orig_shape[:-1], -1)


def apply_int8_quantization(
    model: nn.Module,
    target_keys=_INT8_TARGET_DEFAULT,
    exclude_keys=(),
    compute_device: torch.device = torch.device("cuda"),
) -> int:
    """Quantize the target Linears to int8 (per-output-channel weight scale) IN `module.weight.data`
    and patch their forward to the int8 path. Reuses nf4's source-weight dequant so an fp8 or bf16
    base both work. Call this BEFORE enable_block_swap so the offloader stages int8. Returns count."""
    from training.modules import fp8 as F8
    from training.modules.nf4 import _dequantize_source_weight

    compute_device = torch.device(compute_device)
    count = 0
    for name, module in model.named_modules():
        if not (isinstance(module, nn.Linear) and getattr(module, "weight", None) is not None):
            continue
        if not any(t in name for t in target_keys):
            continue
        if any(e in name for e in exclude_keys):
            continue

        w_bf16 = _dequantize_source_weight(module).to(compute_device).contiguous()   # (N, K)
        w_scale = w_bf16.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0     # (N, 1)
        w_i8 = (w_bf16 / w_scale).round_().clamp_(-127, 127).to(torch.int8)           # (N, K)

        # Keep the natural (N, K) orientation in .weight.data: the offloader still streams it, the
        # shape stays meaningful, and the forward's `.t()` is a free view in the layout the int8
        # tensor cores want (see module docstring — the "pre-transposed" variant is 3.4x slower).
        # requires_grad must be cleared FIRST — a grad-requiring Parameter can't hold an int8 .data.
        F8.detach(module)                   # an fp8 source: its scale and patched forward go with the old weight
        module.weight.requires_grad_(False)
        module.weight.data = w_i8.contiguous()                                       # (N, K) int8
        module.register_buffer("_int8_wscale", w_scale.reshape(1, -1).to(torch.float32), persistent=False)
        module._is_int8 = True
        module.forward = int8_linear_forward_patch.__get__(module, type(module))

        del w_bf16, w_i8
        count += 1

    if count > 0:
        model._int8_quantized = True
    logger.info(f"INT8 quantization: {count} Linears -> W8A8 (weight in .weight.data, per-token activation).")
    return count
