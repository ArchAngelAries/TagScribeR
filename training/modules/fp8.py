# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/krea2/fp8_optimization_utils.py
# (quantize_fp8, quantize_weight, the dequantising branch of fp8_linear_forward_patch, apply_fp8_monkey_patch) and
# src/fizgig/krea2/utils.py (the ComfyUI pre-quantised fp8 layout: `.weight_scale`, `.comfy_quant`), themselves adapted
# from musubi-tuner (https://github.com/kohya-ss/musubi-tuner, Apache-2.0).
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: one small module for every family; the torch._scaled_mm fast paths are not ported (they
# need NVIDIA Ada or newer and are off on ROCm in Fizgig too) - the weights stay fp8 and are dequantised per matmul,
# which is the path Fizgig runs on AMD.
"""fp8 frozen bases: scaled float8 (e4m3) Linear weights, dequantised on the fly.

Two ways a model ends up fp8:
* the checkpoint is ALREADY fp8 (a ComfyUI `..._fp8_scaled.safetensors`, or a plain fp8 cast): the weights are
  kept exactly as stored, with their scales, and never re-quantised (Fizgig: load_krea2_dit, pre-quantised branch);
* a bf16 checkpoint with the fp8 precision chosen: the target Linears are quantised at load with a per-block scale
  (64 input features per block, falling back to per-channel), Fizgig's `quantize_weight` default.

An fp8 Linear holds one byte per weight. Its forward is F.linear(x, weight.to(dtype) * scale, bias). The INT8 and
NF4 quantisers read an fp8 module through dense_weight(), so an fp8 file is a valid source for them too - as in
Fizgig, where the fp8 RAW checkpoint feeds every base precision.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

FP8 = torch.float8_e4m3fn
SCALE_KEYS = (".weight_scale", ".scale_weight")          # ComfyUI's name, musubi's name
DROP_KEYS = (".comfy_quant", ".input_scale")              # markers / activation scales the dequant path never uses


def is_fp8(t) -> bool:
    return isinstance(t, torch.Tensor) and t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


def file_is_fp8(path: str) -> bool:
    """True for a pre-quantised fp8 checkpoint: scale keys, or any fp8 tensor. Header-only read."""
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        if any(k.endswith(SCALE_KEYS + DROP_KEYS) for k in keys):
            return True
        return any("F8" in str(f.get_slice(k).get_dtype()) for k in keys[:400])


def quantize_weight(tensor: torch.Tensor, block_size: int = 64):
    """A Linear weight [out, in] -> (fp8 weight, float32 scale). Block mode: one scale per 64 input features per row
    ([out, blocks, 1]); per-channel ([out, 1]) when `in` is not a multiple of 64; per-tensor for other shapes. The
    scale is that group's absolute maximum over the largest fp8 value (Fizgig quantize_weight, mode "block")."""
    max_value = torch.finfo(FP8).max
    shape = tensor.shape
    t = tensor.to(torch.float32)
    if t.ndim == 2 and shape[1] % block_size == 0:
        t = t.contiguous().view(shape[0], shape[1] // block_size, block_size)
        scale = t.abs().amax(dim=2, keepdim=True) / max_value
    elif t.ndim == 2:
        scale = t.abs().amax(dim=1, keepdim=True) / max_value
    else:
        scale = t.abs().max() / max_value
    scale = scale.clamp(min=1e-8).to(torch.float32)
    q = torch.div(t, scale).nan_to_num_(0.0).clamp_(min=-max_value, max=max_value).to(FP8)
    return q.view(shape), scale


def _reshape_scale(scale: torch.Tensor) -> torch.Tensor:
    """ComfyUI stores per-channel scales as [out] and per-tensor ones as a scalar: make them broadcast against
    [out, in] (Fizgig _reshape_prequant_fp8_scale)."""
    if scale.ndim == 1:
        return scale.unsqueeze(1)
    if scale.ndim == 0:
        return scale.reshape(1)
    return scale


def dequantize(weight: torch.Tensor, scale, dtype=torch.bfloat16) -> torch.Tensor:
    """fp8 weight (+ its scale, or None for a plain fp8 cast) -> a dense weight in `dtype`."""
    if scale is None:
        return weight.to(dtype)
    scale = scale.to(device=weight.device, dtype=torch.float32)
    w = weight.to(torch.float32)
    if scale.ndim < 3:
        return (w * scale).to(dtype)
    out_features, num_blocks, _ = scale.shape
    return (w.contiguous().view(out_features, num_blocks, -1) * scale).view(weight.shape).to(dtype)


def fp8_linear_forward(self, x):
    """F.linear with the weight dequantised for this call (the frozen base: no gradient reaches the weight)."""
    w = dequantize(self.weight, getattr(self, "scale_weight", None), x.dtype)
    bias = self.bias
    if bias is not None and bias.dtype != x.dtype:
        bias = bias.to(x.dtype)
    return F.linear(x, w, bias)


def attach(module: nn.Linear, scale=None) -> None:
    """Mark a Linear whose weight is fp8: keep its scale as a buffer (so .to() and block swap carry it) and patch
    the forward."""
    module.weight.requires_grad_(False)
    if "scale_weight" in module._buffers:
        del module._buffers["scale_weight"]
    module.register_buffer("scale_weight", None if scale is None else scale.to(torch.float32), persistent=False)
    module._is_fp8 = True
    module.forward = fp8_linear_forward.__get__(module, type(module))


def detach(module: nn.Linear) -> None:
    """Forget a module's fp8 state (after another quantiser replaced its weight)."""
    module._buffers.pop("scale_weight", None)
    if getattr(module, "_is_fp8", False):
        module.__dict__.pop("forward", None)          # back to nn.Linear.forward
    module._is_fp8 = False


def dense_weight(module: nn.Linear, dtype=torch.bfloat16) -> torch.Tensor:
    """A module's weight as a dense tensor, whatever it is stored as (fp8 with or without a scale, or float)."""
    w = module.weight.data
    if is_fp8(w):
        return dequantize(w, getattr(module, "scale_weight", None), dtype)
    return w.to(dtype)


def split_scales(sd: dict):
    """A state dict as loaded from a pre-quantised file -> (state dict without scale / marker keys,
    {module path: scale tensor})."""
    out, scales = {}, {}
    for k, v in sd.items():
        if k.endswith(DROP_KEYS):
            continue
        hit = next((s for s in SCALE_KEYS if k.endswith(s)), None)
        if hit is not None:
            scales[k[: -len(hit)]] = _reshape_scale(v)
            continue
        out[k] = v
    return out, scales


def load_state_dict(model: nn.Module, sd: dict, dtype=torch.bfloat16):
    """Load `sd` into `model` (assign=True), keeping fp8 Linear weights fp8 with their scales and casting everything
    else to `dtype`. fp8 tensors that are not a Linear's weight (biases, norms) are dequantised. Returns
    (missing keys, unexpected keys, number of fp8 Linears)."""
    sd, scales = split_scales(sd)
    linears = {n for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    ready = {}
    for k, v in sd.items():
        mod = k.rsplit(".", 1)[0]
        if is_fp8(v):
            if k.endswith(".weight") and mod in linears:
                ready[k] = v                                   # stays fp8, exactly as stored
            else:
                ready[k] = dequantize(v, scales.get(mod) if k.endswith(".weight") else None, dtype)
        else:
            ready[k] = v.to(dtype) if v.is_floating_point() else v
    missing, unexpected = model.load_state_dict(ready, strict=False, assign=True)
    n = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and is_fp8(m.weight):
            attach(m, scales.get(name))
            n += 1
    if n:
        logger.info(f"[precision] fp8 checkpoint: {n} Linears stay fp8 (dequantised per matmul)")
    return list(missing), list(unexpected), n


@torch.no_grad()
def quantize_module(module: nn.Linear, compute_device) -> bool:
    """Quantise one float Linear to scaled fp8 in place. False if it is already fp8."""
    if is_fp8(module.weight):
        return False
    q, scale = quantize_weight(module.weight.data.to(compute_device))
    module.weight.data = q.to(module.weight.device)
    attach(module, scale.to(module.weight.device))
    return True


@torch.no_grad()
def dequantize_model(model: nn.Module, dtype=torch.bfloat16) -> int:
    """Turn every fp8 Linear back into a dense one (an fp8 file with the bf16 precision chosen)."""
    n = 0
    for m in model.modules():
        if isinstance(m, nn.Linear) and is_fp8(m.weight):
            w = dense_weight(m, dtype)
            detach(m)
            m.weight.data = w
            n += 1
    return n
