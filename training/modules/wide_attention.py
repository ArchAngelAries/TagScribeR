"""Attention for heads wider than 256 dimensions - the single-head attention block of image VAEs.

`torch.nn.functional.scaled_dot_product_attention` is wrong on AMD ROCm builds of PyTorch when the head dimension is
above 256. Measured on torch 2.12.0+rocm7.15 (RX 7900 XT), one head, random inputs, against the same maths done by
hand:

    head dim <= 256   every backend correct
    head dim 384      default / flash / memory-efficient backends: garbage in float32 (relative error 1 to 60), NaN in
                      bf16; the "math" backend is correct

The fast kernels were never meant for heads that wide (on CUDA PyTorch falls back to the math path by itself); the
ROCm build runs them anyway. The Qwen-Image VAE (Krea 2, Qwen Image 2.1) has one 384-wide head and the FLUX.2 VAE
(Klein) one 512-wide head over every latent position, so on ROCm each encode and decode went through the broken
path: decoded previews carried bright horizontal streaks at fixed heights, and cached latents were off by a few
percent. Fizgig has the same behaviour; this is a TagScribeR fix.

`attention` uses the built-in function for ordinary heads and computes wide ones by hand, in float32 and in row
chunks so the N x N attention matrix is never held at once (N = 16384 at 1024 px).
"""
import torch
import torch.nn.functional as F

MAX_FAST_HEAD_DIM = 256
_CHUNK_ELEMENTS = 2 ** 27          # attention-matrix elements per chunk: 512 MB in float32


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """softmax(q k^T / sqrt(d)) v for (..., N, d) tensors, no mask. Returns q's dtype."""
    if q.shape[-1] <= MAX_FAST_HEAD_DIM:
        return F.scaled_dot_product_attention(q, k, v)
    dtype = q.dtype
    q, kt, v = q.float() * (q.shape[-1] ** -0.5), k.float().transpose(-1, -2), v.float()
    rows = max(1, _CHUNK_ELEMENTS // max(1, k.shape[-2]))
    out = [torch.softmax(q[..., i:i + rows, :] @ kt, dim=-1) @ v for i in range(0, q.shape[-2], rows)]
    return torch.cat(out, dim=-2).to(dtype)
