# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/krea2/sampling.py (+ the timestep sampler
# `sample_krea2_timesteps` of krea2/trainer.py).
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: einops replaced by plain torch reshapes; the sampler returns latents and leaves the VAE
# decode to the driver (the generic preview loop keeps the VAE resident); cooperative abort is the generic on_step
# callback; the text pad multiple is a constant (Fizgig reads FIZGIG_ATTN_TRIM_MULTIPLE); the training timestep sampler
# takes the loop's seeded generator; the CLI-style encode_prompts helper is not ported.
# Fizgig's own header follows.
#
# Upstream: functional flow-matching sampler ported from ai-toolkit (Ostris, LLC - MIT;
# https://github.com/ostris/ai-toolkit, extensions_built_in/diffusion_models/flux2/src/sampling.py), adapted for
# Fizgig. See THIRD_PARTY_NOTICES.md. The timestep recipe is musubi-tuner's `krea2_shift` (Apache-2.0).
"""Krea 2 flow-matching helpers: patchify / positions / masks, text gathering, the resolution-shifted timestep
schedule, the training timestep sampler and the Euler sampler with optional CFG."""
import logging
import math

import torch

logger = logging.getLogger(__name__)

# Fizgig attention.py _TRIM_MULTIPLE default (env FIZGIG_ATTN_TRIM_MULTIPLE=64): gather_valid_text rounds the padded
# text length up to a multiple of it. Fizgig measured this as a deliberately accepted perturbation (~5e-3 relative on
# short captions, bf16) in exchange for fewer distinct shapes; the value is kept so training sees what Fizgig's does.
TEXT_PAD_MULTIPLE = 64


def roundup(value, multiple, name):
    """Round `value` up to the nearest multiple, logging when padding is applied."""
    aligned = ((value + multiple - 1) // multiple) * multiple
    if aligned != value:
        logger.info(f"[sample] {name}={value} is not a multiple of {multiple}; padding to {aligned}")
    return aligned


def gather_valid_text(txt, mask):
    """Drop masked (invalid) text tokens so the valid ones form a contiguous prefix, then right-pad to the batch
    maximum (rounded up to TEXT_PAD_MULTIPLE). The Qwen3-VL conditioner pads the prompt to max_length and appends the
    template suffix, so its mask is [valid prompt, pad, valid suffix]; dropping the interior padding is lossless for
    the attention (text tokens get zero RoPE position and padding is masked out).

    txt: (B, seq, L, D), mask: (B, seq) bool -> (B, max_valid, L, D), (B, max_valid) bool."""
    valid = [txt[i][mask[i]] for i in range(txt.shape[0])]
    max_len = max(v.shape[0] for v in valid)
    if TEXT_PAD_MULTIPLE > 1:
        max_len = ((max_len + TEXT_PAD_MULTIPLE - 1) // TEXT_PAD_MULTIPLE) * TEXT_PAD_MULTIPLE
    out = txt.new_zeros(txt.shape[0], max_len, txt.shape[2], txt.shape[3])
    newmask = torch.zeros(txt.shape[0], max_len, device=txt.device, dtype=torch.bool)
    for i, v in enumerate(valid):
        out[i, : v.shape[0]] = v
        newmask[i, : v.shape[0]] = True
    return out, newmask


def patchify(img, patch):
    """(B, C, H, W) -> (B, h*w, C*patch*patch), 'b c (h ph) (w pw) -> b (h w) (c ph pw)'."""
    b, c, H, W = img.shape
    h, w = H // patch, W // patch
    return img.reshape(b, c, h, patch, w, patch).permute(0, 2, 4, 1, 3, 5).reshape(b, h * w, c * patch * patch)


def unpatchify(tokens, h, w, patch):
    """(B, h*w, C*patch*patch) -> (B, C, 1, h*patch, w*patch) (the VAE's frame axis added)."""
    b, _, cpp = tokens.shape
    c = cpp // (patch * patch)
    x = tokens.reshape(b, h, w, c, patch, patch).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(b, c, 1, h * patch, w * patch)


def patchify_block(img, patch, frame=0.0):
    """One image block -> (tokens, pos, mask), positions on the 3-axis RoPE grid. `frame` writes the FIRST RoPE axis
    (32 head-dims are allocated to it; every path here uses 0)."""
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    imgids = torch.zeros((h_, w_, 3), device=img.device)
    if frame:
        imgids[..., 0] = float(frame)
    imgids[..., 1] = torch.arange(h_, device=img.device)[:, None]
    imgids[..., 2] = torch.arange(w_, device=img.device)[None, :]
    imgpos = imgids.reshape(1, h_ * w_, 3).expand(b, -1, -1)
    imgmask = torch.ones(b, h_ * w_, device=img.device, dtype=torch.bool)
    return patchify(img, patch), imgpos, imgmask


def prepare(img, txtlen, patch, txtmask):
    """Patchify the latent and build the combined image + text position / mask tensors. Image tokens lead the
    sequence ([img (all valid), text (valid prefix + padding)]). Returns (img_tokens, pos, mask)."""
    b = img.shape[0]
    tokens, imgpos, imgmask = patchify_block(img, patch)
    txtpos = torch.zeros(b, txtlen, 3, device=img.device)
    return tokens, torch.cat((imgpos, txtpos), dim=1), torch.cat((imgmask, txtmask), dim=1)


def timesteps(seq_len, steps, x1, x2, y1=0.5, y2=1.15, sigma=1.0, mu=None):
    """Resolution-aware flow-matching timestep schedule (t: 1 -> 0). `mu` is interpolated linearly in image-sequence
    length between (x1,y1) and (x2,y2), then used to time-shift a uniform 1->0 grid. An explicit `mu` pins a constant
    shift regardless of resolution (the distilled Turbo was trained at a fixed mu=1.15)."""
    ts = torch.linspace(1, 0, steps + 1)
    if mu is None:
        slope = (y2 - y1) / (x2 - x1)
        mu = slope * seq_len + (y1 - slope * x1)
    ts = math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0) ** sigma)
    return ts.tolist()


def _get_lin_function(x1, y1, x2, y2):
    """Linear map through (x1,y1)-(x2,y2): f(x) = m*x + b (musubi's get_lin_function)."""
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


# Krea 2 resolution->mu schedule (musubi `krea2_shift`): token count maps to mu, shift = exp(mu). Endpoints match the
# inference defaults (minres 256, maxres 1280 at align 16): x1 = (256//16)**2 = 256, x2 = (1280//16)**2 = 6400,
# y1 = 0.5, y2 = 1.15 (Fizgig krea2/trainer.py _KREA2_MU).
KREA2_MU = _get_lin_function(256, 0.5, 6400, 1.15)


def sample_krea2_timesteps(bsize: int, num_img_tokens: int, device="cpu", sigmoid_scale: float = 1.0,
                           min_timestep: float = 0.0, max_timestep: float = 1.0, generator=None) -> torch.Tensor:
    """Krea 2 'krea2_shift' timestep sampling (Fizgig krea2/trainer.py sample_krea2_timesteps). The base t is
    logit-normal, then shifted by exp(mu) with mu from the image-token count:

        t_base = sigmoid(randn * sigmoid_scale)
        t      = (t_base * shift) / (1 + (shift - 1) * t_base)

    An optional [min_timestep, max_timestep] window RESCALES t into it (clamping would pile probability mass onto the
    two endpoints)."""
    shift = math.exp(KREA2_MU(num_img_tokens))
    t = (torch.randn(bsize, generator=generator) * sigmoid_scale).sigmoid()
    t = (t * shift) / (1.0 + (shift - 1.0) * t)
    if min_timestep > 0.0 or max_timestep < 1.0:
        lo, hi = max(0.0, float(min_timestep)), min(1.0, float(max_timestep))
        t = lo + t * max(hi - lo, 1e-6)
    return t.to(device)


def initial_noise(seed, width, height, compression=8, channels=16):
    """The seed's starting noise, (1, C, H/8, W/8) float32 on CPU (what sample_latents draws for this seed)."""
    gen = torch.Generator("cpu").manual_seed(int(seed))
    return torch.randn(1, channels, height // compression, width // compression, generator=gen)


@torch.no_grad()
def sample_latents(model, txt, txtmask, *, untxt=None, untxtmask=None, device="cuda", dtype=torch.bfloat16,
                   width=1024, height=1024, steps=28, cfg_scale=5.5, seed=0, minres=256, maxres=1280, y1=0.5,
                   y2=1.15, mu=None, noise=None, on_step=None, compression=8, channels=16):
    """Euler integration of the flow ODE (+ CFG when cfg_scale > 1 and an unconditional text is given). txt (B, L, 12,
    D) gathered text stacks with their masks (sampling.gather_valid_text). Returns the final latents in the VAE's
    layout (B, C, 1, H/8, W/8), normalised (decode with the VAE's decode_to_pixels).

    Fizgig's sample() (without its VAE decode); the DiT stays resident on its device. on_step(done, total) is called
    before every step and may raise to abort."""
    patch = model.config.patch
    align = compression * patch
    width, height = roundup(width, align, "width"), roundup(height, align, "height")
    n = txt.shape[0]
    cfg = cfg_scale > 1.0 and untxt is not None
    txt, txtmask = txt.to(device=device, dtype=dtype), txtmask.to(device)
    if cfg:
        untxt, untxtmask = untxt.to(device=device, dtype=dtype), untxtmask.to(device)
    expected = (n, channels, height // compression, width // compression)
    if noise is not None:
        noise = noise.to(device=device, dtype=dtype)
        if tuple(noise.shape) != expected:
            raise ValueError(f"sample(noise=...) shape {tuple(noise.shape)} != expected {expected}")
    else:
        noise = torch.cat([torch.randn(1, *expected[1:], generator=torch.Generator("cpu").manual_seed(seed + i))
                           for i in range(n)], dim=0).to(device=device, dtype=dtype)
    x, pos, mask = prepare(noise, txt.shape[1], patch, txtmask)
    if cfg:
        _, unpos, unmask = prepare(noise, untxt.shape[1], patch, untxtmask)
    x1 = (minres // align) ** 2
    x2 = (maxres // align) ** 2
    ts = timesteps(x.shape[1], steps, x1, x2, y1=y1, y2=y2, mu=mu)
    img = x
    for i, (tcurr, tprev) in enumerate(zip(ts[:-1], ts[1:])):
        if on_step is not None:
            on_step(i, len(ts) - 1)
        t = torch.full((len(img),), tcurr, dtype=img.dtype, device=img.device)
        cond = model(img=img, context=txt, t=t, pos=pos, mask=mask)
        if cfg:
            uncond = model(img=img, context=untxt, t=t, pos=unpos, mask=unmask)
            v = uncond + cfg_scale * (cond - uncond)
        else:
            v = cond
        img = img + (tprev - tcurr) * v
    return unpatchify(img, height // align, width // align, patch)
