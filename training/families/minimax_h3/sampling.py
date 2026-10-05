# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/minimax/sampling.py (sample_schedule,
# _res_multistep_coeffs, the still-image path of _sample_image_impl) and minimax/trainer.py (sample_sigmas).
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Fizgig's res_multistep step is ComfyUI's (comfy/k_diffusion/sampling.py) - see THIRD_PARTY_NOTICES.md.
# Changes for TagScribeR: one still per call (no clips, references, keyframes, activation cache, slow-step notices or
# the latent-RGB thumbnail); the sampler's noise is drawn from one CPU generator (video, then the audio rows).
"""H3 training timesteps and image sampling.

Flow / sign convention (the easy thing to get backwards, matched to ComfyUI's comfy/ldm/minimax/model.py):
  x0 = clean latent, noise ~ N(0,1), sigma in (0,1) the noise level.
  noised = (1 - sigma) * x0 + sigma * noise            (sigma 0 = clean, 1 = pure noise)
  t = 1 - sigma                                         the "cleanness" fed to the time embedder
  the DiT's raw head output predicts (x0 - noise)       (the reference NEGATES it to get a sampler's velocity)
So the training target is `x0 - noise`, and integrating the RAW head output from sigma = 1 to 0 is
  x += (sigma_curr - sigma_next) * out.
"""
import math

import torch

from training.families.minimax_h3.model import (AUDIO_CHANNELS, AUDIO_SIGMA_SHIFT, VIDEO_SIGMA_SHIFT,
                                                audio_latents_for_frames, remap_sigma)

LOWNOISE_SIGMA = 0.5        # "low noise" = the cleaner half of the sigma range (Fizgig MINIMAX_LOWNOISE_SIGMA)


def lownoise_to_shift(pct):
    """Share of training steps below sigma 0.5 (percent) -> the shift that produces it: shift = (1 - P) / P. 50% -> 1
    (unshifted uniform), 22% -> 3.5, 7.7% -> 12 (H3's own video default). None when unusable (0 and 100 are the
    asymptotes, not values). Fizgig lora_trainer_gui.py minimax_lownoise_to_shift."""
    try:
        p = float(str(pct).strip().rstrip("%")) / 100.0
    except (TypeError, ValueError):
        return None
    if not (0.0 < p < 1.0):
        return None
    return (1.0 - p) / p


def sample_sigmas(batch: int, shift: float = VIDEO_SIGMA_SHIFT, generator=None, device=None) -> torch.Tensor:
    """Training noise levels in (0,1): u ~ uniform, sigma = shift*u / (1 + (shift-1)*u) (Fizgig sample_sigmas with a
    numeric shift)."""
    u = torch.rand(batch, generator=generator)
    s = float(shift)
    sig = (s * u) / (1.0 + (s - 1.0) * u)
    return sig.to(device) if device is not None else sig


def sample_schedule(steps: int, shift: float = VIDEO_SIGMA_SHIFT):
    """Descending sigmas 1 -> 0 as ComfyUI's `simple` scheduler walks ModelSamplingDiscreteFlow(shift=12): a
    1000-entry table sigma = shift*u/(1+(shift-1)*u), u = (i+1)/1000, walked backwards in even strides, then 0.
    `steps` model evaluations, steps + 1 sigmas."""
    if steps < 1:
        raise ValueError("sample_schedule needs at least one step")
    table = [(shift * u) / (1.0 + (shift - 1.0) * u) for u in ((i + 1) / 1000.0 for i in range(1000))]
    stride = len(table) / steps
    return [table[-(1 + int(x * stride))] for x in range(steps)] + [0.0]


def _res_multistep_coeffs(sig_cur, sig_next, sig_prev):
    """(exp(-h), h*b1, h*b2) for ComfyUI's res_multistep second-order step (eta = 0), float64:
        x <- exp(-h) * x + h * (b1 * denoised + b2 * denoised_prev)"""
    t = -math.log(max(float(sig_cur), 1e-12))
    t_next = -math.log(max(float(sig_next), 1e-12))
    t_prev = -math.log(max(float(sig_prev), 1e-12))
    h = t_next - t
    c2 = (t_prev - t) / h
    phi1 = math.expm1(-h) / -h
    phi2 = (phi1 - 1.0) / -h
    b1 = phi1 - phi2 / c2
    b2 = phi2 / c2
    return math.exp(-h), h * b1, h * b2


def latent_grid(width: int, height: int, spatial: int = 16):
    """(lat_h, lat_w): the latent grid, floored to even (the DiT patchifies 2x2)."""
    return (height // spatial // 2) * 2, (width // spatial // 2) * 2


def initial_noise(seed: int, width: int, height: int, channels: int = 24):
    """The seed's starting video noise (1, C, 1, h, w) float32 on the CPU - the first draw of generate()'s generator."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    lat_h, lat_w = latent_grid(width, height)
    return torch.randn(1, channels, 1, lat_h, lat_w, generator=gen, dtype=torch.float32)


@torch.no_grad()
def sample_image(model, text_embeds, *, width=512, height=512, steps=20, cfg_scale=1.0, uncond_embeds=None, seed=0,
                 shift=VIDEO_SIGMA_SHIFT, noise=None, sampler="res_multistep", on_step=None, device=None,
                 dtype=torch.bfloat16):
    """Denoise one still and return its latent [1, 24, 1, H/16, W/16] (decoding is the caller's).

    The audio rows (packed silence in training) are DENOISED jointly on their own shift-3 schedule exactly like the
    reference pipeline, as a carried variable (ComfyUI's ModelSamplingAV scheme). cfg_scale <= 1 runs one forward per
    step (what every shipped H3 workflow does); above 1 a second forward on `uncond_embeds`."""
    device = device or next(model.parameters()).device
    lat_h, lat_w = latent_grid(width, height)
    channels = model.config.latents_dim
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    x0n = torch.randn(1, channels, 1, lat_h, lat_w, generator=gen, dtype=torch.float32)
    x = (x0n if noise is None else noise.to(torch.float32)).to(device)
    joint_audio = bool(getattr(model, "pack_audio_rows", False))
    audio_rows = None
    if joint_audio:
        n_audio = audio_latents_for_frames(1) * AUDIO_CHANNELS
        audio_rows = torch.randn(n_audio, model.config.audio_latents_dim, generator=gen, dtype=torch.float32).to(device)
    use_cfg = cfg_scale > 1.0 and uncond_embeds is not None
    sigmas = sample_schedule(steps, shift=shift)
    n_eval = len(sigmas) - 1
    prev_denoised = prev_denoised_a = None
    for i in range(n_eval):
        if on_step is not None:
            on_step(i, n_eval)
        s_curr, s_next = sigmas[i], sigmas[i + 1]
        t = torch.tensor([1.0 - s_curr], device=device)          # the DiT is conditioned on cleanness
        a_out = None
        if joint_audio:
            _sv = max(float(s_curr), 1e-6)
            _sa = float(remap_sigma(torch.tensor(_sv), shift, AUDIO_SIGMA_SHIFT))
            _ascale = float(shift) / float(AUDIO_SIGMA_SHIFT)
            a_in = (audio_rows * (_sa / _sv)).to(dtype)
            out, a_raw = model(x.to(dtype), t, text_embeds, audio_rows=a_in, return_audio=True)
            out = out.float()
            if use_cfg:
                out_u, a_raw_u = model(x.to(dtype), t, uncond_embeds, audio_rows=a_in, return_audio=True)
                out = out_u.float() + cfg_scale * (out - out_u.float())
                if a_raw is not None and a_raw_u is not None:
                    a_raw = a_raw_u.float() + cfg_scale * (a_raw.float() - a_raw_u.float())
            if a_raw is not None:
                # comfy's formula is written for out = noise - x0; ours is x0 - noise, so the latent term flips once
                a_out = (_ascale - 1.0) * a_in.float() + (1.0 + (_ascale - 1.0) * _sa) * a_raw.float()
        else:
            out = model(x.to(dtype), t, text_embeds).float()
            if use_cfg:
                out_u = model(x.to(dtype), t, uncond_embeds).float()
                out = out_u + cfg_scale * (out - out_u)
        denoised = x + s_curr * out
        if a_out is not None:
            denoised_a = audio_rows + s_curr * a_out
        second = sampler == "res_multistep" and prev_denoised is not None and s_next > 0
        if second:
            a_x, hb1, hb2 = _res_multistep_coeffs(s_curr, s_next, sigmas[i - 1])
            x = a_x * x + hb1 * denoised + hb2 * prev_denoised
        else:
            x = x + (s_curr - s_next) * out
        if a_out is not None:
            if second and prev_denoised_a is not None:
                audio_rows = a_x * audio_rows + hb1 * denoised_a + hb2 * prev_denoised_a
            else:
                audio_rows = audio_rows + (s_curr - s_next) * a_out
            prev_denoised_a = denoised_a
        prev_denoised = denoised
    return x
