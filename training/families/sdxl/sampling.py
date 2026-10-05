# SDXL noise schedule, training objective and sampler for TagScribeR (original code).
#   [sdxl-schedule]  SDXL base scheduler config (stabilityai/stable-diffusion-xl-base-1.0 scheduler/scheduler_config.json):
#                    1000 training steps, scaled_linear betas 0.00085 -> 0.012, epsilon prediction.
#   [ztsnr]          Lin et al. 2023, "Common Diffusion Noise Schedules and Sample Steps are Flawed", Algorithm 1: rescale
#                    sqrt(alpha_bar) so the last step has exactly zero SNR, first step unchanged. Needed by NoobAI-XL's
#                    v-prediction checkpoints (NoobAI model card). diffusers' DDPMScheduler(rescale_betas_zero_snr=True) is
#                    the same algorithm; a test cross-checks this implementation against it.
#   [v-pred]         v = sqrt(alpha_bar) * eps - sqrt(1 - alpha_bar) * x0 (Salimans & Ho 2022; diffusers get_velocity).
#   [min-snr]        Hang et al. 2023 as in kohya sd-scripts train_util.apply_snr_weight: eps loss * min(snr, gamma) / snr,
#                    v-pred loss * min(snr, gamma) / (snr + 1).
#   [noise-offset]   kohya sd-scripts: noise += offset * randn(B, C, 1, 1).
#   [euler]          Euler / Euler-ancestral in sigma space as in diffusers EulerDiscreteScheduler (and k-diffusion):
#                    sigma = sqrt((1 - ab) / ab); model input x / sqrt(sigma^2 + 1); eps: x0 = x - sigma * eps;
#                    v: x0 = x / (sigma^2 + 1) - sigma / sqrt(sigma^2 + 1) * v. "trailing" timestep spacing (Lin et al.).
#   [add-time-ids]   diffusers StableDiffusionXLPipeline._get_add_time_ids: [orig_h, orig_w, crop_top, crop_left,
#                    target_h, target_w].
"""Schedule, loss maths and the Euler sampler. Pure torch, no model code."""
from __future__ import annotations

import math

import torch

TRAIN_STEPS = 1000
BETA_START, BETA_END = 0.00085, 0.012


def rescale_zero_terminal_snr(alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """[ztsnr] Algorithm 1: shift sqrt(alpha_bar) so the last value is 0, rescale so the first is unchanged."""
    s = alphas_cumprod.sqrt()
    first, last = s[0].clone(), s[-1].clone()
    s = (s - last) * (first / (first - last))
    return s ** 2


def alphas_cumprod(zero_terminal_snr: bool = False) -> torch.Tensor:
    """(1000,) float64 cumulative alpha products of the scaled-linear schedule [sdxl-schedule]."""
    betas = torch.linspace(BETA_START ** 0.5, BETA_END ** 0.5, TRAIN_STEPS, dtype=torch.float64) ** 2
    ac = torch.cumprod(1.0 - betas, dim=0)
    return rescale_zero_terminal_snr(ac) if zero_terminal_snr else ac


def draw_timesteps(batch: int, min_t: float, max_t: float, generator) -> torch.Tensor:
    """Integer timesteps in [0, 999]: uniform over the noise window [min_t, max_t] (fractions of the schedule). Fizgig
    semantics: the draw is RESCALED into the window, never clamped. Drawn on the CPU from the loop's generator."""
    u = torch.rand(batch, generator=generator)
    frac = min_t + u * max(max_t - min_t, 1e-6)
    return (frac * TRAIN_STEPS).long().clamp(0, TRAIN_STEPS - 1)


def add_noise(x0, noise, ac_t):
    a = ac_t.view(-1, 1, 1, 1)
    return a.sqrt() * x0 + (1.0 - a).sqrt() * noise


def target_for(prediction: str, x0, noise, ac_t):
    if prediction == "epsilon":
        return noise
    if prediction == "v_prediction":
        a = ac_t.view(-1, 1, 1, 1)
        return a.sqrt() * noise - (1.0 - a).sqrt() * x0                # [v-pred]
    raise ValueError(f"unknown prediction type {prediction!r}")


def min_snr_weights(ac_t: torch.Tensor, gamma: float, prediction: str) -> torch.Tensor:
    """[min-snr] per-sample loss weights."""
    snr = (ac_t / (1.0 - ac_t).clamp_min(1e-12)).float()
    capped = snr.clamp(max=gamma)
    return capped / (snr + 1.0) if prediction == "v_prediction" else capped / snr.clamp_min(1e-8)


def time_ids(height: int, width: int, batch: int = 1, device="cpu", dtype=torch.float32):
    """[add-time-ids] original size = target size = the image's size, crop offset (0, 0): the dataset layer centre-crops
    to the bucket and keeps no crop offsets, so this is the "uncropped, native" conditioning kohya's SDXL training
    also produces for an un-augmented, un-cropped bucket."""
    return torch.tensor([[height, width, 0, 0, height, width]], dtype=dtype, device=device).repeat(batch, 1)


def sigmas_for(steps: int, zero_terminal_snr: bool):
    """(timesteps, sigmas) for `steps` Euler steps with trailing spacing, sigmas ending in 0 [euler]. With zero-terminal
    SNR the last alpha_bar is clamped to 2**-24 (as diffusers does) so the first sigma is finite (4096)."""
    ac = alphas_cumprod(zero_terminal_snr)
    if zero_terminal_snr:
        ac[-1] = max(float(ac[-1]), 2.0 ** -24)
    ts = (torch.arange(TRAIN_STEPS, 0, -TRAIN_STEPS / steps).round().long() - 1).clamp(0, TRAIN_STEPS - 1)
    sig = ((1.0 - ac[ts]) / ac[ts]).sqrt().float()
    return ts, torch.cat([sig, sig.new_zeros(1)])


def initial_noise(seed: int, width: int, height: int):
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(1, 4, height // 8, width // 8, generator=g)


@torch.no_grad()
def euler_sample(predict, noise, steps: int, *, prediction: str, zero_terminal_snr: bool, ancestral: bool = False,
                 seed: int = 0, on_step=None):
    """Denoise `noise` (unit variance, (1, 4, h, w)). `predict(x_in, t)` -> the model output for a scaled input and an
    integer timestep (the caller owns CFG). Returns the final latents (sigma 0)."""
    ts, sig = sigmas_for(steps, zero_terminal_snr)
    x = noise.float() * math.sqrt(float(sig[0]) ** 2 + 1.0)
    g = torch.Generator().manual_seed(int(seed) + 7919)
    for i in range(steps):
        if on_step:
            on_step(i, steps)
        s, s_next = float(sig[i]), float(sig[i + 1])
        out = predict(x / math.sqrt(s * s + 1.0), int(ts[i])).float()
        if prediction == "v_prediction":
            x0 = x / (s * s + 1.0) - s / math.sqrt(s * s + 1.0) * out
        else:
            x0 = x - s * out
        d = (x - x0) / s
        if ancestral and s_next > 0:
            up = min(s_next, math.sqrt(s_next ** 2 * (s ** 2 - s_next ** 2) / s ** 2))
            down = math.sqrt(s_next ** 2 - up ** 2)
            x = x + d * (down - s) + torch.randn(x.shape, generator=g).to(x.device) * up
        else:
            x = x + d * (s_next - s)
    return x
