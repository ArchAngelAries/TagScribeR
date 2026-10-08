# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/klein/position.py (token packing and 4D
# position ids), klein/model_utils.py (get_schedule, denoise, denoise_cfg) and training/trainer.py
# (get_noisy_model_input_and_timesteps, call_dit's target; training/train_utils.py get_lin_function).
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: einops is replaced by reshapes; training noise and timesteps are drawn on the CPU from the
# loop's seeded generator (Fizgig draws on the device from the global RNG); the sampler takes latents / positions in
# the driver's layout and leaves the VAE decode to the driver; the dataset timestep buckets (num_timestep_buckets) and
# the SD3 loss weightings (weighting_scheme sigma_sqrt / cosmap, logit_normal / mode densities) are not ported - Fizgig's
# Klein GUI hides them and always runs "none".
"""Klein flow-matching helpers: packing, position ids, the training timestep samplers and the Euler sampler."""
import logging
import math

import torch

logger = logging.getLogger(__name__)

TIMESTEP_MODES = ("sigma", "uniform", "sigmoid", "shift", "flux_shift", "flux2_shift", "logsnr", "qinglong_flux")


# ---- packing and positions (klein/position.py) ------------------------------------------------------------------
def pack_img(x):
    """(B, C, H, W) latents -> ((B, H*W, C) tokens, (B, H*W, 4) ids [t, h, w, l]). Fizgig prc_img."""
    b, c, h, w = x.shape
    coords = torch.cartesian_prod(torch.arange(1), torch.arange(h), torch.arange(w), torch.arange(1))
    ids = coords.unsqueeze(0).expand(b, -1, -1)
    return x.flatten(2).transpose(1, 2), ids.to(x.device)


def pack_txt(x):
    """(B, T, D) text embeddings -> (same, (B, T, 4) ids [0, 0, 0, l]). Fizgig prc_txt."""
    t = x.shape[-2]
    coords = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(t))
    return x, coords.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.device)


def pack_refs(refs):
    """Reference ("before") latents [(B, C, h, w) or (C, h, w), ...] -> ((B, sum h*w, C) tokens, (B, sum h*w, 4) ids)
    with time coordinates 10, 20, ... so the DiT tells them from the image being made (Fizgig klein/position.py
    pack_control_latent), or (None, None)."""
    if not refs:
        return None, None
    toks, ids = [], []
    for i, r in enumerate(refs):
        r = r[None] if r.dim() == 3 else r
        b, _, h, w = r.shape
        coords = torch.cartesian_prod(torch.tensor([10 + 10 * i]), torch.arange(h), torch.arange(w), torch.arange(1))
        toks.append(r.flatten(2).transpose(1, 2))
        ids.append(coords.unsqueeze(0).expand(b, -1, -1).to(r.device))
    return torch.cat(toks, dim=1), torch.cat(ids, dim=1)


def unpack_img(tokens, h, w):
    """(B, H*W, C) -> (B, C, H, W): the inverse of pack_img (Fizgig: rearrange 'b (h w) c -> b c h w')."""
    return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], h, w)


# ---- training timesteps (trainer.py:791-960) ----------------------------------------------------------------------
def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
    """Fizgig training/train_utils.py:249."""
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def _uniform_to_normal(t):
    eps = 1e-7
    t = torch.clamp(t, eps, 1.0 - eps)
    return math.sqrt(2.0) * torch.erfinv(2.0 * t - 1.0)


def _base_t(mode, n, h, w, g, sigmoid_scale, shift, logit_mean, logit_std):
    """trainer.py compute_sampling_timesteps: n draws in (0, 1) (h, w = the packed latent's size)."""
    def randn(k):
        return torch.randn((k,), generator=g)

    def rand(k):
        return torch.rand((k,), generator=g)

    def logsnr(k, mean, std):
        return torch.normal(mean=mean, std=std, size=(k,), generator=g)

    if mode == "uniform":
        return rand(n)
    if mode == "sigmoid":
        return torch.sigmoid(sigmoid_scale * randn(n))
    if mode in ("shift", "flux_shift", "flux2_shift"):
        if mode == "shift":
            s = shift
        elif mode == "flux_shift":
            s = math.exp(get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2)))
        else:
            s = math.exp(get_lin_function(y1=0.5, y2=1.15)(h * w))
        t = (randn(n) * sigmoid_scale).sigmoid()
        return (t * s) / (1 + (s - 1) * t)
    if mode == "logsnr":
        return torch.sigmoid(-logsnr(n, logit_mean, logit_std) / 2)
    if mode == "qinglong_flux":
        # Qinglong triple hybrid: mid_shift:logsnr:logsnr2 = 0.80:0.075:0.125
        decision = rand(n)
        mid = decision < 0.80
        l1 = (decision >= 0.80) & (decision < 0.875)
        l2 = decision >= 0.875
        t = torch.zeros((n,))
        if mid.any():
            s = math.exp(get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2)))
            tm = (randn(int(mid.sum())) * sigmoid_scale).sigmoid()
            t[mid] = (tm * s) / (1 + (s - 1) * tm)
        if l1.any():
            t[l1] = torch.sigmoid(-logsnr(int(l1.sum()), logit_mean, logit_std) / 2)
        if l2.any():
            t[l2] = torch.sigmoid(-logsnr(int(l2.sum()), 5.36, 1.0) / 2)
        return t
    raise ValueError(f"unknown timestep sampling {mode!r}; choose one of {TIMESTEP_MODES}")


def sample_timesteps(mode, bsz, h, w, generator, *, min_t=0.0, max_t=1.0, sigmoid_scale=1.0, shift=3.0,
                     logit_mean=0.0, logit_std=1.0, preserve=False):
    """Fizgig trainer.get_noisy_model_input_and_timesteps. Returns (t, t_model): `t` (B,) the noise fraction the
    latents are mixed with (x_t = (1 - t) x0 + t noise); `t_model` (B,) what the DiT is told.

    * the window [min_t, max_t] (0-1; Fizgig's boxes are 0-1000) RESCALES the drawn curve into it, or with
      `preserve` (Fizgig's --preserve_distribution_shape) rejection-samples it (1000 tries, then the plain draw);
    * every mode but "sigma": the DiT gets t + 0.001 (Fizgig: `timesteps = t * 1000; timesteps += 1`, later / 1000);
    * "sigma": a uniform draw over the 1000-step schedule restricted to the window; t and the DiT's value are the
      step's sigma, 1 - idx / 1000 (no +0.001)."""
    if mode not in TIMESTEP_MODES:
        raise ValueError(f"unknown timestep sampling {mode!r}; choose one of {TIMESTEP_MODES}")
    if mode == "sigma":
        u = torch.rand((bsz,), generator=generator)
        t_lo, t_hi = round(min_t * 1000.0, 6), round(max_t * 1000.0, 6)
        n = 1000
        lo_idx, hi_idx = n - t_hi, n - t_lo           # the schedule is descending: timesteps[i] = 1000 - i
        idx = (u * (hi_idx - lo_idx) + lo_idx).long().clamp(0, n - 1)
        sigma = torch.linspace(1, 0, n + 1)[idx]
        return sigma, sigma.clone()
    args = (mode, h, w, generator, sigmoid_scale, shift, logit_mean, logit_std)
    if not preserve:
        t = _base_t(mode, bsz, *args[1:]) * (max_t - min_t) + min_t
    else:
        kept = []
        for _ in range(1000):
            for ti in _base_t(mode, bsz, *args[1:]):
                if min_t <= ti <= max_t:
                    kept.append(ti)
                if len(kept) == bsz:
                    break
            if len(kept) == bsz:
                break
        if len(kept) < bsz:
            logger.warning(f"Could not sample {bsz} valid timesteps in 1000 loops")
            t = _base_t(mode, bsz, *args[1:])
        else:
            t = torch.stack(kept, dim=0)
    return t, t + 0.001                                # (t * 1000 + 1) / 1000


# ---- sampling schedule and loop (klein/model_utils.py) -------------------------------------------------------------
def generalized_time_snr_shift(t, mu: float, sigma: float):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def get_schedule(num_steps: int, image_seq_len: int, flow_shift=None) -> list:
    """Fizgig get_schedule: num_steps + 1 values from 1 to 0, shifted by the empirical mu (or a fixed flow shift)."""
    mu = compute_empirical_mu(image_seq_len, num_steps)
    timesteps = torch.linspace(1, 0, num_steps + 1)
    if flow_shift is not None:
        timesteps = (timesteps * flow_shift) / (1 + (flow_shift - 1) * timesteps)
    else:
        timesteps = generalized_time_snr_shift(timesteps, mu, 1.0)
    return timesteps.tolist()


def roundup(value, multiple, name):
    aligned = ((value + multiple - 1) // multiple) * multiple
    if aligned != value:
        logger.info(f"[sample] {name}={value} is not a multiple of {multiple}; padding to {aligned}")
    return aligned


def initial_noise(seed, width, height, channels=128):
    """(1, 128, H/16, W/16) float32 on the CPU, from the seed."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn((1, channels, height // 16, width // 16), generator=g, dtype=torch.float32)


@torch.no_grad()
def sample_latents(dit, ctx, neg_ctx, *, device, width, height, steps, cfg, seed, noise=None, on_step=None,
                   channels=128, refs=None):
    """Fizgig do_inference (Base model, no reference image): Euler over get_schedule; with a negative prompt and
    cfg > 1, classifier-free guidance pred = uncond + cfg * (cond - uncond) (two passes); latents stay bf16 through the
    loop and the DiT runs under bf16 autocast, as denoise() / denoise_cfg() do. ctx / neg_ctx (1, T, D).
    Returns (1, C, H/16, W/16) latents."""
    latents = noise if noise is not None else initial_noise(seed, width, height, channels)
    x, x_ids = pack_img(latents.to(device=device, dtype=torch.bfloat16))
    ctx, ctx_ids = pack_txt(ctx.to(device=device, dtype=torch.bfloat16))
    use_cfg = neg_ctx is not None and cfg > 1.0
    if use_cfg:
        neg_ctx, neg_ids = pack_txt(neg_ctx.to(device=device, dtype=torch.bfloat16))
    timesteps = get_schedule(steps, x.shape[1])
    n = x.shape[1]
    ref_tok, ref_ids = pack_refs([r.to(device=device, dtype=torch.bfloat16) for r in refs] if refs else None)
    total = len(timesteps) - 1
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        if on_step is not None:
            on_step(i, total)
        t_vec = torch.full((x.shape[0],), t_curr, dtype=x.dtype, device=device)
        xin, xin_ids = (torch.cat((x, ref_tok), 1), torch.cat((x_ids, ref_ids), 1)) if ref_tok is not None \
            else (x, x_ids)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            pred = dit(x=xin, x_ids=xin_ids, timesteps=t_vec, ctx=ctx, ctx_ids=ctx_ids, guidance=None)[:, :n]
            if use_cfg:
                pred_uncond = dit(x=xin, x_ids=xin_ids, timesteps=t_vec, ctx=neg_ctx, ctx_ids=neg_ids,
                                  guidance=None)[:, :n]
                pred = pred_uncond + cfg * (pred - pred_uncond)
        x = x + (t_prev - t_curr) * pred
    return unpack_img(x, height // 16, width // 16)
