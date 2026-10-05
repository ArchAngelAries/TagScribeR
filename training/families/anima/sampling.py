# Facts from kohya sd-scripts (Apache-2.0) anima_train_network.py get_noise_pred_and_target / flux_train_utils
# (timestep sampling), anima_minimal_inference.py + library/hunyuan_image_utils.py get_timesteps_sigmas / step
# (the Euler sampler), diffusion-pipe models/cosmos_predict2.py prepare_inputs (the same rectified-flow objective), and
# ComfyUI comfy/supported_models.py Anima sampling_settings (multiplier 1.0, shift 3.0; GPL: facts only). Code is
# TagScribeR's own. See THIRD_PARTY_NOTICES.md.
"""Anima flow-matching helpers: the training timestep sampler and the Euler sampler with optional CFG."""
import torch

# Training: plain logit-normal t = sigmoid(N(0, 1) * scale), NO shift (sd-scripts anima defaults --timestep_sampling
# sigmoid, --sigmoid_scale 1.0, --discrete_flow_shift 1.0 (= identity); diffusion-pipe's logit_normal default likewise).
SIGMOID_SCALE = 1.0
TRAIN_SHIFT = 1.0
# Inference: ComfyUI's Anima model sampling is a flow with shift 3.0 (sd-scripts' minimal inference script defaults
# to 5.0; the model card gives no value). The model receives sigma itself (multiplier 1.0), not sigma * 1000.
DEFAULT_SHIFT = 3.0


def shifted(t, shift):
    return (t * shift) / (1.0 + (shift - 1.0) * t)


def sample_training_timesteps(bsize, min_t=0.0, max_t=1.0, generator=None, sigmoid_scale=SIGMOID_SCALE,
                              shift=TRAIN_SHIFT):
    """t in (0, 1), CPU float32: sigmoid(randn * scale), optionally shifted. A [min_t, max_t] window RESCALES t into it
    (clamping would pile probability mass onto the two endpoints)."""
    t = (torch.randn(bsize, generator=generator) * sigmoid_scale).sigmoid()
    if shift != 1.0:
        t = shifted(t, shift)
    if min_t > 0.0 or max_t < 1.0:
        lo, hi = max(0.0, float(min_t)), min(1.0, float(max_t))
        t = lo + t * max(hi - lo, 1e-6)
    return t


def sigma_schedule(steps, shift=DEFAULT_SHIFT):
    """Euler schedule sigma_0 = 1 -> sigma_steps = 0 on the shifted grid (steps + 1 values)."""
    return shifted(torch.linspace(1.0, 0.0, steps + 1), shift)


def initial_noise(seed, width, height, compression=8, channels=16):
    """The seed's starting noise, (1, C, H/8, W/8) float32 on CPU (what sample_latents draws for this seed)."""
    gen = torch.Generator("cpu").manual_seed(int(seed))
    return torch.randn(1, channels, height // compression, width // compression, generator=gen)


def _batched(cond, device, dtype):
    """One caption's conditioning (unbatched, as the preview loop holds it) or a batch -> model arguments."""
    e, qm, ti, tm = cond["prompt_embeds"], cond["attn_mask"], cond["t5_ids"], cond["t5_mask"]
    if e.dim() == 2:
        e, qm, ti, tm = e[None], qm[None], ti[None], tm[None]
    return (e.to(device=device, dtype=dtype), ti.to(device=device, dtype=torch.long), tm.to(device).bool(),
            qm.to(device).bool())


@torch.no_grad()
def sample_latents(model, cond, *, uncond=None, device="cuda", dtype=torch.bfloat16, width=1024, height=1024,
                   steps=30, cfg_scale=4.0, seed=0, shift=DEFAULT_SHIFT, noise=None, on_step=None, compression=8,
                   channels=16):
    """Euler integration of the flow ODE x <- x - (sigma_i - sigma_{i+1}) * v (+ CFG v = u + g * (c - u) when
    cfg_scale > 1 and an unconditional conditioning is given). Returns the final latents (1, C, 1, H/8, W/8),
    normalised (decode with the VAE's decode_to_pixels). on_step(done, total) is called before every step and may
    raise to abort."""
    width, height = (-(-width // 16)) * 16, (-(-height // 16)) * 16        # 8 (VAE) x 2 (patch)
    expected = (1, channels, height // compression, width // compression)
    if noise is not None:
        if tuple(noise.shape) != expected:
            raise ValueError(f"sample(noise=...) shape {tuple(noise.shape)} != expected {expected}")
        x = noise.to(device=device, dtype=torch.float32)
    else:
        x = initial_noise(seed, width, height, compression, channels).to(device)
    cfg = cfg_scale > 1.0 and uncond is not None
    emb, t5_ids, t5_mask, qmask = _batched(cond, device, dtype)
    if cfg:
        uemb, ut5_ids, ut5_mask, uqmask = _batched(uncond, device, dtype)
    sigmas = sigma_schedule(steps, shift)
    for i in range(steps):
        if on_step is not None:
            on_step(i, steps)
        t = sigmas[i].to(device)[None]
        inp = x.to(dtype).unsqueeze(2)
        v = model(inp, t, emb, t5_ids, t5_mask, qmask).squeeze(2).float()
        if cfg:
            u = model(inp, t, uemb, ut5_ids, ut5_mask, uqmask).squeeze(2).float()
            v = u + cfg_scale * (v - u)
        x = x - float(sigmas[i] - sigmas[i + 1]) * v
    return x.unsqueeze(2)
