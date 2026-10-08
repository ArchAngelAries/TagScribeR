# SDXL-architecture driver for TagScribeR's family layer (original code - Fizgig has no SDXL; the loop, cached text
# conditioning, timestep-window semantics and the driver contract are Fizgig's, the model facts come from diffusers /
# kohya sd-scripts / ComfyUI as cited in unet.py, text.py, vae.py and sampling.py).
# Slider hooks (training_loss diff_ref / diff_weight, noise_latents, predict) ported from Fizgig 7.0.1
# sdxl/driver.py, on this port's own draw order and schedule.
"""SDXL, Pony, Illustrious and NoobAI-XL (epsilon and v-prediction) behind the family interface.

* model file: ONE single-file .safetensors checkpoint holding the UNet, both CLIPs and the VAE (the VAE row may point at
  a separate file such as the fp16-fix VAE; the text-encoder row at another checkpoint)
* conditioning: {"crossattn": (231, 2048), "pooled": (1280,)} - FIXED shape (3 chunks of 75 tokens), so items batch
* latents: SDXL VAE mode x 0.13025 -> (4, h, w) float32
* training: DDPM, 1000 steps, scaled-linear betas, epsilon or v target (+ zero-terminal-SNR for NoobAI v-pred), timestep
  drawn uniformly inside [min_t, max_t]; SDXL size conditioning from the bucket (original = target, crop 0,0)
* LoRA: transformer Linears of the UNet (and optionally the 3x3 convs, "LoCon"), kohya keys with the ORIGINAL (LDM) module
  names A1111 and ComfyUI expect: lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.lora_down.weight
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from training.driver import Block, BlockGroup, FamilyDriver
from training.families.sdxl import sampling as S
from training.families.sdxl import text as T
from training.families.sdxl import unet as U
from training.families.sdxl import vae as V

_GROUPS = (("Input blocks", "in_"), ("Middle", "mid"), ("Output blocks", "out_"))
BF16_MIN_FREE_GB = 4.0


class SDXLDriver(FamilyDriver):
    prediction = "epsilon"            # "epsilon" | "v_prediction" (NoobAI v-pred overrides)
    zero_terminal_snr = False         # NoobAI v-pred overrides
    supports_batching = True          # every cached caption is (231, 2048) + (1280,): items of a bucket stack

    unet_config = U.SDXL_UNET_CONFIG  # tests substitute a tiny config
    clip_l_config = None
    clip_g_config = None
    vae_config = None

    # family options (training/params.py DRIVER_OPTIONS), all off by default
    min_snr_gamma = 0.0
    noise_offset = 0.0
    locon = False

    def configure(self, min_snr_gamma=0.0, noise_offset=0.0, locon=False, **ignored):
        self.min_snr_gamma = max(0.0, float(min_snr_gamma or 0.0))
        self.noise_offset = max(0.0, float(noise_offset or 0.0))
        self.locon = bool(locon)
        self._block_index = None

    # ---- the UNet's structure (names) -----------------------------------------------------------------
    def _adopt(self, config):
        """Remember the UNet config the module names come from (a model passed in wins over the default)."""
        config = dict(config)
        if getattr(self, "_cfg", None) != config:
            self._cfg = config
            self._ldm = U.ldm_module_table(config, locon=True)
            self._block_index = None

    def _config(self, dit=None):
        if dit is not None:
            self._adopt(dit.config)
        elif getattr(self, "_cfg", None) is None:
            self._adopt(self.unet_config)
        return self._cfg

    def lora_key_name(self, module_path):
        """diffusers module path -> the original (LDM / SGM) path A1111 and ComfyUI name LoRA keys after."""
        self._config()
        return self._ldm.get(module_path, module_path)

    # ---- models ---------------------------------------------------------------------------------
    def load_dit(self, path, device):
        return U.load_unet(path, device, config=self.unet_config)

    def load_vae(self, path, device):
        return V.load_vae(path, device, config=self.vae_config)

    def load_text_encoder(self, path, device):
        dev = torch.device(device)
        dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
        clip_l, clip_g = T.load_encoders(path, self.clip_l_config, self.clip_g_config, dtype)
        return T.SDXLTextEncoders(clip_l, clip_g, T.load_tokenizer(path), device=dev, dtype=dtype)

    def unload_text_encoder(self, te):
        te.unload()

    def enable_gradient_checkpointing(self, dit, on=True):
        if on:
            dit.enable_gradient_checkpointing()
        else:
            dit.disable_gradient_checkpointing()

    # ---- encoding -------------------------------------------------------------------------------
    @torch.no_grad()
    def encode_images(self, vae, images):
        x = torch.stack([torch.from_numpy(np.ascontiguousarray(a[..., :3])) for a in images])
        x = x.permute(0, 3, 1, 2).float() / 127.5 - 1.0
        z = vae.encode(x.to(vae.device, dtype=vae.dtype)).latent_dist.mode() * V.SCALING_FACTOR
        return [lat.float().cpu() for lat in z]

    @torch.no_grad()
    def encode_text(self, te, captions):
        cross, pooled = te.encode(list(captions))
        return [{"crossattn": c.to(torch.bfloat16).cpu(), "pooled": p.float().cpu()} for c, p in zip(cross, pooled)]

    # ---- training -------------------------------------------------------------------------------
    def _draw(self, latents, generator, min_t, max_t):
        """A noised training input: the noise (plus the offset noise when set), then a uniform DDPM step inside
        [min_t, max_t]. Fizgig 7.0.1 sdxl/driver.py noise_latents draws the step FIRST; this port keeps its own
        order, so training_loss and the sliders share one draw."""
        device = latents.device
        x0 = latents.float()
        b = x0.shape[0]
        noise = torch.randn(x0.shape, generator=generator)
        if self.noise_offset:                                                  # [noise-offset]
            noise = noise + self.noise_offset * torch.randn(b, x0.shape[1], 1, 1, generator=generator)
        noise = noise.to(device)
        t = S.draw_timesteps(b, min_t, max_t, generator)
        ac = S.alphas_cumprod(self.zero_terminal_snr).float()[t].to(device)
        return {"xt": S.add_noise(x0, noise, ac), "steps": t, "ac": ac, "x0": x0, "noise": noise,
                "t": float(t.float().div(S.TRAIN_STEPS).mean())}

    def _unet(self, dit, noisy, steps, cond):
        """The UNet's output (epsilon, or v for a v-pred checkpoint), with SDXL's size conditioning from the latent."""
        device = noisy.device
        dtype = next(dit.parameters()).dtype
        c, p = self._cond(cond, device, dtype)
        ids = S.time_ids(noisy.shape[-2] * 8, noisy.shape[-1] * 8, noisy.shape[0], device, dtype)
        return dit(noisy.to(dtype), steps.to(device), encoder_hidden_states=c,
                   added_cond_kwargs={"text_embeds": p, "time_ids": ids}).sample

    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None, diff_ref=None,
                      diff_weight=0.0):
        if refs:
            raise RuntimeError("SDXL has no edit / reference training in this port")
        st = self._draw(latents, generator, min_t, max_t)
        x0, ac = st["x0"], st["ac"]
        target = S.target_for(self.prediction, x0, st["noise"], ac)
        pred = self._unet(dit, st["xt"], st["steps"], cond)
        if diff_ref is not None and diff_weight > 0.0:
            # Fizgig 7.0.1 sdxl/driver.py training_loss, image-pair sliders: latent cells where the two poles differ
            # count more (Krea 2 / Qwen's formula); per sample here, so Min-SNR still weighs each sample
            d = (x0 - diff_ref.to(x0.device).float()).abs().mean(dim=1).flatten(1)          # (B, h*w)
            dm = d.mean(dim=1, keepdim=True)
            r = (d / dm.clamp_min(1e-8)).clamp(max=8.0)
            wgt = (1.0 - float(diff_weight)) + float(diff_weight) * r
            wgt = wgt / wgt.mean(dim=1, keepdim=True).clamp_min(1e-8)
            wgt = torch.where(dm > 1e-6, wgt, torch.ones_like(wgt))   # identical pair: uniform, never all-zero
            per = ((pred.float() - target).pow(2).mean(dim=1).flatten(1) * wgt).mean(dim=1)
        else:
            per = F.mse_loss(pred.float(), target, reduction="none").mean(dim=(1, 2, 3))
        if self.min_snr_gamma:
            per = per * S.min_snr_weights(ac, self.min_snr_gamma, self.prediction)
        return per.mean(), {"t": st["t"]}

    # ---- prompt-pair sliders (Fizgig 7.0.1 sdxl/driver.py noise_latents / predict) ------------------------------
    def noise_latents(self, latents, generator, *, min_t=0.0, max_t=1.0):
        """A noised practice latent, drawn exactly as training_loss draws one (see _draw for the order)."""
        return self._draw(latents, generator, min_t, max_t)

    def predict(self, dit, state, cond):
        """The UNet's output (epsilon, or v for a v-pred checkpoint) at a noise_latents() state."""
        return self._unet(dit, state["xt"], state["steps"], cond)

    # ---- sampling -------------------------------------------------------------------------------
    def initial_noise(self, seed, width, height):
        return S.initial_noise(seed, -(-width // 8) * 8, -(-height // 8) * 8)

    @staticmethod
    def _cond(cond, device, dtype):
        c, p = cond["crossattn"], cond["pooled"]
        if c.dim() == 2:
            c, p = c[None], p[None]
        return c.to(device, dtype), p.to(device, dtype)

    @torch.no_grad()
    def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, sigmas=None, options=(),
                 noise=None, on_step=None, refs=None):
        device = next(dit.parameters()).device
        dtype = next(dit.parameters()).dtype
        opts = dict(options)
        w, h = -(-width // 8) * 8, -(-height // 8) * 8
        c, p = self._cond(cond, device, dtype)
        use_cfg = neg_cond is not None and cfg > 1.0
        if use_cfg:
            nc, np_ = self._cond(neg_cond, device, dtype)
            c, p = torch.cat([nc, c]), torch.cat([np_, p])
        ids = S.time_ids(h, w, c.shape[0], device, dtype)
        x = (noise if noise is not None else self.initial_noise(seed, w, h)).float()

        def predict(x_in, t):
            xin = x_in.to(device, dtype)
            if use_cfg:
                xin = torch.cat([xin, xin])
            out = dit(xin, torch.full((xin.shape[0],), t, device=device, dtype=torch.long), encoder_hidden_states=c,
                      added_cond_kwargs={"text_embeds": p, "time_ids": ids}).sample.float()
            if use_cfg:
                uncond, cond_out = out.chunk(2)
                out = uncond + cfg * (cond_out - uncond)
            return out
        sampler = str(opts.get("sampler", self.description.default_sampling().sampler if self.description else "euler"))
        if sampler.startswith("dpmpp_2m_sde"):
            return self._dpmpp(predict, x.to(device), steps, seed, on_step).cpu()
        return S.euler_sample(predict, x.to(device), steps, prediction=self.prediction,
                              zero_terminal_snr=self.zero_terminal_snr, ancestral=sampler in ("euler_a", "euler_ancestral"),
                              seed=seed, on_step=on_step).cpu()

    def _dpmpp(self, predict, x, steps, seed, on_step):
        """DPM++ 2M SDE with Karras sigmas through diffusers' DPMSolverMultistepScheduler (Fizgig 7.0.1 sdxl/driver.py
        _scheduler / generate): the SDE noise from seed + 1; v-prediction and zero-terminal-SNR passed through."""
        from diffusers import DPMSolverMultistepScheduler
        sch = DPMSolverMultistepScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000,
            prediction_type=self.prediction, algorithm_type="sde-dpmsolver++", solver_order=2, use_karras_sigmas=True,
            **({"rescale_betas_zero_snr": True, "timestep_spacing": "trailing"} if self.zero_terminal_snr else {}))
        sch.set_timesteps(int(steps), device=x.device)
        g = torch.Generator("cpu").manual_seed(int(seed) + 1)
        x = x * sch.init_noise_sigma
        for i, t in enumerate(sch.timesteps):
            if on_step is not None:
                on_step(i, len(sch.timesteps))
            out = predict(sch.scale_model_input(x, t), int(t)).to(x.device)
            x = sch.step(out, t, x, generator=g, return_dict=False)[0]
        return x

    @torch.no_grad()
    def decode(self, vae, latents, width, height):
        from PIL import Image
        want = torch.bfloat16 if (vae.device.type == "cuda") else torch.float32       # never fp16: SDXL VAE NaNs there
        if vae.dtype != want:
            vae.to(dtype=want)
        z = latents.to(vae.device, dtype=want) / V.SCALING_FACTOR
        px = (vae.decode(z).sample[0].float() / 2 + 0.5).clamp(0, 1)
        return Image.fromarray((px * 255.0).permute(1, 2, 0).cpu().numpy().round().astype(np.uint8))

    # ---- LoRA and the block map -------------------------------------------------------------------
    def quant_target_names(self, dit):
        """INT8 / NF4 quantise the transformer Linears the LoRA targets except proj_in / proj_out (Fizgig 7.0.1
        sdxl/driver.py); the convolutions a LoCon run adds stay bf16."""
        return [n for n in self.lora_target_names(dit) if not n.endswith(("proj_in", "proj_out"))
                and isinstance(dit.get_submodule(n), torch.nn.Linear)]

    def block_map(self, dit=None):
        """IN01..IN08, MID, OUT00..OUT08 in kohya's block terms; each block lists the target modules it holds under
        the current options (transformer Linears; with LoCon also 3x3 convs). Blocks with no target are left out."""
        config = self._config(dit)
        names = {n for n, _ in dit.named_modules()} if dit is not None else None
        mods = {}
        for u in U.structure(config):
            for d, _ldm in U.unit_modules(u, self.locon):
                if names is None or d in names:
                    mods.setdefault(u.block, []).append(d)
        groups = []
        for label, pref in _GROUPS:
            blocks = [Block(b, U.block_label(b), m) for b, m in mods.items() if b.startswith(pref)]
            if blocks:
                groups.append(BlockGroup(label, blocks))
        return groups


class SDXLVPredDriver(SDXLDriver):
    """NoobAI-XL v-prediction: v target and the zero-terminal-SNR schedule (NoobAI model card; Lin et al. 2023)."""
    prediction = "v_prediction"
    zero_terminal_snr = True
