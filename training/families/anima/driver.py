# Behaviour (loop, loss, previews, caching, presets) follows Fizgig (https://github.com/shootthesound/Fizgig, Apache-2.0,
# Copyright 2026 Peter Neill; see THIRD_PARTY_NOTICES.md) through the generic training layer; Fizgig has NO Anima code,
# so the model facts come from kohya sd-scripts and diffusion-pipe's Anima support (see description.py).
# Slider hooks (training_loss diff_ref / diff_weight, noise_latents, predict) ported from Fizgig 7.0.1
# anima/driver.py, on this port's own draw order and timestep sampler.
"""Anima driver for the training family layer (training/driver.py).

* conditioning (cached per caption, fixed shape): {"prompt_embeds": (512, 1024) bf16 Qwen3-0.6B states, "attn_mask":
  (512,) bool, "t5_ids": (512,) int32, "t5_mask": (512,) bool}. The LLM adapter is part of the DiT checkpoint and runs
  inside the DiT forward (frozen; not a LoRA target), so it is not in the cache.
* latents: Qwen-Image VAE (16 ch, /8), normalised with the VAE's latents_mean / std, (16, h, w); 5-D (B, 16, 1, h, w) only
  inside this driver
* training: rectified flow, x_t = (1 - t) x0 + t noise, target = noise - x0, t ~ sigmoid(N(0, 1)) with no shift, the model
  gets t itself (not x1000); optional [min_t, max_t] window; MSE
* sampling: Euler on the shift-3 flow schedule, optional CFG
* LoRA: the attention and MLP Linears of the 28 blocks (280); not the AdaLN modulation, embedders, final layer or adapter
"""
import numpy as np
import torch
import torch.nn.functional as F

from training.driver import Block, BlockGroup, FamilyDriver
from training.families.anima import sampling as S

# sd-scripts lora_anima.py: target class Block, default exclude r".*(_modulation|_norm|_embedder|final_layer).*"
_BLOCK_MODULES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.output_proj",
                  "cross_attn.q_proj", "cross_attn.k_proj", "cross_attn.v_proj", "cross_attn.output_proj",
                  "mlp.layer1", "mlp.layer2")


class AnimaDriver(FamilyDriver):

    # Every cached caption has the same shape (512 padded tokens + masks, both honoured by the adapter), and samples
    # never interact inside the DiT, so a bucket's items stack into a batch.
    supports_batching = True

    # ---- models ---------------------------------------------------------------------------------
    def load_dit(self, path, device):
        from training.families.anima.model import load_anima_dit
        return load_anima_dit(path, device=device).eval().requires_grad_(False)

    # a 2B DiT (4.2 GB bf16) fits consumer cards: no block swap (max_blocks_to_swap stays 0)

    def load_vae(self, path, device):
        from training.families.krea2.vae_loader import load_vae          # the same Qwen-Image VAE Krea 2 uses
        return load_vae(path, input_channels=3, device=device).eval().requires_grad_(False)

    def load_text_encoder(self, path, device):
        from training.families.anima.embedder import AnimaTextEncoder
        return AnimaTextEncoder(path, device=torch.device(device))

    def unload_text_encoder(self, te):
        te.unload()

    def enable_gradient_checkpointing(self, dit, on=True):
        dit.enable_gradient_checkpointing(on)

    # ---- encoding -------------------------------------------------------------------------------
    @torch.no_grad()
    def encode_images(self, vae, images):
        x = torch.stack([torch.from_numpy(np.ascontiguousarray(a[..., :3])) for a in images])
        x = x.permute(0, 3, 1, 2).unsqueeze(2) / 127.5 - 1.0                  # (B, 3, 1, H, W) in [-1, 1]
        z = vae.encode_pixels_to_latents(x.to(vae.device, dtype=vae.dtype))   # (B, 16, 1, h, w)
        return [lat.squeeze(1).to(torch.bfloat16).cpu() for lat in z]

    @torch.no_grad()
    def encode_text(self, te, captions):
        embeds, qmask, t5_ids, t5_mask = te.encode(list(captions))
        return [{"prompt_embeds": e.to(torch.bfloat16), "attn_mask": qm, "t5_ids": ti.to(torch.int32), "t5_mask": tm}
                for e, qm, ti, tm in zip(embeds, qmask, t5_ids, t5_mask)]

    # ---- training -------------------------------------------------------------------------------
    @staticmethod
    def _draw(latents, generator, min_t, max_t):
        """The noise, then t (this port's order; Fizgig 7.0.1 anima/driver.py draws t first). -> (x0, noise, t, x_t)."""
        bsz = latents.shape[0]
        x0 = latents.float()
        noise = torch.randn(x0.shape, generator=generator).to(latents.device)
        t = S.sample_training_timesteps(bsz, min_t, max_t, generator).to(latents.device)
        t4 = t.view(bsz, 1, 1, 1)
        return x0, noise, t, (1.0 - t4) * x0 + t4 * noise

    @staticmethod
    def _velocity(dit, noised, t, cond):
        """The DiT's velocity (B, 16, h, w) for a noised latent at t (B,) in [0, 1] (Fizgig 7.0.1 anima/driver.py
        _velocity; the adapter runs inside the DiT forward here)."""
        emb, t5_ids, t5_mask, qmask = S._batched(cond, noised.device, dit.dtype)
        return dit(noised.to(dit.dtype).unsqueeze(2), t, emb, t5_ids, t5_mask, qmask).squeeze(2)

    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None, diff_ref=None,
                      diff_weight=0.0):
        """latents (B, 16, h, w); cond as cached, batched. Rectified flow: velocity target = noise - x0. diff_ref /
        diff_weight: an image-pair slider's other pole."""
        if refs:
            raise RuntimeError("Anima has no edit / reference training in this port")
        x0, noise, t, noised = self._draw(latents, generator, min_t, max_t)
        target = noise - x0
        pred = self._velocity(dit, noised, t, cond)
        if diff_ref is not None and diff_weight > 0.0:
            # Fizgig 7.0.1 anima/driver.py training_loss, slider pairs: positions where the two poles differ count
            # more (Krea 2's formula, as Qwen and SDXL)
            ref = diff_ref.to(x0.device).float()
            ref = ref.squeeze(2) if ref.dim() == 5 else ref
            d = (x0 - ref).abs().mean(dim=1).flatten(1)                                 # (B, h*w)
            dm = d.mean(dim=1, keepdim=True)
            r = (d / dm.clamp_min(1e-8)).clamp(max=8.0)
            w = (1.0 - float(diff_weight)) + float(diff_weight) * r
            w = w / w.mean(dim=1, keepdim=True).clamp_min(1e-8)
            w = torch.where(dm > 1e-6, w, torch.ones_like(w))      # identical pair: uniform, never all-zero
            se = (pred.float() - target).pow(2).mean(dim=1).flatten(1)
            loss = (se * w).mean()
        else:
            loss = F.mse_loss(pred.float(), target)
        return loss, {"t": float(t.mean())}

    # ---- prompt-pair sliders (Fizgig 7.0.1 anima/driver.py noise_latents / predict) -----------------------------
    def noise_latents(self, latents, generator, *, min_t=0.0, max_t=1.0):
        """A noised practice latent drawn as training_loss draws one (see _draw for the order)."""
        _x0, _noise, t, noised = self._draw(latents, generator, min_t, max_t)
        return {"xt": noised, "tt": t, "t": float(t.mean())}

    def predict(self, dit, state, cond):
        """The DiT's velocity (B, 16, h, w) at a noise_latents() state."""
        return self._velocity(dit, state["xt"], state["tt"], cond)

    # ---- sampling -------------------------------------------------------------------------------
    @torch.no_grad()
    def initial_noise(self, seed, width, height):
        return S.initial_noise(seed, -(-width // 16) * 16, -(-height // 16) * 16)

    @torch.no_grad()
    def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, sigmas=None, options=(),
                 noise=None, on_step=None, refs=None):
        device = next(dit.parameters()).device
        shift = dict(options).get("shift", S.DEFAULT_SHIFT)
        return S.sample_latents(dit, cond, uncond=neg_cond if cfg > 1.0 else None, device=device, dtype=dit.dtype,
                               width=width, height=height, steps=steps, cfg_scale=cfg, seed=seed, shift=shift,
                               noise=noise, on_step=on_step)

    # ---- LoRA ---------------------------------------------------------------------------------------
    def alias_flat(self, flat):
        """diffusers naming (AI-Toolkit's Anima files before its ComfyUI rename): transformer_blocks_N_attn1_to_q ->
        blocks_N_self_attn_q_proj (Fizgig 7.0.1 anima/driver.py)."""
        import re
        m = re.match(r"^transformer_blocks_(\d+)_(attn1|attn2|ff)_(.+)$", flat)
        if not m:
            return None
        blk, part, leaf = m.groups()
        if part == "ff":
            leaf = {"net_0_proj": "layer1", "net_2": "layer2"}.get(leaf)
            return f"blocks_{blk}_mlp_{leaf}" if leaf else None
        leaf = {"to_q": "q_proj", "to_k": "k_proj", "to_v": "v_proj", "to_out_0": "output_proj"}.get(leaf)
        return f"blocks_{blk}_{'self_attn' if part == 'attn1' else 'cross_attn'}_{leaf}" if leaf else None

    @torch.no_grad()
    def decode(self, vae, latents, width, height):
        from PIL import Image
        px = vae.decode_to_pixels(latents.to(vae.device))[0]                   # (3, H, W) in [0, 1]
        return Image.fromarray((px * 255.0).permute(1, 2, 0).cpu().float().numpy().round().astype(np.uint8))

    # ---- LoRA and the block map -------------------------------------------------------------------
    def block_map(self, dit=None):
        """One group of the 28 blocks; each covers its attention and MLP Linears (with `dit`, only modules it has)."""
        names = {n for n, _ in dit.named_modules()} if dit is not None else None
        n_blocks = len(dit.blocks) if dit is not None else self.description.n_blocks
        blocks = []
        for i in range(n_blocks):
            mods = [f"blocks.{i}.{m}" for m in _BLOCK_MODULES]
            blocks.append(Block(f"block_{i}", f"Block {i}", [m for m in mods if names is None or m in names]))
        return [BlockGroup("Blocks", blocks)]
