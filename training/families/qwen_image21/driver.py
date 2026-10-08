# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/qwen_image21/driver.py
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: import paths; otherwise unchanged. Fizgig 7.0.1's compile_targets added.
"""Qwen Image 2.1 driver for the training family layer (training/driver.py).

Everything Qwen-specific the generic cache/train/preview code needs, behind the FamilyDriver interface:
* conditioning: Qwen3-VL-8B last layer before the final RMSNorm, system tokens dropped -> {"hidden_states": (L, 4096)}
* latents: RGBA VAE (RGB gets an opaque alpha), normalised with the checkpoint's latents_mean/std, (64, h, w) at /16
* training: flow matching, x_t = (1 - t) x0 + t noise, target = noise - x0, logit-normal t with the checkpoint's
  resolution-dependent exponential shift (the sampler's mu); block-causal joint forward, keep the image tokens
* sampling: FlowMatch Euler with the reference schedule (sampling.py)
* edit: reference images go into the text encoder's template as vision tokens (cond["ref_mask"] marks them) and
  into the DiT sequence as clean latents ahead of the target; the checkpoint is one model for both (up to 10 refs)
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from training.driver import FamilyDriver
from training.families.qwen_image21 import sampling as S


class QwenImage21Driver(FamilyDriver):

    # ---- models ---------------------------------------------------------------------------------
    def load_dit(self, path, device):
        from training.families.qwen_image21.model import load_qwen21_dit
        return load_qwen21_dit(path, device=device).eval().requires_grad_(False)

    def max_blocks_to_swap(self, dit=None):
        return len(dit.transformer_blocks) - 2 if dit is not None else self.description.n_blocks - 2

    def enable_block_swap(self, dit, num_blocks, device, supports_backward=True):
        dit.enable_block_swap(num_blocks, device, supports_backward)
        dit.move_to_device_except_swap_blocks(device)

    def block_swap_mode(self, dit, inference):
        if inference:
            dit.switch_block_swap_for_inference()
        else:
            dit.switch_block_swap_for_training()

    def load_vae(self, path, device):
        from training.families.qwen_image21.vae import load_qwen21_vae
        return load_qwen21_vae(path, device=device)

    def load_text_encoder(self, path, device):
        """Text-only (no LM head or vision tower): bf16 when it fits the free VRAM with room to run, else INT8 -
        about 8 GB, which fits a 10 GB card."""
        from training.families.qwen_image21.embedder import Qwen21TextEncoder
        from training.quant import free_vram_gb
        return Qwen21TextEncoder(path, device=device, int8=free_vram_gb() < 19.5)

    supports_references = True

    def load_reference_text_encoder(self, path, device):
        """With its vision tower (about 1 GB more than load_text_encoder's)."""
        from training.families.qwen_image21.embedder import Qwen21TextEncoder
        from training.quant import free_vram_gb
        return Qwen21TextEncoder(path, device=device, int8=free_vram_gb() < 21.0, vision=True)

    def unload_text_encoder(self, te):
        te.unload()

    def compile_targets(self, dit):
        return dit.transformer_blocks

    def enable_gradient_checkpointing(self, dit, on=True):
        dit.enable_gradient_checkpointing(on)

    # ---- encoding -------------------------------------------------------------------------------
    @torch.no_grad()
    def encode_images(self, vae, images):
        x = torch.stack([torch.from_numpy(np.ascontiguousarray(a[..., :3])) for a in images])
        x = x.permute(0, 3, 1, 2).float().div(127.5).sub(1.0)
        p = next(vae.parameters())
        return [z.to(torch.bfloat16).cpu() for z in S.encode_image(vae, x.to(p.device, p.dtype))]

    @torch.no_grad()
    def encode_text(self, te, captions):
        return [{"hidden_states": h.to(torch.bfloat16).cpu()} for h in te.encode(captions)]

    @torch.no_grad()
    def encode_text_with_references(self, te, captions, references):
        from PIL import Image
        out = []
        for cap, imgs in zip(captions, references):
            h, m = te.encode_with_references(cap, [Image.fromarray(np.ascontiguousarray(a[..., :3])) for a in imgs])
            out.append({"hidden_states": h.to(torch.bfloat16).cpu(), "ref_mask": m.cpu()})
        return out

    @staticmethod
    def _ref_sequence(refs, ref_mask, device):
        """Packed reference tokens + their img_shapes entries. Each vision token stands for 2x2 latent tokens, so the
        references' latents must match the size the text encoder saw them at."""
        refs = [r.float().to(device) for r in refs]
        n_ref = sum(r.shape[-2] * r.shape[-1] for r in refs)
        slots = int(ref_mask.sum()) if ref_mask is not None else 0
        if slots * 4 != n_ref:
            raise RuntimeError(f"reference size mismatch: the conditioning holds {slots} image tokens for "
                               f"{n_ref // 4} expected - re-cache the text (Target Megapixels or the before-images "
                               f"changed since it was cached)")
        return torch.cat([S.pack(r) for r in refs], dim=1), [(1, r.shape[-2], r.shape[-1]) for r in refs]

    # ---- training -------------------------------------------------------------------------------
    @staticmethod
    def _sample_t(n_tokens, generator, min_t, max_t):
        t = torch.sigmoid(torch.randn(1, generator=generator)).item()
        mu = S.calculate_mu(n_tokens)
        t = math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0))
        return min_t + (max_t - min_t) * t

    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None):
        device = latents.device
        h, w = latents.shape[-2:]
        n = h * w
        x0 = S.pack(latents.float())
        noise = torch.randn(x0.shape, generator=generator).to(device)
        t = self._sample_t(n, generator, min_t, max_t)
        xt = (1 - t) * x0 + t * noise
        ref_mask = cond["ref_mask"][0] if "ref_mask" in cond else None
        enc, img_mask, enc_mask = S.model_inputs(cond["hidden_states"][0], n, device, ref_mask=ref_mask)
        shapes = [(1, h, w)]
        if refs:
            ref_tokens, ref_shapes = self._ref_sequence(refs, ref_mask, device)
            xt, shapes = torch.cat([ref_tokens, xt], dim=1), ref_shapes + shapes
        elif ref_mask is not None and ref_mask.any():
            raise RuntimeError("edit conditioning without its reference latents - re-cache the latents")
        tt = torch.tensor([t], device=device, dtype=torch.bfloat16)
        pred = dit(xt.to(torch.bfloat16), enc.to(torch.bfloat16), tt, [shapes], img_mask, enc_mask)[:, -n:]
        return F.mse_loss(pred.float(), noise - x0), {"t": t}

    # ---- sampling -------------------------------------------------------------------------------
    @torch.no_grad()
    def initial_noise(self, seed, width, height):
        return S.initial_noise(seed, height, width)

    @torch.no_grad()
    def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, sigmas=None, options=(),
                 noise=None, on_step=None, refs=None):
        device = next(dit.parameters()).device
        neg = neg_cond["hidden_states"] if (neg_cond is not None and cfg > 1.0) else None
        neg_mask = neg_cond.get("mask") if neg is not None else None
        opts = dict(options)
        shift_terminal = opts.get("shift_terminal", S.SHIFT_TERMINAL)
        if sigmas is not None and len(sigmas) != steps:
            sigmas = None                   # an explicit schedule only applies at its own step count
        return S.sample(dit, cond["hidden_states"], height, width, steps=steps, seed=seed, cfg=cfg, neg_emb=neg,
                        device=device, sigmas=sigmas, shift_terminal=shift_terminal, noise=noise, on_step=on_step,
                        text_mask=cond.get("mask"), neg_mask=neg_mask, ref_latents=refs or None,
                        ref_mask=cond.get("ref_mask"), neg_ref_mask=neg_cond.get("ref_mask") if neg is not None else None)

    def pad_conditioning(self, conds):
        """Pad prompts to one length (zeros) with a mask so they can be blended (prompt travel). The DiT masks the
        padded keys and starts the image's positions after the real tokens, so a padded prompt renders as the
        unpadded one does."""
        L = max(c["hidden_states"].shape[0] for c in conds)
        out = []
        for c in conds:
            h = c["hidden_states"]
            pad = L - h.shape[0]
            out.append({"hidden_states": torch.cat([h, h.new_zeros(pad, h.shape[1])]) if pad else h,
                        "mask": torch.cat([torch.ones(h.shape[0], dtype=torch.bool), torch.zeros(pad, dtype=torch.bool)])})
        return out

    @torch.no_grad()
    def decode(self, vae, latents, width, height):
        from PIL import Image
        img = S.decode(vae, latents, height, width)[0, :3].float().clamp(-1, 1)
        return Image.fromarray(((img.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8))
