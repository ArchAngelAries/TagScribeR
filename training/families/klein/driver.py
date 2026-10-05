# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/training/trainer.py (KleinTrainer:
# get_noisy_model_input_and_timesteps, call_dit, do_inference), klein/model_utils.py, scripts/cache_latents.py and
# cache_text.py, and lora_trainer_gui.py (the Model Area patterns, 33372-33405).
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: Fizgig has no FamilyDriver for Klein - this class puts its KleinTrainer code behind the family
# interface (loading, quantisation and the LoRA come from the generic layer; the objective, conditioning, timesteps and
# sampler are Fizgig's). The loss runs in fp32 around a bf16 forward (autocast, as accelerate's bf16 mixed precision
# does) and the noise and timesteps come from the loop's seeded CPU generator.
"""Klein driver for the training family layer (training/driver.py).

* conditioning: Qwen3-8B layers 9 / 18 / 27 -> {"text_embed": (512, 12288)} at a FIXED shape (no mask), so items stack
* latents: FLUX.2 AE (fp32), 2x2-packed and batch-normalised: (128, h/16, w/16)
* training: flow matching, x_t = (1 - t) x0 + t noise, target = noise - x0, default timestep mode flux2_shift
  (sampling.sample_timesteps); the DiT sees packed tokens + 4D position ids and t (+ 0.001 except in sigma mode)
* sampling: Euler over the empirical-mu schedule, CFG with a negative prompt
* LoRA: the 112 Linears of the 8 double + 24 single blocks, in blocks selectable through Fizgig's Model Area
"""
import re

import numpy as np
import torch
import torch.nn.functional as F

from training.driver import Block, BlockGroup, FamilyDriver
from training.families.klein import sampling as S

_DOUBLE_MODULES = ("img_attn.qkv", "img_attn.proj", "img_mlp.0", "img_mlp.2",
                   "txt_attn.qkv", "txt_attn.proj", "txt_mlp.0", "txt_mlp.2")
_SINGLE_MODULES = ("linear1", "linear2")
N_DOUBLE, N_SINGLE = 8, 24
# Below this much free VRAM the text encoder loads with INT8 language-model weights instead of bf16 (8B: ~16.4 GB bf16;
# the same threshold the Qwen Image 2.1 encoder of the same size uses)
TE_BF16_MIN_FREE_GB = 19.5

# Model Area presets: Fizgig's include_patterns (lora_trainer_gui.py:33395-33397), matched against module names. Style and
# Style+Composition share one set (33403-33404); the Style area also narrows the timesteps (the preset's 0-400).
STYLE_COMP_PATTERNS = [r".*double_blocks\..*", r".*single_blocks\.[01]\..*"]
IDENTITY_PATTERNS = [r".*single_blocks\.(1[0-6]|[1-9])\..*"]
DETAILS_PATTERNS = [r".*single_blocks\.(1[2-9]|2[0-3])\..*"]
AREA_PATTERNS = {"Identity": IDENTITY_PATTERNS, "Style": STYLE_COMP_PATTERNS,
                 "Style+Composition": STYLE_COMP_PATTERNS, "Details": DETAILS_PATTERNS}
AREAS = ("Full Model", "Identity", "Style", "Style+Composition", "Details", "Custom")


class KleinDriver(FamilyDriver):

    # Every cached caption has the same shape (512 tokens, no mask) and a bucket's latents share a size, so items stack
    supports_batching = True

    # ---- run options (params.DRIVER_OPTIONS -> configure) -----------------------------------------------------
    target_layers = "Full Model"
    training_blocks = ""
    timestep_sampling = "flux2_shift"
    discrete_flow_shift = 3.0
    sigmoid_scale = 1.0
    logit_mean = 0.0
    logit_std = 1.0
    preserve_distribution = False

    def configure(self, **options):
        for key, value in options.items():
            if not hasattr(type(self), key) or callable(getattr(type(self), key)):
                raise ValueError(f"unknown Klein option {key!r}")
            setattr(self, key, value)
        if self.timestep_sampling not in S.TIMESTEP_MODES:
            raise ValueError(f"timestep sampling must be one of {S.TIMESTEP_MODES}, got {self.timestep_sampling!r}")
        self.trainable_blocks()          # a bad Custom list fails here, before anything loads

    # ---- Model Area ---------------------------------------------------------------------------------------------
    @staticmethod
    def resolve_blocks(area, custom=""):
        """The block ids (block_map ids 'double_blocks.N' / 'single_blocks.N') a Model Area trains, or None for the
        full model. Presets match Fizgig's include_patterns against the module names; Custom is the ticked blocks
        (a list, a 'a, b' string, or Fizgig's {block: bool} dict); an EMPTY Custom trains the full model, as Fizgig
        does (it only warns)."""
        area = str(area or "Full Model")
        if area == "Full Model":
            return None
        if area == "Custom":
            if isinstance(custom, dict):
                tokens = [k for k, on in custom.items() if on]
            elif isinstance(custom, str):
                tokens = [t for t in re.split(r"[,\s]+", custom) if t]
            else:
                tokens = list(custom or [])
            if not tokens:
                return None
            valid = {f"double_blocks.{i}" for i in range(N_DOUBLE)} | {f"single_blocks.{i}" for i in range(N_SINGLE)}
            bad = [t for t in tokens if t not in valid]
            if bad:
                raise ValueError(f"Model area Custom: unknown block(s) {', '.join(bad)} - use double_blocks.0-7 and "
                                 f"single_blocks.0-23")
            return set(tokens)
        pats = AREA_PATTERNS.get(area)
        if pats is None:
            raise ValueError(f"unknown Model Area {area!r}; choose one of {', '.join(AREAS)}")
        ids = [f"double_blocks.{i}" for i in range(N_DOUBLE)] + [f"single_blocks.{i}" for i in range(N_SINGLE)]
        return {b for b in ids if any(re.fullmatch(p, f"{b}.x") for p in pats)}

    def trainable_blocks(self):
        return self.resolve_blocks(self.target_layers, self.training_blocks)

    # ---- models ---------------------------------------------------------------------------------
    def load_dit(self, path, device):
        from training.families.klein.model import load_klein_dit
        return load_klein_dit(path, device=device).eval().requires_grad_(False)

    def max_blocks_to_swap(self, dit=None):
        # Fizgig's swap formula (klein/model.py enable_block_swap) accepts 1-16 for 8 + 24 blocks; its GUI range is 0-16
        return dit.max_block_swap() if dit is not None else 16

    def enable_block_swap(self, dit, num_blocks, device, supports_backward=True):
        dit.enable_block_swap(num_blocks, device, supports_backward)
        dit.move_to_device_except_swap_blocks(device)

    def block_swap_mode(self, dit, inference):
        if inference:
            dit.switch_block_swap_for_inference()
        else:
            dit.switch_block_swap_for_training()

    def load_vae(self, path, device):
        from training.families.klein.vae import load_klein_vae
        return load_klein_vae(path, device=device)

    def load_text_encoder(self, path, device):
        """Language model only (no LM head): bf16 when it fits the free VRAM with room to run, else INT8 weights
        (about 9 GB)."""
        from training.families.klein.embedder import KleinTextEncoder
        from training.quant import free_vram_gb
        dev = torch.device(device)
        return KleinTextEncoder(path, device=dev, int8=dev.type == "cuda" and free_vram_gb() < TE_BF16_MIN_FREE_GB)

    def unload_text_encoder(self, te):
        te.unload()

    def enable_gradient_checkpointing(self, dit, on=True):
        dit.enable_gradient_checkpointing(on)

    # ---- encoding -------------------------------------------------------------------------------
    @torch.no_grad()
    def encode_images(self, vae, images):
        """Fizgig cache_latents.py: pixels / 127.5 - 1 -> ae.encode (fp32) -> (128, h/16, w/16)."""
        x = torch.stack([torch.from_numpy(np.ascontiguousarray(a[..., :3])) for a in images])
        x = x.permute(0, 3, 1, 2).float() / 127.5 - 1.0
        p = next(vae.parameters())
        return [z.float().cpu() for z in vae.encode(x.to(p.device, p.dtype))]

    @torch.no_grad()
    def encode_text(self, te, captions):
        """Fizgig cache_text.py: (512, 12288) per caption, bf16."""
        return [{"text_embed": h.to(torch.bfloat16).cpu()} for h in te.encode(list(captions))]

    # ---- training -------------------------------------------------------------------------------
    @staticmethod
    def _autocast(device):
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)

    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None):
        """Fizgig get_noisy_model_input_and_timesteps + call_dit + the MSE of the train loop. latents (B, 128, h, w);
        cond: text_embed (B, 512, 12288) as cached."""
        if refs:
            raise RuntimeError("Klein has no edit / reference training in this port")
        device = latents.device
        bsz, _, h, w = latents.shape
        x0 = latents.float()
        noise = torch.randn(x0.shape, generator=generator).to(device)
        t, t_model = S.sample_timesteps(
            self.timestep_sampling, bsz, h, w, generator, min_t=min_t, max_t=max_t, sigmoid_scale=self.sigmoid_scale,
            shift=self.discrete_flow_shift, logit_mean=self.logit_mean, logit_std=self.logit_std,
            preserve=self.preserve_distribution)
        t4 = t.view(-1, 1, 1, 1).to(device)
        noised = (1.0 - t4) * x0 + t4 * noise
        target = noise - x0                                                    # flow-matching velocity
        tokens, x_ids = S.pack_img(noised.to(torch.bfloat16))
        txt = cond["text_embed"].to(device=device, dtype=torch.bfloat16)
        if txt.dim() == 2:
            txt = txt[None]
        ctx, ctx_ids = S.pack_txt(txt)
        with self._autocast(device):
            pred = dit(x=tokens, x_ids=x_ids, timesteps=t_model.to(device), ctx=ctx, ctx_ids=ctx_ids, guidance=None)
        loss = F.mse_loss(S.unpack_img(pred.float(), h, w), target)
        return loss, {"t": float(t.mean())}

    # ---- sampling -------------------------------------------------------------------------------
    @torch.no_grad()
    def initial_noise(self, seed, width, height):
        return S.initial_noise(seed, S.roundup(width, 16, "width"), S.roundup(height, 16, "height"))

    @staticmethod
    def _text(cond, device):
        t = cond["text_embed"].to(device=device, dtype=torch.bfloat16)
        return t[None] if t.dim() == 2 else t

    @torch.no_grad()
    def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, sigmas=None, options=(),
                 noise=None, on_step=None, refs=None):
        device = next(dit.parameters()).device
        neg = self._text(neg_cond, device) if neg_cond is not None and cfg > 1.0 else None
        return S.sample_latents(dit, self._text(cond, device), neg, device=device,
                                width=S.roundup(width, 16, "width"), height=S.roundup(height, 16, "height"),
                                steps=steps, cfg=cfg, seed=seed, noise=noise, on_step=on_step,
                                channels=getattr(dit, "in_channels", 128))

    @torch.no_grad()
    def decode(self, vae, latents, width, height):
        """Fizgig do_inference: ae.decode (fp32), (pixels / 2 + 0.5) clamped to [0, 1]."""
        from PIL import Image
        p = next(vae.parameters())
        px = vae.decode(latents.to(p.device, p.dtype))[0].float()
        px = (px / 2 + 0.5).clamp(0, 1)
        return Image.fromarray((px * 255.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8))

    # ---- LoRA and the block map -------------------------------------------------------------------
    def block_map(self, dit=None):
        """8 double blocks and 24 single blocks, ids 'double_blocks.N' / 'single_blocks.N' (Fizgig's TRAINING_BLOCKS
        keys). With `dit`, only modules it has."""
        names = {n for n, _ in dit.named_modules()} if dit is not None else None

        def keep(mods):
            return [m for m in mods if names is None or m in names]
        nd = len(dit.double_blocks) if dit is not None else N_DOUBLE
        ns = len(dit.single_blocks) if dit is not None else N_SINGLE
        dbl = [Block(f"double_blocks.{i}", f"Double block {i}", keep([f"double_blocks.{i}.{m}" for m in _DOUBLE_MODULES]))
               for i in range(nd)]
        sgl = [Block(f"single_blocks.{i}", f"Single block {i}", keep([f"single_blocks.{i}.{m}" for m in _SINGLE_MODULES]))
               for i in range(ns)]
        return [BlockGroup("Double blocks", dbl), BlockGroup("Single blocks", sgl)]
