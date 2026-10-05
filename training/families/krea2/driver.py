# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/krea2/trainer.py (compute_loss,
# load_dit_for_training, encode_sample_prompts, sample_previews_on_dit), krea2/caching.py and krea2/sampling.py.
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: Fizgig has no FamilyDriver for Krea 2 - this class puts its Krea 2 code behind the family
# interface (loading, quantisation and the LoRA come from the generic layer; the objective, conditioning and sampler
# are Fizgig's). The loss runs in fp32 around a bf16 forward instead of Fizgig's bf16 mixing arithmetic.
"""Krea 2 driver for the training family layer (training/driver.py).

* conditioning: Qwen3-VL-4B, 12 hidden-state layers -> {"hidden_states": (512, 12, 2560), "attention_mask": (512,)}
  stored at a FIXED shape (as Fizgig's cache does, ~30 MB per caption), so items stack into batches; the loss drops
  the interior padding per sample (sampling.gather_valid_text) and masks what is left
* latents: Qwen-Image VAE (16 ch, /8), normalised with the VAE's latents_mean / std, (16, h, w)
* training: flow matching, x_t = (1 - t) x0 + t noise, target = noise - x0, logit-normal t with the resolution-
  dependent shift and an optional [min_t, max_t] window (sampling.sample_krea2_timesteps)
* sampling: Euler with the shifted schedule, optional CFG; the Turbo LoRA's schedule pins mu = 1.15
* LoRA: every Linear of the DiT (264); only the 28 main blocks' Linears are quantised (INT8 / NF4)
"""
import logging

import numpy as np
import torch
import torch.nn.functional as F

from training.driver import Block, BlockGroup, FamilyDriver
from training.families.krea2 import sampling as S

logger = logging.getLogger(__name__)

_BLOCK_MODULES = ("attn.wq", "attn.wk", "attn.wv", "attn.gate", "attn.wo", "mlp.gate", "mlp.up", "mlp.down")
# Everything outside the main blocks (Fizgig: LoRA wraps every Linear of the DiT, krea2/trainer.py
# load_dit_for_training -> create_network(None, "lora_unet", ...))
_IO_IN = ("first", "tmlp.0", "tmlp.2", "tproj.1", "txtmlp.1", "txtmlp.3")
_IO_OUT = ("last.linear",)
_TXT_BLOCKS = {"layerwise": "txtfusion.layerwise_blocks", "refiner": "txtfusion.refiner_blocks"}


class Krea2Driver(FamilyDriver):

    compile_blocks = "auto"        # Compile Blocks: auto | on | off | outside (the Train tab's COMPILE_BLOCKS)
    _compile_requested = False

    def configure(self, **options):
        if "compile_blocks" in options:
            self.compile_blocks = str(options["compile_blocks"] or "auto")

    def optimizer_params(self, net, optimizer_type, lr, optimizer_args):
        """Automagic v3 keeps one rate per parameter GROUP, so the LoRA is split by module family (txtfusion / attn / mlp /
        io), each finding its own rate instead of one compromise for all 264 modules, and its sign window is 16: the
        8-step default reads Krea 2's per-step gradient noise as overshoot (batch one, logit-normal timesteps across
        the full range, bucketed resolutions - consecutive steps are genuinely different problems). Fizgig
        krea2/trainer.py:2559-2587 (Peter, 17 Sep 2026). An explicit polarity_history in Optimizer Args wins."""
        if str(optimizer_type or "").lower() != "automagic3":
            return super().optimizer_params(net, optimizer_type, lr, optimizer_args)
        from training.lora import TRAINABLE
        from training.optimizers import family_groups
        items = [("lora_unet_" + self.lora_key_name(full).replace(".", "_"),
                  [p for p in w.adapters[TRAINABLE].parameters() if p.requires_grad])
                 for full, w in net.wrapped.items() if TRAINABLE in w.adapters]
        groups, counts = family_groups(items, lr)
        if "polarity_history" not in (optimizer_args or ""):
            optimizer_args = ((optimizer_args or "") + " polarity_history=16").strip()
            logger.info("[optimizer] Automagic v3: sign window 16 (this family's default - the 8-step window reads "
                        "Krea 2's per-step gradient noise as overshoot). Set polarity_history in Optimizer Args to "
                        "override.")
        return (groups or net.parameters()), optimizer_args, counts

    def prepare_training(self, dit, net, *, precision, blocks_to_swap, total_steps, megapixels, batch_size):
        """torch.compile of the blocks, AFTER the LoRA wrapped their forwards (Fizgig load_dit_for_training, last step)."""
        from training.families.krea2 import compile as C
        do = C.resolve(self.compile_blocks, precision=precision, blocks_to_swap=blocks_to_swap,
                       total_steps=total_steps, mp=megapixels or 0.25, batch=batch_size)
        self._compile_requested = bool(do)
        if do:
            C.compile_blocks(dit, blocks_to_swap, fp8_scaled=precision == "fp8",
                             boundary="outside" if do == "outside" else "inside")

    def after_epoch(self, epoch, steps_remaining):
        """Attention backend: cuDNN's kernel is ~6% faster per step but costs ~1.3 s per distinct sequence shape to plan,
        so it only wins on runs long enough to amortise that. After a full epoch every shape the dataset produces has
        been seen. SUPPRESSED when compile was asked for: flipping the backend mid-run changes the branch every compiled
        block traced (Fizgig krea2/trainer.py:3321-3335)."""
        if self._compile_requested:
            return
        from training.modules.sdpa import consider_training_backend
        switch = consider_training_backend(steps_remaining)
        if switch:
            n_shapes, needed = switch
            logger.info(f"[attention] switching to the cuDNN backend for the rest of the run - {n_shapes} distinct "
                        f"sequence shape(s), which pays back within {needed} steps and this run has more left. "
                        f"Expect a slower first pass over each shape while it plans, then ~6% faster steps.")

    # Every cached caption has the same shape (512 padded tokens + mask), so a bucket's items stack into a batch
    # (Fizgig trains Krea 2 at dataset batch size > 1 the same way; its VRAM is +2.4 GB per extra image).
    supports_batching = True

    # ---- models ---------------------------------------------------------------------------------
    def load_dit(self, path, device):
        from training.families.krea2.model import load_krea2_dit
        return load_krea2_dit(path, device=device).eval().requires_grad_(False)

    def max_blocks_to_swap(self, dit=None):
        return len(dit.blocks) - 2 if dit is not None else self.description.n_blocks - 2

    def enable_block_swap(self, dit, num_blocks, device, supports_backward=True):
        dit.enable_block_swap(num_blocks, device, supports_backward)
        dit.move_to_device_except_swap_blocks(device)

    def block_swap_mode(self, dit, inference):
        if inference:
            dit.switch_block_swap_for_inference()
        else:
            dit.switch_block_swap_for_training()

    def load_vae(self, path, device):
        from training.families.krea2.vae_loader import load_vae
        return load_vae(path, input_channels=3, device=device).eval().requires_grad_(False)

    def load_text_encoder(self, path, device):
        """Qwen3-VL-4B as Fizgig loads it: bf16, or the fp8_scaled file with its language Linears kept fp8 (no
        expansion to bf16 at load)."""
        from training.families.krea2.embedder import Krea2TextEncoder
        return Krea2TextEncoder(path, device=torch.device(device))

    def unload_text_encoder(self, te):
        te.unload()

    def enable_gradient_checkpointing(self, dit, on=True):
        dit.enable_gradient_checkpointing(on)

    # ---- encoding -------------------------------------------------------------------------------
    @torch.no_grad()
    def encode_images(self, vae, images):
        """Fizgig krea2/caching.py encode_and_save_latents: pixels in [-1, 1] -> the VAE's mode, normalised."""
        x = torch.stack([torch.from_numpy(np.ascontiguousarray(a[..., :3])) for a in images])
        x = x.permute(0, 3, 1, 2).unsqueeze(2) / 127.5 - 1.0                  # (B, 3, 1, H, W)
        z = vae.encode_pixels_to_latents(x.to(vae.device, dtype=vae.dtype))   # (B, 16, 1, h, w)
        return [lat.squeeze(1).to(torch.bfloat16).cpu() for lat in z]

    @torch.no_grad()
    def encode_text(self, te, captions):
        hiddens, mask = te.encode(list(captions))
        return [{"hidden_states": h.to(torch.bfloat16).cpu(), "attention_mask": m.cpu()}
                for h, m in zip(hiddens, mask)]

    # ---- training -------------------------------------------------------------------------------
    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None):
        """Fizgig krea2/trainer.py compute_loss. latents (B, 16, h, w); cond: hidden_states (B, 512, 12, 2560) and
        attention_mask (B, 512) as cached."""
        if refs:
            raise RuntimeError("Krea 2 has no edit / reference training in this port")
        device = latents.device
        bsz = latents.shape[0]
        patch = dit.config.patch
        x0 = latents.float()
        noise = torch.randn(x0.shape, generator=generator).to(device)
        n_tokens = (x0.shape[-2] // patch) * (x0.shape[-1] // patch)
        t = S.sample_krea2_timesteps(bsz, n_tokens, "cpu", min_timestep=min_t, max_timestep=max_t, generator=generator)
        # Fizgig hands the DiT (and mixes the noise with) t in bf16: both use the same rounded value here
        t_bf = t.to(torch.bfloat16).to(device)
        t4 = t_bf.float().view(bsz, 1, 1, 1)
        noised = (1.0 - t4) * x0 + t4 * noise
        target = noise - x0                                                    # flow-matching velocity
        txt, txtmask = S.gather_valid_text(cond["hidden_states"].to(device=device, dtype=torch.bfloat16),
                                           cond["attention_mask"].to(device))
        img_tokens, pos, mask = S.prepare(noised.to(torch.bfloat16), txt.shape[1], patch, txtmask)
        pred = dit(img=img_tokens, context=txt, t=t_bf, pos=pos, mask=mask)
        loss = F.mse_loss(pred.float(), S.patchify(target, patch))
        return loss, {"t": float(t_bf.float().mean())}

    # ---- sampling -------------------------------------------------------------------------------
    @torch.no_grad()
    def initial_noise(self, seed, width, height):
        return S.initial_noise(seed, S.roundup(width, 16, "width"), S.roundup(height, 16, "height"))

    @staticmethod
    def _text(cond, device):
        """One caption's conditioning (unbatched, as the preview loop holds it) -> gathered, batched tensors."""
        h, m = cond["hidden_states"], cond["attention_mask"]
        if h.dim() == 3:
            h, m = h[None], m[None]
        return S.gather_valid_text(h.to(device=device, dtype=torch.bfloat16), m.to(device))

    @torch.no_grad()
    def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, sigmas=None, options=(),
                 noise=None, on_step=None, refs=None):
        device = next(dit.parameters()).device
        txt, txtmask = self._text(cond, device)
        untxt = untxtmask = None
        if neg_cond is not None and cfg > 1.0:
            untxt, untxtmask = self._text(neg_cond, device)
        mu = dict(options).get("mu")            # the Turbo LoRA's pinned schedule shift; None = from the resolution
        return S.sample_latents(dit, txt, txtmask, untxt=untxt, untxtmask=untxtmask, device=device, width=width,
                               height=height, steps=steps, cfg_scale=cfg, seed=seed, mu=mu, noise=noise,
                               on_step=on_step)

    @torch.no_grad()
    def decode(self, vae, latents, width, height):
        from PIL import Image
        px = vae.decode_to_pixels(latents.to(vae.device))[0]                   # (3, H, W) in [0, 1]
        return Image.fromarray((px * 255.0).permute(1, 2, 0).cpu().float().numpy().round().astype(np.uint8))

    # ---- LoRA and the block map -------------------------------------------------------------------
    def block_map(self, dit=None):
        """Main blocks, the text-fusion stack and the input / output layers - every Linear of the model (the LoRA is
        all-Linear). With `dit`, only modules it has."""
        names = {n for n, _ in dit.named_modules()} if dit is not None else None

        def keep(mods):
            return [m for m in mods if names is None or m in names]
        n_main = len(dit.blocks) if dit is not None else self.description.n_blocks
        main = [Block(f"block_{i}", f"Block {i}", keep([f"blocks.{i}.{m}" for m in _BLOCK_MODULES]))
                for i in range(n_main)]
        fusion = []
        for kind, prefix in _TXT_BLOCKS.items():
            for i in range(2):
                fusion.append(Block(f"txtfusion_{kind}_{i}", f"{kind.capitalize()} block {i + 1}",
                                    keep([f"{prefix}.{i}.{m}" for m in _BLOCK_MODULES])))
            if kind == "layerwise":
                fusion.append(Block("txtfusion_projector", "Layer projector", keep(["txtfusion.projector"])))
        return [BlockGroup("Main blocks", main), BlockGroup("Text fusion", fusion),
                BlockGroup("Input / output", [Block("io_in", "Input embedders", keep(list(_IO_IN))),
                                              Block("io_out", "Output layer", keep(list(_IO_OUT)))])]

    def quant_target_names(self, dit):
        """Fizgig quantises only `blocks.` minus mod. / norm / txtfusion (krea2/utils.py KREA2_FP8_OPTIMIZATION_*):
        the main blocks' Linears. The text-fusion stack, embedders and output layer stay bf16."""
        return [m for b in self.block_map(dit)[0].blocks for m in b.modules]
