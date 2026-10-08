# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/minimax/driver.py (Fizgig 7.0.1's MiniMaxDriver:
# the options, _shift, the high-noise LR, the adapter ramp, run_metadata, block_map, lora_target_names), minimax/common.py
# (compute_loss, sample_sigmas, parse_block_spec, AdapterRamp), minimax/caching.py and minimax/sampling.py.
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: still images only. Fizgig builds the LoRA on all 50 blocks and freezes the out-of-window ones
# per step; here only the window's modules exist (the others would hold zero-initialised, never-updated adapters, so
# training is identical and the file smaller). The options arrive as configure() keywords (training/params.py H3_*).
"""MiniMax H3 driver for the training family layer (training/driver.py).

* conditioning: Qwen3-VL-32B layer-50 hidden states, (L, 5120) bf16, variable length (batch size 1)
* latents: H3 video VAE encoding of a still (one frame), 24 channels, 16x, (24, h, w) fp32
* training: flow matching, noised = (1 - sigma) x0 + sigma noise, t = 1 - sigma, target = x0 - noise; sigma from the
  shifted-uniform schedule whose shift is set by the "low-noise %" dial: shift = (1 - P) / P
* LoRA: attention + MLP Linears of the 50 blocks (4 per block); the training mode picks the blocks (Default 20-49)
* sampling: ComfyUI's simple schedule at shift 12, res_multistep, joint audio denoising of the packed silence rows
"""
import logging
import re

import numpy as np
import torch
import torch.nn.functional as F

from training.driver import Block, BlockGroup, FamilyDriver
from training.families.minimax_h3 import sampling as S

logger = logging.getLogger(__name__)

_BLOCK_MODULES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
LIKENESS_BLOCKS = "20-49"        # Fizgig MINIMAX_LIKENESS_BLOCKS: "Default" (fast) mode
FULL_MODEL_BLOCKS = "6-49"       # Fizgig MINIMAX_FULL_MODEL_BLOCKS: "More Blocks" mode (blocks 0-5 deform anatomy)
DEFAULT_LOWNOISE_PCT = 60.0      # the preset value (lora_trainer_gui.py settings default "60")
STRUCTURE_PCT = {"likeness": 60.0, "model": 8.0}     # Fizgig families/minimax.py _STRUCTURE (Custom: the box)
N_REFINER = 2                    # MiniMaxH3Config.token_refiner_num_layers


def _block_index(token):
    """A block id as Fizgig 7.0.1 names it ('h3blk_N', the old Repair Studio's; 'block_N', this app's earlier name)
    -> N, or None."""
    m = re.fullmatch(r"(?:h3blk|block)_(\d+)", str(token).strip())
    return int(m.group(1)) if m else None


def parse_block_spec(spec, num_blocks: int = None):
    """"3-12, 14-15, 22,27,31-33" -> sorted unique indices. Raises ValueError on anything unreadable - a typo must stop
    the run, not silently train a different set of blocks (Fizgig trainer.py parse_block_spec)."""
    text = str(spec if spec is not None else "").strip()
    if not text:
        raise ValueError("no blocks given")
    out = set()
    for part in text.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", chunk)
        if _block_index(chunk) is not None:
            out.add(_block_index(chunk))
        elif m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo > hi:
                raise ValueError(f"range runs backwards: {chunk!r}")
            out.update(range(lo, hi + 1))
        elif re.fullmatch(r"\d+", chunk):
            out.add(int(chunk))
        else:
            raise ValueError(f"cannot read {chunk!r} - use numbers and ranges, e.g. '3-12, 14-15, 22, 31-33'")
    if not out:
        raise ValueError("no blocks given")
    if num_blocks is not None:
        bad = sorted(i for i in out if i >= num_blocks)
        if bad:
            raise ValueError(f"block(s) {bad} do not exist - this model has {num_blocks} (0-{num_blocks - 1})")
    return sorted(out)


class H3VAE:
    """The H3 video VAE file, split lazily: the encoder (fp32, caching) and the ViT3D decoder (fp16, previews; kept on
    the CPU between decodes so it never sits beside the training model - Fizgig trainer.py _video_dec_state)."""

    def __init__(self, path, device):
        self.path, self.device = str(path), torch.device(device)
        self._enc = self._dec = None

    def _load(self, model, prefixes):
        from training.families.minimax_h3.weights import SafeReader
        with SafeReader(self.path) as f:
            sd = {k: f.get_tensor(k) for k in f.keys() if k.startswith(prefixes)}
        missing, _ = model.load_state_dict(sd, strict=False)
        if missing:
            raise ValueError(f"{self.path} is not the MiniMax H3 video VAE (missing {missing[:4]})")
        return model

    def encoder(self):
        if self._enc is None:
            from training.families.minimax_h3.vae import MiniMaxH3VideoVAEEncoder
            m = self._load(MiniMaxH3VideoVAEEncoder(), ("encoder.", "quant_conv.", "latents_"))
            self._enc = m.to(self.device, torch.float32).eval().requires_grad_(False)
        return self._enc

    def decoder(self):
        if self._dec is None:
            from training.families.minimax_h3.vae import MiniMaxH3VideoVAEDecoder
            # fp16, not bf16: the weights ship fp16 and ComfyUI permits this VAE only fp16 / fp32 (Fizgig trainer.py)
            m = self._load(MiniMaxH3VideoVAEDecoder(), ("decoder.", "post_quant_conv.", "latents_"))
            self._dec = m.to(torch.float16).eval().requires_grad_(False)
        return self._dec


class MiniMaxH3Driver(FamilyDriver):
    supports_batching = False            # prompts differ in length (the DiT is batch size 1)

    compute_dtype = torch.bfloat16       # Fizgig trains and samples the DiT in bf16

    def __init__(self):
        self.lownoise_pct = DEFAULT_LOWNOISE_PCT
        self.highnoise_lr = 1.0
        self.likeness_mode = "default"
        self.blocks_spec = "all"
        self.adapter_ramp = 0.0
        self.train_refiner = False
        self._ramp = None
        self._warned_range = False

    # ---- run options (driver_options: the Train tab's H3 dials) --------------------------------------------------
    @staticmethod
    def _mode(label) -> str:
        """Fizgig minimax_likeness_mode: 'Default' / 'More Blocks' / 'Off ...' -> default | more | off (anything else
        is default)."""
        s = re.split(r"[·\-]", str(label or ""), maxsplit=1)[0].strip().lower()
        if s in ("more blocks", "all blocks", "ultra", "ultra quality"):
            return "more"
        if s.startswith("off"):
            return "off"
        return "default"

    def configure(self, **options) -> None:
        """The Train tab's H3 options (Fizgig 7.0.1 MiniMaxDriver.set_options / _shift / training_loss / _ramp_setup):
        structure + lownoise_pct (the clean-end share), highnoise_lr_pct, likeness_mode, blocks, adapter_ramp,
        train_token_refiner."""
        pct = options.get("lownoise_pct", self.lownoise_pct)
        if "structure" in options:
            word = str(options["structure"] or "").split()[0].lower() if str(options["structure"] or "").strip() \
                else "likeness"
            pct = STRUCTURE_PCT.get(word, pct)
        if "lownoise_pct" in options or "structure" in options:
            pct = float(str(pct).rstrip("%"))
            if S.lownoise_to_shift(pct) is None:
                raise ValueError(f"The clean-end share must be strictly between 0 and 100 % (got {pct:g})")
            self.lownoise_pct = pct
        if "highnoise_lr_pct" in options:
            raw = options["highnoise_lr_pct"]
            v = 100.0 if raw in (None, "") else float(str(raw).rstrip("%"))
            self.highnoise_lr = max(0.0, min(1.0, v / 100.0))           # Fizgig clamps to 0-100 %
        if "likeness_mode" in options:
            self.likeness_mode = self._mode(options["likeness_mode"])
        if "blocks" in options:
            self.blocks_spec = str(options["blocks"] or "all").split("·")[0].strip() or "all"
        if "adapter_ramp" in options:
            tok = str(options["adapter_ramp"] or "").split()
            try:
                self.adapter_ramp = max(0.0, float(tok[0])) if tok else 0.0
            except ValueError:
                self.adapter_ramp = 0.0                                 # "Off"
        if "train_token_refiner" in options:
            self.train_refiner = str(options["train_token_refiner"]).strip().lower() in ("1", "true", "yes", "on")
        self._block_index = None
        self.trained_blocks()            # a bad block list fails here, before anything loads
        logger.info(f"[h3] clean-end share {self.lownoise_pct:g}% -> schedule shift {self.shift:.4g}; training mode "
                    f"{self.likeness_mode}: blocks {self.trained_blocks_spec()}"
                    + (f"; steps above sigma 0.5 at {self.highnoise_lr * 100:.0f}% of the LR"
                       if self.highnoise_lr != 1.0 else "")
                    + (f"; adapter-relative LR {self.adapter_ramp:g}" if self.adapter_ramp else "")
                    + ("; the token refiner trains" if self.trains_refiner() else ""))

    def trains_refiner(self) -> bool:
        """Fizgig: the refiner is a LoRA target with train_token_refiner=1, but a routed step (Default mode) freezes it
        and a --train_blocks list (More Blocks, a typed list) leaves it out - so it trains in mode Off over every
        block, or when the typed list names h3_rf_N."""
        if not self.train_refiner or self.likeness_mode != "off":
            return False
        spec = self.blocks_spec.lower()
        return spec == "all" or "h3_rf_" in spec

    @property
    def shift(self) -> float:
        return S.lownoise_to_shift(self.lownoise_pct)

    def _n_blocks(self, dit=None) -> int:
        return len(dit.blocks) if dit is not None else self.description.n_blocks

    def trained_blocks(self, n_blocks: int = None) -> list:
        """The block indices the LoRA trains in the current mode (Fizgig: Default = photo blocks 20-49, More Blocks =
        6-49, Off = the typed spec; 'all' = every block). Clipped to the model's block count."""
        n = n_blocks or self.description.n_blocks
        if self.likeness_mode == "default":
            idx = parse_block_spec(LIKENESS_BLOCKS)
        elif self.likeness_mode == "more":
            idx = parse_block_spec(FULL_MODEL_BLOCKS)
        elif self.blocks_spec.lower() == "all":
            idx = list(range(n))
        else:
            spec = ",".join(t for t in self.blocks_spec.split(",") if not t.strip().lower().startswith("h3_rf_"))
            idx = parse_block_spec(spec, n)
        return [i for i in idx if i < n]

    def trained_blocks_spec(self) -> str:
        idx = self.trained_blocks()
        runs, start, prev = [], idx[0], idx[0]
        for i in idx[1:] + [None]:
            if i is not None and i == prev + 1:
                prev = i
                continue
            runs.append(str(start) if start == prev else f"{start}-{prev}")
            if i is not None:
                start = prev = i
        return ",".join(runs)

    # ---- models ---------------------------------------------------------------------------------
    def load_dit(self, path, device):
        from training.families.minimax_h3.model import load_minimax_h3_dit
        return load_minimax_h3_dit(path, device=device)

    def max_blocks_to_swap(self, dit=None):
        return self._n_blocks(dit) - 2

    def enable_block_swap(self, dit, num_blocks, device, supports_backward=True):
        dit.enable_block_swap(num_blocks, device, supports_backward)
        dit.move_to_device_except_swap_blocks(device)

    def block_swap_mode(self, dit, inference):
        if inference:
            dit.switch_block_swap_for_inference()
        else:
            dit.switch_block_swap_for_training()

    def load_vae(self, path, device):
        return H3VAE(path, device)

    def load_text_encoder(self, path, device):
        from training.families.minimax_h3.embedder import load_h3_text_encoder
        from training.quant import free_vram_gb
        dev = torch.device(device)
        return load_h3_text_encoder(path, device=dev, free_gb=free_vram_gb() if dev.type == "cuda" else None)

    def unload_text_encoder(self, te):
        te.unload()

    def enable_gradient_checkpointing(self, dit, on=True):
        dit.enable_gradient_checkpointing(on)

    # ---- encoding -------------------------------------------------------------------------------
    @torch.no_grad()
    def encode_images(self, vae, images):
        """Fizgig minimax/caching.py encode_and_save_latents: pixels in [-1, 1] -> the video VAE's mean, normalised with
        its latents_mean / std; a still is one frame, so (B, 24, 1, h, w) is squeezed to (24, h, w)."""
        enc = vae.encoder()
        x = torch.stack([torch.from_numpy(np.ascontiguousarray(a[..., :3])) for a in images])
        x = x.permute(0, 3, 1, 2).float() / 127.5 - 1.0                        # (B, 3, H, W)
        z = enc.encode(x.to(vae.device, torch.float32))                        # (B, 24, 1, h, w)
        return [lat.squeeze(1).float().cpu() for lat in z]

    @torch.no_grad()
    def encode_text(self, te, captions):
        return [{"hidden_states": h.to(torch.bfloat16).cpu()} for h in te.encode_batch(list(captions))]

    # ---- training -------------------------------------------------------------------------------
    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None):
        """Fizgig minimax/trainer.py compute_loss (image path). latents (1, 24, h, w); cond: hidden_states (1, L, 5120).

        noised = (1 - sigma) x0 + sigma noise;  t = 1 - sigma goes to the DiT;  the head predicts x0 - noise, so the
        target is x0 - noise (sign convention matched to ComfyUI)."""
        if refs:
            raise RuntimeError("MiniMax H3 has no reference / edit training in this port")
        if (min_t > 0.0 or max_t < 1.0) and not self._warned_range:
            self._warned_range = True
            logger.warning("[h3] the noise range boxes do nothing for MiniMax H3: it trains on its own schedule (set "
                           "by Training structure).")
        if latents.shape[0] != 1:
            raise ValueError("MiniMax H3 image training is batch size 1")
        device = latents.device
        dt = self.compute_dtype
        x0 = latents.to(dt).float()                    # Fizgig hands the trainer the cache cast to the training dtype
        if x0.dim() == 4:
            x0 = x0.unsqueeze(2)                       # (1, 24, 1, h, w)
        _pt, ph, pw = dit.patch_size
        hh, ww = (x0.shape[-2] // ph) * ph, (x0.shape[-1] // pw) * pw
        if (hh, ww) != tuple(x0.shape[-2:]):           # an odd latent grid: drop <= 1 row / column so patchify is exact
            x0 = x0[..., :hh, :ww].contiguous()
        noise = torch.randn(x0.shape, generator=generator).to(device)
        sigma = S.sample_sigmas(1, self.shift, generator)
        s = sigma.reshape(1, 1, 1, 1, 1).to(device)
        noised = (1.0 - s) * x0 + s * noise
        t = (1.0 - sigma).to(device)
        text = cond["hidden_states"].to(device=device, dtype=dt)
        a_noise = torch.randn(2 * 2, dit.config.audio_latents_dim, generator=generator).to(device)
        pred = dit(noised.to(dt), t, text, audio_noise=a_noise)
        loss = F.mse_loss(pred.float(), (x0 - noise).float())
        info = {"t": float(sigma[0])}
        if self.highnoise_lr != 1.0:
            # Fizgig's noise-band LR: a step drawn in the noisy half trains at this share of the rate (the loop averages
            # the multiplier over an accumulation window and ignores it under Automagic v3)
            info["lr_mult"] = self.highnoise_lr if float(sigma[0]) >= S.LOWNOISE_SIGMA else 1.0
        return loss, info

    # ---- sampling -------------------------------------------------------------------------------
    @torch.no_grad()
    def initial_noise(self, seed, width, height):
        return S.initial_noise(seed, width, height)

    def _text(self, cond, device):
        h = cond["hidden_states"]
        if h.dim() == 2:
            h = h[None]
        return h.to(device=device, dtype=self.compute_dtype)

    @torch.no_grad()
    def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, sigmas=None, options=(),
                 noise=None, on_step=None, refs=None):
        device = dit.video_patch_proj.weight.device
        text = self._text(cond, device)
        uncond = self._text(neg_cond, device) if (neg_cond is not None and cfg > 1.0) else None
        return S.sample_image(dit, text, width=width, height=height, steps=steps, cfg_scale=cfg, uncond_embeds=uncond,
                              seed=seed, noise=noise, on_step=on_step, device=device, dtype=self.compute_dtype)

    @torch.no_grad()
    def decode(self, vae, latents, width, height):
        from PIL import Image
        dec = vae.decoder()
        dev = vae.device
        dec.to(dev)
        try:
            px = dec.decode(latents.to(dev))[0]                                  # (3, H, W) in [0, 1]
        finally:
            dec.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return Image.fromarray((px * 255.0).permute(1, 2, 0).cpu().float().numpy().round().astype(np.uint8))

    # ---- the adapter-relative LR ramp (Fizgig option adapter_ramp) -------------------------------------
    def prepare_training(self, dit, net, **_run) -> None:
        self._ramp = _AdapterRamp(net, self.adapter_ramp) if self.adapter_ramp > 0 else None
        if self._ramp is not None:
            logger.info(f"[ramp] adapter-relative LR ON - each step held at {100 * self.adapter_ramp:.3f}% of the "
                        f"adapter's current size, starting at 10% of the learning rate")

    def step_policy(self, batch, epoch):
        return False, (self._ramp.mult if self._ramp is not None else 1.0)

    def after_optimizer_step(self) -> None:
        if self._ramp is not None:
            self._ramp.step()

    def after_epoch(self, epoch, steps_remaining) -> None:
        if self._ramp is not None:
            logger.info(self._ramp.epoch_report())

    def run_metadata(self) -> dict:
        """Fizgig 7.0.1 MiniMaxDriver.run_metadata for a still-image run: the clip, voice and distillation entries
        read as off, as they do in Fizgig on a folder of stills."""
        blocks = self.trained_blocks_spec() if self.likeness_mode == "default" else "all"
        md = {"ss_visual_stop": "off", "ss_audio_stop": "off",
              "ss_photo_blocks": blocks, "ss_clip_blocks": blocks, "ss_audio_blocks": blocks, "ss_tread": "off",
              "ss_clip_still_as_photo": "0", "ss_distill": "off",
              "ss_adapter_ramp": f"{self.adapter_ramp:g}" if self.adapter_ramp else "off",
              "ss_train_token_refiner": "1" if self.trains_refiner() else "0",
              "ss_timestep_density": f"{float(self.shift):g}"}
        if self.likeness_mode != "default":
            idx = self.trained_blocks()
            if len(idx) < self.description.n_blocks or self.trains_refiner():
                md["ss_train_blocks"] = ",".join([f"h3blk_{i}" for i in idx] +
                                                 ([f"h3_rf_{i}" for i in range(N_REFINER)] if self.trains_refiner()
                                                  else []))
        return md

    # ---- LoRA and the block map -------------------------------------------------------------------
    def block_map(self, dit=None):
        """The 50 blocks (4 Linears each) and the two token-refiner blocks - the model's map, whatever the training
        mode trains. Ids are Fizgig 7.0.1's (the old Repair Studio's): h3blk_N, h3_rf_N."""
        names = {n for n, _ in dit.named_modules()} if dit is not None else None

        def keep(mods):
            return [m for m in mods if names is None or m in names]
        main = [Block(f"h3blk_{i}", f"Block {i}", keep([f"blocks.{i}.{m}" for m in _BLOCK_MODULES]))
                for i in range(self._n_blocks(dit))]
        n_rf = len(dit.token_refiner.blocks) if dit is not None else N_REFINER
        refiner = [Block(f"h3_rf_{i}", f"Refiner {i}", keep([f"token_refiner.blocks.{i}.{m}" for m in _BLOCK_MODULES]))
                   for i in range(n_rf)]
        return [BlockGroup("Blocks", main), BlockGroup("Token Refiner", refiner)]

    def lora_target_names(self, dit):
        """The trainable adapters' Linears: the training mode's blocks only (Default 20-49), and the token refiner when
        it trains (trains_refiner). The frozen training adapter still reaches every block it adapts."""
        groups = self.block_map(dit)
        keep = set(self.trained_blocks(self._n_blocks(dit)))
        out = [m for i, b in enumerate(groups[0].blocks) if i in keep for m in b.modules]
        if self.trains_refiner():
            out += [m for b in groups[1].blocks for m in b.modules]
        return out

    def quant_target_names(self, dit):
        """Every main-block Linear is quantised (the base is frozen everywhere), not just the trained window."""
        return [m for b in self.block_map(dit)[0].blocks for m in b.modules]


class _AdapterRamp:
    """Fizgig minimax/common.py AdapterRamp, as its driver builds it (_ramp_setup: target R, start at 10% of the LR):
    each step held at a constant fraction of the adapter's current size ||dW||, so the rate climbs toward the configured
    ceiling as the adapter grows. The size reads the family LoRA's trainable adapters (Fizgig _adapter_size)."""

    def __init__(self, net, target_rel):
        self.net, self.target, self.mult = net, float(target_rel), 0.1
        self._smooth = self._prev = None

    @torch.no_grad()
    def _size(self) -> float:
        from training.lora import TRAINABLE
        tot = 0.0
        for w in self.net.wrapped.values():
            m = w.adapters[TRAINABLE] if TRAINABLE in w.adapters else None
            if m is None:
                continue
            if hasattr(m, "lokr_w1"):
                n = (m.lokr_w1.float().norm() * m.lokr_w2.float().norm()) ** 2
            else:
                a, b = m[0].weight.float(), m[1].weight.float()
                n = torch.trace((a @ a.T) @ (b.T @ b)).clamp(min=0)
            tot += float(n) * w.scales[TRAINABLE] ** 2
        return tot ** 0.5

    @torch.no_grad()
    def step(self) -> float:
        cur = self._size()
        if self._prev is None or cur <= 1e-9:
            self._prev = cur
            return self.mult
        rel = max(0.0, cur - self._prev) / cur      # this step as a fraction of what exists
        self._prev = cur
        self._smooth = rel if self._smooth is None else 0.9 * self._smooth + 0.1 * rel
        if self._smooth > 1e-12:
            # Fizgig's damped per-step gain caps: up at most 1.01x, down at most 0.95x, floor 2% of the LR
            err = self._smooth / self.target
            self.mult = min(1.0, max(0.02, self.mult * min(1.01, max(0.95, err ** -0.3))))
        return self.mult

    def epoch_report(self) -> str:
        rel = self._smooth or 0.0
        return (f"[ramp] adapter ||dW||={self._prev or 0:.2f}, growing {100 * rel:.3f}%/step (target "
                f"{100 * self.target:.3f}%) - LR at {100 * self.mult:.0f}% of the configured ceiling")
