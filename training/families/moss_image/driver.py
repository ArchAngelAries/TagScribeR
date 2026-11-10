# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/moss_image/driver.py (commit 9162fb20, v8.7.3).
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: the two family options arrive typed through `configure` (training/params.py DRIVER_OPTIONS:
# MOSS_LORA_SFT -> lora_sft, MOSS_LOWNOISE_PCT -> lownoise_pct) and are stored as Fizgig's option strings, so the
# ported methods read `self.options` unchanged and Fizgig's `set_options` path still works; the VAE is the shared
# training/modules/flux2_vae.py; the tokenizer loads cache-first (training/hf_cache.py); `ft_spec` (the full
# fine-tune) imports training/ft.py; the workbench-only hooks (text_encoder_gb, drop_text_encoder, keep_text_encoder) are
# left out; `cache_stage` becomes the `text_cache_batch_size` attribute (training/cache.py has no stage hook);
# otherwise unchanged.
"""MOSS-Image driver for the training family layer (training/driver.py).

* conditioning: Qwen3.5-9B's final hidden state (after the final norm), the chat template with an open think block,
  one unpadded prompt at a time -> {"hidden_states": (L, 4096)} (embedder.py)
* latents: the FLUX.2 autoencoder (training/modules/flux2_vae.py), BatchNorm-normalised and 2x2-packed -> (128, h, w)
  at /16. MOSS-Image's own VAE file is this encoder (bf16-rounded) with the FLUX.2 small decoder; the latent space is
  the same
* training: flow matching on the Base checkpoint, x_t = (1 - sigma) x0 + sigma noise, target noise - x0, the DiT told
  1000 * sigma; sigma drawn from the final pretraining stage's distribution (technical report Appendix A.2, Table 6:
  truncated logit-normal, mu -1.09, s 0.95, cutoff 0.95 rescaled onto the 1000-step grid, 1% uniform), not AI
  Toolkit's shift-5 uniform. The LoRA trains all 42 blocks, block 22 included (leaving it out, as AI Toolkit does,
  cost likeness in A/B runs: the positive branch always runs it)
* sampling: the reference's snapped shift-5 schedule, Euler, CFG 3 with block 22 skipped in the negative branch

No edit training: the released checkpoints are text-to-image only (technical report section 8, "does not directly
condition on reference images or support image editing"; the pipeline's defaults "match the finalized dense
MOSSImage model, not ... reference-image variants").
"""
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

from training.driver import Block, BlockGroup, FamilyDriver
from training.families.moss_image import sampling as S

logger = logging.getLogger(__name__)

N_BLOCKS = 42                 # transformer/config.json num_layers
N_TEXT_BLOCKS = 2             # the text adapter's blocks (pipeline _TextProjection)
BLOCK_MODULES = ("proj.q", "proj.k", "proj.v", "proj.o", "proj.ffn_up_gate", "proj.ffn_down")
ATTN_MODULES = ("proj.q", "proj.k", "proj.v", "proj.o")
TEXT_MODULES = ("qkv", "proj", "ffn_gate", "ffn_up", "ffn_down")

# technical report Appendix A.2, Table 6, the last Stage 3 interval (steps 274,001-310,000, the released weights)
T_MU, T_S, T_CUTOFF, T_UNIFORM = -1.09, 0.95, 0.95, 0.01
# below this much free VRAM the bf16 text encoder (about 15 GB) streams layer by layer from the CPU (Fizgig's figure)
TE_RESIDENT_MIN_FREE_GB = 18.0


def _t_cdf(x, mu):
    """P(S <= x) for S = 1 - sigmoid(Z), Z ~ N(mu, T_S^2)."""
    return 1.0 - 0.5 * math.erfc(-((math.log((1 - x) / x) - mu) / T_S) / math.sqrt(2))


def mu_for_clean_share(pct):
    """The Z mean that puts `pct`% of training steps below sigma 0.5 (the clean end), the 1% uniform included; the
    curve keeps the reference's shape and width. The reference's own mu -1.09 gives about 11%."""
    want = (float(pct) / 100.0 - T_UNIFORM * 0.5) / (1.0 - T_UNIFORM)
    want = min(max(want, 1e-4), 1.0 - 1e-4)
    lo, hi = -12.0, 12.0
    for _ in range(80):                       # share rises with mu: bisect
        mid = 0.5 * (lo + hi)
        if _t_cdf(0.5 * T_CUTOFF, mid) / _t_cdf(T_CUTOFF, mid) < want:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class MOSSImageDriver(FamilyDriver):

    # TagScribeR: the text cache batch Fizgig's cache_stage asks for on the text stage (driver.py:158-163 upstream), so a
    # streamed encoder moves each layer to the GPU once per group rather than once per 8 captions
    text_cache_batch_size = 4096

    # ---- family options (training/params.py DRIVER_OPTIONS -> configure; Fizgig's set_options strings) -----------
    def configure(self, **options):
        """MOSS_LORA_SFT -> lora_sft (Train attention only), MOSS_LOWNOISE_PCT -> lownoise_pct (Clean-end share, % of
        steps below sigma 0.5; empty = the reference's own schedule). Kept as Fizgig's option strings ("lora_sft=1",
        "lownoise_pct=30") so the ported code below reads them unchanged."""
        options.pop("compile_blocks", None)     # read by the training loop, not the driver
        opts = dict(self.options)
        for key, value in options.items():
            if key == "lora_sft":
                opts["lora_sft"] = "1" if value not in (False, "", "0", "False", "false", None, 0) else ""
            elif key == "lownoise_pct":
                raw = "" if value in (None, "") else str(value).rstrip("%").strip()
                if raw:
                    pct = float(raw)
                    if not 0.0 < pct < 100.0:
                        raise ValueError(f"The low-noise share must be strictly between 0 and 100 % (got {pct:g})")
                    raw = f"{pct:g}"
                opts["lownoise_pct"] = raw
            else:
                raise ValueError(f"unknown MOSS-Image option {key!r}")
        self.options = opts
        self._mu_cached = None

    @property
    def attention_only(self) -> bool:
        return str(self.options.get("lora_sft", "")) == "1"

    # ---- models ---------------------------------------------------------------------------------
    def load_dit(self, path, device):
        from training.families.moss_image.model import load_moss_dit
        return load_moss_dit(path, device=device).eval().requires_grad_(False)

    # ---- training previews on the SFT checkpoint (train_preview_checkpoint; after @Murf-Code, Fizgig PR #195) -------
    def park_for_preview(self, dit, device):
        """Stream all but two of the training model's blocks so the SFT model fits beside it; an NF4 base cannot
        stream and is small - left as it is (token None)."""
        if getattr(dit, "_nf4_quantized", False):
            return None
        orig = int(dit.blocks_to_swap or 0)
        dit.enable_block_swap(len(dit.layers) - 2, torch.device(device), True)
        dit.prepare_block_swap_before_forward()
        return orig

    def load_preview_checkpoint(self, path, device, int8=False):
        """The SFT DiT for this preview round, sized to the VRAM left beside the parked training model: bf16 when it
        fits with room for the render, else INT8, with blocks streamed (forward-only) when even that does not fit.
        ComfyUI's int8 ConvRot file is decoded at load (model.load_moss_dit), so it sizes as the bf16 file."""
        import os
        from training import quant
        cuda = torch.cuda.is_available() and torch.device(device).type == "cuda"
        if cuda:
            torch.cuda.empty_cache()
        free = quant.free_vram_gb() if cuda else float("inf")     # TAGSCRIBER_SIM_VRAM_GB: the simulated card's
        size = os.path.getsize(path) / 1024 ** 3
        if "convrot" in os.path.basename(path).lower():
            size *= 2                       # int8 on disk, bf16 once decoded
        room = 4.0                          # a 1024 px render + the VAE decode
        precision = "int8" if int8 or free < size + room else "bf16"
        need = (size / 2 if precision == "int8" else size) + room
        swap = 0
        if free < need:
            per_block = (size / 2 if precision == "int8" else size) / N_BLOCKS
            swap = min(self.max_blocks_to_swap(), max(1, int((need - free) / max(per_block, 1e-3)) + 2))
        m, swapped = quant.load_base(self, path, device, precision, swap, supports_backward=False)
        if swapped:
            self.block_swap_mode(m, inference=True)
        logger.info(f"[sample] SFT checkpoint: {precision}, {swapped} blocks streamed"
                    + (f" ({free:.1f} GB free)" if cuda else ""))
        return m.eval().requires_grad_(False), swapped

    def unpark_after_preview(self, dit, device, token):
        """The run's own swap back, or with none the preview-time offloader torn down and the model back on the GPU."""
        if token is None:
            return
        device = torch.device(device)
        if token > 0:
            dit.enable_block_swap(token, device, True)
            dit.move_to_device_except_swap_blocks(device)
        else:
            from training import quant
            dit.disable_block_swap()
            quant.move(dit, device)
        dit.switch_block_swap_for_training()

    def max_blocks_to_swap(self, dit=None):
        return (len(dit.layers) if dit is not None else N_BLOCKS) - 2

    def enable_block_swap(self, dit, num_blocks, device, supports_backward=True):
        dit.enable_block_swap(num_blocks, device, supports_backward)
        dit.move_to_device_except_swap_blocks(device)

    def block_swap_mode(self, dit, inference):
        if inference:
            dit.switch_block_swap_for_inference()
        else:
            dit.switch_block_swap_for_training()

    def load_vae(self, path, device):
        """The FLUX.2 autoencoder (flux2-vae_small / flux2-vae.safetensors / Klein's ae.safetensors), fp32 as the
        reference runs it."""
        from training.modules.flux2_vae import load_flux2_vae
        return load_flux2_vae(path, device=device)

    def load_text_encoder(self, path, device):
        """Text-only Qwen3.5-9B (no vision tower, LM head or MTP head) in bf16: resident (about 15 GB) when it fits
        the free VRAM with room to run, else streamed layer-major from the CPU (embedder.MOSSTextEncoder) - still
        exact bf16, which OpenMOSS ask for (their text encoder is "much more unstable than DiT"), not 8-bit."""
        from training.families.moss_image.embedder import MOSSTextEncoder, load_tokenizer
        from training.quant import free_vram_gb
        dev = torch.device(device)
        stream = dev.type == "cuda" and free_vram_gb() < TE_RESIDENT_MIN_FREE_GB
        return MOSSTextEncoder(path, device=dev, stream=stream, tokenizer=load_tokenizer())

    def unload_text_encoder(self, te):
        te.unload()

    def enable_gradient_checkpointing(self, dit, on=True):
        dit.enable_gradient_checkpointing(on)

    def compile_targets(self, dit):
        return dit.layers

    # ---- full fine-tune (training/ft.py; Fizgig 8.7.3 moss_image/driver.py:175-190) ----------------------------
    def ft_spec(self, dit):
        # each block's six Linears are the parts (attention q / k / v / o ~0.018 GB each, the fused MLP up+gate 0.094,
        # the down projection 0.047 in bf16). ComfyUI's file fuses q / k / v into one proj.qkv = [q; k; v] tensor; the
        # diffusers shards keep them apart. The text adapter, embedders, modulations and the final layer stay frozen,
        # as for a LoRA; on SFT or a fine-tune of it (lora_sft) only attention trains, as for a LoRA.
        # Memory: measured upstream 19 Oct on an H100 at 1 MP, the whole model in one window: all six parts peaked
        # 12.6 GB, attention only 8.3 GB. overhead_gb is fitted to each in the planner's frame (4.5 = 0.6 GB above the
        # full run's fit; 6.0 attention only, whose window is credited the whole 4-bit trunk though the FFN stays in
        # it). act_gb_per_mp is not measured yet (2.0, as Anima). Not run by TagScribeR on any hardware.
        from training.ft import FTSpec
        sft = self.attention_only
        return FTSpec(blocks="layers", components=ATTN_MODULES if sft else BLOCK_MODULES,
                      overhead_gb=6.0 if sft else 4.5, calib_mp=1.0, act_gb_per_mp=2.0,
                      file_layout=(("proj.q.weight", "proj.qkv.weight", 0, 3),
                                   ("proj.k.weight", "proj.qkv.weight", 1, 3),
                                   ("proj.v.weight", "proj.qkv.weight", 2, 3)))

    # ---- encoding -------------------------------------------------------------------------------
    @torch.no_grad()
    def encode_images(self, vae, images):
        p = next(vae.parameters())
        out = []
        for a in images:
            x = torch.from_numpy(np.ascontiguousarray(a[..., :3])).permute(2, 0, 1).float().div(127.5).sub(1.0)
            tokens, ids = vae.encode(x.to(p.device, p.dtype))
            h, w = int(ids[:, 1].max()) + 1, int(ids[:, 2].max()) + 1
            out.append(tokens.reshape(h, w, -1).permute(2, 0, 1).float().cpu())        # (128, h, w)
        return out

    @torch.no_grad()
    def encode_text(self, te, captions):
        return [{"hidden_states": h} for h in te.encode(captions)]

    # ---- training -------------------------------------------------------------------------------
    @staticmethod
    def _sample_t(generator, min_t, max_t, mu=T_MU):
        """The final pretraining interval's sigma (report eq. 4): with probability 0.01 uniform over the grid j / 1000,
        else S = 1 - sigmoid(Z), Z ~ N(mu, 0.95^2) (mu -1.09 unless a clean-end share moves it), truncated to
        S <= 0.95 (inverse CDF), rescaled by 1 / 0.95 and taken to the grid point whose cell holds it. Then scaled into
        [min_t, max_t] (the Timestep Range)."""
        T = S.TRAIN_STEPS
        u = torch.rand(2, generator=generator)
        if float(u[0]) < T_UNIFORM:
            sigma = (math.floor(float(u[1]) * T) + 1) / T
        else:
            # draw F(S) uniformly in (0, F(cutoff)], F = _t_cdf
            p = max(float(u[1]), 1e-12) * _t_cdf(T_CUTOFF, mu)
            z = mu + T_S * float(torch.special.ndtri(torch.tensor(1.0 - p, dtype=torch.float64)))
            s = 1.0 / (1.0 + math.exp(z))                 # S = 1 - sigmoid(Z) = sigmoid(-Z)
            sigma = min(max(math.ceil(T * s / T_CUTOFF), 1), T) / T
        return min_t + (max_t - min_t) * sigma

    def _t_mu(self):
        """lownoise_pct=P (the share of steps below sigma 0.5) moves the curve's centre; unset = the reference's."""
        pct = str(self.options.get("lownoise_pct", "") or "").rstrip("%").strip()
        if not pct:
            return T_MU
        cached = getattr(self, "_mu_cached", None)
        if cached is None or cached[0] != pct:
            cached = self._mu_cached = (pct, mu_for_clean_share(pct))
            logger.info(f"MOSS-Image: {pct}% of training steps at the clean end (sigma < 0.5), Z mean "
                        f"{cached[1]:.2f} (reference -1.09, about 11%)")
        return cached[1]

    def _forward(self, dit, xt, t, cond):
        hidden = cond["hidden_states"]
        hidden = hidden if hidden.dim() == 3 else hidden[None]
        tt = torch.tensor([t * S.TRAIN_STEPS], device=xt.device)
        return dit(xt, tt, hidden.to(xt.device))

    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None, diff_ref=None,
                      diff_weight=0.0):
        if refs:
            raise RuntimeError("MOSS-Image has no edit (reference-image) conditioning")
        x0 = latents.float()
        noise = torch.randn(x0.shape, generator=generator).to(x0.device)
        t = self._sample_t(generator, min_t, max_t, self._t_mu())
        xt = (1 - t) * x0 + t * noise
        pred = self._forward(dit, xt, t, cond).float()
        target = noise - x0
        if diff_ref is not None and diff_weight > 0.0:
            # image-pair sliders (Krea 2's weighting): each token's error weighted by how much the two poles differ
            # there, so the slider learns what changes between them, not what they share
            d = (x0 - diff_ref.to(x0.device).float()).abs().mean(dim=1)                       # (1, h, w)
            dm = d.mean().clamp_min(1e-8)
            r = (d / dm).clamp(max=8.0)
            wgt = (1.0 - float(diff_weight)) + float(diff_weight) * r
            wgt = wgt / wgt.mean().clamp_min(1e-8)
            if float(dm) <= 1e-6:
                wgt = torch.ones_like(wgt)                      # identical pair: uniform, never all-zero
            return ((pred - target).pow(2).mean(dim=1) * wgt).mean(), {"t": t}
        # report eq. 3: per-token mean over channels, no sigma weighting
        return F.mse_loss(pred, target), {"t": t}

    def noise_latents(self, latents, generator, *, min_t=0.0, max_t=1.0):
        x0 = latents.float()
        noise = torch.randn(x0.shape, generator=generator).to(x0.device)
        t = self._sample_t(generator, min_t, max_t, self._t_mu())
        return {"xt": (1 - t) * x0 + t * noise, "t": t}

    def predict(self, dit, state, cond):
        return self._forward(dit, state["xt"], state["t"], cond)

    # ---- sampling -------------------------------------------------------------------------------
    @torch.no_grad()
    def initial_noise(self, seed, width, height):
        return S.initial_noise(seed, height, width)

    @staticmethod
    def _real_tokens(cond):
        """A prompt-travel blend (pad_conditioning) keeps only the tokens either waypoint has; the DiT drops padding
        the same way (it evaluates each prompt at its true length)."""
        h = cond["hidden_states"]
        h = h[0] if h.dim() == 3 else h
        if "mask" in cond:
            m = cond["mask"]
            m = (m[0] if m.dim() == 2 else m).to(h.device)
            h = h[m > 0]
        return h

    @torch.no_grad()
    def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, sigmas=None, options=(),
                 noise=None, on_step=None, refs=None):
        if refs:
            raise RuntimeError("MOSS-Image has no edit (reference-image) conditioning")
        if sigmas is not None and len(sigmas) not in (steps, steps + 1):
            sigmas = None                   # an explicit schedule only applies at its own step count
        # the description's default sampler under the caller's options (training previews pass none)
        opts = {**dict(self.description.default_sampling().options), **dict(options)}
        neg = self._real_tokens(neg_cond) if (neg_cond is not None and cfg != 1.0) else None
        if cfg != 1.0 and neg is None:
            raise ValueError("MOSS-Image samples with CFG against an encoded negative (the empty prompt by default); "
                             "none was given")
        skip = tuple(opts.get("uncond_skip", S.UNCOND_SKIP) or ())
        return S.sample(dit, self._real_tokens(cond), height, width, steps=steps, seed=seed, cfg=cfg, neg_text=neg,
                        noise=noise, on_step=on_step, sigmas=sigmas, shift=float(opts.get("shift", S.SHIFT)),
                        uncond_skip=skip, sampler=str(opts.get("sampler", "euler")),
                        scheduler=str(opts.get("schedule", "reference")), slg_end=float(opts.get("slg_end", 1.0)))

    def pad_conditioning(self, conds):
        """Prompt travel: prompts zero-padded to one length with a mask of each one's real tokens. generate keeps only
        the masked-in tokens, so each waypoint renders exactly as its own prompt."""
        L = max(c["hidden_states"].shape[-2] for c in conds)
        out = []
        for c in conds:
            h = c["hidden_states"][0] if c["hidden_states"].dim() == 3 else c["hidden_states"]
            pad = L - h.shape[0]
            out.append({"hidden_states": torch.cat([h, h.new_zeros(pad, h.shape[1])]) if pad else h,
                        "mask": torch.cat([torch.ones(h.shape[0], dtype=torch.bool),
                                           torch.zeros(pad, dtype=torch.bool)])})
        return out

    @torch.no_grad()
    def decode(self, vae, latents, width, height):
        from PIL import Image
        from training.modules.flux2_vae import latent_ids
        z = latents[0] if latents.dim() == 4 else latents                         # (128, h, w)
        h, w = z.shape[-2:]
        tokens = z.permute(1, 2, 0).reshape(h * w, -1)
        p = next(vae.parameters())
        img = vae.decode(tokens.to(p.device), latent_ids(h, w, p.device)).float().clamp(-1, 1)
        return Image.fromarray(((img + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy())

    # ---- LoRA structure -------------------------------------------------------------------------
    def block_map(self, dit=None):
        """The 42 joint blocks, then the two-block text adapter that refines Qwen3.5's states before they join the
        image tokens (the report's "Text adapter: linear projection 4096 -> 3072; two text-only Transformer
        blocks"). Works with no model loaded."""
        names = {n for n, _ in dit.named_modules()} if dit is not None else None

        def keep(mods):
            return [m for m in mods if names is None or m in names]
        main = [Block(f"block_{i}", f"Block {i}", keep([f"layers.{i}.{m}" for m in BLOCK_MODULES]))
                for i in range(N_BLOCKS if dit is None else len(dit.layers))]
        text = [Block(f"text_{i}", f"Text adapter {i}",
                      keep([f"text_embedder.blocks.{i}.{m}" for m in TEXT_MODULES])) for i in range(N_TEXT_BLOCKS)]
        return [BlockGroup("Blocks", main), BlockGroup("Text adapter", text)]

    def lora_target_names(self, dit):
        """The 42 blocks' Linears, block 22 included: the negative branch skips it, but the positive one always runs
        it, and training it gave clearly better likeness than leaving it out as AI Toolkit does. The text adapter
        stays frozen with the text encoder (OpenMOSS's guide: keep the text encoder frozen; @Murf-Code, Fizgig PR
        #195: training it exaggerated expressions epoch by epoch). The block map still lists it, for LoRAs from
        trainers that do adapt it. On SFT or a fine-tune of it (the lora_sft option) only attention trains: its
        feed-forward layers trained gave body horror within a few epochs."""
        names = [m for m in super().lora_target_names(dit) if m.startswith("layers.")]
        if self.attention_only:
            names = [m for m in names if m.endswith(ATTN_MODULES)]
        return names

    def quant_target_names(self, dit):
        """Every block Linear, whichever of them the LoRA trains (an SFT base's feed-forward layers too): block swap
        pairs blocks layer by layer, so every block must hold the same dtype - block 22 left bf16 among INT8 blocks
        got INT8 weights swapped into a plain Linear (Fizgig #201)."""
        return [m for m in super().lora_target_names(dit) if m.startswith("layers.")]
