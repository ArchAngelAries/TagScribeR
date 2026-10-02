# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/families/driver.py
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: `supports_batching` (batch size > 1 for fixed-length conditioning, e.g. SDXL) and
# Conv2d LoRA targets (lora_target_names may name Conv2d modules; training/lora.py wraps both).
"""FamilyDriver: the one interface a new model family implements.

The generic code - caching, training, previews, the LoRA layer and the Train tab - talks to a family ONLY through
its FamilyDescription (facts) and its FamilyDriver (model code). A new family is: a description + a driver + its
model package. Nothing else changes to add it.

Conventions every driver follows:
* Latents are (C, h, w) tensors in the family's normalised space (what the model is trained on).
* Conditioning is a dict of tensors per caption, keys chosen by the driver; the generic cache stores it as-is
  and hands the same dict back (batched, leading dim = batch size) to training_loss / generate.
* Images in and out are uint8 RGB numpy arrays (H, W, 3) / PIL images.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Block:
    """One block of a model's LoRA map: a stable id (used in presets, sidecars and saved states), a display label
    in the model's own terms, and the dotted names (relative to the DiT) of the Linears it covers."""
    id: str
    label: str
    modules: list = field(default_factory=list)


@dataclass
class BlockGroup:
    """A named section of blocks (e.g. "Double blocks", "Text fusion", "Refiner", SDXL's "Input blocks")."""
    label: str
    blocks: list = field(default_factory=list)


class FamilyDriver:
    """Subclass per family. `description` is the family's FamilyDescription."""

    description = None

    # ---- models ---------------------------------------------------------------------------------
    def load_dit(self, path: str, device):
        """The diffusion transformer in bf16, frozen, ready for LoRA wrapping (gradient checkpointing available).
        INT8 / NF4 are applied afterwards by families/quant.py to the block map's Linears; the driver only loads."""
        raise NotImplementedError

    # ---- block swap (optional) --------------------------------------------------------------------
    def max_blocks_to_swap(self, dit=None) -> int:
        """How many blocks may stream between CPU and GPU; 0 = the family has no block swap."""
        return 0

    def enable_block_swap(self, dit, num_blocks: int, device, supports_backward: bool = True) -> None:
        """Stream `num_blocks` blocks: called with the model on CPU; leaves everything else on `device`."""
        raise NotImplementedError

    def block_swap_mode(self, dit, inference: bool) -> None:
        """Forward-only streaming for previews (inference=True), back to training layout after."""

    def load_vae(self, path: str, device):
        raise NotImplementedError

    def load_text_encoder(self, path: str, device):
        """The text encoder, for encode_text only - captioning is TagScribeR's captioners' job, so an encoder can
        leave out anything encoding does not use (an LM head, a vision tower)."""
        raise NotImplementedError

    def unload_text_encoder(self, te) -> None:
        """Free the encoder's VRAM (caching and preview-prompt encoding load it, then drop it)."""
        raise NotImplementedError

    def enable_gradient_checkpointing(self, dit, on: bool = True) -> None:
        raise NotImplementedError

    # ---- encoding (the generic cache calls these) -----------------------------------------------
    def encode_images(self, vae, images: list) -> list:
        """uint8 (H, W, 3) arrays, all the same size -> list of (C, h, w) latents."""
        raise NotImplementedError

    def encode_text(self, te, captions: list) -> list:
        """captions -> list of conditioning dicts (tensors on CPU)."""
        raise NotImplementedError

    # ---- edit training (optional) -----------------------------------------------------------------
    supports_references = False       # reference ("before") images: pair datasets and edit previews

    def load_reference_text_encoder(self, path: str, device):
        """The text encoder able to read reference images (encode_text_with_references)."""
        raise NotImplementedError

    def encode_text_with_references(self, te, captions: list, references: list) -> list:
        """captions + one list of uint8 (H, W, 3) reference images per caption, each already at the size its latent
        uses -> conditioning dicts. training_loss / generate get the references' latents as refs=."""
        raise NotImplementedError

    # ---- training -------------------------------------------------------------------------------
    # True when every caption's conditioning has the same shape (e.g. SDXL's 77-token CLIP chunks), so items of one
    # bucket can be stacked into a batch. False (the Fizgig default) = batch size 1, which the loss watch needs anyway.
    supports_batching = False

    def training_loss(self, dit, latents, cond: dict, generator, *, min_t: float = 0.0, max_t: float = 1.0,
                      refs=None):
        """One training forward. latents (B, C, h, w) on device (B = 1 unless supports_batching), cond = the cached dict (batched), refs = the pair's
        reference latents [(1, C, rh, rw), ...] (edit training) or None.
        Returns (loss tensor, info dict e.g. {"t": 0.63}). Owns the family's noise/target/timestep rules."""
        raise NotImplementedError

    # ---- sampling -------------------------------------------------------------------------------
    def initial_noise(self, seed: int, width: int, height: int):
        """The seed's starting noise (CPU float32), exactly what generate() draws for this seed. The workbench
        slerps two of these for seed travel and passes the result back as generate(noise=...)."""
        raise NotImplementedError

    def generate(self, dit, cond: dict, width: int, height: int, *, steps: int, seed: int, cfg: float = 1.0,
                 neg_cond: Optional[dict] = None, sigmas=None, options=(), noise=None, on_step=None, refs=None):
        """Denoise one image from noise (refs: reference latents for an edit, with conditioning from
        encode_text_with_references); returns latents in the driver's own layout (fed to decode).
        sigmas / options: an explicit schedule and driver-specific sampler options (e.g. from a speed LoRA's
        SamplingSettings); a driver ignores what it doesn't use. noise: a start from initial_noise() (or a blend of
        two) instead of the seed's. on_step(done, total): called before every step; it may raise to abort."""
        raise NotImplementedError

    def pad_conditioning(self, conds: list) -> list:
        """Optional (prompt travel): the conditioning dicts brought to one shape so they can be blended, e.g. padded
        with a validity mask. Tensors that are blended are floating point; booleans are combined as a union. A
        driver without it has no prompt travel."""
        raise NotImplementedError

    def decode(self, vae, latents, width: int, height: int):
        """-> PIL.Image (RGB)."""
        raise NotImplementedError

    # ---- LoRA and the block map -------------------------------------------------------------------
    def block_map(self, dit=None) -> list:
        """The model's LoRA structure in its OWN terms: ordered BlockGroups of Blocks. The workbench builds its
        slider panel, greying, presets and bake mapping from this, so a family with named areas (text fusion,
        refiners, SDXL-style input/middle/output blocks) overrides it. Default: one group of `n_blocks` numbered
        blocks from the description, each covering the description's LoRA modules (filtered to those in `dit`)."""
        d = self.description
        names = {n for n, _ in dit.named_modules()} if dit is not None else None
        blocks = []
        for i in range(d.n_blocks):
            mods = [f"{d.block_prefix}.{i}.{m}" for m in d.lora.block_modules]
            if names is not None:
                mods = [m for m in mods if m in names]
            blocks.append(Block(f"block_{i}", f"Block {i}", mods))
        return [BlockGroup("Blocks", blocks)]

    def lora_target_names(self, dit) -> list:
        """Dotted module names (relative to dit) of the Linears (or Conv2d) a LoRA wraps: every module in the block
        map."""
        return [m for g in self.block_map(dit) for b in g.blocks for m in b.modules]

    def block_of(self, module_name: str) -> Optional[str]:
        """The block id a module belongs to, or None for modules outside the map (e.g. a speed LoRA's extras)."""
        idx = getattr(self, "_block_index", None)
        if idx is None:
            idx = self._block_index = {m: b.id for g in self.block_map() for b in g.blocks for m in b.modules}
        return idx.get(module_name)
