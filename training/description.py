# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/families/description.py
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: entry points are `python -m training.train` / `training.cache`; the Tkinter GUI's
# legacy ARCHITECTURES adapter (architecture_entry) is dropped; model file paths live in TagScribeR settings;
# `ModelFile.default_to` and `family_options` (SDXL: one checkpoint file holds the UNet, VAE and text encoders);
# `auto_swap_order` (a family's own Auto swap base).
"""FamilyDescription: everything the trainer needs to know about one model family, in one object.

A description holds the family's facts once (model files, latent rules, LoRA key format, presets, sampling
recipes); the generic cache / train / UI code reads it. Every value that came from outside carries its source
(`source=` fields), so a later reader can re-check it when the upstream model or a speed LoRA moves on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ModelFile:
    """One model file the user points TagScribeR at (Train tab > Model files)."""
    pref_key: str                     # settings key suffix, e.g. "qwen21_dit" (training.models.<key>)
    label: str                        # Settings row label
    required: bool = True             # training cannot start without it
    repo: str = ""                    # Hugging Face repo for the Download link
    path: str = ""                    # file path inside the repo
    size_gb: float = 0.0
    note: str = ""                    # one plain line shown under the row
    local_name: str = ""              # name in models/ when the repo's own is generic (diffusion_pytorch_model...)
    role: str = ""                    # "dit" | "vae" | "text_encoder" | "training_adapter" | "speed_lora" |
    #                                   "preview_dit" (a second checkpoint previews render on) | ""
    default_to: str = ""              # role whose file stands in when this row is left empty (SDXL: the VAE and the text
    #                                   encoders live inside the checkpoint, so an empty row means "use the checkpoint")


@dataclass(frozen=True)
class SamplingSettings:
    """A complete, citable way to sample the family: steps, CFG, sampler, schedule."""
    name: str
    steps: int
    cfg: float
    sampler: str = "euler"
    scheduler: str = "simple"
    sigmas: Optional[tuple] = None    # explicit schedule when the model needs one
    options: tuple = ()               # driver-specific sampler options as (name, value) pairs
    negative_prompt: bool = False     # whether a negative prompt does anything at this CFG
    note: str = ""
    source: str = ""


@dataclass(frozen=True)
class SpeedLoRA:
    """A Turbo / Lightning / distill LoRA for the family, with the settings it actually wants."""
    name: str
    repo: str
    file: str
    pairs_with: str                   # which base it was trained against
    strength: float
    settings: SamplingSettings
    load_unmerged: bool = False       # merging into the weights loses part of it
    pref_key: str = ""                # the model file (Settings row) holding it
    community_settings: tuple = ()    # (description, source) pairs: how people actually use it
    caveats: tuple = ()
    source: str = ""


@dataclass(frozen=True)
class LoRAFormat:
    """How this family's LoRA keys are written so ComfyUI (and diffusers) load every module."""
    key_template: str                 # e.g. "transformer.transformer_blocks.{block}.{module}.{ab}.weight"
    down: str                         # "lora_A" / "lora_down"
    up: str                           # "lora_B" / "lora_up"
    block_modules: tuple              # per-block Linears the LoRA targets
    alpha_key: str = "{prefix}.alpha"
    kohya: bool = False               # True = the lora_unet_ convention used by Klein/Krea 2/H3
    file_prefix: str = ""             # what precedes a module path in every key, e.g. "transformer."
    # a kohya family's LoKR keys on the lora_unet_ stems (SDXL: ComfyUI names its modules by those) instead of LyCORIS'
    # diffusion_model.<path> (Fizgig 7.0.1 LoRAFormat.lokr_kohya_stems)
    lokr_kohya_stems: bool = False
    note: str = ""
    source: str = ""

    def module_of(self, key: str) -> Optional[str]:
        """Dotted module path of a down-weight key ('transformer.modulation.1.lora_A.weight' -> 'modulation.1'),
        None for any other key."""
        tail = f".{self.down}.weight"
        if not (key.startswith(self.file_prefix) and key.endswith(tail)):
            return None
        return key[len(self.file_prefix):-len(tail)]

    def key(self, block: int, module: str, which: str) -> str:
        """which: "down" or "up"."""
        ab = self.down if which == "down" else self.up
        return self.key_template.format(block=block, module=module, ab=ab)


@dataclass(frozen=True)
class FamilyDescription:
    # identity
    key: str                          # family key (the workbench vocabulary), e.g. "qwen_image21"
    arch_id: str                      # architecture id used in cache filenames, e.g. "qwenimage21"
    display_name: str                 # "Qwen Image 2.1"
    gui_label: str                    # Base Model selector entry
    lora_name_suffix: str
    aliases: tuple = ()
    experimental: bool = True

    # model files (Settings rows, in display order)
    model_files: tuple = ()
    text_encoder_label: str = ""
    vae_label: str = ""

    # latent rules
    latent_channels: int = 16
    spatial_factor: int = 8
    bucket_step: int = 64             # training buckets snap to this many pixels
    image_channels: int = 3           # 4 = RGBA
    native_megapixels: float = 1.0

    # block layout (Repair Studio / block targeting)
    n_blocks: int = 0
    block_prefix: str = ""            # "transformer_blocks"
    block_note: str = ""

    # LoRA
    lora: Optional[LoRAFormat] = None

    # the family's driver ("module.path:ClassName", a FamilyDriver); empty = not built yet, so the family stays
    # hidden from training. Caching and training run through the generic entry points below for every family.
    driver: str = ""
    modelspec_arch: str = ""          # SAI modelspec.architecture, e.g. "Qwen-Image-2.1"
    training_adapter: str = ""        # pref key of the family's frozen training adapter ("" = none)
    training_adapter_note: str = ""   # one line for the Training tab under the adapter toggle
    # a training adapter picked by a choice parameter instead of the on / off tick (MiniMax H3's H3_ADAPTER, Fizgig
    # 7.0.1): the parameter key, and per choice (its first word) (pref key on the standard base, pref key on the
    # alternative base); a choice not listed (Off) trains without one
    adapter_choice: str = ""
    adapter_files: tuple = ()
    # an alternative training base picked by a choice parameter (MiniMax H3's H3_TRAIN_BASE, Fizgig --dit=pref:...):
    # (parameter key, the first word of the choice that picks it, the pref key of its file)
    alt_base: tuple = ()
    # the trainable adapter's dtype ("fp32", or "bf16" where the family's own trainer trained in bf16: MiniMax H3)
    trainable_dtype: str = "fp32"
    ema_default: str = ""             # default EMA decay for the Training tab ("0.98", "Off"); "" = no EMA control
    ema_short_run: bool = False       # the EMA choice also offers Short run (the decay sized to the run - H3's)
    implementation: str = ""          # SAI modelspec.implementation (reference repo URL)
    precisions: tuple = ("bf16",)     # base precisions offered for training: any of "bf16", "int8", "nf4"
    # measured training memory for the Auto plan: {precision: (peak GB with no block swap, GB saved per swapped
    # block)}; the peak may instead be ((megapixels, GB), ...) points, interpolated for the run's resolution.
    # {} = Auto just takes the first precision
    train_memory: dict = field(default_factory=dict)
    # TagScribeR: bumped when the way a family ENCODES captions changes, so text caches written the old way are
    # re-encoded once (the latents' counterpart is training/cache.py LATENT_REV). "" = never changed
    text_cache_rev: str = ""
    # Adaptive LR also treats a grad-clip ratio over 50% of an epoch's steps as a stability signal (Klein's rule,
    # training/adaptive_lr.AdaptiveLR clip_signal)
    adaptive_lr_clip_signal: bool = False
    # Fast Identity Mode: the block ids (driver.block_map) that carry a character's identity, measured. With
    # FAMILY_FAST_ID the LoRA trains these only (Qwen: blocks 10-14). () = no Fast Identity Mode
    identity_blocks: tuple = ()
    # the driver can torch.compile its blocks (FamilyDriver.compile_blocks); the Train tab's Compile Blocks control
    # shows (family_options COMPILE_BLOCKS). The generic Auto rule (FamilyDriver.compile_plan) for a family whose
    # driver does not bring its own: compile_payback_steps {precision: steps} - Auto compiles a run at least that long
    # (a precision not listed is not compiled by Auto; On still compiles). compile_boundary: where the gradient
    # checkpoint sits ("inside" the compiled graph - faster, more memory - or "outside" - eager-level memory);
    # compile_fullgraph: refuse graph breaks; compile_memory {precision: {boundary: ((mp, GB), ...)}}: measured
    # compiled peaks, checked against free VRAM ({} = not measured: the boundary is kept)
    compiles: bool = False
    compile_payback_steps: dict = field(default_factory=dict)
    compile_boundary: str = "inside"
    compile_fullgraph: bool = True
    compile_memory: dict = field(default_factory=dict)
    # optimizer settings a family's own trainer applied (only when Optimizer Args doesn't set them): an Adam-family
    # weight decay, and the 8-bit Adam eps floor of 1e-6 (training/optimizers.create_optimizer eps_floor_8bit)
    optimizer_weight_decay: Optional[float] = None
    optimizer_eps_floor_8bit: bool = False
    # what Auto may choose, in order (() = every offered precision, most precise first). Krea 2: INT8, then NF4 - its
    # original trainer's order; bf16 (and TagScribeR's fp8) stay manual choices
    auto_precisions: tuple = ()
    # TagScribeR: the precisions Auto may block-swap when nothing fits whole (() = Fizgig's rule: int8, then bf16, of
    # those Auto may choose)
    auto_swap_order: tuple = ()
    optimizers: tuple = ("adamw8bit", "adamw")
    network_types: tuple = ("lora",)
    # parameter keys (training/params.py DRIVER_OPTIONS) of the optional family extensions this family offers; the
    # Train tab shows them only here and the run passes them to driver.configure()
    family_options: tuple = ()
    edit_training: bool = False       # Edit LoRA from before/after pairs (the driver's supports_references)
    edit_note: str = ""               # the Edit LoRA section's "What you need" line: pair count and photo size
    # Slider LoRAs (Fizgig 7.0.1): the LoRA's strength is a dial between two looks, trained from image pairs (the
    # driver's training_loss diff_ref) or from prompts (noise_latents / predict). slider_guidance: a prompt slider's
    # default push strength; slider_ultra_blocks: the block ids an "Ultra mode" slider trains (() = no Ultra mode)
    slider_training: bool = False
    slider_guidance: float = 2.0
    slider_ultra_blocks: tuple = ()

    # sampling
    sampling: tuple = ()              # SamplingSettings without any speed LoRA (first = default)
    speed_loras: tuple = ()           # SpeedLoRA entries
    preview_steps: int = 20
    preview_cfg: float = 1.0
    preview_width: int = 1024
    preview_height: int = 1024
    preview_negative: str = ""        # the family's default preview negative prompt ("" = the app default); tag-
    #                                   trained models want tag-style negatives
    preview_speed_lora: str = ""      # name of the SpeedLoRA previews use when its file is set in Settings
    # previews may render on a second checkpoint (the model file with role "preview_dit", Klein's Distilled) with its
    # own recipe, the training model parked meanwhile (Fizgig 7.0.1 train_preview_checkpoint)
    train_preview_checkpoint: bool = False
    preview_checkpoint_sampling: Optional[SamplingSettings] = None
    preview_speed_steps: int = 0      # preview steps with it (0 = the SpeedLoRA's own)
    # the Samples tab sets the speed LoRA's steps in a box of their own (FAMILY_TURBO_STEPS; MiniMax H3's "N steps
    # at M%" row) and the Steps box stays the plain-model count
    samples_turbo_pace: bool = False
    preview_speed_strength: Optional[float] = None   # preview strength for it (None = the SpeedLoRA's own; 0 = off
    # by default: previews render without it until the Samples tab's Turbo strength is raised)
    # (steps, strength) preview defaults an older release shipped; the Samples tab replaces them with the current
    # ones, so a setting saved under the old default moves on instead of sticking
    retired_preview_defaults: tuple = ()
    # a one-time reset: when this tag is new to a user, the Samples tab replaces their saved preview steps and turbo
    # strength with the current defaults once (the tag is remembered, so later choices stick). Change it to reset again.
    preview_reset: str = ""

    # built-in Training-tab presets: ((name, {GUI setting key: value}), ...); the first is applied on a first visit
    presets: tuple = ()

    # small files the text encoder / captioner load by repo name: ((repo, (allow_patterns...)), ...); the model
    # downloader fetches them with the helper models so first use works offline
    helper_files: tuple = ()

    # workbench tools that support this family ("repair", ...); the generic WorkbenchEngine drives them all
    workbench: tuple = ()

    # things a user or a later session must know, with sources
    notes: tuple = ()

    # ---- derived ------------------------------------------------------------------------------
    # generic entry points shared by every described family (the standard layer)
    train_module = "training.train"
    cache_module = "training.cache"

    @property
    def training_ready(self) -> bool:
        return bool(self.driver)

    def load_driver(self):
        """Instantiate the family's FamilyDriver (imported lazily: the description stays importable without torch)."""
        import importlib
        mod, _, cls = self.driver.partition(":")
        drv = getattr(importlib.import_module(mod), cls)()
        drv.description = self
        return drv

    def lora_prefix(self, block: int, module: str) -> str:
        """Key stem of one wrapped module, e.g. 'transformer.transformer_blocks.3.attn.to_q'."""
        return self.lora.key_template.split(".{ab}")[0].format(block=block, module=module)

    @property
    def pref_keys(self) -> tuple:
        return tuple(f.pref_key for f in self.model_files)

    @property
    def required_pref_keys(self) -> tuple:
        return tuple(f.pref_key for f in self.model_files if f.required)

    def pref_for(self, role: str) -> str:
        """Pref key of the model file with this role ("" if the family has none)."""
        return next((f.pref_key for f in self.model_files if f.role == role), "")

    def block_ids(self) -> list:
        return [f"{self.block_prefix}_{i}" for i in range(self.n_blocks)]

    def preview_checkpoint(self):
        """(ModelFile, SamplingSettings) of the family's preview checkpoint, or None."""
        f = next((m for m in self.model_files if m.role == "preview_dit"), None)
        return (f, self.preview_checkpoint_sampling) if f is not None and self.preview_checkpoint_sampling else None

    def preview_speed(self):
        """The SpeedLoRA used for in-training previews, or None."""
        return next((sl for sl in self.speed_loras if sl.name == self.preview_speed_lora), None)

    def preview_speed_defaults(self):
        """(steps, strength) previews use with the preview SpeedLoRA, or None without one."""
        sp = self.preview_speed()
        if sp is None:
            return None
        return (self.preview_speed_steps or sp.settings.steps,
                sp.strength if self.preview_speed_strength is None else self.preview_speed_strength)

    def default_sampling(self) -> Optional[SamplingSettings]:
        return self.sampling[0] if self.sampling else None

    def validate(self) -> list:
        """Internal consistency problems (empty list = fine). Cheap; run by the registry and tests."""
        problems = []
        if not (self.key and self.arch_id and self.display_name and self.gui_label):
            problems.append("identity fields must all be set")
        keys = [f.pref_key for f in self.model_files]
        if len(keys) != len(set(keys)):
            problems.append("duplicate pref keys")
        if self.n_blocks <= 0 or not self.block_prefix:
            problems.append("block layout missing")
        if self.lora is None:
            problems.append("LoRA format missing")
        if not self.sampling:
            problems.append("no sampling settings")
        if self.bucket_step % self.spatial_factor:
            problems.append("bucket_step must be a multiple of spatial_factor")
        for sl in self.speed_loras:
            if not (sl.repo and sl.file and sl.source):
                problems.append(f"speed LoRA {sl.name!r} is missing repo/file/source")
        if self.driver and ":" not in self.driver:
            problems.append("driver must be 'module.path:ClassName'")
        if self.driver and not all(self.pref_for(r) for r in ("dit", "vae", "text_encoder")):
            problems.append("a trainable family needs model files with roles dit, vae and text_encoder")
        roles = {f.role for f in self.model_files}
        for f in self.model_files:
            if f.default_to and (f.default_to not in roles or f.default_to == f.role):
                problems.append(f"{f.pref_key}: default_to must name another row's role")
            if f.default_to and f.required:
                problems.append(f"{f.pref_key}: a row that defaults to another file cannot be required")
        if self.training_adapter and self.pref_for("training_adapter") != self.training_adapter:
            problems.append("training_adapter must name the model file whose role is training_adapter")
        named = [k for _w, a, b in self.adapter_files for k in (a, b)] + list(self.alt_base[2:3])
        if any(k not in keys for k in named):
            problems.append("adapter_files and alt_base must name model files")
        if self.trainable_dtype not in ("fp32", "bf16"):
            problems.append("trainable_dtype must be fp32 or bf16")
        if self.driver and not (self.modelspec_arch and self.implementation):
            problems.append("a trainable family needs modelspec_arch and implementation for LoRA metadata")
        return problems
