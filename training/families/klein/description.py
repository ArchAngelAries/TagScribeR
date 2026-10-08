# Ported from Fizgig (https://github.com/shootthesound/Fizgig): the facts of src/fizgig/klein/ (model_utils.py,
# embedder.py), training/trainer.py, training/metadata.py, networks/lora_klein.py, lora_trainer_gui.py (BUILT_IN_PRESETS
# 857-907, PRESETS 1877-1898, ARCHITECTURES["Flux 2 Klein Base 9B"] 322-345, Model Area patterns 33372-33405),
# scripts/fetch_models.py (files and sizes) and docs/KLEIN.md (memory).
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: Fizgig has no FamilyDescription for Klein (it predates the family layer), so this one is
# assembled from those sources - every value cites its file; the presets keep Fizgig's names, keys and values verbatim
# (legacy keys such as the 0-1000 noise-range boxes are mapped onto this app's parameters by training/presets.py).
# Brought level with Fizgig 7.0.1 families/klein.py (commit 1c8ec88): the Identity preset at rank 8 under its new name,
# NETWORK_TYPE / FAMILY_PRECISION / BLOCKS_SWAP / FAMILY_EMA in every preset, LoKR, the EMA control (Off by default),
# the whole optimizer catalog.
"""FLUX.2 Klein Base 9B: the 9B double / single-stream DiT, trained on the Base (undistilled) checkpoint."""
from training.description import FamilyDescription, LoRAFormat, ModelFile, SamplingSettings

_FETCH = "Fizgig scripts/fetch_models.py FAMILIES['klein']"
_GUI = "Fizgig lora_trainer_gui.py"


def _preset(rank, lr, epochs, lo, hi, area, min_t="", max_t=""):
    # Fizgig BUILT_IN_PRESETS (lora_trainer_gui.py:857-907), keys and values verbatim: LoRA alpha = rank, every epoch
    # saved, seed 42, adamw8bit, Adaptive LR on. MIN / MAX_TIMESTEP are Klein's 0-1000 boxes (presets.migrate_legacy
    # turns "400" into 0.4); blank = the full range.
    return {
        "NETWORK_DIM": rank, "NETWORK_ALPHA": rank, "LEARNING_RATE": lr,
        "MAX_TRAIN_EPOCHS": epochs, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": True, "ADAPTIVE_LR_MIN": lo, "ADAPTIVE_LR_MAX": hi,
        "TARGET_LAYERS": area, "MIN_TIMESTEP": min_t, "MAX_TIMESTEP": max_t,
        "OPTIMIZER_TYPE": "adamw8bit", "NETWORK_TYPE": "LoRA (standard)",
        "FAMILY_PRECISION": "Auto (fits your free VRAM)", "BLOCKS_SWAP": "Auto (detect from GPU)",
        "FAMILY_EMA": "Off",
    }


KLEIN_9B = FamilyDescription(
    key="klein9b",
    arch_id="klein9b",                                       # Fizgig training/metadata.py ARCHITECTURE_KLEIN_9B
    display_name="FLUX.2 Klein Base 9B",
    gui_label="FLUX.2 Klein Base 9B",                        # Fizgig ARCHITECTURES key "Flux 2 Klein Base 9B"
    lora_name_suffix="k9b",                                  # Fizgig settings LORA_NAME default LoraName_TokenName_k9b
    aliases=("flux2_klein", "flux-2-klein", "klein", "klein-base-9b", "flux 2 klein base 9b"),
    experimental=True,          # until the owner's first real run confirms the port

    model_files=(
        ModelFile("klein_dit", "Klein Base 9B DiT", True, "black-forest-labs/FLUX.2-klein-base-9b-fp8",
                  "flux-2-klein-base-9b-fp8.safetensors", 9.5,
                  "The undistilled Base model - what training runs on. Fizgig trains on the fp8 file (~9.5 GB, "
                  "fetch_models.py); the bf16 file (~17 GB) works too. Gated repos: accept Black Forest Labs' "
                  "licence on Hugging Face first. Not the Distilled model.",
                  role="dit"),
        ModelFile("klein_vae", "FLUX.2 AE (ae.safetensors)", True, "black-forest-labs/FLUX.2-dev", "ae.safetensors",
                  0.32,
                  "The AE from the ROOT of the FLUX.2-dev repo (a gated repo) - not the diffusers-format file in its "
                  "vae/ subfolder.", role="vae"),
        ModelFile("klein_text_encoder", "Qwen3-8B text encoder", True, "Comfy-Org/vae-text-encorder-for-flux-klein-9b",
                  "split_files/text_encoders/qwen_3_8b.safetensors", 15.0,
                  "Used for caching and preview prompts only, then unloaded. The tokenizer files come from the "
                  "Hugging Face cache, or a qwen3_tokenizer/ folder next to this file.", role="text_encoder"),
    ),
    text_encoder_label="Qwen3-8B",
    vae_label="FLUX.2 AE",

    latent_channels=128,                                     # model.py Flux2Params.in_channels: 32 AE channels x 2x2
    spatial_factor=16,                                       # dataset/image_dataset.py:129 LATENT_SPATIAL_FACTOR klein9b
    bucket_step=16,                                          # image_dataset.py RESOLUTION_STEPS = 16
    image_channels=3,
    native_megapixels=1.0,                                   # Klein's default preview canvas is 768 (below); BFL native 1 MP

    n_blocks=32,
    block_prefix="double_blocks",
    block_note="8 double-stream blocks (separate image / text streams, attended jointly) and 24 single-stream blocks. "
               "The LoRA covers the Linears of those 32 blocks - 8 x 8 + 24 x 2 = 112 modules; the modulation, "
               "embedder and output layers are not trained (Fizgig networks/lora_klein.py).",

    lora=LoRAFormat(
        key_template="lora_unet_double_blocks_{block}_{module}.{ab}.weight",    # single blocks: lora_unet_single_blocks_N_
        down="lora_down", up="lora_up",
        block_modules=("img_attn.qkv", "img_attn.proj", "img_mlp.0", "img_mlp.2",
                       "txt_attn.qkv", "txt_attn.proj", "txt_mlp.0", "txt_mlp.2"),     # per DOUBLE block
        alpha_key="{prefix}.alpha",
        kohya=True,
        file_prefix="lora_unet_",
        note="kohya keys, as Fizgig writes them: lora_unet_<module path with dots as underscores>.lora_down.weight / "
             ".lora_up.weight / .alpha, e.g. lora_unet_double_blocks_0_img_attn_qkv and lora_unet_single_blocks_3_"
             "linear1.",
        source="Fizgig networks/lora_klein.py (target DoubleStreamBlock / SingleStreamBlock, prefix lora_unet; "
               "modulation and norm Linears excluded) + networks/lora.py LoRANetwork (lora_name = prefix.path, dots "
               "-> underscores)",
    ),

    driver="training.families.klein.driver:KleinDriver",
    modelspec_arch="Flux.2-klein-9b",                        # Fizgig training/metadata.py ARCH_KLEIN_9B
    ema_default="Off",                                       # Fizgig families/klein.py: available, off until an A/B
    implementation="https://github.com/black-forest-labs/flux2",     # Fizgig training/metadata.py IMPL_KLEIN
    precisions=("bf16", "fp8", "int8", "nf4"),
    # Fizgig 7.0.1 (families/klein.py): Auto picks INT8, then NF4, and never plans a block swap (no measured swap
    # saving). When the run is not compiled and the DiT file is fp8, the file trains as it is (KleinDriver.
    # auto_uncompiled_precision): uncompiled that is faster than INT8. Fizgig calls that choice "As the file (bf16 or
    # fp8)" under bf16; TagScribeR names it fp8 and keeps it selectable.
    auto_precisions=("int8", "nf4"),
    adaptive_lr_clip_signal=True,                            # Fizgig families/klein.py: Klein's grad-clip rule
    # Fizgig measured 3 Oct 2026 on a 5090, full model, rank 32, adamw8bit, gradient checkpointing, BFL's fp8 base file,
    # peak reserved over the first epoch: INT8 12.6 GB at 0.25 MP / 17.1 GB at 1 MP; NF4 7.9 / 9.9 GB; the fp8 file as it
    # is 11.1 GB at 0.25 MP (families/klein.py compile notes). Swapped-block savings not measured (0 = Auto plans no
    # swap). TagScribeR's fp8 entry: the 0.25 MP point MEASURED, its 1 MP point INFERRED by carrying INT8's +4.5 GB, and
    # 0.41 GB per unit of "blocks to swap" COMPUTED from the parameter counts (6 double + 18 single blocks at the
    # maximum of 16 = 6.5B weights / 16 at one byte per weight), so a manual fp8 choice with Auto swap still swaps on a
    # small card. bf16 has no entry (never measured; ~17.4 GB of weights), so Auto never picks it.
    train_memory={"int8": (((0.25, 12.6), (1.0, 17.1)), 0.0), "nf4": (((0.25, 7.9), (1.0, 9.9)), 0.0),
                  "fp8": (((0.25, 11.1), (1.0, 15.6)), 0.41)},
    # Fizgig 7.0.1's Train tab offers its whole optimizer catalog for every family (optimizers.available_optimizers);
    # Automagic v3 runs Klein's LoRA as one group (Klein declares no optimizer_families)
    optimizers=("adamw8bit", "adamw", "pagedadamw8bit", "ademamix8bit", "pagedademamix8bit", "lion8bit", "automagic3"),
    network_types=("lora", "lokr"),                          # Fizgig families/klein.py
    # the Train tab shows these only for this family; they reach KleinDriver.configure (params.DRIVER_OPTIONS)
    family_options=("TARGET_LAYERS", "TRAINING_BLOCKS", "TIMESTEP_SAMPLING", "DISCRETE_FLOW_SHIFT", "SIGMOID_SCALE",
                    "LOGIT_MEAN", "LOGIT_STD", "PRESERVE_DISTRIBUTION", "ATTENTION_MECHANISM"),

    sampling=(
        SamplingSettings("Base", steps=40, cfg=4.5, sampler="euler", scheduler="simple", negative_prompt=True,
                         note="Euler over the empirical-mu shifted schedule (get_schedule), classifier-free guidance "
                              "with a negative prompt (two passes per step).",
                         source="Fizgig ARCHITECTURES sample_steps_default 40, sample_cfg_default 4.5 "
                                "(lora_trainer_gui.py:322-345) + klein/model_utils.py get_schedule / denoise_cfg"),
    ),
    # Fizgig previews on the Distilled model (4 steps, a second ~9 GB model) when its file is given - NOT PORTED: the
    # generic layer previews on the resident training model, so this is Fizgig's no-Distilled fallback (Base, 40 steps,
    # CFG 4.5).
    preview_steps=40,
    preview_cfg=4.5,
    preview_width=768,                                       # ARCHITECTURES sample_width_default 768
    preview_height=768,

    presets=(
        # Fizgig BUILT_IN_PRESETS order and names, verbatim. The first is the first-visit default.
        ("✨ Old Reliable (rank 16, full model, single subject)", _preset(16, 1e-4, 55, "1e-4", "4e-4", "Full Model")),
        ("✨ Old Reliable - Flavour 8 (rank 8, full model, single subject)",
         _preset(8, 1e-4, 55, "1e-4", "4e-4", "Full Model")),
        ("✨ Identity (rank 8, single subject)", _preset(8, 4e-4, 15, "2e-4", "4e-4", "Identity")),   # Fizgig 6.8.2
        ("✨ Identity (rank 8, harder dataset)", _preset(8, 4e-4, 20, "2e-4", "4e-4", "Identity")),
        ("✨ Multi-Character (rank 16, multi character or concept)",
         _preset(16, 2e-4, 50, "1e-4", "4e-4", "Identity")),
        ("✨ Style (late timesteps)", _preset(4, 4e-4, 15, "1e-5", "4e-4", "Style", "0", "400")),
        ("✨ Style+Composition (all timesteps)", _preset(4, 4e-4, 15, "1e-5", "4e-4", "Style+Composition")),
    ),

    # Fizgig fetch_models.py: ("hf-config:Qwen/Qwen3-8B", "Qwen3 tokenizer - Klein offline")
    helper_files=(("Qwen/Qwen3-8B", ("generation_config.json", "merges.txt", "tokenizer.json", "tokenizer_config.json",
                                     "vocab.json")),),

    notes=(
        ("Train on the Base checkpoint: the fp8 file as Fizgig does, or bf16; INT8 and 4-bit bases are made "
         "from it. The AE must be ae.safetensors from the root of the FLUX.2-dev repo.", f"{_FETCH}, docs/KLEIN.md"),
        ("Text conditioning is Qwen3-8B hidden states of layers 9, 18 and 27 (each through the final RMSNorm), "
         "concatenated: 512 tokens x 12288, no mask. A cached caption is about 12.6 MB.",
         "Fizgig klein/embedder.py Qwen3Embedder"),
        ("Training timesteps: flux2_shift by default - t = sigmoid(randn x scale), shifted by e^mu with mu from the "
         "packed latent's token count (256 -> 0.5, 4096 -> 1.15). Every mode except sigma tells the DiT t + 0.001.",
         "Fizgig training/trainer.py:791-960"),
        ("Model area: Identity = single blocks 1-16; Style and Style+Composition = all 8 double blocks + single 0-1; "
         "Details = single blocks 12-23; Custom = the ticked blocks; Style also trains only timesteps 0-0.4.",
         f"{_GUI} 33372-33405, 18805-18822"),
        ("Attention mechanism: sdpa (default) is PyTorch SDPA. flash3 is offered because Fizgig's dropdown offers it, but "
         "Fizgig's attention dispatcher has no flash3 branch, so choosing it stops the first step with 'Unsupported "
         "attention mode: flash3' (here too).", f"{_GUI} 10781, Fizgig training/trainer.py 1962, modules/attention.py "
         "dispatch"),
        ("Not part of this port yet: sliders, the full fine-tune, the workbench tools. Not in current Fizgig either: "
         "accelerate, TensorBoard / wandb, LoRA dropout and LoRA+, the SD3 loss weightings.",
         "docs/dev/PORT_PLAN.md"),
    ),
)
