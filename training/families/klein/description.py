# Ported from Fizgig (https://github.com/shootthesound/Fizgig): the facts of src/fizgig/klein/ (model_utils.py,
# embedder.py), training/trainer.py, training/metadata.py, networks/lora_klein.py, lora_trainer_gui.py (BUILT_IN_PRESETS
# 857-907, PRESETS 1877-1898, ARCHITECTURES["Flux 2 Klein Base 9B"] 322-345, Model Area patterns 33372-33405),
# scripts/fetch_models.py (files and sizes) and docs/KLEIN.md (memory).
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: Fizgig has no FamilyDescription for Klein (it predates the family layer), so this one is
# assembled from those sources - every value cites its file; the presets keep Fizgig's names, keys and values verbatim
# (legacy keys such as the 0-1000 noise-range boxes are mapped onto this app's parameters by training/presets.py).
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
        "OPTIMIZER_TYPE": "adamw8bit",
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
    ema_default="",                                          # Fizgig's Klein trainer has no EMA
    implementation="https://github.com/black-forest-labs/flux2",     # Fizgig training/metadata.py IMPL_KLEIN
    precisions=("bf16", "fp8", "int8", "nf4"),
    # Fizgig's Klein default is an fp8 base (FP8 + Scaled on, lora_trainer_gui.py PRESETS), with NF4 on small cards
    # (_klein_small_card: under 15 GiB) and block swap on fp8.
    auto_order=("fp8", "nf4"),
    auto_swap_order=("fp8",),
    # Fizgig measured Klein on an fp8 base (docs/KLEIN.md "VRAM"): the fp8 base stays resident at ~9.6 GB and "a 9B LoRA
    # fits 16 GB (~14 GB observed)"; the NF4 base is ~5.6 GB and "a full LoRA trains in about 8.5 GB at 0.5 MP" (10-12 GB
    # cards, no swap). Mapping:
    #   nf4  8.5 GB at 0.5 MP - MEASURED (Fizgig). Other resolutions are not measured: the figure is used as is.
    #   int8 14.0 GB at 0.5 MP - INFERRED, not measured: INT8 stores one byte per weight like the fp8 base, so the fp8
    #        measurement is used; the 0.5 MP point is Fizgig's NF4 one, the fp8 resolution is not stated.
    #   bf16 has NO entry: Fizgig never measured it (the 8.7B weights alone are ~17.4 GB), so Auto never picks it; it
    #        stays selectable for 24 GB+ cards or with a manual block swap.
    # Swap: GB saved per unit of "blocks to swap" is COMPUTED from the parameter counts (a double block is 436M weights,
    #   a single 218M; one unit at the maximum of 16 swaps 6 double + 18 single blocks = 6.5B weights / 16 = 0.41B
    #   weights = 0.41 GB at one byte per weight), never measured. NF4 cannot swap.
    # fp8: ~14 GB observed (Fizgig docs/KLEIN.md, "a 9B LoRA fits 16 GB") - the measurement the int8 row borrows.
    train_memory={"fp8": (((0.5, 14.0),), 0.41), "int8": (((0.5, 14.0),), 0.41), "nf4": (((0.5, 8.5),), 0.0)},
    # Klein's own list is adamw, adamw8bit and bitsandbytes AdEMAMix8bit / PagedAdEMAMix8bit (lora_trainer_gui.py:2236);
    # Automagic v3 is not offered for Klein (Fizgig: MiniMax H3 and Krea 2 only).
    optimizers=("adamw8bit", "adamw", "ademamix8bit", "pagedademamix8bit"),
    network_types=("lora",),                                 # Fizgig: LoKR is wired for Krea 2 / H3 / Qwen only
    # the Train tab shows these only for this family; they reach KleinDriver.configure (params.DRIVER_OPTIONS)
    family_options=("TARGET_LAYERS", "TRAINING_BLOCKS", "TIMESTEP_SAMPLING", "DISCRETE_FLOW_SHIFT", "SIGMOID_SCALE",
                    "LOGIT_MEAN", "LOGIT_STD", "PRESERVE_DISTRIBUTION"),

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
        ("✨ Identity (rank 4, single subject)", _preset(4, 4e-4, 15, "2e-4", "4e-4", "Identity")),
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
        ("Not part of this port: accelerate, the fp8 base, TensorBoard / wandb, LoRA dropout and LoRA+, "
         "regularisation, the Distilled preview model, edit / reference images, the SD3 loss weightings.",
         "docs/TRAINING_PLAN.md"),
    ),
)
