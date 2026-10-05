# Ported from Fizgig (https://github.com/shootthesound/Fizgig): the facts of src/fizgig/krea2/ (utils.py, trainer.py,
# embedder.py, sampling.py), lora_trainer_gui.py (KREA2_BUILT_IN_PRESETS, the Krea 2 model paths),
# scripts/fetch_models.py (files and sizes) and utils/capabilities.py (recommend_krea2_strategy's measured memory).
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: Fizgig has no FamilyDescription for Krea 2 (it predates the family layer), so this one is
# assembled from those sources - every value cites its file; the presets keep Fizgig's keys and values verbatim (legacy
# keys such as KREA2_EMA / QUANT_4BIT_MODE are mapped onto this app's parameters by training/presets.py).
"""Krea 2: the 12.9B single-stream MMDiT, trained on the RAW (undistilled) checkpoint."""
from training.description import (
    FamilyDescription, LoRAFormat, ModelFile, SamplingSettings, SpeedLoRA,
)

_REPO = "Comfy-Org/Krea-2"
_FETCH = "Fizgig scripts/fetch_models.py FAMILIES['krea2']"
_GUI = "Fizgig lora_trainer_gui.py"


def _preset(rank, epochs, adaptive, lo, hi):
    # Fizgig KREA2_BUILT_IN_PRESETS (lora_trainer_gui.py:913-1005), keys and values verbatim. Every Krea 2 preset:
    # LoRA, lr 1e-4 (ignored while Adaptive LR is on), adamw8bit, 0.25 MP, loss watch + per-image LR on, EMA 0.98.
    return {
        "NETWORK_DIM": rank, "NETWORK_ALPHA": rank, "NETWORK_TYPE": "LoRA (standard)",
        "LEARNING_RATE": 1e-4,
        "MAX_TRAIN_EPOCHS": epochs, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": adaptive, "ADAPTIVE_LR_MIN": lo, "ADAPTIVE_LR_MAX": hi,
        "TARGET_LAYERS": "Full Model", "MIN_TIMESTEP": "", "MAX_TIMESTEP": "",
        "OPTIMIZER_TYPE": "adamw8bit",
        "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0,
        "DATASET_MEGAPIXELS": "0.25",
        "BLOCKS_SWAP": "Auto (detect from GPU)",
        "QUANT_4BIT_MODE": "auto", "COMPILE_BLOCKS": "Auto",
        "KREA2_LOSS_WATCH": True, "KREA2_PER_IMAGE_LR": True,
        "KREA2_AUTO_RECAPTION": False, "KREA2_WARMUP_LOOK": False,
        "KREA2_EMA": "0.98 (recommended)",                 # lora_trainer_gui.py:1004-1005
    }


KREA2 = FamilyDescription(
    key="krea2",
    arch_id="krea2",                                         # Fizgig training/metadata.py ARCHITECTURE_KREA2
    display_name="Krea 2",
    gui_label="Krea 2",
    lora_name_suffix="krea2",
    aliases=("krea-2", "krea_2"),
    experimental=True,          # until the owner's first real run confirms the port

    model_files=(
        ModelFile("krea2_dit", "Krea 2 RAW DiT", True, _REPO, "diffusion_models/krea2_raw_bf16.safetensors", 26.0,
                  "The undistilled 12.9B RAW base - what training runs on. Either the bf16 file (~26 GB) or an "
                  "fp8 / fp8-scaled RAW file (~13 GB) works: an fp8 file stays fp8, and INT8 / 4-bit bases are made "
                  "from whichever you pick. Not the Turbo checkpoint.",
                  role="dit"),
        ModelFile("krea2_vae", "Qwen-Image VAE", True, _REPO, "vae/qwen_image_vae.safetensors", 0.25,
                  "The Qwen-Image VAE (16 latent channels, 8x) used by Krea 2.", role="vae"),
        ModelFile("krea2_text_encoder", "Qwen3-VL-4B text encoder", True, _REPO,
                  "text_encoders/qwen3vl_4b_fp8_scaled.safetensors", 5.2,
                  "Used for caching and preview prompts only, then unloaded. Either the fp8_scaled file (what Fizgig "
                  "downloads) or text_encoders/qwen3vl_4b_bf16.safetensors (8.9 GB) works. The tokenizer files "
                  "come from the Hugging Face cache, or a qwen3vl_tokenizer/ folder next to this file.",
                  role="text_encoder"),
        ModelFile("krea2_turbo_lora", "Krea 2 Turbo LoRA (previews)", False, _REPO,
                  "loras/krea2_turbo_lora_rank_64_bf16.safetensors", 0.47,
                  "Optional: RAW + this LoRA at 1.0 samples like the Turbo model (8 steps, no CFG), so previews "
                  "run on the training model itself. Frozen, off during training.", role="speed_lora"),
    ),
    text_encoder_label="Qwen3-VL-4B",
    vae_label="Qwen-Image VAE",

    latent_channels=16,                                      # utils.py single_mmdit_large_wide.channels
    spatial_factor=8,                                        # vae_loader.py temperal_downsample [F, T, T] -> 2**3 = 8
    bucket_step=16,                                          # 8 x patch 2 (dataset/image_dataset.py RESOLUTION_STEPS 16)
    image_channels=3,
    native_megapixels=1.0,                                   # sampling.sample defaults 1024x1024 (max 1280 px)

    n_blocks=28,                                             # utils.py single_mmdit_large_wide.layers
    block_prefix="blocks",
    block_note="28 identical single-stream blocks (attention + SwiGLU), plus a small text-fusion stack (2 layerwise + "
               "2 refiner blocks and a layer projector) and the input / output layers. The LoRA covers every Linear "
               "in the model - 264 modules.",

    lora=LoRAFormat(
        key_template="lora_unet_blocks_{block}_{module}.{ab}.weight",
        down="lora_down", up="lora_up",
        block_modules=("attn.wq", "attn.wk", "attn.wv", "attn.gate", "attn.wo", "mlp.gate", "mlp.up", "mlp.down"),
        alpha_key="{prefix}.alpha",
        kohya=True,
        file_prefix="lora_unet_",
        note="kohya keys, as Fizgig writes them: lora_unet_<module path with dots as underscores>.lora_down.weight / "
             ".lora_up.weight / .alpha. A LoKR adapter is saved LyCORIS-style: diffusion_model.<dotted path>."
             "lokr_w1 / .lokr_w2 / .alpha (the format ComfyUI LoKRs use).",
        source="Fizgig networks/lora.py create_network (prefix 'lora_unet', every Linear of the DiT) + "
               "krea2/trainer.py _save_lora (comfy_format for LoKR)",
    ),

    driver="training.families.krea2.driver:Krea2Driver",
    modelspec_arch="Krea-2",
    ema_default="0.98",                                      # lora_trainer_gui.py:1004-1005 (Peter, 9 Sep 2026)
    implementation="https://github.com/krea-ai/krea-2",      # Fizgig training/metadata.py IMPL_KREA2
    precisions=("bf16", "fp8", "int8", "nf4"),
    # Fizgig's Auto ladder for Krea 2 (utils/capabilities.py recommend_krea2_strategy): INT8 with no swap, then NF4
    # with no swap, then fp8 with no swap, then fp8 with as few swapped blocks as fit. INT8 and NF4 are skipped on a
    # machine that cannot run them (no torch._int_mm / no bitsandbytes), which leaves fp8.
    auto_order=("int8", "nf4", "fp8"),
    auto_swap_order=("fp8",),
    # Measured by Fizgig (utils/capabilities.py: _INT8_PEAK_GB, _NF4_PEAK_GB, _RES_GB_PER_MP, _SWAP_GB_PER_BLOCK,
    # _MAX_SWAP_KREA2), 5090, 28 Jul 2026, gradient checkpointing, batch 1, rank 32, training-only peaks at 0.25 MP:
    #   INT8 16.2 GB, NF4 11.4 GB fp8 18.7 GB. Resolution: +0.15 GB from 0.25 to 1.05 MP
    #   measured, budgeted at 0.25 GB/MP -> (0.25, base), (1.0, base + 0.2). Batch: +2.4 GB per extra image (not
    #   modelled by the Auto plan: use batch 1 on a tight card). Rank: +0.015 GB per rank above 32.
    # Swap: 0.42 GB saved per swapped block, MEASURED WITH FP8 weights (18.7 - 0.42 * swap); INT8 stores the same one
    #   byte per weight, so the same figure is used for it (an inference, not a measurement). Max swap 26 (28 blocks,
    #   2 stay resident). NF4 cannot swap.
    # bf16 has NO entry: Fizgig never measured it (12.9B x 2 bytes = ~26 GB of weights alone), so Auto never picks
    #   it; it stays selectable for cards with 32 GB or for a manual block swap (about 0.8 GB per block, unmeasured).
    # fp8: 18.7 GB MEASURED (_FP8_PEAK_GB), 0.42 GB per swapped block MEASURED on fp8 (18.7 - 0.42 * swap).
    train_memory={"fp8": (((0.25, 18.7), (1.0, 18.9)), 0.42), "int8": (((0.25, 16.2), (1.0, 16.4)), 0.42),
                  "nf4": (((0.25, 11.4), (1.0, 11.6)), 0.0)},
    # Fizgig's Krea 2 list is its whole optimizer catalogue (optimizers.available_optimizers). Automagic v3 is left
    # out: its per-family parameter groups (family_param_groups) and sign window 16 are not ported, and a single
    # group is the compromise rate Fizgig's own comment warns against.
    optimizers=("adamw8bit", "adamw", "pagedadamw8bit", "ademamix8bit", "pagedademamix8bit", "lion8bit"),
    network_types=("lora", "lokr"),                          # _GUI: Network Type wired for krea2_train

    sampling=(
        SamplingSettings("RAW (undistilled)", steps=28, cfg=5.5, sampler="euler", scheduler="simple",
                         negative_prompt=True,
                         note="Euler on the flow-matching schedule shifted by exp(mu), mu from the image-token count "
                              "(0.5 at 256 tokens to 1.15 at 6400).",
                         source="Fizgig krea2/sampling.py sample() defaults (steps 28, cfg 5.5) + timesteps()"),
    ),
    speed_loras=(
        SpeedLoRA(
            name="Krea 2 Turbo (rank 64 LoRA)",
            repo=_REPO,
            file="loras/krea2_turbo_lora_rank_64_bf16.safetensors",
            pairs_with="Krea 2 RAW transformer",
            strength=1.0,
            settings=SamplingSettings("Turbo 8-step", steps=8, cfg=1.0, sampler="euler", scheduler="simple",
                                      options=(("mu", 1.15),),
                                      note="CFG-free; the distilled model was trained at a fixed mu = 1.15, so the "
                                           "schedule shift is pinned instead of following the resolution.",
                                      source="Fizgig krea2/trainer.py _render_prompt_set (mu=1.15) and "
                                             "sample_previews_on_dit (steps 8, cfg 1.0, turbo at 1.0)"),
            load_unmerged=True,
            pref_key="krea2_turbo_lora",
            caveats=("The file also carries diff_b bias deltas on the input / output layers (first, last, tmlp, tproj, "
                     "txtmlp) where much of the distillation plausibly lives; they are applied with the LoRA while a "
                     "preview renders and restored exactly afterwards.",),
            source=f"{_FETCH}; Fizgig krea2/trainer.py _apply_turbo_lora",
        ),
    ),
    # Previews run on the live RAW training model with the Turbo LoRA at 1.0, 8 steps, CFG 1 (Fizgig's default
    # "RAW + Turbo LoRA" engine, audit 8.3); without the file they render the RAW model at 28 steps / CFG 5.5.
    preview_speed_lora="Krea 2 Turbo (rank 64 LoRA)",
    preview_speed_steps=8,
    preview_steps=28,
    preview_cfg=5.5,
    preview_width=1024,                                      # audit 4.4: SAMPLE_WIDTH per-family default Krea 2 1024
    preview_height=1024,

    presets=(
        # THE DEFAULT (the first visit applies it): rank 8 with Adaptive LR at an aggressive floor. Rank 8 is more than
        # enough for a character on a 12.9B model and lands the right result more reliably than 32 (Peter, 16 Sep 2026).
        ("✨ Krea 2 Ultra Fast (rank 8, adaptive LR)", _preset(8, 30, True, "2e-4", "4e-4")),
        ("✨ Krea 2 Standard (rank 32, full model)", _preset(32, 64, False, "1e-4", "4e-4")),
        # Style: rank 16 (broad global direction; the one published datapoint is "rank 16 for simple styles"), LR
        # ceiling 2e-4 because the watcher probes up on steady descent and style overbakes at 4e-4 (the start is 1e-4,
        # the LR the Krea 2 ecosystem defaults to). Every epoch is saved: scrub for the sweet spot.
        ("✨ Krea 2 Style (rank 16, gentle LR)", _preset(16, 64, True, "5e-5", "2e-4")),
    ),

    # Fizgig's Krea 2 text-encoder tokenizer files (krea2/embedder.py QWEN3_VL_TOKENIZER_FILES), fetched with the models
    helper_files=(("Qwen/Qwen3-VL-4B-Instruct", ("chat_template.json", "generation_config.json", "merges.txt",
                                                  "preprocessor_config.json", "tokenizer.json",
                                                  "tokenizer_config.json", "video_preprocessor_config.json",
                                                  "vocab.json")),),

    notes=(
        ("Train on the RAW checkpoint. Previews use the same RAW model with the Turbo LoRA at strength 1.0 (8 steps, "
         "CFG 1), so no second 13 GB model is loaded.", "Fizgig krea2/trainer.py _apply_turbo_lora"),
        ("Text conditioning is the stack of 12 hidden-state layers (2, 5, ..., 35) of Qwen3-VL-4B under a fixed "
         "descriptor template, padded to 512 tokens, with a validity mask; the DiT folds the stack itself. A cached "
         "caption is about 30 MB.", "Fizgig krea2/embedder.py Qwen3VLConditioner, krea2/caching.py"),
        ("Training timesteps: logit-normal, shifted by exp(mu) with mu from the image-token count; a noise range "
         "(min / max timestep) rescales into the window instead of clamping.",
         "Fizgig krea2/trainer.py sample_krea2_timesteps"),
        ("Not part of this port: fp8 base, torch.compile, rotating-block full fine-tune and regularisation images, "
         "the fp8-Turbo-checkpoint preview engine, Automagic per-family groups, captioning with the encoder, RefMods.",
         "docs/TRAINING_PLAN.md"),
    ),
)
