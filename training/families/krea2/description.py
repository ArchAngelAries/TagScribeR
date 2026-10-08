# Ported from Fizgig (https://github.com/shootthesound/Fizgig): the facts of src/fizgig/krea2/ (utils.py, trainer.py,
# embedder.py, sampling.py), lora_trainer_gui.py (KREA2_BUILT_IN_PRESETS, the Krea 2 model paths),
# scripts/fetch_models.py (files and sizes) and utils/capabilities.py (recommend_krea2_strategy's measured memory); since
# Fizgig 7.0.1 (commit 1c8ec88) also families/krea2.py (Auto order and memory figures).
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
    # Fizgig's Auto for Krea 2 (families/krea2.py auto_precisions, v7.0.1): INT8 with no swap, then NF4 with no swap,
    # then INT8 with as few swapped blocks as fit, then NF4 as the smallest base. bf16 and fp8 are manual choices (fp8 is
    # TagScribeR's: Fizgig dropped it as a Krea 2 base in 6.8.0). INT8 and NF4 are skipped on a machine that cannot run
    # them (no torch._int_mm / no bitsandbytes); with neither, Auto falls back to fp8 (training/quant.py plan).
    auto_precisions=("int8", "nf4"),
    # Fizgig's figures (families/krea2.py train_memory, v7.0.1): the original trainer's measured 0.25 MP peaks
    # (utils/capabilities.py: 5090, batch 1, rank 32; 0.42 GB saved per swapped INT8 block) and the driver's measured
    # growth to 1 MP on full-size photos (5090, rank 8, previews off): INT8 +2.9 GB, NF4 13.4 GB at 1 MP. "The
    # original's +0.25 GB/MP under-plans 1 MP: INT8 + 16 swapped blocks ran out on a 12 GB card." bf16 measured on the
    # driver, rank 8, 0.25 MP, 26.0 GB; its 1 MP point carries INT8's growth. Max swap 26 (28 blocks, 2 stay resident);
    # NF4 cannot swap. Batch: +2.4 GB per extra image (not modelled by the Auto plan: use batch 1 on a tight card).
    # fp8 (TagScribeR only): 18.7 GB at 0.25 MP and 0.42 GB per swapped block MEASURED by the original trainer
    #   (_FP8_PEAK_GB); its 1 MP point is INFERRED the way Fizgig infers bf16's, by carrying INT8's +2.9 GB.
    # Fizgig 7.0.1 caches one caption per forward (krea2/driver.py encode_text); caches written in batches of 8 are
    # re-encoded once
    text_cache_rev="1",
    train_memory={"int8": (((0.25, 16.2), (1.0, 19.1)), 0.42), "nf4": (((0.25, 11.4), (1.0, 13.4)), 0.0),
                  "bf16": (((0.25, 26.0), (1.0, 28.9)), 0.84), "fp8": (((0.25, 18.7), (1.0, 21.6)), 0.42)},
    # Fizgig's Krea 2 list is its whole optimizer catalogue (optimizers.available_optimizers). Automagic v3 runs as
    # Fizgig runs it (krea2/trainer.py:2559-2594): the LoRA split into txtfusion / attn / mlp / io groups that each
    # vote their own rate, sign window 16, the LR box is only its start rate, and Adaptive LR, the LR scheduler, the
    # per-image LR and the look warm-up stand down (Krea2Driver.optimizer_params, training/train.py).
    optimizers=("adamw8bit", "adamw", "pagedadamw8bit", "ademamix8bit", "pagedademamix8bit", "lion8bit", "automagic3"),
    compiles=True,                                           # Krea 2's own rule: Krea2Driver.compile_plan
    family_options=("COMPILE_BLOCKS",),                      # torch.compile of the blocks (Auto / On / Off / Outside)
    network_types=("lora", "lokr"),                          # _GUI: Network Type wired for krea2_train

    sampling=(
        SamplingSettings("RAW (undistilled)", steps=28, cfg=4.5, sampler="euler", scheduler="simple",
                         negative_prompt=True,
                         note="Euler on the flow-matching schedule shifted by exp(mu), mu from the image-token count "
                              "(0.5 at 256 tokens to 1.15 at 6400).",
                         source="Fizgig families/krea2.py RAW recipe, v7.0.1 (steps 28, cfg 4.5) + krea2/sampling.py "
                                "timesteps()"),
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
    # "RAW + Turbo LoRA" engine, audit 8.3); without the file they render the RAW model at 28 steps / CFG 4.5 (Fizgig
    # 7.0.1's RAW recipe; its own no-Turbo preview default is 8 steps / CFG 1, which TagScribeR does not copy).
    preview_speed_lora="Krea 2 Turbo (rank 64 LoRA)",
    preview_speed_steps=8,
    preview_steps=28,
    preview_cfg=4.5,
    preview_width=1024,                                      # audit 4.4: SAMPLE_WIDTH per-family default Krea 2 1024
    preview_height=1024,

    slider_training=True,
    slider_guidance=3.0,              # Fizgig's slider tests (1 Oct 2026): 3 on Krea 2, where Qwen's 2 pushes too little
    # Ultra mode (Fizgig, 1 Oct 2026): blocks 0-7 and the four text-fusion blocks (Fizgig's txt_lw_0/1, txt_rf_0/1)
    slider_ultra_blocks=tuple(f"block_{i}" for i in range(8)) + ("txtfusion_layerwise_0", "txtfusion_layerwise_1",
                                                                 "txtfusion_refiner_0", "txtfusion_refiner_1"),
    presets=(
        # THE DEFAULT (the first visit applies it): rank 8 with Adaptive LR at an aggressive floor. Rank 8 is more than
        # enough for a character on a 12.9B model and lands the right result more reliably than 32 (Peter, 16 Sep 2026).
        ("✨ Krea 2 Ultra Fast (rank 8, adaptive LR)", _preset(8, 30, True, "2e-4", "4e-4")),
        ("✨ Krea 2 Standard (rank 32, full model)", _preset(32, 64, False, "1e-4", "4e-4")),
        # Style: rank 16 (broad global direction; the one published datapoint is "rank 16 for simple styles"), LR
        # ceiling 2e-4 because the watcher probes up on steady descent and style overbakes at 4e-4 (the start is 1e-4,
        # the LR the Krea 2 ecosystem defaults to). Every epoch is saved: scrub for the sweet spot.
        ("✨ Krea 2 Style (rank 16, gentle LR)", _preset(16, 64, True, "5e-5", "2e-4")),
        # Slider (Fizgig 7.0.1 families/krea2.py, its keys and values verbatim): hot and short, as Qwen's (rank 4,
        # 2e-4, 0.5 MP); a happy/sad prompt slider (16 practice pictures, guidance 2) was right at 20 epochs and had
        # turned +1 into an illustration by 25. The loss watch and Adaptive LR stand down for sliders
        ("✨ Krea 2 Slider (rank 4, 2e-4)", {
            "NETWORK_DIM": 4, "NETWORK_ALPHA": 4, "NETWORK_TYPE": "LoRA (standard)", "LEARNING_RATE": 2e-4,
            "MAX_TRAIN_EPOCHS": 20, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42, "FAMILY_SLIDER": True,
            "ADAPTIVE_LR": False, "ADAPTIVE_LR_MIN": "1e-4", "ADAPTIVE_LR_MAX": "4e-4",
            "OPTIMIZER_TYPE": "adamw8bit", "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0,
            "DATASET_MEGAPIXELS": "0.5", "BLOCKS_SWAP": "Auto (detect from GPU)",
            "FAMILY_PRECISION": "Auto (fits your free VRAM)", "FAMILY_EMA": "0.98 (recommended)",
            "KREA2_LOSS_WATCH": True, "KREA2_PER_IMAGE_LR": True, "KREA2_AUTO_RECAPTION": False,
            "KREA2_WARMUP_LOOK": False, "FAMILY_SLIDER_GUIDANCE": "3"}),
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
        ("Not part of this port: the rotating-block full fine-tune and regularisation images, the fp8-Turbo-checkpoint "
         "preview engine, captioning with the encoder, RefMods, and the flash / sageattn / xformers attention "
         "backends Fizgig's Krea 2 never selects.", "docs/TRAINING_PLAN.md"),
        ("torch.compile (Compile Blocks Auto / On / Off): Auto is off on ROCm, with block swap, without a matching "
         "triton or a C compiler, on fp8 / bf16 bases, on short runs and when it will not fit; On overrides what it "
         "can. Compiled per block after the LoRA is in place.",
         "Fizgig utils/capabilities.py should_compile, krea2/trainer.py _compile_blocks"),
    ),
)
