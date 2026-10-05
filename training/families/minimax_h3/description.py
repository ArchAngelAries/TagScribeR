# Ported from Fizgig (https://github.com/shootthesound/Fizgig): the facts of src/fizgig/minimax/ (model.py, loader.py,
# trainer.py, sampling.py, embedder.py), lora_trainer_gui.py (MINIMAX_BUILT_IN_PRESETS, the H3 model paths and the
# MINIMAX_* settings), scripts/fetch_models.py (files and sizes) and training/metadata.py (ARCH_MINIMAX).
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: Fizgig has no FamilyDescription for MiniMax H3, so this one is assembled from those sources -
# every value cites its file; the presets keep Fizgig's keys and values verbatim (the MINIMAX_* keys this app expresses
# differently are mapped by training/presets.py migrate_legacy).
"""MiniMax H3 (image LoRA training only): the 33B omni DiT trained on single still images."""
from training.description import FamilyDescription, LoRAFormat, ModelFile, SamplingSettings

_REPO = "Comfy-Org/MiniMax-H3"
_FETCH = "Fizgig scripts/fetch_models.py FAMILIES['minimax']"

# Fizgig MINIMAX_BUILT_IN_PRESETS (lora_trainer_gui.py:1007-1170), keys and values verbatim. The rank-16 entry is the
# base recipe; Fast is "spread from it" with rank 8 / 50 epochs; Style is Fast with the clip-still off; Fast is
# re-inserted first so it is the default.
_DEFAULTS = {
    "NETWORK_DIM": 16, "NETWORK_ALPHA": 16,
    "NETWORK_TYPE": "LoRA (standard)", "LOKR_FACTOR": 8,
    "LEARNING_RATE": 1e-6,                                  # Automagic's STARTING rate, not the rate (17 Sep 2026)
    "MINIMAX_ADAPTER_RAMP": "Off",
    "MINIMAX_CAPTION_DROPOUT": "0.05 (default)",
    "MAX_TRAIN_EPOCHS": 60, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
    "ADAPTIVE_LR": False, "ADAPTIVE_LR_MIN": "1e-5", "ADAPTIVE_LR_MAX": "4e-4",
    "OPTIMIZER_TYPE": "automagic3",
    "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0,
    "DATASET_MEGAPIXELS": "0.25",
    "MINIMAX_LOWNOISE_PCT": "60", "MINIMAX_HIGHNOISE_LR_PCT": "100",
    "MINIMAX_BLOCKS": "all", "MINIMAX_BASE_QUANT": "Auto (recommended)",
    "MINIMAX_TRAIN_ADALN": False,
    "MINIMAX_TRAIN_REFINER": False,
    "MINIMAX_LIKENESS_MODE": "Default",                     # Fizgig MINIMAX_MODE_FAST: blocks 20-49
    "MINIMAX_ADAPTER": "Circlestone — best for photos",
    "MINIMAX_TREAD": True,
    "MINIMAX_CLIP_STILL": True,
    "MINIMAX_SLOW_BLOCKS": "", "MINIMAX_SLOW_LR_SCALE": "0.2",
    "MINIMAX_BLOCK_LIMIT": "Off",
    "MINIMAX_LR_WARMUP": "Off",
    "MINIMAX_EMA": "0.98 (recommended)",
    "MINIMAX_DISTILL": False,
}
_FAST = {**_DEFAULTS, "NETWORK_DIM": 8, "NETWORK_ALPHA": 8, "MAX_TRAIN_EPOCHS": 50, "LEARNING_RATE": 1e-6,
         "MINIMAX_ADAPTER_RAMP": "Off", "ADAPTIVE_LR": False}
_STYLE = {**_FAST, "LEARNING_RATE": 1e-6, "OPTIMIZER_TYPE": "automagic3", "MINIMAX_LIKENESS_MODE": "Default",
          "MINIMAX_CLIP_STILL": False}

MINIMAX_H3 = FamilyDescription(
    key="minimax_h3",
    arch_id="minimaxh3",                                     # Fizgig training/metadata.py ARCHITECTURE_MINIMAX
    display_name="MiniMax H3",
    gui_label="MiniMax H3",
    lora_name_suffix="minimaxh3",
    aliases=("minimax", "minimax-h3", "h3"),
    experimental=True,

    model_files=(
        ModelFile("minimax_dit", "MiniMax H3 DiT (pruned int8)", True, _REPO,
                  "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors", 21.0,
                  "The 33B DiT as ComfyUI ships it (int8 ConvRot, ~21 GB) - what training runs on. The 66 GB bf16 "
                  "release is refused. Loading needs about 22 GB of free RAM.", role="dit"),
        ModelFile("minimax_vae", "MiniMax H3 video VAE", True, _REPO, "vae/minimax_h3_video_vae_fp16.safetensors", 4.9,
                  "Encodes your photos (caching) and decodes previews; 24 latent channels, 16x.", role="vae"),
        ModelFile("minimax_text_encoder", "Qwen3-VL-32B text encoder (nvfp4)", True, _REPO,
                  "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", 15.7,
                  "Used for caching and preview prompts only, then unloaded: needs about 15 GB of free VRAM (less "
                  "streams its layers from the CPU, slower, ~19 GB of RAM). The tokenizer files come from the "
                  "Hugging Face cache, or a qwen3vl_tokenizer/ folder next to this file.", role="text_encoder"),
        ModelFile("minimax_circlestone_adapter", "Training adapter (Circlestone)", False,
                  "circlestone-labs/MiniMax-H3-Image-Training-Adapter",
                  "minimax_h3_image_training_adapter.safetensors", 0.62,
                  "Optional but recommended for photos: a frozen helper LoRA on for every training step, off for "
                  "previews, never saved into your LoRA.", role="training_adapter"),
    ),
    text_encoder_label="Qwen3-VL-32B",
    vae_label="MiniMax H3 video VAE",

    latent_channels=24,                                      # model.py MiniMaxH3Config.latents_dim
    spatial_factor=16,                                       # vae.py vae_ratio = prod(space_down) = 16
    bucket_step=32,                                          # 16 x patch 2 (Fizgig BUCKET_RESO_STEPS 32)
    image_channels=3,
    native_megapixels=0.6,                                   # H3's 768 short-edge canvas

    n_blocks=50,                                             # MiniMaxH3Config.num_layers
    block_prefix="blocks",
    block_note="50 identical blocks (attention + SwiGLU MLP). The LoRA covers each block's four Linears (qkv, out, fc1, "
               "fc2) - 200 modules; the text refiner and the AdaLN projections are not trained. Default training mode "
               "trains blocks 20-49 (120 modules): the identity blocks.",

    lora=LoRAFormat(
        key_template="lora_unet_blocks_{block}_{module}.{ab}.weight",
        down="lora_down", up="lora_up",
        block_modules=("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"),
        alpha_key="{prefix}.alpha",
        kohya=True,
        file_prefix="lora_unet_",
        note="kohya keys, as Fizgig writes them: lora_unet_blocks_<n>_<attn|mlp>_<qkv_proj|out_proj|fc1|fc2>."
             "lora_down.weight / .lora_up.weight / .alpha. A LoKR adapter is saved LyCORIS-style: diffusion_model."
             "<dotted path>.lokr_w1 / .lokr_w2 / .alpha.",
        source="Fizgig minimax/trainer.py DEFAULT_INCLUDE_PATTERNS + _save_lora (create_network 'lora_unet')",
    ),

    driver="training.families.minimax_h3.driver:MiniMaxH3Driver",
    modelspec_arch="MiniMax-H3",                             # Fizgig training/metadata.py ARCH_MINIMAX
    training_adapter="minimax_circlestone_adapter",
    training_adapter_note="Circlestone's image training adapter: frozen, on for every step, off in previews.",
    ema_default="0.98",                                      # MINIMAX_EMA (Peter, 9 Sep 2026)
    implementation="https://github.com/MiniMax-AI/MiniMax-H3",   # Fizgig training/metadata.py IMPL_MINIMAX
    precisions=("int8", "nf4"),
    # Fizgig minimax/trainer.py measured anchors (5090, gradient checkpointing, batch 1): NF4 pruned base 10.46 GB
    # resident, step peak 13.46 / 13.56 / 13.63 GB at 0.23 / 0.50 / 0.98 MP; the int8 base is 21.07 GB resident. The
    # int8 peak below is an ESTIMATE (resident + ~1 GB for activations, a rank-8 adapter and its optimiser state),
    # and 0.39 GB per swapped block is Fizgig's _H2D_PER_BLOCK_GB (one streamed int8 block) - not measured with this
    # port's offloader. NF4 cannot swap.
    train_memory={"int8": (((0.25, 22.1), (1.0, 22.4)), 0.39), "nf4": (((0.25, 13.5), (1.0, 13.7)), 0.0)},
    # Fizgig locks the optimizer to adamw / automagic3 for H3 (full-precision state: 8-bit state costs fine detail).
    optimizers=("automagic3", "adamw"),
    network_types=("lora", "lokr"),

    sampling=(
        SamplingSettings("H3 default", steps=20, cfg=1.0, sampler="res_multistep", scheduler="simple",
                         note="res_multistep on ComfyUI's simple schedule at shift 12, no CFG (every shipped H3 "
                              "workflow). A preview is a still, which is out of the clip distribution the model "
                              "was mostly trained on: it answers 'is the LoRA learning?', not final quality.",
                         source="Fizgig minimax/sampling.py sample_schedule / _sample_image_impl"),
    ),
    preview_steps=20,
    preview_cfg=1.0,
    preview_width=512,                                       # Fizgig _build_minimax_train_command fallback "512"
    preview_height=512,

    presets=(
        ("✨ MiniMax H3 Fast (LoRA 8, 50 epochs)", dict(_FAST)),
        ("✨ MiniMax H3 (rank 16, 60 epochs)", dict(_DEFAULTS)),
        ("✨ MiniMax H3 Style (LoRA 8)", dict(_STYLE)),
    ),
    family_options=("MINIMAX_LOWNOISE_PCT", "MINIMAX_HIGHNOISE_LR_PCT", "MINIMAX_LIKENESS_MODE", "MINIMAX_BLOCKS"),

    helper_files=(("Qwen/Qwen3-VL-4B-Instruct", ("chat_template.json", "generation_config.json", "merges.txt",
                                                  "preprocessor_config.json", "tokenizer.json",
                                                  "tokenizer_config.json", "video_preprocessor_config.json",
                                                  "vocab.json")),),

    notes=(
        ("Image LoRA training only: video clips, voice, reference images (RefMods), multi-concept, distillation, the "
         "rotation full fine-tune and the Turbo-LoRA previews are not part of this port.", "docs/TRAINING_PLAN.md"),
        ("Training target is x0 - noise on noised = (1 - sigma) x0 + sigma noise, t = 1 - sigma fed to the DiT (sign "
         "convention matched to ComfyUI). The Low-noise training % dial sets the schedule shift (1 - P) / P; the "
         "default 60% is shift 0.667.", "Fizgig minimax/trainer.py:1-14, lora_trainer_gui.py:518-540"),
        ("Medium to High Noise LR % scales the optimizer's learning rate (never the loss) for steps drawn at sigma 0.5 "
         "and above, averaged over an accumulation window. Automagic v3, the preset optimizer, sets its own rate, so the "
         "dial only acts with AdamW - the same in Fizgig.",
         "Fizgig minimax/trainer.py:4359-4369, 5259-5270, 5498; lora_trainer_gui.py:553-565"),
        ("Caption dropout 0.05 swaps in the cached empty-prompt embedding for that share of steps.",
         "Fizgig minimax/trainer.py:5385, scripts/minimax_cache_text.py:298"),
        ("Default training mode trains blocks 20-49 only; blocks 0-19 would only ever hold zero-initialised adapters "
         "in Fizgig's runs (their gradients are cut), so they are left out of the file instead.",
         "Fizgig minimax/trainer.py:3978-4020"),
    ),
)
