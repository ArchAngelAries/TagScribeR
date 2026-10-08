# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/families/minimax.py (Fizgig 7.0.1's MiniMax
# H3 description: model files, presets, options, optimizer settings) and the facts of src/fizgig/minimax/ (model.py,
# loader.py, sampling.py, embedder.py) and training/metadata.py (ARCH_MINIMAX).
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: still images only, so the clip, voice, audio-VAE, distillation, multi-concept, fine-tune and
# Slider parts are left out (port plan stages 5-6); the Turbo preview sampler is res_multistep like the base previews
# (Fizgig's settings text says euler, but its sampler runs res_multistep, minimax/sampling.py); Fizgig's family options are this app's H3_* parameters
# (training/params.py, same keys and labels); the HQQ base, the H2D block rings and the 66 GB bf16 DiT are not ported
# (int8 and NF4 only); the LoRA file keeps this app's "minimaxh3" name suffix (Fizgig: "mmh3").
"""MiniMax H3 (image LoRA training only): the 33B omni DiT trained on single still images."""
from training.description import FamilyDescription, LoRAFormat, ModelFile, SamplingSettings, SpeedLoRA
from training.params import H3_ADAPTERS, H3_STRUCTURES

_REPO = "Comfy-Org/MiniMax-H3"
_OSTRIS = "ostris/minimax_h3_training_adapter"


def _preset(rank, epochs, clip_still=True, slider=False, lr=1e-6, optimizer="automagic3"):
    """Fizgig 7.0.1's H3 built-ins (families/minimax.py _preset), keys and values verbatim. H3_TREAD, H3_CLIP_STILL and
    H3_DISTILL switch on clip / distillation features this port does not have: they are carried (so the preset is
    Fizgig's) and ignored."""
    return {
        "NETWORK_DIM": rank, "NETWORK_ALPHA": rank, "NETWORK_TYPE": "LoRA (standard)", "LOKR_FACTOR": 8,
        "LEARNING_RATE": lr, "MAX_TRAIN_EPOCHS": epochs, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "FAMILY_SLIDER": slider, "FAMILY_SLIDER_GUIDANCE": "2",
        "ADAPTIVE_LR": False, "ADAPTIVE_LR_MIN": "1e-5", "ADAPTIVE_LR_MAX": "4e-4", "OPTIMIZER_TYPE": optimizer,
        "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0, "DATASET_MEGAPIXELS": "0.25",
        "FAMILY_PRECISION": "Auto (recommended)", "BLOCKS_SWAP": "Auto (detect from GPU)",
        "FAMILY_EMA": "0.98 (recommended)",
        "H3_ADAPTER_RAMP": "Off", "H3_CAPTION_DROPOUT": "0.05 (default)", "H3_STRUCTURE": H3_STRUCTURES[0],
        "H3_LOWNOISE_PCT": "60", "H3_HIGHNOISE_LR_PCT": "100",
        "H3_BLOCKS": "all", "H3_TRAIN_REFINER": "", "H3_LIKENESS_MODE": "Default",
        "H3_ADAPTER": H3_ADAPTERS[0], "H3_TREAD": "1", "H3_CLIP_STILL": "1" if clip_still else "",
        "H3_DISTILL": "",
    }


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
                  "The 33B DiT as ComfyUI ships it (int8 ConvRot, ~21 GB): the training base, so your LoRA trains on "
                  "the weights it will run on. The 66 GB bf16 release is not supported here yet. Loading needs about "
                  "22 GB of free RAM.", role="dit"),
        ModelFile("minimax_ref_dit", "MiniMax H3 DiT (reference, ref2va)", False, _REPO,
                  "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors", 21.0,
                  "Optional: used when Training base is Reference (ref2va). A different fine-tune from the one above, "
                  "the model ComfyUI's Reference-to-Video workflow loads; a LoRA trained on it works best there.",
                  role="ref_dit"),
        ModelFile("minimax_vae", "MiniMax H3 video VAE", True, _REPO, "vae/minimax_h3_video_vae_fp16.safetensors", 4.9,
                  "Encodes your photos (caching) and decodes previews; 24 latent channels, 16x.", role="vae"),
        ModelFile("minimax_text_encoder", "Qwen3-VL-32B text encoder (nvfp4)", True, _REPO,
                  "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", 15.7,
                  "Used for caching and preview prompts only, then unloaded: needs about 15 GB of free VRAM (less "
                  "streams its layers from the CPU, slower, ~19 GB of RAM). The tokenizer files come from the "
                  "Hugging Face cache, or a qwen3vl_tokenizer/ folder next to this file.", role="text_encoder"),
        ModelFile("minimax_turbo_lora", "Turbo LoRA (previews)", False, "larryvrh/MiniMax-H3-Turbo-Lora",
                  "minimax_h3_turbo_v4_step600.safetensors", 0.78,
                  "Optional: fast in-training previews. With it set, previews render in 6 steps with the community "
                  "Turbo LoRA at 75% on top of your LoRA, the pairing fast ComfyUI renders use. Previews only: it is "
                  "switched in for the render and out again, and your saved LoRA never contains it. Steps and "
                  "strength are on the Samples settings.", role="speed_lora"),
        ModelFile("minimax_circlestone_adapter", "Training adapter (Circlestone)", False,
                  "circlestone-labs/MiniMax-H3-Image-Training-Adapter",
                  "minimax_h3_image_training_adapter.safetensors", 0.62,
                  "Needed when Training adapter is Circlestone (the default), one file for both bases: sharper eyes, "
                  "cleaner skin and better prompt-following on any dataset with stills. On for every training step, "
                  "off for previews, never saved into your LoRA.", role="training_adapter"),
        ModelFile("minimax_training_adapter", "Training adapter (Ostris, fl2va)", False, _OSTRIS,
                  "minimax_h3_training_adapter_v1.safetensors", 0.16,
                  "Optional: used when Training adapter is Ostris on the standard base."),
        ModelFile("minimax_ref_training_adapter", "Training adapter (Ostris, ref2va)", False, _OSTRIS,
                  "minimax_h3_ref2va_training_adapter_v1.safetensors", 0.16,
                  "Optional: used when Training adapter is Ostris and Training base is Reference (ref2va)."),
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
    # the training adapter is the H3_ADAPTER choice (Circlestone / Ostris per base / Off), not the generic tick
    adapter_choice="H3_ADAPTER",
    adapter_files=(("Circlestone", "minimax_circlestone_adapter", "minimax_circlestone_adapter"),
                   ("Ostris", "minimax_training_adapter", "minimax_ref_training_adapter")),
    alt_base=("H3_TRAIN_BASE", "Reference", "minimax_ref_dit"),
    trainable_dtype="bf16",                                  # the old trainer: network.to(device, dtype=bfloat16)
    ema_default="0.98",                                      # the old H3 default (v5.4.1)
    ema_short_run=True,
    implementation="https://github.com/MiniMax-AI/MiniMax-H3",   # Fizgig training/metadata.py IMPL_MINIMAX
    precisions=("int8", "nf4"),
    # Fizgig minimax/trainer.py measured anchors (5090, gradient checkpointing, batch 1): NF4 pruned base 10.46 GB
    # resident, step peak 13.46 / 13.56 / 13.63 GB at 0.23 / 0.50 / 0.98 MP; the int8 base is 21.07 GB resident. The
    # int8 peak below is an ESTIMATE (resident + ~1 GB for activations, a rank-8 adapter and its optimiser state),
    # and 0.39 GB per swapped block is Fizgig's _H2D_PER_BLOCK_GB (one streamed int8 block) - not measured with this
    # port's offloader. NF4 cannot swap.
    train_memory={"int8": (((0.25, 22.1), (1.0, 22.4)), 0.39), "nf4": (((0.25, 13.5), (1.0, 13.7)), 0.0)},
    # Fizgig 7.0.1's Train tab offers its whole optimizer catalog; Automagic v3 is the presets' optimizer. Adam-family
    # optimizers get the old trainer's weight decay 1e-4 (bitsandbytes' default is 1e-2) and the 8-bit eps floor
    optimizers=("automagic3", "adamw8bit", "adamw", "pagedadamw8bit", "ademamix8bit", "pagedademamix8bit", "lion8bit"),
    optimizer_weight_decay=1e-4,
    optimizer_eps_floor_8bit=True,
    network_types=("lora", "lokr"),

    sampling=(
        SamplingSettings("H3 default", steps=20, cfg=1.0, sampler="res_multistep", scheduler="simple",
                         note="res_multistep on ComfyUI's simple schedule at shift 12, no CFG (every shipped H3 "
                              "workflow). A preview is a still, which is out of the clip distribution the model "
                              "was mostly trained on: it answers 'is the LoRA learning?', not final quality.",
                         source="Fizgig minimax/sampling.py sample_schedule / _sample_image_impl"),
    ),
    speed_loras=(
        SpeedLoRA(
            name="H3 Turbo LoRA (6-step)",
            repo="larryvrh/MiniMax-H3-Turbo-Lora",
            file="minimax_h3_turbo_v4_step600.safetensors",
            pairs_with="MiniMax H3 fl2va",
            strength=0.75,
            settings=SamplingSettings("Turbo 6-step", steps=6, cfg=1.0, sampler="res_multistep", scheduler="simple",
                                      note="CFG-free on the same sampler and schedule; the old previews' 6 steps at "
                                           "0.75.",
                                      source="Fizgig families/minimax.py speed_loras (the original H3 trainer's "
                                             "load_preview_turbo)"),
            load_unmerged=True,
            pref_key="minimax_turbo_lora",
            caveats=("Its AdaLN rows are 2688 wide (the full model's silu(t_emb) space): on the pruned base they are "
                     "injected at run time from the bundled grid (training/families/minimax_h3/turbo.py), as Fizgig "
                     "and larryvrh's ComfyUI node do.",),
            source="https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora",
        ),
    ),
    preview_speed_lora="H3 Turbo LoRA (6-step)",
    samples_turbo_pace=True,                                 # Fizgig: "N steps at M%" (FAMILY_TURBO_STEPS / _PACE)
    preview_steps=20,
    preview_cfg=1.0,
    preview_width=768,                                       # Fizgig 7.0.1 (the old Samples default)
    preview_height=768,

    slider_training=True,                                    # stills: photo pairs or prompts (Fizgig 7.0.1)
    presets=(
        # Fizgig 7.0.1's, in its order (the first is the default)
        ("✨ MiniMax H3 Fast (LoRA 8, 50 epochs)", _preset(8, 50)),
        ("✨ MiniMax H3 (rank 16, 60 epochs)", _preset(16, 60)),
        ("✨ MiniMax H3 Style (LoRA 8)", _preset(8, 50, clip_still=False)),
        # Slider (Fizgig, 3 Oct): rank 8 at 2e-4 for ~160 steps; a prompt smile slider at push 2 was clear by epoch 5
        # of 10 (16 practice pictures an epoch)
        ("✨ MiniMax H3 Slider (rank 8, 2e-4)", _preset(8, 16, clip_still=False, slider=True, lr=2e-4,
                                                         optimizer="adamw8bit")),
    ),
    family_options=("H3_TRAIN_BASE", "H3_STRUCTURE", "H3_LOWNOISE_PCT", "H3_HIGHNOISE_LR_PCT", "H3_LIKENESS_MODE",
                    "H3_BLOCKS", "H3_ADAPTER", "H3_ADAPTER_RAMP", "H3_TRAIN_REFINER"),

    helper_files=(("Qwen/Qwen3-VL-4B-Instruct", ("chat_template.json", "generation_config.json", "merges.txt",
                                                  "preprocessor_config.json", "tokenizer.json",
                                                  "tokenizer_config.json", "video_preprocessor_config.json",
                                                  "vocab.json")),),

    notes=(
        ("Image LoRA training only: video clips, voice, reference images (RefMods), multi-concept, distillation, "
         "the rotation full fine-tune, the HQQ base and the bf16 DiT are not part of this port yet.",
         "docs/dev/PORT_PLAN.md"),
        ("Training target is x0 - noise on noised = (1 - sigma) x0 + sigma noise, t = 1 - sigma fed to the DiT (sign "
         "convention matched to ComfyUI). Training structure sets the clean-end share P and with it the schedule shift "
         "(1 - P) / P; the default 60% is shift 0.667.", "Fizgig minimax/driver.py _shift, families/minimax.py"),
        ("Caption dropout 0.05 swaps in the cached empty-prompt embedding for that share of steps.",
         "Fizgig minimax/trainer.py:5385, scripts/minimax_cache_text.py:298"),
        ("Default training mode trains blocks 20-49 only; blocks 0-19 would only ever hold zero-initialised adapters "
         "in Fizgig's runs (their gradients are cut), so they are left out of the file instead.",
         "Fizgig minimax/trainer.py:3978-4020"),
    ),
)
