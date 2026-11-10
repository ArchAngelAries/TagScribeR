# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/families/moss_image.py (commit 9162fb20,
# v8.7.3). Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: Fizgig's FamilyOption rows (MOSS_LORA_SFT, MOSS_LOWNOISE_PCT) are TagScribeR parameters
# (training/params.py, routed to driver.configure through family_options); the driver path; ModelFile fields TagScribeR
# does not have (hint, download_note, alt_repo / alt_path / alt_label / alt_note, bulk_fetch) are folded into `note`;
# description fields TagScribeR does not have (preview_cfg_note, samples_text, workbench_follows_samples,
# activation_cache, repair_size, the workbench tuple) are left out; Fizgig's stale fourth note ("No speed LoRA ... yet", contradicted by its own Turbo
# row) is dropped; every value otherwise verbatim, with Fizgig's sources. Fizgig's own header follows.
#
# MOSS-Image (OpenMOSS / MOSI, released 16 Oct 2026): a 5B single-stream DiT for text-to-image with strong text
# rendering and JSON layout prompts, Qwen3.5-9B text encoder, the FLUX.2 autoencoder. Apache-2.0 weights.
#
# Sources: github.com/OpenMOSS/MOSS-Image (README, USERGUIDE, paper/MOSS_Image_Technical_Report.pdf, commit c1355e6),
# the model cards and pipeline_moss_image.py (huggingface.co/OpenMOSS-Team/MOSS-Image-Base, -SFT), Comfy-Org's repack
# (huggingface.co/Comfy-Org/MOSS-Image) and ComfyUI's core support (Comfy-Org/ComfyUI PR #17095, merge e087d8f6cb).
"""MOSS-Image: OpenMOSS's 5B single-stream text-to-image DiT, trained on the Base checkpoint, previewed on SFT."""
from training.description import FamilyDescription, LoRAFormat, ModelFile, SamplingSettings, SpeedLoRA

_REPO = "https://github.com/OpenMOSS/MOSS-Image"
_CARD = "https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base"
_COMFY = "Comfy-Org/MOSS-Image"
_GUIDE = f"{_REPO} USERGUIDE.md"
_REPORT = f"{_REPO} paper/MOSS_Image_Technical_Report.pdf"
_PIPE = f"{_CARD} pipeline_moss_image.py"
_TURBO = "lodestones/MOSS-Image-SFT-8-steps-dmd"
_MODULES = ("proj.q", "proj.k", "proj.v", "proj.o", "proj.ffn_up_gate", "proj.ffn_down")   # model.py _Block Linears
# USERGUIDE.md Prompt Guidelines: "we recommend the following effective negative tags"
_NEGATIVE = "worst quality, bad anatomy, watermark, jpeg artifact"
_FZ = "Fizgig families/moss_image.py (v8.7.3)"


def _preset(rank, lr=1e-4, adaptive=None, epochs=30, mp="1.0", slider=False, ft=False):
    # Fizgig families/moss_image.py:23-39, keys and values verbatim. Not yet trained on MOSS-Image when written (the
    # driver was built without a GPU): Flux 3's preset shape (Krea 2's values), at 1 MP (the last pretraining stage's
    # area), with fused AdamW as Fizgig's CHECKLIST.md asks for LoRA presets. EMA 0.98, the per-image loss watch off,
    # saved every epoch. adaptive=(min, max) turns Adaptive LR on.
    lo, hi = adaptive or ("1e-4", "4e-4")
    return {
        "NETWORK_DIM": rank, "NETWORK_ALPHA": rank, "NETWORK_TYPE": "LoRA (standard)", "LEARNING_RATE": lr,
        "MAX_TRAIN_EPOCHS": epochs, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": adaptive is not None, "ADAPTIVE_LR_MIN": lo, "ADAPTIVE_LR_MAX": hi,
        "OPTIMIZER_TYPE": "adamw", "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0,
        "DATASET_MEGAPIXELS": mp, "BLOCKS_SWAP": "Auto (detect from GPU)",
        "FAMILY_PRECISION": "Auto (fits your free VRAM)", "FAMILY_EMA": "0.98 (recommended)",
        "KREA2_LOSS_WATCH": False, "KREA2_PER_IMAGE_LR": False, "KREA2_AUTO_RECAPTION": False,
        "KREA2_WARMUP_LOOK": False, "FAMILY_FT": ft, "FAMILY_EDIT": False, "FAMILY_SLIDER": slider,
        "FAMILY_FAST_ID": False, "FAMILY_SLIDER_GUIDANCE": "3",
        **({"FAMILY_FT_ROTATIONS": "10"} if ft else {}),
    }


# ComfyUI's render of MOSS-Image, the default and the SFT previews' recipe (Fizgig :165-172, :226-231)
_COMFY_OPTIONS = (("shift", 5.0), ("uncond_skip", (22,)), ("sampler", "res_multistep"), ("schedule", "simple"),
                  ("slg_end", 0.5))

MOSS_IMAGE = FamilyDescription(
    key="moss_image",
    arch_id="mossimage",
    display_name="MOSS-Image",
    gui_label="MOSS-Image",
    lora_name_suffix="moss",
    status_note="MOSS-Image support is new and improving: update TagScribeR regularly to get the latest fixes.",
    aliases=("moss-image", "mossimage"),
    experimental=True,          # until the owner's first real run confirms the port

    model_files=(
        ModelFile("moss_image_dit", "MOSS-Image Base DiT", True, _COMFY,
                  "diffusion_models/moss_image_base_bf16.safetensors", 10.16,
                  "bf16 Base for training (the model card: Base is for fine-tuning and LoRAs). The SFT file "
                  "(moss_image_sft_bf16) is for use, not training. ComfyUI's single file; the official diffusers "
                  "shards (OpenMOSS-Team/MOSS-Image-Base transformer/, give the path of any shard) and the "
                  "int8_convrot file load too. So does Kromoss (lodestones/Kromoss "
                  "diffusion_models/kromoss_bf16.safetensors, ~10.2 GB), lodestones' sharper fine-tune of Base, in its "
                  "place: it trains the same way, at the same memory.",
                  role="dit",
                  # Fizgig :51-66, verified 17 Oct 2026: tensor for tensor the official transformer/ shards, q / k / v
                  # fused into proj.qkv = [q; k; v]; the loader splits it back. Kromoss: a full fine-tune of Base at
                  # 1024 px (same architecture, files and recipe); checked 19 Oct 2026 on a 5090 as the Base DiT - int8
                  # auto, 9.4 GB peak at 1 MP, trains, saves and previews
                  fetch_choice=("MOSS-Image Base", "Kromoss"), alt_size_gb=10.16),
        ModelFile("moss_image_vae", "FLUX.2 autoencoder (small decoder)", True, _COMFY,
                  "vae/flux2-vae_small.safetensors", 0.25,
                  "The FLUX.2 autoencoder with BFL's small decoder, the decoder MOSS-Image ships with: FLUX.2's own "
                  "encoder, so the latents are the same as with flux2-vae.safetensors or Klein's ae.safetensors, which "
                  "also load here (full decoder), but the small decoder avoids spots in the pictures.",
                  role="vae",
                  # Fizgig :67-78: MOSS-Image's own checkpoint uses the FLUX.2 small decoder "to help eliminate some
                  # strange artifacts and spots" (a MOSS-Image core developer, r/StableDiffusion AMA, Oct 2026)
                  replaces=("flux2-vae.safetensors", "ae.safetensors")),
        ModelFile("moss_image_turbo_lora", "Turbo LoRA (optional)", False, _TURBO, "lora.safetensors", 1.19,
                  "Optional. A turbo / DMD LoRA for fast previews on the training model: lodestones' 8-step DMD of SFT "
                  "(soft on Base), or Kromoss's own 8-step LoRA (lodestones/Kromoss 8-steps-dmd-lora.safetensors, "
                  "~1.2 GB, 8 steps, CFG 1) for a Kromoss Base. Leave it empty for full-quality previews. With one "
                  "set, raise the Turbo LoRA strength on the Samples tab and set Steps and CFG for it (8 steps, CFG 1).",
                  role="speed_lora"),          # Fizgig :79-88, bulk_fetch=False: its own Download link only
        ModelFile("moss_image_text_encoder", "Qwen3.5-9B text encoder", True, "Comfy-Org/Qwen3.5",
                  "text_encoders/qwen3.5_9b_bf16.safetensors", 19.31,
                  "Used for caching only, then unloaded. Loads text-only (about 15 GB), or streamed layer by layer "
                  "from system RAM on smaller cards (still exact bf16). ComfyUI's int8 ConvRot file "
                  "(Comfy-Org/MOSS-Image text_encoders/qwen3.5_9b_int8_convrot.safetensors, ~9.7 GB) and the official "
                  "text_encoder/ shards also load (decoded at load).",
                  role="text_encoder"),
        # Fizgig :89-100: MOSS-Image SFT (OpenMOSS's aesthetics-tuned checkpoint, same architecture; @Murf-Code, PR
        # #195): training samples render on it (the Samples tab tick, on by default - a Base LoRA carries over to the
        # SFT only partly). Never trained: training runs on the Base
        ModelFile("moss_image_sft_dit", "MOSS-Image SFT DiT (previews)", True, _COMFY,
                  "diffusion_models/moss_image_sft_int8_convrot.safetensors", 5.41,
                  "OpenMOSS's everyday checkpoint: training samples render on it (untick it on the Samples tab for "
                  "samples on the Base), showing the LoRA where most people use it. Never trained. ComfyUI's int8 "
                  "ConvRot file (~5.4 GB, Apache-2.0) is decoded at load; the bf16 SFT file "
                  "(diffusion_models/moss_image_sft_bf16.safetensors, ~10.2 GB) works as well.",
                  role="preview_dit"),
    ),
    text_encoder_label="Qwen3.5-9B",
    vae_label="FLUX.2 autoencoder",
    helper_files=(("OpenMOSS-Team/MOSS-Image-Base", ("tokenizer/*",)),),   # tokenizer.json + chat template

    latent_channels=128,              # transformer/config.json latent_channels: 32 FLUX.2 channels 2x2-packed
    spatial_factor=16,                # report Table 5: effective image stride 16 pixels
    bucket_step=16,                   # pipeline __call__: "height and width must be positive multiples of 16"
    native_megapixels=1.0,            # USERGUIDE.md: 1, 1.5 or 2 MP recommended, 0.25 to 4 MP supported

    n_blocks=42,                      # transformer/config.json num_layers
    block_prefix="layers",
    block_note="42 joint blocks with one shared modulation (so per-block sliders act on attention and MLP), after a "
               "two-block text adapter that refines the Qwen3.5 states before they meet the image.",

    lora=LoRAFormat(
        key_template="lora_unet_layers_{block}_{module}.{ab}.weight",
        down="lora_down", up="lora_up",
        block_modules=_MODULES,
        alpha_key="{prefix}.alpha",
        kohya=True,
        file_prefix="lora_unet_",
        note="kohya keys over the reference module names. ComfyUI's MOSSImage branch maps lora_unet_layers_N_proj_q "
             "/ _k / _v onto thirds of the fused proj.qkv in Comfy-Org's file; its generic map takes the same keys "
             "on an unfused model (the OpenMOSS ComfyUI plugin). LoKR saves as diffusion_model.layers.N.proj.q, "
             "which both maps also take.",
        source="Comfy-Org/ComfyUI comfy/lora.py model_lora_keys_unet (PR #17095, merge e087d8f6cb)",
    ),

    driver="training.families.moss_image.driver:MOSSImageDriver",
    modelspec_arch="MOSS-Image",
    implementation=_REPO,
    precisions=("int8", "nf4", "bf16"),
    auto_precisions=("int8", "nf4"),  # AI Toolkit's MOSS-Image default is an 8-bit base; bf16 stays a manual choice
    # Fizgig :134-141, MEASURED 17 Oct 2026 on a 5090 (STILLS.md 4's calculation, 4 one-epoch runs + the 2-epoch smoke
    # run). The preview round (1024², 50 steps, CFG 3) sets the peak at every precision, so the figures are the
    # preview's peak after training (optimizer state resident; bf16 smoke run 15.1 GB vs 14.6 before training, +0.5
    # applied to INT8 / NF4). Training alone: bf16 9.2 GB at 0.25 MP with 10 blocks swapped (7.97 resident -> 1.2 GB
    # overhead, ~11.3 GB unswapped), 13.4 at 1 MP; INT8 8.8 and NF4 7.3 at 1 MP. Per swapped block: 0.211 GB bf16
    # (10 blocks cut the preview peak 2.1 GB), 0.105 INT8. Not run on the owner's AMD card yet.
    train_memory={"bf16": (((0.25, 15.1), (1.0, 15.1)), 0.211), "int8": (((0.25, 10.5), (1.0, 10.5)), 0.105),
                  "nf4": (((0.25, 8.5), (1.0, 8.5)), 0.0)},
    optimizers=("adamw", "adamw8bit"),
    network_types=("lora", "lokr"),
    ema_default="0.98",
    compiles=True,                    # driver.compile_targets: the 42 blocks
    compile_boundary="outside",       # as Qwen 2.1's and Flux 3's: the checkpoint wraps the compiled block
    # Fizgig :147-150, measured 17 Oct 2026, 5090, bf16, 1 MP, one bucket: 1.45 -> 1.07 s/step compiled (-26%), the
    # first epoch 13 s slower (compiling) = paid back in ~35 steps; 200 leaves room for runs with several bucket shapes.
    # INT8 / NF4 not measured yet: bf16's figure, and Klein's NF4 one, so Auto compiles the default INT8 base.
    compile_payback_steps={"bf16": 200, "int8": 200, "nf4": 400},
    # Fizgig :163-164: finetune=True (driver.ft_spec: every block's six Linears on an NF4 trunk, attention only with
    # Train attention only); ft_learning_rate is FINETUNE.md's default (the report's pretraining LR was 5e-5, Muon +
    # AdamW). Not run by TagScribeR on any hardware
    finetune=True,
    ft_learning_rate=1e-5,
    slider_training=True,             # photo pairs (training_loss diff weighting) or prompts (noise_latents / predict)
    # edit_training stays off: the released checkpoints don't condition on reference images (report section 8: "does
    # not directly condition on reference images or support image editing"; pipeline docstring: defaults are "not
    # ... reference-image variants"). The RoPE keeps a reference-index axis for a later release.
    # the Train tab shows these only for this family; they reach MOSSImageDriver.configure (params.DRIVER_OPTIONS):
    # Fizgig's FamilyOption rows MOSS_LORA_SFT ("Train attention only", lora_sft=1) and MOSS_LOWNOISE_PCT ("Clean-end
    # share", lownoise_pct, default 30) (Fizgig :246-257)
    family_options=("COMPILE_BLOCKS", "MOSS_LORA_SFT", "MOSS_LOWNOISE_PCT"),

    sampling=(
        SamplingSettings("res_multistep, 30 steps, CFG 4", steps=30, cfg=4.0, sampler="res_multistep",
                         scheduler="simple", options=_COMFY_OPTIONS, negative_prompt=True,
                         note="As ComfyUI renders it: res_multistep on ComfyUI's simple schedule (shift 5), CFG 4, and "
                              "the SLG switch on (SkipLayerGuidanceDiTSimple: block 22 skipped in the negative branch "
                              "for the first half, sigma >= 0.83).",
                         source=f"{_FZ}: Peter's ComfyUI comparison, 18 Oct 2026 (ComfyUI's MOSS-Image template, SLG on)"),
        SamplingSettings("Euler, 30 steps", steps=30, cfg=3.0, sampler="euler", scheduler="simple",
                         options=(("shift", 5.0), ("uncond_skip", (22,))), negative_prompt=True,
                         note="AI Toolkit's MOSS-Image sample settings: the reference sampler at 30 steps.",
                         source="ostris/ai-toolkit 7b94dc5, extensions_built_in/diffusion_models/ui.tsx (moss_image)"),
        SamplingSettings("Reference (diffusers pipeline)", steps=50, cfg=3.0, sampler="euler", scheduler="simple",
                         options=(("shift", 5.0), ("uncond_skip", (22,))), negative_prompt=True,
                         note="Flow-matching Euler on the shift-5 grid snapped to the 1000 training sigmas; CFG 3 with "
                              "block 22 skipped in the negative branch; the negative is empty by default.",
                         source=f"{_PIPE} (MOSSImageScheduler, __call__ defaults); {_CARD} model card"),
        SamplingSettings("ComfyUI workflow", steps=50, cfg=3.0, sampler="euler", scheduler="sgm_uniform",
                         options=(("shift", 5.0), ("uncond_skip", (22,))), negative_prompt=True,
                         note="SkipLayerGuidanceDiTSimple layer 22 on the negative, and no ModelSamplingSD3 node: ComfyUI's "
                              "MOSS-Image model brings its own sampling, and adding one renders badly (Peter, 18 Oct 2026). "
                              "ComfyUI's scheduler does not snap to the training grid, so the same seed differs slightly.",
                         source=f"{_REPO} comfyui/moss-image/example_workflows/moss-image-api.json, comfyui/README.md"),
    ),
    speed_loras=(
        SpeedLoRA(
            name="MOSS-Image Turbo LoRA",
            repo=_TURBO,
            file="lora.safetensors",
            pairs_with="MOSS-Image SFT",
            strength=1.0,             # lodestones' card: lora.safetensors at 1.0 (lora_latest at 0.7)
            settings=SamplingSettings("Turbo 8-step", steps=8, cfg=1.0, sampler="euler", scheduler="simple",
                                      options=(("shift", 5.0), ("uncond_skip", ()), ("sampler", "euler")),
                                      note="8 steps, no CFG, the reference's shift-5 grid.",
                                      source=f"https://huggingface.co/{_TURBO} (load in ComfyUI's LoRA loader)"),
            load_unmerged=True,       # an fp32 rank-128 adapter: its delta is below bf16's rounding step
            pref_key="moss_image_turbo_lora",
            community_settings=(("12-16 steps for stronger prompt adherence; strength 0.8 at 16 steps",
                                 "r/StableDiffusion 'MOSS-Image INT8 ComfyUI' thread; dskjal's ComfyUI guide"),),
            caveats=("Distilled from MOSS-Image SFT, and soft on Base (Peter's comparison, 18 Oct 2026).",),
            source=f"https://huggingface.co/{_TURBO}",
        ),
    ),
    preview_speed_lora="MOSS-Image Turbo LoRA",      # used only when its Model files row is set (empty by default)
    preview_speed_strength=0.0,       # the Samples tab's Turbo LoRA strength starts at 0 (full-quality previews);
    #                                   raise it with the Steps / CFG boxes set for the turbo (1 at 8 steps, CFG 1)
    preview_steps=30,                 # res_multistep 30 at CFG 4: the best of the side-by-side (Peter, 18 Oct)
    preview_cfg=4.0,                  # USERGUIDE.md: 3-6 recommended; 4 won the side-by-side
    preview_negative=_NEGATIVE,
    # training samples on the SFT checkpoint (the driver's park_for_preview / load_preview_checkpoint /
    # unpark_after_preview; @Murf-Code, PR #195): on by default - where most people use a MOSS LoRA (Peter)
    train_preview_checkpoint=True,
    preview_checkpoint_tick="Use the SFT model for samples (set it under Model files - a Base LoRA carries over "
                            "only partly)",
    preview_checkpoint_on=True,
    # the released Base's last-block norm (Fizgig :233-236, measured 19 Oct 2026: Base 3658.7021484375, SFT bf16 and
    # int8 3658.7646484375, Kromoss 3658.6904296875); any other Base DiT previews on itself (base_check.py)
    standard_base=("layers.41.norm2.weight", 3658.7021484375),
    preview_checkpoint_steps_note="Base samples only - the SFT samples use their own 30 steps, CFG 4",
    preview_checkpoint_sampling=SamplingSettings(
        "MOSS-Image SFT", steps=30, cfg=4.0, sampler="res_multistep", scheduler="simple",
        options=_COMFY_OPTIONS, negative_prompt=True,
        note="The Base samples' sampler (ComfyUI's res_multistep / simple 30, CFG 4, shift 5, SLG block 22 for the "
             "first half) on the SFT checkpoint.",
        source=f"https://huggingface.co/{_COMFY}; OpenMOSS/MOSS-Image-SFT model card"),
    preview_width=1024,
    preview_height=1024,
    presets=(
        # Fizgig :234-245, names, keys and values verbatim. Standard at 1 MP (MOSS-Image pretrained to 1024²); Fast is
        # Standard at 0.25 MP and rank 16, which does very well on face-cropped datasets (Peter's runs, 18 Oct 2026).
        # 70 epochs: a 53-image run was still improving at 1,590 steps (epoch 30), so a 25-image set needs ~65 epochs
        # to get there (Peter, 18 Oct 2026). A fixed learning rate: Adaptive LR walks to its floor on MOSS-Image's flat
        # loss (@Murf-Code's runs, PR #195; Peter's runs agree that 1e-4 suits it). Adaptive LR stays on the Training
        # tab, off by default.
        ("◆ Balanced (rank 32, 1e-4)", _preset(32, epochs=70)),
        ("◆ Quick start (rank 16, 0.25 MP)", _preset(16, epochs=70, mp="0.25")),
        ("◆ Style (rank 16, 5e-5)", {**_preset(16, lr=5e-5, epochs=50), "FAMILY_EMA": "Off"}),   # no EMA: a style at 5e-5 is slow enough
        ("◆ Slider (rank 8, 2e-4)", _preset(8, lr=2e-4, epochs=20, slider=True)),
        ("◆ Full fine-tune (1e-5)", _preset(16, lr=1e-5, ft=True)),
    ),

    notes=(
        ("Training timesteps follow the released weights' last pretraining interval: sigma = 1 - sigmoid(Z), "
         "Z ~ N(-1.09, 0.95^2), truncated at 0.95 and rescaled onto the 1000-step grid, mixed with 1% uniform (the "
         "Clean-end share moves the curve's centre, 30% of steps below sigma 0.5 by default; the reference is about "
         "11%); the DiT is told 1000 * sigma; loss is the unweighted per-token MSE on noise - x0.",
         f"{_REPORT} Appendix A.2, eq. 3-4, Table 6"),
        ("Text conditioning: Qwen3.5-9B's final hidden state through the chat template with an open <think> block, "
         "unpadded, up to 4096 tokens. Keep the text encoder frozen.",
         f"{_PIPE} encode_prompt; Comfy-Org/ComfyUI comfy/text_encoders/moss_image.py; {_GUIDE} Post-Training"),
        ("No edit or reference-image training: the released checkpoints are text-to-image only. 'Layout editing' "
         "means regenerating from an edited layout prompt, not conditioning on a picture.",
         f"{_REPORT} section 6 and 8; {_GUIDE}"),
        ("Train on Base, use the LoRA on SFT (where previews render by default). Training on SFT or a fine-tune of "
         "it wants Train attention only: training its feed-forward layers breaks anatomy. With Kromoss or any other "
         "Base DiT, previews render on that model instead (the SFT tick turns off once).",
         "Fizgig docs/MOSS_IMAGE.md (v8.7.3), families/base_check.py"),
        ("Not part of this port yet: the workbench tools.", "docs/dev/PORT_PLAN.md"),
    ),
)
