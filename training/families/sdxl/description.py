# Descriptions of the SDXL-architecture families for TagScribeR (original code, written before Fizgig had SDXL).
# Merged with Fizgig 7.0.1 families/sdxl.py (commit 1c8ec88; owner's decision 4): Fizgig's MEASURED presets come first
# (rank 32 / alpha 16 and rank 16 / alpha 8 at a flat 5e-5, fused AdamW - measured on Juggernaut XL, a photographic
# checkpoint, so for Pony / Illustrious / NoobAI they are an extrapolation), then TagScribeR's earlier three, which are
# COMMUNITY STARTING POINTS, not measured. Also from Fizgig: INT8 / NF4 bases with its measured memory, LoKR on the
# kohya stems, the DPM++ 2M SDE Karras preview sampler (SDXL 1.0). TagScribeR's five variants, NoobAI v-pred with
# zero-terminal-SNR, LoCon, offline loading, batching and the timestep window are kept.
# Model facts: see unet.py / text.py / vae.py / sampling.py. Sampling settings and file names below are from the model
# cards as remembered by the author of this port and were NOT re-verified against the live pages: see each `source=`.
"""SDXL 1.0, Pony Diffusion V6 XL, Illustrious-XL and NoobAI-XL (eps / v-pred) family descriptions."""
from training.description import FamilyDescription, LoRAFormat, ModelFile, SamplingSettings

_DRIVER = "training.families.sdxl.driver:SDXLDriver"
_DRIVER_VPRED = "training.families.sdxl.driver:SDXLVPredDriver"
_VAE_REPO = "madebyollin/sdxl-vae-fp16-fix"


def _preset(rank, alpha, epochs, lo, hi, *, adaptive=True, lr=1e-4, scheduler="constant"):
    # Fizgig's key set and shape (KREA2_BUILT_IN_PRESETS in lora_trainer_gui.py), SDXL values NOT measured by Fizgig:
    # community starting points. 1.0 MP (SDXL's native training size), batch 1 (the loss watch and per-image LR need 1;
    # raise it in the Train tab for speed), AdamW 8-bit, EMA 0.98, loss watch on.
    return {
        "NETWORK_DIM": rank, "NETWORK_ALPHA": alpha, "NETWORK_TYPE": "LoRA (standard)",
        "LEARNING_RATE": lr,
        "MAX_TRAIN_EPOCHS": epochs, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": adaptive, "ADAPTIVE_LR_MIN": lo, "ADAPTIVE_LR_MAX": hi,
        "MIN_TIMESTEP": 0.0, "MAX_TIMESTEP": 1.0,
        "OPTIMIZER_TYPE": "adamw8bit", "LR_SCHEDULER": scheduler,
        "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0,
        "DATASET_MEGAPIXELS": "1.0", "DATASET_BATCH_SIZE": 1,
        "KREA2_LOSS_WATCH": True, "KREA2_PER_IMAGE_LR": True,
        "KREA2_AUTO_RECAPTION": False, "KREA2_WARMUP_LOOK": False,
        "FAMILY_EMA": "0.98 (recommended)",
    }


def _fizgig_preset(rank, alpha, lr=5e-5, epochs=20):
    # Fizgig 7.0.1 families/sdxl.py _preset: flat LR (SDXL's per-epoch loss swings with its uniform timesteps, so
    # Adaptive LR reacted to noise), fused AdamW (1493 -> 1046 ms/step against 8-bit AdamW, measured on a 5090), the
    # loss watch with per-image LR and auto-recaption on
    return {
        "NETWORK_DIM": rank, "NETWORK_ALPHA": alpha, "NETWORK_TYPE": "LoRA (standard)", "LEARNING_RATE": lr,
        "MAX_TRAIN_EPOCHS": epochs, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": False, "ADAPTIVE_LR_MIN": "1e-4", "ADAPTIVE_LR_MAX": "4e-4",
        "MIN_TIMESTEP": 0.0, "MAX_TIMESTEP": 1.0,
        "OPTIMIZER_TYPE": "adamw", "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0,
        "DATASET_MEGAPIXELS": "1.0", "BLOCKS_SWAP": "Auto (detect from GPU)",
        "FAMILY_PRECISION": "Auto (fits your free VRAM)", "FAMILY_EMA": "0.98 (recommended)",
        "KREA2_LOSS_WATCH": True, "KREA2_PER_IMAGE_LR": True, "KREA2_AUTO_RECAPTION": True,
        "KREA2_WARMUP_LOOK": False,
    }


def _presets(name):
    # NOT measured by Fizgig; community starting points:
    #  * Fast: rank 16 / alpha 16 with Adaptive LR between 5e-5 and 2e-4 (around the usual 1e-4 UNet rate), 20 epochs.
    #  * Standard: rank 32 / alpha 16 (the common kohya "dim 32, alpha 16"), fixed 1e-4 with a cosine decay, 30 epochs.
    #  * Style: rank 16 / alpha 8, gentle adaptive range 5e-5..1e-4, 40 epochs, every epoch saved to scrub for the sweet spot.
    return (
        # Fizgig's (Peter, 5 Oct 2026): the first is a first visit's preset. alpha at half the rank ("SDXL LoRAs gain
        # from the halved alpha"); 5e-5 "clearly better results than 1e-4 in his comparisons". The Slider and
        # Fine-tune presets wait for those modes (port plan stage 5)
        (f"✨ {name} Strong (rank 32, alpha 16, 5e-5)", _fizgig_preset(32, 16)),
        (f"✨ {name} Standard (rank 16, alpha 8, 5e-5)", _fizgig_preset(16, 8)),
        # TagScribeR's earlier presets, community starting points (not measured)
        (f"✨ {name} Fast (rank 16, adaptive LR)", _preset(16, 16, 20, "5e-5", "2e-4")),
        (f"✨ {name} Cosine (rank 32, community)", _preset(32, 16, 30, "1e-4", "4e-4", adaptive=False,
                                                           scheduler="cosine")),
        (f"✨ {name} Style (rank 16, gentle LR)", _preset(16, 8, 40, "5e-5", "1e-4")),
    )


def _lora():
    return LoRAFormat(
        key_template="lora_unet_{module}.{ab}.weight", down="lora_down", up="lora_up",
        block_modules=("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0", "attn2.to_q", "attn2.to_k",
                       "attn2.to_v", "attn2.to_out.0", "ff.net.0.proj", "ff.net.2"),
        alpha_key="{prefix}.alpha", kohya=True, file_prefix="lora_unet_",
        lokr_kohya_stems=True,                  # Fizgig 7.0.1: LoKR on the lora_unet_ (LDM) stems ComfyUI reads
        note="kohya SDXL keys with the ORIGINAL (LDM / SGM) module names, as A1111 and ComfyUI load them: "
             "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.lora_down.weight / .lora_up.weight / .alpha. "
             "The trainer's in-memory (diffusers) names are mapped to these on save and load.",
        source="ComfyUI comfy/utils.py unet_to_diffusers + comfy/lora.py model_lora_keys_unet (key NAMES only; "
               "verified against training/families/sdxl/unet.py for all 760 SDXL target modules)")


def _variant(key, arch, display, *, repo, file, size_gb, ckpt_note, sampling, preview_steps, preview_cfg,
             driver=_DRIVER, extra_notes=()):
    return FamilyDescription(
        key=key, arch_id=arch, display_name=display, gui_label=display, lora_name_suffix=arch,
        aliases=(), experimental=True,
        model_files=(
            ModelFile(f"{arch}_dit", f"{display} checkpoint", True, repo, file, size_gb, ckpt_note, role="dit"),
            ModelFile(f"{arch}_vae", "Separate VAE (optional)", False, _VAE_REPO, "sdxl_vae.safetensors", 0.33,
                      "Leave empty to use the VAE inside the checkpoint. The fp16-fix VAE is a drop-in that is "
                      "numerically safer in half precision; training itself runs the VAE in float32 / bf16 either way.",
                      role="vae", default_to="dit"),
            ModelFile(f"{arch}_text_encoder", "Text encoders (optional)", False, "", "", 0.0,
                      "Leave empty: the two CLIP text encoders inside the checkpoint are used (caching and previews "
                      "only, then unloaded). Set it only to take them from another SDXL checkpoint. The CLIP "
                      "tokenizer comes from a clip_tokenizer/ folder next to the checkpoint or the Hugging Face cache.",
                      role="text_encoder", default_to="dit"),
        ),
        text_encoder_label="CLIP-L + OpenCLIP-bigG", vae_label="SDXL VAE",
        latent_channels=4,                       # [sdxl-vae] diffusers vae/config.json latent_channels
        spatial_factor=8,                        # 3 downsamples in the VAE
        bucket_step=64,                          # kohya's SDXL bucket step (audit 11.3.1); 32 also works
        image_channels=3, native_megapixels=1.0,  # SDXL's native 1024 x 1024
        n_blocks=19, block_prefix="unet",        # IN00-IN08, MID, OUT00-OUT08 (block_map() is overridden by the driver)
        block_note="The UNet's input blocks, middle block and output blocks (kohya's IN / MID / OUT names). The LoRA "
                   "covers the attention, cross-attention and feed-forward layers of every transformer block (722 "
                   "modules); the optional LoCon switch adds the 3x3 convolutions (760 modules in all).",
        lora=_lora(),
        driver=driver,
        modelspec_arch="stable-diffusion-xl-v1-base",    # audit 11.3.1 (SAI modelspec architecture id)
        ema_default="0.98",                      # Fizgig's family default; the same recipe applies
        implementation="https://github.com/Stability-AI/generative-models",
        # Fizgig 7.0.1: bf16, INT8 and NF4 (2.6B UNet: 5.1 GB in bf16). Measured 5 Oct 2026 on a 5090 (Juggernaut v9,
        # rank 16, adamw8bit, gradient checkpointing, 1024 previews), peak GB including the preview, which sets it
        # (training alone: bf16 6.0 / 6.5, INT8 4.0 / 4.4, NF4 3.2 / 3.6 at 0.5 / 1 MP). No block swap.
        precisions=("bf16", "int8", "nf4"),
        train_memory={"bf16": (((0.5, 10.3), (1.0, 10.3)), 0.0), "int8": (((0.5, 8.2), (1.0, 8.2)), 0.0),
                      "nf4": (((0.5, 7.4), (1.0, 7.4)), 0.0)},
        optimizers=("adamw8bit", "adamw", "pagedadamw8bit", "ademamix8bit", "pagedademamix8bit", "lion8bit",
                    "automagic3"),
        network_types=("lora", "lokr"),
        family_options=("SDXL_MIN_SNR_GAMMA", "SDXL_NOISE_OFFSET", "SDXL_LOCON"),
        sampling=sampling,
        preview_steps=preview_steps, preview_cfg=preview_cfg, preview_width=1024, preview_height=1024,
        preview_negative="worst quality, low quality, lowres, bad anatomy, bad hands, jpeg artifacts, watermark, "
                         "signature, text",
        presets=_presets(display),
        helper_files=(("openai/clip-vit-large-patch14",
                       ("vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json")),),
        notes=(
            ("The checkpoint is one .safetensors file holding the UNet, both CLIP text encoders and the VAE; the "
             "UNet loads from the vendored SDXL config, so nothing is downloaded.", "diffusers SDXL unet/config.json"),
            ("Text conditioning is the penultimate hidden state of CLIP-L and OpenCLIP-bigG (2048 wide), cached for a "
             "FIXED 3 x 75 = 225 tokens (231 positions), plus the pooled vector; captions longer than 225 tokens are "
             "cut. A cached caption is about 1 MB. The text encoders are not trained.",
             "diffusers StableDiffusionXLPipeline.encode_prompt; A1111 / ComfyUI chunking"),
            ("Size conditioning uses the bucket's size as original and target size with no crop offset.",
             "diffusers StableDiffusionXLPipeline._get_add_time_ids"),
            ("Training timesteps are drawn uniformly over the 1000-step DDPM schedule; Noise range min / max rescale "
             "into the window. Min-SNR gamma, noise offset and LoCon are optional extensions, off by default.",
             "kohya sd-scripts (options); Fizgig timestep-window semantics"),
            ("The Strong and Standard presets are Fizgig's, measured on Juggernaut XL (a photographic checkpoint): flat "
             "5e-5, alpha at half the rank, fused AdamW, Adaptive LR off (it reacts to SDXL's noisy per-epoch loss). On "
             "Pony, Illustrious and NoobAI they are an extrapolation. The Fast, Cosine and Style presets are community "
             "starting points, not measured.", "Fizgig families/sdxl.py presets (5 Oct 2026)"),
        ) + tuple(extra_notes),
    )


_NEG_GENERIC = "worst quality, low quality, blurry, jpeg artifacts, watermark, text"

SDXL = _variant(
    "sdxl", "sdxl", "SDXL 1.0",
    repo="stabilityai/stable-diffusion-xl-base-1.0", file="sd_xl_base_1.0.safetensors", size_gb=6.94,
    ckpt_note="SDXL 1.0 base as one file (about 6.9 GB; the fp16 file is fine - it is loaded into bf16).",
    sampling=(SamplingSettings("SDXL (DPM++ 2M SDE Karras)", steps=30, cfg=6.0, sampler="dpmpp_2m_sde",
                               scheduler="karras", options=(("sampler", "dpmpp_2m_sde_karras"),), negative_prompt=True,
                               note="DPM++ 2M SDE with Karras sigmas, Fizgig 7.0.1's SDXL preview sampler (in ComfyUI: "
                                    "dpmpp_2m_sde / karras); the SDE noise follows the seed.",
                               source="Fizgig families/sdxl.py sampling; CFG and steps: diffusers SDXL defaults and the "
                                      "common CFG 5-8 range - community values"),
              SamplingSettings("SDXL base (Euler)", steps=30, cfg=6.0, sampler="euler", scheduler="simple",
                               negative_prompt=True,
                               note="Euler on the 1000-step scaled-linear schedule, trailing spacing: deterministic, "
                                    "so epochs compare cleanly.",
                               source="diffusers StableDiffusionXLPipeline defaults (guidance 5.0) and the common "
                                      "SDXL range CFG 5-8, 25-40 steps - community values, not verified here"),),
    preview_steps=30, preview_cfg=6.0)

PONY = _variant(
    "pony", "pony", "Pony Diffusion V6 XL",
    repo="LyliaEngine/Pony_Diffusion_V6_XL", file="ponyDiffusionV6XL_v6StartWithThisOne.safetensors", size_gb=6.94,
    ckpt_note="Pony Diffusion V6 XL as one file (the CivitAI download works too; about 6.9 GB).",
    sampling=(SamplingSettings("Pony V6", steps=25, cfg=7.0, sampler="euler", scheduler="simple",
                               negative_prompt=True,
                               note="Pony expects quality score tags: start prompts with 'score_9, score_8_up, "
                                    "score_7_up' and put 'score_6, score_5, score_4' in the negative prompt.",
                               source="Pony Diffusion V6 XL model card (CivitAI / Hugging Face), from memory: Euler a, "
                                      "about 25 steps, CFG about 7 - not re-verified"),),
    preview_steps=25, preview_cfg=7.0)

ILLUSTRIOUS = _variant(
    "illustrious", "illustrious", "Illustrious-XL",
    repo="OnomaAIResearch/Illustrious-xl-early-release-v0", file="Illustrious-XL-v0.1.safetensors", size_gb=6.94,
    ckpt_note="Illustrious-XL as one file (v0.1 or a later release; about 6.9 GB).",
    sampling=(SamplingSettings("Illustrious-XL", steps=28, cfg=6.0, sampler="euler", scheduler="simple",
                               negative_prompt=True,
                               note="Danbooru-style tags; 'masterpiece, best quality' in the prompt and 'worst quality, "
                                    "low quality' in the negative prompt are the usual quality tags.",
                               source="Illustrious-XL model card, from memory (Euler a, 28 steps, CFG 5-7) - not "
                                      "re-verified; the exact values are a community default"),),
    preview_steps=28, preview_cfg=6.0)

NOOBAI_EPS = _variant(
    "noobai_eps", "noobaieps", "NoobAI-XL (eps)",
    repo="Laxhar/noobai-XL-1.1", file="NoobAI-XL-v1.1.safetensors", size_gb=6.94,
    ckpt_note="NoobAI-XL epsilon-prediction checkpoint (v1.1 or similar; about 6.9 GB). Not the V-Pred one.",
    sampling=(SamplingSettings("NoobAI-XL eps", steps=28, cfg=5.5, sampler="euler", scheduler="simple",
                               negative_prompt=True,
                               note="Danbooru / e621 tags; the model card recommends quality tags such as 'masterpiece, "
                                    "best quality, newest, absurdres, highres'.",
                               source="NoobAI-XL model card, from memory (Euler, 28-32 steps, CFG 5-7) - not re-verified"),),
    preview_steps=28, preview_cfg=5.5)

NOOBAI_VPRED = _variant(
    "noobai_vpred", "noobaivpred", "NoobAI-XL (v-pred)",
    repo="Laxhar/noobai-XL-Vpred-1.0", file="NoobAI-XL-Vpred-v1.0.safetensors", size_gb=6.94,
    ckpt_note="NoobAI-XL V-Prediction checkpoint (about 6.9 GB). Trained and sampled with zero-terminal-SNR; do not "
              "pick the epsilon checkpoint here (or the other way round): the pictures come out as noise.",
    driver=_DRIVER_VPRED,
    sampling=(SamplingSettings("NoobAI-XL v-pred", steps=28, cfg=4.5, sampler="euler", scheduler="simple",
                               negative_prompt=True,
                               note="v-prediction with a zero-terminal-SNR schedule and trailing timesteps. The "
                                    "model card suggests a lower CFG (about 4-5); a CFG-rescale (about 0.7) used by "
                                    "some UIs is not implemented in previews.",
                               source="NoobAI-XL V-Pred model card, from memory (Euler, CFG 4-5.5) - not re-verified; "
                                      "zero-terminal-SNR: Lin et al. 2023 Algorithm 1"),),
    preview_steps=28, preview_cfg=4.5,
    extra_notes=(("v-prediction target v = sqrt(alpha_bar) * noise - sqrt(1 - alpha_bar) * x0 on the zero-terminal-SNR "
                  "schedule (the last step is pure noise). Leave Noise offset off for this model.",
                  "NoobAI-XL V-Pred model card; Lin et al. 2023; diffusers get_velocity"),))

VARIANTS = (SDXL, PONY, ILLUSTRIOUS, NOOBAI_EPS, NOOBAI_VPRED)
