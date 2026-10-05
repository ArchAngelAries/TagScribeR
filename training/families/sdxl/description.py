# Descriptions of the SDXL-architecture families for TagScribeR (original; Fizgig has no SDXL FamilyDescription).
# Fizgig is the truth for training BEHAVIOUR; its presets' SHAPES (key set, "fast adaptive default / standard / style",
# EMA 0.98, loss watch + per-image LR on) are reused here. EVERY NUMBER IN THE PRESETS BELOW IS A COMMUNITY STARTING
# POINT, NOT MEASURED BY FIZGIG and not measured by this port either: they come from widespread kohya-style SDXL LoRA
# practice (rank 16-32, UNet LR ~1e-4, 1024 px training) and must be revisited after the first real runs.
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


def _presets(name):
    # NOT measured by Fizgig; community starting points:
    #  * Fast: rank 16 / alpha 16 with Adaptive LR between 5e-5 and 2e-4 (around the usual 1e-4 UNet rate), 20 epochs.
    #  * Standard: rank 32 / alpha 16 (the common kohya "dim 32, alpha 16"), fixed 1e-4 with a cosine decay, 30 epochs.
    #  * Style: rank 16 / alpha 8, gentle adaptive range 5e-5..1e-4, 40 epochs, every epoch saved to scrub for the sweet spot.
    return (
        (f"✨ {name} Fast (rank 16, adaptive LR)", _preset(16, 16, 20, "5e-5", "2e-4")),
        (f"✨ {name} Standard (rank 32, cosine)", _preset(32, 16, 30, "1e-4", "4e-4", adaptive=False,
                                                           scheduler="cosine")),
        (f"✨ {name} Style (rank 16, gentle LR)", _preset(16, 8, 40, "5e-5", "1e-4")),
    )


def _lora():
    return LoRAFormat(
        key_template="lora_unet_{module}.{ab}.weight", down="lora_down", up="lora_up",
        block_modules=("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0", "attn2.to_q", "attn2.to_k",
                       "attn2.to_v", "attn2.to_out.0", "ff.net.0.proj", "ff.net.2"),
        alpha_key="{prefix}.alpha", kohya=True, file_prefix="lora_unet_",
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
        # bf16 only: the 2.6B UNet is ~5.1 GB in bf16. fp32 / fp16 bases are not offered: fp16 overflows in the VAE and
        # is less stable for LoRA training; int8 / nf4 are unnecessary at this size.
        precisions=("bf16",),
        # training memory was NOT measured (no hardware access when this was written), so Auto simply takes bf16.
        train_memory={},
        optimizers=("adamw8bit", "adamw", "pagedadamw8bit", "ademamix8bit", "pagedademamix8bit", "lion8bit"),
        network_types=("lora",),
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
            ("Training starting points (rank, learning rate, epochs) are community values, not measured by Fizgig.",
             "docs/TRAINING_PLAN.md"),
        ) + tuple(extra_notes),
    )


_NEG_GENERIC = "worst quality, low quality, blurry, jpeg artifacts, watermark, text"

SDXL = _variant(
    "sdxl", "sdxl", "SDXL 1.0",
    repo="stabilityai/stable-diffusion-xl-base-1.0", file="sd_xl_base_1.0.safetensors", size_gb=6.94,
    ckpt_note="SDXL 1.0 base as one file (about 6.9 GB; the fp16 file is fine - it is loaded into bf16).",
    sampling=(SamplingSettings("SDXL base", steps=30, cfg=6.0, sampler="euler", scheduler="simple",
                               negative_prompt=True,
                               note="Euler on the 1000-step scaled-linear schedule, trailing spacing. Previews are "
                                    "deterministic Euler so epochs compare cleanly; Euler-a / DPM++ are what most UIs use.",
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
