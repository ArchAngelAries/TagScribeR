# Anima (circlestone-labs/Anima): a 2B Cosmos-Predict2 fine-tune for anime / illustration.
#
# Fizgig has NO Anima code, so the architecture facts below were established for this port from primary sources
# (checked 5 Oct 2026). Training BEHAVIOUR (loop, Adaptive LR, EMA, loss watch, cached conditioning, previews, preset
# shapes) is Fizgig's, through the generic layer. Two independent trainers agree on every training-critical fact.
#
# VERIFIED FACTS (confidence: H = two independent trainers/loaders agree, M = one source or inferred)
#  1. DiT = NVIDIA Cosmos-Predict2-2B-Text2Image "MiniTrainDIT": 28 blocks, 2048 wide, 16 heads, patch 2, in/out 16 ch
#     (+1 zero padding-mask input channel), AdaLN-LoRA dim 256, 3-axis RoPE (h/w extrapolation 4.0), crossattn width 1024,
#     "net." (base) or "model.diffusion_model." (aesthetic/turbo) key prefix, fixed config.                          [H]
#     https://huggingface.co/circlestone-labs/Anima  (base_model nvidia/Cosmos-Predict2-2B-Text2Image)
#     https://github.com/kohya-ss/sd-scripts/blob/main/library/anima_utils.py (load_anima_model dit_config, Apache-2.0)
#     https://github.com/tdrussell/diffusion-pipe/blob/main/models/cosmos_predict2.py (get_dit_config, Apache-2.0 header)
#     diffusers' CosmosTransformer3DModel does NOT load it: no LLM adapter and different module names, so this port
#     carries its own DiT (training/families/anima/model.py) in the checkpoint's key names.                          [H]
#  2. Text: Qwen3-0.6B (base) last hidden state (after the final norm), right-padded to 512, padding zeroed; PLUS the
#     prompt tokenised with the T5 (google/t5-v1_1-xxl vocabulary) tokenizer to 512 ids. The T5 ids are NOT encoded by
#     T5: the "LLM adapter" (6 layers, 1024 wide, 32128-token embedding, lives INSIDE the DiT checkpoint as
#     llm_adapter.*) embeds them and cross-attends to the Qwen3 states; its output (zeroed at T5 padding) is the DiT's
#     cross-attention context. Runs inside the DiT forward.                                                          [H]
#     sd-scripts library/strategy_anima.py + library/anima_models.py; diffusion-pipe models/llm_adapter.py;
#     docs https://github.com/kohya-ss/sd-scripts/blob/main/docs/anima_train_network.md; ComfyUI
#     comfy/text_encoders/anima.py (GPL - facts only: Qwen3 pad id 151643, T5 ids carried beside the Qwen states).
#     The model card says: do NOT train the adapter ("outsized influence"); this port freezes it and no LoRA targets it.
#     https://huggingface.co/circlestone-labs/Anima/blob/main/README.md                                              [H]
#  3. VAE = Qwen-Image VAE (16 ch, /8, the Wan 2.1 latent statistics), ordinary mean/std normalisation of the VAE.
#     Model card file list (split_files/vae/qwen_image_vae.safetensors); sd-scripts AnimaLatentsCachingStrategy
#     (vae.encode_pixels_to_latents); ComfyUI Anima latent_format = Wan21. Shares Krea 2's VAE code here.           [H]
#  4. Objective = plain rectified flow: x_t = (1 - t) x0 + t noise, target = noise - x0, MSE. The model is fed t in
#     [0, 1] itself (sd-scripts divides its 0-1000 timesteps by 1000; diffusion-pipe passes t directly). Training t =
#     sigmoid(N(0, 1) * 1.0) with NO shift (sd-scripts defaults --timestep_sampling sigmoid, --discrete_flow_shift
#     1.0; diffusion-pipe logit_normal). (Nvidia's EDM-style weighting t^2 + (1 - t)^2 is dropped by both.)        [H]
#     https://github.com/kohya-ss/sd-scripts/blob/main/anima_train_network.py (get_noise_pred_and_target)
#     https://github.com/tdrussell/diffusion-pipe/blob/main/models/cosmos_predict2.py (prepare_inputs + note)
#  5. Sampling: Euler on a flow with shift 3.0 and the model receiving sigma (ComfyUI Anima: multiplier 1.0, shift 3.0;
#     sd-scripts' minimal script defaults shift 5.0); 30-50 steps, CFG 4-5 (model card); card's preferred samplers are
#     er_sde / euler_a / dpmpp_2m_sde - this port previews with plain Euler.                                          [M]
#     https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/supported_models.py (class Anima)
#  6. LoRA: ComfyUI loads `lora_unet_<module path, dots -> underscores>.lora_down/.lora_up/.alpha` (its generic key map
#     lora_unet_ + every diffusion_model.* weight) and `diffusion_model.<path>.lora_A/lora_B`. sd-scripts writes the kohya
#     form with the prefix "lora_unet" ("ComfyUI compatible") over class `Block` Linears, excluding modulation / norm /
#     embedder / final layer by default; diffusion-pipe writes `diffusion_model.` + PEFT keys.                       [H]
#     https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora_anima.py ; ComfyUI comfy/lora.py
#  7. Licences: sd-scripts Apache-2.0, diffusion-pipe code Apache-2.0 file headers (repo LICENSE is GPL-3.0 - nothing
#     was copied from it), NVIDIA Cosmos modeling Apache-2.0, ComfyUI GPL (facts only). Anima weights are
#     NON-COMMERCIAL (circlestone-labs-non-commercial-license; generated images may be used commercially).         [H]
#
# NOT VERIFIED (open points; the model has never been run here against the real files):
#  * the exact checkpoint header (key list / extra keys) - the loader is strict and names any mismatch;
#  * the T5 tokenizer: sd-scripts bundles "t5_old" and documents google/t5-v1_1-xxl's vocabulary; both share the 32128 ids
#    but equivalence was not diffed;
#  * recommended preview shift (3.0 here vs 5.0 in sd-scripts' script); training t has no shift in either trainer.
"""Anima: the 2B Cosmos-Predict2 anime / illustration model, trained on the base checkpoint."""
from training.description import (
    FamilyDescription, LoRAFormat, ModelFile, SamplingSettings,
)

_REPO = "circlestone-labs/Anima"
_CARD = "https://huggingface.co/circlestone-labs/Anima"


def _preset(rank, epochs, adaptive, lo, hi, lr, mp):
    # Preset SHAPES are Fizgig's (lora_trainer_gui.py KREA2_BUILT_IN_PRESETS: LoRA, alpha = rank, adamw8bit, loss watch +
    # per-image LR on, EMA 0.98, every epoch saved). The rank / LR / epoch VALUES are NOT measured by Fizgig - they are
    # community starting points (see the comments at each preset).
    return {
        "NETWORK_DIM": rank, "NETWORK_ALPHA": rank, "NETWORK_TYPE": "LoRA (standard)",
        "LEARNING_RATE": lr,
        "MAX_TRAIN_EPOCHS": epochs, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": adaptive, "ADAPTIVE_LR_MIN": lo, "ADAPTIVE_LR_MAX": hi,
        "MIN_TIMESTEP": "", "MAX_TIMESTEP": "",
        "OPTIMIZER_TYPE": "adamw8bit",
        "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0,
        "DATASET_MEGAPIXELS": mp,
        "BLOCKS_SWAP": "Auto (detect from GPU)",
        "KREA2_LOSS_WATCH": True, "KREA2_PER_IMAGE_LR": True,
        "KREA2_AUTO_RECAPTION": False, "KREA2_WARMUP_LOOK": False,
        "FAMILY_EMA": "0.98 (recommended)",
    }


ANIMA = FamilyDescription(
    key="anima",
    arch_id="anima",
    display_name="Anima",
    gui_label="Anima",
    lora_name_suffix="anima",
    aliases=("anima-base", "cosmos-predict2-anima"),
    experimental=True,                                       # never run against the real checkpoint (see top of file)

    model_files=(
        ModelFile("anima_dit", "Anima base DiT", True, _REPO, "split_files/diffusion_models/anima-base-v1.0.safetensors",
                  4.2,
                  "The 2B base (bf16, 4.2 GB) - what the model card recommends for LoRA training. The aesthetic / turbo "
                  "releases load too (same architecture) but are tuned looks, not a neutral base. Includes the LLM "
                  "adapter, which stays frozen. Pre-quantised fp8 files are refused.", role="dit"),
        ModelFile("anima_vae", "Qwen-Image VAE", True, _REPO, "split_files/vae/qwen_image_vae.safetensors", 0.25,
                  "The Qwen-Image VAE (16 latent channels, 8x) - the same file Krea 2 uses.", role="vae"),
        ModelFile("anima_text_encoder", "Qwen3-0.6B text encoder", True, _REPO,
                  "split_files/text_encoders/qwen_3_06b_base.safetensors", 1.2,
                  "Qwen3-0.6B base (1.2 GB); used for caching and preview prompts only, then unloaded. Tokenizers come "
                  "from the Hugging Face cache, or qwen3_tokenizer/ and t5_tokenizer/ folders next to this file "
                  "(tokenizer files only: the T5 one is just the token vocabulary, no T5 model is loaded).",
                  role="text_encoder"),
    ),
    text_encoder_label="Qwen3-0.6B",
    vae_label="Qwen-Image VAE",

    latent_channels=16,                                      # fact 3
    spatial_factor=8,
    bucket_step=16,                                          # 8 (VAE) x patch 2 (sd-scripts verify_bucket_reso_steps(16))
    image_channels=3,
    native_megapixels=1.0,                                   # model card: 512^2 to 1536^2

    n_blocks=28,                                             # fact 1
    block_prefix="blocks",
    block_note="28 AdaLN-LoRA blocks (self-attention, cross-attention to the text, MLP). The LoRA covers the attention "
               "and MLP Linears - 280 modules; the AdaLN modulation, embedders, output layer and the LLM adapter are not "
               "trained (the model card: leave the adapter alone).",

    lora=LoRAFormat(
        key_template="lora_unet_blocks_{block}_{module}.{ab}.weight",
        down="lora_down", up="lora_up",
        block_modules=("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.output_proj",
                       "cross_attn.q_proj", "cross_attn.k_proj", "cross_attn.v_proj", "cross_attn.output_proj",
                       "mlp.layer1", "mlp.layer2"),
        alpha_key="{prefix}.alpha",
        kohya=True,
        file_prefix="lora_unet_",
        note="kohya keys as sd-scripts writes them and ComfyUI loads: lora_unet_<module path with dots as underscores>"
             ".lora_down.weight / .lora_up.weight / .alpha.",
        source="sd-scripts networks/lora_anima.py (LORA_PREFIX_ANIMA 'lora_unet', ComfyUI compatible) + ComfyUI "
               "comfy/lora.py model_lora_keys_unet (lora_unet_ + diffusion_model.* key with dots as underscores)",
    ),

    driver="training.families.anima.driver:AnimaDriver",
    modelspec_arch="Anima",
    ema_default="0.98",                                      # Fizgig's Krea 2 default; Fizgig never measured it on Anima
    implementation="https://huggingface.co/circlestone-labs/Anima",
    precisions=("bf16",),                                    # 2B (4.2 GB): no quantisation or block swap needed
    train_memory={},                                         # not measured: Auto takes bf16
    optimizers=("adamw8bit", "adamw", "pagedadamw8bit", "ademamix8bit", "pagedademamix8bit", "lion8bit"),
    network_types=("lora",),

    sampling=(
        SamplingSettings("Anima base", steps=30, cfg=4.0, sampler="euler", scheduler="simple",
                         options=(("shift", 3.0),), negative_prompt=True,
                         note="Euler on the flow schedule shifted by 3 (ComfyUI's Anima setting). The model card "
                              "suggests 30-50 steps, CFG 4-5 and the er_sde / euler_a / dpmpp_2m_sde samplers; Turbo "
                              "checkpoints want CFG 1 and 8-12 steps. Anima negatives are TAGS (worst quality, low "
                              "quality, score_1, blurry, ...).",
                         source=f"{_CARD} (README generation settings); ComfyUI supported_models.Anima "
                                f"sampling_settings"),
    ),
    preview_steps=30,
    preview_cfg=4.0,
    preview_negative="worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, watermark, "
                     "signature, text",
    preview_width=1024,
    preview_height=1024,

    presets=(
        # NOT measured by Fizgig; community starting points. The model author's own guidance: "a light touch" - rank 32
        # starts at LR 2e-5 (model card); sd-scripts' documented example is rank 8, LR 1e-4 at alpha 1 (a 1/8 scale -
        # these presets use alpha = rank, scale 1, so they start lower).
        # THE DEFAULT: rank 8, adaptive LR between 1e-5 and 2e-4 (starts at ~4.5e-5), 24 epochs, 0.5 MP.
        ("✨ Anima Fast (rank 8, adaptive LR)", _preset(8, 24, True, "1e-5", "2e-4", 4.5e-5, "0.5")),
        # Standard: rank 32 at the model card's 2e-5, fixed, 1 MP.
        ("✨ Anima Standard (rank 32, LR 2e-5)", _preset(32, 40, False, "1e-5", "1e-4", 2e-5, "1.0")),
        # Style: rank 16, adaptive 1e-5..1e-4 (starts ~3.2e-5), every epoch saved to scrub for the sweet spot.
        ("✨ Anima Style (rank 16, gentle LR)", _preset(16, 40, True, "1e-5", "1e-4", 3.2e-5, "1.0")),
    ),

    # tokenizer files only (no model weights): fetched with the helper models so first use works offline
    helper_files=(("Qwen/Qwen3-0.6B", ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")),
                  ("google/t5-v1_1-xxl", ("spiece.model", "tokenizer_config.json", "special_tokens_map.json"))),

    notes=(
        ("EXPERIMENTAL: the architecture, conditioning and objective were verified from two independent trainers "
         "(kohya sd-scripts, diffusion-pipe) and ComfyUI, but never run against the real checkpoint here. The loader is "
         "strict and reports any key mismatch.", "description.py header"),
        ("Conditioning = Qwen3-0.6B hidden states + T5 token ids (512 each, padded, with masks). The LLM adapter inside "
         "the DiT turns them into the cross-attention context at every step; it is frozen and not a LoRA target "
         "(model card: do not train it). A cached caption is ~1 MB.", f"{_CARD}; sd-scripts strategy_anima.py"),
        ("Training: rectified flow, t = sigmoid(N(0,1)) without shift, the model receives t in [0, 1]; a noise range "
         "(min / max timestep) rescales into the window.", "sd-scripts anima_train_network.py; diffusion-pipe"),
        ("Anima is trained on Danbooru-style tags and natural language together (lowercase tags with spaces, score_* "
         "quality tags only on the base model, artists as @name). Tag-based captions work well; the optional "
         "'Shuffled tag variants' extension (CAPTION_SHUFFLE_VARIANTS, off by default, keep the first N tags) suits "
         "them but is not enabled by default.", f"{_CARD} (prompting section)"),
        ("Train on the base model for a neutral LoRA; the weights are non-commercial (images you generate may be used "
         "commercially).", f"{_CARD} (licence)"),
        ("Previews sample the training model with plain Euler; the card prefers er_sde / euler_a, so a ComfyUI render "
         "will look slightly different. Previews use the Train tab's negative prompt - Anima wants a tag negative.",
         "model card"),
        ("Not part of this port: fp8 / quantised bases, block swap, torch.compile, training the LLM adapter, LoKR, "
         "Automagic per-family groups.", "docs/TRAINING_PLAN.md"),
    ),
)
