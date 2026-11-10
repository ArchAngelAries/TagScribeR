"""MOSS-Image (ported from Fizgig v8.7.3: description, driver, DiT, sampler, Qwen3.5-9B text encoder, base check).

Fizgig (https://github.com/shootthesound/Fizgig, Apache-2.0) is the reference for every training behaviour here: the
flow-matching objective on the Base checkpoint, the technical report's truncated logit-normal timestep law with the
Clean-end share, the Qwen3.5-9B conditioning (chat template with an open think block, unpadded, 4096 tokens), the
kohya LoRA keys over all 42 blocks (block 22 included), Train attention only, the five presets, the SFT preview
checkpoint, the Turbo LoRA previews and the measured memory figures. The FLUX.2 autoencoder is the shared
training/modules/flux2_vae.py (with the ROCm wide-attention fix). See docs/TRAINING_PLAN.md and THIRD_PARTY_NOTICES.md.

The full fine-tune is the shared one (training/ft.py; driver.ft_spec: every block's six Linears, attention only with
Train attention only, q / k / v written back fused for ComfyUI's file).

Not ported (Fizgig has them; this family does not yet):
  * the comfy-kitchen INT8 attention hook (workbench renders only; Fizgig skips it on AMD too)
  * the workbench pieces: the text encoder kept in pinned RAM between prompts (keep_text_encoder / text_encoder_gb /
    drop_text_encoder), the activation cache, Repair Studio / Royale settings (workbench_follows_samples, repair_size)
"""
