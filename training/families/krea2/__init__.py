"""Krea 2 (ported from Fizgig: description, driver, DiT, VAE, sampler, Qwen3-VL-4B text encoder).

Fizgig (https://github.com/shootthesound/Fizgig, Apache-2.0) is the reference for every training behaviour here: the
flow-matching objective and timestep sampler, the 12-layer text conditioning, the all-Linear kohya LoRA, the presets,
the Turbo-LoRA preview recipe and the measured memory figures. See docs/TRAINING_PLAN.md and THIRD_PARTY_NOTICES.md.

Not ported (Fizgig has them; this family does not):
  * the fp8 _scaled_mm fast path (NVIDIA Ada and newer only; off on ROCm in Fizgig too). The fp8 base itself IS
    supported: fp8 / fp8-scaled RAW files load as fp8 and are dequantised per matmul (training/modules/fp8.py),
    a bf16 file can be quantised to fp8, and either file feeds the INT8 / 4-bit bases
  * torch.compile of the blocks
  * the rotating-block full fine-tune and regularisation images
  * the classic fp8-Turbo-checkpoint preview engine (previews run on the training model with the Turbo LoRA)
  * Automagic v3's per-family parameter groups (so Automagic v3 is not offered for this family)
  * captioning with the text encoder (the encoder is encode-only here), reference images and RefMods
  * the Repair Studio activation cache (forward_cached) and Fizgig's flash / sageattn / xformers attention backends
"""
