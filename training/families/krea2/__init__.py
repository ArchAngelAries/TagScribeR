"""Krea 2 (ported from Fizgig: description, driver, DiT, VAE, sampler, Qwen3-VL-4B text encoder).

Fizgig (https://github.com/shootthesound/Fizgig, Apache-2.0) is the reference for every training behaviour here: the
flow-matching objective and timestep sampler, the 12-layer text conditioning, the all-Linear kohya LoRA, the presets,
the Turbo-LoRA preview recipe and the measured memory figures. See docs/TRAINING_PLAN.md and THIRD_PARTY_NOTICES.md.

Not ported (Fizgig has them; this family does not):
  * the fp8 _scaled_mm fast path (NVIDIA Ada and newer only; off on ROCm in Fizgig too). The fp8 base itself IS
    supported: fp8 / fp8-scaled RAW files load as fp8 and are dequantised per matmul (training/modules/fp8.py),
    a bf16 file can be quantised to fp8, and either file feeds the INT8 / 4-bit bases
  * the rotating-block full fine-tune and regularisation images
  * the classic fp8-Turbo-checkpoint preview engine (previews run on the training model with the Turbo LoRA)
  * captioning with the text encoder (the encoder is encode-only here), reference images and RefMods
  * the Repair Studio activation cache (forward_cached)
  * attention backends Fizgig's Krea 2 never selects: krea2/utils.py:92 defaults attn_mode="torch" and nothing passes
    another, so flash / sageattn / xformers / split attention are unreachable there. The torch-SDPA mode, its
    uniform-length trim and the cuDNN backend switch are ported (attention.py, training/modules/sdpa.py)

Ported beyond the objective: Automagic v3 with Fizgig's per-family parameter groups (driver.optimizer_params),
torch.compile of the blocks with Fizgig's Auto rules (compile.py), the fp8_scaled text encoder kept fp8 (embedder.py).
"""
