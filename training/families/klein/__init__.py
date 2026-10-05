"""FLUX.2 Klein Base 9B (ported from Fizgig: description, driver, DiT, AutoEncoder, sampler, Qwen3-8B text encoder).

Fizgig (https://github.com/shootthesound/Fizgig, Apache-2.0) is the reference for every training behaviour here: the
flow-matching objective and the timestep modes, the Qwen3-8B conditioning (layers 9 / 18 / 27), the Model Area block
targeting, the kohya LoRA keys, the seven built-in presets and the measured memory figures. See docs/TRAINING_PLAN.md
and THIRD_PARTY_NOTICES.md.

Not ported (Fizgig has them; this family does not):
  * accelerate (the loop is the generic one) and torch.compile
  * the fp8 _scaled_mm fast path (NVIDIA Ada and newer). The fp8 base itself IS supported: the fp8 Base file
    loads as fp8 and is dequantised per matmul (training/modules/fp8.py), and feeds the INT8 / 4-bit bases
  * TensorBoard / wandb logging, LoRA neuron / rank / module dropout, LoRA+ (Fizgig's GUI keeps it at 1), weight-norm
    scaling, regularisation images
  * the Distilled preview model (4 steps; a second ~9 GB model load): previews run on the resident Base model at 40
    steps, CFG 4.5 (Fizgig's no-Distilled path); reference-image (edit) previews and edit training
  * the SD3 loss weightings (WEIGHTING_SCHEME / MODE_SCALE: hidden in Fizgig's Klein GUI, always "none") and dataset
    timestep buckets
  * the Repair Studio activation cache, the profiler and extraction, Fizgig's flash / sage / xformers attention
"""
