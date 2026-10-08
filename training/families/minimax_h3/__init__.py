"""MiniMax H3, image LoRA training only (ported from Fizgig: description, driver, DiT, video VAE, sampler, Qwen3-VL-32B).

Fizgig (https://github.com/shootthesound/Fizgig, Apache-2.0) is the reference for every training behaviour here: the
flow-matching objective (target x0 - noise), the low-noise % dial -> schedule shift, the 20-49 block window, the frozen
training adapter, caption dropout, EMA, the kohya LoRA keys and the presets. See docs/TRAINING_PLAN.md and
THIRD_PARTY_NOTICES.md.

Not ported yet (Fizgig has them; this family does not): video clips, audio / voice items, reference images (RefMods),
multi-concept, reference distillation, sliders, the rotation full fine-tune, TREAD token routing and "clip still" (clip
features), HQQ and the H2D streaming rings, the int8 attention kernel and the 66 GB bf16 base. See docs/dev/PORT_PLAN.md.
"""
