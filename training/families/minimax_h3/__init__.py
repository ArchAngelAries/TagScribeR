"""MiniMax H3, image LoRA training only (ported from Fizgig: description, driver, DiT, video VAE, sampler, Qwen3-VL-32B).

Fizgig (https://github.com/shootthesound/Fizgig, Apache-2.0) is the reference for every training behaviour here: the
flow-matching objective (target x0 - noise), the low-noise % dial -> schedule shift, the 20-49 block window, the frozen
training adapter, caption dropout, EMA, the kohya LoRA keys and the presets. See docs/TRAINING_PLAN.md and
THIRD_PARTY_NOTICES.md.

Not ported (Fizgig has them; this family does not): video clips, audio / voice items, reference images (RefMods),
multi-concept, reference distillation, the rotation full fine-tune, TREAD token routing (Fizgig runs it on clip steps
only, never on stills), "clip still" (a clip feature), the high-noise LR % dial (Fizgig never applies it under Automagic
v3, the preset optimiser), HQQ / NF4-streamed H2D rings and the int8 attention kernel, the Turbo-LoRA previews, the
adapter ramp, the movement limiter and the 66 GB bf16 base.
"""
