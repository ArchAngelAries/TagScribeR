"""Native LoRA training for TagScribeR.

A port of Fizgig's standard family layer (https://github.com/shootthesound/Fizgig, Apache-2.0, (c) 2026 Peter
Neill); Fizgig is the reference for every training behaviour here: the family interface, the generic loop, caching,
LoRA format and metadata, Adaptive LR, the per-image loss watch, presets, optimizers, EMA and the subprocess /
file-sentinel run contract. See docs/TRAINING_PLAN.md and docs/FIZGIG_TRAINING_AUDIT.md.

Layout
  description.py driver.py registry.py   family facts + model-code interface + the family list
  families/<family>/                      one package per model family (description, driver, model code)
  dataset.py cache.py                     bucketing, cache files, `python -m training.cache`
  train.py                                the generic loop, `python -m training.train`
  lora.py quant.py modules/               LoRA/LoKR adapters (Linear + Conv2d), INT8/NF4 bases, block swap
  adaptive_lr.py loss_logger.py loss_watch.py optimizers.py ema.py automagic3.py
  metadata.py progress.py train_utils.py
  params.py presets.py pipeline.py        UI-independent parameter schema, presets, run builder / control files

Nothing here imports Qt. Modules that need torch import it themselves; params / presets / pipeline / registry do
not, so the app can build runs without loading torch.
"""
