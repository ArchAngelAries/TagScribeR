# Third-party notices

TagScribeR includes code adapted from the projects below. Each component stays under its upstream licence; the
full Apache License 2.0 text is in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt). Every ported file carries a
header naming its source file and the changes made for TagScribeR.

---

## Fizgig: Apache License 2.0

Upstream: https://github.com/shootthesound/Fizgig

```
Fizgig - Klein 9B LoRA Trainer & Workbench
Copyright 2026 Peter Neill

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

TagScribeR's native training package (`training/`) is a port of Fizgig's standard family layer and the modules it
uses:

| TagScribeR | Fizgig source |
|---|---|
| `training/description.py`, `driver.py`, `registry.py` | `src/fizgig/families/description.py`, `driver.py`, `registry.py` |
| `training/train.py`, `cache.py`, `lora.py`, `quant.py`, `loss_watch.py` | `src/fizgig/families/train.py`, `cache.py`, `lora.py`, `quant.py`, `loss_watch.py` |
| `training/adaptive_lr.py` | `src/fizgig/krea2/trainer.py` (`AdaptiveLR`) + the clip-ratio signal of `src/fizgig/training/trainer.py` |
| `training/dataset.py` | `src/fizgig/dataset/image_dataset.py`, `config.py` (image-only subset) |
| `training/ema.py`, `optimizers.py`, `metadata.py`, `progress.py`, `train_utils.py`, `loss_logger.py` | the same-named modules in `src/fizgig/training/` |
| `training/modules/offloading.py`, `int8_train.py`, `nf4.py` | `src/fizgig/modules/` |
| `training/families/qwen_image21/` | `src/fizgig/families/qwen_image.py` and `src/fizgig/qwen_image21/` |

The training presets, parameter names and defaults (`training/params.py`) follow Fizgig's Training tab.

Not copied: Fizgig's `detect_gpu.py` is GPL-3.0 (from comfyui-rocm) and is deliberately **not** included.
TagScribeR detects GPUs with its own `core/hardware.py`.

## Diffusers / Qwen-Image 2.1: Apache License 2.0

Upstream: https://github.com/huggingface/diffusers. Copyright 2026 The Qwen Team, the Qwen-Image Team and The
HuggingFace Team.

`training/families/qwen_image21/model.py` (the Qwen Image 2.1 transformer) and `vae.py` (its VAE) were ported from
diffusers by Fizgig (`transformer_qwenimage21.py`, `autoencoder_kl_qwenimage21.py`) and are carried over with their
headers. The sampler follows diffusers' `QwenImage21Pipeline` / `FlowMatchEulerDiscreteScheduler`.

Model weights are not included and have their own licences: Qwen Image 2.1 is under the Qwen Research License
(non-commercial).

## musubi-tuner: Apache License 2.0

Upstream: https://github.com/kohya-ss/musubi-tuner. Copyright the musubi-tuner authors (kohya-ss and contributors).

Fizgig credits musubi-tuner for its block-swap offloader design. `training/modules/offloading.py` follows the same
design. Fizgig's audit lists it as kohya / musubi lineage by inspection.

## ai-toolkit (Ostris, LLC): MIT License

Upstream: https://github.com/ostris/ai-toolkit

`training/automagic3.py` (the Automagic v3 optimizer) is kept as upstream's `toolkit/optimizers/automagic3.py`,
vendored by Fizgig, under a header describing how it is constructed.

```
MIT License

Copyright (c) 2024 Ostris, LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
