# Third-party notices

TagScribeR as a whole is licensed under the GNU General Public License v3.0 (see [LICENSE](LICENSE)). It includes code adapted from the projects below; Apache-2.0 and MIT code may be combined into a GPL-3.0 work, and the notices here are kept as those licences require.

Each component stays under its upstream licence; the
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
| `training/modules/offloading.py`, `int8_train.py`, `nf4.py`, `sdpa.py`, `compile_util.py` | `src/fizgig/modules/` |
| `training/modules/fp8.py` | `src/fizgig/krea2/fp8_optimization_utils.py`, `krea2/utils.py` (adapted by Fizgig from musubi-tuner) |
| `training/families/qwen_image21/` | `src/fizgig/families/qwen_image.py` and `src/fizgig/qwen_image21/` |
| `training/families/krea2/` | `src/fizgig/krea2/` (`model.py`, `sampling.py`, `vae.py`, `vae_loader.py`, `embedder.py`, `trainer.py`, `caching.py`, `utils.py`), `lora_trainer_gui.py` (presets, model paths), `scripts/fetch_models.py`, `utils/capabilities.py` (measured memory) |

The training presets, parameter names and defaults (`training/params.py`) follow Fizgig's Training tab.

Not copied: Fizgig's `detect_gpu.py` is GPL-3.0 (from comfyui-rocm) and is deliberately **not** included.
TagScribeR detects GPUs with its own `core/hardware.py`.

## Krea 2 (ai-toolkit, musubi-tuner, diffusers)

`training/families/krea2/` is Fizgig's Krea 2 code, ported, and carries its upstream credits:

- **ai-toolkit (Ostris, LLC), MIT** (the licence text is in the ai-toolkit section below): the single-stream MMDiT
  backbone (`model.py`, from `extensions_built_in/diffusion_models/krea2/src/mmdit.py`) and the functional
  flow-matching sampler (`sampling.py`, from `extensions_built_in/diffusion_models/flux2/src/sampling.py`).
- **musubi-tuner (kohya-ss and contributors), Apache License 2.0**: the training hooks in `model.py` (gradient
  checkpointing, block-swap wiring), the flow-matching timestep recipe (`krea2_shift`) in `sampling.py`, the attention
  dispatch and length trim in `attention.py`, and the Qwen-Image VAE key conversion and loader in `vae_loader.py`.
- **Diffusers / the Qwen-Image Team, Wan Team and The HuggingFace Team, Apache License 2.0**: `vae.py` is Fizgig's copy
  of diffusers' `AutoencoderKLQwenImage`, kept with its original header.

The Krea 2, Qwen-Image VAE and Qwen3-VL-4B weights are not included; the publishers' licences apply when you download
them.

## MiniMax H3 (Fizgig, with code that derives from ComfyUI)

`training/families/minimax_h3/` ports Fizgig's MiniMax H3 trainer for still-image LoRA training
(`src/fizgig/minimax/`). Its training recipe follows Ostris's ai-toolkit (MIT), which is listed below.

Provenance to be aware of: Fizgig's own file headers describe its H3 model and VAE code as faithful ports of
ComfyUI's `comfy/ldm/minimax/model.py` and `vae.py`, and the sampler's res_multistep step follows
`comfy/k_diffusion`. ComfyUI (https://github.com/comfyanonymous/ComfyUI) is licensed under the GNU General Public
License v3.0. `training/families/minimax_h3/model.py`, `vae.py` and `sampling.py` carry that provenance in their
headers. This concerns the source code only. It is separate from the licence of the MiniMax H3 model weights, which
are not included and whose terms the MiniMax team sets.

## FLUX.2 Klein (Black Forest Labs FLUX): Apache License 2.0

Upstream: https://github.com/black-forest-labs/flux. Copyright Black Forest Labs.

`training/families/klein/model.py` and `vae.py` are ported from Fizgig's Klein model code (`src/fizgig/klein/`), which
is based on FLUX's reference model code. `driver.py`, `sampling.py` and `embedder.py` port Fizgig's Klein trainer
recipe (`src/fizgig/training/trainer.py`, `src/fizgig/networks/lora_klein.py`).

Model weights are not included and have their own licences (FLUX.2 Klein Base 9B and the FLUX.2 autoencoder are gated
by Black Forest Labs).

## SDXL family (diffusers, kohya-ss sd-scripts): Apache License 2.0

`training/families/sdxl/` is original TagScribeR code with no Fizgig counterpart. The loop, the caching design and the
timestep-window semantics around it are Fizgig's. It uses and references:

- **diffusers** (Apache-2.0, https://github.com/huggingface/diffusers): the UNet, VAE and CLIP model classes and the
  single-file checkpoint converters are called as a library, and the Euler and DDPM formulas follow its schedulers.
  The vendored UNet, VAE and text-encoder configs come from `stabilityai/stable-diffusion-xl-base-1.0`.
- **kohya-ss/sd-scripts** (Apache-2.0, https://github.com/kohya-ss/sd-scripts): the LoRA target lists, the kohya key
  format, and the min-SNR and noise-offset formulas.
- **ComfyUI** (GPL-3.0) was read only as a reference for LoRA key names. No ComfyUI code is included.
- Zero-terminal-SNR follows Lin et al. 2023, "Common Diffusion Noise Schedules and Sample Steps are Flawed".

## Anima (kohya-ss sd-scripts, NVIDIA Cosmos-Predict2): Apache License 2.0

`training/families/anima/` is original TagScribeR code with no Fizgig counterpart. Its model is written to load the
Anima checkpoint's own layout, and its facts come from:

- **kohya-ss/sd-scripts** (Apache-2.0): the architecture, tokenisation, training objective and LoRA key facts. The
  attention and adapter structure is re-implemented, not copied.
- **NVIDIA Cosmos-Predict2** (Apache-2.0, Copyright 2025 NVIDIA): module and parameter names and the RoPE formula,
  re-implemented.
- **diffusion-pipe** and **ComfyUI** (GPL-3.0) were read for facts only. No code from either is included.

The VAE is the Qwen-Image VAE already listed under Krea 2. Anima's weights are under the circlestone-labs
non-commercial licence.

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
