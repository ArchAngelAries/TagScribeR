# TagScribeR native training plan

**Ground rule (from the project owner):** Fizgig is the grounded truth for how training works in TagScribeR. That covers the training features, presets (built-in and user-created), adaptive learning rate, parameter surface, efficiency and memory features, and optional sample previews. Other tools (kohya sd-scripts, diffusers, ai-toolkit) are consulted only for model-architecture facts that Fizgig has no code for, never to override Fizgig's design.

The source analysis is in [FIZGIG_TRAINING_AUDIT.md](FIZGIG_TRAINING_AUDIT.md), with file and line citations into Fizgig.

## 1. Architecture: port Fizgig's standard family layer

Fizgig has three generations of trainer code. The newest, the standard family layer in `src/fizgig/families/` (used by Qwen-Image 2.1), is the clean template. A model family is a `FamilyDescription` (facts) plus a `FamilyDriver` (code). The driver owns the training objective, so epsilon- and v-prediction models fit without changing the loop.

TagScribeR gets a `training/` package that ports that layer. It's Apache-2.0, ported with attribution headers. `detect_gpu.py` is GPL-3.0 and is never copied.

| TagScribeR module | Ported from Fizgig | Notes |
|---|---|---|
| `training/description.py`, `driver.py` | `families/description.py`, `driver.py` | Same interface |
| `training/train.py` | `families/train.py` (generic loop) | Extended with gradient accumulation, batch > 1 and Conv2d LoRA (needed for SDXL) |
| `training/cache.py` | `families/cache.py` | Latent and text-encoder caches, same naming scheme |
| `training/lora.py` | `families/lora.py` | Plus Conv2d targets; ComfyUI-compatible keys and SAI metadata |
| `training/adaptive_lr.py` | `krea2/trainer.py: AdaptiveLR` | Faithful port, including Klein's clip-ratio signal |
| `training/loss_watch.py` | `families/loss_watch.py` + `training/loss_logger.py` | Per-image loss watch; auto-recaption uses TagScribeR's captioners |
| `training/optimizers.py`, `ema.py`, `quant.py`, `metadata.py`, `progress.py` | Same-named Fizgig modules | |
| `training/presets.py` | Fizgig preset lifecycle | Built-ins in code; user presets in `user_data/training_presets/<family>/*.json`; strict validation; unknown keys ignored |

**Process model (same as Fizgig).** The UI launches cache-latents → cache-text → train as child processes and reads their progress from stdout. This keeps VRAM and crashes isolated from the app. Control is file-based through the run folder: a pause sentinel, a sample override, problem-image and caption-edit files. Stop kills the process tree.

**Adaptive LR (exact Fizgig behaviour).**
- The user sets min and max LR; the run starts at √(min·max).
- At the end of each epoch it compares the epoch loss and the LoRA weight-norm growth (plus the clip ratio where clipping is on).
- Probe up ×1.25 after 2 improving epochs, and reduce ×0.5 on a plateau (patience 1 in epochs 2–3, otherwise 2).
- On instability (weight growth > 30%), blend the weights 70/30 back toward the previous snapshot and restore the optimizer state.
- The step scheduler is forced to constant while it's on, and self-scheduling optimizers disable it.

**Presets (exact Fizgig behaviour).** Built-in presets per family, a first-visit default, and user presets saved as flat JSON without a family field. Every parameter Fizgig exposes is carried (audit §4.4). Values a family doesn't support are refused, not silently changed. The last run is snapshotted for resume.

**Sample previews (exact Fizgig behaviour).** Previews render at the configured cadence with the adapter on and EMA swapped in. They park VRAM on low-memory cards, accept an override file, and use the same file naming. A preview failure never kills the run.

## 2. Model families and build order

| Phase | Family | Basis | Why this order |
|---|---|---|---|
| T1 | Training core + **Qwen-Image 2.1** | Direct port of Fizgig's standard layer and its existing family | Proves the port against a family Fizgig already runs on that layer |
| T2 | **SDXL family**: SDXL, PonyXL, IllustriousXL, NoobAI | New `SDXLDriver` (ε- and v-prediction variants; dual CLIP; UNet input/middle/output block map; Conv2d LoRA) | The most-requested older models; architecture facts from SDXL references, behaviour from Fizgig |
| T3 | **Anima** (Cosmos-Predict2-2B finetune) | New `AnimaDriver` (DiT; reuses the Qwen-Image VAE path) | Architecture facts from Anima / Cosmos references |
| T4 | **Krea 2, MiniMax H3, FLUX.2 Klein** | Fizgig's standalone trainers moved onto the family layer | The largest ports; done once the shared layer is proven |

## 3. Training UI in TagScribeR

The training UI is a new **Train** tab on the shared workspace:
- the dataset is the open folder or a collection, so captions, health flags and buckets feed straight in
- the family picker and preset system
- Fizgig's parameter groups, with the same names and defaults
- run queue, live loss chart, samples gallery and Problem Images window
- pause, stop and resume

TagScribeR's own captioners take over the loss watch's auto-recaption role, where Fizgig uses Qwen3-VL-4B.

## 4. Decisions for the owner

1. **Caption shuffle and dropout.** Fizgig caches text conditioning once per caption, so it has no tag shuffle or caption dropout. These are standard for SDXL-family tag datasets (Pony, Illustrious, NoobAI) and Anima. Options:
   - **(a) Strict Fizgig:** no shuffle.
   - **(b) Optional extension, off by default:** cache several shuffled variants per caption, keeping Fizgig's design untouched when it's off.

   Recommendation: **(b)**.
2. **Build order.** Is T1 → T4 above right, or should the SDXL family come first?
3. **Dependencies.** The SDXL driver needs `diffusers` (UNet, VAE, schedulers) unless the UNet is ported as plain PyTorch. Fizgig already uses diffusers for some paths. Recommendation: use `diffusers`.

## 5. Verification policy

The owner runs all real training. Development verification uses:
- unit tests (adaptive-LR decisions on synthetic loss curves, preset validation, cache keys, LoRA key formats)
- dry-run config checks
- tiny CPU-only smoke tests with randomly initialised mini-models

No real model training is run during development.
