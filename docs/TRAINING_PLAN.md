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
| T2 | **Krea 2** | Fizgig's standalone Krea 2 trainer moved onto the family layer, without changing what it does | The owner's main training model; Fizgig's most mature recipe (measured presets, EMA, Turbo-LoRA previews, loss watch); shares much with Qwen 2.1 |
| T3 | **SDXL family**: SDXL, PonyXL, IllustriousXL, NoobAI | New `SDXLDriver` (ε- and v-prediction variants; dual CLIP; UNet input/middle/output block map; Conv2d LoRA) | The most-used older models; architecture facts from SDXL references, behaviour from Fizgig |
| T4 | **MiniMax H3, FLUX.2 Klein** | Fizgig's standalone trainers moved onto the family layer | The largest ports; done once the layer has carried Krea 2 |
| T5 | **Anima** (Cosmos-Predict2-2B finetune) | New `AnimaDriver` (DiT; reuses the Qwen-Image VAE path) | The least-used of the older families; architecture facts from Anima / Cosmos references |

**Reordered (2026-10-02):** the owner's main focus is Krea 2, so it moved from T4 to T2. The modern families (Krea 2,
Qwen 2.1, MiniMax H3) come first; the older families stay supported for the people who still use them.

## 3. Training UI in TagScribeR

The training UI is a new **Train** tab on the shared workspace:
- the dataset is the open folder or a collection, so captions, health flags and buckets feed straight in
- the family picker and preset system
- Fizgig's parameter groups, with the same names and defaults
- run queue, live loss chart, samples gallery and Problem Images window
- pause, stop and resume

TagScribeR's own captioners take over the loss watch's auto-recaption role, where Fizgig uses Qwen3-VL-4B.

## 4. Decisions

**Decided (2026-10-02):** all three recommendations accepted:
- caption shuffle and dropout as an optional extension, off by default
- build order T1 → T4 as listed (since reordered: see section 2)
- use `diffusers` for the SDXL family

The options as originally presented:

1. **Caption shuffle and dropout.** Fizgig caches text conditioning once per caption, so it has no tag shuffle or caption dropout. These are standard for SDXL-family tag datasets (Pony, Illustrious, NoobAI) and Anima. Options:
   - **(a) Strict Fizgig:** no shuffle.
   - **(b) Optional extension, off by default:** cache several shuffled variants per caption, keeping Fizgig's design untouched when it's off.

   Recommendation: **(b)**.
2. **Build order.** Is T1 → T4 above right, or should the SDXL family come first?
3. **Dependencies.** The SDXL driver needs `diffusers` (UNet, VAE, schedulers) unless the UNet is ported as plain PyTorch. Fizgig already uses diffusers for some paths. Recommendation: use `diffusers`.

## 5. Progress

- **T1 training core: done (2026-10-02).** `training/` holds the family interface and the generic loop. The loop
  adds gradient accumulation, batching for drivers that support it, Conv2d LoRA, the clip-ratio Adaptive LR signal
  and the preview-failure policy. The package also has:
  - caching, with the optional caption shuffle and dropout extension (off by default);
  - LoRA save and SAI metadata, EMA and the optimizers;
  - the loss watch, with TagScribeR's captioners for auto-recaption;
  - presets, the run builder and the control files;
  - the Qwen Image 2.1 family.

  This is verified by unit tests and CPU smoke tests on a tiny random family. The ported Qwen DiT and driver also
  run forward and backward on a 2-layer random config.
- **T1 Train tab: done (2026-10-02).** `tabs/train.py` covers:
  - the family picker, presets (load, save, delete, import, last run) and the model-file rows (with Get links);
  - the parameter groups, built from `training/params.py` with tooltips, and the auto-recaption captioner;
  - start, queue, pause, resume and stop, running the stages as QProcess child processes;
  - the loss chart (with the Adaptive LR marks), the samples gallery, the console and `run.log`;
  - Problem Images (caption fixes) and the preview override.

  It has Help Center topics, Ctrl+6, command-palette entries and a quit guard. It is verified by an offscreen
  test that runs the real child processes on the CPU test family, including pause and resume.
- **T2 Krea 2: done (2026-10-03), awaiting the owner's first real run.** `training/families/krea2/` ports Fizgig's Krea 2
  onto the family layer; the objective, conditioning, presets and recipe values are Fizgig's, cited per value.
  - *Ported:* the 12.9B DiT (28 blocks, text fusion, 3-axis RoPE) with SDPA attention, key-padding masks, gradient
    checkpointing and block swap; the Qwen-Image VAE; an encode-only Qwen3-VL-4B (12 hidden-state layers, fixed
    512-token cache of about 30 MB per caption, so batch size above 1 works); the flow-matching loss with the
    logit-normal, resolution-shifted timestep sampler and the `[min_t, max_t]` window; the Euler sampler with CFG;
    the all-Linear LoRA (264 modules) in Fizgig's kohya keys, and LoKR saved as `diffusion_model.*`; Turbo-LoRA
    previews on the training model (8 steps, strength 1.0, `mu` 1.15) including the file's `diff_b` bias deltas;
    the three built-in presets, EMA 0.98, and INT8 / NF4 bases with Fizgig's measured memory for Auto.
  - *Generic-layer changes:* kohya key writing in `lora.py`, `diff_b` bias deltas for frozen adapters, a
    `quant_target_names` hook (only the 28 main blocks are quantised, the LoRA covers more), legacy-key migration in
    `presets.apply` (`KREA2_EMA`, `QUANT_4BIT_MODE`, blank noise boxes), and Turbo steps in `pipeline.train_kwargs`.
  - *fp8 base (added 2026-10-05):* fp8 and fp8-scaled RAW files load as fp8 and are dequantised per matmul, a
    bf16 file can be quantised to fp8, and either feeds INT8 / NF4 (`training/modules/fp8.py`). Auto follows
    Fizgig's order: INT8, NF4, fp8, then fp8 with block swap, skipping what the machine cannot run.
  - *Not ported:* the fp8 `_scaled_mm` fast path (NVIDIA Ada and newer), torch.compile, the
    rotating-block full fine-tune and regularisation images, the fp8-Turbo-checkpoint preview engine, Automagic v3's
    per-family groups (so it is not offered here), captioning with the encoder, reference images and RefMods.
  - *Unverified without weights:* the real checkpoints' key layout, the fp8_scaled text-encoder dequantisation, the
    tokenizer download, the Turbo LoRA's key and `diff_b` naming, INT8 / swap memory at 12.9B, and bf16 memory.
    CPU tests cover the objective, masks, batching, timestep sampler, LoRA keys and round trips, the loader's refusals
    and the text encoder's hidden-state layout on tiny random models.

- **T3 SDXL family: done (2026-10-05), awaiting a first real run.** `training/families/sdxl/` adds SDXL 1.0, Pony
  Diffusion V6 XL, Illustrious-XL and NoobAI-XL (eps and v-pred). It is original code with no Fizgig counterpart; the
  loop, caching and timestep-window behaviour are Fizgig's.
  - One checkpoint file, loaded offline through diffusers. Dual-CLIP conditioning is cached at a fixed shape (three
    75-token chunks plus the pooled vector), so batch size above 1 works.
  - Epsilon and v-prediction objectives, with zero-terminal-SNR for v-pred. Euler previews with CFG.
  - LoRA keys in the kohya / LDM naming A1111 and ComfyUI load, checked against ComfyUI's mapping for all 760 target
    modules.
  - Optional extensions, off by default: LoCon (convolutions), min-SNR gamma, noise offset.
  - The presets are community starting points, not measured by Fizgig. Training memory is not measured.
- **T4 FLUX.2 Klein Base 9B: done (2026-10-05), awaiting a first real run.** `training/families/klein/` moves
  Fizgig's Klein recipe onto the family layer: all seven presets, Model Area block targeting, eight timestep modes
  (default `flux2_shift`), kohya keys, INT8 / NF4 bases. Not ported: accelerate, fp8, torch.compile, the Distilled
  preview model, LoRA dropout and LoRA+, logging back ends. One deviation: `shift` mode honours the shift box, where
  Fizgig's GUI never passes it.
- **T5 Anima: done (2026-10-05), experimental.** `training/families/anima/` is original code. Its architecture facts
  were verified against two trainers and ComfyUI (table in `description.py`). Rectified flow, Qwen3-0.6B plus T5
  token ids feeding the checkpoint's frozen LLM adapter, the Qwen-Image VAE, kohya keys. The presets are community
  starting points.
- **MiniMax H3: on hold for a licence decision (2026-10-05).** A still-image port was written and tested on CPU,
  but it is not in the repository. Fizgig's own headers describe its MiniMax DiT and VAE as faithful ports of
  ComfyUI's `comfy/ldm/minimax` code, and ComfyUI is GPL-3.0. Including that code would bring GPL terms with it.
  The options are: leave H3 out; re-implement the model from a permissively licensed reference; or ship those
  files under GPL-3.0 and accept what that means for distributing the app. The parameter definitions and the
  preset-key migration for H3 (`MINIMAX_*` in `training/params.py` and `training/presets.py`) are TagScribeR's own
  code and stay in, unused until a family lists them.
## 6. Verification policy

The owner runs all real training. Development verification uses:
- unit tests (adaptive-LR decisions on synthetic loss curves, preset validation, cache keys, LoRA key formats)
- dry-run config checks
- tiny CPU-only smoke tests with randomly initialised mini-models

No real model training is run during development.
