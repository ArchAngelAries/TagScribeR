# TagScribeR architecture

TagScribeR is a PySide6 desktop app for image-dataset work: browsing, caption
and tag editing, AI captioning/tagging, image prep, and metadata. It runs from
a git checkout with a project-local virtual environment.

```
main.py                 entry point: env setup → logging → theme → MainWindow
core/                   UI-independent foundation (no Qt widgets, no torch at import)
  paths.py              app-root-relative paths; user state lives in user_data/
  config.py             versioned, validated settings (user_data/settings.json)
  secrets.py            API profiles; keys in Windows Credential Manager via keyring
  log.py                rotating log file (user_data/logs/) + console
  hardware.py           ROCm / CUDA / CPU detection, dtype selection, ROCm env defaults
  captions.py           pure tag/caption transforms (split, dedupe, merge, normalize…)
  caption_io.py         atomic caption reads/writes with daily backups
  fileops.py            collision-safe copies, Recycle Bin deletes
  dataset.py            natural-sorted folder scans, header-only image info
  image_utils.py        Qt image conversion + thumbnail decoding (QImage is thread-safe)
  image_ops.py          Pillow transforms for the editor; saves keep ICC/EXIF, atomic
  metadata.py           metadata audit + lossless JPEG/PNG/WebP writers, prompt extraction
  health.py             duplicates (hash + dHash), blur, resolution, aspect buckets
  dataset_session.py    in-memory dataset: dirty tracking, change-set transforms, saves
  query.py              filter language (tag:, missing:, res:, ar:, flag: …)
  presets.py            built-in + user presets (caption, tagger, authorship)
  projects.py           per-folder settings (subject, preset, filters) in user_data/projects
  widgets.py            tag bubbles, flow layout, auto-tag dialog
inference/              all AI work; tabs talk to it only via specs / worker
  base.py               Provider interface, ProviderSpec, GenerationParams, errors
  transformers_vlm.py   local VLMs via Hugging Face Transformers (generic)
  openai_api.py         OpenAI-compatible servers (LM Studio, llama.cpp, Ollama, cloud)
  wd_tagger.py          WD (SmilingWolf) ONNX booru taggers
  models.py             model catalog + discovery (app models/, Stability Matrix, extra dirs)
  manager.py            caches loaded models; one heavy VLM resident at a time
  worker.py             QThread batch worker: failure-tolerant, cancellable, saves as it goes
  prompts.py            caption instruction presets + per-image prompt builder (subject, tag hints)
training/               native LoRA training: a port of Fizgig's standard family layer (no Qt; see below)
  description.py driver.py registry.py   family facts, the model-code interface, the family list
  families/qwen_image21/                  Qwen Image 2.1: description + presets, driver, DiT, VAE, sampler, encoder
  dataset.py cache.py                     buckets, cache files, `python -m training.cache`
  train.py                                the generic loop, `python -m training.train`
  lora.py quant.py modules/               LoRA/LoKR (Linear + Conv2d), INT8/NF4 bases, block swap
  adaptive_lr.py loss_logger.py loss_watch.py optimizers.py ema.py automagic3.py metadata.py progress.py
  params.py presets.py pipeline.py        parameter schema, presets, run builder + control files (torch-free)
tabs/                   UI only
  common.py             shared helpers (background tasks, quick-tag list, collapsible section, dialogs)
  workspace/            shared dataset workspace used by every tab
    context.py          WorkspaceContext: open folder, model, thumbnails, undo, saving, job/health state
    browser.py          DatasetBrowser: toolbar + virtualized grid + filters + context menu
    model.py delegate.py thumbs.py commands.py jobs.py panels.py
  gallery.py caption.py editor.py datasets.py metadata.py settings.py
  review.py             AI caption review window (diff, accept/reject)
  palette.py            Ctrl+K command palette
  help.py help_content.py   Help Center (searchable, context-aware)
tools/install.py        environment installer (GPU detection + pinned torch builds)
tests/                  pytest suite (core + inference, no GPU required)
```

## Shared workspace

`tabs/workspace/context.py` holds the one open dataset. Every tab embeds its own
`DatasetBrowser`, with its own selection, filter and zoom, over the shared list
model, so edits, unsaved state and undo history stay consistent everywhere.
Captions written by AI jobs are picked up through `reload_from_disk`; entries the
user is editing are never overwritten. Per-folder preferences live in
`core/projects.py`. Global preferences stay in settings.json.

## Key design rules

**Data safety.** Every caption write goes through `core.caption_io.write_caption`:
atomic temp-file-and-replace, no empty files created, unchanged text not
rewritten, and the first version of each overwritten caption per day copied to
`user_data/caption_backups/<date>/`. Deletes go to the Recycle Bin
(`core.fileops`). Copies never overwrite; collisions get ` (n)` suffixes.
Image edits default to copies in `Image Edits/<source folder>/`.

**Hardware neutrality.** ROCm PyTorch exposes AMD GPUs through `torch.cuda`,
so device strings are `cuda:N` on both vendors. `core.hardware` detects the
backend (`torch.version.hip`) and probes capabilities: bf16 support picks
bf16 vs fp16, and SDPA attention is used everywhere (AOTriton kernels on
ROCm). Optional accelerators (flash-attn, bitsandbytes, ONNX GPU providers)
are used only if importable, never required. `apply_runtime_env()` sets the
ROCm defaults (MIOpen fast find, AOTriton experimental kernels, allocator
config, pip ROCm SDK paths) before torch loads, and never overrides
user-set variables.

**Providers.** A provider turns images plus a `CaptionRequest` into one
caption, or an exception, per image. Capabilities are explicit (`CAP_PROMPT`,
`CAP_TAGS`, …). Adding a backend means adding a provider and one line in
`manager.create_provider`. The UI is unchanged.

**Model lifetime.** `ModelManager` caches providers by spec, ignoring
per-call options such as batch size, so repeated runs reuse the loaded model.
Switching VLMs unloads the previous one first. A job holds a manager
*session*, so "Free VRAM" can't unload a model mid-batch. Idle unloading is
configurable.

**Batch jobs.** `BatchWorker` decodes the next chunk while the GPU works on
the current one. It saves each result immediately, which keeps completed work
on cancel or crash. It records per-image failures without aborting and
reports them at the end. OOM in a batch retries one image at a time.
Cancellation is checked inside generation through a stopping criterion.

## Training

`training/` ports Fizgig's standard family layer. Fizgig is the reference for all training behaviour: see
[TRAINING_PLAN.md](TRAINING_PLAN.md) and [FIZGIG_TRAINING_AUDIT.md](FIZGIG_TRAINING_AUDIT.md). Every ported file
carries an Apache-2.0 attribution header, listed in `THIRD_PARTY_NOTICES.md`.

**Families.** A model family is a `FamilyDescription` and a `FamilyDriver`:
- The description holds the facts: model files, latent rules, the LoRA key format, presets, sampling recipes and
  measured memory.
- The driver holds the model code: loading, encoding, the training objective, sampling and the block map.
- The loop never sees noise schedules, so flow-matching, epsilon and v-prediction models all fit.

Adding a family means a package under `training/families/` plus one line in `registry.py`.

**Process model.**
- The app process never trains. The Train tab freezes a run folder: `dataset.json`, `train_config.json` and the
  preview prompts.
- It then starts three child processes in turn, with this interpreter: `training.cache --stage latents`, then
  `--stage text`, then `training.train`. It reads their stdout (`training.progress` parses the lines).
- Control goes through files in the run folder:
  - `.pause_requested`: save state at the next epoch boundary and exit 0;
  - `.sample_override.json`: the prompt for the next preview;
  - `loss_log/caption_updates.json`: caption fixes, applied at the next boundary.
- Stop kills the process tree.

**Data.** Training reads caches, never images. Cache files use Fizgig's naming in
`user_data/training_cache/<folder>-<hash>/`. A re-cache skips latents that are still valid and captions that haven't
changed, and it deletes the caches of images that are gone.

**Presets.** Built-in presets come from the family description. User presets are flat JSON in
`user_data/training_presets/<family>/`, with Fizgig's keys, so Fizgig presets import as they are. They are applied
with Fizgig's rules:
- unknown keys are ignored;
- values match their option by first token;
- strict choices are refused when the family doesn't offer them.

**Verification.** `tests/tiny_family.py` is a tiny random-weight family. The smoke tests run caching, training,
pause/resume, previews and the loss watch on CPU in seconds. No real model is trained in development.

## Environment

The environment matches the Fizgig trainer so both apps share one known-good AMD stack:

* Python 3.12. Packages are installed with `uv`.
* AMD on Windows: AMD's pinned multi-arch ROCm nightlies,
  `torch[device-gfx1100]==2.12.0+rocm7.15.0a20260728`.
* NVIDIA: CUDA 12.8 wheels. No GPU: CPU wheels.

The difference is Transformers. Fizgig pins 4.57.x for its Qwen3-VL encoder.
TagScribeR needs 5.x, because every 2026 VLM family requires it: Qwen3.5,
3.6 and 3.8 (`qwen3_5`) and Gemma 4 (`gemma4`). Qwen3-VL still works in 5.x.

## Known AMD notes

* Qwen3.5-family models use Gated DeltaNet layers. Their fast kernels
  (`flash-linear-attention`, `causal-conv1d`) have no ROCm builds, so
  Transformers uses its slower PyTorch reference path. The models run
  correctly. Batch size 2–4 recovers most of the throughput. Don't install
  `flash-linear-attention` on ROCm: it has produced garbage output.
* The first generation after loading a model is slow (~20 s) while MIOpen and
  AOTriton select kernels. Later images take about 1 s.
* FP8 / AWQ / GPTQ checkpoints generally need CUDA kernels. Use bf16, or GGUF
  quants served by LM Studio / llama-server through the API provider.
* `onnxruntime` is CPU-only by default. Installing `onnxruntime-directml`
  runs the WD tagger on the GPU on Windows.
