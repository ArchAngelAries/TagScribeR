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
  image_utils.py        Qt pixmap conversion + thumbnail decoding
  widgets.py            tag bubbles, flow layout, auto-tag dialog
inference/              all AI work; tabs talk to it only via specs / worker
  base.py               Provider interface, ProviderSpec, GenerationParams, errors
  transformers_vlm.py   local VLMs via Hugging Face Transformers (generic)
  openai_api.py         OpenAI-compatible servers (LM Studio, llama.cpp, Ollama, cloud)
  wd_tagger.py          WD (SmilingWolf) ONNX booru taggers
  models.py             model catalog + discovery (app models/, Stability Matrix, extra dirs)
  manager.py            caches loaded models; one heavy VLM resident at a time
  worker.py             QThread batch worker: failure-tolerant, cancellable, saves as it goes
  prompts.py            caption instruction presets
tabs/                   UI only
  common.py             shared widgets/helpers (thumbnail worker, collapsible section, dialogs)
  gallery.py caption.py editor.py datasets.py metadata.py settings.py help.py
tools/install.py        environment installer (GPU detection + pinned torch builds)
tests/                  pytest suite (core + inference, no GPU required)
```

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
