# Speed, memory and responsiveness: improvement ideas

Consulting notes, 2026-10-19. Read-only review of the code on `port/stage1-bt67en` plus Fizgig v8.7.1 (`.claude/fizgig-src`,
738 commits past the port baseline `1c8ec88`). Development notes, not user documentation; nothing here is decided.
The standing rules apply: keep what works, Fizgig is the training reference, anything that could change what a run
learns is opt-in or proven bit-identical, and a GPU claim is an estimate until the owner has run it.

How to read an idea: **what** / **why** (mechanism, with a measurement where one exists and *(estimate)* where not) /
**risk** / **effort** S (hours), M (a day or two), L (a week+) / **files** / **verify** (what a CPU test or benchmark
proves, and what only a real run on the owner's AMD card or an NVIDIA user's card can confirm).

Tiers: **(a) safe wins** change no training numerics and no defaults that affect output; **(b) architectural bets**
are larger pieces of work that pay back over time; **(c) speculative** needs a measurement or a platform answer first.

---

## 0. What was measured (CPU, this session)

Two throwaway CPU scripts (generated synthetic data only, not committed; worth turning into `tools/bench_app.py` and
`tools/bench_train.py` so the owner can run them on Windows). Linux sandbox, 4 cores, project venv (torch 2.14 CPU).
Absolute numbers will differ on the owner's machine; the *ratios* are the point.

| What | Measured | Note |
|---|---|---|
| `DatasetSession.open`, 10k PNGs + 6.7k captions | 519 ms | `core/dataset_session.py:66-77`; on the UI thread (`tabs/workspace/context.py:120`) |
| one `scandir` + `DirEntry.stat` + caption presence from the listing | 403 ms | 1.3x here; on Windows `DirEntry.stat()` is free (comes with the listing) so the gap widens |
| sort 10k entries, `natural_key` recomputed per comparison (what `FilterProxy.lessThan` does) | 58 ms | `tabs/workspace/model.py:114-118` |
| same with one cached key per entry | 17 ms | 3.3x |
| thumbnail of a 4000x3000 JPEG, current PIL path (`load_qimage`, 384 tier) | 103-124 ms | `core/image_utils.py:48-55` |
| same via `QImageReader.setScaledSize` (Qt's libjpeg, scaled decode) | 65-75 ms | 1.6x |
| 768 tier (a second full decode when the user zooms in) | 247-312 ms | `tabs/workspace/thumbs.py:27,105-119` |
| 384 tier derived from the cached 768 JPEG instead | 17 ms | 7x |
| 4000x3000 PNG thumbnail (no draft mode possible) | 428-532 ms | decode-bound; same for WebP (525-742 ms) |
| thumbnail disk cache read (384 JPEG) / write | 0.8-1.3 ms / 3.6-6.9 ms | cache hits are already cheap |
| `health.analyze` of one 4000x3000 JPEG / PNG | 63 ms / 843 ms | `core/health.py:38-57`; of the PNG: 73 ms blake2b of 24 MB + 409 ms decode |
| dhash + Laplacian from the 384 thumbnail cache instead | 1.9-2.3 ms | 30-180x on the decode part |
| size + first/last 1 MB fingerprint instead of hashing the whole file | 3.6-4.9 ms | full hash only for files sharing a fingerprint |
| `health.analyze` x8, serial vs 4 threads | 1263 ms vs 692 ms | 1.8x on 4 cores (`tabs/workspace/jobs.py:37-57` runs one at a time) |
| `write_caption` x300 (atomic + fsync each) | 1029 ms | `core/caption_io.py:64-78,100-116` |
| same on 8 threads | 224 ms | 4.6x; Windows NTFS fsync is far dearer than this sandbox's, so expect more |
| WD tagger post-processing loop, 10861 tags x 8 images | 76 ms | `inference/wd_tagger.py:185-198` |
| same vectorised with numpy (identical tag lists) | 12 ms | 6.4x |
| Krea 2 text cache per caption: as stored (512 padded tokens) | 31.5 MB | `training/families/krea2/embedder.py:138-153` |
| same trimmed to the valid tokens (71 of 512) | 4.4 MB | 7.2x smaller; `gather_valid_text` output **bit-identical** (also for a batch of 2 with 71 and 130 tokens) |
| `get_batch` cost with a warm page cache | 1.0 ms (full) / 0.15 ms (trimmed) | the per-step *disk* cost is negligible once the OS has the files cached |
| one-step-ahead prefetch over a simulated 300 ms step | 303 -> 300 ms | no win with a warm cache; only cold reads / slow disks / datasets larger than RAM benefit |
| fp8 dequant 6144x6144, current fp32 temp vs direct bf16 | 1494 ms vs 491 ms (CPU) | 3x less memory traffic, but **not bit-identical**: 22% of weights differ by one bf16 ulp (max rel 7.8e-3) |

Things only a real GPU run can give: the split of the 2.5 min/epoch preview time, whether `expandable_segments` is
honoured by the ROCm Windows build, INT8 vs fp8 vs bf16 matmul speed at Krea 2's shapes, and every s/step figure.

---

## 1. The app: startup, folder loading, thumbnails, memory, responsiveness, batch operations

### A1. Dataset health scan: 10-20x by reusing what is already decoded *(a, S-M)*
**What.** `core/health.analyze` decodes every image a second time (`core/health.py:47-54`), hashes the whole file
(`:42-46`), and `tabs/gallery.py:199-209` runs it through a `FileJob`, one image at a time. Instead: (1) take the
grey image for dHash and sharpness from the thumbnail cache (`tabs/workspace/thumbs.py:_disk_path`) when a 384 tier
exists (it does for every card the user has seen; the scan can request the rest through the same loader);
(2) exact duplicates in two stages: group by (file size, blake2b of the first and last 1 MB), hash the whole file
only inside groups of two or more; (3) run `analyze` on a thread pool (PIL decode and blake2b release the GIL).
**Why.** Measured: 843 ms -> ~5 ms per 4000x3000 PNG for the decode part; 73 -> 5 ms for the hash; 1.8x more from
4 threads. A 10k PNG dataset goes from ~2.3 h to a few minutes *(estimate)*.
**Risk.** Sharpness is computed on a 384 px grey image instead of 512: the blur verdict is relative to the dataset
median (`health.py:162-169`), so verdicts stay consistent within a dataset but can move against the old absolute
floor of 40. Keep the old path for images with no cached thumbnail and compute both once on a demo set to pick the
floor. dHash at 384 vs 512 is a wash (it resizes to 9x8 anyway).
**Files.** `core/health.py`, `tabs/gallery.py:178-211`, `tabs/workspace/jobs.py` (a pooled `FileJob` variant),
`tabs/help_content.py` (what the scan reads).
**Verify.** `tests/test_health.py` with generated images: same duplicate groups, same near-duplicate groups, blur
verdicts within tolerance; a benchmark line in `tools/bench_app.py`. Nothing GPU-related.

### A2. Thumbnail pipeline: decode once, decode less, decode on more cores *(a, S-M)*
**What.** Four independent pieces, all in `tabs/workspace/thumbs.py` / `core/image_utils.py`:
1. *One decode per image*: when the 384 tier is requested and no cache exists, decode once at 768, write both tier
   files, and produce the 384 from the 768 in memory (`QImage.scaled`, 17 ms vs a second 250-300 ms decode when the
   user later zooms). `_Job.run` (`thumbs.py:50-69`) already has everything it needs.
2. *Qt's scaled JPEG decode* for `.jpg`: `QImageReader` with `setAutoTransform(True)` and `setScaledSize` was
   1.6x faster than the PIL draft path in the measurement; PNG/WebP stay on PIL (no gain). EXIF orientation is
   honoured by `setAutoTransform`.
3. *Thread count*: `pool.setMaxThreadCount(max(2, cpu_count // 2))` (`thumbs.py:80`) leaves half the cores idle while
   PNG decoding is CPU-bound and GIL-free; `cpu_count - 1` with the UI thread left alone *(estimate: ~1.8x on a
   first open of a PNG folder)*.
4. *Background pre-build* (opt-in setting, off by default): after the visible cards are done, queue the rest of the
   folder at the lowest priority, so the first scroll through a 10k folder is smooth. The priority scheme
   (`thumbs.py:118-119`) already puts on-screen requests first; the pre-build just feeds the pool behind them.
**Risk.** None to data. Qt's JPEG decoder and PIL's can differ by a pixel in chroma upsampling; thumbnails only.
**Files.** `tabs/workspace/thumbs.py`, `core/image_utils.py`, `core/config.py` (the setting), `tabs/settings.py`,
`tabs/help_content.py`.
**Verify.** `tests/test_workspace_core.py`: both tier files written from one decode; a scaled Qt read equals the
PIL read's size and orientation on a generated EXIF-rotated JPEG. Benchmark line for each format.

### A3. Open a folder off the UI thread, with one directory listing *(a, S)*
**What.** `WorkspaceContext.open_folder` (`tabs/workspace/context.py:118-140`) calls `DatasetSession.open` on the
UI thread: a scandir, then per image an `is_file` on the caption (`core/caption_io.py:50`), a `read_bytes`, and a
`stat` (`core/dataset_session.py:70-76`). Two changes: (1) one `os.scandir` whose `DirEntry` objects supply
`stat()` (free on Windows: the listing carries the attributes) and tell which captions exist, so the per-image
`is_file` and `stat` syscalls disappear (`core/dataset.py:37-41` can return entries, not just paths);
(2) do the open in a background task (`tabs/common.run_in_background`) with the model populated when it returns, or
in chunks of 1000 via `beginInsertRows`, so the window stays responsive on a network drive or a 50k folder.
**Why.** 519 -> 403 ms here for 10k files on local disk; on NTFS each avoided `stat`/`is_file` is ~50-100 us, so
10k images save a second or more *(estimate)*; a network share saves far more. The UI freeze goes away regardless.
**Risk.** The `generation` counter already guards against a folder change mid-scan (`context.py:125`); the
background open must honour it. Recursive scans (`dataset.py:29-35`) keep the walk.
**Files.** `core/dataset.py`, `core/dataset_session.py`, `tabs/workspace/context.py`, `tabs/workspace/model.py`.
**Verify.** `tests/test_fileops_dataset.py` / `test_workspace_core.py`: identical entry lists, sizes and mtimes
from both paths; a 10k-file timing in `tools/bench_app.py`. Windows timing is the owner's.

### A4. Saving many captions: parallel writes off the UI thread *(a, S)*
**What.** `DatasetSession.save` (`core/dataset_session.py:122-138`) writes serially on the UI thread; each
`write_caption` reads the old file, copies a backup and does `mkstemp` + `fsync` + `replace`
(`core/caption_io.py:64-78`). Run the writes on a small thread pool (8) and report progress; keep every guarantee
(atomic replace, fsync, backup) per file.
**Why.** 4.6x measured on 8 threads in this sandbox; on NTFS, where `FlushFileBuffers` costs 10-50 ms, a 10k-caption
save is minutes today *(estimate)* and would be seconds. The pool makes the fsyncs overlap, it does not drop them.
**Risk.** Backups go to one folder per day; `_backup_once_per_day` checks `dest.exists()` then copies, so two
threads cannot race on the same file (each caption is written by one task). Report failures as today.
**Files.** `core/dataset_session.py`, `tabs/workspace/context.py:191-204`, `tabs/workspace/browser.py:367-372`.
**Verify.** `tests/test_caption_io.py`: same file contents, same backups, same `SaveReport` counts, failures
collected. Benchmark line.

### A5. Window first, AI runtime later *(a, S-M)*
**What.** `main.py:252` imports torch (`preload_ai_runtime`, `main.py:210-234`) behind a splash before the window
is built. Show the window first and run the same main-thread import from a `QTimer.singleShot(0)` after
`window.show()`, with "Loading AI runtime…" in the status area; or go further and import only on the first AI
action (the Train tab never needs torch in the app process: training runs in children; the memory bar reads VRAM
without torch, `core/vram_monitor.py:59-75`). Cache `detect_devices()` to `user_data/cache/devices.json` keyed by
the torch version so device labels appear without the import.
**Why.** A torch import on Windows ROCm is several seconds (large DLLs, antivirus scanning); users who only browse
and edit pay it every launch. The main-thread-only rule in the `preload_ai_runtime` docstring is kept: the deferred
import still runs on the main thread, after the other imports are done.
**Risk.** Callers that assume `detect_devices()` has run before the first AI action must be checked (grep
`detect_devices` in `tabs/`); the Caption tab's device combo is the likely one.
**Files.** `main.py`, `core/hardware.py:150-154`, `tabs/caption.py`.
**Verify.** `tests/test_no_gpu.py` style: the window builds with torch absent from `sys.modules`; a device cache
round-trip test. Startup timing is the owner's.

### A6. The Train tab's samples list decodes on the UI thread *(a, S)*
**What.** `_refresh_samples` (`tabs/train.py:1001-1016`) decodes up to 60 preview PNGs with `QPixmap(...).scaled`
on the UI thread whenever the list changes (every preview, and a 5 s timer, `train.py:83-85`); a 1024x1024 PNG is
~30 ms each, so a changed list freezes the window for a second or two. Use the workspace `ThumbnailLoader` (disk
cache, pool) or decode only the new files.
**Files.** `tabs/train.py`. **Verify.** `tests/test_train_tab.py` with generated PNGs; no GPU.

### A7. Cache the sort key, cache the stats *(a, S)*
**What.** `FilterProxy.lessThan` (`tabs/workspace/model.py:114-118`) recomputes two `natural_key` regex splits per
comparison (3.3x slower than a cached key); `update_status` (`tabs/workspace/browser.py:412-425`) walks all entries
on every selection change (2.5 ms per 10k; fine today, but it runs on every drag-select step). Compute the sort key
once per entry per sort (invalidate in `set_sort_mode`), keep running counts in the session.
**Files.** `tabs/workspace/model.py`, `core/dataset_session.py`. **Verify.** existing widget tests; benchmark line.

### A8. Watch the folder instead of polling *(b, M)*
**What.** `pick_up_external_changes` (`tabs/workspace/context.py:224-228`, called from `tabs/gallery.py:143`)
re-reads every caption to find what another app changed. A `QFileSystemWatcher` on the open folder, debounced
500 ms, re-lists the directory once and diffs names/mtimes against the session: new and removed images become
`beginInsertRows` / `beginRemoveRows` (today `DatasetModel.remove_keys` resets the whole model,
`tabs/workspace/model.py:67-72`, which drops the selection and scroll position), changed captions are re-read only
for the files whose mtime moved (never for dirty entries, as today).
**Why.** Scales with what changed, not with the folder; new images appear without Rescan; selection survives deletes.
**Risk.** Watcher limits on Windows are per directory (fine); recursive scans need one watch per subfolder.
**Files.** `tabs/workspace/context.py`, `tabs/workspace/model.py`, `core/dataset.py`.
**Verify.** offscreen widget test writing files into a temp folder; benchmark the diff on 10k names.

### A9. Batch operations: nothing urgent *(a)*
`session.transform` over 10k captions is pure Python and tens of ms; the undo command coalesces repaints
(`tabs/workspace/model.py:46-61`). The slow part of a batch op is the save (A4). `tag_counts` (`dataset_session.py:147-166`)
over 10k captions is ~100-200 ms *(estimate)* and runs on selection change in the Gallery; cache per entry if it shows.

---

## 2. Inference: captioning and tagging

### I1. Avoid re-captioning unchanged images: a provenance index *(a, S-M)*
**What.** "Skip images that already have a caption" (`inference/worker.py:139-144`) is the only skip rule. Add a
per-folder index `user_data/projects/<folder>/caption_index.json` written by `BatchWorker._save` (`worker.py:213-230`):
per image `{image mtime, image size, caption sha1, provider kind + model, prompt hash, params hash}`. A new save
mode "Skip images already captioned with these settings" skips an image whose index row matches and whose caption
file still has that sha1 (edited captions are re-done or kept, the user's choice). The Caption tab's
"Overwrite captions?" count (`tabs/caption.py:910-917`) then says how many are unchanged.
**Why.** Re-running a job after adding ten images to a 500-image folder costs ten captions, not 500. Resumable after
a crash by construction.
**Risk.** None to data: the index is advisory; the caption file is the truth.
**Files.** `inference/worker.py`, `core/projects.py`, `tabs/caption.py`, `tabs/help_content.py`.
**Verify.** `tests/test_inference.py` with the fake provider: skips, re-does on prompt change, re-does on edited
caption.

### I2. Transformers VLMs: cheaper prefill and faster decode without changing the default output *(a/b, S-M)*
1. *Image token budget* (S, opt-in): `max_image_side` (default 1536, `inference/base.py:53`) bounds pixels before
   the processor, but Qwen-style processors then tile at their own `max_pixels`; a 1536-px image is ~2300 visual
   tokens on Qwen3-VL. An "Image detail: Standard / High" choice that sets the processor's `max_pixels`
   (`transformers_vlm.py:112`, after `AutoProcessor.from_pretrained`) to ~1024 tokens halves prefill
   *(estimate: 20-40% of a short caption's time)*. Default stays as today; the catalog can carry a per-model value.
2. *Prompt-lookup speculative decoding* (M): `self.model.generate(..., prompt_lookup_num_tokens=10)`
   (`transformers_vlm.py:277`) when the batch is one image. The decoder proposes n-grams found in the prompt; with
   tag hints in the prompt (`tabs/caption.py:929-936`) or tag-style outputs the acceptance rate is high. No draft
   model, no extra VRAM; greedy output is identical, sampling keeps the same distribution. Only for batch size 1
   (assisted generation's limit). *(estimate: 1.3-2x on decode for tag-style outputs, ~1.0x on prose.)*
3. *Batch size by default for GDN models on AMD*: the catalog notes already say batch 2-4 recovers Qwen3.5's
   throughput on ROCm (`inference/models.py:41-47`); make the Caption tab suggest it when a `qwen3_5` model is
   picked on a ROCm device (a hint, not a change).
**Risk.** (1) changes caption detail, so it is a visible choice; (2) touches the generation call: guard behind a
try/except that falls back to plain `generate` on any `ValueError` from an unsupported combination.
**Files.** `inference/transformers_vlm.py`, `inference/models.py`, `tabs/caption.py`, `tabs/help_content.py`.
**Verify.** a tiny randomly initialised `LlavaForConditionalGeneration` built from a config in a test exercises
the generate kwargs on CPU; real throughput is the owner's (AMD) and an NVIDIA user's.

### I3. OpenAI-compatible servers: already ordered right; add server-aware concurrency *(a, S)*
`_one` puts the text before the image (`inference/openai_api.py:103-106`), which is what llama.cpp / LM Studio prefix
caching wants (the shared system + instruction prefix is reused; the image differs). Two cheap additions: (1) the
JPEG sent is `max_image_side` (1536) at q92 (`image_prep.py:33-36`): for a local server whose model tiles at
~1024 px, a 1024 cap for the API path saves encode, base64 and the server's vision prefill *(estimate: 10-30%)*; make
it the API profile's own "Upload size" option; (2) the "Concurrency" option (`openai_api.py:67`) only helps when the
server has parallel slots (`llama-server -np N`, LM Studio's setting); probe once with two concurrent tiny requests
and warn when the second waits for the first.
**Files.** `inference/openai_api.py`, `inference/image_prep.py`, `tabs/caption.py`.

### I4. WD tagger: vectorise the post-processing, overlap the preprocessing *(a, S)*
**What.** `predict` (`inference/wd_tagger.py:185-198`) walks 10861 tags per image in Python (76 ms per batch of 8);
numpy masking + `argsort` gives identical lists in 12 ms. `_prep` (pad to square, resize, `np.asarray`,
`wd_tagger.py:165-173`) runs on the worker thread between ONNX runs; a `Provider.prepare(image)` hook called from
the worker's decode pool (`inference/worker.py:161-169`) overlaps it with inference.
**Why.** On the CPU provider inference dominates (hundreds of ms per image); on DirectML (the owner's Windows AMD
route) inference is tens of ms, so ~10 ms of Python per image is 20-30%.
**Risk.** None: the tag lists are identical (checked in the benchmark).
**Files.** `inference/wd_tagger.py`, `inference/base.py`, `inference/worker.py`. **Verify.** `tests/test_inference.py`
equality of old and new `format_tags` input on random scores.

### I5. Model lifetime: free the captioner before a training run starts *(a, S)*
**What.** The Train tab starts child processes with whatever the app has resident (`tabs/train.py:710-760` never
calls `inference.manager`); a loaded 9-17 GB VLM in the app process halves the free VRAM the child's Auto plan sees
(`training/quant.py:194-262`) and, on Windows, pushes the trainer into shared-memory spill. Offer to unload (or
unload automatically, a setting) on Start when `manager().loaded_model_name()` is set and the device is the training
device; the reverse warning when a caption job starts during training.
**Files.** `tabs/train.py`, `inference/manager.py`, `tabs/help_content.py`. **Verify.** `tests/test_train_tab.py`
with the fake manager.

### I6. ONNX providers *(c)*
`GPU_PROVIDERS` (`wd_tagger.py:45-46`) covers CUDA, ROCm, MIGraphX, DirectML, CoreML. On Windows AMD only
`onnxruntime-directml` exists; the installer could offer it (it is noted in `docs/ARCHITECTURE.md` but not
installed). A float16 copy of the model for DirectML would be a quality question (the tagger's thresholds are tuned
on fp32 outputs); not recommended without a check against the fp32 tags on a demo set.

---

## 3. Training: run-to-result time without changing what a run learns

### T0. Diagnostics first (the plan's own rule: measurements before defaults) *(a, S)*
Three log lines that cost nothing and settle the owner's three leads:
1. **Preview phase timings.** `previews()` measures only the total (`training/train.py:1133-1135`). Log per preview:
   text gather, `driver.generate` per prompt (and steps/s), VAE decode, park/restore and EMA swap, Turbo LoRA move
   (`_render_previews`, `train.py:316-406`). Then the 2.5 min/epoch has an owner.
2. **Allocator line per epoch.** `train.py:1110-1113` logs peak reserved only. Add `memory_allocated` peak,
   `inactive_split_bytes.all.peak` and `num_alloc_retries` from `torch.cuda.memory_stats()`, and once per run the
   allocator config actually in force (`PYTORCH_ALLOC_CONF` / `PYTORCH_CUDA_ALLOC_CONF`, `torch.cuda.get_allocator_backend()`).
   The 17.5 GB reserved / 12.9 GB in use lead becomes a fragmentation number (inactive splits) rather than a guess.
   Note: the child only gets `expandable_segments` through the app's environment (`core/hardware.py:81`, inherited by
   `training/pipeline.py:child_env`); `training/train.py:1162-1164` sets it itself only off Windows (as Fizgig
   `families/train.py:2013`), so a command-line run on Windows has no allocator config at all. Whether the ROCm
   Windows build honours `expandable_segments` is unknown: PyTorch logs a warning when it does not, so the line
   should quote it.
3. **Matmul speed probe.** `quant.available("int8")` (`training/quant.py:124-136`) proves `_int_mm` exists; time it
   and bf16 `mm` and the fp8 dequant at Krea 2's shape (M=1152, K=N=6144, ~20 ms total) and log the ratios:
   "INT8 matmul 0.9x bf16 on this GPU: INT8 saves memory, not time." The owner's lead (INT8 not faster than fp8 on
   AMD) becomes a number every user sees.
**Files.** `training/train.py`, `training/quant.py`. **Verify.** smoke tests still pass on CPU (lines are guarded by
`torch.cuda.is_available()`); the numbers are the owner's.

### T1. Preview cost: a preview profile, and Fizgig's eager stance *(a, S each)*
**What.**
1. *Preview profile* (visible choice, default unchanged): "Standard (family default, 1024², Turbo 8 steps)" and
   "Fast (768², Turbo 4-6 steps)". Krea 2 at 1024² is ~4100 image tokens padded to a multiple of 256
   (`families/krea2/model.py:341-345`); at 768² it is ~2300: attention cost drops ~3x and the Linears ~1.8x
   *(estimate: a preview at ~0.45x the time)*. Previews do not change the LoRA.
2. *RAW vs Turbo*: without the Turbo file the preview runs 28 RAW steps at CFG 4.5, i.e. 56 DiT calls versus 8
   (`families/krea2/description.py:157-163`). If the 2.5 min/epoch run used RAW previews, pointing at the Turbo
   file is the whole fix; the Train tab's status line should say which it is rendering with (T0.1 tells).
3. *Eager previews under Compile Blocks* (NVIDIA; Fizgig `d9a896d2`): TagScribeR renders previews through the
   compiled blocks, so each preview shape and the Turbo LoRA are new graphs ("~5 minutes of Triton/MSVC work on the
   first preview after every start or resume", Fizgig's note). Wrap `_render_previews` in
   `torch.compiler.set_stance("force_eager")` as Fizgig does.
4. *Decode while the next epoch starts* is not worth it: one GPU, and the decode is a few seconds.
**Risk.** (1) is a user choice; (3) changes nothing numerically (eager and compiled are the same maths) and only
applies when compile is on.
**Files.** `training/train.py:316-406`, `training/params.py` (the profile parameter), `tabs/train.py`,
`tabs/help_content.py`. **Verify.** smoke test with the tiny family for the profile and the stance (CPU); the
seconds are the owner's.

### T2. Allocator behaviour across bucket shapes *(a for the diagnostics, c for the remedies, S)*
**What.** After T0.2 says how much of the 4.6 GB gap is inactive splits, an A/B of one epoch each, logged, on the
owner's card: (i) as today; (ii) `expandable_segments:True` confirmed active or, if the backend refuses it,
(iii) `max_split_size_mb:512` (keeps the allocator from carving large blocks it later cannot reuse for a different
bucket shape); (iv) a `torch.cuda.empty_cache()` at every epoch boundary (once per epoch, cheap; the preview path
already does it, `train.py:403`). (v) *Shape-descending first epoch* (opt-in, changes the sample order): visit the
largest bucket first so the biggest blocks are reserved before smaller shapes split them. (vi) Fewer shapes:
`TAGSCRIBER_ATTN_TRIM=0` (`families/krea2/attention.py:21-24`) collapses the per-caption sequence lengths to the
bucket count at the price of attending over padding; `TAGSCRIBER_ATTN_TRIM_MULTIPLE=128` halves the shape count
with the same accepted perturbation class as 64 (Fizgig's note in `sampling.py:22-25`).
**Why.** On Windows, fragmentation is a speed problem, not only a capacity one: once reserved memory plus the
desktop exceeds the card, the driver spills to system memory over PCIe and the step time jumps several-fold.
**Risk.** (v) changes which image is seen when; the objective is unchanged but the trajectory is not: opt-in,
documented. (vi) is already an env knob; a parameter would need the owner's agreement on the default.
**Files.** `training/train.py`, `training/dataset.py:420-428` (the order), `core/hardware.py:81`, `training/pipeline.py:741-747`.
**Verify.** CPU: the order option gives the documented permutation (test with the tiny family). The rest is the
owner's A/B, read from the T0.2 lines.

### T3. Trimmed text cache and CPU-side gather *(a, M)* — bit-identical, proven on random data
**What.** A Krea 2 caption is cached as `(512, 12, 2560)` bf16 = 31.5 MB, mostly zeros: the mask is
`[prompt, padding, suffix]` (`families/krea2/embedder.py:138-153`). The DiT never sees that tensor: `_text`
(`krea2/driver.py:243-248`) runs `gather_valid_text` (`krea2/sampling.py:36-52`), which drops the interior padding and
zero-pads to the batch maximum rounded up to 64. Two steps: (1) *now*: gather on the CPU in `batch_cond` before
`.to(device)` (`training/driver.py:93-96`, Krea 2 override), so 4 MB cross the bus per step instead of 31.5 MB;
(2) *cache format*: `encode_text` stores `hidden_states[mask]` (the valid rows, 71 of 512 for a typical caption) and an
all-true mask, under `text_cache_rev="2"` so old caches re-encode once; `batch_cond` right-pads items to the batch
maximum before stacking (batch > 1 is Krea 2's `supports_batching`). Measured: the gathered tensor is bit-identical
in both layouts, for one caption and for a batch of 71 and 130 tokens. The run reads caches, so the DiT input is
unchanged to the bit.
**Why.** 7.2x smaller caches (a 1000-image dataset: 31 GB -> 4.4 GB on disk and in the page cache), 7x less host-to-device
copy per step, and the door to T4. The per-step *disk* cost is already ~1 ms with a warm page cache, so the s/step
gain on a small dataset on NVMe is small *(estimate: 0.5-2%)*; the win is datasets that no longer fit the page cache,
slower disks, and the storage.
**Risk.** A mistake in the pad/mask handling would silently change the conditioning: guard it with a parity test
that encodes random stacks through both layouts and asserts `torch.equal` on `gather_valid_text`'s output and mask,
single and batched. The Krea 2 preview path (`_text` on an unbatched dict) must accept both layouts. Other families
with fixed-shape text (Qwen Image 2.1?) are checked one by one; a family whose cache has no mask is left alone.
**Files.** `training/families/krea2/driver.py` (`encode_text`, `batch_cond`, `_text`), `training/families/krea2/description.py`
(`text_cache_rev`), `training/cache.py` (`cached_matches` already keys on the rev), `tests/test_training_krea2_parity.py`.
**Verify.** the parity test above on CPU; a cache-size assertion in the smoke test; the owner's next run confirms the
loss curve is unchanged (it must be, by construction).

### T4. RAM-resident conditioning and a one-step prefetch *(b, M)*
**What.** With T3, a whole dataset's text caches fit in RAM (4.4 MB x 1000). Load them once at `prepare_for_training`
(`training/dataset.py:366-410`) into pinned CPU tensors, hand `get_batch` (`dataset.py:446-466`) views instead of
`load_file`, and move the per-step `.to(device)` to `non_blocking=True`. Independently, a one-deep prefetch thread
that calls `get_batch(i + 1)` while step `i` runs (the `data_rng` is consumed in the same order, so caption dropout
and shuffle variants draw the same numbers).
**Why.** Removes every per-step file read and the synchronous host-to-device copy. Measured honestly: with a warm
page cache the read is ~1 ms and prefetch gained nothing; the win is on cold caches (the first epoch), HDDs, network
shares and datasets larger than RAM, and it removes a stall that `loss.item()` (`train.py:1103-1104`) otherwise
hides. Caption fixes at a boundary (`training/loss_watch.py:206-224`) must invalidate the RAM copy of the fixed item.
**Risk.** Pinned memory on Windows counts against the driver's shared budget; cap the RAM cache (e.g. 4 GB) and fall
back to disk. The prefetch must never run ahead across an epoch boundary (the shuffle and `watch.boundary` come first).
**Files.** `training/dataset.py`, `training/train.py:1033-1106`, `training/loss_watch.py`.
**Verify.** smoke test: identical losses step by step with and without the RAM cache / prefetch (CPU, tiny family,
same seed): this is the bit-identity proof.

### T5. Selective gradient checkpointing, planned from free VRAM *(a numerically, b in effort; M)*
**What.** Every Krea 2 block is checkpointed (`families/krea2/model.py:359-361`), so the forward runs twice per step.
Checkpointing is a memory/time trade with identical maths (the recompute reproduces the same activations; the
INT8/fp8 paths are deterministic per call). Offer "Checkpoint blocks: All (default) / Half / Auto": Auto measures the
peak at the largest bucket during the first steps and un-checkpoints as many blocks as the free VRAM allows. On a
24-32 GB NVIDIA card at 0.25 MP INT8 (16.2 GB peak, `krea2/description.py:117`) there is room for most blocks.
**Why.** The recompute is about a third of the step; un-checkpointing half the blocks saves ~15%, all of them ~25-30%
*(estimate)*. Fizgig has only on/off.
**Risk.** None to numerics (bit-identical in principle; a smoke test should assert it on CPU with the tiny family,
which has `enable_gradient_checkpointing` as a no-op today, so the test needs a checkpointing tiny block). A wrong
Auto estimate means an OOM on the first step, not a silent change: fall back to All and continue.
**Files.** `training/families/krea2/model.py`, `training/driver.py` (a per-block flag), `training/train.py`,
`training/params.py`, `tabs/help_content.py`. **Verify.** CPU equality test; the speed is the owner's and NVIDIA
users'.

### T6. fp8 dequant: a lighter path, opt-in *(c numerically, S)*
**What.** `fp8_linear_forward` (`training/modules/fp8.py:77-95`) dequantises through an fp32 temporary
(`w.to(float32) * scale -> bf16`) on every call, twice per step per Linear under checkpointing. A direct
`w.to(bf16) * scale.to(bf16)` moved 3x less memory on CPU (1494 -> 491 ms for 6144x6144) and is bandwidth-bound on
the GPU too *(estimate: 5-15% of a 0.25 MP fp8 step, less at 1 MP where the matmul dominates)*. It is **not**
bit-identical: 22% of weights move by one bf16 ulp (max rel 7.8e-3), an order of magnitude below fp8's own rounding
(3 mantissa bits) but a change nonetheless. So: `TAGSCRIBER_FP8_DEQUANT=fast`, off by default, for the owner to
A/B on a run the owner already knows; promote only with the owner's verdict. The same applies to a bf16 INT8 dequant in the
backward (`int8_train.py:132-135` already does it in the compute dtype).
**Files.** `training/modules/fp8.py`. **Verify.** `tests/test_training_fp8.py`: the fast path's max error against the
exact path stays under a stated bound; the quality call is the owner's.

### T7. INT8 on AMD: name it, do not change Auto *(c, S)*
T0.3 tells whether `_int_mm` is slower than bf16 on the ROCm build. If it is, INT8 is still the right *memory* choice
on a 20 GB card (16.2 vs 18.7 GB for fp8 at 0.25 MP), so Auto's order stays as decided (PORT_PLAN decision 2); the
log line and the Help text should say "on this GPU, INT8 saves memory, not time", so nobody picks INT8 over bf16 on
a 32 GB AMD card for speed. A question for the owner, not a change: should Auto prefer bf16 over INT8 when bf16 fits
with margin *and* the probe says INT8 is not faster? (Fizgig's Auto never picks bf16 for Krea 2.)

### T8. Compile on ROCm: the shape problem is the real blocker *(c, M)*
Auto refuses compile on ROCm because HIP recompiles per bucket shape (`training/compile.py:153-155`); On overrides.
Two facts narrow the question: (1) the shape count is the per-caption trimmed sequence length times the bucket count
(`attention.py:56-77`); `TAGSCRIBER_ATTN_TRIM=0` collapses it to the bucket count; (2) Fizgig tried dynamic shapes
and found "the trigger-free captions' extra text lengths sent Krea 2's compiled blocks to a dynamic graph that
failed in inductor at step 2" (Fizgig `a0c1a52e`), so `mark_dynamic` is not the answer; static graphs, few shapes,
is. Whether triton runs on the Windows ROCm (TheRock) wheels at all is the first question for the owner; if it does,
a run with `TAGSCRIBER_ATTN_TRIM=0`, Compile On, Target MP fixed and `bucket_no_upscale` off (one shape) would say
what compile is worth on gfx1100. Fizgig's measured compile wins (Z-Image 1.49x INT8 outside-boundary,
`dea949b9`; Flux 3 1.19x, `e4c4dc3c`) are NVIDIA numbers.

### T9. hipBLASLt / rocBLAS preference for bf16 GEMMs *(c, S to try)*
`core/hardware.py:77` removes `ROCBLAS_USE_HIPBLASLT_BATCHED`; nothing sets `TORCH_BLAS_PREFER_HIPBLASLT`. On recent
ROCm builds the bf16 GEMM backend for gfx11 can go either way; the T0.3 probe run twice (with and without
`TORCH_BLAS_PREFER_HIPBLASLT=1`) is a one-minute experiment for the owner. Not a default until measured; Fizgig does
not set it either.

### T10. Block swap overlap *(c, M; NVIDIA 12-16 GB users)*
The offloader streams through small pinned staging buffers with a synchronous CPU copy on each side
(`training/modules/offloading.py:181-237`, the two `BOTTLENECK` comments: pageable block weights to the staging
buffer and back) because `use_pinned_memory` defaults off (`families/krea2/model.py:290` passes nothing). Keeping the
swapped weights themselves pinned (`offloading.py:238-287`) makes the copies asynchronous at the price of pinned RAM. Fizgig's newer
"H2D rings" for H3 are listed in PORT_PLAN stage 6. The owner's card does not swap for Krea 2, so this is for other
users: expose `TAGSCRIBER_OFFLOADER_DEBUG=1` timings in the docs and ask an NVIDIA tester for a run with pinned on.

### T11. Small Fizgig takes worth having *(a, S each)*
- *Latent cache re-encode when the image is newer than its cache* (Fizgig `59638a09`): `run_latents`'s skip check
  (`training/cache.py:193-197`) compares size, revision and controls, not mtime; an edited photo trains on old latents.
  Correctness first, but it also avoids a stale-cache re-run by hand.
- *Text encoder kept in pinned RAM between uses* (Fizgig `4001f1f0`, MOSS workbench): `_encode_override`
  (`train.py:270-291`) and the loss watch's `_reencode` (`loss_watch.py:206-224`) reload the 8 GB Qwen3-VL from disk at
  every boundary that needs it. Keeping it parked in pinned RAM when free RAM allows (Fizgig's 18 GB rule from
  `_CheckpointPreviews._cache_on`, `train.py:426-437`) turns a 30-60 s reload into a 1-2 s move *(estimate)*.
- *Checkpoint thumbnail refresh* (`training/metadata.py:92-117`) rewrites the whole LoRA file to change one metadata
  string; `safetensors` has no in-place metadata edit, but the rewrite could be skipped when the preview is the one
  already embedded (it is: `:106-107`) and done on a thread so the next epoch starts at once. Seconds per epoch.
- *Caching latents in batches*: `run_latents` encodes one image at a time (`cache.py:202`, `batch_size=1`) although
  `latent_batches` groups by bucket. Batch 4 on the VAE encode is ~2-3x on the caching stage *(estimate)*; the VAE
  encode is deterministic per sample (`posterior.mode()`), but kernels can round a batched conv differently, so this
  is the same class of question as decision 2's one-caption-per-forward: ask before changing it.

---

## 4. Ranking by value / effort

### (a) Safe wins (no training numerics change; defaults that affect output stay)
| # | Idea | Effort | Expected win |
|---|---|---|---|
| 1 | A1 health scan: thumbnails + fingerprint + pool | S-M | 10-20x on the scan (measured parts) |
| 2 | A2 thumbnails: one decode, Qt JPEG, more threads, pre-build | S-M | 1.6-7x per image, smooth first scroll |
| 3 | T0 diagnostics: preview timings, allocator line, matmul probe | S | turns three leads into numbers |
| 4 | T3 trimmed text cache + CPU gather (bit-identical) | M | 7x smaller caches, 7x less H2D |
| 5 | A3 folder open off the UI thread, one scandir | S | no freeze; ~2x on Windows (estimate) |
| 6 | A4 parallel caption saves | S | 4.6x measured, more on NTFS |
| 7 | I1 provenance index (skip unchanged) | S-M | re-runs cost only the new images |
| 8 | T1 preview profile + eager stance + "RAW or Turbo" in the status | S | 2x per preview (estimate); minutes on NVIDIA compile runs |
| 9 | I4 WD tagger vectorised post-processing + prep overlap | S | 6x on 10 ms/image of Python |
| 10 | I5 unload the captioner before a run | S | the Auto plan sees the real free VRAM |
| 11 | A6 samples list off the UI thread; A7 cached sort key | S | no 1-2 s freezes |
| 12 | T11 Fizgig takes (mtime re-encode, TE in RAM, thumbnail refresh on a thread) | S each | correctness + tens of seconds per boundary |
| 13 | I2.1 image token budget (visible choice) | S | 20-40% of a caption (estimate) |

### (b) Architectural bets
| # | Idea | Effort | Expected win |
|---|---|---|---|
| 1 | T5 selective gradient checkpointing, Auto from free VRAM | M | 15-30% s/step where VRAM allows (estimate) |
| 2 | T4 RAM-resident conditioning + prefetch | M | cold epochs, big datasets, slow disks |
| 3 | A8 file watcher + incremental model | M | scales with changes, keeps selection |
| 4 | I2.2 prompt-lookup speculative decoding | M | 1.3-2x decode for tag-style output (estimate) |
| 5 | A5 window first, torch on demand, device cache | S-M | perceived startup; no torch for non-AI use |

### (c) Speculative / needs a measurement or a platform answer
| # | Idea | Effort | Open question |
|---|---|---|---|
| 1 | T2 allocator remedies (expandable segments, max_split, shape order) | S | does ROCm-Windows honour expandable segments? |
| 2 | T6 fast fp8 dequant (opt-in, 1 ulp) | S | the owner's A/B |
| 3 | T7 INT8 on AMD: wording, maybe Auto's bf16 question | S | the probe's ratio; the owner's call |
| 4 | T8 compile on ROCm with few shapes | M | does triton run on the Windows ROCm wheels? |
| 5 | T9 hipBLASLt preference | S | one timed run |
| 6 | T10 pinned block swap | M | an NVIDIA tester |
| 7 | I6 DirectML tagger, fp16 copy | S | fp32 tag parity |
| 8 | T11 batched VAE encode when caching | S | a decision-2 class question |

## 5. Top ten, one line each

1. **A1** Health scan reads the thumbnail cache, fingerprints before hashing, runs on a pool: 10-20x.
2. **A2** Thumbnails decode once for both tiers, JPEGs through Qt's scaled decoder, all cores, optional pre-build.
3. **T0** Preview phase timings, an allocator line with inactive splits, a matmul speed probe: the three leads get numbers.
4. **T3** Trimmed Krea 2 text cache + gather on the CPU: 7x smaller caches, bit-identical DiT input (proven on random data).
5. **A3 + A4** Folder open and caption saves off the UI thread, one scandir, parallel writes.
6. **I1** Provenance index so re-runs caption only what changed.
7. **T1** Preview profile (768² / fewer Turbo steps) and Fizgig's eager stance for previews under compile.
8. **T5** Selective gradient checkpointing planned from free VRAM: 15-30% s/step where there is headroom.
9. **I4 + I2** WD post-processing vectorised, prep overlapped; image token budget and prompt-lookup decoding for VLMs.
10. **I5 + T11** Unload the captioner before a run; re-encode latents when the image is newer; keep the text encoder in RAM between boundaries.

## 6. What I would do first, and why

1. **T0 diagnostics** (half a day): every training decision the owner wants (previews, allocator, INT8 vs fp8) is
   blocked on a number only the owner's card can produce, and the plan says "bring findings and measurements before changing
   any default". Three log lines do that with zero risk and no default touched.
2. **A1 + A2** (a day): the largest measured wins in the whole review, in the part of the app every user touches
   daily, with no connection to training numerics and a complete CPU test story.
3. **T3 trimmed text cache with CPU-side gather** (a day or two): proven bit-identical on random stacks including the
   batched case, 7x less storage and bus traffic, and the foundation for a RAM-resident data path (T4). It is the one
   training-side change that is both substantial and provably neutral.

## 7. Questions for the owner (not decisions)

- T1: which preview engine was the 2.5 min/epoch run using, RAW 28 steps or Turbo 8? (T0.1 will say for the next.)
- T2: may a shape-descending first epoch be offered as an opt-in (it changes the sample order, not the objective)?
- T6: is a 1-bf16-ulp fp8 dequant acceptable behind an env switch for an A/B on a known run?
- T7: should Auto consider bf16 on AMD when the probe shows INT8 is not faster and bf16 fits with margin?
- T8: do the Windows ROCm wheels in use ship a working triton? (Decides whether compile on ROCm is worth a day.)
- T11: batched VAE encode when caching latents - acceptable, or the same answer as one-caption-per-forward?
- I2.1: should the catalog carry a per-model image token budget, with "High" as the current behaviour?
