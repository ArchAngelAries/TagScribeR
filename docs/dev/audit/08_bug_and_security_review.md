# 08 - Adversarial bug and exploit review: the training side

Scope: everything under `training/` (every family, including code older than the `port/stage1-bt67en` branch), the Train
tab (`tabs/train.py`, `tabs/train_dialogs.py`) and what it calls, `tools/check_release.py`, `tools/make_screenshots.py`.
Reviewed at commit `98a48a2` against Fizgig 7.0.1 (`.claude/fizgig-src`, commit `1c8ec88`). Read-only: nothing in the
tracked tree was changed. The core/, inference/ and the other tabs are covered by their own reviews.

Method: read the run-folder launcher, the trainer loop, the LoRA layer, the cache, the dataset, presets, the loss watch
and all six family drivers; grepped for unsafe primitives (`torch.load`, pickle, `trust_remote_code`, `subprocess`,
`shell=True`, `eval`); diffed the ported maths against Fizgig where a sign or a draw order could silently ruin a run;
wrote three reproductions on the tiny random family (`tests/tiny_family.py`), no real weights, no real datasets. The
full test suite passed before the review (exit code 0, `pytest -q -p no:cacheprovider`).

Reproduction scripts (scratch, not tracked): `repro_stale_latents.py`, `repro_prepare_items.py` in the session's
scratchpad; the one-liners are quoted inline below.

Severity scale: **critical** = silent wrong training or loss of user data with no recovery; **high** = a real run dies or
user data is overwritten; **medium** = silently wrong data in a reachable case, recoverable; **low** = hardening.

---

## Findings, most severe first

### 1. HIGH - a mid-run caption fix on an Edit LoRA run drops the reference from the text cache (Qwen Image 2.1 then dies)

**Where.** `training/dataset.py:394-398` (`TrainingDataset.prepare_for_training`) rebuilds every item as
`ItemInfo(item_key, "", (w, h), bucket, cache_file)`: no caption and no `control_paths`, although the run's
`control_directory` is right there in `d`. `training/loss_watch.py:209` then decides
`refs = any(getattr(item, "control_paths", None) ...)` -> `False`, and `_reencode` calls
`cache.encode_captions(..., references=False)`, i.e. the plain `driver.encode_text`.

**Scenario.** Qwen Image 2.1 Edit LoRA run (before/after pairs). During training the user opens Problem Images and
clicks "Save fix" on any image (manual fixes are applied at every boundary even with the loss watch off:
`Watch.boundary` -> `_captions`), or auto-recaption fires. The item's `<stem>_qwenimage21_te.safetensors` is rewritten
with `cond__hidden_states` only - no `cond__ref_mask`, no `reference_sizes` - while its latent cache still carries
`latent_control_0`. Next epoch, `get_batch` hands the trainer `refs` and a cond without `ref_mask`;
`training/families/qwen_image21/driver.py:102-105` `_ref_sequence` raises `reference size mismatch ... re-cache the
text`. The exception leaves `train_family`, the trainer exits with code 1, the run is over (saved epochs stay; a resume
re-runs into the same cache unless cache preparation is on, which re-encodes it because `reference_sizes` no longer
matches). On Klein the same re-encode is harmless (`encode_text_with_references` == `encode_text`).

**Reproduced.** `repro_prepare_items.py` (tiny family with a reference-aware driver): the cache stage writes
`['cond__hidden_states', 'cond__ref_mask'] | reference_sizes: 96x96`; after `prepare_for_training` the item has
`caption='' control_paths=[]`; after `Watch._reencode` the file holds `['cond__hidden_states'] | reference_sizes: None`.

**Fix.** In `prepare_for_training`, set `it.control_paths = _control_for(d.get("control_directory") or "", item_key)`
and `it.caption` from the text cache's `caption1` header (one `safe_open` per item; it is already opened for
`latent_cache_matches_reso`). Finding 4 is fixed by the same change. A unit test on the tiny family can assert the
re-encoded file keeps `reference_sizes`.

### 2. MEDIUM - an image edited in place at the same pixel size keeps its old latent cache forever

**Where.** `training/cache.py:193-197`: with `--skip_existing` (always on from the Train tab) a latent cache is reused
when its resolution, control presence and `latent_rev` match. Nothing records the source image's mtime, size in bytes
or hash; the cache file name only encodes the ORIGINAL pixel size (`training/dataset.py:146-148`, Fizgig's naming).

**Scenario.** The preflight itself tells the user to fix EXIF-rotated photos by saving them in the Image Editor
(`training/pipeline.py:416-437`). For a square photo, or an orientation-3 (180 degree) photo, the saved file has the
same width and height, so the cache name is unchanged and the stale latents (the unrotated pixels) train again. The
same holds for any in-place colour / retouch edit, and for an edited partner ("before") image of an edit pair or a
slider's -1 end (`reference_sizes` only compares sizes). The user sees "N image(s), 0 to encode" and a LoRA trained on
pixels they no longer have. Fizgig's cache has the same rule, but TagScribeR ships the editor that makes this a
one-click path.

**Reproduced.** `repro_stale_latents.py`: cache 2 images; overwrite `img00.png` with black pixels at the same size;
re-run `run_latents(..., skip_existing=True)` -> `re-encoded 0; latent unchanged: True`.

**Fix.** Write `source_mtime` and `source_bytes` (or a cheap content hash) into the latent cache metadata in
`save_latents` and compare them in the `skip_existing` test (bumping `LATENT_REV` is not needed - a cache without the
keys can be treated as stale once, or accepted once to avoid a one-time full re-encode; say which in the help). The
text cache's `reference_sizes` check should compare the same stamp for partner images. Alternatively (or additionally)
the Image Editor's save could delete the stem's caches under `pipeline.cache_dir_for(folder)`.

### 3. MEDIUM (data loss) - a caption extension equal to an image extension makes caption writes overwrite the image

**Where.** `training/params.py:327` `DATASET_CAPTION_EXT` is a free TEXT preset key with no validation
(`P.coerce` returns any string); `core/caption_io.py:100-116` `write_caption(image_path, text, ext)` writes to
`stem + ext` with no check that the target is not the image itself. Callers on the training side:
`tabs/train_dialogs.py:224` (Problem Images "Save fix") and `training/loss_watch.py:271` (auto-recaption).

**Scenario.** An imported preset (or a typo in the advanced Dataset section) sets `DATASET_CAPTION_EXT` to `.png`.
Preflight passes (every image "has a caption": `pipeline.py:311` matches the images themselves), the cache stage reads
each PNG's bytes as its caption (`decode_caption` falls back to latin-1), and the first "Save fix" or auto-recaption
replaces the photo with its caption text. `_backup_once_per_day` keeps one copy per day under
`user_data/caption_backups`, so the picture is recoverable only if the user knows to look there.

**Reproduced.** `write_caption('photo.png', 'a caption', '.png')` -> the file then holds `b'a caption'`;
`'DATASET_CAPTION_EXT' in P.PRESET_KEYS` is True and `P.coerce` accepts `'.png'`.

**Fix.** Refuse the value in three places: `P.coerce` for `DATASET_CAPTION_EXT` (must start with `.`, must not be in
`IMAGE_EXTENSIONS`), `pipeline.preflight` (error, not warning), and `caption_io.write_caption` (raise if the caption
path equals the image path). The first two also belong to the Gallery / Caption tabs if they read the same setting.

### 4. LOW - image-pair slider previews never use the pairs' caption

**Where.** `training/train.py:119-127` `_pair_slider_caption(dataset)` counts `it.caption` over `dataset.items()`;
those captions are `""` after `prepare_for_training` (finding 1), so it always returns `None`, `shared` is never set
(`train.py:603-606`) and the preview prompt stays whatever the Samples tab holds ("A high quality photo" by default).
Fizgig's `_pair_slider_caption` (`families/train.py:372`) reads the caption files from the folder instead.

**Reproduced.** `repro_prepare_items.py`: `_pair_slider_caption -> None (expected 'shared caption')`.

**Fix.** Same as finding 1 (carry `caption1` into the trainer's items), or read the caption files as Fizgig does.

### 5. LOW - the stale-cache sweep can delete another family's caches when an image stem contains `_<arch>`

**Where.** `training/cache.py:164` globs `*_{arch}*.safetensors` and deletes every match not in this family's keep
set. Arch ids are `anima, klein9b, krea2, minimaxh3, qwenimage21, sdxl, pony, illustrious, noobaieps, noobaivpred`
(no id prefixes another, so families do not collide by name), but an IMAGE STEM can: a dataset with
`portrait_sdxl_v2.png` cached for Krea 2 produces `portrait_sdxl_v2_1024x1024_krea2.safetensors`, which matches the
SDXL family's glob and is removed by the next SDXL cache stage (and `_te_s` in a stem would be mistaken for a shuffle
variant). Caches rebuild on the next run, so no user data is at risk; a long re-encode is.

**Reasoned** from the code; not reproduced.

**Fix.** Anchor the patterns: `*_{arch}.safetensors`, `*_{arch}_te.safetensors`, `*_{arch}_te_s*.safetensors`, and
parse the stem with the same split `prepare_for_training` uses.

### 6. LOW - run-folder helpers trust `LORA_NAME` before preflight validates it

**Where.** `training/pipeline.py:55-57` `run_dir_for` joins `LORA_OUTPUT_DIR / LORA_NAME` verbatim.
`validate_output_name` (`train_utils.py:104-117`) rejects separators and `..` - but only in `preflight`, at Start.
`tabs/train.py:978-999` `_run_dir()` / `open_run_folder` (`mkdir(parents=True)`), `open_override`
(`write_sample_override` writes `.sample_override.json`), `open_problem_images` and `_latest_state` use `run_dir_for`
unvalidated while idle.

**Scenario.** An imported preset carries `LORA_NAME` = `..\\..\\Users\\Public\\x` (LORA_NAME is a preset key).
Clicking "Open run folder" or saving a preview override before Start creates that folder / drops a small JSON file
there. Start itself is refused. Impact is a stray directory or JSON file at a chosen relative path - a nuisance, not a
takeover.

**Reasoned** from the code; not reproduced.

**Fix.** Make `run_dir_for` call `validate_output_name` and fall back to a sanitised name (or raise) so every caller is
covered; `check_name`-style character filtering as presets already do for preset names.

### 7. LOW / question for the owner - presets carry filesystem paths

**Where.** `training/params.py`: `LORA_OUTPUT_DIR`, `FAMILY_EDIT_DIR`, `CONTEXT_LORA_PATH`, `METADATA_THUMBNAIL`
(a path or "off") are `preset=True`; `presets.apply` applies them like any other value. Fizgig does the same
(`families/form.py:686` keeps `LORA_OUTPUT_DIR` unless empty), so this is parity, listed as a question.

**Scenario.** A shared preset redirects the run's output to an arbitrary folder (the run then writes checkpoints there
and `prune_state_dirs` / `Watch._cleanup` delete only its own `<name>-NNNNNN-state` folders and `loss_log/*.json`, so
no foreign files are removed), or sets `METADATA_THUMBNAIL` to a local image path that gets base64-embedded into every
checkpoint the user later publishes. Both values are visible in the UI after "Load", under Output and Metadata.

**Suggestion.** Keep applying them (Fizgig parity) but name them in the `[preset] applied` console line, e.g.
"also set Output folder, Thumbnail", so a redirected output or a foreign thumbnail is noticed. Owner's call.

### 8. LOW (hardening) - `torch.load` on a resume state relies on torch's default

**Where.** `training/train.py:200, 204, 867` load `optimizer.pt`, `rng.pt`, `ema.pt` from the state folder the user
points Resume at, without `weights_only=True`. The installed pins (`tools/install.py`: torch 2.10 CUDA, 2.12 ROCm) have
`weights_only=True` as the default since torch 2.6, so today a state folder from someone else cannot run code; a
downgrade or a stray older torch would reopen the pickle path on a shared `-state` folder.

**Reasoned**; checked that the saved objects (optimizer state dicts, RNG tensors, EMA shadow lists) are tensors and
ints, so the explicit flag will not break resume.

**Fix.** Pass `weights_only=True` (and `map_location="cpu"` for `rng.pt`, whose CUDA tensors already fail loudly on a
CPU box and are caught).

---

## Checked and found sound (so the owner need not re-check)

Security surface:
* Child processes: argv lists only (`pipeline.build_run`, `kill_tree`), no shell; `compile.py:272` is the one
  `shell=True`, on a `vcvars64.bat` path discovered via vswhere / fixed VS roots, not from user files.
* No pickle, `trust_remote_code`, `eval`/`exec` or `hf_hub_download` of user-named repos under `training/`; model and
  LoRA files go through safetensors (`safe_open` header reads for metadata, `load_file` for tensors); the Anima
  tokenizer helper is a fixed repo id.
* `optimizer_args` is parsed with `ast.literal_eval` (`optimizers.py:178`); `OPTIMIZER_TYPE` is a strict choice, so a
  preset cannot reach the `module.path.ClassName` import path (the UI combo is non-editable for strict choices).
* Presets: strict choices refused, numbers range-checked, slider / fine-tune presets refused as a whole, Fizgig key
  migration (`QUANT_4BIT*`, `MINIMAX_*`, `TARGET_LAYERS`, `TRAINING_BLOCKS` dict, 0-1000 timesteps) read as intended.
* Caption writes go through `core.caption_io.write_caption` (atomic, daily backup) in both the Problem Images dialog
  and auto-recaption; preset deletes go to the Recycle Bin; caption fixes are claimed atomically (`os.replace`).
* Dataset folder content: stems come from `os.listdir` (no separators), captions are only metadata and text-encoder
  input, broken images are skipped at `source_items` (a truncated image still fails the cache stage loudly, not
  silently); `tagscriber_excluded.json` / look-score files are read as plain JSON with per-field type checks.

Training correctness (branch and older code):
* Objectives and targets per family (`noise - x0` for Krea 2 / Klein / Qwen / Anima, `x0 - noise` for H3 matching
  ComfyUI, epsilon / v for SDXL) and the `noise_latents` / `predict` pairs draw in the same order as `training_loss`
  within each driver, so a prompt slider's state is a training state.
* `_prompt_slider_step` is Fizgig's line for line; both poles backpropagate before the flip; `set_trainable_multiplier`
  is restored in `finally` in the step, the practice-bank render and `_render_previews` (`done` set).
* Preview state restore: training adapter off/on, speed LoRA on/off and moved back to CPU, EMA swap in/out, block-swap
  inference/training mode, low-memory park, `dit.train(was_training)` - each undone only if it was done; H3's AdaLN
  rows: `(context, speed)` for the render, `(adapter, context)` restored in `finally`; a failed preview disables
  previews and never ends the run.
* The H3 adapter ramp's size is `||B A||_F^2 * scale^2` in both codebases (TagScribeR `trace((A A^T)(B^T B))`, Fizgig
  `trace((B^T B)(A A^T))`, equal by cyclicity); gain caps 1.01 / 0.95 / floor 0.02 / start 0.1 match. The short-run EMA
  (`1 - 4/steps`, clamped to [0.5, 0.995], ramp 2) matches Fizgig; a tiny run lands at decay 0.5, which is fine.
* H3's noise-range boxes rescale sigma exactly as Fizgig 7.0.1 (`minimax/driver.py:428`); the noise-band LR
  multiplier averages over an accumulation window and composes into the group LR for the step only.
* Every key `pipeline.train_kwargs` emits is a `train_family` parameter; `driver_options` reach `configure`, which
  rejects unknown keys for Klein and bad block lists for H3 before anything loads.
* bf16 trainable adapters (H3's `trainable_dtype`): bitsandbytes 8-bit optimizers, fused AdamW and Automagic v3
  (state in `p.dtype`, maths in fp32) accept them; EMA and Adaptive LR snapshots cast on copy.
* Batch > 1 is refused for drivers without `supports_batching` both in preflight and in `train_family`; the loss watch
  stands down at batch > 1 and after resume replays its history.
* Caches: text caches invalidate on caption, `reference_sizes` and `text_cache_rev`; latents on resolution, control
  presence and `latent_rev` (but not on content - finding 2); shuffle-variant leftovers are removed.
* Resume: `training_state.json` is the commit marker, pruning is anchored to the LoRA name, a mismatched rank fails
  loudly, Adaptive LR state round-trips (its rollback snapshot is not persisted, as in Fizgig).

Only a real run can confirm (not reproducible on the tiny family): a prompt slider's three `no_grad` forwards per step
under block swap (the offloader's forward/backward pairing); Klein's `park_for_preview` / `unpark_after_preview`
re-splitting the swapped blocks exactly as the run had them; INT8 `_int_mm` fallbacks at odd preview sizes.

Observation, not a finding: a resumed run with gradient accumulation 1 fast-forwards the LR scheduler by `global_step`,
which counts steps the loss watch skipped; the schedule lands a few steps ahead of an unbroken run only when images
were excluded. Fizgig behaves the same.
