# Audit 05 - shared layer: Fizgig v8.7.1 vs TagScribeR (the delta since v7.0.1)

Read-only audit, 2026-10-19. Upstream = `.claude/fizgig-src` at v8.7.1 (commit `5d31b2f`); TagScribeR was ported
against v7.0.1 (commit `1c8ec88`), so every upstream citation here is a change between those two commits
(`git -C .claude/fizgig-src diff 1c8ec88 5d31b2f -- <path>`), 738 commits. Paths: "FZ:" is relative to
`.claude/fizgig-src/src/fizgig/` unless it starts with a repo-root file (`write_rocm_env.py`, `src/install/...`);
"TS:" is relative to the TagScribeR repo root. Line numbers are v8.7.1's and the work branch `port/stage1-bt67en`'s
(HEAD `98a48a2`).

Scope: the shared training layer only (`families/{train,lora,lorafile,description,driver,launch,ft,loss_watch,quant,
compile,epochs,multi_person,trigger_check,offline_check,refmod,checks,form,cache,clips,registry}.py`, `training/`,
`dataset/`, `modules/`, `networks/`, `rocm/`, `write_rocm_env.py`, requirements). Per-family drivers, descriptions
and new families (Z-Image Turbo, ACE-Step, Flux 3 Image Klein, MOSS-Image) are other agents' audits; they are named
here only where a shared change exists because of them.

**Limits**

- Nothing was run. Every "crashes" / "mispairs" below is reasoned from the code and says so; the unit test that
  would prove it is named.
- The release-notes digest the brief mentioned did not exist when this was written; the notes v7.1.0 to v8.7.1 were
  read directly (`docs/RELEASE_NOTES_v7.*.md`, `v8.*.md` in the Fizgig checkout).
- Audit 01 (`01_shared_layer.md`) covers v7.0.1 against TagScribeR. Items it listed as missing and that upstream
  did not change since are **not repeated** here (fine-tune loop, `is_reg`, `lorafile.py`, `extract.py`, clips, the
  Preferences `ModelFile` extras, `FamilyOption`, `cache_stage`, legacy state resume, ...). Where an upstream change
  moves the facts behind a settled decision, section 9 says so.
- Sizes: S = under ~50 lines, M = 50-250, L = more or a design decision, XL = several files plus UI and help.

---

## Summary - the items worth doing first

1. **Block swap pairs Linears through LoRA wrappers (#199, 8.5.8).** FZ `modules/offloading.py:50-58` keys the
   streamed Linears with a wrapper's `.base` dropped, so a block whose Linears carry a LoRA pairs with one whose
   don't. TagScribeR's copy (`TS:training/modules/offloading.py:61-63`, `:161-176`) pairs by full module name
   (CUDA / HIP path) or by `zip(modules())` (no-CUDA path). TagScribeR's MiniMax H3 wraps **only the trained
   window's** Linears (`TS:training/families/minimax_h3/driver.py:496-504`, the settled "window-only modules"
   decision), so blocks 0-19 are bare and blocks 20-49 wrapped; with any block swap the stock pairing `(i, i + 50 -
   N)` meets a bare block on one side and a wrapped one on the other, and `module_to_cpu.weight` is read on a
   `LoRALinear` that has no `weight` (`TS:training/lora.py:109-182` defines none) - the exact message upstream fixed.
   Reasoned, not run; the owner has not trained H3 yet. **S fix, highest value for the owner** (H3 on a 24 GB card
   needs swap). A tiny-family test: a 4-block model, LoRA on blocks 2-3, swap 2, one forward + backward.
2. **An AMD PC with built-in graphics trains on the card, not the iGPU (#186, 8.1.4; #188/#192, 8.5.1).**
   Fizgig's `src/install/pick_gpu.py:15-50` skips `is_integrated` / "AMD Radeon(TM) Graphics" and takes the most
   VRAM; the launcher puts it in `HIP_VISIBLE_DEVICES` (`Launch Fizgig AMD.bat:103-107`), and `write_rocm_env.py:
   105-170` bakes the same card's name / VRAM. TagScribeR: `core/hardware.py:157-196` lists devices in index order
   and `resolve_device("auto")` returns `devices[0]`; the trainer child uses `torch.device("cuda")` = device 0
   (`TS:training/train.py:546`) and `_small_card_previews` reads device 0 (`:248`); nothing sets
   `HIP_VISIBLE_DEVICES`. **Missing; S-M, own implementation** (`pick_gpu.py` carries no GPL header, unlike
   `detect_gpu.py`, but the rule is three lines and TagScribeR must not copy `detect_gpu.py` anyway).
3. **A failed resume-state save no longer ends the run (8.5.9).** FZ `families/train.py:1552-1574` retries once
   after 5 s, removes the half-written `-state` folder so pruning keeps the last good one, warns and trains on; a
   pause still stops (`required=True`, `:1849`). TS `:993-995`, `:1128-1131`, `:1141` has no retry. S.
4. **A picture edited after it was cached is encoded again (8.4.0).** FZ `scripts/cache_latents.py:109-122`
   re-encodes when the image's mtime is newer than its latent file. TS `training/cache.py:185-200` checks only
   existence, bucket, controls and `latent_rev`. Silent data error on the verified path (an edited photo trains on
   its old latent). S.
5. **Compile: a test compile before compiling blocks (8.3.1), and previews run uncompiled (7.2.2).** FZ
   `families/compile.py:104-137`, `:185` compiles one tiny function on the GPU first (an old Visual Studio passes
   `find_host_compiler` and fails Triton's link at the first kernel launch, outside dynamo's `suppress_errors`); FZ
   `families/train.py:554-557`, `:595-596` wraps preview generation in `torch.compiler.set_stance("force_eager")`
   (the first preview after every start / resume spent minutes compiling preview graphs). TS has neither
   (`training/compile.py`, `training/train.py:318-406`). S + S. NVIDIA users mostly (Auto leaves compile off on
   ROCm, `TS:training/compile.py:153`), but "On" on ROCm meets the same preview recompile.
6. **Per-step loss log and "Sample now" / "Save now" (8.0.0).** FZ `training/step_log.py` (new, 98 lines) writes
   `.fizgig_loss_<name>.jsonl` in batches of 20, resumable; `.sample_now` / `.save_now` flags are taken once between
   optimizer steps (`families/train.py:43`, `:1586-1588`, `:1718-1720`, `:1773-1780`, saved as
   `<name>-stepNNNNNNN.safetensors`). TS has only the epoch summary line (`training/progress.py
   parse_epoch_summary`). Trainer side S; the Train tab's loss graph and two buttons M.
7. **Krea 2 Auto: NF4 at 1 MP is 15.0 GB, not 13.4 (#185, 7.3.0).** FZ `families/krea2.py train_memory` (measured
   at rank 16 with previews on); TS `training/families/krea2/description.py:117` still 13.4, so a 16 GB card at 1 MP
   gets NF4 (fills the card) where Fizgig now gives INT8 with block swap. Family data, shared Auto plan; Auto only
   (the owner picks INT8). S - for the Krea 2 agent / owner.
8. **Sliders since 7.0.1** (TagScribeR ported 7.0.1's): the override prompt is ignored on a slider run (FZ
   `:1502-1504`, 7.2.0) - TS `:947-964` applies it; live prompt changes at an epoch boundary (`_read_slider_live`,
   `slider_live`, `render_bank`, `ss_slider_prompt_history`; FZ `:250-265`, `:1224-1252`, `:1373-1420`, `:1448`,
   `:1852-1855`; 8.0.3) - missing, M; the strip's label band sized to the text (FZ `:349-371`; 8.0.3) - S;
   `driver.practice_latent` (FZ `families/driver.py:210-216`) - TS inlines the same decode + encode (`:795-796`), S;
   slider presets at rank 8 on Qwen / Krea 2 / Anima (7.1.0) - family presets, other agents.
9. **DOP, Multi Concept on every model, Fizgig LoRA Isolation (8.4.0-8.4.2, 8.5.7)**: `families/multi_person.py`
   (new, 313 lines), `families/trigger_check.py` (new, 154), `scripts/multi_person.py` (new), DOP in the loop
   (`families/train.py:414-516`, `:1058-1127`, `:1693-1697`, `:1747-1762`), `--lr_boost`, `caption_replace` /
   `photos_only` / `item_te_paths` in the dataset, `save_cond_file` in the cache, the launch rules
   (`families/launch.py:1022-1120`, `:634-640`, `:745`, `:1254`). All missing. DOP alone is separable (M, model-
   agnostic, "Leave the rest of the model alone"); Isolation is XL (composites stage, a pinned 21 MB Hugging Face
   dataset of 32 generated strangers, Subjects card, help).
10. **Edit LoRAs with extra reference pictures (8.1.1 / 8.1.2)**: `description.edit_references` (Klein 4, Qwen 2.1
    and Flux 3 10), `extra_reference_directory`, extras at their own shape (`dataset/image_dataset.py:207-214`,
    `:1080-1085`), controls encoded one at a time (`families/cache.py:164-170`, `:187`; `families/train.py:314-324`,
    `:1160-1161`), `--sample_reference` takes several. Missing; M shared + per-driver work for Klein / Qwen.

Also found: the fine-tune is still documented NVIDIA-only (section 7); `retired_preview_defaults` / `preview_reset`
were removed upstream and TagScribeR's Qwen still uses them (section 2); upstream tried batch size > 1 (`supports_batch`,
commit `2016b80b`) and removed it again a day later (`658a728d`), so the settled "keep batch > 1" stands with no new
counter-argument (section 9).

---

## 1. `families/train.py` (1675 -> 2070 lines) vs `training/train.py` (1187)

| Upstream change (FZ:families/train.py) | What | TagScribeR | Verdict | Size |
|---|---|---|---|---|
| :43, :1583-1588, :1718-1720, :1793, :1831 + `training/step_log.py` | per-step loss log `.fizgig_loss_<name>.jsonl` (`{"s","e","l","lr"}`, flushed every 20 steps and at each epoch end; a resume keeps the lines up to its step) | none; `progress.py:69 parse_epoch_summary` reads one line an epoch | **missing** | S trainer, M Train tab graph |
| :1588, :1773-1780 | `.sample_now` / `.save_now` taken once between optimizer steps; save as `<name>-step{N:07d}.safetensors`; preview time left out of s/it | none (`.pause_requested` and `.sample_override.json` only, TS:25-27) | **missing** | S trainer, S Train tab |
| :1552-1574, :1830-1833, :1849 | state save retried once (5 s), half-written `-state` dir removed, warning instead of a crash; a pause's save still raises | TS:993-995 one try; TS:1128-1131 prunes after it | **missing** (8.5.9 fix) | S |
| :232-247, :1498-1513 | override holds a `prompts` list (or one `prompt`), `#` lines dropped; the encoding is cached while the prompts do not change | TS:254-267 one prompt; TS:947-964 re-encodes every preview | different (TagScribeR's dialog writes one prompt, `training/pipeline.py:652`) | S |
| :1502-1504 | a slider run ignores the override ("slider previews always show the slider's own prompt", 7.2.0) | TS:947-964 applies it to sliders too | **missing** | S |
| :554-557, :595-596 | previews under `torch.compiler.set_stance("force_eager")`: eval mode, the preview size and the speed LoRA are each a new graph (7.2.2) | none | **missing** | S |
| :519, :549-551, :588-591, :836, :1534, parser :1932 | `sample_strengths`: each preview again at these LoRA strengths, saved `<...>_s050.png`; not before epoch 1 | none | missing; only ACE-Step declares `preview_strengths` (FZ:families/acestep.py:174) | S, low |
| :874, :1527-1530 | checkpoint previews (Klein Distilled) take the Samples tab's Steps / CFG when given (`asked_steps`, `asked_cfg`), and the negative when CFG > 1 | TS:968-973 always the checkpoint recipe's steps / CFG, `neg=None` | **different** (8.1.x `a9fc8104`: "the distilled tick works as the Turbo LoRA does ... stay editable, and previews use what they say") | S |
| :314-324 `_load_references`, :1160-1161 | extra reference pictures keep their own shape at the preview's area; reference latents encoded one at a time | TS:308-315 all at the first picture's shape; TS:699 batch encode | missing (edit extras, section 6) | S here |
| :250-265, :1224-1252 `render_bank`, :1373-1420 `slider_live`, :1448, :1852-1855 | a prompt slider's prompts change at an epoch boundary (`.slider_prompts.json` -> `.slider_prompts_applied.json`), poles re-encoded, a new base re-renders the bank (optimizer parked on CPU when `preview_park_optimizer`), `ss_slider_prompt_history` in the metadata | TS:764-811 the bank inline, no live change | **missing** (8.0.3) | M |
| :1250 `driver.practice_latent(vae, lat, res)` | the bank item through a driver hook (ACE-Step's songs) | TS:795-796 decode + `encode_images` inline (what the default hook does) | different, same result for pictures | S |
| :349-371 `_slider_strip` | the label band fits the text (its height from `textbbox`, padded) | TS:100-116 fixed 30 px band | different (8.0.3 "strength labels no longer clip") | S |
| :414-516, :1058-1127, :1693-1697, :1747-1762, :1794-1797, parser :1994-2006 | **DOP** (`_dop_pairs`, `_dop_caption`, `_dop_captions`, `_dop_path`, `_dop_step`; trigger-free captions encoded once to `<text cache>_dop.safetensors`; a preservation pass every N steps at weight W with the adapter at 0 for the target; `block_swap_mode(inference=True)` around the no-backward forward with swap (#8.4.1); `automatic_dynamic_shapes = False` while DOP is on, :1105-1108); `--lr_boost PATTERN=MULT` composed into the step's LR (:1693-1695); a fine-tune's DOP targets made from the original model before step 1 and saved for a continuation (:486-516, :1656-1676) | none | **missing** | L (DOP + lr_boost M; fine-tune DOP with the fine-tune) |
| :1210 `held_grads=bool(dop_cond)` | the fine-tune planner sizes windows for a held gradient per weight (+8.3 GB measured on Krea 2) | n/a until the fine-tune exists | missing | with ft |
| :973-976 | batch size != 1 still refused for every family (the wording now points at `--gradient_accumulation_steps`) | TS:594-596 allows it for `supports_batching` | TagScribeR extension, settled; see section 9 (upstream added and removed `supports_batch`) | - |
| :983-1000 | `is_reg` regularisation blocks (7.0.1 code, unchanged) | missing (audit 01, 3.1) | unchanged | - |
| :1616-1640 `_update` | `m = 1.0` for `Automagic3` by class name | TS:1016 `owns_its_rate(optimizer)` | same | - |
| :1693 `_lrm *= lr_boost` | items whose `item_keys` hold a pattern train at MULT x LR | TS:1052-1058 has `step_policy` and `lr_acc` but no name rule | missing (needed by Isolation's composites) | S |

Not changed since 7.0.1 in this file, so audit 01's verdicts stand: the warm-up note conditions (01, 3.2 item 2),
`build_metadata` call shape, the DataLoader vs `dataset.shuffle` item order, the log-line format, the device
fallback.

## 2. `families/description.py` (622 -> 713 lines) vs `training/description.py` (331)

New fields and their consumers. "GUI" = read only by Fizgig's web form (`families/form.py`) or Preferences.

| FZ field | What | TS | Verdict | Size |
|---|---|---|---|---|
| :48 `ModelFile.replaces`, :54 `bulk_fetch`, :63-69 `fetch_is_optional` (a `training_adapter` row is never optional by default; 7.2.1) | downloads (Preferences) | `ModelFile` TS:20-33 has none of the download extras (audit 01, 1.1) | missing; n/a until TagScribeR fetches models | S data |
| :159-190 `VideoCheckpoint`, `VideoWorkbench`, :338 `video_controls` | H3 Studio pickers, Speed Boost | none | n/a (workbench, stage 6/7) | - |
| :201-203 `FamilyOption.warn`, `warn_link`, `featured`, :227 `isolation_off` | web form wording; a check greyed while Isolation is on | TS has no `FamilyOption` (its parameters are `training/params.py`) | n/a / with Isolation | - |
| :300 `status_note` | an orange line under the model picker ("new and improving daily", MOSS-Image) | none | missing, GUI | S |
| :331 `multi_concept: bool = True` (was False), :334 `subject_isolation: bool = True` | Multi Concept on every family; the Subjects card's DOP / Isolation offered | TS has neither field; its pipeline writes one dataset block (`training/pipeline.py:461-466`) though `training/dataset.py` takes several | missing | S data, M pipeline + tab for Multi Concept |
| :404 `ft_regularisation` | the fine-tune card offers a regularisation set (ACE-Step: no) | n/a until fine-tune | missing | S data |
| :408 `lr_hint = (2e-4, 125)`, :411 `long_run_hint`, :414 `song_captions` | Training-tab tip cards (7.2.0 large-dataset tip; ACE-Step long-run tip); Caption Every Song | none | missing, GUI; the large-dataset tip is a good Train tab item | S each |
| :420-422 `preview_checkpoint_tick`, `preview_checkpoint_on`, `preview_checkpoint_steps_note` | the Samples tick's words and default for a preview checkpoint (Klein on, Qwen Turbo off, MOSS SFT) | TS Klein ported with the tick (stage 2); wording hard-coded | different (data vs code) | S |
| :460 `edit_references: int = 1` | how many before-images an edit takes | none | missing (section 6) | S data |
| :464 `slider_sources`, :467 `kind_notes`, :470 `slider_prompt_notes` | a family with prompt-only sliders (ACE-Step); family wording | none | missing, GUI | S |
| :478 `preview_every`, :481 `preview_strengths`, :701 `sample_every_default` | Samples: a preview every N epochs kept per family (ACE-Step 40); extra strengths | none (`SAMPLE_EVERY` is one global parameter) | missing | S |
| :498-499 `block_activity`, `repair_default_strength`, :529 `lucky_zones`, :532 `detail_blocks` | Repair Studio | none | n/a (workbench) | - |
| :607 `audio_workbench`, :646-654 `cfg_note()` | sound-only family; the CFG line names the Turbo's CFG where it differs (Anima, Qwen 2.1) | none | `cfg_note` is a Train tab help line: missing, S | S |
| removed: `retired_preview_defaults`, `preview_reset` (were :396-402 in 7.0.1) | replaced by 8.6.1's rule "Steps and CFG always match what the previews render with" (the Turbo LoRA's 8 / 1, a ticked checkpoint's own, else the plain model's) | TS:217-222 still declares both; `training/families/qwen_image21/description.py:190-191` sets them | **obsolete-upstream**: TagScribeR keeps a mechanism Fizgig dropped. Remove once the Train tab applies 8.6.1's rule (see also the Qwen agent) | S |

Still missing from 7.0.1 and unchanged since (audit 01, 1.1): `GENERAL_NEGATIVE`, `ClipSpec`, `FamilyOption`,
`hidden`, `media`, `clip_spec`, `finetune`, `ft_learning_rate`, `preview_image`, `preview_park_optimizer`,
`options`, `settings_aliases`, `train_areas`, `architecture_entry`.

## 3. `families/driver.py` (439 -> 486) vs `training/driver.py` (390)

| FZ hook | What | TS | Verdict | Size |
|---|---|---|---|---|
| :74 `keep_text_encoder`, :76 `drop_text_encoder`, :79 `text_encoder_gb(path)` | the workbench keeps the encoder in RAM between prompts (8.5.9); the park check sizes it by what it loads | `_encode_override` sizes by file (`TS:training/train.py:273`, the default) | n/a (workbench) / same | - |
| :85-98 `from_pretrained(cls, repo_id)`, `helper_dir(repo_id)` | cache-first loads as static helpers on the base driver (7.1.0, with `offline_check.py`) | TagScribeR's families call `training/hf_cache.py` directly (stage 1) | present, different shape | S if wanted |
| :112 `text_cache_key()` | the workbench's prompt cache key (ACE-Step's song length) | none | n/a (workbench) | - |
| :210-216 `practice_latent(vae, latents, res)` | a prompt slider's bank item (songs) | inline (section 1) | different, same result | S |
| :475 `activity_axis(module_name)` | block activity (Repair Studio) | none | n/a | - |

All 7.0.1 hooks TagScribeR lacked (audit 01, 2.1) are unchanged upstream: still missing `cache_stage`'s callers,
`media_problem`, `clip_bucket_cap`, `expand_train_blocks`, `legacy_state_order`, `step_frozen_blocks`, `ft_*`,
`install_ft_streamer`, `encode_text_with_image`, `keep_vae_resident`, `decode_audio`.

## 4. `families/lora.py` (595 -> 668) vs `training/lora.py` (660)

| FZ change | What | TS | Verdict | Size |
|---|---|---|---|---|
| :120-124 `_base_device()` used at :135, :144, :151 | a new adapter's device read off the base's storage: a ConvRot INT8 Linear's `.weight` decodes the whole matrix (Qwen 2.1 Turbo's int8 checkpoint as a preview model, 7.2.0) | TS:161, :170, :177 `self.base.weight.device` | different; harmless today (TagScribeR's H3 ConvRot keeps `qdata` inside an autograd function, `training/families/minimax_h3/model.py:324-336`, no `.weight` property) - mirror when a quantised base exposes a decoding `.weight` | S |
| :82-109 `BlockMeter`, `METER`, :118 `_meter_block`, :157-160, :176-196 `_forward_metered` | Repair Studio block activity for songs | none | n/a (workbench) | - |

The fused frozen add (settled, mirrored) and everything else in `LoRALinear.forward` are unchanged.

## 5. `modules/offloading.py` (#199) - the block-swap pairing

FZ `modules/offloading.py:50-58` `_streamed_linears(layer)`: `{name: module}` of the Linears a swap streams, with a
wrapper's trailing `.base` dropped from the key; used by both swap paths (`:67-70`, `:169-176`). TS
`training/modules/offloading.py:61-63` (no-CUDA path: `zip(layer_to_cpu.modules(), layer_to_cuda.modules())`, any
module with a `weight`) and `:161-176` (CUDA / HIP path: `named_modules()` keyed by the full name, `endswith("Linear")`
+ `hasattr(weight)`).

Where TagScribeR meets it (reasoned from code, not run):

- `FamilyLoRA` wraps every **target** Linear at init (`TS:training/lora.py:289-311` `_wrap`, called for
  `driver.lora_target_names`), so a family whose targets cover every block (Krea 2, Qwen 2.1, Klein, SDXL, Anima)
  has the same wrapper layout in every block and pairs correctly, even with `train_blocks` (which only decides which
  wrappers get an adapter, `:318-321`).
- MiniMax H3's `lora_target_names` returns the trained window's modules only (`TS:training/families/minimax_h3/
  driver.py:496-504`, "Default 20-49"), so blocks outside the window have bare `nn.Linear`s. With block swap on
  (`training/families/minimax_h3/model.py:430-431` uses the shared `ModelOffloader`), the stock pairing swaps block
  `i` out as block `i + 50 - N` comes in; for every N in 1-49 at least one pair is bare-with-wrapped. On the CUDA /
  HIP path the wrapped side's `attn.q.base` finds no `attn.q.base` in the bare block (`:161-166`) and only gets a
  `.to(device)`; in the other direction the bare side's `attn.q` finds the `LoRALinear` named `attn.q` and reads
  `.weight.shape` on it -> `AttributeError: 'LoRALinear' object has no attribute 'weight'`, upstream's #199 message.
  Fizgig's own H3 never hit it because it writes zero adapters to every block (the settled decision kept
  TagScribeR's smaller file instead).
- The frozen H3 training adapter (`add_file`) wraps the Linears it reaches in every block, which may mask the
  mismatch for those Linears and leave it for the others; not traced further.

**Verdict: missing, S, highest value** (section 9: the settled window-only decision needs this fix to stay safe).
Verification: a tiny-family test with LoRA targets on a subset of blocks and `blocks_to_swap` > 0 on the CPU device
(the no-CUDA path hits the same mismatch through `zip(modules())`, so the test needs no GPU).

TagScribeR's CUDA guard (`:119-120`, `:301-306`, `:317-319`) is untouched by the upstream change; `_streamed_linears`
slots into both paths.

## 6. `families/cache.py`, `dataset/`, `scripts/cache_latents.py`

| FZ change | What | TS | Verdict | Size |
|---|---|---|---|---|
| `scripts/cache_latents.py:109-122` `_newer(item)` | re-encode when the image's mtime is newer than its latent file (8.4.0 "A picture edited after it was cached is encoded again") | `training/cache.py:185-200` checks existence, bucket, controls, `latent_rev` | **missing**; keep `latent_rev` and `_remove_stale` | S |
| `families/cache.py:79-90` `save_cond_file`, `cond_file_matches` | a text encoding at a path of its own in the text-cache format (DOP's trigger-free captions beside each item's cache) | `save_cond(..., path=, caption=)` TS:69 already takes a path and caption | present, different shape (TagScribeR's `save_cond` can serve DOP) | - |
| `families/cache.py:164-170` `limit = 1 if slider else edit_references`, :187 one control at a time | a multi-reference edit; a driver may stack its batch (Klein) | TS:273-281 refuses more than one partner; TS:203-204 batch-encodes controls | missing (edit extras) | S |
| `dataset/config.py:79, :85, :88`, `:149-152` | `extra_reference_directory`, `caption_replace [[from, to], ...]`, `photos_only` per dataset block | `training/dataset.py` config has none | missing (edit extras; Isolation) | S data each |
| `dataset/image_dataset.py:207-214` `_own_shape`, :1080-1085 | extra references at the target's pixel count in their own aspect, snapped to the bucket step; the first control at the target's crop | TS resizes every control to the bucket (not re-read here; audit 01, 14) | missing | S |
| `dataset/image_dataset.py:598` `item_te_paths` in every batch | the item's own text-cache path (two folders' same-named files share an `item_key`) | `training/dataset.py:446-465` returns `item_keys` only | missing (DOP keys on it) | S |
| `dataset/image_dataset.py:628-629`, `:795-797` `caption_replace` | whole-word, case-insensitive rewrites as a caption is read; the `.txt` files untouched | none | missing (Isolation's fused triggers) | S |
| `dataset/image_dataset.py:891-899` `photos_only` | a clip / voice family's folder trains from its stills | n/a (image-only dataset) | n/a | - |
| `dataset/image_dataset.py:691-696` | extra references globbed from their own folder, matched by `casefold` stem | TS:261-277 `partners` / `_control_for` do the case-insensitive match (stage 1) for one control folder | present for one folder | - |

## 7. `families/ft.py` (775 -> 797) - the full fine-tune, and is it still NVIDIA-only

Changes since 7.0.1 (all planner data, nothing in the training maths):

| FZ | What | Verdict |
|---|---|---|
| :67 `FTSpec.streams: bool = True`, :477 | a model whose forward cannot stream out-of-window blocks (Flux 3: its blocks do not run in one ordered list) plans only windows that fit with every frozen block resident (8.1.0) | n/a until the fine-tune is ported; a field every family port's `ft_spec` must consider |
| :86 `window_factor: float = 1.0`, :478-479 | a window costs this many times its bf16 weights (Z-Image 2.0: measured peaks grow by twice the window) (7.2.0) | same |
| :734-741 `plan(..., held_grads=False)` | DOP on a fine-tune holds every trained weight's gradient at once -> planned as `window_factor` 2.0 (8.4.0) | same; H3's own planner had to take the argument (8.5.7, `800ff5db`) |
| `families/launch.py:638` | **a fine-tune never sends `--adaptive_lr`** (8.1.3 / 8.4.0 "Fine-tunes no longer use Adaptive LR on any model") and 8.0.7 hides rank, alpha, Network Type, Model Area, the timestep range and Resume for a fine-tune | stage 5 must mirror both |
| `families/launch.py:151-200` `latest_state`, `latest_ft_checkpoint`, `ft_continuation`, `ft_rotations_left` | a paused fine-tune carries on from its last checkpoint (`fizgig_next_start_window`, `fizgig_ft_epochs_done`, `fizgig_ft_n_windows` in the checkpoint's metadata), shared by both apps (8.0.0) | n/a until the fine-tune; the metadata keys are the resume contract |
| `families/launch.py:676-690`, `:778-781` | the continuation checkpoint replaces `--dit` after the family options (H3's reference base, 8.0.x) | same |

**NVIDIA-only?** Upstream still says so: `docs/FINETUNE.md:138` "NVIDIA only. Fine-tuning is untested on AMD/ROCm:
every measured tier is NVIDIA, and the NF4 default relies on bitsandbytes 4-bit, the least-travelled part of the ROCm
stack". The code has no explicit vendor check: `families/ft.py` imports `krea2/rotation.py`, whose `RotationOffloader`
turns prefetch on whenever `device.type == "cuda"` (`krea2/rotation.py:640-645`, `torch.cuda.Stream`, a worker thread,
`cuda.Event`), which is also true on a HIP build. TagScribeR's own `training/modules/offloading.py` streams on HIP the
same way (`:119-120`), and the owner's verified Krea 2 runs used **no block swap**, so neither streaming path has
been run on the owner's card. TagScribeR's NF4 (`training/modules/nf4.py`, bitsandbytes 4-bit) has not been run on
real weights on ROCm either. So: not NVIDIA-only by code, NVIDIA-only by evidence. A port must say "not run on AMD"
in the docs (standing rule 8 / 9) and keep the fine-tune's streamer behind the same CUDA guard as the offloader.

## 8. Multi-person mode, DOP, LoRA Isolation, trigger check (new upstream files)

All missing in TagScribeR. What each piece is, with its shared-layer dependencies:

| FZ | What | Dependencies | Size |
|---|---|---|---|
| `families/train.py` DOP (section 1) | Differential Output Preservation (AI-Toolkit's, by Ostris): each image also trains with its trigger swapped for its class word, the target the base model's own prediction with the adapter at 0; a pass every 2 steps at weight 2 is the launched recipe (`launch.py:634-636`), the loss's gradient joins the step's own | text encoder at the start (`launch.py:788-789`), `item_te_paths`, a `_dop` text-cache file per item, `set_trainable_multiplier(0)`, `block_swap_mode(inference=True)` around the no-backward forward | M |
| `families/multi_person.py:22-34` constants, :55 `dop_caption`, :73 `fuse`, :198 `subject`, :266 `build`, :148 `strangers` | fused triggers ("zwxem, a woman" -> `zwxemwoman` via `caption_replace`, the user's files untouched); 16 side-by-side 1024x512 composites per person next to one of 32 fixed, generated strangers (a pinned Hugging Face dataset revision, 21 MB, downloaded once; the copy on disk is used only at the pinned revision), captions stranger-first; the composites train at 2x LR (`--lr_boost fizgig_mp_=2`); people / animal-or-thing / style recipes | `caption_replace`, several dataset blocks (Multi Concept), `lr_boost`, DOP, a pre-cache stage (`scripts/multi_person.py`, 26 lines), a Subjects card, help; loss watch / per-image LR / look warm-up and auto-recaption forced off (`launch.py:694-699`); `isolation_off` options | XL |
| `families/trigger_check.py` | a trigger word checked on the model's own tokenizer (does `zwxem` + `woman` keep "woman" whole; shared pieces / first letters with other subjects; five made-up suggestions), tokenizer only, CPU | `training/hf_cache.py`; per-family tokenizer specs (:14-25; SDXL's CLIP, Anima's, Qwen's processor, Klein's Qwen3-8B, Krea 2's bundled Qwen3-VL) | S-M (a Caption / Train tab tool) |
| `families/launch.py:1022-1120` | `automagic()` (Automagic v3 owns the rate: Adaptive LR, per-image LR and the look warm-up forced off at plan time, `:1229-1231`; a LR above 1e-3 with Automagic refused, `:1143-1150`), `dop_on`, `multi_person_on`, `multi_person_subjects`, `multi_concept_problems`, `_balance_repeats` (Multi Concept's balance: smaller folders repeated to about the largest's count) | TagScribeR's `training/pipeline.py preflight` has the Automagic stand-down in the loop (`train.py:827-837`) but no plan-time refusal of an AdamW-sized LR under Automagic | S each |

Numerics: none of this changes a run with the Subjects card off. With DOP on, every step gains a second forward
(and a no-grad forward) and the loss changes; Fizgig measured 48/50 -> 0/50 stranger bleed on Krea 2 at equal
likeness (8.4.0). "Tested on Krea 2, with more model testing to follow" - the owner trains Krea 2.

## 9. Settled decisions that newer upstream code touches

| Decision (PORT_PLAN) | New fact | Effect |
|---|---|---|
| H3: the LoRA holds only the trained window's modules (kept) | #199 (section 5): a swap pairing a bare block with a wrapped one crashes without `_streamed_linears` | the decision stands **only with the offloading fix**; alternatively write zero adapters as Fizgig does (not recommended: bigger files, same training) |
| Batch size > 1 kept | upstream added `FamilyDriver.supports_batch` for MOSS-Image (`2016b80b`, 17 Oct) and removed it the next day (`658a728d`: "Left out for now: batch size ... shared training features, for their own issues"); `families/train.py:973-976` refuses != 1 again | no new argument either way; keep. TagScribeR's `supports_batching` name stays its own |
| Krea 2 memory figures taken from 7.0.1 (verified) | NF4 at 1 MP is 15.0 GB upstream (#185, measured rank 16 with previews) | Auto on a 16 GB card at 1 MP: take 15.0 (TS `krea2/description.py:117`); the owner's INT8 choice is unaffected |
| Klein's Attention Mechanism dropdown removed (a dead widget) | 8.0.6 removed the Attention choice for every model ("every model renders with torch attention") | confirms the call |
| Shared compile decided on the empty card (stage 1) | previews now run uncompiled (7.2.2); a test compile precedes `compile_blocks` (8.3.1) | audit 01 risk R5 shrinks: the first preview no longer recompiles; both are S additions to `training/compile.py` / `train.py` |
| Fine-tune refuses pre-quantised sources (stage 5 note) | plus: no Adaptive LR on any fine-tune; `streams` / `window_factor` / `held_grads` planner fields; the resume contract keys | stage 5 scope grows by S items |
| Sliders ported from 7.0.1 (just landed) | 7.2.0 override ignored on sliders; 8.0.3 live prompts and strip band; 7.1.0 slider presets at rank 8 (Qwen / Krea 2 / Anima) | section 1 rows; presets are the family agents' |
| Qwen `retired_preview_defaults` / `preview_reset` (ported with Qwen) | both fields removed upstream; 8.6.1 replaced them with "Steps and CFG always match what the previews render with" | remove the fields and the Qwen values once the Train tab applies 8.6.1's rule (coordinate with the Qwen agent) |
| Caching skips existing latents (`latent_rev`, settled) | 8.4.0 re-encodes a picture whose mtime is newer than its latent | add the mtime test beside the `latent_rev` test; nothing settled changes |

## 10. ROCm / AMD

| Item | FZ | TS | Verdict |
|---|---|---|---|
| VAE wide-head attention (#179, 7.1.0) | `modules/wide_attention.py` (new): on ROCm only, heads > 256 computed in fp32 in row slices; SDPA everywhere else; `rocm_vae_processor()` for a diffusers VAE (SDXL); used by `krea2/vae.py:372`, `qwen_image21/vae.py:497`, `klein/model.py:144`, `sdxl/driver.py:137-139`; Anima shares the Krea 2 VAE | `training/modules/wide_attention.py` (TagScribeR wrote it first): the by-hand path for every backend above head 256 (`:27-35`), used by Krea 2, Qwen 2.1, Klein VAEs; Anima shares the Krea 2 VAE loader (`training/families/anima/driver.py:44-46`); SDXL uses diffusers' classic `AttnProcessor` instead (`training/families/sdxl/vae.py:46-49`, correct on ROCm, runs in the model dtype) | **present** (preserved item). Difference: on NVIDIA Fizgig keeps SDPA (PyTorch falls back to its math path itself); TagScribeR's fp32 chunked path runs there too - correct, slower. Recommend gating on `torch.version.hip` as Fizgig does (S). Confirms the owner's fix was right: upstream measured heads 64 / 128 / 256 correct, 384 / 512 wrong |
| Discrete card over built-in graphics (#186, 8.1.4) | `src/install/pick_gpu.py` (own code, no GPL header; `detect_gpu.py` next to it is the GPL one): skip `is_integrated` / "AMD Radeon(TM) Graphics", most VRAM; `Launch Fizgig AMD.bat:103-107` sets `HIP_VISIBLE_DEVICES` unless already set; Linux launcher does not (not ported there) | `core/hardware.py:157-196`: index order, `devices[0]`; trainer and cache children: device 0; `core/vram_monitor.py:686` reads `CUDA_VISIBLE_DEVICES` | **missing**, S-M: an own rule in `core/hardware.py` (prefer non-integrated, most VRAM; `is_integrated` is a `torch.cuda.get_device_properties` field on recent torch), passed to child processes as `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES`, honoured when the user set one |
| GPU name / VRAM baked into `rocm_env` (#188, #192, 8.5.1) | `write_rocm_env.py:105-170`, `:300-307`, `:336-341`: `FIZGIG_AMD_GFX`, `FIZGIG_AMD_VRAM_TOTAL`, `FIZGIG_AMD_GPU_NAME` for a web server that must not import torch | TagScribeR reads the card in-process (`core/hardware.py`, `core/vram_monitor.py`) | n/a (web UI architecture); the *card choice* is item 2 |
| `rocm_env.bat` / `.sh` moved to `user_data/` (8.0.0); launchers renamed | `write_rocm_env.py:22-23`; `launch_fizgig_amd.sh`; `rocm/cache_exit.py` comment | TagScribeR has no launcher files; `core/hardware.apply_runtime_env` sets the same env (`:68-80`) | n/a |
| Pins | unchanged since 7.0.1 (`docs/INSTALL.md:82-86`: torch `2.12.0+rocm7.15.0a20260728`, the 0xDELUXA bitsandbytes wheel) | `tools/install.py:41-60` the same | present |
| ROCm env in the launchers | `Launch Fizgig AMD.bat:33-37`: `MIOPEN_FIND_MODE=2`, `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, `PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.8`; RDNA1/2 math-only SDPA by gfx | `core/hardware.py:74-77`, `:220-231`: `MIOPEN_FIND_MODE=2`, AOTriton experimental, `PYTORCH_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8`, RDNA1/2 math SDPA | present; `FLASH_ATTENTION_TRITON_AMD_ENABLE` and `max_split_size_mb:512` are not set by TagScribeR (both predate 7.0.1 - not a delta item; noted for completeness, unverified which matters) |
| H3 long captions on pre-RTX-30 cards (7.2.0, by a contributor on 4xV100) | `minimax/` chunked text encoding + lighter attention path | family scope | for the H3 agent |
| Fine-tune on ROCm | still documented NVIDIA-only (section 7) | - | - |
| Compile on ROCm | `utils/capabilities.py` unchanged since 7.0.1 | TS Auto leaves compile off on HIP (`training/compile.py:153`) | present (TagScribeR's own rule) |

## 11. Other new or changed upstream files in scope

| FZ | What | TS | Verdict |
|---|---|---|---|
| `training/step_log.py` (new) | section 1: `StepLog`, `take_request`, `read()` for the graph (every k-th point above 4000, epoch starts) | none | missing, S (+ Train tab) |
| `families/epochs.py` (new, 43) | torch-free: a run's saved epochs beside a file (`run_epochs`), and `ema_weights(epochs, up_to, history)` - Repair Studio's "Epochs of this run" moving-average mix | none | n/a (workbench); S if an epoch blend tool is wanted |
| `families/offline_check.py` (new, 80) | proves a family's text encoder and VAE load with `HF_HUB_OFFLINE=1` in a child process, after checking every `helper_files` repo is cached (7.1.0 driver-guide requirement) | TagScribeR has `training/hf_cache.py` but no proof tool | missing; S as `tools/offline_check.py` - worth having (the owner trains behind a flaky connection; "works offline" is a standing promise) |
| `families/refmod.py` (new, 447) | H3 RefMod maker options and stages, shared by both Fizgig apps | none | stage 6 (RefMods) |
| `families/form.py` (new, 1145) | the Training / Samples forms as data for the web UI | TagScribeR's `training/params.py` is its equivalent | n/a (its *rules* matter: 8.0.7 hides Network Type, Model Area, Adaptive LR, EMA and the loss watch for a slider, `:858`; rank / alpha / Network Type / Model Area / timesteps / Resume for a fine-tune, `:749`) |
| `dataset/prep.py` (new, 506) | Image Prep (resize / PNG / face crops) and the Look Consistency Filter maths, shared by both apps; writes `fizgig_look_scores.json` which the look warm-up reads | TagScribeR has its own image tools; `training/loss_logger.py` reads `tagscriber_look_scores.json` or `fizgig_look_scores.json` (audit 01, 10.2) | roadmap (README "image cleanup"); the scores-file format is the contract to keep |
| `families/launch.py` (+451) | sections 6-8; plus `recache` (resume at a new Target Megapixels: the cache stages run again and keep what fits, `:154-165`; 8.0.0), `--sample_strengths`, `preview_extras`, slider preview cadence at least every checkpoint (`:838-845`) | `training/pipeline.py:626-630` never caches on a resume | resume-at-new-size missing, S (the cache already keeps what fits by bucket) |
| `families/registry.py` | four new families registered (Z-Image, ACE-Step, Flux 3, MOSS-Image) | - | other agents |
| `families/description.py:159` `VideoWorkbench` etc. | section 2 | - | n/a |
| `training/automagic3.py` | header comment only (`THIRD_PARTY_NOTICES.md` moved to `docs/`) | identical code | present |
| `requirements.txt` | + `pywebview` (the app window), + `soundfile` (song lengths) | neither needed | n/a. Still not in TagScribeR from 7.0.1: `einops`, `hqq`, `insightface`, `imageio-ffmpeg` (PORT_PLAN) |

**Unchanged between 1c8ec88 and 5d31b2f** (so audit 01's verdicts stand as written): `families/quant.py`,
`families/checks.py`, `families/loss_watch.py`, `families/lorafile.py`, `families/extract.py`, `families/clips.py`,
`training/loss_logger.py`, `training/adaptive_lr.py`, `training/ema.py`, `training/optimizers.py`,
`training/metadata.py`, `training/progress.py`, `training/train_utils.py`, `networks/lora.py`,
`modules/{int8_train,nf4,sdpa,fp8,compile_util,int8,int8_attention,attention,schedulers}.py`,
`utils/capabilities.py`, `utils/hf_cache.py`.

## 12. Changes that move a real run's numerics

Only these change what a run computes; everything else in this audit is control flow, files or UI.

| Change | When it applies | Direction |
|---|---|---|
| DOP / `lr_boost` (section 8) | only with the Subjects card on | new loss term; opt-in |
| `caption_replace` | only in multi-person mode | the captions the text encoder sees |
| Krea 2 NF4 1 MP memory 15.0 (#185) | Auto on a 16 GB card at 1 MP | picks INT8 + swap instead of NF4: different base weights for that card class |
| Checkpoint previews at the Samples tab's steps / CFG | Klein Distilled previews with a typed Steps / CFG | previews only |
| Previews uncompiled (`force_eager`) | compiled runs | previews only (the eager graph is the reference the compiled one approximates) |
| Edit extras at their own shape | multi-reference edits | the reference latents' shapes |
| mtime re-encode (8.4.0) | an edited picture under the same name | the run trains on the new pixels (a fix) |
| No Adaptive LR on a fine-tune | fine-tunes (stage 5) | the LR trajectory |

Nothing changed upstream since 7.0.1 in the objective, timestep sampling, optimizer defaults, EMA, Adaptive LR's
rules, LoRA init / scale / dtype, INT8 / NF4 quantisation, the cache formats or the LoRA key formats (section 11's
unchanged list).

## 13. Dependency-ordered port list

Steps 1-6 need nothing else and fix or protect real runs; 7-10 are features; 11-13 are the large items.

1. **Block swap through LoRA wrappers** (`_streamed_linears` in both swap paths of `training/modules/offloading.py`,
   keeping the CUDA guard) + a tiny-family test with a block subset and swap on CPU. **S.**
2. **Latent cache: re-encode when the image is newer than its latent** (`training/cache.py run_latents`), keeping
   `latent_rev`. S. Unblocks nothing; fixes silent staleness.
3. **State save retry / half-written state removed / pause still raises** (`training/train.py state`). S.
4. **Discrete AMD card over built-in graphics**: an own rule in `core/hardware.py` (`is_integrated`, name,
   most VRAM), the device list ordered by it, `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` for the cache and
   train children (`training/pipeline.py`) unless the user set one, a console line saying which card. S-M. Help
   text.
5. **Compile**: `_compile_works()` before `compile_blocks` (with the Windows Build Tools message), and previews under
   `torch.compiler.set_stance("force_eager")` in `_render_previews`. S + S.
6. **Small loop fixes**: override ignored on sliders; checkpoint previews take typed Steps / CFG (and the negative
   when CFG > 1); the slider strip band sized to the text; Krea 2 NF4 1 MP 15.0 (family data). S each.
7. **Per-step loss log and Sample now / Save now**: `training/step_log.py` port (TagScribeR file names), the loop
   hooks, `pipeline.request_sample_now / request_save_now`; then the Train tab's graph and buttons. S + M.
8. **Description data** (pure data, nothing reads them until a tab does): `edit_references`, `preview_every`,
   `preview_strengths`, `slider_sources`, `kind_notes`, `slider_prompt_notes`, `lr_hint`, `long_run_hint`,
   `status_note`, `cfg_note()`, `multi_concept`, `subject_isolation`, `ft_regularisation`, the preview-checkpoint
   tick wording; drop `retired_preview_defaults` / `preview_reset` with 8.6.1's Samples rule (coordinate with the
   Qwen agent). S each.
9. **Resume at a new Target Megapixels** (`recache`): the Train tab offers it when the size changed; the cache
   stages run with skip-existing and the resumed run goes on. S-M.
10. **Slider live prompts** (`.slider_prompts.json`, `render_bank` as a function, `slider_live` at the epoch
    boundary, `ss_slider_prompt_history`, a Train tab panel) and `practice_latent` as a hook. M.
11. **Edit LoRAs with extra reference pictures**: `extra_reference_directory`, `_own_shape`, one control at a time
    in the cache, `_load_references` and `ref_latents` per picture, the pipeline's extras folder checks
    (`launch.py:83-130` rules), `edit_references` per family; Klein's and Qwen's drivers must take several
    references (family agents). M shared.
12. **Multi Concept on every model** (several dataset blocks from the Train tab, balance repeats, per-folder
    captions checks) -> **DOP** (`item_te_paths`, the `_dop` text-cache file through `save_cond`, `_dop_step`,
    `dop_every` / weight recipe 2 / 2, `block_swap_mode` around the no-grad forward, dynamic shapes off while on)
    -> **`lr_boost`** -> **Fizgig LoRA Isolation** (`multi_person.py`, the composites stage, `caption_replace`, the
    strangers download at a pinned revision into TagScribeR's own data folder, the Subjects card, the trigger
    check tool, loss-watch stand-down, help). M + M + S + XL. Each step is usable on its own.
13. **Full fine-tune** (stage 5, unchanged plan) now including: no Adaptive LR, `streams` / `window_factor` /
    `held_grads`, the continuation metadata keys, 8.0.7's hidden controls, the fine-tune's DOP targets if 12 landed,
    the streamer behind the CUDA guard, docs saying "not run on AMD". L.
14. `tools/offline_check.py` (S) and, when a family needs it, `trigger_check` as a caption tool (S-M).

---

## 14. Questions for the owner (recommendations, not open questions)

1. **#199 fix.** Recommend: port `_streamed_linears` now (S) and keep the window-only H3 LoRA. The alternative
   (zero adapters in every block, as Fizgig) only grows the file.
2. **AMD card choice.** Recommend: an own rule in `core/hardware.py` (not a copy of `pick_gpu.py`, which is Fizgig's
   own code but sits beside the GPL `detect_gpu.py`; the rule is trivial), applied to the app's device list and to
   the training children's environment. Honour a user-set `HIP_VISIBLE_DEVICES`.
3. **wide_attention on NVIDIA.** Recommend: gate the by-hand fp32 path on `torch.version.hip` as Fizgig does; NVIDIA
   keeps SDPA (PyTorch's own math fallback). No change for the owner's card.
4. **Loss graph and Sample now / Save now.** Recommend: the trainer side now (S, no UI risk), the Train tab graph and
   buttons as a stage-5 UI item.
5. **DOP before Isolation.** Recommend: DOP + Multi Concept (M) with stage 5; Isolation (XL, a 21 MB pinned
   download of generated strangers, a Subjects card) after the fine-tune. Fizgig's measurements are on Krea 2, the
   owner's family.
6. **`sample_strengths`.** Recommend: skip until a TagScribeR family declares `preview_strengths` (only ACE-Step does).
7. **Krea 2 NF4 1 MP = 15.0.** Recommend: take it (Auto only; the owner's INT8 runs are untouched).
8. **`retired_preview_defaults` / `preview_reset`.** Recommend: remove, replacing them with 8.6.1's rule in the
   Train tab (Steps / CFG follow what previews render with; a typed value is kept).
9. **Batch size > 1.** No action: upstream tried it for MOSS-Image and backed it out within a day; the settled call
   stands.
10. **Fine-tune on AMD.** Recommend: port it in stage 5 with the streamer behind the CUDA guard and the docs saying
    "unit-tested only; not run on AMD or NVIDIA by the owner" until a real run reports back.

## 15. What I did not check

- Nothing was executed; the #199 reasoning (section 5) is from reading the two swap paths and TagScribeR's H3
  targets, not from a run.
- Per-family files (`krea2/`, `klein/`, `qwen_image21/`, `minimax/`, `sdxl/`, `anima/`, `zimage/`, `acestep/`,
  `flux3/`, `moss_image/`, their descriptions and drivers): other agents; cited only for `train_memory` (#185),
  the VAE wide-head call sites and `lora_target_names`.
- `families/form.py` beyond its hide rules; `families/refmod.py` and `dataset/prep.py` beyond their headers;
  `scripts/studio_worker.py`, `profile_run.py`, `extract_*`, `transcribe_clip.py`, `translate_captions.py`
  (workbench / web / captioning, out of scope).
- `families/launch.py` flag by flag against `training/pipeline.py` (audit 01 left this open; only the new 8.x
  functions were read).
- Whether the H3 frozen training adapter's wrappers mask part of the #199 mismatch (section 5, third bullet).
- The web UI's `.sample_override.json` writer and `.slider_prompts.json` format beyond what the trainer reads.
- `docs/drivers/README.md`'s new checklist items beyond the grep in this audit (memory-from-file method,
  "Second environments" rule) - relevant to how family ports are done, not to the shared layer.
- Whether `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` or `max_split_size_mb:512` (Fizgig's AMD launcher, pre-7.0.1)
  matter on the owner's card.
