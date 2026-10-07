# Audit 01 - shared layer: current Fizgig (v7.0.1) vs TagScribeR `training/`

Read-only audit. Upstream = `.claude/fizgig-src/src/fizgig/` (HEAD `1c8ec88`, "Release notes v7.0.1"). Paths below
are relative to that folder (upstream, "F:") or to the TagScribeR repo root ("T:").

**Limits of this audit**

- The Fizgig copy is a depth-1 clone: baseline commit `746ddab` is not in it, so "which side changed" is inferred
  from TagScribeR's file headers ("Changes for TagScribeR: ...") and the release notes, not from a git diff. Where I
  could not tell, I say so.
- Nothing was run. No tests, no GPU.
- Family drivers (`krea2/driver.py`, `klein/driver.py`, ...) are out of scope; they are cited only where a shared-layer
  hook exists because of them.
- Size estimates: S = under ~50 lines, M = 50-250, L = more, or needs a design decision.

The audit is complete: sections 1-16 are the file pairs, 17 the port plan, 18 the risks, 19 the questions for the
owner, 20 what was not checked.

---

## 1. `families/description.py` (622 lines) -> `training/description.py` (265 lines)

### 1.1 Upstream has, TagScribeR lacks

| Upstream (F:families/description.py) | What | Size |
|---|---|---|
| :17 `GENERAL_NEGATIVE` | shared default negative prompt | S |
| :36-54 `ModelFile.hint/download_label/download_note/alt_repo/alt_path/alt_label/alt_note/gated/fetch_optional/announce/inside`, :56 `fetch_is_optional` | Preferences-row extras; `inside` = a part that ships inside another file (SDXL) | S |
| :105 `LoRAFormat.lokr_kohya_stems` | LoKR keys on `lora_unet_` stems for SDXL | S |
| :123-136 `ClipSpec` | video clip contract (H3) | S (data), L with clips |
| :139-219 `FamilyOption` (+ `pick`, `resolve`) | a family's own controls as launch tokens (`name=value`, `--flag`, `aux:`) | M |
| :233 `hidden` | registered but not offered | S |
| :241-269 `shares_prefs_with`, `model_note`, `preset_notes`, `reference_strength`, `extract_presets`, `samples_text`, `samples_cfg_free`, `prefs_*`, `fetch_*`, `announce_intro` | GUI wording / Preferences | S each |
| :272-280 `media`, `multi_concept*`, `clip_spec` | photo / clip / voice items, Multi Concept | S (data), L behaviour |
| :304 `ema_short_run`, :307 `resumes_untagged_states` | EMA "Short run"; resume of untagged states | S |
| :312 `auto_precisions` | Auto's own order (Krea 2: INT8 then NF4) | S |
| :315-316 `precision_labels`, `precision_hint` | per-family precision wording (Klein) | S |
| :319-333 `compiles`, `compile_payback_steps`, `compile_boundary`, `compile_fullgraph`, `compile_memory`, `compile_hint` | generic torch.compile rule | S (data) |
| :336 `preview_image`, :339 `preview_checkpoint_sampling`, :341 `preview_park_optimizer`, :349 `train_preview_checkpoint` | vision-path preview image; preview checkpoint (Klein Distilled); optimizer parked on CPU for previews (H3) | S (data) |
| :344-345 `finetune`, `ft_learning_rate` | full fine-tune offered | S (data) |
| :358-361 `optimizer_weight_decay`, `optimizer_eps_floor_8bit`, `trainable_dtype` | per-family optimizer defaults; bf16 trainable adapter (H3) | S |
| :365-367 `optimizer_families`, `automagic_sign_window` | Automagic v3 parameter groups declared as data | S |
| :371-378 `adaptive_lr_clip_signal`, `adaptive_lr`, `loss_watch`, `network_hint`, `ema_hint`, `ema_section`, `precision_label`, `precision_after_states`, :381 `samples_turbo_pace` | Klein's clip signal as a per-family flag; controls hidden per family | S |
| :384-388 `slider_training`, `slider_guidance`, `slider_ultra_blocks` | slider LoRAs | S (data) |
| :394 `preview_cfg_note`, :397 `preview_negative: Optional[str] = None` | `None` = previews take no negative | S |
| :400-404 `int8_attention`, `activation_cache`, `workbench_follows_samples`, :409 `repair_size` | workbench | S |
| :433-453 `workbench_engine`, `clip_regimes`, `repair_presets`, `block_categories`, `category_masters`, `identity_blocks`, `train_areas` | workbench + Fast Identity Mode + Model Area to Train as data | S (data) |
| :457 `options`, :460 `settings_aliases` | `FamilyOption` list; old-preset key aliases | S |
| :472 `reference_kind`, :477 `preview_checkpoint()`, :490-511 `workbench_engine_class`, `make_workbench_engine`, `video_workbench`, :529 `model_path()` | derived helpers | S |
| :558-593 `architecture_entry()` | Tk GUI adapter | not needed (T header says it was dropped on purpose) |

### 1.2 Differ where both have them

- `precisions` comment: F:309 lists `"bf16", "int8", "nf4"`; T:training/description.py:131 the same comment, but
  T:training/quant.py:27 adds `"fp8"` (TagScribeR-only, preserved). Upstream `PRECISIONS` is now
  `("bf16", "int8", "nf4", "hqq")` (F:families/quant.py:22). Upstream side changed (added `hqq`); TagScribeR side
  added `fp8`.
- Auto order: F:312 `auto_precisions` (one tuple) vs T:138-139 `auto_order` + `auto_swap_order` (two tuples).
  TagScribeR invented its own names before upstream had the field; upstream then added `auto_precisions`. A port must
  map one onto the other (TagScribeR's `auto_swap_order` has no upstream counterpart - upstream always swaps
  `("int8", "bf16")`, F:families/quant.py:159).
- "A part inside another file": F:54 `ModelFile.inside` holds a **pref key**; T:30 `ModelFile.default_to` holds a
  **role**. Same purpose (SDXL), different field and different value. Upstream added `inside` with SDXL in 7.0.0;
  TagScribeR's `default_to` predates it.
- Family options: F:457 `options` (tuple of `FamilyOption`) vs T:144 `family_options` (tuple of parameter keys from
  `training/params.py` `DRIVER_OPTIONS`). Different design on each side.
- `preview_negative`: F:397 `Optional[str] = None` (None greys the box) vs T:155 `str = ""` ("" = the app default).
- Entry points: F:464-465 `train_script` / `cache_script` (paths) vs T:183-184 `train_module` / `cache_module`
  (TagScribeR change, intended).
- `validate()`: identical rules, plus TagScribeR's `default_to` checks (T:255-260).

### 1.3 Bug fixes to pick up

None in this file (it is data).

### 1.4 Upstream removed, TagScribeR still has

Nothing removed. (`architecture_entry` is still upstream; TagScribeR dropped it.)

---

## 2. `families/driver.py` (439 lines) -> `training/driver.py` (202 lines)

### 2.1 Upstream has, TagScribeR lacks

| Upstream (F:families/driver.py) | What | Size |
|---|---|---|
| :83 `cache_stage(stage, datasets, args, device, aux)` | a family caches in its own layout (H3) | S hook, L for H3 |
| :89 `media_problem`, :95 `clip_bucket_cap` | clip / voice checks | S hooks |
| :101-105 `options`, `set_options(options)` | `--family_option KEY=VALUE`, set before the dataset is built | S |
| :107 `prepare_training(dit, group, net=None)` | see 2.2 | - |
| :111 `step_frozen_blocks(batch)` | per-step block routing | S hook (+ loop code, see 3) |
| :115 `expand_train_blocks(items)` | `--train_blocks` spec -> block ids | S |
| :120 `legacy_state_order(dit)` | remap an older trainer's optimizer / EMA order on resume | S hook |
| :126 `after_optimizer_step()` | H3 adapter-relative LR ramp | S |
| :129 `run_metadata()` | extra `ss_*` keys | S |
| :133 `step_policy(batch, epoch)` | (skip, lr multiplier) per step | S hook |
| :139 `batch_cond(batch, device)` | maps cached `cond__` keys to the conditioning dict | S |
| :157-158 `training_loss(..., diff_ref=None, diff_weight=0.0)` | image-pair slider weighting | S (signature) |
| :167-188 `slider_setup`, `still_renders`, `slider_preview`, `noise_latents`, `predict` | prompt-pair sliders | S hooks, M per driver |
| :196-198 `generate(..., frames=1, audio=False)` | video | S |
| :221-234 `keep_vae_resident`, `decode_audio`, `save_preview(result, path)` | preview written by the driver (clips) | S |
| :252 `alias_flat(flat)` | other trainers' module names (fix in 6.8.2) | S hook |
| :258 `convert_lora_state_dict(sd)` | Klein diffusers q/k/v -> qkv | S hook |
| :268 `encode_text_with_image(te, captions, image, megapixels)` | vision-path preview image | S hook |
| :274-314 `ft_spec`, `ft_backend`, `ft_card_plan`, `ft_cycle`, `ft_source_unfit`, `install_ft_streamer` | full fine-tune | S hooks, L feature |
| :316 `frozen_file_added(dit, path, strength, role)` | after a frozen LoRA file is attached | S |
| :323-331 `park_for(dit, device, need_gb, purpose)`, `unpark` | driver-owned parking for decode / override encode | S |
| :334-345 `park_for_preview`, `load_preview_checkpoint`, `unpark_after_preview` | previews on a preview checkpoint (Klein Distilled) | S hooks, M feature |
| :347-358 `compile_targets`, `compile_blocks` | shared torch.compile | S |
| :360 `plan_run(precision, blocks_to_swap, *, group, run)` | family's own Auto plan (H3) | S |
| :368 `load_planned(path, device, precision, blocks_to_swap)` | driver-owned load (H3 tiers) | S |
| :373 `auto_uncompiled_precision(dit_path, precision)` | Klein: fp8 file trains as it is when uncompiled | S |
| :378-427 `compile_plan`, `_compile_fit` | generic Auto compile rule (payback steps + measured memory) | M |

Attributes read by the shared layer with `getattr`, not declared in the base class: `int8_fp32_scales`
(F:families/quant.py:44), `loads_quantized` (F:families/quant.py:100).

### 2.2 Differ where both have them

- **`prepare_training`**: F:107 `prepare_training(dit, group, net=None)`, called right after `FamilyLoRA(...)` and
  **before** any adapter file or the trainable adapter (F:families/train.py:951). T:training/driver.py:57
  `prepare_training(dit, net, *, precision, blocks_to_swap, total_steps, megapixels, batch_size)`, called **after**
  `add_trainable` (T:training/train.py:432). TagScribeR's is its own hook (used for Krea 2's compile). Upstream now
  does compile in the shared loop (`driver.compile_blocks`, F:families/train.py:1039-1041), so the two hooks have
  the same name and different contracts. Both sides changed. A port must rename or split one.
- **`quant_target_names`**: present on both (F:429, T:191). T header calls it a TagScribeR change; upstream now has
  the same method with the same default. Mirrors.
- **`block_map`, `lora_target_names`, `block_of`, `enable_block_swap`, `block_swap_mode`, `encode_*`,
  `initial_noise`, `pad_conditioning`, `decode`**: same.

### 2.3 Bug fixes to pick up

- `alias_flat` is the hook behind 6.8.2 "Krea 2 LoRAs from OneTrainer, AI-Toolkit and other diffusers-format
  trainers load again" (used by F:families/lora.py:288-291). TagScribeR has no alias path, so such a file as a
  Context LoRA matches nothing.

### 2.4 TagScribeR-only hooks (no upstream counterpart; decide per hook when porting)

`configure(**options)` (T:47; upstream: `set_options`), `optimizer_params` (T:51; upstream: description
`optimizer_families` + `_optimizer_family_groups`, F:families/train.py:100-119), `after_epoch` (T:63; upstream has
no epoch hook in the base driver - not checked whether the Krea 2 cuDNN attention switch still exists upstream),
`trainable_blocks` (T:66; upstream: `train_blocks` argument), `lora_key_name` (T:184; upstream: SDXL handled by
`LoRAFormat.lokr_kohya_stems` and the SDXL driver, not checked in detail), `supports_batching` (T:128; upstream
refuses batch size > 1 for every family, F:families/train.py:825-828), `warmup_note` (T:129, preserved item).

**Upstream change touching a preserved item**: the warm-up note is now unconditional in the shared loop for every
family (F:families/train.py:1336-1349), with no driver attribute. See 3.2.

---
## 3. `families/train.py` (1675 lines) -> `training/train.py` (702 lines)

### 3.1 Upstream has, TagScribeR lacks

| Upstream (F:families/train.py) | What | Size |
|---|---|---|
| :683-707 new `train_family` arguments: `sample_image`, `slider_pairs`, `slider_diff_weight`, `slider_prompts`, `slider_guidance`, `train_blocks`, `slider_bank`, `slider_bank_res`, `compile_blocks`, `recaption_instruction`, `recaption_instruction_detailed`, `finetune`, `ft_rotations`, `ft_save_every_rotations`, `ft_rotate_every`, `ft_max_parts`, `ft_start_window`, `ft_epochs_done`, `ft_fused_backward`, `reg_lr_multiplier`, `preview_checkpoint`, `preview_checkpoint_cache`, `preview_int8`, `family_options` | the argument surface of everything below | - |
| :742-769, :304-384, :1004-1038, :1363-1393, :1167-1174 | **Slider LoRAs** (image pairs and prompt pairs): `_SliderBank`, `_slider_strip`, `_pair_slider_caption`, `_prompt_slider_step`, practice-bank render, two-pole step, `ss_slider*` metadata, -1 / 0 / +1 preview strip (`SLIDER_PREVIEW_MULTIPLIERS`) | L |
| :571-680, :771-801, :984-993, :1339-1341, :1439-1459, :1488-1496 | **Full fine-tune** loop side: `_FineTune` (windows, rotation, fresh optimizer per window, park, full-checkpoint save), forced NF4 trunk, rotation-only saves and pauses | L (plus `families/ft.py`, section 15) |
| :834-851, :1407-1410 | regularisation blocks (`is_reg`) at `reg_lr_multiplier` for fine-tunes; warning for LoRA runs | S (needs dataset `is_reg`) |
| :859-867 `driver.plan_run(...)` | family's own Auto plan before `quant.plan` | S |
| :870-887, :1039-1041 | **shared torch.compile**: `desc.compiles`, `driver.compile_plan(...)` decided on the empty card, `driver.compile_blocks(...)` applied last | M (TagScribeR does this inside the Krea 2 driver today) |
| :888-893 `driver.auto_uncompiled_precision` | Klein: Auto picks the fp8 file as it is when the run is not compiled | S |
| :923-930 `sample_image` + `driver.encode_text_with_image` | preview conditioned on a picture through the text encoder's vision path (Krea 2 `preview_image=True`) | S here, M in the driver |
| :951 `driver.prepare_training(dit, group, net)` | see section 2.2 | - |
| :956, :963, :969 `driver.frozen_file_added(...)` | after each frozen file | S |
| :995-1003 `train_blocks` (+ `driver.expand_train_blocks`, :717-718), `ss_train_blocks` (:1175-1176) | train a subset of blocks (Ultra sliders, Fast Identity Mode, Model Area) | S |
| :100-119 `_optimizer_family_groups`, :1052-1061 | Automagic v3 groups from `desc.optimizer_families`; `desc.automagic_sign_window` -> `polarity_history` | S (TagScribeR: `driver.optimizer_params`, section 2.4) |
| :1062-1067 | `desc.optimizer_weight_decay` appended to Adam-family args; `eps_floor_8bit=desc.optimizer_eps_floor_8bit` | S |
| :1085-1093 | EMA **Short run** mode: decay `min(0.995, max(0.5, 1 - 4/steps))`, `ramp=2` | S |
| :122-139 `_legacy_perm`, :142-194 `_load_state` (`own_state`, `untagged_own`, legacy `model.safetensors` + `adaptive_lr_state.json`), :1103-1114 | resume from another trainer's / an older trainer's state: fresh optimizer unless the state is this family's own; optimizer moments and EMA shadow remapped | M |
| :1200-1220 `desc.preview_park_optimizer` | optimizer tensors parked on CPU for a preview (H3) | S |
| :475-560 `_CheckpointPreviews`, :1240-1246, :1254-1264 | previews on a preview checkpoint (Klein Distilled) with RAM cache | M |
| :247-275 `_encode_override` uses `driver.park_for` / `unpark`; :431-433 same for the decode | driver-owned parking | S |
| :441-450 `driver.save_preview`, `driver.slider_preview` | the driver writes the preview file(s) | S |
| :1280-1302 `_step_freeze` + `driver.step_frozen_blocks` | per-step block freezing | S |
| :1355-1361 `driver.step_policy` (skip / LR multiplier), :1314-1325 multiplier applied to the LR for one step, :1403-1404 `_info["lr_mult"]` | per-step LR policy (H3) | S |
| :1326 `driver.after_optimizer_step()` | - | S |
| :1177 `md.update(driver.run_metadata())` | - | S |
| :1453, :1470 `progress.start_t += time.time() - _tp` | the bar's s/it leaves out preview time (6.7.1) | S |
| :1498-1501 | final save also copied to `<name>-<epochs:06d>.safetensors` (#176, 6.8.3) | S |
| :1508-1619 `setup_parser` | CLI flags. TagScribeR uses `--config train_config.json` (its own design, T:673-698) | not needed |

### 3.2 Differ where both have them

1. **Clip signal for Adaptive LR.** F:1310-1313 calls `adaptive.record_clip(...)` only `if adaptive.clip_signal`,
   and `clip_signal` comes from `desc.adaptive_lr_clip_signal` (F:1082), which only Klein sets
   (F:families/klein.py:146). T:training/train.py:580-583 calls `adaptive.note_clip(...)` for **every** family
   whenever `max_grad_norm` is non-zero, and T:training/adaptive_lr.py:163 acts on it for every family. So on
   TagScribeR a Krea 2 run with Adaptive LR and Max grad norm 1.0 (the preset value,
   T:training/families/krea2/description.py:28) can REDUCE on ">50% of steps clipped"; on Fizgig it cannot.
   TagScribeR side changed (its header says "Klein's grad-clip-ratio signal fed to Adaptive LR"); upstream then
   added the same signal as a per-family flag. **This is on the owner's verified Krea 2 path - see Risks.**
2. **Warm-up note** (preserved item). F:1336-1349: every family, `epoch - start_epoch < 2` (so the first two epochs
   of a *resumed* run too), and the text adds ", compiles the blocks" when the run is compiled. T:574, :600-603:
   only when `driver.warmup_note` is true, `epoch < 2` (absolute, so never on a resume past epoch 2), fixed text.
   Upstream changed (the note moved from the Krea 2 trainer into the shared loop in 6.8.3: "the warm-up note is
   back").
3. **Preview with a speed LoRA and CFG.** F:422-427: `_cfg = cfg if cfg and cfg > 1.0 else speed.cfg`, and the
   negative is passed when `_cfg > 1.0`. T:244-246: always `cfg=speed.cfg`, no negative. Upstream changed (6.8.1
   "CFG works with Turbo previews"). Same result at the default CFG 1.
4. **Preview failure handling.** Both switch previews off for the rest of the run (F:1193-1217, T:527-558).
   Upstream additionally tracks what the preview changed in a `done` set (F:400-470) so the `finally` undoes exactly
   that even when the failure is in `move_adapter` or `swap_in`; TagScribeR does the set-up **before** its `try`
   (T:227-236), so an exception in `net.move_adapter(SPEED, device)` or `ema.swap_in()` leaves the adapter disabled /
   the speed LoRA on. Upstream changed (6.8.3). Upstream also `gc.collect()`s after a failure (F:1216).
5. **Checkpoint thumbnail refresh.** F:1473-1476: only `if cadence and not (metadata_thumbnail or "").strip()`.
   T:560-561: whenever a checkpoint and a preview exist, even when the user set an explicit thumbnail or "off".
   TagScribeR differs from upstream here (TagScribeR-side simplification; looks like a bug: an explicit thumbnail
   is overwritten).
6. **Thumbnail at save time.** F:1148-1149 `metadata_thumbnail or latest_sample_image(output_dir)`. T:495-496 adds
   `sample_for_epoch(...)` first. TagScribeR-only addition, harmless.
7. **`build_metadata` call.** F:1150 `build_metadata(None, arch, time.time(), ...)` (first arg `state_dict`), no
   `reso`. T:498-504 `build_metadata(arch, time.time(), ..., reso=...)`. TagScribeR changed the signature
   (T:training/metadata.py:22). Title / description use `is not None` upstream (F:1151, :1154) and truthiness in
   TagScribeR (T:499, :502).
8. **Scheduler fast-forward on resume with accumulation.** F:1142
   `start_epoch * updates_per_epoch if accum > 1 else global_step`. T:488 `global_step // gradient_accumulation`.
   These differ when `steps_per_epoch % accum != 0` (upstream counts `ceil` per epoch, which is what the loop
   actually steps). TagScribeR's count is short by up to `start_epoch` steps. TagScribeR wrote accumulation before
   upstream had it; upstream's is the correct one.
9. **Gradient accumulation naming.** F: `gradient_accumulation_steps`, off for sliders (F:1126). T:
   `gradient_accumulation` (T:293). Same arithmetic otherwise (loss / N, flush at epoch end).
10. **Batch size.** F:825-828 refuses batch size != 1 for every family. T:329-339 allows it for
    `driver.supports_batching` and stands the loss watch down. TagScribeR-only extension.
11. **Conditioning for a step.** F:1395 `driver.batch_cond(batch, device)`; T:611 inline `cond__` strip (the
    upstream default does the same).
12. **Data loop.** F:829-831 `DataLoader(group, shuffle=True)` + `_Collator` / `shared_epoch`. T:594-604
    `dataset.shuffle(seed + epoch + 1)` + `dataset.get_batch(i, data_rng)`. TagScribeR's own dataset layer
    (intended). Item order per epoch therefore differs from Fizgig for the same seed.
13. **Speed / log line.** F:1427-1431 uses `optimizer_lr(optimizer)` and prints `group_rates` in brackets before
    the peak; T:632-635 uses `param_groups[0]['lr']` and appends `group_rates` after. Cosmetic, but
    `training/progress.py` parses this line (T:69 `parse_epoch_summary`) - keep TagScribeR's format or update the
    parser together.
14. **Adaptive start LR.** F:1077 `math.sqrt(min * max)`; T:456 `AdaptiveLR.start_lr(...)`. Same value.
15. **EMA construction.** F:1096 `EMAWeights(net, decay)`; T:465 `EMAWeights(net.trainable_modules(), decay)`. Both
    reach the same parameters in the same order (`FamilyLoRA.parameters()` iterates `trainable_modules()`).
16. **Pause.** F:1484 `sys.exit(0)` inside `train_family`; T:662 `return None`, `main` exits 0 (T:697-698). Same
    contract.
17. **`_load_state` RNG restore.** T:102-108 wraps it in `try/except` (TagScribeR hardening); F:188-193 does not.
18. **Device.** F:737 `torch.device("cuda")` always; T:321 CPU fallback for smoke tests (intended).
19. **Auto plan megapixels.** F:857-858 from `user_config["general"]["resolution"]`; T:348-349 from
    `dataset.config["resolution"]`. Same meaning.
20. **Speed LoRA log line** F:980-981 adds the strength; T:417-418 does not. Cosmetic.

### 3.3 Bug fixes to pick up (release-note wording)

- 6.8.3 "Extending a finished run with Resume no longer overwrites its last epoch" (#176) - F:1498-1501. **Missing.**
- 6.8.3 "a preview that fails no longer ends the run ... with the model put back exactly as it was, however early in
  the preview it failed" - the `done` set, F:400-470. **Partly present** (item 4 above).
- 6.8.3 "the warm-up note is back" - F:1336-1349. Present as a TagScribeR port with different conditions (item 2).
- 6.8.1 "CFG works with Turbo previews" - F:422-427. **Missing.**
- 6.7.1 "The training speed on the progress bar no longer counts preview time" - F:1453, :1470. **Missing.**
- 6.7.2 "Resuming a Qwen run no longer prints a harmless PyTorch warning about the learning-rate schedule" -
  F:1137-1143. Present (T:485-489, broader filter).
- 6.8.0 "Each epoch checkpoint's thumbnail shows its own epoch's preview" (#122) - present (T:560-561), but see
  item 5.
- 6.8.0 "Gradient Accumulation works on Krea 2 and Qwen Image 2.1" - present; scheduler fast-forward differs
  (item 8).
- 6.8.0 "A run paused in an older Fizgig resumes from its saved LoRA, epoch and adaptive LR, with a fresh
  optimizer" - F:142-194. Missing; only matters for states written by old Fizgig trainers.
- 6.8.1 "A fine-tune no longer picks up a Context LoRA" - F:712-714. Not applicable until fine-tune is ported.

### 3.4 Upstream removed, TagScribeR still has

Nothing in this file was removed upstream. TagScribeR-only pieces: `driver.configure` / `driver_options`,
`driver.optimizer_params`, `driver.after_epoch`, `driver.trainable_blocks`, batch size > 1, `--config`.

---

## 4. `families/lora.py` (595), `families/lorafile.py` (139), `families/extract.py` (107) -> `training/lora.py` (589)

### 4.1 Upstream has, TagScribeR lacks

| Upstream | What | Size |
|---|---|---|
| F:families/lora.py:65-79 `LoHa`, :112-117 `add_loha`, :303-313 read, :362-365 attach, :532 bake | frozen LoHa files load (T:training/lora.py:323-324 refuses them) | S |
| F:families/lora.py:89, :105, :241-253 `trainable_dtype` | bf16 trainable adapter for a family (`desc.trainable_dtype`, H3) | S |
| F:families/lora.py:254-263 `_trainable_scale`, `set_trainable_multiplier(m)` | signed strength of the trainable adapter (sliders) | S |
| F:families/lora.py:288-291 `driver.alias_flat` in `_module_for` | other trainers' names (6.8.2 fix) | S |
| F:families/lora.py:300 `driver.convert_lora_state_dict` | Klein diffusers layouts | S |
| F:families/lora.py:326, :330 `lora.down` / `lora.up` key spelling | diffusers' own training scripts (7.0.0) | S |
| F:families/lora.py:337-344 + F:families/lorafile.py:30-45 `fused_parts` | a LoRA extracted from a file whose tensors the model splits (Qwen `img_mlp.gate_up`) | S (needs `FTSpec.file_layout`) |
| F:families/lora.py:24 `"unet."` in `_PREFIXES` | diffusers SDXL files | S |
| F:families/lora.py:217 `LoRAFormat.lokr_kohya_stems` | SDXL LoKR on `lora_unet_` stems | S |
| F:families/lorafile.py (whole file) | read a LoRA's pairs / LoKR / LoHa modules without loading the model: `lora_pairs`, `lokr_modules`, `lokr_factors`, `loha_modules`, `loha_delta`, `get_up`, `block_of`, `family_keys` | M |
| F:families/extract.py (whole file) | `reduce_pair`, `extract_weight_only`: weight-only rank reduction to the family's key format | M |

TagScribeR has no equivalent of `lorafile.py` or `extract.py` (see section 15).

### 4.2 Differ where both have them

1. **Frozen adapter add in the model's dtype.** F:families/lora.py:127-133: when the adapter output dtype equals
   the base output dtype (a frozen bf16 adapter in a bf16 model) it does `torch.add(out, lx, alpha=float(s))` - one
   fused add, scale formed in fp32, rounded once. T:training/lora.py:161: always
   `out + (s * ad(x.to(...))).to(out.dtype)`. Upstream changed (comment: "so a preview at strength 0.75 matches"
   the old loaders). The trainable fp32 adapter path is the same on both sides. **Affects the training adapter,
   context LoRA and the Turbo LoRA in previews - see Risks.**
2. **Conv2d.** F:137-178 `LoRAConv`: frozen-only, accepts a 1x1 or full-kernel down weight, raises if asked to
   train. T:165-208 `LoRAConv2d` + `LoRAConvFactor`: trainable (SDXL LoCon), `fits()` requires the full kernel,
   only wraps `groups == 1` (T:264). TagScribeR added Conv2d first for SDXL; upstream then added a frozen-only one
   in 7.0.0 ("Community SDXL LoRAs load fully ... LoCon, and speed LoRAs such as Lightning and LCM"). Upstream's
   SDXL does not train convs ("Never trained: training targets the block map's Linears", F:141). **Question for the
   owner: keep TagScribeR's trainable LoCon, which upstream SDXL does not have?**
3. **Kohya stems.** F:217 hard-codes `lora_unet_`; T:245-246 uses `f.file_prefix` +
   `driver.lora_key_name(full)`. TagScribeR-only (`lora_key_name`). T:228-230 also reads those names.
4. **Bias deltas (`diff_b`).** Same idea on both sides (snapshot / restore). F:382-415 keeps `(module name, delta)`
   and resolves the bias each time, scales `delta.float() * load` when load != 1; T:354-376, :414-423 keeps the
   `Parameter` and uses `add_(d, alpha=load)`; T also applies only `if st["on"] and st["load"]`. Upstream resolves
   names through `_module_for` (so aliases work); TagScribeR strips prefixes by hand. TagScribeR's header says it
   ported this from `krea2/trainer.py` before the shared layer had it; upstream then added it to the shared layer.
   Numerically equal at load strength 1.
5. **`_module_for`.** F:272-292 collects flattened candidates and tries aliases last; T:303-315 returns on first
   hit. Same result without aliases.
6. **`self.linears`.** T:226-227 excludes `LoRAFactor` / `LoRAConvFactor`; F:193 does not (a second `FamilyLoRA`
   on an already-wrapped model would see adapter Linears). TagScribeR hardening.
7. **LoKR helpers.** F imports `factorization`, `_lokr_forward_update`, `lycoris_scale_from_keys` from
   `fizgig/networks/lora.py` (F:39, :57, :310); T inlines them (T:46-84). Not compared line by line against
   `networks/lora.py` in this audit.

### 4.3 Bug fixes to pick up

- 6.8.2 "Krea 2 LoRAs from OneTrainer, AI-Toolkit and other diffusers-format trainers load again ... as a Context
  LoRA" - `alias_flat` (F:families/lora.py:288-291, F:krea2/driver.py:305). **Missing.**
- 7.0.0 "LoHa LoRAs ..." and "SDXL LoRAs saved in diffusers naming ... now work everywhere" - `LoHa`, `lora.down`,
  `"unet."`. **Missing.**
- 6.8.0 Repair Studio "Saving with a donor looks exactly as previewed" - `bake` is the same on both sides apart
  from LoHa.

### 4.4 Upstream removed

Nothing.

---

## 5. `families/quant.py` (196) -> `training/quant.py` (267)

### 5.1 Upstream has, TagScribeR lacks

| Upstream (F:families/quant.py) | What | Size |
|---|---|---|
| :22 `"hqq"` in `PRECISIONS` | a precision a driver loads itself (H3) | S |
| :31, :51, :104 `store_device` | INT8 weights quantised on the GPU but **stored on CPU** when the base is block-swapped, so a swapped INT8 base never has to fit whole | S |
| :44-47 `driver.int8_fp32_scales` | a driver may compute INT8 scales in bf16 (Krea 2: `int8_fp32_scales = False`, F:krea2/driver.py:36-38) | S |
| :52 `scale...to(torch.float32)` | the stored scale buffer is always fp32 | S |
| :86-88 `driver.load_planned(...)` | driver-owned load | S |
| :100-102 `driver.loads_quantized` | the file already is the base precision: load and return, no quantise, no swap | S |
| :140-141 `desc.auto_precisions` | Auto's order | S (TagScribeR: `auto_order`) |

### 5.2 Differ where both have them

1. **INT8 scales for Krea 2.** F:44-48: with `int8_fp32_scales = False` the weight stays bf16
   (`w.contiguous()`), so `scale = w.abs().amax(...)/127` and `w / scale` are computed in **bf16**, then the scale
   is stored as fp32. T:training/quant.py:67-72: always `.float()` first, so scale and rounding are in fp32.
   Upstream changed (the flag arrived with Krea 2's move to the driver system in 6.8.0, "so a driver run starts
   from the same INT8 weights as a Krea 2 run"). TagScribeR's Krea 2 INT8 weights therefore differ slightly from
   Fizgig's. **Owner's verified path - see Risks.**
2. **fp8** (preserved item). T:27 `PRECISIONS = ("bf16", "fp8", "int8", "nf4")`, T:42-60 `fp8` branch,
   `F8.dequantize_model` for bf16, `F8.detach(m)` before INT8 / NF4, T:154-159 fp8 loads straight to the device.
   Upstream has none of this in `quant.py`; upstream Krea 2 dropped fp8 as a base precision in 6.8.0 ("Base
   precision: fp8 is gone. Auto picks INT8, then NF4") and its description offers `("int8", "nf4", "bf16")`
   (F:families/krea2.py:91-92). Upstream Klein keeps an fp8 *file* trained as it is under the "bf16" choice
   (F:families/description.py:313-314 `precision_labels`, F:families/driver.py:373 `auto_uncompiled_precision`).
   **Upstream change that touches a preserved item**: mirroring upstream literally would remove TagScribeR's fp8
   base precision for Krea 2. Checked: upstream's Krea 2 loader still keeps a pre-quantised ComfyUI fp8 file as
   fp8 with a dequantising forward whatever `fp8_scaled` says (F:krea2/utils.py:129-157, the
   `is_prequantized_fp8` branch; the driver calls it with `fp8_scaled=False`, F:krea2/driver.py:68), and
   `quantize` then reads it through `_dequantize_source_weight` (F:families/quant.py:46). So an fp8-scaled RAW
   file is still a valid INT8 / NF4 source upstream, and under upstream's "bf16" choice such a file in fact trains
   as fp8 - the same result as TagScribeR's explicit "fp8" choice, without the name.
3. **`available()` probe** (T:105-128) and the "asked for X but unavailable -> fp8" fallback (T:233-239) exist only
   in TagScribeR (ported from `utils/capabilities.py` per its docstring). Upstream `plan` has no probe.
4. **Auto swap.** F:159 swaps `("int8", "bf16")`; T:194 `auto_swap_order` or the same default. T:225-228 adds a
   "maximum swap, tight" result where upstream falls through to "nothing fits ... the smallest base" (F:165-166).
   TagScribeR-only.
5. **`_prequantized` / `dense_weight`** (T:65-66, :82-83, MiniMax H3 ConvRot) - TagScribeR-only; upstream H3 uses
   `load_planned` instead.
6. `_peak`, `swap_for`, `apply_vram_cap`, `free_vram_gb`, `move`: same (env prefix differs).

### 5.3 Bug fixes to pick up

- 6.7.2 "Krea 2 Auto plans higher resolutions correctly" is description data (`train_memory` points at 0.25 and
  1.0 MP, F:families/krea2.py:125) plus `_peak` interpolation, which TagScribeR already has (T:164-177). Whether
  TagScribeR's Krea 2 figures match is a description question (not checked here).
- 7.0.1 "Fine-tuning an FP32 checkpoint: memory and disk are now sized correctly" - in `families/ft.py`, n/a.

### 5.4 Upstream removed

fp8 as a Krea 2 base precision (see 5.2 item 2). TagScribeR keeps it by the owner's instruction.

---

## 6. `families/cache.py` (194) -> `training/cache.py` (282)

TagScribeR's cache is built on its own dataset layer (`training/dataset.py`), so the code shapes differ; the file
formats are the same (`latent_{h}x{w}`, `latent_control_{i}_{h}x{w}`, `cond__<key>`, `caption1`,
`reference_sizes`, `format_version 1.0.0`).

### 6.1 Upstream has, TagScribeR lacks

| Upstream (F:families/cache.py) | What | Size |
|---|---|---|
| :45-48 `_latent_key` | clip latents `latent_{T}x{h}x{w}`; T:44-46 unpacks exactly 3 dims | S |
| :51-56 `save_latents(..., extra=None)` | further tensors beside the latent (audio rows) | S |
| :128-130 `--slider`, :146-149, :175 | the control folder is a slider's other pole: cache its latents, encode captions plainly | S |
| :131-132 `--aux KEY=VALUE`, :156-158 `driver.cache_stage(...)` | a family caches in its own layout (H3) | S hook |
| :124-125 `--batch_size`, `--num_workers` | T encodes latents at batch 1 (T:198) and text in chunks of 8 (T:138) | S |

### 6.2 Differ

- **`latent_rev`** (preserved item): T:27 `LATENT_REV = "2"`, T:51 metadata, T:54-61, T:192. Upstream has no such
  mark (F:59-61 metadata has no `latent_rev`). No upstream change touches it; a re-port of `save_latents` must keep
  the extra metadata key and the skip-existing check.
- Text cache is written atomically in TagScribeR (T:78-80 tmp + `os.replace`); upstream writes in place (F:74-75).
- Caption-shuffle variants and the caption-dropout empty caption (T:103-145, :224-232) are TagScribeR-only
  (owner-approved extension per the skills file).
- Stale-cache cleanup: upstream delegates to `scripts/cache_text.post_process` / `cache_latents.encode_datasets`
  (F:100, :160, :174); T:148-178 has its own `_remove_stale`. Not compared against `scripts/` in this audit.
- `needs_reencode` when pairs are added / removed: F:163; T:191 (same rule, plus the resolution and `latent_rev`
  checks).

### 6.3 Bug fixes to pick up

- 6.7.0 "Edit and slider pairs ignore upper and lower case in file names" - that is in the dataset layer
  (section 14), not here.
- 6.7.0 "MiniMax H3: caching clips no longer stalls near the memory limit" - `driver.clip_bucket_cap`, H3 only.

### 6.4 Upstream removed

Nothing.

---

## 7. `families/registry.py` (84) -> `training/registry.py` (65)

- Upstream has, TagScribeR lacks: `by_gui_label` (F:26), `workbench_families` (F:48), **`family_of_lora(path)`**
  (F:53-84: which family a LoRA file belongs to, from its header; needs `lorafile.py`) - S/M; `hidden` filter in
  `training_families` (F:45).
- TagScribeR-only: `register`, `_extra_from_env`, test families hidden by a leading "_" (T:31-62).
- Family list: upstream `(KLEIN, MINIMAX, KREA2, QWEN_IMAGE_21, SDXL, ANIMA)` - **SDXL and Anima now exist
  upstream** (7.0.0), so the skills-file statement "The SDXL family and Anima have no Fizgig counterpart" is out of
  date. TagScribeR registers `*SDXL_VARIANTS` (several SDXL keys) where upstream has one `SDXL`.
- Nothing removed upstream.

---

## 8. `training/adaptive_lr.py` (202) -> `training/adaptive_lr.py` (234)

Decision logic, constants (`CLIP_RATIO_THRESHOLD 0.5`, weight growth, blend, x1.25 / x0.5, patience) are the same.

Differences (TagScribeR ported the class from `krea2/trainer.py` and added the clip signal itself; upstream has
since added it too, differently):

1. **Who gets the clip signal** - F:28 `__init__(min_lr, max_lr, clip_signal=False)`, F:132
   `if self.clip_signal and clip_ratio > ...`. T:training/adaptive_lr.py:163 `if clip_ratio is not None and ...`
   (any family that clips). See 3.2 item 1.
2. **Klein baseline rule** - F:115-116: with `clip_signal`, the weight-norm baseline is **not** set at epoch 1
   ("Klein's baseline starts at the first comparison"), and the first epoch's clip counts are **not** reset at
   ARMED (F:113-120 has no reset), so they carry into epoch 2's ratio. T:146-150 always sets the baseline at
   epoch 1 and resets the counters (`_reset_clip()`). So TagScribeR's Klein behaviour differs from upstream's
   Klein in the first comparison.
3. Method name: F:45 `record_clip`; T:80 `note_clip`. T adds `clip_ratio` property, `start_lr`, `last_action`,
   a return value from `epoch_boundary`, `FACTOR_UP/DOWN` constants, and a `ValueError` for bad min / max
   (T:45-46).
4. Log line: F:197 prints `clip=NN% ` only with `clip_signal`; T:229 always prints `clip=...` ("—" when no
   clipping). `training/progress.py` `parse_adaptive_line` (T:81) reads this line.
5. `state_dict` keys: the diff shows no change in `state_dict` / `load_state_dict` (six keys, F:54-68).

Bug fixes: none named in the release notes. Removed upstream: nothing.

## 9. `training/ema.py`, `training/optimizers.py`, `training/automagic3.py`

- `ema.py`: identical apart from the header (67 vs 70 lines). `ramp` argument exists on both. Only the *use* of
  Short-run mode is missing (section 3.1).
- `automagic3.py`: identical apart from the header and one comment.
- `optimizers.py`: identical apart from TagScribeR's `family_groups(items, lr)` helper
  (T:training/optimizers.py:122) that `family_param_groups` now delegates to. `create_optimizer(...,
  eps_floor_8bit=False)` exists on both; TagScribeR's loop never passes it (T:training/train.py:439), upstream
  passes `desc.optimizer_eps_floor_8bit` (F:families/train.py:1066-1067). Neither side has a LoRA+ ratio in this
  file (see the commit check, section 16).

---


## 10. `families/loss_watch.py` (347) -> `training/loss_watch.py` (349); `training/loss_logger.py` (976) -> `training/loss_logger.py` (982)

### 10.1 Upstream has, TagScribeR lacks

| Upstream | What | Size |
|---|---|---|
| F:training/loss_logger.py:170 `family: str = ""`, :245-248 `_known_hard`, :276-285, :309-336, :350-357, :705, :963; F:families/loss_watch.py:86-87 `family=self.driver.description.key` | **Per-family exclusions, and earlier exclusions train again** (7.0.0): `fizgig_excluded.json` entries hold `{"families": {key: {epoch, date, reason}}, "caption"}`; an image excluded by an earlier run of the same family is **not** skipped - it trains normally, is marked as past its two recaptions (`_incorrigible`), and is excluded at once if confirmed stuck again; the report row carries `earlier_exclusion` | M |
| F:families/loss_watch.py:56-61 `recaption_instruction`, `recaption_instruction_detailed` | edited recaption instructions | n/a (TagScribeR passes its own captioner dict with `prompt` / `prompt_detailed`, T:training/loss_watch.py:255-256) |

### 10.2 Differ where both have them

1. **Persistent exclusions.** T:training/loss_logger.py:321-322 puts every entry of `tagscriber_excluded.json` into
   `self._excluded` (skipped from step 1, T:329-333 "they will be skipped"), keyed by nothing but the image, with
   a flat entry `{epoch, date, reason, caption}` (T:350-353). Upstream: see the table. Upstream changed (7.0.0
   "Problem images excluded in an earlier run train again ... Exclusions are now kept per model"). TagScribeR is
   at the pre-7.0.0 behaviour. File formats differ, so an old flat entry has to be read as "counts for every
   family" (upstream does exactly that: F:324 `"families" not in entry`).
2. **"Every image excluded" guard.** T:training/loss_logger.py:379-384 ignores the file when it excludes every
   image. Upstream removed that guard (not needed once earlier exclusions train again; F:362-376 has only the
   ghost pruning). Listed under 10.4.
3. **Captioner.** F:families/loss_watch.py:244-283 loads Krea 2's Qwen3-VL-4B (`generate_caption`) for every
   family; T:training/loss_watch.py:226-285 uses TagScribeR's `inference` providers and `core.caption_io`.
   TagScribeR design, stated in its header.
4. **Trigger word.** F:265-267 always adds the trigger word; T:266 adds it only when it is not already in the
   caption. TagScribeR-side change.
5. **Re-encode.** F:210-217 encodes in chunks of 4 with `cache.save_cond`; T:205-224 `_reencode` goes through
   `cache.encode_captions` (so shuffle variants refresh) and handles edit pairs
   (`load_reference_text_encoder`). TagScribeR-side change; upstream does not re-encode with references here.
6. **Look scores file.** F:98 `fizgig_look_scores.json`; T:67-78 reads `tagscriber_look_scores.json` or
   `fizgig_look_scores.json`.
7. **Image lookup for a recaption.** F:175-179 by extension in `image_dir`; T:151-153 the item's own
   `image_path` (works with several dataset folders).
8. Env var and file names: `FIZGIG_PERIMAGE_LOSS_LOG` / `fizgig_excluded.json` vs `TAGSCRIBER_...` /
   `tagscriber_excluded.json`.

Everything else in `loss_logger.py` (verdict thresholds, plateau rules, per-image LR multipliers, resume replay,
report rows) is unchanged between the two: the whole diff is the lines cited above.

### 10.3 Bug fixes to pick up

- 7.0.0 "Problem images excluded in an earlier run train again. They contribute as normal until they get stuck,
  and if they do, they're excluded straight away without new recaptions. Exclusions are now kept per model." -
  **Missing.** Touches the owner's verified "per-image loss watch" path only for datasets that already have a
  `tagscriber_excluded.json` (see Risks).

### 10.4 Upstream removed, TagScribeR still has

- The "excludes EVERY image - ignoring it for this run" guard (T:training/loss_logger.py:379-384).
- Skipping previously excluded images from step 1 (T:321-322, :329-333).

---

## 11. `training/metadata.py`, `training/progress.py`, `training/train_utils.py`

- **metadata.py** (F 321 / T 149): `latest_sample_image`, `sample_for_epoch`, `refresh_checkpoint_thumbnail`,
  `thumbnail_data_uri` are the same code. `build_metadata`: F:96 takes `state_dict` first and maps Klein / Krea 2 /
  MiniMax constants before falling back to `_described_family` (F:88); T:22 resolves every family through
  `training.registry.by_arch_id`. `resolve_title`: F:290-299 swaps a placeholder output name
  (`PLACEHOLDER_OUTPUT_NAMES`) for the trigger phrase; T:138-140 `output_name or trigger_phrase`. Upstream-only
  helpers not in TagScribeR: `precalculate_safetensors_hashes`, `load_bytes_in_safetensors`, `get_title`,
  `load_metadata_from_safetensors`, `build_merged_from` (hashing / merge tools; TagScribeR's header says they were
  dropped on purpose). No bug fix to pick up. Not checked: whether upstream `build_metadata` writes any field
  TagScribeR's does not (F:130-196 was not read line by line).
- **progress.py** (F 178 / T 157): TagScribeR dropped the RefMod parsers (F:39, :88) and added
  `parse_epoch_summary`, `parse_adaptive_line`, `parse_cache_line` (T:69-96). `parse_preview_phase` matches
  TagScribeR's own log wording (`[sample] ... preview failed`, T:104-106) where upstream matches `[preview]`.
  **Port note:** upstream's preview-failure line is now
  `[previews] the epoch N preview failed (...)` (F:families/train.py:1214); if the loop is re-ported, keep
  TagScribeR's wording or update T:104 with it. Same for the epoch line (section 3.2 item 13) and the adaptive
  line (section 8 item 4).
- **train_utils.py** (F 253 / T 125): `LossRecorder`, `list_state_dirs`, `prune_state_dirs`,
  `validate_output_name`, `get_epoch_ckpt_name`, `get_last_ckpt_name` are the same logic with shorter comments
  and "LoRA name" instead of "output name" in the messages. TagScribeR adds `latest_state_dir` (T:77). Upstream's
  accelerate-era helpers (`save_state_on_epoch_end`, `get_sanitized_config_or_none`, `get_lin_function`, ...,
  F:148-253) are not in TagScribeR and are not used by `families/train.py`.

---

## 12. `families/compile.py` (220) -> `training/families/krea2/compile.py` (400), `training/modules/compile_util.py`

- `modules/compile_util.py`: identical apart from the header and an unused import.
- TagScribeR's `krea2/compile.py` was ported from `krea2/trainer.py` + `utils/capabilities.py` **before** upstream
  made compile a shared-layer feature (6.8.3: "Krea 2 and Qwen Image 2.1 now share the same compile code, and
  Krea 2 behaves exactly as before"). It holds two things that upstream now keeps in different places:
  the machine / plan rules (`triton_matches_torch`, `has_host_c_compiler`, `compile_boundary`, `should_compile`,
  T:68-203 - upstream: still `utils/capabilities.py`) and the compile itself (`CheckpointedBlock`,
  `find_host_compiler`, `compile_blocks`, T:206-375 - upstream: `families/compile.py`).

### Upstream has, TagScribeR lacks

| Upstream | What | Size |
|---|---|---|
| F:families/compile.py:27-30 `CheckpointedBlock.forward(self, *args)` | any block signature; T:219 is `forward(self, x, vec, freqs, attn_params=None)` (Krea 2 only) | S |
| F:families/compile.py:104-169 `ready_to_compile(blocks_to_swap, fp8_scaled)` | the guard + settings as its own function, for drivers that compile their own way (SDXL) | S |
| F:families/compile.py:162-168 | two `warnings.filterwarnings` for expected inductor notices (TF32, complex operators) | S |
| F:families/compile.py:172-173 `compile_blocks(dit, blocks, ..., fullgraph=True)` | the block list comes from `driver.compile_targets`; `fullgraph` from `desc.compile_fullgraph` | S |
| F:families/driver.py:378-427 `compile_plan` / `_compile_fit` + description `compile_payback_steps`, `compile_memory` | the generic Auto rule for families without their own (Qwen: ~300 steps INT8, ~800 bf16) | M |
| F:utils/capabilities.py:693 `compile_blocker(blocks_to_swap, caps)` | "why this machine cannot compile" as one function (used by F:families/driver.py:393-396) | S |
| Compile for Qwen Image 2.1 and Klein (`compiles=True` in their descriptions) | TagScribeR compiles Krea 2 only | M per family |

### Differ

- Constants in TagScribeR's copy (`_TRITON_FOR_TORCH`, `_HEADROOM_GB 1.5`, `_BATCH_GB_PER_IMAGE 2.4`,
  `_COMPILE_WARMUP_S 90`, `_COMPILE_SAVING_S {"int8": 0.300, "nf4": 0.153}`, `_COMPILE_MARGIN 2.0`,
  `_COMPILE_GB_PER_MP 15.0`, `_COMPILE_OUTSIDE_ANCHOR_MP 1.05`; T:33-60) equal upstream's
  (F:utils/capabilities.py:35-40, :221, :238, :638-666). The bodies of `should_compile` / `compile_boundary` were
  **not** compared line by line in this audit (they live in `utils/`, outside the listed scope).
- Where compile is decided and applied: upstream in the shared loop (F:families/train.py:870-887 decide on the
  empty card, :1039-1041 apply last); TagScribeR inside the Krea 2 driver's `prepare_training`
  (T:training/families/krea2/driver.py:68, `compile.resolve`, T:training/families/krea2/compile.py:376). Upstream
  changed.
- TagScribeR names the base by precision including `"fp8"` (T header, lines 5-8); upstream's Krea 2
  `compile_plan` only distinguishes NF4 / INT8 / other (F:krea2/driver.py:52-63), because fp8 is no longer a
  Krea 2 choice.

Bug fixes: none named. Removed upstream: nothing.

---

## 13. `modules/` -> `training/modules/`

| Upstream file | TagScribeR | Finding |
|---|---|---|
| `modules/offloading.py` (493) | `training/modules/offloading.py` (498) | Identical apart from the header, the env prefix (T:112) and the **CUDA guard** (T:317, preserved item). No upstream change touches the guard. |
| `modules/int8_train.py` (185) | `training/modules/int8_train.py` (188) | Identical (import path only). |
| `modules/nf4.py` (152) | `training/modules/nf4.py` (158) | Identical plus TagScribeR's fp8 branch in `_dequantize_source_weight` (T:49-51, part of the preserved fp8 work). |
| `modules/sdpa.py` (173) | `training/modules/sdpa.py` (177) | Identical apart from env prefixes and TagScribeR's `sdpa_backend_ctx(device_type=...)` CPU guard (T:117-121). |
| `modules/compile_util.py` (77) | `training/modules/compile_util.py` (78) | Identical. |
| `modules/fp8.py` (733) | `training/modules/fp8.py` (188) | **Different files.** Upstream's is the musubi-style loader + `_FP8ScaledMMLinear` / `_try_fp8_scaled_mm_train` (the `_scaled_mm` training path) + `fp8_linear_forward_patch` / `apply_fp8_monkey_patch`, used by `klein/model_utils.py` and `krea2/fp8_optimization_utils.py`. TagScribeR's is its own small module (dequantise per matmul, `attach` / `detach` / `dense_weight` / `load_state_dict`), a preserved item. Not compared function by function: they share only `quantize_weight`'s idea. See the `d10022d` verdict (section 16). |
| `modules/attention.py` (256) | none in `training/modules/` | Unified attention (torch / flash / xformers / sage) with `AttentionParams`; used by `klein/model.py` only. TagScribeR's Klein has its own `training/families/klein/model.py` (not audited here). |
| `modules/int8.py` (105) | none | INT8 **inference** quantisation (`apply_int8_quantization`), used by `klein/driver.py` (`preview_int8`, the Distilled preview checkpoint). Missing; needed only with checkpoint previews. S. |
| `modules/int8_attention.py` (88) | none | comfy-kitchen INT8 SDPA for workbench renders (`attend`, `renders`); imported by every family model upstream (`krea2/attention.py`, `qwen_image21/model.py`, `anima/model.py`, `minimax/model.py`, `modules/attention.py`). Training never turns it on. A family-model re-port will meet the import. S (can be a stub that returns None until a workbench exists - **ask the owner**, this would be a reduced version). |
| `modules/schedulers.py` (201) | none | `FlowMatchDiscreteScheduler`, `RexLR`. No importer found under `src/fizgig` (searched for `modules.schedulers`). Not needed. |
| none | `training/modules/wide_attention.py` (35) | TagScribeR-only, preserved. Not checked: whether any upstream VAE file changed since the baseline in a way that touches the attention TagScribeR patched (VAE files are family scope). |

---

## 14. `dataset/config.py` (401) + `dataset/image_dataset.py` (1401) -> `training/dataset.py` (459)

TagScribeR's dataset layer is its own design (JSON config, one object for caching and training, image-only), a
selective port per its header. Not compared line by line; the checks below are by feature.

### Upstream has, TagScribeR lacks

| Upstream | What | Size |
|---|---|---|
| F:dataset/image_dataset.py:677-691 `.casefold()` pairing | **6.7.0 fix: "Edit and slider pairs ignore upper and lower case in file names"**. T:training/dataset.py:261-270 `_control_for` compares `os.path.splitext(n)[0] == stem` (case-sensitive). Also upstream accepts `<stem>_<n>` suffixed controls and sorts them; TagScribeR takes the exact stem only | S |
| F:dataset/config.py:24, :80, :141 `is_reg`; F:dataset/image_dataset.py:822, :843 | a dataset block marked as regularisation (fine-tunes) | S |
| F:dataset/config.py:69, :133 `clip_megapixels`; F:dataset/image_dataset.py:905-947 `_clip_target_bucket`, `_plan_clip_bucket` | 6.7.0 "MiniMax H3: Clip Target Megapixels" | M (video only) |
| F:dataset/image_dataset.py:111, :595-797 video / audio items (`VIDEO_EXTENSIONS`, `is_audio_path`, clip decode through `families/clips.py`), :1319-1343 `latent_cache_has_still`, `latent_cache_frames` | clips and voice items | L (H3 video; TagScribeR header: "image-only") |
| F:dataset/config.py:339-354 `load_user_config` reads `.json` or `.toml` | TagScribeR takes a dict | n/a |
| Several dataset blocks with their own settings (`DatasetBlueprint`, Multi Concept) | TagScribeR has a `datasets` list with `num_repeats` and `control_directory` per folder, one shared resolution (T:12-15) | not compared |

### Differ

- `decode_caption`: present on both (F:36, T:67). Not diffed line by line.
- Batch contents: both return `latents`, `latents_control_<i>`, `cond__<key>`, `item_keys`
  (F:dataset/image_dataset.py:586, T:440-458).
- Shuffle: upstream `BucketBatchManager.shuffle` via the DataLoader; TagScribeR `shuffle(seed)` (T:413). Item
  order differs for the same seed (section 3.2 item 12).

### Bug fixes to pick up

- 6.7.0 case-insensitive pair matching (above). **Missing**; TagScribeR's own preflight
  (`training/pipeline.py:144 edit_pairs`) was not checked for the same rule.

---

## 15. New upstream files with no TagScribeR file of the same name

| Upstream | What it is | TagScribeR equivalent | Size |
|---|---|---|---|
| `families/checks.py` (190) | Start's shared refusals as plain functions: `number`, `learning_rate`, `network`, `numbers`, `dataset_config`, `learning_rate_range`, `context_lora`, `tidy_name`, `run`, `training_folder` | **Partial**: `training/pipeline.py:189 preflight(...)` returns `Check` rows (its own design). The two were not compared rule by rule - a follow-up should diff `checks.py` against `preflight` so both refuse the same runs | M (to compare), S per rule |
| `families/launch.py` (866) | settings -> launch plan: `problems`, `pair_problems`, `option_tokens`, `cache_command`, `train_command`, `_preview_flags`, `dataset_toml`, `start_problems`, `clip_problems`, `plan`, plus `edit_on` / `slider_on` / `ft_on`, `area_blocks`, `timestep_flags` | **Yes, by design**: `training/pipeline.py` (`build_run`, `train_kwargs`, `dataset_config`, `edit_pairs`, `edit_instruction`; preserved). `launch.py` is now the upstream reference for "what is actually passed to the trainer" (the skills file still points at the GUI command builder). Not compared flag by flag | M (to compare) |
| `families/clips.py` (160) | clips against a `ClipSpec`: `probe`, `validate`, `problem`, `read_frames`, `hold_to_grid`, `read_audio` (ffmpeg) | none (image-only) | M, video only |
| `families/ft.py` (775) | full fine-tune: `FTSpec`, `DiskMaster`, `Rotator`, `plan_windows` / `plan_from_file`, `make_optimizer`, `FusedSteps`, `save_checkpoint`, `SharedBackend`, `schedule` | none (searched `training/` for `FTSpec`: no hit) | L |
| `families/act_cache.py` (180) | Repair Studio's activation cache (Turbo Preview) | none; workbench-only | not needed until a workbench exists |
| `families/lorafile.py`, `families/extract.py` | section 4 | none | M each |
| `families/workbench.py` (745), `video_workbench.py` (423), `block_profile.py` (548) | skipped as briefed: the generic Repair Studio / Explorer / Royale engine, its video variant, and the render-based Profiler | none | - |

---

## 16. The three unmerged TagScribeR commits against current Fizgig

Searched the whole current Fizgig tree, not only `families/`.

### `ffc75bb` "Klein: Network dropout (Fizgig NETWORK_DROPOUT)" - **obsolete as a mirror of upstream**

- The commit ports `networks/lora.py` `LoRAModule.forward` dropout (neuron / rank / module) into
  `training/lora.py`, adds `network_dropout`, `rank_dropout`, `module_dropout` to `train_family`, and
  `FamilyLoRA.param_groups` (Fizgig `prepare_optimizer_params`); source named in the commit:
  `src/fizgig/training/trainer.py`.
- Current Fizgig: `src/fizgig/training/trainer.py` **no longer exists** (the `training/` folder holds only
  `adaptive_lr, automagic3, ema, loss_logger, metadata, optimizers, progress, train_utils`). Klein trains through
  `families/train.py` (7.0.0 "Klein: trains on the new engine"). `NETWORK_DROPOUT` / `network_dropout` appears
  **nowhere** in the current tree (`*.py`, `*.json`, `*.md`; zero hits). `families/lora.py` has no dropout
  (F:families/lora.py:119-134), `families/train.py` has no such argument (F:683-707). The dropout code survives
  only in `networks/lora.py:35-37, :108-130`, whose `LoRANetwork` no trainer instantiates any more (its remaining
  importers use `factorization`, `_lokr_forward_update`, `lycoris_scale_from_keys`, format detection and
  `create_network_from_weights` in `minimax/common.py:1196`).
- Verdict: upstream removed the setting. Merging it would add something Fizgig does not have. **Owner question**
  (a departure either way): drop the commit, or keep network dropout as a TagScribeR-only extension.

### `2089c5d` "Klein: LoRA+ LR ratio (Fizgig LORA_LR_RATIO)" - **obsolete as a mirror of upstream**

- Current Fizgig keeps the key only for preset compatibility: `lora_trainer_gui.py:3895-3897` "LoRA LR Ratio -
  hidden, always 1 (LoRA+ default). Widget exists for preset/save compat.", default 1 at `:1442`;
  `families/checks.py:49` only checks it is an integer >= 1. `families/train.py` has no LoRA+ argument and builds
  one flat parameter list (F:1046-1051; groups only for Automagic v3, F:1052-1057). `prepare_optimizer_params`
  (`networks/lora.py:1118`) has no caller in the tree.
- So upstream today: the ratio is never applied, for any family. The commit's default (ratio 1 = one group) gives
  the same updates as upstream; any other value is a TagScribeR-only behaviour.
- Verdict: needs no merge to mirror upstream. If kept, it must be described as a TagScribeR extension, and
  `LORA_LR_RATIO` in imported Fizgig presets is always 1 anyway. **Owner question.**

### `d10022d` "Krea 2: port Fizgig's Fast FT (fp8 `torch._scaled_mm` on the frozen base)" - **obsolete on the trainer path; the library code it ports still exists**

- Still upstream: `modules/fp8.py:355-574` (`_train_scaled_mm_supported`, `_FP8ScaledMMLinear`,
  `_try_fp8_scaled_mm_train`), `krea2/fp8_optimization_utils.py:382-389, :453, :511-517` (`_fp8_fast`,
  `apply_fp8_monkey_patch(..., fast=False)`), `krea2/utils.py:96, :177-182` (`fp8_fast`, per-tensor
  `quantization_mode="tensor"`), and the probe `utils/capabilities.py:116, :174`. The commit mirrors that code.
- No longer reachable upstream: `KREA2_FAST_FT` / "Fast FT" has **zero** hits in the tree, the GUI included. The
  only training caller of `load_krea2_dit` is `krea2/driver.py:68`, with `fp8_scaled=False` and no `fp8_fast`.
  Fine-tuning now runs on an NF4 trunk for every family (`families/train.py:783-786` "the fine-tune trunk is
  always 4-bit"; `families/ft.py` header: "an NF4 trunk only"), and fp8 is no longer a Krea 2 base precision
  (6.8.0). So the feature the commit ports (Fast FT sent with base-model fine-tune) has no switch and no code path
  in v7.0.1.
- It also changes two preserved files (`training/quant.py`, `training/modules/fp8.py`) and adds a
  `resolve_fast_ft()` to `quant.py` that has no upstream counterpart any more.
- Verdict: obsolete as a mirror of the current trainer. Worth keeping only if the owner wants the fp8 `_scaled_mm`
  speed-up for the TagScribeR-only fp8 base precision on NVIDIA Ada+ cards - a TagScribeR extension, untested on
  hardware by the commit's own note. **Owner question.** The fine-tune it was a first step toward should be
  re-planned from `families/ft.py`, not from the old Krea 2 fine-tune.

---

## 17. Dependency-ordered port plan for the shared layer

Each step lists what it unblocks. Steps 1-4 are needed before family drivers can be re-ported against upstream's
driver interface; 5 onward are features.

1. **Description fields** (section 1.1): add the data fields with upstream names and defaults
   (`auto_precisions`, `compiles` + `compile_*`, `optimizer_families`, `automagic_sign_window`,
   `optimizer_weight_decay`, `optimizer_eps_floor_8bit`, `trainable_dtype`, `adaptive_lr_clip_signal`,
   `adaptive_lr`, `loss_watch`, `ema_short_run`, `resumes_untagged_states`, `preview_image`,
   `preview_park_optimizer`, `train_preview_checkpoint`, `preview_checkpoint_sampling`, `slider_*`, `finetune`,
   `ft_learning_rate`, `identity_blocks`, `train_areas`, `media`, `clip_spec`, `options` / `FamilyOption`,
   `ModelFile.inside`, `LoRAFormat.lokr_kohya_stems`). Pure data, no behaviour change while nothing reads them.
   Decide the three name clashes first: `auto_order` / `auto_swap_order` vs `auto_precisions`; `default_to` vs
   `inside`; `family_options` vs `options`.
2. **Driver interface** (section 2.1): add the optional hooks with upstream defaults (`set_options`,
   `batch_cond`, `step_policy`, `step_frozen_blocks`, `expand_train_blocks`, `after_optimizer_step`,
   `run_metadata`, `frozen_file_added`, `park_for` / `unpark`, `save_preview`, `alias_flat`,
   `convert_lora_state_dict`, `plan_run`, `load_planned`, `auto_uncompiled_precision`, `compile_targets` /
   `compile_blocks` / `compile_plan`, `legacy_state_order`, `cache_stage`). Resolve the `prepare_training`
   clash (section 2.2) and decide the fate of the TagScribeR-only hooks (section 2.4). Defaults reproduce today's
   behaviour.
3. **LoRA layer** (section 4): `alias_flat` + `convert_lora_state_dict` in the reader, `lora.down` and `unet.`
   spellings, LoHa, `trainable_dtype`, `set_trainable_multiplier`, the fused frozen add (risk item R2), keeping
   TagScribeR's trainable Conv2d and `lora_key_name` if the owner confirms. Then `lorafile.py` (needed by
   `family_of_lora` and Extract).
4. **Quant** (section 5): `store_device`, `int8_fp32_scales` (risk item R1), `load_planned`, `loads_quantized`,
   `hqq`, with TagScribeR's fp8 branch, `available()` probe and `_prequantized` handling kept.
5. **Loop fixes that need nothing else** (section 3.3): numbered copy of the final save (#176); the `done`-set
   preview restore; CFG with a speed LoRA; `progress.start_t` after previews; scheduler fast-forward with
   accumulation; thumbnail refresh only without an explicit thumbnail; per-family clip signal (risk item R3);
   warm-up note conditions.
6. **Loop features on steps 1-4**: `train_blocks`; description-driven optimizer defaults and Automagic groups;
   EMA Short run; `plan_run`; shared compile (move TagScribeR's Krea 2 compile behind `compile_targets` /
   `compile_plan`, add `families/compile.py`'s generic block wrapper); legacy / foreign state resume;
   `preview_park_optimizer`; `sample_image`.
7. **Loss watch** (section 10): per-family exclusions and "earlier exclusions train again", with a reader for the
   existing flat `tagscriber_excluded.json` entries.
8. **Dataset** (section 14): case-insensitive pair matching; `is_reg`.
9. **Cache** (section 6): `--slider`, `driver.cache_stage`, `_latent_key` / `extra` (with `latent_rev` kept).
10. **Sliders** (needs 3, 6, 9): the loop code, `noise_latents` / `predict` / `training_loss(diff_ref=...)` per
    driver, pipeline + Train tab + help.
11. **Checkpoint previews** (Klein Distilled; needs 2 and `modules/int8.py`).
12. **Full fine-tune** (needs 2, 4, 8): `families/ft.py`, `_FineTune`, driver `ft_spec`, pipeline + Train tab +
    help. Largest single item.
13. **Extract / Checkpoint to LoRA** (`families/extract.py`; needs `lorafile.py`).
14. **Video (clips, voice)** for MiniMax H3: `ClipSpec`, `families/clips.py`, dataset video branches,
    `clip_megapixels`. Only if the owner wants H3 video in TagScribeR.
15. Compare `families/checks.py` and `families/launch.py` with `training/pipeline.py` rule by rule and flag by
    flag (not done in this audit).

---

## 18. Risks to the owner's verified Krea 2 path

Path: fp8-scaled RAW checkpoint -> INT8 base, no swap, AdamW 8-bit, Adaptive LR, EMA 0.98, per-image loss watch,
base and Turbo-LoRA previews.

| # | Change | What would move | Severity |
|---|---|---|---|
| R1 | **INT8 scales in bf16 for Krea 2** (`int8_fp32_scales = False`, F:families/quant.py:44-48, F:krea2/driver.py:36-38) | The quantised weights of every block Linear change slightly (scale and rounding computed in bf16 instead of fp32, T:training/quant.py:67-72). Training results and previews shift by a small amount from step 0. Upstream says this is what makes a driver run "start from the same INT8 weights as a Krea 2 run". Not measured here. Note the source is the owner's fp8 file read through `_dequantize_source_weight` -> `dense_weight(..., bf16)` (T:training/modules/nf4.py:49-51), so the input to the quantiser is already bf16 on both sides | High (numerics of the base) |
| R2 | **Fused frozen-adapter add** (F:families/lora.py:127-133 vs T:training/lora.py:161) | Turbo-LoRA previews, and the training adapter / a context LoRA during training, round differently (one fused bf16 add with an fp32 strength instead of scale-then-add). Small numeric change in previews; the trainable adapter path is unchanged | Medium (previews), low (training without adapters) |
| R3 | **Clip signal only for Klein** (section 3.2 item 1, section 8) | Today a Krea 2 run with Adaptive LR and Max grad norm 1.0 can REDUCE (and roll back 70/30) when more than half an epoch's steps clip. Mirroring upstream removes that trigger for Krea 2; only weight-norm growth and the loss plateau remain. This changes the LR trajectory of exactly the runs the owner verified, if any of them ever logged a "grad clip NN% > 50%" REDUCE. Check an old `run.log` for that line before deciding | High if those lines exist, none otherwise |
| R4 | **Earlier exclusions train again** (section 10) | For a dataset that already has `tagscriber_excluded.json`, images skipped today would train again (until confirmed stuck). Changes which images a run sees | Medium, only with an existing exclusions file |
| R5 | **Shared compile replaces the driver's** (sections 3.1, 12) | If the Krea 2 compile decision moves from `prepare_training` (after the model is loaded) to the empty-card point upstream uses, the free-VRAM reading the Auto rule sees changes, so Auto may pick a different boundary or decline. Constants are equal on both sides. On the owner's AMD card Auto is off on ROCm (TagScribeR's own docstring, T:training/families/krea2/compile.py:11-13), so this matters for NVIDIA users | Low for the owner, medium for NVIDIA |
| R6 | **Warm-up note** becomes unconditional and also fires on the first two epochs after a resume | Console text only; the Train tab keeps log lines glued to the step bar (commit `b3658d2`), so check that the extra lines after a resume do not disturb it | Low |
| R7 | **Speed-LoRA preview CFG** (section 3.2 item 3) | No change at Turbo defaults (CFG 1). If the owner's preview CFG is above 1, Turbo previews would start using CFG and the negative prompt | Low |
| R8 | **Preview restore via the `done` set**, `gc.collect()` after a failure | Behaviour only differs after a failed preview | Low |
| R9 | **`prepare_training` contract change** (section 2.2) | TagScribeR's Krea 2 driver compiles and reads `precision`, `total_steps`, `megapixels`, `batch_size` there. A straight re-port of the hook signature breaks it unless compile moves to the shared loop in the same change | High if done in two steps, none if done together |
| R10 | **`after_epoch` / `optimizer_params` / `configure` have no upstream counterpart** | TagScribeR's Krea 2 uses all three (cuDNN attention switch, Automagic groups, Compile Blocks option). Dropping them during a re-port silently removes behaviour. Not checked: whether the cuDNN switch still exists in upstream's Krea 2 code | Medium |
| R11 | **fp8 base precision** (section 5.2 item 2) | Upstream removed fp8 for Krea 2; the owner's instruction keeps it. A literal mirror of `quant.py` / the Krea 2 description would drop it. Upstream's Krea 2 loader still keeps a pre-quantised fp8 file as fp8 with a dequantising forward (F:krea2/utils.py:129-157), so the owner's file remains a valid INT8 source upstream too | High if mirrored literally |
| R12 | **Item order per epoch** | Not a change to make: TagScribeR's own shuffle already differs from upstream's DataLoader shuffle. Keep it, or the verified runs' data order changes | n/a (leave as is) |
| R13 | **Auto precision order** (`auto_precisions=("int8", "nf4")` upstream vs TagScribeR `auto_order` / `auto_swap_order`) | Only for Auto. The owner picks INT8 explicitly | Low |
| R14 | **`train_memory` figures for Krea 2** changed upstream in 6.7.2 (F:families/krea2.py:125) | Only for Auto and auto swap. TagScribeR's figures were not compared (description scope) | Low for the owner |

Unchanged on this path (verified identical in this audit): `int8_train.py` (the INT8 forward and backward),
`ema.py`, `optimizers.py` (AdamW 8-bit creation), `automagic3.py`, the Adaptive LR decision rules apart from R3,
the loss-watch verdict logic apart from R4, `offloading.py`, `sdpa.py`, `compile_util.py`.

---

## 19. Questions for the owner (departures that need a decision, not settled here)

1. Krea 2 INT8 scales: switch to upstream's bf16 scales (R1), accepting slightly different base weights from the
   runs already verified?
2. Clip signal: restrict it to Klein as upstream does (R3)?
3. Trainable Conv2d (LoCon) for SDXL: upstream's SDXL trains Linears only. Keep TagScribeR's?
4. `ffc75bb`, `2089c5d`, `d10022d`: drop, or keep as TagScribeR-only extensions (section 16)?
5. fp8 as a Krea 2 base precision: keep (your standing instruction) although upstream removed it - confirm.
6. Batch size > 1, caption shuffle / dropout, `lora_key_name`, `after_epoch`, `optimizer_params`, `configure`:
   keep as TagScribeR extensions, or move to upstream's mechanisms (`FamilyOption`, `optimizer_families`)?
7. SDXL and Anima now exist upstream (7.0.0). TagScribeR's are original code with community presets. Re-port them
   from Fizgig?
8. Video (clips / voice) for MiniMax H3, the workbench tools, Extract and full fine-tune: which of these does
   TagScribeR want, and in what order?
9. `modules/int8_attention.py`: port it with the family models, or leave it until a workbench exists?

---

## 20. What was not checked

- Baseline diff (`746ddab`) - not available in the copy; direction of change is inferred.
- `utils/capabilities.py` function bodies (`should_compile`, `compile_boundary`, `estimate_krea2_peak`,
  `recommend_krea2_strategy`) against TagScribeR's copies; only the constants were compared.
- `networks/lora.py` helpers (`factorization`, `_lokr_forward_update`, `lycoris_scale_from_keys`) against
  TagScribeR's inlined copies.
- `scripts/cache_latents.py`, `scripts/cache_text.py` (upstream's batching and stale-cache cleanup) against
  TagScribeR's `_remove_stale`.
- `families/checks.py` and `families/launch.py` rule by rule against `training/pipeline.py`.
- `dataset/` line by line (feature-level only); `decode_caption`, `BucketSelector`, `resize_image_to_bucket`
  bodies.
- `metadata.build_metadata` fields F:130-196.
- Every family driver, model, VAE and description (other audits), including whether upstream VAE code changed
  around the attention that `wide_attention.py` replaces.
- `modules/fp8.py` function by function.
- Nothing was executed.
