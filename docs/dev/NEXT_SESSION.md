# Next session: where the work stands (handoff, 2026-10-19, end of the local session)

Read `CLAUDE.md` (standing rules and the orchestration rule), then `docs/dev/PORT_PLAN.md` (the one tracked roadmap,
with every decision already made, including the owner's product calls and the lead's technical calls on
`VISION.md` / `UI_UX_RESEARCH.md`). This file is the short version; PORT_PLAN wins where they differ.

## State

- Work branch **`port/v8-wave2`** (pushed). `main` is untouched at `aaf604b`; merge only with the owner's go.
- Fizgig reference: `.claude/fizgig-src`. **Port baseline: v8.7.3, commit `9162fb20`** (read files with
  `git -C .claude/fizgig-src show 9162fb20:<path>`; the working tree may still be at `5d31b2f`). Audit 11 digests
  v8.7.1 -> v8.7.3. Check Fizgig for newer commits before moving the baseline again.
- Committed and pushed this session (each reviewed by Fable before the commit, full suite green, unit-tested only):
  `bc2d02d` installer bitsandbytes SHA-256; `2886e61` wave 2 trainer side; `8e717ac` VISION + UI/UX reports;
  `46910fc` multi-reference Edit LoRAs; `5e32434` Train tab wave 2 UI (loss graph, Sample now / Save now,
  Checkpoints tab, slider live prompts, per-preview prompts, Distilled follow, edit extras, description fields).
- The owner is "the idea guy": decide technical / package questions by the standing rules and record them in
  PORT_PLAN; ask only product questions, with a recommendation.

## Uncommitted work in the working tree (three agents stopped at a clean point)

Nothing below is reviewed yet; the suite is NOT green with it (two SDXL / Anima preset tests fail until
`FAMILY_FT` is declared in `training/params.py`). A backup of all of it is in the git-ignored `.claude/wip/`
(`2026-10-19-tracked.patch` = `git apply` it; `2026-10-19-untracked.tgz` = the new files). Finish each piece, then a
Fable review, then commit it on its own (test the exact commit set in a temporary worktree, as this session did).

1. **Full fine-tune (Fizgig `families/ft.py`, stage 5).** Done, importable: `training/modules/safetensors_io.py`,
   `training/modules/rotation.py` (prefetch only on CUDA / HIP, behind the guard), `training/ft.py`, the `_FineTune`
   branches of `training/train.py`, `training/driver.py` ft hooks, `training/pipeline.py` ft helpers / preflight /
   `is_reg` / train_kwargs (a fine-tune drops Adaptive LR, context LoRA, resume, train_blocks, preview checkpoint),
   ft_spec in the Krea 2 / Klein / Qwen / SDXL / Anima drivers, `finetune=True` and Fizgig's Fine-tune presets on
   those families (Fizgig's metadata keys `fizgig_finetune`, `fizgig_next_start_window`, ...). H3 declares ft_spec
   but stays `finetune=False` (needs Fizgig's `minimax/ft_backend.py`, ~950 lines: later).
   **Left:** (a) params in `training/params.py`, `family_only="finetune"`, from Fizgig form.py:65-66, :468-485:
   `FAMILY_FT` (bool False, "Fine-tune the whole model"), `FAMILY_FT_ROTATIONS` ("10", "Rotations"),
   `FAMILY_FT_SAVE_EVERY` ("1", "Checkpoint + preview every N rotations"), `FAMILY_FT_ROTATE_EVERY` ("1", "Epochs per
   part before the next"), `FAMILY_FT_MAX_PARTS` (choice from `pipeline.FT_WINDOW_SIZES`, default first),
   `FAMILY_FT_FUSED` (bool True, "Free each gradient as soon as it's used (less memory - recommended)"),
   `FAMILY_FT_REG_DIR` (dir, "Regularisation images (optional)", into PATH_KEYS), `FAMILY_FT_REG_MULT` ("0.2",
   "Regularisation LR x"); (b) Train tab: choosing Fine-tune sets the LR to `desc.ft_learning_rate`, hides rank /
   alpha / Network Type / Model Area / timesteps / Resume (Fizgig 8.0.7), Resume continues via
   `pipeline.ft_rotations_left` (`FAMILY_FT_CONTINUE`); (c) `training/presets.py`: allow `FAMILY_FT` presets when
   `desc.finetune` and the key is declared, keep `KREA2_*` / `MINIMAX_FINETUNE` refused; (d) a `TinyFTDriver` in
   `tests/tiny_family.py` and `tests/test_finetune.py` (an ordinary LoRA run unchanged; rotation / save / continuation
   on CPU); (e) `THIRD_PARTY_NOTICES.md` entries for rotation.py (krea2/rotation.py), safetensors_io.py
   (krea2/safetensors_utils.py, musubi-tuner origin), ft.py (families/ft.py); (f) help topic and docs saying
   "not run on AMD (or by the owner on NVIDIA)". Lead's calls: the dense-trunk CPU path is test-only; add the
   `model.diffusion_model.` prefix for Anima masters only if Fizgig does (it does not: mirror, note it).
2. **MOSS-Image family (Fizgig 8.5.0-8.7.3).** Done, 33 tests passing (1 xfail): `training/modules/flux2_vae.py`
   (BFL autoencoder with `wide_attention`), `training/families/moss_image/` (model, sampling, embedder, driver,
   description with Fizgig's presets / memory / Turbo / SFT tick / standard_base, torch-free `base_check.py`),
   registry entry, THIRD_PARTY_NOTICES section, `tests/test_training_moss_image.py`. `finetune=False` until the
   fine-tune lands (its FTSpec: blocks "layers", overhead 6.0 / 4.5 GB, calib 1 MP, 2.0 GB per MP, q / k / v ->
   qkv thirds). **Left:** params `MOSS_LORA_SFT` (BOOL False, "Train attention only", Training Parameters) and
   `MOSS_LOWNOISE_PCT` (30, "Clean-end share (% of steps)", Timesteps; Fizgig allows empty = the model's own
   schedule, so TEXT or a FLOAT with 0 = own schedule) with `family_only="option"` and `DRIVER_OPTIONS` entries
   `lora_sft` / `lownoise_pct` (tooltips in the agent's words: attention only is recommended on SFT, may cost
   quality on Base; the clean-end share is experimental, Fizgig's default 30); `training/cache.py` text chunk ->
   `getattr(driver, "text_cache_batch_size", 8)`; the Train tab unticks the SFT preview once when
   `base_check.is_standard_base` says the Base is not the released one (Fizgig form.py:890-901); help (train on
   Base, use on SFT; 70-epoch patience; clean-end share; long captions; AMD: Qwen3.5's Gated DeltaNet runs the slow,
   correct PyTorch path); a Fable review against Fizgig. Questions it raised: offer the fp8 base as TagScribeR does
   for others (recommend yes, later); base_check on diffusers shards is an unverified extension.
3. **Theme system (UI/UX research stage 1).** `tabs/theme.py` (tokens, two token themes `tagscriber_dark` /
   `tagscriber_light`, legacy overlays reproducing today's colours for the qt_material themes, `build_qss`, button
   variants by dynamic property) is written and checked offscreen, but **nothing is wired**: the app looks as before.
   **Next:** `tabs/settings.py` apply / theme choices; `main.py` sidebar object names, qtawesome icons instead of
   emoji, tooltips "Gallery (Ctrl+1)"; inline styles routed through tokens in common, palette, help, memory_bar,
   train_dialogs chart, metadata `LEVEL_COLORS`, caption, datasets, review, editor, train (Start / Run = primary,
   Stop / Abort / Delete = danger); sentence-case labels mirrored in help; tests/test_theme.py; a visual check of every
   tab per theme. Default stays qt_material `dark_teal`; the owner switches after seeing the new theme. Later:
   `tabs/workspace/*`, `core/widgets.py`. Lead's call on contrast: white on the dark theme's accent is 2.7:1; use
   dark text on accent in the dark token theme.

## For the owner to check on real hardware

1. An ordinary Krea 2 run on `port/v8-wave2`: confirms stage 1's numerics and the new preview recipe (Turbo 0.7,
   10 steps, CFG 1); watch the new Loss tab and try Save now / Sample now once. If good, lift Krea 2's
   "experimental" label.
2. The training console's GPU line on the AMD machine (the discrete Radeon is chosen); `core/vram_monitor.py` with
   `HIP_VISIBLE_DEVICES` pinned.
3. Resume from a state saved with AdamW 8-bit.
4. A fresh install or `update.bat` on Windows ROCm: the bitsandbytes wheel's SHA-256 check passes.
5. The camera presets' EXIF strings marked `# verify` in `core/camera_presets.py`.

## Next work, in order

1. Finish and commit the three uncommitted pieces above (fine-tune, MOSS-Image, theme), each reviewed.
2. Stage 5 rest: DOP, then Multi Concept, then LoRA Isolation; the fine-tune's DOP targets; Checkpoint-to-LoRA
   extraction (`families/extract.py`); the H3 fine-tune backend.
3. New families: Flux 3 Image Klein (gated weights: licence link, no stored token; forced cuDNN attention through
   `training/modules/sdpa.py` for ROCm; reuse `training/modules/flux2_vae.py`), then Z-Image Turbo. Move Klein onto
   the shared `flux2_vae.py`. ACE-Step is skipped.
4. VISION do-first list (remote first, testable on one PC: the headless runner CLI and portable run bundle), the
   theme's later stages, the `IMPROVEMENT_IDEAS.md` first three.
5. Documentation debts (PORT_PLAN "Documentation ledger").

## Known, harmless, needs the owner's call

`tools/check_release.py` lists history findings for example values in pushed commits (a private-range IP example in
`dd7974e`, a made-up drive path in `21f8931`); both were replaced later. Removing them means rewriting history (a
force push), which only the owner can approve.

## Working style (the owner's)

The lead plans, integrates and checks; Sonnet subagents do narrow execution (one set of files each); Fable subagents
audit, design harder ports, propose improvements, review every batch before it is committed, and hunt bugs and
exploits. Run independent agents in parallel, never two on one file. No agent instantiates the app against the real
`user_data/` (tests use the isolated fixtures in `tests/conftest.py`).
