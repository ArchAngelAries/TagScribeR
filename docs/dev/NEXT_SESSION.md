# Next session: where the work stands (handoff, 2026-10-19)

Read `CLAUDE.md` (standing rules and the orchestration rule), then `docs/dev/PORT_PLAN.md` (the one tracked roadmap,
with every decision already made). This file is the short version; PORT_PLAN wins where they differ.

## State

- `main` now holds everything: stages 1-5 sliders of the Fizgig port, the 2026-10-19 bug-fix batches and the
  Fizgig v8.7.1 catch-up wave 1. The work branch was `port/stage1-bt67en`; continue on a new work branch
  (`port/...` or `wip/...`), push it freely, and merge to `main` only with the owner's go.
- Fizgig reference: clone https://github.com/shootthesound/Fizgig into the git-ignored `.claude/fizgig-src` and check
  out `5d31b2f` (v8.7.1). The audits' line numbers (FZ:) refer to it.
- Everything since stage 1 is unit-tested only (tiny random models, CPU). Only Krea 2 has run on real hardware (the
  owner's AMD Radeon on ROCm, before stage 1's numeric changes).
- Reports in `docs/dev/`: `audit/05`-`07` (Fizgig v8.7.1 delta), `audit/08`-`10` (bug and security hunts, all
  confirmed findings fixed), `IMPROVEMENT_IDEAS.md` (measured efficiency ideas).

## For the owner to check on real hardware

1. An ordinary Krea 2 run: confirms stage 1's numerics (bf16 loss order, bf16 INT8 scales, one caption per forward)
   and the new preview recipe (Turbo 0.7, 10 steps, CFG 1). If good, lift Krea 2's "experimental" label.
2. On the AMD machine: the training console's GPU line (the discrete Radeon is chosen, not integrated graphics);
   `core/vram_monitor.py` reads only `CUDA_VISIBLE_DEVICES` when mapping devices - check its AMD reading while a run
   pins `HIP_VISIBLE_DEVICES`.
3. Resume from a state saved with AdamW 8-bit (state now loads with `weights_only=True`; bitsandbytes was not
   installed in the cloud test environment).
4. The camera presets' EXIF strings marked `# verify` in `core/camera_presets.py` against real camera files.

## Next work, in order (details and decisions in PORT_PLAN)

1. **Fizgig v8 shared-layer wave 2** (`audit/05`): resume-state save retry; a test compile before
   `compile_blocks` and eager previews under compile; per-step loss log and Sample now / Save now; slider override
   ignored and live prompt changes at epoch boundaries; multi-reference Edit LoRAs (Klein 4, Qwen 10, with the
   dataset / cache change: `training/cache.py` refuses more than one partner today).
2. **Installer:** pin a SHA-256 for the Windows ROCm bitsandbytes wheel (`tools/install.py`, audit 10 #2).
3. **Stage 5 rest:** the full fine-tune (`families/ft.py`; upstream documents it NVIDIA-only: port with the streamer
   behind the CUDA guard, docs say "not run on AMD"), DOP, Multi Concept, then LoRA Isolation.
4. **New families:** MOSS-Image, then Flux 3 Image Klein (gated weights: licence link, no stored token; route its
   forced cuDNN attention through `training/modules/sdpa.py` for ROCm), then Z-Image Turbo. ACE-Step is skipped.
5. **Improvements** (`IMPROVEMENT_IDEAS.md`), first three: training diagnostics that measure the owner's speed leads
   on the real card; the health scan and thumbnails (10-20x / up to 7x measured); the trimmed Krea 2 text cache.
6. **Documentation debts** (PORT_PLAN "Documentation ledger"): README What's new / roadmap / "what has actually
   been run", `docs/TRAINING_PLAN.md`, `docs/ARCHITECTURE.md`.

## Known, harmless, needs the owner's call

`tools/check_release.py` lists history findings for example values in pushed commits (a private-range IP example in `dd7974e`,
a made-up drive path in `21f8931`); both were replaced in later commits. Removing them from history means rewriting
it (a force push), which only the owner can approve. They are made-up examples, no real data.

## Working style (the owner's)

The lead plans, integrates and checks; Sonnet subagents do narrow execution (one set of files each); Fable subagents
audit, design harder ports, propose improvements, review every batch before it is committed, and hunt bugs and
exploits. Run independent agents in parallel, never two on one file. Decide by the standing rules and record the
call in PORT_PLAN instead of asking; ask only for decisions that are genuinely new, with a recommendation.
