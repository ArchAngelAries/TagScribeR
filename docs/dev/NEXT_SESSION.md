# Next session: where the work stands (handoff, 2026-11-09)

Read `CLAUDE.md` (standing rules and the orchestration rule), then `docs/dev/PORT_PLAN.md` (the one tracked roadmap:
its "Now (2026-11-09)" section has the decisions, the order of work and the questions waiting for the owner). This
file is the short version; PORT_PLAN wins where they differ. The session ended early on usage: everything below is
a clean stopping point.

## State

- Work branch **`port/v8-wave2`**. `main` is untouched at `aaf604b`; merge only with the owner's go.
- Fizgig reference: `.claude/fizgig-src`, fetched on 2026-11-09. Read files with
  `git -C .claude/fizgig-src show <commit>:<path>`; the working tree is still at `5d31b2f`.
  - **Baseline of what is committed: v8.7.3, `9162fb20`.**
  - **Target for new work: just past v9.4.0, `c11134a2`** (694 commits later). Two audits digest the delta:
    `audit/12_fizgig_v9_4_0_training_core.md` and `audit/13_fizgig_v9_4_0_new_families_and_tools.md`.
- Committed this session (unit-tested only; nothing run on real weights):
  - Image grid: Ctrl+A toggles select all / none, Ctrl+Shift+A deselects (owner-reported).
  - The full fine-tune (Fizgig `families/ft.py`) for Krea 2, Qwen Image 2.1, Klein, MOSS-Image, the SDXL family and
    Anima, with its Train tab side, pause / resume from a checkpoint (which now also survives an app restart) and a
    tiny CPU fine-tune family for tests. Reviewed by Fable against `9162fb20`; its two blockers (the hidden timestep
    range reached the trainer; a run could write over its own source file) and most should-fix items are fixed and
    pinned in `tests/test_review_fixes_ft_moss.py`.
  - MOSS-Image at Fizgig 8.7.3, reviewed by Fable (no blocker; three should-fix items fixed).
  - Fizgig 9 fixes F1, F2's guard, F8, F13 and the onnxruntime telemetry switch (`tests/test_fizgig_v9_fixes.py`).

## Uncommitted work in the working tree: the theme wiring (stage 1 written, not reviewed)

`tabs/theme.py`, `tests/test_theme.py` and the wiring in `main.py`, `tabs/settings.py`, `tabs/common.py`,
`tabs/palette.py`, `tabs/help.py`, `tabs/memory_bar.py`, `tabs/train_dialogs.py`, `tabs/metadata.py`,
`tabs/caption.py`, `tabs/datasets.py`, `tabs/review.py`, `tabs/editor.py` are NOT committed and NOT reviewed. A
backup is in the git-ignored `.claude/wip/` (`2026-11-09-theme.patch` for tracked files,
`2026-11-09-theme-untracked.tgz` for the two new ones) with the agent's report (`2026-11-09-theme-report.md`).
By that report: the theme is selectable in Settings and applied at once and at startup; the default stays qt_material
`dark_teal`, pixel-identical to before on 19 offscreen shots; the sidebar has icons and hotkey tooltips; inline
styles in ten tab files go through tokens. Left: a Fable review; the Help Center text for the setting (rule 7); the
button variants in `tabs/train.py` (Start = primary, Stop and preset Delete = danger); `tabs/workspace/*`,
`core/widgets.py`, the export dialog and `tabs/icons.py` still carry dark inline colours on the light theme;
switching from a token theme back to qt_material at runtime leaves small layout / icon differences until a restart;
nothing has been looked at on a real screen.

## Review findings recorded but not fixed (fine-tune / MOSS-Image)

- Regularisation images are matched by file stem: a subject picture sharing a stem with a regularisation picture
  trains at the wrong rate (as upstream).
- No disk-space check before a multi-GB save; the scratch master (`.tagscriber_ft_master`) survives Stop or a crash;
  a `.tmp` is left when Stop lands mid-save (as upstream).
- Rotations done are miscounted on Resume if the window plan changed between legs (as upstream).
- `ft_card_plan` is never called: the Train tab has no "on this card" plan line before Start.
- MOSS-Image: the text cache encodes 4096 captions per call (as upstream): Stop or an out-of-memory error loses the
  whole chunk and no progress prints until it ends.
- MOSS-Image on ROCm is unchecked where a wrong result would silently corrupt every caption embedding: the bf16
  depthwise conv1d with 8192 groups, batched `solve_triangular`, SDPA at head dim 256. The owner's first run should
  compare one caption's embedding on CPU and GPU.
- Test gaps named by the reviews: a fused-parts / fp16-source checkpoint round trip, a disk-master round trip, the
  regularisation multiplier, a tab test for the once-per-Base untick, the vendored gated-delta rule against
  transformers' fallback.

## For the owner to check on real hardware

1. An ordinary Krea 2 run on this branch (still pending from 2026-10-19): stage 1's numerics, the preview recipe, the
   Loss tab, Save now / Sample now. Auto precision is unchanged for it unless NF4 fits with only 0.5 to 1.5 GB to
   spare (then NF4 without swap instead of INT8 with swapped blocks).
2. A fine-tune is NOT ready to be trusted: it has run only on a tiny CPU model. It needs a bf16 model file (the
   fp8-scaled Krea 2 file is refused before the run starts).
3. The older list: the GPU line in the training console, resume from an AdamW 8-bit state, a fresh install's
   bitsandbytes check, the camera presets marked `# verify`.

## Next work, in order

See PORT_PLAN "Next, in order". In short: the remaining audit-12 fixes (F4, F5, F11, F12; F6 / F7 with one cache
revision; F3), then Muon and the preset move (owner's answer needed), F9 / F10, the fine-tune and MOSS-Image at the
target, mask awareness and start warnings, then the new families (Z-Image Turbo and Base first, if the owner agrees).
Finish and commit the theme wiring in parallel (different files).

## Questions waiting for the owner

Listed in PORT_PLAN "Questions waiting for the owner"; the audits end with the full lists and recommendations.

## Known, harmless, needs the owner's call

`tools/check_release.py` lists history findings for example values in pushed commits (a private-range IP example in
`dd7974e`, a made-up drive path in `21f8931`); both were replaced later. Removing them means rewriting history (a
force push), which only the owner can approve.

## Working style (the owner's)

The lead plans, integrates and checks; Sonnet subagents do narrow execution; Fable subagents audit, design harder
ports, review every batch before it is committed, and hunt bugs. Run independent agents in parallel, never two on one
file. No agent instantiates the app against the real `user_data/`. Usage is limited: check it before starting a
phase, prefer inline work for small pieces, and stop at a commit when the owner asks.
