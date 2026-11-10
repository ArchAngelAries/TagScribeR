# Next session: where the work stands (handoff, 2026-11-10)

Read `CLAUDE.md` (standing rules; rule 10 is new: TagScribeR's own names, see `docs/dev/NAMING.md`), then
`docs/dev/PORT_PLAN.md` (the one tracked roadmap: its "Now" section has the decisions, the order of work and the
questions waiting for the owner). This file is the short version; PORT_PLAN wins where they differ.

## State

- Work branch **`port/v8-wave2`**, pushed. `main` is untouched at `aaf604b`; merge only with the owner's go.
- Upstream reference: `.claude/fizgig-src`. Committed code is adapted from v8.7.3 (`9162fb20`); the target for new
  work is just past v9.4.0 (`c11134a2`). Read files with `git -C .claude/fizgig-src show <commit>:<path>`.
- Studies to work from: `audit/12` (training core, fixes F1-F13, port order), `audit/13` (new model families and
  tools), `audit/14` (security, privacy, safety), `UI_UX_REDESIGN.md` (the "Trace" direction, measured freezes, a
  nine-stage plan, `mockups/tagscriber_trace_mockup.html`), `CAPTION_SPEED.md` (captioning speed).
- Nothing in training has been run on real weights by this project since the owner's Krea 2 runs; the full
  fine-tune and MOSS-Image are tested on tiny CPU models only.

## Done since the last handoff (all pushed unless listed under "In the working tree")

- The full fine-tune and MOSS-Image, reviewed; audit-12 fixes F1, F2's guard, F4, F5, F8, F12, F13.
- Ctrl+A toggles select all / none.
- TagScribeR's own names: labels, built-in preset names (diamond marker), help, messages; one Credits topic; the
  README's credits; `tests/test_naming.py` guards it.
- Token themes "TagScribeR dark" / "TagScribeR light" in Settings (reviewed, nine findings fixed); `dark_teal`
  stays the default and is pinned by tests. Not yet looked at on a real screen.
- Patreon and Ko-fi links at the bottom of the sidebar (`core/support.py`, `tabs/support_bar.py`).
- The security and privacy pass (audit 14): keys bound to their host and really deleted, a redacting log filter,
  allow-list metadata cleaning with a re-read check, untrusted text shown as plain text, "Copy log for sharing" and
  "Export for sharing", preset imports that ask before applying paths, a validated pause sidecar, bounded
  safetensors header readers, atomic LoRA saves, windows / menus deleted after use, a stricter release check.

## In the working tree when this was written (check `git status`)

1. **Captioning speed** (`inference/`, `core/config.py`, `tools/bench_caption.py`, `tests/test_caption_speed.py`,
   `tests/test_caption_security.py`, `tests/caption_sim.py`, `docs/dev/CAPTION_SPEED.md`): a Fable agent's work on
   the request pipeline for API servers (adaptive parallel requests, prepare-ahead, an image pixel budget, a loop
   guard, a server probe). Before commit: read its report, run the suite, fix the redirect / key findings the
   security review named for `inference/server_probe.py` and `inference/openai_api.py` if it has not, review, then
   wire its options into `tabs/caption.py` with tooltips and Help Center text (the lead's job; the agent lists the
   controls). The owner runs `tools/bench_caption.py` against their own LM Studio; no agent sends requests to it.
2. **UI stage 0, "stop the freezes"** (`tabs/workspace/*`, `tabs/gallery.py`, `core/image_utils.py`,
   `tabs/settings.py`, `main.py`, `tools/bench_ui.py`): a Sonnet agent's performance-only changes. Before commit:
   its before / after benchmark, the full suite, and a Fable review for "same order, same results, no lost edits".

## Verifying a commit while agents edit in parallel

Run the suite on the exact commit in a temporary worktree, not in the working tree:
`git worktree add --detach <temp>/wt HEAD`, run pytest there with the repo's venv, then
`git worktree remove --force <temp>/wt`. Widget tests must not leave windows open (`tests/conftest.py` closes them
per module; before that fix the theme tests took minutes after the tab tests).

## For the owner to check on real hardware

1. An ordinary Krea 2 run on this branch (still pending): stage 1's numerics, the Loss tab, Save now / Sample now.
   Built-in presets have new names (the first is now "Quick start"); their values did not change.
2. The two new themes on a real screen (Settings, Appearance): text sharpness, the sidebar icons, switching back
   and forth.
3. `tools/bench_caption.py` against LM Studio once the captioning work is committed (`CAPTION_SPEED.md` says how
   and which LM Studio settings matter).
4. A fine-tune is NOT ready to be trusted (tiny CPU model only; it needs a bf16 model file).

## Next work, in order

1. Commit the two pieces above (review first).
2. UI stages 1-4 of `UI_UX_REDESIGN.md` (the workspace joins the theme; sort / filter / open off the UI thread; the
   shell with job chips; the thumbnail pipeline), each after the owner's answers to its questions.
3. Training catch-up per PORT_PLAN: F6 / F7 with one cache revision, F3, then Muon and the preset move, F9 / F10,
   the fine-tune and MOSS-Image at the target, mask awareness and start warnings, then new families.
4. Audit 14's open items: a pixel limit for very large pictures, cache cleanup in Settings, the app-crash case.

## Questions waiting for the owner

PORT_PLAN lists the training ones. `UI_UX_REDESIGN.md` ends with thirteen (layout opt-in, the default theme for new
installs, softer outlines, job notifications ...). Audit 14 ends with six. The commit e-mail in the public history
is the owner's decision (a history rewrite needs a force push).

## Working style (the owner's)

The lead plans, integrates and checks; Sonnet subagents do narrow execution (one set of files each); Fable subagents
audit, design, and review every batch before it is committed. Never two agents on one file. No agent runs the app
against the real `user_data/`, opens the owner's images, loads a model or sends a request to the owner's LM Studio.
Usage is limited: commit early, keep the number of Fable agents small, and stop at a commit when the owner asks.
