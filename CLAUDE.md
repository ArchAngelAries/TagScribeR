# Working on TagScribeR

TagScribeR is a PySide6 desktop app for image-dataset work (browse, caption, tag, edit, metadata, health) with native
LoRA training ported from the [Fizgig](https://github.com/shootthesound/Fizgig) trainer. Read `docs/ARCHITECTURE.md`
first. The current work plan is `docs/dev/PORT_PLAN.md`; the parity audits it is based on are in `docs/dev/audit/`.

## Standing rules (the owner's, not suggestions)

1. **Fizgig is the reference for training.** Keep pace with it where it gets things right: features, presets,
   parameters, adaptive LR, previews, supported file formats, required packages. Preset keys are Fizgig's GUI keys so
   its preset files import unchanged.
2. **Keep what works.** Never narrow, skip, replace, refuse or make optional something that works in TagScribeR, or
   something Fizgig supports, on your own judgement. A good technical reason is a reason to ASK the owner, with a
   recommendation, not a reason to decide. Remove only controls that no longer do anything. TagScribeR may support
   more than Fizgig (fp8 base, NoobAI v-pred, batch sizes above 1, the ROCm VAE attention fix ...): keep those.
3. **Never run real model training and never load real model weights.** Verify with unit tests on tiny random models
   (`tests/tiny_family.py`). Reading a checkpoint's header is fine. The owner runs real training on their own machine
   and reports back; a change to a verified path is not "verified" until they have run it.
4. **Never use the owner's real datasets or images** for tests, screenshots or anything published.
5. **Pushing.** Push work branches (`port/...`, `wip/...`) freely so the owner can pull and test. NEVER push to
   `main`, never merge into `main`, never force-push, without the owner's explicit go in the current conversation. Run
   `tools/check_release.py` before every push and fix anything it finds.
6. **Privacy.** The repo is public. No absolute local paths, user names, e-mail addresses, machine names or dataset
   names in code, comments, tests, docs or commit messages. Use neutral examples such as `D:\Datasets\demo_dataset`.
   Refer to Fizgig by its GitHub URL. Screenshots come only from `tools/make_screenshots.py` (generated demo data).
7. **Every feature ships with help:** tooltips, Help Center text (`tabs/help_content.py`), hotkeys and docs in the
   same change. Serve both beginners (simple view) and power users ("Show all settings").
8. **TagScribeR is for NVIDIA users too.** "Not needed on AMD" is not a reason to cut something. It was built and
   tested on Windows 11 with an AMD Radeon card on ROCm; NVIDIA and Linux paths cannot be run by the owner, so be
   honest in docs about what has and has not been run on real hardware.
9. **Report faithfully.** Say what was verified, what was only unit-tested, and what only a real run can confirm.
   List differences from Fizgig as questions for the owner. Do not overstate.

## Layout

- `core/`: UI-independent foundation. No Qt widgets, no torch at import.
- `inference/`: captioning providers (Transformers VLMs, OpenAI-compatible servers, WD tagger).
- `tabs/`: the UI. One shared dataset workspace across tabs.
- `training/`: native training. No Qt. `params.py`, `presets.py`, `pipeline.py` and `registry.py` are torch-free.
- `training/families/<family>/`: one package per model family (description, driver, model code).
- `tools/`: `install.py`, `make_screenshots.py`, `check_release.py`.
- `tests/`: pytest. `tests/conftest.py` hides the GPU on purpose: the suite must never open a GPU context.

## Running things

```bash
QT_QPA_PLATFORM=offscreen PYTHONIOENCODING=utf-8 python -m pytest -q -p no:cacheprovider
python -m pyflakes training tabs main.py
python tools/check_release.py
```

On the owner's Windows machine the interpreter is `venv\Scripts\python.exe` (Python 3.12, packages installed with
`uv`, pinned ROCm torch wheels). In another environment, a CPU build of torch plus `requirements.txt` is enough for
the test suite. Widget tests need one offscreen `QApplication` for the whole session.

## Porting from Fizgig

- Port from a checkout of Fizgig at the commit named in `docs/dev/PORT_PLAN.md`. Keep it outside the tracked tree
  (for example `.claude/fizgig-src`, which is git-ignored). Never copy `detect_gpu.py` (GPL, from comfyui-rocm):
  TagScribeR detects GPUs with its own `core/hardware.py`.
- Every ported file carries a header naming its Fizgig source file and the changes made. Keep upstream headers.
  Record new ported files in `THIRD_PARTY_NOTICES.md`. TagScribeR is GPL-3.0; do not hold features back over licence
  worries, record the provenance and move on.
- Training never runs in the app process. The Train tab freezes a run folder and starts child processes
  (`training.cache`, then `training.train`); control is by files in the run folder.
- TagScribeR-only fixes that any port must preserve: `training/modules/wide_attention.py` and its use in the VAEs
  (PyTorch's built-in attention is wrong on AMD ROCm for heads wider than 256); the `latent_rev` mark in
  `training/cache.py`; the CUDA guard in `training/modules/offloading.py`; the fp8 base precision
  (`training/quant.py`, `training/modules/fp8.py`); the run-folder launcher `training/pipeline.py`.
- The SDXL family and Anima were written before Fizgig had them. Where their presets are not Fizgig's measured ones,
  never describe them as measured.

## Conventions

- Data safety: caption writes go through `core.caption_io.write_caption` (atomic, with backups). Deletes go to the
  Recycle Bin. Copies never overwrite.
- Match the surrounding code's style and comment density. Run pyflakes on files you touch.
- Commit each finished piece separately with a clear message. The full test suite must pass before each commit.
- Offscreen Qt cannot render emoji; buttons use qtawesome icons. In Qt titles `&` is a mnemonic: write `&&` or "and".
- When delegating to subagents: give each its own files, tell it to mirror Fizgig and to report anything it cannot
  do as a question, and check the parts that would silently ruin a real run (objective, timestep sampling, presets,
  key formats, defaults) against the source yourself before calling the work done.
