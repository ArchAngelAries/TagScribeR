# Audit 11: Fizgig 5d31b2f (v8.7.1) to origin/HEAD (v8.7.3, 9162fb20)

Source: https://github.com/shootthesound/Fizgig. 20 commits, 21 files, +733/-45. Line numbers are in origin/HEAD.
Everything below was read from the diffs and release notes unless marked (inferred).

## Summary

- **Training numerics change: none.** No objective, timestep-sampling, LR, optimizer or trained-parameter-set change for any
  family. The one trainer-behaviour change is the MOSS-Image fine-tune now honours `lora_sft` (attention only), which is an
  opt-in tick, plus the planner's memory constants for MOSS-Image fine-tunes.
- Everything else is UI (Samples tab greying, per-epoch prompt record, News card, download picker), preview routing
  (previews on the Base DiT when it is not the released Base), a fine-tune preview bug fix, and docs.
- Shared layer touched: `families/description.py`, `families/form.py`, `families/train.py`, `families/base_check.py` (new),
  `scripts/fetch_models.py`. MOSS-Image only: `families/moss_image.py`, `moss_image/driver.py`.
- Clean-end share 30 and the MOSS-Image 70-epoch presets were already in 5d31b2f (v8.7.1); only their release notes were added.

## 1. "Train attention only" (MOSS-Image only; changes what is trained, opt-in)

- `src/fizgig/families/moss_image.py:260`: `FamilyOption("MOSS_LORA_SFT", "Train attention only", kind="check",
  tokens="lora_sft=1", section="after", summary="attention only")`. Default: unticked (no `default` given). The GUI key
  `MOSS_LORA_SFT` and the family-option token `lora_sft=1` are unchanged from 5d31b2f; only the label, hint and summary
  changed (was "Base is SFT or a fine-tune of it", summary "SFT base").
- The removed `mode="lora"` restriction means the tick now shows for **fine-tunes as well as LoRAs** (previously LoRA mode only).
- LoRA: `moss_image/driver.py:355-362` `lora_target_names()` keeps only names ending in `ATTN_MODULES`
  (`proj.q`, `proj.k`, `proj.v`, `proj.o`, defined at driver.py:34) when `lora_sft == "1"`. This is not new (already in 5d31b2f).
- Fine-tune (new): `moss_image/driver.py:175-190` `ft_spec()`: `components=ATTN_MODULES if sft else BLOCK_MODULES`, i.e. the
  trainable parts are the four attention Linears per block instead of all six (q, k, v, o, fused MLP up+gate, down). Text
  adapter, embedders, modulations, final layer stay frozen either way.
- Variants: the hint says recommended on SFT or a fine-tune of it (training SFT's feed-forward layers "breaks anatomy"); on
  Base it "may cost quality or slow convergence". Fizgig's measurement (docs/MOSS_IMAGE.md "Fine-tuning"): after 10
  rotations on one dataset a full Base fine-tune scored 49% ArcFace likeness, attention-only 19% (both rendered on the
  fine-tuned Base); attention-only is about 25% faster per step.
- It is a user tick, not auto-set from the detected Base (inferred from code: nothing sets it from `is_standard_base`).

## 2. Kromoss and "not the released Base" detection

What it is: lodestones' full fine-tune of MOSS-Image Base at 1024 px, same architecture/files/recipe, made to fix Base's
blurriness. Repo id `lodestones/Kromoss` (https://huggingface.co/lodestones/Kromoss). Checked by the author on a 5090 as the
Base DiT: int8 auto, 9.4 GB peak at 1 MP, trains, saves, previews (comment at moss_image.py:62-63).

- Files (moss_image.py:64-66, on the `moss_image_dit` ModelFile): `alt_repo="lodestones/Kromoss"`,
  `alt_path="diffusion_models/kromoss_bf16.safetensors"`, `alt_label="Download Kromoss"`,
  `fetch_choice=("MOSS-Image Base", "Kromoss")`, `alt_size_gb=10.16`.
- Turbo LoRA (moss_image.py:79-88, `moss_image_turbo_lora`): alt `alt_repo="lodestones/Kromoss"`,
  `alt_path="8-steps-dmd-lora.safetensors"`, `alt_label="Download Kromoss Turbo"`, ~1.2 GB; 8 steps, CFG 1. The LoRA is
  distilled from Kromoss/Base (the default turbo LoRA row is lodestones' 8-step DMD of SFT, "soft on Base"). `bulk_fetch=False`
  unchanged: the Turbo LoRA is never in "Download models for me", only its own link.
- New description fields: see section 5.
- Detection, `src/fizgig/families/base_check.py` (new, 32 lines):
  - `is_standard_base(desc, path)` (line 24) returns True = released Base, False = another model, None = unknown (no
    `desc.standard_base`, empty/missing path, unreadable file).
  - `_check()` (lines 13-21, `lru_cache` keyed on path, mtime, size, key, want): opens the safetensors header with
    `safe_open`; if the metadata contains `fizgig_finetune` returns False (a Fizgig fine-tune save keeps its norms frozen, so the
    norm sum would match Base; the metadata names it instead); else finds the tensor whose name is `key` or ends in `"." + key`;
    not found returns None (treated as Base); else compares `abs(float(tensor.double().sum()) - want) < 1e-6`.
  - MOSS-Image fingerprint (moss_image.py:233-236): `standard_base=("layers.41.norm2.weight", 3658.7021484375)`. Measured
    19 Oct 2026: Base 3658.7021484375; SFT (bf16 and int8) 3658.7646484375; Kromoss 3658.6904296875. The sums differ only in
    the 5th-6th significant digit, so the 1e-6 tolerance on a double-precision sum is exact-match territory. (Inferred: the
    int8 SFT file still stores this norm in bf16, as the docstring says norms are kept by quantised files. It reads a tensor
    of 1 norm vector, so it only loads a tiny tensor.)
  - Caveat to ask Fizgig/owner: the fingerprint covers the ComfyUI single-file layout (key suffix match). The notes say the
    diffusers shards also load; a shard layout would only match if that shard holds `layers.41.norm2.weight` (unverified).
- Preview routing, `families/form.py:890-901` in `state()` (shared layer, gated on `desc.train_preview_checkpoint and
  desc.standard_base`):
  - Not the released Base (False): note on `SAMPLE_CHECKPOINT` ("Your Base DiT isn't ...'s own Base, so previews render on
    it ..."), and once per Base file (tracked in the hidden form value `_SAMPLE_CHECKPOINT_BASE` = the file path) forces
    `SAMPLE_CHECKPOINT` to False. The user can tick it back on and it stays.
  - Back on the released Base (True) with a stale mark: forces `SAMPLE_CHECKPOINT` to `desc.preview_checkpoint_on` and clears the mark.
  - `state()` gained the parameter `base_file=""` (line 712); the web server passes the `role == "dit"` pref path (server.py:2991-2996).
  - Training itself is unaffected (it only changes which model previews render on).
- Commit 8c8edb01 title: "previews render on the Base DiT itself when it isn't the released Base (Kromoss, a fine-tune, SFT)".
- Download picker (shared layer): `ModelFile.fetch_choice`/`alt_size_gb` (description.py:46-48);
  `scripts/fetch_models.py`: `Weight.alt` / `Weight.switch_from` (lines 55-56), `_described_families()` builds `w.alt` (lines
  161-179), `_chosen()` (line ~449) picks main or alt and sets `switch_from` to the other build's filename, `fetch_weight()`
  (line ~332) moves a pref pointing at the other build to the picked one, `fetch(..., choices=None)`, and a new CLI flag
  `--choose PREF=main|alt` (repeatable). Web: `server.py:2082-2101` `_start_fetch(family, token, optional, choices)`, the
  `/api/fetch` body takes `choices`, `app.js:698-715` draws a select beside "Download models for me", defaulting to "alt" when the
  pref's basename equals the alt file's basename.
  A pref pointing at a user's own file is kept unless the user picks the other build (`switch_from` only matches the other
  build's own filename).

## 3. Fine-tune memory measurements (MOSS-Image, v8.7.3)

Measured 19 Oct 2026 on an H100 at 1 MP, whole model in one window (driver.py:175-190 comment):
- All six parts: **12.6 GB** peak. Train attention only: **8.3 GB** peak. A preview peaked at **9.9 GB** (under training).
- Planner constants in `ft_spec()`: `overhead_gb=6.0 if sft else 4.5` (4.5 = 0.6 GB above the full run's fit; 6.0 for attention
  only, "whose window is credited the whole 4-bit trunk though the FFN stays in it"), `calib_mp=1.0`, `act_gb_per_mp=2.0`
  ("not measured yet, as Anima and Flux 3"). Before: no `overhead_gb` (cautious defaults, "memory figures not measured yet").
  FTSpec is `families/ft.py:36` (the three field names are existing FTSpec fields; defaults in that file, not re-read here).
- Planner outcome (docs/MOSS_IMAGE.md "Fine-tuning"; the 16 GB row is measured, 12 and 10 GB rows are the planner's, NOT run):

| Card | Base (all parts) | Train attention only |
|:--|:--|:--|
| 16 GB and up | whole model at once, nothing rotates | whole model at once |
| 12 GB | 2 parts per rotation | whole model at once |
| 10 GB | 9 parts per rotation | many small parts, blocks streamed from system RAM (slow) |

- Fine-tune bf16 master copy: about 9.5 GB system RAM. EMA off for a fine-tune. MOSS-Image fine-tune preset LR 1e-5 (the
  "MOSS-Image Fine-tune" preset, `_preset(16, lr=1e-5, ft=True)`, already in 5d31b2f). Fine-tune Base, not SFT.
- docs/FINETUNE.md: adds a MOSS-Image column (16/24/32 GB all supported) and says 16 GB is the floor "for these families"
  (MOSS-Image lighter). Other-family figures there are unchanged (Klein 14.0 GB at 1 MP / 12.6 GB at 0.25 MP; Krea 2 up to 11.4 GB;
  H3 8.8-12.3 GB).

## 4. Preview-checkpoint fine-tune fix (shared layer, bug fix)

`families/train.py:1562-1566`: `if preview_checkpoint and encoded is not None:` has a new branch
`elif finetune: logger.info("[sample] a fine-tune previews on the model it trains, not the preview checkpoint")`. Before, a
fine-tune with a preview checkpoint (MOSS-Image's SFT previews, on by default) failed its first preview and then trained on
with previews off, because the checkpoint carries a LoRA and a fine-tune changes the model itself. Applies to every family
with `train_preview_checkpoint`, not just MOSS-Image (inferred from code location: the branch is in the shared `train_family`).
The form side of this (the tick state for fine-tunes) is not changed in form.py beyond section 2 (inferred; not checked
whether the web UI also greys the tick in fine-tune mode).

## 5. New / changed description fields and keys

`families/description.py`:
- `ModelFile.fetch_choice: tuple = ()` (line 47): names for the two builds, e.g. `("MOSS-Image Base", "Kromoss")`.
- `ModelFile.alt_size_gb: float = 0.0` (line 48): the alt's size (falls back to `size_gb`).
- `FamilyDescription.standard_base: tuple = ()` (line 429): `(tensor name, sum)`; `()` = no check.
- Existing, reused: `alt_repo`, `alt_path`, `alt_label`, `alt_note`.
No new preset keys. No default changes. New GUI/form keys: `_SAMPLE_CHECKPOINT_BASE` (hidden form value, a path string).
Changed GUI key semantics: `SAMPLE_CHECKPOINT` is now force-set by `state()` as above. `MOSS_LORA_SFT` relabelled (see 1).
`form.state()` signature gained `base_file`.

## 6. Samples tab: Steps / Turbo strength (UI, shared)

- `families/form.py:~902-911` (inside the `if desc.train_preview_checkpoint and checkpoint_file and
  values.get("SAMPLE_CHECKPOINT", True)` block): if the speed (Turbo) LoRA is set and on disk
  (`speed_file`), `desc.preview_speed() is not None` and `not desc.samples_turbo_pace`, then `FAMILY_TURBO_STRENGTH` is added to
  `disabled` with a note "The <name> applies to previews on the training model: untick "<tick label>" to use it.". Reason:
  previews on the preview checkpoint never take the speed LoRA.
- `web/static/app.js` `turboFollow()` (~line 487-530): the auto-follow of Steps/CFG to the mode's values now runs only on
  the form's first load for a model (`S._turboFollowFor`) or when a switch (flags) changes, so Steps typed by hand stay.
  Previously they snapped back to the mode's default. (Commit b1a473ca.)
- Behaviour relevant to a PySide port: same two rules in the Samples tab logic.

## 7. Per-preview prompt recording (trainer + UI)

- `families/train.py:232-247` `_record_prompts(sample_dir, output_name, epoch, prompts)`: writes
  `<sample_dir>/.prompts_<output_name>.json` as `{"<epoch>": [prompts...]}`, atomically (`.tmp` then `os.replace`), warns on
  OSError. Called after each preview set at train.py:1549 (checkpoint-preview path) and train.py:1558 (normal path).
  Reads existing file tolerantly (`OSError`, `ValueError` give `{}`).
- Reader: `web/server.py:1598-1608` `epoch_prompts()`; gallery endpoint (server.py ~2840) returns `epoch_prompts` plus `prompts`
  = the newest epoch's prompts (falling back to the old `run_prompts`). `app.js` `previewFigure(it, prompts, eager, byEpoch)` uses the
  epoch's own prompt for `it.idx`, else the run's. Commit 38ad3f36 skips malformed entries (non-digit keys, non-list values).
- No effect on training. Hidden dotfile in the samples folder.

## 8. Other (UI/docs only)

- Start tab News card: `web/news.py` (new, `NEWS` list, newest first, shown 3), `schema.py` adds `"news"` to the schema,
  `app.js` `newsCard()`; the web UI now always opens on the Start tab. `docs/NEWS.md` (new): "MOSS-Image training, fixed" and
  "Kromoss and its 8-step Turbo LoRA" (both 19 Oct 2026). Not relevant to a Qt port except as release-notes content.
- README / docs/CLI.md: wording only; CLI.md notes `--family_option lora_sft=1` is "the GUI's Train attention only; it also
  lowers a fine-tune's VRAM on Base".
- RELEASE_NOTES_v8.7.2.md: Clean-end share 30% applies to fine-tunes (already in code at 5d31b2f); ArcFace tables (epoch 3/6/9/12/15/20:
  Before 40/44/52/58/63/60%; Now 42/59/63/66/69/69%); Standard preset (rank 32, 1 MP) reached 66% by epoch 12, 72% by epoch 32, vs 71%
  for the dataset's own photos against the same 3 reference photos. v8.7.3 notes: the memory table, fine-tune preview fix, the tick
  rename, per-preview prompts.
- docs/MOSS_IMAGE.md: the "How long to train" paragraph rewritten with the 66%/72%/71% figures; Clean-end paragraph adds
  by-eye notes; Previews section adds the "any other Base DiT" line; new "Fine-tuning" section.

## 9. What a fresh MOSS-Image port should take from origin/HEAD instead of 5d31b2f

Take origin/HEAD versions of: `families/moss_image.py` (Kromoss alt rows, `standard_base`, relabelled tick, now offered in fine-tune
mode), `moss_image/driver.py` (`ft_spec` with `lora_sft`, `overhead_gb` 4.5/6.0, `calib_mp` 1.0, `act_gb_per_mp` 2.0),
`families/base_check.py` (new file, needs a `THIRD_PARTY_NOTICES.md` entry), and the shared pieces `description.py`
(`fetch_choice`, `alt_size_gb`, `standard_base`), `form.py` `state(base_file=...)` + the Turbo-strength greying, `train.py`
(`_record_prompts`, the `elif finetune` preview-checkpoint branch), `fetch_models.py` (choices). Docs: take the new MOSS_IMAGE.md
Fine-tuning section, FINETUNE.md MOSS-Image column. Re-check that the TagScribeR-only fixes (wide_attention in the VAE, CUDA
guard in offloading, `latent_rev`, fp8 base) still apply on the 5d31b2f-to-HEAD MOSS files; the diffs here do not touch the
MOSS VAE or offloading code.

## 10. Recommendation

Safe to move the port baseline from 5d31b2f to origin/HEAD for the remaining work: the delta touches no training objective,
timestep sampling, LR logic or presets, the shared-layer changes are additive (new optional fields and a bug fix), and the only
trainer-behaviour change is MOSS-Image's opt-in attention-only fine-tune plus its planner constants. Port the shared
`train.py` preview-checkpoint fix and `_record_prompts` with the rest of wave 2.

## Open questions

1. `FTSpec` defaults for `overhead_gb`, `calib_mp`, `act_gb_per_mp` and how the planner uses them were not read (ft.py not in the
   diff); TagScribeR's fine-tune port will need them from `src/fizgig/families/ft.py` at HEAD.
2. Does the SFT/Kromoss norm fingerprint hold for the diffusers shard layout (the key may live in a particular shard)? Fizgig
   documents shards as loadable but only measured single files.
3. The 12 GB and 10 GB fine-tune rows are Fizgig's planner output, not run on those cards; TagScribeR should say so.
4. Should TagScribeR's port show "Train attention only" in fine-tune mode too (Fizgig now does)? Recommend yes (parity).
5. Only the web UI path was read for the tick/greying behaviour; the Fizgig desktop (Tk) GUI path, if any, was not checked.
