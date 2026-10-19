# Port plan: bringing TagScribeR's training level with current Fizgig

Status on 2026-10-19. Development notes, not user documentation. Read `CLAUDE.md` first for the standing rules.

This file is the ONE tracked roadmap: the Fizgig port, TagScribeR's own fixes and improvements, the longer-term
roadmap, and the documentation ledger. Update it in the same commit as the work it describes.

## Now (2026-10-19): Fizgig v8.7.1, fixes first

Fizgig moved 738 commits from v7.0.1 to **v8.7.1 (commit `5d31b2f`)**; the checkout in `.claude/fizgig-src` is at
that commit. **New port source: `5d31b2f`.** Stages 1-5 below were ported against `1c8ec88`. Seven read-only
reports (2026-10-19) cover the delta and the app as a whole:

- `audit/05_shared_layer_v8.md`: shared training layer. Numerics, quantisation, optimizers, EMA, Adaptive LR and key
  formats are unchanged upstream since 7.0.1; the gaps are features and robustness.
- `audit/06_existing_families_v8.md`: per-family drivers of Krea 2, Klein, Anima and H3 are byte-identical upstream;
  the gaps are presets, preview recipes and memory figures. No text-cache revision bump is needed.
- `audit/07_new_families_v8.md`: Z-Image Turbo (7.1.0), ACE-Step (7.3.0, audio), Flux 3 Image Klein (8.1.0),
  MOSS-Image (8.5.0).
- `audit/08_bug_and_security_review.md`, `09_app_core_review.md`, `10_inference_install_review.md`: bug and exploit
  hunts over training, the app core / tabs, and inference / installer / network.
- `IMPROVEMENT_IDEAS.md`: efficiency ideas for the app, captioning and training, with measurements.

**Baseline moved to v8.7.3 (commit `9162fb20`, local session 2026-10-19).** Fizgig moved 20 commits past `5d31b2f`;
`audit/11_fizgig_v8_7_3_delta.md` digests them: nothing changes an objective, timestep sampling, LR or preset; the
shared additions are `_record_prompts` (each preview round's prompts, `.prompts_<name>.json`), a fine-tune previewing
on the model it trains instead of a preview checkpoint, and the description fields `ModelFile.fetch_choice`,
`alt_size_gb`, `FamilyDescription.standard_base`. Everything else is MOSS-Image (the "Train attention only" label for
`MOSS_LORA_SFT`, now also on fine-tunes; Kromoss as an alternative Base; `families/base_check.py`). Port the remaining
work, MOSS-Image included, from `9162fb20`; audit line numbers from `5d31b2f` still hold for every unchanged file.

**Handoff (2026-10-19, end of the cloud session):** all fix batches and catch-up wave 1 are done and merged to
`main`; `docs/dev/NEXT_SESSION.md` lists what the owner should check on real hardware and the next work in order.

Order of work: (1) confirmed bugs (below), (2) the v8 catch-up for the existing families and shared layer, (3) the
rest of stage 5 (fine-tune, DOP, Multi Concept), (4) new families, (5) improvements, interleaved where independent.

### Fixes (from the reviews; each lands with a regression test)

- [x] EXIF blocks written with a doubled / forbidden `Exif\0\0` header (09 #1): every metadata write.
- [x] Training (08; plus the review's: the Train tab tolerates a half-typed name, FAT32 / cloud-sync time shifts re-encode only when a file is newer, pre-stamp text caches notice a newer partner): mid-run caption fix on an Edit LoRA run re-encodes without its reference (#1, high); latent
  cache reused after an in-place edit (#2; Fizgig 8.4.0 re-encodes on mtime); a caption extension that names an image
  type overwrites images (#3); image-pair slider previews never use the pairs' caption (#4); stale-cache glob can
  delete another family's cache (#5); `run_dir_for` unvalidated (#6); resume `torch.load` without
  `weights_only=True` (#8); path keys applied from an imported preset are named in the report (#7).
- [x] Shared layer (05 #1): block swap pairs a bare block with a LoRA-wrapped one (Fizgig #199) - H3 with swap.
- [x] App core (09 #2-#9; and the review's findings: export leaked XMP, XMP orientation, 16-bit orientation; caption writes refuse image names): editor "Overwrite originals" convert without confirmation; collection copy overwrites a
  caption; in-place edits drop PNG text / XMP and ignore the quality slider; 16-bit PNGs clipped; header-size guard
  before decoding huge images; UTF-16 captions; file modes and fsync in atomic writers; assorted (inspector shows a
  trashed image, `../` collection names, `.json` sidecars, caption extension in counts).
- [x] Inference (10; plus the review's: old profiles' 120 s / 3 retries become automatic, a stale .incomplete never hides a model, key bound to the first address, resume into the partial folder, an unencodable image fails alone; #6 revision pins: the field exists, no pins known yet): `trust_remote_code` per model, never global (#1); partial downloads shown as installed (#3); API
  timeouts / retries / cancel and the blocking unload on close (#4); non-InferenceError aborts a batch (#5); the key
  kept when the server URL changes (#7); `items_started` never emitted (#8); gated-model message (#9).
- [x] Owner-reported: a failed image's red outline hides the selection highlight (now both show).

### v8 catch-up, decided by the standing rules (recorded so later sessions do not ask)

Mirrored (Fizgig corrected or improved it): Krea 2 NF4 1 MP memory 15.0 GB; Slider presets at rank 8 (Krea 2, Qwen,
Anima); SDXL Strong / Standard 50 epochs; Krea 2 Turbo previews at strength 0.7, 10 steps (the strength box keeps
1.0 one edit away); Anima previews on the Turbo LoRA by default (strength 1, CFG 1, 10 steps); Qwen's preview recipe
(Comfy-Org Turbo LoRA, CFG 3 on the plain model, training adapter v2 with `replaces`); `wide_attention`'s fp32 path
gated on ROCm (NVIDIA keeps SDPA; no change on the owner's card); resume-state save retry; latent re-encode on mtime;
test compile before `compile_blocks` and eager previews under compile; per-step loss log and Sample now / Save now;
slider override ignored and live prompt changes at epoch boundaries; multi-reference edit LoRAs (Klein 4, Qwen 10);
H3's pre-sm_80 NVIDIA paths (unit-tested only, documented as unrun here).
TagScribeR's own (not GPL code): pick the discrete GPU over integrated graphics in `core/hardware.py`, honouring a
user-set `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES`.
Kept as TagScribeR's: batch size above 1 (upstream added and reverted it); the window-only H3 LoRA (with the
block-swap fix above); `retired_preview_defaults` / `preview_reset` removed (dead at HEAD); 8.6.1's "Steps / CFG
follow what previews render with" rule is in `pipeline.train_kwargs` (an untouched Steps / CFG box means the Turbo's).
Deferred: Qwen Turbo DiT previews (workbench stage); `sample_strengths` (only ACE-Step declares them).
Stage 5 gains: DOP and Multi Concept on every family; LoRA Isolation after the fine-tune.
New families, in this order: MOSS-Image, Flux 3 Image Klein (gated, non-commercial weights: a licence link on the
row, no stored token; its forced cuDNN attention routed through `training/modules/sdpa.py` for ROCm), Z-Image Turbo.
ACE-Step (a music trainer): skipped (the owner, 2026-10-19). Do not port it unless the owner asks again.

## Where things stand

- `main` is the published release. Its training code was ported from Fizgig as it was on 2026-09-29
  (commit `746ddab`).
- Fizgig has since moved 251 commits to v7.0.1 (commit `1c8ec88`, 2026-10-05). Krea 2, FLUX.2 Klein and MiniMax H3
  now all train through Fizgig's driver layer (the architecture TagScribeR's `training/` was ported from), the
  standalone trainers are deleted, and Fizgig has its own SDXL and Anima families, sliders, a shared full fine-tune,
  shared clip handling and torch.compile for every family.
- **Port source: Fizgig at `1c8ec88`.** Clone https://github.com/shootthesound/Fizgig and check out that commit into a
  git-ignored folder (`.claude/fizgig-src`). The audits' line numbers refer to it. When the port has caught up, check
  Fizgig's newer commits before moving the baseline.
- Four read-only parity audits are in `docs/dev/audit/`. They were made by reading, not by running anything. The
  claims marked "verified" below were re-checked against the source by the lead; treat the rest as leads to confirm
  while porting, not as facts.
- Real hardware results so far: Krea 2 has been trained end to end twice on the owner's machine (AMD Radeon, ROCm,
  fp8-scaled RAW checkpoint, Auto -> INT8 base, no block swap, AdamW 8-bit, adaptive LR, EMA 0.98, loss watch, base and
  Turbo-LoRA previews). No other family has been run on real weights.
- Branch `wip/h3-stills` held two commits for stage 4 (the H3 noise-band LR dial and Turbo-LoRA previews). Both are
  now on the work branch, re-keyed as the audit asked; `wip/h3-stills` is superseded and can be deleted.

## Stage 1 status (2026-10-08)

Done on branch `port/stage1-bt67en`, unit-tested only (tiny random models, CPU; no GPU, no real weights, inductor not
run): Krea 2 memory figures and `auto_precisions`; the loop fixes; cache-first tokenizer loading; Slider / Fine-tune
presets refused; LoRA reader (`alias_flat`, LoHa, `lora.down`, `unet.`); loss-watch exclusions per family; Krea 2
numerics of decision 2 (bf16 INT8 scales, bf16 loss order under autocast, one caption per forward with a text-cache
revision, clip signal Klein only, RAW CFG 4.5); driver hooks (`batch_cond`, `step_policy`, `after_optimizer_step`,
`run_metadata`, `frozen_file_added`, `park_for`, `save_preview`, `plan_run`, `load_planned`, INT8 `store_device`);
Qwen Fast Identity Mode and torch.compile with compile shared by every family (`training/compile.py`, decided on the
empty card). The owner's next ordinary Krea 2 run on this branch confirms the numeric changes (see Settled).

Left for later stages on purpose: the Krea 2 and Qwen Slider presets (sliders are stage 5); hooks of features not
ported yet (sliders, fine-tune, clips, preview checkpoints, legacy state order, `cache_stage`).

## Stage 2 status (2026-10-08)

Done on `port/stage1-bt67en`, unit-tested only: Klein's
presets (Identity at rank 8), LoKR, EMA control (Off), the whole optimizer catalog (Qwen too); Auto INT8 then NF4 with
no Auto swap, the fp8-file-as-is rule when uncompiled, Fizgig's measured memory; block ids `double_N` / `single_N`
(old spelling accepted) and `ss_train_blocks`; torch.compile of both block lists; other trainers' Klein LoRAs
(diffusers q / k / v fusion); Edit LoRAs and reference previews; Distilled 4-step previews with the park / restore
handoff, RAM cache and INT8 option. Needs a real Klein run (Distilled previews on and off, an Edit LoRA) before it is
called verified.

The owner does not train Klein, so its real-hardware check waits on other users; it stays marked experimental.

## Stage 3 status (2026-10-08)

Done on `port/stage1-bt67en`, unit-tested only. SDXL: Fizgig's Strong / Standard presets first (TagScribeR's kept),
INT8 / NF4 with Fizgig's memory table, LoKR with kohya stems, the whole optimizer catalog, DPM++ 2M SDE Karras previews
for SDXL 1.0 (diffusers' scheduler); every variant, v-pred, LoCon and the 225-token captions kept. Anima: the Qwen3
text encoder in float32 (text-cache revision "1", so old caches re-cache once), the Anima repo's T5 tokenizer
(offline first, google/t5-v1_1-xxl as the last fallback), Fizgig's three presets first with TagScribeR's three kept,
INT8 / NF4, LoKR, AI-Toolkit LoRA names, 20-step CFG 4.5 previews and the optional Turbo LoRA (strength 0 by default).
Needs real SDXL and Anima runs before either is called verified.

## Stage 4 status (2026-10-08)

Done on `port/stage1-bt67en`, unit-tested only: MiniMax H3's Fizgig 7.0.1 options under their `H3_*` keys and labels
(the old `MINIMAX_*` keys migrate): Training base fl2va / ref2va (new reference-DiT row), Training structure with the
clean-end share, Medium to High Noise LR (clamped, as upstream), the Circlestone / Ostris / Off adapter choice (two
new Ostris rows, picked per base), the adapter-relative LR ramp, the token refiner; Fizgig's block ids
(`h3blk_N`, `h3_rf_N`) and run metadata; bf16 trainable adapters, weight decay 1e-4 and the 8-bit eps floor, the
whole optimizer catalog, EMA Short run (shared: any family with `ema_short_run`), 768 previews; Turbo-LoRA previews
(`FAMILY_TURBO_STEPS` / `FAMILY_TURBO_PACE`) with run-time AdaLN rows for the adapter, a Context LoRA and the Turbo.
The H3 Slider preset waits for stage 5. Needs a real H3 run before any of it is called verified.

Left for later H3 work (audit B11 steps 4-10): HQQ, the H2D rings and upstream's own Auto plan, the 66 GB bf16 DiT and
bf16 text encoder, clips, audio, voice, multi-concept, distillation, RefMods.

## Stage 5 status (2026-10-11)

Done: sliders (image pairs and prompt pairs) for every family, Fizgig 7.0.1's shared loop, cache flag, settings,
checks, -1 / 0 / +1 previews, presets and per-family hooks (the slider presets move to rank 8 with the v8 catch-up).
Unit-tested only. Left: the full fine-tune (`families/ft.py`; upstream still documents it NVIDIA-only: port with the
streamer behind the CUDA guard and docs saying "not run on AMD"), DOP, Multi Concept, LoRA Isolation.

## Decisions the owner has made

1. **Keep what works.** When Fizgig removed something that still works in TagScribeR, keep it (fp8 as a selectable
   base precision, batch sizes above 1, the old Klein timestep modes, NoobAI v-pred, LoCon ...). Remove only controls
   that no longer do anything (for example Klein's Attention Mechanism dropdown, a dead widget upstream).
2. **Mirror upstream's Krea 2 numerics.** INT8 scales and the loss target in bf16
   (`int8_fp32_scales = False`, verified in `krea2/driver.py:38`), one caption per forward when caching text, RAW
   preview CFG 4.5, the gradient-clip signal to Adaptive LR for Klein only (per description, verified in
   `families/train.py:1082`). Take upstream's corrected memory figures (verified in `families/krea2.py:125`: INT8
   16.2 GB at 0.25 MP and 19.1 GB at 1 MP; NF4 11.4 / 13.4). This changes a verified path, so it needs one more real
   run by the owner before it is called verified.
3. **Dropped:** three earlier port commits that current Fizgig abandoned (network dropout, LoRA+ ratio, the fp8
   `_scaled_mm` fast path). Do not resurrect them without asking.
4. **SDXL and Anima: merge.** Keep everything TagScribeR supports (five SDXL variants and their loaders, NoobAI v-pred
   with zero-terminal-SNR, LoCon, offline strict loading, batch above 1, the timestep window, kohya LDM key names, the
   VAE attention fix). Adopt Fizgig's measured presets (verified in `families/sdxl.py:170-179`: default rank 32,
   alpha 16, LR 5e-5, Adaptive LR off), INT8 / NF4 bases with its memory table, LoKR, and the DPM++ 2M SDE Karras
   preview sampler. Anima: run the Qwen3 text encoder in float32 (verified in `anima/driver.py:78-79`; bf16 puts the
   first token 16% off), use the Anima repo's T5 tokenizer, take Fizgig's six presets and preview settings, add
   INT8 / NF4, LoKR and the Turbo LoRA.
5. **Guiding line:** ensure state-of-the-art capabilities and the best user experience, keep what works, learn from
   Fizgig and keep pace with it where it gets things right, and improve on our own implementations.

## Stages

Every stage lands on the one work branch, passes the full suite, and is pushed for the owner to pull and test. Nothing
goes to `main` without the owner's explicit go.

1. **Shared-layer fixes, then Krea 2 and Qwen Image 2.1 level with upstream** (`port/stage1`).
   - Memory figures and the Auto plan (`auto_precisions`).
   - Loop fixes: the final save also kept under its epoch number, a failed preview fully restored, CFG honoured with
     Turbo previews, preview time left out of s/it, scheduler fast-forward on resume with accumulation, an explicit
     thumbnail not overwritten.
   - LoRA reader: `alias_flat` (OneTrainer / AI-Toolkit files), LoHa, `lora.down` keys.
   - Loss watch: exclusions per family; an image excluded in an earlier run trains normally again.
   - Offline tokenizer / processor loading (text caching must not need the network).
   - A guard so an imported Fizgig Slider or Fine-tune preset is refused or clearly flagged until those features
     exist, instead of silently running as an ordinary LoRA.
   - The Krea 2 numerics of decision 2. Qwen: Fast Identity Mode, compile, the two new presets.
   - Driver interface: bring `training/driver.py` and `training/description.py` level with upstream's hooks and
     fields in a way that keeps TagScribeR's own hooks (`after_epoch`, `optimizer_params`, `configure`).
2. **Klein:** Distilled 4-step previews (default on upstream), reference-image previews, LoKR, Edit LoRAs, EMA
   control, upstream's Auto plan (INT8 then NF4; keep fp8 selectable), the renamed Custom block ids
   (`double_N` / `single_N`), the new Identity preset.
3. **SDXL and Anima merge** (decision 4).
4. **MiniMax H3 stills parity:** the options layer and loop hooks, `H3_*` keys with migration from `MINIMAX_*`, the
   two commits on `wip/h3-stills`, adapter choice and ramp, the Slider preset.
5. **Sliders and full fine-tune for every family** (`FAMILY_SLIDER`, `FAMILY_FT` and their keys; `families/ft.py`).
   Fine-tune refuses pre-quantised sources upstream (verified in `families/ft.py:142-148`).
6. **MiniMax H3 clips, audio and voice, RefMods, HQQ.** Several weeks. Needs a media-aware dataset layer.
7. **Workbench tools,** starting with face-aware tools and the face likeness compare, then LoRA comparison, repair
   and the rest.

Packages Fizgig requires that TagScribeR does not install yet: `einops`, `hqq`, `insightface`, `imageio-ffmpeg`. Add
each to `requirements.txt` and the installer in the stage that first needs it.

## Also on the list

- **Speed investigation (asked for by the owner):** ways to make captioning and training faster without giving up
  much or any quality. Bring findings and measurements before changing any default. Leads from the first real runs:
  PyTorch's allocator held 17.5 GB reserved with 12.9 GB in use during Krea 2 training (cache growth across bucket
  shapes); previews at 1024 x 1024 cost about 2.5 minutes per epoch; INT8 was not faster than fp8 in a bare matmul
  benchmark on the AMD card, though it is the measured fast path on NVIDIA.
- **Settled, so later sessions do not ask again** (the owner, 2026-10-08: decide by the standing rules - keep what
  works, mirror Fizgig where it fixes something - record the call here, and stop re-asking; answers given in local
  sessions are not visible to cloud sessions, so record every decision in this file):
  - Branches: one work branch (`port/stage1-bt67en`) for all stages; the owner pulls, tests and merges to `main`.
  - Krea 2: the owner's last real runs were good (sample artifacts gone, trains well). They predate stage 1's
    numeric changes (bf16 loss order, bf16 INT8 scales, one caption per forward), so the next ordinary Krea 2 run on
    this branch is the one that confirms them; no special test run is asked for.
  - Kept as TagScribeR's (works today; changing it would only move pixels or re-cache for no gain): CPU fp32 preview
    start noise and round-up sizes (Krea 2, Klein); INT8 + maximum swap when even that is short; 28 RAW preview steps
    without the Turbo LoRA; fp8's inferred 1 MP memory point until measured; Klein's INT8 text encoder on small cards;
    the `text_embed` / `klein9b` and `krea2` cache names; Klein's extra timestep modes, Preserve Distribution and
    Attention Mechanism; one t per image at batch size above 1; SDXL's 225-token captions and the empty caption
    encoded as text (as ComfyUI / A1111; Fizgig uses zeros); the H3 time-embedding grid asset; a Turbo LoRA that matches nothing is a warning, the run continues.
  - Mirrored from Fizgig (a fix): the fused, once-rounded add for frozen adapters (training/lora.py).
  - MiniMax H3 (stage 4): Fizgig's options are TagScribeR parameters with Fizgig's `H3_*` keys and labels (its
    presets import unchanged), not a port of its FamilyOption layer, which TagScribeR's parameter system already
    covers. Kept as TagScribeR's: the LoRA holds only the trained window's modules (Fizgig writes zero adapters for
    blocks 0-19; training is identical, the file smaller), the `minimaxh3` file-name suffix (Fizgig `mmh3`), the
    run-seeded silence-row noise, travel-noise previews, res_multistep for Turbo previews (what Fizgig's sampler
    actually runs). Mirrored: bf16 trainable adapters, weight decay 1e-4, 768 previews, the clamped noise-band dial.
  - The H3 tokenizer: Fizgig bundles Qwen/Qwen3-VL-4B-Instruct's files unmodified (its `assets/qwen3vl_tokenizer/
    ATTRIBUTION.md`), the repo TagScribeR's helper files name - the audit's open question is closed.
  - Krea 2 extras beyond the audit: port them as their stages come.
- **Orchestration (the owner, 2026-10-19):** see `CLAUDE.md`. The lead plans, integrates and checks; Sonnet for narrow
  execution; Fable for audits, design, improvements, objective review before commits, and bug / exploit hunts.
- **App features asked for by the owner:** camera EXIF presets in the Metadata tab (done 2026-10-19; the exact EXIF
  strings of the presets marked "verify" in `core/camera_presets.py` still need checking against real camera files);
  the mouse wheel never changes a setting it scrolls past (done 2026-10-19); failed-image outline plus selection
  highlight (in the fixes list).
- **Improvements** (`IMPROVEMENT_IDEAS.md`, Fable, 2026-10-19). First three: training diagnostics that measure the
  owner's three speed leads on the real card (preview phases, allocator, a matmul probe); the health scan and
  thumbnails (10-20x and up to 7x measured on CPU, no training numerics involved); the trimmed Krea 2 text cache with
  a CPU-side gather (bit-identical DiT input, 7x smaller). Anything that could change what a run learns is opt-in or
  proven bit-identical; defaults change only after the owner's measurements.
- **README:** remove the roadmap line about a Fizgig tracker (it is a development aid, not a user feature), and keep
  the "what has actually been run" section true as stages land. The README roadmap also predates stage 2 (it still
  lists the Klein Distilled previews and H3 Turbo previews as to do): refresh it with this plan.
- **Longer-term roadmap** (after the port, see the README): image cleanup and background removal tools, bucket
  preview and a fit check before a run, dataset balance and caption consistency reports, a guided first run.

## Documentation ledger

Every user-visible change updates, in the same commit: tooltips, the Help Center (`tabs/help_content.py`), and where
it applies `README.md` (features, roadmap, "what has actually been run"), `docs/TRAINING_PLAN.md` (training status
per family), `docs/ARCHITECTURE.md` (new modules or layers) and `THIRD_PARTY_NOTICES.md` (ported files). Dev notes
(this file, `docs/dev/audit/`) record decisions and status. Open documentation debts:

- [ ] README: What's new and the roadmap for stages 2-5, the v8 catch-up, camera presets, the wheel fix.
- [ ] README "what has actually been run": still only Krea 2 on real hardware.
- [ ] `docs/TRAINING_PLAN.md`: sliders, the v8 catch-up per family.
- [ ] `docs/ARCHITECTURE.md`: sliders in the loop, `core/camera_presets.py`, the wheel guard.
- [ ] `THIRD_PARTY_NOTICES.md`: entries for each new family as it is ported.
