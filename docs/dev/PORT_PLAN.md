# Port plan: bringing TagScribeR's training level with current Fizgig

Status on 2026-10-08. Development notes, not user documentation. Read `CLAUDE.md` first for the standing rules.

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
- Branch `wip/h3-stills` holds two unreviewed commits for stage 4 (the H3 "Medium to High Noise LR" dial and H3
  Turbo-LoRA previews). The audit says both still match upstream with changes: re-key to `H3_HIGHNOISE_LR_PCT`, clamp
  instead of raising, `FAMILY_TURBO_STEPS` / `FAMILY_TURBO_PACE`, and the general frozen-file AdaLN hook.

## Stage 1 status (2026-10-08)

Done on branch `port/stage1-bt67en`, unit-tested only (tiny random models, CPU; no GPU, no real weights, inductor not
run): Krea 2 memory figures and `auto_precisions`; the loop fixes; cache-first tokenizer loading; Slider / Fine-tune
presets refused; LoRA reader (`alias_flat`, LoHa, `lora.down`, `unet.`); loss-watch exclusions per family; Krea 2
numerics of decision 2 (bf16 INT8 scales, bf16 loss order under autocast, one caption per forward with a text-cache
revision, clip signal Klein only, RAW CFG 4.5); driver hooks (`batch_cond`, `step_policy`, `after_optimizer_step`,
`run_metadata`, `frozen_file_added`, `park_for`, `save_preview`, `plan_run`, `load_planned`, INT8 `store_device`);
Qwen Fast Identity Mode and torch.compile with compile shared by every family (`training/compile.py`, decided on the
empty card). **Needs one real Krea 2 run by the owner** (short, 0.25 MP, sample-at-first) before it is called
verified.

Left for later stages on purpose: the Krea 2 and Qwen Slider presets (sliders are stage 5); hooks of features not
ported yet (sliders, fine-tune, clips, preview checkpoints, legacy state order, `cache_stage`).

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

Each stage lives on its own work branch, passes the full suite, and is pushed for the owner to pull and test. Nothing
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
- **Open questions from stage 1** (ask, do not decide): (a) Krea 2 preview start noise - Fizgig draws it on the GPU in
  bf16 and runs previews under autocast, TagScribeR on the CPU in fp32 (same seed, different picture); mirror it?
  (b) Auto when even maximum swap is short: TagScribeR keeps INT8 + maximum swap (about 5 GB on Krea 2), Fizgig falls
  to NF4 (11.4 GB) - keep TagScribeR's? (c) the fused bf16 add for frozen adapters (Fizgig families/lora.py:127-133:
  Turbo / context LoRA previews round once, as the old loaders) - port it? It changes preview pixels slightly. (d)
  Fizgig's no-Turbo Krea 2 preview default is 8 steps / CFG 1; TagScribeR keeps 28 steps (now CFG 4.5) - keep? (e)
  fp8's 1 MP memory point is inferred (INT8's +2.9 GB carried over, as Fizgig does for bf16) - fine until measured?
- **Open questions for the owner** raised by the audits (ask, do not decide): unlimited prompt chunks versus the
  fixed 225 tokens for SDXL, and zeros for an empty caption; whether the 5.5 MB H3 time-embedding grid asset stays
  in the repo; whether a bad Turbo LoRA file should be a warning instead of a stopped run; which Krea 2 extras
  outside the audit's scope are wanted.
- **README:** remove the roadmap line about a Fizgig tracker (it is a development aid, not a user feature), and keep
  the "what has actually been run" section true as stages land.
- **Longer-term roadmap** (after the port, see the README): image cleanup and background removal tools, bucket
  preview and a fit check before a run, dataset balance and caption consistency reports, a guided first run.
