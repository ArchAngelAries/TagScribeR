# Audit 03 - FLUX.2 Klein and MiniMax H3: TagScribeR vs current Fizgig (v7.0.1)

Read-only audit. Upstream = `.claude/fizgig-src/` (paths below prefixed `FZ:`). TagScribeR paths prefixed `TS:`.
Nothing was run on the GPU; no tests were run; no file other than this one was written.

Status: complete. Part A = Klein, Part B = MiniMax H3, Part C = the four unmerged commits, Part D = what was not
checked. Everything is from reading source; no code was executed.

## 0. Headline: the upstream shape changed

- Upstream deleted the standalone Klein trainer (`src/fizgig/training/trainer.py`, `scripts/train.py`,
  `networks/lora_klein.py`) and the standalone H3 trainer (`minimax/trainer.py`, `scripts/minimax_train.py`) on
  4 Oct 2026 (`FZ:src/fizgig/families/klein.py:1-5`, `FZ:src/fizgig/families/minimax.py:1-5`).
- Both families are now a `FamilyDescription` + a `FamilyDriver` run by the shared `families/cache.py`,
  `families/train.py` and `families/launch.py`. `FZ:src/fizgig/training/` now holds only `adaptive_lr.py`,
  `automagic3.py`, `ema.py`, `loss_logger.py`, `metadata.py`, `optimizers.py`, `progress.py`, `train_utils.py`.
- TagScribeR's two descriptions say "Fizgig has no FamilyDescription for Klein / MiniMax H3"
  (`TS:training/families/klein/description.py:6`, `TS:training/families/minimax_h3/description.py:5`). That is no longer
  true: the upstream descriptions are now the thing to mirror, field by field.
- The upstream `FamilyDescription` gained many fields TagScribeR's `TS:training/description.py` does not have
  (`FZ:src/fizgig/families/description.py:222-460`): `media`, `clip_spec` (`ClipSpec`), `options` (`FamilyOption`),
  `train_areas`, `identity_blocks`, `block_categories`, `extract_presets`, `auto_precisions`, `precision_labels`,
  `precision_hint`, `compiles` + `compile_*`, `finetune`, `ft_learning_rate`, `train_preview_checkpoint`,
  `preview_checkpoint_sampling`, `reference_strength`, `slider_training`, `slider_guidance`, `slider_ultra_blocks`,
  `adaptive_lr`, `adaptive_lr_clip_signal`, `loss_watch`, `optimizer_weight_decay`, `optimizer_eps_floor_8bit`,
  `trainable_dtype`, `multi_concept`, `samples_turbo_pace`, `preview_park_optimizer`, `settings_aliases`,
  `resumes_untagged_states`, `ema_short_run`, `LoRAFormat.lokr_kohya_stems`. TagScribeR instead has its own
  `family_options` tuple + `params.DRIVER_OPTIONS` (`TS:training/description.py:142-144`, `TS:training/params.py:329-337`)
  and `auto_order` / `auto_swap_order` (`TS:training/description.py:136-139`).

---

# PART A - FLUX.2 Klein 9B

Upstream files: `FZ:src/fizgig/families/klein.py` (208 lines, the description), `FZ:src/fizgig/klein/driver.py` (417),
`klein/model.py`, `klein/model_utils.py`, `klein/embedder.py`, `klein/position.py`.
TagScribeR files: `TS:training/families/klein/{description,driver,sampling,model,vae,embedder}.py`.

## A1. Training objective and timestep sampling

Same objective, with these exact differences.

| Item | Upstream | TagScribeR | Verdict |
|---|---|---|---|
| Flow target | `noisy = (1 - t) x0 + t noise`, `target = noise - x0`, unweighted `F.mse_loss` (`FZ:klein/driver.py:204-219`) | same (`TS:klein/driver.py:181-190`) | identical |
| Timestep mode | ONE mode only: flux2_shift, sigmoid scale 1: `shift = exp(mu(h*w))`, `t = sigmoid(randn)`, `t = t*shift / (1 + (shift-1) t)` (`FZ:klein/driver.py:160-166`), `mu` linear through (256, 0.5) and (4096, 1.15) (`:31-34`) | 8 modes (`sigma, uniform, sigmoid, shift, flux_shift, flux2_shift, logsnr, qinglong_flux`), default `flux2_shift` with the same formula (`TS:klein/sampling.py:18, 56-98`) | default identical; 7 extra modes are no longer upstream (see A9) |
| Timestep window | always rescale: `t * (max - min) + min` (`FZ:klein/driver.py:166`) | rescale, or rejection sampling with `PRESERVE_DISTRIBUTION` (`TS:klein/sampling.py:122-138`) | default identical; "preserve" is no longer upstream |
| Value told to the DiT | `(t * 1000 + 1) / 1000` (`FZ:klein/driver.py:195`) | `t + 0.001` (`TS:klein/sampling.py:139`) | identical |
| Draws per batch | noise first, then ONE `t` for the whole batch: `torch.randn(1, generator=...)` (`FZ:klein/driver.py:164, 224-225`) | noise first, then `bsz` separate `t` values (`TS:klein/sampling.py:58-59, 78`; `TS:klein/driver.py:175-179`) | **differs at batch size > 1** (same at 1) |
| Input dtype | tokens, text and guidance passed as float32 under bf16 autocast (`FZ:klein/driver.py:187-197`) | noised latents and text cast to bf16 before packing (`TS:klein/driver.py:183-184`) | small numeric difference (rounding before `img_in` instead of inside it); not checked on real weights |
| Guidance | vector of 1.0 (`FZ:klein/driver.py:194`) | `guidance=None` (`TS:klein/driver.py:189`) | no effect: `Klein9BParams.use_guidance_embed = False` (`FZ:klein/model.py:100, 676`) |
| Edit / reference training | `supports_references = True`; reference latents packed after the image tokens with `pack_control_latent` (`FZ:klein/driver.py:150, 168-174, 191-193`); `edit_training=True` (`FZ:families/klein.py:167`) | refused: `raise RuntimeError("Klein has no edit / reference training in this port")` (`TS:klein/driver.py:170-171`) | **missing** |
| Slider loss | `diff_ref` / `diff_weight` per-token weighting (`FZ:klein/driver.py:209-218`), `noise_latents` + `predict` (`:229-237`); `slider_training=True` (`FZ:families/klein.py:171`) | none | **missing** |
| Gradient accumulation, grad clip | shared loop (`FZ:families/train.py:1309-1330`) | shared loop | same mechanism (shared-layer audit, not re-checked here) |

## A2. Auto precision / memory plan

Upstream (`FZ:families/klein.py:130-143`, planner `FZ:families/quant.py:129-172`, `FZ:families/train.py:853-896`):

- `precisions=("bf16", "int8", "nf4")`, `auto_precisions=("int8", "nf4")`, label override
  `precision_labels={"bf16": "As the file (bf16 or fp8)"}`.
- There is **no "fp8" precision** any more. The fp8 Base file is loaded by choosing "bf16" ("As the file"): the loader keeps
  fp8 weights resident with their scales (`FZ:klein/model_utils.py:244-245, 281-313`).
- `train_memory={"int8": (((0.25, 12.6), (1.0, 17.1)), 0.0), "nf4": (((0.25, 7.9), (1.0, 9.9)), 0.0)}` - measured 3 Oct
  2026 on a 5090. The swap saving is `0.0`, so **Auto never plans a block swap for Klein** (`quant.plan` `swap_for`
  returns 0 when `per_block <= 0`, `FZ:families/quant.py:147-152`).
- Auto order: INT8 if 12.6-17.1 GB fits (budget = free - 1.5 GB), else NF4. Then, if the run is NOT compiled and the file
  is an fp8 file, Auto switches INT8 -> "bf16" (file as it is): `auto_uncompiled_precision`
  (`FZ:klein/driver.py:59-64`, applied at `FZ:families/train.py:888-893`). Reason given: uncompiled, the fp8 file is
  0.77 s/step vs 1.14 for INT8 at 11.1 vs 12.8 GB.
- torch.compile: `compiles=True`, `compile_fullgraph=False`, `compile_boundary="outside"`,
  `compile_payback_steps={"int8": 200, "nf4": 400}`, `compile_memory={"int8": {"inside": ((0.25, 20.7),), "outside":
  ((0.25, 11.8), (1.0, 15.7))}, "nf4": {"inside": ((0.25, 7.3), (1.0, 9.3))}}` (`FZ:families/klein.py:153-158`); the
  driver compiles both block lists and forces the checkpoint inside for an fp8-resident base
  (`FZ:klein/driver.py:105-126`). Generic rule in `FZ:families/driver.py:378-427`.
- Manual block swap maximum 16 (`FZ:klein/driver.py:66-69`); NF4 never swaps (`FZ:families/quant.py:88-90`).

TagScribeR (`TS:training/families/klein/description.py:89-106`, planner `TS:training/quant.py:180-243`):

- `precisions=("bf16", "fp8", "int8", "nf4")`, `auto_order=("fp8", "nf4")`, `auto_swap_order=("fp8",)`.
- `train_memory={"fp8": (((0.5, 14.0),), 0.41), "int8": (((0.5, 14.0),), 0.41), "nf4": (((0.5, 8.5),), 0.0)}` - one point
  each, from `docs/KLEIN.md`; the INT8 figure is marked "INFERRED" and the 0.41 GB / block is "COMPUTED" (`:98-105`).
- Auto: fp8 if 14.0 GB fits, else NF4 if 8.5 fits, else fp8 with block swap.
- No torch.compile for Klein (`COMPILE_BLOCKS` is "Krea 2 only", `TS:training/params.py:159-167`).

Differences:

1. Auto order: upstream INT8 -> NF4 (-> file-as-is when uncompiled on an fp8 file); TagScribeR fp8 -> NF4 -> fp8 + swap.
2. Upstream's figures are measured at two resolutions per precision; TagScribeR's are single-point, partly inferred.
3. Upstream Auto never swaps on Klein; TagScribeR Auto swaps on fp8.
4. Upstream has no fp8 entry in the precision list; TagScribeR lists "fp8" as its own precision. (TagScribeR keeping an
   explicit fp8 choice is consistent with the owner's "fp8 checkpoints must work" rule; the upstream equivalent is the
   "As the file (bf16 or fp8)" label. Mirror the behaviour, keep the file support.)
5. torch.compile for Klein: **missing** in TagScribeR.
6. `docs/KLEIN.md` still says "Auto picks [NF4] on cards under 16 GB" and "fp8 ... Automatic" (lines 12, 57): the doc is
   older than the code. The code above is what runs.

## A3. Presets

Upstream: 7 built-ins (`FZ:families/klein.py:192-202`), built by `_preset` (`:14-24`). Every preset carries:
`NETWORK_ALPHA = rank`, `NETWORK_TYPE "LoRA (standard)"`, `SAVE_EVERY_N_EPOCHS 1`, `SEED 42`, `ADAPTIVE_LR True`,
`OPTIMIZER_TYPE "adamw8bit"`, `FAMILY_PRECISION "Auto (fits your free VRAM)"`, `BLOCKS_SWAP "Auto (detect from GPU)"`,
`FAMILY_EMA "Off"`.

| # | Upstream name | rank | LR | epochs | TARGET_LAYERS | Adaptive min / max | MIN / MAX_TIMESTEP |
|---|---|---|---|---|---|---|---|
| 1 | `✨ Old Reliable (rank 16, full model, single subject)` | 16 | 1e-4 | 55 | Full Model | 1e-4 / 4e-4 | "" / "" |
| 2 | `✨ Old Reliable - Flavour 8 (rank 8, full model, single subject)` | 8 | 1e-4 | 55 | Full Model | 1e-4 / 4e-4 | "" / "" |
| 3 | `✨ Identity (rank 8, single subject)` | **8** | 4e-4 | 15 | Identity | 2e-4 / 4e-4 | "" / "" |
| 4 | `✨ Identity (rank 8, harder dataset)` | 8 | 4e-4 | 20 | Identity | 2e-4 / 4e-4 | "" / "" |
| 5 | `✨ Multi-Character (rank 16, multi character or concept)` | 16 | 2e-4 | 50 | Identity | 1e-4 / 4e-4 | "" / "" |
| 6 | `✨ Style (late timesteps)` | 4 | 4e-4 | 15 | Style | 1e-5 / 4e-4 | "0" / "400" |
| 7 | `✨ Style+Composition (all timesteps)` | 4 | 4e-4 | 15 | Style+Composition | 1e-5 / 4e-4 | "" / "" |

TagScribeR (`TS:training/families/klein/description.py:16-26, 130-141`): same 7 in the same order, with these differences:

- **Preset 3 is stale**: TagScribeR has `✨ Identity (rank 4, single subject)` with rank 4 / alpha 4
  (`TS:...description.py:135`). Upstream renamed it and moved it to rank 8 in v6.8.2 (release notes; code
  `FZ:families/klein.py:196`). Note upstream presets 3 and 4 now differ only in epochs (15 vs 20).
- TagScribeR presets lack the four keys upstream now sets in every Klein preset: `NETWORK_TYPE`, `FAMILY_PRECISION`,
  `BLOCKS_SWAP`, `FAMILY_EMA "Off"`.
- All other values match.

## A4. GUI settings upstream exposes for Klein NOW

How to read this: for a described family the launch is built ONLY by `FZ:families/launch.py train_command` (`:436-589`)
and `_preview_flags` (`:592-691`) from the dict made by `_family_launch_inputs` (`FZ:lora_trainer_gui.py:6090-6165`).
A key that is not read there does not reach the trainer. Visibility comes from `_apply_training_arch_visibility` /
`_generic_training_visibility` / `_family_train_area_rows` (`FZ:lora_trainer_gui.py:6971-7121`) and `_generic_samples_ui`
(`:11225-11350`). "TS" column: P = present, M = missing, D = different.

Training tab:

| Upstream key | Default | Visible for Klein | Passed / read | TS |
|---|---|---|---|---|
| `LORA_NAME`, `LORA_OUTPUT_DIR` | `LoraName_TokenName_k9b` | yes | `--output_name`, `--output_dir` (`launch.py:443`) | P |
| `LEARNING_RATE` | 4e-4 | yes (greyed with Adaptive LR) | `--learning_rate` | P |
| `ADAPTIVE_LR`, `_MIN`, `_MAX` | False, 1e-5, 4e-4 | yes (`adaptive_lr` default True) | `--adaptive_lr --adaptive_lr_min/max` (`launch.py:456-459`); clip-ratio signal on (`adaptive_lr_clip_signal=True`, `klein.py:146`) | P (clip signal per the parity skill; not re-checked) |
| `NETWORK_DIM`, `NETWORK_ALPHA` | 4, 4 | yes | passed | P |
| `NETWORK_TYPE`, `LOKR_FACTOR` | LoRA, 8 | **yes** (`network_types=("lora","lokr")`, `klein.py:145`) | `--network_type lokr --lokr_factor` (`launch.py:530-532`) | **M**: TS `network_types=("lora",)` (`TS:...description.py:110`, comment "LoKR is wired for Krea 2 / H3 / Qwen only" is out of date) |
| `MAX_TRAIN_EPOCHS`, `SAVE_EVERY_N_EPOCHS`, `SEED` | 12, 1, 42 | yes | passed | P |
| `TARGET_LAYERS` (Model Area) | first area | yes; choices `Full Model, Identity, Style, Style+Composition, Details, Custom` (`klein.py:106-112`, `gui:7110`) | as `FAMILY_TRAIN_AREA` -> `--train_blocks` (`launch.py:110-117, 562-565`); not sent for a slider or fine-tune | P (same areas, `TS:klein/driver.py:37-44`) |
| `TRAINING_BLOCKS` (Custom picker) | none ticked | yes when Custom; block ids `double_0..7`, `single_0..23` (`klein/driver.py:412-417`) | `FAMILY_TRAIN_BLOCKS` -> `--train_blocks` | **D**: TS ids are `double_blocks.N` / `single_blocks.N` (`TS:klein/driver.py:94, 232-235`), and TS takes a typed text list; an upstream preset's `TRAINING_BLOCKS` dict uses `double_N` keys and would be refused by TS `resolve_blocks` |
| `MIN_TIMESTEP`, `MAX_TIMESTEP` | "", "" (0-1000 boxes) | yes (the Timestep Range section shows for a family with `train_areas`) | `--min_timestep/--max_timestep` as 0-1 (`launch.py:120-129, 566-567`) | P (0-1 in TS, legacy 0-1000 mapped, `TS:training/presets.py:184-187`) |
| `OPTIMIZER_TYPE` | adamw8bit | yes | `--optimizer_type` (`launch.py:544-545`). The dropdown lists the WHOLE installed catalog for every family (`gui:6957-6965`, `training/optimizers.py:39-52`: adamw8bit, adamw, pagedadamw8bit, ademamix8bit, pagedademamix8bit, lion8bit, automagic3). `desc.optimizers` is not read anywhere in the GUI or trainer (grep: no `.optimizers` use). | **D**: TS offers Klein `adamw8bit, adamw, ademamix8bit, pagedademamix8bit` only (`TS:...description.py:109`) and filters by it (`TS:training/params.py:349-350`); upstream now also lets Klein pick pagedadamw8bit, lion8bit and automagic3 |
| `OPTIMIZER_ARGS` | "" | yes | `--optimizer_args` | P |
| `GRADIENT_ACCUMULATION` | 1 | yes | passed when > 1 | P |
| `MAX_GRAD_NORM` | 1.0 | yes | passed when != 1 | P |
| `LR_SCHEDULER`, `LR_WARMUP_STEPS` | constant, "" | yes | only when Adaptive LR is off (`launch.py:460-468`) | P |
| `FAMILY_PRECISION` | Auto | yes; choices Auto / `As the file (bf16 or fp8)` / INT8 / 4-bit NF4 | `--precision` | **D** (see A2) |
| `BLOCKS_SWAP` | auto | yes | `--blocks_to_swap -1` or N (`launch.py:533-539`); max 16 | P (max 16 both) |
| `COMPILE_BLOCKS` | auto | **yes** (`compiles=True`) | `--compile_blocks auto/on/off/outside` | **M** for Klein |
| `FAMILY_EMA` | "Off" (`ema_default="Off"`, `klein.py:129`) | **yes** (any non-empty `ema_default` shows it, `gui:7058`) | `--ema_decay` unless Off (`launch.py:540-543`) | **M**: TS `ema_default=""` with the comment "Fizgig's Klein trainer has no EMA" (`TS:...description.py:87`) - no longer true |
| `CONTEXT_LORA_PATH`, `_STRENGTH` | "", 1.0 | yes | passed (not for a fine-tune) | P |
| `SAVE_STATE`, `SAVE_STATE_ON_TRAIN_END`, `KEEP_LAST_N_STATES` | True, True, 2 | yes | passed | P |
| Loss watch: `KREA2_LOSS_WATCH`, `KREA2_PER_IMAGE_LR`, warm-up look, `KREA2_AUTO_RECAPTION` | False | **yes** (`loss_watch` default True; Klein does not turn it off) | `--log_per_image_loss`, `--per_image_lr`, `--warmup_look_outliers`, `--auto_recaption` at batch size 1 (`launch.py:504-529`) | P in the shared layer (TS params `:120-131`); whether TS shows them for Klein was not checked |
| Kind of training: `FAMILY_EDIT` (+ `FAMILY_EDIT_DIR`, `_REF`, `_CAPTION`) | off | **yes** (`edit_training=True`) | control folder in the dataset TOML, `--sample_reference` | **M** for Klein (TS `edit_training` not set) |
| Kind of training: `FAMILY_SLIDER` (+ `_SOURCE`, `_DIR`, `_CAPTION`, `_BASE`, `_POS`, `_NEG`, `_GUIDANCE`) | off | **yes** (`slider_training=True`) | `--slider_pairs` or `--slider_prompts ... --slider_guidance --slider_bank_res` (`launch.py:568-586`) | **M** (TS has no slider training at all - grep of `training/` not run for sliders in other families; none in the Klein driver) |
| Kind of training: `FAMILY_FT` (+ `_ROTATIONS`, `_SAVE_EVERY`, `_ROTATE_EVERY`, `_REG_DIR`, `_REG_MULT`, `_MAX_PARTS`, `_FUSED`) | off | **yes** (`finetune=True`, `klein.py:133`; new in v7.0.0) | `--finetune --ft_rotations ...` (`launch.py:488-503`); driver `ft_spec` / `install_ft_streamer` (`klein/driver.py:382-397`) | **M** |
| `METADATA_*` (7 keys) | "" | yes | passed | P |
| `ATTENTION_MECHANISM` | sdpa | **shown** in Other Options (`gui:8175-8185`) | **NOT passed, NOT read**: no `--sdpa/--flash3` in `launch.py`, no such argument in `families/train.py:1510-1618`; the driver hard-codes `attn_mode="torch"` (`klein/driver.py:56`) | TS has it and passes it to the model (`TS:klein/driver.py:43, 113`) - now a TS-only live setting for something upstream made dead |
| `LOGGING_DIR`, `LOG_WITH`, `LOG_PREFIX` | "", none, "" | shown (`gui:8187-8205`) | not passed, not read | not in TS; dead upstream, nothing to port |
| `LORA_LR_RATIO` (LoRA+) | 1 | **hidden** ("hidden, always 1 ... Widget exists for preset/save compat", `gui:3895-3897`) | validated as a number (`families/checks.py:49`) but **never passed**; no LoRA+ code in `families/lora.py` or `families/train.py` | see the commit check below |
| `IMG_IN_TXT_IN_OFFLOADING` | False | no widget shown ("no-op for Klein 9B", `gui:8207-8209`) | not passed | absent in TS; fine |
| `GRADIENT_CHECKPOINTING`, `FP8_TEXT_ENCODER`, `QUANT_4BIT` | True, True, False | settings-dict defaults only (`gui:1479-1482`), no widget, no reader | not passed; checkpointing is always on in the driver path, fp8 TE is automatic (A5) | absent in TS; fine |
| `NETWORK_DROPOUT` | - | **key does not exist** upstream any more (grep: 0 hits in the GUI and in `src/`) | - | see the commit check below |
| `TIMESTEP_SAMPLING`, `DISCRETE_FLOW_SHIFT`, `SIGMOID_SCALE`, `LOGIT_MEAN`, `LOGIT_STD`, `PRESERVE_DISTRIBUTION` | - | **keys do not exist** upstream any more (0 hits, except one comment mentioning PRESERVE_DISTRIBUTION at `gui:4976`) | - | TS has all six as live Klein options (`TS:training/params.py:210-231`, `TS:...description.py:112-113`) - see A9 |

Samples tab:

| Upstream key | Default (Klein) | Visible | Passed / read | TS |
|---|---|---|---|---|
| `SAMPLE_ENABLED`, `SAMPLE_PROMPT`, `SAMPLE_EVERY_N_EPOCHS`, `SAMPLE_AT_FIRST`, `SAMPLE_SEED` | True, ..., 1, True, 1234 | yes | passed | P |
| `SAMPLE_WIDTH`, `SAMPLE_HEIGHT` | 768, 768 (`klein.py:190-191`) | yes | passed | P (768) |
| `SAMPLE_STEPS`, `SAMPLE_CFG_SCALE`, `SAMPLE_NEGATIVE` | 40, 4.5, `GENERAL_NEGATIVE` (`klein.py:185-187`) | yes; greyed while "Use Distilled" is on (note "Base samples only - Distilled is locked at 4 steps", `gui:11321-11322`) | passed, but with a preview checkpoint the Distilled recipe replaces them (`families/train.py:1241-1244`) | P for Base (40 / 4.5); Distilled M |
| "Use Distilled model for samples" (`use_distilled_samples_var`) | **True** | yes (`gui:10652-10658, 11231-11233`) | `--preview_checkpoint <distilled_dit> --preview_checkpoint_cache auto/on/off [--preview_int8]` (`launch.py:682-690`) | **M** |
| `CACHE_SAMPLE_MODEL` | auto | yes | `--preview_checkpoint_cache` | **M** |
| Reference image (`sample_ref_image`) | "" | yes | `--sample_reference` (Klein's `reference_kind` is "edit" via `edit_training`; `launch.py:669-672`) | **M** |
| `SAMPLE_EVERY_N_STEPS` | 0 | widget exists (`gui:10639-10640`) | **not passed** (no flag in `launch.py`) | not in TS; dead upstream |
| `SAMPLE_FLOW_SHIFT` | "" | widget exists (`gui:10699`) | **not passed** (the driver supports `("flow_shift", s)` but nothing sends it) | not in TS; dead upstream |
| `FAMILY_TURBO_STRENGTH` | - | hidden for Klein (no speed LoRA) | - | n/a |

So the earlier finding ("several settings existed but did nothing") is now: the old Klein-only timestep knobs and network
dropout are **gone**; LoRA+ ratio, attention mechanism, logging, flow shift and every-N-steps sampling **still exist as
widgets or keys and do nothing**.

## A5. Text encoder, VAE, cache format and ids

| Item | Upstream | TagScribeR | Verdict |
|---|---|---|---|
| Text encoder model | Qwen3-8B, chat template with `enable_thinking=False`, `max_length=512` padded, hidden states of layers 9 / 18 / 27 through the final RMSNorm, concatenated to (512, 12288) (`FZ:klein/embedder.py:23, 80-121`) | same recipe per the file header (`TS:klein/embedder.py:27, 130-150`) - body not diffed line by line | same design |
| TE low-VRAM path | fp8 weights (`torch.float8_e4m3fn`) when free VRAM < 19.5 GB, else bf16 (`FZ:klein/driver.py:86-92`), with patched RMSNorm / Embedding (`FZ:klein/embedder.py:178-193`) | **INT8** language-model weights below the same 19.5 GB (`TS:klein/driver.py:33, 134-140`; header: "Fizgig's fp8 text-encoder option ... is replaced by weight-only INT8", `TS:klein/embedder.py:5`) | **different** (a known TS choice; the parity skill lists "the fp8 text encoder" as something that was dropped without asking on Krea 2 - same question applies here) |
| Text cache tensor | key `cond__ctx_vec`, shape (512, 12288), dtype as produced under bf16 autocast (`FZ:klein/driver.py:142-148`, `FZ:families/cache.py:68-75`) | key `cond__text_embed`, bf16 (`TS:klein/driver.py:158-160`) | **different key name** (`ctx_vec` vs `text_embed`) |
| VAE | FLUX.2 AE in float32, pixels / 127.5 - 1, `vae.encode` -> (128, H/16, W/16) (`FZ:klein/driver.py:82-84, 135-140`) | same (`TS:klein/driver.py:149-155`) | identical recipe |
| Latent cache key | `latent_{h}x{w}`, metadata `architecture`, `width`, `height`, `dtype`, `format_version "1.0.0"` (`FZ:families/cache.py:45-61`) | same keys plus `latent_rev` (`TS:training/cache.py:42-51`) | same layout, TS adds a field |
| Cache arch id | `klein9bdrv` - deliberately new so the driver's cache never mixes with the old trainer's (`FZ:families/klein.py:3-4, 31`) | `klein9b` (`TS:...description.py:31`) | **different id**. TS caches are TS's own format, so nothing breaks, but the ids no longer match upstream |
| Family key | `klein` (`FZ:families/klein.py:30`) | `klein9b`, alias `klein` (`TS:...description.py:30, 35`) | different key, alias covers it |
| Model files | 4 rows: `base_dit` (fp8 recommended, bf16 alt), `distilled_dit` (role `preview_dit`, required=False but `fetch_optional=False`), `vae`, `text_encoder` (`FZ:families/klein.py:38-61`); fetch list the same four + the caption TE (`FZ:scripts/fetch_models.py:104-117`) | 3 rows, no Distilled DiT (`TS:...description.py:38-53`) | **Distilled DiT row missing** |
| Tokenizer helper | `hf-config:Qwen/Qwen3-8B` (`FZ:scripts/fetch_models.py:172`) | `helper_files` Qwen/Qwen3-8B (`TS:...description.py:144-145`) | present |

## A6. LoRA targets, key format, network types, saved metadata

- Targets: identical. 8 double blocks x 8 Linears + 24 single blocks x 2 Linears = 112
  (`FZ:klein/driver.py:25-28, 406-417`; `TS:klein/driver.py:27-30, 223-236`).
- Key format: identical kohya keys `lora_unet_double_blocks_{n}_{module}` / `lora_unet_single_blocks_{n}_linear1|2`,
  `.lora_down.weight / .lora_up.weight / .alpha` (`FZ:families/klein.py:114-124`; `TS:...description.py:69-83`).
- Block ids: upstream `double_N` / `single_N` (kept from the old Repair Studio); TagScribeR `double_blocks.N` /
  `single_blocks.N`. This matters for `--train_blocks`, the `ss_train_blocks` metadata and Custom presets.
- Network types: upstream `("lora", "lokr")`; LoKR is written as LyCORIS `diffusion_model.<module path>` keys
  (`LoRAFormat.lokr_kohya_stems` default False, `FZ:families/description.py:102-105`). TagScribeR: LoRA only for Klein
  (the shared layer already writes LoKR the same way, `TS:training/lora.py:240-246`).
- Reading other trainers' Klein LoRAs: upstream `convert_lora_state_dict` -> `ensure_kohya_lora_state_dict` (kohya,
  OneTrainer `lora_transformer_`, PEFT, diffusers Flux with split q/k/v fused, `FZ:klein/driver.py:400-404`). TagScribeR
  driver has no such hook (the shared `read_file` handles prefixes only) - a context LoRA in diffusers-Flux layout would
  not load. **Missing.**
- Metadata: upstream `ss_network_module = "fizgig.families (klein, lora)"`, `ss_network_dim`, `ss_network_alpha`,
  `ss_lokr_factor`, `ss_architecture`, `ss_epoch`, `ss_optimizer`, `ss_learning_rate`, `ss_training_adapter`,
  `ss_context_lora(+_strength)`, `ss_slider*`, `ss_train_blocks` (the comma-joined block ids when a Model Area is set),
  plus `driver.run_metadata()` (`FZ:families/train.py:1147-1178`). TagScribeR writes the same generic set with
  `ss_network_module = "tagscriber.training (klein9b, lora)"` and **no `ss_train_blocks`** on main
  (`TS:training/train.py:494-514`). `modelspec.architecture` `Flux.2-klein-9b` and implementation URL match
  (`FZ:families/klein.py:127-128`; `TS:...description.py:86-88`).

## A7. Preview engines

| Engine | Upstream | TagScribeR |
|---|---|---|
| Base | Euler on the empirical-mu schedule, CFG with a negative prompt when cfg > 1; defaults 40 steps / CFG 4.5 / 768x768 (`FZ:klein/driver.py:255-304`, `FZ:families/klein.py:179-191`) | same (`TS:klein/sampling.py:147-208`, `TS:...description.py:115-128`) - present |
| Distilled 4-step (default ON) | separate checkpoint `distilled_dit`, 4 steps, CFG 1, `("schedule","simple")`, `("shift", 2.02)`, `("guidance", 1.0)` (`FZ:families/klein.py:172-177`), ComfyUI's simple schedule `get_simple_euler_schedule` (`FZ:klein/model_utils.py:105-126`); memory handoff `park_for_preview` (max swap 6 double + 22 single on the training DiT), `load_preview_checkpoint` (swap by card: >= 23 GB none, 15-22 GB 16, < 15 GB 6 + 22; optional INT8), `unpark_after_preview` (`FZ:klein/driver.py:315-379`); runner at `FZ:families/train.py:476-553, 1255-1263` | **missing** - `TS:...description.py:122-124` says "NOT PORTED" |
| Reference image | Samples-tab picture -> `--sample_reference`; reference latents after the image tokens in every step (`FZ:klein/driver.py:276-278, 293-294`), resized by the trainer (`_load_references`, `FZ:families/train.py:915-916`) | **missing** (`generate` accepts `refs` but ignores it, `TS:klein/driver.py:204-211`) |
| Edit-LoRA previews | the edit instruction on the test photo (`FZ:families/launch.py:619-624, 673-681`) | missing |
| Slider previews | -1 / 0 / +1 strip (`FZ:families/train.py` `_slider_strip`) | missing |
| Noise | drawn on the GPU when available: `torch.Generator(device=dev)`, bf16 (`FZ:klein/driver.py:244-249`) | CPU generator, float32 (`TS:klein/sampling.py:177-180`) - **different noise for the same seed on a GPU machine**, so previews will not match Fizgig's seed for seed |
| Size rounding | floor to 16, min 16 (`FZ:klein/driver.py:240-242`) | round UP to 16 (`TS:klein/sampling.py:170-174`) - different for sizes off the grid |

## A8. Other Klein capabilities upstream that TagScribeR lacks

- Full fine-tune (`finetune=True`; `ft_spec` windows `img_attn, img_mlp, txt_attn, txt_mlp, linear1, linear2`,
  `FZ:klein/driver.py:382-397`; shared runner `FZ:families/ft.py`, 775 lines). New for Klein in v7.0.0.
- Workbench: `workbench=("repair", "explorer", "profiler", "extract", "royale")`, `block_categories`, `extract_presets`,
  `category_masters`, `int8_attention`, `activation_cache` (`FZ:families/klein.py:90-102, 131-132`). The memory file says
  the workbench is "left for later" by the owner's approval, so this is listed, not counted as a gap.

## A9. What TagScribeR has for Klein that upstream removed or changed

1. Timestep sampling modes `sigma, uniform, sigmoid, shift, flux_shift, logsnr, qinglong_flux` and their knobs
   `DISCRETE_FLOW_SHIFT`, `SIGMOID_SCALE`, `LOGIT_MEAN`, `LOGIT_STD`, `PRESERVE_DISTRIBUTION`
   (`TS:klein/sampling.py:18-139`, `TS:training/params.py:210-231`). Upstream: removed with the old trainer; only
   flux2_shift at scale 1 remains.
2. `ATTENTION_MECHANISM` as a live setting (`TS:klein/driver.py:42-43, 61, 113`); upstream: dead widget, driver fixed to
   `"torch"`. flash3 never worked upstream either (the TS note at `TS:...description.py:159-162` says so).
3. Optimizer list restricted to 4 (`TS:...description.py:109`); upstream: unrestricted catalog.
4. An explicit "fp8" precision and fp8-first Auto with block swap; upstream: INT8-first, fp8 only as "As the file".
5. Preset 3 at rank 4 (old name).
6. Block ids `double_blocks.N` (old `TRAINING_BLOCKS` dict keys); upstream `double_N`.
7. Notes text "Not part of this port: ... the fp8 base ..." (`TS:...description.py:163-165`) contradicts line 148 and the
   code (the fp8 base IS supported in TS).

Whether items 1-2 should be removed to mirror upstream, or kept as TagScribeR extensions, is an owner question (the
rule says mirror; these were ported from Fizgig as it was on 29 Sept and Fizgig has since dropped them).

## A10. TagScribeR-only fix to preserve: wide single-head VAE attention

- Upstream's Klein AE attention is **still a plain single-head call**:
  `h_ = nn.functional.scaled_dot_product_attention(q, k, v)` on `(b, 1, h*w, c)` tensors, `c = in_channels`
  (`FZ:klein/model.py:121-146`, the call at line 143). Upstream did not touch this code path.
- TagScribeR routes it through `training/modules/wide_attention.py` (`TS:klein/vae.py:20, 84-85`). Keep it; nothing
  upstream conflicts.

## A11. Port plan for Klein (dependency order)

Sizes: S = under a day of agent work, M = 1-3 days, L = more.

1. **Description refresh (S)**: preset 3 -> rank 8 + new name; add `NETWORK_TYPE / FAMILY_PRECISION / BLOCKS_SWAP /
   FAMILY_EMA` to the presets; `network_types=("lora","lokr")`; `ema_default="Off"`; add the Distilled DiT model file
   (`preview_dit`); fix the stale notes. Decide block ids (`double_N`) and arch id with the owner.
2. **Auto plan (S-M)**: upstream's measured `train_memory`, `auto_precisions=("int8","nf4")`, the
   `auto_uncompiled_precision` rule for fp8 files, no Auto swap. Needs the shared layer to grow `auto_precisions` /
   `precision_labels` (or map them onto TS's `auto_order`). Keep explicit fp8-file support.
3. **Driver alignment (S)**: one `t` per batch; `ctx_vec` cache key (or keep TS's and record it); GPU-side preview
   noise and floor-to-16 if seed parity with Fizgig matters; `convert_lora_state_dict`; `ss_train_blocks`.
4. **Optimizer list (S)**: open the Klein list to the catalog, as upstream.
5. **Reference previews + Edit LoRA (M)**: `position.pack_control_latent`, `supports_references`,
   `encode_text_with_references`, refs in `training_loss` / `generate`, the Samples-tab reference row, Edit card. The
   shared TS layer already has edit plumbing for Qwen (`FAMILY_EDIT*` params exist).
6. **Distilled previews (M)**: `get_simple_euler_schedule`, the `("schedule","simple")` branch in `generate`,
   `park_for_preview / load_preview_checkpoint / unpark_after_preview`, the preview-checkpoint runner in the shared
   loop, `CACHE_SAMPLE_MODEL`, the "Use Distilled" tick (default on), explicit per-type block swap
   (`double_blocks_to_swap` / `single_blocks_to_swap`, which TS `enable_block_swap` lacks, `TS:klein/model.py:178`).
7. **torch.compile for Klein (M)**: `compile_blocks` for both block lists, the fp8 "inside" rule, payback and memory
   tables. Depends on TS's shared compile support (exists for Krea 2).
8. **Sliders (M-L)**: shared-layer feature (`--slider_pairs`, `--slider_prompts`, bank renders, strip previews) plus the
   driver's `diff_ref` loss, `noise_latents`, `predict`. Shared with Krea 2 / Qwen / H3.
9. **Full fine-tune (L)**: `families/ft.py` runner + Klein `ft_spec` / `install_ft_streamer`. Shared with every family.
10. **Owner decision**: remove or keep the old timestep modes, preserve-distribution and attention mechanism (A9).

---

# PART B - MiniMax H3

Upstream files: `FZ:src/fizgig/families/minimax.py` (396, description + `OPTIONS` + `PRESETS`),
`FZ:src/fizgig/minimax/driver.py` (972), `minimax/common.py` (1619: loss, sigma sampler, VRAM planners, preview
helpers, AdaLN injection, `AdapterRamp`, distillation loss), plus the modules listed in B8.
TagScribeR files: `TS:training/families/minimax_h3/{description,driver,sampling,model,vae,embedder,weights}.py`
(still images only, stated at `TS:training/families/minimax_h3/__init__.py:8-12`).

## B1. Training objective and timestep sampling

| Item | Upstream | TagScribeR | Verdict |
|---|---|---|---|
| Flow target (still) | `noised = (1 - s) x0 + s noise`, `t = 1 - sigma`, target `x0 - noise`, `F.mse_loss` (`FZ:minimax/common.py:978-993`) | same (`TS:minimax_h3/driver.py:240-248`) | identical |
| Sigma density | `sigma = shift*u / (1 + (shift-1) u)`, `u ~ U(0,1)` (`FZ:minimax/common.py:923-926`); shift from `lownoise_pct=P` as `(1-P)/P` (`FZ:minimax/driver.py:395-403`); with no option at all the shift is 12 (`VIDEO_SIGMA_SHIFT_TRAIN`, `common.py:19, 906-907`) | same formula (`TS:minimax_h3/sampling.py:27-46`), default P = 60 (`TS:minimax_h3/driver.py:33`) | identical for the GUI default (Training Structure "Likeness and Style - 60% clean-end" sends `lownoise_pct=60`, `FZ:families/minimax.py:15-17`) |
| Other densities | `shift="sigmoid"`, `"resolution"`, `"lognorm:<s>"` exist in `sample_sigmas` (`common.py:908-922`) but no GUI option sends them (CLI `--family_option shift=` only) | not present | CLI-only upstream; minor gap |
| Draw order | sigma first, then video noise (`FZ:minimax/driver.py:427-430`) | noise first, then sigma, then 4 rows of audio noise (`TS:minimax_h3/driver.py:240-246`) | **different RNG order**: same statistics, different numbers for the same seed |
| Silence-row noise (still) | drawn inside the model from the global RNG when no `audio_noise` is passed (`FZ:minimax/model.py:863-866`; the driver passes none, `common.py:987`) | drawn from the run's seeded generator and passed in (`TS:minimax_h3/driver.py:246-247`) | TS is more reproducible; not bit-equal |
| Timestep window | `sigma = min_t + (max_t - min_t) * sigma` (`FZ:minimax/driver.py:428-429`); the GUI never sends it for H3 (no `train_areas`, `FZ:families/launch.py:566-567`) | ignored with a warning (`TS:minimax_h3/driver.py:225-228`) | CLI-only upstream; minor gap |
| High-noise LR | `info["lr_mult"] = hn if sigma >= 0.5 else 1.0`; not for voice items (`FZ:minimax/driver.py:441-448`); applied as the window mean on the optimizer LR, skipped for `Automagic3` (`FZ:families/train.py:1313-1324, 1403-1404`) | **missing on main** (commit `3feb963`, see part C) | missing |
| Per-modality block routing | LoRA is built on all 50 blocks; per step the out-of-window blocks' adapters get `requires_grad_(False)` so the backward stops at the window (`step_frozen_blocks`, `FZ:minimax/driver.py:252-266`; `FZ:families/train.py:1284-1300, 1400`). Default mode = `photo_blocks=20-49 clip_blocks=20-49 audio_blocks=20-49` (`FZ:families/minimax.py:80`) | adapters are only created for the window's blocks (`lora_target_names`, `TS:minimax_h3/driver.py:295-299`) | same training for stills; **different saved file**: upstream writes all 200 modules (blocks 0-19 with zero up-weights, `FZ:families/lora.py:551-566`), TS writes 120 |
| Caption dropout | option `caption_dropout` (GUI default 0.05), Python `random.random()` per step, the cached `uncond_minimaxh3_te.safetensors` embed (`FZ:minimax/driver.py:230-241, 380-385`) | generic `CAPTION_DROPOUT` extension, default 0.0, set to 0.05 only through the preset key `MINIMAX_CAPTION_DROPOUT` (`TS:training/params.py:272-274`, `TS:training/presets.py:121`) | present, different default path |
| Clip steps (T > 1), audio loss on its own shift-3 schedule, voice items (video term 0) | `compute_loss` (`FZ:minimax/common.py:995-1024`), `batch_cond` (`FZ:minimax/driver.py:380-393`) | not present (model is T = 1 only, `TS:minimax_h3/model.py:468-500`) | **missing** |
| TREAD, adapter ramp, retirement, distillation, sliders, fine-tune | see B8 | not present | **missing** |
| Trainable adapter dtype | bf16 (`trainable_dtype="bf16"`, `FZ:families/minimax.py:368`) | fp32 (`TS:training/lora.py:140, 149`) | **different** |
| AdamW defaults | weight decay 1e-4 for Adam-family optimizers unless Optimizer Args sets it; 8-bit Adam eps floor 1e-6 (`FZ:families/minimax.py:366-367`, `FZ:families/train.py:1062-1066`) | no family weight-decay default in `TS:training/train.py` (grep: none); `eps_floor_8bit` exists in `TS:training/optimizers.py:203-237` but adamw8bit is not offered for H3 | **different** when the user picks adamw (PyTorch's default decay is 1e-2) |

## B2. Auto precision / memory plan

Upstream: the driver has its own planner (`plan_run`, `FZ:minimax/driver.py:67-125`) calling `plan_base_quant`
(`FZ:minimax/common.py:596-691`) and `plan_vram` (`:822-846`). Inputs: free VRAM, whether the file is the pruned build,
the heaviest dataset item as effective megapixels (spatial x clip frames, `_max_effective_mp`), and the adapter's real
size (`adapter_param_count`, `plan_adapter_gb`: rank, LoKR factor, optimizer, frozen LoRAs, EMA shadow).

Constants (`FZ:minimax/common.py:356-446`): `_RESIDENT_GB 17.5` (bf16 file as NF4), `_RESIDENT_PRUNED_GB 10.5`,
`_RESIDENT_INT8_GB 21.0`, `_RESIDENT_HQQ_GB 22.0`, `_RESIDENT_HQQ_PRUNED_GB 15.0`, `_INT8_TRANSIENT_GB 1.0`,
`_PER_BLOCK_GB 0.34`, `_ACT_GB_NOCKPT 5.5`, `_ACT_GB_CKPT 0.5`, `_SWAP_TRANSIENT_GB 7.5`, `_H2D_PER_BLOCK_GB 0.39`,
`_H2D_TRANSIENT_GB 2.0`, `_MIN_INT8_H2D_FREE_GB 13.5`, `_RESERVE_GB 1.5`, `_NOCKPT_MARGIN_GB 3.0`,
`_ACT_EXTRAP_FRAC 0.15`.

Order for the pruned int8 file:
1. int8, no swap, if it fits (also decides gradient checkpointing off / on).
2. int8 with `ceil((need + 2.0 - free) / 0.39)` blocks streamed host-to-device (max 40), if free VRAM >= 13.5 GB and
   available system RAM >= staged GB + 14.
3. otherwise NF4 (+ classic swap from `plan_vram`).
HQQ is never picked by Auto. A bf16 file is always NF4. A hand-set swap skips the planner and takes the file's own
precision (`FZ:minimax/driver.py:100-104`). The plan also trims previews (22 frames when streaming; 768x640 / 22 frames
under 20 GB total, `_plan_previews`, `:127-141`). Loading: `load_planned` (`:143-169`) - int8 ConvRot codes, NF4 or HQQ,
last n blocks through an H2D ring (`h3_h2d_offload.py`, `h3_nf4_h2d_offload.py`, `h3_hqq_h2d_offload.py`).
`max_blocks_to_swap` = 40 (`:64-65`).

TagScribeR (`TS:minimax_h3/description.py:108-114`): `precisions=("int8", "nf4")`; the generic `quant.plan` over
`train_memory={"int8": (((0.25, 22.1), (1.0, 22.4)), 0.39), "nf4": (((0.25, 13.5), (1.0, 13.7)), 0.0)}`. The comment
says the int8 peak is "an ESTIMATE" and 0.39 is "not measured with this port's offloader". Block swap uses the shared
`ModelOffloader` (round-trip), max `n_blocks - 2` = 48 (`TS:minimax_h3/driver.py:174-175`).

Differences: no adapter-size term, no clip term, no RAM check, no 13.5 GB floor, no gradient-checkpointing decision, no
HQQ, no H2D one-way streaming (upstream calls it ~6.4x faster than round-trip swap, `docs/MINIMAX_H3.md`), swap cap 48
vs 40, and the 66 GB bf16 DiT is refused (`TS:minimax_h3/description.py:59-60`) where upstream trains LoRAs on it as NF4.

## B3. Presets

Upstream (`FZ:families/minimax.py:24-51`): four built-ins, all from `_preset`. Shared values: `NETWORK_ALPHA = rank`,
`NETWORK_TYPE "LoRA (standard)"`, `LOKR_FACTOR 8`, `SAVE_EVERY_N_EPOCHS 1`, `SEED 42`, `FAMILY_SLIDER_GUIDANCE "2"`,
`ADAPTIVE_LR False`, `ADAPTIVE_LR_MIN "1e-5"`, `ADAPTIVE_LR_MAX "4e-4"`, `GRADIENT_ACCUMULATION 1`, `MAX_GRAD_NORM 1.0`,
`DATASET_MEGAPIXELS "0.25"`, `FAMILY_PRECISION "Auto (recommended)"`, `BLOCKS_SWAP "Auto (detect from GPU)"`,
`FAMILY_EMA "0.98 (recommended)"`, `H3_ADAPTER_RAMP "Off"`, `H3_CAPTION_DROPOUT "0.05 (default)"`,
`H3_STRUCTURE "Likeness and Style — 60% clean-end"`, `H3_LOWNOISE_PCT "60"`, `H3_HIGHNOISE_LR_PCT "100"`,
`H3_BLOCKS "all"`, `H3_TRAIN_REFINER ""`, `H3_LIKENESS_MODE "Default"`, `H3_ADAPTER "Circlestone — best for photos"`,
`H3_TREAD "1"`, `H3_DISTILL ""`.

| Upstream name | rank | epochs | LR | optimizer | `H3_CLIP_STILL` | `FAMILY_SLIDER` |
|---|---|---|---|---|---|---|
| `✨ MiniMax H3 Fast (LoRA 8, 50 epochs)` | 8 | 50 | 1e-6 | automagic3 | "1" | False |
| `✨ MiniMax H3 (rank 16, 60 epochs)` | 16 | 60 | 1e-6 | automagic3 | "1" | False |
| `✨ MiniMax H3 Style (LoRA 8)` | 8 | 50 | 1e-6 | automagic3 | "" | False |
| `✨ MiniMax H3 Slider (rank 8, 2e-4)` | 8 | 16 | 2e-4 | adamw8bit | "" | True |

TagScribeR (`TS:minimax_h3/description.py:17-45, 131-135`): the first three, same names, order and numbers. Differences:

- The **Slider preset is missing**.
- Keys are the OLD `MINIMAX_*` names (`MINIMAX_ADAPTER_RAMP`, `MINIMAX_LOWNOISE_PCT`, `MINIMAX_BASE_QUANT`,
  `MINIMAX_EMA`, ...). Upstream now uses `H3_*` option keys plus `FAMILY_PRECISION` / `FAMILY_EMA`, with
  `settings_aliases` and each option's `setting=` mapping the old names (`FZ:families/minimax.py:286-288`). A preset
  exported by current Fizgig carries `H3_*` keys, which `TS:training/presets.py` does not map (it handles `MINIMAX_*`
  only, `:110-146, 173`) - not tested here; by reading, they would land in "ignored".
- TS presets still carry keys upstream deleted (0 hits anywhere upstream): `MINIMAX_TRAIN_ADALN`,
  `MINIMAX_SLOW_BLOCKS`, `MINIMAX_SLOW_LR_SCALE`, `MINIMAX_BLOCK_LIMIT`, `MINIMAX_LR_WARMUP`.
- New upstream preset keys with no TS counterpart: `H3_STRUCTURE`, `FAMILY_SLIDER`, `FAMILY_SLIDER_GUIDANCE`.

RefMod presets are a separate family entry, see B8.

## B4. GUI settings upstream exposes for H3 NOW

Generic controls (same reader as Klein: `FZ:families/launch.py`). Visibility facts: Adaptive LR hidden
(`adaptive_lr=False`), loss watch hidden (`loss_watch=False`), Model Area and Timestep Range hidden (no `train_areas`),
Compile Blocks hidden (`compiles` not set), the generic training-adapter tick hidden (`training_adapter` not set: the
adapter is the `H3_ADAPTER` option), Samples "Advanced" card hidden (`samples_cfg_free=True`), reference row and
"Use Distilled" hidden (`FZ:lora_trainer_gui.py:11215-11224`).

| Upstream key | Default | Visible | Passed / read | TS |
|---|---|---|---|---|
| `LEARNING_RATE`, `NETWORK_DIM/ALPHA`, `MAX_TRAIN_EPOCHS`, `SAVE_EVERY_N_EPOCHS`, `SEED`, state keys, metadata keys | preset | yes | yes | P |
| `NETWORK_TYPE`, `LOKR_FACTOR` | LoRA, 8 | yes (hint "LoRA recommended for MiniMax") | yes | P |
| `OPTIMIZER_TYPE` | automagic3 | yes; whole catalog | yes; weight decay 1e-4 and eps floor added for Adam types | **D**: TS offers `automagic3, adamw` only (`TS:minimax_h3/description.py:115-116`; its comment "Fizgig locks the optimizer to adamw / automagic3" is out of date: upstream lists `("automagic3","adamw8bit","adamw")` and the Slider preset uses adamw8bit) |
| `OPTIMIZER_ARGS`, `GRADIENT_ACCUMULATION`, `MAX_GRAD_NORM`, `LR_SCHEDULER`, `LR_WARMUP_STEPS` | "", 1, 1.0, constant, "" | yes | yes (scheduler off under Automagic) | P |
| `FAMILY_PRECISION` ("Base Precision") | "Auto (recommended)" | yes; `int8 · most accurate, needs ~30 GB free`, `4-bit · fits smaller cards`, `4-bit HQQ · lower error than 4-bit, slower` (`minimax.py:359-364`) | `--precision auto/int8/nf4/hqq` | **D**: no HQQ; a preset's HQQ becomes NF4 with a note (`TS:training/presets.py:132-136`) |
| `BLOCKS_SWAP` | auto | yes | `--blocks_to_swap` (max 40) | D (round-trip swap, max 48) |
| `FAMILY_EMA` (in Other Options) | "0.98 (recommended)" | yes; extra choice "Short run" (`ema_short_run=True`) | `--ema_decay 0.98 / short` | P for the numeric values; **M** "Short run" |
| `CONTEXT_LORA_PATH/STRENGTH` | "", 1.0 | yes | yes; a context LoRA's full-model AdaLN rows are injected at run time (`frozen_file_added`, `FZ:minimax/driver.py:638-660`) | P for Linears; **M** AdaLN-row injection for context / adapter files |
| `FAMILY_SLIDER` + slider keys | off | yes (`slider_training=True`) | yes | **M** |
| `FAMILY_FT` + fine-tune keys | off | yes (`finetune=True`, `ft_learning_rate=3e-5`) | yes | **M** |
| `FAMILY_MULTICONCEPT` + extra folders | off | yes (`multi_concept=True`) | extra `[[datasets]]` blocks (`launch.py:723, 743-746`); ticking sets `H3_CAPTION_DROPOUT` to "0.10 (strong)" | **M** |
| Clip Target Megapixels (`CLIP_MEGAPIXELS`) | 0.25 | when the folder has clips | `clip_megapixels` in the dataset TOML (`launch.py:729-734`) | **M** |
| `SAMPLE_WIDTH/HEIGHT/STEPS/SEED/...` | 768 x 768, 20 | yes | yes | **D**: TS preview default 512 x 512 (`TS:minimax_h3/description.py:128-129`, citing the deleted `_build_minimax_train_command` fallback); upstream is 768 x 768 (`minimax.py:394-395`) |
| `FAMILY_TURBO_STEPS`, `FAMILY_TURBO_PACE` | 6, 75 | when the Turbo LoRA file is set (`samples_turbo_pace=True`) | `--speed_lora`, `--speed_lora_strength` (only when != 0.75), `--sample_steps` (`launch.py:598-616, 651-652`) | **M on main** (commit `99a35a0`, part C) |

Family options (`FZ:families/minimax.py:53-171`; sent as `--family_option k=v`, a trainer flag, or cache `--aux`):

| Option key (old key) | Label | Default | Tokens | Shown when | TS |
|---|---|---|---|---|---|
| `H3_TRAIN_BASE` (`MINIMAX_TRAIN_BASE`) | Training Base | First/last frame (fl2va) | "" / `--dit=pref:minimax_ref_dit` | always (Base Model card) | **M** |
| `H3_STRUCTURE` (`MINIMAX_LOWNOISE_PCT`) | Training Structure | Likeness and Style - 60% | `lownoise_pct=60` / `lownoise_pct=8` / Custom | always | **D**: TS has only the raw number `MINIMAX_LOWNOISE_PCT` 60 (`TS:training/params.py:234-238`) |
| `H3_LOWNOISE_PCT` | Clean-end share | 60 | `lownoise_pct={}` | Structure = Custom | P (as the only control) |
| `H3_HIGHNOISE_LR_PCT` (`MINIMAX_HIGHNOISE_LR_PCT`) | Medium to High Noise LR | 100 | `highnoise_lr_pct={}` | always | **M on main** |
| `H3_MIXED_STOP_CATEGORY` / `_EPOCH` / `_MODE` | Finish one category early | voice / blank / anchor at 10% LR | `stop_category=audio|visual`, `stop_epoch={}`, `stop_mode=anchor|stop` | mixed voice + picture datasets only | **M** |
| `H3_LIKENESS_MODE` (`MINIMAX_LIKENESS_MODE`) | Training mode | Default | Default: `photo_blocks=20-49 clip_blocks=20-49 audio_blocks=20-49`; More Blocks: `--train_blocks=6-49`; Off: "" | always | P (`TS:training/params.py:239-243`; same block sets, `TS:minimax_h3/driver.py:31-32`) |
| `H3_ADAPTER` (`MINIMAX_ADAPTER`) | Training adapter | Circlestone | `--training_adapter=pref:minimax_circlestone_adapter` / Ostris (fl2va or ref2va file by base) / Off | always | **D**: TS has an on/off tick for Circlestone only; Ostris refused with a note (`TS:training/presets.py:125-128`) |
| `H3_TREAD` (`MINIMAX_TREAD`) | TREAD token routing | on | `tread=0.5@2-47` | clips present, LoRA runs | **M** |
| `H3_CLIP_STILL` (`MINIMAX_CLIP_STILL`) | Also train each clip's sharpest face frame as a photo | on | `clip_still_as_photo=1 aux:clip_still=1` | clips present | **M** |
| `H3_ADAPTER_RAMP` (`MINIMAX_ADAPTER_RAMP`) | Adapter-relative LR | Off | `adapter_ramp=0.003|0.005|0.01` | always (Other Options) | **M** |
| `H3_CAPTION_DROPOUT` (`MINIMAX_CAPTION_DROPOUT`) | Caption dropout | 0.05 (default) | `caption_dropout=0|0.05|0.1` | always | P via the generic `CAPTION_DROPOUT` |
| `H3_BLOCKS` (`MINIMAX_BLOCKS`) | Blocks to Train | all | `--train_blocks={}` | live only with Training mode Off | P (`MINIMAX_BLOCKS`) |
| `H3_DISTILL` + `_WEIGHT` (0.8) + `_REFS` (2) + `_PHASE1` (Auto) | Learn identity from my dataset | off | `distill=1 aux:distill=1 --dit=pref:minimax_ref_dit`, `distill_weight={}`, `aux:distill_refs={}`, `distill_phase1=-1|0|2|4|8|16|30` | always (Other Options) | **M** |
| `H3_TRAIN_REFINER` (`MINIMAX_TRAIN_REFINER`) | Train the text token refiner | off | `train_token_refiner=1` | always | **M** (preset value refused with a note, `TS:training/presets.py:139-141`) |
| `H3_SAMPLE_FRAMES` (`SAMPLE_FRAMES`) | Sample length (Samples tab) | Still (1 frame) is the first choice; `docs/MINIMAX_H3.md` and `model_note` say previews default to 56 frames with sound - which one a fresh install gets was not resolved (`lora_trainer_gui.py:10857-10876`) | `preview_frames=1|22|56|124 preview_audio=1` | always | **M** (stills only) |
| `H3_FT_SCOPE`, `H3_FT_BLOCKS` | Train on / Fine-tune blocks | All media / empty | `ft_scope=photo`, `ft_blocks={}` | fine-tune only | **M** |
| `AUDIO_VAE` (fixed) | - | - | `audio_vae=pref:minimax_audio_vae aux:audio_vae=...` | always sent | **M** |

Every option above is read by the driver (`set_options`, `prepare_training`, `step_policy`, `batch_cond`,
`training_loss`, `_generate`, `run_metadata`, `FZ:minimax/driver.py:203-449, 676-691`) - none is a dead control.
Keys that existed in the old H3 GUI and are now gone upstream (0 hits): `MINIMAX_TRAIN_ADALN`, `MINIMAX_SLOW_BLOCKS`,
`MINIMAX_SLOW_LR_SCALE`, `MINIMAX_BLOCK_LIMIT`, `MINIMAX_LR_WARMUP`, `MINIMAX_LIKENESS_OPT`, `MINIMAX_LOGNORM`,
`MINIMAX_TURBO_STEPS`, `MINIMAX_TURBO_STRENGTH`.

## B5. Text encoder, VAE, cache format and ids

| Item | Upstream | TagScribeR | Verdict |
|---|---|---|---|
| Text conditioning | Qwen3-VL-32B, layer-50 states, (L, 5120), variable length (`FZ:minimax/driver.py:588-590`) | same (`TS:minimax_h3/driver.py:213-215`, `TS:minimax_h3/embedder.py` header) | same design |
| TE loader | `load_minimax_h3_te_planned` (`FZ:minimax/embedder.py:857`): nvfp4 kept packed, or bf16 -> NF4; pinned host-to-device layer streaming on small cards (`embedderH2D.py`, 3258 lines); vision tower for reference images (`build_qwen3vl_te`, `build_reference_tokens`); tokenizer bundled in `src/fizgig/assets/qwen3vl_tokenizer` | nvfp4 only, text only, per-layer `.to()` instead of the pinned streamer, tokenizer from a folder next to the file or the HF cache (`TS:minimax_h3/embedder.py:4-9`) | reduced: **bf16 TE file, vision path and H2D streamer missing** |
| Text cache file | keys `hidden_states` + `attention_mask`, metadata `format_version "2.0.0"`, `caption1` (`FZ:minimax/caching.py:104-121`); empty-prompt embed `uncond_minimaxh3_te.safetensors` per cache dir (`FZ:scripts/minimax_cache_text.py:298-318`); distillation refs in `..._teref{N}.safetensors` | generic `cond__hidden_states`, `format_version "1.0.0"` (`TS:training/cache.py:69-75`), empty-prompt file from `empty_cache_path` (`:227-230`) | **different layout under the same arch id `minimaxh3`** - the two apps' H3 caches are not interchangeable |
| Latent cache | `latent_{H}x{W}` (still) / `latent_{T}x{H}x{W}` (clip), optional `audio_latent` (packed rows (2T, 32)), `audio_only`, `still_latent`, `still_frame`, `latent_control_*`; `format_version "2.0.0"` (`FZ:minimax/caching.py:29-101`) | `latent_{h}x{w}` only | stills compatible in shape; clip / audio keys missing |
| Video VAE | encoder fp32 (`encode`, `encode_clip`, `plan_clip_bucket`), ViT3D decoder fp16 (`decode`, `decode_clip`, `decode_middle_frame`) (`FZ:minimax/vae.py:164-260, 415-600`) | encoder `encode` and decoder `decode` only (`TS:minimax_h3/vae.py:1-5`) | **clip encode / decode missing** |
| Audio VAE | `audio_vae.py` (459 lines): encoder, BigVGAN decoder, `pack_audio` / `unpack_audio` | none | **missing** |
| Arch id / key / suffix | `minimaxh3` / `minimax` / `mmh3` (`FZ:families/minimax.py:174-178`) | `minimaxh3` / `minimax_h3` (alias `minimax`) / `minimaxh3` (`TS:minimax_h3/description.py:48-53`) | suffix differs (`mmh3`) |
| Model files | 9 rows: `minimax_dit`, `minimax_ref_dit` (optional), `minimax_text_encoder` (nvfp4, bf16 alt), `minimax_vae`, `minimax_audio_vae`, `minimax_turbo_lora`, `minimax_circlestone_adapter`, `minimax_training_adapter` (Ostris fl2va), `minimax_ref_training_adapter` (Ostris ref2va) (`FZ:families/minimax.py:183-266`) | 4 rows: DiT, VAE, TE, Circlestone adapter (`TS:minimax_h3/description.py:56-73`) | **5 rows missing** |
| Tokenizer helper | bundled `assets/qwen3vl_tokenizer` | `helper_files` lists `Qwen/Qwen3-VL-4B-Instruct` (`TS:minimax_h3/description.py:138-141`) - a 4B repo for a 32B encoder; whether that tokenizer is byte-identical to upstream's bundled one was **not checked** | question |

## B6. LoRA targets, key format, network types, saved metadata

- Targets: 50 blocks x `attn.qkv_proj, attn.out_proj, mlp.fc1, mlp.fc2` = 200; upstream can add the 2 token-refiner
  blocks with `train_token_refiner=1` (`FZ:minimax/driver.py:967-972`); AdaLN is not a target. TagScribeR: the same four
  Linears, window only, no refiner.
- Keys: identical kohya `lora_unet_blocks_{n}_{attn|mlp}_...` (`FZ:families/minimax.py:338-348`;
  `TS:minimax_h3/description.py:89-100`); LoKR as `diffusion_model.<path>` on both sides.
- Block ids: upstream `h3blk_N`, `h3_rf_N` (`FZ:minimax/driver.py:876-885`); TagScribeR `block_N`
  (`TS:minimax_h3/driver.py:292`).
- Network types: `("lora", "lokr")` both.
- Metadata upstream adds (`run_metadata`, `FZ:minimax/driver.py:305-326`): `ss_visual_stop`, `ss_audio_stop`,
  `ss_photo_blocks`, `ss_clip_blocks`, `ss_audio_blocks`, `ss_tread`, `ss_caption_dropout`, `ss_clip_still_as_photo`,
  `ss_distill`, `ss_adapter_ramp`, `ss_train_token_refiner`, `ss_timestep_density` (the shift as `f"{shift:g}"`, e.g.
  `0.666667`), `ss_base_quant`; plus the generic `ss_train_blocks` when `--train_blocks` is used. TagScribeR main
  writes none of these (`TS:training/train.py:494-514`). `ss_network_module` differs
  (`fizgig.families (minimax, lora)` vs `tagscriber.training (minimax_h3, lora)`).
- Reading old H3 LoRAs with full-model AdaLN rows (context LoRA, training adapter): upstream injects them at run time
  (`_adaln_pairs`, `frozen_file_added`); TagScribeR main drops rows that do not fit a Linear.

## B7. Preview engines

| Engine | Upstream | TagScribeR |
|---|---|---|
| Base still | res_multistep on ComfyUI's simple schedule at shift 12, joint audio-row denoising, 20 steps, CFG 1 (`FZ:minimax/sampling.py:56, 168-171`; description `minimax.py:371-375`) | same (`TS:minimax_h3/sampling.py:49-153`) - present. Note: upstream's `SamplingSettings` text says `sampler="euler"` but `sample_image` defaults to `res_multistep` and the driver does not override it (`FZ:minimax/driver.py:737-740`) |
| Turbo LoRA | generic speed-LoRA path: file added as the `speed_lora` adapter, on only while previews render; its AdaLN rows injected with `turbo_adaln_patch` from `src/fizgig/assets/h3_silu_temb_grid.safetensors` (`FZ:minimax/common.py:1101-1179`, `FZ:minimax/driver.py:662-674`); 6 steps at 0.75; CFG: the Samples-tab value if > 1, else the Turbo's (`FZ:families/train.py:422-427`) | missing on main (commit `99a35a0`) |
| Clips | `preview_frames` 22 / 56 / 124, `decode_clip`, output contract: every 2nd frame as JPEG in `<stem>.clip/`, a wav, an mp4 (ffmpeg), then the middle-frame PNG last (`save_preview`, `FZ:minimax/driver.py:837-873`); OOM ladder (shorter clip, then resolution down to 512, `_ladder`, `:693-766`); small-card caps | **missing** |
| Sound | `preview_audio=1` + audio VAE decoder (`decode`, `decode_audio`, `:795-835`) | **missing** |
| Memory handoff | `park_for` / `unpark` (partial tail-block park for the decode, `:768-793`), optimizer state parked on CPU (`preview_park_optimizer=True`) | decoder kept on CPU between decodes (`TS:minimax_h3/driver.py:271-283`); no partial park |
| Travel noise | `initial_noise` raises `NotImplementedError` (`:607-608`) | implemented (`TS:minimax_h3/sampling.py:80-84`) - TS-only |
| Default size | 768 x 768 | 512 x 512 |

## B8. What each missing H3 capability needs

Sizes are for bringing TagScribeR to upstream behaviour, including Train-tab controls, help text and CPU tests:
S = under a day of agent work, M = 1-3 days, L = 3-7 days, XL = more. "Files" are upstream sources to port.

| Capability | Upstream files (lines) | Model files | Packages | Size | What TagScribeR's still-only layers need |
|---|---|---|---|---|---|
| **Family options layer** (prerequisite for most rows) | `families/description.py` `FamilyOption`, `ClipSpec` (`:123-220`); `families/launch.py` `option_tokens`, `option_applies`, `dataset_media` (`:302-384`); `families/driver.py` `set_options` | - | - | M | `training/description.py` gains `options`, `media`, `clip_spec`; `training/params.py` / `pipeline.py` build tokens from them; the Train tab renders them (choice / entry / check, `requires`, `show_if_media`, `mixed_only`, `mode`) |
| **Per-step hooks in the loop** (prerequisite) | `families/train.py` `step_policy`, `step_frozen_blocks`, `after_optimizer_step`, `lr_acc`, `batch_cond`, `prepare_training`, `run_metadata` (`:1276-1420`) | - | - | M | `training/train.py` and `training/driver.py`; LoRA built on all blocks with per-step freeze |
| **High-noise LR dial** | `minimax/driver.py:441-448`; `families/train.py:1313-1324, 1403-1404` | - | - | S (commit exists) | driver + loop |
| **Turbo-LoRA previews** | `minimax/common.py:1101-1179`; `minimax/driver.py:610-674`; asset `assets/h3_silu_temb_grid.safetensors` | `minimax_h3_turbo_v4_step600.safetensors` (0.78 GB, `larryvrh/MiniMax-H3-Turbo-Lora`) | - | S (commit exists) | driver hook for frozen files; also covers context / adapter AdaLN rows |
| **Training adapter choice + two bases** | `families/minimax.py:13, 54-58, 94-102` | Ostris fl2va + ref2va adapters (0.16 GB each, `ostris/minimax_h3_training_adapter`); `minimax_h3_ref2va_pruned_int8_convrot.safetensors` (21 GB) | - | S-M | options layer; `--dit` override; adapter AdaLN rows |
| **Adapter ramp** | `minimax/common.py` `AdapterRamp` (`:1534-1607`); `minimax/driver.py:268-303` | - | - | S | loop hook `after_optimizer_step` + `step_policy` multiplier |
| **Train the token refiner** | `minimax/driver.py:876-885, 967-972` | - | - | S | block map gains the refiner group; freeze it on routed steps |
| **HQQ 4-bit base** | `minimax/hqq4.py` (224), `h3_hqq_h2d_offload.py` (268), `loader.py` (277) | - | `hqq==0.2.8.post1` (installed with `DISABLE_CUDA=1`, `FZ:requirements.txt:22-26`) | M | `training/quant.py` gains "hqq" for H3 (driver-loaded); installer line; untested on ROCm upstream as far as the docs say |
| **H2D one-way block streaming + upstream Auto plan** | `h3_h2d_offload.py` (347), `h3_nf4_h2d_offload.py` (381), `common.py` planners (`:356-860`), `driver.py` `plan_run` / `load_planned`; fused int8 kernels `convrot.py` (294), `convrot_w8a16_triton.py` (161), `convrot_w8a16_backward_triton.py` (146) | - | triton (already), `psutil` (already), pinned RAM | L | `training/driver.py` gains `plan_run` / `load_planned`; TS model keeps ConvRot codes in `weight` for the shared offloader, which the upstream rings do not expect - the model's int8 Linear has to follow upstream's layout. The Triton kernels are NVIDIA paths; ROCm behaviour unknown |
| **bf16 66 GB DiT and bf16 TE support** | `loader.py`, `embedder.py:659-860` | `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors` (51.5 GB) | bitsandbytes (NF4) | M | loader branches TS dropped |
| **Clips (video training)** | `families/clips.py` (160); `dataset/image_dataset.py` clip paths (bucket per clip, `clip_megapixels`, `_plan_clip_bucket`, 4-D latents, clip pairs; ~300 of 1401 lines); `minimax/vae.py` `encode_clip`, `plan_clip_bucket`, `decode_clip`, `decode_middle_frame`; `minimax/model.py` T > 1 layout (`image_position_ids` `latent_t`, `pixel_frames_for_latent`, `latent_frames_for_pixels`); `minimax/caching.py` `_encode_clip` (`:190-289`); `scripts/minimax_cache_latents.py` (155); `common.py` `_max_effective_mp`, `clip_fallback_frames`, `write_preview_mp4`; `driver.py` `_ladder`, `save_preview` | - (video VAE already used) | `imageio-ffmpeg>=0.5` (bundled ffmpeg; decode, probe, mp4 writing) - **not in `TS:requirements.txt`** | XL | **Dataset layer**: `training/dataset.py` is image-only (`IMAGE_EXTENSIONS`, `TS:training/dataset.py:41`); needs media kinds, the ClipSpec refusal checks (24 fps, 17n+5 frames up to 124, edges multiple of 32, 32 kHz stereo), clip buckets and 4-D latent caches. **VAE**: causal 3-D encode over frames and temporal decode. **Model**: frame axis in patchify, position ids and audio row count (`audio_latents_for_frames(1)` is hard-coded, `TS:minimax_h3/model.py:500`). **Sampler**: `num_frames`, audio rows for the clip length, the OOM ladder. **Previews**: clip files + a gallery that can play them (the Train tab shows PNGs today - not checked in detail) |
| **Clip stills** ("sharpest face frame as a photo") | `minimax/still_pick.py` (126); `image_dataset.py:501-518, 801-807` | insightface detector model (Fizgig's `face_utils.FaceDetector`) | `insightface>=0.7.0`, `onnxruntime` (present), `opencv-python` (present) | M (after clips) | derived still items in the dataset; face detector (the owner's roadmap already lists face tools) |
| **TREAD token routing** | `minimax/model.py:936-985` | - | - | S-M (after clips) | model forward (clip steps only) |
| **Audio in clips / voice-only items** | `minimax/audio_vae.py` (459), `audio.py` (113), `clip.py` (46), `clips.read_audio`; `caching.py` `_encode_audio_item` (`:290-333`); `common.compute_loss` audio branch (`:995-1024`); `model.remap_sigma`; dataset audio sentinel (`AUDIO_EXTENSIONS`, `AUDIO_SENTINEL_RESO`, `image_dataset.py:98-112`) | `vae/minimax_h3_audio_vae_fp32.safetensors` (0.61 GB) | `imageio-ffmpeg` (audio decode through the ffmpeg binary); no torchaudio / soundfile used | L (after clips) | dataset voice items (`.wav .mp3 .flac .m4a`, strict durations), audio latent in the cache, `audio_rows` + `return_audio` already exist in the TS model forward (`TS:minimax_h3/model.py:468-469`) but only for the still's 4 rows; preview wav + mp4 |
| **Category retirement** ("Finish one category early") | `driver.py` `step_policy` (`:351-378`), `ft_cycle` (`:936-959`); `common.py` `ANCHOR_LR_SCALE`, `snap_ft_stop` | - | - | S (after voice) | loop hook |
| **Multi Concept** | `families/launch.py:723, 743-746`; dataset per-folder blocks | - | - | M | TS dataset is one folder per run (`TS:training/dataset.py:274-297`, not read in full) |
| **Reference distillation** | `common.py` `compute_distill_loss` (`:1270-1358`), `lora_disabled`; `driver.py` `_distill_setup`, `_teacher_phase` (`:328-349, 406-422`); `scripts/minimax_cache_text.py` reference encodes (330); `caching.py` `save_reference_text_cache`; `embedder.py` vision path (`build_qwen3vl_te`, `build_reference_tokens`); `reference.py` (98); `model.py` `ref_latents` / `text_token_tags` rows | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` (21 GB) | - | L | text encoder vision tower, reference rows in the model, `teref` caches |
| **Sliders** | `driver.py` `_slider_loss`, `noise_latents`, `predict`, `slider_setup`, `still_renders`, `slider_preview` (`:451-572`); shared `families/train.py` slider code | - | - | M-L (shared with Klein / Krea 2 / Qwen); clip pairs need clips | shared loop + driver |
| **Full fine-tune** | `minimax/ft_backend.py` (343), `rotation_ft.py` (605), `families/ft.py` (775), `convrot.py` encode side + Triton kernels; `driver.py:903-959` | none new (needs the pruned int8 file) | bitsandbytes NF4; upstream says "NVIDIA only ... untested on AMD/ROCm" (`docs/FINETUNE.md`) | XL | everything: master copy, rotating windows, ConvRot re-encode on save |
| **RefMods (the maker)** | `minimax/refmod.py` (1309), `scripts/minimax_refmod.py` (137), `reference.py`; GUI entry "MiniMax H3 RefMod" with `REFMOD_BUILT_IN_PRESETS` (`FZ:lora_trainer_gui.py:541-680`): 5 presets (community 8 refs 1 MP; Fizgig recipe 16 refs 1 MP optimised; lite 0.5 MP; style community 16x16; style high fidelity), keys `MINIMAX_REFMOD_GRID / _REFS / _STEPS / _CLIPS / _TOKEN_CAP / _AUDIO / _AUDIO_CONCEPT / _AUDIO_SECONDS / _CONCEPT / _DESC` | ref2va DiT for the optimised kind (default base); audio VAE for audio mods | insightface (face-centred crops) | L (plain encode alone: M) | a new "family-like" entry with its own card; reference rows in the model for the optimiser; no text-encoder pass for plain encodes |
| **RefMod Studio / Repair Studio / Profiler / Extract / Royale on H3** | `minimax/refmod_apply.py` (541), `minimax/workbench.py` (117), `families/video_workbench.py` (423), `families/workbench.py` (745) | - | `comfy-kitchen==0.2.31` (int8 attention on NVIDIA) | XL | workbench is "left for later" by the owner's earlier approval (parity skill); listed for completeness |
| **Gizmo (clip / voice prep tool)** | `gizmo.py` (repo root, not read) + Whisper helper | Whisper transcriber | not checked | not sized | TagScribeR has its own dataset tools; without a prep tool, users must bring on-spec clips |

## B9. What TagScribeR has for H3 that upstream removed or changed

1. Preset keys `MINIMAX_TRAIN_ADALN`, `MINIMAX_SLOW_BLOCKS`, `MINIMAX_SLOW_LR_SCALE`, `MINIMAX_BLOCK_LIMIT`,
   `MINIMAX_LR_WARMUP` (in the TS presets, `TS:minimax_h3/description.py:30-38`, and the ignore list
   `TS:training/presets.py:110-112`) - deleted upstream.
2. All `MINIMAX_*` option keys - renamed to `H3_*` upstream (old names still accepted there as aliases).
3. Optimizer restriction to `automagic3, adamw` - upstream lists adamw8bit too and the GUI offers the whole catalog.
4. Preview default 512 x 512 - upstream 768 x 768.
5. `lora_name_suffix "minimaxh3"` - upstream `mmh3`.
6. Refusing the bf16 DiT - upstream accepts it for LoRA runs.
7. `initial_noise` (travel previews) works in TS; upstream raises for H3.
8. Generic training-adapter tick for H3 - upstream replaced it with the three-way `H3_ADAPTER` choice.
9. The note at `TS:minimax_h3/__init__.py:10-11` ("the high-noise LR % dial (Fizgig never applies it under Automagic
   v3 ...)") as a reason not to port: upstream still exposes the dial; it acts with AdamW.

## B10. TagScribeR-only fix check: wide attention in the H3 VAEs

- H3 **video VAE**: the only attention is the ViT3D decoder's `_VaeAttention`, 32 heads x 64 = 2048
  (`FZ:minimax/vae.py:329-353, 376-378`; `TS:minimax_h3/vae.py:250-271, 298`). Head width 64: not a wide head. The
  encoder (`EncoderFCN3D`) is convolutional, no attention.
- H3 **audio VAE** (not in TagScribeR): `CausalAttention(in_dim=2048, num_heads=8)` -> head width **256 exactly**, a
  plain `F.scaled_dot_product_attention(q, k, v, is_causal=True)` (`FZ:minimax/audio_vae.py:121-148, 408`). That is at,
  not above, 256. If the audio VAE is ported, check it against `training/modules/wide_attention.py`'s threshold on
  ROCm before trusting voice caches (I did not read that module's exact condition).

## B11. Port plan for MiniMax H3 (dependency order)

1. **Description refresh (S)**: upstream names and facts - 768 preview, `mmh3` suffix decision, optimizer list, the
   Slider preset (hidden until sliders exist, or ask), drop deleted preset keys, map `H3_*` keys in
   `training/presets.py`, the 5 missing model-file rows.
2. **Family options layer (M)** and **loop hooks (M)** - B8 rows 1-2. Everything below sits on these.
3. **Stills-only parity on the existing model (S each, M together)**: high-noise LR (`3feb963`, renamed), Turbo
   previews (`99a35a0`, re-keyed), Training Structure dropdown, training-adapter choice (Circlestone / Ostris / Off),
   Training Base (fl2va / ref2va), adapter ramp, token refiner, EMA "Short run", bf16 trainable adapters, AdamW weight
   decay 1e-4, all-block LoRA with per-step freeze, `run_metadata` keys, context / adapter AdaLN-row injection.
4. **Base tiers (M + L)**: HQQ, then the H2D rings and upstream's `plan_run` / `load_planned`, bf16 DiT / TE files.
5. **Clips (XL)**: dataset media + ClipSpec checks, VAE clip encode / decode, model frame axis, sampler frames, clip
   previews with the OOM ladder, Clip Target Megapixels. Then TREAD (S-M) and clip stills (M).
6. **Audio (L)**: audio VAE, sound in clips, voice items, previews with sound; then category retirement (S).
7. **Multi Concept (M)**, then **reference distillation (L)** (needs the TE vision path and reference rows).
8. **Sliders (M-L, shared)**.
9. **RefMod maker (L)**; plain-encode presets first (M).
10. **Full fine-tune (XL, shared runner)**.
11. Workbench / RefMod Studio (XL) - only if the owner lifts the "later" decision.

Realistic total for steps 1-10: several weeks of agent work (roughly 3 S-M groups, 5 L, 3 XL), with clips, audio and
the fine-tune as the three large blocks, and every one of them unverifiable without the owner's real runs. Steps 1-3
(about a week) bring the still-image trainer level with upstream's still-image behaviour.

---

# PART C - The four unmerged TagScribeR commits against current Fizgig

| Commit | What it ports | Upstream now | Verdict |
|---|---|---|---|
| `ffc75bb` Klein network dropout (`NETWORK_DROPOUT`; neuron / rank / module dropout in `training/lora.py`; `FamilyLoRA.param_groups`) | the old `networks/lora.py LoRAModule.forward` dropouts | The key `NETWORK_DROPOUT` has **0 hits** in `lora_trainer_gui.py` and in `src/`. `families/lora.py` (the layer every family trains through) has no dropout: `LoRALinear.add / forward` (`FZ:families/lora.py:82-135`), and `families/train.py` has no dropout argument (`:1510-1618`). | **Obsolete.** Upstream removed the setting when Klein moved to the driver. Merging it would add a Klein control Fizgig no longer has. |
| `2089c5d` Klein LoRA+ ratio (`LORA_LR_RATIO`, up matrices in their own optimizer group) | the old `prepare_optimizer_params` | The key survives only as a hidden widget "always 1 ... for preset/save compat" (`FZ:lora_trainer_gui.py:3895-3897`), a number check (`FZ:families/checks.py:49`) and a docstring mention (`FZ:families/launch.py:29`). `train_command` never passes it; `families/train.py` builds one parameter group (or Automagic family groups) with no ratio (`:1046-1066`). | **Obsolete.** Upstream's ratio is fixed at 1 and unreachable. At ratio 1 the commit changes nothing; above 1 it does something current Fizgig cannot. (`param_groups` from `ffc75bb` is only needed by this.) |
| `3feb963` H3 Medium to High Noise LR | per-step LR multiplier for sigma >= 0.5, window mean on the optimizer LR, off under Automagic | Same mechanism, still live: `FZ:minimax/driver.py:441-448`, `FZ:families/train.py:1313-1324, 1403-1404`; option `H3_HIGHNOISE_LR_PCT` (`FZ:families/minimax.py:64-68`). | **Still mirrors upstream; needs small changes**: (1) key name `H3_HIGHNOISE_LR_PCT` (old `MINIMAX_HIGHNOISE_LR_PCT` is the alias) and info key `lr_mult`; (2) upstream **clamps** the percent to 0-100, the commit raises `ValueError` outside it; (3) metadata: upstream writes `ss_timestep_density` as the bare number (`f"{shift:g}"`), the commit writes `"shift{...}"`; upstream writes **no** `ss_highnoise_lr_scale` (not in `run_metadata`) and `ss_train_blocks` only for `--train_blocks` (block ids), with `ss_photo_blocks` etc. for Default mode; (4) upstream skips the multiplier for voice items (n/a until audio exists); (5) upstream tests `optimizer.__class__.__name__ == "Automagic3"`, the commit uses `owns_its_rate` - same result today; (6) upstream's window mean also carries `step_policy` multipliers (ramp, anchor) in the same list - keep one list when those are ported. |
| `99a35a0` H3 Turbo-LoRA previews (+ `assets/h3_silu_temb_grid.safetensors`, `turbo.py`, `MINIMAX_TURBO_STEPS / _STRENGTH`) | 6-step previews with the Turbo at 75%, AdaLN rows injected at run time | Same technique, still live: `_load_h3_egrid`, `_turbo_adaln_forward`, `turbo_adaln_patch / unpatch` (`FZ:minimax/common.py:1101-1179`), and the asset is still bundled at `FZ:src/fizgig/assets/h3_silu_temb_grid.safetensors`. | **Still mirrors upstream; needs changes**: (1) keys are now `FAMILY_TURBO_STEPS` (6) and `FAMILY_TURBO_PACE` (75) (`FZ:lora_trainer_gui.py:1506-1507`); `MINIMAX_TURBO_*` have 0 hits; (2) the hook is now general - `frozen_file_added(dit, path, strength, role)` for roles `adapter`, `context` and `speed`, with one combined patch per phase (training: adapter + context; preview: context + speed; `FZ:minimax/driver.py:638-674`) - the commit handles the speed LoRA only, so a context LoRA's or adapter's full-model AdaLN rows are still dropped; (3) CFG rule: upstream uses the Samples-tab CFG when > 1, else the Turbo's own (`FZ:families/train.py:422-427`), the commit always keeps the Samples-tab CFG through a `keep_cfg` option - same result at the default CFG 1; (4) upstream sends `--speed_lora_strength` clamped 0-2 and treats strength 0 as "no Turbo" (`FZ:families/launch.py:611-616`) - the commit clamps the same range; strength 0 handling not checked; (5) upstream reads the file through `ensure_kohya_lora_state_dict` and caches rows per (file, mtime) (`_ADALN_ROWS`) - the commit parses prefixes itself; (6) preview size default 768 upstream. The asset itself is unchanged in purpose (same file name, key `silu_t_emb_grid`); byte equality with upstream's copy was **not checked**. |

Neither H3 commit conflicts with the clip / audio work; both are good first steps of B11 step 3 after re-keying.

---

# PART D - Things I did not check

- No code was executed: every "identical" above is by reading, not by comparing numbers.
- `FZ:lora_trainer_gui.py` was grepped, not read; the visibility tables rely on the functions cited.
- TagScribeR's Train tab (Qt) was not opened: "present" means the parameter and driver support exist in `training/`.
- TagScribeR `klein/model.py`, `klein/embedder.py`, `minimax_h3/model.py`, `vae.py`, `embedder.py` bodies were not
  diffed line by line against upstream.
- Upstream `gizmo.py`, `embedderH2D.py`, the three offload rings, `refmod.py`, `rotation_ft.py`, `ft_backend.py` and
  `families/ft.py` were sized from outlines and docstrings, not read in full.
- `FZ:src/fizgig/utils/capabilities.py` has no Klein or H3 planner entries any more (one comment mentioning the H3
  fine-tune at line 464); the plans live in the descriptions and drivers cited above.
- Whether upstream's H3 Sample length defaults to a still or to 56 frames on a fresh install (B4).
- Whether the 4B tokenizer repo TagScribeR lists equals upstream's bundled 32B tokenizer (B5).
- ROCm behaviour of HQQ, the Triton int8 kernels and the H2D rings: upstream documents NVIDIA measurements only.
