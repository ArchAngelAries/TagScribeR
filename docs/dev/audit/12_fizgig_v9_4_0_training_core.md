# Audit 12: Fizgig 9162fb20 (v8.7.3) to c11134a2 (v9.4.0 + 4 commits), training core

Source: https://github.com/shootthesound/Fizgig. 694 commits, 321 files, +139,760 / -2,730. Read-only audit made on
2026-11-09 by reading release notes (`docs/RELEASE_NOTES_v8.7.4.md` to `v9.4.0.md`), `git log`, `git diff` and
`git show` of the two commits. Nothing was run: no training, no weights, no test suite. Fizgig line numbers are at
`c11134a2`; TagScribeR line numbers are the working tree on `port/v8-wave2` at the time of reading (the lead was
editing some of these files, so re-check a line before relying on it).

Scope: the shared training layer, the families TagScribeR has (Krea 2, Qwen Image 2.1, FLUX.2 Klein, MiniMax H3
stills, SDXL, Anima, MOSS-Image), the full fine-tune, and the packaging training depends on. The new families (Vimoo
1.0 Lite, Fibo 2.0, Z-Image Base, Flux 3) and the new tools (Studio, training maps, mask making, detail crops, Royale,
Profiler, helpers, pods, themes) belong to another audit and are named here only where they touch shared code.

Markers: **[S]** read in Fizgig's source at `c11134a2`; **[M]** from a commit message or release note only. Every
speed, memory or likeness figure is **Fizgig's measurement, not ours**, on NVIDIA hardware; none was run here.

## Summary

- **No objective, noise or target changed for an existing family at batch size 1.** The loss arithmetic of Krea 2,
  Klein, Qwen, SDXL, Anima and H3 is the same code path as before when no mask and no batch is in play [S: the
  per-driver diffs add `loss_mask`, `prior` and `per` branches and keep the old `F.mse_loss` return].
- **Three things do change what a default run learns**, all mirrored upstream in the last ten days:
  1. every picture family's built-in presets train on **Muon** instead of AdamW / AdamW 8-bit (same learning rates);
  2. the trainable LoRA does its two multiplies in the activations' **bf16** (fp32 weights kept) instead of fp32;
  3. **MOSS-Image** has a new timestep schedule by epoch (`30:7:60:8:3`), half learning rate on some epochs, a new
     preview recipe and a different default preset. TagScribeR's MOSS port (uncommitted, from `9162fb20`) predates
     all of it.
- **Fixes TagScribeR very likely shares**: 13 found (section 1). The ones that change results without an error are the
  full-matrix LoKR scale (#304), the Optimizer Args left over from a Muon preset (reachable today by importing a
  Fizgig 9.0.8+ preset), pictures with an EXIF rotation flag, transparent PNGs trained on their hidden colour, the
  preview canvas cap on cards under 20 GB, the loss watch throttling a third of a dataset, and Adaptive LR reacting to
  one jittery epoch.
- **Largest new capabilities in the shared layer**: Muon / Prodigy / Automagic v4, Batch size for every family with
  the loss watch kept, masked training (the loss side), caption dropout, Start at a smaller size, preview and save
  schedules, Resume taking the learning rate from the form, a fine-tune optimizer choice, compile for any sequence
  length, and Auto turning gradient checkpointing off on 40 GB+ cards.
- **Packaging**: Fizgig's CUDA install moved to PyTorch 2.15.0 / CUDA 13.4 / torchvision 0.29.2, bitsandbytes 0.50.2,
  triton-windows 3.7.0.post26, Python 3.11+, NVIDIA driver 580+. Its ROCm install is untouched. Nothing here applies
  to the AMD Windows wheels TagScribeR pins.

## 1. Fixes to something TagScribeR already has

Each entry: what Fizgig fixed, where TagScribeR has the same code, and how to verify without real weights.

**F1. A full-matrix LoKR is scaled by the alpha stored in the file (#304).** [S] `src/fizgig/networks/lora.py:326-351`
(`lycoris_scale_from_keys`: `if not dims: return 1.0`) and `:517` (`LoKRInfModule`). LyCORIS writes the network dim
into `alpha`, so a LyCORIS LoKR used as a Context LoRA played 16x or more too strong and "samples came out as static"
[M]. TagScribeR: `training/lora.py:75-89` still returns `alpha / 1` when there are no `lokr_w1_a` / `lokr_w2_a`
factors; it feeds every frozen LoKR file read at `training/lora.py:399-401`. TagScribeR's own LoKR saves fold the
scale into `w2` (`training/lora.py:585-590`) and should be unaffected. Verify: a unit test building a key dict with
`lokr_w1`, `lokr_w2` and `alpha = 16` must give scale 1.0.

**F2. Optimizer Args left from a Muon / Prodigy preset silently turn the run into plain AdamW (9.2.0).** [S]
`src/fizgig/training/optimizers.py:250-254` (the args are dropped at build time) and
`src/fizgig/families/form.py:1107-1113` (cleared in the form). Reachable in TagScribeR **today**: a Fizgig 9.0.8+
preset carries `OPTIMIZER_TYPE = "muon"` and `OPTIMIZER_ARGS = "adjust_lr_fn=match_rms_adamw weight_decay=0.01
momentum=0.95 nesterov=True ns_steps=5"`. `training/presets.py:273-274` refuses the unknown optimizer and keeps the
current one, but `OPTIMIZER_ARGS` is free text (`training/params.py:253`) and is applied; `training/optimizers.py:
241-271` then fails to build AdamW 8-bit with those keywords and falls back to `torch.optim.AdamW(params, lr=lr)`
with only a log warning. This also breaks standing rule 1 ("its preset files import unchanged"). Verify: apply such a
preset dict in a unit test, then `create_optimizer("adamw8bit", ..., args)` must not return the `"adamw (fallback)"`
label. Fix: port Muon (section 3) and the guard.

**F3. Block swap with a LoRA on some blocks only crashes on the second step with "mat2 is on cpu" (#302).** [S]
`src/fizgig/modules/offloading.py:387-394, 437, 467-481, 489`: the backward stops at the first trained block, so the
hooks that bring the earliest swapped blocks back never fire; the offloader now counts fired hooks and restores every
block's place before the next forward. TagScribeR: `training/modules/offloading.py:394` (hook registration), `:433-441`
(hook), `:444-460`, `:463-466` (`wait_for_block`) have none of it. Reachable with Qwen Fast Identity Mode, Klein Custom
blocks / Model Area, `train_blocks`, and possibly TagScribeR's window-only H3 LoRA with swap (to check: whether H3
goes through this offloader). It fails loudly, not silently. Verify: read-through against upstream; the offloader sits
behind the CUDA guard, so only a real run with swap and a block subset confirms it.

**F4. Resume with another Optimizer Type (#246).** [S] `src/fizgig/families/train.py:104-105` (the state records
`"optimizer"`), `:194-209` (a different type, or a different group count, keeps the LoRA weights and epoch and starts
a fresh optimizer). TagScribeR: `training/train.py:290-311` calls `optimizer.load_state_dict` unconditionally
(`:298`). A different type either raises on the group count or loads moments that do not fit. Verify: unit test on
the tiny family, save with AdamW, resume with Lion 8-bit or Automagic v3.

**F5. Training previews are shrunk to 768 px on every family on a card under 20 GB (#309).** [S]
`src/fizgig/families/train.py:1160-1173`; `src/fizgig/families/description.py:666` (`preview_small_card_cap`, 0 =
never) and `src/fizgig/families/krea2.py:195` (`768`, Krea 2 only). Fizgig's report: a 768x1024 MOSS-Image preview
rendered at 576x768 "looked soft, oddly coloured and broken in the face" [M]. TagScribeR: `training/train.py:822-826`
and `:1297` cap every family (`_small_card_previews`, `:335-349`, total memory under 20e9 bytes). Verify: unit test
with `TAGSCRIBER_SIM_VRAM_GB=16` on a non-Krea tiny family: the preview size must stay as asked. The model still
parks on the CPU for the decode on every small card (unchanged).

**F6. Pictures with an EXIF rotation flag train sideways (#232).** [S] `src/fizgig/utils/upright.py` (31 lines:
`open_upright`, `upright_size`); `src/fizgig/dataset/image_dataset.py` reads every picture through it and
`source_stamp` adds `":upright"` so a picture cached sideways re-encodes once. TagScribeR: `training/dataset.py:400-401`
reads `im.size` as stored and `:420`, `:427` open pixels without `ImageOps.exif_transpose`, while
`core/image_ops.py:103` and `core/health.py:59` do apply it elsewhere, so a caption may describe an upright picture
that trains rotated and in the wrong bucket. Verify: a generated JPEG with orientation 6 (never a real photo) must
come out of `load_pixels` upright and land in the portrait bucket.

**F7. Transparent PNGs train on whatever colour hides behind the transparent pixels (9.0.3).** [S] Fizgig lays a
see-through picture on white (`src/fizgig/utils/image.py` `flatten`; `src/fizgig/families/train.py:471-477`
`_as_trained`) on every family but Qwen Image 2.1, and marks the cache so old ones re-encode once [M].
TagScribeR: `training/dataset.py:421` keeps RGBA and every driver then drops the alpha channel
(`[..., :3]`: `training/families/krea2/driver.py:151`, `klein/driver.py:221`, `anima/driver.py:61`,
`sdxl/driver.py:100`, `minimax_h3/driver.py:260`, `moss_image/driver.py:206`, `qwen_image21/driver.py:78`). Verify: a
generated RGBA PNG whose transparent area holds a strong colour; the encoded pixels must be white there. Needs a
`latent_rev` bump (or an alpha mark in the cache metadata) so affected caches re-encode.

**F8. Auto precision swaps many INT8 blocks where NF4 with no swap fits (#216, Krea 2 on 16 GB).** [S]
`src/fizgig/families/quant.py:208` (`TIGHT_MARGIN_GB = 0.5`) and `:257-259`: when nothing fits with the 1.5 GB
margin, NF4 goes first if it fits with 0.5 GB to spare. Fizgig measured the swapped INT8 base at 4.4x the step time
and 4x the CPU [S: the docstring at `:224-229`]. TagScribeR: `training/quant.py:232-246` goes straight from "fits with
the margin" to the swap. TagScribeR's "INT8 + maximum swap when even that is short" (kept, PORT_PLAN) is the later
branch and is not touched by this. Verify: unit test of `quant.plan` with Krea 2's figures at 13.9 GB free and 1 MP.

**F9. Adaptive LR halves the rate on one jittery epoch (9.0.0).** [S] `src/fizgig/training/adaptive_lr.py:33`
(`SMOOTH_EPOCHS = 3`), `:147-149` (plateau / improving judged on the mean of the last 3 epochs' losses, carried
through pause / resume in `recent_losses`). The source comment says an epoch's loss "jitters by about +-0.005, which on
its own read as a plateau every other epoch and walked the LR to Min". TagScribeR: `training/adaptive_lr.py:151-230`
compares the raw epoch loss. This changes a path the owner has verified on Krea 2, so it is "mirrored, unit-tested,
waiting for a real run". Verify: unit test replaying a loss series with +-0.005 jitter.

**F10. The per-image loss watch convicts about a third of every dataset late in a run (9.3.0).** [S]
`src/fizgig/training/loss_logger.py:647` (a stuck vote now needs `mean_residual >= ext_hi`, the median + 1.5 IQR
fence, not `>= hi`) and `:185` (`stuck_floor` 0.1 -> 0.25). Fizgig replayed 39 of its own runs: 1026 convictions
became 249 [S: comment at `:633-637`; the measurement is Fizgig's]. TagScribeR: `training/loss_logger.py:605`
(`>= hi`) and `:172` (`stuck_floor: float = 0.1`); `ext_hi` already exists at `:529`. Same caveat as F9: the owner's
Krea 2 runs use the loss watch.

**F11. A Context LoRA that adapts H3's patch projections or output heads crashes at step 0 (#298).** [S]
`src/fizgig/minimax/model.py` `_wdtype()` reads a Linear's weight dtype through a LoRA wrapper; the same one-line
change in `src/fizgig/moss_image/model.py`. TagScribeR reads `.weight.dtype` directly at
`training/families/minimax_h3/model.py:185, 366, 372, 555, 576` and `training/families/moss_image/model.py:91`.
Whether TagScribeR's `FamilyLoRA.add_file` can wrap those modules was not traced (open question 3).

**F12. A Context LoRA that matches nothing trains on as if it were not there (9.0.4).** [S]
`src/fizgig/families/train.py:1688-1692` raises when `net.add_file` plus `driver.frozen_file_added` (which now returns
a count, `src/fizgig/families/driver.py:494-499`) is 0. TagScribeR: `training/train.py:1054-1058` logs "0 Linears" and
continues. PORT_PLAN's settled rule ("a Turbo LoRA that matches nothing is a warning, the run continues") is about the
preview Turbo LoRA, which upstream also still only warns about (`train.py:1702-1709`).

**F13. The fine-tune's "free each gradient" stepper ignores the step's LR multiplier (uncommitted `training/ft.py`).**
[S] `src/fizgig/families/ft.py:663-686` (`FusedSteps.scale`, applied to every per-weight optimizer before its step)
and `src/fizgig/families/train.py:2908-2909`. Without it a family's `step_policy` / `lr_mult` (H3's noise-band dial,
`lr_boost`, MOSS-Image's half-rate epochs) does nothing in a fine-tune with that tick on. TagScribeR:
`training/ft.py:694` (`FusedSteps`) and `training/train.py:707`. The step readout under that mode also showed no
rate upstream (`train.py:2474-2482`).

Also likely shared, smaller:

- **Resume after deleting cache files stops with "No training items" (#210).** [M] Fizgig caches the missing pictures
  again before resuming (in `families/launch.py`). TagScribeR's pipeline skips the cache stage on resume
  (`training/pipeline.py:13`, module docstring); not traced further.
- **Slider previews were tied to the save cadence** [S: `families/launch.py`, commit b9ce9c5e] and every Slider
  preset now sets previews every 5 epochs (`src/fizgig/families/form.py`, `SLIDER_PREVIEW_EVERY = "5"`).
  TagScribeR's `training/pipeline.py:780-786` passes `SAMPLE_EVERY_N_EPOCHS` as set (already right); its slider
  presets do not set the cadence.
- **An SDXL prompt slider with CFG above 1 stopped at its first practice picture** [S: `train.py:1793-1795`, the
  negative is moved to the GPU with the prompt]. TagScribeR renders the bank in its own `_render_slider_bank`
  (`training/train.py:1120`); not checked.
- **Speed readout**: s/it now leaves out everything between epochs (saves, state, loss watch, Adaptive LR, previews)
  [S: `train.py:2384, 2784-2786, 2955`]. TagScribeR subtracts previews and slider updates only
  (`training/train.py:1535, 1571, 1591, 1604`).
- **Warm-up note**: measured instead of "the first two epochs" [S: `train.py:2544-2577`]; TagScribeR
  `training/train.py:1458-1461` is the old fixed text.

## 2. Changed defaults, presets and numbers

### 2.1 Every picture preset trains on Muon (v9.0.8)

[S] `src/fizgig/training/optimizers.py:35`:
`MUON_DEFAULT_ARGS = "adjust_lr_fn=match_rms_adamw weight_decay=0.01 momentum=0.95 nesterov=True ns_steps=5"`.
Each family's `_preset()` now sets `"OPTIMIZER_TYPE": "muon", "OPTIMIZER_ARGS": MUON_DEFAULT_ARGS`. Learning rates,
ranks, alphas, epochs, EMA and Adaptive LR bounds of the existing families' presets are **unchanged** [S: the diffs of
the six description files touch only that line, plus the items below].

| Family | Was (9162fb20) | Now | Fizgig file:line | TagScribeR now |
|:--|:--|:--|:--|:--|
| Krea 2 (all presets, Slider too) | adamw8bit | muon + args | `families/krea2.py:23` | adamw8bit, `training/families/krea2/description.py:28`, `:201` |
| Qwen Image 2.1 (all, Edit, Slider) | adamw8bit | muon + args | `families/qwen_image.py:32` | adamw8bit, `qwen_image21/description.py:41` |
| Klein 9B (all) | adamw8bit | muon + args | `families/klein.py:22` | adamw8bit, `klein/description.py:28` |
| SDXL (Fizgig's presets) | adamw | muon + args | `families/sdxl.py:22` | adamw, `sdxl/description.py:46` (TagScribeR's own presets: adamw8bit, `:28`) |
| Anima (Fizgig's presets) | adamw | muon + args | `families/anima.py:20` | adamw, `anima/description.py:74` (TagScribeR's own: adamw8bit, `:98`) |
| MOSS-Image (all but Fine-tune) | adamw | muon + args | `families/moss_image.py:46` | adamw, `moss_image/description.py:43` |
| MiniMax H3 LoRA presets | automagic3 | automagic3 (kept) | `families/minimax.py:24` | automagic3 |
| MiniMax H3 Slider | adamw8bit | **adamw** | `families/minimax.py:50-51` | adamw8bit, `minimax_h3/description.py:179-180` |
| Any Fine-tune preset | the family's LoRA optimizer in the form, Adafactor in the trainer | **adafactor**, args "" | `families/registry.py:22-33` | Adafactor in the trainer (`training/ft.py:679`) |

Notes:
- Muon's step is scaled to AdamW's (`adjust_lr_fn=match_rms_adamw`: `lr * 0.2 * sqrt(max(rows, cols))`,
  `src/fizgig/training/muon.py:117-127`), which is why the rates stay. Fizgig's own caution: with Adaptive LR on,
  "watch for a run that climbs above about 3.5e-4" (one Qwen run did better once it came back to 2e-4) [M]; the
  learned ceiling in section 3 is the follow-up.
- Fizgig's claims (likeness matched AdamW over 20 epochs on Qwen at rank 8; "12x faster" batched step; "about 0.2 GB"
  more state than AdamW 8-bit on a rank-32 Krea 2 LoRA) are its measurements on NVIDIA [M].
- The family `optimizers=(...)` tuples upstream are unchanged (`krea2.py:159` etc.); the Optimizer Type row lists
  `available_optimizers()` (`families/form.py:131-133`), so Muon, Prodigy, Automagic v4 and Adafactor show for every
  family.
- SDXL with LoCon in TagScribeR has 4-D conv factors: upstream's dispatch (`optimizers.py:314-327`) would send such a
  run to the per-parameter `Muon` class, not `BatchedMuon` (slower; Fizgig has no Conv2d LoRA so this is unmeasured).

### 2.2 MOSS-Image (TagScribeR's port is from 9162fb20 and uncommitted: take these before it lands)

| Item | 9162fb20 / TagScribeR now | c11134a2 | Fizgig file:line |
|:--|:--|:--|:--|
| Clean-end share, LoRA and slider | `30` (one value) | `30:7:60:8:3` (Slide) | `families/moss_image.py:30, :54, :374` |
| Clean-end share, fine-tune | `30` | `11:4:30:8:2` | `moss_image.py:34, :54, :223-224` |
| Clean-end share, Single mode start | n/a | `40` | `moss_image.py:384` |
| Slide semantics | n/a | `A:H:B:E[:R]`, see below | `moss_image/driver.py:239-271` |
| Optimizer | adamw | muon + args | `moss_image.py:46` |
| Default (first) preset | Standard (rank 32, 1 MP) | Fast (rank 16, 0.25 MP) | `moss_image.py:355-356` |
| New preset | none | Turbo Train (Experimental): rank 32, alpha 16, Adaptive LR 2e-4 to 4e-4, 70 epochs, 0.25 MP, loss watch and per-image LR on | `moss_image.py:361-363` |
| Turbo LoRA file | `lodestones/MOSS-Image-SFT-8-steps-dmd` `lora.safetensors` | `lodestones/Kromoss` `8-steps-dmd-lora.safetensors`, a default download | `moss_image.py:20, :98-106, :264` |
| Turbo sampling | 8 steps, CFG 1 | 8 steps, **CFG 2**, negative prompt on | `moss_image.py:270-274` |
| Turbo preview strength | 0.0 (off) | **0.6** | `moss_image.py:293` |
| Previews on SFT by default | True | **False** (Base + Turbo 0.6 + SFT look LoRA) | `moss_image.py:327` |
| SFT look LoRA (previews only) | none | `ShootTheSound/Fizgig-MOSS-Image-SFT-Look-LoRA` `moss_image_sft_look_lora.safetensors`, 2.07 GB, strength 1.5, on | `moss_image.py:19, :136-142, :317-318` |
| Compile boundary | outside | **inside** ("on torch 2.15 it costs no extra memory") | `moss_image.py:201` |
| Kromoss as a Base / preview download | offered (Base row `fetch_choice`) | withdrawn; rows carry `replaces=(kromoss files)` | `moss_image.py:81-85, :125-129` |
| Fast Identity Mode | none | blocks 24-35 (`identity_blocks`) | `moss_image.py:214-215` |
| Small-card previews | capped at 768 (shared bug F5) | full size; opt-in tick `preview_small_card_option=768` | `moss_image.py:291` |
| Batch memory | none | `batch_gb=(0.45, 1.35)` | `moss_image.py:65` |
| Training memory | unchanged | unchanged, plus no-checkpointing figures | `moss_image.py:189-196` |

Slide semantics [S: `moss_image/driver.py:239-271`]: `A:H:B:E` holds share A until epoch H, slides in a straight line
to B by epoch E; with a fifth number R, after E the run trains at the **higher** of A and B and every R-th epoch at
the **lower** one, and those lower-share epochs train at **half the learning rate** (`step_policy` returns
`(False, 0.5)`). The epoch comes from a new driver hook `begin_epoch(epoch)` (`families/driver.py:187-191`), called
at `families/train.py:2706`; a fine-tune counts rotations instead (`driver.py:200-203`, `ft_cycle`). The default
`30:7:60:8:3` therefore means 30% until epoch 7, 60% from epoch 8, then 60% with every 3rd epoch at 30% and half LR.
Fizgig's notes say this order "gave the best results" with Muon and suits datasets up to about 55 images [M].

TagScribeR today: the driver takes one number only (`training/families/moss_image/driver.py:86-95`, `float(raw)`), so
a Fizgig 9.x MOSS preset's `MOSS_LOWNOISE_PCT` is refused with an error rather than mis-trained; there is no
`begin_epoch` in `training/driver.py` or `training/train.py`; `training/families/moss_image/description.py:26, :212,
:220-222` still carry the SFT Turbo LoRA at strength 0, `:228-230` previews on SFT, `:159` compile outside, `:73-81`
the Kromoss choice, `:251-256` the old preset order. No `MOSS_*` parameter exists in `training/params.py` yet.

Other MOSS-Image fixes upstream that the port should start with [M unless marked]: the SFT preview checkpoint loads
with plain reads (mapping it beside the training model crashed Windows with 32 GB of RAM) [S: `moss_image/driver.py:
62, :100-105`]; the text encoder streams from the file on computers under 30 GB of RAM (3.4 GB instead of 14 GB)
[S: `driver.py:149-155`]; quantised files with `.input_scale` keys load; a cached tokenizer missing its chat template
no longer stops caching (#305); ComfyUI's 4-bit text encoder file loads; a LoRA covering the output layer renders.

### 2.3 Other per-family changes

- **Krea 2** [S `families/krea2.py`]: `preview_small_card_cap=768` (`:195`, the cap is now Krea 2's alone);
  `batch_gb=(1.2, 2.0)` (`:39`); `text_cache_trim=True` (`:200`, used by the helper share only); training memory
  table unchanged (`:151-152`); no-checkpointing figures added (`:154-158`). Turbo previews at 0.7 / 10 steps are
  unchanged, and TagScribeR already passes the 0.7 (section 4).
- **Qwen Image 2.1** [S `families/qwen_image.py`]: `rgba=True` (`:50`: transparent pictures keep their alpha through
  the RGBA VAE); `batch_gb=(0.55, 1.35)` (`:49`); wording of the training-adapter note; memory table unchanged.
- **Klein** [S `families/klein.py`]: the VAE is saved as `flux2-vae.safetensors` with a header signature
  `("decoder.conv_in.weight", (512, 32, 3, 3))` (`:71-72`); `batch_gb=(0.8, 1.1)` (`:36`). TagScribeR's row is still
  `ae.safetensors` (`training/families/klein/description.py:56`), which only matters if a Z-Image family (whose
  different VAE has the same file name) is ever added.
- **SDXL** [S `families/sdxl.py`]: `batch_gb=(0.1, 0.55)` (`:39`), `batch_auto=((0.25, 8), (1.0, 4))` (`:43`).
- **Anima** [S `families/anima.py`]: `batch_gb=(0.25, 0.65)` (`:35`), `batch_auto=((0.25, 4), (1.0, 1))` (`:39`).
- **MiniMax H3** [S `families/minimax.py`]: Slider on AdamW (above); `batch_training=True` for pictures;
  `caption_dropout=False`, `resolution_stages=False`, `training_maps=False`; `masked_training=True` for stills.
- **Start checks**: each family declares the learning rates it calls usual (`start_lr_bands`), e.g. Krea 2 LoRA
  3e-5 to 3e-4, Qwen 3e-5 to 4.5e-4, Klein 3e-5 to 1.2e-3, SDXL 1.7e-5 to 1.5e-4, Anima 7e-6 to 3e-4, MOSS-Image
  1.7e-5 to 3e-4 [S: the first lines of each description]. A warning at Start, never a block.

### 2.4 Shared-layer numbers

| What | Was | Now | Fizgig file:line | TagScribeR |
|:--|:--|:--|:--|:--|
| Loss watch stuck floor | x0.1 | x0.25 | `training/loss_logger.py:185` | 0.1, `training/loss_logger.py:172` |
| Loss watch stuck vote | residual >= top third | >= median + 1.5 IQR | `loss_logger.py:647` | `>= hi`, `:605` |
| Adaptive LR plateau test | this epoch's loss | mean of last 3 epochs | `training/adaptive_lr.py:33, :147-149` | raw, `training/adaptive_lr.py:151` |
| Adaptive LR probe up | `min(lr*1.25, max)` if higher | must exceed `lr*1.001`, capped by a learned ceiling | `adaptive_lr.py:202-203` | `:206-207` |
| Auto precision | swap INT8 when nothing fits | NF4 first if it fits with 0.5 GB | `families/quant.py:208, :257` | `training/quant.py:232-246` |
| Gradient checkpointing | always on | "auto": off (or partly off) on cards of about 40 GB and up when measured figures fit | `families/train.py:1500-1522`, `quant.py:308-310, :340-396` | always on |
| Trainer's optimizer default | "adamw" | None: adamw for a LoRA, adafactor for a fine-tune | `train.py:1133` | "adamw", `training/train.py:785` |
| Turbo LoRA strength when not given | the LoRA's own (1.0) | the family's preview strength | `train.py:1147-1149` | already right (section 4) |
| Step log LR precision | 4 significant digits | 6 | `training/step_log.py:47` | not checked |
| Target megapixels choices | ..., 1.0 | adds 1.05 (exactly 1024 x 1024); 1.0 is 992 x 992 | [M] 8.7.5 notes | TagScribeR's own dataset code; sizes not compared |
| Preview / save cadence | every N | every N, then from epoch Y every Z | `train.py:390-396` | every N |
| Auto-recaption first try | default token budget | `max_new_tokens=160` | `families/loss_watch.py:324` | not checked |
| INT8 backward option | "bf16" or "int8" | bf16 only (option removed) | `modules/int8_train.py:15-17` | never exposed (section 4) |

### 2.5 The trainable LoRA's multiplies run in bf16 (v9.2.0, commit a667a9d0)

[S] `src/fizgig/families/lora.py:113-131` (`_LoraLowp`) and `:190-193`: when the trainable adapter's weights are fp32
and the layer's input is bf16 / fp16, the down and up multiplies run in the input's dtype with a hand-written
backward that keeps the input once in bf16; the weights and their gradients stay fp32. Before, the input was cast to
fp32 (`lx = ad(x.to(ad[0].weight.dtype))`), which is what TagScribeR still does (`training/lora.py:190`). This is a
numerics change for every family whose trainable adapter is fp32 (all but H3, whose adapters are bf16). Fizgig's
reason is memory ("what made the checkpoint-inside-compile placement double MOSS's peak") and it reports likeness
"the same as 9.0.8 or better" [M]. It is a silent change to a verified path: mirror it only together with a real run
by the owner, and never mix it into a commit that also changes the optimizer.

## 3. New capabilities in the shared layer or an existing family

Sizes are upstream lines; "deps" names what each needs first.

1. **Muon, BatchedMuon** [S]. `src/fizgig/training/muon.py` (216 lines, adapted from `torch.optim.Muon`, BSD-3-Clause;
   needs a `THIRD_PARTY_NOTICES.md` entry) plus about 40 lines in `optimizers.py` (`:35-36`, `:314-327`, `:339-340`).
   No new package. Uses `torch._foreach_lerp_`, `torch.baddbmm` and a bf16 Newton-Schulz iteration; not run on ROCm
   by anyone here. The new optimizers never fall back to AdamW (`optimizers.py:339-340`).
2. **Prodigy** (`training/prodigy.py`, 261 lines, vendored from prodigyopt 1.1.2, MIT) and **Automagic v4**
   (`training/automagic4.py`, 270 lines, vendored from ai-toolkit, MIT; subclasses the Automagic v3 TagScribeR already
   has). Prodigy owns its rate (`owns_its_rate`, `optimizers.py:184-187`): Adaptive LR, the scheduler and the look
   warm-up stand down, per-image LR stays (`train.py:1852-1859`); the LR box is locked to 1.0 in the form. Fizgig:
   "No preset uses them"; both "got there more slowly" than Muon / AdamW in its Qwen test [M].
3. **Adaptive LR's learned ceiling** [S] `training/adaptive_lr.py:23-27, :32, :84-102`: a cut within 3 epochs of a
   probe up, from the rate that probe set, caps later probes at that rate / 1.25. State persists across resume.
   About 60 lines with F9's smoothing. Deps: none.
4. **Batch size above 1, Fizgig's way** [S]. Driver hooks `training_loss_batch` and `stack_conds`
   (`families/driver.py:259-283`), `picture_losses` (`:54-63`), `_batch_loss` in the trainer (`train.py:1292-1334`),
   the loss watch at any batch (`families/loss_watch.py:179-204`: `weigh`, `observe_each`), planner
   (`quant.py:276-280`, `batch_gb`), a resume at another batch size (`train.py:1893-1898`), masked attention
   preferring cuDNN with captions padded to a multiple of 16 (`modules/sdpa.py:112-120`). Per family 20 to 40 lines.
   Each picture's noise and t are drawn in the order batch-1 steps would draw them. Now covers Qwen, MOSS-Image and
   H3 stills (packed, `forward_packed`), which TagScribeR's own batching does not (`supports_batching` is True only
   for Krea 2, Klein, SDXL, Anima). Not at batch above 1: sliders, DOP, edits, masked prior, Start at a smaller size.
   "Batch size Auto" and the measured figures are for 40 GB+ NVIDIA cards.
5. **Masked training, the loss side** [S]: `weighted_loss`, `prior_base`, `with_prior` (`families/driver.py:37-79`),
   `loss_mask` / `prior` arguments on every `training_loss`, `_MaskPrior` (`train.py:693-713`), `_loss_mask`
   (`train.py:1336-1342`), `dataset/masks.py` (203 lines), metadata `ss_masked_training`, `ss_mask_floor`,
   `ss_masked_images`, `ss_mask_prior`. Default floor 0.1. The mask-making tools are the other audit's.
6. **Caption dropout for every picture family** [S] `train.py:1583-1605`, `:2871-2873`: Off / 0.05 / 0.10; the empty
   caption is encoded once into `uncond_<arch>_te.safetensors` in the cache folder; the draw is seeded from the step
   number so a resume drops the same steps. About 40 lines plus two cache helpers. H3 keeps its own.
7. **Start at a smaller size** [S] `train.py:1369-1399`, `:2584-2600`: the first 20 / 30 / 50% of the epochs at a
   smaller Target Megapixels from a second dataset config and cache, blended over a tenth of the run. About 70 lines
   plus the launcher's second cache stage. Standard LoRAs only.
8. **Preview and save schedules** [S] `preview_due` (`train.py:390-396`) with `sample_until_epoch`,
   `sample_every_n_epochs_after`, `save_until_epoch`, `save_every_n_epochs_after`. About 30 lines plus two controls.
9. **Resume takes the Learning Rate from the form (#214)** [S] `training/resume_lr.py` (43 lines) and
   `train.py:1888, :1901-1903`; the state now stores `learning_rate`. With Adaptive LR the saved rate is clamped into
   the new Min / Max. Fizgig also lets Resume take block swap from the form (#208) [M]; PORT_PLAN's technical call
   ("keep the saved precision and block swap") is about Auto planning differently on another card, not a user edit.
10. **Live preview changes carry steps, CFG, Turbo strength, cadence and the preview-model tick** [S]
    `train.py:274-305`, `:2089-2278`. A Turbo strength set mid-run loads the Turbo LoRA at the next preview. The
    negative prompt is now always encoded unless the family samples CFG-free (`train.py:1646-1659`), so raising CFG
    mid-run cannot kill previews. About 200 lines. TagScribeR's override carries prompt, seed and size only
    (`training/train.py:352-364`).
11. **A preview-only "look" LoRA** [S] `train.py:46`, `:1712-1739`, description fields `preview_look_*`
    (`families/description.py:480-493`). Used by MOSS-Image only. About 60 lines.
12. **Compile for any sequence length** [S] `families/compile.py:33-48, :266-268, :277-278` (a forward pre-hook marks
    every axis that carries the sequence length dynamic), 18 lines, plus Krea 2 keeping the padded length under
    compile (`krea2/attention.py:48, :74-79`; `krea2/driver.py:52-61`). Fizgig: 81 recompiles in the first 26 steps
    became 4 on a MOSS run with DOP [M]. Uses `torch._dynamo.maybe_mark_dynamic`; inductor has not been run on the
    owner's card, and the call was not checked against the pinned ROCm torch 2.12.
13. **The step's loss is read one step late** [S] `train.py:2467-2504`, `:2922-2936`: loss, per-picture losses and
    grad norm are copied to the CPU without blocking (`torch.cuda.Event`) and read at the next step. About 60 lines.
    Pure speed; the recorded numbers are the same. Not checked on ROCm.
14. **LoKR forward as two large matrix multiplies** [S] `networks/lora.py:353-367`: "4-6x faster", same result
    (Fizgig, on a 5090). Three lines; TagScribeR has the old form at `training/lora.py:65-72`.
15. **Gradient checkpointing "auto"** [S] `quant.py:308-396`, `train.py:1500-1522`, `compile.py:263`
    (`gc_off_blocks`), per-family `train_memory_nogc` / `train_memory_partial`. Only on cards of about 40 GB and up
    (`NOGC_MIN_CARD_GB = 38.0`), rank up to 32, LoRA only. About 150 lines plus figures. No effect on the owner's
    card; cannot be run here.
16. **Krea 2: low-RAM load and ComfyUI int8 / ConvRot / fp8 builds as the DiT (#273)** [S] `quant.py:94-146`
    (`_stream_load`: a Linear at a time when the bf16 file does not fit free RAM), `krea2/driver.py:82-100`
    (`stream_dit`, `dit_gb`), `krea2/utils.py` (+136 lines, `ComfyDecodeReader`, `dit_layout`). Fizgig: peak RAM about
    2 GB instead of 25 GB [M]. Interacts with TagScribeR's own fp8 base (`training/modules/fp8.py`): port with care.
17. **Qwen Image 2.1 trains transparent pictures with their alpha** [S] `qwen_image21/driver.py` (encode keeps a
    4-channel picture, decode returns RGBA when the render is see-through), `families/cache.py:62-63` (`alpha` in the
    cache metadata).
18. **Prompt-slider anchors (#252)** [S] `train.py:558-589`: prompts the dial must leave alone, one a step in turn,
    with a weight. About 60 lines.
19. **Family description fields** added for the above [S `families/description.py`]: `batch_training`, `batch_gb`,
    `batch_auto`, `batch_gb_nogc`, `train_memory_small`, `auto_precisions_small`, `train_memory_nogc`,
    `train_memory_partial`, `nogc_uncompiled`, `own_plan`, `masked_training`, `rgba`, `caption_dropout`,
    `resolution_stages`, `detail_crops`, `training_maps`, `trigger_attention`, `not_offered`, `start_lr_bands`,
    `kind_values`, `value_moves`, `small_dataset_hint`, `sample_prompt`, `preview_small_card_cap`,
    `preview_small_card_option`, `samples_turbo_tick`, `preview_look_*`, `preview_labels`,
    `checkpoint_speed_except`, `text_cache_trim`, `workbench_checkpoint_follows_tick`; `ModelFile.fetch_on_use`,
    `same_as`, `signature`; `SpeedLoRA.gallery_note`; `LoRAFormat.save_fused`; `FamilyOption.batch_off`, `split`,
    `slide`, `single`. Driver hooks: `begin_epoch`, `training_loss_batch`, `stack_conds`, `map_target`, `dit_gb`,
    `ft_load_mp`, `nogc_load`, `plan_pictures`, `export_lora_state_dict`, the trigger-map hooks;
    `frozen_file_added` returns an int.

Touching shared code but owned by the other audit: training maps (`training/train_maps.py`, 329 lines, called at
`train.py:2687-2704` and `:3026-3027`), helpers and training share (`train.py:308-388`, `:2411-2459`, `:2584-2780`),
sub-folders (`item_prefix` keys through the loss watch, `loss_logger.py:139-152`), detail crops.

## 4. Not applicable, already done, or TagScribeR keeps its own

- **Turbo previews at the family's strength (9.2.2)**: already done. `training/pipeline.py:802-809` passes the family
  default whenever it differs from the LoRA's own, with a comment naming Fizgig's bug.
- **A folder refilled with other photos under the same names (8.7.6)**: already done. `training/cache.py:45-51,
  :92-105` stamps `mtime_ns:bytes`.
- **Removal of the INT8 backward option and `apply_int8_training`**: never exposed in TagScribeR
  (`training/quant.py:83` hard-codes `"bf16"`). The dead branch and function remain in
  `training/modules/int8_train.py:114-131, :146-188`; removing them mirrors upstream and changes nothing.
- **Weight-only INT8** (`modules/int8_train.py:109-114`): used by Vimoo only.
- **Fine-tune previews on the trained model, `_record_prompts`, the state-save retry**: ported in wave 2.
- **Kromoss detection (`base_check.py`)**: `checkpoint_takes_speed` was added for the day Kromoss was a preview model
  and is now dead for MOSS-Image (`checkpoint_speed_except` is unset at the target).
- **EMA**: no EMA code changed upstream in this range (no EMA file in the diff; the trainer's EMA calls are unchanged).
- **Timestep sampling of Krea 2, Qwen, Klein, SDXL, Anima, H3**: unchanged [S: no `_sample_t` hunk in their diffs].
- **Batch size above 1**: TagScribeR keeps its own for Krea 2, Klein, SDXL, Anima (PORT_PLAN); upstream's return is a
  new capability (section 3 item 4), not a reason to remove it.
- **Helpers, training share, Move a run, pods, Vast.ai, RunPod API, themes, News card, numbered prompt list, Run
  summary, table view of captions**: not training core.
- **H3 clips**: checkpointing rules for clips, TREAD, reference distillation fix, the Auto plan reading only H3's own
  caches (commit 09717e70; TagScribeR does not use upstream's H3 Auto plan yet, take the fix with it).
- **Python 3.10 `tomllib` fix, container RAM limit, torch.compile cache on a pod volume**: not applicable.
- **`text_encoder_builds`, `fetch_on_use`, `same_as`, `signature`, `hf_public` (retry a public download without an
  expired login)**: download plumbing; TagScribeR has its own downloader. The expired-login retry is worth a look in
  the inference / install audit's area.
- **`FTSpec.file_layout`'s fifth field and `LoRAFormat.save_fused`**: Fibo 2.0 only.
- **Auto-recaption task picker, sub-folder triggers, caption tasks for objects / concepts**: captioning UI, out of
  scope here; the loss-watch signature changes (`families/loss_watch.py:67-70`) come with sub-folders.

## 5. The full fine-tune (`families/ft.py`, 829 lines at the target; diff +54 / -28)

- **Optimizer choice (#312)** [S] `training/optimizers.py:55` (Adafactor back in the catalog at a fixed rate:
  `relative_step=False, scale_parameter=False, warmup_init=False`, `:292-296`), `:83-91`
  (`FT_OPTIMIZERS = {"adafactor": 0.0, "adamw8bit": 1.0, "pagedadamw8bit": 1.0, "ademamix8bit": 1.5,
  "pagedademamix8bit": 1.5, "lion8bit": 0.5, "adamw": 2.0}`: GB of optimizer state per GB of window weights),
  `families/ft.py:638-661` (`make_optimizer(params, lr, name)`), `:748` and `:767-777` (the planner adds the state
  factor to the window factor), `families/train.py:989-1014` (a LoRA optimizer picked for a fine-tune trains with
  Adafactor and says so; a backend that plans its own windows, H3's, keeps Adafactor), `:1133`.
  `families/registry.py:22-33` rewrites every built-in Fine-tune preset to `adafactor` with empty args.
- **Free-each-gradient takes the step's LR multiplier** (F13).
- **Batch size for fine-tunes (9.0.6)** [S] `train.py:989-990, :1013, :1752-1754`: the pictures past the first are
  taken off the free memory before the windows are sized; a regularisation picture keeps its own multiplier inside a
  batch (`loss_watch.py:185-192`).
- **`ft_load_mp(group)`** (`families/driver.py:471-476`): the megapixels the planner uses, overridable by a video
  family.
- **MOSS-Image fine-tune counts rotations for its schedule** (section 2.2) and continues it across a resumed
  fine-tune (`train.py:2706`).
- Unchanged: `FTSpec` defaults, the rotation logic, the streamer, the refusal of pre-quantised sources.

TagScribeR: `training/ft.py` (848 lines, uncommitted) has `make_optimizer(params, lr)` at `:679`, `FusedSteps(lr)` at
`:694`, `plan(...)` without `state_factor` at `:789`; `training/train.py:707-708, :749` call them the old way. Port
the #312 hunks before the fine-tune is committed, so it is not committed in a state the next audit must undo.

## 6. Packaging

| Item | Fizgig 9162fb20 | Fizgig c11134a2 | TagScribeR now | Applies to |
|:--|:--|:--|:--|:--|
| torch (CUDA) | 2.10.0, cu128 | **2.15.0, cu134** (`requirements.txt:7-9`) | 2.10.0, cu128 (`tools/install.py:52-54`) | NVIDIA only |
| torchvision | 0.25.0 | 0.29.2 | 0.25.0 | NVIDIA only |
| NVIDIA driver | 555+ / 550+ | 580+ (CUDA 13) | not stated | NVIDIA only |
| bitsandbytes | 0.48.2 | **0.50.2** ("0.48.2 has no CUDA 13.4 binary") | `>=0.48` on CUDA (`tools/install.py:275`); 0.50.2.dev0 community wheel on AMD Windows (`:62-63`) | the pin change: NVIDIA |
| triton-windows | `>=3.5.1,<3.7` | `==3.7.0.post26` | `>=3.5.1,<3.7` (`requirements.txt:34`) | NVIDIA Windows |
| Python | 3.10+ | **3.11+** (`install_fizgig.py`, `MIN_PYTHON_VERSION`) | targets 3.12 | none |
| New package | - | `ai-edge-litert>=2.2,<3` (Make masks) | - | the other audit |
| Telemetry | - | `ORT_DISABLE_TELEMETRY=1`, `HF_HUB_DISABLE_TELEMETRY=1` (`src/fizgig/__init__.py`), `onnxruntime.disable_telemetry_events()` (`utils/ort_telemetry.py`) | `HF_HUB_DISABLE_TELEMETRY` set (`core/hardware.py:84`); nothing for onnxruntime | every platform |
| ROCm install | own script | unchanged (`filter_requirements_rocm.py`: only the cu128 -> cu134 comment) | pinned ROCm wheels | AMD: nothing to do |

Points to keep straight:
- Fizgig says AMD ROCm installs "keep their own PyTorch, so nothing changes there" [M], and its diff confirms the
  ROCm path was not touched [S].
- TagScribeR's `triton-windows` pin sits in the shared `requirements.txt`, so it also lands on AMD Windows installs.
  Moving it to `==3.7.0.post26` for torch 2.15 must not drag the ROCm torch 2.12 install along: the pin needs to
  become per backend in `tools/install.py`.
- Two upstream defaults lean on torch 2.15 and should not be copied onto older torch blindly: MOSS-Image's
  `compile_boundary="inside"` (`moss_image.py:201`) and the claim that the bf16 LoRA path removes the extra memory.
  Fizgig's commit says torch 2.10 "kept 7-10 full-width activations per block for the backward and 2.15 keeps 2"
  [M]. On torch 2.10 (TagScribeR's CUDA pin) and 2.12 (ROCm) the old "outside" boundary is the safe one until
  measured.
- onnxruntime (TagScribeR's WD tagger) on Linux from 1.29 "report[s] a device ID and machine details" unless the
  variable is set [M]. TagScribeR pins `onnxruntime>=1.20` with no switch. A two-line privacy fix.
- Fizgig's 9.2 figures (Krea 2 22.9 -> 14.9 GB with checkpointing; MOSS at batch 3 21.7 -> 7.1 GB; 11-23% faster)
  are its own, on NVIDIA, on torch 2.15 [M].

## 7. Checklist: what would silently ruin or change a real run if ported carelessly

1. Importing a Fizgig 9.0.8+ preset before Muon exists here (F2): the run trains on plain AdamW.
2. Porting Muon without `MUON_DEFAULT_ARGS` exactly as above: `torch.optim.Muon`'s own default is
   `adjust_lr_fn=None` ("original" scaling) and `weight_decay=0.1` in `BatchedMuon.__init__` (`muon.py:71-73`);
   Fizgig's presets rely on `match_rms_adamw` and 0.01 to keep AdamW's rates.
3. Porting MOSS-Image's `30:7:60:8:3` without `begin_epoch` (the share would stay at its first value) or without the
   half-rate `step_policy`.
4. Taking MOSS-Image's Turbo recipe (CFG 2, strength 0.6) without the new Turbo file: the old SFT-distilled LoRA was
   tuned for CFG 1.
5. Mixing the bf16 LoRA path (2.5) into the same change as the optimizer switch: two numerics changes, one run.
6. Flattening transparent PNGs or applying EXIF rotation without invalidating old latent caches (F6, F7).
7. Changing the loss watch and Adaptive LR (F9, F10) and calling Krea 2 "verified" before the owner's next run.
8. Adopting upstream's batch path alongside TagScribeR's own without deciding which one a given family uses.

## Open questions (could not be determined by reading)

1. Does `torch._dynamo.maybe_mark_dynamic` exist and behave on the pinned ROCm torch 2.12, and does
   `torch.cuda.Event` work on HIP? Needed for items 12 and 13 of section 3.
2. Do Muon's batched bf16 Newton-Schulz multiplies (`torch.baddbmm` on stacks of small matrices) run correctly on
   AMD ROCm? PyTorch's built-in attention is already known to be wrong there for wide heads; nobody has run Muon on
   this card.
3. Can TagScribeR's `FamilyLoRA.add_file` wrap H3's `video_patch_proj` / `audio_patch_proj` / output heads (F11)?
4. Does TagScribeR's window-only H3 LoRA with block swap go through `ModelOffloader`'s backward hooks (F3)?
5. Does a resume in TagScribeR rebuild caches deleted while paused (#210)? The pipeline skips caching on resume.
6. Does TagScribeR's SDXL prompt slider at CFG above 1 have the "negative on the CPU" bug (its own
   `_render_slider_bank`)?
7. Do upstream's batched draws (noise, then t, per picture) match the order of TagScribeR's own batching, i.e. would
   a batch of N reproduce N batch-1 steps here?
8. What do Target Megapixels 1.0 and the new 1.05 resolve to in TagScribeR's own bucket code, compared with
   Fizgig's 992 x 992 and 1024 x 1024?
9. Fizgig's release notes say every existing optimizer "behaves exactly as before" after #292; only the dispatch and
   the args guard were read, not a line-by-line comparison of the eps-floor and fused paths.
10. The Krea 2 preview Turbo LoRA name differs between the two projects' descriptions (unchanged in this range);
    earlier audits own that.

## Questions for the owner

Product decisions only; each has a recommendation. Technical and package calls (torch pins, where the triton pin
lives, port mechanics) are the lead's and are listed under the port order instead.

1. **Muon as the default optimizer of the built-in presets.** Fizgig moved every picture preset to Muon; your
   verified Krea 2 recipe is AdamW 8-bit. Recommendation: add Muon and move the Fizgig-named presets to it, as Fizgig
   has (rule 1), and keep one "AdamW 8-bit" preset per family so the recipe you have actually run stays one click
   away. The docs say plainly that Muon has not been run on AMD until you have run it.
2. **MOSS-Image at Fizgig's current defaults.** The port is not committed yet. Recommendation: land it straight at
   v9.4.0's values (Slide `30:7:60:8:3`, Fast as the first preset, previews on Base with the Turbo LoRA at 0.6 /
   CFG 2 and the SFT look LoRA, the Turbo Train preset marked experimental) instead of committing 8.7.3's and
   changing them a week later.
3. **Kromoss.** Fizgig withdrew it as an offered download (as a Base it "began to overbake after about 750 steps" in
   its test; as a preview model it was unreliable). TagScribeR's MOSS-Image Base row still offers it. Recommendation:
   remove the download choice and the wording that recommends it, keep any Kromoss file loadable by hand (nothing is
   refused), and say why in the row's help.
4. **A Context LoRA that matches no layer.** Fizgig now stops the run with a message; TagScribeR trains on without
   it. Recommendation: stop, as Fizgig does, for the Context LoRA only. The Turbo LoRA keeps your settled rule (a
   warning, the run continues).
5. **Previews on cards under 20 GB.** Recommendation: mirror Fizgig: full size on every family except Krea 2, plus
   MOSS-Image's "Faster, smaller previews (768 px)" tick. Users with 16 GB cards get slower but correct previews.
6. **Transparent PNGs.** Recommendation: mirror Fizgig (laid on white; Qwen Image 2.1 keeps the alpha). It re-encodes
   the affected pictures' caches once, and a dataset that relied on the hidden colour will train differently.
7. **Batch size.** Fizgig's version now covers Qwen, MOSS-Image and H3 stills and keeps the loss watch on at any
   batch. Recommendation: adopt its driver hooks for those three families and for the loss watch, keep TagScribeR's
   existing batching where it already works, and keep whatever maximum TagScribeR allows today (Fizgig's box stops at
   8).
8. **Which new training feature first.** Candidates in the shared layer: masked training (needs masks; the tools are
   the other audit's), caption dropout, Start at a smaller size, preview / save schedules, Resume with a new learning
   rate. Recommendation: Resume LR and the schedules first (small, no numerics), then caption dropout, then masked
   training together with the mask tools.

## Recommended port order

Sizes are rough upstream lines to port, not effort. Every step is unit-tested on the tiny family only; anything that
touches a verified path waits for the owner's run before a doc calls it verified.

1. **Silent-wrong fixes, no numerics change for a normal run** (about 150 lines). F1 LoKR scale
   (`networks/lora.py:326-351`), F2 args guard (`training/optimizers.py:250-254`), F4 resume with another optimizer
   (`families/train.py:104-105, :194-209`), F5 preview cap (`train.py:1160-1173`, `description.py:666`,
   `krea2.py:195`), F8 tight NF4 (`quant.py:208, :257-259`), F12 Context LoRA (`train.py:1688-1692`,
   `driver.py:494-499`), F11 `_wdtype` (`minimax/model.py`, `moss_image/model.py`), the onnxruntime telemetry switch
   (`src/fizgig/__init__.py`, `utils/ort_telemetry.py`).
2. **Dataset reads** (about 80 lines plus a cache revision). F6 `utils/upright.py`, F7 `utils/image.py` `flatten`,
   `dataset/image_dataset.py` `source_stamp`. Bump `latent_rev` once for both.
3. **Block swap with a block subset** (25 lines). F3, `modules/offloading.py`, keeping TagScribeR's CUDA guard.
4. **Optimizers** (about 800 lines, mostly vendored). `training/muon.py`, `training/prodigy.py`,
   `training/automagic4.py`, `training/optimizers.py`; `owns_its_rate` handling in `train.py:1849-1859`; Help Center
   text and tooltips; `THIRD_PARTY_NOTICES.md`. Then the preset change of section 2.1 per the owner's answer 1, and
   H3 Slider on AdamW.
5. **Loss watch and Adaptive LR** (about 90 lines). F9, F10, the learned ceiling: `training/adaptive_lr.py`,
   `training/loss_logger.py:185, :647`. One commit, flagged for the owner's next Krea 2 run.
6. **MOSS-Image at the target** before the family is committed (about 350 lines over 5 files).
   `families/moss_image.py`, `moss_image/driver.py`, `moss_image/model.py`, `moss_image/embedder.py`, the shared
   `begin_epoch` hook, the preview look LoRA (`train.py:1712-1739`), `preview_small_card_option`. Keep
   `compile_boundary="outside"` unless the torch in use is 2.15 (section 6).
7. **Fine-tune at the target** before it is committed (about 120 lines). `families/ft.py:638-686, :748-777`,
   `train.py:956-1014, :2474-2482, :2908-2909`, `registry.py:22-33`, F13.
8. **Small trainer features** (about 150 lines): `training/resume_lr.py` and its call, `preview_due` and the four
   schedule arguments, s/it without between-epoch work, the measured warm-up note, slider presets previewing every
   5 epochs, `empty_cache` before previews.
9. **The bf16 LoRA path and the two-GEMM LoKR forward** (30 lines): `families/lora.py:113-131, :190-193`,
   `networks/lora.py:353-367`. Alone in its commit; needs a real run.
10. **Compile for any sequence length and the one-step-late loss read** (about 100 lines): `families/compile.py`,
    `krea2/attention.py`, `krea2/driver.py:52-61`, `train.py:2467-2504`. After open question 1 is answered.
11. **Batch size per Fizgig** for Qwen, MOSS-Image, H3 stills and the loss watch (about 300 lines):
    `families/driver.py:37-63, :259-283`, each driver's `training_loss_batch` / `stack_conds`,
    `families/loss_watch.py:179-204`, `train.py:1292-1334`, `quant.py:276-280`, `modules/sdpa.py:112-120`.
12. **Caption dropout, then masked training's loss side, then Start at a smaller size** (about 400 lines with
    `dataset/masks.py`), each with its Train tab control and help text.
13. **Live preview changes (steps, CFG, Turbo strength, cadence, preview-model tick)** (about 200 lines):
    `train.py:274-305, :2089-2278`.
14. **Krea 2 low-RAM load and ComfyUI int8 / ConvRot builds** (about 250 lines): `quant.py:61-146`,
    `krea2/driver.py:82-100`, `krea2/utils.py`.
15. **NVIDIA stack and big-card planning**, unrunnable here: the CUDA pins of section 6 with the triton pin split per
    backend, then gradient checkpointing "auto" and Batch size Auto (`quant.py:283-396`, `train.py:593-617,
    :1500-1522`). Documented as not run on real hardware by this project.

Lead's technical calls recorded with this audit (decide and note in PORT_PLAN, do not ask the owner): whether the
CUDA install moves to torch 2.15 / cu134 now or stays on 2.10 until an NVIDIA user reports back; moving the
`triton-windows` pin out of the shared requirements; the cache-revision mechanism for steps 2 and 6; dropping the dead
INT8-backward code.
