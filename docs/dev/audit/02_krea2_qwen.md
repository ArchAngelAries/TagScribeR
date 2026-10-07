# Audit 02 - Krea 2 and Qwen Image 2.1: TagScribeR vs current Fizgig (v7.0.1)

Read-only audit. Fizgig paths are relative to `.claude/fizgig-src/`; TagScribeR paths are relative to the repo root.
"F:" = Fizgig, "T:" = TagScribeR. Nothing was run; no weights were loaded. The report is complete; items that were
not verified say "not checked" and are collected at the end.

Order of the parts (written Krea 2 first): 0 headline facts - Part A Krea 2 (A1-A9) - Part C Krea 2 port plan and the
check of commit d10022d - Part D TagScribeR-only fixes to preserve - Part B Qwen Image 2.1 (B1-B9) - questions for the
owner - what was not checked.

## 0. Headline facts (established first)

1. **Krea 2 upstream moved onto the driver layer (v6.8.0).** `src/fizgig/krea2/trainer.py` is gone; the family is
   `src/fizgig/families/krea2.py` (description) + `src/fizgig/krea2/driver.py` (driver). TagScribeR's
   `training/families/krea2/` is a port of the deleted standalone trainer onto its own driver layer.
2. **`arch_id` changed: F `krea2drv` (families/krea2.py:34) vs T `krea2` (description.py:40).** Upstream says the
   cache layout differs from the original trainer's, "so its caches are rebuilt" (families/krea2.py:1-4,
   RELEASE_NOTES_v6.8.0 "Your dataset is re-cached once").
3. **fp8 base precision was removed upstream for Krea 2.** F `precisions=("int8", "nf4", "bf16")`,
   `auto_precisions=("int8", "nf4")` (families/krea2.py:91-92). T `precisions=("bf16", "fp8", "int8", "nf4")`,
   `auto_order=("int8", "nf4", "fp8")`, `auto_swap_order=("fp8",)` (description.py:99-104).
   An fp8-scaled RAW **file** is still accepted upstream: `load_krea2_dit` keeps its `is_prequantized_fp8` branch
   (krea2/utils.py:129-157) and the generic quantiser dequantises fp8 sources (`modules/nf4.py:43-49`,
   families/quant.py:46, 62). So the owner's file still loads and still feeds INT8 / NF4. What is gone is fp8 as a
   *selectable base* and as Auto's fallback; with precision "bf16" an fp8-scaled file is simply left fp8
   (families/quant.py:35-36 returns before touching the weights).
4. **INT8 scales: F bf16, T fp32.** F `Krea2Driver.int8_fp32_scales = False` (krea2/driver.py:36-38) makes
   `families/quant.py:44-48` compute the per-row scale and the rounding in bf16 ("exactly as the original trainer's
   apply_int8_training"). T `training/quant.py:67-71` always does `.float()`; there is no `int8_fp32_scales` anywhere
   in `training/` or `tests/`. This changes the INT8 weights of the owner's verified run (see the port plan, group a).
5. **Fast FT is dead upstream.** No GUI key (`KREA2_FAST_FT` does not occur anywhere in fizgig-src), no caller passes
   `fp8_fast=True`; only the parameter and `_fp8_fast` code remain (krea2/utils.py:96, 179, 182, 203, 239;
   krea2/fp8_optimization_utils.py:383-388, 513). See the commit verdict.
6. `recommend_krea2_strategy` / `estimate_krea2_peak` / `swap_for_budget` (utils/capabilities.py:254-292, 500-626)
   have **no caller left** in fizgig-src: Krea 2's Auto plan is now `families/quant.py:129-172 plan()` over
   `train_memory`.

---

# PART A - KREA 2

Files compared. F: `src/fizgig/families/krea2.py` (181 lines), `src/fizgig/krea2/driver.py` (313),
`krea2/utils.py`, `krea2/sampling.py`, `krea2/model.py`, `krea2/embedder.py`, `krea2/attention.py`,
`krea2/vae.py`, `krea2/vae_loader.py`, plus the generic layer Krea 2 now runs on: `families/train.py`,
`families/quant.py`, `families/cache.py`, `families/launch.py`, `families/compile.py`, `families/description.py`,
`families/lora.py`, `utils/capabilities.py`. T: `training/families/krea2/*`, `training/train.py`, `training/quant.py`,
`training/cache.py`, `training/optimizers.py`, `training/presets.py`, `training/params.py`.
Not read in full: F `krea2/offloading.py`, `krea2/rotation.py`, `krea2/lora_utils.py`, `krea2/fp8_optimization_utils.py`,
`krea2/safetensors_utils.py`, `families/ft.py` (grep only); T `training/modules/fp8.py`, `training/lora.py` (grep only).

## A1. Training objective and timestep sampling

**Objective: the same formula, different dtype order (not bit-identical).**

| | Fizgig | TagScribeR |
|---|---|---|
| Formula | `noised = (1 - t) * latent + t * noise`, target `noise - latent`, unweighted `F.mse_loss(pred.float(), target.float())` - `krea2/driver.py:168-190` | same formula - `training/families/krea2/driver.py:147-169` |
| Mixing dtype | latent, noise and t are cast to **bf16 first**; the mix and the target are computed **in bf16** (`driver.py:173-178`: `latent = latents.to(dtype=DTYPE)`, `noise.to(dtype=DTYPE)`, `t_ = ...to(DTYPE)`, `target = patchify_block(noise - latent)`) | mix and target computed **in fp32** (`driver.py:155-163`: `x0 = latents.float()`, fp32 `noise`, `t4 = t_bf.float()`, `target = noise - x0`), then `noised.to(torch.bfloat16)` for the DiT |
| t handed to the DiT | `t.to(DTYPE)` (`driver.py:166`) | `t_bf = t.to(torch.bfloat16)` (`driver.py:160`) - same value |
| Autocast | DiT call wrapped in `torch.autocast(device, dtype=bf16)` (`driver.py:165-166`) | no autocast (`driver.py:167`) |
| RNG order | `noise = torch.randn(latents.shape, generator)` then `t` (`driver.py:198-199`) | noise then t (`driver.py:156-158`) - same order |

The T header says so itself: "The loss runs in fp32 around a bf16 forward instead of Fizgig's bf16 mixing arithmetic"
(`training/families/krea2/driver.py:6`). Upstream's docstring says its arithmetic is "the original compute_loss's
arithmetic, in the same dtype order" (`krea2/driver.py:169-170`), so this is a departure T made from the original
trainer, not something upstream changed since. Effect: the target differs by the bf16 rounding of `noise` and of
`noise - latent` (relative ~4e-3 per element at worst), the noised input by rounding order only. Small, but it is in
every step.

Whether the missing autocast matters depends on the dtype of the non-quantised layers after loading. F keeps them in
the file's dtype on the pre-quantised path (`krea2/utils.py:146` `dit_weight_dtype=None`) and relies on autocast;
T's loader passes a dtype to `fp8.load_state_dict(dit, sd, dtype)` (`model.py:388`). **Not checked**: what
`training/modules/fp8.py` does with non-fp8 tensors, and what dtype the owner's file stores them in.

**Timestep sampling: identical.** F `Krea2Driver._sample_t` (`krea2/driver.py:140-149`) and `_mu`
(`driver.py:28-31`): `shift = exp(mu)`, `mu` linear through (256, 0.5) and (6400, 1.15) in image tokens,
`t = sigmoid(randn(1))`, `t = t*shift / (1 + (shift-1)*t)`, an optional window rescales (`lo + t*(hi-lo)`), never
clamps. T `sample_krea2_timesteps` (`training/families/krea2/sampling.py:119-135`) is the same arithmetic.
One difference: F draws **one** t per step (`torch.randn(1)`), T draws `bsize` - identical at batch 1.

**Batch size.** F refuses anything but batch 1 for every driver family: `families/train.py:825-828` ("trains at batch
size 1 here"). T allows batch > 1 for Krea 2 (`driver.py:95` `supports_batching = True`, `training/train.py:329-339`).
The original standalone trainer allowed it (`utils/capabilities.py:238` `_BATCH_GB_PER_IMAGE = 2.4` is its measurement).
See A9.

**Gradient checkpointing gate.** F `self.gradient_checkpointing and self.training` (`krea2/model.py:487`);
T `torch.is_grad_enabled() and self.gradient_checkpointing` (`training/families/krea2/model.py:359`). Same in a
training step; they differ only for a no-grad forward on a model left in train mode (T skips checkpointing, F does not).

**Text gather / trim.** Identical: `gather_valid_text` rounds the padded length to the trim multiple (64) on both
sides (F `krea2/sampling.py:30-73`, T `sampling.py:36-52`); the combined sequence pads to a multiple of 256 on both
(F `krea2/model.py:463-468`, T `model.py:340-344`).

**Gradient accumulation.** Both divide the loss by N and step every N; both flush a partial group at the epoch end
(F `families/train.py:1123-1130, 1412-1425`; T `training/train.py:618-629`). The scheduler runs in optimizer updates
on both. Same.

**EMA / adaptive LR start.** Both start Adaptive LR at `sqrt(min*max)` (F `train.py:1076-1081`, T `train.py:455-460`)
and update EMA after the clipped step (F `train.py:1329-1330`, T `train.py:587-588`). F also offers a "Short run" EMA
mode (`train.py:1085-1093`) for families with `ema_short_run` - Krea 2 does not set it. Not compared: the internals of
`ema.py`, `adaptive_lr.py`, `loss_watch.py` (generic layer, outside this audit's scope).

## A2. Auto precision / memory plan

**Upstream now** (`families/krea2.py:91-92, 120-126`; planner `families/quant.py:129-172`, margin 1.5 GB):

```
precisions      = ("int8", "nf4", "bf16")
auto_precisions = ("int8", "nf4")
train_memory    = {"int8": (((0.25, 16.2), (1.0, 19.1)), 0.42),
                   "nf4":  (((0.25, 11.4), (1.0, 13.4)), 0.0),
                   "bf16": (((0.25, 26.0), (1.0, 28.9)), 0.84)}
```

Ladder: INT8 no swap -> NF4 no swap -> INT8 with the fewest swapped blocks that fit (`quant.py:159-164`; the swap
candidates are `("int8", "bf16")` filtered by `auto_precisions`, so INT8) -> NF4 as "the smallest base"
(`quant.py:165-166`). bf16 is manual only. fp8 is not in the list at all.

**TagScribeR** (`training/families/krea2/description.py:99-117`; planner `training/quant.py:180-243`):

```
precisions      = ("bf16", "fp8", "int8", "nf4")
auto_order      = ("int8", "nf4", "fp8")
auto_swap_order = ("fp8",)
train_memory    = {"fp8":  (((0.25, 18.7), (1.0, 18.9)), 0.42),
                   "int8": (((0.25, 16.2), (1.0, 16.4)), 0.42),
                   "nf4":  (((0.25, 11.4), (1.0, 11.6)), 0.0)}
```

Differences:

| Item | Fizgig | TagScribeR | Effect |
|---|---|---|---|
| INT8 at 1 MP | 19.1 GB | 16.4 GB | T under-plans by 2.7 GB. Upstream: "The original's +0.25 GB/MP under-plans 1 MP: INT8 + 16 swapped blocks ran out on a 12 GB card" (`families/krea2.py:120-124`), and the v6.7.2 notes. |
| NF4 at 1 MP | 13.4 GB | 11.6 GB | T under-plans by 1.8 GB |
| bf16 | 26.0 / 28.9 GB, 0.84 GB per swapped block (measured on the driver) | no entry ("Fizgig never measured it", `description.py:113-114`) | upstream has since measured it |
| fp8 | not a precision | 18.7 / 18.9 GB, the swap base | removed upstream; upstream's last fp8 slope was 5.1 GB/MP (`utils/capabilities.py:239`), so T's 18.9 at 1 MP is ~3.6 GB light even by the old trainer's final figures |
| Auto's swap base | INT8 (0.42 GB/block) | fp8 | different fallback on small cards |
| Auto when nothing fits | NF4, swap 0 (`quant.py:165-166`) | fp8 + max swap (`training/quant.py:225-228`) | different |
| INT8 / NF4 availability | not checked by the planner (the old `caps.int8_matmul_train` / `caps.bitsandbytes` gates in `utils/capabilities.py:589, 596` are dead code now) | probed: `training/quant.py:105-128, 195-201`; INT8 falls back to fp8 (`quant.py:233-237`) | T keeps a guard upstream dropped. **Not checked**: whether upstream gates this in the GUI before launch. |
| INT8 block-swapped storage | quantised weights kept on CPU, `store_device="cpu" if swap` (`quant.py:31-34, 51, 104`) | quantised and left on the compute device (`training/quant.py:71`) | on a card that needs swap, T must hold the whole INT8 base on the GPU before swap starts |
| INT8 scale arithmetic | bf16 for Krea 2 (`krea2/driver.py:36-38`, `quant.py:44-48`) | fp32 (`training/quant.py:67-69`) | different INT8 weights - see port plan (a) |

**The owner's setup (20 GB card, 0.25 MP presets, Auto):** budget = free - 1.5 GB. INT8 needs 16.2 on both sides, so
both pick INT8, swap 0, whenever at least 17.7 GB is free. Same result at 0.25 MP. At 1 MP upstream needs 19.1 + 1.5
= 20.6 GB free and would fall to NF4 on a 20 GB card (v6.7.2: "a card with about 19 GB free now uses NF4 instead of
INT8"), while T would still pick INT8 (16.4 + 1.5) and, by upstream's measurements, run out of memory.

**The old `recommend_krea2_strategy`** is still in `utils/capabilities.py:500-626` with updated constants
(`_RES_GB_PER_MP = {"nf4": 3.0, "int8": 2.6, "fp8": 5.1}`, `_NF4_HEADROOM_GB = 1.0`, lines 225, 239) but has no caller.
T's figures were taken from its 29 Sept state (0.25 GB/MP).

**Compile decision timing.** F decides on the empty card, before the DiT loads (`families/train.py:870-893`: "the
free VRAM its checks read would be the loaded DiT's leftovers later") and applies after the LoRA is attached
(`train.py:1039-1041`). T decides inside `prepare_training`, after the load (`training/train.py:432-434`,
`training/families/krea2/driver.py:68-76`), and `compile.py:117-119, 152` reads free VRAM at that moment - so the
INT8 fit checks are compared against what is left after the model is resident. On NVIDIA this makes Auto decline
compile where upstream would enable it. No effect on ROCm (Auto is off there on both sides: F
`utils/capabilities.py:697-699`, T `compile.py:149-151`). The rule constants themselves match (90 s warm-up,
0.300 / 0.153 s saved, margin 2.0, 15 GB/MP, outside anchor 1.05 MP: F `capabilities.py:638-666`, T
`compile.py:40-60`). The compile code itself moved upstream to the shared `families/compile.py` (220 lines); T's copy
is `training/families/krea2/compile.py` (400 lines) - **not diffed line by line**.

## A3. Presets

Upstream built-ins (`families/krea2.py:14-29, 168-176`). Common to all: `NETWORK_ALPHA` = rank,
`NETWORK_TYPE` "LoRA (standard)", `SAVE_EVERY_N_EPOCHS` 1, `SEED` 42, `OPTIMIZER_TYPE` "adamw8bit",
`GRADIENT_ACCUMULATION` 1, `MAX_GRAD_NORM` 1.0, `BLOCKS_SWAP` "Auto (detect from GPU)",
`FAMILY_PRECISION` "Auto (fits your free VRAM)", `FAMILY_EMA` "0.98 (recommended)", `KREA2_LOSS_WATCH` True,
`KREA2_PER_IMAGE_LR` True, `KREA2_AUTO_RECAPTION` False, `KREA2_WARMUP_LOOK` False, `FAMILY_SLIDER_GUIDANCE` "3".

| Preset | rank | LR | Adaptive LR (min - max) | epochs | MP | FAMILY_SLIDER | In T? |
|---|---|---|---|---|---|---|---|
| ✨ Krea 2 Ultra Fast (rank 8, adaptive LR) - the default | 8 | 1e-4 | on, 2e-4 - 4e-4 | 30 | 0.25 | False | yes, same values (`description.py:166`) |
| ✨ Krea 2 Standard (rank 32, full model) | 32 | 1e-4 | off (boxes 1e-4 / 4e-4) | 64 | 0.25 | False | yes, same (`description.py:167`) |
| ✨ Krea 2 Style (rank 16, gentle LR) | 16 | 1e-4 | on, 5e-5 - 2e-4 | 64 | 0.25 | False | yes, same (`description.py:171`) |
| ✨ Krea 2 Slider (rank 4, 2e-4) | 4 | 2e-4 | off | 20 | 0.5 | True | **missing** (added upstream) |

Key-level differences in the three shared presets (the values are equal, the keys moved):

- T carries the legacy keys `TARGET_LAYERS` "Full Model", `MIN_TIMESTEP` "", `MAX_TIMESTEP` "", `QUANT_4BIT_MODE`
  "auto", `COMPILE_BLOCKS` "Auto", `KREA2_EMA` "0.98 (recommended)" (`description.py:26-34`). Upstream dropped all six
  from the presets and writes `FAMILY_PRECISION` / `FAMILY_EMA` directly, plus the new `FAMILY_SLIDER` and
  `FAMILY_SLIDER_GUIDANCE`. `training/presets.py:159-178` already maps the legacy keys, so old Fizgig preset files keep
  importing; current Fizgig files import as they are for the keys T has. `FAMILY_SLIDER*` are unknown to T.
- `COMPILE_BLOCKS` is no longer a preset value upstream (it is a settings default "auto", `lora_trainer_gui.py:1480`).
- No preset was removed or renamed. Fine-tune has no built-in Krea 2 preset upstream (choosing Fine-tune sets LR 1e-5:
  `families/description.py:345` `ft_learning_rate`, docs/FINETUNE.md "Learning rates").

## A4. Text encoder, VAE, cache format and cache ids

**Text encoder (Qwen3-VL-4B).**

| | Fizgig | TagScribeR |
|---|---|---|
| Template, `max_length` 512, layers (2, 5, ..., 35), prefix 34 / suffix 5 tokens | `krea2/embedder.py:166-168, 519-522, 544-572` | `training/families/krea2/embedder.py:55-63, 137-152` - identical |
| fp8_scaled file kept fp8 | `embedder.py:205-242` | `embedder.py:75-98` (`training/modules/fp8.py:140-164`) - same intent |
| Captions per forward at cache time | **one** (`krea2/driver.py:120-128`: "a batched forward rounds a bf16 step or two differently, so a caption's conditioning would depend on which captions shared its batch") | **chunks of 8, batched** (`training/cache.py:139-145` -> `driver.py:141-144` -> `embedder.py:137-152`) |
| Model call | `self.qwen(...)` the LM wrapper (`embedder.py:566`) | `self.model.model(...)`, LM head dropped (`embedder.py:90, 150`) - same hidden states |
| Vision path (reference image, captioning) | present: `_forward_with_images` (`embedder.py:617-646`), `generate_caption` (`embedder.py:448`) | not ported (`embedder.py:4`) |
| Tokenizer source | 1. `qwen3vl_tokenizer/` next to the checkpoint, 2. **the copy bundled with Fizgig** (`src/fizgig/assets/qwen3vl_tokenizer/`, 8 files, ~11 MB), 3. the Hub, cache first (`embedder.py:66-93, 282-283`) | 1. `qwen3vl_tokenizer/` next to the checkpoint, 2. the Hub id via `AutoTokenizer.from_pretrained` (`embedder.py:66-72, 125-127`), with the files fetched by the model downloader (`description.py:175-178` `helper_files`) |

So T's cached conditioning for a caption is not exactly what Fizgig caches: it depends on the 7 other captions in
its chunk. Upstream calls the difference "a bf16 step or two". This is the one cache-content difference found.

**VAE.** `krea2/vae.py` and `krea2/vae_loader.py` are unchanged upstream apart from what T altered on purpose: the
only functional diff is the attention call (F `krea2/vae.py:373` `F.scaled_dot_product_attention(q, k, v)`, T
`training/families/krea2/vae.py:390` `wide_attention(q, k, v)`) - see section D. Latents: both encode pixels in
[-1, 1] and take the posterior mode, normalised (F `krea2/driver.py:114-118`, T `driver.py:133-138`). F stores the
latent in the VAE's dtype, T casts to bf16 before saving (`driver.py:138`); F casts to bf16 at train time anyway
(`krea2/driver.py:173`), so training sees the same values.

**Cache layout: the same.** Latents `latent_{h}x{w}`, conditioning `cond__hidden_states` / `cond__attention_mask`,
metadata `architecture`, `caption1`, `format_version` "1.0.0" (F `families/cache.py:45-75`, T `training/cache.py:42-51`
and `save_cond`). T adds `latent_rev` (keep it, section D).

**Cache ids.** F `arch_id="krea2drv"`, T `arch_id="krea2"`; the id is part of every cache file name. Upstream changed
the id because its **original trainer's** layout differed from the driver's. T's Krea 2 was written on the driver
layout from the start, so T does not need the rename for correctness. If T adopts `krea2drv`:

- every existing T Krea 2 cache (latents and text) is orphaned and re-encoded once, as upstream's was;
- T's stale-cache sweep globs `*_{arch}*.safetensors` (`training/cache.py:160`): with arch `krea2` the pattern also
  matches `..._krea2drv...` files, so mixing both ids in one cache folder needs a look at `_remove_stale` first
  (**not traced further**);
- `ss_architecture` in saved LoRAs changes from `krea2` to `krea2drv` (F `families/train.py:1161`, T
  `training/train.py:509`). **Not checked**: whether any Fizgig tool keys on that value.

If T keeps `krea2`: existing latent caches stay valid (the encode is unchanged). Text caches would still need one
re-encode **if** the one-caption-per-forward rule is ported, and T has no text-side revision mark to trigger it
(`latent_rev` covers latents only) - one would have to be added, or the owner clears the text cache by hand.
This is a question for the owner (Q1 below).

## A5. LoRA targets, key format, alpha, network types, metadata

| Item | Fizgig | TagScribeR | Same? |
|---|---|---|---|
| Targets | every Linear, 264 modules: 28 blocks x 8 (`attn.wq/wk/wv/gate/wo`, `mlp.gate/up/down`), the text-fusion stack, `first`, `tmlp.0`, `tmlp.2`, `txtmlp.1`, `txtmlp.3`, `tproj.1`, `last.linear`, `txtfusion.projector` (`krea2/driver.py:24-25, 277-293`) | same modules (`driver.py:29-34, 204-223`) | yes |
| Quantised targets | the 28 blocks' Linears only (`driver.py:311-313`) | same (`driver.py:225-228`) | yes |
| Key format | `lora_unet_<path with _>.lora_down.weight` / `.lora_up.weight` / `.alpha` (`families/krea2.py:76-85`, `families/lora.py:211-218`) | same (`description.py:81-93`, `training/lora.py:240-246`) | yes |
| LoKR keys | `diffusion_model.<dotted path>.lokr_w1` / `.lokr_w2` / `.alpha` = 1.0 (`families/lora.py:217, 556-559`) | same (`training/lora.py:245, 551-553`) | yes |
| Alpha | preset alpha = rank; scale `alpha / rank` (`families/lora.py:103`) | same | yes |
| Network types | `("lora", "lokr")` (`families/krea2.py:134`) | `("lora", "lokr")` (`description.py:124`) | yes |
| Adapter dtype / save dtype | trainable fp32, saved bf16 (`families/lora.py:89, 568`; `description.py:361` `trainable_dtype="fp32"`) | fp32 / bf16 (`training/lora.py:140, 562`) | yes |
| Init | kaiming down, zero up; LoKR w1 kaiming, w2 zero (`families/lora.py:48-51, 96-97`) | same (`training/lora.py:110, 137-138`) | yes |
| **Block ids** | `block_0..27`, `txt_lw_0`, `txt_lw_1`, `txt_rf_0`, `txt_rf_1`, `io` (the projector rides in `io`) - `krea2/driver.py:277-293` | `block_0..27`, `txtfusion_layerwise_0/1`, `txtfusion_projector`, `txtfusion_refiner_0/1`, `io_in`, `io_out` - `driver.py:204-223` | **no** |
| Reading diffusers-named Krea 2 LoRAs (OneTrainer, AI-Toolkit) as Context LoRA | `alias_flat` + `_ALIASES` (`krea2/driver.py:295-309`, restored in v6.8.2) | none (no `alias_flat` in `training/`) | **gap** |
| `ss_network_module` | `fizgig.families (krea2, lora)` (`train.py:1157`) | `tagscriber.training (krea2, lora)` (`train.py:505`) | intended |
| `ss_architecture` | `krea2drv` | `krea2` | follows arch id |
| Slider / block metadata | `ss_slider`, `ss_slider_prompts`, `ss_slider_guidance`, `ss_slider_diff_weight`, `ss_train_blocks` (`train.py:1167-1176`), `driver.run_metadata()` (`train.py:1177`) | none | follows the missing features |
| Final file | `<name>.safetensors` **plus a numbered copy** `<name>-<max_epochs:06d>.safetensors` (`train.py:1497-1501`, #176) | `<name>.safetensors` only (`training/train.py:665-666`) | **gap** (small) |
| modelspec | `modelspec_arch="Krea-2"`, implementation `https://github.com/krea-ai/krea-2` | same (`description.py:96-98`) | yes |

The block ids matter as soon as anything names blocks: Ultra mode's `slider_ultra_blocks`
(`families/krea2.py:139`), `--train_blocks`, `ss_train_blocks`, Repair Studio presets (`families/krea2.py:117-119`).
A port of any of those must take upstream's ids.

Automagic's groups are **not** affected by the id change (they match module names, A7).

## A6. Preview engines

| Setting | Fizgig | TagScribeR |
|---|---|---|
| Engines for training previews | the training model, with the Turbo LoRA when its file is set and strength > 0 (`families/launch.py:598-616`, `families/train.py:966-981`). **The Turbo checkpoint is no longer a training-preview option** (v6.8.0 notes; `train_preview_checkpoint` is False for Krea 2, so `--preview_checkpoint` is ignored, `train.py:1255-1257`) | the same two: RAW, or RAW + Turbo LoRA (`description.py:133-161`). No Turbo-checkpoint engine - which now matches upstream |
| Turbo LoRA recipe | 8 steps, CFG 1.0, euler / simple, `mu` pinned 1.15, strength 1.0, loaded unmerged (`families/krea2.py:145-159`) | same (`description.py:133-153`) |
| CFG with the Turbo LoRA on | "the speed LoRA's own CFG, unless the Samples tab asks for more (then the negative applies too)": `_cfg = cfg if cfg and cfg > 1.0 else speed.cfg`, `neg_cond=neg if _cfg > 1.0` (`train.py:422-427`, v6.8.1) | always `speed.cfg`, no negative (`training/train.py:244-246`) |
| Defaults without the Turbo LoRA | `preview_steps=8`, `preview_cfg=1.0` (`families/krea2.py:162-163`); the Samples tab's Steps / CFG boxes are what is sent (`launch.py:651-659`) | `preview_steps=28`, `preview_cfg=5.5`, `preview_speed_steps=8` (`description.py:156-159`) |
| RAW recipe listed | 28 steps, **CFG 4.5** (`families/krea2.py:142-143`) | 28 steps, **CFG 5.5** (`description.py:127-131`, from `sampling.sample`'s default, which is still 5.5 at `krea2/sampling.py:167`) |
| Default negative prompt | `preview_negative=GENERAL_NEGATIVE` (`families/krea2.py:164`, text at `families/description.py:18-19`); sent only when CFG > 1 (`launch.py:660-662`) | the same text as the `SAMPLE_NEGATIVE` default (`training/params.py:311-314`), sent only when CFG > 1 (`pipeline.py:422-424`) |
| Schedule shift | `mu` from the image-token count, (256, 0.5)-(6400, 1.15); pinned 1.15 with the Turbo LoRA (`krea2/driver.py:251-253`, `krea2/sampling.py:110-123`) | same (`sampling.py:94-103, 173-175`) |
| Preview size | 1024 x 1024; capped to 768 on cards under 20 GB, DiT parked for the decode (`train.py:732-736, 434-440`) | same (`training/train.py:316-320, 239-254`) |
| Start noise | drawn **on the GPU in bf16**: `torch.Generator(device=dev).manual_seed(seed)`, `torch.randn(..., device=dev, dtype=bf16)` (`krea2/driver.py:226-231`) | drawn **on the CPU in fp32** (`sampling.py:138-141, 168-169`) |
| Reference image | `preview_image=True` (`families/krea2.py:105`): the Samples tab's picture goes through Qwen3-VL's vision path, `--sample_image` (`launch.py:669-672`, `train.py:923-930`, `krea2/driver.py:130-137`, 1 MP) | none |
| A failed preview | previews go off, training continues, the model is put back exactly (`train.py:1195-1220, 400-470`, v6.8.3) | previews go off, training continues (`training/train.py:548-556`); the restore is a plain `finally` (`train.py:259-273`), not upstream's "undo exactly what was done" set - a failure inside `move_adapter` would skip nothing here but is **not verified** equivalent |
| Slider previews | strip of three at -1 / 0 / +1 (`train.py:323-342, 417-421, 441-448`) | none |
| Thumbnail of the epoch's own preview | `refresh_checkpoint_thumbnail` (`train.py:1471-1476`) | present (`training/train.py:560-561`) |

Consequences for the owner's two preview modes:

- **Turbo-LoRA previews**: same recipe. Two visible differences: the start noise (a given seed gives a different
  picture than in Fizgig), and a CFG above 1 on the Samples tab is ignored by T where upstream now applies it.
- **Base previews**: T renders RAW at 28 steps / CFG 5.5 by default. Upstream's description default for a Krea 2 run
  without the Turbo LoRA is 8 steps / CFG 1.0 unless the user types other values, and its listed RAW recipe is CFG 4.5.
  T's default is the more sensible picture; the mismatch is the 5.5 vs 4.5 (Q4 below).

## A7. Optimizers and parameter groups

- **Offered.** F's description lists `("adamw8bit", "adamw")` (`families/krea2.py:127`), but the GUI fills the
  dropdown for every family from the whole catalogue, `available_optimizers()` (`lora_trainer_gui.py:1548, 6957-6968`;
  catalogue `training/optimizers.py:38-52`): adamw8bit, adamw, pagedadamw8bit, ademamix8bit, pagedademamix8bit,
  lion8bit, automagic3. T lists the same seven (`description.py:122`; `training/optimizers.py:44-57` is the same
  catalogue). Same set. `automagic3.py` differs from upstream by 5 lines (header only, **not read line by line**).
- **Automagic v3 groups for Krea 2.** F: `optimizer_families=(("txtfusion", ("txtfusion.",)), ("attn", (".attn.",)),
  ("mlp", (".mlp.",)))`, the rest "other"; `automagic_sign_window=16` (`families/krea2.py:132-133`), applied by the
  generic `_optimizer_family_groups` on dotted module names (`families/train.py:100-120, 1052-1061`). T:
  `KREA2_LORA_FAMILIES` txtfusion / attn (`_attn_`) / mlp (`_mlp_`), the rest "io", on kohya-flattened names, sign
  window 16 (`training/optimizers.py:96-111`, `training/families/krea2/driver.py:47-66`). Membership is the same on
  all 264 modules (txtfusion first; `txtmlp.*`, `tmlp.*`, `tproj.1`, `first`, `last.linear` fall to the last group on
  both sides). Only the fourth group's **name** differs: "other" upstream, "io" in T. It appears in the per-epoch
  rate line only.
- Stand-downs with Automagic (adaptive LR, per-image LR, look warm-up, the scheduler): same on both
  (F `train.py:1068-1075`, T `train.py:444-454`).
- **New upstream, unused by Krea 2:** `optimizer_weight_decay`, `optimizer_eps_floor_8bit`
  (`families/description.py:358-359`, `train.py:1062-1067`) - both left at their defaults for Krea 2.

**Adaptive LR - one difference that reaches the owner's run** (generic layer, reported here because the owner's
verified setup uses it): upstream counts the gradient-clip ratio as a stability signal **only for a family that sets
`adaptive_lr_clip_signal`** - Klein (`training/adaptive_lr.py:28-36, 131-133`; `families/train.py:1082, 1312-1313`;
`families/description.py:369-371`). Krea 2 does not set it, so on Krea 2 the only stability signal is LoRA
weight-norm growth above 30%. T applies the clip-ratio signal to **every** family whenever clipping is on
(`training/train.py:581-583` `adaptive.note_clip(...)`; `training/adaptive_lr.py:163-164`
`if clip_ratio is not None and clip_ratio > 0.5`). With the presets' `MAX_GRAD_NORM` 1.0, an epoch in which more
than half the steps clip makes T halve the rate and roll back; Fizgig would not. `adaptive_lr.py` differs from
upstream by 118 lines; only this point was traced.

## A8. Features upstream has that TagScribeR's Krea 2 lacks

Size: S = under a day, M = a few days, L = a week or more (including tests, help text and Train-tab controls).

| # | Feature | Fizgig source | GUI key(s) | Size |
|---|---|---|---|---|
| 1 | **Slider LoRAs** (image pairs and prompt pairs) | trainer: `families/train.py:742-769, 304-384, 1004-1038, 1363-1393`; Krea 2 driver hooks `loss_at(diff_ref, diff_weight)` `krea2/driver.py:168-190`, `noise_latents` `:203-209`, `predict` `:211-212`; description `slider_training=True`, `slider_guidance=3.0` `families/krea2.py:135-136`; launch `families/launch.py:568-586`, caches `--slider` `families/cache.py:128-130` | `FAMILY_SLIDER`, `FAMILY_SLIDER_SOURCE`, `FAMILY_SLIDER_DIR`, `FAMILY_SLIDER_CAPTION`, `FAMILY_SLIDER_BASE`, `FAMILY_SLIDER_POS`, `FAMILY_SLIDER_NEG`, `FAMILY_SLIDER_GUIDANCE` (`lora_trainer_gui.py:4198, 5834-5837`) | L (generic layer + dataset pairs + previews; M per further family once it exists) |
| 2 | **Ultra mode** (slider trains blocks 0-7 + the 4 text-fusion blocks) | `slider_ultra_blocks` `families/krea2.py:137-139`; `--train_blocks` `families/launch.py:569-570`, `families/train.py:717-718, 995-1003`; needs upstream's block ids | `FAMILY_SLIDER_ULTRA` (`lora_trainer_gui.py:4339-4342`) | S once 1 and `--train_blocks` exist |
| 3 | **Full fine-tune** (rotating component windows, NF4 trunk, bf16 master, optional disk master, regularisation images, Pause at rotation boundaries, window-size cap) | `families/ft.py` (775 lines), `families/train.py:571-681, 771-801, 984-993, 1339-1341, 1439-1459, 1488-1496`; Krea 2 `ft_spec` `krea2/driver.py:40-47` (`components=("attn", "mlp.gate", "mlp.up", "mlp.down")`, `always_on=("txtfusion",)`, `overhead_gb=9.5`, `trunk_gb_per_block=0.217`, `stream_base_gb=3.2`, `calib_mp=0.25`, `act_gb_per_mp=3.2`); `finetune=True` `families/krea2.py:112` | `FAMILY_FT`, `FAMILY_FT_ROTATIONS` (10), `FAMILY_FT_SAVE_EVERY` (1), `FAMILY_FT_ROTATE_EVERY` (1), `FAMILY_FT_MAX_PARTS`, `FAMILY_FT_FUSED` (default True), `FAMILY_FT_REG_DIR`, `FAMILY_FT_REG_MULT` (0.2), `FAMILY_FT_CONTINUE` (`families/launch.py:86-108, 488-503`; `lora_trainer_gui.py:4199, 4427`) | L |
| | - accepted checkpoint formats | `source_unfit_reason` (`families/ft.py:140-151`): **refuses** any file with `.weight_scale` / `.scale_weight` keys ("is a pre-quantized checkpoint") or any tensor stored as `F8*`, `I8`, `U8`. Accepts bf16, fp16, fp32 (`ft.py:115-121`). **The owner's fp8-scaled RAW cannot be fine-tuned upstream**; the bf16 RAW (~26 GB) is required. Upstream also says fine-tune is NVIDIA-only / untested on ROCm (docs/FINETUNE.md "What it costs") | | |
| 4 | **Checkpoint to LoRA** (extract a LoRA from a fine-tune) | `diff_to_lora_gui.py`, `families/extract.py` (107 lines) | button on the fine-tune card | M (**not read**) |
| 5 | **Fast FT** | removed upstream - see section C | (`KREA2_FAST_FT` no longer exists) | n/a |
| 6 | **Fast Identity Mode** | Qwen only (`identity_blocks`, B8). Krea 2 sets none: "Krea 2's per-block roles aren't charted yet" (docs/KREA2.md) | `FAMILY_FAST_ID` | n/a for Krea 2 |
| 7 | **Edit LoRAs** | Qwen only. Krea 2 refuses references: `krea2/driver.py:194-195` ("Krea 2 has no edit training") | `FAMILY_EDIT*` | n/a for Krea 2 |
| 8 | **Train only some blocks** (`--train_blocks`, `expand_train_blocks`, per-step block freeze) | `families/train.py:717-718, 995-1003, 1284-1302`, `families/driver.py:115` | `FAMILY_TRAIN_AREA`, `FAMILY_TRAIN_BLOCKS` (families with `train_areas`; Krea 2 has none - only Ultra uses it) | M |
| 9 | **Block map with upstream's ids** + Repair Studio text-fusion presets | `krea2/driver.py:277-293`, `families/krea2.py:117-119` | workbench | S (ids); the workbench itself is out of scope |
| 10 | **Reference image for previews** (vision path) | `preview_image=True` `families/krea2.py:105`; `krea2/driver.py:130-137`; `krea2/embedder.py:574-646`; `families/train.py:923-930`; `families/launch.py:669-672` | Samples tab reference row, `--sample_image` | M (needs the vision tower kept at load and the image processor) |
| 11 | **CFG + negative with the Turbo LoRA** | `families/train.py:422-427` | Samples CFG / Negative | S |
| 12 | **One caption per forward when caching** | `krea2/driver.py:120-128` | - | S (plus a cache-invalidation decision) |
| 13 | **Bundled tokenizer** (offline with no setup) | `krea2/embedder.py:73-93`, `src/fizgig/assets/qwen3vl_tokenizer/` | - | S (11 MB of files + ATTRIBUTION.md; Apache-2.0) |
| 14 | **INT8 scales in bf16** | `krea2/driver.py:36-38`, `families/quant.py:44-48` | - | S |
| 15 | **INT8 kept on CPU for a block-swapped base** | `families/quant.py:31-34, 51, 104` | - | S |
| 16 | **Updated memory figures, bf16 measured, Auto = INT8 then NF4** | `families/krea2.py:91-92, 120-126` | `FAMILY_PRECISION` | S |
| 17 | **Compile decided before the model loads; shared compile module; Qwen compile** | `families/train.py:870-893, 1039-1041`, `families/compile.py`, `krea2/driver.py:49-63` | `COMPILE_BLOCKS` | S-M |
| 18 | **Diffusers-named Krea 2 LoRAs load** (Context LoRA) | `krea2/driver.py:295-309`, `families/lora.py:273` | Context LoRA path | S |
| 19 | **Final epoch also saved under its number** | `families/train.py:1497-1501` | - | S |
| 20 | **Krea 2 slider preset** | `families/krea2.py:172-175` | preset | S (needs 1) |
| 21 | **Turbo DiT model row** (workbench previews only) | `families/krea2.py:52-55`, `preview_checkpoint_sampling` `:106-109` | Preferences `krea2_turbo_dit` | n/a without a workbench |
| 22 | **Captioning / auto-recaption with Qwen3-VL-4B** | `krea2/embedder.py:292-496`, `families/loss_watch.py` | `KREA2_AUTO_RECAPTION`, Captions tab | **not checked** what T's auto-recaption uses instead (T's Krea 2 encoder is encode-only, `embedder.py:4`) |
| 23 | Workbench: Repair Studio, LoRA the Explorer, Profiler (renders), Extract, LoRA Royale, INT8 attention, activation cache | `families/workbench.py`, `families/block_profile.py`, `families/act_cache.py`, `families/krea2.py:110-113` | workbench tabs | L (out of this audit's scope) |

Gradient-checkpointing changes: none in the Krea 2 model itself (`krea2/model.py:483-490` is what T ported). What
changed is around it: the compile boundary "outside" keeps the checkpoint wrapper eager (`families/compile.py:207-215`,
which T has, `compile.py:290-375`), and a fine-tune may turn clipping / accumulation off (`train.py:793-801`).

Compile rules: unchanged for Krea 2 ("Krea 2 behaves exactly as before", v6.8.3); the code moved to
`families/compile.py` and the decision moved before the load (A2).

## A9. What TagScribeR has that upstream removed or changed its mind about

| Item | TagScribeR | Upstream now |
|---|---|---|
| **fp8 as a Krea 2 base precision** and as Auto's fallback / swap base | `description.py:99-104`, `training/quant.py:53-60` | Removed: "Base precision: fp8 is gone. Auto picks INT8, then NF4; INT8 is faster" (v6.8.0); `precisions=("int8", "nf4", "bf16")`. An fp8-scaled file still loads and still feeds INT8 / NF4; chosen with "bf16" it simply runs as fp8 (section 0, item 3) |
| **bf16 on an fp8 file dequantises to bf16** | `training/quant.py:45-50` | no such step: the file's fp8 Linears stay fp8 with the dequantising forward (`krea2/utils.py:129-157`, `families/quant.py:35-36`) |
| **Batch size > 1** | `driver.py:95`, `training/train.py:329-339` | refused for all driver families (`families/train.py:825-828`); Gradient Accumulation is the replacement ("Gradient Accumulation works on Krea 2", v6.8.0) |
| **Mid-run switch to the cuDNN attention backend after epoch 1** | `driver.py:78-91` -> `training/modules/sdpa.py:88-113` | `consider_training_backend` still exists (`modules/sdpa.py:88`) but nothing calls it; training stays on the default SDPA backend |
| **Clip-ratio stability signal in Adaptive LR for Krea 2** | `training/adaptive_lr.py:163-164` | Klein only (A7) |
| **Fast FT** (unmerged commit d10022d) | worktree branch only | removed (section C) |
| **`recommend_krea2_strategy`'s figures** (0.25 GB/MP, fp8 18.7) | `description.py:105-117` | superseded by per-base slopes and then by `train_memory` (A2) |
| **Legacy preset keys in built-ins** (`QUANT_4BIT_MODE`, `KREA2_EMA`, `COMPILE_BLOCKS`, `TARGET_LAYERS`, `MIN/MAX_TIMESTEP`) | `description.py:26-34` | replaced by `FAMILY_PRECISION`, `FAMILY_EMA` (A3) |
| **INT8 / NF4 availability probe with an fp8 fallback** | `training/quant.py:105-128, 195-201, 233-237` | not in the driver planner. A T-side safety net; with fp8 gone upstream, its fallback target has no upstream counterpart |
| **RAW default CFG 5.5** | `description.py:127` | 4.5 (`families/krea2.py:142`) |
| **The Turbo-checkpoint training preview** | never ported | now also gone upstream for training previews - no longer a gap |
| **`[warm-up]` note** | `training/train.py:600-603`, `driver.py:41` | back upstream in v6.8.3 for every driver family (section D) |

---

# PART C - port plan for Krea 2 (dependency order)

## (a) Changes that alter what a standard LoRA run does on the owner's verified setup

Setup: fp8-scaled RAW checkpoint, Auto -> INT8, no swap, AdamW 8-bit, Adaptive LR, EMA 0.98, per-image loss watch,
base and Turbo-LoRA previews, ROCm. Each item changes numbers the owner has already verified; none should be made
without the owner's go-ahead, and each wants a before/after note in the commit.

| Order | Change | Why it alters the run | Files |
|---|---|---|---|
| a1 | **INT8 scales in bf16 for Krea 2** (`int8_fp32_scales = False`) | different INT8 base weights from step 0. Upstream does this so the driver "starts from the same INT8 weights as a Krea 2 run" | `training/quant.py:61-77`, `training/families/krea2/driver.py` |
| a2 | **Loss arithmetic in bf16 order** (mix, target and noise in bf16, as `krea2/driver.py:168-190`), optionally with the autocast wrapper | changes every step's target by bf16 rounding | `training/families/krea2/driver.py:147-169` |
| a3 | **Adaptive LR: clip-ratio signal off for Krea 2** (a per-family `adaptive_lr_clip_signal`, on for Klein only) | removes a REDUCE / rollback trigger that Fizgig does not have on Krea 2; can change the LR path of the whole run | `training/adaptive_lr.py`, `training/train.py:455-461, 581-583`, `training/description.py` |
| a4 | **One caption per forward in the text cache** | changes cached conditioning slightly; needs the text cache re-encoded (add a text revision mark, or change the arch id) | `training/families/krea2/driver.py:141-144`, `training/cache.py:139-145` |
| a5 | **Auto ladder and memory figures**: `auto` = INT8 then NF4, swap base INT8, `train_memory` as upstream, bf16 entry added | at the owner's 0.25 MP nothing changes (INT8, swap 0). Above ~0.9 MP on 20 GB the plan flips from INT8 to NF4. Depends on the decision about keeping fp8 (Q2) | `training/families/krea2/description.py:99-117`, `tests/test_training_krea2.py:60` |
| a6 | **Remove the after-epoch cuDNN backend switch** | no numeric effect expected on ROCm (the cuDNN SDPA context is a no-op where the kernel is missing - **not verified on the owner's build**); on NVIDIA it changes the attention kernel from epoch 2 | `training/families/krea2/driver.py:78-91` |
| a7 | **Preview behaviour**: CFG + negative honoured with the Turbo LoRA; start noise drawn as upstream (GPU, bf16); RAW default CFG 4.5 | previews only - training is untouched, but the owner's reference pictures for a seed change | `training/train.py:244-249`, `training/families/krea2/sampling.py:138-141, 168-169`, `description.py:127, 159` |
| a8 | **arch id `krea2drv`** (only if the owner wants cache / metadata ids identical to Fizgig's) | full re-cache once; `ss_architecture` changes | `description.py:40`, tests |

Order: a1 and a2 are independent and smallest. a3 needs a description field first. a4 and a8 should be decided
together (one re-cache, not two). a5 waits on Q2. a6, a7 can go any time.

Not in (a) because they do not touch the owner's run: batch > 1 (the owner's loss watch already forces batch 1),
compile timing (Auto is off on ROCm), INT8-on-CPU for swap (the owner runs without swap).

## (b) Pure additions (no effect on a standard LoRA run)

In dependency order:

1. **Block ids as upstream** (`txt_lw_*`, `txt_rf_*`, `io`) - prerequisite for 3, 5, 6. S.
2. **Small parity items**: numbered copy of the final epoch (A5); diffusers-name aliases for Context LoRAs (A5);
   bundled tokenizer (A4); INT8 stored on CPU when swapping (A2); compile decided before the load (A2). S each.
3. **`--train_blocks` plumbing** in the trainer (`families/train.py:717-718, 995-1003`). M.
4. **Slider training** in the generic layer: pair datasets (`control_directory` as the other pole), the prompt-pair
   bank, `_prompt_slider_step`, the image-pair step, slider previews, metadata, the driver hooks
   (`noise_latents`, `predict`, `training_loss(diff_ref, diff_weight)`), Train-tab card and help. L.
5. **Krea 2 slider preset + Ultra mode** (needs 1, 3, 4). S.
6. **Reference image for previews** (vision path in the encoder). M.
7. **Full fine-tune** (`families/ft.py`, the fine-tune card, regularisation folder, rotation-boundary pause,
   continuation). L. Note for the owner: upstream refuses fp8 / pre-quantised sources here and calls ROCm untested.
8. **Checkpoint to LoRA** (needs 7). M.

## Check of unmerged commit `d10022d` ("Krea 2: port Fizgig's Fast FT (fp8 torch._scaled_mm on the frozen base)")

**Verdict: obsolete against current Fizgig. Do not merge as it is.**

What the commit does (`git show d10022d`, 10 files, +537 / -28): adds `KREA2_FAST_FT` to
`training/params.py` and to Krea 2's `family_options`, a `fast_ft` argument to `train_family`, `resolve_fast_ft()`
in `training/quant.py`, a per-tensor fp8 quantisation mode and a `_scaled_mm` training path in
`training/modules/fp8.py`, and Train-tab visibility tied to a `KREA2_FINETUNE` option. Its own message cites
Fizgig's `krea2/trainer.py:2228-2246` and `lora_trainer_gui.py:5038-5060, 34206-34246`.

What upstream is now:

- `krea2/trainer.py` no longer exists; `lora_trainer_gui.py` is 30,365 lines, so the cited GUI lines are gone.
- `KREA2_FAST_FT`, `KREA2_FINETUNE`, `--fast_ft` occur **nowhere** in fizgig-src (searched `*.py`, `*.md`, `*.json`).
  The only `KREA2_*` keys left are the four loss-watch toggles.
- A fine-tune's frozen base is forced to NF4: `families/train.py:783-786` (`precision, blocks_to_swap = "nf4", 0`,
  "the fine-tune trunk is always 4-bit"). There is no fp8 trunk to accelerate.
- fp8 is not a Krea 2 base precision any more (A2), so the condition the commit gates on ("only on an fp8 base")
  cannot be met through upstream's options.
- Left behind upstream as unreachable code: `load_krea2_dit(..., fp8_fast=False)` and its `quantization_mode="tensor"`
  branch (`krea2/utils.py:96, 177-182, 203, 239`), `_fp8_fast` in `krea2/fp8_optimization_utils.py:383-388, 513`.
  `Krea2Driver.load_dit` calls `load_krea2_dit(path, device=device, dtype=DTYPE, fp8_scaled=False,
  loading_device=device)` (`krea2/driver.py:66-69`) and never passes `fp8_fast`.
- Still live upstream, but for **Klein**, not Krea 2: `modules/fp8.py:337-572` (`_FP8ScaledMMLinear`,
  `_try_fp8_scaled_mm_train`, used when an fp8 file "trains as it is", `families/driver.py:375`).

Per part:

| Part of d10022d | Status |
|---|---|
| `KREA2_FAST_FT` parameter, help text, Train-tab visibility, `finetune_kwargs`, `fast_ft` argument, `resolve_fast_ft` | obsolete - no upstream counterpart |
| `KREA2_FINETUNE`-based `finetune_on` | obsolete - upstream's key is `FAMILY_FT` and the fine-tune is a different design (rotations, NF4 trunk) |
| `training/modules/fp8.py` per-tensor mode + `_scaled_mm` training path | not needed for Krea 2. Could be re-homed for the Klein fp8 path if that audit finds T's Klein lacks it (**not checked here**) - it would need re-checking against `modules/fp8.py`, not against the Krea 2 file it was ported from |
| "a bf16 file quantised to fp8 is staged on the CPU instead of loaded whole to the GPU" (`training/quant.py`) | only relevant while T keeps fp8 as a Krea 2 precision (Q2) |
| `tests/test_training_fp8.py` (+193 lines) | follows whichever parts survive |

---

# PART D - TagScribeR-only fixes to preserve

1. **VAE wide-head attention fix** (`training/modules/wide_attention.py`, used at
   `training/families/krea2/vae.py:390` and `training/families/qwen_image21/vae.py:506`). Upstream's VAE attention is
   **still the plain single-head call**: `x = F.scaled_dot_product_attention(q, k, v)` at
   `src/fizgig/krea2/vae.py:373` and `src/fizgig/qwen_image21/vae.py:498`. A full diff of the three VAE files
   (`krea2/vae.py`, `krea2/vae_loader.py`, `qwen_image21/vae.py`) against T's shows no upstream change besides T's own
   edits (header, the inlined `get_activation`, the dropped `logging.basicConfig`, `load_file` in place of
   `load_safetensors`, and the attention line). Upstream has not touched this code; a re-port of the VAE files would
   silently undo the fix, so they should be excluded from any bulk copy.
2. **`latent_rev` cache mark** (`training/cache.py:23-27, 51, 54-61, 192`). Upstream's `save_latents`
   (`families/cache.py:51-61`) writes `architecture`, `width`, `height`, `dtype`, `format_version` and has no such
   mark. Upstream did change this function (a `_latent_key` helper for clips and an `extra` argument,
   `families/cache.py:45-56`), so a port of those must keep the `latent_rev` entry in the metadata dict and the check
   in `run_latents`. One small bug seen in T: `latent_rev()` returns `False` instead of `""` on a read error
   (`training/cache.py:60-61`); harmless today because `False != "2"` still forces a re-encode.
3. **`[warm-up]` note** (`training/train.py:600-603`, `driver.py:41`). Upstream brought it back in v6.8.3 in the
   generic trainer (`families/train.py:1333-1349`), for every driver family, with three differences: it is gated on
   `epoch - start_epoch < 2` (so a resumed run shows it again; T uses `epoch < 2`), it has no per-driver flag (T shows
   it only for drivers with `warmup_note = True`), and it adds ", compiles the blocks" when compile is on. T's port
   need not be removed; aligning the three details is optional.

---

# PART B - QWEN IMAGE 2.1

Method: a whitespace-insensitive diff of each upstream file against T's port. Differing lines (both directions):
`driver.py` 64, `model.py` 18, `embedder.py` 59, `sampling.py` 3 (header only), `vae.py` 10 (header + the attention
fix). The description was read side by side. T's Qwen port is close to current upstream; everything below is either
an upstream addition or a generic-layer difference already described under Krea 2.

## B1. Objective and timestep sampling - identical

F `qwen_image21/driver.py:116-152`, T `training/families/qwen_image21/driver.py:105-130`: `x0 = pack(latents.float())`,
fp32 noise, `t = sigmoid(randn(1))` shifted by `exp(mu)` with `mu = S.calculate_mu(n_tokens)`, window
`min_t + (max_t - min_t) * t`, `xt = (1 - t) * x0 + t * noise`, references prepended, bf16 forward,
`F.mse_loss(pred.float(), noise - x0)`. The only diff in `training_loss` is upstream's added slider branch
(`diff_ref` / `diff_weight`, `driver.py:142-151`), which does not run for a normal LoRA. `sampling.py` is verbatim.
The DiT forward differs only by the compile hook (`model.py:425-427` upstream: a block with `_handles_checkpointing`
is called directly) and the workbench-only INT8 attention (`model.py:166-172`); neither changes a standard run.

## B2. Auto precision / memory plan

Description figures are **identical**: `precisions=("bf16", "int8", "nf4")`, `train_memory={"bf16": (((0.25, 14.9),
(1.0, 19.0)), 0.44), "int8": (((0.25, 8.6), (1.0, 11.9)), 0.19), "nf4": (((0.25, 6.0), (1.0, 9.5)), 0.0)}`
(F `families/qwen_image.py:137-145`, T `description.py:106-114`). No `auto_precisions` on either side, so Auto is
bf16 -> INT8 -> NF4, then INT8 with swap.

Planner differences (generic layer, `training/quant.py` vs `families/quant.py`):

- When even the swap needed does not fit, F falls to NF4 "the smallest base" (`quant.py:163-166`); T returns the
  swappable base with the **maximum** swap, "tight" (`training/quant.py:225-228`). Different pick on very small cards.
- T probes INT8 / NF4 availability and drops what the machine cannot run (`training/quant.py:195-201`); F does not.
- INT8 on a swapped base is stored on the CPU upstream (`quant.py:51, 104`), on the GPU in T (`training/quant.py:71`).
- INT8 scales are fp32 for Qwen on both sides (F default `int8_fp32_scales = True`, `quant.py:44-47`).
- Text encoder: bf16 when >= 19.5 GB is free, else INT8; with the vision tower the threshold is 21.0 GB - same on
  both (F `driver.py:57-70`, T `driver.py:50-63`).

## B3. Presets

Upstream (`families/qwen_image.py:17-35, 220-248`). Common: alpha = rank, LoRA, save every epoch, seed 42,
adamw8bit, accumulation 1, max grad norm 1.0, **0.5 MP**, swap Auto, precision Auto, training adapter on, EMA 0.98,
loss watch on, per-image LR **off**, auto-recaption off, look warm-up off, `FAMILY_SLIDER_GUIDANCE` "2",
`FAMILY_FAST_ID` False.

| Preset | rank | LR | Adaptive LR | epochs | flags | In T? |
|---|---|---|---|---|---|---|
| ✨ Qwen 2.1 Fast (rank 8, adaptive LR) - default | 8 | 1e-4 | 2e-4 - 4e-4 | 30 | | yes, same (`description.py:186`) |
| ✨ Qwen 2.1 Fast Identity Mode (rank 8) - very close to full-model likeness, ~1.5x faster | 8 | 1e-4 | 2e-4 - 4e-4 | 30 | `DATASET_MEGAPIXELS` "0.25", `FAMILY_FAST_ID` True | **missing** (new, v6.8.3) |
| ✨ Qwen 2.1 Standard (rank 16, adaptive LR) | 16 | 1e-4 | 1e-4 - 2e-4 | 30 | | yes, same (`:189`) |
| ✨ Qwen 2.1 Style (rank 16, 1.5e-4) | 16 | 1.5e-4 | off | 30 | | yes, same (`:193`) |
| ✨ Qwen 2.1 Edit (rank 8, adaptive LR) | 8 | 1e-4 | 2e-4 - 4e-4 | 12 | `FAMILY_EDIT` True | yes, same (`:196`) |
| ✨ Qwen 2.1 Edit Strong (rank 16, adaptive LR) - trickier edits | 16 | 1e-4 | 1e-4 - 2e-4 | 12 | `FAMILY_EDIT` True | yes, same (`:198-199`) |
| ✨ Qwen 2.1 Slider (rank 4, 2e-4) | 4 | 2e-4 | off | 30 | `FAMILY_SLIDER` True | **missing** (new, v6.7.0) |

No preset changed value or name. Every upstream preset now also carries `FAMILY_SLIDER`, `FAMILY_SLIDER_GUIDANCE`
and `FAMILY_FAST_ID`, which T's presets and `training/params.py` do not have.

## B4. Text encoder, VAE, cache

- **Text encoder** (Qwen3-VL-8B): encode logic unchanged. Upstream changes are the loader helpers' home and
  **offline loading**: `from_pretrained_cache_first(AutoTokenizer, TOKENIZER_REPO, subfolder="processor")` and the
  same for `Qwen3VLProcessor` (`qwen_image21/embedder.py:92-94, 148-149`; helper `utils/hf_cache.py:22-53`, which
  loads from the cached snapshot's own folder first because "since transformers 4.57 its loader asks the Hub whether
  the repo is a Mistral model ... so a cached Qwen tokenizer still failed offline (#174)"). T calls plain
  `AutoTokenizer.from_pretrained(TOKENIZER_REPO, subfolder="processor")` and
  `Qwen3VLProcessor.from_pretrained(...)` (`embedder.py:132, 186`): **with no network, T's Qwen caching fails even
  when the files are cached** - the bug upstream fixed in v6.8.0. T's own Krea 2 and Klein loaders have the same
  exposure in part (`krea2/embedder.py:127` uses the repo id with no cache-first step; Klein uses
  `local_files_only=True` at `klein/embedder.py:103`, which upstream says is not enough for Qwen tokenizers).
- **VAE**: unchanged upstream; only T's wide-attention line differs (`qwen_image21/vae.py:498` vs T `:506`).
- **Cache format and id**: unchanged. `arch_id="qwenimage21"` on both; latents `latent_{h}x{w}` (+
  `latent_control_{i}_{h}x{w}`), conditioning `cond__hidden_states` (+ `cond__ref_mask`), `reference_sizes` in the
  metadata. **Existing T Qwen caches stay valid.** Captions are encoded in batches of 8 on both sides here (Qwen's
  driver passes the list through, `driver.py:90-91`); the one-caption rule is Krea 2's only.

## B5. LoRA targets, keys, alpha, network types, metadata - identical

`transformer.transformer_blocks.{block}.{module}.lora_A/lora_B.weight`, modules `attn.to_q`, `attn.to_k`,
`attn.to_v`, `attn.to_out.0`, `img_mlp.gate_layer`, `img_mlp.proj`, `img_mlp.out`, `kohya=False`,
`file_prefix="transformer."`, `("lora", "lokr")` (F `families/qwen_image.py:114-127, 147`, T `description.py:83-96,
116`). Metadata differs only by the generic additions listed in A5 (slider keys, `ss_train_blocks`, the numbered
final copy). Upstream added a `lokr_kohya_stems` switch in `families/lora.py:217` (**not traced**; no family in this
audit sets it as far as the two descriptions show).

## B6. Preview engines

| Setting | Fizgig | TagScribeR |
|---|---|---|
| Default | plain model, 25 steps, CFG 1.0; Turbo strength 0 (`families/qwen_image.py:207-215`) | same (`description.py:167-175`) |
| Viggle turbo LoRA | 6 steps, CFG 1.0, sigmas (1.0, 0.9375, 0.875, 0.75, 0.5, 0.25), `shift_terminal` None, strength 1.0, unmerged (`:178-203`) | same (`:138-163`) |
| Default negative | `preview_negative=GENERAL_NEGATIVE` (`:216`) | no description field; the same text is the `SAMPLE_NEGATIVE` default (`training/params.py:311-314`) |
| CFG with the turbo LoRA on | honoured above 1, with the negative (`families/train.py:422-427`, v6.8.1) | ignored (`training/train.py:244-246`) |
| CFG hint | `preview_cfg_note` "1 = no CFG ... about 1.5 to 3 ..." (`:153-154`); v6.8.1 recommends "Turbo strength 0, 20 steps and CFG 3" | none |
| Edit previews (reference photo) | `--sample_reference`, `FAMILY_EDIT_REF`, the Samples tab's picture also edits (`families/launch.py:669-681`) | `FAMILY_EDIT_REF` present (`training/params.py:318-320`, `pipeline.py:444`); the Samples-tab picture as a general edit reference was **not checked** |
| Slider previews | -1 / 0 / +1 strips | none |
| `workbench_follows_samples`, INT8 attention, activation cache | set (`:150, 155-156`) | workbench not ported |

## B7. Optimizers

F `optimizers=("adamw", "adamw8bit")` (`:146`) with the GUI offering the whole catalogue for every family
(A7); T `("adamw", "adamw8bit")` (`description.py:115`). **Not checked**: whether T's Train tab limits the dropdown
to the description's tuple for Qwen - if it does, Qwen users cannot pick Automagic v3 in T, which upstream allows
(docs/QWEN_IMAGE.md: "Picking Automagic as the optimizer turns Adaptive LR, per-image LR and the look warm-up off").
No per-family Automagic groups for Qwen on either side (`optimizer_families` empty).

## B8. Features upstream has that TagScribeR's Qwen lacks

| # | Feature | Fizgig source | GUI key(s) | Size |
|---|---|---|---|---|
| 1 | **Slider LoRAs** (pairs and prompts; push strength default 2) | `slider_training=True` `families/qwen_image.py:149`; driver `training_loss(diff_ref, diff_weight)` `qwen_image21/driver.py:123-152`, `noise_latents` `:154-159`, `predict` `:161-168`; generic trainer as A8.1 | `FAMILY_SLIDER*` | L once (shared with Krea 2), then S for Qwen |
| 2 | **Fast Identity Mode** (train blocks 10-14 only) | `identity_blocks=tuple(f"block_{i}" for i in range(10, 15))` `:107`; `families/launch.py:559-561` -> `--train_blocks`; `families/train.py:995-1003` | `FAMILY_FAST_ID` (`lora_trainer_gui.py:4224-4226`) | S after `--train_blocks` (A8.8) |
| 3 | **Block map / categories** (identity = blocks 10-14, the rest "look") and Repair Studio presets | `block_categories` `:89`, `repair_presets` `:108-112`, `block_note` `:80-83` | workbench | S for the data; T's `block_note` still says "No block map (style / identity) exists yet" (`description.py:80-81`) |
| 4 | **Full fine-tune** | `finetune=True` `:157`; `ft_spec` `qwen_image21/driver.py:42-51` (`components=("attn", "img_mlp.gate_layer", "img_mlp.proj", "img_mlp.out")`, `file_layout` for the fused `gate_up`, `overhead_gb=5.5`, `stream_base_gb=2.7`, `calib_mp=0.5`, `act_gb_per_mp=4.2`); accepts bf16 / fp16 / fp32 sources only (`families/ft.py:140-151`); the training adapter stays on | `FAMILY_FT*` | L once (shared), then S |
| 5 | **torch.compile** for Qwen | `compiles=True`, `compile_boundary="outside"`, `compile_fullgraph=False`, `compile_payback_steps={"int8": 300, "bf16": 800}` `:96-99`; `compile_targets` `driver.py:75-76`; model hook `model.py:425-427`; generic rule `families/driver.py:378-427`; measured 1.43x on INT8, 1.15x on bf16 | `COMPILE_BLOCKS` | M (move T's Krea 2 compile module to a shared one first) |
| 6 | **Offline tokenizer / processor loading** | `utils/hf_cache.py`, `embedder.py:92-94, 148-149` | - | S |
| 7 | **CFG + negative with the turbo LoRA**; default negative on the description; CFG hint | `families/train.py:422-427`, `:153-154, 216` | Samples CFG / Negative | S |
| 8 | **Qwen slider and Fast Identity presets** | `:230-231, 247` | presets | S (after 1, 2) |
| 9 | **Checkpoint to LoRA** | as A8.4 | fine-tune card | M |
| 10 | **Workbench** (Profiler with renders, Repair Studio colours, INT8 attention, activation cache) | `:150-156, 249` | workbench tabs | L (out of scope) |

Edit LoRAs are already in T (`edit_training=True`, `description.py:117`; `FAMILY_EDIT*` in `training/params.py:113-119`).
**Not checked**: v6.7.0-6.7.1's edit fixes - case-insensitive pair matching (`families/launch.py:140-183`) and
"Start now saves the dataset settings just before every Qwen run" - against T's pair handling.

## B9. Anything T has that upstream removed

Nothing family-specific. The generic items in A9 apply (clip-ratio signal in Adaptive LR - Qwen does not set
`adaptive_lr_clip_signal` either, so T can halve the rate on a clip-heavy epoch where Fizgig would not; the planner's
max-swap fallback and availability probe).

---

# Questions for the owner

1. **Cache id.** Adopt upstream's `arch_id="krea2drv"` (one full Krea 2 re-cache, ids identical to Fizgig) or keep
   `krea2` and add a text-cache revision mark so only the text cache is redone when the one-caption-per-forward rule
   is ported? Recommendation: keep `krea2` + a text revision mark; T's layout was never the old trainer's.
2. **fp8 as a Krea 2 base precision.** Upstream removed it ("fp8 is gone. Auto picks INT8, then NF4"). T still offers
   it and uses it as Auto's fallback and swap base. Mirror upstream (drop it from the list; an fp8-scaled *file* keeps
   working as the source for INT8 / NF4, and chosen with "bf16" it runs as fp8), or keep it as a T extension? The
   owner's Auto -> INT8 run is unaffected either way. Recommendation: mirror upstream's Auto order and figures now;
   keep "fp8" selectable only if the owner wants it, since removing it narrows what T offers today.
3. **The (a) list.** Which of a1-a4 (INT8 scales in bf16, bf16-order loss, clip-ratio signal off for Krea 2, one
   caption per forward) should be applied? Each moves T towards Fizgig and away from the exact numbers of the owner's
   verified runs. Recommendation: all four, in one commit series, followed by one real run by the owner.
4. **RAW preview CFG.** Upstream's description lists 28 steps / CFG 4.5 for RAW and defaults a Krea 2 run without
   the Turbo LoRA to 8 steps / CFG 1.0 (user-editable). T defaults base previews to 28 / 5.5. Change to 4.5?
5. **Batch size > 1 on Krea 2.** Upstream now refuses it for all driver families. Keep T's support or mirror?
6. **Commit d10022d.** Drop it (recommended), or keep only the fp8 module work for Klein if the Klein audit wants it?
7. **Fine-tune.** Upstream refuses fp8 / pre-quantised checkpoints and says ROCm is untested. Port it anyway for
   NVIDIA users and bf16 files (size L)?

# What was not checked

- No weights or checkpoint headers were read; the dtype of the non-quantised tensors in the owner's fp8-scaled RAW file
  is unknown, so the autocast point in A1 is unresolved.
- The 746ddab baseline was not available (the Fizgig copy is a one-commit clone), so "what upstream changed since
  the port" is inferred from the release notes, file comments and the diff against T - not from upstream's history.
- Generic-layer internals: `loss_watch.py`, `ema.py`, `automagic3.py`, `lora.py` (grep only), the dataset layer,
  `_load_state` / resume, `families/ft.py` (grep only), `families/workbench.py`, `block_profile.py`, `act_cache.py`,
  `extract.py`, `diff_to_lora_gui.py`.
- `krea2/offloading.py`, `krea2/rotation.py`, `krea2/lora_utils.py`, `krea2/safetensors_utils.py`,
  `krea2/fp8_optimization_utils.py` (grep only). `krea2/rotation.py` is still imported by `families/ft.py`.
- `training/families/krea2/compile.py` against `families/compile.py` line by line (constants and rules compared).
- `tests/test_training_krea2*.py` were only grepped: `tests/test_training_krea2.py:56-60` pins `arch_id` "krea2" and
  `auto_order == ("int8", "nf4", "fp8")`, and `:150-163` tests batch 2 - all three would need updating with the
  corresponding changes.
- The Train tab (`tabs/train.py`) and how it limits optimizer / precision choices per family.
