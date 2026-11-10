# Audit 13 - Fizgig v8.7.3 to v9.4.0: new families, new tools, UI lessons, and where TagScribeR can lead

Read-only study, 2026-11-09. Upstream = https://github.com/shootthesound/Fizgig, read from the git-ignored checkout
in `.claude/fizgig-src` with `git log` / `git diff` / `git show` only (the checkout was not changed). Baseline =
`9162fb20` (v8.7.3, TagScribeR's port source). Target = `c11134a2` (origin/HEAD, just after v9.4.0). The range is 694
commits, 321 files, +139,760 / -2,730 lines (about 98,000 of the added lines are a bundled CLIP tokenizer vocabulary).

Paths: `FZ:` = `src/fizgig/...` in the Fizgig repository at `c11134a2` (or `docs/...` where a `docs/` path is given);
`TS:` = this repository. Nothing was run: no tests, no model loads, no downloads, no GPU, no app launch.

**How each claim is marked.**

- **(S)** read in source at `c11134a2` (a file and, where useful, a line is given).
- **(N)** from release notes, `docs/NEWS.md`, a guide under `docs/` or a commit message only; not confirmed in code.
- Every memory, speed or likeness figure is **Fizgig's own measurement** on Fizgig's hardware (mostly RTX 5090,
  RTX PRO 6000 and a 16 GB laptop RTX 4090). None was reproduced. No figure here comes from an AMD card.
- Licences and gating of weights, file sizes and repository names are what Fizgig's descriptions and guides say
  ("per upstream"). Nothing was checked on Hugging Face.

Scope: what is NEW since the baseline and is not the already-ported training core. The shared loop, optimizers
(Muon, Prodigy, Automagic v4), the presets of the families TagScribeR already has, the fine-tune and packaging
(PyTorch 2.15, Python 3.11) belong to the other audit of this range and are only named here where a family or a
tool depends on them.

Sections: 0 summary; 1 what a family now declares; 2 the families; 3 the new tools and features; 4 UI / UX lessons;
5 where TagScribeR is ahead or can pass Fizgig; 6 not verified; 7 questions for the owner; 8 recommended order.

---

## 0. Summary

### Families

Fizgig's registry at the target lists 13 families (S, `FZ: families/registry.py:8-21`). New since the baseline:
**Z-Image Base** (v8.9.95), **Fibo 2.0** (v9.0.4), **Vimoo 1.0 Lite** (v9.0.7). No other family was added.
Still unported in TagScribeR and changed since audit 07: **Flux 3 Image Klein**, **Z-Image Turbo**. MOSS-Image, which
TagScribeR is porting from `9162fb20`, moved a great deal (section 2.5).

| Family | What | Weights licence (per upstream) | Gated | Download | Upstream lines | Port size | Order |
|---|---|---|---|---|---|---|---|
| Z-Image Turbo | Tongyi-MAI 6B image DiT, 8-step distilled | Apache-2.0 | no | ~21 GB | 1,212 (shared with Base) | S-M | 1 |
| Z-Image Base | the undistilled Z-Image, same DiT, same driver | Apache-2.0 | no | +12.3 GB on top of Turbo | +110 (one `dataclasses.replace`) and three driver hooks | S on top of Turbo | 1 (same change) |
| Flux 3 Image Klein | Black Forest Labs 5.6B image + edit DiT | FLUX Kommunity (non-commercial) | yes, every file | ~31 GB | 1,321 | M-L | 2 |
| Fibo 2.0 | BRIA 8.2B image + edit DiT, structured JSON captions | CC BY-NC 4.0 (non-commercial) | no (Comfy-Org repo) | ~28 GB | 1,898 | M-L | 3 |
| Vimoo 1.0 Lite | Vimoo-AI 9.5B video model (text- and image-to-video) | Apache-2.0 (Glyph-ByT5: no licence stated) | no | ~40 GB | 2,598 | L stills only, XL with clips | 4, after stage 6 |
| ACE-Step 1.5 XL | music (skipped by the owner) | - | - | - | - | - | - |

ACE-Step: 19 lines changed (the slider "leave these alone" texts, a song-style first preview prompt and "not offered"
flags); nothing that changes the owner's decision (S, `git diff --stat`, `FZ: families/acestep.py`).

### The five most valuable new tools for TagScribeR

1. **Masked training with a mask workflow** (make masks in one click, SAM 3.1 from words, click-to-mask, a mask
   painter). Fizgig reports a significant likeness gain from it (its claim; no figures given). TagScribeR today
   does not know mask files at all: they appear as dataset images and block Start (section 3.1).
2. **Training maps and the trigger-word map** (heatmaps over three training pictures, epoch by epoch). A diagnosis
   tool no other trainer has; small upstream code (459 lines) on hooks TagScribeR already ported for sliders.
3. **Detail crops and the Style score** for style LoRAs, plus caption tasks per kind of LoRA (person, object, concept,
   style). Dataset-preparation features, TagScribeR's own ground.
4. **Start warnings, Continue, the run summary, per-run log files, preview and save schedules, prompt sets.** The
   "runs that survive things going wrong" group: each is small and independent.
5. **Face tools for drawn characters** (a YOLO head finder plus DINOv2 look match) beside InsightFace, and the
   per-image loss window with caption editing during a run.

Helpers, Move a run, Training share and Pods are the largest new body of code (4,505 lines in one file) and map onto
VISION theme G, which the owner already decided (runner service first, on one PC).

### Top UI / UX lessons (section 4)

Warn at Start and let the user start anyway; a missing ability is a written decision, never an oversight; a note
appears only when it is true ("the warm-up note is measured"); every enlarged preview says how it was made; a
changed default moves a saved form once and says so; long jobs grey their buttons and name each stage; a number
never shown without its ceiling ("peak ArcFace is about 72%").

### Top "surpass" ideas (section 5)

Masks as a first-class dataset layer in the Gallery and Editor; mask health checks; a multi-provider mask and
caption pipeline (Fizgig is tied to one captioner and one segmenter per job); a pre-run report that joins Health,
masks, trigger words and buckets; detail crops placed by hand in the Editor; structured (JSON) captions for Fibo 2.0
from any VLM; training maps and per-image loss shown on the Gallery grid.

---

## 1. What a family declares now (shared abilities)

`FZ: families/description.py` grew by 182 lines (S, full diff read). The fields a new family port meets:

| Field | Meaning | Declared by |
|---|---|---|
| `masked_training` | `training_loss` takes `loss_mask` (and `prior`) | Klein, Krea 2, Qwen 2.1, Flux 3, MOSS, Z-Image x2, SDXL, Anima, H3 stills, Fibo 2.0, Vimoo (stills) |
| `batch_training`, `batch_gb`, `batch_auto`, `batch_gb_nogc` | `training_loss_batch` + `stack_conds`; memory per extra picture | every picture family; Vimoo `False` |
| `training_maps` (default True), `trigger_attention` | the maps (section 3.2); the model calls `attn_probe.layout / see`, the driver has `trigger_text` and `map_target` | maps: all stills families; trigger map: all but SDXL |
| `not_offered: {ability: reason}` | an ability a picture family leaves out must be named with its reason; `checks/test_family_abilities.py` fails otherwise | Vimoo (5 entries), Z-Image Base, MOSS |
| `train_memory_small`, `auto_precisions_small` | figures for cards under 20 GB, where the trainer parks the model for the preview decode | Fibo 2.0, Vimoo |
| `train_memory_nogc`, `train_memory_partial`, `nogc_uncompiled` | gradient checkpointing off (or partly off) on 40 GB+ cards | Z-Image, MOSS, Flux 3, Fibo 2.0 |
| `start_lr_bands` | the learning-rate range the Start warning calls usual, per kind | every family |
| `resolution_stages`, `detail_crops`, `caption_dropout`, `rgba` | Start at a smaller size; Detail crops; caption dropout; transparent pictures (Qwen 2.1 only) | defaults True / True / True / False |
| `ModelFile.same_as`, `signature`, `fetch_on_use` | reuse another family's file; a tensor name and shape only the right file has; download on first use | Z-Image Base, Flux 3, Z-Image, MOSS |
| `TextEncoderBuild` (`text_encoder_builds`) | other builds of a text encoder the loader takes, with hash and measured cosine | Flux 3, Fibo 2.0, MOSS |
| `LoRAFormat.save_fused` | every LoRA file is written through `driver.export_lora_state_dict` and read through `convert_lora_state_dict` | Fibo 2.0 |
| `preview_look_*`, `preview_labels`, `samples_turbo_tick`, `gallery_note`, `preview_small_card_cap / option`, `workbench_render_choice` | preview-only look LoRA, the "how this preview was made" line, a Turbo tick, small-card preview size | MOSS, Z-Image Base |
| `kind_values`, `value_moves`, `small_dataset_hint`, `sample_prompt`, `status_note` | per-kind defaults, a changed default moved once in saved forms, hints | MOSS, ACE-Step |

The "declared by" column is (S) for Flux 3, Z-Image, MOSS-Image, Fibo 2.0 and Vimoo, whose descriptions were read
in full or as a diff; for the other families it is what the release notes say (N).

Two cross-dependencies with the other audit:

- **Every built-in LoRA, slider and edit preset of every picture family is on Muon** with
  `OPTIMIZER_ARGS = MUON_DEFAULT_ARGS` (S, `FZ: training/optimizers.py:35`, each family's `_preset`). TagScribeR has
  no Muon (S, `TS: training/optimizers.py`). A new family's presets cannot be mirrored verbatim until Muon is
  ported; until then they would have to run on another optimizer, which is a difference from Fizgig to record.
- **Built-in Fine-tune presets are rewritten to Adafactor** at registry import (S, `FZ: families/registry.py:23-31`).

---

## 2. The families

### 2.1 Z-Image Turbo and Z-Image Base

**What changed for Turbo since audit 07 (section 4 there still describes the model)** (S, diff of
`FZ: zimage/driver.py`, `families/zimage.py`):

- Abilities added: masked training with prior preservation, batch training (`stack_conds` keeps each caption apart,
  no padding), the trigger-word map (`trigger_text`), the big-card memory tables.
- The Qwen3-4B text encoder loads in 8-bit when less than 11 GB of VRAM is free (S, `zimage/driver.py:117-121`;
  about 4.5 GB instead of 7.7, Fizgig's figure).
- Presets moved from `adamw` to Muon; their names and every other value are unchanged since the baseline (S, the
  diff and `families/zimage.py:22-39, 163-181`). Slider push 2. `start_lr_bands` lora 3e-5 to 4.5e-4.
- The VAE row carries `signature=("decoder.conv_in.weight", (512, 16, 3, 3))`: Fizgig found Klein's FLUX.2 VAE and
  Z-Image's FLUX.1 VAE both saved as `ae.safetensors`, one overwriting the other (N v9.0.5; S for the field and
  `FZ: scripts/fetch_models.py:344 repair_links`). TagScribeR should check the same shape before loading.
- Objective and timestep sampling unchanged: `x_t = (1 - s) x0 + s noise`, the model runs at `t = 1 - s` and
  predicts `x0 - noise`; `s = sigmoid(randn)` shifted by the static 3, unweighted MSE (S, `zimage/driver.py:1-13,
  154-157`).

**Z-Image Base** (v8.9.95; S, `families/zimage.py:197-276`):

- The description is `dataclasses.replace(ZIMAGE, ...)`: same driver, same S3-DiT ("transformer/config.json
  identical", upstream's comment), same FLUX.1 VAE and Qwen3-4B. Key `zimage_base`, arch id `zimagebase`, file
  suffix `zib`, modelspec "Z-Image".
- Files: `Comfy-Org/z_image` `split_files/diffusion_models/z_image_bf16.safetensors` (12.31 GB); the VAE and text
  encoder rows are `same_as` Turbo's; the Turbo DiT is an optional `preview_dit` row. Apache-2.0, no gating flag
  (per upstream, `docs/ZIMAGE.md`).
- **No training adapter** (`training_adapter=""`): Base is not distilled. Presets are Turbo's six with the adapter
  key removed and "Turbo" renamed "Base" (S, `_base_presets`).
- **Previews render on Turbo by default** (`train_preview_checkpoint=True`, `preview_checkpoint_on=True`, tick
  "Use Z-Image Turbo for samples (set it in Preferences)", 8 steps, CFG 1, shift 3). With the tick off: 25 steps,
  CFG 4, shift 3, the general negative prompt, "ComfyUI's Z-Image Base template". The driver gained
  `park_for_preview / load_preview_checkpoint / unpark_after_preview` (S, `zimage/driver.py:50-99`), the same
  handoff TagScribeR already has for Klein (`TS: training/train.py`, `_CheckpointPreviews`).
- `int8_attention=False` with the reason recorded in `not_offered` (workbench only; not a port concern).
- Training timestep sampling is the same code as Turbo's, shift 3. One upstream comment says "Turbo's shift 3,
  Base's 6" (`zimage/driver.py:237`) while the Base description's sampling says shift 3 and its comment says
  Tongyi's reference schedule uses 6 (`families/zimage.py:203-204, 245-250`). The code path gives 3 for both. A
  stale comment, noted so nobody "fixes" it to 6 while porting.

**Reuse in TagScribeR.** INT8 / NF4 / block swap / compile, the preview-checkpoint handoff, the per-family tick
(`TS: training/params.py:60-61`), sliders, LoKR, the training-adapter mechanism (Qwen 2.1's), the offline tokenizer
loader. New: the vendored `zimage/model.py` (485) and `pipeline.py` (133), the driver (318), a 16-channel FLUX.1 VAE.

**ROCm.** (a) The FLUX.1 VAE loads through diffusers `AutoencoderKL` (S, `zimage/pipeline.py:73-86`): one 512-wide
attention head, the case `TS: training/modules/wide_attention.py` exists for. Fizgig's own fix
(`FZ: modules/wide_attention.py`) is used by Klein, Krea 2, Qwen 2.1 and SDXL only (S, grep: four call sites), not by
Z-Image. TagScribeR must install its attention processor here as SDXL does. (b) Complex-valued RoPE
(`torch.view_as_complex`, `torch.polar`, S `zimage/model.py:82, 211, 274`): not verified on the owner's ROCm build;
a CPU unit test proves the maths, only a real run proves the card. (c) Masked attention uses
`sdpa_kernel([CUDNN, EFFICIENT, MATH], set_priority=True)` (S, `zimage/model.py:91, 153`): a preference list with
fallbacks, so it should not raise on ROCm, but route it through `TS: training/modules/sdpa.py` like every other
family. (d) Compile "requires Triton and, on Windows, a C++ compiler" (upstream's `compile_hint`).

**Port size.** Turbo S-M (as audit 07), Base S more. Tests as audit 07 section 4.7 plus: the Base description
validates and has no adapter key in any preset; the preview handoff with a tiny "Turbo" file; the VAE signature
check refuses a 32-channel file.

### 2.2 Flux 3 Image Klein (state at v9.4.0)

Audit 07 section 2 holds. Changes since the baseline (S, diff of `FZ: families/flux3.py`, `flux3/driver.py`):

- **The files were renamed upstream on Hugging Face** (N v9.0.5; S for the new paths):
  `diffusion_models/flux3-image-klein-base-bf16.safetensors` (Base, trained on) and
  `diffusion_models/flux3-image-klein-bf16.safetensors` (distilled, previews). Audit 07's file names
  (`flux3_image_klein_bf16`, `..._sd_bf16`) no longer download. Still `gated=True` on all four rows.
- VAE `signature=("decoder.conv_in.weight", (512, 32, 3, 3))`.
- `text_encoder_builds`: Krea 2's bf16 Qwen3-VL-4B copy and the fp8-scaled build are accepted (cosine 1.00000,
  worst token 0.9945 for fp8: Fizgig's measurement); the int8 ConvRot build "doesn't load in Flux 3's encoder loader".
- Abilities added: masked training with prior, batch training (each picture packed as its own sample, no padding),
  `map_target`, `trigger_text`, `batch_auto=((0.25, 4), (1.0, 1))`, `train_memory_nogc` bf16 16.7 / 24.0 GB.
- Presets: the same seven, now on Muon. Upstream still says they "start from Krea 2's proven settings and haven't
  yet been tuned on Flux 3 itself" (S, `docs/FLUX3.md:7`): do not describe them as measured.
- **The forced cuDNN attention is still there** (S, `flux3/model.py:89, 96`:
  `sdpa_kernel(SDPBackend.CUDNN_ATTENTION)` with no fallback list). On ROCm this raises. PORT_PLAN's decision
  (route it through `training/modules/sdpa.py`) stands.
- Licence text unchanged: FLUX Kommunity, non-commercial, outputs commercial under US$5M revenue (per upstream).

Port size unchanged: M-L. TagScribeR already has the FLUX.2 autoencoder with the wide-attention fix
(`TS: training/modules/flux2_vae.py`), multi-reference edit pairs and the per-family preview tick, which were three
of audit 07's five TagScribeR-side items.

### 2.3 Fibo 2.0 (new, v9.0.4; contributed upstream through a pull request)

**What it is.** BRIA's 8.2B text-to-image and editing model, "trained on long structured captions" (JSON); plain
text works too (N, `docs/FIBO2.md`). Marked `experimental=True` (S). A source comment says the family was shown in
the GUI on request while "the training-run proofs that gate the un-hide are deferred" (S, `families/fibo2.py:57-61`):
upstream's own checklist for a finished model was not completed when it shipped.

**Licence and gating (per upstream).** Weights CC BY-NC 4.0: non-commercial only, and "a LoRA or fine-tune trained
on it is a derivative and carries the same restriction" (`docs/FIBO2.md`, Licence). Fizgig downloads from
`Comfy-Org/Fibo-2.0` with no token; BRIA's own repository `briaai/Fibo-2.0-dev` is gated (S, the text-encoder
row's `download_note`). Model code: Apache-2.0, ported from diffusers 0.42.1 (S, `fibo2/model.py:1-2`).

**Files** (S, `families/fibo2.py:63-103`):

| pref key | role | path in `Comfy-Org/Fibo-2.0` | GB |
|---|---|---|---|
| `fibo2_dit` | dit | `diffusion_models/fibo2_dev_bf16.safetensors` (alt: `fibo2_dev_int8_convrot.safetensors`, 8.41 GB, decoded at load) | 16.45 |
| `fibo2_vae` | vae | `vae/flux2-vae.safetensors` | 0.34 |
| `fibo2_text_encoder` | text encoder | `text_encoders/qwen3vl_4b_bf16.safetensors` (saved as `fibo2_text_encoder.safetensors`) | 8.88 |
| `fibo2_turbo_lora` | speed LoRA | `loras/fibo_2.0_turbo_lora_avg_rank_270_bf16.safetensors` | 1.99 |

**Text encoder.** Qwen3-VL-4B, "the same encoder Krea 2 and FLUX.3 use", loaded whole (vision tower and LM head
kept) because edits pass the before-images through its vision path. Conditioning is six bundles of three hidden
layers each, every layer scaled to unit RMS per token in float32, 7680 wide: `{"tokens": (6, L, 7680), "mask"}`
(S, `fibo2/driver.py:398-413`, `fibo2/model.py:40-41`). Up to 4,096 tokens.

**VAE.** The FLUX.2 autoencoder, loaded with Flux 3's loader (S, `fibo2/driver.py:331-332`). **TagScribeR's
`training/modules/flux2_vae.py` is that file, with the ROCm fix.** Latents are 32 channels at /8, normalised with
the VAE's BatchNorm over the 2x2-packed latents; the DiT patchifies 2x2 itself.

**Model.** 44 main blocks (five of them cross-attend the text), 2 noise refiners, a 2-layer Perceiver resampler;
hidden 3584, 28 heads (head width 128), complex RoPE (S, `fibo2/model.py:32-36, 189-230, 340-360`). Fizgig trains
on split Linears and **writes every LoRA in ComfyUI's fused layout** (`attention.qkv`, `ffn.w13`, `to_kv`; the FFN
hidden padded to 9728), reading it back split: `export_lora_state_dict` / `convert_lora_state_dict`, about 230
lines (S, `fibo2/driver.py:844-1074`). LoRA only: "a LoKR change can't be written exactly onto ComfyUI's fused
tensors". LoRA modules `to_q, to_k, to_v, to_out.0, w1, w2, w3`, kohya names, 415 adapted Linears.

**Objective and timestep sampling** (S, `fibo2/driver.py:14, 45-49, 511-519, 558-595`). Flow matching,
`x_t = (1 - sigma) x0 + sigma noise`, target `noise - x0`, MSE. The draw follows BRIA's reference trainer: 10% of
steps uniform, otherwise `sigmoid(N(mu, 1))` with `mu` rising from 0.5 to 1.15 across 256 to 4,096 image tokens. A
`lownoise_pct` Clean-end option exists in the driver (`_t_mu`) but the description declares no `options`, so the
GUI does not offer it.

**Presets** (S, `families/fibo2.py:26-45, 296-306`; upstream names carry a sparkle prefix; all Muon, EMA 0.98,
alpha = rank, save every epoch, 0.25 MP unless noted):

| Preset | rank | LR | epochs | flags |
|---|---|---|---|---|
| Fibo 2.0 (rank 32, full model) | 32 | 1e-4 | 40 | first = default |
| Fibo 2.0 Edit (rank 16) | 16 | 1e-4 | 6 | `FAMILY_EDIT` |
| Fibo 2.0 Fine-tune (1e-5) | 16 | 1e-5 | epochs key 40; a fine-tune runs by rotations | `FAMILY_FT`, rotations 3, rotate every 2, optimizer rewritten to Adafactor by the registry |
| Fibo 2.0 Slider (rank 8, 2e-4) | 8 | 2e-4 | 20 | `FAMILY_SLIDER`, 0.5 MP |

Optimizers offered: `adamw`, `adamw8bit`, `automagic3`. The first preset was Automagic from 1e-6 until v9.0.8
moved it to Muon at a flat 1e-4.

**Memory (Fizgig's figures, previews included, RTX 5090):** bf16 26.6 / 27.8 GB at 0.25 / 1 MP; INT8 18.7 / 20.5;
NF4 15.5 / 17.5. Under 20 GB the preview renders at 768 with the model parked: INT8 12.4 / 15.9, NF4 10.1 / 13.0
(16 GB laptop RTX 4090, 2.09 s a step at 0.25 MP). Auto plans INT8 then NF4.

**Previews.** Comfy-Org's Turbo LoRA, 4 steps, CFG 1, strength 1, on by default; without it 30 steps at CFG 5
against the reference's "empty structured caption" negative (a JSON string, S `families/fibo2.py:285-287`).

**Abilities.** Masked training (with prior), batch training (captions padded to a multiple of 16 with a mask),
sliders, edit LoRAs (10 references; the caption is the edit instruction or a JSON caption with an
`edit_instruction` key), fine-tune, training maps and the trigger map, compile. Not offered: LoKR, Model Area, Fast
Identity (no measured block map).

**Reuse in TagScribeR.** The FLUX.2 VAE module; Krea 2's Qwen3-VL-4B builder (`TS: training/families/krea2/embedder.py`;
audit 07 notes it keeps the vision tower); the ConvRot decode (`TS: training/families/minimax_h3/weights.py`);
multi-reference edit pairs; Automagic v3; INT8 / NF4 / swap / compile. New shared work: the `save_fused` hook pair
in the LoRA writer and resume reader.

**ROCm.** Complex RoPE (as Z-Image). Masked attention uses the cuDNN-first preference list with fallbacks (S,
`fibo2/model.py:102, 114`). The INT8 attention kernel is workbench only. The VAE fix is already in TagScribeR.
Nothing else platform-specific was found in the package.

**Port size: M-L** (1,896 upstream lines; the fused-layout round trip is the risky part, because a wrong fuse
writes a LoRA that loads in ComfyUI and does nothing: upstream measured "0/210 keys" for the first format it
tried). What only a real run confirms: the saved LoRA in ComfyUI, the memory plan, the Turbo previews.

### 2.4 Vimoo 1.0 Lite (new, v9.0.7; video)

**What it is.** Vimoo-AI's 9.5B video transformer: one checkpoint for text-to-video and image-to-video, 720p at
24 fps, 40 dual-stream blocks over a 2x2 hidden patch merge (S, `families/vimoo.py:1-8`, `vimoo/model.py:1-14`).
`experimental=True`. Fizgig's own words: "its quality and performance sit closer to Wan 2.2 5B than to the newer
video models" (N).

**Licence (per upstream, `docs/VIMOO.md`).** Model, distilled LoRA and ComfyUI nodes Apache-2.0; Qwen2.5-VL-7B,
SigLIP and the Wan 2.2 VAE Apache-2.0; "the Glyph-ByT5 weights come from Glyph-SDXL-v2, which states no licence on
Hugging Face". Not gated.

**Files** (S, `families/vimoo.py:71-97`; all in `Vimoo-AI/Vimoo-1.0-Lite-ComfyUI`): DiT 19.3 GB, Wan 2.2 TI2V VAE
2.82, Qwen2.5-VL-7B text tower 14.1, Glyph-ByT5 0.89 (required), SigLIP so400m 0.86, distilled LoRA 2.27. About
40 GB.

**Training** (S, `vimoo/driver.py:1-8, 23-38, 359-420`). Photos train as one-frame clips; clips must be 24 fps,
`4n + 1` frames (up to 241), edges in multiples of 32. Text-to-video objective: the 49 conditioning channels stay
zero; `t = sigmoid(ln 8 + 2.5 z)`; target `noise - clean`; MSE. An "Image-to-video share" option (25% or 50% of
clip steps start from the clip's own first frame). Latents 48 channels, /16 spatial, /4 temporal.

**Presets** (S): Vimoo 1.0 Lite (rank 16, 1e-4, 40 epochs, 0.25 MP, Muon, compile Off); Slider (rank 8, 2e-4, 16
epochs); Fine-tune (1e-5, 10 epochs). Optimizers `adamw`, `adamw8bit`, `automagic3`; LoRA and LoKR.

**Memory (Fizgig's figures).** With the default 2-second preview on an RTX 5090: bf16 28.0 GB, INT8 19.5, NF4
15.9. On a 16 GB laptop RTX 4090: NF4 9.6 GB (3.0 s a step), INT8 13.1 (3.8 s). Under 20 GB Auto picks NF4 first.
INT8 here is weight-only, because the faster INT8 "showed as mottled, flickering backgrounds" on this model.

**Previews.** Clips: 12 steps, CFG 1, on the distilled LoRA's own sigma list; or 50 steps, CFG 4 with guidance
rescale 0.85. Sample length is an option, "Still (1 frame)" included.

**Declared abilities.** Masked training (stills), sliders, fine-tune, loss watch, caption dropout. `not_offered`
names five with reasons: training maps, batch training, INT8 attention, detail crops, resolution stages.

**What TagScribeR lacks for it.** A clip dataset layer (PORT_PLAN stage 6); 5-D latents through the cache and
loop; three new encoders (the Qwen2.5-VL text tower through transformers' `Qwen2_5_VLTextModel`, Glyph-ByT5 on the
CPU, SigLIP); the Wan 2.2 TI2V VAE (916 lines, vendored from Wan-Video, Apache-2.0); **`einops`** (S,
`vimoo/vae.py:9`), which PORT_PLAN already lists as a package to add. TagScribeR's Qwen Image 2.1 VAE is described
as "Wan 2.2-style" (`TS: training/families/qwen_image21/vae.py:21`) but is a different network (RGBA, 64 channels);
not reusable as is.

**ROCm.** The Wan 2.2 VAE's `AttentionBlock` is one head over the whole channel width (S, `vimoo/vae.py:237-268`),
512 channels or more: the wide-head bug applies and Fizgig has no fix on this path. The causal 3-D convolutions
were not assessed on ROCm. The DiT's attention already goes through `fizgig.modules.sdpa.sdpa_backend_ctx` (S,
`vimoo/model.py:158-166`), the pattern TagScribeR uses.

**Port size.** L for stills and one-frame previews; XL with clips, image-to-video and clip previews. Value for an
image-dataset app: low today. Recommendation: after stage 6, and only if users ask.

### 2.5 MOSS-Image since the baseline (for the port in progress)

TagScribeR's MOSS-Image port is from `9162fb20`. Upstream then changed, in order (N unless marked):

- Kromoss removed as preview model and as a base to train on (v8.7.7); the rows carry `replaces=` so a saved
  Kromoss path moves back (S, `families/moss_image.py`).
- Clean-end share became a schedule: Slide (v8.8.1), `60:7:30:8:3` (v8.9.94), then **reversed to `30:7:60:8:3`**
  with Muon (v9.2.5); fine-tunes `11:4:30:8:2`; the alternate epochs train at half the learning rate (S,
  `_LORA_SHARE`, `_FT_SHARE`, `kind_values`). A fine-tune's schedule counts rotations, not epochs (v8.9.94 fix).
- Previews: the SFT look LoRA (Fizgig's own, 2.07 GB, `fetch_on_use`) and, from v9.2.2, the default recipe "Base
  + Turbo LoRA at 0.6, 8 steps, CFG 2 + the SFT look LoRA"; the Turbo LoRA is now `lodestones/Kromoss`
  `8-steps-dmd-lora.safetensors` and pairs with Base (S). Faster 768 px previews are an opt-in under 20 GB.
- ComfyUI's 4-bit text encoder (`w4a8`) loads (N v8.9.4), although the description's comment still says the w4a8
  build "doesn't load" (S, `families/moss_image.py`, `text_encoder_builds` comment). Contradiction, not resolved.
- Identity blocks 24-35: Fast Identity Mode, Extract's Identity mode and Studio presets (v9.0.6; S).
- Presets: Fast (rank 16, 0.25 MP) is the default (v8.9.93); "Turbo Train (Experimental)" added (v9.2.5; rank 32,
  alpha 16 since v9.3.0); all on Muon; `compile_boundary` moved to "inside" for PyTorch 2.15.
- 16 GB of system RAM is enough (v8.7.5); a Windows access violation at the first preview fixed (v8.9.4).

Recommendation: finish the port at `9162fb20` as planned, then move MOSS-Image to the target in one follow-up
change together with Muon, because the preset values, the Clean-end default and the preview recipe all changed
together and upstream says the new share order was chosen for Muon.

### 2.6 What on AMD ROCm is likely to be a problem

| Item | Families | Status |
|---|---|---|
| Forced cuDNN SDPA, no fallback | Flux 3 | raises on ROCm; replace with TagScribeR's `sdpa_backend_ctx` |
| cuDNN-first preference list for masked attention | Z-Image, Fibo 2.0 (a comment in `fibo2/model.py` says it was first measured on MOSS-Image; MOSS's own file was not read) | should fall back; route through `training/modules/sdpa.py`; unverified |
| VAE single head wider than 256 | Z-Image (FLUX.1 VAE via diffusers), Vimoo (Wan 2.2 VAE) | wrong on ROCm without TagScribeR's fix; Fizgig has none on these paths |
| FLUX.2 autoencoder | Flux 3, Fibo 2.0, MOSS | already fixed in `TS: training/modules/flux2_vae.py` |
| Complex-valued RoPE | Z-Image, Fibo 2.0 | not verified on the owner's build |
| `torch.compile` (Triton, MSVC) | all | as today: bails to eager; Vimoo's presets set compile Off |
| comfy-kitchen INT8 attention | all | workbench only; leave out, as audit 07 decided |
| 3-D causal convolutions | Vimoo | not assessed |
| PyTorch 2.15 memory gains (gradient checkpointing keeps far fewer activations) | all | "AMD ROCm installs keep their own PyTorch" (N v9.2.0): Fizgig's new memory figures were measured on 2.15 and may not hold on the ROCm wheel TagScribeR pins |

The last row matters for honesty in docs: figures taken from `c11134a2` were measured on PyTorch 2.15 with CUDA.

### 2.7 Ranking and port order

1. **Z-Image Turbo and Base, in one change.** Apache-2.0, not gated, smallest code, popular, trains from 12 GB
   (Fizgig's plan), and Base costs almost nothing once Turbo exists.
2. **Flux 3 Image Klein.** Already planned and audited; three of its five TagScribeR-side items are done.
3. **Fibo 2.0.** Shares the FLUX.2 VAE and the Qwen3-VL-4B loader with Flux 3; adds the fused-layout LoRA writer.
   Its JSON captions suit TagScribeR's captioning (section 5, idea 6).
4. **Vimoo 1.0 Lite.** After the clip layer of stage 6.

PORT_PLAN's recorded order is MOSS-Image, Flux 3, Z-Image Turbo. Moving Z-Image ahead of Flux 3 is a change of a
recorded decision: question 1 in section 7.

---

## 3. New tools and features

For each: what it does for the user; where it lives upstream; what it needs; how it maps onto TagScribeR.
Several tools named in the brief (Studio, LoRA Royale, the Profiler, Extract, LoRA Isolation, Multi Concept, the Look
Consistency Filter, the loss watch) **predate the baseline**; section 3.10 lists only what changed in them.

### 3.1 Masked training, Make masks, SAM 3.1, click-to-mask and the mask painter (v8.9.6 to v9.0.3)

**For the user.** A greyscale mask beside a picture says which part counts: white fully, grey partly, black at a
floor the user sets (0.1 by default). The picture still trains whole. Optional prior preservation outside the mask
(Off, 0.5, 1, 2): one extra forward with the LoRA off teaches the LoRA to leave the outside as the model draws it.
Fizgig reports that masks "significantly raised both the best likeness a run reached ... and how quickly it got
there" on character LoRAs (Fizgig's claim, N; no figures given). Masks are made three ways: a person segmenter with
eight choices (person, person without clothes, face and hair, body and clothes, skin only, clothes only, everything
but faces, background only), SAM 3.1 from typed words with a "leave these out" list and a two-picture preview, and a
painter (brush, eraser, softness, strength, undo, three views, SAM click to add or right-click to remove). "Only my
subject" keeps the person whose face matches the Look Filter's baselines.

**Upstream.**

| File | Lines | Role |
|---|---|---|
| `FZ: dataset/masks.py` | 203 | naming (`photo-masklabel.png`, OneTrainer's suffix), `is_mask_file`, `load_mask` onto the latent grid at train time (never cached), problems (all black, wrong shape) (S) |
| `FZ: dataset/automask.py` | 221 | MediaPipe Selfie Multiclass on LiteRT, CPU; six classes; `keep_figure` for "only my subject" (S) |
| `FZ: dataset/sammask.py` | 248 | SAM 3.1 from words, preview, click point, idle unload after 120 s (S) |
| `FZ: sam3/` | 3,130 | SAM 3.1 detector, tracker and text encoder, ported from ComfyUI (S, headers) |
| `FZ: web/mask_edit.py` | 96 | serve picture and mask upright, save, SAM click (S); the painter itself is in `web/static/app.js` (not read) |
| `FZ: families/driver.py` | +158 | `weighted_loss`, `picture_losses`, `prior_base`, `with_prior` used by every driver (S, the imports in each driver) |
| `checks/test_masks.py`, `test_mask_prior.py`, `test_mask_edit.py`, `test_sammask.py`, `test_minimax_masks.py` | 452 | upstream tests |

**Needs.** `ai-edge-litert` (new requirement, "numpy-only, wheels for Python 3.10-3.14", S `requirements.txt`);
the segmenter model, 16 MB, Apache-2.0, fetched from Google's address and checked against a pinned SHA-256 (S);
SAM 3.1 `Comfy-Org/sam3.1` `checkpoints/sam3.1_multiplex_fp16.safetensors`, 1.75 GB, "the SAM License" (per
upstream's notices, not read); a bundled CLIP tokenizer (MIT). **Licence of the code:** the five `sam3/` files are
GPL-3.0, taken from ComfyUI, and upstream keeps them outside its Apache-2.0 licence (S, file headers and
`docs/THIRD_PARTY_NOTICES.md`). TagScribeR is GPL-3.0, so they can be ported with their headers. SAM 3.1 runs on
the GPU ("make masks when nothing is training"); its attention paths on ROCm were not assessed.

**In TagScribeR today.** Nothing knows a mask. `TS: core/dataset.py scan_images` lists `photo-masklabel.png` as an
image, so it shows in the Gallery and gets captioned; `TS: training/pipeline.py preflight` then counts it as "image
with no caption" and refuses to start (S, both read). **A dataset brought from OneTrainer with masks cannot be
trained without moving the masks out.** That is worth fixing on its own, before any mask feature.

**Mapping.** Trainer: the loss helpers and the `loss_mask` / `prior` arguments per driver, `load_mask` in
`training/dataset.py`, three `ss_` metadata keys. App: `is_mask_file` in `core/dataset.py` and
`training/dataset.py`; a mask badge and overlay in the Gallery inspector; `has:mask` / `missing:mask` in
`core/query.py`; masks follow their pictures in `core/fileops.py`, export and the Editor's crop and resize (Fizgig
lists the same rule); a "Masks" mode in `tabs/editor.py` for the painter (a `QPainter` canvas suits this better
than a web page); mask jobs through `tabs/workspace/jobs.py`. Size: trainer M, mask awareness S, make-masks M,
SAM 3.1 M-L, painter M.

### 3.2 Training maps and the trigger-word map (v9.3.0 to v9.4.0)

**For the user.** With "Training maps" ticked, three training pictures (chosen evenly through the sorted list, so
the same folder gives the same ones) are probed before training and after every epoch, "a few seconds an epoch"
(Fizgig's words). The viewer overlays: Learned since the start (green better, magenta worse), Not learned yet,
Where the LoRA acts, and Trigger word (the share of each area's attention that goes to the trigger word). Three
fixed noise levels (0.25, 0.5, 0.75) with fixed noise, so only the LoRA changes between epochs. With several
subjects each gets a picture. v9.4.0 adds the same maps in Studio for any LoRA, plus "What your tweaks change" and
a per-block-group map.

**Upstream.** `FZ: training/train_maps.py` (329; S, header and class list): "loss" = per-position squared error
against `driver.map_target`, "effect" = |prediction with the LoRA at 1 minus at 0|; no gradients, EMA swapped in,
RNG restored, a failure is never fatal. `FZ: training/attn_probe.py` (130; S): hooks `layout` and `see` called
from each model's forward and attention. Output: `<output>/training_maps/<name>/probes.json` and `maps.jsonl`.
The viewer is in `web/static/app.js` (not read). Studio's part: `FZ: scripts/studio_worker.py:928 cmd_map`.

**Needs.** No package, no download. Per family: `noise_latents` and `predict` (TagScribeR ported these for prompt
sliders), `map_target` (a few lines), and for the trigger map two hook calls in the model code plus `trigger_text`.

**Mapping.** `training/train_maps.py` ported nearly unchanged (torch, no Qt); a viewer dialog beside
`TS: tabs/train_dialogs.py LossChart` drawn with `QPainter` (PORT_PLAN: charts stay on `QPainter`). Size: maps M,
trigger map M more (it touches every family's model file), Studio maps wait for a workbench (stage 7).

### 3.3 Per-image loss window and loss-watch changes (v8.8.3 to v9.3.0)

**For the user.** A "Per-image loss" button on the run card: each picture's thumbnail, its residual by epoch (its
loss minus the dataset's mean at the same noise level), its verdict, and its caption to edit; the edit is
re-encoded at the next epoch. A "Worth a look" filter and a Refresh button. Auto-recaption lets the user pick the
caption task for the first and the second try. The watch now works at any batch size (v9.0.6) and throttles less:
a picture is marked stuck only when it is a real outlier, and its learning rate stops at x0.25 instead of x0.1
("replaying 39 real runs, the old rule had held back about a third of every dataset late in a run": Fizgig's
figure, N).

**Upstream.** `FZ: web/image_loss.py` (218; S header), `FZ: families/loss_watch.py` (+118 / -36),
`FZ: training/loss_logger.py` (+70 / -24) (diffs not read line by line: the other audit's ground).

**In TagScribeR.** `TS: tabs/train_dialogs.py ProblemImagesDialog` already lists problem images with caption
editing, and `TS: training/loss_watch.py` already recaptions with any provider from `inference/` (its header says
so), where Fizgig's head commit states auto-recaption "always uses Qwen3-VL ... whatever the Captions tab uses"
(S, the target commit's message). TagScribeR is ahead on the captioner and behind on the per-image chart and on
the throttle change. Size: chart S-M; the throttle rule is a numeric change for the core audit.

### 3.4 Detail crops and the Style score (v9.0.0)

**Detail crops, for the user.** For a style LoRA: each picture also trains as one square crop at the picture's own
scale, so brushwork and grain train pixel for pixel. The spot is random but fixed per picture (seeded from its
name), moved off flat areas, the same centre at every size. Crops are captioned once and the caption is shared
across sizes; "train on the crops only" parks the full pictures in a folder that never trains.

**Upstream.** `FZ: dataset/detail_crops.py` (343; S): "No torch: Pillow and numpy only"; crops live in
`fizgig_detail_crops/<size>mp/` with a manifest; a switch file in the training folder; made at Start for the run's
size. `FZ: scripts/detail_crops.py` (68).

**Style score, for the user.** A 0-100% number per preview: how close it is to the training set's look (texture,
palette, line, medium) rather than to a face. **Upstream.** `FZ: web/gallery_style.py` (216; S): DINOv2-base patch
tokens' mean and spread, on the CPU; a preview's score is the mean similarity to its 8 nearest of up to 256
training pictures; calibrated 0.75 to 0.95 raw (Fizgig's calibration on one artist's 240 pictures). **Needs.**
`facebook/dinov2-base`, about 350 MB, Apache-2.0 (per upstream).

**Mapping.** Detail crops: a pure dataset operation, `core/` plus the Editor and Datasets tab; TagScribeR can add
what Fizgig cannot (section 5, idea 5). Style score: `inference/` gets a small embedder; the same vectors serve
VISION theme B's embedding index (PORT_PLAN decided "opt-in per folder, about 400 MB"): DINOv2 and that index
should be one decision, not two downloads. Size: crops M, style score M.

### 3.5 Captioning and caption editing

- **Caption tasks per kind of LoRA** (v8.9.96; S, `FZ: krea2/caption_text.py:86-200`): every length in a person
  version and an object or product version; a concept task that names the concept in the user's own words in every
  caption; a style task. TagScribeR has twelve instruction presets (`TS: inference/prompts.py`), none for objects
  or concepts. Size S; a clear gain.
- **A secondary captioning model** (v8.9.95; S for the 4B / 8B detection in `FZ: krea2/embedder.py:194-210`):
  Qwen3-VL 8B (fp8 10.6 GB or bf16 17.5 GB). TagScribeR already loads many VLMs: ahead.
- **Table view and "add to every caption"** (v8.9.98; S, `FZ: web/captions.py add_text`): thumbnail, name,
  caption, word count, size, mask, sortable; add text at the start (after the trigger word) or the end with a
  preview. TagScribeR has batch prefix / suffix (`TS: tabs/workspace/panels.py BatchPanel`) and a grid, no table.
  A table view with a mask and loss column is S-M.
- **Captions in any encoding** (v8.9.0; S, `FZ: web/captions.py read_text`). TagScribeR fixed UTF-16 captions in
  its own review (PORT_PLAN, fixes list). Level.
- **Prompt boxes no longer spellcheck** (N). Not relevant to Qt.

### 3.6 Faces: drawn characters, likeness and the Look Filter

- **Face backends** (v8.9.0; S, `FZ: face_utils.py:1-62`, `docs/FACE_BACKENDS.md`): the face finder can be
  InsightFace, a YOLO head model run through `onnxruntime` (no ultralytics, no PyTorch), or both merged; the look
  match can be ArcFace, DINOv2 on the face crop, or decided per dataset from 24 probed pictures. Default YOLO
  model: an anime head model stated as MIT; a furry model the user converts once
  (`FZ: scripts/convert_yolo_to_onnx.py`, 77). `face_utils.py` grew from 441 to 1,077 lines; 784 lines of tests.
- **Likeness** against up to three chosen photos in LoRA Royale; a line under every ArcFace score saying the
  practical ceiling is about 72% (S, `web/static/app.js:2347-2349`).
- **Look Filter** fixes: it no longer scores mask files; baselines follow a folder change (N).

**Mapping.** PORT_PLAN stage 7 (face tools) and the decided interim (YuNet through `onnxruntime`). The YOLO-via-ONNX
route fits what TagScribeR already installs. Port the face module at the target, not the baseline. Size M-L.

### 3.7 Dataset handling in the trainer

- **Train on sub-folders** (v8.9.92; `checks/test_subfolders.py`, 297 lines of tests): each sub-folder its own set
  with its own cache; Fizgig's own folders skipped; Start names an uncaptioned sub-folder. TagScribeR's workspace
  can open a folder recursively (`TS: core/dataset.py`) but the trainer reads one level
  (`TS: training/dataset.py glob_images`, "not recursive"). Size M.
- **EXIF rotation read upright everywhere, training included** (v8.9.5; S, `FZ: utils/upright.py`). TagScribeR's
  trainer still reads pixels as stored and warns (`TS: training/pipeline.py:619-641`, whose text says "as Fizgig
  does": no longer true). A changed upstream behaviour in the core; flagged for the other audit. A latent cached
  sideways must be re-encoded once (upstream does).
- **Transparent pictures** (v9.0.3): on Qwen Image 2.1 a PNG trains with its alpha (`rgba`); on every other model
  it is laid on white instead of on whatever colour sits behind the transparent pixels. TagScribeR converts or
  passes RGBA as stored (`TS: training/dataset.py:419-422`); what happens to hidden colour was not traced.
- **Cache keyed on size and date** (v8.7.6), **Resume rebuilds a deleted cache** (N).
- **Start at a smaller size** (v8.9.98) and **caption dropout for every picture model** (Off, 0.05, 0.10; S,
  `FZ: families/train.py:1585-1605`). TagScribeR already has caption dropout as its own extension
  (`TS: training/params.py`, CAPTION_DROPOUT) through the dataset layer; Fizgig's uses the empty caption's
  encoding made once. Two mechanisms for one idea: a question for the core audit.

### 3.8 Runs: starting, watching, changing, continuing

All (N) unless marked; each is small and independent.

- **Warnings before Start with "Start anyway"** (S, `FZ: families/launch.py:282-345`): pictures with no caption,
  captions without the trigger word, a learning rate outside `start_lr_bands`. TagScribeR's preflight
  (`TS: training/pipeline.py:423-608`) has errors, warnings and infos but no trigger-word or learning-rate check,
  and treats a missing caption as an error.
- **Start downloads a missing model, then starts** (v9.0.1). **Start cannot fire twice** (v8.9.92).
- **Run summary** card: time, speed, peak VRAM, the live learning rate, grad norm, loss, likeness, and the run's
  own settings (S, `web/static/app.js:2131-2188`). UI_UX_RESEARCH already proposes run tiles and a finished-run
  card; this confirms the content.
- **Each run's log saved to a file**, newest 20 kept; an out-of-memory end points at Continue (v8.9.7, v8.9.93).
- **Continue** any run from the runs list, queue it, edit a queued run (v8.9.92); **Resume takes a changed learning
  rate, save schedule, block swap, picture size and previews**, each marked "Applies on Resume" (v8.9.98; S,
  `FZ: families/form.py:287`). PORT_PLAN decided the opposite for precision and block swap on another card ("keep
  the saved precision and block swap"); block swap on the same card is a different case, worth one line there.
- **Preview and save schedules** ("every N, then from epoch Y every Z"; v8.7.8); **prompt sets** saved by name and
  shared across models (S, `FZ: web/server.py:542-549`); **prompts as a numbered list**; **several edit
  instructions previewed per round** (v9.2.4); **arrow keys in the preview viewer** step epochs and prompts
  (v9.0.0); **the loss graph shows the epoch average, a running average and the live learning rate** (v9.0.3);
  **speed excludes everything between epochs** (v9.2.4).
- **Sidebar meters**: free disk on the output drive, background jobs, what is loaded on the GPU (S,
  `web/server.py:240`). Matches UI_UX_RESEARCH M2 (the global job chip).

### 3.9 Helpers, Move a run, Training share, Pods, Vast.ai (v8.9.0 to v9.0.2)

**For the user.** Another computer running Fizgig (a second PC, a second GPU in the same PC, a rented pod) can
render previews so training never pauses, take over a run ("Move to...", "Bring it back", "Carry on here" from the
last save), or train a share of each epoch with the two LoRAs combined every N epochs (experimental). Checkpoints
and previews come home as they are saved; a pod can delete itself once everything is home and checked; a pod is put
on the same version as the user's computer.

**Upstream.** `FZ: web/helpers.py` (4,505, new; S header), `FZ: families/preview_worker.py` (184; S),
`web/pods.py` (461), `web/pod_version.py` (211; S header), `web/pod_stop.py`, `web/lan.py`, guides `HELPERS.md`,
`HELPER_PREVIEWS.md`, `MOVE_RUN.md`, `TRAINING_SHARE.md`, `VAST.md`. Control is by files in the run folder
(`.helper_previews.json`, `.helper_rounds/<epoch>/`), the same idea as TagScribeR's run folder.

**Mapping.** VISION theme G, decided: the headless runner and the portable run bundle first, then a localhost
runner service, with a security review before `main`. Two parts are separable and cheap once a render tool exists
(VISION section 4, item 2): **previews in a second process on a second GPU of the same PC**, and **Move a run** as
"export the bundle, import it there". Fizgig's deploy links carry a referral code (N, the Vast.ai link in the
v8.9.0 note); PORT_PLAN already decided "no referral codes". Size: XL as a whole.

### 3.10 The workbench tools (all predate the baseline): what changed

- **Studio** (N): opens a LoRA on the model read from its file; settings wait for an Update button; the Baseline
  shows while the Tweaked render is still coming; untouched sliders reuse the Baseline (one render instead of two);
  settings travel inside the saved PNG; block values can be typed; seed 0 means a fresh seed; **Maps** (3.2);
  "Turbo Preview" renamed "Reuse unchanged blocks". `web/studio.py` +168, `scripts/studio_worker.py` +89.
- **LoRA Royale**: likeness from up to 3 photos; a sheet above 24 pictures renders again (N).
- **Profiler**: starts with the LoRA's own first sample prompt; a report for LoRAs with no face; the "Open in
  Studio" button removed because "its block suggestions rarely helped on the newer models" (N).
- **Extract**: any rank; **Identity mode** keeps 4x the rank in the identity blocks, for MOSS-Image (24-35) and
  Qwen Image 2.1 (10-14) (S, `FZ: extraction/model_diff.py` +56, `checks/test_extract_identity.py`).
- **LoRA Isolation and Multi Concept**: Isolation needs two or more subjects; side-by-side pictures use face crops
  first; per-subject folders fixed; training maps per subject (N; `families/multi_person.py` +73).
- **Prompt sliders: "Leave these alone"** anchors with a weight (v8.9.93; `checks/test_slider_anchors.py`).

All of these sit behind PORT_PLAN stage 7 and VISION's render tool. Nothing here changes that order.

### 3.11 taeh3 early previews (v9.2.0)

A 22.7 MB tiny decoder for MiniMax H3 latents (MIT), used only for Studio's mid-render look: 0.14 s against 1.55 s
for the full 4.5 GB decoder, PSNR 30.2 dB (Fizgig's figures). Pinned to one upstream commit and checked by SHA-256
(S, `FZ: minimax/taeh3.py:1-28`); `minimax/taehv.py` (534, vendored). It is a workbench feature upstream; a
"preview of the preview" during H3 training would be TagScribeR's own use of it. Size S once there is a place to
show it.

### 3.12 Installer, updater and downloads

- **Updater health check** (S, `update_fizgig.py:207-375`): before updating it probes the Python environment in a
  child process (torch imports, sees the GPU), says what this update does to PyTorch, and counts down 10 s; Space
  updates, R rebuilds the environment from scratch after a second confirm; after the update it probes again and
  offers the rebuild if something broke. It also detects a ROCm torch so the wrong vendor's updater refuses
  (`torch_kind`, three distinct exit codes). TagScribeR's `tools/install.py` (350 lines) has no health probe.
- **Telemetry off** (S, `FZ: utils/ort_telemetry.py`, `checks/test_no_telemetry.py`): `onnxruntime` reports a
  device id unless told not to. TagScribeR sets `HF_HUB_DISABLE_TELEMETRY` (`TS: core/hardware.py:84`) but not
  `ORT_DISABLE_TELEMETRY` or `onnxruntime.disable_telemetry_events()` (S, grep), and it uses `onnxruntime` for
  the WD tagger. A two-line privacy fix that fits TagScribeR's "nothing left your machine" stance.
- **Public downloads survive an expired Hugging Face login** (S, `FZ: utils/hf_public.py`): retry once without the
  token on a 401, and say so in one line.
- **A model row pointing at another model's file is repaired** by tensor signature (section 2.1).
- **`ai-edge-litert`** is the one new package of this scope (masks); `einops` is needed only by Vimoo's VAE.

### 3.13 Themes, scroll memory, News, loading tips

- **Ten themes** (S: nine `data-theme` values in `web/static/style.css` plus the default), three "for eyes":
  colour-blind safe (never red against green), high contrast, dim. TagScribeR's token theme module
  (`TS: tabs/theme.py`) is the place for the same three.
- **Each tab keeps its scroll place** (S, `web/static/app.js:9402-9421`). In Qt each tab is its own widget and
  keeps its scroll bar; nothing to port unless a tab rebuilds its contents.
- **News card** (S, `FZ: web/news.py`, a list baked into the release) and a **loading-screen tip** (S,
  `web/static/index.html:51`). For TagScribeR: a "What's new" entry in the Help Center, written with each release.

---

## 4. UI / UX lessons (ideas, not layouts)

| # | Idea | Where upstream shows it | Suggestion for TagScribeR |
|---|---|---|---|
| 1 | **Warn, never block, on the usual mistakes.** One line each, with examples and "Start anyway" | `FZ: families/launch.py:282-345`; `app.js:1618-1624` | Add the trigger-word and learning-rate-band checks to `training/pipeline.py preflight` as warnings; show warnings in the Ready check card (UI_UX M3) with a "Start anyway" button |
| 2 | **A missing ability is a decision written down.** `not_offered = {ability: reason}`, and a test fails when a family lacks a common ability without naming it | `FZ: families/description.py` (`not_offered`), `checks/test_family_abilities.py` | Add the field and the test; show the reason as the tooltip of the greyed control ("Not for Vimoo: clips train one a step") |
| 3 | **A note appears only when it is true.** The warm-up note waits 30 s, shows the speed now against the start, ends with "Up to speed" | `FZ: families/train.py:2536-2575` | Same rule for the Train tab's status line: measure, then say; never a standing "this may be slow" |
| 4 | **Say how each result was made.** A line over an enlarged preview: which model, which look LoRA, Turbo or not, and which gives the better read | `preview_labels`, `gallery_note` in `FZ: families/description.py`; `.prompts_<name>.json` (already ported) | Show it in the samples viewer and write it into the preview PNG's metadata (TagScribeR's Metadata tab can then read it back) |
| 5 | **Never a score without its ceiling.** "Peak ArcFace is about 72%" under every likeness number | `web/static/app.js:2347-2349` | Any score TagScribeR adds (blur, style, likeness, readiness) carries one line on what a good value is |
| 6 | **A changed default moves saved forms once, and says so.** | `value_moves` (`FZ: families/form.py:1288`), `replaces=` on model rows | When a mirrored default changes, migrate saved presets once with a note in the import report (`training/presets.py` already reports) |
| 7 | **Defaults that follow the card.** Small-card figures, NF4 first under 20 GB, checkpointing off above 40 GB, a note under Batch size saying what Auto would pick now | `train_memory_small`, `auto_precisions_small`, `FZ: web/batch_plan.py` (the planner runs in a short child process, never in the UI) | The Train tab's memory line could show the plan "at this setting" live, computed in a child as upstream does |
| 8 | **Long jobs grey their buttons and name each stage.** "Sending the LoRA, loading the model there, waiting its turn, rendering"; cache stages count up; the button reads "Starting..." | N v8.9.93, v8.9.97; `app.js:1632` | Stage names in the Train status tiles; disable Start while a start is pending |
| 9 | **Destructive or costly actions ask twice, in place.** Stop turns into "Confirm stop" for 30 s | `app.js:10189` | Replaces a modal dialog; fits UI_UX's button-state proposal |
| 10 | **Tell the user what applies when.** Rows marked "Applies on Resume" while a run is paused | `FZ: families/form.py:287`, `app.js:1663` | A small badge on those rows while a run is paused |
| 11 | **Wording for every kind of LoRA.** Hints say what applies to a person, a style or an object; face tools say they score faces and what to do otherwise | N v8.9.96 | A pass over `tabs/help_content.py` and the Train tab tooltips with the same three cases |
| 12 | **Hints tied to the dataset in front of the user.** "With {n} images, this model can take longer to converge" beside Epochs | `small_dataset_hint` (`FZ: families/moss_image.py:309`) | TagScribeR knows the open folder at all times: put count-aware hints on the Train tab |
| 13 | **Show only what the current choice uses.** Cards closed by default, extras under a divider, rows hidden per mode | N v8.9.98, v9.0.0 | Already TagScribeR's "simple view"; apply it per kind of LoRA |
| 14 | **A run never dies silently.** OOM ends with what to do; a failed save retries; every log is a file; the last line names the file | N v8.9.7, v8.9.93 | Per-run log file in the run folder and a "what to do" line for the three common failures |
| 15 | **Check before you change the user's environment.** The updater's health probe, countdown and rebuild | `update_fizgig.py:334-375` | `tools/install.py --check`: probe torch and the GPU in a child, print what an update would change |

---

## 5. Where TagScribeR is ahead, and nine ways to pass Fizgig on dataset preparation

**Ahead today** (read in both code bases): a workspace shared by every tab with a filter language
(`TS: core/query.py`) against Fizgig's one-folder-at-a-time Captions and Image Prep ("to caption or mask a
sub-folder, choose it as the training folder", N v8.9.92); captioning with any provider, where Fizgig captions and
recaptions with one model family; the health scan, metadata and privacy tools, the editor and export, none of which
Fizgig has; atomic caption writes with backups.

**Behind today:** masks, crops for style, face tools, per-kind caption tasks, the pre-run warnings.

Ideas, each grounded in a gap read above:

1. **Masks as a dataset layer, not a trainer option** (M). Mask badge and overlay on Gallery cards, `has:mask`,
   `missing:mask`, `mask:<30%` filters, masks that follow every file operation, and export that includes them.
   Fizgig shows a mask through a "Show mask" button on one picture at a time.
2. **Mask health in the Health scan** (S-M). Fizgig names two problems at Start (all black, wrong shape). Add:
   coverage far from the folder's median, a mask that misses the detected face, a mask older than its picture, and
   "the latent grid sees N cells" (a small mask at 0.25 MP can be a handful of cells; Fizgig's own guide warns that
   a very small mask "trains hard on a few spots").
3. **Any segmenter, like any captioner** (M-L). A mask-provider layer in `inference/` beside the caption providers:
   the 16 MB person segmenter on the CPU, SAM 3.1, and later the owner-approved RMBG 2.0 from the feature roadmap.
   Fizgig hard-wires two.
4. **A desktop mask painter with the Editor's tools** (M). Pressure-free brush on a `QPainter` canvas, zoom to
   pixels, before / after, undo, and batch actions over the selection ("grow by 8 px", "feather", "invert") that a
   web page makes awkward. Click-to-mask through the same provider layer.
5. **Detail crops the user can see and steer** (M). Fizgig picks a random spot and moves it off flat areas. In the
   Editor: show the proposed square on the picture, drag it, add a second crop, and keep the choice in a sidecar so
   every size uses it. Health can say which pictures are too small for a 1:1 crop at the target size.
6. **Structured captions for Fibo 2.0 from any VLM** (S-M). Fibo is trained on long JSON captions and its edit
   LoRAs take an `edit_instruction` key. A caption preset that asks a VLM for that JSON, a validator, and the review
   queue's diff. Fizgig offers plain captions only for it.
7. **Caption tasks per kind of LoRA, with a consistency check** (S-M). Port the object and concept tasks as
   presets, then use the tag statistics panel to verify what Fizgig only instructs: that the concept is named the
   same way in every caption and the trigger word comes first.
8. **One pre-run report** (M). Join Fizgig's three Start warnings with what only TagScribeR has: health flags,
   bucket fit, token limits, mask problems, EXIF rotation, hidden colour behind transparency. This is VISION's
   readiness page with concrete rows from this audit.
9. **Training evidence on the Gallery grid** (M). Per-image loss verdicts and the training-map pictures as
   filters and badges (`flag:stuck`, `map:probe`), so the fix happens where the picture lives. VISION lists loss
   badges; the maps and the trigger-word map add "this logo is holding the trigger word", which points at a crop or
   a caption edit in the same window.

Smaller: `onnxruntime` telemetry off (S, section 3.12); a "What's new" Help topic (S); a table view with mask,
loss and token columns (S-M).

---

## 6. Not verified

- Anything on Hugging Face: licences and gating of weights, file names after the Flux 3 rename, sizes.
- Every measurement. All are Fizgig's, on NVIDIA cards and, from v9.2, on PyTorch 2.15.
- `web/static/app.js` (10,267 lines) and `index.html` were only searched, not read: the mask painter, the maps
  viewer, the run summary, the caption table and the themes are described from release notes and from the server
  side.
- `web/helpers.py` beyond its header; the pods code; the Docker and Vast.ai files.
- The diffs of `families/train.py` (+1,404), `loss_watch.py` and `loss_logger.py`: the other audit's ground.
- Whether transformers 5.x (TagScribeR's pin) has `Qwen2_5_VLTextModel` as Vimoo's text encoder needs; Fizgig pins
  another transformers line.
- The Wan 2.2 VAE's exact attention width; SAM 3.1's attention paths on ROCm; complex RoPE on ROCm.
- Whether an uncaptioned picture trains on an empty caption (`FZ: families/launch.py:303-304`) or is left out
  (`FZ: dataset/image_dataset.py:193`): the two upstream texts disagree.
- The MOSS-Image `w4a8` text encoder: the release note says it loads, a source comment says it does not.
- SAM 3.1's weights licence text; the licence of the anime head model; InsightFace's model licence.

---

## 7. Questions for the owner (product decisions; each with a recommendation)

1. **Family order.** PORT_PLAN records MOSS-Image, Flux 3, Z-Image Turbo. Should Z-Image Turbo and Base move ahead
   of Flux 3? *Recommendation: yes.* Open licence, no token, least code, and Base comes almost free with Turbo.
2. **Fibo 2.0.** Port it, given non-commercial weights whose licence also binds the LoRAs trained on them?
   *Recommendation: yes, after Flux 3, with the licence stated on the model row and in Help,* as for Flux 3.
3. **Vimoo 1.0 Lite.** A video model in an image-dataset app. *Recommendation: not now; revisit after stage 6
   (clips) and only if users ask.* Fizgig itself calls it experimental.
4. **Masked training and the mask tools.** Bring them in ahead of the remaining stage-5 work (DOP, Multi Concept,
   Isolation)? *Recommendation: yes.* Fizgig claims a significant effect on likeness (unmeasured here), and its
   dataset half is TagScribeR's own ground. Start with mask awareness (masks stop appearing as images and stop
   blocking Start), which is small and fixes a real problem today.
5. **Where masks are made and edited.** A "Masks" mode in the Editor tab, or a new tab? *Recommendation: the Editor,*
   with an overlay toggle in the Gallery inspector, so Ctrl+1..7 keep their tabs (a recorded decision).
6. **SAM 3.1.** A 1.75 GB download under Meta's SAM licence, GPL-3.0 code from ComfyUI, GPU only in practice.
   *Recommendation: yes, optional, downloaded on first use, after the CPU person segmenter,* and not while a run
   trains.
7. **One embedding model or two.** DINOv2 (Style score, drawn-character look match) and the SigLIP / CLIP index
   already approved as opt-in. *Recommendation: decide on DINOv2-base for all three uses if it serves search well
   enough;* otherwise accept two downloads. Needs a short technical comparison first.
8. **Uncaptioned pictures at Start.** TagScribeR refuses; Fizgig now warns and lets the user start. *Recommendation:
   keep refusing by default* (it prevents the commonest silent mistake) *and add "Start anyway" only if the core
   audit confirms what Fizgig then trains on.*
9. **Sub-folder training.** Fizgig's rule is "each sub-folder its own set". TagScribeR can already open a folder
   recursively. Should the trainer follow the workspace's recursive switch, or have its own tick as upstream?
   *Recommendation: its own tick, mirroring Fizgig,* so a run never trains more than the Train tab says.
10. **Themes for eyes.** Add colour-blind-safe, high-contrast and dim themes with the token theme?
    *Recommendation: yes, with the token theme's first stage.*
11. **Second GPU in the same PC for previews.** Worth doing before the runner service? *Recommendation: only after
    the render tool exists; then it is small.*

---

## 8. Recommended order of work

Sizes: S = a day or less, M = a few days, L = a week or more, XL = several weeks. Every item ships with tests on
tiny random models or generated images, help text and docs, and is "unit-tested only" until the owner runs it.

| # | Work | Size | Depends on |
|---|---|---|---|
| 1 | Mask awareness: mask files are never dataset images, never block Start, follow their pictures; `has:mask` filter; overlay in the inspector | S-M | - |
| 2 | `onnxruntime` telemetry off; a model-file signature check before loading a VAE | S | - |
| 3 | Start warnings: trigger word missing, learning rate outside the family's band (`start_lr_bands`); per-run log file | S | - |
| 4 | Caption presets for objects, products and concepts; a trigger-first and concept-consistency check | S-M | - |
| 5 | Masked training in the trainer: loss helpers, `loss_mask` and prior per family, metadata keys | M | 1 |
| 6 | Make masks on the CPU (person segmenter, eight kinds), as a job over the selection | M | 1 |
| 7 | Z-Image Turbo and Z-Image Base | S-M + S | Muon for verbatim presets (other audit) |
| 8 | Training maps (loss, effect) and a viewer; then the trigger-word map | M + M | - |
| 9 | Mask painter in the Editor; mask health rows | M + S-M | 1 |
| 10 | Flux 3 Image Klein (new file names, cuDNN route) | M-L | - |
| 11 | Detail crops (with hand placement) and the Style score | M + M | question 7 |
| 12 | Fibo 2.0, with the JSON caption preset | M-L | 10 (shared VAE and encoder work) |
| 13 | SAM 3.1 from words and click-to-mask | M-L | 6, 9 |
| 14 | Sub-folder training; preview and save schedules; prompt sets; Continue from the runs list | M | - |
| 15 | MOSS-Image moved from the baseline to the target (Clean-end schedules, preview recipe, presets) | M | Muon |
| 16 | Face tools at the target (InsightFace, YOLO via ONNX, DINOv2), Look Filter | M-L | stage 7, question 7 |
| 17 | Helpers and Move a run, as VISION theme G plans them | XL | render tool, runner service, security review |
| 18 | Vimoo 1.0 Lite | L-XL | stage 6, question 3 |
