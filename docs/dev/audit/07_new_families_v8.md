# Audit 07 - Fizgig's families since v7.0.1: Flux 3 Image Klein, MOSS-Image, Z-Image Turbo, ACE-Step 1.5 XL

Read-only porting study. Upstream = the Fizgig copy at `.claude/fizgig-src` (HEAD `5d31b2f`, "v8.7.1 notes", tag
v8.7.1). TagScribeR = the repo at HEAD `98a48a2` (work branch, stages 1-4 and the slider hooks landed). Paths:
`FZ:` means `.claude/fizgig-src/src/fizgig/...` (or `.claude/fizgig-src/docs/...` where a `docs/` path is given);
`TS:` is TagScribeR. Nothing was run: no tests, no model loads, no GPU. The Hugging Face API was not reachable from
the session (proxy), so every statement about a repo's gating, a licence of weights or a file size is what Fizgig's
descriptions and docs say, marked "per upstream", not something checked online.

Scope: the four families `FZ: families/registry.py:13-18` lists that `TS: training/registry.py:18` does not:
`ZIMAGE` (Fizgig v7.1.0), `ACESTEP15` (v7.3.0), `FLUX3` (v8.1.0), `MOSS_IMAGE` (v8.5.0). No other family was added
between v7.0.1 and v8.7.1 (registry: Klein, MiniMax, Krea 2, Qwen Image 2.1, SDXL, Anima are the six TagScribeR has).

Status of this file: complete. Section 0 is the summary and the order; 1 the gaps all four share; 2-5 one family
each; 6 the ranking; 7 the owner questions, as recommendations; 8 what was not checked.

---

## 0. Summary

| Family | What | Code licence (Fizgig's port) | Weights licence (per upstream) | Gated | Download | Value to image users | Effort | Order |
|---|---|---|---|---|---|---|---|---|
| MOSS-Image | OpenMOSS 5B single-stream text-to-image DiT, Qwen3.5-9B text encoder, FLUX.2 autoencoder (16 Oct 2026) | Apache-2.0 (OpenMOSS pipeline, `FZ: moss_image/model.py:1-3`; Qwen3.5 text model from transformers, `embedder.py:1-5`); Fizgig's driver Apache-2.0 | Apache-2.0 (`FZ: docs/MOSS_IMAGE.md:5`) | No (`MOSS_IMAGE.md:13`) | ~35 GB (10.2 Base + 5.4 SFT + 19.3 TE + 0.25 VAE) | High: open weights, measured presets, 12 GB cards | M | 1 |
| Flux 3 Image Klein | Black Forest Labs 5.6B text-to-image + edit DiT (up to 10 references), Qwen3-VL-4B, FLUX.2 autoencoder (15 Oct 2026) | Apache-2.0 (BFL reference, `FZ: flux3/model.py:3`, `autoencoder.py:1`, `flux3/LICENSE`) | FLUX Kommunity License: non-commercial, outputs commercial under US$5M (`FZ: docs/FLUX3.md:96`) | Yes, every file (`FZ: families/flux3.py:47-55`, `gated=True`) | ~31 GB (11.1 Base + 11.1 distilled + 8.9 TE + 0.3 VAE) | High: BFL name, edit LoRAs | M-L | 2 |
| Z-Image Turbo | Tongyi-MAI 6B S3-DiT, 8-step distilled, Qwen3-4B, FLUX.1 VAE; trains on Turbo with Fizgig's own frozen adapter | Apache-2.0 (Alibaba Z-Image, `FZ: zimage/model.py:1-2`, `pipeline.py:1-2`; Fizgig NOTICE) | Apache-2.0 (Fizgig `NOTICE`, "Z-Image ... Apache-2.0") | No flag in the description | ~20.8 GB (12.3 DiT + 8.0 TE + 0.34 VAE + 0.07 adapter) | Medium: popular 8-step model, Turbo only | S-M | 3 |
| ACE-Step 1.5 XL | 4B text-to-music flow DiT (songs with lyrics), Oobleck audio VAE, Qwen3-Embedding-0.6B | Header says "MIT / Apache-2.0" (`FZ: acestep/model.py:1-3`); Fizgig's notices have no ACE-Step entry | MIT, code and weights (`FZ: families/acestep.py:211`) | No flag | ~21.8 GB (10 Turbo + 10 SFT optional + 1.2 TE + 0.34 VAE + 0.34 EQ LoRA) | None for an image-dataset app | XL (audio dataset layer, audio previews, ffmpeg) | 4 (ask first) |

Recommended order: **MOSS-Image, Flux 3 Image Klein, Z-Image Turbo, then ACE-Step only if the owner wants a music
trainer**. Reasons in section 6. All three image families need the same two shared-layer additions first (section
1.3): a preview checkpoint whose Samples-tab tick and sampling come from the description, and the FLUX.2 autoencoder
loader extended to the small decoder. Flux 3 also needs multi-reference edit pairs (TagScribeR takes one partner per
image today, `TS: training/cache.py:276-280`).

---

## 1. What all four share

### 1.1 Where the shared layer stands

TagScribeR's `training/description.py` and `driver.py` were brought level with Fizgig 7.0.1 (`TS: training/driver.py:8-14`
lists the hooks). Fizgig's description grew between 7.0.1 and 8.7.1; the fields the four new families use that
TagScribeR's `FamilyDescription` (`TS: training/description.py:97-235`) does not have:

| Upstream field | `FZ: families/description.py` | Used by | TagScribeR today | Needed for a port? |
|---|---|---|---|---|
| `ModelFile.gated`, `hint`, `download_note`, `alt_repo/alt_path/alt_label/alt_note`, `replaces`, `bulk_fetch`, `fetch_optional` | `:36-60` | Flux 3 (gated), MOSS (hint, alt int8 files, replaces, bulk_fetch), Z-Image and ACE (hint) | `TS: training/description.py:20-34` has `note`, `role`, `default_to` only | `gated` yes (a token-less download cannot work for Flux 3); the rest are Settings-row cosmetics: fold `hint` into `note`, drop `alt_*` or add as one optional field |
| `hidden`, `status_note`, `model_note`, `preset_notes` | `:283-302` | MOSS (`status_note`) | none | Optional (a help line) |
| `media`, `subject_isolation`, `multi_concept*`, `clip_spec` | `:324-338` | ACE-Step (`media=("voice",)`, `subject_isolation=False`) | none; the dataset layer is image-only (`TS: training/dataset.py:41`) | ACE-Step only |
| `finetune`, `ft_learning_rate`, `ft_regularisation` | `:400-404` | all four | none (stage 5) | Stage 5; the stage-1 guard refuses imported Fine-tune presets meanwhile |
| `preview_checkpoint_tick`, `preview_checkpoint_on`, `preview_checkpoint_steps_note` | `:418-422` | Flux 3 (distilled, 16 steps), MOSS (SFT, on by default) | `train_preview_checkpoint` + `preview_checkpoint_sampling` exist (`TS: description.py:209-210`); the tick's label is hard-coded to Klein's "4-step" (`TS: training/params.py:398`) | Yes: the label and default per family |
| `preview_every`, `preview_strengths` | `:478-481` | ACE-Step (40 epochs; strength 0.5 too) | none | ACE-Step only |
| `preview_negative: Optional[str] = None` ("None = no negative, box greys") | `:483-485` | Z-Image, ACE (CFG-free) | `preview_negative: str = ""` (`TS: description.py:204`) | Small: allow None |
| `samples_cfg_free`, `samples_text`, `preview_cfg_note`, `workbench_follows_samples` | `:310-315, :482, :492` | all four | none | `samples_cfg_free` (grey CFG and negative) is a Samples-section nicety; the rest are help text |
| `edit_references`, `reference_strength` | `:457-460, :306` | Flux 3 (10) | none; one partner per image (`TS: cache.py:276-280`, `pipeline.py:232`) | Yes for Flux 3 (and it lifts Klein to upstream's 4, Qwen 2.1 to 10: `FZ: families/klein.py:171`, `qwen_image.py:161`) |
| `slider_sources`, `kind_notes`, `slider_prompt_notes`, `ema_hint`, `long_run_hint`, `song_captions` | `:464-470, :448, :411, :414` | ACE-Step | none | ACE-Step only |
| `adaptive_lr`, `loss_watch` (hide the controls) | `:445-446` | ACE-Step (both False) | none (H3 stage 4 handled its hidden controls in params) | ACE-Step only |
| `auto_precisions` | `:370` | MOSS (`("int8", "nf4")`) | present (`TS: description.py:179`) | - |
| `compile_hint`, `compile_fullgraph` | `:391`, (TS `:171`) | Z-Image | `compile_fullgraph` present; hint none | Help text |
| `int8_attention`, `activation_cache`, `repair_*`, `lucky_zones`, `block_activity`, `workbench*` | `:486-499, :512-529` | all four | none (no workbench: stage 7) | No: workbench-only. Leave the fields out and keep `int8_attention.attend()` out of the model code (see 1.4) |
| `FamilyOption` (`options=`) | `:186-235, :550` | MOSS (2), ACE-Step (2) | TagScribeR parameters + `DRIVER_OPTIONS` (`TS: training/params.py:422-431`); settled for H3 in stage 4 (`docs/dev/PORT_PLAN.md`, "Settled") | Same rule: `MOSS_*` / `ACESTEP_*` parameters with upstream's keys, routed to `driver.configure()` |

Driver hooks the four use that `TS: training/driver.py` lacks:

| Hook | `FZ: families/driver.py` | Used by | Notes |
|---|---|---|---|
| `cache_stage(stage, datasets, args, device, aux) -> bool` | `:117-121`; called at `FZ: families/cache.py:174-175` | MOSS (text batch size 4096 for the streamed encoder, `FZ: moss_image/driver.py:158-163`), ACE-Step (its own latent and text layouts, `FZ: acestep/driver.py:235-291`) | Named in `TS: driver.py:13-14` as "not yet". MOSS's use is a batch-size hint; ACE-Step's is a whole cache layout |
| `text_cache_key()` | `:112-115` | ACE-Step (the caption carries the song length) | workbench prompt cache; not needed without the workbench |
| `media_problem(path)` | `:123-127` | ACE-Step (a bad `.json` sidecar) | audio only |
| `keep_text_encoder`, `drop_text_encoder()`, `text_encoder_gb(path)` | `:74-83` | MOSS (the 19 GB encoder kept in pinned RAM between workbench prompts, `FZ: moss_image/embedder.py:293-328`) | workbench-only; training caches once. Skip |
| `from_pretrained(cls, repo, **kw)`, `helper_dir(repo)` | `:85-98` | MOSS (tokenizer), ACE (tokenizer, silence latent) | TagScribeR has `training/hf_cache.py from_pretrained_cache_first` (used by Krea 2, `TS: training/families/krea2/embedder.py:124-128`) |
| `practice_latent`, `slider_preview`, `decode_audio`, `activity_axis` | `:210-220, :264, :475` | ACE-Step | `slider_preview` is already in TS (`TS: training/train.py:380`); the rest audio-only |
| `ft_spec`, `ft_backend`, `ft_card_plan`, `install_ft_streamer` | `:315-342` | all four | stage 5 |
| `lora_target_names`, `quant_target_names` overrides | present in TS (`:356-383`) | MOSS (SFT: attention only; quantise every block Linear) | - |

### 1.2 Packages

Flux 3, MOSS-Image and Z-Image need nothing TagScribeR does not install: torch, transformers (Qwen3-VL-4B,
Qwen3-4B / `Qwen3Model`), diffusers (`AutoencoderKL` for Z-Image's FLUX.1 VAE), accelerate (`init_empty_weights`),
safetensors, numpy, Pillow, psutil - all in `TS: requirements.txt`. None of the four imports `einops` (grep of the
four packages). ACE-Step adds an **ffmpeg binary** (`FZ: acestep/driver.py:87-106` decodes and probes through
`families/clips.ffmpeg`, which is imageio-ffmpeg's bundled binary, `FZ: lora_royale/export.py:382-384`) and
diffusers' `AutoencoderOobleck` (`FZ: acestep/vae.py:46-49`; present in diffusers >= 0.35, not checked). `PORT_PLAN.md`
already lists `imageio-ffmpeg` for stage 6.

### 1.3 Two shared pieces of work that the three image families reuse

1. **The FLUX.2 autoencoder.** Flux 3 vendors BFL's `autoencoder.py` (`FZ: flux3/autoencoder.py:1-3`) and MOSS loads
   its VAE through the same module (`FZ: moss_image/driver.py:133-136`). TagScribeR already has this autoencoder as
   Klein's (`TS: training/families/klein/vae.py`), with the wide-attention fix in its `AttnBlock`
   (`TS: klein/vae.py:73-85`). What is new upstream: a `decoder_ch` argument (128 full, 96 for BFL's small decoder
   that MOSS ships, `FZ: flux3/autoencoder.py:121-125, :157, :232`), a diffusers-name converter for Comfy-Org's
   `flux2-vae_small.safetensors` (`:195-219`), and the `encode()` / `decode()` pair that returns packed tokens with
   ids (`:164-180`). Port: one shared loader (either extend `klein/vae.py` or move it to
   `training/modules/flux2_vae.py` and import it from Klein, Flux 3 and MOSS), keeping the wide-attention call.
   The three families then share one VAE file row in spirit; MOSS's `replaces=("flux2-vae.safetensors", "ae.safetensors")`
   (`FZ: families/moss_image.py:69`) says the full-decoder file is accepted too.
2. **Preview checkpoint per family.** `TS: training/train.py:409-495 _CheckpointPreviews` already does the park / load
   / unpark handoff for Klein's Distilled. Flux 3 (distilled flow map, 16 steps, no CFG) and MOSS (SFT, 30 steps,
   CFG 4, negative) differ only in the tick's label, its default and the sampling recipe, all of which upstream
   reads from the description (`FZ: families/description.py:418-422`, `preview_checkpoint_sampling :397`). TagScribeR's
   `SAMPLE_USE_DISTILLED` label is Klein's text (`TS: training/params.py:398-399`): make the label, default and the
   Steps note come from the description.

### 1.4 CUDA assumptions in the new code, and what runs on ROCm

| Item | Where | On AMD ROCm | Port action |
|---|---|---|---|
| cuDNN SDPA forced: `sdpa_kernel(SDPBackend.CUDNN_ATTENTION)` whenever the tensor is on the GPU | `FZ: flux3/model.py:87, :94` (`packed_attention`) | ROCm PyTorch has no cuDNN attention backend; forcing one backend raises "No available kernel" (the reason TagScribeR made cuDNN a *preference* in a priority list, `TS: training/modules/sdpa.py:159-165`). Every Flux 3 training step and preview would fail | Replace with plain `F.scaled_dot_product_attention` under `training.modules.sdpa.sdpa_backend_ctx` (Krea 2's pattern) |
| comfy-kitchen INT8 attention `attend()` before SDPA | `FZ: flux3/model.py:19, :92`; `moss_image/model.py:51-53`; `zimage/model.py:20, :140` | The module itself says "AMD builds skip it" (`FZ: modules/int8_attention.py:16-17`); workbench renders only, never training | Leave the call out (TagScribeR has no workbench; `int8_attention` field not ported) |
| VAE single-head attention wider than 256 | FLUX.2 autoencoder `AttnBlock` at 512 channels (`FZ: flux3/autoencoder.py:68-82`, `CH * CH_MULT[-1]` = 512) for Flux 3 and MOSS; FLUX.1 VAE for Z-Image through diffusers `AutoencoderKL` (`FZ: zimage/pipeline.py:66-79`), also one 512-wide head | Wrong on ROCm: streaks in decoded previews, latents off by a few percent (`TS: training/modules/wide_attention.py:1-18`). Fizgig has no fix | Flux 3 / MOSS: the shared loader of 1.3 keeps `wide_attention`. Z-Image: `vae.set_attn_processor(AttnProcessor())` as SDXL does (`TS: training/families/sdxl/vae.py:45-49`), or load the FLUX.1 VAE through the same vendored class with a 16-channel config (it is the same architecture family). ACE-Step's Oobleck VAE is convolutional (Snake + conv, `FZ: acestep/vae.py:15-44`): no attention, nothing to fix |
| `torch.autocast("cuda", ...)` / `torch.amp.autocast("cuda", enabled=False)` | `FZ: flux3/driver.py:181`, `flux3/sampling.py:119`, `flux3/model.py:64, :70`, `zimage/model.py:41, :77` | HIP tensors report device type `cuda`; fine (every TagScribeR family does this) | none |
| DiT attention head widths | Flux 3 3072/24 = 128 (`FZ: flux3/model.py:23-24`); MOSS 3072/24 = 128 (`moss_image/model.py:17`); Z-Image 3840/30 = 128 (`zimage/model.py:208-209`); ACE 128 (`acestep/model.py:34`) | All within the fast kernels' range | none |
| Complex-valued RoPE (`torch.view_as_complex`, `torch.polar`, complex64 buffers) | `FZ: zimage/model.py:76-79, :192-204, :255-263` | Complex tensors work on ROCm in general; **not verified on the owner's build** (TagScribeR's other families use real-valued rotations) | A unit test on CPU proves the maths; the first real run proves ROCm. Could be rewritten real-valued if it fails |
| Block swap through `ModelOffloader` | `FZ: flux3/model.py:195-234`, `moss_image/model.py:166-205`, `zimage/model.py:266-299` (ACE: none) | TagScribeR's `training/modules/offloading.py` carries the CUDA guard (`CLAUDE.md`) | Point the vendored models at TagScribeR's offloader |
| INT8 / NF4 bases | descriptions | bitsandbytes NF4 and TagScribeR's INT8 have run for real on Krea 2 (`PORT_PLAN.md`, "Real hardware results") | none |
| ComfyUI int8 ConvRot files (MOSS SFT preview file, the int8 text encoder) decoded at load | `FZ: moss_image/model.py:293-301`, `embedder.py:276-284` via `fizgig.minimax.convrot` | Pure torch; TagScribeR has the decode half (`TS: training/families/minimax_h3/weights.py:101-113`) | Import from `minimax_h3/weights.py` (or move it to `training/modules/`) |
| Qwen3.5-9B Gated DeltaNet (MOSS text encoder) | `FZ: moss_image/embedder.py:77-115` is transformers' pure-torch chunked rule | Exactly the path TagScribeR's AMD note describes for Qwen3.5 VLMs ("no ROCm kernels, the slower PyTorch reference path, correct"); and "don't install flash-linear-attention on ROCm: garbage" (`docs/ARCHITECTURE.md`, Known AMD notes). Caching is one pass per caption, so the slow path is acceptable | none; note it in Help |
| ffmpeg subprocess (ACE-Step) | `FZ: acestep/driver.py:87-106` | Platform-neutral binary | imageio-ffmpeg |

Everything else in the four packages is plain PyTorch. No family uses flash-attn, xformers or CUDA extensions.

### 1.5 Provenance for `THIRD_PARTY_NOTICES.md`

Fizgig's own `docs/THIRD_PARTY_NOTICES.md` (read in full) has sections for FLUX 3 Image Klein (Apache-2.0,
`src/fizgig/flux3/`) and Z-Image (Apache-2.0, `src/fizgig/zimage/model.py`, `pipeline.py`) but **none for
MOSS-Image or ACE-Step**; their provenance is only in file headers: `FZ: moss_image/model.py:1-3` ("Ported from
OpenMOSS-Team/MOSS-Image-Base pipeline_moss_image.py ... Copyright 2026 OpenMOSS ... Apache License, Version 2.0
(LICENSE beside this file)", and `moss_image/LICENSE` is the Apache text), `FZ: moss_image/embedder.py:1-5`
(transformers' `modeling_qwen3_5.py`, Apache-2.0, Qwen Team and HuggingFace), `FZ: acestep/model.py:1-3` ("adapted
from the official ACE-Step 1.5 repository ... MIT / Apache-2.0" - the slash is upstream's ambiguity; the description
note says the repository is MIT, `FZ: families/acestep.py:211`), `FZ: acestep/vae.py` (diffusers' `AutoencoderOobleck`
used as a library). MOSS's sampler imports ComfyUI-derived code: `_res_multistep_coeffs` from
`fizgig/minimax/sampling.py` (`FZ: moss_image/sampling.py:94`), which Fizgig's notices mark GPL-3.0 (ported from
`comfy/k_diffusion`). TagScribeR is GPL-3.0 and already carries that function (`TS: training/families/minimax_h3/sampling.py:60`),
so this is a provenance line, not a problem. The Flux 3 package's `LICENSE` (`FZ: flux3/LICENSE`) is Apache-2.0 with
a different copyright line from Fizgig's root `LICENSE` (the two files differ; both Apache-2.0 text).

TagScribeR entries to write: "FLUX 3 Image Klein (Black Forest Labs), Apache-2.0" (DiT, autoencoder, text-encoder
wrapper, sampler); "MOSS-Image (OpenMOSS), Apache-2.0" and "Qwen3.5 text model (transformers), Apache-2.0";
"Z-Image (Alibaba Z-Image Team), Apache-2.0"; for ACE-Step, "ACE-Step 1.5 (ace-step), MIT per the upstream
repository" after checking the upstream `LICENSE` file (not done here). Every ported file keeps its upstream header and
gets TagScribeR's line, as the standing rule asks. Weights are never included; each section states the weights'
licence as upstream's docs do (FLUX Kommunity for Flux 3; Apache-2.0 for MOSS and Z-Image; MIT for ACE-Step - all
per upstream, unverified).

---

## 2. Flux 3 Image Klein

### 2.1 The model

Black Forest Labs' 5.6B DiT for text-to-image and editing with up to 10 reference images, released 15 Oct 2026
(`FZ: families/flux3.py:1-6`). Structure: five "per-mode" blocks each for the image being made, the reference images
and the text, run before the streams meet, then 28 shared single-stream blocks (`FZ: flux3/model.py:21-28, :182-186`;
`families/flux3.py:65-68`). Hidden 3072, 24 heads, MLP ratio 3, no biases, QK RMSNorm with RoPE over four axes
(`flux3/model.py:23-29, :146-167`). Text conditioning is Qwen3-VL-4B's hidden states at layers 4, 8, ..., 32 stacked
to 20480 per token, right-padded to a multiple of 80 tokens, and the DiT attends every token including the padding
(`FZ: flux3/embedder.py:1-11, :20-22, :73-74`). Latents: the FLUX.2 autoencoder, BatchNorm-normalised, 2x2-packed to
128 channels at /16 (`FZ: flux3/driver.py:4`; `autoencoder.py:14-19, :164-171`). Two checkpoints: Base (undistilled,
trained on) and distilled (a "flow map": each step jumps to a target time, no CFG; recognised by the
`ref_time_in.in_layer.weight` key, `FZ: flux3/model.py:30, :311-317`).

Fizgig added it in v8.1.0 (`FZ: docs/RELEASE_NOTES_v8.1.0.md`), multi-reference edits in v8.1.1-8.1.2, and calls it
experimental: "the presets start from Krea 2's proven settings and haven't yet been tuned on Flux 3 itself"
(`FZ: docs/FLUX3.md:7`).

### 2.2 Model files (`FZ: families/flux3.py:43-56`; per upstream, all gated)

| pref key | role | repo / path | GB | note |
|---|---|---|---|---|
| `flux3_dit` | dit, required | `black-forest-labs/flux-3-image-klein-comfy` `diffusion_models/flux3_image_klein_bf16.safetensors` | 11.08 | "Files with _sd_ in the name are the distilled model" |
| `flux3_vae` | vae, required | same repo `vae/flux2-vae.safetensors` | 0.34 | the FLUX.2 autoencoder, as Klein 9B's |
| `flux3_text_encoder` | text_encoder, required | same repo `text_encoders/qwen3vl_4b_bf16.safetensors` | 8.88 | caching only; the same file Krea 2 uses (`FZ: flux3/embedder.py:9-11`) |
| `flux3_distilled_dit` | preview_dit, optional | same repo `diffusion_models/flux3_image_klein_sd_bf16.safetensors` | 11.10 | 16 steps, no CFG |

Download needs an accepted licence and a read token; Fizgig's Preferences has a token box used for that download only
and never saved, honouring `HF_TOKEN` (`FZ: docs/FLUX3.md:11-12`). Quantised Comfy files (int8 ConvRot, NVFP4, FP8)
are refused by the loader: "use a bf16 file (Fizgig quantises to INT8 / NF4 itself)" (`FZ: flux3/model.py:320-324, :346-348`).
Helper files: none declared (`helper_files` absent); the tokenizer comes through Krea 2's resolver (`FZ: flux3/embedder.py:48-51`).

### 2.3 Description fields (`FZ: families/flux3.py`)

| Field | Value | Line |
|---|---|---|
| key / arch_id / display / suffix / aliases | `flux3` / `flux3klein` / "Flux 3 Image Klein" / `f3k` / `("flux-3-image-klein", "flux3-klein")` | `:36-41` |
| latents | 128 channels, `spatial_factor=16`, `bucket_step=16`, native 1.0 MP | `:60-63` |
| blocks | `n_blocks=28`, `block_prefix="single_blocks"`; the driver's block map adds 3 x 5 per-mode blocks | `:65-68` |
| LoRA | kohya: `lora_unet_single_blocks_{block}_{module}.{ab}.weight`, `lora_down` / `lora_up`, modules `q_proj, k_proj, v_proj, attn_out, mlp_in, mlp_out`, `alpha_key "{prefix}.alpha"`, `lokr_kohya_stems=True`, `file_prefix="lora_unet_"`; source "ComfyUI comfy/lora.py model_lora_keys_unet, Flux3 branch (15 Oct 2026)" | `:12, :70-81` |
| driver / modelspec / implementation | `fizgig.flux3.driver:Flux3Driver` / "Flux.3-image-klein" / the BFL reference repo | `:83-85` |
| precisions | `("bf16", "int8", "nf4")`; no `auto_precisions` (Auto may pick any, most precise first) | `:86` |
| train_memory | bf16 `((0.25, 12.3), (1.0, 12.7))`, 0.245 GB per swapped block; int8 `((0.25, 8.7), (1.0, 9.1))`, 0.12; nf4 `((0.25, 5.8), (1.0, 6.2))`, 0.0. "training peaks, LoRA rank 16, adamw (15 Oct 2026, RTX 5090). 1.0 MP measured for each precision; 0.25 MP from the measured swap run ... minus the 0.4 GB bf16 grew" | `:87-92` |
| optimizers / network types / EMA | `("adamw", "adamw8bit")` / `("lora", "lokr")` / `"0.98"` | `:93-95` |
| compile | `compiles=True`, boundary "outside", payback bf16 600 / int8 600; "28 single blocks compiled: 0.31 -> 0.26 s/step (1.19x) at 0.25 MP bf16" | `:96-99` |
| fine-tune | `finetune=True`, `ft_learning_rate=5e-5` | `:101-102` |
| sliders | `slider_training=True` | `:103` |
| edit | `edit_training=True`, `edit_references=10`, `reference_strength=True`, `edit_note` "Pairs of an original and its edited version, the same crop and shape - about 40 pairs is a good start (20 at least). Each photo 1 MP or larger" | `:104-108` |
| preview checkpoint | `train_preview_checkpoint=True`, tick "Use the distilled model for samples (16 steps, no CFG - quick)", steps note "Base samples only - the distilled samples use their own 16 steps" | `:109-111` |
| sampling | "Reference Base": 50 steps, CFG 3.5, euler, simple, `alpha 6.93`, negative prompt on; "Undistilled: Euler, shifted schedule alpha 6.93, CFG against the empty prompt" | `:114-119` |
| preview checkpoint sampling | "Reference distilled": 16 steps, CFG 1.0, euler, alpha 6.93, "Flow map: each step jumps to the next time, no CFG" | `:120-123` |
| previews | 50 steps, CFG 3.5, negative `""` (the empty prompt is the reference's unconditional), 1024 x 1024 | `:124-128` |
| workbench / activation cache / int8 attention | workbench-only | `:100, :142-143` |
| notes | padded text kept in the cache; "Training timestep distribution is not published; the driver samples logit-normal shifted by the sampler's alpha 6.93 until measured otherwise" (assumption) | `:145-150` |

**Presets, verbatim** (`FZ: families/flux3.py:129-141`; `_preset` `:15-32`). Every preset carries `NETWORK_ALPHA = rank`,
`NETWORK_TYPE "LoRA (standard)"`, `SAVE_EVERY_N_EPOCHS 1`, `SEED 42`, `ADAPTIVE_LR_MIN / MAX` ("1e-4" / "4e-4" unless
adaptive is on), `OPTIMIZER_TYPE "adamw8bit"`, `GRADIENT_ACCUMULATION 1`, `MAX_GRAD_NORM 1.0`,
`BLOCKS_SWAP "Auto (detect from GPU)"`, `FAMILY_PRECISION "Auto (fits your free VRAM)"`, `FAMILY_EMA "0.98 (recommended)"`,
`KREA2_LOSS_WATCH True`, `KREA2_PER_IMAGE_LR True`, `KREA2_AUTO_RECAPTION False`, `KREA2_WARMUP_LOOK False`,
`FAMILY_FAST_ID False`, `FAMILY_SLIDER_GUIDANCE "3"`, and `FAMILY_FT / FAMILY_EDIT / FAMILY_SLIDER` as below.
Upstream's own words: "Krea 2's preset values ... Not yet measured on Flux 3" (`:16-19`).

| Preset | rank | LR | Adaptive LR | epochs | MP | flags |
|---|---|---|---|---|---|---|
| `✨ Flux 3 Ultra Fast (rank 16, adaptive LR)` (first = default) | 16 | 1e-4 (ignored: adaptive) | True, 1e-4 to 2e-4 | 50 | 0.25 | - |
| `✨ Flux 3 Standard (rank 32, full model)` | 32 | 1e-4 | False | 64 | 0.25 | - |
| `✨ Flux 3 Style (rank 16, gentle LR)` | 16 | 1e-4 (ignored) | True, 5e-5 to 2e-4 | 64 | 0.25 | - |
| `✨ Flux 3 Edit (rank 16, adaptive LR)` | 16 | 1e-4 (ignored) | True, 1e-4 to 2e-4 | 12 | 0.25 | `FAMILY_EDIT True` |
| `✨ Flux 3 Edit Strong (rank 32) - trickier edits` | 32 | 1e-4 | False | 12 | 0.25 | `FAMILY_EDIT True` |
| `✨ Flux 3 Slider (rank 8, 2e-4)` | 8 | 2e-4 | False | 20 | 0.5 | `FAMILY_SLIDER True` |
| `✨ Flux 3 Fine-tune (5e-5)` | 16 | 5e-5 | False | 30 | 0.25 | `FAMILY_FT True`, `FAMILY_FT_ROTATIONS "10"` |

The Slider and Fine-tune presets are what the stage-1 guard refuses until stages 5 (fine-tune) land; sliders are now
in TagScribeR (commit `32d9d6e`), so the Slider preset can import.

### 2.4 The driver (`FZ: flux3/driver.py`)

| Hook | Lines | What it does |
|---|---|---|
| `load_dit` | `:31-33` | `load_flux3_dit` (meta-device build, strict keys, fused `qkv` split into thirds, `FZ: flux3/model.py:327-357`) |
| `load_vae` | `:35-37` | `load_flux3_vae` fp32 |
| `load_text_encoder` | `:39-44` | `Flux3TextEncoder`, INT8 language Linears when free VRAM < 11 GB (`embedder.py:43-46`, via Qwen Image 2.1's `_int8_weights`) |
| `unload_text_encoder`, `enable_gradient_checkpointing` | `:46-50` | |
| `compile_targets` | `:52-54` | the 28 single blocks |
| `ft_spec` | `:57-68` | stage 5: per-mode blocks then single blocks, NF4 trunk resident, "12.6 GB at 0.25 MP, 14.0 at 1.0 (RTX 5090)" |
| `max_blocks_to_swap`, `enable_block_swap`, `block_swap_mode` | `:71-82` | single blocks only, at most 26 |
| `park_for_preview`, `load_preview_checkpoint`, `unpark_after_preview` | `:85-136` | Klein's Distilled handoff adapted from Qwen 2.1's: stream all but two single blocks (or park an NF4 base on CPU under 8 GB free); the distilled sized to the VRAM left: bf16, else INT8, else INT8 with blocks streamed forward-only |
| `encode_images` | `:140-148` | per image: `vae.encode` -> tokens -> `(128, h, w)` fp32 |
| `encode_text` | `:151-152` | `{"tokens": (L, 20480) bf16}` |
| `supports_references`, `load_reference_text_encoder`, `encode_text_with_references` | `:155-162` | the text encoder never sees references; they reach the DiT as latents (the `image_cond` stream) |
| `_sample_t` | `:166-170` | `t = sigmoid(randn)`, then `shift(t) = 6.93 t / (1 + 5.93 t)`, scaled into `[min_t, max_t]`. An assumption, flagged by upstream |
| `_forward` | `:172-184` | packs `[text, references (l = -(1+i), time 0), image]`, `timesteps["image"] = t`, bf16 autocast; refuses the distilled checkpoint for training |
| `training_loss` | `:186-206` | `x_t = (1 - t) x0 + t noise`, target `noise - x0`, `F.mse_loss`; image-pair sliders weight each token by `|x0 - diff_ref|` (Krea 2's formula, clamp 8, normalised) |
| `noise_latents`, `predict` | `:208-217` | prompt sliders |
| `initial_noise` | `:221-222` | `(h*w, 128)` bf16-rounded fp32 on CPU from the seed (`sampling.py:89-93`) |
| `generate` | `:225-239` | `S.sample`: Euler on the alpha-shifted schedule; Base with CFG on a two-sample pack `[uncond, cond]`; the distilled as a flow map; references; returns `(1, 128, h, w)` |
| `pad_conditioning` | `:241-254` | prompt travel (workbench) |
| `decode` | `:257-263` | tokens -> `vae.decode` -> PIL |
| `block_map` | `:266-280` | four groups: "Image blocks" `image_0..4`, "Reference blocks" `ref_0..4`, "Text blocks" `text_0..4`, "Single blocks" `single_0..27`; each the six Linears. So a LoRA covers 43 blocks x 6 = 258 Linears |

Objective check against the sampler: `sample` integrates `x += dt * v` from t = 1 to 0 with `v = dit(...)`
(`FZ: flux3/sampling.py:120-134`), so the velocity the model predicts is `d x / d t = noise - x0`: the training target
matches (`driver.py:4-6`).

### 2.5 What TagScribeR has, and what it needs

Has: the FLUX.2 autoencoder with the ROCm fix (1.3); the Qwen3-VL-4B loader (`TS: training/families/krea2/embedder.py:75-98
build_language_model`, `:66-72 tokenizer_source`; `_int8_weights` at `TS: qwen_image21/embedder.py:98`); the preview
checkpoint handoff (`TS: training/train.py:409-495`); kohya LoRA writing with `lokr_kohya_stems` (`TS: training/lora.py:276-282`);
sliders; INT8 / NF4 / block swap / compile; Edit LoRAs with one reference (Klein, stage 2).

Needs (TagScribeR-side):

1. **Multi-reference edit pairs** (`edit_references`): `TS: training/cache.py:276-280` refuses more than one partner;
   `TS: training/dataset.py:261-277` already collects `stem_<anything>.<ext>` partners, so the data side is there.
   Upstream's rule (`FZ: families/cache.py:164-171`): a slider pairs one with one, an edit takes up to the family's
   count, `photo.png` first then `photo_1.png`, `photo_2.png` in order. The loop already passes `latents_control_*`
   as `refs` (`TS: training/train.py:1088-1091`) and caches several controls (`TS: cache.py:203-204`). Mostly a limit
   to lift plus the ordering rule, and Klein (4) and Qwen 2.1 (10) gain the same.
2. **Per-family preview-checkpoint tick** (1.3).
3. **Gated download**: `ModelFile.gated` and a Help / Settings line pointing at the licence page; TagScribeR's
   Download links (`TS: tabs/train.py:440, :729`) cannot fetch a gated file without a token. Recommendation in 7.
4. **The vendored model must take a config** for tiny-model tests: `FZ: flux3/model.py:22-29` hard-codes HIDDEN,
   HEADS, DEPTH, DEPTH_SINGLE, AXES_DIM, IN_CHANNELS (Klein's port has `Flux2Params`, `TS: training/families/klein/model.py:39-53`;
   its tests build a 64-wide 2+3-block DiT, `TS: tests/test_training_klein.py:22-23`). A `Flux3Params` dataclass with
   the reference values as defaults, recorded in the header as a TagScribeR change.
5. **Attention through `training.modules.sdpa`** instead of the forced cuDNN backend (1.4).
6. The text encoder: `Flux3TextEncoder` drops the vision tower and LM head and fixes the RoPE `inv_freq` to fp32
   ("In bf16 the rotary angles drift with position: measured 1.25% off the reference at 160 tokens",
   `FZ: flux3/embedder.py:37-42`). TagScribeR's Krea 2 builder keeps the vision tower (unused) and loads fp8 files;
   check whether its `model.to(dtype)` also casts `inv_freq` (Krea 2's own encoder would then have the same drift -
   worth a look, not in scope here). Position ids are the mask's cumsum given as four rows (Qwen3-VL's mRoPE,
   `:68-71`); TagScribeR's `Krea2TextEncoder.encode` does not pass `position_ids` (`TS: krea2/embedder.py:151`), so
   the Flux 3 encoder needs its own `encode_one`, as upstream has.
7. Cache size: a caption is `L x 20480` bf16, 3.3 MB per 80 tokens (`FZ: flux3/embedder.py:65, :74`); fine.

### 2.6 Port plan

Files to create: `training/families/flux3/{__init__,description,driver,model,autoencoder_or_shared,sampling,embedder}.py`
(upstream headers kept, TagScribeR change lines added), plus the shared FLUX.2 loader of 1.3. Registry line, Help Center
text, hotkeys none, docs (`docs/` family page as the others). Tests (CPU, tiny random):

- description validity and registry, the seven presets verbatim and their import through `training/presets.py`
  (Fine-tune refused until stage 5) - the pattern of `TS: tests/test_training_klein.py:48-95`;
- `Flux3Params` tiny DiT: forward shape, gradient checkpointing, block swap on CPU does not change the loss
  (`test_block_swap_does_not_change_the_loss` pattern), the flow-map refusal for training, `is_flow_map` on a tiny
  saved file, `convert_state_dict`'s qkv split and the strict-key refusal;
- `_sample_t` against a transcription of `shift()`; loss = MSE to `noise - x0` at the drawn t; the slider diff
  weighting sums to its mean;
- references: `_forward` with two tiny references gives ids `l = -1, -2` and a different output from none;
- LoRA keys: `lora_unet_single_blocks_3_q_proj.lora_down.weight` and the per-mode names
  (`lora_unet_content_mode_blocks_image_0_q_proj...`, `lora_unet_txt_mode_blocks_0_...`) round-trip through
  `FamilyLoRA`; LoKR under the same stems;
- sampler: `schedule(steps, alpha)` ends at 0, CFG two-sample pack returns the conditional half, the flow map calls
  the DiT with `timesteps_ref`;
- text encoder layout on a tiny Qwen3-VL (as `test_text_encoder_layout_on_a_tiny_qwen3` does for Klein): 8 layers
  stacked, padding to 80, four-row position ids;
- the preview-checkpoint handoff with a tiny flow-map file through `_CheckpointPreviews`;
- the autoencoder: pack / unpack round trip, `decoder_ch` 96 build, diffusers-name conversion on a tiny dict;
- the smoke run (`TS: tests/test_training_smoke.py` pattern) with a Flux 3-shaped tiny family: cache, train, preview,
  pause / resume.

What only a real run confirms: the ROCm attention path, the 12.7 GB bf16 / 9.1 INT8 / 6.2 NF4 peaks, the distilled
previews, an edit LoRA in ComfyUI. The owner does not train Klein; Flux 3 may be the same case (experimental until a
user reports).

Size: **M-L** (about 1,100 upstream lines, mostly vendored; the TagScribeR-side items 1-5 are the work).

---

## 3. MOSS-Image

### 3.1 The model

OpenMOSS's 5B single-stream text-to-image DiT, released 16 Oct 2026 under Apache-2.0, strong at text rendering and JSON
layout prompts (`FZ: families/moss_image.py:1-7`; `docs/MOSS_IMAGE.md:5`). 42 joint blocks at width 3072 (24 x 128)
with one shared modulation, after a two-block text adapter over Qwen3.5-9B's final hidden states; FLUX.2 latents
2x2-packed to 128 channels (`FZ: moss_image/model.py:8-9, :17, :140-157`). Text: Qwen3.5-9B through the chat template
with an open `<think>` block, unpadded, up to 4096 tokens, the final hidden state after the norm
(`FZ: moss_image/embedder.py:6-13`). No edit or reference-image conditioning: "the released checkpoints are
text-to-image only" (`FZ: families/moss_image.py:156-158, :269-271`). Two checkpoints: Base (trained on; "the model
card: Base is for fine-tuning and LoRAs") and SFT (OpenMOSS's everyday checkpoint; previews render on it, never
trained; `:52-60, :88-99`). Fizgig added it in v8.5.0 through a community pull request (`FZ: docs/RELEASE_NOTES_v8.5.0.md`),
and its description still carries `status_note` "new and improving daily" (`:48`).

### 3.2 Model files (`FZ: families/moss_image.py:51-100`; per upstream, none gated)

| pref key | role | repo / path | GB | note |
|---|---|---|---|---|
| `moss_image_dit` | dit, required | `Comfy-Org/MOSS-Image` `diffusion_models/moss_image_base_bf16.safetensors` | 10.16 | also loads the official diffusers shards (`OpenMOSS-Team/MOSS-Image-Base transformer/`) and ComfyUI's int8 ConvRot file (`:57-60`; loader `moss_image/model.py:277-310`) |
| `moss_image_vae` | vae, required | same repo `vae/flux2-vae_small.safetensors` | 0.25 | FLUX.2's encoder with BFL's small decoder ("avoids spots"); `replaces=("flux2-vae.safetensors", "ae.safetensors")` - the full-decoder files load too, same latents (`:61-72`) |
| `moss_image_turbo_lora` | speed_lora, optional, `bulk_fetch=False` | `lodestones/MOSS-Image-SFT-8-steps-dmd` `lora.safetensors` | 1.19 | "work in progress; soft on Base" (`:73-78`) |
| `moss_image_text_encoder` | text_encoder, required | `Comfy-Org/Qwen3.5` `text_encoders/qwen3.5_9b_bf16.safetensors` | 19.31 | "Loads text-only (about 15 GB), or 8-bit (about 9 GB)"; alt: `Comfy-Org/MOSS-Image text_encoders/qwen3.5_9b_int8_convrot.safetensors` ~9.7 GB (`:79-87`) |
| `moss_image_sft_dit` | preview_dit, **required=True** | `Comfy-Org/MOSS-Image` `diffusion_models/moss_image_sft_int8_convrot.safetensors` | 5.41 | decoded at load; alt bf16 10.2 GB (`:91-99`) |

Helper files: `OpenMOSS-Team/MOSS-Image-Base tokenizer/*` (tokenizer + chat template, `:103`). Note the SFT preview
file is the only `preview_dit` in Fizgig marked required (Klein's and Flux 3's are optional); see 7.

### 3.3 Description fields (`FZ: families/moss_image.py`)

| Field | Value | Line |
|---|---|---|
| key / arch_id / display / suffix / aliases | `moss_image` / `mossimage` / "MOSS-Image" / `moss` / `("moss-image", "mossimage")` | `:43-49` |
| latents | 128, /16, bucket 16, native 1.0 MP ("USERGUIDE.md: 1, 1.5 or 2 MP recommended, 0.25 to 4 MP supported") | `:105-108` |
| blocks | `n_blocks=42`, `block_prefix="layers"` | `:110-113` |
| LoRA | kohya `lora_unet_layers_{block}_{module}`, `lora_down` / `lora_up`, modules `proj.q, proj.k, proj.v, proj.o, proj.ffn_up_gate, proj.ffn_down`; "ComfyUI's MOSSImage branch maps lora_unet_layers_N_proj_q / _k / _v onto thirds of the fused proj.qkv"; LoKR saves as `diffusion_model.layers.N.proj.q` (no `lokr_kohya_stems`) | `:18, :115-127` |
| driver / modelspec / implementation | `fizgig.moss_image.driver:MOSSImageDriver` / "MOSS-Image" / the OpenMOSS repo | `:129-131` |
| precisions | `("int8", "nf4", "bf16")`, `auto_precisions=("int8", "nf4")` ("AI Toolkit's MOSS-Image default is an 8-bit base; bf16 stays a manual choice") | `:132-133` |
| train_memory | bf16 `((0.25, 15.1), (1.0, 15.1))`, 0.211 per block; int8 `((0.25, 10.5), (1.0, 10.5))`, 0.105; nf4 `((0.25, 8.5), (1.0, 8.5))`, 0.0. "MEASURED 17 Oct 2026 on a 5090 ... The preview round (1024², 50 steps, CFG 3) sets the peak at every precision"; training alone bf16 9.2 GB at 0.25 MP with 10 blocks swapped, 13.4 at 1 MP; INT8 8.8 and NF4 7.3 at 1 MP | `:134-141` |
| optimizers / network types / EMA | `("adamw", "adamw8bit")` / `("lora", "lokr")` / `"0.98"` | `:142-144` |
| compile | outside; payback bf16 200 / int8 200 / nf4 400; "1.45 -> 1.07 s/step compiled (-26%)" | `:145-150` |
| fine-tune | `finetune=True`, `ft_learning_rate=1e-5` | `:153-154` |
| sliders / edit | `slider_training=True`; edit off (text-to-image only) | `:155-158` |
| sampling (4 recipes, first = default) | (1) "res_multistep, 30 steps, CFG 4": `shift 5.0`, `uncond_skip (22,)`, `sampler res_multistep`, `schedule simple`, `slg_end 0.5`, negative on - "As ComfyUI renders it ... SLG switch on (block 22 skipped in the negative branch for the first half)"; (2) "Euler, 30 steps" CFG 3 (AI Toolkit's); (3) "Reference (diffusers pipeline)" 50 steps CFG 3; (4) "ComfyUI workflow" 50 steps CFG 3 sgm_uniform | `:163-188` |
| speed LoRA | "MOSS-Image Turbo LoRA", pairs with SFT, strength 1.0, 8 steps CFG 1 shift 5 `uncond_skip ()`, `load_unmerged=True` ("an fp32 rank-128 adapter: its delta is below bf16's rounding step"), community 12-16 steps / 0.8 at 16; `preview_speed_lora` set, `preview_speed_strength=0.0` (off until raised) | `:189-210` |
| previews | 30 steps, CFG 4.0 ("4 won the side-by-side"; guide says 3-6), negative "worst quality, bad anatomy, watermark, jpeg artifact", 1024 x 1024 | `:20, :211-215, :232-233` |
| preview checkpoint | `train_preview_checkpoint=True`, tick "Use the SFT model for samples (set it in Preferences - a Base LoRA carries over only partly)", **on by default**, steps note "Base samples only - the SFT samples use their own 30 steps, CFG 4", sampling = recipe (1) on the SFT | `:220-231` |
| options | `MOSS_LORA_SFT` (check, `lora_sft=1`: "the LoRA then trains attention only (q, k, v, o). Training its feed-forward layers breaks anatomy", mode lora); `MOSS_LOWNOISE_PCT` (entry, default "30", `lownoise_pct={}`, suffix "% of steps", "Experimental figure", featured) | `:246-257` |
| notes | the timestep law (below); text conditioning; no edit; and a stale note "No speed LoRA or distilled checkpoint yet" that the Turbo row above contradicts | `:260-274` |

**Presets, verbatim** (`FZ: families/moss_image.py:234-245`; `_preset` `:23-39`): every preset carries
`NETWORK_ALPHA = rank`, `NETWORK_TYPE "LoRA (standard)"`, `SAVE_EVERY_N_EPOCHS 1`, `SEED 42`, `ADAPTIVE_LR False`,
`ADAPTIVE_LR_MIN "1e-4"`, `ADAPTIVE_LR_MAX "4e-4"`, `OPTIMIZER_TYPE "adamw"`, `GRADIENT_ACCUMULATION 1`,
`MAX_GRAD_NORM 1.0`, `BLOCKS_SWAP "Auto (detect from GPU)"`, `FAMILY_PRECISION "Auto (fits your free VRAM)"`,
`KREA2_LOSS_WATCH False`, `KREA2_PER_IMAGE_LR False`, `KREA2_AUTO_RECAPTION False`, `KREA2_WARMUP_LOOK False`,
`FAMILY_EDIT False`, `FAMILY_FAST_ID False`, `FAMILY_SLIDER_GUIDANCE "3"`. "a fixed learning rate: Adaptive LR walks
to its floor on MOSS-Image's flat loss" (`:238-239`); "70 epochs: a 53-image run was still improving at 1,590 steps"
(`:236-237`).

| Preset | rank | LR | epochs | MP | EMA | flags |
|---|---|---|---|---|---|---|
| `✨ MOSS-Image Standard (rank 32, 1e-4)` (default) | 32 | 1e-4 | 70 | 1.0 | 0.98 | - |
| `✨ MOSS-Image Fast (rank 16, 0.25 MP)` | 16 | 1e-4 | 70 | 0.25 | 0.98 | - |
| `✨ MOSS-Image Style (rank 16, 5e-5)` | 16 | 5e-5 | 50 | 1.0 | **Off** | - |
| `✨ MOSS-Image Slider (rank 8, 2e-4)` | 8 | 2e-4 | 20 | 1.0 | 0.98 | `FAMILY_SLIDER True` |
| `✨ MOSS-Image Fine-tune (1e-5)` | 16 | 1e-5 | 30 | 1.0 | 0.98 | `FAMILY_FT True`, `FAMILY_FT_ROTATIONS "10"` |

These are measured (Peter's runs, 18 Oct 2026, `:235-239`), unlike Flux 3's.

### 3.4 The driver (`FZ: moss_image/driver.py`)

| Hook | Lines | What it does |
|---|---|---|
| `load_dit` | `:64-66` | `load_moss_dit`: strict keys; any of ComfyUI's fused file, diffusers shards, int8 ConvRot (`model.py:277-322`) |
| `park_for_preview`, `load_preview_checkpoint`, `unpark_after_preview` | `:69-118` | the SFT previews: stream all but two blocks (NF4 base left as is); the SFT sized to the VRAM left (bf16 / INT8 / INT8 streamed); a `convrot` file counts double on disk-to-bf16 |
| block swap | `:120-131` | up to 40 blocks |
| `load_vae` | `:133-136` | Flux 3's autoencoder loader |
| `load_text_encoder` | `:138-147` | `MOSSTextEncoder`: bf16, resident when >= 18 GB free, else **streamed layer-major from the CPU** ("still exact bf16, which OpenMOSS ask for ... not 8-bit"); tokenizer via `from_pretrained(AutoTokenizer, repo, subfolder="tokenizer")` |
| `text_encoder_gb`, `drop_text_encoder` | `:149-156` | workbench RAM cache; skip |
| `cache_stage` | `:158-163` | text stage: batch size 4096 so a streamed encoder moves each layer once per group; returns False (the shared path runs) |
| `compile_targets` | `:171-172` | the 42 blocks |
| `ft_spec` | `:175-184` | stage 5; `file_layout` writes q / k / v back fused |
| `encode_images` | `:188-196` | `(128, h, w)` fp32 |
| `encode_text` | `:199-200` | `{"hidden_states": (L, 4096) bf16}` |
| `_sample_t` | `:204-218` | with p 0.01 uniform on the 1000-step grid, else `S = 1 - sigmoid(Z)`, `Z ~ N(mu, 0.95²)`, truncated to `S <= 0.95` by inverse CDF, rescaled by 1/0.95, taken to the grid; then into `[min_t, max_t]` |
| `_t_mu` / `mu_for_clean_share` | `:41-58, :220-230` | `lownoise_pct` = the share of steps below sigma 0.5 moves `mu` (bisection on the truncated CDF); reference mu -1.09 ~ 11%, default 30% |
| `_forward` | `:232-236` | `dit(xt, 1000 * t, hidden)` |
| `training_loss` | `:238-260` | `x_t = (1 - t) x0 + t noise`, target `noise - x0`, `F.mse_loss` ("report eq. 3: per-token mean over channels, no sigma weighting"); slider diff weighting; refuses refs |
| `noise_latents`, `predict` | `:262-269` | |
| `initial_noise` | `:273-274` | `[1, 128, h, w]` fp32 CPU generator |
| `generate` | `:289-305` | options over the description's default sampler: shift, uncond_skip, sampler (euler / res_multistep), schedule (reference / simple), slg_end; CFG needs an encoded negative |
| `pad_conditioning` | `:307-318` | workbench |
| `decode` | `:321-329` | |
| `block_map` | `:332-344` | "Blocks" `block_0..41` (six Linears) and "Text adapter" `text_0..1` (`qkv, proj, ffn_gate, ffn_up, ffn_down`) |
| `lora_target_names` | `:346-356` | **the 42 blocks only, block 22 included** ("leaving it out, as AI Toolkit does, cost likeness"); the text adapter stays frozen; `lora_sft=1` keeps attention only |
| `quant_target_names` | `:358-362` | every block Linear whatever the LoRA trains (block swap pairs blocks dtype for dtype, #201) |

### 3.5 Model, embedder, sampler

- `FZ: moss_image/model.py`: ported from the diffusers custom pipeline shipped with the weights (Apache-2.0,
  OpenMOSS); `MOSS_CONFIG` is a dict and the constructor takes every size (`:17-18, :143-144, :313-316`), so tiny
  tests need no change. Per-sample forward (the batch loop at `:234-259`) with the text mask, `skip_layers` for the
  SLG negative branch (`:247`), RoPE built on CPU fp32 (`:124-127`), shared modulation (`:153, :230`), block swap and
  checkpointing on the 42 blocks (`:166-205, :244-255`). Attention: the `attend()` INT8 hook before SDPA (`:51-55`): drop.
- `FZ: moss_image/embedder.py`: Fizgig vendors transformers' Qwen3.5 text model because it pins 4.57 (`:1-5`).
  TagScribeR runs transformers 5.x, which has `qwen3_5` (`docs/ARCHITECTURE.md`, Environment). Two routes: (a) port
  Fizgig's vendored model as is (pure torch, no KV cache, `read_text_state_dict` for the three file layouts
  `:263-287`, and the streamed layer-major `encode` for cards under 18 GB free `:331-411`); (b) build transformers'
  `Qwen3_5` text model from a config and feed it the same state dict. (a) keeps parity and the streaming path, which
  the owner's card may need (a 15 GB bf16 encoder beside the display; upstream's threshold is 18 GB free); (b) saves
  ~410 lines but has no streaming and relies on transformers' own GDN fallback (the same pure-torch path on ROCm).
  Recommendation: (a).
- `FZ: moss_image/sampling.py`: the reference's snapped shift-5 grid (`:24-41`), ComfyUI's simple schedule (`:44-51`),
  `slg_sigma_end` (`:58-64`), Euler and res_multistep (coefficients from `minimax/sampling._res_multistep_coeffs`,
  `:94, :126-131`; TagScribeR has them at `TS: training/families/minimax_h3/sampling.py:60`). Upstream ships a sampler
  test with an exact Gaussian denoiser (`.claude/fizgig-src/checks/test_moss_sampler.py`): port it.

### 3.6 What TagScribeR needs

1. The shared FLUX.2 loader with the small decoder (1.3) and the ConvRot decode import (1.4).
2. The preview checkpoint tick from the description, **on by default** for MOSS (1.3).
3. `MOSS_LORA_SFT` and `MOSS_LOWNOISE_PCT` as parameters in `training/params.py` with upstream's keys and labels,
   routed through `DRIVER_OPTIONS` to `configure()` (the H3 rule); `lora_target_names` reads them.
4. A speed LoRA that is a Settings row left empty by default, with `preview_speed_strength=0.0`: TagScribeR's
   description supports both (`TS: training/description.py:206, :215-216`), check the Samples section greys the
   Turbo strength while the row is empty.
5. Text caching with the streamed encoder: the shared text stage encodes in chunks (`TS: training/cache.py:144`);
   upstream raises the batch to 4096 so the streamed encoder moves each layer once per group (`cache_stage`). Either
   add the hook or make the chunk size a driver attribute.
6. `preview_negative` is a tag-style string - supported (`TS: description.py:204`).
7. Help: the "train on Base, use on SFT" rule, the 70-epoch patience note, the clean-end share, the 4,096-token
   captions ("describe the picture fully, including any text in it exactly as written", `FZ: docs/MOSS_IMAGE.md:56`),
   and the AMD note on Qwen3.5's slow-but-correct path.

### 3.7 Port plan

Files: `training/families/moss_image/{__init__,description,driver,model,embedder,sampling}.py` (+ the shared VAE).
Tests (CPU, tiny):

- description validity, the five presets verbatim and their import (Fine-tune refused until stage 5); `auto_precisions`;
- tiny `MOSS_CONFIG` DiT: forward, `skip_layers=(1,)` changes the output, batch loop with a text mask, block swap
  on CPU, the loader's fused-qkv split, the diffusers-shard glob, the ConvRot decode on a tiny quantised tensor;
- `_sample_t`: the 1% uniform branch, every draw on the grid and <= 0.95 before the window; `mu_for_clean_share(11)`
  near -1.09, the share at the default 30 measured by sampling; `lownoise_pct` reaches `_t_mu` through `configure()`;
- loss = MSE to `noise - x0` with the DiT told `1000 * t`; the slider weighting;
- `lora_target_names` with and without `lora_sft=1`; `quant_target_names` is every block Linear; keys
  `lora_unet_layers_22_proj_q.lora_down.weight` round-trip; LoKR as `diffusion_model.layers.N.proj.q`;
- sampler: `schedule()` is on the training grid and ends at 0; `comfy_simple_schedule` length; `slg_sigma_end(0.5)`;
  the Gaussian-denoiser test from upstream's `checks/test_moss_sampler.py` (res_multistep closer than Euler);
- text encoder: a tiny `QWEN35_9B`-shaped config (2 layers: one linear-attention, one full) encoded resident and
  streamed gives the same hidden states; the chat template ends with `<think>\n`;
- the SFT preview handoff through `_CheckpointPreviews` with a tiny file, and the speed LoRA loaded unmerged;
- the smoke run.

Only a real run confirms: the 10.5 GB INT8 peak with SFT previews, the streamed text encoder on the owner's card,
compile, the likeness behaviour Peter measured.

Size: **M** (about 1,250 upstream lines, nearly all vendored; the TagScribeR side is the shared pieces plus two
parameters).

---

## 4. Z-Image Turbo

### 4.1 The model

Tongyi-MAI's 6B single-stream DiT (S3-DiT): 30 layers at width 3840 (30 heads x 128), two image-only noise refiners
and two text-only context refiners ahead of them, 3-axis RoPE (32 / 48 / 48, theta 256), AdaLN from the timestep only
(`FZ: zimage/model.py:5-10`). Text: Qwen3-4B, chat template with thinking on, `hidden_states[-2]`, real tokens only, at
most 512 (`FZ: zimage/pipeline.py:6-8`). Latents: the FLUX.1 VAE, `(z - 0.1159) * 0.3611`, 16 channels at /8
(`:12, :18`). Turbo is distilled to 8 steps, no CFG, shift 3 (`FZ: families/zimage.py:132-136`). Fizgig trains
**on Turbo itself** with its own frozen training adapter (rank 16 on attention + MLP, "7.6k real photos + 1k of
Turbo's own renders", `:91-97`), because "a plain LoRA on it undoes the distillation (blurry, washed-out renders at 8
steps)" (`:4-5`); the adapter is off for previews and never in the saved file (`:55-57`). Added in Fizgig v7.1.0
(`FZ: docs/RELEASE_NOTES_v7.1.0.md`: "We're open to adding training on Z-Image Base, but in our tests Turbo now
responds very well"). Marked experimental (`:41`).

### 4.2 Model files (`FZ: families/zimage.py:43-58`; per upstream, no `gated` flag)

| pref key | role | repo / path | GB |
|---|---|---|---|
| `zimage_dit` | dit, required | `Comfy-Org/z_image_turbo` `split_files/diffusion_models/z_image_turbo_bf16.safetensors` | 12.31 |
| `zimage_vae` | vae, required | same repo `split_files/vae/ae.safetensors` (FLUX.1's, "Not the FLUX.2 VAE Klein uses") | 0.34 |
| `zimage_text_encoder` | text_encoder, required | same repo `split_files/text_encoders/qwen_3_4b.safetensors` | 8.04 |
| `zimage_training_adapter` | training_adapter, optional | `ShootTheSound/Fizgig-Z-Image-Turbo-Training-Adapter` `fizgig_z_image_turbo_training_adapter.safetensors` | 0.07 |

Helper files: `Tongyi-MAI/Z-Image-Turbo` `tokenizer/*` and `vae/config.json` (`:130`) - the VAE config is needed
because the VAE loads through diffusers `from_single_file(config=repo)` (`FZ: zimage/pipeline.py:66-79`, cache-first).

### 4.3 Description fields (`FZ: families/zimage.py`)

| Field | Value | Line |
|---|---|---|
| key / arch_id / display / suffix / aliases | `zimage` / `zimageturbo` / "Z-Image Turbo" / `zit` / `("z-image", "z-image-turbo", "zimage")`; experimental | `:35-41` |
| latents | 16, /8, `bucket_step=16` ("8x VAE x 2x2 patches"), native 1.0 MP | `:63-66` |
| blocks | `n_blocks=30`, `block_prefix="layers"`; refiners not trained | `:68-71` |
| LoRA | `diffusion_model.layers.{block}.{module}.{ab}.weight`, `lora_A` / `lora_B`, modules `attention.to_q, .to_k, .to_v, .to_out.0, feed_forward.w1, .w2, .w3`, `kohya=False`, `file_prefix="diffusion_model."`; "as AI-Toolkit writes Z-Image LoRAs; ComfyUI slices to_q / to_k / to_v onto its fused attention.qkv" | `:74-85` |
| driver / modelspec / implementation / adapter | `fizgig.zimage.driver:ZImageDriver` / "Z-Image-Turbo" / Tongyi-MAI repo / `training_adapter="zimage_training_adapter"` with the note | `:87-97` |
| EMA / precisions | `"0.98"`; `("bf16", "int8", "nf4")` | `:98-99` |
| train_memory | bf16 `((0.5, 17.1), (1.0, 17.2))`, 0.36 per layer; int8 `((0.5, 12.3), (1.0, 12.3))`, 0.18; nf4 `((0.5, 10.0), (1.0, 10.1))`, 0.0 - "Measured 6 Oct 2026 (RTX PRO 6000 ...), peak GB including the preview, which sets it - training alone 13.7 / 15.0 (bf16), 8.8 / 10.0 (INT8), 7.0 / 8.1 (NF4) at 0.5 / 1 MP; s/step bf16 0.50 / 0.92" | `:100-105` |
| optimizers / network types | `("adamw", "adamw8bit")` / `("lora", "lokr")` | `:106-107` |
| sliders / fine-tune | `slider_training=True`; `finetune=True`, `ft_learning_rate=5e-5` ("weights ~3x Qwen's") | `:108-112` |
| compile | outside, `compile_fullgraph=False`, payback int8 300 / bf16 600; "INT8 1.03 -> 0.69 s/step (1.49x)"; `compile_hint` text | `:113-127` |
| sampling | "Turbo": 8 steps, CFG 1.0, euler, simple, `shift 3.0` ("the reference: 9 scheduler steps whose last is skipped") | `:132-136` |
| previews | 8 steps, CFG 1.0, `preview_negative=None` (CFG-free), `samples_cfg_free=True`, 1024 x 1024 | `:137-143` |
| notes | train with the adapter; the text recipe | `:163-168` |

**Presets, verbatim** (`FZ: families/zimage.py:145-161`; `_preset` `:15-31`): every preset carries `NETWORK_ALPHA = rank`,
`NETWORK_TYPE "LoRA (standard)"`, `SAVE_EVERY_N_EPOCHS 1`, `SEED 42`, `ADAPTIVE_LR_MIN / MAX` ("2e-4" / "4e-4"
unless adaptive), `OPTIMIZER_TYPE "adamw"`, `GRADIENT_ACCUMULATION 1`, `MAX_GRAD_NORM 1.0`, `BLOCKS_SWAP "Auto ..."`,
`FAMILY_PRECISION "Auto ..."`, **`FAMILY_TRAINING_ADAPTER True`**, `FAMILY_EMA "0.98 (recommended)"`,
`KREA2_LOSS_WATCH True`, `KREA2_PER_IMAGE_LR False`, `KREA2_AUTO_RECAPTION False`, `KREA2_WARMUP_LOOK False`,
`FAMILY_FT False`, `FAMILY_FAST_ID False`, `FAMILY_SLIDER_GUIDANCE "2"`. "Qwen Image 2.1's preset set" (`:16`).

| Preset | rank | LR | Adaptive LR | epochs | MP | flags |
|---|---|---|---|---|---|---|
| `✨ Z-Image Turbo Fast (rank 8, adaptive LR)` (default) | 8 | 1e-4 (ignored) | True, 2e-4 to 4e-4 | 30 | 0.25 | - |
| `✨ Z-Image Turbo Standard (rank 16, adaptive LR)` | 16 | 1e-4 (ignored) | True, 1e-4 to 2e-4 | 30 | 0.5 | - |
| `✨ Z-Image Turbo Style (rank 16, 1.5e-4)` | 16 | 1.5e-4 | False | 30 | 0.5 | - |
| `✨ Z-Image Turbo Slider (rank 8, 2e-4)` | 8 | 2e-4 | False | 30 | 0.5 | `FAMILY_SLIDER True` |
| `✨ Z-Image Turbo Fine-tune (5e-5)` | 16 | 5e-5 | False | 30 | 0.5 | `FAMILY_FT True`, `FAMILY_FT_ROTATIONS "10"` |
| `✨ Z-Image Turbo Fine-tune Strong (1e-4)` | 16 | 1e-4 | False | 30 | 0.5 | `FAMILY_FT True`, `FAMILY_FT_ROTATIONS "10"` |

### 4.4 The driver (`FZ: zimage/driver.py`)

| Hook | Lines | What it does |
|---|---|---|
| `load_dit` | `:45-47` | `load_zimage_dit` from ComfyUI's single file (fused `attention.qkv` split, `out` -> `to_out.0`, `q_norm` -> `norm_q`, `x_embedder` -> `all_x_embedder.2-1`, `model.py:426-463`) |
| block swap | `:49-60` | 30 layers, at most 28 |
| `load_vae` | `:62-63` | diffusers `AutoencoderKL.from_single_file(..., config=<Hub repo>, subfolder="vae")` |
| `load_text_encoder` | `:65-66` | `Qwen3Model(Qwen3Config(**QWEN3_4B))` from the ComfyUI file; tokenizer cache-first from the Hub repo |
| `encode_images` | `:79-82` | stacked batch, bf16 latents |
| `encode_text` | `:85-86` | `{"cap": (L, 2560) bf16}` real tokens only |
| `_velocity` | `:90-96` | the model runs at `t = 1 - sigma` and returns `x0 - noise`; negated into `noise - x0` |
| `_sample_sigma` | `:99-102` | `sigmoid(randn)` then `3 s / (1 + 2 s)` (the model's own shift), per item |
| `training_loss` | `:105-125` | `x_t = (1 - s) x0 + s noise`, target `noise - x0`, `F.mse_loss`; slider weighting per item |
| `noise_latents`, `predict` | `:127-136` | |
| `initial_noise` | `:140-142` | `(1, 16, H/8, W/8)` fp32 CPU |
| `generate` | `:145-163` | Euler over `P.sigmas(steps, shift)`; CFG `u + cfg (v - u)` when `cfg > 1` and a negative is given (Turbo: never) |
| `pad_conditioning` | `:165-177` | workbench prompt travel |
| `convert_lora_state_dict`, `alias_flat` | `:180-205` | ComfyUI-named LoRAs (fused qkv) read back |
| `ft_spec` | `:208-222` | stage 5 |
| `decode` | `:225-228` | |

The training adapter is applied by the shared layer (`training_adapter` pref, `FAMILY_TRAINING_ADAPTER`), exactly as
Qwen Image 2.1's is in TagScribeR; the driver has no adapter code.

### 4.5 Model code

`FZ: zimage/model.py` (adapted from Tongyi's `transformer.py`, Apache-2.0): the constructor takes every size
(`:208-210`), so a tiny config is `ZImageTransformer2DModel(dim=64, n_layers=2, n_heads=4, cap_feat_dim=24, ...)`
with `axes_dims` summing to the head dim. Items are lists of `(C, F, H, W)` latents padded to multiples of 32 tokens
(`:23, :317-353`); the forward handles a batch with masks (`:355-422`). Complex RoPE (`:76-79, :192-204`): see 1.4.
About 170 of its 463 lines are prompt-travel plumbing for the workbench (`_travel_layout`, `_key_bias`,
`_shift_frame`, the `cap_weights` path, `:91-107, :243-263, :322-336, :371-373, :386-389, :398-399, :412-413`): port
as is (harmless, keeps the file close to upstream) or leave out with a header note. `init_empty_weights` from
accelerate (`:455`).

### 4.6 What TagScribeR needs

1. The FLUX.1 VAE without a Hub call and with the ROCm fix: either diffusers `AutoencoderKL.from_config` with a
   vendored config (16 latent channels; the loader pattern of `TS: training/families/sdxl/vae.py:23-50`, with
   `set_attn_processor(AttnProcessor())`), or the vendored FLUX autoencoder class with `EMBED_DIM=16` and no
   BatchNorm (its `encode` returns the posterior mean, scaled by `(z - 0.1159) * 0.3611`). The former is less code.
2. `preview_negative=None` and `samples_cfg_free` (grey CFG and negative in the Samples section).
3. The Qwen3-4B text encoder: `Qwen3Model` + `Qwen3Config` from transformers 5.x (Anima and Klein already build Qwen3
   encoders in TagScribeR, `TS: training/families/anima/embedder.py`, `klein/embedder.py`) - reuse their offline
   tokenizer pattern (`helper_files` -> a `tokenizer/` folder next to the checkpoint, as Krea 2's `qwen3vl_tokenizer`).
4. A training-adapter row: present (`TS: training/description.py:132-133`); Fizgig's adapter file is hosted on
   Fizgig's Hugging Face account (licence of that file not checked).
5. Help: "train on Turbo with the adapter; without it Turbo LoRAs go soft"; previews are 8 steps with no CFG.

### 4.7 Port plan

Files: `training/families/zimage/{__init__,description,driver,model,pipeline,vae}.py`. Tests (CPU, tiny):

- description validity, six presets verbatim and import (two Fine-tune presets refused until stage 5),
  `FAMILY_TRAINING_ADAPTER True` in every preset;
- tiny S3-DiT: forward on a list of two differently sized latents, padding to 32, refiners run, block swap on CPU,
  `convert_comfy_state_dict` (fused qkv, renames) and the strict refusal;
- `_sample_sigma` against `3 s / (1 + 2 s)`; the loss target is `noise - x0` while the model returns `x0 - noise`
  (sign test: a model that returns `-(noise - x0)` exactly gives zero loss);
- LoRA keys `diffusion_model.layers.N.attention.to_q.lora_A.weight` + alpha round-trip; `convert_lora_state_dict`
  splits a ComfyUI `attention.qkv` LoRA into thirds with the shared down weight; `alias_flat`;
- the training adapter frozen at 1.0, off in previews and absent from the saved file (TagScribeR's Qwen 2.1 tests
  cover this path; re-run them for Z-Image);
- sampler: `sigmas(8, 3.0)` has 9 entries ending at 0; `generate` at CFG 1 calls the DiT once per step;
- text encoder on a tiny `Qwen3Config`: `hidden_states[-2]`, real tokens only, the template ends with the
  generation prompt and thinking on;
- VAE: scaling and shift round-trip on a tiny `AutoencoderKL`; the attention processor is the classic one;
- the smoke run.

Only a real run confirms: the complex RoPE on ROCm, the 12.3 GB INT8 peak, previews at 8 steps.

Size: **S-M** (about 820 upstream lines; little TagScribeR-side work).

---

## 5. ACE-Step 1.5 XL (music)

### 5.1 The model

ACE-Step 1.5 XL: a 4B flow DiT (32 layers, 2560 wide, sliding-window and full self-attention alternating, cross-attention
to caption, lyrics and timbre) over an Oobleck audio VAE (48 kHz stereo to 64 channels at 25 frames per second),
Qwen3-Embedding-0.6B for caption and lyric conditioning (`FZ: families/acestep.py:1-7`; `acestep/model.py:22-50`).
Two bases: XL Turbo (8 steps, the default) and XL SFT (30 guided steps, `:182-186`). Fizgig v7.3.0 added it with
Repair Studio as "a DAW for AI music" (`FZ: docs/RELEASE_NOTES_v7.3.0.md`). Training data is **audio files with
sidecars**: `song.wav/.mp3/.flac/.m4a`, `song.txt` caption, optional `song.lyrics.txt`, optional `song.json` with
bpm / timesignature / keyscale / language; songs over 240 s are cut (`FZ: docs/ACESTEP.md:7-14`; `acestep/driver.py:66-77`).

### 5.2 Model files (`FZ: families/acestep.py:36-55`; per upstream, no `gated` flag)

| pref key | role | repo / path | GB |
|---|---|---|---|
| `acestep15xl_dit` | dit, required | `Comfy-Org/ace_step_1.5_ComfyUI_files` `split_files/diffusion_models/acestep_v1.5_xl_turbo_bf16.safetensors` | 9.97 |
| `acestep15xl_sft_dit` | (alt base), optional | same repo `..._xl_sft_bf16.safetensors` | 9.97 |
| `acestep15xl_vae` | vae, required | same repo `split_files/vae/ace_1.5_vae.safetensors` | 0.34 |
| `acestep15xl_model_eq` | model_eq (workbench), optional | `ShootTheSound/Fizgig-ACE-Step-1.5-XL-Model-EQ` | 0.34 |
| `acestep15xl_text_encoder` | text_encoder, required | same Comfy repo `split_files/text_encoders/qwen_0.6b_ace15.safetensors` | 1.19 |

Helper files: `ACE-Step/Ace-Step1.5` `Qwen3-Embedding-0.6B/*.json, *.txt, *.jinja` and `acestep-v15-turbo/silence_latent.pt`
(`:162-163`). The tokenizer is loaded straight from the Hub (`FZ: acestep/driver.py:147`, not cache-first) and the
silence latent through `hf_hub_download` (`:131-138`).

### 5.3 Description fields (`FZ: families/acestep.py`)

| Field | Value | Line |
|---|---|---|
| key / arch_id / gui label / suffix | `acestep15xl` / `acestep15xl` / "ACE-Step 1.5 XL (music)" / `acestep15xl`; experimental | `:28-34` |
| media / subjects | `media=("voice",)`, `subject_isolation=False` | `:60-61` |
| "latents" | `latent_channels=64`, `spatial_factor=1`, `bucket_step=1`, native 1.0 (a song is `(64, T)`; upstream's image dataset pins audio to a sentinel 128 x 128 bucket, `FZ: dataset/image_dataset.py:100-112, :1282-1283`) | `:62-65` |
| blocks | 32 layers; the LoRA wraps q/k/v/o of self- and cross-attention ("the official trainer's targets") | `:67-71` |
| workbench | `("repair", "extract")`, `lucky_zones`, three `repair_presets` by ear, `repair_size=30` (seconds), `block_activity`, `repair_default_strength=0.5` | `:72-103` |
| EMA | `"0.98"`, `ema_short_run=True`, `ema_hint` (few steps per epoch) | `:104-105, :138-140` |
| hints | `long_run_hint=(400, ...)` ("try the final LoRA at a lower strength, around 0.5"), `song_captions` example | `:106-112` |
| fine-tune | `finetune=True`, 1e-5, `ft_regularisation=False` | `:113-115` |
| sliders | `slider_training=True`, `slider_sources=("prompts",)` only, `kind_notes` and `slider_prompt_notes` in music words | `:118-137` |
| LoRA | PEFT: `base_model.model.layers.{block}.{module}.{ab}.weight`, `lora_A` / `lora_B`, modules `self_attn.q_proj ... cross_attn.o_proj` (8), alpha `{prefix}.alpha`, `file_prefix="base_model.model."`; "ComfyUI maps base_model.model.<path> to diffusion_model.decoder.<path>" | `:142-152` |
| driver / modelspec / implementation | `fizgig.acestep.driver:AceStepDriver` / "ACE-Step-1.5-XL" / the ACE-Step repo | `:154-156` |
| precisions / optimizers / network types | `("bf16",)` only / `("adamw", "adamw8bit")` / `("lora",)` only | `:157-159` |
| controls hidden | `adaptive_lr=False`, `loss_watch=False` | `:160-161` |
| sampling | "ACE-Step 1.5 XL Turbo": 8 steps, CFG 1, explicit `sigmas = (1.0, 0.9545, 0.9, 0.8333, 0.75, 0.6429, 0.5, 0.3, 0.0)` (the official shift-3 Turbo schedule) | `:13, :165-169` |
| previews | `samples_cfg_free`, 8 steps, **`preview_every=40`**, **`preview_strengths=(0.5,)`**, CFG 1, `samples_text` banner: a sample prompt is `"caption | lyrics"` with lyric lines separated by `" / "` | `:170-180` |
| options | `ACESTEP_TRAIN_BASE` (tab "model": "XL Turbo (8 steps) — standard" / "XL SFT (30 steps + guidance)" -> `--dit=pref:acestep15xl_sft_dit base=sft`); `ACESTEP_PREVIEW_SECONDS` (tab "samples": 10 / 30 / 60 / 120, default 60 -> `preview_seconds=N`) | `:181-191` |
| notes | MIT; the sidecar format; conditioning as the official handler's text-to-music; the XL CLS token ("ComfyUI reads the first frame without it, which moves the velocity ~2%"); Turbo trains on its own 8 timesteps with no CFG dropout, SFT on logit-normal (mu -0.4, sigma 1) with 15% null-condition dropout, previews 30 steps, shift 1, APG guidance 7 | `:210-226` |

**Presets, verbatim** (`FZ: families/acestep.py:193-208`; `_preset` `:16-24`): every preset carries
`NETWORK_TYPE "LoRA (standard)"`, `SEED 42`, `ADAPTIVE_LR False`, `OPTIMIZER_TYPE "adamw"`, `MAX_GRAD_NORM 1.0`,
`BLOCKS_SWAP "Auto ..."`, `FAMILY_PRECISION "Auto ..."`, `KREA2_LOSS_WATCH False`, `KREA2_PER_IMAGE_LR False`,
`KREA2_AUTO_RECAPTION False`, `KREA2_WARMUP_LOOK False`; no `DATASET_MEGAPIXELS`.

| Preset | rank / alpha | LR | epochs | save every | accumulation | EMA | flags |
|---|---|---|---|---|---|---|---|
| `✨ ACE-Step Official LoRA (rank 64, 800 epochs)` (default) | 64 / 128 | 1e-4 | 800 | 40 | 4 | 0.98 | - ("the official LoRA recipe ... measured in Fizgig 11 Oct 2026") |
| `✨ ACE-Step Slider (rank 8, 20 epochs)` | 8 / 8 | 2e-4 | 20 | 4 | 1 | Off | `FAMILY_SLIDER True`, `FAMILY_SLIDER_GUIDANCE "2"` |
| `✨ ACE-Step Fine-tune (1e-5, 800 rotations)` | 64 / 16 | 1e-5 | 800 | 40 | 4 | Off | `FAMILY_FT True`, `FAMILY_FT_ROTATIONS "800"`, `FAMILY_FT_SAVE_EVERY "50"` |

### 5.4 The driver (`FZ: acestep/driver.py`, 678 lines; a third of it is workbench audio tooling)

| Hook | Lines | What it does |
|---|---|---|
| `read_audio`, `audio_seconds` | `:87-106` | ffmpeg subprocess: any audio -> fp32 (2, n) at 48 kHz, first 240 s; the duration from ffmpeg's banner |
| `silence_latent` | `:136-138` | the official `(1, 15000, 64)` latent from the Hub helper |
| `_TextEncoder` | `:141-181` | `Qwen3Model` 0.6B from the ComfyUI file; caption -> last hidden state (<= 256 tokens); lyrics -> token embeddings (<= 2048) |
| `load_dit` | `:194-204` | `AceStep15(AceConfig())`, `load_checkpoint` keeps `decoder.*` (rooted), `encoder.*`, `null_condition_emb`; drops the 5 Hz tokenizer path |
| `cover_guide` | `:206-219` | workbench (audio-to-audio cover) |
| `load_vae` | `:221-223` | diffusers `AutoencoderOobleck` after renaming stable-audio-tools keys (`vae.py:20-63`) |
| **`cache_stage`** | `:235-291` | both stages in its own layout: latents `(64, T)` sampled from the posterior with a seed-0 generator, metadata `seconds` and a size:mtime stamp; text = `caption_prompt` (the official instruction + caption + bpm / timesignature / keyscale / duration metas) and `lyrics_prompt` ("# Languages / # Lyric"), re-encoded when the composed key changes |
| `media_problem` | `:293-301` | an invalid `.json` sidecar |
| `encode_text`, `text_cache_key`, `_preview_seconds` | `:303-315` | sample prompts parsed as `caption | lyrics`, encoded at the preview length (and under the cover / repaint instructions for the workbench) |
| `_condition`, `_context` | `:318-334` | the frozen condition encoder runs **inside every forward** (text projector, lyric encoder, timbre encoder on the silence latent as the 30 s reference); context = silence latent + ones mask |
| `_draw_t` | `:347-355` | Turbo: uniform over its 8 timesteps inside the window; SFT: logit-normal (mu -0.4, sigma 1) |
| `training_loss` | `:357-369` | `x_t = t noise + (1 - t) x0`, target `noise - x0`, MSE; SFT swaps in the learned null condition with p 0.15 |
| `noise_latents`, `predict` | `:371-380` | |
| `initial_noise` | `:387-389` | `(1, frames, 64)`, frames = seconds x 25 |
| `encode_reference`, `source_wave`, `repaint_*`, `splice_source`, `encode_source`, `source_start` | `:396-476` | workbench: sound-like reference, audio-to-audio, repaint |
| `schedule`, `generate` | `:478-574` | Euler; Turbo over its schedule; SFT 30 steps with APG guidance 7 against the null condition (`apg` `:119-128`); prompt blend, source, cover, repaint |
| `decode`, `decode_audio` | `:593-600` | `Song` (waveform + a waveform picture) |
| `practice_latent`, `slider_preview` | `:602-612` | prompt sliders: the practice bank stays latents; the dial plays -1, 0, +1 with a second of silence between |
| `save_preview` | `:614-620` | writes `<name>.wav` (16-bit stereo via `minimax/common.write_wav`) and the waveform PNG last |
| `block_map`, `activity_axis`, `quant_target_names`, `ft_spec`, `alias_flat` | `:623-656` | 32 "DiT layers" `layer_N` with 8 Linears each |

Model (`FZ: acestep/model.py`): `AceConfig` dataclass (`:22-50`) so tiny configs are trivial; builds on transformers'
Qwen3 internals (`Qwen3MLP`, `Qwen3RMSNorm`, `Qwen3RotaryEmbedding`, `apply_rotary_pos_emb`, `:17-18`) - their
import paths and signatures under transformers 5.x are **not checked**; SDPA with additive sliding-window / key masks
(`:71-80, :108`); the condition encoder packs lyrics, timbre and text with real tokens first (`:270-295`); the
cover guide (`:343-459`) is workbench-only.

### 5.5 What TagScribeR has, and what it needs

Nothing of the data side exists: `TS: training/dataset.py` scans `IMAGE_EXTENSIONS` (`:41`), buckets by resolution,
and the Train tab, `DatasetBrowser`, captions, thumbnails and health checks are all about images. ACE-Step needs:

1. **An audio dataset layer**: scanning `.wav / .mp3 / .flac / .m4a` with the `.txt` / `.lyrics.txt` / `.json`
   sidecars, no buckets (one item per step, `(64, T)` latents of different lengths), `cache_stage`-style custom cache
   files or a generalised cache that stores non-image latents; `PORT_PLAN.md` stage 6 names this "media-aware dataset
   layer" for H3 clips, audio and voice. Upstream pins audio to a sentinel 128 x 128 bucket to reuse its image dataset
   (`FZ: dataset/image_dataset.py:100-112, :1282-1283`); TagScribeR's smaller `dataset.py` could do the same or grow a
   media kind.
2. **Previews as audio**: `save_preview` returning several files is supported (`TS: training/driver.py:112-115`), but
   the run panel shows PNGs; a `.wav` beside the waveform picture needs a player (Qt Multimedia) or at least a
   "open folder" affordance. `preview_every=40`, `preview_strengths=(0.5,)` and `preview_seconds` need parameters.
3. **ffmpeg** (imageio-ffmpeg) for decoding and probing; `write_wav` (stdlib `wave`) is trivial.
4. The Train tab for a family with no megapixels, no buckets, hidden Adaptive LR and loss watch, LoRA only, bf16 only,
   accumulation 4 default, 800 epochs; Help for the sidecar format, the "worse before better" note, strength 0.5.
5. The DatasetBrowser would show songs (or the user would point the Train tab at a folder the browser cannot open).
6. The workbench half of the driver (cover, repaint, reference, Model EQ, Repair presets) has no home: leave it out
   with a header note (about 250 lines).

Size: **XL**. Everything except the model and VAE loader is TagScribeR-side work that touches the app's data model.

---

## 6. Ranking: value to an image-dataset app's users against effort

| Rank | Family | Value | Effort | Why this position |
|---|---|---|---|---|
| 1 | MOSS-Image | High: Apache-2.0 weights, no gating, a 5B model that trains on 12 GB cards (`FZ: docs/MOSS_IMAGE.md:52`), measured presets from real runs, text rendering and long captions that reward a good captioning tool (TagScribeR's own strength: "describe the picture fully, including any text in it exactly as written") | M | The port is mostly vendoring plus two shared pieces (the small-decoder VAE loader, the per-family preview tick) that Flux 3 reuses. Everything is pure torch and fits the owner's ROCm stack (Qwen3.5's slow path is the known one). Not experimental upstream, but its `status_note` says it changes daily: port from this commit and mark the first TagScribeR version experimental |
| 2 | Flux 3 Image Klein | High: BFL's current model, the only one of the four with edit LoRAs (up to 10 references); INT8 / NF4 fit 8-16 GB cards | M-L | More TagScribeR-side work: multi-reference pairs, gated downloads, a config for the vendored model, the cuDNN backend; the presets are untuned and the licence is non-commercial. Doing it after MOSS means the VAE and preview-tick work is already there |
| 3 | Z-Image Turbo | Medium: still widely used (`FZ: docs/RELEASE_NOTES_v7.1.0.md`), 8-step previews are fast; but it trains only on a distilled model with Fizgig's own adapter, which is a narrower proposition than a base model | S-M | Cheapest port; could be done first as a warm-up if the owner prefers a quick win. Its Hub-config VAE loader and complex RoPE are the two things to watch on ROCm |
| 4 | ACE-Step 1.5 XL | None for image datasets: a music trainer whose data is songs with lyrics | XL | Only worth it if TagScribeR is to become a general media trainer (stage 6). Ask before starting |

Recommended order: **MOSS-Image -> Flux 3 Image Klein -> Z-Image Turbo**, each as its own stage on the work branch,
with the shared-layer changes of section 1 landing in the MOSS stage. ACE-Step: not now (see 7.1).

Rough line counts to port: MOSS ~1,250 (driver 362, model 335, embedder 410, sampling 133, description 275);
Flux 3 ~1,100 (driver 280, model 357, autoencoder 239 shared, sampling 135, embedder 78, description 151);
Z-Image ~820 (driver 228, model 463, pipeline 126, description 169); ACE-Step ~1,440 (driver 678, model 459,
vae 79, description 227) plus the data layer.

---

## 7. Questions for the owner (as recommendations)

1. **ACE-Step.** Recommendation: do not port it in this cycle. It is a music trainer and needs stage 6's media-aware
   dataset layer, audio previews and ffmpeg first. If the owner wants TagScribeR to train music later, the driver is
   a clean port once that layer exists; record the decision in `PORT_PLAN.md`.
2. **Order.** Recommendation: MOSS-Image, Flux 3, Z-Image (section 6). Alternative if a quick first result matters:
   Z-Image first (S-M), then MOSS, then Flux 3.
3. **Flux 3's gated files and licence.** TagScribeR's Download links cannot fetch gated files. Recommendation: mark
   the rows gated, show the licence page link and a "sign in to Hugging Face, accept the licence, download by hand or
   with a token in `HF_TOKEN`" line in Settings and Help; do not add token storage (Fizgig does not save it either,
   `FZ: docs/FLUX3.md:12`). State the FLUX Kommunity terms (non-commercial; outputs commercial under US$5M) in Help
   and in the family's docs page, as upstream does.
4. **MOSS's SFT preview checkpoint is `required=True` upstream** (`FZ: families/moss_image.py:91`). Recommendation:
   optional in TagScribeR (previews fall back to the Base model, as Klein's Distilled does), with the tick on by
   default when the file is set; the standing rule is to keep what works, and a required 5.4 GB preview-only file
   blocks training for no gain.
5. **MOSS text encoder: vendor Fizgig's Qwen3.5 text model or use transformers 5.x's?** Recommendation: vendor
   Fizgig's (parity; the streamed layer-major path for cards under 18 GB free; the ConvRot int8 file) and note in the
   header that transformers 5.x has the same architecture. Revisit if the vendored copy drifts from upstream.
6. **Multi-reference edits** (`edit_references`). Recommendation: lift TagScribeR's one-partner limit per family
   description in the Flux 3 stage, with upstream's counts (Flux 3 10, Qwen 2.1 10, Klein 4:
   `FZ: families/klein.py:171`, `qwen_image.py:161`) and upstream's ordering rule (`photo.png`, then `photo_1.png` ...).
7. **Flux 3's forced cuDNN attention.** Recommendation (a "mirror the fix" case): route it through
   `training/modules/sdpa.py`'s priority context, as Krea 2 is. Upstream's code would fail outright on ROCm.
8. **Fine-tune presets** (one each for Flux 3 and MOSS, two for Z-Image, one for ACE). Recommendation: ship them in
   the descriptions as upstream has them so Fizgig preset files import unchanged, and let the stage-1 guard refuse
   them until stage 5; then stage 5 brings `ft_spec` for all families at once.
9. **Workbench-only fields and code** (`int8_attention`, `activation_cache`, `repair_*`, `lucky_zones`,
   `workbench_follows_samples`, prompt-travel plumbing in the Z-Image and MOSS models, ACE's cover / repaint).
   Recommendation: leave them out of the descriptions and keep the INT8 `attend()` calls out of the model code (the
   module is CUDA-only and inference-only); keep the prompt-travel code in the vendored models only where removing it
   would mean rewriting the forward (Z-Image). Record each omission in the file header; stage 7 can bring them back.
10. **Provenance.** Recommendation: write the four `THIRD_PARTY_NOTICES.md` sections from the file headers (1.5),
    and for ACE-Step check the upstream repository's `LICENSE` before shipping, since Fizgig's own header says
    "MIT / Apache-2.0" and its notices file has no entry.
11. **Z-Image's training adapter** is a Fizgig-published file. Recommendation: use it as upstream does (the H3
    adapters set the precedent) and record its repo in the notes; its licence was not checked.
12. **Flux 3's timestep law is an assumption upstream** (`FZ: families/flux3.py:148-149`). Recommendation: port it as
    is and keep the note; if BFL publishes the training distribution, mirror upstream's update.
13. **Krea 2's own text encoder** may have the bf16 `inv_freq` drift Flux 3's embedder corrects
    (`FZ: flux3/embedder.py:37-42`, "1.25% off the reference at 160 tokens"). Not in scope; recommendation: check
    whether Fizgig's `krea2/embedder.py` applies the same fix at v8.7.1 and, if so, mirror it in the Krea 2 stage
    (it changes cached text, so it needs a `text_cache_rev` bump and the owner's real run).

---

## 8. Not checked

- Anything online: repo gating, weights licences, file sizes, the ComfyUI LoRA key maps the descriptions cite, the
  ACE-Step upstream `LICENSE`, Fizgig's Hugging Face adapter files.
- Transformers 5.x compatibility of ACE-Step's imports (`Qwen3RotaryEmbedding`, `apply_rotary_pos_emb`,
  `Qwen3MLP`, `Qwen3RMSNorm`) and of `Qwen3Model(Qwen3Config(...))` for Z-Image and ACE; `AutoProcessor` for
  Qwen3-VL in Flux 3's embedder (TagScribeR's Krea 2 uses `AutoTokenizer`).
- Complex-valued RoPE (Z-Image) on the owner's ROCm build.
- Any VRAM figure on the owner's card; every memory number above is upstream's on NVIDIA.
- The Fizgig web form (`FZ: families/form.py`) and launch (`families/launch.py`) handling of `FamilyOption` tokens
  beyond what the descriptions declare; the `--dit=pref:` alternate-base mechanism for ACE's SFT base (TagScribeR's
  `alt_base` field, `TS: training/description.py:139-141`, looks like the same idea).
- Fizgig's `families/ft.py` for the four `ft_spec` implementations (stage 5).
- Whether `diffusers>=0.35` ships `AutoencoderOobleck` and whether its `from_single_file` for the FLUX.1 VAE works
  offline with a local config folder (Z-Image).
- The MOSS loader's diffusers-shard glob (`FZ: moss_image/model.py:264-274`) against the actual repo layout.
- Fizgig commits after v8.7.1.
