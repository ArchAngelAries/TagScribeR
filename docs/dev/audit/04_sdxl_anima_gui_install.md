# Audit 04 - SDXL, Anima, GUI settings keys, environment, workbench map

Read-only audit. Upstream = the Fizgig copy at `.claude/fizgig-src` (HEAD `1c8ec88`, "Release notes v7.0.1",
5 Oct 2026). TagScribeR = the repo at HEAD `d426cab`. Paths are repo-relative: `fizgig-src/...` means
`.claude/fizgig-src/...`; everything else is TagScribeR. Nothing was run: no tests, no model loads, no GPU.
"Not checked" is said where it applies.

Status of this file: complete. Sections appear in the order they were written: A, C, B, D, then the owner
questions and the list of what was not checked. A.3 corrects two items of A.1 (items 9 and 12).

---

## Part A - SDXL and Anima

### A.0 The one structural fact that drives both recommendations

Upstream now has ONE SDXL family (`key="sdxl"`, `fizgig-src/src/fizgig/families/sdxl.py:30-37`,
`gui_label="SDXL (any checkpoint, experimental)"`) and one Anima family (`families/anima.py:27-34`).
TagScribeR has FIVE SDXL families (`training/families/sdxl/description.py:127-192`: `sdxl`, `pony`, `illustrious`,
`noobai_eps`, `noobai_vpred`) and one Anima family. Upstream's numbers for both families are dated "5 Oct 2026" and
are MEASURED on an RTX 5090 (memory, speed, an A/B of Min-SNR and noise offset). TagScribeR's numbers are labelled
"community starting point, not measured" in its own headers (`sdxl/description.py:1-7`,
`anima/description.py:62-65`).

---

### A.1 SDXL

#### 1. Checkpoint loading and variants

| | Fizgig | TagScribeR |
|---|---|---|
| File | One single-file `.safetensors`; VAE and text-encoder rows are optional overrides via `ModelFile.inside="sdxl_checkpoint"` (`families/sdxl.py:40-57`; `families/description.py:52-54, 529-538`) | Same idea via `ModelFile.default_to="dit"` (`training/families/sdxl/description.py:66-77`; `training/description.py:30-31`) |
| UNet loader | `UNet2DConditionModel.from_single_file(path, config="stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", torch_dtype=bf16)` (`sdxl/driver.py:127-130`). Needs the helper files from the Hub or HF cache: `helper_files=((_SDXL, ("model_index.json", "*/config.json", "tokenizer/*", "tokenizer_2/*", "scheduler/*")),)` (`families/sdxl.py:132`) | Own offline loader: vendored `SDXL_UNET_CONFIG` (`sdxl/unet.py:25-41`), reads only `model.diffusion_model.*` lazily, `convert_ldm_unet_checkpoint`, builds on the meta device, STRICT (any missing / unexpected key aborts) (`sdxl/unet.py:175-200`). No Hub access. |
| VAE loader | `AutoencoderKL.from_single_file(..., subfolder="vae", torch_dtype=torch.float32)` (`sdxl/driver.py:132-137`) | Own loader, vendored config, accepts LDM or diffusers-layout standalone VAE files, fp32, and replaces the attention processor with the classic `AttnProcessor` because SDPA is wrong for the VAE's single 512-wide head on AMD ROCm (`sdxl/vae.py:23-50`) |
| Text encoders | `StableDiffusionXLPipeline.from_single_file(path, torch_dtype=bf16, config=CONFIG_REPO)`, then drops UNet and VAE ("about 10 s at the cache stage") (`sdxl/driver.py:68-83`). Two tokenizers from the helper repo. | Own loader: reads only `conditioner.embedders.*`, vendored CLIP-L / bigG configs, `convert_ldm_clip_checkpoint` / `convert_open_clip_checkpoint`, strict (`sdxl/text.py:86-104`). ONE tokenizer (`openai/clip-vit-large-patch14`) serves both, padding id differs (`text.py:47-66, 116-120`) |
| Default download | Juggernaut XL v9 (`RunDiffusion/Juggernaut-XL-v9`, `Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors`, 7.11 GB); alternative link Illustrious XL v0.1 (`families/sdxl.py:7, 40-49`) | Per variant: SDXL base 1.0, Pony V6 (`LyliaEngine/Pony_Diffusion_V6_XL`), Illustrious v0.1, NoobAI-XL 1.1, NoobAI-XL Vpred 1.0 (`sdxl/description.py:127-177`) |
| Separate VAE default | `stabilityai/sdxl-vae` `sdxl_vae.safetensors` (`families/sdxl.py:50`) | `madebyollin/sdxl-vae-fp16-fix` `sdxl_vae.safetensors` (`sdxl/description.py:13, 68`) |
| Variants | "Any SDXL checkpoint"; Pony / Illustrious / RealVis named in the hint only. One family, one preset set, one sampler recipe (Juggernaut's). | Five families, each with its own download, sampling recipe, CFG, steps, notes (score tags for Pony, quality tags for Illustrious / NoobAI). |
| v-prediction | The driver HAS the code: `_v_pred()` reads `self.options.get("prediction", "epsilon") == "v"` (`sdxl/driver.py:171-172`), used for the target (`:217`), Min-SNR (`:233`) and the scheduler (`:245`). But NO `FamilyOption` sends `prediction` - `options=` holds only `SDXL_MIN_SNR` and `SDXL_NOISE_OFFSET` (`families/sdxl.py:184-194`). So v-pred is not reachable from upstream's GUI. Not checked: whether `--family_option prediction=v` typed on the CLI reaches the driver (see Part B / CLI). No zero-terminal-SNR rescale anywhere in the driver. | A real family: `SDXLVPredDriver` with `prediction="v_prediction"`, `zero_terminal_snr=True` (`sdxl/driver.py:202-205`), the Lin et al. rescale (`sdxl/sampling.py:28-40`), ZTSNR-aware sampler sigmas (`sampling.py:79-87`). |

**TagScribeR supports, Fizgig does not (KEEP all of these):**

1. NoobAI-XL v-prediction as a selectable family, with zero-terminal-SNR in training and in the preview sampler.
2. Per-variant families (Pony, Illustrious, NoobAI eps) with their own downloads, preview CFG / steps and prompt notes.
3. LoCon: the 3x3 convolutions of resnets and up / down samplers as LoRA targets (`SDXL_LOCON`, `sdxl/unet.py:111-129`). Upstream trains Linears only.
4. Offline loading (vendored configs, no Hub call) and strict key checking.
5. Batch size above 1 (`supports_batching = True`, fixed-shape conditioning, `sdxl/driver.py:34`). Upstream's `noise_latents` draws ONE timestep per call (`sdxl/driver.py:195`, `t = int(...item())`), i.e. one `t` for the whole batch.
6. Timestep window (Noise range min / max) for SDXL. Upstream's driver accepts `min_t` / `max_t` but the launch only sends `--min_timestep` / `--max_timestep` for a family with `train_areas` (`families/launch.py:566-567`); SDXL has none.
7. Adaptive LR on SDXL (upstream switched it off: `adaptive_lr=False`, `families/sdxl.py:121-123`, "reacts to noise").
8. Six optimizers (`adamw8bit, adamw, pagedadamw8bit, ademamix8bit, pagedademamix8bit, lion8bit`, `sdxl/description.py:97`) against upstream's `("adamw", "adamw8bit")` (`families/sdxl.py:119`).
9. Kohya LDM-layout LoRA keys on save (see 4).
10. Euler-ancestral preview sampler (`sdxl/driver.py:168-171`).
11. The AMD ROCm VAE attention fix (`sdxl/vae.py:45-49`). Upstream's SDXL driver has no equivalent (`AutoencoderKL.from_single_file` with the default SDPA processor). Not checked whether upstream patches SDPA globally for ROCm elsewhere (see Part C).

**Fizgig supports, TagScribeR does not:**

1. INT8 and NF4 bases (`precisions=("bf16", "int8", "nf4")`, `families/sdxl.py:98`) and a measured memory table.
2. LoKR (`network_types=("lora", "lokr")`, `:120`; `lokr_kohya_stems=True`, `:87`).
3. Slider LoRAs from prompt pairs or photo pairs (`slider_training=True`, `:126`; driver `noise_latents` / `predict` / `diff_ref`, `sdxl/driver.py:189-234`).
4. Full fine-tune (`finetune=True`, `:128`; `ft_spec`, `sdxl/driver.py:338-354`).
5. `torch.compile` of the 70 transformer blocks (`compiles=True`, `:108-116`; `compile_blocks`, `sdxl/driver.py:308-325`).
6. Unlimited prompt length: 75-token chunks, as many as the caption needs (`sdxl/driver.py:85-114`), with `pad_conditioning` (LCM repeat) for CFG (`:283-299`). TagScribeR caches a FIXED 3 chunks = 225 tokens and cuts the rest (`sdxl/text.py:30-32, 69-77`).
7. DPM++ 2M SDE Karras preview sampler (`sdxl/driver.py:242-249`).
8. Loading community LDM-layout LoRAs through `alias_flat` (`sdxl/driver.py:327-331`) - for the workbench and context LoRAs.
9. Workbench support: `workbench=("repair", "explorer", "profiler", "extract", "royale")` (`:77`).

#### 2. Objective, schedule, timesteps, Min-SNR, noise offset, v-pred

| Item | Fizgig | TagScribeR | Same? |
|---|---|---|---|
| Schedule | scaled-linear betas `0.00085`-`0.012`, 1000 steps, float64 cumprod (`sdxl/driver.py:164-169`) | identical formula (`sdxl/sampling.py:24-25, 36-40`) | Same |
| Noising | `a.sqrt()*x0 + (1-a).sqrt()*noise` (`:203`) | same (`sampling.py:51-53`) | Same |
| Timestep draw | `torch.randint(lo, min(hi, 1000), (1,))` with `lo=round(min_t*999)`, `hi=round(max_t*999)+1`: ONE integer per call, uniform (`:193-195`) | `u = torch.rand(batch)`; `frac = min_t + u*(max_t-min_t)`; `(frac*1000).long().clamp(0, 999)`: one per sample (`sampling.py:43-48`) | Same distribution at batch 1 (uniform over 0..999). NOT the same random stream: `randint` vs `rand`, and the order of draws differs (see next row), so the same seed gives a different run. |
| Draw order | timestep, then noise, then offset noise (`:195-200`) | noise, then offset noise, then timestep (`sdxl/driver.py:116-120`) | Differs (seed reproducibility against upstream only) |
| Target (eps) | `noise` (`:217`) | `noise` (`sampling.py:57-58`) | Same |
| Target (v) | `a.sqrt()*noise - (1-a).sqrt()*x0` (`:217`), plain schedule | same formula (`sampling.py:59-61`) on the zero-terminal-SNR schedule | Formula same; TagScribeR adds ZTSNR (needed by NoobAI v-pred) |
| Loss | `F.mse_loss(pred, target)`, unweighted (`:228`) | per-sample MSE mean (`sdxl/driver.py:127-130`) | Same at batch 1 |
| Min-SNR | `loss * (min(snr, gamma) / (snr+1 if v else snr))`, `snr = a/(1-a)` (`:229-233`). GUI: choice `off` / `γ 5` only (`families/sdxl.py:188`) | same weights (`sampling.py:65-69`). GUI: a float 0-20 (`training/params.py:183-187`) | Same maths; TagScribeR allows any gamma |
| Noise offset | `noise + off * randn(B, C, 1, 1)` (`:197-200`), entry box default `"0"` (`families/sdxl.py:191`) | same (`sdxl/driver.py:117-118`), float default 0.0 | Same |
| Size conditioning | `[h, w, 0, 0, h, w]` from the latent size x 8 (`:174-176, 204`) | same (`sampling.py:72-76`) | Same |
| Empty caption | hidden states and pooled are ZEROS (`sdxl/driver.py:97-101`, "force_zeros_for_empty_prompt") | the empty string is tokenised and ENCODED like any caption (`sdxl/text.py:116-133`) | DIFFERS. Affects caption dropout and an empty negative prompt. SDXL base and most fine-tunes were trained with the zero vector for a dropped caption (ComfyUI and A1111 instead encode the empty string). Owner question. |
| Padding of CLIP-G chunks | pads with `tok2.pad_token_id` of the SDXL tokenizer_2 (id 0, "!") (`sdxl/driver.py:89-91`) | pads bigG with id 0 explicitly (`sdxl/text.py:120`) | Same in effect |
| Slider / diff weighting | yes (`:218-226`) | no | Fizgig only |

Upstream's measured verdict on the two options: "Min-SNR 5 + offset 0.0357 reached the likeness no sooner and
wandered at epoch 4; plain MSE locked it from epoch 4. One run each, so off by default" (`families/sdxl.py:185-187`).
Both sides default them off.

#### 3. Presets

Upstream (`families/sdxl.py:14-27, 163-182`). Every preset carries: `NETWORK_TYPE "LoRA (standard)"`,
`SAVE_EVERY_N_EPOCHS 1`, `SEED 42`, `ADAPTIVE_LR False`, `ADAPTIVE_LR_MIN "1e-4"`, `ADAPTIVE_LR_MAX "4e-4"`,
`OPTIMIZER_TYPE "adamw"`, `GRADIENT_ACCUMULATION 1`, `MAX_GRAD_NORM 1.0`, `DATASET_MEGAPIXELS "1.0"`,
`BLOCKS_SWAP "Auto (detect from GPU)"`, `FAMILY_PRECISION "Auto (fits your free VRAM)"`,
`FAMILY_EMA "0.98 (recommended)"`, `KREA2_LOSS_WATCH True`, `KREA2_WARMUP_LOOK False`.

| Upstream preset | rank | alpha | LR | epochs | slider | FT | per-image LR | auto-recaption |
|---|---|---|---|---|---|---|---|---|
| `✨ SDXL Strong (rank 32, alpha 16, 5e-5)` (first = default) | 32 | 16 | 5e-5 | 20 | no | no | True | True |
| `✨ SDXL Standard (rank 16, alpha 8, 5e-5)` | 16 | 8 | 5e-5 | 20 | no | no | True | True |
| `✨ SDXL Slider (rank 32, alpha 16, 5e-5)` | 32 | 16 | 5e-5 | 30 | `FAMILY_SLIDER True` | no | False | False |
| `✨ SDXL Fine-tune (1e-5)` | 32 | 16 | 1e-5 | 20 | no | `FAMILY_FT True`, `FAMILY_FT_ROTATIONS "10"` | False | False |
| `✨ SDXL Fine-tune Official (3e-6)` | 32 | 16 | 3e-6 | 20 | no | `FAMILY_FT True`, `FAMILY_FT_ROTATIONS "10"` | False | False |

Upstream's own comments: "5e-5 for every SDXL preset (Peter, 5 Oct 2026: clearly better results than 1e-4 in his
comparisons)", "SDXL LoRAs gain from the halved alpha", and Adaptive LR removed for SDXL (`families/sdxl.py:164-169`).

TagScribeR (`training/families/sdxl/description.py:16-45`), the same three for each of the five variants. Every preset
carries: `NETWORK_TYPE "LoRA (standard)"`, `SAVE_EVERY_N_EPOCHS 1`, `SEED 42`, `MIN_TIMESTEP 0.0`, `MAX_TIMESTEP 1.0`,
`OPTIMIZER_TYPE "adamw8bit"`, `GRADIENT_ACCUMULATION 1`, `MAX_GRAD_NORM 1.0`, `DATASET_MEGAPIXELS "1.0"`,
`DATASET_BATCH_SIZE 1`, `KREA2_LOSS_WATCH True`, `KREA2_PER_IMAGE_LR True`, `KREA2_AUTO_RECAPTION False`,
`KREA2_WARMUP_LOOK False`, `FAMILY_EMA "0.98 (recommended)"`.

| TagScribeR preset | rank | alpha | LR | epochs | Adaptive LR | min / max | scheduler |
|---|---|---|---|---|---|---|---|
| `✨ <name> Fast (rank 16, adaptive LR)` (default) | 16 | 16 | 1e-4 | 20 | True | `5e-5` / `2e-4` | constant |
| `✨ <name> Standard (rank 32, cosine)` | 32 | 16 | 1e-4 | 30 | False | `1e-4` / `4e-4` | cosine |
| `✨ <name> Style (rank 16, gentle LR)` | 16 | 8 | 1e-4 | 40 | True | `5e-5` / `1e-4` | constant |

Differences that matter:

* Default preset: upstream rank 32 / alpha 16 at a FLAT 5e-5; TagScribeR rank 16 / alpha 16 with Adaptive LR
  5e-5..2e-4 (starts at the geometric middle, 1e-4). That is twice upstream's rate at twice the LoRA scale
  (alpha / rank 1.0 against 0.5), on a mechanism upstream measured as harmful for SDXL.
* Optimizer: upstream `adamw` (fused; measured 1493 -> 1046 ms/step against 8-bit AdamW, `families/sdxl.py:102-105`);
  TagScribeR `adamw8bit`.
* Auto-recaption: upstream on for the LoRA presets; TagScribeR off.
* `ADAPTIVE_LR_MIN "5e-5"` and `ADAPTIVE_LR_MAX "2e-4"` are valid options in `training/params.py:77-81`; fine.

#### 4. LoRA targets, key format, LoCon, LoKR, alpha, metadata

| | Fizgig | TagScribeR |
|---|---|---|
| Targets | 11 attention modules `IN04, IN05, IN07, IN08, MID, OUT00..OUT05`; each: `proj_in`, `proj_out` + per transformer block the 10 Linears `attn1.to_q/k/v/to_out.0`, `attn2.to_q/k/v/to_out.0`, `ff.net.0.proj`, `ff.net.2` (`sdxl/driver.py:26-41, 60-65`). 70 blocks x 10 + 22 = 722 modules (upstream's speed note says "1,444 tensors" = 722 x 2, `families/sdxl.py:103`). | The same 722 Linears (`sdxl/unet.py:111-121`; "722 modules", `sdxl/description.py:85-86`); with LoCon 760 (`+38` 3x3 convs). |
| Block ids | `IN04`.. per ATTENTION module, 11 (`n_blocks=11`) | per kohya block `in_01..in_08, mid, out_00..out_08`, `n_blocks=19`; blocks without a target are left out (`sdxl/driver.py:184-199`) |
| Key names on save | diffusers module names: `lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight` (`families/sdxl.py:80, 88-90`) - ComfyUI maps them; A1111 / Forge support for this layout was NOT checked here | original LDM names: `lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.lora_down.weight` (`sdxl/description.py:48-58`; `lora_key_name`, `sdxl/driver.py:68-71`) - what kohya sd-scripts writes |
| Reading the other layout | yes: `alias_flat` LDM -> diffusers (`sdxl/driver.py:44-51, 327-331`) | maps its own LDM names back on load (description note); reading a Fizgig-written diffusers-name file: NOT CHECKED (see open questions) |
| `.alpha` | `{prefix}.alpha` (`:84`) | `{prefix}.alpha` (`:53`) |
| LoCon / conv | none | `SDXL_LOCON` (`training/params.py:193-197`) |
| LoKR | yes, on the kohya stems (`lokr_kohya_stems=True`) | no (`network_types=("lora",)`, `:98`) |
| `modelspec.architecture` | `stable-diffusion-xl-v1-base/lora` (`families/sdxl.py:95`) | `stable-diffusion-xl-v1-base` (`sdxl/description.py:89`). SAI's spec uses the `/lora` suffix for a LoRA file; whether TagScribeR's `training/metadata.py` appends it was not yet checked at the time of writing (see A.3). |

#### 5. Text encoders, VAE precision, caching, cache ids

| | Fizgig | TagScribeR |
|---|---|---|
| Cached text dict | `{"hidden_states": (77*k, 2048) bf16, "pooled": (1280,) bf16}`, k = as many chunks as needed (`sdxl/driver.py:99-113`) | `{"crossattn": (231, 2048) bf16, "pooled": (1280,) fp32}`, fixed k = 3 (`sdxl/driver.py:104-106`; `text.py:30-34`) |
| Dict key names | `hidden_states`, `pooled` | `crossattn`, `pooled` - a cache made by one cannot be read by the other |
| Text-encoder dtype | bf16 on any device (`DTYPE`, `:75`) | bf16 on CUDA, fp32 on CPU (`sdxl/driver.py:80-84`) |
| VAE encode | fp32, `latent_dist.mode() * scaling_factor`, latents stored bf16 (`:153-157`) | fp32, `mode() * 0.13025`, latents stored fp32 (`sdxl/driver.py:97-101`) |
| VAE decode | in the VAE's own dtype = fp32 (`:302-306`) | bf16 on CUDA ("the training UNet is resident"), fp32 on CPU (`sdxl/driver.py:174-181`) |
| arch id in cache names | `sdxl` | `sdxl`, `pony`, `illustrious`, `noobaieps`, `noobaivpred` (`sdxl/description.py:127-175`) - so the SAME dataset is cached once per variant although the latents are identical across the four eps variants and the text differs only by checkpoint |

Upstream's note: "The VAE runs in fp32: SDXL's VAE overflows in half precision" (`families/sdxl.py:201`). TagScribeR
decodes previews in bf16. bf16 has fp32's exponent range, so the fp16 overflow does not apply, but it is a deviation
from upstream and from the model card; nothing here verified preview quality in bf16.

#### 6. Samplers and previews

| | Fizgig | TagScribeR |
|---|---|---|
| Default sampler | DPM++ 2M SDE, Karras sigmas (`DPMSolverMultistepScheduler(algorithm_type="sde-dpmsolver++", solver_order=2, use_karras_sigmas=True)`), SDE noise from `seed + 1` (`sdxl/driver.py:242-257`) | own Euler in sigma space, trailing spacing (`sdxl/sampling.py:79-119`) |
| Second sampler | Euler, `timestep_spacing="trailing"` (`:246-247`), preset "Euler (softer)" | Euler ancestral when the sampler option is `euler_a` / `euler_ancestral` (`sdxl/driver.py:168-171`) |
| Steps / CFG | `preview_steps=30`, `preview_cfg=3.0` (`families/sdxl.py:147-148`) | SDXL 30 / 6.0, Pony 25 / 7.0, Illustrious 28 / 6.0, NoobAI eps 28 / 5.5, NoobAI v-pred 28 / 4.5 (`sdxl/description.py:137-187`) |
| Negative | upstream's long photo negative (`families/sdxl.py:150-155`, about 130 tokens, encoded in chunks) | `"worst quality, low quality, lowres, bad anatomy, bad hands, jpeg artifacts, watermark, signature, text"` (`:102-103`) |
| CFG without a negative | zeros as the negative (`sdxl/driver.py:259-265`) | CFG is skipped unless `neg_cond` is given (`sdxl/driver.py:151`) |
| Size | 1024 x 1024, `repair_size=1024` | 1024 x 1024 |
| Workbench follows Samples tab | `workbench_follows_samples=True` | n/a |

#### 7. Memory plan and precisions

| | Fizgig | TagScribeR |
|---|---|---|
| Precisions | `("bf16", "int8", "nf4")` (`families/sdxl.py:98`) | `("bf16",)` (`sdxl/description.py:94`) |
| `train_memory` (peak GB incl. the 1024 preview) | `bf16 10.3`, `int8 8.2`, `nf4 7.4` at 0.5 and 1.0 MP, 0.0 saved per swapped block (`:117-118`) | `{}` - "NOT measured" (`:95-96`) |
| Training alone (comment) | bf16 6.0 / 6.5, INT8 4.0 / 4.4, NF4 3.2 / 3.6 GB at 0.5 / 1 MP (`:99-101`) | - |
| Quantised layers | LoRA targets except `proj_in` / `proj_out` (`sdxl/driver.py:334-336`) | - |
| Block swap | none ("No block swap") | none |
| Compile | Auto after 200 steps on bf16, boundary "outside", `fullgraph=False` (`:108-116`) | none for SDXL |

#### 8. Recommendation for SDXL: MERGE (keep TagScribeR's family set and loaders, take upstream's recipe and features)

Do not adopt upstream's implementation wholesale: it would drop v-pred / ZTSNR, the per-variant families, LoCon,
offline loading, batching, the timestep window and the ROCm VAE fix. Do not keep TagScribeR's as it is: its default
recipe contradicts upstream's measurements.

Concrete changes, in order of value:

1. **Presets: take upstream's values.** For every variant make the default `rank 32, alpha 16, LEARNING_RATE 5e-5,
   ADAPTIVE_LR False, OPTIMIZER_TYPE "adamw", 20 epochs` and the second `rank 16, alpha 8, 5e-5`, with
   `KREA2_PER_IMAGE_LR True`, `KREA2_AUTO_RECAPTION True` as upstream (`families/sdxl.py:170-171`). Keep TagScribeR's
   three as extra presets if the owner wants them, labelled community. Gain: the measured recipe. Loss: none.
   OWNER QUESTION: upstream measured on Juggernaut (photographic). For Pony / Illustrious / NoobAI the 5e-5 figure is
   an extrapolation; say so in the preset note.
2. **Adaptive LR on SDXL.** Upstream hides it (`adaptive_lr=False`). TagScribeR must not remove the control (narrowing),
   but should stop defaulting to it and show upstream's reason. OWNER QUESTION: hide as upstream, or keep visible and
   off?
3. **Add a generic "SDXL (any checkpoint)" entry** with upstream's Juggernaut download, sampling presets
   (`families/sdxl.py:135-146`), `preview_cfg=3.0` and negative, so a Fizgig user finds the same thing. Keep the five
   existing families. Decide whether the existing `sdxl` key becomes the generic one (upstream's key is `sdxl`).
4. **Precisions `int8`, `nf4`** with upstream's `train_memory` table and `quant_target_names` rule (not `proj_in` /
   `proj_out`). TagScribeR's shared quant layer already has both (`training/quant.py`). Gain: 8 GB cards.
5. **LoKR** for SDXL (`network_types`, `lokr_kohya_stems`). Needs `training/lora.py` to write LoKR on the kohya stems;
   not checked whether it can today.
6. **Long prompts.** Either move to upstream's variable-length chunks (loses stacking at batch > 1 unless padded with
   upstream's LCM-repeat `pad_conditioning`, which is exact), or keep the fixed 225 and say so. OWNER QUESTION.
7. **Empty caption = zeros** as upstream (`sdxl/driver.py:97-101`). OWNER QUESTION (it changes caption dropout and the
   empty negative).
8. **DPM++ 2M SDE Karras preview sampler** as an added sampler option; keep Euler / Euler-a.
9. **`alias_flat` in reverse:** accept diffusers-name SDXL LoRAs (Fizgig-written files) wherever TagScribeR reads a
   LoRA (context LoRA, resume).
10. **Slider, fine-tune, torch.compile** for SDXL: upstream features that live in the shared layer
    (`families/train.py`, `families/ft.py`, `families/compile.py`); port with the shared layer, not per family.
    These are missing for every family in TagScribeR (see Part B), so they are a separate project.
11. **Key layout on save: keep TagScribeR's LDM names** (kohya's own; loads in ComfyUI, A1111, Forge). OWNER QUESTION:
    upstream writes diffusers names; a TagScribeR SDXL LoRA and a Fizgig SDXL LoRA are therefore not byte-compatible,
    though ComfyUI loads both.
12. Fix `modelspec_arch` to carry `/lora` if `training/metadata.py` does not add it (see A.3).

---

### A.2 Anima

#### 1. Checkpoint loading and variants

| | Fizgig | TagScribeR |
|---|---|---|
| Files | DiT `split_files/diffusion_models/anima-base-v1.0.safetensors` (4.18 GB), VAE `split_files/vae/qwen_image_vae.safetensors` (0.25), TE `split_files/text_encoders/qwen_3_06b_base.safetensors` (1.19), optional Turbo LoRA `circlestone-labs/Anima-Official-LoRAs` `anima-turbo-lora-v0.2.safetensors` (0.15, role `speed_lora`) (`families/anima.py:36-50`) | the same three files (`training/families/anima/description.py:91-105`); NO Turbo LoRA row |
| DiT code | `anima/model.py`, 1389 lines, VENDORED from kohya sd-scripts `library/anima_models.py` (Apache-2.0), with SDPA, non-reentrant checkpointing, no block swap, and an INT8-attention hook (`anima/model.py:1-7, 245-261`) | `training/families/anima/model.py`, 405 lines, a compact RE-IMPLEMENTATION in the checkpoint's key names (`model.py:1-8`): T = 1 only, RoPE on the fly |
| DiT loader | `Anima(**DIT_CONFIG)` under `init_empty_weights`, strips `net.` / `model.diffusion_model.` / `diffusion_model.`, `strict=False, assign=True`, tolerates missing keys containing `seq`, `dim_spatial_range`, `dim_temporal_range`, `inv_freq`, refuses anything else (`anima/driver.py:102-114`) | meta-device build, strips `net.` / `model.diffusion_model.`, fully strict, and `refuse_fp8()` rejects a pre-quantised file (`anima/model.py:367-405`) |
| Variants | Base (train on it); Aesthetic and Turbo checkpoints load by prefix | same |
| fp8 | not offered (`precisions=("bf16","int8","nf4")`, "the community reports fp8 bases train badly", `families/anima.py:84`). No explicit refusal in the loader: an fp8 file would be cast to bf16 by `v.to(DTYPE)` (`anima/driver.py:108`) - without its scale keys it would fail on unexpected keys or load wrong. | explicit refusal with a message |

**TagScribeR supports, Fizgig does not (KEEP):**

1. Batch size above 1 (`supports_batching = True`, fixed 512-token conditioning with masks, `anima/driver.py:31-33`).
   Upstream's conditioning is variable length per caption.
2. Timestep window for Anima (upstream's launch does not send it without `train_areas`; the driver accepts it).
3. Adaptive LR presets (upstream's Anima presets all have `ADAPTIVE_LR False`, but the control is still shown since
   `adaptive_lr` defaults True - so this is a preset difference, not a capability one).
4. Six optimizers against upstream's two.
5. Explicit fp8 refusal, offline tokenizer folders (`qwen3_tokenizer/`, `t5_tokenizer/` next to the file,
   `anima/embedder.py:37-41`).
6. Caption shuffle variants note for tag datasets (a TagScribeR extension that applies to every family).

**Fizgig supports, TagScribeR does not:**

1. INT8 and NF4 bases with a measured memory table (`families/anima.py:84, 104-105`).
2. LoKR (`network_types=("lora","lokr")`, `:107`).
3. Slider LoRAs (`slider_training=True`, `:88`) and full fine-tune (`finetune=True`, `:87`; `ft_spec`, `anima/driver.py:250-262`).
4. `torch.compile` of the 28 blocks: measured 0.71 -> 0.31 s/step on a 5090 (`:95-103`).
5. The Turbo LoRA for previews (`speed_loras`, `preview_speed_lora`, `:121-142`), and it applies to the LLM adapter
   because the adapter runs in the grad-capable forward (`anima/driver.py:8-9, 146-152`).
6. Loading the official LoRAs' PEFT keys (`diffusion_model.*.lora_A/B`) and AI-Toolkit's diffusers names via
   `alias_flat` (`anima/driver.py:278-290`; `families/anima.py:75-76`).
7. Workbench support (`workbench=("repair","explorer","profiler","extract","royale")`, `int8_attention=True`).
8. Explicit `sigmas` passed to `generate` (`anima/driver.py:218-221`).

#### 2. Objective and timesteps

| Item | Fizgig | TagScribeR | Same? |
|---|---|---|---|
| Flow | `xt = (1-t)*x0 + t*noise`, target `noise - x0`, MSE (`anima/driver.py:169-187`) | same (`anima/driver.py:77-86`) | Same |
| Timestep | `t = sigmoid(randn(n))`, then `min_t + (max_t-min_t)*t` (`:169-170`) | `sigmoid(randn(b) * 1.0)`, no shift, rescaled into the window only when the window is not 0..1 (`anima/sampling.py:22-32`) | Same |
| Draw order | t, then noise (`:169-171`) | noise, then t (`anima/driver.py:78-79`) | Differs (seed stream only) |
| Model time input | `t.to(DTYPE)` = bf16 (`:160`), so the sinusoid is computed from a bf16-rounded t and cast back to bf16 (`anima/model.py:454-468`) | `t.float()`, sinusoid and its RMS norm in fp32, then cast (`anima/model.py:349-350`) | Small numeric difference (TagScribeR is the more precise one; upstream is what was measured) |
| Padding-mask channel | zeros at latent size (`:159`) | zeros (`anima/model.py:342`) | Same |
| Slider diff weighting | yes (`:175-186`) | no | Fizgig only |
| Noise offset | none; note: "noise_offset washes colours out" (`families/anima.py:177`) | none | Same |

#### 3. Presets

Upstream (`families/anima.py:15-24, 153-168`). Every preset: `NETWORK_ALPHA = rank`, `NETWORK_TYPE "LoRA (standard)"`,
`SAVE_EVERY_N_EPOCHS 1`, `SEED 42`, `ADAPTIVE_LR False`, `OPTIMIZER_TYPE "adamw"`, `GRADIENT_ACCUMULATION 1`,
`MAX_GRAD_NORM 1.0`, `DATASET_MEGAPIXELS "1.0"`, `BLOCKS_SWAP "Auto (detect from GPU)"`,
`FAMILY_PRECISION "Auto (fits your free VRAM)"`, `FAMILY_EMA "0.98 (recommended)"`, `KREA2_LOSS_WATCH True`,
`KREA2_PER_IMAGE_LR False`, `KREA2_AUTO_RECAPTION False`, `KREA2_WARMUP_LOOK False`, `FAMILY_FT False`.

| Upstream preset | rank | LR | epochs | extra |
|---|---|---|---|---|
| `✨ Anima Character (rank 16, 1e-4)` (default) | 16 | 1e-4 | 50 | |
| `✨ Anima Style (rank 16, 5e-5)` | 16 | 5e-5 | 30 | |
| `✨ Anima Official (rank 32, 2e-5)` | 32 | 2e-5 | 30 | |
| `✨ Anima Slider (rank 4, 2e-4)` | 4 | 2e-4 | 30 | `FAMILY_SLIDER True` |
| `✨ Anima Fine-tune (1e-5)` | 16 | 1e-5 | 30 | `FAMILY_FT True`, `FAMILY_FT_ROTATIONS "10"` |
| `✨ Anima Fine-tune Official (1e-6)` | 16 | 1e-6 | 30 | `FAMILY_FT True`, `FAMILY_FT_ROTATIONS "10"` |

Upstream says of these: "community values (Oct 2026) ... Not yet measured in Fizgig" (`families/anima.py:154-155`).
So for Anima PRESETS neither side is measured; upstream's memory and speed figures are.

TagScribeR (`training/families/anima/description.py:62-79, 162-172`). Every preset: `NETWORK_ALPHA = rank`,
`OPTIMIZER_TYPE "adamw8bit"`, `MIN_TIMESTEP ""`, `MAX_TIMESTEP ""`, `KREA2_LOSS_WATCH True`,
`KREA2_PER_IMAGE_LR True`, `KREA2_AUTO_RECAPTION False`, `KREA2_WARMUP_LOOK False`, `FAMILY_EMA "0.98 (recommended)"`,
`BLOCKS_SWAP "Auto (detect from GPU)"`, `SEED 42`, `SAVE_EVERY_N_EPOCHS 1`, `MAX_GRAD_NORM 1.0`.

| TagScribeR preset | rank | LR | epochs | Adaptive | min / max | MP |
|---|---|---|---|---|---|---|
| `✨ Anima Fast (rank 8, adaptive LR)` (default) | 8 | 4.5e-5 | 24 | True | `1e-5` / `2e-4` | 0.5 |
| `✨ Anima Standard (rank 32, LR 2e-5)` | 32 | 2e-5 | 40 | False | `1e-5` / `1e-4` | 1.0 |
| `✨ Anima Style (rank 16, gentle LR)` | 16 | 3.2e-5 | 40 | True | `1e-5` / `1e-4` | 1.0 |

Differences: upstream's default is rank 16 at a flat 1e-4, 50 epochs, 1 MP; TagScribeR's is rank 8, adaptive around
4.5e-5, 24 epochs, 0.5 MP - markedly gentler and shorter. Upstream turns per-image LR OFF for Anima; TagScribeR has it
ON. Upstream uses fused `adamw`. Only the "rank 32, 2e-5" preset agrees (the model card's value).
`ADAPTIVE_LR_MAX "2e-4"` / `"1e-4"` and `ADAPTIVE_LR_MIN "1e-5"` are valid options in `training/params.py:77-81`.

#### 4. LoRA

| | Fizgig | TagScribeR |
|---|---|---|
| Targets | per block: `self_attn.q_proj/k_proj/v_proj/output_proj`, `cross_attn.q_proj/k_proj/v_proj/output_proj`, `mlp.layer1`, `mlp.layer2` (`families/anima.py:69-71`) | identical tuple (`anima/description.py:124-126`; `anima/driver.py:24-26`), 280 modules |
| Key | `lora_unet_blocks_{block}_{module}.lora_down/.lora_up.weight`, `.alpha` (`:67-74`) | identical (`:122-129`) |
| LLM adapter | never trained | never trained, and its forward is under `no_grad` (`anima/model.py:314-334`) |
| LoKR | yes (LyCORIS `diffusion_model.<path>` names by default: `lokr_kohya_stems` False) | no |
| Reads other layouts | PEFT `diffusion_model.*.lora_A/B`, AI-Toolkit `transformer_blocks_N_attn1_to_q` (`anima/driver.py:278-290`) | not checked beyond its own format |
| `modelspec.architecture` | `"anima"` (`:81`) | `"Anima"` (`:137`) - case differs |
| `implementation` | the model card URL | the model card URL |

LoRA files of the two are key-compatible.

#### 5. Text encoder, VAE, caching

| | Fizgig | TagScribeR |
|---|---|---|
| Qwen3 0.6B dtype | **fp32**: "in bf16 the first token (Qwen's very large activation) is 16% off and the rest ~1%" (`anima/driver.py:78-79`) | **bf16** on the device (`anima/embedder.py:53, 95`) - upstream measured this as a real error in the conditioning |
| Qwen3 config | spelled out, `rope_theta=1000000.0` passed explicitly because the HF file stores it under `rope_parameters` (transformers 5) (`anima/driver.py:32-38`) | vendored dict with `"rope_theta": 1000000.0` via `Qwen3Config.from_dict` (`anima/embedder.py:27-34, 89`). TagScribeR runs transformers 5.x; whether `Qwen3Config` there honours a top-level `rope_theta` or wants `rope_parameters` was NOT checked. If it is ignored the encoder falls back to 10000 and every position after the first is wrong. CHECK ON A REAL RUN (compare one caption's states against ComfyUI). |
| Qwen tokenisation | raw caption, `add_special_tokens=False`, cut at 512, NO padding, no mask; empty -> one pad id `151643` (`:86`) | padded to 512 on the right with a mask, padding states zeroed; empty -> one pad token (`anima/embedder.py:44-47, 102-116`). Equivalent for the real tokens (causal model, right padding). |
| T5 ids | the Anima repo's own `t5_tokenizer/tokenizer.json` (helper `circlestone-labs/Anima-Base-v1.0-Diffusers`), with EOS, cut at 512, no padding (`:68-72, 88`) | `google/t5-v1_1-xxl` via `AutoTokenizer` (needs `spiece.model`, i.e. sentencepiece), padded to 512 with a mask (`anima/embedder.py:22-24, 74-79, 113`). Same 32128-id vocabulary per the header; equivalence "was not diffed" (`anima/description.py:50-51`). |
| Adapter call | `dit.llm_adapter(qwen, ids)` with no masks, output zero-padded to 512 (`anima/driver.py:146-152`) - "ComfyUI's path exactly" | adapter called with target and source masks, output zeroed at T5 padding (`anima/model.py:326-334`) - sd-scripts' path. Mathematically the same context for the real tokens. |
| Cached dict | `{"qwen": (Lq, 1024) bf16, "t5_ids": (Lt,) int64}` - small, variable | `{"prompt_embeds": (512, 1024) bf16, "attn_mask", "t5_ids" int32, "t5_mask"}` - about 1 MB per caption, fixed |
| Helper files | `circlestone-labs/Anima-Base-v1.0-Diffusers`: `tokenizer/*`, `t5_tokenizer/*` (`families/anima.py:108-109`) | `Qwen/Qwen3-0.6B` tokenizer files and `google/t5-v1_1-xxl` `spiece.model` etc. (`anima/description.py:175-176`) |
| VAE | Krea 2's `load_vae(path, input_channels=3, device, disable_mmap=True)`; latents stored in the VAE's dtype; decode in bf16 (`anima/driver.py:116-118, 134-138, 241-248`) | Krea 2's `load_vae(path, input_channels=3, device)`; latents stored bf16; decode in the VAE's dtype (`anima/driver.py:42-44, 58-62, 103-106`) |
| arch id | `anima` | `anima` (same id, but the text cache layouts are incompatible) |

#### 6. Samplers and previews

| | Fizgig | TagScribeR |
|---|---|---|
| Sampler | Euler, `sigma = 3t / (1 + 2t)`, steps + 1 values from 1 to 0 (`anima/driver.py:208-211`) | identical (`anima/sampling.py:18-19, 35-37`) |
| Sampling preset | `"Anima Base"`: steps 30, cfg 4.5, shift 3.0 (`families/anima.py:115-119`) | `"Anima base"`: steps 30, cfg 4.0, shift 3.0 (`anima/description.py:146-153`) |
| Preview defaults | `preview_steps=20`, `preview_cfg=4.5` (`:143-144`) | `preview_steps=30`, `preview_cfg=4.0` (`:155-156`) |
| Negative | `"worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration"` ("the model card's", `:145-146`) | `"worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, watermark, signature, text"` (`:157-158`) |
| CFG without a negative | a synthetic uncond: `{"qwen": zeros(1, 1024), "t5_ids": [1]}` (`anima/driver.py:225-227`) | CFG skipped unless an uncond is given (`anima/driver.py:98`) |
| Turbo | LoRA v0.2, 10 steps, CFG 1, strength default 0 (off) with 20 plain steps (`:121-142`) | none |
| Initial noise | `randn(1, 16, 1, h/8, w/8)` from the seed (`:204-206`) | `randn(1, 16, h/8, w/8)` (`anima/sampling.py:40-43`) - the same numbers (same element count and generator) |

#### 7. Memory and precisions

| | Fizgig | TagScribeR |
|---|---|---|
| Precisions | `("bf16", "int8", "nf4")` | `("bf16",)` |
| `train_memory` (peak GB incl. preview) | `bf16 8.9`, `int8 7.3`, `nf4 6.6` at 0.5 and 1.0 MP (`families/anima.py:104-105`) | `{}` |
| Training alone | bf16 5.1 / 5.7, INT8 3.4 / 4.0, NF4 2.9 / 3.5 GB at 0.5 / 1 MP (`:89-91`) | - |
| Compile | Auto after 300 steps (bf16), boundary "inside", +0.4 GB (`:95-103`) | none ("Not part of this port", `anima/description.py:196-197`) |
| Block swap | none | none |

#### 8. Recommendation for Anima: ADOPT upstream's recipe and conditioning details; keep TagScribeR's model code and its extras

TagScribeR's Anima header says the family "has never been run against the real checkpoint". Upstream's has been run
and measured. Where they differ in a way that changes the numbers, upstream should win:

1. **Text encoder in fp32** (`anima/driver.py:78-79`). TagScribeR's bf16 is, by upstream's measurement, 16% off on the
   first token. One-line change in `training/families/anima/embedder.py` (load and run in fp32, store bf16).
   Invalidates existing Anima text caches (bump the cache id).
2. **Verify `rope_theta` under transformers 5.x** (see row above) before anyone trains.
3. **T5 tokenizer from the Anima repo** (`circlestone-labs/Anima-Base-v1.0-Diffusers`, `t5_tokenizer/tokenizer.json`
   through `PreTrainedTokenizerFast`), as upstream: removes the sentencepiece dependency and the un-diffed
   equivalence. Keep the local-folder fallback.
4. **Presets: take upstream's six** (first = `✨ Anima Character (rank 16, 1e-4)`), per-image LR off, `adamw`.
   Keep TagScribeR's three as additional community presets if wanted. Slider and fine-tune presets only once those
   modes exist.
5. **Preview defaults** `preview_steps=20`, `preview_cfg=4.5`, upstream's negative; sampling preset cfg 4.5.
6. **Precisions int8 / nf4 + `train_memory`**, **LoKR**, **Turbo LoRA row and `speed_loras`** (adapter must then leave
   `no_grad` or the Turbo LoRA's adapter keys must be ignored - upstream applies them), **`alias_flat`** and PEFT-key
   reading, **`modelspec_arch="anima"`** (lower case, as upstream).
7. **Keep** TagScribeR's `model.py` (it is strict, smaller, and builds on meta) OR replace it with upstream's vendored
   file. OWNER QUESTION. Upstream's is the tested one and carries the INT8-attention hook, `compile_blocks` and the
   fine-tune spec; TagScribeR's fixed-shape masked path is what makes batch > 1 possible. A merge keeps TagScribeR's
   file and adds the missing pieces. What was compared here: module names, norm epsilons (`1e-6` for q/k norms and
   `t_embedding_norm` on both sides), the sinusoid (`cat([cos, sin])`, exponent `i / half`), adapter structure
   (6 layers, self-attention on, RMSNorm, `o_proj`, GELU MLP with biases), RoPE for the adapter (theta 10000). All
   agree. NOT compared line by line: the 3-axis RoPE of the DiT (`anima/model.py:281-391` upstream vs
   `rope_3d` in TagScribeR), `FinalLayer`, `PatchEmbed` ordering.
8. Keep: batching, timestep window, fp8 refusal, optimizer list, offline tokenizer folders.

Lost by this: nothing TagScribeR supports. Gained: measured memory plan, 8 GB cards, correct text precision, LoKR,
Turbo previews.

---

### A.3 Open checks for Part A (filled in below as they were done)

* **`prediction=v` upstream is CLI-only.** `families/train.py:1560` defines `--family_option KEY=VALUE`, `:1671`
  builds the dict, `:716` hands it to `driver.set_options` (`families/driver.py:104-105`). So
  `--family_option prediction=v` reaches `SDXLDriver._v_pred()`, but no GUI control sends it, `docs/CLI.md` and
  `README.md` do not mention it (grep for "prediction" / "v-pred": no hits), and there is no zero-terminal-SNR.
  TagScribeR's v-pred family is therefore strictly more than upstream has.
* **modelspec.** TagScribeR: `training/metadata.py:37` appends `"/lora"` to `fam.modelspec_arch`, giving
  `stable-diffusion-xl-v1-base/lora` - correct. Upstream: `families/sdxl.py:95` already says
  `"stable-diffusion-xl-v1-base/lora"` and `src/fizgig/training/metadata.py:136, 141-142` appends `/lora` again for a
  LoRA, i.e. `stable-diffusion-xl-v1-base/lora/lora` (read from the code, not from a written file). Do NOT copy
  upstream's `modelspec_arch` string for SDXL. A.1 item 12 is void.
* **Reading Fizgig-written SDXL LoRAs.** `training/lora.py:228-230` indexes every module under BOTH its diffusers
  name and its `lora_key_name` (LDM) name, and `_module_for` (`:303-316`) resolves `lora_unet_<either>`. So a
  diffusers-name SDXL LoRA (upstream's layout) already loads in TagScribeR. A.1 item 9 is already done.
* **LoKR for SDXL.** `training/lora.py:240-246` writes a kohya family's LoKR as `diffusion_model.<diffusers path>`.
  For SDXL that path is not what ComfyUI calls the module (ComfyUI uses LDM names), which is exactly why upstream
  added `LoRAFormat.lokr_kohya_stems=True` (`families/description.py:102-105`). Enabling LoKR for SDXL in TagScribeR
  needs that flag ported (write LoKR on `lora_unet_<LDM name>` stems). LoKR is Linear-only in TagScribeR
  (`training/lora.py:26-27, 286-293`), so LoKR + LoCon trains no convs.
* **`rope_theta` under transformers 5.18.0** (the version in `venv/Lib/site-packages`): the RoPE config mixin takes a
  top-level `rope_theta` kwarg and moves it into `rope_parameters`
  (`transformers/modeling_rope_utils.py:752-771, 783-789`). So TagScribeR's vendored dict should be honoured. Read
  only, not executed.
* **ROCm VAE attention.** Upstream has no counterpart of `training/modules/wide_attention.py`: a grep of
  `fizgig-src/src` for `wide_attention`, "512-wide" and "heads above 256" finds nothing, and `src/fizgig/rocm/` holds
  only `cache_exit.py`. The fix (commit `4267d2e`) is TagScribeR's own and must survive any merge of the SDXL VAE path.
  It may be worth reporting upstream.
* **`fetch_models.py`** has no SDXL- or Anima-specific code: it walks the registry's descriptions
  (`src/fizgig/scripts/fetch_models.py:123-152`) and fetches each family's `model_files` and `helper_files`. So what
  upstream "trains on" is exactly the `ModelFile` rows quoted in A.1.1 and A.2.1.

---

## Part C - Environment and packages

Files read: upstream `fizgig-src/requirements.txt` (74 lines), `install_fizgig_rocm.bat` (328), `run_fizgig_rocm.bat`
(75), `filter_requirements_rocm.py` (118), `write_rocm_env.py` and `install_fizgig.py` (grep only),
`docs/INSTALL.md` (grep); TagScribeR `requirements.txt` (52), `tools/install.py` (304), `core/hardware.py` (301),
and the installed versions in `venv/Lib/site-packages` (directory listing only).
`rocm_env.bat` does not exist in the upstream copy: it is GENERATED by `write_rocm_env.py` at install time.
The upstream copy is a shallow clone (one commit), so "did upstream move" is judged against TagScribeR's pins, which
were copied from upstream at the port, not against upstream's history.

### C.1 torch / ROCm / bitsandbytes pins: upstream has NOT moved

| Pin | Fizgig | TagScribeR | Same |
|---|---|---|---|
| Windows ROCm index | `https://rocm.nightlies.amd.com/whl-multi-arch/` (`install_fizgig_rocm.bat:28`) | same (`tools/install.py:39`) | yes |
| Windows ROCm torch | `2.12.0+rocm7.15.0a20260728` (`:32`) | same (`:41`) | yes |
| torchvision | `0.27.0+rocm7.15.0a20260728` (`:33`) | same (`:42`) | yes |
| `rocm-sdk-devel` | `7.15.0a20260728` (`:34`) | same (`:43`) | yes |
| Experimental index | `https://nightly.repo.amd.com/rocm/whl-next/` (`:31`) | same (`:40`) | yes |
| bitsandbytes (Win ROCm) | 0xDELUXA `bitsandbytes-0.50.2.dev0-cp312-cp312-win_amd64.whl`, release `0.50.2.dev0-py3.12-rocm7.16-win_amd64_all` (`:35`) | same URL (`:57-61`) | yes |
| Linux ROCm stable | `torch==2.12.0+rocm7.14.0`, `rocm-sdk==7.14.0` (`docs/INSTALL.md:101`); the Linux default is now NIGHTLY (`:99`) | `2.12.0+rocm7.14.0`, `rocm-sdk-devel==7.14.0` (`tools/install.py:45-47`), stable only | pins same; upstream's default Linux channel differs. Not checked: `install_fizgig_rocm.sh` line by line. |
| CUDA | `torch==2.10.0`, `torchvision==0.25.0`, cu128 (`requirements.txt:5-7`) | same (`tools/install.py:49-51`) | yes |
| bitsandbytes (CUDA) | `==0.48.2` (`requirements.txt:21`) | `>=0.48` (`tools/install.py:229`) | TagScribeR floats upward |
| bitsandbytes (Linux ROCm) | `>=0.50.0` (`docs/INSTALL.md:116`) | `>=0.50.0` (`tools/install.py:220`) | yes |
| triton | `triton-windows>=3.5.1,<3.7` (`requirements.txt:39`) | same (`requirements.txt:34`); installed `3.6.0.post26` | yes |
| Python | 3.10-3.13 (CUDA), 3.12 for Windows ROCm (`docs/INSTALL.md:17, 74`) | 3.12 (`tools/install.py:268-270`) | yes |

Installed in TagScribeR's venv (dist-info names): `torch-2.12.0+rocm7.15.0a20260728`,
`torchvision-0.27.0+rocm7.15.0a20260728`, `bitsandbytes-0.50.2.dev0`, `triton_windows-3.6.0.post26`,
`transformers-5.18.0`, `diffusers-0.40.0`, `accelerate-1.15.0`, `safetensors-0.8.0`, `tokenizers-0.23.2`,
`huggingface_hub-1.33.0`, `sentencepiece-0.2.1`, `numpy-2.4.4`, `onnxruntime-1.30.0`, `opencv_python-4.10.0.84`.

### C.2 Packages upstream installs that TagScribeR does not

| Package (upstream pin) | What needs it upstream | TagScribeR status | Matters when |
|---|---|---|---|
| `einops==0.7.0` (`requirements.txt:13`) | model code: `fizgig-src/src/fizgig/anima/model.py:14-15` (`rearrange`, `Rearrange`), Klein and others | not installed; TagScribeR's ports replace it with reshapes (`training/families/klein/model.py:4`, `anima/model.py:5`) | ANY upstream model file adopted verbatim (e.g. upstream's `anima/model.py`) needs it. Cheapest fix: add `einops` to `requirements.txt`. |
| `hqq==0.2.8.post1` (`:27`), installed with `DISABLE_CUDA=1` (`install_fizgig_rocm.bat:210-211`) | MiniMax H3 "4-bit HQQ" base (`families/minimax.py:359`, `precisions=("int8","nf4","hqq")`; `launch.py:45` label "4-bit HQQ (more accurate 4-bit)") | absent; `training/presets.py:134-136` maps HQQ to NF4 with a note | H3 parity. Not on the list of approved departures in the skills: OWNER QUESTION. |
| `comfy-kitchen==0.2.31` (`:28`) | INT8 attention for workbench renders (`src/fizgig/modules/int8_attention.py`; `int8_attention=True` on Anima) | absent | workbench only. Upstream STRIPS it on ROCm (`filter_requirements_rocm.py:24`: `SKIP_PACKAGES` holds `comfy-kitchen` and `nvidia-ml-py`), so on AMD it is never installed. An NVIDIA-only feature. |
| `imageio-ffmpeg>=0.5` (`:49`) | LoRA Royale MP4 export | absent | workbench (Part D) |
| `insightface>=0.7.0` (`:62`) + the `buffalo_l` download at install (`install_fizgig_rocm.bat:238-239`) | `face_utils.py`, Royale's likeness compare (`src/fizgig/lora_royale/likeness.py`) | absent (face tools are on the approved "later" list) | face tools |
| `timm>=0.9.0` (`:66`) | Florence-2 captioning | absent (TagScribeR has its own captioners) | not needed |
| `toml==0.10.2`, `voluptuous==0.15.2` (`:42-43`) | dataset TOML + schema | absent by design (JSON dataset config is an approved decision) | not needed |
| `tensorboard==2.20.0` (`:52`) | `LOG_WITH` / `LOGGING_DIR` | absent | the logging keys (Part B) |
| `tqdm==4.67.1` (`:53`) | progress | one import in TagScribeR, not in `requirements.txt` (arrives with `huggingface-hub` / `transformers`) | fine |
| `pyyaml==6.0.3`, `packaging==25.0` (`:69-70`) | misc; `packaging` for version parsing | not listed (both arrive transitively) | fine |
| `tokenizers==0.22.2` (`:17`) | pinned with transformers 4.57.6 | follows transformers 5.x | fine |

Version differences on shared packages: `transformers` 4.57.6 vs `>=5.18,<6` (approved); `diffusers` 0.32.1 vs
`>=0.35` (0.40.0 installed); `accelerate` 1.6.0 vs `>=1.6.0`; `huggingface-hub` 0.34.3 vs unpinned;
`safetensors` 0.5.3 vs `>=0.5.3`.

A risk in TagScribeR's own choice: its SDXL loaders call `diffusers.loaders.single_file_utils.convert_ldm_*` directly
(`training/families/sdxl/unet.py:180`, `text.py:88`, `vae.py:27`). That is a private diffusers module whose
signatures can change between releases, and `diffusers>=0.35` has no upper bound. Upstream uses the public
`from_single_file` on a pinned 0.32.1. Suggest an upper bound on `diffusers` or a unit test on the three converters.

Packages TagScribeR has that upstream does not: `PySide6`, `qt-material`, `qtawesome`, `openai`, `piexif`,
`send2trash`, `keyring` (the app's own).

### C.3 Environment variables

Upstream sets them in `run_fizgig_rocm.bat` before the GUI starts; TagScribeR sets them in
`core/hardware.py: apply_runtime_env` (`:68-83`), called once from `main.py:10`; the training child processes
inherit them (`training/pipeline.py:585` builds the child env from `os.environ`).

| Variable | Fizgig (`run_fizgig_rocm.bat`) | TagScribeR | Gap |
|---|---|---|---|
| `MIOPEN_FIND_MODE=2` | `:8` | `hardware.py:75` | none |
| `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` | `:9` | NOT SET | Read only by the `flash-attn` package's AMD Triton backend. TagScribeR offers `ATTENTION_MECHANISM = flash3` for Klein (`training/params.py:226-228`); if flash-attn is ever used on AMD this is needed. Harmless otherwise. Add it. |
| `PYTORCH_ALLOC_CONF` | `expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.8` (`:11`) | `expandable_segments:True,garbage_collection_threshold:0.8` (`hardware.py:81`) | TagScribeR lacks `max_split_size_mb:512`. Upstream sets its value for ROCm only; TagScribeR sets its shorter one on every backend. The owner's card runs at its memory limit: align the ROCm value. |
| `FIZGIG_GPU_BACKEND=rocm` | `:12` | n/a | Upstream's own switch (VRAM monitor `utils/vram_monitor.py:591`; the Linux cache fast-exit). No TagScribeR equivalent needed on Windows. |
| `BNB_ROCM_VERSION` | from the generated `rocm_env.bat` (`:17-22`); unset with `--experimental` | computed at start from the torch version, set only if bitsandbytes ships that library (`hardware.py:110-125`) | equivalent in intent. Not checked: which `libbitsandbytes_rocm*` files the wheel holds (torch is `+rocm7.15`, the wheel's release name says `rocm7.16 ... all`). |
| `ROCM_PATH`, `HIP_PATH` | `rocm_env.bat`, fallback `venv\Lib\site-packages\_rocm_sdk_core` (`:25-26`) | `_expose_pip_rocm_sdk` (`hardware.py:86-98`) | none |
| `PATH` += `%ROCM_PATH%\bin`, `_rocm_sdk_devel\bin`, `venv\Scripts` | `:29-31` | `hardware.py:100-107` | none |
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` | only when NOT gfx101x / gfx103x (`:44-56`) | always on ROCm (`hardware.py:76`) | TagScribeR also sets it on RDNA1/2 |
| `TORCH_BACKENDS_CUDA_FLASH_SDP_ENABLED=0`, `..._MEM_EFF_SDP_ENABLED=0`, `..._MATH_SDP_ENABLED=1` on RDNA1/2 | `:46-48` | through the torch API in `configure_torch_backends` (`hardware.py:220-232`) | That function runs where it is called. NOT CHECKED: whether the training child processes (`training.train`, `training.cache`) call it. If they do not, RDNA1/2 cards train with flash / mem-efficient SDPA enabled. Setting the three variables in `apply_runtime_env` for gfx101x / gfx103x would cover the children. |
| `MIOPEN_SYSTEM_DB_PATH`, `ROCBLAS_TENSILE_DB_PATH`, `ROCBLAS_TENSILE_LIBPATH` -> `_rocm_sdk_devel\bin`, `...\rocblas`, `...\rocblas\library`; skipped on `gfx1100` | `:60-65` | NOT SET | Missing for every AMD card except gfx1100. If the owner's 20 GB card is an RX 7900 XT it is gfx1100, where upstream skips them too: no effect there. Other AMD users (RDNA4, 7800 / 7700 / 7600, APUs) do not get the MIOpen / rocBLAS database paths upstream gives them. MIOpen serves convolutions, i.e. the SDXL UNet and every VAE. Add. |
| `ROCBLAS_USE_HIPBLASLT_BATCHED` cleared | `:67` | cleared (`hardware.py:77`) | none |
| `DISABLE_CUDA=1` (install time, for `hqq`) | `install_fizgig_rocm.bat:211` | n/a (no hqq) | needed if hqq is added |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (non-Windows, in the trainer) | `families/train.py:1623-1625` | `training/train.py:678-680` | none |
| `KMP_BLOCKTIME=0`, `OMP_WAIT_POLICY=PASSIVE` | `families/train.py:1626-1627` | `training/train.py:681-682` | none |
| MSVC environment for `torch.compile` (vcvars64 sourced into the process) | `families/compile.py:60-95` | `training/families/krea2/compile.py` (not compared line by line) | not checked |
| `HF_HUB_DISABLE_TELEMETRY=1` | - | `hardware.py:83` | TagScribeR only |

### C.4 Other points relevant to AMD ROCm on Windows

1. **`nvidia-ml-py` on AMD.** Upstream strips it from the ROCm install (`filter_requirements_rocm.py:24`) and reads
   VRAM through `typeperf` on Windows ROCm (`docs/INSTALL.md:134`, `utils/vram_monitor.read_amd_gpu_vram`).
   TagScribeR installs `nvidia-ml-py` on every backend (`requirements.txt:52`). Harmless. Not checked: which reader
   TagScribeR's memory bar uses on ROCm.
2. **Linux ROCm fast exit.** `src/fizgig/rocm/cache_exit.py`: on Linux with `FIZGIG_GPU_BACKEND=rocm` the cache
   scripts call `os._exit(0)` after success to skip HIP teardown (opt out: `FIZGIG_ROCM_NO_FAST_EXIT=1`). Linux only;
   TagScribeR's `training/cache.py` has no equivalent.
3. **Data-centre targets.** Upstream installs gfx942 / gfx950 from
   `https://rocm.nightlies.amd.com/v2-staging/gfx942-dcgpu/` (and `gfx950-dcgpu/`) (`install_fizgig_rocm.bat:151-178`);
   TagScribeR's detector (`tools/install.py:66-90`) does not know those ids.
4. **GPU detection.** Upstream uses `detect_gpu.py` (332 lines, from comfyui-rocm, GPL-3.0); TagScribeR its own PCI /
   name table. Not diffed entry by entry.
5. **InsightFace models** are pre-downloaded by upstream's installer (CPU provider). Needed only for the face tools.
6. **No ROCm-specific model fixes upstream.** `fizgig-src/src/fizgig/rocm/` holds one file (the Linux fast exit).
   TagScribeR's ROCm fixes (`training/modules/wide_attention.py`, the SDPA backend selection in
   `training/modules/sdpa.py:71-120`) are its own: upstream neither provides nor conflicts with them.

### C.5 A finding outside the brief that the owner must see

Release notes 6.8.0: "**Base precision: fp8 is gone.** Auto picks INT8, then NF4" for Krea 2
(`fizgig-src/docs/RELEASE_NOTES_v6.8.0.md:13`). `families/krea2.py:91-92` now has
`precisions=("int8", "nf4", "bf16")`, `auto_precisions=("int8", "nf4")`, and `families/launch.py:43-45` has no fp8
label. TagScribeR offers `fp8` (`training/params.py:20-22`) and the owner trains Krea 2 on an fp8-scaled RAW file.
Mirroring upstream here would NARROW TagScribeR and break the owner's own workflow: keep TagScribeR's fp8 base.
Not checked: whether upstream's Krea 2 driver still accepts an fp8-scaled file under another choice, the way Klein's
does ("As the file (bf16 or fp8)", `families/klein.py:135`). This belongs to the Krea 2 audit.

---

## Part B - GUI settings keys and presets, all families

### B.0 What changed in the shape of upstream's GUI

* `ARCHITECTURES = {}` (`fizgig-src/lora_trainer_gui.py:372`): there is no hand-written architecture table any more.
  Every training family is a `FamilyDescription` and gets its entry from `architecture_entry()`
  (`lora_trainer_gui.py:427-430`; `families/description.py:558-593`). The one hand-written entry left is
  `"MiniMax H3 RefMod"` (`:381-411`).
* Klein and MiniMax H3 moved to the driver system in 7.0 (`docs/RELEASE_NOTES_v7.0.0.md:12`), Krea 2 in 6.8.0
  (`RELEASE_NOTES_v6.8.0.md:5-16`). There are no `BUILT_IN_PRESETS` / `KREA2_BUILT_IN_PRESETS` /
  `MINIMAX_BUILT_IN_PRESETS` tables left in the GUI: built-ins are `desc.presets`
  (`lora_trainer_gui.py:4873-4878`). Only `REFMOD_BUILT_IN_PRESETS` (`:614`) remains.
* One command builder for every family: `families/launch.py` (`train_command` `:436-589`, `cache_command`
  `:406-416`, `_preview_flags` `:592-691`, `dataset_toml` `:713-753`). The old per-family builders are gone.
* A family's own controls are `FamilyOption` objects (`families/description.py:139-219`) whose values become launch
  tokens; only MiniMax H3 (23 options, `families/minimax.py:54-170`) and SDXL (2, `families/sdxl.py:188-193`) declare
  any. Their keys travel in presets (`lora_trainer_gui.py:6208-6213`).
* Line references in TagScribeR that point into the old GUI (e.g. `training/params.py:6`
  "lora_trainer_gui.py:2083-2211", `:158`, `:198`, `:232`; `training/presets.py:73, 93`) no longer match upstream.

### B.1 Every training-related settings key upstream has now

Sources: the default settings dict `lora_trainer_gui.py:1429-1508`; the widget keys (`self.entries[...]`, grep);
`_FAMILY_ENTRY_KEYS` `:5832-5840`; `_collect_preset_values` `:6172-6236`; `_NON_TRAINING_ENTRY_KEYS` `:5822-5829`;
`families/launch.py`; the family files. "Sent" = `launch.train_command` puts it on the command line.
"Shown for" uses the description flags: K = Klein, R = Krea 2, Q = Qwen Image 2.1, H = MiniMax H3, S = SDXL, A = Anima.
TagScribeR column: the entry in `training/params.py` (line), "-" = no such key.

**Core recipe**

| Key | Upstream default | Shown for | Sent as | TagScribeR |
|---|---|---|---|---|
| `LORA_OUTPUT_DIR` | `output_loras` in Fizgig, remembered per family (`:1438, 1519-1527`) | all | `--output_dir` | yes `:62`, default `""` (= `user_data/training_runs`) |
| `LORA_NAME` | `"LoraName_TokenName_k9b"` | all | `--output_name` | yes `:65`, default `"my_lora"` |
| `LEARNING_RATE` | `4e-4` | all | `--learning_rate` | yes `:69`, same |
| `NETWORK_DIM` | `4` | all | `--network_dim` | yes `:85`, same |
| `NETWORK_ALPHA` | `4` | all | `--network_alpha` | yes `:88`, same |
| `NETWORK_TYPE` | `"LoRA (standard)"` | all six offer LoKR (`network_types=("lora","lokr")` in every family file) | `--network_type lokr --lokr_factor N` (`launch.py:530-532`), never for a slider | yes `:82`; LoKR offered only where the family's `network_types` has it - NOT for TagScribeR's SDXL or Anima |
| `LOKR_FACTOR` | `8` | with LoKR | `--lokr_factor` | yes `:90`, same |
| `MAX_TRAIN_EPOCHS` | `12` | all | `--max_train_epochs` | yes `:93`, same |
| `SAVE_EVERY_N_EPOCHS` | `1` | all | `--save_every_n_epochs` | yes `:96`, same |
| `SEED` | `42` | all | `--seed` | yes `:99`, same |
| `ADAPTIVE_LR` | `False` | all except S and H (`adaptive_lr=False`: `families/sdxl.py:123`, `minimax.py:312`) | `--adaptive_lr` only when `desc.adaptive_lr` (`launch.py:456`) | yes `:72`, same default; shown for EVERY family incl. SDXL |
| `ADAPTIVE_LR_MIN` | `"1e-5"`; options `1e-5, 5e-5, 1e-4, 2e-4 - rank 4/8 only, 3e-4 - low-rank only` (`:3876`) | as above | `--adaptive_lr_min` | yes `:77`, same options |
| `ADAPTIVE_LR_MAX` | `"4e-4"` | as above | `--adaptive_lr_max` | yes `:80`, same |
| `LR_SCHEDULER` | `"constant"`; `constant, constant_with_warmup, cosine, cosine_with_restarts, linear, polynomial` (`:3158`) | all | `--lr_scheduler` when not constant and Adaptive LR is off (`launch.py:460-463`) | yes `:144`, same |
| `LR_WARMUP_STEPS` | `""` | all | `--lr_warmup_steps` when > 0 | yes `:147`, default `0` (INT). A Fizgig preset holding `""` is REFUSED by `coerce` (`params.py:381-382`) with a message; harmless but noisy. See B.4. |
| `OPTIMIZER_TYPE` | `"adamw8bit"` | all; per family: K, R `("adamw8bit","adamw")`; Q, S, A `("adamw","adamw8bit")`; H `("automagic3","adamw8bit","adamw")` | `--optimizer_type` | yes `:133`; TagScribeR's families list more optimizers than upstream's |
| `OPTIMIZER_ARGS` | `""` | all | `--optimizer_args` | yes `:136` |
| `GRADIENT_ACCUMULATION` | `1` | all | `--gradient_accumulation_steps` when > 1 | yes `:138` |
| `MAX_GRAD_NORM` | `1.0` | all | `--max_grad_norm` when not 1.0 | yes `:141` |
| `CONTEXT_LORA_PATH` / `CONTEXT_LORA_STRENGTH` | `""` / `"1.0"` | all; hidden and not sent for a fine-tune (`launch.py:452-455`) | `--context_lora_path`, `--context_lora_strength` | yes `:108, 111` |
| `MIN_TIMESTEP` / `MAX_TIMESTEP` | `""` / `""`; units are 0-1000 in the Training tab | ONLY a family with `train_areas` = Klein (`launch.py:566-567`, `timestep_flags` `:120-129` divides by 1000) | `--min_timestep`, `--max_timestep` (0-1) | yes `:175, 178`, units 0-1, shown for every family. `migrate_legacy` divides values above 1.0 by 1000 (`presets.py:186-187`) |
| `BLOCKS_SWAP` | `"auto"` in the dict; label `"Auto (detect from GPU)"` in presets | all | `--blocks_to_swap -1` for Auto, else the number (`launch.py:533-539`) | yes `:155`, default `"Auto (detect from GPU)"` |
| `COMPILE_BLOCKS` | `"auto"`; `auto, on, off, outside` | families with `compiles=True`: K, R, Q, S, A (not H) | `--compile_blocks` (`launch.py:469-472`) | yes `:159`, but `family_only="option"` and described as "Krea 2 only" |
| `SAVE_STATE`, `SAVE_STATE_ON_TRAIN_END`, `KEEP_LAST_N_STATES` | `True`, `True`, `2` | all | `--save_state`, `--save_state_on_train_end`, `--keep_last_n_states` (`launch.py:419-433`) | yes `:168, 170, 172`, same |
| `RESUME_TRAINING` | `""` (run-only: in `_NON_TRAINING_ENTRY_KEYS`) | all; ignored for a fine-tune | `--resume` | NOT a key in `params.py` (resume may live elsewhere in the Train tab; not checked) |

**Standard-layer family keys** (`_FAMILY_ENTRY_KEYS`)

| Key | Upstream default | Shown for | Sent as | TagScribeR |
|---|---|---|---|---|
| `FAMILY_PRECISION` | first label; `auto / bf16 / int8 / nf4 / hqq` labels in `launch.py:43-45` | families with more than one precision: all six | `--precision` (`launch.py:483-487`) | yes `:151`. Labels differ: TagScribeR has `fp8` and no `hqq`. Klein's `"As the file (bf16 or fp8)"` and H3's `"int8 · most accurate, ..."` do not match any TagScribeR option by first token and the key is strict: REFUSED (B.4). |
| `FAMILY_EMA` | the family's `ema_default`: `"0.98"` (R, Q, H, S, A), `"Off"` (K) | all | `--ema_decay` unless Off (`launch.py:540-543`) | yes `:104`. H3 also offers `"Short run (window = ¼ of the run)"` (`lora_trainer_gui.py:5842`, `ema_short_run`): not in TagScribeR's `EMA_OPTIONS`; the key is not strict, so the string is stored as is and `training/pipeline.py:121` takes its first token, `"Short"`. NOT CHECKED what happens next. |
| `FAMILY_TRAINING_ADAPTER` | `True` | Q (`training_adapter="qwen21_training_adapter"`); H3 uses the `H3_ADAPTER` option instead | `--training_adapter <file>` | yes `:101` |
| `FAMILY_EDIT`, `FAMILY_EDIT_DIR`, `FAMILY_EDIT_REF`, `FAMILY_EDIT_CAPTION` | `False`, `""`, `""`, `""` | K, Q (`edit_training=True`) | control folder in the dataset TOML; `--sample_reference` | yes `:113, 116, 318, 321` |
| `FAMILY_SLIDER` | `False` | all six (`slider_training=True`) | `--slider_pairs` or `--slider_prompts ...` (`launch.py:568-586`) | **-** |
| `FAMILY_SLIDER_SOURCE` | `"pairs"` (`pairs` / `prompts`) | all six | selects the branch | **-** |
| `FAMILY_SLIDER_DIR`, `FAMILY_SLIDER_CAPTION` | `""` | all six | control folder; the slider preview prompt | **-** |
| `FAMILY_SLIDER_BASE`, `FAMILY_SLIDER_POS`, `FAMILY_SLIDER_NEG` | `""` | all six | `--slider_prompts base "base pos" "base neg"` | **-** |
| `FAMILY_SLIDER_GUIDANCE` | the family's `slider_guidance` (2.0) | all six | `--slider_guidance` | **-** |
| `FAMILY_SLIDER_ULTRA` | `False` | R (`slider_ultra_blocks`, `families/krea2.py:139`) | `--train_blocks <ultra blocks>` | **-** |
| `FAMILY_FAST_ID` | `False` | Q (`identity_blocks` = blocks 10-14, `families/qwen_image.py:107`) | `--train_blocks block_10,...,block_14` (`launch.py:559-561`) | **-** |
| `FAMILY_FT` | `False` | all six (`finetune=True`) | `--finetune` | **-** |
| `FAMILY_FT_ROTATIONS` | `"10"` | all six | `--ft_rotations` | **-** |
| `FAMILY_FT_SAVE_EVERY` | `"1"` | all six | `--ft_save_every_rotations` | **-** |
| `FAMILY_FT_ROTATE_EVERY` | `"1"` | all six | `--ft_rotate_every` | **-** |
| `FAMILY_FT_FUSED` | `True` | all six | `--ft_fused_backward` | **-** |
| `FAMILY_FT_REG_DIR`, `FAMILY_FT_REG_MULT` | `""`, `0.2` (`launch.py:96-100`) | all six | a reg dataset block; `--reg_lr_multiplier` | **-** |
| `FAMILY_FT_MAX_PARTS` | `"Auto - as few windows as fit"`; choices `launch.py:207-208` | all six | `--ft_max_parts` | **-** |
| `FAMILY_TRAIN_AREA`, `FAMILY_TRAIN_BLOCKS` | launch-input names for `TARGET_LAYERS` / the ticked blocks (`lora_trainer_gui.py:6100-6102`) | K | `--train_blocks` (`launch.py:562-565`) | as `TARGET_LAYERS` / `TRAINING_BLOCKS` below |
| `TARGET_LAYERS` (preset key) | `"Full Model"`; Klein areas `Full Model, Identity, Style, Style+Composition, Details` + `Custom` (`families/klein.py:106-112`) | K | via the area | yes `:200`, same six values |
| `TRAINING_BLOCKS` (preset key) | `{block id: bool}` with ids `double_N` / `single_N` (`fizgig-src/src/fizgig/klein/driver.py:412-415`) | K, area Custom | `--train_blocks` | yes `:206`, but TagScribeR's ids are `double_blocks.N` / `single_blocks.N` (`training/families/klein/driver.py:94-98`): a current Fizgig Custom preset imports as `"double_0, single_5"` and the driver raises "unknown block(s)". See B.4. |
| `FAMILY_MULTICONCEPT`, `MINIMAX_CONCEPT_DIRS` | `False`, list | H (`multi_concept=True`) | extra `[[datasets]]` blocks | **-** (image-only H3) |
| `FAMILY_OPTIONS` | launch-input dict of the `FamilyOption` values | H, S | tokens | TagScribeR uses `DRIVER_OPTIONS` (`params.py:329-337`) |

**Loss watch**

| Key | Upstream default | Shown for | Sent as | TagScribeR |
|---|---|---|---|---|
| `KREA2_LOSS_WATCH` | `False` | all except H (`loss_watch=False`, `minimax.py:314`) | `--log_per_image_loss` (batch size 1 only) | yes `:120` |
| `KREA2_PER_IMAGE_LR` | `False` | as above | `--per_image_lr` | yes `:123` |
| `KREA2_AUTO_RECAPTION` | `False` | as above | `--auto_recaption --captioner ... [--trigger_word] [--recaption_instruction...]` | yes `:126` |
| `KREA2_WARMUP_LOOK` | `False` | as above | `--warmup_look_outliers` | yes `:129` |

**Dataset**

| Key | Upstream default | Shown for | TagScribeR |
|---|---|---|---|
| `DATASET_MEGAPIXELS` | `"0.25"`; `0.25 ... 4.2` (`:1358, 4007`) | all | yes `:249`, same list |
| `CLIP_MEGAPIXELS` | `"0.25"` (`:1360`) | H with clips | **-** |
| `DATASET_BATCH_SIZE` | `"1"` | all | yes `:253` |
| `DATASET_CAPTION_EXT` | `".txt"` | all | yes `:256` |
| `ENABLE_BUCKET`, `BUCKET_NO_UPSCALE` | `True`, `True` | all | yes `:258, 260` |
| `ENABLE_CACHE` | `True` (`:4749`) | all | **-** (TagScribeR always runs the cache stages, skipping unchanged items) |
| `GRADIENT_MINING` | a var grabbed into presets (`:6231`); not sent by `launch.py` | - | **-** |

**Metadata** `METADATA_TITLE`, `METADATA_AUTHOR`, `METADATA_DESCRIPTION`, `METADATA_LICENSE`, `METADATA_TAGS`,
`METADATA_TRIGGER_PHRASE`, `METADATA_THUMBNAIL`: all `""` upstream, all sent (`launch.py:548-557`), all in
TagScribeR (`params.py:276-289`). Same.

**Samples (run-only upstream: `_NON_TRAINING_ENTRY_KEYS`)**

| Key | Upstream default | TagScribeR |
|---|---|---|
| `SAMPLE_ENABLED` | `True` | yes `:291` |
| `SAMPLE_PROMPT` | `"A high quality photo"` | yes `:294` |
| `SAMPLE_WIDTH` / `SAMPLE_HEIGHT` | `768` / `768` (the family's `preview_width/height` when blank) | yes `:296, 298`, default `0` = family default |
| `SAMPLE_STEPS` | `40` | yes `:300`, default `0` = family default |
| `SAMPLE_SEED` | `1234` | yes `:302` |
| `SAMPLE_EVERY_N_EPOCHS` | `1` | yes `:304` |
| `SAMPLE_EVERY_N_STEPS` | `0` | **-** |
| `SAMPLE_AT_FIRST` | `True` | yes `:306` |
| `SAMPLE_CFG_SCALE` | `1.0` | yes `:308`, default `0.0` = family default |
| `SAMPLE_NEGATIVE` | the long general negative; now PER FAMILY (`preview_negative`, "Each model has its own default and remembers your edits", `RELEASE_NOTES_v7.0.0.md:55`) | yes `:311`, one global default; TagScribeR's description has `preview_negative` too |
| `SAMPLE_FLOW_SHIFT` | `""` | **-** |
| `SAMPLE_FRAMES` | H3 preview length (now the `H3_SAMPLE_FRAMES` option) | **-** |
| `CACHE_SAMPLE_MODEL` | `"auto"` (Klein's Distilled preview model in RAM) | **-** |
| `FAMILY_TURBO_STRENGTH` | per family (`last_used["turbo_strengths"]`); never written by a preset (`:4930-4933`) | yes `:315`, `preset=False` |
| `FAMILY_TURBO_STEPS`, `FAMILY_TURBO_PACE` | `6`, `75` (H3's "N steps at M%" row) | **-** |

**MiniMax H3 option keys** (`families/minimax.py:54-170`; each names the old key it replaces in `setting=`)

| New key | Old key (`setting=`) | Default in the built-in presets (`minimax.py:28-41`) | TagScribeR |
|---|---|---|---|
| `H3_TRAIN_BASE` | `MINIMAX_TRAIN_BASE` | first choice (fl2va); `tab="model"`, never in a preset | - |
| `H3_STRUCTURE` | `MINIMAX_LOWNOISE_PCT` | first structure choice | - |
| `H3_LOWNOISE_PCT` | `MINIMAX_LOWNOISE_PCT` | `"60"` | has `MINIMAX_LOWNOISE_PCT` (`params.py:234`), default 60.0 |
| `H3_HIGHNOISE_LR_PCT` | `MINIMAX_HIGHNOISE_LR_PCT` | `"100"` | ignored (`presets.py:110`) |
| `H3_MIXED_STOP_CATEGORY`, `H3_MIXED_STOP_EPOCH`, `H3_MIXED_STOP_MODE` | `MIXED_STOP_*` | - | - |
| `H3_LIKENESS_MODE` | `MINIMAX_LIKENESS_MODE` | `"Default"`; choices `Default`, `More Blocks`, `Off · hand-pick the blocks in Other Options` | has `MINIMAX_LIKENESS_MODE` (`params.py:239`); third label is `"Off - hand-pick the blocks below"` |
| `H3_ADAPTER` | `MINIMAX_ADAPTER` | `"Circlestone — best for photos"` | migrated to `FAMILY_TRAINING_ADAPTER` (`presets.py:125-131`) - from the OLD key only |
| `H3_TREAD` | `MINIMAX_TREAD` | `"1"` | ignored |
| `H3_CLIP_STILL` | `MINIMAX_CLIP_STILL` | `"1"` / `""` | ignored |
| `H3_ADAPTER_RAMP` | `MINIMAX_ADAPTER_RAMP` | `"Off"` | ignored |
| `H3_CAPTION_DROPOUT` | `MINIMAX_CAPTION_DROPOUT` | `"0.05 (default)"` | migrated to `CAPTION_DROPOUT` - from the OLD key only |
| `H3_BLOCKS` | `MINIMAX_BLOCKS` | `"all"` | has `MINIMAX_BLOCKS` (`params.py:244`) |
| `H3_DISTILL`, `H3_DISTILL_WEIGHT`, `H3_DISTILL_REFS`, `H3_DISTILL_PHASE1` | `MINIMAX_DISTILL*` | `""` | ignored with a note - OLD keys only |
| `H3_TRAIN_REFINER` | `MINIMAX_TRAIN_REFINER` | `""` | ignored with a note - OLD key only |
| `H3_SAMPLE_FRAMES` | `SAMPLE_FRAMES` | Samples tab | - |
| `H3_FT_SCOPE`, `H3_FT_BLOCKS` | `MINIMAX_FT_SCOPE`, `MINIMAX_FT_BLOCKSPEC` | fine-tune only | - |
| `AUDIO_VAE` | - | fixed | - |

Upstream also keeps `settings_aliases` for H3 (`minimax.py:286-288`): `FAMILY_EMA <- MINIMAX_EMA`,
`FAMILY_PRECISION <- MINIMAX_BASE_QUANT`, `FAMILY_FT <- MINIMAX_FINETUNE`, `FAMILY_FT_ROTATE_EVERY <- MINIMAX_FT_EVERY`,
`FAMILY_FT_FUSED <- MINIMAX_FT_FUSED`, `FAMILY_FT_REG_DIR <- MINIMAX_REG_DIR`, `FAMILY_FT_REG_MULT <- MINIMAX_REG_MULT`.

**SDXL option keys** (`families/sdxl.py:188-193`)

| Upstream key | Kind / values | TagScribeR |
|---|---|---|
| `SDXL_MIN_SNR` | choice: `"off"` -> `min_snr=0`, `"γ 5"` -> `min_snr=5` | DIFFERENT KEY: `SDXL_MIN_SNR_GAMMA`, a float 0-20 (`params.py:183`). An upstream SDXL preset's `SDXL_MIN_SNR` is ignored. |
| `SDXL_NOISE_OFFSET` | entry, default `"0"` -> `noise_offset={}` | same key (`params.py:188`), float. `"0"` and `"0.0357"` coerce. Same. |
| - | - | `SDXL_LOCON` (`params.py:193`): TagScribeR only |

**Keys still in upstream's GUI that `launch.py` no longer sends** (widgets or dict entries left from the old Klein
trainer; grep of `families/launch.py` finds no use): `LORA_LR_RATIO` (default `1`; only named in the number checks),
`ATTENTION_MECHANISM` (`"sdpa"`), `LOGGING_DIR`, `LOG_WITH` (`"none"`), `LOG_PREFIX`, `IMG_IN_TXT_IN_OFFLOADING`
(`False`), `MODEL_TYPE`, `QUANT_4BIT` (`False`), `GRADIENT_CHECKPOINTING` (`True`), `FP8_TEXT_ENCODER` (`True`),
`VAE_MODEL`, `CLIP_MODEL`, `T5_MODEL`, `TEXT_ENCODER`, `DIT_MODEL`. NOT CHECKED: whether any of them is read by a
code path outside `launch.py` (the GUI is 30k lines; only the launch plan was traced).

### B.2 Keys TagScribeR has that upstream no longer has

Zero hits for the key name anywhere in `fizgig-src/lora_trainer_gui.py` and `fizgig-src/src/fizgig`:

| TagScribeR key (`params.py`) | What it was upstream | Status upstream now |
|---|---|---|
| `TIMESTEP_SAMPLING` (`:210`) | Klein's timestep sampler choice | gone (0 hits). The Klein driver decides. |
| `DISCRETE_FLOW_SHIFT` (`:215`), `SIGMOID_SCALE` (`:218`), `LOGIT_MEAN` (`:222`), `LOGIT_STD` (`:224`) | Klein sampler knobs | gone (0 hits) |
| `PRESERVE_DISTRIBUTION` (`:229`) | Klein noise-range mode | gone as a setting (1 hit, a code comment at `lora_trainer_gui.py:4976`) |
| `MINIMAX_LOWNOISE_PCT`, `MINIMAX_LIKENESS_MODE`, `MINIMAX_BLOCKS` (`:234-247`) | H3 dials | RENAMED to `H3_LOWNOISE_PCT`, `H3_LIKENESS_MODE`, `H3_BLOCKS`; the old names survive only as `setting=` aliases |
| `ATTENTION_MECHANISM` (`:226`) | Klein attention backend | still a widget upstream, no longer sent |
| `SDXL_MIN_SNR_GAMMA` (`:183`), `SDXL_LOCON` (`:193`) | TagScribeR's own | upstream's key is `SDXL_MIN_SNR`; no LoCon |
| `DATASET_REPEATS` (`:262`) | - | 0 hits (upstream writes `num_repeats = 1`, `launch.py:726`) |
| `CAPTION_SHUFFLE_VARIANTS`, `CAPTION_KEEP_TOKENS`, `CAPTION_DROPOUT` (`:265-274`) | TagScribeR extension (approved) | upstream has caption dropout only as `H3_CAPTION_DROPOUT` |

Also gone upstream (TagScribeR already treats them as legacy in `presets.py:69-74, 159-170`): `KREA2_EMA`,
`QUANT_4BIT_MODE`, `FP8`, `NETWORK_DROPOUT`, `WEIGHTING_SCHEME`, `MODE_SCALE`, `LR_DECAY_STEPS`, `KREA2_FINETUNE*`
(0 hits each; `"SCALED"` 1 hit in `src`, not traced).

Per the owner's rule none of these should be REMOVED from TagScribeR because upstream dropped them (the Klein
sampler knobs and fp8 are capabilities). They are listed so that nobody "re-syncs" them away.

### B.3 Features upstream's shared layer has that TagScribeR's does not (the source of most missing keys)

| Feature | Upstream | Keys | TagScribeR |
|---|---|---|---|
| Slider LoRAs (pairs or prompts) | `families/train.py` (`--slider_pairs`, `--slider_prompts`, `--slider_guidance`, `--slider_bank_res`), driver `noise_latents` / `predict` / `diff_ref`; all six families | `FAMILY_SLIDER*` (9 keys) | none |
| Full fine-tune with rotation windows | `families/ft.py` (775 lines), driver `ft_spec`; all six | `FAMILY_FT*` (8 keys) | none (`presets.py:66-74` ignores the old Krea 2 fine-tune keys) |
| Fast Identity Mode | `identity_blocks` (Qwen) | `FAMILY_FAST_ID` | none |
| Model Area to Train as a description field | `train_areas` (Klein) | `TARGET_LAYERS`, `TRAINING_BLOCKS`, timestep window | TagScribeR has Klein's area as driver options |
| torch.compile for every family | `families/compile.py` (220 lines), `compiles`, `compile_payback_steps`, `compile_boundary`, `compile_memory` | `COMPILE_BLOCKS` | Krea 2 only (`training/families/krea2/compile.py`) |
| Preview checkpoint (Klein Distilled previews) | `train_preview_checkpoint`, `--preview_checkpoint`, `--preview_checkpoint_cache`, `--preview_int8` | Samples tab | not checked in TagScribeR's Klein |
| FamilyOption system | `description.py:139-219`, `launch.option_tokens` | per family | `family_options` + `DRIVER_OPTIONS` (simpler; no `requires`, `mode`, `show_if_media`) |
| Multi Concept, clips, voice | H3 | `FAMILY_MULTICONCEPT`, `CLIP_MEGAPIXELS`, `H3_TREAD`, ... | image-only H3 |
| HQQ 4-bit | `precisions=("int8","nf4","hqq")` (H3) | `FAMILY_PRECISION` | none |
| EMA "Short run" | `ema_short_run` (H3) | `FAMILY_EMA` | none |
| Per-family sample negative, CFG note, sampler text | `preview_negative: Optional[str]` (None greys the box), `preview_cfg_note`, `samples_text`, `samples_cfg_free` | Samples tab | `preview_negative: str` only |
| Regularisation dataset for a fine-tune | `is_reg = true` block (`launch.py:747-752`) | `FAMILY_FT_REG_DIR` | none |
| Description fields TagScribeR's `FamilyDescription` lacks | `hidden`, `inside` (TagScribeR: `default_to`), `auto_precisions` (TagScribeR: `auto_order`), `precision_labels`, `precision_hint`, `compiles` + 5 compile fields, `finetune`, `ft_learning_rate`, `slider_training`, `slider_guidance`, `slider_ultra_blocks`, `identity_blocks`, `train_areas`, `adaptive_lr`, `adaptive_lr_clip_signal`, `loss_watch`, `optimizer_weight_decay`, `optimizer_eps_floor_8bit`, `trainable_dtype`, `optimizer_families`, `automagic_sign_window`, `ema_short_run`, `media`, `clip_spec`, `multi_concept`, `options`, `settings_aliases`, `extract_presets`, `repair_presets`, `block_categories`, `category_masters`, `workbench_engine`, `int8_attention`, `activation_cache`, `workbench_follows_samples`, `repair_size`, `preview_checkpoint_sampling`, `train_preview_checkpoint`, `preview_image`, `preview_cfg_note`, `samples_text`, `LoRAFormat.lokr_kohya_stems` | | compare `families/description.py:222-460` with `training/description.py:92-179` |

Three of those description flags are BEHAVIOUR, not GUI, and are worth checking against TagScribeR's ports in their
own audits: `adaptive_lr_clip_signal` (Klein), `optimizer_eps_floor_8bit` and `trainable_dtype="bf16"` (H3),
`optimizer_families` / `automagic_sign_window` (Automagic v3 per-group learning rates).

### B.4 Would a current Fizgig preset file still import into TagScribeR?

`TrainingPresets.import_file` (`training/presets.py:284-297`) copies the JSON verbatim and requires only that one key
is known, so every current Fizgig preset IMPORTS. What matters is `apply` (`:193-221`) + `migrate_legacy`
(`:154-190`). Read from the code, not executed:

| Case | What happens | Verdict |
|---|---|---|
| Core keys (rank, alpha, LR, epochs, seed, optimizer, EMA, Adaptive LR, megapixels, loss watch, states, metadata) | applied unchanged | OK |
| `COMPILE_BLOCKS: "Auto"/"On"/"Off"/"Outside"` | `_legacy_compile` (`:91-96`) | OK |
| `MIN_TIMESTEP` / `MAX_TIMESTEP`: `""` -> defaults; `"400"` -> 0.4 (`:184-187`) | OK for upstream's Klein Style preset (`("0", "400")`, `families/klein.py:200`). Edge: a value of exactly `"1"` in upstream's 0-1000 units (0.001) is read as 1.0 = full range. | OK, one edge |
| `TARGET_LAYERS` | current names equal TagScribeR's options | OK |
| `TRAINING_BLOCKS: {"double_0": true, ...}` | joined to `"double_0, ..."` (`:182-183`); TagScribeR's Klein driver accepts only `double_blocks.N` / `single_blocks.N` and raises | **BROKEN for Klein Custom-area presets.** Map `double_N` -> `double_blocks.N`, `single_N` -> `single_blocks.N` in `migrate_legacy`. |
| `FAMILY_PRECISION: "As the file (bf16 or fp8)"` (Klein) | no option matches by first token; strict -> refused, current value kept, message shown | **Wrong for Klein**: should map to bf16 (or fp8 when the file is fp8). |
| `FAMILY_PRECISION: "int8 · most accurate, needs ~30 GB free"` (H3, `minimax.py:360`) | first token `int8` vs option `INT8 (8-bit, fastest)`: `match_option` is case-sensitive (`params.py:362-372`) -> refused | **Wrong for H3**: an INT8 preset silently stays on the current precision. `_migrate_minimax` handles this only for the OLD key `MINIMAX_BASE_QUANT`. |
| `FAMILY_PRECISION: "4-bit HQQ ..."` | first token `4-bit` matches `4-bit NF4 (smallest)` -> NF4 with NO note | silent substitution |
| `FAMILY_PRECISION: "Auto (recommended)"` (H3 presets) | first token `Auto` matches | OK |
| `FAMILY_EMA: "Short run (window = ¼ of the run)"` | stored as is (not strict) | NOT CHECKED downstream; likely an error or EMA off |
| `H3_*` keys (17 in every H3 built-in, `minimax.py:36-40`) | not in `P.BY_KEY`, do not start with `MINIMAX_` -> `rep.ignored` | **LOST**: `H3_LOWNOISE_PCT`, `H3_LIKENESS_MODE`, `H3_BLOCKS`, `H3_ADAPTER`, `H3_CAPTION_DROPOUT` carry real settings TagScribeR supports under the old names. Add an `H3_* -> MINIMAX_*` step before `_migrate_minimax` (upstream's own `setting=` table gives the mapping), and map `H3_LIKENESS_MODE`'s third label (`"Off · hand-pick ..."` vs TagScribeR's `"Off - hand-pick the blocks below"`; first-token `Off` matches, since that key is not strict it is stored verbatim if no match - check). |
| `SDXL_MIN_SNR: "γ 5"` | unknown key -> ignored | LOST: map to `SDXL_MIN_SNR_GAMMA = 5.0` (`"off"` -> 0). |
| `SDXL_NOISE_OFFSET: "0.0357"` | float coerce | OK |
| `FAMILY_SLIDER*`, `FAMILY_FT*`, `FAMILY_FAST_ID`, `FAMILY_SLIDER_ULTRA`, `FAMILY_MULTICONCEPT`, `CLIP_MEGAPIXELS`, `ENABLE_CACHE`, `GRADIENT_MINING`, `LORA_LR_RATIO`, `LOG_*`, `MODEL_TYPE` | unknown or legacy -> ignored silently | **DANGEROUS for `FAMILY_SLIDER: true` and `FAMILY_FT: true`**: upstream's Slider and Fine-tune presets (`✨ SDXL Slider ...`, `✨ SDXL Fine-tune (1e-5)`, `✨ Anima Fine-tune Official (1e-6)`, `✨ Krea 2 Slider ...`, `✨ Qwen 2.1 Slider ...`, `✨ MiniMax H3 Slider ...`) would import and run as an ORDINARY LoRA at a slider's or a fine-tune's learning rate (e.g. 1e-6), with no warning. `migrate_legacy` should add a note - or refuse the preset - when `FAMILY_SLIDER`, `FAMILY_FT` or `FAMILY_FAST_ID` is truthy. |
| `NETWORK_TYPE: "LoKR (Kronecker)"` on SDXL / Anima | strict; TagScribeR's SDXL / Anima offer only LoRA -> refused with a message | honest, but a narrowing (A.1, A.2) |
| `OPTIMIZER_TYPE: "adamw"` | offered by every TagScribeR family | OK |
| `ADAPTIVE_LR: true` in an SDXL preset from an older Fizgig | applied (TagScribeR shows Adaptive LR for SDXL) | OK |
| `LR_WARMUP_STEPS: ""`, `KEEP_LAST_N_STATES: ""`, other blank number boxes | `coerce` raises -> refused with a `[preset]` line each | noisy, harmless. Treat `""` as "keep the default" for INT / FLOAT params. |
| Family: presets do not carry it (same on both sides) | a Fizgig `sdxl` preset can be applied to TagScribeR's `pony`, etc. | OK |

Preset FOLDERS: upstream keeps presets per architecture label under `presets/`; TagScribeR per family key under
`user_data/training_presets/<family>/`. Import is by file, so the folder layout does not matter. Not checked:
upstream's `get_preset_dir_for_architecture` naming.

**Most important missing settings keys, in order:** `FAMILY_SLIDER` (+8), `FAMILY_FT` (+7), the `H3_*` renames,
`FAMILY_FAST_ID`, `SDXL_MIN_SNR` (name mismatch), `COMPILE_BLOCKS` for families other than Krea 2, `CLIP_MEGAPIXELS` /
`FAMILY_MULTICONCEPT` (H3 video, out of the image-only scope), `SAMPLE_EVERY_N_STEPS`.

---

## Part D - Fizgig's workbench tools as they are now (a map only)

Every one of the six families declares `workbench=("repair", "explorer", "profiler", "extract", "royale")`
(`families/klein.py:102`, `krea2.py:113`, `qwen_image.py:249`, `minimax.py:289`, `sdxl.py:77`, `anima.py:64`). The
image families share ONE engine, `src/fizgig/families/workbench.py` (`WorkbenchEngine`, 745 lines), built on the
family driver and the family LoRA layer; a video family gets `families/video_workbench.py` (423 lines), and MiniMax H3
brings its own class (`workbench_engine="fizgig.minimax.workbench:H3WorkbenchEngine"`, `minimax.py:290`) over
`src/fizgig/repair_studio/h3_engine.py` (1826 lines). The tabs live in `lora_trainer_gui.py` (order at `:1595-1619`:
Profiler, Repair Studio, RefMod Studio, LoRA the Explorer, LoRA Royale, Extract, Metadata). Nothing below was read in
depth: file headers, tab docstrings, README lines 48-89 and the 6.8 / 7.0 release notes.

**LoRA Royale** (`src/fizgig/lora_royale/`: `scan.py` 70 lines, `export.py` 533, `prompt_travel.py` 425,
`likeness.py` 124; tab `create_lora_royale_tab`, `lora_trainer_gui.py:20040`). Finds a run's epoch checkpoints
(`<name>-NNNNNN.safetensors`, `scan.py`), renders every epoch on one seed and prompt through the workbench engine, and
crossfades between them to find the best epoch. It also does seed travel, prompt travel (interpolating the TEXT
EMBEDDING between waypoint words such as dawn -> night, `prompt_travel.py`) and strength travel, and exports the
result as a GIF or an H.264 MP4 with an epoch ticker (`export.py`; ffmpeg from `imageio-ffmpeg`). **Face likeness
compare** (`likeness.py`): a reference (training) photo and each rendered epoch are embedded with InsightFace ArcFace
(`buffalo_l` recognition model, CPU only, `ctx_id=-1`), the largest face of each is taken, and the cosine similarity
is the score (`score_renders(ref_path, renders)`, `likeness.py:116`). Needs `insightface` + `onnxruntime`. Families:
all six; H3 renders clips.

**Repair Studio** (`src/fizgig/repair_studio/`: `state.py` 189, `bake.py` 463, `metrics.py` 99, `h3_blocks.py` 69,
`h3_engine.py` 1826, `h3_render_cache.py` 319; `families/act_cache.py` 180; tab `:18047`). One slider per block of the
family's block map: switch a LoRA's blocks on or off or scale them, optionally blend a DONOR LoRA per block, and see
the render beside the baseline. `SliderState` (`state.py`) is the whole configuration; `save_repaired_lora`
(`bake.py`) bakes it into a new file (disabled blocks dropped, strengths baked into `lora_up`, a donor merged by rank
concatenation). "Turbo Preview" (`act_cache.py`) replays the unchanged blocks on step 1 so a slider move re-renders
only what follows it. `metrics.py` gives overbake measures on a same-seed pair (patch grid etc.). Per-family extras
come from the description: `repair_presets` (Krea 2 text-fusion x2 / x3, Qwen "Identity only" / "Look only" /
"Identity x0.5"), `block_categories` + `category_masters` (Klein's five category sliders; Qwen ID / Look colours),
`repair_size`, its own negative-prompt box for CFG families (SDXL, Anima). Families: all six.

**LoRA the Explorer** (tab `create_explorer_tab`, `lora_trainer_gui.py:14599`; no package of its own - it drives the
same workbench engine and `SliderState`). "Evolutionary LoRA discovery via human-guided selection": the app randomly
perturbs block strengths, shows four variants, the user picks one and it mutates again; a Structure control
stabilises composition and Freeze locks blocks already tuned. The result is a block configuration that can be baked
like a Repair Studio state. Families: all six.

**Profiler** (`src/fizgig/families/block_profile.py` 548 lines, `families/lorafile.py` 139; CLI
`src/fizgig/scripts/profile_lora.py` 41; tab `:17511`). Two instruments in one report. Weights (no model, instant):
every module's real update through its singular values, from two QR factorisations and an r x r SVD - each block's
size and how much of the LoRA sits in its top 1, 2, 4, 8 ... directions, i.e. how small Extract can make it.
Rendering (minutes, through the workbench engine): the LoRA whole, off, each group of neighbouring blocks alone and
left out (and, "Thorough", each single block left out, two seeds), scored as a share of what the whole LoRA does,
with likeness when photos are given. Modes: Weights only / Quick / Thorough (`RELEASE_NOTES_v6.8.3.md:5-11`). Reads
kohya, PEFT / diffusers and LoHa files. Families: all six (new for Klein and H3 in 7.0).

**RefMod Studio** (`src/fizgig/minimax/refmod.py`, `refmod_apply.py`, `reference.py`; CLI
`src/fizgig/scripts/minimax_refmod.py` 137; tab `create_refmod_studio_tab`, `:25860`; the training side is the
separate Base Model entry "MiniMax H3 RefMod", `lora_trainer_gui.py:381-411`, with its own
`REFMOD_BUILT_IN_PRESETS`, `:614`). A RefMod is a reference for H3 saved as a file (the community
`ComfyUI-MiniMaxH3Mod` format): the reference pictures run through the video VAE and stored as a latent. Fizgig also
OPTIMISES that latent against the frozen H3 base for a few hundred flow-matching steps ("textual inversion, in the
reference channel", `refmod.py:1-14`). The Studio tab lists mods, applies them, previews and acts on them.
Families: MiniMax H3 only.

**Face utilities** (`face_utils.py`, repo root, 441 lines). InsightFace on the CPU: `FaceDetector` (detection +
gender / age, `:26`), `crop_to_face` (`:229`), `draw_face_boxes` (`:287`), `FaceEmbedder` (ArcFace embeddings, `:344`),
`is_face_detection_available` (`:433`). Used by Image Prep (face-centred crops), the sample gallery's live likeness
score (README line 87) and, through the same `buffalo_l` models, Royale's likeness compare. Model-family independent
(it works on pictures, not on models). `src/fizgig/minimax/still_pick.py` (a clip's sharpest face frame) was not read.

**Gizmo** (`gizmo.py`, repo root, 5275 lines, `gizmo.pyw`, launcher `Launch Gizmo (Video clip prep tool).bat`). A
separate app: the clip prep tool for MiniMax H3. H3's clips must be exact (24 fps, a frame count on the 17n+5 grid,
width and height multiples of 32, 32 kHz stereo sound - the same numbers as `ClipSpec`, `families/description.py:123-136`)
and Fizgig refuses anything off-spec rather than converting it, so Gizmo produces them: cutting with scene detection,
cropping to the subject, and recording or segmenting a voice dataset with Whisper transcription (README line 89).
Families: MiniMax H3 only (clips and voice).

Also in the workbench, not asked for but adjacent: **Extract** (`families/extract.py` 107 lines: exact weight-SVD rank
reduction per family, with `extract_presets` block groups; `src/fizgig/extraction/model_diff.py` 270 lines:
checkpoint-minus-base to a LoRA at several ranks from one SVD; `diff_to_lora_gui.py` 327 lines, a standalone window;
CLI `scripts/extract_lora.py`) and the **Metadata** tab.

Packages the workbench adds over training: `insightface`, `onnxruntime` (likeness, face tools),
`imageio-ffmpeg` (MP4 export), `comfy-kitchen` (INT8 attention for renders, NVIDIA only), `opencv-python`
(`metrics.py`, exports).

---

## Open questions for the owner (collected)

1. SDXL presets: replace TagScribeR's community defaults with upstream's measured `rank 32 / alpha 16 / 5e-5 / flat
   / adamw` for all five variants, keeping the old three as extras? (Recommended: yes.)
2. Adaptive LR on SDXL: upstream hides it. Keep it visible but off by default, with upstream's warning? (Recommended.)
3. Add a generic "SDXL (any checkpoint)" entry with Juggernaut defaults beside the five variants?
4. SDXL long prompts: move to upstream's unlimited chunks (with exact LCM padding for batches) or keep the fixed 225
   tokens?
5. SDXL empty caption: zeros (upstream / SDXL base training) or the encoded empty string (TagScribeR / ComfyUI)?
6. SDXL LoRA key layout: keep kohya LDM names (recommended) although upstream writes diffusers names?
7. Anima: switch the text encoder to fp32 and the T5 tokenizer to the Anima repo's file (invalidates Anima text
   caches)? (Recommended: yes, before anyone trains.)
8. Anima model code: keep TagScribeR's re-implementation and add what is missing, or vendor upstream's file (needs
   `einops`)?
9. Sliders, fine-tuning, Fast Identity Mode, torch.compile for every family: these are upstream shared-layer
   features TagScribeR lacks for ALL families. Schedule as their own port?
10. `hqq` (H3's HQQ 4-bit base) is not installed and is mapped to NF4. Mirror upstream?
11. Environment: add `FLASH_ATTENTION_TRITON_AMD_ENABLE`, `max_split_size_mb:512` on ROCm, and the MIOpen / rocBLAS
    database paths for non-gfx1100 cards?
12. Krea 2 fp8: upstream removed the fp8 base precision. Confirm TagScribeR keeps it (it must, for the owner's own
    runs).

## What was not checked

* Nothing was executed. Every statement about behaviour is read from source.
* `fizgig-src/lora_trainer_gui.py` was grepped, not read; only the settings dict, the preset apply / collect code,
  the entry keys and `_family_launch_inputs` were read. Keys a widget sets without `self.entries[...]` may be missing
  from B.1.
* `families/train.py` (1675 lines): only the argument parser tail, `--family_option` and the env lines were read. The
  slider, fine-tune and loss-watch loops were not compared with `training/train.py`.
* `families/lora.py`, `families/quant.py`, `families/cache.py`, `families/driver.py`, `utils/capabilities.py`: not
  compared with TagScribeR's `training/lora.py`, `quant.py`, `cache.py`, `driver.py` (cache ids and file naming in
  particular were not traced beyond `arch_id` and the cached dict keys).
* The DiT 3-axis RoPE, `FinalLayer` and `PatchEmbed` of the two Anima model files were not compared line by line.
* No checkpoint header was read (SDXL or Anima), so key-level compatibility of the loaders with real files is
  unverified on both sides.
* `install_fizgig_rocm.sh`, `run_fizgig_rocm.sh`, `uv_install_deps.py`, `detect_gpu.py`, `update_fizgig*.py/.bat`
  and `docker/`: not read.
* `docs/CLI.md` was grepped for SDXL / Anima / prediction only (no hits beyond the generic `--family` usage).

