# Audit 06 - The existing families against Fizgig v8.7.1 (per-family code only)

Read-only audit, 2026-10-19. Fizgig paths are relative to `.claude/fizgig-src/` at v8.7.1 (commit `5d31b2f`);
TagScribeR paths are relative to the repo root (HEAD `7615a63`, branch `port/stage1-bt67en`). "FZ:" = Fizgig,
"TS:" = TagScribeR. TagScribeR was ported against v7.0.1 (`1c8ec88`); every upstream change below is from
`git diff 1c8ec88 5d31b2f` over the family files, read line by line, plus the v7.1.0-v8.7.1 release notes and the
family docs. Nothing was run; no weights were loaded.

Scope: `src/fizgig/families/{krea2,qwen_image,klein,sdxl,anima,minimax}.py` and
`src/fizgig/{krea2,qwen_image21,klein,sdxl,anima,minimax}/` against `training/families/{krea2,qwen_image21,klein,
sdxl,anima,minimax_h3}/`. The shared layer (`families/train.py` +512 lines, `launch.py`, `form.py`, `workbench.py`,
DOP / LoRA Isolation, Multi Concept, `multi_person.py`) and the four new families (Z-Image Turbo, ACE-Step 1.5 XL,
Flux 3 Image Klein, MOSS-Image) are other agents' audits; where a shared change alters a fact about one of these
families it is cross-referenced in section 8, not re-audited.

Verdicts: **present** (TS already has it), **missing** (upstream has it, TS does not), **different** (both have it,
not the same), **obsolete-upstream** (TS carries something upstream dropped), **n/a** (workbench / clips / web UI,
outside TS's scope). Line numbers inside large diff hunks are marked "about" where the hunk, not the file, was read.

---

## 1. Summary - the top items

The per-family *driver* code barely moved: `krea2/driver.py`, `klein/driver.py`, `anima/driver.py` and
`minimax/driver.py` are byte-identical between `1c8ec88` and `5d31b2f` (they are absent from the diff stat);
`sdxl/driver.py` gained 3 lines (the ROCm VAE processor); only `qwen_image21/driver.py` (+74) and `model.py` (+66)
changed in substance. So the objective, target, timestep sampling, noise draw order, loss dtype order, LoRA target
lists and key formats, `ft_spec` and the slider hooks (`noise_latents` / `predict` / `diff_ref`) of every family
are exactly what the 7.0.1 audits compared and stages 1-5 ported. **No text-encoder change anywhere: no
`text_cache_rev` bump is needed for any family** (Krea 2's `ENCODE_SYSTEM_DESCRIPTOR` moved file, byte for byte;
H3's chunked prefill is exact; Qwen's adapter v2 and Turbo files do not touch the encoder).

What did move is in the description files (presets, memory, preview recipes, model files) and a few model-side
fixes. In order of value to the owner:

1. **Krea 2 NF4 memory at 1 MP: 13.4 -> 15.0 GB** (FZ `families/krea2.py:125-129`, #185, v7.3.0: "rank 16 with
   previews on peaked 14.9 training / 15.2 previewing on a simulated 16 GB card - the old 13.4 put 16 GB cards on NF4
   at 1 MP and filled them"). TS still has 13.4 (`training/families/krea2/description.py:88`). The Auto plan on a
   16 GB card at 1 MP now picks INT8 + swap instead of NF4. **The fact behind decision 2's "take upstream's corrected
   memory figures" changed again.** S.
2. **Slider presets at rank 8** on Krea 2, Qwen Image 2.1 and Anima (were rank 4; v7.1.0, "as on MiniMax H3 and
   Z-Image"). TS ported the rank-4 versions days ago (`krea2/description.py:165-174`, `qwen_image21/description.py:
   225-226`, `anima/description.py:222`). S.
3. **Krea 2 Turbo-LoRA preview default: strength 0.7 at 10 steps** (FZ `families/krea2.py:165-166`, "Peter's
   setup"); TS previews at the LoRA's own 1.0 / 8 steps (`krea2/description.py:138`, test
   `tests/test_training_krea2.py:67`). Previews only; a question for the owner (Q1).
4. **Qwen Image 2.1 previews**: Comfy-Org's Turbo LoRA replaces Viggle's (8 steps, CFG 1, its own 8 sigmas with the
   dynamic shift OFF), on by default at strength 1; the plain model previews at **CFG 3** (was 1); the training
   adapter is **v2**; and an optional **Turbo DiT (int8 ConvRot) preview checkpoint** with Klein's park / load /
   unpark handoff, off by default (v7.2.0-v7.3.1, v8.6.1). TS has none of this (`qwen_image21/description.py:
   62-70, 156-193`). M for the LoRA / CFG / adapter, L for the Turbo DiT.
5. **Anima previews default to the Turbo LoRA** (strength 1, CFG 1, 10 steps; FZ `families/anima.py:138-141`, "Peter,
   13 Oct: down from 20"). TS keeps "plain model, strength 0, 20 steps" (`anima/description.py:204-205`) - stage 3's
   call, whose upstream fact changed. S, question Q2.
6. **Multi-reference edits**: Klein takes up to 4 before-images per edit, Qwen up to 10 (`edit_references`, FZ
   `families/klein.py:167-171`, `families/qwen_image.py:158-161`; v8.1.1 / v8.1.2). The drivers already handled a
   list of references in 7.0.1; the new work is the dataset (an extra-pictures folder, own-shape resize), the cache
   limit and the GUI card. TS refuses more than one partner image (`training/cache.py:276-279`). M-L, shared with
   the dataset layer.
7. **SDXL Strong / Standard presets now run 50 epochs** (FZ `families/sdxl.py:170-171`; the helper's default is 20,
   `:14`). TS has 20 (`sdxl/description.py:37, 64-65`). S.
8. **MiniMax H3 on pre-RTX-30 NVIDIA cards (V100)**: the text encoder prefills long captions in chunks through
   `past_key_values` (exact), and the DiT's attention runs items too long for the bf16 math kernel through the fp16
   memory-efficient kernel (FZ `minimax/embedder.py` about `:463-545`, `minimax/model.py` about `:414-467`; v7.2.0,
   by a contributor on 4xV100). TS has neither. NVIDIA-only; cannot be run by the owner. M.
9. **Fizgig now has TagScribeR's ROCm VAE fix** (v7.1.0, #179): `modules/wide_attention.py` computes heads wider
   than 256 by hand in float32 on ROCm, wired into the Qwen-Image VAE (Krea 2, Anima), the Qwen 2.1 VAE, the FLUX.2
   AE and SDXL's diffusers VAE. It is the same method as `training/modules/wide_attention.py`; nothing upstream
   conflicts with TS's fix (section 7).
10. **Obsolete upstream**: the Viggle turbo LoRA and its community settings; the `retired_preview_defaults` /
    `preview_reset` fields (TS `training/description.py:217-222`, used only by Qwen); "LoRA the Explorer" in
    `workbench=` tuples; "(experimental)" in GUI labels (v7.2.1: "No model is called experimental any more" - TS's
    flags are its own "until a real run" criterion, see Q3).

The TagScribeR-only fixes CLAUDE.md lists (wide attention, `latent_rev`, the CUDA guard in offloading, fp8 base
precision, the run-folder launcher) are all untouched by the upstream delta - see section 7.

---

## 2. Krea 2 (`families/krea2.py`, `krea2/`)

### 2.1 Every upstream change since 1c8ec88

| # | Change | FZ | TS | Verdict |
|---|---|---|---|---|
| 1 | `experimental=False` (v7.2.1) | `families/krea2.py:38` | `experimental=True` "until the owner's first real run confirms the port" (`krea2/description.py:40`); the Train tab appends "(experimental)" (`tabs/train.py:115`) | different - TS's own criterion; the owner has run it twice (PORT_PLAN "Real hardware results"). Q3 |
| 2 | Turbo LoRA fetched by default (`fetch_optional=False`, note "8-step previews on the training model") | `families/krea2.py:49-51` | `required=False`, note "Optional: ..." (`krea2/description.py:60-63`); TS's Preferences mark non-required rows "(optional)" (`tabs/train.py:460`) | different (download policy). Not checked how `tools/install.py` picks files. S |
| 3 | Turbo DiT note loses "LoRA the Explorer"; `repair_presets` lose "(experimental)"; `workbench` loses `"explorer"` (v8.0.0: the Explorer is gone) | `:52-55, 113, 116-118` | no workbench, no `repair_presets` | n/a |
| 4 | **`train_memory` NF4 at 1 MP 13.4 -> 15.0** (#185, v7.3.0; INT8 + 16 swapped blocks peaked 12.7 / 12.9 on the same run) | `:125-129` | `nf4: ((0.25, 11.4), (1.0, 13.4))` (`:88`); `tests/test_training_krea2_parity.py:372` asserts a 14 GB card at NF4 / 1 MP with the old figure | **missing**. Effect: Auto on a 16 GB card at 1 MP moves from NF4 to INT8 with swap (`training/quant.py` plan, unchanged ladder). At the owner's 0.25 MP nothing changes. S |
| 5 | **Preview defaults with the Turbo LoRA: `preview_speed_steps=10`, `preview_speed_strength=0.7`** ("Samples previews: Turbo 0.7, 10 steps, CFG 1 (Peter's setup)") | `:164-168` | `preview_speed_steps=8`, strength None = the LoRA's own 1.0 (`:137-138`, `training/description.py:287-288`); `tests/test_training_krea2.py:67` pins `(8, 1.0)`; help says "renders in 8 steps" (`tabs/help_content.py:534-535`) | **different**. Previews only. The bias deltas follow the load strength on both sides (FZ `families/lora.py:465-466`, TS `training/lora.py:485-493`), so 0.7 is applied the same way. Q1 |
| 6 | Plain-model preview stays `preview_steps=8, preview_cfg=1.0` | `:167-168` | 28 steps / CFG 4.5 (`:139-140`) | settled ("28 RAW preview steps without the Turbo LoRA" kept) - upstream unchanged, nothing to re-open |
| 7 | **Slider preset rank 4 -> 8** (`_preset(8, lr=2e-4, epochs=20, slider=True, mp="0.5")`) | `:177-180` | rank 4, alpha 4 (`:165-174`, a hand-written dict) | **missing**. S |
| 8 | `_preset` carries `"FAMILY_SLIDER_GUIDANCE": "3"` in every Krea 2 preset | `:14-30` | only the Slider preset has it (`:174`) | different, cosmetic (the key is read for sliders only) |
| 9 | `krea2/caption_text.py` (new): the Qwen3-VL caption instructions moved out of `embedder.py` so the web UI lists them without torch; `ENCODE_SYSTEM_DESCRIPTOR` moved verbatim | `krea2/caption_text.py:1-126`, `krea2/embedder.py:292-294` | TS's Krea 2 encoder is encode-only; captioning is `inference/` | n/a (refactor). **Verified** the encode descriptor string is unchanged: no `text_cache_rev` bump |
| 10 | VAE attention through `wide_head_attention` (#179) | `krea2/vae.py:372-374` | `training/families/krea2/vae.py:34, 389-390` | present (TS's own fix; section 7) |
| 11 | `THIRD_PARTY_NOTICES.md` path in headers -> `docs/THIRD_PARTY_NOTICES.md` | `krea2/{attention,fp8_optimization_utils,lora_utils,model,offloading,safetensors_utils,sampling}.py:1-7` | - | n/a |
| 12 | Compile test before a run (v8.3.1: VS 2019 PCs failed at step 0) | `families/compile.py:104-136, 185-186` | shared `training/compile.py` | shared layer (section 8); Krea 2 is the family it was reported on |

### 2.2 Exactly checked, unchanged upstream (TS matches 7.0.1's port)

| Item | FZ (5d31b2f) | TS | Verdict |
|---|---|---|---|
| Objective: `noised = (1-t) latent + t noise` in bf16, `target = noise - latent` in bf16, `mse(pred.float(), target.float())`, DiT under bf16 autocast | `krea2/driver.py:165-190` | `krea2/driver.py:197-224` (`_mix` `:181`, autocast `:194`) | present (stage 1) |
| `int8_fp32_scales = False` | `:38` | `:111` | present |
| Timesteps: `t = sigmoid(randn(1))`, `shift = exp(mu)`, window rescales | `:141-149` | `_draw` `:173` -> `sampling.py` | present |
| Draw order: noise, then t | `:198-199`, `:206-207` | `:173`, `:227-231` | present |
| `noise_latents` / `predict` (prompt sliders, DOP) | `:203-212` | `:227-235` | present (the state keys differ, `t` vs `tt`; internal) |
| `loss_at(diff_ref, diff_weight)` per-token weighting | `:168-190` | `:206-222` | present |
| One caption per forward in `encode_text` | `:121-128` | `:157-163`, `text_cache_rev="1"` (`description.py:87`) | present |
| `ft_spec` (attn, mlp.gate/up/down, `always_on=("txtfusion",)`, overhead 9.5, trunk 0.217, stream 3.2, calib 0.25 MP, 3.2 GB/MP) | `:40-47` | not ported (stage 5) | missing, unchanged since 02 A8 #3 |
| `precisions=("int8","nf4","bf16")`, `auto_precisions=("int8","nf4")` | `:91-92` | `("bf16","fp8","int8","nf4")`, auto INT8 then NF4 (`:72-75`) | settled (fp8 kept) |
| `optimizers=("adamw8bit","adamw")` - the web form still lists the whole catalog (`families/form.py:82` `available_optimizers`) | `:130` | the whole catalog (`:95`) | settled (stage 1-2) |
| `slider_guidance=3.0`, Ultra blocks 0-7 + 4 text-fusion | `:139-142` | `:145-149` | present |
| LoRA keys `lora_unet_blocks_N_<module>` + LyCORIS LoKR | unchanged | `:51-64` | present |

---

## 3. Qwen Image 2.1 (`families/qwen_image.py`, `qwen_image21/`)

### 3.1 Every upstream change since 1c8ec88

| # | Change | FZ | TS | Verdict |
|---|---|---|---|---|
| 1 | `gui_label="Qwen Image 2.1"`, `experimental=False` | `families/qwen_image.py:45, 48` | "(experimental)" label and flag (`qwen_image21/description.py:45, 48`) | different (TS's own criterion; never run here) |
| 2 | **Training adapter v2** `fizgig_qwen_image_2.1_training_adapter_v2.safetensors`, `replaces=(v1,)` (v7.3.1: a pref pointing at v1 is moved to v2; a user's own file is kept). `ModelFile.replaces` is a new field | `:62-67`; `families/description.py:46-48` | v1 file (`:62-66`); no `replaces` in `training/description.py` | **missing**. S (file name + a one-time pref migration) |
| 3 | **Turbo LoRA = Comfy-Org `loras/qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors`** (0.91 GB), fetched by default; sampling 8 steps, CFG 1, `sigmas=_TURBO_SIGMAS` = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568), `shift_terminal None`, **`dynamic_shift False`** (the Turbo's scheduler_config: use_dynamic_shifting false, shift 1.0) | `:13-16, 68-71, 211-225` | Viggle 6-step r128 with its community settings and caveats (`:67-70, 156-181`) | **missing / obsolete-upstream**. M (with #4, #5) |
| 4 | `sigma_schedule(..., dynamic_shift=True)`: with False the sigmas are used as given; `sample()` and the driver's `generate()` pass `opts["dynamic_shift"]` | `qwen_image21/sampling.py:22-32, 68-79`; `qwen_image21/driver.py:256-258` | `sigma_schedule(steps, image_seq_len, sigmas, shift_terminal)` always applies the exponential shift (`qwen_image21/sampling.py:25`) | **missing**. Without it the new Turbo sigmas would be shifted by `mu` - wrong previews. S |
| 5 | **Preview defaults: Turbo LoRA strength 1.0 at 8 steps when its file is set; else the plain model at 25 steps, `preview_cfg=3.0`** (v7.2.0 table; v8.6.1 fixed the Samples tab putting 25 / 3 back under the Turbo). `retired_preview_defaults` / `preview_reset` deleted from `FamilyDescription` | `:229-233`; `families/description.py:497-503` (old) | `preview_speed_steps=25, preview_speed_strength=0.0, retired_preview_defaults=((10, 0.7),), preview_reset=..., preview_cfg=1.0` (`:185-193`); the fields live in `training/description.py:217-222` | **different / obsolete-upstream**. Q4 |
| 6 | `preview_cfg_note` and `samples_text` (Samples-tab wording: banner, "flow: not used", "neg: ... the Turbo samples don't use it", cfg) | `:164-182` | none (TS has its own help, `tabs/help_content.py`) | n/a (web UI text); the CFG facts belong in TS's help when #5 lands |
| 7 | **Turbo DiT preview checkpoint**: `ModelFile("qwen21_turbo_dit", ... "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors", 7.26, role="preview_dit", fetch_optional=False)`; `train_preview_checkpoint=True`, tick "Use the Turbo model for samples (8-step, CFG 1 ...)", **`preview_checkpoint_on=False`**, `preview_checkpoint_sampling` = the 8 Turbo sigmas, euler, CFG 1, no dynamic shift | `:72-77, 170-173, 183-187`; `families/description.py:420-422` (the three new tick fields) | none; TS's `pipeline.py:589` turns the checkpoint on by default for any family with `train_preview_checkpoint` (Klein's rule) | **missing**. L: needs #8-#10 and a per-family default for the tick |
| 8 | Driver handoff: `park_for_preview` (stream all but 2 blocks; NF4 left alone -> token None), `load_preview_checkpoint` (ConvRot file costs its size; bf16 / INT8 / INT8 + streamed blocks by free VRAM, `room = 4.0` GB), `unpark_after_preview` (swap back, or `disable_block_swap()` + move back) | `qwen_image21/driver.py:65-111` | no `park_for_preview` in the Qwen driver (grep) | **missing** (part of #7) |
| 9 | `QwenImage21DiT.disable_block_swap()` (removes the offloader's backward hooks) | `qwen_image21/model.py:349-355` | not present | **missing** (part of #7) |
| 10 | **ConvRot int8 file support**: `is_convrot()` (a `<module>.comfy_quant` key), `_take_convrot()` (int8_tensorwise only; the fused `img_mlp.gate_up` split by rows into gate_layer / proj with their scales), `load_qwen21_dit(..., keep_convrot=True)` keeps `ConvRotInt8Linear` from `minimax/convrot.py`, `keep_convrot=False` decodes to bf16; the `.comfy_attention.config` keys of the 9 Oct re-upload are dropped (#184, v7.2.3); `load_planned()` loads a ConvRot DiT as stored, or decodes + INT8 + swap | `qwen_image21/model.py:464-540`; `qwen_image21/driver.py:33-49` | none; TS has `ConvRotInt8Linear` inside `training/families/minimax_h3/model.py:342` (family-local) | **missing** (part of #7; the training base can also be the Turbo file through `load_planned`, which is a workbench use upstream - not needed for training on the base) |
| 11 | **`edit_references=10`** ("the edited photo first, then up to 9 supporting references ... The text encoder sees them too: the dataset's resize (own shape, the 32 px grid) is what both the vision pass and the latents get") | `:158-161` | `edit_training=True` with "(up to 10 references)" in a comment (`:131`); the driver's `encode_text_with_references` / `_ref_sequence` take a list (`qwen_image21/driver.py:87-108`); the cache refuses > 1 partner (`training/cache.py:276-279`) | **missing** (dataset + cache + GUI; section 8 #3). M |
| 12 | **Slider preset rank 4 -> 8**; `_preset(..., slider=False)` carries `FAMILY_SLIDER` and `"FAMILY_SLIDER_GUIDANCE": "2"` in every preset | `:20-38, 263-265` | rank 4 with `FAMILY_SLIDER` / guidance "2" added by hand (`:225-226`) | **missing** (rank). S |
| 13 | `workbench` loses `"explorer"`; `preview_dit` note | `:267` | `workbench=("repair", "explorer", ...)` (`:228`, unused in TS) | obsolete-upstream, cosmetic |
| 14 | VAE attention through `wide_head_attention` | `qwen_image21/vae.py:497-499` | `training/families/qwen_image21/vae.py:31, 505-506` | present |
| 15 | Fast Identity Mode, compile, identity blocks, memory table, LoRA keys, `ft_spec` (four windows per block, gate_up fused) | unchanged (`:113-160`, `qwen_image21/driver.py:113-160`) | present / `ft_spec` missing (stage 5) | unchanged since 02 B |

Objective, `_sample_t`, draw order, `noise_latents` / `predict`, `diff_ref` weighting: `qwen_image21/driver.py:
194-240` unchanged; TS `qwen_image21/driver.py:112-160` is the 7.0.1 port. **present**.

---

## 4. FLUX.2 Klein 9B (`families/klein.py`, `klein/`)

| # | Change | FZ | TS | Verdict |
|---|---|---|---|---|
| 1 | `experimental=False` (unchanged since 7.0.1: Klein was never experimental upstream) | `families/klein.py:35` | `experimental=True` (`klein/description.py:41`) | TS's own criterion (the owner does not train Klein; "stays marked experimental", PORT_PLAN stage 2) |
| 2 | **`edit_references=4`** ("BFL's FLUX.2 overview: [klein] 'Up to 4' references ... its code sets no count (each its own t offset, capped at 1 MP when several). The text encoder never sees them"); v8.1.1 / v8.1.2 | `:167-171` | `edit_training=True` (`:127`), no `edit_references`; `pack_refs` already gives each reference its own t offset 10, 20, ... (`klein/sampling.py:37-49`, as FZ `klein/position.py:120-145`); the cache refuses > 1 partner (`training/cache.py:276-279`) | **missing** at the dataset / cache / GUI level; the driver side is present. M (shared with Qwen, section 8 #3). The "capped at 1 MP when several" clause is BFL's inference code, not something the trainer does - **not checked** whether upstream's dataset caps extra references at 1 MP (`_own_shape` keeps the target's pixel count, `dataset/image_dataset.py:207-213`) |
| 3 | `workbench` loses `"explorer"` | `:102` | none | n/a |
| 4 | AE attention through `wide_head_attention` ("one 512-wide head") | `klein/model.py:143-144` | `training/families/klein/vae.py:20, 84-85` | present |
| 5 | Everything else: presets (7, Identity at rank 8), Auto INT8 then NF4 with no swap, `auto_uncompiled_precision` (fp8 file as is), memory table, `train_areas`, block ids `double_N` / `single_N`, Distilled 4-step previews on by default (`preview_checkpoint_on` default True, `families/description.py:421`), park / load / unpark, `ft_spec`, LoKR, EMA Off, `adaptive_lr_clip_signal` | `families/klein.py` unchanged apart from #1-#3; `klein/driver.py` unchanged | stage 2 (`klein/description.py:50-55, 149-154, 161-172`; `klein/driver.py:232-290, 346-375`) | present (`ft_spec` missing, stage 5) |

Objective and timesteps: FZ `klein/driver.py:160-237` unchanged; TS `klein/driver.py:318-375` is the 7.0.1 port
(noise, then one t; `(t * 1000 + 1) / 1000`; refs after the image tokens; `diff_ref` weighting). **present**.

---

## 5. SDXL (`families/sdxl.py`, `sdxl/`)

| # | Change | FZ | TS | Verdict |
|---|---|---|---|---|
| 1 | `gui_label="SDXL (any checkpoint)"` (no "experimental") | `families/sdxl.py:34` | five variants, `experimental=True` (`sdxl/description.py:94`) | TS's own criterion (never run here) |
| 2 | **Strong and Standard presets: `epochs=50`** (the `_preset` default stays 20, `:14`; the Slider keeps 30, the Fine-tunes 20) | `:170-171` | `_fizgig_preset(32, 16)` / `(16, 8)` with `epochs=20` (`:37, 64-65`) | **missing**. S. No measurement is cited for 50 (the diff adds only the number) |
| 3 | `workbench` loses `"explorer"` | `:77` | none | n/a |
| 4 | **ROCm VAE processor**: `vae.set_attn_processor(rocm_vae_processor())` when `ROCM` (diffusers `AttnProcessor2_0` with the call through `wide_head_attention`) | `sdxl/driver.py:137-139`; `modules/wide_attention.py:32-73` | TS sets diffusers' classic `AttnProcessor` (plain matmul + softmax) on every platform, VAE in fp32 (`sdxl/vae.py:45-49`) | present (TS's own fix, platform-independent). Equivalent on ROCm: both do softmax(q k^T / sqrt d) v in fp32 (TS: the VAE's fp32; FZ: `.float()`). Difference: on NVIDIA TS also takes the classic path; upstream keeps SDPA there. No conflict |
| 5 | Everything else: presets' values (rank 32 / alpha 16 / 5e-5, fused AdamW, loss watch + per-image LR + auto-recaption), `preview_cfg=3.0`, DPM++ 2M SDE Karras, INT8 / NF4 memory table, LoKR on kohya stems, slider hooks with epsilon prediction | unchanged (`:98-179`, `sdxl/driver.py:163-341`) | stage 3 (`sdxl/description.py:37-73, 124-137`) | present (SDXL `ft_spec` `sdxl/driver.py:341` missing, stage 5) |

TS keeps (settled, decision 4): five variants and loaders, NoobAI v-pred with zero-terminal-SNR, LoCon, offline
strict loading, batch above 1, the timestep window, Adaptive LR, Euler previews, the 225-token captions. Nothing
upstream touches those paths.

---

## 6. Anima (`families/anima.py`, `anima/`)

| # | Change | FZ | TS | Verdict |
|---|---|---|---|---|
| 1 | `gui_label="Anima"` | `families/anima.py:31` | `gui_label="Anima"`, `experimental=True` (`anima/description.py:106-109`) | present (label); flag is TS's own |
| 2 | **Turbo-LoRA previews on by default**: `preview_speed_steps=10`, `preview_speed_strength` removed (None = the SpeedLoRA's `strength=1.0`, `:126`); comment "With the Turbo LoRA file set, previews render on it by default (strength 1, CFG 1) at its card's 10 steps (Peter, 13 Oct: down from 20)". The Samples CFG box follows the Turbo on and off (`cfg_note()`, `families/description.py:646-654`) | `:138-141` | `preview_speed_steps=20, preview_speed_strength=0.0` (`:204-205`), the Turbo ModelFile note "Previews still start on the plain model (Turbo strength 0)" (`:125-128`), help `tabs/help_content.py:670-671`, test `tests/test_training_anima.py:67` `(20, 0.0)` | **different**. Stage 3's "strength 0 by default" mirrored 7.0.1; the upstream fact changed. S, Q2 |
| 3 | **Slider preset rank 4 -> 8** | `:159-160` | rank 4 (`:222`) | **missing**. S |
| 4 | `workbench` loses `"explorer"` | `:64` | none | n/a |
| 5 | Qwen-Image VAE attention through `wide_head_attention` (shared with Krea 2) | `krea2/vae.py:372-374` | Anima shares `training/families/krea2/vae.py` (`anima/description.py` fact 3) | present |
| 6 | Everything else: Character / Style / Official presets, `preview_steps=20`, `preview_cfg=4.5`, the card's negative, fp32 text encoder, the repo's T5 tokenizer, INT8 / NF4 memory, `ft_spec` | unchanged (`:84-165`; `anima/driver.py` unchanged) | stage 3 (`anima/description.py:167-232`) | present (`ft_spec` `anima/driver.py:250` missing, stage 5) |

Objective and timesteps: FZ `anima/driver.py:163-205` unchanged; TS `anima/driver.py:90-125`. **present**.

---

## 7. MiniMax H3 (`families/minimax.py`, `minimax/`)

### 7.1 Every upstream change since 1c8ec88

| # | Change | FZ | TS | Verdict |
|---|---|---|---|---|
| 1 | `H3_BLOCKS` `unmet_notes` reworded ("Set by Training mode (Default): blocks 20-49 for photos, clips and voice. To choose the blocks yourself, set Training mode to Off.") | `families/minimax.py:132-134` | TS's own help text says the same facts (`training/params.py:299-303`, `H3_LIKENESS_MODE` `:298`) | present (different wording) |
| 2 | `H3_DISTILL` gets `isolation_off=True` (can't run with LoRA Isolation) | `:135-139`; `families/description.py:227` | distillation not ported (stage 6) | n/a until stage 6 / Isolation |
| 3 | `H3_SAMPLE_FRAMES` default "56 frames with sound (~2.3s)" | `:156-161` | stills only; no such option | n/a (clips, stage 6) |
| 4 | `detail_blocks=("h3blk_46".."h3blk_49")` (Studio's Alternate / Toggle buttons) | `:182` | no workbench | n/a |
| 5 | `prefs_title="Model Paths (MiniMax H3)"` | `:270` | - | n/a |
| 6 | `workbench` adds `"refmod"`, loses `"explorer"`; `video_controls=VideoWorkbench(...)` with `speed_boost=True` | `:292, 330-346` | no workbench | n/a (stage 6-7) |
| 7 | **Text encoder: chunked prefill on pre-sm_80 CUDA cards** (`_plan_chunk`: None on non-CUDA, on sm_80+, or when B*L^2 <= 1024^2; else a chunk size from live free VRAM; chunk k passes `past_key_values` from chunk k-1, "causal attention makes this EXACT"); `_cfg_attr` reads `text_config` too (v7.2.0, #172, 4xV100) | `minimax/embedder.py:432-440`, about `:463-545` | `training/families/minimax_h3/embedder.py` has no chunking (grep: no `past_key_values`, no `get_device_capability`) | **missing**. NVIDIA pre-Ampere only; output identical where it triggers. M |
| 8 | **DiT attention: pre-sm_80 bf16 fallback** (`_BF16_SDPA` fit boundary from free VRAM at the first call, 16 B/element, 80% margin; items with S above it run SDPA in fp16 and cast back; "q and k are RMS-normalized right before this call") | `minimax/model.py` about `:414-467` (`h3_attention`) | `minimax_h3/model.py` `h3_attention` without it | **missing**. Same scope as #7; **changes numerics only on those cards** for long items. S-M |
| 9 | Speed Boost for clips (`speed_boost=` in `_sample_image_impl`; `minimax/speed_boost.py`: SPEED spectral shrink / grow, half the steps at half size) | `minimax/sampling.py:178-182, 275-292, 301-312`; `minimax/speed_boost.py:1-72` | stills only | n/a (clips; v8.3.0 says "a seed frames a little differently with it on") |
| 10 | GPL-3.0 headers on the ComfyUI-derived files (`model.py`, `vae.py`, `audio_vae.py`, `reference.py`) and on `_res_multistep_coeffs` | `minimax/model.py:1-3`, `minimax/vae.py:1-3`, `minimax/sampling.py:371-373` | `THIRD_PARTY_NOTICES.md:66-76` already records the ComfyUI derivation of `model.py`, `vae.py` and the res_multistep step; TS is GPL-3.0 | present (TS's notice is complete for the files it has) |
| 11 | ReFLoRA files (`minimax/reflora.py`, new; `refmod_meta.py` split out of `refmod_apply.py`) | `minimax/reflora.py:1-87`, `minimax/refmod_meta.py` (+406) | RefMods not ported | n/a (stage 6) |
| 12 | `H3FTBackend.plan(..., held_grads=False)` (v8.5.7: the fine-tune stopped with "unexpected keyword argument 'held_grads'" since DOP) | `minimax/ft_backend.py:194-199`; `families/ft.py:734-742` | fine-tune not ported | n/a until stage 5 - **the H3 backend's signature must take `held_grads` when it is** |
| 13 | `still_pick.py` imports `fizgig.face_utils` | `minimax/still_pick.py:36` | not ported | n/a |
| 14 | Unchanged: presets (Fast 8 / 50, rank 16 / 60, Style, Slider rank 8 at 2e-4), options and their `H3_*` keys, `precisions=("int8","nf4","hqq")`, `optimizers=("automagic3","adamw8bit","adamw")`, weight decay 1e-4, 768 previews, `lora_target_names`, `ft_spec`, `block_map` ids, the training objective and `_shift` | `families/minimax.py:14-125, 373-416`; `minimax/driver.py` unchanged | stage 4 (`minimax_h3/description.py:18-35, 123-183`) | present (settled calls stand: `minimaxh3` suffix, trained window only, res_multistep Turbo previews) |

The H3 driver is unchanged, so the "Fizgig's options are TagScribeR parameters with Fizgig's `H3_*` keys" call
(settled) has nothing new to absorb. `H3_LOWNOISE_PCT` default "60" is unchanged (`:62`); only MOSS-Image got the
new 30% default (v8.7.1) - not an H3 change.

---

## 8. TagScribeR-only fixes: nothing upstream conflicts

| TS fix (CLAUDE.md) | Upstream now | Verdict |
|---|---|---|
| `training/modules/wide_attention.py` (heads > 256 computed in fp32 row chunks; used by the Krea 2 / Anima, Qwen 2.1 and Klein VAEs; SDXL uses the classic processor) | **Mirrored upstream in v7.1.0** (#179, reported with a diagnosis by a Fizgig user): `modules/wide_attention.py` - same method (`q.float() * d^-0.5`, `softmax(q k^T) v` in float32, 2^27-element row chunks), gated on `ROCM and head > 256` (`:12-29`); SDXL through a diffusers processor on ROCm only (`:32-73`). TS gates on head width alone (`training/modules/wide_attention.py:25, 30-32`) | compatible. TS's header "Fizgig has the same behaviour; this is a TagScribeR fix" (`:17`) is now out of date: Fizgig has the fix. Keep TS's (it also protects NVIDIA from a future kernel change; CUDA already falls back to math for wide heads per TS's header - **not re-verified**) |
| `latent_rev` mark in `training/cache.py` | no upstream counterpart; `families/cache.py` changed only for multi-reference edits and DOP's `save_cond_file` (`:79-85, 164-187`) | untouched |
| CUDA guard in `training/modules/offloading.py` | `modules/offloading.py` changed only `_streamed_linears` (#199: pair Linears with the LoRA wrapper's `.base` dropped, `:50-58, 67-71, 169-178`) - a MOSS-Image block-swap fix that also applies to any swapped block whose Linears carry a LoRA | untouched; the `_streamed_linears` fix is a shared-layer item (section 9) |
| fp8 base precision (`training/quant.py`, `training/modules/fp8.py`) | still gone upstream for Krea 2 (`families/krea2.py:91`); nothing new | settled, untouched |
| Run-folder launcher `training/pipeline.py` | upstream's `families/launch.py` grew +451 lines (web form, DOP, Subjects card) - another agent's scope | untouched |
| Batch size > 1 (SDXL, Krea 2 ...) | v7.2.0: the GUI pins Batch Size at 1 ("a higher value stopped some runs"); `families/train.py:973-975` still refuses > 1 | settled (kept); FYI only |

---

## 9. Shared-layer changes that alter a fact about these families (cross-references, not audited here)

| Change (version) | Where | Family facts it touches |
|---|---|---|
| DOP and Fizgig LoRA Isolation (v8.4.0-v8.4.2); `multi_concept=True` and `subject_isolation=True` on every family by default (`families/description.py:326-334`) | `families/train.py:414-514, 1058-1128, 1747-1762`, `families/multi_person.py`, `trigger_check.py` | "tested on Krea 2" (0 of 50 strangers vs 48 of 50; ArcFace 0.78 vs 0.77); H3 trains people from photos only (`dataset/image_dataset.py:888-901`), distillation off under Isolation (7.1 #2); needs every driver's `noise_latents` / `predict` - present in all six TS drivers |
| Multi Concept on every model's Standard LoRA and fine-tune (v8.4.0) | `families/description.py:326-331` | was H3-only in 7.0.1 |
| Multi-reference edits (v8.1.1-v8.1.2): `extra_reference_directory`, `_own_shape`, `caption_replace`, `photos_only`, the cache limit `edit_references` and one-at-a-time control encoding | `dataset/image_dataset.py:207-213, 624-633, 694-696, 1078-1085`; `families/cache.py:164-187` | Klein 4, Qwen 10 (sections 3 #11, 4 #2) |
| Slider steering mid-run (v8.0.3), `practice_latent` default hook, `slider_setup(group)`, `slider_sources`, `kind_notes`, `slider_prompt_notes` | `families/driver.py:201-215`; `families/train.py:1377-1412`; `families/description.py:464-470` | every slider family; no per-family code |
| `lr_hint=(2e-4, 125)`, `long_run_hint`, `preview_every`, `preview_strengths`, `ft_regularisation` fields (v7.2.0, v7.3.0) | `families/description.py:404-411, 478-481` | GUI hints; defaults apply to all six families |
| Fine-tunes never use Adaptive LR (v8.1.3, v8.4.0) | launch / form | stage 5 |
| Compile test before a run (v8.3.1); previews run uncompiled under Compile Blocks (v7.2.2) | `families/compile.py:104-136, 185-186`; `families/train.py` | Krea 2 and Qwen (the compiling families) |
| Block-swap pairing with LoRA wrappers (#199, v8.5.8) | `modules/offloading.py:50-58` | any swapped family |
| Samples steps / CFG always match the Turbo state (v7.2.1, v8.6.1) | web form | Qwen, MOSS; the facts are in section 3 #5 |
| Driver base additions: `keep_text_encoder` / `drop_text_encoder` / `text_encoder_gb` (workbench), `from_pretrained` / `helper_dir` cache-first statics, `text_cache_key()`, `activity_axis()` | `families/driver.py:72-99, 112-115, 475-479` | no family in scope overrides them (ACE-Step does); TS has cache-first loading in `training/hf_cache.py` |
| `ModelFile.fetch_is_optional`: a `training_adapter` is never optional by default (v7.2.1); `bulk_fetch`; `replaces` | `families/description.py:46-69` | Qwen's adapter, Krea 2's and Qwen's Turbo LoRAs |
| Resume restores the paused run's settings (v7.3.0); re-encode an edited picture (v8.4.0); state-save retry (v8.5.9) | `families/train.py`, launch | all |

---

## 10. Port list (dependency order, sizes S / M / L / XL)

| # | Item | Families | Files (TS) | Size | Depends on |
|---|---|---|---|---|---|
| P1 | NF4 1 MP memory 15.0 (FZ `families/krea2.py:125-129`) + the parity test + the help / README memory text | Krea 2 | `krea2/description.py:88`, `tests/test_training_krea2_parity.py:372`, `tabs/help_content.py` | S | - |
| P2 | Slider presets at rank 8 (names "✨ ... Slider (rank 8, 2e-4)") | Krea 2, Qwen, Anima | the three `description.py` presets; any test naming them | S | - |
| P3 | SDXL Strong / Standard `epochs=50` | SDXL (all five variants) | `sdxl/description.py:37, 64-65` | S | - |
| P4 | Krea 2 Turbo preview default 0.7 / 10 steps (`preview_speed_steps=10, preview_speed_strength=0.7`); help and the test | Krea 2 | `krea2/description.py:137-138`, `tests/test_training_krea2.py:67`, `tabs/help_content.py:534-535` | S | Q1 |
| P5 | Anima Turbo previews on by default (10 steps, strength 1, CFG 1); ModelFile note, help, test | Anima | `anima/description.py:125-128, 204-205`, `tests/test_training_anima.py:67`, `tabs/help_content.py:670-671` | S | Q2 |
| P6 | Qwen preview recipe: Comfy-Org Turbo LoRA (file, 0.91 GB, sigmas, `dynamic_shift` False), `sigma_schedule` / `sample` / `generate` `dynamic_shift`, Turbo on by default at 1.0 / 8, plain model CFG 3, drop `retired_preview_defaults` / `preview_reset` (and the two fields from `training/description.py:217-222` if nothing else uses them), help | Qwen | `qwen_image21/description.py:67-70, 156-193`, `qwen_image21/sampling.py:22-79`, `qwen_image21/driver.py` generate | M | Q4 |
| P7 | Qwen training adapter v2 + `ModelFile.replaces` (a pref pointing at the v1 file name moves to v2; another file is left alone) | Qwen | `qwen_image21/description.py:62-66`, `training/description.py` ModelFile, the Preferences / installer path that resolves files | S | not checked where TS resolves model paths (`tools/install.py`, `tabs/train.py:442-460`) |
| P8 | Turbo LoRA fetched by default (Krea 2, Qwen); `fetch_is_optional` rule for training adapters | Krea 2, Qwen | `ModelFile` + installer | S | P7; not checked how `tools/install.py` picks optional files |
| P9 | Multi-reference edits: `edit_references` field (Klein 4, Qwen 10), extra-pictures folder, `_own_shape` resize for references 2..n, cache limit, one-at-a-time control encoding, previews with the test photo's extras, the Edit card + help | Klein, Qwen (+ shared dataset) | `training/description.py`, `training/dataset.py:262-275`, `training/cache.py:195-210, 276-279`, `training/pipeline.py`, `tabs/train.py`, `tabs/help_content.py` | M-L | shared dataset layer (other agent) |
| P10 | H3 pre-sm_80 NVIDIA paths: chunked TE prefill (exact) and the fp16 attention fallback for long items | H3 | `minimax_h3/embedder.py`, `minimax_h3/model.py` `h3_attention` | M | tiny-model unit tests only; a V100 user must confirm |
| P11 | Qwen Turbo DiT previews: `qwen21_turbo_dit` ModelFile (`preview_dit`), `train_preview_checkpoint` with a per-family tick default (off) / label / steps note, ConvRot loader (`is_convrot`, `_take_convrot`, `comfy_attention.config` drop, `keep_convrot`), `ConvRotInt8Linear` shared out of `minimax_h3/model.py:342`, `disable_block_swap`, `park_for_preview` / `load_preview_checkpoint` / `unpark_after_preview`, `load_planned`, `preview_checkpoint_sampling`, help | Qwen (+ `training/description.py`, `training/pipeline.py:589`) | `qwen_image21/{description,driver,model}.py`, `training/description.py:208-210`, `training/pipeline.py:586-595` | L | P6; Q5 |
| P12 | Cosmetic: `FAMILY_SLIDER_GUIDANCE` in every Krea 2 / Qwen preset (so an exported preset matches Fizgig's key set); `workbench` tuples without "explorer"; the wide-attention header no longer says Fizgig lacks the fix | Krea 2, Qwen, all | `krea2/description.py:150-174`, `qwen_image21/description.py:22-38, 228`, `training/modules/wide_attention.py:17` | S | - |
| P13 | `experimental` flags: lift Krea 2's (the owner has run it twice); the others stay until a real run | Krea 2 | `krea2/description.py:40` | S | Q3 |
| P14 | Stage 5 reminder: H3's fine-tune backend `plan()` must accept `held_grads`; `FTSpec.streams` / `window_factor` (DOP + fine-tune memory) are shared-layer fields, none set by these six families' `ft_spec` | H3, all | stage 5 | - | stage 5 |

Nothing in this list changes a training step on the owner's verified Krea 2 setup (0.25 MP, INT8, no swap): P1
alters the plan only at 1 MP on a 16 GB card; P4 alters previews only; P2 alters the Slider preset only.

---

## 11. Questions for the owner (as recommendations)

- **Q1 - Krea 2 Turbo previews at 0.7 / 10 steps?** Fizgig's Samples default since v7.2.x ("Peter's setup",
  `families/krea2.py:166`); TS previews at the LoRA's own 1.0 / 8. Previews only - training is untouched, but the
  owner's reference previews for a seed will change. Recommendation: mirror (keep the Turbo strength box, so 1.0 is a
  one-field change back), since it is the one Krea 2 preview recipe Fizgig's author runs.
- **Q2 - Anima previews on the Turbo LoRA by default (strength 1, CFG 1, 10 steps)?** Fizgig changed this on 13 Oct
  (`families/anima.py:139-141`). Recommendation: mirror; Anima is unverified here either way, and the Turbo preview
  is what the card recommends for speed.
- **Q3 - Lift Krea 2's "(experimental)" label?** TS's own rule was "until the owner's first real run confirms the
  port"; two runs are done (PORT_PLAN). Recommendation: lift it for Krea 2 once the next ordinary run on this branch
  confirms the stage-1 numerics (the run the plan already asks for); keep the others.
- **Q4 - Qwen preview recipe (Comfy-Org Turbo LoRA on by default, plain CFG 3, adapter v2)?** Recommendation: mirror
  all three; the Viggle LoRA and the CFG-1 default are obsolete upstream and the adapter v2 is "the one Fizgig uses"
  (v7.3.1). The adapter changes what the LoRA trains against: a Qwen run on v1 and one on v2 are not comparable -
  worth a line in the docs.
- **Q5 - Qwen Turbo DiT previews (P11, L)?** A 7.3 GB optional file, previews only, off by default upstream.
  Recommendation: defer to the workbench stage (7) unless a Qwen user asks; the Turbo LoRA (P6) gives the 8-step
  preview speed without it. If ported, move `ConvRotInt8Linear` to `training/modules/` first.
- **Q6 - H3 pre-sm_80 paths (P10)?** Rule 8 says NVIDIA users count; a V100 contributor wrote and tested them
  upstream. Recommendation: port, unit-test on tiny models, and say in the docs that they are unrun here.
- **Q7 - Multi-reference edits (P9)?** Recommendation: port with the shared dataset change for Klein and Qwen
  together (one dataset mechanism, two `edit_references` values); Krea 2 has no edit training either side.
- FYI, no action: upstream pins Batch Size at 1 in its GUI (v7.2.0); TS's batching stays (settled).

---

## 12. What I did not check

- Nothing was run: no tests, no model loads, no GPU. Every verdict is from reading.
- The shared layer (`families/train.py` +512, `launch.py` +451, `form.py` +1145, `workbench.py` +438, `ft.py`,
  `multi_person.py`, `trigger_check.py`, `refmod.py`, `step_log.py`, `epochs.py`, `prep.py`) beyond the lines cited
  in section 9; the four new families; the web UI.
- `qwen_image21/driver.py` lines 112-160 (`ft_spec`) and 161-240 were taken as unchanged from the diff (the diff
  touches only `load_planned`, the three preview hooks and `generate`'s `dynamic_shift`); not re-read in full.
- Line numbers "about" in `minimax/embedder.py` and `minimax/model.py` are from the diff hunks, not the files.
- Whether `minimax/convrot.py`'s `parse_comfy_quant` / `dequantize_int8_convrot` have TS equivalents
  (`training/families/minimax_h3/weights.py` was not read); P11's size assumes they do not.
- How `tools/install.py` and the Preferences rows decide which files to fetch (P7, P8), and whether TS has any
  pref-migration mechanism a `replaces` field could use.
- Whether TS's `training/dataset.py` could take an extra-reference folder without a format change to the run
  folder's dataset TOML (P9).
- The exactness claim of H3's chunked prefill (causal attention over `past_key_values`) and the fp16 fallback's
  numerics - upstream's measurements on a V100 were taken as stated.
- Whether upstream's dataset caps extra edit references at 1 MP for Klein (`_own_shape` keeps the target's pixel
  count; the "capped at 1 MP" remark in `families/klein.py:169` reads as BFL's inference behaviour).
- `docs/QWEN_IMAGE.md`, `docs/KREA2.md`, `docs/KLEIN.md`, `docs/MINIMAX_H3.md` were read as diffs only; the
  `docs/drivers/*` additions (EXCERPTS.md, STILLS.md) not at all.
- Whether CUDA SDPA really falls back to the math kernel for heads wider than 256 (TS's wide-attention header says
  so; upstream gates on ROCm only, which suggests the author did not rely on it).
- The GPL headers upstream added to `minimax/model.py` and `vae.py`: TS's `THIRD_PARTY_NOTICES.md:66-76` already
  records the derivation; whether TS's own file headers say it was not re-read (`minimax_h3/model.py:1-11` names
  ComfyUI's loader and ConvRot, so at least partly).
