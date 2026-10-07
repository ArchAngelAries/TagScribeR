# Fizgig training system: implementation audit for TagScribeR native training

**Subject:** Fizgig (https://github.com/shootthesound/Fizgig, Apache-2.0, © 2026 Peter Neill), branch as checked out on 2026-10-02.
**Purpose:** TagScribeR will add native LoRA training. The project owner has made Fizgig's design the
reference for training features, presets (built-in and user-created), adaptive learning rate,
parameters, memory features and in-training sample previews, including when the work extends to
families Fizgig does not support (SDXL / Pony / Illustrious / NoobAI, and Anima). This document records
how Fizgig actually does each of these things, from the code.

**Method:** read-only source audit. Nothing was run. The GUI (`lora_trainer_gui.py`, ~35.4k lines /
~2 MB) was read with targeted greps and reads only. All paths are relative to the Fizgig root unless
stated otherwise. Line numbers are as of this checkout and will drift.

**Confidence notes:** if a statement is inferred rather than read directly, it is marked *(inferred)*.
If the code did not settle something, it is marked *(unverified)*.

---

## Contents

1. [Architecture overview: how a run flows](#1-architecture-overview-how-a-run-flows)
2. [The model "families" abstraction](#2-the-model-families-abstraction)
3. [Networks: LoRA / LoKR implementation and file format](#3-networks-lora--lokr-implementation-and-file-format)
4. [Presets, profiles, prefs and other persisted state](#4-presets-profiles-prefs-and-other-persisted-state)
5. [Adaptive learning rate and the per-image loss watch](#5-adaptive-learning-rate-and-the-per-image-loss-watch)
6. [Optimizers, schedulers, precision, memory, caching, ROCm](#6-optimizers-schedulers-precision-memory-caching-rocm)
7. [Dataset handling for training](#7-dataset-handling-for-training)
8. [In-training sample previews](#8-in-training-sample-previews)
9. [The GUI's training parameter surface](#9-the-guis-training-parameter-surface)
10. [Other notable subsystems](#10-other-notable-subsystems)
11. [Implications for TagScribeR](#11-implications-for-tagscriber)

---

## 1. Architecture overview: how a run flows

### 1.1 Three generations of trainer code

Fizgig has three trainer stacks, and they do not share much code. This is the most important
structural fact for reuse:

| Stack | Families | Entry point | Shape |
|---|---|---|---|
| **Klein trainer** (kohya/musubi lineage) | Flux 2 Klein Base 9B | `accelerate launch src/fizgig/scripts/train.py` → `src/fizgig/training/trainer.py` (`KleinTrainer`, 3276 lines) | Accelerate-based, argparse with ~200 flags, kohya-style `LoRANetwork` (`src/fizgig/networks/lora.py`) |
| **Per-family monoliths** | Krea 2, MiniMax H3 (+ RefMod) | `src/fizgig/scripts/krea2_train.py` → `src/fizgig/krea2/trainer.py` (3704 lines); `src/fizgig/scripts/minimax_train.py` → `src/fizgig/minimax/trainer.py` (5814 lines) | Plain single-process Python, its own loop, reuses `networks/lora.py` |
| **Standard layer ("families")** | Qwen Image 2.1 (the only described family so far) | `src/fizgig/families/cache.py` + `src/fizgig/families/train.py` (724 lines) | Small, generic, family-agnostic loop. All model-specific code sits behind a `FamilyDriver` |

`src/fizgig/families/registry.py:1-5` says outright that Klein, Krea 2 and H3 are "deliberately
absent" from the standard layer and "keep their existing code paths". The standard layer is the newest
design (September 2026), and it is the template TagScribeR should follow (see §11).

### 1.2 Process model

- **GUI:** Tkinter, single process (`lora_trainer_gui.py:1-16`). It does not import torch for training.
- **Training and caching run as child processes** started by `run_subprocess`
  (`lora_trainer_gui.py:32736-32800`):
  - `subprocess.Popen(cmd, stdout=PIPE, stderr=PIPE, text=True, bufsize=1, encoding='utf-8')`.
  - Env: `PYTHONIOENCODING=utf-8`, `PYTHONUNBUFFERED=1`, plus `CUDA_VISIBLE_DEVICES` from the GPU pref
    (`_cuda_env_for_subprocess`, `:10273`).
  - On Windows: `CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS`. The
    comment says below-normal priority lets DWM pre-empt the GPU and fixes desktop judder for ~1% speed
    cost. On POSIX: `preexec_fn=os.setsid`.
  - Two daemon threads read stdout and stderr line by line and marshal each line to the Tk thread with
    `master.after(0, update_console, line)`.
  - A third thread `wait()`s, then routes the exit code to the pause/queue state machine
    (`_pipeline_exit_routes_to_state_machine`, `:32719`) and fires the next pipeline stage's callback on
    the Tk thread.
- **Interpreter:** `_venv_python()` (`:33744`) uses `venv/Scripts/python.exe` when it exists, otherwise
  `sys.executable`. Klein alone goes through `venv/Scripts/accelerate.exe launch --num_cpu_threads_per_process 2 --mixed_precision bf16|fp16`
  (`:33313-33335`). Mixed precision is picked from the DiT filename: `fp16` in the name → fp16, else bf16.

### 1.3 End-to-end pipeline (Start Training)

`start_training` (`:32802`) → `_start_training_launch` (`:32884`):

1. **Re-entrancy:** if a process is already running, Start means *queue this run*
   (`_queue_current_run`).
2. **Validation:** `validate_inputs()` refuses to start when, for example, a media file has no caption
   (`:32695-32716`). It also checks disk headroom and whether a resume has epochs left.
3. A stale `.pause_requested` sentinel left by an earlier session is deleted.
4. `_save_last_train_settings()` writes `presets/.last_train_settings.json` (§4.6).
5. VRAM is freed: the workbench engines are unloaded (Repair Studio / Explorer / Royale), and so is the
   caption worker.
6. A legacy cache directory is cleared on a fresh (non-resume) run (`:32913-32925`). The variable
   behind it, `dataset_cache_dir_var`, is "legacy/back-compat — UI removed" (`:2013`), so this is
   normally a no-op. Real cache invalidation works per item, by staleness checks (§6.9).
7. **Block swap Auto** is resolved in the GUI for Klein and Krea 2 (`_parse_blocks_swap`, `:10071`).
   For MiniMax and described families the trainer resolves it (`--blocks_to_swap -1` / `auto`).
8. The Tk widgets are collected into `self.settings` (`:32990-33108`).
9. **Dataset TOML freeze (#98):** the live `dataset/Fizgig_train.toml` is copied to
   `dataset/run_snapshots/<lora_name>-<ms>.toml` (`_snapshot_dataset_config_for_run`, `:31841`).
   This stops Start-tab edits made during a run from retargeting it. Pruning keeps the newest 12, and
   older files go only once they are also >30 days old. On resume, the run's own snapshot is reused.
   `_verify_frozen_dataset_config` refuses to launch if the frozen TOML does not list the Start-tab
   folders.
10. Three commands are built: `build_training_command`, `build_cache_latents_command` and
    `build_cache_text_command` (`:33295`, `:33638`, `:33697`).
11. **Chain:** `cache latents → cache text → train`, each stage started from the previous stage's
    success callback (`:33166-33212`). On resume, or with "Enable Cache Preparation" off, the cache
    stages are skipped and training starts directly. A nonzero exit stops the pipeline.
12. `training_state = "running"`. The buttons become Pause / Stop / Queue Train.

### 1.4 GUI ↔ trainer communication

There is no socket or pipe protocol beyond stdout. Everything else is **files in the output directory**:

| Channel | Direction | File / mechanism | Code |
|---|---|---|---|
| Console log + progress | trainer → GUI | stdout/stderr lines. `training/progress.py` parses tqdm lines (`steps: N/M [.. <eta, x s/it, avr_loss=..]`) and `epoch N/M` into the progress card | `src/fizgig/training/progress.py:14-80` |
| Pause request | GUI → trainer | empty sentinel `<output_dir>/.pause_requested`, checked at each **epoch boundary** | GUI `_pause_training` `:34856`; families `train.py:536,592-599`; Klein `--pause_flag_path` (`:33457`) |
| Paused-state metadata | GUI internal | `<output_dir>/.fizgig_paused.json` sidecar | `_paused_sidecar_path` `:34814` |
| Live sample override | GUI → trainer | `<output_dir>/.sample_override.json` `{prompt, seed, width, height[, ref_image]}`, read at each preview | families `train.py:150-163`; Krea 2 `trainer.py:1208` |
| Problem images | trainer → GUI | `<output_dir>/loss_log/problem_images.json` (atomic tmp+replace), polled by the Problem Images window | `training/loss_logger.py:573-592` |
| Caption edits mid-run | GUI → trainer | `<output_dir>/loss_log/caption_updates.json`. The trainer claims it atomically (rename to `.processing`), re-encodes, and acks into `caption_updates_applied.json` | `families/loss_watch.py:157-241, 328-346` |
| Samples gallery | trainer → GUI | PNGs in `<output_dir>/sample/`. A GUI watcher thread rebuilds `files.json` every 5 s for a local HTTP gallery | `start_samples_watcher` `:15191`; `update_gallery_html` `:14910` |
| Per-image loss research log | trainer → disk | `loss_log/per_image_loss.jsonl` (rotated to `.bak` on a fresh run) | `training/loss_logger.py:67-129` |

### 1.5 Stop, pause, resume, checkpoints

- **Stop** (`stop_training`, `:35323`): on Windows, `taskkill /F /T /PID` (a hard kill of the process
  tree; CTRL_BREAK cannot work under CREATE_NO_WINDOW). On POSIX, `killpg(SIGTERM)`. Then
  `wait(timeout=5)`, then `kill()`. Nothing is saved.
- **Pause:** the GUI writes `.pause_requested`. At the end of the current epoch the trainer saves a full
  state dir, unless the cadence save already wrote one, and calls `sys.exit(0)`
  (`families/train.py:592-599`; Krea 2 `trainer.py:3577+`). The GUI sees the clean exit, records the
  paused state, and offers Resume. Resume relaunches with `--resume <name>-NNNNNN-state` and skips the
  cache stages.
- **Checkpoint files:**
  - Per epoch: `<name>-NNNNNN.safetensors` (6-digit epoch), saved when
    `done % save_every_n_epochs == 0` and `done < max_epochs`.
  - Final: `<name>.safetensors` (`families/train.py:581-606`).
  - Shared name helpers: `training/train_utils.py:15-19`.
- **State dirs:** `<name>-NNNNNN-state/`. Families contents (`families/train.py:79-93`):
  - `lora.safetensors` (fp32)
  - `optimizer.pt`
  - `ema.pt`
  - `rng.pt` (torch + cuda RNG)
  - `training_state.json`, written **last** as a commit marker: `{epoch, global_step, architecture,
    adaptive_lr_state}`
  - Klein instead uses `accelerator.save_state()` plus an `adaptive_lr_state.json` sidecar
    (`training/trainer.py:2462-2507`).
  - Pruning (`prune_state_dirs`, `train_utils.py:88-110`) keeps the newest `keep_last_n_states`. The
    count is clamped to ≥1, the regex is anchored to the output name, and each `rmtree` is guarded on
    its own.
- **Resume validation:** the state dir must contain `lora.safetensors`, `optimizer.pt` and
  `training_state.json`, and the LoRA must match at least one module, otherwise the error asks
  "different rank or target modules?" (`families/train.py:96-113`).
  - Resume restores optimizer state, RNG, EMA and the adaptive-LR counters.
  - The adaptive-LR **rollback snapshot is not persisted**, so the first epoch after a resume cannot
    roll back (`krea2/trainer.py:805-807`).
  - The step scheduler is fast-forwarded by calling `scheduler.step()` `global_step` times
    (`families/train.py:463-464`).
  - The data-order generator is reseeded with `seed + start_epoch` (`:535`).
- **Training queue:** `presets/training_queue.json` is a list of full settings snapshots
  (preset values + architecture + dataset folder + Samples-tab values). It is never auto-started at
  launch, and restoring an item replays the normal Start path (`:6820-6870`).

---

## 2. The model "families" abstraction

`src/fizgig/families/` is the standard layer. A new family consists of a **description** (facts), a
**driver** (model code) and its own model package. The module docstrings
(`families/driver.py:1-14`, `families/description.py:1-10`) say "nothing else in Fizgig changes to
add it."

### 2.1 Files and roles

| File | Role |
|---|---|
| `registry.py` | `FAMILIES = {d.key: d}` (currently just `QWEN_IMAGE_21`). Runs `validate()` at import and raises on inconsistency (`:11-16`). Lookups: `get`, `by_gui_label`, `by_arch_id` (cache filenames, metadata), `training_families()` (driver set), `workbench_families(tool)`, and `family_of_lora(path)`, which sniffs a LoRA header against each family's block map (`:50-70`). |
| `description.py` | Frozen dataclasses `ModelFile`, `SamplingSettings`, `SpeedLoRA`, `LoRAFormat` and `FamilyDescription` (all facts, each external value with a `source=`). `architecture_entry()` produces the legacy `ARCHITECTURES`-shaped dict the GUI expects (`:220-256`). `validate()` at `:258-285`. |
| `driver.py` | `FamilyDriver` base class (the interface, §2.3), plus `Block` / `BlockGroup`. |
| `train.py` | Generic LoRA/LoKR training loop (§2.5). |
| `cache.py` | Generic latent and text caching CLI (`--stage latents|text`). Stores `latent_{h}x{w}` and `cond__<key>` tensors (§6.9). |
| `lora.py` | `FamilyLoRA`: wraps a family's Linears with a multi-adapter `LoRALinear` (one trainable plus any number of frozen adapters), and provides bake/save/load (§3.2). |
| `quant.py` | INT8 / NF4 quantisation of the block-map Linears, `load_base`, and the Auto `plan()` for precision and block swap from free VRAM (§6.4). |
| `loss_watch.py` | Family-agnostic wrapper over `PerImageLossWatch`: detection, per-image LR, look warm-up, auto-recaption (§5.3). |
| `weight_profile.py` | Weight-only Profiler for any family: per-block sum of `‖up‖·‖down‖`, written as an HTML report plus a JSON sidecar. |
| `lorafile.py` | Reads LoRA files without the model: family, kohya or PEFT keys, plus LoKR. |
| `extract.py` | Weight-only rank reduction (exact QR+SVD of `B@A`; LoKR densely), saved in the family's own key format. |
| `workbench.py` | `WorkbenchEngine`: Repair Studio / Explorer / Royale for any family through the driver. Same protocol as the old per-family engines. |
| `qwen_image.py` | The `QWEN_IMAGE_21` description, including its presets (§4.2). |

### 2.2 `FamilyDescription` fields (the facts a family must declare)

From `families/description.py:89-167`:

```python
@dataclass(frozen=True)
class FamilyDescription:
    # identity
    key: str; arch_id: str; display_name: str; gui_label: str; lora_name_suffix: str
    aliases: tuple = (); experimental: bool = True
    # model files (Preferences rows)
    model_files: tuple = ()            # ModelFile(pref_key, label, required, repo, path, size_gb, note, local_name, role)
    text_encoder_label: str = ""; vae_label: str = ""
    # latent rules
    latent_channels: int = 16; spatial_factor: int = 8
    bucket_step: int = 64              # training buckets snap to this many pixels (must be multiple of spatial_factor)
    image_channels: int = 3; native_megapixels: float = 1.0
    # block layout
    n_blocks: int = 0; block_prefix: str = ""; block_note: str = ""
    # LoRA
    lora: Optional[LoRAFormat] = None
    driver: str = ""                   # "module.path:ClassName"; empty = not trainable yet
    modelspec_arch: str = ""; implementation: str = ""
    training_adapter: str = ""; training_adapter_note: str = ""
    ema_default: str = ""              # "0.98" | "Off" | "" (no control)
    precisions: tuple = ("bf16",)      # subset of bf16/int8/nf4
    train_memory: dict = {}            # {precision: (peak_GB or ((mp, GB),...), GB_saved_per_swapped_block)}
    optimizers: tuple = ("adamw8bit", "adamw"); network_types: tuple = ("lora",)
    edit_training: bool = False; edit_note: str = ""
    # sampling
    sampling: tuple = ()               # SamplingSettings(name, steps, cfg, sampler, scheduler, sigmas, options, negative_prompt, note, source)
    speed_loras: tuple = ()            # SpeedLoRA(name, repo, file, pairs_with, strength, settings, load_unmerged, pref_key, ...)
    preview_steps: int = 20; preview_cfg: float = 1.0; preview_width: int = 1024; preview_height: int = 1024
    preview_speed_lora: str = ""; preview_speed_steps: int = 0; preview_speed_strength: Optional[float] = None
    retired_preview_defaults: tuple = (); preview_reset: str = ""
    presets: tuple = ()                # ((name, {GUI_KEY: value}), ...) — first = applied on first visit
    helper_files: tuple = (); workbench: tuple = (); notes: tuple = ()
    train_script = "src/fizgig/families/train.py"; cache_script = "src/fizgig/families/cache.py"
```

`LoRAFormat` (`:62-86`):

```python
LoRAFormat(key_template, down, up, block_modules, alpha_key="{prefix}.alpha",
           kohya=False, file_prefix="", note, source)
module_of(key) -> dotted module | None
key(block, module, which)
```

`validate()` enforces the following (`:258-285`):

- identity fields are set and pref keys are unique;
- `n_blocks > 0` and `block_prefix` is set;
- a `lora` format is present and `sampling` is non-empty;
- `bucket_step % spatial_factor == 0`;
- each speed LoRA has repo, file and source;
- `driver` is in `module:Class` form;
- a trainable family has model files with the roles `dit`, `vae` and `text_encoder`;
- `training_adapter` names the file whose role is `training_adapter`;
- a trainable family sets `modelspec_arch` and `implementation`.

### 2.3 The `FamilyDriver` interface (exact signatures)

`families/driver.py:107-224`. Conventions (`:79-83`):

- Latents are `(C, h, w)` in the DiT's normalised space.
- Conditioning is a `dict` of CPU tensors per caption. The cache stores it verbatim and hands it back
  batched (leading dim 1).
- Images are `uint8 (H,W,3)` numpy arrays or PIL images.

```python
class FamilyDriver:
    description = None
    # models
    def load_dit(self, path: str, device): ...                     # bf16, frozen, LoRA-wrappable; quant applied later by families/quant.py
    def max_blocks_to_swap(self, dit=None) -> int: return 0         # 0 = no block swap
    def enable_block_swap(self, dit, num_blocks: int, device, supports_backward: bool = True) -> None: ...
    def block_swap_mode(self, dit, inference: bool) -> None: ...    # forward-only streaming for previews
    def load_vae(self, path: str, device): ...
    def load_text_encoder(self, path: str, device): ...             # encode-only; captioning is a shared job
    def unload_text_encoder(self, te) -> None: ...
    def enable_gradient_checkpointing(self, dit, on: bool = True) -> None: ...
    # encoding (generic cache calls these)
    def encode_images(self, vae, images: list) -> list: ...         # same-size uint8 arrays -> [(C,h,w)]
    def encode_text(self, te, captions: list) -> list: ...          # -> [dict of CPU tensors]
    # edit training (optional)
    supports_references = False
    def load_reference_text_encoder(self, path: str, device): ...
    def encode_text_with_references(self, te, captions: list, references: list) -> list: ...
    # training
    def training_loss(self, dit, latents, cond: dict, generator, *, min_t: float = 0.0,
                      max_t: float = 1.0, refs=None): ...           # -> (loss_tensor, {"t": float})
    # sampling
    def initial_noise(self, seed: int, width: int, height: int): ...
    def generate(self, dit, cond: dict, width: int, height: int, *, steps: int, seed: int, cfg: float = 1.0,
                 neg_cond=None, sigmas=None, options=(), noise=None, on_step=None, refs=None): ...  # -> latents
    def pad_conditioning(self, conds: list) -> list: ...            # optional (prompt travel)
    def decode(self, vae, latents, width: int, height: int): ...    # -> PIL.Image
    # LoRA + block map (defaults provided)
    def block_map(self, dit=None) -> list:  # [BlockGroup(label, [Block(id, label, modules)])]
    def lora_target_names(self, dit) -> list
    def block_of(self, module_name: str) -> Optional[str]
```

The **driver owns the training objective.** `training_loss` decides noise, target, timestep
distribution and forward. The loop never sees sigmas or schedulers. This is what makes the layer
extensible to epsilon- and v-prediction models such as SDXL (§11.3).

The default `block_map()` builds one group of `n_blocks` blocks from
`f"{block_prefix}.{i}.{m}" for m in lora.block_modules` (`:200-213`). The docstring explicitly
anticipates overriding it for "SDXL-style input/middle/output blocks" (`:101-102`, `:201-204`).

### 2.4 Shared vs per-family

| Shared (standard layer) | Per family (driver + description) |
|---|---|
| Dataset/bucketing (`fizgig.dataset`), caching CLI, state/resume, pause contract, Adaptive LR, step schedulers, EMA, grad clipping, the loss watch, previews orchestration (adapter on/off, EMA swap, low-VRAM parking, override, file naming), SAI metadata, LoRA wrapping/saving/baking, INT8/NF4 quantisation, Auto VRAM planner, Profiler/Extract/Repair/Explorer/Royale engines, GUI Training/Samples tabs | Model loading, VAE encode/decode, text encode, noise/timestep/target rules, forward, sampler, block swap, block map, LoRA key template, measured memory table, presets, sampling recipes, speed LoRA, training adapter |

### 2.5 The generic training loop (`families/train.py:280-607`)

```text
desc = registry.get(family); driver = desc.load_driver()
sample_* defaults from desc (speed LoRA recipe if --speed_lora)
lowmem = total VRAM < 20 GB  -> cap preview canvas at 768 px, park DiT for decode
dataset = BlueprintGenerator(ConfigSanitizer()).generate(toml, architecture=desc.arch_id) -> DatasetGroup
  -> requires batch_size == 1 (conditioning lengths differ per image)             (:333-336)
DataLoader(batch_size=1, shuffle=True, num_workers=0)
if precision=="auto" or blocks_to_swap<0: quant.plan(desc, driver, ..., megapixels from TOML resolution)
if previews: load TE -> encode prompts (+neg if cfg>1) -> unload TE -> load VAE (kept resident)
dit, swapped = quant.load_base(driver, dit_path, device, precision, blocks_to_swap)
driver.enable_gradient_checkpointing(dit, True)   # default True; not exposed on the CLI
net = FamilyLoRA(dit, driver)
  add_file(training_adapter, "training_adapter")  # frozen, ON in training, OFF in previews, never saved
  add_file(context_lora, "context")               # frozen, ON in training AND previews, never saved
  add_file(speed_lora, "speed_lora") -> disabled + parked on CPU except during previews
  add_trainable(rank, alpha, kind="lora"|"lokr", factor)
optimizer = create_optimizer(type, params, lr, args)
if owns_its_rate(optimizer) (Automagic v3): adaptive/per-image LR/look warm-up OFF, no scheduler
if adaptive_lr: lr = sqrt(min*max); AdaptiveLR(min,max)
else: LambdaLR step scheduler (constant|constant_with_warmup|cosine|cosine_with_restarts|linear|polynomial)
ema = EMAWeights(net, decay) if decay>0
resume -> restore lora/optimizer/rng/adaptive/ema
watch = loss_watch.Watch(...)
for epoch:
  for batch:
    if watch.excluded(batch): recorder.drop(i); continue
    loss, info = driver.training_loss(dit, latents, cond, gen, min_t, max_t, refs)
    zero_grad; (loss * watch.multiplier(batch)).backward()   # per-image LR = loss scaling at bs=1
    clip_grad_norm_(params, max_grad_norm); optimizer.step(); scheduler.step(); ema.update()
    recorder.add(loss); watch.observe(epoch+1, step, batch, info["t"], loss)
  adaptive.epoch_boundary(epoch, recorder.moving_average, net.trainable_modules(), optimizer)
  watch.boundary(epoch+1, dit, device)          # verdicts, caption repair + re-encode
  save cadence checkpoint (+ state, prune); previews if done % sample_every == 0; pause check
save final <name>.safetensors (+ state if save_state_on_train_end)
```

Things the generic loop does **not** have (all are present in the Klein and/or Krea 2 stacks):

- gradient accumulation;
- batch size > 1;
- step-based sampling or saving;
- noise offset;
- network dropout, rank dropout or LoRA+;
- block targeting from the GUI (`add_trainable(blocks=...)` exists but is not wired);
- regularisation images;
- caption dropout;
- TensorBoard / wandb logging.

### 2.6 How each existing family plugs in

| Family | Path | Text encoder | VAE / latents | Objective & timesteps | Notes |
|---|---|---|---|---|---|
| **Flux 2 Klein Base 9B** | Klein trainer (not a driver) | Qwen3-8B (`qwen_3_8b.safetensors`) | FLUX.2 AE `ae.safetensors`, latents /16 (2×2 packed after /8) (`dataset/image_dataset.py:129`) | Flow matching; `timestep_sampling` modes `sigma, uniform, sigmoid, shift, flux_shift, flux2_shift, logsnr, qinglong_flux`. Klein default `flux2_shift`: `mu = lin(256→0.5, 4096→1.15)(h*w)`, `shift = e^mu`, `t = σ(randn·s)`, `t = t·shift/(1+(shift−1)t)` (`training/trainer.py:791-900`) | Trains on Base, previews on Distilled (4 steps). Block targeting via `include_patterns`. Context LoRA. fp8 / NF4 base. |
| **Krea 2** (12.9B single-stream MMDiT) | Krea 2 monolith | Qwen3-VL-4B (fp8 or bf16, ComfyUI layout). Cache stores the **multi-layer** stack `(seq, 12 layers, dim)` (layers 2,5,…,35) plus a validity mask (`krea2/caching.py:1-11`, `krea2/embedder.py:168`) | Qwen-Image VAE, 16 ch, /8, frame axis squeezed | Flow matching, `noised = (1−t)x0 + t·noise`, target `noise − x0`, logit-normal t with resolution shift `mu = lin(256→0.5, 6400→1.15)(tokens)`. Optional window `[min_t,max_t]` is **rescaled into** the window, not clamped (`krea2/trainer.py:464-487, 490-592`) | Trains RAW. Previews use the RAW model with the Turbo LoRA at 1.0, or the fp8 Turbo checkpoint. The same Qwen3-VL-4B is Fizgig's shared **captioner** for all families. |
| **MiniMax H3** (33B omni DiT, photos/clips/sound/voice) | MiniMax monolith | Qwen3-VL-32B | MiniMax video VAE (16× spatial), audio VAE | Flow matching. Target is `x0 − noise` (sign convention matched to ComfyUI); "low-noise %" dial maps to shift via `shift = (1−P)/P` (`minimax/trainer.py:1-14`, GUI `:518-540`) | NF4/INT8/HQQ base, TREAD token routing, caption dropout 0.05, frozen training adapters (circlestone/Ostris), RefMods, multi-concept, Automagic v3 default. |
| **Qwen Image 2.1** | **Standard layer** (`qwen_image21/driver.py`) | Qwen3-VL-8B. Last layer **before** the final RMSNorm, system tokens dropped → `{"hidden_states": (L, 4096)}`. bf16 if free VRAM ≥ 19.5 GB, else INT8 (`driver.py:46-51`) | Own RGBA VAE, 64 ch, /16, normalised with latents_mean/std. Bucket step 32 | Flow matching. `t = σ(randn)`, exponential shift `t = e^mu/(e^mu + 1/t − 1)` with `mu = calculate_mu(tokens)`, then mapped to `[min_t,max_t]`. MSE on `noise − x0` over image tokens; block-causal joint forward (`driver.py:102-127`) | Frozen Fizgig training adapter (on in training, off in previews). Viggle turbo LoRA (unmerged) for fast previews. Edit LoRAs from before/after pairs (refs as vision tokens + clean latents). |

---

## 3. Networks: LoRA / LoKR implementation and file format

### 3.1 kohya-style `LoRANetwork` (Klein, Krea 2, MiniMax): `src/fizgig/networks/lora.py`

- **`LoRAModule`** (`:23-174`) replaces the `forward` of the original Linear or Conv2d (monkey-patch,
  not module replacement). `lora_down` uses Kaiming-uniform init with `a=√5`; `lora_up` is zero-init.
  - Scale: `scale = alpha / rank`. If `alpha` is 0 or None, `alpha = rank` (`:93-103`).
  - Forward: `org(x) + up(down(x)) · multiplier · scale`, with optional dropout, rank dropout (rescaled
    by `1/(1−p)`) and module dropout.
  - `split_dims` mimics split q/k/v on a fused Linear.
  - Rank is deliberately **not** capped at `min(in,out)`. The comment cites measured faster
    optimisation from over-complete factorisations (`:54-62`).
- **LoRA+**: `prepare_optimizer_params` puts `lora_up` in a "plus" group at `lr × loraplus_ratio`
  (`:1118-1164`). The GUI keeps `LORA_LR_RATIO` hidden at 1.
- **`LoKRModule`** (trainable LoKR, `:382-460`):
  - `w1 (a,b)` small, `w2 (c,d)` full-matrix; `a,c = factorization(out, factor)` and
    `b,d = factorization(in, factor)`, where `factorization` picks the largest divisor ≤ factor, the
    LyCORIS convention (`:366-379`).
  - Init: `w1` Kaiming, `w2` zero, so the delta is exactly 0 at step 0. `alpha = 1`, `scale = 1`.
  - The Kronecker product is never materialised. The update is computed as `vec(w2 · X · w1ᵀ)`
    (`_lokr_forward_update`, `:351-363`).
  - Dropout and rank dropout do not apply.
- **Inference-only**: `LoRAInfModule`, `LoKRInfModule`, `LoHaInfModule` (`:176`, `:463`, `:569`).
  **LoHa can be loaded but not trained. DoRA is not supported**: `.dora_scale` keys are skipped with a
  comment (`:1520`, `:1748-1749`).
- **Format detection and conversion:** `detect_lora_format`, `peft_to_kohya`, diffusers-Flux and
  diffusers-Krea2 converters, `ensure_kohya_lora_state_dict`, `lora_family_from_file` /
  `assert_lora_family_matches` (`:1310-1938`). Fizgig loads kohya, PEFT, OneTrainer, AI-Toolkit and
  LyCORIS files.
- **Save** (`save_weights`, `:1182-1220`): safetensors with `sshs_model_hash` / `sshs_legacy_hash`. The
  hashes are skipped on MemoryError or PanicException so a checkpoint is never lost over optional
  metadata.
- **Klein targets** (`networks/lora_klein.py:18-59`):
  - modules: `DoubleStreamBlock`, `SingleStreamBlock`, prefix `lora_unet`;
  - excluded: `.*(img_mod\.lin|txt_mod\.lin|modulation\.lin).*` and `.*(norm).*`;
  - Model Area presets add `include_patterns` regexes (GUI `:33372-33405`):
    - Identity: `single_blocks.(1-16)`
    - Style / Style+Comp: `double_blocks.*` + `single_blocks.[01]`
    - Details: `single_blocks.(12-23)`
    - Custom: per-block ticks
- **Krea 2 targets:** every `nn.Linear` in the DiT (`create_network(None, "lora_unet", …)`,
  `krea2/trainer.py:216-238`), described as "all-Linear", 264 modules.
  - Stacking order: Turbo LoRA (innermost, disabled except for previews), then Context LoRA, then the
    trainable LoRA (`:200-214`).
  - `torch.compile` runs per block **after** the LoRAs patch the forwards.

### 3.2 The standard layer's `FamilyLoRA` (`src/fizgig/families/lora.py`)

- `LoRALinear(base)` holds an `nn.ModuleDict` of adapters and computes
  `W x + Σ_i s_i · B_i(A_i(x))` (`:65-104`).
  - `LoRAFactor` is a subclass of `nn.Linear`, so block-swap offloaders, which stream modules whose
    class name ends in "Linear", leave the adapters resident (`:27-29`).
  - Trainable adapters are **fp32**; frozen adapters are **bf16** (`:82`). The input is cast to the
    adapter dtype and the result cast back.
  - LoKR is computed in the input's dtype (bf16) for speed, with fp32 weights. The note records 7.46
    vs 2.04 s/step in fp32 (`:53-59`).
- Frozen adapter scale = `alpha/rank · load_strength · block_strength · on · block_on`. Modules outside
  the block map follow load strength and on/off only (`:7-11`, `:253-259`).
- `read_file()` accepts any common layout: the family's own keys, kohya `lora_unet_<flattened>`, PEFT
  `lora_A/B` or `lora_down/up` (bare or under `transformer.` / `diffusion_model.` /
  `model.diffusion_model.` / `base_model.model.`), and LyCORIS LoKR (low-rank factors multiplied out,
  LyCORIS scale rule). **LoHa is refused in the standard layer** (`:188-223`).
- `bake(names)` turns any live combination into one standard LoRA by rank-concatenation, with scales
  folded into the up weights and `alpha = total rank`. A lone LoKR stays LoKR. A LoKR mixed with a LoRA
  is SVD'd to rank ≤ 64 (`:336-374`).
- `state_dict()` / `save()`. LoRA: `{file_prefix}{module}.{down}.weight`,
  `{file_prefix}{module}.{up}.weight`, alpha `= self.alpha`. LoKR: `{prefix}{module}.lokr_w1`,
  `.lokr_w2`, `.alpha = 1.0` (`:377-396`). Saved in **bf16** by default; state-dir copies in fp32.

### 3.3 Key formats and ComfyUI compatibility

| Family | Key example | Why |
|---|---|---|
| Klein / Krea 2 / H3 (LoRA) | `lora_unet_<dotted_path_with_underscores>.lora_down.weight`, `.lora_up.weight`, `.alpha` | kohya convention ComfyUI loads |
| Krea 2 LoKR final save | `diffusion_model.<dotted>.lokr_w1/.lokr_w2/.alpha` (LyCORIS standard) — state dirs keep native names (`krea2/trainer.py:1137-1179`) | "the format every ComfyUI LoKR in the wild uses" |
| Qwen Image 2.1 | `transformer.transformer_blocks.{i}.{module}.lora_A.weight` / `lora_B` / `.alpha`, modules `attn.to_q/to_k/to_v/to_out.0`, `img_mlp.gate_layer/proj/out` (`families/qwen_image.py:80-93`) | **Approved exception** to the kohya rule: ComfyUI maps `gate_layer`/`proj` onto its fused `gate_up` only for bare or `transformer.`-prefixed keys. `lora_unet_` keys silently drop the MLP input (`:88-91`) |

The lesson Fizgig recorded, which TagScribeR should keep: **the key format is a per-family fact, checked
against ComfyUI's `comfy/lora.py` mapping**, and it lives in the description (`LoRAFormat.source`).

### 3.4 Metadata

- `training/metadata.py:96-200` `build_metadata()` writes **SAI ModelSpec 1.0.0**:
  - `modelspec.architecture = "<arch>/lora"` (e.g. `Flux.2-klein-9b/lora`, `Krea-2/lora`,
    `MiniMax-H3/lora`, or `desc.modelspec_arch` for described families);
  - `implementation`, `title`, `date`, `resolution`;
  - optional `author`, `description`, `license`, `tags`, `merged_from`, `trigger_phrase`, `thumbnail`
    (base64 data URI of the newest sample, ≤512 px JPEG), `usage_hint` (auto
    "Include '<trigger>' in your prompt.").
- Families add `ss_network_module`, `ss_network_dim` (factor for LoKR), `ss_network_alpha`,
  `ss_lokr_factor`, `ss_architecture`, `ss_epoch`, `ss_optimizer`, `ss_learning_rate`,
  `ss_training_adapter`, and `ss_context_lora[_strength]` (`families/train.py:468-488`).
- The description defaults to the last preview prompt.
- Each epoch checkpoint's thumbnail is refreshed with **its own** preview once the preview exists
  (`refresh_checkpoint_thumbnail`, `metadata.py:240`; Krea 2 `trainer.py:3528-3534`).
- **Cache-id gotcha:** arch ids used in cache filenames must contain **no underscore**
  (`klein9b`, `krea2`, `minimaxh3`, `qwenimage21`), because the cache filename is parsed with
  `split("_")` (`metadata.py:22-27`).

---

## 4. Presets, profiles, prefs and other persisted state

### 4.1 What each persisted file is

| File | Location | Content | Writer / reader |
|---|---|---|---|
| **Built-in presets** | in code | `BUILT_IN_PRESETS` (Klein), `KREA2_BUILT_IN_PRESETS`, `MINIMAX_BUILT_IN_PRESETS`, `REFMOD_BUILT_IN_PRESETS` in `lora_trainer_gui.py:857-1310`; described families in `FamilyDescription.presets` | `_builtins_for_arch` (`:6403-6415`) |
| **User presets** | `presets/<architecture label>/<name>.json` (e.g. `presets/Krea 2/`, `presets/Flux 2 Klein Base 9B/`) | flat JSON of GUI setting keys → values (same keys as built-ins) | `save_custom_preset` `:7765`, `load_custom_preset` `:7812`, `delete_custom_preset` `:7845` |
| **Reset-to-defaults preset** | in code | `PRESETS = {"Flux 2 Klein Base 9B": {...}}` (`:1877-1898`) | `load_default_preset` (`:6446`) |
| **Last launch snapshot** | `presets/.last_train_settings.json` | `_collect_preset_values()` + `__architecture__` + `__minimax_train_base__` | `_save_last_train_settings` `:6766`; "Load Settings From Last Train" `:6789` |
| **Training queue** | `presets/training_queue.json` | list of `{preset snapshot, architecture, dataset folder, samples values}` | `:6834-6870` |
| **Repair Studio presets** | `presets/repair_studio/<family>/<name>.json` | slider states | `_repair_preset_dir` `:31578` |
| **prefs.json** | Fizgig root | **model file paths** (per family pref keys such as `krea2_raw_dit`, `qwen21_dit`, …), dirs (`lora_output_dir`, `profiles_dir`, `cache_dir`, stored relative when inside the repo), inference block swap / INT8, `cuda_device` (GPU UUID), RunPod settings, `caption_qwen_instructions` overrides | `load_prefs` `:1663`, `save_prefs` `:1841`, `DEFAULT_PREFS` `:1485` |
| **.last_used.json** | Fizgig root | folders (image folder, prep source, concept folders), caption trigger, sample prompt, `lora_output_dirs` **per family**, `architecture`, Krea 2 preview engine, prep MP, … (atomic tmp+replace) | `load_last_used` `:1356`, `save_last_used` `:1385` |
| **profiles/** | `prefs["profiles_dir"]` | **Profiler HTML reports + JSON sidecars**, not training profiles | Profiler tab (`:21148-21335`) |
| **Dataset TOML** | `dataset/Fizgig_train.toml` (live), `dataset/run_snapshots/*.toml` (frozen per run) | §7 | `auto_save_dataset_config_silent` `:31819` |
| **Dataset-side files** | in the image folder | `fizgig_look_scores.json` (Look Filter), `fizgig_excluded.json` (persistent loss-watch exclusions, each entry snapshots the caption and auto-prunes when the caption changes) | §5.3 |

So "profiles" in Fizgig means *LoRA block-activation profiles*. There is no separate "training profile"
concept. Presets are the only training-settings bundles.

### 4.2 Built-in presets and their key values

**Klein** (`lora_trainer_gui.py:857-907`). All use `SAVE_EVERY_N_EPOCHS=1`, `SEED=42` and
`OPTIMIZER_TYPE=adamw8bit`.

| Preset | Rank/α | LR box | Epochs | Adaptive (min–max) | Model area | Timesteps |
|---|---|---|---|---|---|---|
| ✨ Old Reliable (rank 16, full model, single subject) | 16/16 | 1e-4 | 55 | on, 1e-4–4e-4 | Full Model | all |
| ✨ Old Reliable – Flavour 8 | 8/8 | 1e-4 | 55 | on, 1e-4–4e-4 | Full Model | all |
| ✨ Identity (rank 4, single subject) | 4/4 | 4e-4 | 15 | on, 2e-4–4e-4 | Identity | all |
| ✨ Identity (rank 8, harder dataset) | 8/8 | 4e-4 | 20 | on, 2e-4–4e-4 | Identity | all |
| ✨ Multi-Character (rank 16) | 16/16 | 2e-4 | 50 | on, 1e-4–4e-4 | Identity | all |
| ✨ Style (late timesteps) | 4/4 | 4e-4 | 15 | on, 1e-5–4e-4 | Style | MIN 0 / MAX 400 |
| ✨ Style+Composition (all timesteps) | 4/4 | 4e-4 | 15 | on, 1e-5–4e-4 | Style+Composition | all |

Klein "Reset Defaults" (`PRESETS`, `:1877-1898`): LR 4e-4, rank/α 4/4, 12 epochs, adamw8bit,
`flux2_shift`, blocks swap auto, FP8 + scaled on, NF4 off, compile auto, gradient checkpointing on.

**Krea 2** (`:913-984`). All use `NETWORK_TYPE="LoRA (standard)"`, `OPTIMIZER_TYPE=adamw8bit`,
`GRADIENT_ACCUMULATION=1`, `MAX_GRAD_NORM=1.0`, `DATASET_MEGAPIXELS="0.25"`,
`BLOCKS_SWAP="Auto (detect from GPU)"`, `QUANT_4BIT_MODE=auto`, `COMPILE_BLOCKS=Auto`,
`KREA2_LOSS_WATCH=True`, `KREA2_PER_IMAGE_LR=True`, auto-recaption and look warm-up off, and
`KREA2_EMA="0.98 (recommended)"` (`:1004-1005`).

| Preset | Rank | Epochs | LR |
|---|---|---|---|
| ✨ Krea 2 Ultra Fast (rank 8, adaptive LR) — **default** | 8 | 30 | adaptive 2e-4–4e-4 (start = geometric mid 2.83e-4) |
| ✨ Krea 2 Standard (rank 32, full model) | 32 | 64 | flat 1e-4 (adaptive off) |
| ✨ Krea 2 Style (rank 16, gentle LR) | 16 | 64 | adaptive 5e-5–2e-4 (start = 1e-4) |

**MiniMax H3** (`:1007-1171`). Fast is re-inserted first, so it is the default.
- ✨ MiniMax H3 Fast (LoRA 8, 50 epochs):
  - LoRA 8/8, `automagic3` with start LR 1e-6, adaptive off, 0.25 MP;
  - low-noise 60%, high-noise LR 100%;
  - Fast training mode (blocks 20–49), circlestone training adapter;
  - TREAD on, clip still on, EMA 0.98, caption dropout 0.05;
  - block limit off, AdaLN off, refiner off, distill off.
- ✨ MiniMax H3 (rank 16, 60 epochs): the same with rank 16 and 60 epochs.
- ✨ MiniMax H3 Style (LoRA 8): Fast, with clip still off.

**Qwen Image 2.1** (`families/qwen_image.py:17-31, 176-197`). `_preset()` sets:
- 0.5 MP, adamw8bit, 30 epochs, save every epoch, seed 42, grad-acc 1, max-norm 1.0;
- block swap Auto, precision Auto, training adapter on, EMA 0.98;
- loss watch on, per-image LR **off**, auto-recaption off, warm-up off.

| Preset | Rank | LR | Epochs | Edit |
|---|---|---|---|---|
| ✨ Qwen 2.1 Fast (rank 8, adaptive LR) — default | 8 | adaptive 2e-4–4e-4 | 30 | no |
| ✨ Qwen 2.1 Standard (rank 16, adaptive LR) | 16 | adaptive 1e-4–2e-4 | 30 | no |
| ✨ Qwen 2.1 Style (rank 16, 1.5e-4) | 16 | flat 1.5e-4 | 30 | no |
| ✨ Qwen 2.1 Edit (rank 8, adaptive LR) | 8 | adaptive 2e-4–4e-4 | 12 | **yes** (`FAMILY_EDIT`) |
| ✨ Qwen 2.1 Edit Strong (rank 16, adaptive LR) | 16 | adaptive 1e-4–2e-4 | 12 | yes |

The "adaptive 2e-4 to 4e-4" wording means `ADAPTIVE_LR=True, ADAPTIVE_LR_MIN="2e-4", ADAPTIVE_LR_MAX="4e-4"`.
The LR box is then ignored and the run starts at √(2e-4·4e-4) ≈ 2.83e-4 (§5.1).

### 4.3 Preset lifecycle (create / save / load / validate)

- **First visit to a family** applies its **first** built-in preset. A later visit within the same
  session restores that family's last values (`_arch_settings_memory`, `:13440-13495`). The LoRA name
  suffix is re-tagged after the preset is applied (`_apply_lora_name_suffix`, `:13398`).
- **Save** (`:7765-7810`):
  - asks for a name and rejects `<>:"/\|?*` and empty names;
  - confirms before overwriting;
  - writes `json.dump(self._collect_preset_values(), indent=4)` to
    `presets/<arch>/<name>.json`.
  - Presets deliberately **do not carry the architecture**, so a Krea 2 preset cannot switch your model.
    Only the last-train snapshot and queue items store it, as namespaced `__architecture__`.
- **What gets collected** (`_collect_preset_values`, `:7666-7763`):
  - every `self.entries` widget **except** `_NON_TRAINING_ENTRY_KEYS` (all `SAMPLE_*`, MiniMax turbo,
    `RESUME_TRAINING`; `:7484-7491`);
  - `FAMILY_*` keys only for described families (`:7494-7496`);
  - every Training-tab Boolean/StringVar held outside `entries` (FP8, SCALED, QUANT_4BIT,
    QUANT_4BIT_MODE stored as the **canonical key**, not the display label, COMPILE_BLOCKS, SAVE_STATE*,
    KREA2_* toggles, FT settings, reg dirs, GRADIENT_CHECKPOINTING, ADAPTIVE_LR, TARGET_LAYERS,
    TIMESTEP_SAMPLING, WEIGHTING_SCHEME, dataset bucket/caption/MP/batch, TRAINING_BLOCKS dict,
    MINIMAX_CONCEPT_DIRS list).
  - Sample settings are therefore **not** part of a preset.
- **Load** (`load_custom_preset`, `:7812`): built-ins are looked up first, then the disk file.
  - `json.JSONDecodeError` → "Preset file is corrupted".
  - A user preset with the same name as a built-in appears once in the combo (`:6432-6444`).
- **Validation on apply** (`_apply_preset_values_inner`, `:6482-6560`). This is the part to copy:
  - Unknown keys are **silently ignored** (only keys present in `self.entries` or the explicit vars are
    applied). That gives forward/backward compatibility.
  - A combobox value matches its option by **first token**, so a saved `"2e-4"` selects
    `"2e-4 - rank 4/8 only"`.
  - `_STRICT_COMBO_KEYS` (`OPTIMIZER_TYPE`, `ADAPTIVE_LR_MIN/MAX`, `LR_SCHEDULER`, `NETWORK_TYPE`, RefMod
    boxes; `:6463-6470`) and `_STRICT_VAR_OPTIONS`: a value the current family does not offer is
    **rejected** with a console line `[preset] KEY: saved value … isn't offered here — keeping …`,
    rather than being set onto a readonly combobox where it would fail at launch.
  - `TIMESTEP_SAMPLING` is validated against the trainer's accepted list.
  - Legacy migrations: e.g. `MINIMAX_LIKENESS_OPT` → `MINIMAX_LIKENESS_MODE`; old architecture labels
    via `_ARCH_ALIASES` (`:472-478`).
  - After applying, dependent UI is refreshed (network type row swap, timestep fields,
    Automagic gating).

### 4.4 Every parameter a preset can carry

Defaults come from `self.settings` (`lora_trainer_gui.py:2083-2211`) and widget initialisers. Values
are stored as the widget's text, so most numbers are serialised as strings in user presets (see the
real snapshot in `presets/.last_train_settings.json`). "Fam" lists the families that read the key:
K = Klein, K2 = Krea 2, H3 = MiniMax H3, Q = Qwen Image 2.1 (standard layer).

**Core training**

| Key | Type | Default | Meaning | Fam |
|---|---|---|---|---|
| `LORA_OUTPUT_DIR` | path | `output_loras` (per-family memory) | output folder | all |
| `LORA_NAME` | str | `LoraName_TokenName_k9b` | output name; suffix swapped per family | all |
| `MODEL_TYPE` | str | "" | legacy (Wan task), unused | — |
| `LEARNING_RATE` | float | 4e-4 | LR (ignored under Adaptive LR; Automagic start) | all |
| `ADAPTIVE_LR` | bool | False | enable bi-directional plateau tracker | K, K2, Q (H3 via its trainer) |
| `ADAPTIVE_LR_MIN` | combo str | "1e-5" | options `1e-5, 5e-5, 1e-4, "2e-4 - rank 4/8 only", "3e-4 - low-rank only"` | same |
| `ADAPTIVE_LR_MAX` | combo str | "4e-4" | options `1e-4, 2e-4, 3e-4, 4e-4` | same |
| `LORA_LR_RATIO` | int | 1 | LoRA+ ratio for `lora_up` (hidden) | K |
| `NETWORK_DIM` | int | 4 | rank | all |
| `NETWORK_ALPHA` | float | 4 | alpha (scale = α/r) | all |
| `NETWORK_TYPE` | combo | "LoRA (standard)" | or "LoKR (Kronecker)" | K2, H3, Q |
| `LOKR_FACTOR` | int | 8 | LoKR factor (w1 ≈ f×f) | K2, H3, Q |
| `MAX_TRAIN_EPOCHS` | int | 12 | epochs | all |
| `SAVE_EVERY_N_EPOCHS` | int | 1 | checkpoint cadence | all |
| `SEED` | int | 42 | training seed | all |
| `TARGET_LAYERS` | str | "Full Model" | Model Area: Full / Identity / Style / Style+Composition / Details / Custom | K |
| `TRAINING_BLOCKS` | dict[str,bool] | all false | per-block ticks for Custom (`double_blocks.N`, `single_blocks.N`) | K |
| `CONTEXT_LORA_PATH` | path | "" | frozen + active LoRA to train on top of | K, K2, Q |
| `CONTEXT_LORA_STRENGTH` | str float | "1.0" | its strength | same |
| `RESUME_TRAINING` | path | "" | state dir (not saved in presets) | all |

**Optimizer and schedule**

| Key | Type | Default | Meaning | Fam |
|---|---|---|---|---|
| `OPTIMIZER_TYPE` | str | "adamw8bit" | see §6.1 | all |
| `OPTIMIZER_ARGS` | str | "" | `key=value key=value` (`ast.literal_eval`) | all |
| `GRADIENT_ACCUMULATION` | int | 1 | micro-batches per step | K, K2, H3 (not Q) |
| `MAX_GRAD_NORM` | float | 1.0 | clip norm, 0 = off | all |
| `NETWORK_DROPOUT` | float | 0 | LoRA neuron dropout | K |
| `LR_SCHEDULER` | str | "constant" | constant / constant_with_warmup / cosine / cosine_with_restarts / linear / polynomial (Klein accepts more) | all |
| `LR_WARMUP_STEPS` | str int | "" | warmup steps | all (ignored under adaptive) |
| `LR_DECAY_STEPS` | str int | "" | Klein only | K |

**Memory and precision**

| Key | Type | Default | Meaning | Fam |
|---|---|---|---|---|
| `BLOCKS_SWAP` | combo | "auto" / "Auto (detect from GPU)" | blocks streamed CPU↔GPU; Auto rules §6.5 | all |
| `FP8` / `SCALED` | bool | True / True | fp8 base / scaled fp8 | K |
| `QUANT_4BIT` | bool | False | NF4 base | K, K2 |
| `QUANT_4BIT_MODE` | str | "auto" | canonical base-precision key (auto/nf4/int8/fp8/no_4bit) | K, K2 |
| `FAMILY_PRECISION` | combo | "Auto (fits your free VRAM)" | bf16 / INT8 / 4-bit NF4 | Q |
| `COMPILE_BLOCKS` | str | "auto" | Auto / On / Off (/outside) | K2 |
| `GRADIENT_CHECKPOINTING` | bool | True | | K (always on for others) |
| `FP8_TEXT_ENCODER` | bool | True | fp8 text encoder | K |
| `ATTENTION_MECHANISM` | str | "sdpa" | sdpa / flash_attn / xformers / … | K |
| `IMG_IN_TXT_IN_OFFLOADING` | bool | False | no-op for Klein | K |
| `SAVE_STATE` | bool | True | state dir at each checkpoint | all |
| `SAVE_STATE_ON_TRAIN_END` | bool | True | final state | all |
| `KEEP_LAST_N_STATES` | str int | 2 | prune older states | all |

**Timesteps and noise schedule**

| Key | Type | Default | Meaning | Fam |
|---|---|---|---|---|
| `TIMESTEP_SAMPLING` | str | "shift" (Klein preset flux2_shift) | | K |
| `DISCRETE_FLOW_SHIFT` | str | "3.0" | fixed shift for `shift` | K |
| `SIGMOID_SCALE` | str | "1.0" | randn scale before sigmoid | K |
| `MIN_TIMESTEP` / `MAX_TIMESTEP` | str | "" | Klein: 0–1000 ints; K2/Q: 0–1 floats | K, K2, Q |
| `PRESERVE_DISTRIBUTION` | bool | False | rescale (not clip) into the window | K |
| `WEIGHTING_SCHEME` | str | "none" | none / logit_normal / mode / … (SD3-style) | K |
| `LOGIT_MEAN` / `LOGIT_STD` / `MODE_SCALE` | str | 0.0 / 1.0 / 1.29 | | K |

**Logging and metadata**

| Key | Type | Default | Meaning | Fam |
|---|---|---|---|---|
| `LOGGING_DIR` / `LOG_WITH` / `LOG_PREFIX` | str | "" / "none" / "" | TensorBoard / wandb | K |
| `METADATA_TITLE/AUTHOR/DESCRIPTION/LICENSE/TAGS/TRIGGER_PHRASE/THUMBNAIL` | str | "" | ModelSpec fields; trigger falls back to the Captions-tab trigger; thumbnail "off"/"none" disables | all |

**Dataset**

| Key | Type | Default | Meaning | Fam |
|---|---|---|---|---|
| `ENABLE_CACHE` | bool | True | run cache stages before training | all |
| `ENABLE_BUCKET` / `BUCKET_NO_UPSCALE` | bool | True / True | aspect bucketing / never upscale | all |
| `DATASET_CAPTION_EXT` | str | ".txt" | caption extension | all |
| `DATASET_MEGAPIXELS` | str | "0.25" | Target MP → `resolution=[s,s]`, `s=floor(√(MP·1e6)/16)·16` | all |
| `DATASET_BATCH_SIZE` | str | "1" | batch size (Q and the loss watch need 1) | all |

**Loss watch, EMA and families**

| Key | Type | Default | Meaning | Fam |
|---|---|---|---|---|
| `KREA2_LOSS_WATCH` | bool | False (presets True) | detect problem images | K2, Q |
| `KREA2_PER_IMAGE_LR` | bool | False | per-image LR multipliers | K2, Q |
| `KREA2_AUTO_RECAPTION` | bool | False | Qwen3-VL recaption stuck images | K2, Q |
| `KREA2_WARMUP_LOOK` | bool | False | Look-Filter outlier warm-up | K2, Q |
| `KREA2_EMA` | combo | "0.98 (recommended)" | EMA decay or Off | K2 |
| `FAMILY_TRAINING_ADAPTER` | bool | True | use description's training adapter | Q |
| `FAMILY_EMA` | combo | `desc.ema_default` | EMA decay | Q |
| `FAMILY_TURBO_STRENGTH` | str | desc preview speed strength | speed LoRA strength for previews (0 = not loaded) | Q (Samples tab) |
| `FAMILY_EDIT` / `FAMILY_EDIT_DIR` / `FAMILY_EDIT_REF` / `FAMILY_EDIT_CAPTION` | bool/path/path/str | False/""/""/"" | Edit LoRA, originals folder, preview test photo, edit instruction | Q |

**Krea 2 full fine-tune**

| Key | Type | Default | Meaning | Fam |
|---|---|---|---|---|
| `KREA2_FINETUNE`, `KREA2_FAST_FT`, `KREA2_FT_MODE`, `KREA2_FT_BLOCKS`, `KREA2_FT_EVERY`, `KREA2_FT_FUSED`, `KREA2_REG_DIR`, `KREA2_REG_MULT` | various | off, "Auto (by VRAM)", 14, 1, True, "", 0.2 | rotating-block full fine-tune and its regularisation set | K2 |

**MiniMax H3**

| Key | Type | Default | Meaning | Fam |
|---|---|---|---|---|
| `MINIMAX_*` (~30 keys: `LOWNOISE_PCT`, `HIGHNOISE_LR_PCT`, `BLOCKS`, `BASE_QUANT`, `TRAIN_ADALN`, `TRAIN_REFINER`, `LIKENESS_MODE`, `ADAPTER`, `ADAPTER_RAMP`, `TREAD`, `CLIP_STILL`, `EMA`, `CAPTION_DROPOUT`, `SLOW_BLOCKS`, `SLOW_LR_SCALE`, `BLOCK_LIMIT`, `LR_WARMUP`, `DISTILL*`, `MULTICONCEPT`, `CONCEPT_DIRS`, `FINETUNE`, `FT_*`, `REG_*`, `MIXED_STOP_*`) | various | see `:2105-2143` | H3-specific | H3 |
| `MINIMAX_REFMOD_*` | various | `REFMOD_DEFAULTS` `:1299-1309` | RefMod maker | RefMod |

**Not in presets (Samples tab; captured by queue items only)**

| Key | Default |
|---|---|
| `SAMPLE_ENABLED` | True |
| `SAMPLE_PROMPT` | "A high quality photo" |
| `SAMPLE_WIDTH` / `SAMPLE_HEIGHT` | 768 / 768 (per family defaults: Klein 768, Krea 2 1024, Qwen 1024) |
| `SAMPLE_STEPS` | 40 (Krea 2 8, Qwen 25) |
| `SAMPLE_SEED` | 1234 (0 = random per round) |
| `SAMPLE_EVERY_N_EPOCHS` | 1 |
| `SAMPLE_EVERY_N_STEPS` | 0 (Klein only) |
| `SAMPLE_AT_FIRST` | True |
| `SAMPLE_CFG_SCALE` | 1.0 (Klein arch default 4.5) |
| `SAMPLE_NEGATIVE` | long default negative |
| `SAMPLE_FLOW_SHIFT` | "" |
| `CACHE_SAMPLE_MODEL` | "auto" |

---

## 5. Adaptive learning rate and the per-image loss watch

### 5.1 Adaptive LR (epoch-level, bi-directional plateau tracker)

**Implementations.** Klein, inline: `training/trainer.py:2408-2880`. Constants at `:2508-2511`.
Standard layer and Krea 2: `class AdaptiveLR` in `krea2/trainer.py:796-965`, imported by
`families/train.py:36`. The Krea 2 class is described as "a faithful port of Klein's adaptive_lr logic"
(`:797`).

**User inputs.** Only `min_lr` and `max_lr`. The Learning Rate box is **ignored**: the run starts at the
geometric midpoint, `lr0 = √(min·max)` (Klein `:2424-2440`, families `train.py:431-436`). A resumed run
keeps its restored LR. Examples: 2e-4–4e-4 → 2.83e-4; 1e-4–4e-4 → 2e-4; 5e-5–2e-4 → 1e-4.

**Signal.** `current_loss = LossRecorder.moving_average`. Despite the name, this is the **mean of the
latest loss recorded at each in-epoch step slot**, so in practice it is the mean loss over the last
epoch. Excluded steps are dropped from the average (`training/train_utils.py:22-61`).

**Constants**

| Constant | Value |
|---|---|
| `BLEND` (rollback) | 0.7, i.e. 70% previous snapshot + 30% current |
| `WEIGHT_GROWTH_THRESHOLD` | 0.30 (LoRA weight-norm growth per epoch) |
| `CLIP_RATIO_THRESHOLD` | 0.5 (Klein only: fraction of steps whose pre-clip grad norm > `max_grad_norm`) |
| factor up / down | ×1.25 / ×0.5, clamped to [min, max] |

**Pseudocode** (identical in both implementations except the clip signal):

```text
at end of epoch e (0-indexed):
  if e == 0:                                    # "ARMED"
      best_loss = loss; prev_wn = ||θ_lora||_2; snapshot(θ_lora, optimizer_state -> CPU); return
  patience_up   = 2
  patience_down = 2 if (stability_triggered or e == 1 or e >= 4) else 1   # fast-reactive in epochs 2-3
  wn = ||θ_lora||_2 ; growth = (wn - prev_wn)/prev_wn
  stability = (clip_ratio > 0.5  [Klein only])  or  (growth > 0.30)
  if stability:
      stability_streak += 1
      need = 1 if not stability_triggered else 2
      if stability_streak >= need:
          θ ← 0.7·θ_snapshot + 0.3·θ ; optimizer.load_state_dict(snapshot.optim)   # kills bad Adam moments
          lr ← max(lr·0.5, min)                  # "REDUCE+ROLLBACK" (or "HOLD (floored)")
          reset all streaks; stability_triggered = True     # patience pinned at 2 from now on
      else: "WAIT"
  elif loss < best_loss:                         # improving
      stability_streak = 0; best_loss = loss; good += 1; bad = 0
      if good >= patience_up: lr ← min(lr·1.25, max) ("PROBE UP" / "HOLD (capped)"); good = 0
  else:                                          # plateau
      stability_streak = 0; bad += 1; good = 0
      if bad >= patience_down: lr ← max(lr·0.5, min) ("REDUCE" / "HOLD (floored)"); bad = 0
  prev_wn = wn; reset clip counters; snapshot(θ, optimizer -> CPU)   # snapshot AFTER the decision
log: "[adaptive_lr] epoch N: loss=… lr=a→b clip=…% wnorm_Δ=…% | ACTION (reason)"
```

**Interactions.**
- The **step scheduler is off** while adaptive is on: the GUI forces `constant` and omits warmup/decay
  (`lora_trainer_gui.py:33533-33550`), and families skip `_step_scheduler` (`train.py:459-464`).
- **Automagic v3 disables adaptive**, per-image LR and look warm-up (`owns_its_rate`, `train.py:423-430`).
- Klein also disables adaptive if the optimizer stores `lr=None`, as Adafactor's relative_step does
  (`:2412-2423`).
- Krea 2 has no clip-ratio signal in the class version. The docstring says "krea2 has no grad clipping",
  yet `max_grad_norm` does exist in the families and Krea 2 loops. The class simply does not count clip
  events. *(Observed inconsistency; harmless.)*
- **Persistence.** `state_dict()` holds `best_loss`, the streaks, `stability_triggered` and
  `prev_weight_norm`, and it rides in `training_state.json` (families) or `adaptive_lr_state.json`
  (Klein). The CPU snapshot is not persisted.

### 5.2 Step schedulers (non-adaptive)

Families / Krea 2 `_step_scheduler` (`families/train.py:60-76`) uses `LambdaLR` with
`f(s)`:
- warmup: `(s+1)/warmup` for `s < warmup`;
- then by kind, with `prog = (s−warmup)/(total−warmup)`:
  - `constant*`: `1`
  - `cosine`: `0.5(1+cos(π·prog))`
  - `cosine_with_restarts`: `0.5(1+cos(π·((prog·cycles) mod 1)))`
  - `linear`: `1−prog`
  - `polynomial`: `(1−prog)^power`

`total = steps_per_epoch × epochs`.

Klein uses the kohya/diffusers scheduler factory (`training/trainer.py:646-760`). It adds
`lr_scheduler_type` (any `torch.optim.lr_scheduler` class or a dotted path), `lr_scheduler_args`,
`timescale`, `min_lr_ratio`, and decay steps.

### 5.3 The per-image loss watch (`training/loss_logger.py` + `families/loss_watch.py`)

This is an **image-level** controller, separate from Adaptive LR. It requires batch size 1, and it
records the **raw** (unscaled) loss.

**Normalisation** (`PerImageLossWatch`, `loss_logger.py:138-290`):
- Each step's loss is binned by its timestep into **40 buckets** over [0,1] (`_N_BUCKETS`, `:54-60`).
- At each epoch boundary the residual is `loss − bucket_mean[b]`, recomputed against the current
  whole-run bucket means, so it does not depend on order.

**Per image per epoch** (`epoch_boundary`, `:472-598`; starts once `warmup_epochs=2` epochs are seen):
- `series` = mean residual per epoch.
- `mean_residual` over the last `window=5` epochs.
- Trend test over the last `trend_window=8` epochs, split 4 vs 4:
  `improving = drop ≥ max(0.12·max(first,0), 0.02, se)`, with `se = √(2·pooled_var/half)`.
  It only **counts** after 2 consecutive passes.
- `good_run = total_drop ≥ max(0.30·baseline, 0.04)` (baseline = mean of the first 3 epochs).
- Quantiles: `hi` = 66th percentile, `lo` = 33rd percentile of `mean_residual`. Robust outliers:
  `ext_hi = max(med + 1.5·IQR, hi)`, `ext_vhi = max(med + 3·IQR, hi)`.

**Verdict ladder and multiplier** (multiplier applied only when `per_image_lr` is on; warm-up always
applies):

| Verdict | Condition | Multiplier |
|---|---|---|
| excluded | incorrigible (2 failed AI recaptions) + re-confirmed stuck + never healthy ≥3 epochs + cap not reached (≤ half the dataset) | 0 (step skipped, loss slot dropped) |
| stuck | `votes_stuck` (trend ≥4 epochs, `mean_residual ≥ hi`, `> se`, `last > 0`, not improving, no good run, not in warm-up hold) confirmed after 2 consecutive votes; released after 3 clear votes | `max(0.1, 0.5 · 0.5^((tenure−1)//2))` → ×0.5, ×0.25, ×0.125, floor ×0.1 |
| suspect | extreme magnitude (`≥ ext_hi` 2 boundaries, or `≥ ext_vhi` once) | ×0.7 |
| exhausted | good run then plateaued for 2 boundaries while still above average (or "retired" instead of excluded) | ×0.6 |
| watch | stuck vote this epoch, not yet confirmed | ×1.0 |
| learning | `mean_residual ≥ hi` | ×1.0 |
| easy | `mean_residual ≤ lo` and ≥3 epochs | ×1.1 (the only boost) |
| mid | else | ×1.0 |
| warmup | Look-Filter outlier: `0.4 + 0.6·(epoch−1)/4`, released early on proof of improvement | ramp |

**Plateau and best epoch** (`:499-553`):
- Each image's "finish epoch" is `epoch − trend_window/2` at its last confirmed improving pass.
- The run has **plateaued** when, for 2 consecutive boundaries, the improving count is ≤ max(1, 5% of
  tracked images), after at least 8 epochs.
- The best-epoch estimate is the 75th percentile of finish epochs. The console suggests scrubbing
  epochs `be−2 … be+2` in LoRA Royale.
- A plateau declared while images are still being adjudicated is labelled **provisional**.

**Mid-run caption repair** (`families/loss_watch.py:148-285`):
- At each boundary the trainer atomically claims `loss_log/caption_updates.json` (manual edits from the
  Problem Images window) and collects stuck images.
- When auto-recaption is on, it loads the **shared captioner** (Krea 2's Qwen3-VL-4B), writes a new
  `.txt` with the trigger at the start or end, and frees the captioner. Attempt 2 uses the "detailed"
  instruction.
- It parks the DiT on CPU (unless block-swapped), loads the family's text encoder, re-encodes in
  chunks of 4 with `driver.encode_text`, and overwrites the cache file with `cache.save_cond`.
- It resets those images' watch history, marks attempt ≥2 images incorrigible, and appends to
  `caption_updates_applied.json`.
- A captioner OOM turns auto-recaption off for the rest of the run.

**Resume** replays `per_image_loss.jsonl` plus the applied-caption ledger, so verdicts survive a pause
(`loss_watch.py:121-134`; `resume_from_jsonl`, `loss_logger.py:826`). A fresh run clears the previous
run's problem files and rotates the JSONL (`loss_watch.py:288-299`).

Loss-watch flags need batch size 1. With batch size > 1, the GUI skips them with a console note
(`lora_trainer_gui.py:33890-33909`) and the watcher latches `_batched`, which disables per-image LR.

---

## 6. Optimizers, schedulers, precision, memory, caching, ROCm

### 6.1 Optimizers (`src/fizgig/training/optimizers.py`)

`_CATALOG` (`:39-54`):

| Name | Backend | Note |
|---|---|---|
| `adamw8bit` (default) | bitsandbytes `AdamW8bit` | the validated recipe |
| `adamw` | `torch.optim.AdamW(fused=cuda)` | fp32 state |
| `pagedadamw8bit`, `ademamix8bit`, `pagedademamix8bit` | bitsandbytes | |
| `lion8bit` | bitsandbytes | warns when LR > 5e-5 (wants about 1/10 of an AdamW LR) |
| `automagic3` | vendored Ostris (MIT) `training/automagic3.py`, `fused=False` | owns its LR (per param group); the LR box is the **start** (1e-6 recommended) |
| any `module.path.ClassName` | import | exotic optimizers |

- Prodigy, CAME and Adafactor were **removed** on purpose because they manage their own LR and fight
  Adaptive LR (`:31-38`).
- `create_optimizer` falls back to AdamW, with a logged warning, if construction fails
  (`:190-263`).
- `eps_floor_8bit` (eps=1e-6 for the 8-bit Adam family) is opt-in, used by MiniMax only.
- Krea 2 Automagic splits params into families (`txtfusion`, `attn`, `mlp`, `io`) so each family gets
  its own rate (`:91-128`).
- Klein's dropdown offers `adamw, adamw8bit, bitsandbytes.optim.AdEMAMix8bit, …PagedAdEMAMix8bit`
  (`lora_trainer_gui.py:2236`).

### 6.2 Mixed precision

| Stack | Approach |
|---|---|
| Klein | Accelerate `--mixed_precision bf16` (fp16 if the DiT filename says fp16); `--fp8_base [--fp8_scaled]` or `--quant_4bit` |
| Krea 2 | base in fp8 (default), INT8 (W8A8) or NF4; network in bf16 (`network.to(dtype)`); autocast bf16 around the DiT forward (`krea2/trainer.py:570`) |
| Families | base bf16 / INT8 / NF4; **trainable adapters fp32**, frozen adapters bf16; the driver casts inputs to bf16 (Qwen forward `xt.to(bf16)`); loss in fp32 (`F.mse_loss(pred.float(), target)`) |

### 6.3 Gradient checkpointing

- Klein: GUI toggle, on by default (`--gradient_checkpointing`).
- Krea 2: forced on, via `_CheckpointedBlock`.
- Families: always on (`train_family(gradient_checkpointing=True)` default, not exposed on the CLI).
- Qwen note: with only block Linears LoRA-wrapped, checkpointing records no graph unless the joint
  hidden states require grad (`qwen_image.py:209-213`, citing ai-toolkit#1054).

### 6.4 Base quantisation

- **INT8 W8A8 for training** (`modules/int8_train.py`; families `quant.py:31-68`): per-output-channel
  absmax scale `w/127`, int8 forward via `torch._int_mm`, **bf16 backward**. Grad w.r.t. input only;
  the base is frozen.
- **NF4** (`modules/nf4.py`; bitsandbytes `quantize_nf4`): dequantise per matmul. **NF4 cannot block-swap**
  (packed weights are plain attributes), so swap is forced to 0 (`quant.py:77-99`).
- **fp8** (Klein/Krea 2; `modules/fp8.py`, `krea2/fp8_optimization_utils.py` from musubi): uses
  `_scaled_mm` on Ada+ only. On ROCm, `fp8_matmul` is forced False (`capabilities.py:174`).
- **HQQ 4-bit** and ConvRot int8 for MiniMax (`minimax/hqq4.py`, `minimax/convrot*.py`).
- Families quantise **only the block-map Linears**. Everything else stays bf16. The base loads on CPU
  first and is quantised one Linear at a time on the GPU, so the full bf16 model is never resident
  (`quant.py:1-12`).
- **Families Auto plan** (`quant.plan`, `quant.py:118-162`):

```text
budget = free_vram_GB - 1.5
peak(p) = desc.train_memory[p][0] interpolated linearly over (megapixels, GB) points
auto precision: first of (bf16, int8, nf4) ∩ desc.precisions with peak <= budget, swap 0
  else: int8 (or bf16) + n swapped blocks, n = ceil((peak - budget)/GB_per_block) capped at max_blocks_to_swap
  else: nf4 (smallest)
explicit precision + swap -1: fewest blocks that fit
```

Qwen's measured table (`qwen_image.py:110-111`):

| Precision | Peak at 0.25 MP | Peak at 1.0 MP | GB saved per swapped block |
|---|---|---|---|
| bf16 | 14.9 | 19.0 | 0.44 |
| int8 | 8.6 | 11.9 | 0.19 |
| nf4 | 6.0 | 9.5 | 0 |

Example for an RX 7900 XT with ~18.5 GB free at 0.5 MP *(inferred from the formula)*: bf16 needs ≈16.3,
the budget is ≈17.0, so **bf16 with no swap**.

### 6.5 Block swap and offloading

- **Mechanism.** A musubi-derived `ModelOffloader` (`modules/offloading.py:88-490`;
  `krea2/offloading.py` is explicitly credited to musubi in THIRD_PARTY_NOTICES). It streams the
  weights of N blocks between CPU and GPU with forward and backward hooks on a thread pool, and has a
  forward-only mode for previews (`set_forward_only`, driver `block_swap_mode`).
- **torch.compile and block swap are mutually exclusive.** Compile is ignored while swap is on
  (`capabilities.should_compile`, `:707-709`; GUI `:34103-34125`).

**Auto rules (GUI)**

*Klein training* (`_auto_training_blocks_swap`, `:10239`; uses total VRAM):

| Total VRAM | Blocks swapped |
|---|---|
| ≥ 15 GiB | 0 |
| ≥ 10 GiB | 12 |
| < 10 GiB | 16 |

Klein Auto base: NF4 below 15 GiB (`_klein_small_card`, `:10229`; `_parse_blocks_swap`, `:10080-10085`).

*Klein Distilled sample model* (`training/trainer.py:1313-1341`):

| Total VRAM | Blocks swapped |
|---|---|
| ≥ 23 GiB | 0 |
| ≥ 15 GiB | 16 |
| < 15 GiB | maximum |

*Krea 2* (`_auto_krea2_strategy`, `:10093` → `capabilities.recommend_krea2_strategy`, `:490-625`):
- Decides on **free** VRAM, read fresh. Preference order: INT8 with no swap, then NF4 with no swap, then
  fp8 with no swap, then fp8 + swap.
- `peak = base + 2.4 GB per extra batch image + 0.25 GB/MP above 0.25 + 0.015 GB/rank` (rank measured
  at 32); LoKR extra by factor.
- Bases (training-only, measured): fp8 18.7, INT8 16.2, NF4 11.4 GB. Headroom 1.5 GB. Swap saves
  0.42 GB/block, capped at 26.
- INT8 needs `torch._int_mm` (probed); NF4 needs bitsandbytes.
- Legacy fallback by total VRAM: ≥30 → 0, ≥22 → 12, ≥15 → 20, else 26 (`:10185`).

*Krea 2 inference/preview swap* (`:10204-10227`): ≥30 → 0, ≥22 → 4, ≥18 → 12, ≥15 → 20, else 26.

*Workbench / inference default* (`_auto_detect_blocks_to_swap`, `:1632`): ≥15 → 0, ≥10 → 12, else 16.

*MiniMax and families*: resolved in the trainer from free VRAM (`--blocks_to_swap -1|auto`).

**VRAM simulation.** `FIZGIG_SIM_VRAM_GB=N` caps the process with `set_per_process_memory_fraction` and
reports simulated free memory (`quant.py:165-183`), which makes it possible to test small-card plans on
a big card.

### 6.6 torch.compile (Krea 2)

- `should_compile` (`capabilities.py:683-763`) returns **off on ROCm** under Auto: "recompiles per bucket
  shape on HIP; set Compile Blocks to On to override" (`:697-701`).
- It is also off when block swap is on, triton is missing or mismatched, there is no C compiler, or the
  base is not quantised (fp8/bf16 unmeasured).
- Otherwise it compiles only if `steps ≥ 90 s / saving_per_step × 2` (INT8 saves 0.30 s → ~600 steps;
  NF4 saves 0.153 → ~1176). VRAM is checked with 15 GB/MP over 0.25.
- The "outside" checkpoint boundary is used when inside-the-graph does not fit.
- Compilation is **per block** after the LoRA patch (`krea2/trainer.py:347`).

### 6.7 Attention

- `modules/sdpa.py`: SDPA uses the default backend while training and cuDNN under `no_grad`
  (inference), probed once.
- Override with `FIZGIG_SDPA_BACKEND`. `FIZGIG_ATTN_TRIM=0` reduces the number of distinct shapes.
- Klein has flags `--sdpa/--flash_attn/--xformers/--flash3/--split_attn`.

### 6.8 Other memory hygiene

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set by the trainers on **non-Windows** only
  (`families/train.py:684-686`; `scripts/train.py:15-30`). The Windows CUDA allocator rejects it.
- The ROCm launcher sets `PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.8`
  (`run_fizgig_rocm.bat`).
- `KMP_BLOCKTIME=0`, `OMP_WAIT_POLICY=PASSIVE`.
- Text encoders are loaded only for caching and preview-prompt encoding, then unloaded.
- The EMA raw-weight backup lives on CPU (`ema.py:40-49`).
- `[preview-vram]` log lines are written at every preview waypoint.

### 6.9 Latent and text-encoder caching

- **Cache directory per dataset folder**: `<prefs.cache_dir>/<sanitised folder name>-<sha1(lowercased
  path)[:8]>` (`_cache_dir_for`, `lora_trainer_gui.py:31778-31791`). For example `cache/demo_dataset-1a2b3c4d/`.
  The dataset layer refuses a shared `cache_directory`.
- **Filenames** (`dataset/image_dataset.py:907-916`):
  - latents: `<image basename>_<WWWW>x<HHHH>_<arch>.safetensors`, where W×H is the **original** image
    size, zero-padded to 4 digits;
  - text: `<image basename>_<arch>_te.safetensors`;
  - examples: `image (1)_0800x0800_krea2.safetensors` and `image (1)_krea2_te.safetensors`.
- **Contents:**
  - latents: `latent_{h}x{w}` (the **bucket** latent size), optional `latent_control_{i}_{h}x{w}`
    (edit/paired), metadata `architecture, width, height, dtype, format_version`;
  - families text: `cond__<driver key>` tensors + metadata `architecture, caption1 (the caption text),
    format_version[, reference_sizes]` (`families/cache.py:45-67`);
  - Krea 2 text: `hidden_states (seq, 12, dim)` + `attention_mask`.
- **Staleness:**
  - `latent_cache_matches_reso` rejects caches written at a different Target MP, because the filename
    encodes the original size, not the bucket (`image_dataset.py:1078`, `:1193-1202`);
  - `prepare_for_training` skips caches whose image no longer exists in the folder (`:1146-1160`);
  - families text caching with `--skip_existing` re-encodes when `caption1` or `reference_sizes`
    differ (`families/cache.py:70-78`);
  - latents are re-encoded when the control (pair) presence changed (`:144-145`);
  - NaNs in cached tensors are replaced with 0 and a warning is logged (`:37-42`);
  - `post_process` deletes cache files whose images are gone, unless `--keep_cache`.
- **Training reads caches, never images.** The item list is built by **globbing the cache dir**
  (`prepare_for_training`, `:1139+`). The loader maps cache keys to batch keys: `latent_*` → `latents`,
  `latent_control_i_*` → `latents_control_i`, `cond__*` verbatim, plus `item_keys` for the loss watch
  (`:494-592`).

### 6.10 ROCm-specific paths and workarounds

- **Backend detection.** `utils/gpu_backend.is_rocm()` checks `torch.version.rocm`, then
  `torch.version.hip`, then `"+rocm"` in the version string (`gpu_backend.py:13-30`).
- **Capability probes are real kernels, not tables** (`capabilities.py:116-144`): `_scaled_mm` for fp8
  (skipped on ROCm), `_int_mm` for INT8 training. On ROCm, cuDNN attention is False.
  *(Whether `torch._int_mm` works on gfx1100 under torch 2.12+rocm7.15 is unverified here. The probe
  decides at runtime.)*
- **Compile Auto off on ROCm** (see §6.6).
- **ROCm Windows launcher** `run_fizgig_rocm.bat` (Apache-2.0, adapted from comfyui-rocm):
  - `MIOPEN_FIND_MODE=2`, `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, `FIZGIG_GPU_BACKEND=rocm`;
  - calls the generated `rocm_env.bat` (`ROCM_PATH`/`HIP_PATH` = venv `_rocm_sdk_core`,
    `BNB_ROCM_VERSION=715`);
  - RDNA1/2 (`gfx101*`/`gfx103*`): disables flash and mem-efficient SDP;
  - otherwise `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`;
  - sets MIOpen/rocBLAS DB paths for non-gfx1100 GPUs;
  - clears `ROCBLAS_USE_HIPBLASLT_BATCHED`.
  - The gfx code comes from the **GPL-3.0 `detect_gpu.py`**.
- **Install pins** (docs/INSTALL.md:79-85): `torch==2.12.0+rocm7.15.0a20260728`, a pinned community
  bitsandbytes Windows ROCm wheel (0xDELUXA), and Python 3.12 (the bnb wheel is cp312-only). **This is
  exactly TagScribeR's runtime stack.**
- **Linux ROCm:** cache scripts skip HIP teardown on success (`src/fizgig/rocm/cache_exit.py`).
- **VRAM status bar on AMD:** `typeperf` (Windows) or `amd-smi` / `rocm-smi` (Linux)
  (`utils/vram_monitor.py`).
- Full fine-tune emits an "UNTESTED on AMD/ROCm" warning (`krea2/trainer.py:2404-2411`).
- Fizgig's general position: "on AMD, NF4 and INT8 are the primary quant paths" (fp8 base savings need
  NVIDIA Ada+).

---

## 7. Dataset handling for training

- **TOML schema** (`dataset/config.py:1-140`, validated with voluptuous):
  - `[general]` and/or per `[[datasets]]`: `resolution` (int or [w,h]), `caption_extension`,
    `batch_size`, `num_repeats`, `enable_bucket`, `bucket_no_upscale`;
  - per dataset: `image_directory`, `cache_directory`, `control_directory` (pairs), `is_reg`
    (regularisation).
- **GUI-written TOML** (`_build_dataset_toml_text`, `:31916-32060`):
  - `resolution = [s, s]` with `s = floor(√(MP·10⁶)) // 16 · 16` (0.25 MP → 496, 0.5 MP → 704);
  - **`num_repeats = 1` hardcoded** ("UI removed");
  - one `[[datasets]]` block per folder: the Start folder, plus extra concept folders (MiniMax Multi
    Concept);
  - `control_directory` for Qwen Edit;
  - an optional `is_reg` block only when fine-tune is on.
  - Repeats are supported by the dataset layer (each item is appended `num_repeats` times,
    `image_dataset.py:1215-1217`) but not exposed in the GUI.
- **Captions:** sidecar `<stem><ext>` (default `.txt`). The decode is tolerant: UTF-16 BOM → UTF-8(-sig)
  → cp1252 → latin-1, with warnings (`image_dataset.py:36-97`). Images without a caption are **left out**,
  with a loud warning (`glob_images`, `:168-200`). The GUI refuses to start in that case (`:32695-32709`).
- **Trigger words:** there is no training-time token injection. The trigger is written into the caption
  files by the Captions tab (start or end). It also:
  - becomes `modelspec.trigger_phrase` and `usage_hint` when the metadata field is blank
    (`:33580-33583`);
  - is prepended or appended to auto-recaptions (`loss_watch.py:264-266`).
- **Caption shuffle / keep_tokens: not supported anywhere.** Text conditioning is **pre-cached per
  caption**, so per-step text augmentation is impossible in this design. Bilingual captions (EN + ZH via
  Helsinki-NLP) are offered as "text-level augmentation" at caption time.
- **Caption dropout:** MiniMax H3 only. Default 0.05 (`MINIMAX_CAPTION_DROPOUT`). It swaps in a cached
  empty-prompt embedding for that fraction of steps (`minimax/trainer.py:4191-4206, 5385-5386`;
  `scripts/minimax_cache_text.py:298-315`).
- **Regularisation images:** fine-tune only (Krea 2 / H3 rotation FT). They train at a fixed LR
  multiplier (default 0.2) and are exempt from the loss watch. **On LoRA runs, reg blocks are ignored
  with a warning**: they would otherwise train as ordinary images at full LR (`krea2/trainer.py:1978-2010`).
- **Multi-concept:** MiniMax only. There is a "Subject 2 folder" (more possible), each with its own
  dataset block and cache dir (`_dataset_folders`, `:31793-31817`). Klein has a "Multi-Character" preset,
  which is just rank 16 with Identity blocks.
- **Bucketing** (`BucketSelector`, `image_dataset.py:364-440`):
  - Target area `A = W·H`. Candidate widths run from `divisible_by(√A/2, step)` to `√A` in `step`
    increments, with `h = divisible_by(A//w, step)`; both orientations are added.
  - An image picks the bucket with the **closest aspect ratio**.
  - `no_upscale`: an image smaller than the target area keeps its own size, floored to `step`.
  - Resize uses scale-to-cover (Lanczos up, INTER_AREA down) plus a **centre crop**
    (`resize_image_to_bucket`, `:207-237`).
  - Steps: `RESOLUTION_STEPS = 16` (Klein, Krea 2), MiniMax 32 (`BUCKET_RESO_STEPS`), described families
    `desc.bucket_step` (Qwen 32) (`:113-130`, `:819-820`).
  - Qwen 0.5 MP examples (docs/QWEN_IMAGE.md): 1:1 → 704×704, 4:5 → 624×784, 2:3 → 576×848,
    9:16 → 528×928.
  - `BucketBatchManager` shuffles items within buckets and the bucket-batch order each epoch
    (`:470-489`).
- **Edit pairs (Qwen):** the before-image has the **same file name** as the after-image in the
  originals folder. One before-image per after-image is enforced (`families/cache.py:135-139`). The
  before-image is cached at the target's bucket. Validation refuses a missing original, a missing
  caption, or a different crop or shape (docs/QWEN_IMAGE.md).
- **Image prep tools** (Image Prep tab, not training): batch resize to an **area** target (default
  1.0 MP, never below the training target, issue #44), PNG conversion, InsightFace face crops, and the
  Look Consistency Filter (ArcFace; writes `fizgig_look_scores.json` `{cutoff, scores{stem: score}}`).

---

## 8. In-training sample previews

### 8.1 What, when, where

- **Frequency:** every `sample_every_n_epochs` (GUI default 1), plus `sample_at_first` (epoch 0, before
  training). Klein also supports `sample_every_n_steps`. Families and Krea 2 sample at epoch boundaries
  only.
- **Prompts:** the Samples tab text, one prompt per line.
  - Klein's prompt file supports inline flags: `--w --h --d(seed) --s(steps) --g --fs --l(cfg) --n(neg) --ci(control image)`
    (`training/trainer.py:199-249`; generated by `generate_sample_prompt_file`, `:13861`).
  - Krea 2 and families get clean lines with flags stripped (`_write_krea2_sample_prompts`, `:33790`).
  - Edit LoRA previews use the **edit instruction** instead.
- **Seeds:** `seed + i` per prompt. Seed 0 means a fresh random seed each round.
- **Output:** `<output_dir>/sample/<name>_e<epoch:06d>_<idx:02d>_<YYYYmmddHHMMSS>_<seed>.png`. The same
  pattern is used by every trainer so the gallery can parse it (`families/train.py:220-277`;
  `krea2/trainer.py:1593-1625`).
- **Gallery:** the GUI watcher regenerates `files.json` every 5 s, and once more at stop, and serves a
  local HTTP gallery ("View Samples Gallery"). The gallery offers live ArcFace likeness badges and a
  per-epoch trend once 3 reference photos are chosen (CPU, `lora_royale/likeness.py`), plus the Training
  Run Visualiser.
- **Live sample override:** the `.sample_override.json` prompt, seed and size replace the configured set
  at the next preview. The override prompt is encoded mid-run, and the DiT is parked on CPU first if the
  text encoder will not fit beside it (`families/train.py:166-191`).

### 8.2 How a preview is rendered (standard layer, `_render_previews`)

```text
_preview_vram("preview start", reset_peak)
net.set_enabled("training_adapter", False)      # deployment setup: adapter OFF
if speed LoRA: move to GPU + enable              # e.g. Viggle turbo at its own steps/sigmas
ema.swap_in()                                    # previews show the EMA weights
dit.eval(); if swapped: driver.block_swap_mode(dit, inference=True)
lats = [driver.generate(dit, cond_i, w, h, steps, seed+i, cfg | speed.cfg, sigmas, options, refs)]
if lowmem and not swapped: park DiT on CPU for the decode     # never hold DiT + VAE decode together
pngs = [driver.decode(vae, lat, w, h).save(...)]
finally: restore DiT, swap mode, ema.swap_out(), speed LoRA off+CPU, adapter ON, train mode, empty_cache
```

- **Prompt encoding happens once at startup.** The text encoder is loaded, used and unloaded, and the
  VAE stays resident for the whole run (`families/train.py:351-379`).
- **Low-VRAM rule:** total VRAM < 20 GB (`/1e9`) → preview canvas capped at 768 px long side (multiples
  of 32), and the DiT parks on CPU for the decode. `FIZGIG_PREVIEW_LOWMEM=0/1` forces it
  (`families/train.py:133-147, 194-199`). An RX 7900 XT reports ~21.5 GB in that unit, so it is **not**
  low-mem. *(Inferred from the formula.)*
- **Context LoRA stays on** for previews. **The training adapter is off.** The speed LoRA is parked on
  CPU between previews.
- **Failure policy (Krea 2):** any preview exception, usually OOM, logs a warning and **disables previews
  for the rest of the run**. Training and saving continue (`krea2/trainer.py:3444-3452, 3480-3489,
  3555-3567`). The families loop does not wrap previews in a try. *(A preview OOM there would end the
  run; this is a gap.)*

### 8.3 Per-family preview engines

| Family | Engine |
|---|---|
| Klein | Optionally the **Distilled** DiT (`--sample_dit`) at 4 steps. The base is max-swapped to make room; the Distilled is block-swapped by VRAM (§6.5) and optionally cached in CPU RAM when ≥18 GB is free (`_should_cache_sample_model`, `:1343-1374`). Optional reference image for edit-conditioned previews. Default 40 steps CFG 4.5 without the Distilled. |
| Krea 2 | Default "RAW + Turbo LoRA": the Turbo LoRA (rank 64) is staged disabled on the **training DiT** and enabled at 1.0 for previews (8 steps, mu = 1.15, CFG 1). Classic mode instead loads the fp8 Turbo checkpoint and parks the trainer on CPU. Preview INT8 and preview block swap are configurable. |
| Qwen 2.1 | Plain model at 25 steps (turbo strength 0 by default) or Viggle turbo at strength 1 / 6 steps with explicit sigmas `(1, .9375, .875, .75, .5, .25)` and `shift_terminal=None` (`qwen_image.py:131-176`). |
| MiniMax H3 | Short clip previews judged by the middle frame; turbo LoRA steps/strength (`MINIMAX_TURBO_*`). |

---

## 9. The GUI's training parameter surface

The tabs run left to right: **1 Start** (training folder), **2 Image Prep**, **3 Captions**,
**4 Samples**, **5 Training**. Unnumbered workbench tabs follow: Profiler, Repair Studio, LoRA the
Explorer, LoRA Royale, Extract, Preferences. The Training tab is assembled around
`lora_trainer_gui.py:4600-6320`:

- **Header row:** Base Model selector; Load Preset (✨ built-ins first, then user presets); Save Preset;
  Delete; "Load Settings From Last Train" (`:4628-4652`).
- **Output** (expanded, `:4655`): Output Directory (remembered per family), LoRA Name.
- **Training Parameters** (expanded, `:4676`):
  - Learning Rate.
  - **Adaptive LR** checkbox with Min LR / Max LR and Reset Defaults. The help text says the LR box is
    ignored and the run starts at the geometric midpoint (`:4686-4725`).
  - Network Dim (Rank), Network Alpha, Max Epochs, Save Every N Epochs, Seed.
  - Model Area to Train (Klein: Full / Identity / Style / Style+Comp / Details / Custom, with a block
    grid of 8 double and 24 single blocks).
  - Context LoRA (path, strength).
  - Target Megapixels.
  - Network Type (LoRA / LoKR), LoKR Factor.
  - Loss-watch toggles: Detect problem images, Per-image adaptive LR, Auto-recaption, Warm up look
    outliers, and a "👁 View Problem Images" window.
  - Krea 2 EMA.
  - Described families: Training adapter (recommended), Weight averaging (EMA), Edit LoRA (originals
    folder, test photo, edit caption + Write captions) (`:5488-5575`).
  - Krea 2 / H3 full fine-tune sub-panels and regularisation folder × LR.
  - Many MiniMax rows.
- **Optimizer** (collapsed, `:5592`): Optimizer Type (family-filtered), Optimizer Args, Gradient
  Accumulation, Max Grad Norm, Network Dropout *(inferred contents from presets and settings keys)*.
- **Other Options** (collapsed, `:5606`):
  - Dataset: Caption Extension (.txt), Batch Size (recommended 1), Bucket Options (Enable Bucket,
    No Upscale).
  - LR Scheduler with Warmup / Decay (steps).
  - Metadata fields.
- **Memory & Precision (INT8 / FP8 / NF4)** (`:5689`):
  - Blocks Swap (Auto (detect from GPU) or a number).
  - Weight Optimization (FP8 Base, FP8 Scaled), FP8 Text Encoder (Klein).
  - Base precision (Klein/Krea 2: Auto / NF4 / INT8 / fp8; families: Auto / bf16 / INT8 / 4-bit NF4).
  - Grad Checkpoint, Compile Blocks (Auto/On/Off).
  - Save State (At each checkpoint / At end of training), Keep Last N.
  - MiniMax Base Precision.
- **Timestep & Noise Schedule** (collapsed, `:6134`):
  - Quick Presets (Full range / Structure / Detail / Sigmoid; `_ts_preset_*`, `:10448-10480`).
  - Timestep Sampling, Discrete Flow Shift, Sigmoid Scale, Timestep Range Min/Max, Preserve Distribution
    Shape.
  - Weighting Scheme (Logit Normal mean/std, Mode Scale).
- **Bottom:** Enable Cache Preparation; Start / Pause / Resume / Stop; View Samples Gallery; Open Samples
  Folder; progress bar (`:6279-6314`).
- **Samples tab:** enable; prompts (one per line); width/height (`SAMPLE_RESOLUTIONS` 512–1536); steps;
  seed; every N epochs (Klein also steps); sample at first; CFG; negative; flow shift; reference image;
  Klein "use Distilled"; Krea 2 preview engine; family Turbo strength.

Defaults are listed in §4.4. Per-family visibility is handled by `update_ui_for_architecture`
(`:7870+`), which hides controls a family does not read.

---

## 10. Other notable subsystems

- **Repair Studio** (`repair_studio/`, `families/workbench.py`): a per-block slider for each LoRA block
  (32 on Klein, up to 50 plus refiners on H3), a side-by-side preview, a donor-LoRA blend, and an
  **exact bake**.
  - The bake rank-concatenates primary and donor with multipliers folded into `lora_up` and
    `alpha = new rank`, so that `baked_up @ baked_down` equals the live sum (`repair_studio/bake.py:1-20`).
  - LoKR and LoHa are baked losslessly in native form.
  - Turbo Preview caches per-block activations.
  - Krea 2 "Text fusion ×2/×3" presets.
  - Presets live in `presets/repair_studio/<family>/`.
- **LoRA the Explorer:** evolutionary mutation of block strengths, with four variants per round.
- **LoRA Royale:** renders every epoch on a fixed seed with a crossfade, optional ArcFace likeness
  scoring, and exports (epoch morph, seed/prompt/strength travel, comparison sheet) as MP4/GIF via
  ffmpeg (`lora_royale/export.py`, `prompt_travel.py`).
- **Profiler:** activation profile (Klein) or weight-only profile (others), colour-coded HTML plus a
  JSON sidecar keyed by the LoRA hash. Repair Studio auto-loads the matching sidecar
  (`families/weight_profile.py`).
- **Extract:**
  - weight-only rank reduction (exact SVD via QR; `families/extract.py:14-26`);
  - Klein activation-weighted extraction targeted by blocks and timesteps (`extraction/extractor.py`);
  - **checkpoint diff → LoRA** (`extraction/model_diff.py`, standalone `diff_to_lora_gui.py`), multi-rank
    from one SVD per layer, kohya-flattened keys, pre-quantised checkpoints decoded first.
- **Full fine-tuning (Krea 2 / H3):** rotating-block fine-tune (LISA-style). The base stays frozen in
  fp8/NF4, a window of N blocks is promoted to trainable bf16 from a CPU bf16 master copy, and the
  window rotates every K epochs (`krea2/rotation.py:1-40`). Fused backward is optional. Output is a full
  checkpoint, then diff-to-LoRA.
- **Weight profiles:** see Profiler. `families/weight_profile.py` computes `Σ‖up‖_F·‖down‖_F` per block,
  plus LoKR `‖w1‖·‖w2‖·scale`.
- **Qwen Edit presets:** the Edit and Edit Strong presets set `FAMILY_EDIT=True`. Training uses the
  before-image as reference latents plus vision tokens in the text encoder template (`driver.py:80-99`).
  Previews apply the edit to a held-out test photo.
- **Validation and loss tracking:**
  - **There is no held-out validation loss.** Quality is tracked with the training loss moving average
    (console), the per-image loss JSONL, `problem_images.json`, plateau / best-epoch estimation,
    `scripts/analyze_loss_log.py` (offline analysis), and live ArcFace likeness on previews.
  - TensorBoard / wandb exist only in the Klein trainer (`--log_with`).
- **Context LoRA:** trains a new LoRA over a frozen, active LoRA, so the new one learns to coexist with
  it (e.g. a face on a style). The context LoRA is never saved into the output.
- **Training adapters:** a frozen LoRA that is on during training and off in previews and saves (Qwen,
  H3). The measured A/B is recorded in `qwen_image.py:200-205`.
- **EMA** (`training/ema.py`): fp32 shadow, decay ramp `min(decay, (1+n)/(10+n))`, swapped in for saves
  and previews only. Default 0.98 where measured (Krea 2, H3, Qwen).
- **Tests:** the GUI comments reference a `tests/` suite (e.g. `tests/test_minimax_likeness_gui.py`) and
  the env var `FIZGIG_NO_PERSIST` for headless tests. *(The tests directory was not inspected.)*

---

## 11. Implications for TagScribeR

### 11.1 Licensing ground rules

- Fizgig is **Apache-2.0** (`LICENSE`, `NOTICE`). Copying or adapting its files requires keeping the
  copyright and licence notice, stating changes, and carrying `NOTICE` content into TagScribeR's
  attribution (e.g. a `THIRD_PARTY_NOTICES.md` entry: "Portions adapted from Fizgig © 2026 Peter Neill,
  Apache-2.0").
- **Inherited third-party terms travel with the files:**
  - musubi-tuner (Apache-2.0): `krea2/offloading.py`, `krea2/fp8_optimization_utils.py`,
    `krea2/safetensors_utils.py`, `krea2/lora_utils.py`, `krea2/attention.py`, `krea2/vae_loader.py`,
    and the Krea 2 training recipe. `modules/offloading.py` and the Klein trainer/network are kohya/musubi
    lineage by inspection. *(Inferred; NOTICE lists musubi for Krea 2 only.)*
  - ai-toolkit (MIT): `krea2/model.py`, `krea2/sampling.py`, `training/automagic3.py`. The MIT text must
    ship with them.
  - Diffusers (Apache-2.0): `krea2/vae.py` (Qwen-Image VAE).
- **GPL-3.0 — do not copy:** `detect_gpu.py` (verbatim from comfyui-rocm; THIRD_PARTY_NOTICES
  `:104-118`). `run_fizgig_rocm.bat` is Apache-2.0 but adapted from comfyui-rocm patterns and *calls* the
  GPL script. TagScribeR already has its own `core/hardware.py` ROCm detection and should keep using it.
  `detect_gpu_linux.py` is Fizgig-authored *(by its header; no GPL notice)*.
- Model weights (Fizgig training adapter, Viggle turbo, etc.) have their own licences. Qwen Image 2.1
  is under the Qwen Research License (non-commercial).

### 11.2 What is cleanly reusable vs tightly coupled

**Reusable almost as-is** (small, torch-only, few or no GUI deps):

| Component | File(s) | Notes |
|---|---|---|
| Family interface | `families/driver.py`, `families/description.py` | Copy the **design**. The dataclasses are generic; drop Qwen-specific fields TagScribeR does not need. |
| Generic train loop | `families/train.py` | Imports `AdaptiveLR` from `krea2/trainer.py`. Move that class into its own module. Add the gaps listed in §2.5. |
| Adaptive LR | `krea2/trainer.py:796-965` (`AdaptiveLR`), plus the Klein clip-ratio signal (`training/trainer.py:2605-2618, 2735-2750`) | Self-contained. Re-add the clip-ratio stability signal (cheap to count in the loop). |
| Step schedulers | `families/train.py:60-76` | Trivial. |
| Loss recorder / state pruning / output-name validation | `training/train_utils.py:22-145` | Depends on `accelerate` only at import (for unused helpers). Split it. |
| Per-image loss watch | `training/loss_logger.py`, `families/loss_watch.py` | Watch core is pure Python. `loss_watch.py` imports the Krea 2 captioner for auto-recaption; TagScribeR should inject **its own VLM captioner** there (its core competence). |
| EMA | `training/ema.py` | Self-contained. |
| Optimizer factory | `training/optimizers.py` (+ `automagic3.py`, MIT) | bitsandbytes on ROCm needs the pinned wheel (same as Fizgig). |
| Multi-adapter LoRA layer | `families/lora.py` | Depends on `networks.lora.factorization` / `_lokr_forward_update` / `lycoris_scale_from_keys`. Extend to Conv2d for SDXL (§11.3). |
| INT8 / NF4 frozen base + Auto plan | `families/quant.py`, `modules/int8_train.py`, `modules/nf4.py` | Linear-only. Verify `_int_mm` on gfx1100. |
| Block-swap offloader | `modules/offloading.py` | Generic over a list of blocks. |
| Dataset + bucketing + cache I/O | `dataset/config.py`, `dataset/image_dataset.py` | 1.4k lines with MiniMax video/audio branches. Usable, but a trimmed image-only copy is cleaner. TagScribeR's `core/health.py` already has aspect-bucket logic to align with. |
| Generic caching CLI | `families/cache.py` + `scripts/cache_latents.py`, `scripts/cache_text.py` helpers | |
| Metadata | `training/metadata.py` | Extend `build_metadata` with SDXL / Anima arch strings. |
| Progress parser | `training/progress.py` | Pure regex over tqdm lines; works for a QProcess-driven design. |
| Capability probes / strategy | `utils/capabilities.py`, `utils/gpu_backend.py` | Probe pattern reusable; numbers are per-model. |
| Workbench tools | `families/workbench.py`, `families/weight_profile.py`, `families/extract.py`, `families/lorafile.py` | For later. |

**Tightly coupled, re-implement** (keep only the behaviour):

- **`lora_trainer_gui.py`** (Tkinter, 35k lines). The preset system, last-train snapshot, queue,
  per-family settings memory, pipeline chaining and file-sentinel IPC all live in it. Re-implement them
  in PySide6, following the contracts in §1.4 and §4.3.
  - TagScribeR already has `core/presets.py`, `core/projects.py` and a QThread worker. The training
    presets should reuse that persistence style: user_data JSON, versioned and validated.
  - Port the **validation rules** from §4.3: ignore unknown keys; strict combos for optimizer, LR
    bounds, scheduler and network type; first-token matching; legacy migrations; no architecture in
    presets; architecture stored in last-run/queue snapshots.
- **Klein trainer** (`training/trainer.py`, accelerate) and the **Krea 2 / MiniMax monoliths**. They are
  rich, but reuse means reading them for recipes, not importing them. The standard layer is the intended
  future.
- **Family model code** (`klein/`, `krea2/`, `qwen_image21/`, `minimax/`): only relevant if TagScribeR
  wants those families. The Qwen-Image VAE (`krea2/vae.py`, `krea2/vae_loader.py`) **is** relevant to
  Anima (§11.4).

**Process model recommendation (keep Fizgig's):** run caching and training as **child processes**
(QProcess or `subprocess`) using the same venv, and parse stdout. Keep the file sentinels:

- `.pause_requested`
- `.sample_override.json`
- `loss_log/problem_images.json`
- `loss_log/caption_updates.json`
- `sample/*.png`

This keeps the GUI process free of torch and allocator state, survives OOMs, makes Stop a process-tree
kill (`taskkill /F /T` on Windows), and matches TagScribeR's design rule of no torch at import in
`core/`.

### 11.3 What SDXL-architecture families need to fit the family interface (SDXL, Pony, Illustrious, NoobAI)

The interface is DiT-flavoured in naming (`load_dit`, `block_prefix`) but not in substance: the driver
owns loading, encoding, the objective and the sampler. An `SDXLDriver` would need the following.
Specifics below are from general knowledge of kohya sd-scripts and diffusers; framed for Fizgig's
design, but verify before implementing.

1. **Description facts:**
   - `latent_channels=4`, `spatial_factor=8`, `bucket_step=64` (kohya's SDXL convention; 32 also
     works), `native_megapixels≈1.0` (1024²);
   - `n_blocks` / `block_prefix` replaced by an overridden **`block_map()`** with groups "Input blocks"
     (IN00–IN08), "Middle", "Output blocks" (OUT00–OUT08). The `BlockGroup` docstring already names this
     case;
   - `precisions=("bf16",)`: the 2.6B UNet fits 20 GB in bf16. Optionally `int8`;
   - `modelspec_arch="stable-diffusion-xl-v1-base"`; `ss_base_model_version="sdxl_base_v1-0"` for kohya
     tooling;
   - `sampling`: e.g. Euler-a / DPM++ 2M Karras, 25–30 steps, CFG 5–7, `negative_prompt=True`. Pony and
     Illustrious recipes differ (Pony `score_9, score_8_up…` tags; NoobAI-XL ships ε-pred and
     **v-pred** checkpoints).
2. **`load_dit` → load the UNet** from a single-file SDXL checkpoint (split the
   `model.diffusion_model.*` keys; the CLIP and VAE keys come from the same file). That means the three
   "model files" may be **one file**. The description's `ModelFile` roles assume separate files, so add
   a "checkpoint" role, or let the VAE and text-encoder roles point at the same path.
3. **`load_text_encoder` / `encode_text`:**
   - two encoders: CLIP-L (penultimate layer) and OpenCLIP-bigG (penultimate layer + pooled
     projection);
   - conditioning dict `{"crossattn": (77k, 2048), "pooled": (1280,)}`, with 75-token chunking for long
     tags (kohya `max_token_length` 150/225).
   - The SDXL **size conditioning** (`original_size`, `crop_coords`, `target_size` → `time_ids`) depends
     on the **bucket and crop**, not the caption. Compute it in `training_loss` from latent shape and
     metadata, rather than caching it per caption. The latent cache metadata already stores the
     original width and height, but **crop offsets are not stored** today.
4. **`encode_images`:** SDXL VAE, scale 0.13025, no shift. The stock SDXL VAE NaNs in fp16; use bf16/fp32
   or the fp16-fix VAE.
5. **`training_loss`:** DDPM objective with 1000 discrete timesteps and the scaled-linear schedule.
   - ε-prediction (SDXL / Pony / Illustrious) or **v-prediction** (NoobAI v-pred: also zero-terminal-SNR
     rescale).
   - Expected knobs beyond Fizgig's set: **min-SNR-γ weighting** (γ≈5), **noise offset** (≈0.0357),
     multires noise, `min_t` / `max_t` as integer timesteps.
   - Fizgig's flow-matching `min_t` / `max_t` floats map onto this naturally (fraction of 1000).
6. **LoRA layer gaps:**
   - SDXL LoRAs target attention `to_q/to_k/to_v/to_out.0`, `ff.net.*` and `proj_in/proj_out` inside
     `Transformer2DModel` blocks, optionally **Conv2d** (LoCon: resnets, up/down samplers).
   - `families/lora.py` wraps `nn.Linear` only. Add a Conv2d `LoRAFactor` pair (down = conv k×k, up =
     1×1), as `networks/lora.py:LoRAModule` already does for Conv2d (`:66-76`).
   - `quant.py` likewise only quantises Linears, which is fine.
7. **Text encoder training (optional):** kohya trains CLIP LoRAs too (`lora_te1_*`, `lora_te2_*`
   keys). Fizgig never trains text encoders and caches their output. Supporting it means **not caching
   text** for those runs, and adding TE modules to the trainable set with a separate LR.
8. **Caption augmentation:**
   - Booru-tag datasets (Pony, Illustrious, NoobAI) usually expect **shuffle_caption + keep_tokens +
     caption/tag dropout**. Fizgig's cached-conditioning design cannot do this, and it is the biggest
     *design* mismatch.
   - Options:
     - (a) re-encode the text each step. The SDXL encoders are ~1.6 GB and fast, so this is affordable
       on 20 GB; keep them resident.
     - (b) cache K shuffled variants per caption.
   - Either way, keep the cache path for the no-augmentation default.
9. **Batch size:** `families/train.py` hardcodes batch 1 because conditioning lengths vary. SDXL
   conditioning is fixed-length (77·k), so batched training is natural and expected (bs 2–4 on 20 GB).
   Relax the check per driver (e.g. a `supports_batching` flag) and stack within buckets
   (`BucketBatchManager` already batches). Per-image LR and the loss watch would then switch off, as
   designed.
10. **Key format:** kohya `lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q` (SGM naming, what
    A1111 and ComfyUI load) or diffusers-named keys. Put it in `LoRAFormat` with `kohya=True` and
    verify against ComfyUI's `comfy/lora.py` SDXL mapping, as Fizgig did for Qwen.
11. **Previews:** `generate()` = Euler / DPM++ with CFG and a negative prompt, at 20–30 steps. No speed
    LoRA is required, although DMD2 / Lightning / Hyper SDXL LoRAs fit the `SpeedLoRA` slot exactly.
    Block swap is unnecessary on 20 GB (`max_blocks_to_swap = 0`).
12. **Likeness tooling:** ArcFace-based likeness and the Look Filter are meaningless for anime / Pony
    datasets. A CLIP or anime-face embedder would be needed for those features.

### 11.4 What Anima (circlestone-labs/Anima, Cosmos-Predict2-2B finetune) needs

Facts below are from general knowledge and are **unverified against Anima's official code**. Confirm
before implementing.

- **Components:**
  - a 2B Cosmos-Predict2 `MiniTrainDIT`-style DiT (28 blocks, hidden 2048, self-attn + cross-attn +
    MLP with AdaLN-LoRA modulation);
  - the text encoder is **Qwen3-0.6B**, plus a small **LLM adapter** inside the DiT checkpoint that maps
    Qwen3 states into the cross-attention space the original T5-based Cosmos expected;
  - the VAE is the **Qwen-Image / Wan-2.1 VAE** (16 channels, /8, causal 3D with a frame axis).
- **Fit with Fizgig's pieces:**
  - **VAE:** Fizgig already ships a Qwen-Image VAE implementation and loader (`krea2/vae.py` from
    diffusers, Apache-2.0; `krea2/vae_loader.py` from musubi, Apache-2.0) and a cache path that squeezes
    the 1-frame axis (`krea2/caching.py:5-9`). Directly reusable for `encode_images` / `decode`. Mind the
    latent normalisation constants (Qwen-Image VAE mean/std per channel).
  - **Text:** `encode_text` returns the Qwen3-0.6B hidden states (last layer, or the layer Anima's
    reference uses) plus a mask, cached as `cond__hidden_states` / `cond__mask`. If the LLM adapter is
    trained or LoRA'd it must stay inside the DiT forward, not the cache. Variable length → batch 1 or
    padding with masks (`pad_conditioning` already exists for this).
  - **`training_loss`:** community trainers (diffusion-pipe, sd-scripts Anima support) treat
    Cosmos-Predict2 as a **rectified-flow** model: `x_t = (1−t)x0 + t·ε`, velocity target, logit-normal
    timesteps with a shift. This mirrors the Qwen/Krea driver almost exactly. Cosmos's own code uses an
    EDM-style σ parameterisation with preconditioning, so check which convention Anima's released
    weights expect (and whether `t` is fed ×1000) before trusting previews.
  - **Block map:** one group of 28 blocks; modules `self_attn.{q_proj,k_proj,v_proj,output_proj}`,
    `cross_attn.*`, `mlp.layer1/layer2`. Decide whether AdaLN modulation and `llm_adapter.*` are targets
    (H3's experience: modulation Linears soak up capacity, so exclude them by default).
  - **LoRA keys:** match ComfyUI's Cosmos/Anima loader, likely `diffusion_model.blocks.N.…` bare or
    kohya `lora_unet_blocks_N_…`. Verify against `comfy/lora.py` and record it in `LoRAFormat.source`.
  - **Memory:** a 2B DiT in bf16 is ~4 GB, so `precisions=("bf16",)`, no swap or quantisation needed on
    20 GB. `train_memory` should still be measured so the Auto plan works on smaller cards.
  - **Sampling:** Anima's recommended sampler (community: ~30–50 steps, CFG ~4–5, a negative prompt,
    quality / booru tags) goes in `SamplingSettings` with sources. Caption-augmentation concerns from
    §11.3 item 8 apply here too, because Anima is tag-trained.
  - **5D latents:** Cosmos patchifies (T,H,W). Add the T=1 axis inside the driver, as Krea 2 does.

### 11.5 Suggested shape for TagScribeR (summary)

1. Port the **standard layer** as `training/`:
   - `driver.py` + `description.py` (trimmed);
   - `train.py` (+ grad accumulation, optional batching, step sampling, preview try/except, Conv2d LoRA,
     caption-augmentation hook);
   - `cache.py`, `lora.py`, `quant.py`, `adaptive_lr.py` (from `AdaptiveLR`, with the clip signal),
     `loss_watch.py`, `ema.py`, `optimizers.py`, `metadata.py`, `progress.py`.
   - Add Fizgig attribution headers.
2. Write **drivers**: `SDXLDriver` (ε/v-pred variants parameterised for SDXL / Pony / Illustrious /
   NoobAI) and `AnimaDriver` (reusing the Qwen-Image VAE).
3. Re-implement the GUI-side contracts in PySide6:
   - preset JSON per family, with built-ins in code / descriptions, first-visit default, and strict
     validation;
   - last-run snapshot and queue;
   - frozen dataset config per run;
   - subprocess pipeline cache → train with stdout progress;
   - pause sentinel, sample override, Problem Images window, samples gallery.
4. TagScribeR's captioner (WD tagger / VLMs) slots into the loss watch's auto-recaption in place of
   Fizgig's Krea 2 Qwen3-VL-4B, which is a natural advantage for TagScribeR.
5. Do not copy `detect_gpu.py`. Keep `core/hardware.py`. Mirror Fizgig's ROCm env decisions
   (aotriton experimental flag, MIOpen find mode, compile off by default on HIP) as configuration, not
   copied scripts.
