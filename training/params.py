"""The Train tab's parameters: Fizgig's setting keys, defaults, option labels and groups, plus help text.

Keys are Fizgig's GUI keys (NETWORK_DIM, ADAPTIVE_LR_MIN, KREA2_LOSS_WATCH, ...) so a Fizgig preset JSON applies
here unchanged, and values are stored the way Fizgig stores them (combo labels such as "2e-4 - rank 4/8 only" or
"Auto (detect from GPU)"; numbers as typed). Defaults come from Fizgig's settings defaults
(lora_trainer_gui.py:2083-2211) and widget initialisers (docs/FIZGIG_TRAINING_AUDIT.md section 4.4).

`preset=False` marks values that are not part of a preset (Fizgig's _NON_TRAINING_ENTRY_KEYS: the Samples tab,
resume) - they belong to the run, not the recipe. The TagScribeR caption-augmentation extension (CAPTION_*) is off
by default; with it off, runs are Fizgig's.

Importable without torch or Qt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

BOOL, INT, FLOAT, TEXT, CHOICE, PATH, DIR, MULTILINE = "bool", "int", "float", "text", "choice", "path", "dir", "multiline"

PRECISION_LABELS = {"auto": "Auto (fits your free VRAM)", "bf16": "bf16 (full precision)",
                    "fp8": "fp8 (8-bit float, as fp8 checkpoints ship)",
                    "int8": "INT8 (8-bit, fastest)", "nf4": "4-bit NF4 (smallest)"}
NETWORK_LABELS = {"lora": "LoRA (standard)", "lokr": "LoKR (Kronecker)"}
EMA_OPTIONS = ("Off", "0.98 (recommended)", "0.99 (stronger)", "0.995 (long runs only)")
SWAP_AUTO = "Auto (detect from GPU)"
SCHEDULERS = ("constant", "constant_with_warmup", "cosine", "cosine_with_restarts", "linear", "polynomial")
OPTIMIZER_NOTES = {
    "adamw8bit": "AdamW, 8-bit state (default - the validated recipe; needs bitsandbytes)",
    "adamw": "AdamW, fp32 state",
    "pagedadamw8bit": "AdamW8bit that pages state to CPU under pressure (bitsandbytes)",
    "ademamix8bit": "AdEMAMix - second slow EMA, aimed at long runs (bitsandbytes)",
    "pagedademamix8bit": "AdEMAMix8bit with CPU paging (bitsandbytes)",
    "lion8bit": "Lion - sign updates; use about 1/10 of the AdamW learning rate (bitsandbytes)",
    "automagic3": "Automagic v3 - sets its own learning rate; the LR box is its start (1e-6 recommended)",
}

GROUPS = ("Output", "Training Parameters", "Loss watch", "Optimizer", "Memory & Precision", "Timesteps",
          "Dataset", "Caption augmentation (extension)", "Metadata", "Samples")


@dataclass(frozen=True)
class Param:
    key: str
    label: str
    kind: str
    default: object
    group: str
    tip: str = ""
    options: tuple = ()
    strict: bool = False           # a saved value the family doesn't offer is refused (Fizgig _STRICT_COMBO_KEYS)
    preset: bool = True            # carried by presets (False = run-only, like Fizgig's Samples tab)
    minimum: float | None = None
    maximum: float | None = None
    advanced: bool = False         # collapsed by default in the UI
    family_only: str = ""          # "adapter" | "ema" | "precision" | "lokr" | "edit" | "speed" | "option": shown when the
    #                                family declares that feature


P = Param
PARAMS: tuple[Param, ...] = (
    # ---- output ----------------------------------------------------------------------------------
    P("LORA_OUTPUT_DIR", "Output folder", DIR, "", "Output",
      "Where checkpoints, samples and the run's logs are written. Each run gets its own subfolder named after the "
      "LoRA. Empty = user_data/training_runs."),
    P("LORA_NAME", "LoRA name", TEXT, "my_lora", "Output",
      "The file name of the trained LoRA (no extension). Epoch checkpoints are <name>-000012.safetensors, the final "
      "one <name>.safetensors."),
    # ---- training parameters ----------------------------------------------------------------------
    P("LEARNING_RATE", "Learning rate", FLOAT, 4e-4, "Training Parameters",
      "How big each training step is. Ignored while Adaptive LR is on (the run starts at the geometric middle of "
      "Min and Max). With Automagic v3 this is only its starting rate.", minimum=0.0),
    P("ADAPTIVE_LR", "Adaptive LR", BOOL, False, "Training Parameters",
      "Fizgig's adaptive learning rate. Each epoch it compares the loss and how fast the LoRA's weights grow: it "
      "probes up x1.25 after two improving epochs, halves on a plateau, and on instability (weights growing >30% "
      "in an epoch, or more than half the steps clipped) halves AND blends the weights 70/30 back to the previous "
      "epoch. The run starts at sqrt(Min x Max); the LR box and the scheduler are ignored."),
    P("ADAPTIVE_LR_MIN", "Min LR", CHOICE, "1e-5", "Training Parameters",
      "The lowest rate Adaptive LR may use. The higher floors are only safe for low-rank LoRAs.",
      options=("1e-5", "5e-5", "1e-4", "2e-4 - rank 4/8 only", "3e-4 - low-rank only"), strict=True),
    P("ADAPTIVE_LR_MAX", "Max LR", CHOICE, "4e-4", "Training Parameters",
      "The highest rate Adaptive LR may probe up to.", options=("1e-4", "2e-4", "3e-4", "4e-4"), strict=True),
    P("NETWORK_TYPE", "Network type", CHOICE, NETWORK_LABELS["lora"], "Training Parameters",
      "LoRA: two small matrices per layer (rank x alpha). LoKR: a Kronecker-product adapter (LyCORIS), sized by its "
      "factor instead of a rank.", options=tuple(NETWORK_LABELS.values()), strict=True, family_only="lokr"),
    P("NETWORK_DIM", "Network rank", INT, 4, "Training Parameters",
      "LoRA rank: capacity. 4-8 for one subject, 16 for larger or mixed datasets, 32 for big style sets. Higher "
      "ranks learn faster but overcook sooner and make bigger files.", minimum=1, maximum=1024),
    P("NETWORK_ALPHA", "Network alpha", FLOAT, 4, "Training Parameters",
      "Scales the LoRA's effect: scale = alpha / rank. Fizgig's presets use alpha = rank (scale 1).", minimum=0),
    P("LOKR_FACTOR", "LoKR factor", INT, 8, "Training Parameters",
      "LoKR only: the small Kronecker factor is about factor x factor. Larger = smaller file, less capacity.",
      minimum=1, maximum=64, family_only="lokr"),
    P("MAX_TRAIN_EPOCHS", "Epochs", INT, 12, "Training Parameters",
      "How many passes over the dataset. Every saved epoch is a usable LoRA, so you can pick an earlier one if the "
      "last is overcooked.", minimum=1, maximum=10000),
    P("SAVE_EVERY_N_EPOCHS", "Save every N epochs", INT, 1, "Training Parameters",
      "Write a checkpoint <name>-NNNNNN.safetensors every N epochs (the final file is always written).",
      minimum=0, maximum=1000),
    P("SEED", "Seed", INT, 42, "Training Parameters",
      "Makes noise and data order repeatable: the same seed and settings give the same run.", minimum=0),
    P("FAMILY_TRAINING_ADAPTER", "Use the training adapter", BOOL, True, "Training Parameters",
      "A frozen helper LoRA that is on for every training step, off for previews and never saved into your file. "
      "It keeps training stable on families that need it.", family_only="adapter"),
    P("FAMILY_EMA", "Weight averaging (EMA)", CHOICE, "0.98 (recommended)", "Training Parameters",
      "Checkpoints and previews come from a running average of the adapter's recent steps instead of whichever "
      "step the epoch ended on - smoother, usually better. Training itself runs on the raw weights.",
      options=EMA_OPTIONS, family_only="ema"),
    P("CONTEXT_LORA_PATH", "Context LoRA", PATH, "", "Training Parameters",
      "Optional: an existing LoRA kept frozen and active while you train, so the new LoRA learns to work on top "
      "of it (e.g. a face on a style). It is never saved into the output.", advanced=True),
    P("CONTEXT_LORA_STRENGTH", "Context LoRA strength", FLOAT, 1.0, "Training Parameters",
      "The context LoRA's strength during training and previews.", minimum=0.0, maximum=2.0, advanced=True),
    P("FAMILY_EDIT", "Edit LoRA (before/after pairs)", BOOL, False, "Training Parameters",
      "Train an edit (e.g. a colour grade) from pairs: the dataset folder holds the edited images, the originals "
      "folder holds the matching originals with the same file names.", family_only="edit"),
    P("FAMILY_EDIT_DIR", "Originals folder", DIR, "", "Training Parameters",
      "Edit LoRA: the folder of original ('before') images, each named like its edited version.",
      family_only="edit"),
    # ---- loss watch -------------------------------------------------------------------------------
    P("KREA2_LOSS_WATCH", "Detect problem images", BOOL, False, "Loss watch",
      "Watch each image's loss through the run (normalised for the noise level) and flag images that never learn "
      "(stuck), learn oddly (suspect) or are done early (exhausted). See them in Problem Images. Batch size 1 only."),
    P("KREA2_PER_IMAGE_LR", "Per-image adaptive LR", BOOL, False, "Loss watch",
      "Also scale each image's step by its verdict: stuck images train gentler (x0.5 down to x0.1), easy ones a "
      "little harder (x1.1). Needs Detect problem images."),
    P("KREA2_AUTO_RECAPTION", "Auto-recaption stuck images", BOOL, False, "Loss watch",
      "Between epochs, re-caption stuck images with the captioner chosen below, re-encode them, and continue. Two "
      "failed attempts and an image is set aside. Captions are saved (with backups)."),
    P("KREA2_WARMUP_LOOK", "Warm up look outliers", BOOL, False, "Loss watch",
      "Images marked as look outliers in a look-score file (tagscriber_look_scores.json or Fizgig's "
      "fizgig_look_scores.json) start at x0.4 learning rate and ramp to x1.0 over four epochs.", advanced=True),
    # ---- optimizer --------------------------------------------------------------------------------
    P("OPTIMIZER_TYPE", "Optimizer", CHOICE, "adamw8bit", "Optimizer",
      "How the weights are updated. AdamW 8-bit is Fizgig's validated recipe (it needs the bitsandbytes package; "
      "without it the run falls back to AdamW with a warning).", options=tuple(OPTIMIZER_NOTES), strict=True),
    P("OPTIMIZER_ARGS", "Optimizer args", TEXT, "", "Optimizer",
      "Extra optimizer settings as key=value pairs, e.g. weight_decay=0.01 betas=0.9,0.99", advanced=True),
    P("GRADIENT_ACCUMULATION", "Gradient accumulation", INT, 1, "Optimizer",
      "Average the gradients of N steps before each update - a bigger effective batch without more VRAM.",
      minimum=1, maximum=256),
    P("MAX_GRAD_NORM", "Max grad norm", FLOAT, 1.0, "Optimizer",
      "Clip the gradient's size to this (0 = no clipping). Also feeds Adaptive LR's clip-ratio stability signal.",
      minimum=0.0),
    P("LR_SCHEDULER", "LR scheduler", CHOICE, "constant", "Optimizer",
      "How the learning rate changes over the run (off while Adaptive LR or Automagic is on).",
      options=SCHEDULERS, strict=True, advanced=True),
    P("LR_WARMUP_STEPS", "Warmup steps", INT, 0, "Optimizer",
      "Ramp the learning rate up from 0 over this many optimizer steps (not with Adaptive LR).", minimum=0,
      advanced=True),
    # ---- memory & precision -----------------------------------------------------------------------
    P("FAMILY_PRECISION", "Base precision", CHOICE, PRECISION_LABELS["auto"], "Memory & Precision",
      "The frozen base model's precision. Auto picks the most precise one that fits your free VRAM (quantising "
      "costs no speed; swapping blocks costs 25-65%).", options=tuple(PRECISION_LABELS.values()), strict=True,
      family_only="precision"),
    P("BLOCKS_SWAP", "Blocks to swap", CHOICE, SWAP_AUTO, "Memory & Precision",
      "Stream this many model blocks between system RAM and the GPU to fit a smaller card (slower). Auto uses as "
      "few as fit. Not with 4-bit NF4.", options=(SWAP_AUTO, "0", "4", "8", "12", "16", "20", "24", "30")),
    P("SAVE_STATE", "Save state at each checkpoint", BOOL, True, "Memory & Precision",
      "Also save the optimizer and training state with each checkpoint, so a run can be resumed or extended."),
    P("SAVE_STATE_ON_TRAIN_END", "Save state at the end", BOOL, True, "Memory & Precision",
      "Save a resumable state when the run finishes (to train more epochs later)."),
    P("KEEP_LAST_N_STATES", "Keep last N states", INT, 2, "Memory & Precision",
      "Older state folders are deleted to save disk space (at least 1 is always kept).", minimum=1, maximum=1000),
    # ---- timesteps --------------------------------------------------------------------------------
    P("MIN_TIMESTEP", "Noise range min", FLOAT, 0.0, "Timesteps",
      "The lowest noise level trained (0-1). Raising it trains only coarse structure.", minimum=0.0, maximum=1.0,
      advanced=True),
    P("MAX_TIMESTEP", "Noise range max", FLOAT, 1.0, "Timesteps",
      "The highest noise level trained (0-1). Lowering it trains only fine detail / style.", minimum=0.0,
      maximum=1.0, advanced=True),
    # SDXL-family extensions (TagScribeR, NOT Fizgig: kohya sd-scripts options). All default OFF, so a run with them
    # untouched is the plain recipe. Passed to the driver as driver_options (see DRIVER_OPTIONS).
    P("SDXL_MIN_SNR_GAMMA", "Min-SNR gamma", FLOAT, 0.0, "Timesteps",
      "Extension (off = 0). Down-weights the easy, low-noise timesteps so the loss is balanced across noise levels "
      "(Hang et al. 2023, kohya's --min_snr_gamma). 5 is the usual value; try it if training looks unstable or "
      "converges slowly. Applies to SDXL-family models only.", minimum=0.0, maximum=20.0, advanced=True,
      family_only="option"),
    P("SDXL_NOISE_OFFSET", "Noise offset", FLOAT, 0.0, "Timesteps",
      "Extension (off = 0). Adds a small per-channel brightness shift to the training noise so the LoRA can learn "
      "very dark and very bright images (kohya's --noise_offset). 0.03-0.05 is typical; leave off for v-prediction "
      "models that already use zero-terminal-SNR. Applies to SDXL-family models only.", minimum=0.0, maximum=1.0,
      advanced=True, family_only="option"),
    P("SDXL_LOCON", "Also train convolutions (LoCon)", BOOL, False, "Training Parameters",
      "Extension (off). Besides the attention and feed-forward layers, also train the 3x3 convolutions of the "
      "ResNet blocks and the up / down samplers (kohya's conv_dim / LyCORIS LoCon), at the same rank. Bigger file, "
      "more capacity for style; the file still loads in A1111 and ComfyUI. Applies to SDXL-family models only.",
      advanced=True, family_only="option"),
    # FLUX.2 Klein extensions: Fizgig's keys and defaults (lora_trainer_gui.py settings 2158-2165, PRESETS 1877-1898,
    # ARCHITECTURES["Flux 2 Klein Base 9B"] timestep_sampling flux2_shift). Passed to the driver as driver_options.
    P("TARGET_LAYERS", "Model area to train", CHOICE, "Full Model", "Training Parameters",
      "Which part of the model the LoRA trains (Klein). Identity = single blocks 1-16, Style and Style+Composition = "
      "all 8 double blocks + single 0-1, Details = single blocks 12-23, Custom = the blocks named below. Style also "
      "pairs with late timesteps (0-0.4).",
      options=("Full Model", "Identity", "Style", "Style+Composition", "Details", "Custom"), strict=True,
      family_only="option"),
    P("TRAINING_BLOCKS", "Custom blocks", TEXT, "", "Training Parameters",
      "Model area Custom only: comma-separated blocks, e.g. double_blocks.0, double_blocks.1, single_blocks.5 (8 "
      "double blocks 0-7, 24 single blocks 0-23). Empty trains the full model, as Fizgig does.", advanced=True,
      family_only="option"),
    P("TIMESTEP_SAMPLING", "Timestep sampling", CHOICE, "flux2_shift", "Timesteps",
      "How noise levels are drawn each step. flux2_shift (default) is the resolution-shifted sigmoid Klein uses; "
      "the others are Fizgig's alternatives: sigma, uniform, sigmoid, shift, flux_shift, logsnr, qinglong_flux.",
      options=("sigma", "uniform", "sigmoid", "shift", "flux_shift", "flux2_shift", "logsnr", "qinglong_flux"),
      strict=True, advanced=True, family_only="option"),
    P("DISCRETE_FLOW_SHIFT", "Flow shift", FLOAT, 3.0, "Timesteps",
      "Only for timestep sampling 'shift': the fixed shift t = t*s / (1 + (s-1)*t).", minimum=0.0, maximum=20.0,
      advanced=True, family_only="option"),
    P("SIGMOID_SCALE", "Sigmoid scale", FLOAT, 1.0, "Timesteps",
      "Scales the random number before the sigmoid (sigmoid, shift, flux_shift, flux2_shift, qinglong_flux); "
      "larger spreads the noise levels toward both ends.", minimum=0.0, maximum=20.0, advanced=True,
      family_only="option"),
    P("LOGIT_MEAN", "Logit mean", FLOAT, 0.0, "Timesteps", "logsnr and qinglong_flux sampling: the mean of the "
      "log signal-to-noise draw.", minimum=-20.0, maximum=20.0, advanced=True, family_only="option"),
    P("LOGIT_STD", "Logit std", FLOAT, 1.0, "Timesteps", "logsnr and qinglong_flux sampling: the spread of the "
      "log signal-to-noise draw.", minimum=0.0, maximum=20.0, advanced=True, family_only="option"),
    P("PRESERVE_DISTRIBUTION", "Preserve distribution shape", BOOL, False, "Timesteps",
      "With a noise range set: keep drawing until the noise levels fall inside it (the natural curve, cut off) "
      "instead of squeezing the whole curve into the range.", advanced=True, family_only="option"),
    # MiniMax H3 extensions: Fizgig's keys and defaults (lora_trainer_gui.py settings 2113-2143, MINIMAX_* presets).
    # Passed to the driver as driver_options.
    P("MINIMAX_LOWNOISE_PCT", "Low-noise training %", FLOAT, 60.0, "Timesteps",
      "MiniMax H3: the share of training steps drawn from the clean half of the noise range (below sigma 0.5), where "
      "detail and identity are learned. 60 is the tuned default for stills; 50 is the plain uniform schedule; 8 is "
      "the model's own video schedule (mostly composition and movement). Fizgig maps it to the schedule shift "
      "(1 - P) / P.", minimum=1.0, maximum=99.0, family_only="option"),
    P("MINIMAX_LIKENESS_MODE", "Training mode", CHOICE, "Default", "Training Parameters",
      "MiniMax H3: which of the 50 blocks the LoRA trains. Default = blocks 20-49 (the identity blocks: quickest steps "
      "and the measured best for characters and styles). More Blocks = 6-49 (slower, holds the dataset's global "
      "traits out of the LoRA longer). Off = the blocks you type below.",
      options=("Default", "More Blocks", "Off - hand-pick the blocks below"), family_only="option"),
    P("MINIMAX_BLOCKS", "Blocks to train", TEXT, "all", "Training Parameters",
      "MiniMax H3, training mode Off only: blocks as numbers and ranges, e.g. 6-49 or 3-12, 14-15, 22, 31-33 "
      "(0-49; 'all' = every block). A typo stops the run instead of training a different set.", advanced=True,
      family_only="option"),
    # ---- dataset ----------------------------------------------------------------------------------
    P("DATASET_MEGAPIXELS", "Target megapixels", CHOICE, "0.25", "Dataset",
      "Training resolution as an area. Images are bucketed by aspect ratio at about this many pixels (0.5 MP is "
      "704x704 for square images). Higher = sharper but slower and more VRAM.",
      options=("0.25", "0.37", "0.5", "0.75", "1.0", "1.5", "2.0", "2.4", "3.0", "4.2")),
    P("DATASET_BATCH_SIZE", "Batch size", INT, 1, "Dataset",
      "Images per step. Families with variable-length conditioning (Qwen Image) train at 1 - use Gradient "
      "accumulation for a bigger effective batch. The loss watch needs 1.", minimum=1, maximum=64),
    P("DATASET_CAPTION_EXT", "Caption extension", TEXT, ".txt", "Dataset",
      "The caption file next to each image (image.png + image.txt).", advanced=True),
    P("ENABLE_BUCKET", "Aspect-ratio buckets", BOOL, True, "Dataset",
      "Group images by aspect ratio so nothing is squashed (off = everything cropped to a square).", advanced=True),
    P("BUCKET_NO_UPSCALE", "Never upscale", BOOL, True, "Dataset",
      "Images smaller than the target keep their own size instead of being enlarged.", advanced=True),
    P("DATASET_REPEATS", "Repeats", INT, 1, "Dataset",
      "Show each image this many times per epoch (Fizgig keeps this at 1).", minimum=1, maximum=100, advanced=True),
    # ---- caption augmentation (TagScribeR extension, off by default) ----------------------------------
    P("CAPTION_SHUFFLE_VARIANTS", "Shuffled tag variants", INT, 0, "Caption augmentation (extension)",
      "TagScribeR extension (off = 0). Cache this many shuffled-tag versions of each caption and pick one at random "
      "each step - common for booru-tag datasets. Costs one text-encoder pass per variant at caching time.",
      minimum=0, maximum=16, advanced=True),
    P("CAPTION_KEEP_TOKENS", "Keep first N tags", INT, 0, "Caption augmentation (extension)",
      "With shuffling: the first N comma-separated tags (e.g. your trigger word) stay in front.", minimum=0,
      maximum=32, advanced=True),
    P("CAPTION_DROPOUT", "Caption dropout", FLOAT, 0.0, "Caption augmentation (extension)",
      "TagScribeR extension (off = 0). The fraction of steps trained with an empty caption (MiniMax H3 uses 0.05 "
      "in Fizgig). Helps a LoRA work without its exact caption.", minimum=0.0, maximum=0.5, advanced=True),
    # ---- metadata ---------------------------------------------------------------------------------
    P("METADATA_TITLE", "Title", TEXT, "", "Metadata", "Shown by model browsers. Empty = the LoRA name.",
      advanced=True),
    P("METADATA_AUTHOR", "Author", TEXT, "", "Metadata", "Your name or handle.", advanced=True),
    P("METADATA_DESCRIPTION", "Description", TEXT, "", "Metadata",
      "Empty = the last preview prompt.", advanced=True),
    P("METADATA_LICENSE", "License", TEXT, "", "Metadata",
      "e.g. the base model's licence. Qwen Image 2.1 is under the Qwen Research License (non-commercial).",
      advanced=True),
    P("METADATA_TAGS", "Tags", TEXT, "", "Metadata", "Comma-separated keywords.", advanced=True),
    P("METADATA_TRIGGER_PHRASE", "Trigger phrase", TEXT, "", "Metadata",
      "The word that activates the LoRA. Empty = the dataset's subject / trigger from Auto Caption.", advanced=True),
    P("METADATA_THUMBNAIL", "Thumbnail", TEXT, "", "Metadata",
      "Empty = each checkpoint carries its own epoch's preview. 'off' = no thumbnail, or an image path.",
      advanced=True),
    # ---- samples (run-only, not in presets) ----------------------------------------------------------
    P("SAMPLE_ENABLED", "Render previews", BOOL, True, "Samples",
      "Render sample images during training with the LoRA so far (on the training model itself; a preview failure "
      "never stops the run).", preset=False),
    P("SAMPLE_PROMPT", "Preview prompts", MULTILINE, "A high quality photo", "Samples",
      "One prompt per line. Include your trigger word. Lines starting with # are ignored.", preset=False),
    P("SAMPLE_WIDTH", "Width", INT, 0, "Samples", "Preview width (0 = the family default).", preset=False,
      minimum=0, maximum=4096),
    P("SAMPLE_HEIGHT", "Height", INT, 0, "Samples", "Preview height (0 = the family default).", preset=False,
      minimum=0, maximum=4096),
    P("SAMPLE_STEPS", "Steps", INT, 0, "Samples", "Sampling steps (0 = the family default).", preset=False,
      minimum=0, maximum=200),
    P("SAMPLE_SEED", "Seed", INT, 1234, "Samples",
      "Preview seed (each prompt adds 1). 0 = a new random seed every round.", preset=False, minimum=0),
    P("SAMPLE_EVERY_N_EPOCHS", "Every N epochs", INT, 1, "Samples", "0 = no previews during training.",
      preset=False, minimum=0, maximum=1000),
    P("SAMPLE_AT_FIRST", "Preview before training", BOOL, True, "Samples",
      "Render a set before the first step - the base model's look, for comparison.", preset=False),
    P("SAMPLE_CFG_SCALE", "CFG", FLOAT, 0.0, "Samples",
      "Guidance scale (0 = the family default). Above 1 the negative prompt is used.", preset=False, minimum=0.0,
      maximum=30.0),
    P("SAMPLE_NEGATIVE", "Negative prompt", TEXT,
      "blurry, low detail, noisy, washed out, oversaturated, distorted anatomy, extra limbs, duplicate objects, text, "
      "watermark, logo, frame, cropped subject, flat lighting, muddy colors", "Samples",
      "Used only when CFG is above 1.", preset=False),
    P("FAMILY_TURBO_STRENGTH", "Turbo LoRA strength", FLOAT, -1.0, "Samples",
      "The family's speed LoRA for fast previews (needs its file under Model files). 0 = previews without it; "
      "-1 = the family default.", preset=False, minimum=-1.0, maximum=2.0, family_only="speed"),
    P("FAMILY_EDIT_REF", "Edit preview photo", PATH, "", "Samples",
      "Edit LoRA: the photo every preview applies the edit to (empty = the first original).", preset=False,
      family_only="edit"),
    P("FAMILY_EDIT_CAPTION", "Edit instruction", TEXT, "", "Samples",
      "Edit LoRA: the preview instruction (empty = the first edited image's caption).", preset=False,
      family_only="edit"),
)

BY_KEY: dict[str, Param] = {p.key: p for p in PARAMS}

# family-extension parameters -> the keyword `driver.configure()` receives (see FamilyDescription.family_options)
DRIVER_OPTIONS = {"SDXL_MIN_SNR_GAMMA": "min_snr_gamma", "SDXL_NOISE_OFFSET": "noise_offset", "SDXL_LOCON": "locon"}
DRIVER_OPTIONS.update({"TARGET_LAYERS": "target_layers", "TRAINING_BLOCKS": "training_blocks",
                       "TIMESTEP_SAMPLING": "timestep_sampling", "DISCRETE_FLOW_SHIFT": "discrete_flow_shift",
                       "SIGMOID_SCALE": "sigmoid_scale", "LOGIT_MEAN": "logit_mean", "LOGIT_STD": "logit_std",
                       "PRESERVE_DISTRIBUTION": "preserve_distribution"})
DRIVER_OPTIONS.update({"MINIMAX_LOWNOISE_PCT": "lownoise_pct", "MINIMAX_LIKENESS_MODE": "likeness_mode",
                       "MINIMAX_BLOCKS": "blocks"})
PRESET_KEYS = tuple(p.key for p in PARAMS if p.preset)


def defaults() -> dict:
    return {p.key: p.default for p in PARAMS}


def options_for(param: Param, desc=None) -> tuple:
    """The options a family offers for a choice parameter (Fizgig filters optimizers, network types, precisions)."""
    if desc is None:
        return param.options
    if param.key == "OPTIMIZER_TYPE":
        return tuple(desc.optimizers)
    if param.key == "NETWORK_TYPE":
        return tuple(NETWORK_LABELS[t] for t in desc.network_types if t in NETWORK_LABELS)
    if param.key == "FAMILY_PRECISION":
        return (PRECISION_LABELS["auto"],) + tuple(PRECISION_LABELS[p] for p in desc.precisions)
    return param.options


def first_token(value) -> str:
    return str(value).split(" ")[0]


def match_option(value, options) -> str | None:
    """Fizgig's combobox rule: an exact option, else the option whose first token matches ("2e-4" selects
    "2e-4 - rank 4/8 only"), else None."""
    s = str(value)
    if s in options:
        return s
    tok = first_token(s)
    for opt in options:
        if first_token(opt) == tok:
            return str(opt)
    return None


def coerce(param: Param, value):
    """A stored value as the parameter's type (Fizgig stores numbers as typed strings). Raises ValueError."""
    if param.kind == BOOL:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if param.kind == INT:
        v = int(float(str(value).strip()))
    elif param.kind == FLOAT:
        v = float(str(value).strip())
    else:
        return "" if value is None else str(value)
    if param.minimum is not None and v < param.minimum:
        raise ValueError(f"{param.label} must be at least {param.minimum:g}")
    if param.maximum is not None and v > param.maximum:
        raise ValueError(f"{param.label} must be at most {param.maximum:g}")
    return v


def family_shows(param: Param, desc) -> bool:
    """Whether the family declares the feature a family_only parameter belongs to."""
    f = param.family_only
    if not f or desc is None:
        return True
    if f == "adapter":
        return bool(desc.training_adapter)
    if f == "ema":
        return bool(desc.ema_default)
    if f == "precision":
        return len(desc.precisions) > 1
    if f == "lokr":
        return "lokr" in desc.network_types
    if f == "edit":
        return bool(desc.edit_training)
    if f == "speed":
        return bool(desc.preview_speed())
    if f == "option":
        return param.key in desc.family_options
    return True


@dataclass
class ApplyReport:
    applied: dict = field(default_factory=dict)
    refused: list = field(default_factory=list)      # (key, value, reason) - shown as console / status lines
    ignored: list = field(default_factory=list)      # unknown keys (forward / backward compatibility)
    notes: list = field(default_factory=list)        # info lines, e.g. a legacy key that was mapped (not a refusal)

    def messages(self) -> list:
        return [f"[preset] {k}: saved value {v!r} isn't offered here - keeping the current one ({why})"
                for k, v, why in self.refused] + list(self.notes)
