"""Building and controlling a training run, independent of the UI - Fizgig's run contract (audit sections 1.3-1.5).

A run is a folder (<output folder>/<LoRA name>/) holding everything the child processes need, frozen at Start so
edits made in the Train tab during a run cannot retarget it:

    dataset.json          the dataset settings (folders, resolution, buckets, captions, augmentation)
    train_config.json     every trainer argument (training.train.train_family keyword arguments)
    sample_prompts.txt    the preview prompts
    run.log               the console, appended by the Train tab
    <name>-NNNNNN.safetensors, <name>.safetensors, <name>-NNNNNN-state/, sample/, loss_log/

Stages run as child processes with this interpreter: cache latents -> cache text -> train (the cache stages are
skipped on resume or with cache preparation off). Control is file-based: .pause_requested (honoured at the next epoch
boundary), .sample_override.json (the next preview), loss_log/caption_updates.json (caption fixes at the next
boundary). Stop kills the process tree.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from training import params as P
from training.dataset import IMAGE_EXTENSIONS, resolution_for_megapixels

PAUSE_FILE = ".pause_requested"
OVERRIDE_FILE = ".sample_override.json"
PAUSED_SIDECAR = ".tagscriber_paused.json"


# ---- locations ----------------------------------------------------------------------------------------------
def default_output_dir() -> Path:
    from core import paths
    return paths.USER_DATA / "training_runs"


def cache_root() -> Path:
    from core import paths
    return paths.USER_DATA / "training_cache"


def cache_dir_for(folder: str, root: Path | None = None) -> Path:
    """One cache folder per dataset folder: <root>/<sanitised name>-<sha1(lowercased path)[:8]> (Fizgig's rule)."""
    p = os.path.abspath(folder)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(p.rstrip("\\/")) or "dataset")[:40]
    digest = hashlib.sha1(p.lower().encode("utf-8")).hexdigest()[:8]
    return Path(root or cache_root()) / f"{name}-{digest}"


def run_dir_for(values: dict) -> Path:
    base = Path(str(values.get("LORA_OUTPUT_DIR") or "").strip() or default_output_dir())
    return base / str(values.get("LORA_NAME") or "my_lora").strip()


def model_setting_key(pref_key: str) -> str:
    return f"training.models.{pref_key}"


# ---- value helpers ------------------------------------------------------------------------------------------
def _num(values, key, kind=float):
    param = P.BY_KEY[key]
    v = values.get(key, param.default)
    try:
        return kind(float(str(v).strip())) if kind is int else kind(str(v).strip())
    except (TypeError, ValueError):
        return param.default


def precision_key(values, desc) -> str:
    if len(desc.precisions) <= 1:
        return desc.precisions[0]
    lab = str(values.get("FAMILY_PRECISION") or "")
    key = next((k for k, v in P.PRECISION_LABELS.items() if v == lab or k == P.first_token(lab).lower()), "auto")
    return key if key == "auto" or key in desc.precisions else "auto"


def model_path(desc, models: dict, role: str) -> str:
    """The file a role resolves to: its own row, else the row it defaults to (SDXL: the VAE and text-encoder rows
    fall back to the checkpoint, which holds both), else ""."""
    row = next((f for f in desc.model_files if f.role == role), None)
    if row is None:
        return ""
    path = (models.get(row.pref_key) or "").strip()
    if not path and row.default_to and row.default_to != role:
        return model_path(desc, models, row.default_to)
    return path


def slider_on(desc, values, source=None) -> bool:
    """Slider mode for a family that offers it; source "pairs" / "prompts" narrows it to one kind (Fizgig 7.0.1)."""
    on = bool(desc.slider_training and values.get("FAMILY_SLIDER"))
    if on and source:
        on = (str(values.get("FAMILY_SLIDER_SOURCE") or "pairs").strip() or "pairs") == source
    return on


def edit_on(desc, values) -> bool:
    return bool(desc.edit_training and values.get("FAMILY_EDIT"))


def _choice_word(values, key) -> str:
    return P.first_token(values.get(key) or P.BY_KEY[key].default).lower()


def alt_base_on(desc, values) -> bool:
    """Whether the run trains on the family's alternative base (MiniMax H3: Training base Reference (ref2va))."""
    return bool(desc.alt_base) and _choice_word(values, desc.alt_base[0]) == desc.alt_base[1].lower()


def dit_key(desc, values) -> str:
    """The pref key of the base model file this run trains on."""
    return desc.alt_base[2] if alt_base_on(desc, values) else desc.pref_for("dit")


def training_adapter_key(desc, values) -> str:
    """The pref key of the training adapter this run uses ("" = none): the family's adapter choice (MiniMax H3's
    Circlestone / Ostris per base / Off), else its one adapter when the tick is on."""
    if desc.adapter_choice:
        word = _choice_word(values, desc.adapter_choice)
        for w, std, alt in desc.adapter_files:
            if w.lower() == word:
                return alt if alt_base_on(desc, values) else std
        return ""
    return desc.training_adapter if values.get("FAMILY_TRAINING_ADAPTER", True) else ""


def turbo_strength(desc, values) -> float:
    """The preview speed LoRA's strength (0 = previews without it): the Samples tab's Turbo strength (-1 = the family
    default), or for a family with the "N steps at M%" row (samples_turbo_pace) its percentage."""
    default = desc.preview_speed_defaults()[1] if desc.preview_speed() else 0.0
    if desc.samples_turbo_pace:
        raw = values.get("FAMILY_TURBO_PACE")
        return default if raw in (None, "") else float(raw) / 100.0
    ts = _num(values, "FAMILY_TURBO_STRENGTH")
    return default if ts < 0 else ts


def driver_options(desc, values) -> dict:
    """The family-extension settings (P.DRIVER_OPTIONS) this family offers, as driver.configure() keywords."""
    out = {}
    for key in desc.family_options:
        name, param = P.DRIVER_OPTIONS.get(key), P.BY_KEY.get(key)
        if name is None or param is None:
            continue
        if param.kind == P.BOOL:
            out[name] = bool(values.get(key, param.default))
        elif param.kind in (P.CHOICE, P.TEXT):          # Klein: timestep sampling, model area, custom blocks
            out[name] = str(values.get(key, param.default) or "")
        else:
            out[name] = _num(values, key)
    return out


def blocks_to_swap(values) -> int:
    raw = str(values.get("BLOCKS_SWAP") or "").strip()
    if not raw or raw.lower().startswith("auto"):
        return -1
    m = re.match(r"\d+", raw)
    return int(m.group()) if m else -1


def ema_decay(values, desc):
    """The run's EMA decay: a number, 0.0 for off, or "short" (Short run, families with ema_short_run)."""
    if not desc.ema_default:
        return 0.0
    tok = P.first_token(values.get("FAMILY_EMA") or desc.ema_default)
    if tok.lower() == "off":
        return 0.0
    if tok.lower() == "short":
        return "short" if desc.ema_short_run else float(P.first_token(desc.ema_default))
    try:
        return float(tok)
    except ValueError:
        return 0.0


def network_type(values, desc) -> str:
    return "lokr" if "lokr" in desc.network_types and str(values.get("NETWORK_TYPE", "")).startswith("LoKR") \
        else "lora"


def sample_prompts(values) -> list[str]:
    lines = []
    for raw in str(values.get("SAMPLE_PROMPT") or "").splitlines():
        ln = raw.strip()
        if ln and not ln.startswith("#"):
            lines.append(ln)
    return lines


def edit_pairs(image_folder: str, originals: str) -> tuple[list, list, str | None]:
    """(matched stems, images with no partner, the first partner's path) for an Edit LoRA's or an image-pair slider's
    two folders, matched as the dataset matches them (training.dataset.partners)."""
    from training.dataset import partners
    def names(folder):
        try:
            return sorted(n for n in os.listdir(folder) if os.path.splitext(n)[1].lower() in IMAGE_EXTENSIONS)
        except OSError:
            return []
    before = names(originals)
    matched, missing, first = [], [], None
    for n in names(image_folder):
        stem = os.path.splitext(n)[0]
        m = partners(stem, before)
        if m:
            matched.append(stem)
            first = first or os.path.join(originals, m[0])
        else:
            missing.append(stem)
    return matched, missing, first


def pair_problems(image_folder: str, other: str) -> tuple[list, list]:
    """(images matching more than one partner, pairs whose sizes differ) - a slider's two ends must be one partner
    each, at the same framing (Fizgig 7.0.1 pair_problems)."""
    from PIL import Image

    from training.dataset import partners
    try:
        before = sorted(os.listdir(other))
        after = sorted(n for n in os.listdir(image_folder) if os.path.splitext(n)[1].lower() in IMAGE_EXTENSIONS)
    except OSError:
        return [], []
    multiple, shapes = [], []
    for n in after:
        m = partners(os.path.splitext(n)[0], before)
        if len(m) > 1:
            multiple.append(n)
        elif len(m) == 1:
            try:
                with Image.open(os.path.join(image_folder, n)) as a, Image.open(os.path.join(other, m[0])) as b:
                    if a.size != b.size:
                        shapes.append(n)
            except OSError:
                pass
    return multiple, shapes


def edit_instruction(values, image_folder: str) -> str:
    text = str(values.get("FAMILY_EDIT_CAPTION") or "").strip()
    if text:
        return text
    ext = str(values.get("DATASET_CAPTION_EXT") or ".txt")
    try:
        for f in sorted(os.listdir(image_folder)):
            b, e = os.path.splitext(f)
            cap = os.path.join(image_folder, b + ext)
            if e.lower() in IMAGE_EXTENSIONS and os.path.exists(cap):
                with open(cap, encoding="utf-8", errors="replace") as fh:
                    line = fh.read().strip().splitlines()
                if line and line[0].strip():
                    return line[0].strip()
    except OSError:
        pass
    return ""


def bitsandbytes_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("bitsandbytes") is not None


# ---- preflight ----------------------------------------------------------------------------------------------
@dataclass
class Check:
    level: str            # "error" (Start refused) | "warning" | "info"
    message: str


def preflight(desc, values: dict, image_folder: str, models: dict, *, captioner: dict | None = None,
              resume: str = "") -> list[Check]:
    """Everything that would fail later, checked before Start (Fizgig's validate_inputs and friends)."""
    from training.train_utils import validate_output_name
    out: list[Check] = []
    err = lambda m: out.append(Check("error", m))  # noqa: E731
    warn = lambda m: out.append(Check("warning", m))  # noqa: E731
    info = lambda m: out.append(Check("info", m))  # noqa: E731
    try:
        validate_output_name(str(values.get("LORA_NAME") or ""))
    except ValueError as e:
        err(str(e))
    prompt_slider = slider_on(desc, values, "prompts")
    if prompt_slider:
        for k, what in (("FAMILY_SLIDER_BASE", "what the picture is"), ("FAMILY_SLIDER_POS", "what the +1 end adds"),
                        ("FAMILY_SLIDER_NEG", "what the -1 end adds")):
            if not str(values.get(k) or "").strip():
                err(f"Slider from prompts: fill in {what}.")
        info("Slider from prompts: no dataset is used; the model renders its own practice pictures first.")
    if edit_on(desc, values) and slider_on(desc, values):
        err("Edit LoRA and Slider are both on - a run is one kind of LoRA: pick Edit or Slider.")
    if prompt_slider:
        image_folder = ""
    elif not image_folder or not os.path.isdir(image_folder):
        err("Open a dataset folder first (the Train tab trains the folder open in the workspace).")
        return out
    ext = str(values.get("DATASET_CAPTION_EXT") or ".txt")
    if not image_folder:
        names = []
    else:
        try:
            names = os.listdir(image_folder)
        except OSError as e:
            err(f"Can't read the dataset folder: {e}")
            return out
    imgs = [n for n in names if os.path.splitext(n)[1].lower() in IMAGE_EXTENSIONS]
    have = {os.path.splitext(n)[0] for n in names if n.lower().endswith(ext.lower())}
    missing = [n for n in imgs if os.path.splitext(n)[0] not in have]
    if prompt_slider:
        pass
    elif not imgs:
        err("The dataset folder has no images.")
    elif missing:
        err(f"{len(missing)} image(s) have no {ext} caption (e.g. {', '.join(missing[:3])}). Caption them (Gallery: "
            f"filter missing:caption) or move them out - Fizgig refuses to start in this case too.")
    else:
        info(f"{len(imgs)} captioned image(s).")
    if slider_on(desc, values, "pairs"):
        other = str(values.get("FAMILY_SLIDER_DIR") or "").strip()
        if not other or not os.path.isdir(other):
            err("Slider: set the -1 end folder (each picture's other end, the same file names).")
        else:
            matched, missing_p, _ = edit_pairs(image_folder, other)
            multiple, shapes = pair_problems(image_folder, other)
            if missing_p:
                err(f"Slider: {len(missing_p)} +1 picture(s) have no -1 picture with the same file name in the -1 "
                    f"end folder (e.g. {', '.join(missing_p[:3])}).")
            if multiple:
                err(f"Slider: {len(multiple)} +1 picture(s) match more than one -1 picture (e.g. "
                    f"{', '.join(multiple[:3])}) - keep one per picture, with the same file name.")
            if shapes:
                err(f"Slider: {len(shapes)} picture(s) have a different size at the two ends (e.g. "
                    f"{', '.join(shapes[:3])}) - use the same framing for both.")
            if matched and not (missing_p or multiple or shapes):
                info(f"Slider: {len(matched)} image pair(s).")
    adapter_key, base_key = training_adapter_key(desc, values), dit_key(desc, values)
    for f in desc.model_files:
        path = (models.get(f.pref_key) or "").strip()
        if f.pref_key == base_key and not f.required and not os.path.isfile(path):
            err(f"Model file missing: {f.label} - the chosen training base; set it under Model files" +
                (f" ({f.repo})" if f.repo else ""))
            continue
        used = f.role in ("vae", "text_encoder") or f.pref_key in (adapter_key, base_key) or \
            (f.role == "speed_lora" and values.get("SAMPLE_ENABLED") and turbo_strength(desc, values) != 0) or \
            (f.role == "preview_dit" and values.get("SAMPLE_ENABLED") and values.get("SAMPLE_USE_DISTILLED", True))
        if f.required and not os.path.isfile(path):
            err(f"Model file missing: {f.label} - set it under Model files" + (f" ({f.repo})" if f.repo else ""))
        elif used and path and not os.path.isfile(path):
            warn(f"{f.label}: {path} not found")
        elif f.pref_key == adapter_key and not path:
            warn(f"{f.label} is not set - {desc.training_adapter_note or 'training runs without it'}")
    try:
        bs = int(values.get("DATASET_BATCH_SIZE") or 1)
    except ValueError:
        bs = 1
    driver_batches = _driver_supports_batching(desc)
    if bs > 1 and not driver_batches:
        err(f"{desc.display_name} trains at batch size 1 - use Gradient accumulation for a bigger effective batch.")
    watch_on = any(values.get(k) for k in ("KREA2_LOSS_WATCH", "KREA2_PER_IMAGE_LR", "KREA2_AUTO_RECAPTION",
                                            "KREA2_WARMUP_LOOK"))
    if watch_on and bs > 1:
        warn("Problem-image detection, per-image LR and auto-recaption need batch size 1 - they are skipped.")
    if values.get("KREA2_AUTO_RECAPTION") and not (captioner or {}).get("model"):
        warn("Auto-recaption is on but no captioner is chosen - it will be off for this run.")
    if values.get("KREA2_PER_IMAGE_LR") and not values.get("KREA2_LOSS_WATCH"):
        info("Per-image LR uses the problem-image detector; it runs even with 'Detect problem images' off.")
    opt = str(values.get("OPTIMIZER_TYPE") or "")
    if "8bit" in opt and not bitsandbytes_available():
        warn(f"{opt} needs the bitsandbytes package, which isn't installed - the run will fall back to AdamW "
             f"(fp32 state; a little more VRAM, same recipe otherwise).")
    if values.get("ADAPTIVE_LR"):
        lo, hi = float(P.first_token(values.get("ADAPTIVE_LR_MIN", "1e-5"))), \
            float(P.first_token(values.get("ADAPTIVE_LR_MAX", "4e-4")))
        if lo > hi:
            err("Adaptive LR: Min LR is above Max LR.")
    if values.get("FAMILY_EDIT"):
        orig = str(values.get("FAMILY_EDIT_DIR") or "")
        if not os.path.isdir(orig):
            err("Edit LoRA: choose the originals folder (the 'before' images).")
        else:
            matched, missing_o, _ = edit_pairs(image_folder, orig)
            if missing_o:
                err(f"Edit LoRA: {len(missing_o)} edited image(s) have no original with the same name "
                    f"(e.g. {', '.join(missing_o[:3])}).")
            elif matched:
                info(f"Edit LoRA: {len(matched)} before/after pair(s).")
    if resume:
        if not os.path.isfile(os.path.join(resume, "training_state.json")):
            err(f"Resume: {resume} is not a saved training state.")
        else:
            try:
                with open(os.path.join(resume, "training_state.json"), encoding="utf-8") as f:
                    done = int(json.load(f).get("epoch", 0))
                if done >= int(values.get("MAX_TRAIN_EPOCHS") or 0):
                    err(f"Resume: that state already has {done} epoch(s) - raise Epochs above {done} to train more.")
            except (OSError, ValueError):
                err(f"Resume: {resume} has an unreadable training_state.json.")
    out += _exif_rotation_check(image_folder, imgs)
    return out


def _driver_supports_batching(desc) -> bool:
    """The driver's supports_batching flag, read without importing torch-heavy model code where possible."""
    try:
        import importlib
        mod, _, cls = desc.driver.partition(":")
        return bool(getattr(getattr(importlib.import_module(mod), cls), "supports_batching", False))
    except Exception:
        return False


def _exif_rotation_check(folder, names, limit=400) -> list[Check]:
    """Training reads pixels as stored (as Fizgig does), ignoring EXIF rotation - photos shown upright here would
    train sideways. Bake the rotation in with the Image Editor first."""
    try:
        from PIL import Image
    except ImportError:
        return []
    rotated = []
    for n in names[:limit]:
        if os.path.splitext(n)[1].lower() not in (".jpg", ".jpeg", ".webp"):
            continue
        try:
            with Image.open(os.path.join(folder, n)) as im:
                if im.getexif().get(0x0112, 1) not in (1, 0):
                    rotated.append(n)
        except Exception:
            continue
    if rotated:
        return [Check("warning", f"{len(rotated)} photo(s) are rotated only by EXIF (e.g. {', '.join(rotated[:3])}); "
                                 f"training would see them sideways. Edit and save them in the Image Editor (saving "
                                 f"bakes the rotation into the pixels).")]
    return []


# ---- building a run -----------------------------------------------------------------------------------------
@dataclass
class Run:
    family: str
    run_dir: Path
    output_name: str
    stages: list = field(default_factory=list)       # [(label, argv)]
    total_epochs: int = 1
    resume: str = ""

    def to_dict(self):
        return {"family": self.family, "run_dir": str(self.run_dir), "output_name": self.output_name,
                "stages": [[lab, list(argv)] for lab, argv in self.stages], "total_epochs": self.total_epochs,
                "resume": self.resume}


def dataset_config(desc, values: dict, image_folder: str, cache_root_dir: Path | None = None) -> dict:
    mp = float(str(values.get("DATASET_MEGAPIXELS") or "0.25").split(" ")[0])
    side = resolution_for_megapixels(mp)
    ds = {"image_directory": os.path.abspath(image_folder),
          "cache_directory": str(cache_dir_for(image_folder, cache_root_dir)),
          "num_repeats": max(1, int(values.get("DATASET_REPEATS") or 1))}
    if edit_on(desc, values) and not slider_on(desc, values) and values.get("FAMILY_EDIT_DIR"):
        ds["control_directory"] = os.path.abspath(str(values["FAMILY_EDIT_DIR"]))
    elif slider_on(desc, values, "pairs") and values.get("FAMILY_SLIDER_DIR"):
        ds["control_directory"] = os.path.abspath(str(values["FAMILY_SLIDER_DIR"]))   # the -1 end
    return {"datasets": [ds], "resolution": [side, side],
            "caption_extension": str(values.get("DATASET_CAPTION_EXT") or ".txt"),
            "batch_size": max(1, int(values.get("DATASET_BATCH_SIZE") or 1)),
            "enable_bucket": bool(values.get("ENABLE_BUCKET", True)),
            "bucket_no_upscale": bool(values.get("BUCKET_NO_UPSCALE", True)),
            "caption_shuffle_variants": max(0, int(values.get("CAPTION_SHUFFLE_VARIANTS") or 0)),
            "caption_keep_tokens": max(0, int(values.get("CAPTION_KEEP_TOKENS") or 0)),
            "caption_dropout": max(0.0, float(values.get("CAPTION_DROPOUT") or 0.0)),
            "caption_separator": ","}


def train_kwargs(desc, values: dict, run_dir: Path, models: dict, *, captioner: dict | None = None,
                 trigger: str = "", trigger_position: str = "start", image_folder: str = "",
                 resume: str = "") -> tuple[dict, list[str]]:
    """training.train.train_family keyword arguments for these settings (Fizgig's _generic_train_command), and the
    preview prompts."""
    m = lambda role: model_path(desc, models, role)  # noqa: E731
    kw = {
        "family": desc.key, "output_dir": str(run_dir),
        "dit_path": (models.get(dit_key(desc, values)) or "").strip() if alt_base_on(desc, values) else m("dit"),
        "output_name": str(values.get("LORA_NAME")).strip(),
        "network_dim": _num(values, "NETWORK_DIM", int), "network_alpha": _num(values, "NETWORK_ALPHA"),
        "learning_rate": _num(values, "LEARNING_RATE"), "max_train_epochs": _num(values, "MAX_TRAIN_EPOCHS", int),
        "save_every_n_epochs": _num(values, "SAVE_EVERY_N_EPOCHS", int), "seed": _num(values, "SEED", int),
        "save_state": bool(values.get("SAVE_STATE", True)),
        "save_state_on_train_end": bool(values.get("SAVE_STATE_ON_TRAIN_END", True)),
        "keep_last_n_states": max(1, _num(values, "KEEP_LAST_N_STATES", int)),
        "precision": precision_key(values, desc), "blocks_to_swap": blocks_to_swap(values),
        "min_timestep": _num(values, "MIN_TIMESTEP"), "max_timestep": _num(values, "MAX_TIMESTEP"),
        "max_grad_norm": _num(values, "MAX_GRAD_NORM"), "ema_decay": ema_decay(values, desc),
        "optimizer_type": str(values.get("OPTIMIZER_TYPE") or "adamw8bit").strip(),
        "optimizer_args": str(values.get("OPTIMIZER_ARGS") or "").strip(),
        "lr_scheduler": str(values.get("LR_SCHEDULER") or "constant"),
        "lr_warmup_steps": _num(values, "LR_WARMUP_STEPS", int),
        "network_type": network_type(values, desc), "lokr_factor": _num(values, "LOKR_FACTOR", int),
        "gradient_accumulation": max(1, _num(values, "GRADIENT_ACCUMULATION", int)),
        "vae_path": m("vae") or None, "te_path": m("text_encoder") or None,
        "resume_state_dir": resume or None,
    }
    if desc.identity_blocks and values.get("FAMILY_FAST_ID") and not (edit_on(desc, values) or
                                                                       slider_on(desc, values)):
        kw["train_blocks"] = list(desc.identity_blocks)          # Fast Identity Mode (Fizgig --train_blocks)
    if slider_on(desc, values):                                  # Fizgig 7.0.1 launch.py's slider flags
        if desc.slider_ultra_blocks and values.get("FAMILY_SLIDER_ULTRA"):
            kw["train_blocks"] = list(desc.slider_ultra_blocks)
        if slider_on(desc, values, "prompts"):
            base = str(values.get("FAMILY_SLIDER_BASE") or "").strip()
            g = _num(values, "FAMILY_SLIDER_GUIDANCE")
            kw.update(slider_prompts=[base, f"{base} {str(values.get('FAMILY_SLIDER_POS') or '').strip()}".strip(),
                                      f"{base} {str(values.get('FAMILY_SLIDER_NEG') or '').strip()}".strip()],
                      slider_guidance=g if g > 0 else desc.slider_guidance)
            # the practice pictures (and so the training) at Target megapixels, a square on the 16 px grid
            mp = float(str(values.get("DATASET_MEGAPIXELS") or "0.25").split(" ")[0])
            side = int((mp * 1_000_000) ** 0.5) // 16 * 16
            if side >= 256:
                kw["slider_bank_res"] = side
        else:
            kw["slider_pairs"] = True
    opts = driver_options(desc, values)
    if opts:
        kw["driver_options"] = opts
    if values.get("ADAPTIVE_LR"):
        kw.update(adaptive_lr=True, adaptive_lr_min=float(P.first_token(values.get("ADAPTIVE_LR_MIN", "1e-5"))),
                  adaptive_lr_max=float(P.first_token(values.get("ADAPTIVE_LR_MAX", "4e-4"))))
    akey = training_adapter_key(desc, values)
    adapter = (models.get(akey) or "").strip() if akey else ""
    if adapter and os.path.isfile(adapter):
        kw["training_adapter"] = adapter
    ctx = str(values.get("CONTEXT_LORA_PATH") or "").strip()
    if ctx:
        kw.update(context_lora_path=ctx, context_lora_strength=_num(values, "CONTEXT_LORA_STRENGTH"))
    bs1 = int(values.get("DATASET_BATCH_SIZE") or 1) <= 1
    if bs1:
        kw.update(log_per_image_loss=bool(values.get("KREA2_LOSS_WATCH")),
                  per_image_lr=bool(values.get("KREA2_PER_IMAGE_LR")),
                  warmup_look_outliers=bool(values.get("KREA2_WARMUP_LOOK")))
        if values.get("KREA2_AUTO_RECAPTION") and (captioner or {}).get("model"):
            kw.update(auto_recaption=True, captioner=dict(captioner))
    if trigger:
        kw.update(trigger_word=trigger, trigger_position=trigger_position)
    for key in ("TITLE", "AUTHOR", "DESCRIPTION", "LICENSE", "TAGS", "THUMBNAIL"):
        v = str(values.get(f"METADATA_{key}") or "").strip()
        if v:
            kw[f"metadata_{key.lower()}"] = v
    trig = str(values.get("METADATA_TRIGGER_PHRASE") or "").strip() or trigger
    if trig:
        kw["metadata_trigger_phrase"] = trig
    prompts: list[str] = []
    every = _num(values, "SAMPLE_EVERY_N_EPOCHS", int)
    if values.get("SAMPLE_ENABLED") and every > 0:
        edit = edit_on(desc, values) and not slider_on(desc, values)
        prompts = [edit_instruction(values, image_folder)] if edit else sample_prompts(values)
        prompts = [p for p in prompts if p]
        if prompts:
            kw.update(sample_every_n_epochs=every, sample_at_first=bool(values.get("SAMPLE_AT_FIRST")),
                      sample_seed=_num(values, "SAMPLE_SEED", int))
            for key, arg in (("SAMPLE_WIDTH", "sample_width"), ("SAMPLE_HEIGHT", "sample_height"),
                             ("SAMPLE_STEPS", "sample_steps")):
                v = _num(values, key, int)
                if v > 0:
                    kw[arg] = v
            cfg = _num(values, "SAMPLE_CFG_SCALE")
            kw["sample_cfg_scale"] = cfg if cfg > 0 else desc.preview_cfg
            neg = str(values.get("SAMPLE_NEGATIVE") or "").strip()
            if neg and kw["sample_cfg_scale"] > 1.0:
                kw["sample_negative"] = neg
            sp = desc.preview_speed()
            speed_path = (models.get(sp.pref_key) or "").strip() if sp and sp.pref_key else ""
            if speed_path and os.path.isfile(speed_path):
                default = desc.preview_speed_defaults()[1]
                ts = turbo_strength(desc, values)
                if ts > 0:
                    kw["speed_lora"] = speed_path
                    if abs(ts - default) > 1e-9:
                        kw["speed_lora_strength"] = max(0.0, min(2.0, ts))
                    if desc.samples_turbo_pace:             # H3: the Turbo row's own steps (Fizgig 7.0.1)
                        kw["sample_steps"] = _num(values, "FAMILY_TURBO_STEPS", int) or \
                            desc.preview_speed_defaults()[0]
                    # The Train tab fills Steps with the family's plain-model default; with the speed LoRA on, the
                    # untouched default means "the LoRA's own steps" (Krea 2: 28 -> 8; unchanged for Qwen, whose
                    # two are equal)
                    elif kw.get("sample_steps") == desc.preview_steps:
                        kw["sample_steps"] = desc.preview_speed_defaults()[0]
            ck = desc.preview_checkpoint()
            if desc.train_preview_checkpoint and ck and values.get("SAMPLE_USE_DISTILLED", True):
                ck_path = (models.get(ck[0].pref_key) or "").strip()
                if ck_path and os.path.isfile(ck_path):        # Klein's "Use Distilled model for samples"
                    kw.update(preview_checkpoint=ck_path,
                              preview_checkpoint_cache=str(values.get("CACHE_SAMPLE_MODEL") or "auto"),
                              preview_int8=bool(values.get("PREVIEW_INT8")))
            if edit:
                ref = str(values.get("FAMILY_EDIT_REF") or "").strip() or \
                    (edit_pairs(image_folder, str(values.get("FAMILY_EDIT_DIR") or ""))[2] or "")
                if ref:
                    kw["sample_reference"] = [ref]
    return kw, prompts


def build_run(desc, values: dict, image_folder: str, models: dict, *, captioner: dict | None = None,
              trigger: str = "", trigger_position: str = "start", resume: str = "", enable_cache: bool = True,
              python: str | None = None, cache_root_dir: Path | None = None) -> Run:
    """Write the frozen run folder and return the stages to launch. Call preflight() first."""
    run_dir = run_dir_for(values)
    run_dir.mkdir(parents=True, exist_ok=True)
    image_folder = image_folder or ""                     # a prompt slider trains without a dataset folder
    ds = dataset_config(desc, values, image_folder, cache_root_dir)
    ds_path = run_dir / "dataset.json"
    ds_path.write_text(json.dumps(ds, indent=2), encoding="utf-8")
    kw, prompts = train_kwargs(desc, values, run_dir, models, captioner=captioner, trigger=trigger,
                               trigger_position=trigger_position, image_folder=image_folder, resume=resume)
    prompts_path = run_dir / "sample_prompts.txt"
    prompts_path.write_text("\n".join(prompts) + ("\n" if prompts else ""), encoding="utf-8")
    cfg_path = run_dir / "train_config.json"
    cfg_path.write_text(json.dumps({"family": desc.key, "dataset_config": str(ds_path),
                                    "sample_prompts_file": str(prompts_path), "train": kw,
                                    "settings": {k: values.get(k) for k in P.BY_KEY}}, indent=2, default=str),
                        encoding="utf-8")
    clear_pause(run_dir)
    py = python or sys.executable
    stages = []
    if enable_cache and not resume and not slider_on(desc, values, "prompts"):   # a prompt slider has no dataset
        for stage, role in (("latents", "vae"), ("text", "text_encoder")):
            stages.append((f"Caching {stage}", [py, "-m", "training.cache", "--family", desc.key, "--stage", stage,
                                                "--dataset", str(ds_path), "--model",
                                                model_path(desc, models, role), "--skip_existing"]
                           + (["--slider"] if slider_on(desc, values, "pairs") else [])))
    stages.append(("Training", [py, "-m", "training.train", "--config", str(cfg_path)]))
    return Run(desc.key, run_dir, kw["output_name"], stages, kw["max_train_epochs"], resume)


# ---- control files ------------------------------------------------------------------------------------------
def request_pause(run_dir) -> None:
    Path(run_dir, PAUSE_FILE).write_text("", encoding="utf-8")


def clear_pause(run_dir) -> None:
    try:
        Path(run_dir, PAUSE_FILE).unlink()
    except FileNotFoundError:
        pass


def pause_requested(run_dir) -> bool:
    return Path(run_dir, PAUSE_FILE).exists()


def write_sample_override(run_dir, prompt: str, seed: int = 1234, width: int = 1024, height: int = 1024) -> None:
    from core.caption_io import atomic_write_text
    atomic_write_text(Path(run_dir, OVERRIDE_FILE), json.dumps({"prompt": prompt, "seed": int(seed),
                                                                 "width": int(width), "height": int(height)}))


def read_sample_override(run_dir) -> dict | None:
    try:
        d = json.loads(Path(run_dir, OVERRIDE_FILE).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) and str(d.get("prompt", "")).strip() else None
    except (OSError, ValueError):
        return None


def clear_sample_override(run_dir) -> None:
    try:
        Path(run_dir, OVERRIDE_FILE).unlink()
    except FileNotFoundError:
        pass


def queue_caption_updates(run_dir, updates: dict) -> None:
    """Merge caption fixes into loss_log/caption_updates.json (the trainer claims it at the next epoch boundary)."""
    path = Path(run_dir, "loss_log", "caption_updates.json")
    cur = {}
    try:
        cur = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    from core.caption_io import atomic_write_text
    atomic_write_text(path, json.dumps({**cur, **updates}, indent=2, ensure_ascii=False))


def read_problem_images(run_dir) -> dict:
    try:
        return json.loads(Path(run_dir, "loss_log", "problem_images.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def read_applied_captions(run_dir) -> dict:
    try:
        return json.loads(Path(run_dir, "loss_log", "caption_updates_applied.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def samples(run_dir) -> list[Path]:
    d = Path(run_dir, "sample")
    if not d.is_dir():
        return []
    return sorted((p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg", ".webp")),
                  key=lambda p: (p.stat().st_mtime, p.name))


def parse_sample_name(name: str):
    """<name>_e<epoch:06d>_<idx:02d>_<timestamp>_<seed>.png -> (epoch, index, seed) or None."""
    m = re.search(r"_e(\d{6})_(\d{2})_(\d{14})_(\d+)\.\w+$", name)
    return (int(m.group(1)), int(m.group(2)), int(m.group(4))) if m else None


def checkpoints(run_dir, output_name) -> list[Path]:
    d = Path(run_dir)
    if not d.is_dir():
        return []
    pat = re.compile(rf"^{re.escape(output_name)}(-\d{{6}})?\.safetensors$")
    return sorted(p for p in d.iterdir() if pat.match(p.name))


def write_paused(run_dir, info: dict) -> None:
    from core.caption_io import atomic_write_text
    atomic_write_text(Path(run_dir, PAUSED_SIDECAR), json.dumps(info, indent=2))


def read_paused(run_dir) -> dict | None:
    try:
        return json.loads(Path(run_dir, PAUSED_SIDECAR).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def clear_paused(run_dir) -> None:
    try:
        Path(run_dir, PAUSED_SIDECAR).unlink()
    except FileNotFoundError:
        pass


# ---- processes ----------------------------------------------------------------------------------------------
def child_env(base: dict | None = None) -> dict:
    env = dict(base if base is not None else os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    from core import paths
    env["PYTHONPATH"] = str(paths.APP_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def lower_priority(pid: int) -> None:
    """Below-normal priority for the training process (Fizgig: lets the desktop compositor pre-empt the GPU, which
    fixes desktop judder for about 1% speed)."""
    try:
        import psutil
        p = psutil.Process(pid)
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:
        pass


def kill_tree(pid: int) -> None:
    """Stop: a hard kill of the process tree (nothing is saved - Pause is the clean exit)."""
    if not pid:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return
    try:
        import psutil
        parent = psutil.Process(pid)
        procs = parent.children(recursive=True) + [parent]
        for p in procs:
            try:
                p.terminate()
            except psutil.NoSuchProcess:
                pass
        _gone, alive = psutil.wait_procs(procs, timeout=5)
        for p in alive:
            p.kill()
    except Exception:
        pass
