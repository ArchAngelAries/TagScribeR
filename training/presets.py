"""Training presets, the last-run snapshot and the run queue - Fizgig's preset lifecycle (audit section 4.3).

* Built-in presets come from the family description (the first one is what a first visit to the family applies).
* User presets are flat JSON files of setting keys -> values in user_data/training_presets/<family>/<name>.json.
  They deliberately do NOT carry the family, so a preset can never switch your model; only the last-run snapshot
  and queue items record it (as the namespaced "__architecture__" key). A user preset with a built-in's name shows
  once (the built-in wins when loading).
* Applying a preset: unknown keys are ignored (forward / backward compatibility); a choice value matches its option
  by first token ("2e-4" selects "2e-4 - rank 4/8 only"); for strict choices (optimizer, Adaptive LR bounds,
  scheduler, network type, precision) a value the family doesn't offer is REFUSED with a message instead of being
  set; values of the wrong type are refused the same way.

Fizgig preset JSON files apply here unchanged (same keys and value formats). Fizgig's Krea 2 keys that this app
expresses differently are migrated first (migrate_legacy): KREA2_EMA -> FAMILY_EMA, QUANT_4BIT_MODE -> FAMILY_PRECISION
(fp8 maps to the fp8 base), blank noise-range boxes -> the defaults; torch.compile and the
fine-tune keys do nothing here and are ignored.

A Fizgig preset that switches on a training mode TagScribeR does not have yet (a slider LoRA, a full fine-tune) is
refused as a whole (UNSUPPORTED_MODES): applied without it, its settings would run as an ordinary LoRA.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from training import params as P

log = logging.getLogger(__name__)

ILLEGAL = '<>:"/\\|?*'
ARCH_KEY = "__architecture__"
DATASET_KEY = "__dataset__"


class PresetError(ValueError):
    pass


def _root() -> Path:
    from core import paths
    return paths.USER_DATA / "training_presets"


def _atomic_json(path: Path, data) -> None:
    from core.caption_io import atomic_write_text
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(data, indent=4, ensure_ascii=False))


def check_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise PresetError("Preset name can't be empty.")
    bad = sorted({c for c in name if c in ILLEGAL or c < " "})
    if bad:
        raise PresetError(f"Preset names can't contain {' '.join(bad)}")
    if len(name) > 120:
        raise PresetError("Preset name is too long (120 characters max).")
    return name


def collect(values: dict) -> dict:
    """The preset part of the current settings (Fizgig's _collect_preset_values): every preset key, nothing run-only."""
    return {k: values[k] for k in P.PRESET_KEYS if k in values}


# Fizgig keys that have no counterpart here (the Krea 2 rotating fine-tune; Klein's fp8 base, attention
# backend, LoRA dropout / LoRA+, fp8 text encoder, logging, LR decay, and the hidden loss-weighting boxes - the family's
# docstring lists each as not ported). TARGET_LAYERS / TRAINING_BLOCKS are real parameters (Klein's Model Area).
LEGACY_IGNORED = ("FP8", "SCALED", "NETWORK_DROPOUT", "LORA_LR_RATIO",
                  "FP8_TEXT_ENCODER", "IMG_IN_TXT_IN_OFFLOADING", "LOGGING_DIR", "LOG_WITH", "LOG_PREFIX",
                  "LR_DECAY_STEPS", "GRADIENT_CHECKPOINTING", "WEIGHTING_SCHEME", "MODE_SCALE")
LEGACY_AREAS = {"All Layers": "Full Model", "Identity Blocks": "Identity", "Style+Composition Blocks":
                "Style+Composition", "Details Blocks": "Details"}      # Fizgig lora_trainer_gui.py:6717-6722
LEGACY_IGNORED_PREFIXES = ("KREA2_FINETUNE",)


def _legacy_precision(value) -> str:
    """Fizgig's _normalize_base_precision: a canonical key, a combobox label or a legacy Auto / On / Off -> one of
    auto / int8 / nf4 / fp8. "Off" meant "not 4-bit", which Fizgig resolved to INT8; unknown values mean Auto."""
    v = str(value or "").strip().lower()
    if v in ("no_4bit", "off"):
        return "int8"
    if v == "on":
        return "nf4"
    for key, starts in (("int8", ("int8",)), ("nf4", ("nf4", "4-bit")), ("fp8", ("fp8",)), ("auto", ("auto",))):
        if v.startswith(starts):
            return key
    return "auto"


def _legacy_compile(value) -> str:
    """Fizgig's COMPILE_BLOCKS: the combobox label ("Auto" / "On" / "Off") or the settings value in lower case
    ("auto" | "on" | "off" | "outside", lora_trainer_gui.py:6589-6592, 34102-34108). Unknown values mean Auto."""
    v = str(value or "").strip().lower()
    return {"auto": "Auto", "on": "On", "off": "Off", "outside": "Outside", "true": "On", "false": "Off",
            "1": "On", "0": "Off"}.get(v, "Auto")


def _is_number(value) -> bool:
    try:
        float(str(value).strip())
        return True
    except ValueError:
        return False


# Fizgig's old MiniMax H3 keys: Fizgig 7.0.1 renamed them H3_* and reads the old names as aliases (families/minimax.py
# FamilyOption setting=...); so does this app
_MINIMAX_RENAMED = ("LOWNOISE_PCT", "HIGHNOISE_LR_PCT", "LIKENESS_MODE", "BLOCKS", "ADAPTER", "ADAPTER_RAMP",
                    "TRAIN_REFINER", "TRAIN_BASE", "CAPTION_DROPOUT", "TREAD", "CLIP_STILL", "DISTILL",
                    "DISTILL_WEIGHT", "DISTILL_REFS", "DISTILL_PHASE1", "FT_SCOPE")
# old H3 keys Fizgig itself deleted (0 hits in 7.0.1)
_MINIMAX_IGNORED = ("MINIMAX_TRAIN_ADALN", "MINIMAX_SLOW_BLOCKS", "MINIMAX_SLOW_LR_SCALE", "MINIMAX_BLOCK_LIMIT",
                    "MINIMAX_LR_WARMUP", "MINIMAX_LIKENESS_OPT", "MINIMAX_LOGNORM", "MINIMAX_TURBO_STEPS",
                    "MINIMAX_TURBO_STRENGTH")
# H3 options for clips, voice, distillation and the fine-tune: machinery this still-image port does not have. A
# switched-on one is reported; the clip ones (on in every Fizgig preset, inert on a folder of stills) pass quietly
_H3_NOT_HERE = ("H3_DISTILL", "H3_DISTILL_WEIGHT", "H3_DISTILL_REFS", "H3_DISTILL_PHASE1", "H3_MIXED_STOP_CATEGORY",
                "H3_MIXED_STOP_EPOCH", "H3_MIXED_STOP_MODE", "H3_FT_SCOPE", "H3_FT_BLOCKS", "H3_SAMPLE_FRAMES")
_H3_CLIP_ONLY = ("H3_TREAD", "H3_CLIP_STILL")


def _precision_label(value, key, notes):
    """Fizgig's H3 base-precision labels ("int8 · most accurate ...", "4-bit · ...", "4-bit HQQ · ...") -> this app's;
    HQQ is not offered, so it becomes NF4 with a note."""
    v = str(value or "").split("·")[0].strip().lower()
    prec = "int8" if v.startswith("int8") else ("nf4" if v.startswith(("4-bit", "nf4", "hqq")) else "auto")
    if "hqq" in v:
        notes.append(f"[preset] {key}: HQQ 4-bit isn't offered - using plain 4-bit NF4")
    return P.PRECISION_LABELS[prec]


def _h3_structure(pct) -> str:
    """The Training structure a clean-end share stands for (Fizgig's _STRUCTURE: 60, 8, else Custom)."""
    try:
        v = float(str(pct).rstrip("%"))
    except ValueError:
        return P.H3_STRUCTURES[0]
    return {60.0: P.H3_STRUCTURES[0], 8.0: P.H3_STRUCTURES[1]}.get(v, P.H3_STRUCTURES[2])


def _migrate_h3(key, value, preset, out, notes, ignored) -> None:
    """Fizgig's MiniMax H3 keys: the old MINIMAX_* names -> H3_* (an explicit H3_* key in the same preset wins), and the
    H3 keys this app expresses differently (EMA, base precision, caption dropout) or does not have."""
    if key.startswith("MINIMAX_") and key[len("MINIMAX_"):] in _MINIMAX_RENAMED:
        new = "H3_" + key[len("MINIMAX_"):]
        if new in preset:
            return
        if key == "MINIMAX_LOWNOISE_PCT" and "H3_STRUCTURE" not in preset:
            out["H3_STRUCTURE"] = _h3_structure(value)
        key = new
    if key == "MINIMAX_EMA":
        if "FAMILY_EMA" not in preset:
            out["FAMILY_EMA"] = value
    elif key == "MINIMAX_BASE_QUANT":
        if "FAMILY_PRECISION" not in preset:
            out["FAMILY_PRECISION"] = _precision_label(value, key, notes)
    elif key == "H3_CAPTION_DROPOUT":
        tok = P.first_token(value)
        if "CAPTION_DROPOUT" not in preset:
            out["CAPTION_DROPOUT"] = float(tok) if _is_number(tok) else 0.0
    elif key in _H3_CLIP_ONLY:
        ignored.append(key)
    elif key in _H3_NOT_HERE:
        if key in ("H3_DISTILL", "H3_MIXED_STOP_CATEGORY", "H3_FT_SCOPE") and _on(value):
            notes.append(f"[preset] {key}: not available in the still-image MiniMax H3 trainer - ignored")
        ignored.append(key)
    elif key in P.BY_KEY:
        out[key] = value
    elif key.startswith("MINIMAX_"):
        if key.startswith(("MINIMAX_REFMOD", "MINIMAX_CONCEPT", "MINIMAX_MULTICONCEPT", "MINIMAX_REG_",
                           "MINIMAX_FT_")) and _on(value):
            notes.append(f"[preset] {key}: not available in the still-image MiniMax H3 trainer - ignored")
        ignored.append(key)
    else:
        ignored.append(key)


def migrate_legacy(preset: dict) -> tuple[dict, list, list]:
    """Fizgig's Krea 2 preset keys -> this app's parameters. Returns (preset, notes, ignored). Explicit new-style keys
    in the same preset win over their legacy twins."""
    out, notes, ignored = {}, [], []
    for key, value in (preset or {}).items():
        if key == "KREA2_EMA":
            if "FAMILY_EMA" not in preset:
                out["FAMILY_EMA"] = value
        elif key in ("QUANT_4BIT_MODE", "QUANT_4BIT"):
            if key == "QUANT_4BIT" and "QUANT_4BIT_MODE" in preset:
                continue
            if key == "QUANT_4BIT":              # legacy boolean: False means "no opinion", not "fp8"
                prec = "nf4" if value in (True, "True", "true", 1, "1") else None
            else:
                prec = _legacy_precision(value)
            if prec and "FAMILY_PRECISION" not in preset:
                out["FAMILY_PRECISION"] = P.PRECISION_LABELS[prec]
        elif key in LEGACY_IGNORED or key.startswith(LEGACY_IGNORED_PREFIXES):
            ignored.append(key)
        elif key in _MINIMAX_IGNORED:
            ignored.append(key)
        elif key.startswith("MINIMAX_") or key in _H3_NOT_HERE + _H3_CLIP_ONLY + ("H3_CAPTION_DROPOUT",):
            _migrate_h3(key, value, preset, out, notes, ignored)
        elif key == "FAMILY_PRECISION" and "·" in str(value or ""):
            out[key] = _precision_label(value, key, notes)       # Fizgig 7.0.1's H3 labels
        elif key == "FAMILY_PRECISION" and str(value or "").strip().lower().startswith("as the file"):
            # Fizgig 7.0.1 Klein's "As the file (bf16 or fp8)": its recommended Base file is fp8, which TagScribeR's
            # fp8 choice keeps exactly as stored (a bf16 file is quantised to fp8 instead - pick bf16 for that file)
            out[key] = P.PRECISION_LABELS["fp8"]
            notes.append("[preset] FAMILY_PRECISION: Fizgig's 'As the file' is TagScribeR's fp8 (the fp8 Base file "
                         "as it is); with a bf16 Base file pick bf16")
        elif key == "COMPILE_BLOCKS":
            out[key] = _legacy_compile(value)
        elif key == "TARGET_LAYERS":
            out[key] = LEGACY_AREAS.get(value, value)
        elif key == "TRAINING_BLOCKS" and isinstance(value, (dict, str)):
            # Fizgig stores {block: ticked}; its older Training tab named Klein's blocks double_blocks.N, Fizgig 7.0.1
            # (and TagScribeR) double_N
            items = [k for k, on in value.items() if on] if isinstance(value, dict) else \
                [t for t in value.replace(",", " ").split() if t]
            out[key] = ", ".join(re.sub(r"^(double|single)_blocks[._](\d+)$", r"\1_\2", k.strip()) for k in items)
        elif key in ("MIN_TIMESTEP", "MAX_TIMESTEP") and str(value).strip() == "":
            out[key] = P.BY_KEY[key].default      # Fizgig: an empty noise-range box means the full range
        elif key in ("MIN_TIMESTEP", "MAX_TIMESTEP") and _is_number(value) and float(value) > 1.0:
            out[key] = float(value) / 1000.0      # Klein's boxes are 0-1000 (Fizgig); the generic range is 0-1
        else:
            out[key] = value
    return out, notes, ignored


# Fizgig preset keys whose True turns the run into something other than a LoRA - not available here yet (port plan
# stage 5: the full fine-tune); a slider preset is refused only for a family without slider training
UNSUPPORTED_MODES = {"FAMILY_SLIDER": "slider training", "FAMILY_FT": "full fine-tuning",
                     "KREA2_FINETUNE": "full fine-tuning", "MINIMAX_FINETUNE": "full fine-tuning"}


def _on(value) -> bool:
    return value not in (False, "False", "false", 0, "0", "", None, "Off", "off")


def unsupported_modes(preset: dict, desc=None) -> list:
    """(key, mode) for each training mode `preset` switches on that this app (or this family) cannot train."""
    return [(k, what) for k, what in UNSUPPORTED_MODES.items() if _on((preset or {}).get(k))
            and not (k == "FAMILY_SLIDER" and (desc is None or desc.slider_training))]


def apply(preset: dict, current: dict, desc=None) -> tuple[dict, P.ApplyReport]:
    """`current` updated with `preset` under Fizgig's validation rules. Returns (new values, report). A preset for
    a training mode this app does not have (unsupported_modes) changes nothing: report.blocked says why."""
    out = dict(current)
    rep = P.ApplyReport()
    rep.blocked = unsupported_modes(preset, desc)
    if rep.blocked:
        return out, rep
    preset, rep.notes, rep.ignored = migrate_legacy(preset)
    for key, value in (preset or {}).items():
        if key.startswith("__"):
            continue
        param = P.BY_KEY.get(key)
        if param is None:
            rep.ignored.append(key)
            continue
        if param.kind == P.CHOICE:
            opts = P.options_for(param, desc)
            hit = P.match_option(value, opts)
            if hit is None:
                if param.strict:
                    rep.refused.append((key, value, f"offered: {', '.join(map(str, opts))}"))
                    continue
                hit = str(value)
            out[key] = hit
        else:
            try:
                out[key] = P.coerce(param, value)
            except (TypeError, ValueError) as e:
                rep.refused.append((key, value, str(e)))
                continue
        rep.applied[key] = out[key]
    return out, rep


class TrainingPresets:
    """Built-in + user presets for one family."""

    def __init__(self, desc, root: Path | None = None):
        self.desc = desc
        self.dir = Path(root or _root()) / desc.key
        self.builtins = {name: dict(values) for name, values in desc.presets}

    @property
    def default_name(self) -> str | None:
        return next(iter(self.builtins), None)

    def user_names(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        return sorted((p.stem for p in self.dir.glob("*.json")), key=str.lower)

    def names(self) -> list[str]:
        """Built-ins first (in the family's order), then user presets; a user name equal to a built-in shows once."""
        return list(self.builtins) + [n for n in self.user_names() if n not in self.builtins]

    def is_builtin(self, name: str) -> bool:
        return name in self.builtins

    def path(self, name: str) -> Path:
        return self.dir / f"{check_name(name)}.json"

    def exists(self, name: str) -> bool:
        return name in self.builtins or self.path(name).is_file()

    def load(self, name: str) -> dict:
        if name in self.builtins:
            return dict(self.builtins[name])
        p = self.path(name)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise PresetError(f"No preset named '{name}'.") from None
        except (ValueError, UnicodeDecodeError) as e:
            raise PresetError(f"Preset file is corrupted: {p} ({e})") from None
        if not isinstance(data, dict):
            raise PresetError(f"Preset file is corrupted: {p}")
        return data

    def save(self, name: str, values: dict) -> Path:
        name = check_name(name)
        if name in self.builtins:
            raise PresetError(f"'{name}' is a built-in preset. Save your version under another name.")
        p = self.path(name)
        _atomic_json(p, collect(values))
        return p

    def delete(self, name: str) -> None:
        if name in self.builtins:
            raise PresetError("Built-in presets can't be deleted.")
        p = self.path(name)
        if p.is_file():
            from core.fileops import trash
            trash(p)

    def import_file(self, src: Path, name: str | None = None) -> str:
        """Copy a preset file (TagScribeR's or Fizgig's - same format) in as a user preset."""
        try:
            data = json.loads(Path(src).read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError) as e:
            raise PresetError(f"Couldn't read {src}: {e}") from None
        if not isinstance(data, dict) or not any(k in P.BY_KEY for k in data):
            raise PresetError("That file doesn't look like a training preset.")
        base = check_name(name or Path(src).stem)
        final, n = base, 2
        while self.exists(final):
            final, n = f"{base} ({n})", n + 1
        _atomic_json(self.path(final), data)       # verbatim: keys this app doesn't know survive a round trip
        return final


# ---- last run + queue (these DO record the family) -----------------------------------------------------
def last_run_path() -> Path:
    return _root() / ".last_train_settings.json"


def save_last_run(family_key: str, values: dict, dataset_folder: str = "") -> None:
    _atomic_json(last_run_path(), {**values, ARCH_KEY: family_key, DATASET_KEY: dataset_folder})


def load_last_run() -> dict | None:
    try:
        data = json.loads(last_run_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def queue_path() -> Path:
    from core import paths
    return paths.USER_DATA / "training_queue.json"


def load_queue() -> list:
    try:
        data = json.loads(queue_path().read_text(encoding="utf-8"))
        return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def save_queue(items: list) -> None:
    _atomic_json(queue_path(), list(items))


def queue_item(family_key: str, values: dict, dataset_folder: str) -> dict:
    """A full settings snapshot (preset values + run-only values + family + dataset), as Fizgig's queue stores."""
    return {**values, ARCH_KEY: family_key, DATASET_KEY: dataset_folder}


def env_path_ok(path: str) -> bool:
    return bool(path) and os.path.exists(path)
