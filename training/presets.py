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

Fizgig preset JSON files apply here unchanged (same keys and value formats).
"""
from __future__ import annotations

import json
import logging
import os
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


def apply(preset: dict, current: dict, desc=None) -> tuple[dict, P.ApplyReport]:
    """`current` updated with `preset` under Fizgig's validation rules. Returns (new values, report)."""
    out = dict(current)
    rep = P.ApplyReport()
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
