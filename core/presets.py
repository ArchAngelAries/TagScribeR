"""Named presets (built-in + user) with safe persistence, plus change notification.

Used for caption instruction presets and auto-tag presets. Built-in presets
ship in code and can't be overwritten or deleted; user presets live in a JSON
file in user_data and persist until the user deletes them.

File format: {"_schema": 1, "presets": {name: {...}}}. A corrupt file is moved
aside (``.corrupt-N``) rather than overwritten, so user presets are never lost
silently.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

from core.caption_io import atomic_write_text, read_text_file

log = logging.getLogger(__name__)

SCHEMA = 1


class PresetError(ValueError):
    pass


class PresetStore:
    def __init__(self, path: Path, builtins: dict[str, dict] | None = None):
        self.path = Path(path)
        self.builtins = dict(builtins or {})
        self.user: dict[str, dict] = {}
        self._listeners: list[Callable[[], None]] = []
        self.load()

    # -- persistence --------------------------------------------------------
    def load(self) -> None:
        self.user = {}
        if not self.path.is_file():
            return
        try:
            raw = json.loads(read_text_file(self.path))
            presets = raw.get("presets", {}) if isinstance(raw, dict) else {}
            self.user = {str(k): dict(v) for k, v in presets.items() if isinstance(v, dict)}
        except (OSError, ValueError) as e:
            n = 1
            while (dest := self.path.with_name(f"{self.path.name}.corrupt-{n}")).exists():
                n += 1
            try:
                self.path.rename(dest)
            except OSError:
                pass
            log.error("Preset file %s was unreadable (%s); moved aside.", self.path, e)

    def _save(self) -> None:
        atomic_write_text(self.path, json.dumps({"_schema": SCHEMA, "presets": self.user}, indent=2,
                                                ensure_ascii=False))
        for fn in list(self._listeners):
            try:
                fn()
            except Exception:
                log.exception("Preset listener failed")

    def subscribe(self, fn: Callable[[], None]) -> None:
        self._listeners.append(fn)

    # -- queries -----------------------------------------------------------
    def names(self) -> list[str]:
        """Built-ins (in definition order) followed by user presets (alphabetical)."""
        return list(self.builtins) + sorted(self.user, key=str.lower)

    def is_builtin(self, name: str) -> bool:
        return name in self.builtins

    def get(self, name: str) -> dict | None:
        if name in self.user:
            return dict(self.user[name])
        if name in self.builtins:
            return dict(self.builtins[name])
        return None

    # -- edits --------------------------------------------------------------
    @staticmethod
    def _clean(name: str) -> str:
        name = " ".join((name or "").split())
        if not name:
            raise PresetError("Preset name can't be empty.")
        if len(name) > 80:
            raise PresetError("Preset name is too long (80 characters max).")
        return name

    def save(self, name: str, data: dict) -> str:
        name = self._clean(name)
        if name in self.builtins:
            raise PresetError(f"'{name}' is a built-in preset. Save your version under a different name.")
        self.user[name] = dict(data)
        self._save()
        return name

    def delete(self, name: str) -> None:
        if name in self.builtins:
            raise PresetError("Built-in presets can't be deleted.")
        if self.user.pop(name, None) is not None:
            self._save()

    def rename(self, old: str, new: str) -> str:
        if old in self.builtins:
            raise PresetError("Built-in presets can't be renamed.")
        new = self._clean(new)
        if new != old and (new in self.user or new in self.builtins):
            raise PresetError(f"A preset named '{new}' already exists.")
        if old not in self.user:
            raise PresetError(f"No preset named '{old}'.")
        self.user[new] = self.user.pop(old)
        self._save()
        return new

    def unique_name(self, base: str) -> str:
        base = self._clean(base)
        if base not in self.user and base not in self.builtins:
            return base
        n = 2
        while f"{base} ({n})" in self.user or f"{base} ({n})" in self.builtins:
            n += 1
        return f"{base} ({n})"

    # -- sharing ------------------------------------------------------------
    def export(self, path: Path, names: list[str] | None = None) -> int:
        chosen = {n: self.get(n) for n in (names or list(self.user)) if self.get(n) is not None}
        atomic_write_text(Path(path), json.dumps({"_schema": SCHEMA, "presets": chosen}, indent=2,
                                                 ensure_ascii=False))
        return len(chosen)

    def import_file(self, path: Path) -> list[str]:
        """Import presets; name clashes get ' (2)' suffixes so nothing is overwritten."""
        try:
            raw = json.loads(read_text_file(Path(path)))
        except (OSError, ValueError) as e:
            raise PresetError(f"Couldn't read presets from {path}: {e}") from e
        presets = raw.get("presets") if isinstance(raw, dict) else None
        if not isinstance(presets, dict):
            raise PresetError("That file doesn't contain TagScribeR presets.")
        added = []
        for name, data in presets.items():
            if not isinstance(data, dict):
                continue
            if any(data == v for v in list(self.user.values()) + list(self.builtins.values())):
                continue  # identical preset already present (under any name): don't duplicate
            final = self.unique_name(str(name))
            self.user[final] = dict(data)
            added.append(final)
        if added:
            self._save()
        return added


# -- the app's preset libraries ------------------------------------------------
TAG_OUTPUT_PRESETS = {"Booru Tags", "Stable Diffusion Tags"}

TAGGER_BUILTINS: dict[str, dict] = {
    "Balanced": {"general_threshold": 0.35, "character_threshold": 0.85, "max_tags": 50},
    "Precise (fewer, surer tags)": {"general_threshold": 0.5, "character_threshold": 0.9, "max_tags": 30},
    "Broad (more tags)": {"general_threshold": 0.25, "character_threshold": 0.8, "max_tags": 80},
    "No character names": {"general_threshold": 0.35, "character_threshold": 1.0, "max_tags": 50},
}

_caption_store: PresetStore | None = None
_tagger_store: PresetStore | None = None


def caption_presets() -> PresetStore:
    """Caption instruction presets. Data keys: prompt, system_prompt, output ('prose'|'tags'),
    and optionally max_tokens, temperature, top_p, top_k, repetition_penalty, save_mode."""
    global _caption_store
    if _caption_store is None:
        from core import paths
        from inference.prompts import PROMPT_PRESETS
        builtins = {name: {"prompt": text, "system_prompt": "",
                           "output": "tags" if name in TAG_OUTPUT_PRESETS else "prose"}
                    for name, text in PROMPT_PRESETS.items()}
        _caption_store = PresetStore(paths.CAPTION_PRESETS_FILE, builtins)
        _migrate_legacy_custom_prompt(_caption_store)
    return _caption_store


def tagger_presets() -> PresetStore:
    """Auto-tag presets (thresholds, limits, blacklist, forced tags, formatting)."""
    global _tagger_store
    if _tagger_store is None:
        from core import paths
        _tagger_store = PresetStore(paths.TAGGER_PRESETS_FILE, TAGGER_BUILTINS)
    return _tagger_store


def _migrate_legacy_custom_prompt(store: PresetStore) -> None:
    """Earlier versions kept one 'Custom' prompt in settings; keep it as a named preset."""
    from core.config import settings
    cfg = settings()
    custom = (cfg.get("caption.custom_prompt") or "").strip()
    if custom and not store.path.exists() and not cfg.get("caption.custom_prompt_migrated", False):
        store.save("My custom prompt", {"prompt": custom, "system_prompt": cfg.get("caption.system_prompt"),
                                        "output": "prose"})
        cfg.set("caption.custom_prompt_migrated", True)
