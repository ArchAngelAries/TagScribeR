"""Application settings: defaults, validation, versioned migration, atomic saves.

Settings are a flat dict of dotted keys (``"caption.max_tokens"``) persisted to
``user_data/settings.json``. Defaults live in code; the file only needs to hold
what the user changed, but we write the full merged set for readability.

* Unknown keys in the file are preserved (forward compatibility).
* Values of the wrong type fall back to the default instead of crashing.
* A corrupt file is renamed to ``settings.json.corrupt-<n>`` and defaults are used.
* On first run, the pre-modernization ``config.json`` / ``user_tags.txt`` are
  imported (the old files are left in place).
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from core import paths
from core.caption_io import atomic_write_text, read_text_file

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

DEFAULTS: dict[str, Any] = {
    # Appearance
    "ui.theme": "dark_teal.xml",
    "ui.thumbnail_size": 220,
    "ui.last_folder": "",
    # Captions
    "caption.extension": ".txt",
    "caption.separator": ",",
    "caption.backups": True,
    # AI generation defaults
    "caption.max_tokens": 512,
    "caption.temperature": 0.7,
    "caption.top_p": 0.9,
    "caption.max_image_side": 1536,
    "caption.prompt_template": "Detailed Description",
    "caption.custom_prompt": "",
    "caption.system_prompt": "",
    "caption.strip_thinking": True,
    # Local inference
    "local.model_dirs": [],          # extra folders to scan for HF-format models
    "local.device": "auto",          # auto | cuda:N | cpu
    "local.dtype": "auto",           # auto | bfloat16 | float16 | float32
    "local.quantization": "none",    # none | 8bit | 4bit (requires bitsandbytes)
    "local.attention": "auto",       # auto | sdpa | eager | flash_attention_2
    "local.idle_unload_minutes": 0,  # 0 = keep loaded until the user frees it
    # Tagger
    "tagger.model": "SmilingWolf/wd-eva02-large-tagger-v3",
    "tagger.general_threshold": 0.35,
    "tagger.character_threshold": 0.85,
    "tagger.max_tags": 50,
    "tagger.underscores_to_spaces": True,
    "tagger.escape_parentheses": False,
    "tagger.include_rating": False,
    "tagger.blacklist": "",
    "tagger.device": "auto",         # auto | cpu
    # Datasets / outputs
    "paths.collections_dir": "",     # '' = <app>/Dataset Collections
    "paths.edits_dir": "",           # '' = <app>/Image Edits
}


def _coerce(key: str, value: Any) -> Any:
    """Return value if compatible with the default's type, else the default."""
    default = DEFAULTS.get(key)
    if default is None:
        return value
    if isinstance(default, bool):
        return value if isinstance(value, bool) else default
    if isinstance(default, int):
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return int(value)
        return default
    if isinstance(default, float):
        if isinstance(value, bool):
            return default
        return float(value) if isinstance(value, (int, float)) else default
    if isinstance(default, str):
        return value if isinstance(value, str) else default
    if isinstance(default, list):
        return list(value) if isinstance(value, list) else list(default)
    return value


class Settings:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else paths.SETTINGS_FILE
        self._lock = threading.RLock()
        self._values: dict[str, Any] = {}
        self.load()

    # -- persistence ---------------------------------------------------
    def load(self) -> None:
        with self._lock:
            data: dict[str, Any] = {}
            if self.path.is_file():
                try:
                    raw = json.loads(read_text_file(self.path))
                    if not isinstance(raw, dict):
                        raise ValueError("settings root is not an object")
                    data = raw
                except (OSError, ValueError) as e:
                    self._quarantine(e)
            else:
                data = self._import_legacy()
            data = self._migrate(data)
            values = dict(DEFAULTS)
            for k, v in data.items():
                if k.startswith("_"):
                    continue
                values[k] = _coerce(k, v)
            self._values = values

    def _quarantine(self, err: Exception) -> None:
        n = 1
        while (dest := self.path.with_name(f"{self.path.name}.corrupt-{n}")).exists():
            n += 1
        try:
            self.path.rename(dest)
            log.error("Settings file was unreadable (%s); moved to %s and using defaults.", err, dest)
        except OSError:
            log.error("Settings file was unreadable (%s); using defaults.", err)

    def _migrate(self, data: dict[str, Any]) -> dict[str, Any]:
        version = data.get("_schema", 0)
        if not isinstance(version, int):
            version = 0
        # Future schema changes go here as `if version < N: ...; version = N`.
        data["_schema"] = SCHEMA_VERSION
        return data

    def _import_legacy(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        legacy = paths.LEGACY_CONFIG_FILE
        if legacy.is_file():
            try:
                old = json.loads(read_text_file(legacy))
                mapping = {
                    "theme": "ui.theme",
                    "ai_max_tokens": "caption.max_tokens",
                    "ai_temperature": "caption.temperature",
                    "ai_top_p": "caption.top_p",
                    "default_prompt_template": "caption.prompt_template",
                }
                for old_k, new_k in mapping.items():
                    if old_k in old:
                        data[new_k] = old[old_k]
                log.info("Imported legacy settings from %s", legacy)
            except (OSError, ValueError) as e:
                log.warning("Could not import legacy config %s: %s", legacy, e)
        return data

    def save(self) -> None:
        with self._lock:
            out = {"_schema": SCHEMA_VERSION, **dict(sorted(self._values.items()))}
            try:
                atomic_write_text(self.path, json.dumps(out, indent=2, ensure_ascii=False))
            except OSError as e:
                log.error("Could not save settings to %s: %s", self.path, e)

    # -- access --------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key in self._values:
                return self._values[key]
            return DEFAULTS.get(key, default)

    def set(self, key: str, value: Any, save: bool = True) -> None:
        with self._lock:
            self._values[key] = _coerce(key, value)
            if save:
                self.save()

    def update(self, values: dict[str, Any]) -> None:
        with self._lock:
            for k, v in values.items():
                self._values[k] = _coerce(k, v)
            self.save()


_settings: Settings | None = None


def settings() -> Settings:
    """Process-wide settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# -- quick tags (simple newline-delimited list) -----------------------------
DEFAULT_QUICK_TAGS = ["masterpiece", "best quality", "4k", "photo", "illustration",
                      "scenery", "portrait", "simple background", "solo", "1girl", "1boy"]


def load_quick_tags() -> list[str]:
    for candidate in (paths.QUICK_TAGS_FILE, paths.LEGACY_TAGS_FILE):
        if candidate.is_file():
            try:
                tags = [ln.strip() for ln in read_text_file(candidate).splitlines() if ln.strip()]
                return list(dict.fromkeys(tags))
            except OSError as e:
                log.warning("Could not read quick tags %s: %s", candidate, e)
    return list(DEFAULT_QUICK_TAGS)


def save_quick_tags(tags: list[str]) -> None:
    atomic_write_text(paths.QUICK_TAGS_FILE, "\n".join(dict.fromkeys(t.strip() for t in tags if t.strip())) + "\n")
