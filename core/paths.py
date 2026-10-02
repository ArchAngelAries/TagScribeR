"""Well-known application paths.

Everything is resolved relative to the application root rather than the
current working directory, so launching from another folder (a shortcut, an
IDE, a different shell) still finds the user's settings and models.

TagScribeR is a portable, git-cloned application, so per-user state lives in
``<app root>/user_data`` (gitignored) rather than %APPDATA%.
"""
from __future__ import annotations

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent

# Overridable for tests and for users who want their state elsewhere.
USER_DATA = Path(os.environ.get("TAGSCRIBER_USER_DATA", APP_ROOT / "user_data"))

SETTINGS_FILE = USER_DATA / "settings.json"
API_PROFILES_FILE = USER_DATA / "api_profiles.json"
QUICK_TAGS_FILE = USER_DATA / "quick_tags.txt"
CAPTION_PRESETS_FILE = USER_DATA / "caption_presets.json"
TAGGER_PRESETS_FILE = USER_DATA / "tagger_presets.json"
LOG_DIR = USER_DATA / "logs"
CAPTION_BACKUP_DIR = USER_DATA / "caption_backups"
THUMBNAIL_CACHE_DIR = USER_DATA / "cache" / "thumbnails"

# Default content locations (user-configurable in settings).
DEFAULT_MODELS_DIR = APP_ROOT / "models"
DEFAULT_COLLECTIONS_DIR = APP_ROOT / "Dataset Collections"
DEFAULT_EDITS_DIR = APP_ROOT / "Image Edits"
RESOURCES_DIR = APP_ROOT / "resources"

# Pre-modernization files, read once for migration and otherwise left untouched.
LEGACY_CONFIG_FILE = APP_ROOT / "config.json"
LEGACY_API_PRESETS_FILE = APP_ROOT / "api_presets.json"
LEGACY_TAGS_FILE = APP_ROOT / "user_tags.txt"


def ensure_user_dirs() -> None:
    for d in (USER_DATA, LOG_DIR, CAPTION_BACKUP_DIR, THUMBNAIL_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def resource(name: str) -> Path | None:
    """Find a bundled resource case-insensitively (Linux filesystems are case-sensitive)."""
    direct = RESOURCES_DIR / name
    if direct.exists():
        return direct
    if RESOURCES_DIR.is_dir():
        lower = name.lower()
        for p in RESOURCES_DIR.iterdir():
            if p.name.lower() == lower:
                return p
    return None
