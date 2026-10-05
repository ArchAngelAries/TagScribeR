import os
import sys
from pathlib import Path

import pytest

# The suite must never open a GPU context: a card that is full during a training run has no room for one, and a test
# run beside it can push that training into shared memory. Hidden before anything imports torch.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"       # not "": Windows treats an empty value as unset
os.environ["HIP_VISIBLE_DEVICES"] = "-1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolated_user_data(tmp_path, monkeypatch):
    """Point every user_data path at a temp dir so tests never touch real settings/backups."""
    from core import paths
    ud = tmp_path / "user_data"
    monkeypatch.setattr(paths, "USER_DATA", ud)
    monkeypatch.setattr(paths, "SETTINGS_FILE", ud / "settings.json")
    monkeypatch.setattr(paths, "API_PROFILES_FILE", ud / "api_profiles.json")
    monkeypatch.setattr(paths, "QUICK_TAGS_FILE", ud / "quick_tags.txt")
    monkeypatch.setattr(paths, "LOG_DIR", ud / "logs")
    monkeypatch.setattr(paths, "CAPTION_BACKUP_DIR", ud / "caption_backups")
    monkeypatch.setattr(paths, "THUMBNAIL_CACHE_DIR", ud / "cache" / "thumbnails")
    legacy = tmp_path / "legacy"
    monkeypatch.setattr(paths, "LEGACY_CONFIG_FILE", legacy / "config.json")
    monkeypatch.setattr(paths, "LEGACY_API_PRESETS_FILE", legacy / "api_presets.json")
    monkeypatch.setattr(paths, "LEGACY_TAGS_FILE", legacy / "user_tags.txt")
    return ud


def make_image(path: Path, size=(32, 24), color=(200, 30, 30), mode="RGB", **save_kwargs):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    fill = color if mode != "RGBA" else color + (128,)
    Image.new(mode, size, fill).save(path, **save_kwargs)
    return path
