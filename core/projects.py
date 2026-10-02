"""Per-dataset project settings (stored in user_data, never inside the dataset folder).

Each folder gets its own small JSON file keyed by a hash of its path, holding
things that belong to *that* dataset rather than to the app: the trigger /
subject word, preferred caption preset, tag-hint choice, health-check training
size, named filters and free-form notes. Global preferences stay in settings.json.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from core import paths
from core.caption_io import atomic_write_text, read_text_file

log = logging.getLogger(__name__)

PROJECT_DEFAULTS: dict[str, Any] = {
    "subject": "",
    "subject_first": False,
    "tag_hints": False,
    "caption_preset": "",
    "health_resolution": 0,      # 0 = use the global default
    "filters": {},               # name -> query
    "notes": "",
}

BUILTIN_FILTERS = {
    "Missing captions": "missing:caption",
    "Unsaved edits": "is:unsaved",
    "Flagged by health scan": "flag:any",
    "Tag-style captions": "is:tagged",
    "Sentence captions": "is:prose",
    "Small images (< 768 px)": "res:<768",
}


def projects_dir() -> Path:
    return paths.USER_DATA / "projects"


def _key(folder: str | os.PathLike) -> str:
    norm = os.path.normcase(os.path.abspath(os.fspath(folder)))
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


class Project:
    def __init__(self, folder: str | os.PathLike):
        self.folder = os.path.abspath(os.fspath(folder))
        self.path = projects_dir() / f"{_key(folder)}.json"
        self.data: dict[str, Any] = dict(PROJECT_DEFAULTS)
        self.data["filters"] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(read_text_file(self.path))
            if isinstance(raw, dict):
                for k, default in PROJECT_DEFAULTS.items():
                    v = raw.get(k, default)
                    if isinstance(default, bool):
                        self.data[k] = v if isinstance(v, bool) else default
                    elif isinstance(default, int):
                        self.data[k] = int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else default
                    elif isinstance(default, dict):
                        self.data[k] = {str(a): str(b) for a, b in v.items()} if isinstance(v, dict) else {}
                    else:
                        self.data[k] = v if isinstance(v, str) else default
        except (OSError, ValueError) as e:
            log.warning("Project settings for %s unreadable (%s); using defaults.", self.folder, e)

    def save(self) -> None:
        out = {"_folder": self.folder, **self.data}
        try:
            atomic_write_text(self.path, json.dumps(out, indent=2, ensure_ascii=False))
        except OSError as e:
            log.error("Could not save project settings for %s: %s", self.folder, e)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        if self.data.get(key) != value:
            self.data[key] = value
            self.save()

    # -- named filters ------------------------------------------------------
    def filters(self) -> dict[str, str]:
        return dict(self.data.get("filters", {}))

    def save_filter(self, name: str, query: str) -> None:
        name = " ".join(name.split())
        if not name or not query.strip():
            raise ValueError("A filter needs a name and a query.")
        f = self.filters()
        f[name] = query.strip()
        self.set("filters", f)

    def delete_filter(self, name: str) -> None:
        f = self.filters()
        if f.pop(name, None) is not None:
            self.set("filters", f)
