"""API profiles and secret storage.

API keys are stored in the OS credential store (Windows Credential Manager,
macOS Keychain, Secret Service on Linux) via ``keyring`` when available. The
profile file (``user_data/api_profiles.json``) only holds non-secret fields.
If no credential store is usable, keys fall back to the profile file and a
warning is logged — the app keeps working, just less securely.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

from core import paths
from core.caption_io import atomic_write_text, read_text_file

log = logging.getLogger(__name__)

KEYRING_SERVICE = "TagScribeR"


@dataclass
class ApiProfile:
    name: str
    base_url: str
    model: str
    api_key: str = ""
    timeout: float = 120.0
    max_retries: int = 3


def _keyring():
    try:
        import keyring
        from keyring.backends import fail
        if isinstance(keyring.get_keyring(), fail.Keyring):
            return None
        return keyring
    except Exception:
        return None


def get_secret(name: str) -> str | None:
    kr = _keyring()
    if kr is None:
        return None
    try:
        return kr.get_password(KEYRING_SERVICE, name)
    except Exception as e:
        log.warning("Credential store read failed for %s: %s", name, e)
        return None


def set_secret(name: str, value: str) -> bool:
    kr = _keyring()
    if kr is None:
        return False
    try:
        if value:
            kr.set_password(KEYRING_SERVICE, name, value)
        else:
            try:
                kr.delete_password(KEYRING_SERVICE, name)
            except Exception:
                pass
        return True
    except Exception as e:
        log.warning("Credential store write failed for %s: %s", name, e)
        return False


class ApiProfileStore:
    def __init__(self, path=None):
        self.path = path or paths.API_PROFILES_FILE
        self.profiles: dict[str, ApiProfile] = {}
        self.load()

    def load(self) -> None:
        self.profiles = {}
        raw: dict = {}
        if self.path.is_file():
            try:
                raw = json.loads(read_text_file(self.path))
            except (OSError, ValueError) as e:
                log.error("Could not read API profiles %s: %s", self.path, e)
        elif paths.LEGACY_API_PRESETS_FILE.is_file():
            raw = self._import_legacy()
        for name, d in raw.items():
            if not isinstance(d, dict):
                continue
            key = get_secret(f"api:{name}") or d.get("api_key", "")
            self.profiles[name] = ApiProfile(
                name=name, base_url=d.get("base_url", ""), model=d.get("model", d.get("model_name", "")),
                api_key=key, timeout=float(d.get("timeout", 120.0)), max_retries=int(d.get("max_retries", 3)),
            )

    def _import_legacy(self) -> dict:
        try:
            raw = json.loads(read_text_file(paths.LEGACY_API_PRESETS_FILE))
        except (OSError, ValueError) as e:
            log.warning("Could not import legacy API presets: %s", e)
            return {}
        log.info("Importing legacy API presets from %s", paths.LEGACY_API_PRESETS_FILE)
        self.profiles = {
            n: ApiProfile(name=n, base_url=d.get("base_url", ""), model=d.get("model_name", ""),
                          api_key=d.get("api_key", ""))
            for n, d in raw.items() if isinstance(d, dict)
        }
        self.save()
        return json.loads(read_text_file(self.path)) if self.path.is_file() else {}

    def save(self) -> None:
        out = {}
        for name, p in self.profiles.items():
            d = asdict(p)
            d.pop("name")
            key = d.pop("api_key")
            if key and not set_secret(f"api:{name}", key):
                log.warning("No OS credential store available; API key for '%s' stored in plain text.", name)
                d["api_key"] = key
            out[name] = d
        atomic_write_text(self.path, json.dumps(out, indent=2))

    def upsert(self, profile: ApiProfile) -> None:
        self.profiles[profile.name] = profile
        self.save()

    def delete(self, name: str) -> None:
        if name in self.profiles:
            del self.profiles[name]
            set_secret(f"api:{name}", "")
            self.save()
