"""API profiles and secret storage.

API keys are stored in the OS credential store (Windows Credential Manager,
macOS Keychain, Secret Service on Linux) via ``keyring`` when available. The
profile file (``user_data/api_profiles.json``) only holds non-secret fields.
If no credential store is usable, keys fall back to the profile file (mode 0600
on POSIX) and a warning is logged — the app keeps working, just less securely.

A saved key belongs to ONE server host. The profile file records it as
``key_host`` next to the profile; a key is never loaded for a profile whose URL
now names another host. Migration: profiles saved before this field existed have
no ``key_host``; their key is accepted for the URL they hold today and is bound
to that host the next time the profile is saved.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass

from core import paths
from core.caption_io import atomic_write_text, read_text_file

log = logging.getLogger(__name__)

KEYRING_SERVICE = "TagScribeR"

KEY_PROBLEM_TEXT = ("The API key contains a line break or another character that cannot be sent. "
                    "Copy the key again without extra lines.")

# Keys this session has seen (typed, loaded, saved): the log filter blanks them out of any log line.
_known_keys: set[str] = set()


def clean_key(key) -> str:
    """The key as it is stored and sent: no leading or trailing whitespace."""
    return key.strip() if isinstance(key, str) else ""


def key_problem(key: str) -> str:
    """Plain message when a key cannot be a header value (line breaks, control characters); else ''."""
    key = clean_key(key)
    if key and not key.isprintable():
        return KEY_PROBLEM_TEXT
    return ""


def remember_secret(key) -> None:
    """Tell the log filter about a key so it is redacted from every log line."""
    raw = key if isinstance(key, str) else ""
    for k in {raw, clean_key(raw)}:
        if len(k) >= 4:
            _known_keys.add(k)


def known_secrets() -> list[str]:
    return sorted(_known_keys, key=len, reverse=True)


def host_of(url: str) -> str:
    from urllib.parse import urlparse
    try:
        return (urlparse((url or "").strip()).hostname or "").lower()
    except ValueError:
        return ""


@dataclass
class ApiProfile:
    name: str
    base_url: str
    model: str
    api_key: str = ""
    timeout: float = 180.0
    max_retries: int | None = None   # None = automatic: none for local servers, a couple for cloud APIs

    def __post_init__(self):
        self.api_key = clean_key(self.api_key)
        remember_secret(self.api_key)


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
    """Store a secret; an empty value deletes the entry. False when there is no usable credential store."""
    kr = _keyring()
    if kr is None:
        return False
    try:
        if value:
            kr.set_password(KEYRING_SERVICE, name, value)
        else:
            try:
                kr.delete_password(KEYRING_SERVICE, name)
            except Exception as e:
                if type(e).__name__ != "PasswordDeleteError":      # "no such entry" is fine
                    log.warning("Credential store delete failed for %s: %s", name, e)
        return True
    except Exception as e:
        log.warning("Credential store write failed for %s: %s", name, e)
        return False


def _restrict_file(path) -> None:
    """Owner-only permissions for a file that holds a key (no-op on Windows)."""
    if sys.platform == "win32":
        return
    try:
        os.chmod(path, 0o600)
    except OSError as e:
        log.warning("Could not restrict permissions of %s: %s", path.name, e)


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
            base_url = d.get("base_url", "")
            key = clean_key(get_secret(f"api:{name}") or d.get("api_key", ""))
            key_host = d.get("key_host")            # absent in profiles saved before the binding existed
            if key and key_host is not None and key_host != host_of(base_url):
                log.warning("API key of profile '%s' was saved for another server; not loading it.", name)
                key = ""
            if key and key_problem(key):
                log.warning("API key of profile '%s' contains unusable characters; ignoring it.", name)
                key = ""
            timeout, retries = d.get("timeout"), d.get("max_retries")
            if (timeout, retries) in ((120.0, 3), (120, 3)):
                # the old defaults every earlier profile file holds (no UI ever set them): automatic now
                timeout, retries = None, None
            self.profiles[name] = ApiProfile(
                name=name, base_url=base_url, model=d.get("model", d.get("model_name", "")),
                api_key=key, timeout=float(timeout if timeout is not None else 180.0),
                max_retries=int(retries) if retries is not None else None,
            )

    def _import_legacy(self) -> dict:
        legacy = paths.LEGACY_API_PRESETS_FILE
        try:
            raw = json.loads(read_text_file(legacy))
        except (OSError, ValueError) as e:
            log.warning("Could not import legacy API presets: %s", e)
            return {}
        if not isinstance(raw, dict) or raw.get("_imported"):
            return {}                                # already imported once: never repeated
        log.info("Importing legacy API presets from %s", legacy.name)
        self.profiles = {
            n: ApiProfile(name=n, base_url=d.get("base_url", ""), model=d.get("model_name", ""),
                          api_key=d.get("api_key", "") if isinstance(d.get("api_key", ""), str) else "")
            for n, d in raw.items() if isinstance(d, dict)
        }
        try:
            self.save()
        except ValueError as e:
            log.warning("Legacy API presets not imported: %s", e)
            return {}
        if not self.path.is_file():
            return {}
        self._scrub_legacy(raw)
        return json.loads(read_text_file(self.path))

    @staticmethod
    def _scrub_legacy(raw: dict) -> None:
        """The keys now live in the credential store (or the profile file): remove them from the old file."""
        out = {}
        for n, d in raw.items():
            if isinstance(d, dict):
                d = {k: v for k, v in d.items() if k != "api_key"}
            out[n] = d
        out["_imported"] = True
        out["_note"] = "The API keys were moved to TagScribeR's own profile storage and removed from this file."
        try:
            atomic_write_text(paths.LEGACY_API_PRESETS_FILE, json.dumps(out, indent=2))
            log.info("Legacy API presets imported; their keys were removed from the old file.")
        except OSError as e:
            log.warning("Could not remove the keys from the old presets file: %s", e)

    def save(self) -> None:
        out = {}
        holds_key = False
        for name, p in self.profiles.items():
            d = asdict(p)
            d.pop("name")
            key = clean_key(d.pop("api_key"))
            problem = key_problem(key)
            if problem:
                raise ValueError(problem)
            # an empty key must reach the store too: it deletes the entry
            stored = set_secret(f"api:{name}", key)
            if key:
                d["key_host"] = host_of(p.base_url)
                if not stored:
                    log.warning("No OS credential store available; API key for '%s' stored in plain text.", name)
                    d["api_key"] = key
                    holds_key = True
            out[name] = d
        atomic_write_text(self.path, json.dumps(out, indent=2))
        if holds_key:
            _restrict_file(self.path)

    def upsert(self, profile: ApiProfile) -> None:
        problem = key_problem(profile.api_key)
        if problem:
            raise ValueError(problem)
        self.profiles[profile.name] = profile
        self.save()

    def delete(self, name: str) -> None:
        if name in self.profiles:
            del self.profiles[name]
            set_secret(f"api:{name}", "")
            self.save()
