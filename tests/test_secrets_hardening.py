"""Key handling: clearing really deletes, keys are bound to their host, log redaction, legacy import scrub."""
import json
import logging
import os
import sys

import pytest

from core import log as applog
from core import paths, secrets


class FakeKeyring:
    def __init__(self):
        self.data = {}

    def get_password(self, service, name):
        return self.data.get((service, name))

    def set_password(self, service, name, value):
        self.data[(service, name)] = value

    def delete_password(self, service, name):
        if (service, name) not in self.data:
            raise KeyError("missing")
        del self.data[(service, name)]


@pytest.fixture
def kr(monkeypatch):
    fake = FakeKeyring()
    monkeypatch.setattr(secrets, "_keyring", lambda: fake)
    return fake


def _profile(name="Cloud", url="https://api.example.test/v1", key="test-key-123"):
    return secrets.ApiProfile(name=name, base_url=url, model="m", api_key=key)


def test_empty_key_deletes_from_store_and_file(kr):
    s = secrets.ApiProfileStore()
    s.upsert(_profile())
    assert kr.data[("TagScribeR", "api:Cloud")] == "test-key-123"
    s.upsert(_profile(key=""))
    assert ("TagScribeR", "api:Cloud") not in kr.data
    assert secrets.ApiProfileStore().profiles["Cloud"].api_key == ""
    assert "test-key-123" not in paths.API_PROFILES_FILE.read_text()


def test_empty_key_deletes_plaintext_fallback(monkeypatch):
    monkeypatch.setattr(secrets, "_keyring", lambda: None)
    s = secrets.ApiProfileStore()
    s.upsert(_profile())
    assert "test-key-123" in paths.API_PROFILES_FILE.read_text()
    s.upsert(_profile(key=""))
    assert "test-key-123" not in paths.API_PROFILES_FILE.read_text()
    assert secrets.ApiProfileStore().profiles["Cloud"].api_key == ""


def test_changed_url_then_save_leaves_no_key(kr):
    s = secrets.ApiProfileStore()
    s.upsert(_profile())
    s.upsert(_profile(url="http://other.example.test/v1", key=""))
    again = secrets.ApiProfileStore()
    assert again.profiles["Cloud"].api_key == ""
    assert not kr.data


def test_key_not_loaded_for_another_host(kr):
    s = secrets.ApiProfileStore()
    s.upsert(_profile())
    raw = json.loads(paths.API_PROFILES_FILE.read_text())
    assert raw["Cloud"]["key_host"] == "api.example.test"
    raw["Cloud"]["base_url"] = "http://attacker.example.test/v1"       # edited file / changed URL, key still stored
    paths.API_PROFILES_FILE.write_text(json.dumps(raw))
    assert secrets.ApiProfileStore().profiles["Cloud"].api_key == ""
    raw["Cloud"]["base_url"] = "https://API.example.test:8443/other"   # same host, other path / port / case
    paths.API_PROFILES_FILE.write_text(json.dumps(raw))
    assert secrets.ApiProfileStore().profiles["Cloud"].api_key == "test-key-123"


def test_old_profile_without_key_host_keeps_working(kr):
    kr.data[("TagScribeR", "api:Old")] = "test-key-123"
    paths.API_PROFILES_FILE.parent.mkdir(parents=True, exist_ok=True)
    paths.API_PROFILES_FILE.write_text(json.dumps({"Old": {"base_url": "http://127.0.0.1:1234/v1", "model": "m"}}))
    s = secrets.ApiProfileStore()
    assert s.profiles["Old"].api_key == "test-key-123"
    s.save()                                                           # rebinds to the host it has today
    assert json.loads(paths.API_PROFILES_FILE.read_text())["Old"]["key_host"] == "127.0.0.1"


def test_key_is_stripped_and_bad_keys_refused(kr):
    assert secrets.ApiProfile("a", "u", "m", api_key="  test-key-123 \r\n").api_key == "test-key-123"
    s = secrets.ApiProfileStore()
    with pytest.raises(ValueError):
        s.upsert(_profile(key="test\nkey"))
    assert "Cloud" not in s.profiles
    assert secrets.key_problem("test-key\x00") and not secrets.key_problem(" test-key-123\n")
    # a bad key already on disk is not loaded
    kr.data[("TagScribeR", "api:Bad")] = "bad\tkey"
    paths.API_PROFILES_FILE.parent.mkdir(parents=True, exist_ok=True)
    paths.API_PROFILES_FILE.write_text(json.dumps({"Bad": {"base_url": "http://x/v1", "model": "m"}}))
    assert secrets.ApiProfileStore().profiles["Bad"].api_key == ""
    # a stored key with a trailing space is cleaned on load
    kr.data[("TagScribeR", "api:Bad")] = "test-key-123 "
    assert secrets.ApiProfileStore().profiles["Bad"].api_key == "test-key-123"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_plaintext_fallback_file_is_private(monkeypatch):
    monkeypatch.setattr(secrets, "_keyring", lambda: None)
    secrets.ApiProfileStore().upsert(_profile())
    assert os.stat(paths.API_PROFILES_FILE).st_mode & 0o777 == 0o600


def test_legacy_keys_removed_after_import_and_not_repeated(kr):
    legacy = paths.LEGACY_API_PRESETS_FILE
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({"Old": {"base_url": "http://127.0.0.1:1234/v1", "api_key": "test-key-123",
                                          "model_name": "qwen", "extra": 7}}))
    s = secrets.ApiProfileStore()
    assert s.profiles["Old"].api_key == "test-key-123"
    text = legacy.read_text()
    assert "test-key-123" not in text
    data = json.loads(text)
    assert data["Old"]["model_name"] == "qwen" and data["Old"]["extra"] == 7 and "api_key" not in data["Old"]
    # profile file lost: the scrubbed legacy file is not imported again
    paths.API_PROFILES_FILE.unlink()
    kr.data.clear()
    assert secrets.ApiProfileStore().profiles == {}


# ---------------------------------------------------------------- log redaction
def _record(msg, args=None, exc_info=None):
    return logging.LogRecord("tagscriber", logging.DEBUG, __file__, 1, msg, args, exc_info)


@pytest.mark.parametrize("text", [
    "Illegal header value b'Bearer test-key-123 '",
    "headers {'Authorization': 'Bearer test-key-123'}",
    "Authorization: Basic dGVzdC1rZXktMTIz",
    "GET http://h/v1?api_key=test-key-123&x=1",
    "x-api-key: test-key-123",
    # made-up tokens, put together here so the release check's own token patterns do not flag this file
    "token " + "sk-" + "abcdefghijklmnop1234" + " used",
    "hf_" + "abcdefghijklmnopqrstuv",
])
def test_redact_patterns(text):
    out = applog.redact(text)
    assert "test-key-123" not in out and "sk-" + "abcdefghijklmnop" not in out
    assert "dGVzdC1rZXktMTIz" not in out and "hf_" + "abcdefghijklmnopqrst" not in out
    assert "<redacted>" in out


def test_filter_covers_args_exception_and_known_keys():
    secrets.remember_secret("plainsecretvalue")
    f = applog.RedactionFilter()
    r = _record("call failed for %s with %s", ("Bearer test-key-123", "plainsecretvalue"))
    f.filter(r)
    assert "test-key-123" not in r.getMessage() and "plainsecretvalue" not in r.getMessage()
    try:
        raise RuntimeError("Illegal header value b'Bearer test-key-123 ' and plainsecretvalue")
    except RuntimeError:
        r2 = _record("failed", exc_info=sys.exc_info())
    f.filter(r2)
    text = logging.Formatter("%(message)s").format(r2)
    assert "test-key-123" not in text and "plainsecretvalue" not in text and "RuntimeError" in text


def test_profile_keys_are_known_to_redaction(kr):
    secrets.ApiProfileStore().upsert(_profile(key="zz-unusual-key-77"))
    assert "zz-unusual-key-77" not in applog.redact("the key was zz-unusual-key-77 here")


def test_handler_filters_drop_library_debug_and_redact(tmp_path):
    h = logging.FileHandler(tmp_path / "t.log", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
    applog.install_filters(h)
    lg_app, lg_lib = logging.getLogger("inference.x"), logging.getLogger("httpcore2.http11")
    root = logging.getLogger()
    old = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(h)
    try:
        lg_lib.debug("receive_response_headers openai-organization: org-1")
        lg_lib.info("HTTP Request: GET http://h/v1")
        logging.getLogger("httpx2").debug("response headers")
        lg_app.debug("app debug kept Bearer test-key-123")
        lg_lib.warning("library warning kept")
    finally:
        root.removeHandler(h)
        root.setLevel(old)
        h.close()
    out = (tmp_path / "t.log").read_text(encoding="utf-8")
    assert "openai-organization" not in out and "HTTP Request" not in out and "response headers" not in out
    assert "app debug kept" in out and "library warning kept" in out and "test-key-123" not in out
