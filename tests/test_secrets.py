import json

from core import paths, secrets


def _fake_store(monkeypatch, available=True):
    store = {}
    monkeypatch.setattr(secrets, "get_secret", lambda n: store.get(n) if available else None)
    monkeypatch.setattr(secrets, "set_secret",
                        lambda n, v: (store.__setitem__(n, v) or True) if available else False)
    return store


def test_profiles_keep_keys_out_of_file(monkeypatch):
    _fake_store(monkeypatch)
    s = secrets.ApiProfileStore()
    s.upsert(secrets.ApiProfile(name="LM", base_url="http://x/v1", model="m", api_key="sk-123"))
    on_disk = json.loads(paths.API_PROFILES_FILE.read_text())
    assert "api_key" not in on_disk["LM"]
    assert secrets.ApiProfileStore().profiles["LM"].api_key == "sk-123"


def test_plaintext_fallback_without_keyring(monkeypatch):
    _fake_store(monkeypatch, available=False)
    s = secrets.ApiProfileStore()
    s.upsert(secrets.ApiProfile(name="P", base_url="u", model="m", api_key="k"))
    assert secrets.ApiProfileStore().profiles["P"].api_key == "k"


def test_legacy_presets_imported(monkeypatch):
    _fake_store(monkeypatch)
    paths.LEGACY_API_PRESETS_FILE.parent.mkdir(parents=True)
    paths.LEGACY_API_PRESETS_FILE.write_text(json.dumps(
        {"Old": {"base_url": "http://127.0.0.1:1234/v1", "api_key": "lm", "model_name": "qwen"}}))
    s = secrets.ApiProfileStore()
    assert s.profiles["Old"].model == "qwen" and s.profiles["Old"].api_key == "lm"
    assert "api_key" not in json.loads(paths.API_PROFILES_FILE.read_text())["Old"]
