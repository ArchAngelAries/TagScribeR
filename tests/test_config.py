import json

from core import config, paths


def test_defaults_when_no_file():
    s = config.Settings()
    assert s.get("caption.max_tokens") == 512


def test_roundtrip_and_type_validation():
    s = config.Settings()
    s.set("caption.max_tokens", 1024)
    s.set("caption.temperature", "hot")  # wrong type -> default
    s2 = config.Settings()
    assert s2.get("caption.max_tokens") == 1024
    assert s2.get("caption.temperature") == 0.7


def test_unknown_keys_preserved():
    paths.SETTINGS_FILE.parent.mkdir(parents=True)
    paths.SETTINGS_FILE.write_text(json.dumps({"_schema": 1, "future.key": 5}))
    s = config.Settings()
    s.save()
    assert json.loads(paths.SETTINGS_FILE.read_text())["future.key"] == 5


def test_corrupt_file_quarantined():
    paths.SETTINGS_FILE.parent.mkdir(parents=True)
    paths.SETTINGS_FILE.write_text("{not json")
    s = config.Settings()
    assert s.get("ui.theme") == "dark_teal.xml"
    assert (paths.SETTINGS_FILE.parent / "settings.json.corrupt-1").exists()


def test_legacy_import():
    paths.LEGACY_CONFIG_FILE.parent.mkdir(parents=True)
    paths.LEGACY_CONFIG_FILE.write_text(json.dumps({"theme": "light_blue.xml", "ai_max_tokens": 300}))
    s = config.Settings()
    assert s.get("ui.theme") == "light_blue.xml"
    assert s.get("caption.max_tokens") == 300
    assert paths.LEGACY_CONFIG_FILE.exists()  # old file untouched


def test_quick_tags_fallbacks():
    assert "masterpiece" in config.load_quick_tags()
    paths.LEGACY_TAGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    paths.LEGACY_TAGS_FILE.write_text("one\ntwo\none\n")
    assert config.load_quick_tags() == ["one", "two"]
    config.save_quick_tags(["x", "y"])
    assert config.load_quick_tags() == ["x", "y"]
