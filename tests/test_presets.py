import json

import pytest

from core import config, paths
from core.presets import PresetError, PresetStore


@pytest.fixture
def store(tmp_path):
    return PresetStore(tmp_path / "p.json", {"Built In": {"prompt": "b"}})


def test_user_presets_persist_across_instances(store, tmp_path):
    store.save("Mine", {"prompt": "x", "temperature": 0.3})
    again = PresetStore(tmp_path / "p.json", {"Built In": {"prompt": "b"}})
    assert again.get("Mine") == {"prompt": "x", "temperature": 0.3}
    assert again.names() == ["Built In", "Mine"]


def test_builtins_protected(store):
    with pytest.raises(PresetError):
        store.save("Built In", {"prompt": "hijack"})
    with pytest.raises(PresetError):
        store.delete("Built In")
    with pytest.raises(PresetError):
        store.rename("Built In", "x")
    assert store.get("Built In") == {"prompt": "b"}


def test_delete_and_rename(store):
    store.save("A", {"prompt": "a"})
    store.rename("A", "B")
    assert store.get("A") is None and store.get("B") == {"prompt": "a"}
    store.save("C", {"prompt": "c"})
    with pytest.raises(PresetError):
        store.rename("B", "C")  # no silent overwrite
    store.delete("B")
    assert "B" not in store.names()


def test_names_validated(store):
    with pytest.raises(PresetError):
        store.save("   ", {})
    assert store.save("  spaced   name ", {}) == "spaced name"


def test_corrupt_file_moved_aside_not_overwritten(tmp_path):
    f = tmp_path / "p.json"
    f.write_text("{broken")
    s = PresetStore(f)
    assert s.user == {}
    assert (tmp_path / "p.json.corrupt-1").read_text() == "{broken"


def test_export_import_renames_clashes(store, tmp_path):
    store.save("Mine", {"prompt": "x"})
    out = tmp_path / "share.json"
    assert store.export(out) == 1
    other = PresetStore(tmp_path / "other.json")
    other.save("Mine", {"prompt": "different"})
    added = other.import_file(out)
    assert added == ["Mine (2)"] and other.get("Mine") == {"prompt": "different"}
    assert other.import_file(out) == []  # re-importing the same file doesn't duplicate


def test_import_rejects_garbage(store, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"something": 1}))
    with pytest.raises(PresetError):
        store.import_file(bad)


def test_listeners_notified(store):
    calls = []
    store.subscribe(lambda: calls.append(1))
    store.save("X", {})
    store.delete("X")
    assert len(calls) == 2


def test_quick_tags_add_remove_persist_and_sync():
    calls = []
    config.on_quick_tags_changed(lambda: calls.append(1))
    config.save_quick_tags(["a", "b"])
    assert config.add_quick_tag("c") and not config.add_quick_tag("c")
    assert config.remove_quick_tags(["a"]) == 1
    assert config.load_quick_tags() == ["b", "c"]
    # A stale copy elsewhere can't resurrect a deleted tag: add starts from disk.
    stale = ["a", "b", "c"]
    config.add_quick_tag("d")
    assert "a" not in config.load_quick_tags() and stale  # stale list unused by the API
    config.save_quick_tags([])
    assert config.load_quick_tags() == []  # deleting everything stays empty (no defaults resurrected)
    assert len(calls) >= 4
    config._quick_tag_listeners.clear()


def test_legacy_custom_prompt_migrated(monkeypatch):
    import core.presets as presets
    monkeypatch.setattr(presets, "_caption_store", None)
    monkeypatch.setattr(paths, "CAPTION_PRESETS_FILE", paths.USER_DATA / "caption_presets.json")
    monkeypatch.setattr(config, "_settings", None)
    config.settings().set("caption.custom_prompt", "Describe the cat.")
    store = presets.caption_presets()
    assert store.get("My custom prompt")["prompt"] == "Describe the cat."
    assert store.get("Booru Tags")["output"] == "tags"
    monkeypatch.setattr(presets, "_caption_store", None)
    monkeypatch.setattr(config, "_settings", None)
