"""core.privacy.scrub and training.presets.export_for_sharing (audit 14, H5)."""
from __future__ import annotations

import os
import platform

import pytest

from core import privacy
from training import params as P
from training import presets


@pytest.fixture
def folders(monkeypatch):
    monkeypatch.setattr(os.path, "expanduser", lambda p: "C:\\Users\\someone" if p == "~" else p)
    from core import paths
    monkeypatch.setattr(paths, "APP_ROOT", __import__("pathlib").Path("D:/Apps/TagScribeR"))
    monkeypatch.setattr(paths, "USER_DATA", __import__("pathlib").Path("D:/Apps/TagScribeR/user_data"))


def test_windows_home_both_slashes(folders):
    out = privacy.scrub("a C:\\Users\\someone\\data\\x.png and C:/Users/someone/data/y.png")
    assert "someone" not in out
    assert out == "a <home>\\data\\x.png and <home>/data/y.png"


def test_nested_folder_longest_first(folders):
    out = privacy.scrub("D:\\Apps\\TagScribeR\\user_data\\logs\\a.log; D:\\Apps\\TagScribeR\\main.py")
    assert out == "<user_data>\\logs\\a.log; <app>\\main.py"


def test_extra_paths_numbered_and_nested(folders):
    out = privacy.scrub("E:\\Data\\sets\\one\\1.png E:\\Data\\sets\\x.png", ["E:\\Data\\sets", "E:/Data/sets/one"])
    assert out == "<path2>\\1.png <path1>\\x.png"
    assert "Data" not in out


def test_mixed_slashes_doubled_and_case(folders):
    if os.name == "nt":
        assert privacy.scrub("c:\\users\\SOMEONE\\a") == "<home>\\a"
    assert privacy.scrub("C:\\\\Users\\\\someone\\\\a") == "<home>\\\\a"            # JSON-escaped


def test_long_path_prefix(folders):
    assert privacy.scrub("\\\\?\\C:\\Users\\someone\\a.txt") == "<home>\\a.txt"


def test_not_a_prefix_of_a_longer_name(folders):
    assert "<home>" not in privacy.scrub("C:\\Users\\someone2\\a")


def test_other_user_names_are_replaced():
    assert privacy.scrub("X:\\Users\\bob\\Desktop\\a") == "X:\\Users\\<user>\\Desktop\\a"
    assert privacy.scrub("/home/alice/data/a") == "/home/<user>/data/a"
    assert privacy.scrub("/Users/carol/data/a") == "/Users/<user>/data/a"


def test_posix_extra_path():
    out = privacy.scrub("loading /mnt/work/models/a.safetensors", ["/mnt/work/models"])
    assert out == "loading <path1>/a.safetensors"


def test_machine_name(monkeypatch):
    monkeypatch.setattr(platform, "node", lambda: "FAKEHOST-77")
    assert privacy.scrub("running on fakehost-77 now") == "running on <machine> now"
    monkeypatch.setattr(platform, "node", lambda: "")
    assert privacy.scrub("running on x") == "running on x"


def test_plain_text_unchanged(monkeypatch):
    monkeypatch.setattr(platform, "node", lambda: "FAKEHOST-77")
    text = "Epoch 3/10 loss 0.123, step 400. Saved my_lora-000003.safetensors (rank 16)"
    assert privacy.scrub(text) == text


def test_credentials_redacted():
    out = privacy.scrub("Authorization: Bearer abcdef1234567890XYZ api_key=SECRETVALUE123 "
                        "hf_abcdefghijklmnop1234 sk-abcdefghijklmnop1234")
    for leaked in ("abcdef1234567890XYZ", "SECRETVALUE123", "hf_abcdefghijklmnop1234", "sk-abcdefghijklmnop1234"):
        assert leaked not in out
    assert out.count("<redacted>") == 4


def test_never_raises():
    assert privacy.scrub("") == ""
    assert isinstance(privacy.scrub(None), str)
    assert isinstance(privacy.scrub("x", [None, 5, ""]), str)
    privacy.scrub("\x00\\\\?\\", ["[", "(", "\\"])


def test_export_for_sharing_drops_personal_keys():
    values = {k: "x" for k in P.PRESET_KEYS}
    values.update({"LORA_NAME": "mine", "METADATA_AUTHOR": "me", "NOT_A_PRESET_KEY": 1, "LORA_OUTPUT_DIR": "C:\\out"})
    out = presets.export_for_sharing(values)
    for k in (*P.PATH_KEYS, "LORA_NAME", "METADATA_AUTHOR", "METADATA_TITLE", "METADATA_DESCRIPTION",
              "METADATA_TAGS", "METADATA_TRIGGER_PHRASE", "METADATA_THUMBNAIL", "NOT_A_PRESET_KEY"):
        assert k not in out
    assert values["LORA_NAME"] == "mine"                      # the original is untouched
    assert out and all(k in P.PRESET_KEYS for k in out)


def test_export_for_sharing_reimports():
    cur = {p.key: p.default for p in P.PARAMS}
    values = presets.collect(cur)
    exported = presets.export_for_sharing(values)
    assert set(exported) < set(values) or set(exported) == set(values)
    new, rep = presets.apply(exported, cur)
    assert not rep.refused and not rep.blocked
