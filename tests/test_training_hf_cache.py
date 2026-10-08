"""training/hf_cache.py (Fizgig utils/hf_cache.py): tokenizers and processors load from the local cache before the
Hub is asked anything, so text caching works offline once the files are cached (Fizgig #26, #174). No network."""
import pytest

from training import hf_cache


class _Loader:
    """Stands in for AutoTokenizer / a processor class: records each from_pretrained call; `ok` decides which load."""
    __name__ = "FakeTokenizer"

    def __init__(self, ok):
        self.calls, self.ok = [], ok

    def from_pretrained(self, src, **kw):
        self.calls.append((src, kw))
        if not self.ok(src, kw):
            raise OSError("not available")
        return ("loaded", src)


def test_cached_snapshot_folder_first(monkeypatch, tmp_path):
    monkeypatch.setattr(hf_cache, "cached_snapshot_dir", lambda repo: str(tmp_path))
    cls = _Loader(lambda src, kw: True)
    assert hf_cache.from_pretrained_cache_first(cls, "org/repo", subfolder="processor") == ("loaded", str(tmp_path))
    assert cls.calls == [(str(tmp_path), {"local_files_only": True, "subfolder": "processor"})]   # never the Hub


def test_falls_back_to_repo_id_offline_then_the_hub(monkeypatch):
    monkeypatch.setattr(hf_cache, "cached_snapshot_dir", lambda repo: None)
    cls = _Loader(lambda src, kw: not kw.get("local_files_only"))      # nothing cached: only the Hub has it
    assert hf_cache.from_pretrained_cache_first(cls, "org/repo") == ("loaded", "org/repo")
    assert cls.calls == [("org/repo", {"local_files_only": True}), ("org/repo", {})]


def test_local_folder_is_not_a_snapshot(tmp_path):
    assert hf_cache.cached_snapshot_dir(str(tmp_path)) is None


def test_missing_everywhere_raises(monkeypatch):
    monkeypatch.setattr(hf_cache, "cached_snapshot_dir", lambda repo: None)
    with pytest.raises(OSError):
        hf_cache.from_pretrained_cache_first(_Loader(lambda src, kw: False), "org/repo")
