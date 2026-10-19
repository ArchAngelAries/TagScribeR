"""The Windows ROCm bitsandbytes wheel is verified against a pinned SHA-256 before it is installed."""
import hashlib
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import install  # noqa: E402

DATA = b"fake wheel bytes"
URL = "https://example.invalid/dl/bitsandbytes-0-cp312-cp312-win_amd64.whl"


def _fake_urlopen(data):
    def opener(url, timeout=0):
        return io.BytesIO(data)
    return opener


def test_pin_is_a_sha256():
    assert len(install.BNB_ROCM_WIN_SHA256) == 64
    int(install.BNB_ROCM_WIN_SHA256, 16)


def test_matching_digest_returns_file(tmp_path, monkeypatch):
    monkeypatch.setattr(install.urllib.request, "urlopen", _fake_urlopen(DATA))
    p = install.fetch_verified_wheel(URL, hashlib.sha256(DATA).hexdigest(), tmp_path)
    assert p.read_bytes() == DATA and p.name.endswith(".whl")


def test_mismatch_stops_and_names_digests(tmp_path, monkeypatch):
    monkeypatch.setattr(install.urllib.request, "urlopen", _fake_urlopen(DATA))
    with pytest.raises(SystemExit) as e:
        install.fetch_verified_wheel(URL, "0" * 64, tmp_path)
    msg = str(e.value)
    assert "0" * 64 in msg and hashlib.sha256(DATA).hexdigest() in msg and "--no-bnb" in msg


def _win_rocm(monkeypatch):
    monkeypatch.setattr(install, "IS_WINDOWS", True)
    monkeypatch.setattr(install.sys, "version_info", (3, 12, 0, "final", 0))


def test_mismatch_never_calls_uv(monkeypatch):
    _win_rocm(monkeypatch)
    monkeypatch.setattr(install.urllib.request, "urlopen", _fake_urlopen(DATA))
    calls = []
    monkeypatch.setattr(install.subprocess, "call", lambda *a, **k: calls.append(a) or 0)
    with pytest.raises(SystemExit):
        install.install_bitsandbytes("rocm", False)
    assert calls == []


def test_match_installs_local_file(monkeypatch):
    _win_rocm(monkeypatch)
    monkeypatch.setattr(install.urllib.request, "urlopen", _fake_urlopen(DATA))
    monkeypatch.setattr(install, "BNB_ROCM_WIN_SHA256", hashlib.sha256(DATA).hexdigest())
    calls = []
    monkeypatch.setattr(install.subprocess, "call", lambda cmd, **k: calls.append(cmd) or 0)
    install.install_bitsandbytes("rocm", False)
    assert len(calls) == 1 and calls[0][-1].endswith(".whl") and not calls[0][-1].startswith("http")


def test_dry_run_downloads_nothing(monkeypatch, capsys):
    _win_rocm(monkeypatch)
    def boom(*a, **k):
        raise AssertionError("network used in dry run")
    monkeypatch.setattr(install.urllib.request, "urlopen", boom)
    install.install_bitsandbytes("rocm", True)
    assert "SHA-256" in capsys.readouterr().out


def test_custom_wheel_needs_its_own_digest(monkeypatch):
    _win_rocm(monkeypatch)
    monkeypatch.setattr(install, "BNB_ROCM_WIN_WHEEL", URL)
    monkeypatch.delenv("BNB_WHEEL_SHA256", raising=False)
    with pytest.raises(SystemExit) as e:
        install.install_bitsandbytes("rocm", True)
    assert "BNB_WHEEL_SHA256" in str(e.value)
