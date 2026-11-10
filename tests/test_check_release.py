"""tools/check_release.py on a throw-away git repository (audit 14, L1). This repository is never touched."""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("check_release", Path(__file__).resolve().parent.parent / "tools"
                                               / "check_release.py")
cr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cr)


def _git(root, *a):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *a], cwd=root, check=True,
                   capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("hello\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    _git(tmp_path, "tag", "basetag")
    monkeypatch.setattr(cr, "ROOT", tmp_path)
    return tmp_path


def test_missing_base_is_an_error(repo, capsys):
    assert cr.main(["--base", "origin/nonexistent"]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_clean_repo_passes(repo):
    assert cr.main(["--base", "basetag"]) == 0


def test_commit_body_and_subject_are_scanned(repo, capsys):
    (repo / "b.txt").write_text("fine\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "subject at C:\\Users\\jdoe\\x", "-m",
         "body with Bearer abcdefghijklmnopqrstuvwx")
    assert cr.main(["--base", "basetag"]) == 1
    out = capsys.readouterr().out
    assert "commit message" in out and "Windows user folder" in out and "bearer token" in out


def test_binary_and_large_files_reported(repo):
    (repo / "m.ckpt").write_bytes(b"0" * 10)
    (repo / "big.dat").write_bytes(b"0" * (cr.MAX_TRACKED_BYTES + 1))
    asset = repo / "training" / "families" / "x" / "assets"
    asset.mkdir(parents=True)
    (asset / "intended.safetensors").write_bytes(b"0" * 10)
    (repo / "other.safetensors").write_bytes(b"0" * 10)
    _git(repo, "add", "-A")
    text = "\n".join(cr.binary_findings(cr.git("ls-files").splitlines()))
    assert "m.ckpt" in text and "big.dat" in text and "other.safetensors" in text
    assert "intended.safetensors" not in text


def test_credential_urls_and_unc_paths():
    assert cr.scan_line("git clone https://bob:hunter2@example.org/x.git", [])
    assert cr.scan_line("see \\\\fileserver\\share\\data", [])
    assert not cr.scan_line("a plain https://example.org/x line", [])


def test_machine_name_term(repo, monkeypatch, capsys):
    monkeypatch.setattr(cr.platform, "node", lambda: "fakehost-77")
    (repo / "c.txt").write_text("built on FAKEHOST-77\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "more")
    assert cr.main(["--base", "basetag"]) == 1
    assert "fakehost-77" in capsys.readouterr().out
