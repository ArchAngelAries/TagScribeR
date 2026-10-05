"""Pre-push privacy check: what would a `git push` publish, and does any of it look personal?

    venv\\Scripts\\python.exe tools\\check_release.py            (checks the current branch against origin/main)
    venv\\Scripts\\python.exe tools\\check_release.py --base origin/main --extra "MyRealName,D:\\\\Private"

It reads git only (nothing is changed) and checks:
  * the tracked file list: nothing from user_data, settings, API presets, logs, models or datasets;
  * every tracked text file AND every line added by every commit not yet on the base branch, for API keys / tokens,
    private keys, e-mail addresses, home-folder and drive paths, private IP addresses and any --extra terms;
  * image metadata of tracked PNG / JPEG files (embedded text, EXIF);
  * that the private local files are git-ignored;
  * the author / committer identities on the commits to be pushed (shown, for you to judge).
Exit code 0 = nothing found, 1 = findings to review. A finding is a prompt to look, not proof of a leak.
"""
from __future__ import annotations

import argparse
import getpass
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SECRET_PATTERNS = {
    "OpenAI-style key": r"sk-[A-Za-z0-9_-]{16,}",
    "Hugging Face token": r"hf_[A-Za-z0-9]{20,}",
    "Google API key": r"AIza[0-9A-Za-z_-]{20,}",
    "GitHub token": r"gh[pousr]_[A-Za-z0-9]{20,}",
    "Slack token": r"xox[baprs]-[A-Za-z0-9-]{10,}",
    "AWS access key": r"AKIA[0-9A-Z]{16}",
    "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY",
    "bearer token": r"Bearer [A-Za-z0-9._-]{20,}",
    "literal password / key": r"(?i)(password|passwd|api[_-]?key|secret)\s*[:=]\s*[\"'][A-Za-z0-9_\-/+]{10,}[\"']",
    "e-mail address": r"[A-Za-z0-9._%+-]+@(?!example\.|anthropic\.com|users\.noreply)[A-Za-z0-9.-]+\.[a-z]{2,}",
    "Windows user folder": r"(?i)[a-z]:[\\/]+users[\\/]+(?!public|default)[^\\/\s\"']+",
    "Unix home folder": r"/(?:home|Users)/(?!user|you|name)[A-Za-z0-9._-]+",
    "private IP address": r"\b(?:10|192\.168|172\.(?:1[6-9]|2[0-9]|3[01]))\.\d{1,3}\.\d{1,3}(?:\.\d{1,3})?\b",
    "Windows machine name": r"\b(?:DESKTOP|LAPTOP)-[A-Z0-9]{5,}\b",
}
# drive paths that are clearly placeholders in docs, tooltips and the screenshot tool
PLACEHOLDER_PATHS = re.compile(r"(?i)^[a-z]:[\\/]+(datasets|models|tagscriber|path|program files)\b")
DRIVE_PATH = re.compile(r"(?i)(?<![A-Za-z0-9])[a-z]:\\{1,2}[A-Za-z0-9 _.-]+(?:\\{1,2}[A-Za-z0-9 _.-]+)+")
SENSITIVE_NAMES = re.compile(r"(?i)(^|/)(user_data/|config\.json$|api_presets\.json$|user_tags\.txt$|\.env$|"
                             r".*\.log$|.*\.pem$|.*\.key$|settings\.json$|models/|dataset collections/|image edits/)")
MUST_BE_IGNORED = ("config.json", "api_presets.json", "user_tags.txt", "user_data", "models", "Dataset Collections",
                   "Image Edits", "venv", ".claude")
ALLOWED_LINES = (re.compile(r"SKIP_PATTERNS|SECRET_PATTERNS|PLACEHOLDER_PATHS|MUST_BE_IGNORED"),)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          errors="replace").stdout


def scan_line(line: str, extra: list[str]) -> list[str]:
    if any(a.search(line) for a in ALLOWED_LINES):
        return []
    hits = [name for name, pat in SECRET_PATTERNS.items() if re.search(pat, line)]
    for m in DRIVE_PATH.finditer(line):
        if not PLACEHOLDER_PATHS.match(m.group(0)):
            hits.append(f"drive path {m.group(0)[:40]!r}")
    low = line.lower()
    hits += [f"term {t!r}" for t in extra if t and t.lower() in low]
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="origin/main", help="what is already public (default origin/main)")
    ap.add_argument("--extra", default="", help="comma-separated extra terms to look for (names, folders)")
    args = ap.parse_args(argv)
    extra = [t.strip() for t in args.extra.split(",") if t.strip()]
    user = getpass.getuser()
    if len(user) >= 4:
        extra.append(f"\\{user}\\")
        extra.append(f"/{user}/")
    findings: list[str] = []
    files = [f for f in git("ls-files").splitlines() if f]
    print(f"Tracked files: {len(files)}")

    for f in files:
        if SENSITIVE_NAMES.search(f) and not f.endswith(".py") and "/families/" not in f:
            findings.append(f"tracked file with a private-looking name: {f}")

    for f in files:
        p = ROOT / f
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            try:
                from PIL import Image
                with Image.open(p) as im:
                    info = {k: v for k, v in im.info.items() if k not in ("dpi", "gamma", "chromaticity", "srgb",
                                                                          "icc_profile", "jfif", "jfif_version",
                                                                          "jfif_unit", "jfif_density")}
                    if im.getexif():
                        info["exif"] = "present"
                if info:
                    findings.append(f"image metadata in {f}: {sorted(info)}")
            except Exception as e:  # noqa: BLE001
                findings.append(f"could not read image {f}: {e}")
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for h in scan_line(line, extra):
                findings.append(f"{f}:{n}: {h}: {line.strip()[:110]}")

    ahead = [c for c in git("rev-list", f"{args.base}..HEAD").splitlines() if c]
    print(f"Commits not yet on {args.base}: {len(ahead)}")
    commit = ""
    for line in git("log", f"{args.base}..HEAD", "-p", "--format=COMMIT %h %s").splitlines():
        if line.startswith("COMMIT "):
            commit = line[7:]
        elif line.startswith("+") and not line.startswith("+++"):
            for h in scan_line(line[1:], extra):
                findings.append(f"history, commit {commit[:60]}: {h}: {line[1:].strip()[:100]}")

    for name in MUST_BE_IGNORED:
        if (ROOT / name).exists() and subprocess.run(["git", "check-ignore", "-q", name], cwd=ROOT).returncode != 0:
            findings.append(f"private local path is NOT git-ignored: {name}")

    who = sorted(set(git("log", f"{args.base}..HEAD", "--format=%an <%ae> / %cn <%ce>").splitlines()))
    print("Identities on the commits to be pushed (they become public):")
    for w in who:
        print(f"  {w}")

    seen, unique = set(), []
    for f in findings:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    if unique:
        print(f"\n{len(unique)} finding(s) to review:")
        for f in unique:
            print(f"  - {f}")
        return 1
    print("\nNo findings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
