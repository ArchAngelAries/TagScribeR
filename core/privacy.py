"""Scrubbing text before it is shared: a pasted console, a log, a report. No Qt, no torch.

`scrub(text, extra)` replaces the user's home folder, the application folder, the user-data folder and each folder in
`extra` by placeholders, then the user-name component of any other Windows or POSIX home-folder path, the machine
name, and anything that looks like a credential. It is pure and never raises.

Placeholders: <home> <app> <user_data> <path1> <path2> ... (extra, in the order given) <user> <machine> <redacted>.
"""
from __future__ import annotations

import os
import platform
import re
from typing import Iterable, List, Tuple

_SEP = r"[\\/]+"
_UNC_PREFIX = r"(?:[\\/]{2}\?[\\/])?"           # the \\?\ long-path prefix, in either slash spelling
_FLAGS = re.I if os.name == "nt" else 0


def _path_regex(path: str):
    """A regex matching `path` with either slash spelling (and doubled slashes, as in JSON), or None."""
    p = str(path).strip()
    if p.startswith("\\\\?\\") or p.startswith("//?/"):
        p = p[4:]
    parts = [c for c in re.split(r"[\\/]+", p) if c]
    if not parts or len("".join(parts)) < 3:    # a drive root, "/" or a one-letter name: nothing to protect
        return None
    drive = bool(re.fullmatch(r"[A-Za-z]:", parts[0]))
    head = _UNC_PREFIX if drive else (_SEP if p[:1] in "\\/" else "")
    body = _SEP.join(re.escape(c) for c in parts)
    return re.compile(r"(?<!\w)" + head + body + r"(?![\w-])", _FLAGS)


def _known_folders(extra: Iterable[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    try:
        out.append((os.path.expanduser("~"), "<home>"))
    except Exception:       # noqa: BLE001
        pass
    try:
        from core import paths
        out.append((str(paths.APP_ROOT), "<app>"))
        out.append((str(paths.USER_DATA), "<user_data>"))
    except Exception:       # noqa: BLE001
        pass
    try:
        for i, e in enumerate(extra or (), 1):
            if e:
                out.append((str(e), f"<path{i}>"))
    except Exception:       # noqa: BLE001
        pass
    return out


_WIN_USER = re.compile(r"([A-Za-z]:[\\/]+Users[\\/]+)(?!<)[^\\/\s\"'<>|]+", re.I)
_NIX_USER = re.compile(r"((?<![\w.])/(?:home|Users)/)(?!<)[^/\s\"'<>|\\]+")
_CREDENTIALS = (
    (re.compile(r"(?i)\b(Bearer)\s+[A-Za-z0-9._~+/=-]{8,}"), r"\1 <redacted>"),
    (re.compile(r"(?i)\b(api[_-]?key|apikey|access[_-]?token|auth[_-]?token|secret|password|passwd)"
                r"""(["']?\s*[:=]\s*["']?)[^\s"',;&]+"""), r"\1\2<redacted>"),
    (re.compile(r"\bhf_[A-Za-z0-9]{10,}"), "<redacted>"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{10,}"), "<redacted>"),
)


def scrub(text: str, extra: Iterable[str] = ()) -> str:
    """`text` without the user's folders, user name, machine name or credentials (see the module docstring)."""
    try:
        if not isinstance(text, str):
            text = str(text)
        pairs = []
        for path, label in _known_folders(extra):
            rx = _path_regex(path)
            if rx is not None:
                pairs.append((len(str(path).strip()), rx, label))
        pairs.sort(key=lambda t: -t[0])            # longest first: a nested folder is not half-replaced
        for _n, rx, label in pairs:
            text = rx.sub(lambda _m, _l=label: _l, text)
        text = _WIN_USER.sub(lambda m: m.group(1) + "<user>", text)
        text = _NIX_USER.sub(lambda m: m.group(1) + "<user>", text)
        try:
            node = platform.node()
        except Exception:   # noqa: BLE001
            node = ""
        if node and len(node) >= 2:
            text = re.sub(r"(?<![\w-])" + re.escape(node) + r"(?![\w-])", "<machine>", text, flags=re.I)
        for rx, repl in _CREDENTIALS:
            text = rx.sub(repl, text)
        return text
    except Exception:       # noqa: BLE001
        try:
            return str(text)
        except Exception:   # noqa: BLE001
            return ""
