"""Dataset filter query language.

Examples::

    1girl smile                 caption or filename contains both words
    tag:"long hair" -tag:blurry exact tag present / absent (underscores = spaces)
    tag:*hair                   wildcard tag match
    missing:caption             no caption (has:caption for the opposite)
    is:unsaved                  edited but not saved
    res:<768                    shorter side under 768 px  (w:, h:, mp: also work)
    ar:>1.5                     aspect ratio (width / height) above 1.5
    name:img_00*  ext:png       filename pattern / extension
    tags:>40  len:<20           tag count / caption length in characters
    flag:blurry                 Health scan results: duplicate, similar, blurry, lowres, crop

Terms are ANDed; prefix any term with ``-`` to negate it. Pure Python so it
can be unit tested and reused by a future training dataset filter.
"""
from __future__ import annotations

import fnmatch
import operator
import re
import shlex
from dataclasses import dataclass
from typing import Callable, Protocol

from core import captions

_CMP = {"<=": operator.le, ">=": operator.ge, "<": operator.lt, ">": operator.gt, "=": operator.eq}
_NUM_RE = re.compile(r"^(<=|>=|<|>|=)?\s*([0-9]*\.?[0-9]+)$")


class EntryLike(Protocol):
    name: str
    text: str
    width: int
    height: int
    dirty: bool


@dataclass
class Query:
    predicate: Callable[[EntryLike], bool]
    errors: list[str]
    needs_info: bool  # true if the query uses dimensions (which load lazily)


def _norm_tag(t: str) -> str:
    return " ".join(t.replace("_", " ").split()).lower()


def _numeric(value: str, getter: Callable[[EntryLike], float]) -> Callable[[EntryLike], bool] | None:
    m = _NUM_RE.match(value.strip())
    if not m:
        return None
    op = _CMP[m.group(1) or "="]
    target = float(m.group(2))
    return lambda e: op(getter(e), target)


def _term(key: str, value: str) -> tuple[Callable[[EntryLike], bool] | None, bool]:
    """Build a predicate for key:value. Returns (predicate or None if invalid, uses_dimensions)."""
    key = key.lower()
    if key == "tag":
        pattern = _norm_tag(value)
        if any(c in pattern for c in "*?["):
            return (lambda e: any(fnmatch.fnmatchcase(_norm_tag(t), pattern) for t in captions.split_tags(e.text))), False
        return (lambda e: any(_norm_tag(t) == pattern for t in captions.split_tags(e.text))), False
    if key in ("missing", "has"):
        if value.lower() not in ("caption", "captions", "text", "tags"):
            return None, False
        has = lambda e: bool(e.text.strip())  # noqa: E731
        return ((lambda e: not has(e)) if key == "missing" else has), False
    if key == "is":
        v = value.lower()
        if v in ("unsaved", "dirty", "modified", "edited"):
            return (lambda e: e.dirty), False
        if v in ("saved", "clean"):
            return (lambda e: not e.dirty), False
        if v in ("tagged", "tags"):
            return (lambda e: captions.looks_like_tags(e.text)), False
        if v in ("prose", "sentence", "natural"):
            return (lambda e: bool(e.text.strip()) and not captions.looks_like_tags(e.text)), False
        return None, False
    if key == "flag":
        f = value.lower()
        aliases = {"dup": "duplicate", "duplicates": "duplicate", "near": "similar", "blur": "blurry",
                   "low": "lowres", "small": "lowres", "any": "*"}
        f = aliases.get(f, f)
        if f == "*":
            return (lambda e: bool(getattr(e, "flags", None))), False
        return (lambda e: f in getattr(e, "flags", set())), False
    if key == "name":
        pat = value.lower()
        if any(c in pat for c in "*?["):
            return (lambda e: fnmatch.fnmatchcase(e.name.lower(), pat)), False
        return (lambda e: pat in e.name.lower()), False
    if key == "ext":
        ext = value.lower().lstrip(".")
        return (lambda e: e.name.lower().rsplit(".", 1)[-1] == ext), False
    getters: dict[str, tuple[Callable[[EntryLike], float], bool]] = {
        "w": (lambda e: e.width, True),
        "width": (lambda e: e.width, True),
        "h": (lambda e: e.height, True),
        "height": (lambda e: e.height, True),
        "res": (lambda e: min(e.width, e.height), True),
        "mp": (lambda e: e.width * e.height / 1_000_000, True),
        "ar": (lambda e: e.width / e.height if e.height else 0.0, True),
        "len": (lambda e: len(e.text.strip()), False),
        "tags": (lambda e: len(captions.split_tags(e.text)), False),
        "words": (lambda e: len(e.text.split()), False),
    }
    if key in getters:
        getter, dims = getters[key]
        pred = _numeric(value, getter)
        if pred is None:
            return None, dims
        if dims:  # unknown dimensions (not yet loaded) never match a size filter
            return (lambda e, p=pred: bool(e.width and e.height) and p(e)), True
        return pred, False
    return None, False


def compile_query(text: str) -> Query:
    try:
        tokens = shlex.split(text, posix=True)
    except ValueError:  # unbalanced quotes: fall back to plain whitespace split
        tokens = text.replace('"', " ").split()
    preds: list[Callable[[EntryLike], bool]] = []
    errors: list[str] = []
    needs_info = False
    for tok in tokens:
        negate = tok.startswith("-") and len(tok) > 1
        body = tok[1:] if negate else tok
        if ":" in body and not body.startswith(":"):
            key, value = body.split(":", 1)
            pred, dims = _term(key, value)
            needs_info |= dims
            if pred is None:
                errors.append(f"Didn't understand '{tok}'")
                continue
        else:
            word = body.lower()
            pred = lambda e, w=word: w in e.text.lower() or w in e.name.lower()  # noqa: E731
        preds.append((lambda e, p=pred: not p(e)) if negate else pred)

    def predicate(e: EntryLike) -> bool:
        return all(p(e) for p in preds)

    return Query(predicate=predicate, errors=errors, needs_info=needs_info)
