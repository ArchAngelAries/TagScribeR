"""An open dataset folder: images, their in-memory captions, and dirty state.

This is the single source of truth the workspace UI edits. All batch
operations are expressed as text transforms that return a change set
``{path: (old, new)}`` so the UI can push them onto an undo stack. Nothing is
written to disk until ``save`` (which uses the safe caption writer).
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from core import caption_io, captions, dataset

log = logging.getLogger(__name__)

Changes = dict[str, tuple[str, str]]


@dataclass
class Entry:
    path: Path
    text: str = ""
    saved_text: str = ""
    width: int = 0
    height: int = 0
    file_size: int = 0
    mtime: float = 0.0
    info_loaded: bool = False

    @property
    def key(self) -> str:
        return str(self.path)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def dirty(self) -> bool:
        return self.text != self.saved_text

    @property
    def has_caption(self) -> bool:
        return bool(self.text.strip())


@dataclass
class SaveReport:
    saved: int = 0
    unchanged: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)


class DatasetSession:
    def __init__(self, folder: Path, entries: list[Entry], caption_ext: str = ".txt"):
        self.folder = Path(folder)
        self.caption_ext = caption_ext
        self.entries = entries
        self._index = {e.key: i for i, e in enumerate(entries)}

    @classmethod
    def open(cls, folder: str | Path, recursive: bool = False, caption_ext: str = ".txt") -> "DatasetSession":
        entries = []
        for p in dataset.scan_images(folder, recursive=recursive):
            text = caption_io.read_caption(p, caption_ext)
            try:
                st = p.stat()
                size, mtime = st.st_size, st.st_mtime
            except OSError:
                size, mtime = 0, 0.0
            entries.append(Entry(path=p, text=text, saved_text=text, file_size=size, mtime=mtime))
        return cls(Path(folder), entries, caption_ext)

    # -- lookup -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    def row_of(self, key: str) -> int | None:
        return self._index.get(key)

    def get(self, key: str) -> Entry | None:
        i = self._index.get(key)
        return self.entries[i] if i is not None else None

    def remove(self, keys: Iterable[str]) -> None:
        drop = set(keys)
        self.entries = [e for e in self.entries if e.key not in drop]
        self._index = {e.key: i for i, e in enumerate(self.entries)}

    @property
    def dirty_entries(self) -> list[Entry]:
        return [e for e in self.entries if e.dirty]

    # -- editing ----------------------------------------------------------
    def set_texts(self, texts: dict[str, str]) -> None:
        for key, text in texts.items():
            e = self.get(key)
            if e is not None:
                e.text = text

    def transform(self, keys: Iterable[str], fn: Callable[[str], str]) -> Changes:
        """Apply ``fn`` to each caption; returns only the captions that changed (not yet applied)."""
        changes: Changes = {}
        for key in keys:
            e = self.get(key)
            if e is None:
                continue
            new = fn(e.text)
            if new != e.text:
                changes[key] = (e.text, new)
        return changes

    def apply(self, changes: Changes, undo: bool = False) -> None:
        self.set_texts({k: (old if undo else new) for k, (old, new) in changes.items()})

    # -- persistence ------------------------------------------------------
    def save(self, keys: Iterable[str] | None = None) -> SaveReport:
        report = SaveReport()
        targets = [self.get(k) for k in keys] if keys is not None else self.entries
        for e in targets:
            if e is None:
                continue
            if not e.dirty:
                report.unchanged += 1
                continue
            try:
                caption_io.write_caption(e.path, e.text, self.caption_ext)
                e.saved_text = e.text
                report.saved += 1
            except OSError as err:
                report.failed.append((e.key, str(err)))
                log.warning("Save failed for %s: %s", e.path, err)
        return report

    def reload_from_disk(self, key: str) -> None:
        """Refresh one caption after an external writer (e.g. a caption job) changed it."""
        e = self.get(key)
        if e is not None:
            e.text = e.saved_text = caption_io.read_caption(e.path, self.caption_ext)

    # -- statistics -------------------------------------------------------
    def tag_counts(self, keys: Iterable[str] | None = None) -> list[tuple[str, int]]:
        """Tag → number of captions containing it, most frequent first.

        Tags are grouped case/underscore-insensitively and shown in their most
        common spelling.
        """
        targets = [self.get(k) for k in keys] if keys is not None else self.entries
        counts: Counter = Counter()
        spellings: dict[str, Counter] = {}
        for e in targets:
            if e is None:
                continue
            seen = set()
            for t in captions.split_tags(e.text):
                norm = " ".join(t.replace("_", " ").split()).lower()
                spellings.setdefault(norm, Counter())[t] += 1
                if norm not in seen:
                    seen.add(norm)
                    counts[norm] += 1
        return [(spellings[n].most_common(1)[0][0], c) for n, c in counts.most_common()]

    def stats(self) -> dict[str, int]:
        return {
            "images": len(self.entries),
            "captioned": sum(1 for e in self.entries if e.has_caption),
            "missing": sum(1 for e in self.entries if not e.has_caption),
            "unsaved": sum(1 for e in self.entries if e.dirty),
        }


# -- batch transform factories (return text -> text functions) -----------------
def op_add_tags(tags: list[str], position: str = "append") -> Callable[[str], str]:
    return lambda text: captions.add_tags(text, tags, position)


def op_remove_tags(tags: list[str]) -> Callable[[str], str]:
    norm = {" ".join(t.replace("_", " ").split()).lower() for t in tags}
    return lambda text: captions.join_tags(
        t for t in captions.split_tags(text) if " ".join(t.replace("_", " ").split()).lower() not in norm)


def op_replace_tag(old: str, new: str) -> Callable[[str], str]:
    target = " ".join(old.replace("_", " ").split()).lower()

    def fn(text: str) -> str:
        tags = captions.split_tags(text)
        if not any(" ".join(t.replace("_", " ").split()).lower() == target for t in tags):
            return text
        out = []
        for t in tags:
            if " ".join(t.replace("_", " ").split()).lower() == target:
                if new.strip():
                    out.append(new.strip())
            else:
                out.append(t)
        return captions.join_tags(captions.dedupe_tags(out))
    return fn


def op_find_replace(find: str, replace: str, regex: bool = False, case_sensitive: bool = True) -> Callable[[str], str]:
    return lambda text: captions.find_replace(text, find, replace, regex=regex, case_sensitive=case_sensitive)


def op_normalize(underscores_to_spaces: bool = False, lowercase: bool = False, dedupe: bool = True,
                 sort: bool = False, escape_parentheses: bool = False) -> Callable[[str], str]:
    def fn(text: str) -> str:
        if not text.strip():
            return text
        return captions.normalize_caption(text, dedupe=dedupe, underscores_to_spaces=underscores_to_spaces,
                                          lowercase=lowercase, sort=sort, escape_parentheses=escape_parentheses)
    return fn


def op_prefix_suffix(prefix: str = "", suffix: str = "") -> Callable[[str], str]:
    """Raw-text prefix/suffix (works for prose captions too)."""
    def fn(text: str) -> str:
        t = text.strip()
        if prefix and not t.startswith(prefix):
            t = f"{prefix}{t}"
        if suffix and not t.endswith(suffix):
            t = f"{t}{suffix}"
        return t
    return fn


def op_ascii() -> Callable[[str], str]:
    return captions.to_ascii


def op_clear() -> Callable[[str], str]:
    return lambda text: ""


def op_set(text_value: str) -> Callable[[str], str]:
    return lambda text: text_value
