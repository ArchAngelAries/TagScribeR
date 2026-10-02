"""Pure caption / tag manipulation.

Captions are stored as plain text. A caption is "tag-style" when it is a
separator-delimited list (``1girl, solo, red hair``) and "prose" otherwise;
every function here works on either, treating the text as a tag list split on
the separator. Nothing in this module touches the filesystem or Qt, so it is
fully unit-testable.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, Sequence

DEFAULT_SEPARATOR = ","


def split_tags(text: str, sep: str = DEFAULT_SEPARATOR) -> list[str]:
    """Split caption text into stripped, non-empty tags. Newlines also separate."""
    if not text:
        return []
    parts: list[str] = []
    for line in text.splitlines():
        parts.extend(line.split(sep))
    return [p.strip() for p in parts if p.strip()]


def join_tags(tags: Iterable[str], sep: str = DEFAULT_SEPARATOR) -> str:
    """Join tags with the separator followed by a space (``a, b, c``)."""
    joiner = sep if sep.endswith(" ") or sep == "\n" else sep + " "
    return joiner.join(t for t in tags if t)


def _key(tag: str, case_sensitive: bool) -> str:
    t = " ".join(tag.split())  # collapse internal whitespace
    return t if case_sensitive else t.lower()


def dedupe_tags(tags: Sequence[str], case_sensitive: bool = False) -> list[str]:
    """Remove duplicate tags, keeping the first occurrence and original order."""
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        k = _key(t, case_sensitive)
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


def contains_tag(text: str, tag: str, sep: str = DEFAULT_SEPARATOR, case_sensitive: bool = False) -> bool:
    k = _key(tag, case_sensitive)
    return any(_key(t, case_sensitive) == k for t in split_tags(text, sep))


def add_tags(text: str, new_tags: Sequence[str], position: str = "append",
             sep: str = DEFAULT_SEPARATOR, case_sensitive: bool = False) -> str:
    """Add tags that aren't already present. ``position`` is 'append' or 'prepend'."""
    existing = split_tags(text, sep)
    keys = {_key(t, case_sensitive) for t in existing}
    to_add = [t for t in dedupe_tags([t.strip() for t in new_tags if t.strip()], case_sensitive)
              if _key(t, case_sensitive) not in keys]
    if not to_add:
        return text
    merged = to_add + existing if position == "prepend" else existing + to_add
    return join_tags(merged, sep)


def remove_tags(text: str, tags: Sequence[str], sep: str = DEFAULT_SEPARATOR,
                case_sensitive: bool = False) -> str:
    keys = {_key(t, case_sensitive) for t in tags}
    kept = [t for t in split_tags(text, sep) if _key(t, case_sensitive) not in keys]
    return join_tags(kept, sep)


def replace_tag(text: str, old: str, new: str, sep: str = DEFAULT_SEPARATOR,
                case_sensitive: bool = False) -> str:
    """Replace a whole tag (not substrings). Replacing with '' removes it."""
    k = _key(old, case_sensitive)
    out = []
    for t in split_tags(text, sep):
        if _key(t, case_sensitive) == k:
            if new.strip():
                out.append(new.strip())
        else:
            out.append(t)
    return join_tags(dedupe_tags(out, case_sensitive), sep)


def find_replace(text: str, find: str, replace: str, *, regex: bool = False,
                 case_sensitive: bool = True) -> str:
    """Substring / regex find-and-replace over the raw caption text."""
    if not find:
        return text
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = find if regex else re.escape(find)
    return re.sub(pattern, replace, text, flags=flags)


def normalize_tag(tag: str, *, underscores_to_spaces: bool = False, lowercase: bool = False,
                  escape_parentheses: bool = False) -> str:
    t = " ".join(tag.split())
    if underscores_to_spaces:
        # Keep emoticon-style tags like ^_^ or >_< intact.
        if not re.fullmatch(r"\W+_\W+", t):
            t = t.replace("_", " ")
    if lowercase:
        t = t.lower()
    if escape_parentheses:
        t = re.sub(r"(?<!\\)([()])", r"\\\1", t)
    return t


def normalize_caption(text: str, *, sep: str = DEFAULT_SEPARATOR, dedupe: bool = True,
                      underscores_to_spaces: bool = False, lowercase: bool = False,
                      escape_parentheses: bool = False, sort: bool = False,
                      case_sensitive: bool = False) -> str:
    tags = [normalize_tag(t, underscores_to_spaces=underscores_to_spaces, lowercase=lowercase,
                          escape_parentheses=escape_parentheses) for t in split_tags(text, sep)]
    if dedupe:
        tags = dedupe_tags(tags, case_sensitive)
    if sort:
        tags = sorted(tags, key=str.lower)
    return join_tags(tags, sep)


def to_ascii(text: str) -> str:
    """Fold accented characters to ASCII (``é`` -> ``e``); drops what can't be folded."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def merge_generated_tags(existing: str, generated: Sequence[str], mode: str = "append", *,
                         prepend: Sequence[str] = (), append: Sequence[str] = (),
                         sep: str = DEFAULT_SEPARATOR) -> str:
    """Combine auto-generated tags with an existing caption.

    mode: 'overwrite' replaces the caption; 'ignore' keeps a non-empty caption
    untouched; 'append' adds only tags that are not already present.
    Forced ``prepend`` tags always lead and ``append`` tags always trail.
    """
    existing = existing.strip()
    if mode == "ignore" and existing:
        return existing
    base = [] if mode == "overwrite" else split_tags(existing, sep)
    body = dedupe_tags(base + list(generated))
    forced_front = [t for t in prepend if t.strip()]
    forced_back = [t for t in append if t.strip()]
    front_keys = {_key(t, False) for t in forced_front}
    back_keys = {_key(t, False) for t in forced_back}
    middle = [t for t in body if _key(t, False) not in front_keys | back_keys]
    return join_tags(dedupe_tags(forced_front + middle + forced_back), sep)


def tag_frequencies(captions: Iterable[str], sep: str = DEFAULT_SEPARATOR,
                    case_sensitive: bool = False) -> Counter:
    """Count in how many captions each tag appears (once per caption)."""
    counts: Counter = Counter()
    for text in captions:
        counts.update({_key(t, case_sensitive) for t in split_tags(text, sep)})
    return counts


def looks_like_tags(text: str, sep: str = DEFAULT_SEPARATOR) -> bool:
    """Heuristic: mostly short, separator-delimited fragments rather than sentences."""
    tags = split_tags(text, sep)
    if len(tags) < 2:
        return False
    avg_words = sum(len(t.split()) for t in tags) / len(tags)
    return avg_words <= 4


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def clean_model_output(text: str) -> str:
    """Tidy raw VLM output: drop reasoning blocks, code fences and boilerplate lead-ins."""
    if not text:
        return ""
    t = _THINK_RE.sub("", text)
    # An unterminated <think> (generation cut off mid-thought) leaves nothing usable after it.
    if "<think>" in t.lower():
        t = t[: t.lower().index("<think>")]
    # A stray closing tag with the reasoning before it (some templates omit the opener).
    if "</think>" in t.lower():
        t = t[t.lower().rindex("</think>") + len("</think>"):]
    t = t.strip()
    fence = re.fullmatch(r"```[a-zA-Z]*\n?(.*?)\n?```", t, re.DOTALL)
    if fence:
        t = fence.group(1).strip()
    t = re.sub(r"^(sure|certainly|of course)[,!.][^\n]*?:\s*", "", t, flags=re.IGNORECASE)
    return t.strip()


_TOKEN_RE = re.compile(r"[A-Za-z]+|\d|[^\sA-Za-z\d]")


def estimate_clip_tokens(text: str) -> int:
    """Approximate CLIP BPE token count (no tokenizer download needed).

    Common English words are one token; long or rare words split into several
    pieces (~1 per 6 letters); every digit and punctuation mark is its own token.
    Good enough to warn about SD1.5 / SDXL's 75-token chunk limit.
    """
    total = 0
    for tok in _TOKEN_RE.findall(text or ""):
        total += max(1, -(-len(tok) // 6)) if tok.isalpha() else 1
    return total


def word_diff(old: str, new: str) -> list[tuple[str, str]]:
    """Word-level diff as [(op, text)], op in {'same', 'add', 'del'}; whitespace is kept with words."""
    import difflib
    a = re.findall(r"\S+\s*", old or "")
    b = re.findall(r"\S+\s*", new or "")
    out: list[tuple[str, str]] = []
    sm = difflib.SequenceMatcher(a=[w.strip().lower() for w in a], b=[w.strip().lower() for w in b],
                                 autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.append(("same", "".join(b[j1:j2])))
        else:
            if i2 > i1:
                out.append(("del", "".join(a[i1:i2])))
            if j2 > j1:
                out.append(("add", "".join(b[j1:j2])))
    return out
