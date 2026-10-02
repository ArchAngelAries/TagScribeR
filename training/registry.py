# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/families/registry.py
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: GUI-label lookup and the workbench filter are dropped (no workbench yet); test families
# can be registered for the CPU smoke tests.
"""The model families TagScribeR can train, keyed by family key. Importable without torch."""
from typing import Optional

from training.description import FamilyDescription
from training.families.qwen_image21.description import QWEN_IMAGE_21

FAMILIES = {d.key: d for d in (QWEN_IMAGE_21,)}


def _check(d: FamilyDescription) -> None:
    problems = d.validate()
    if problems:
        raise ValueError(f"family description {d.key!r} is inconsistent: {'; '.join(problems)}")


for _d in FAMILIES.values():
    _check(_d)


def _extra_from_env() -> None:
    """TAGSCRIBER_TRAINING_EXTRA_FAMILIES="pkg.module:function,...": call each to register extra families in child
    processes (the CPU test family uses this; nothing in the app sets it)."""
    import importlib
    import os
    for spec in filter(None, os.environ.get("TAGSCRIBER_TRAINING_EXTRA_FAMILIES", "").split(",")):
        mod, _, fn = spec.strip().partition(":")
        getattr(importlib.import_module(mod), fn)()


def register(desc: FamilyDescription) -> None:
    """Add a family at runtime (tests register a tiny CPU family this way)."""
    _check(desc)
    FAMILIES[desc.key] = desc


def get(key: str) -> Optional[FamilyDescription]:
    return FAMILIES.get(key)


def by_arch_id(arch_id: str) -> Optional[FamilyDescription]:
    """The description whose architecture id (cache filenames, metadata) is arch_id."""
    for d in FAMILIES.values():
        if d.arch_id == arch_id:
            return d
    return None


def training_families(include_hidden: bool = False) -> list:
    """Descriptions whose driver exists (shown in the Train tab's family picker). Families whose key starts with
    "_" (test families) are hidden unless asked for."""
    return [d for d in FAMILIES.values() if d.training_ready and (include_hidden or not d.key.startswith("_"))]


_extra_from_env()
