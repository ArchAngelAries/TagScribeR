"""Undo/redo for caption edits."""
from __future__ import annotations

from typing import Callable

from PySide6.QtGui import QUndoCommand

from core.dataset_session import Changes, DatasetSession

_TYPING_ID = 0x7A51


class ChangeCommand(QUndoCommand):
    """Applies a change set ``{key: (old, new)}``. Consecutive typing in one caption merges."""

    def __init__(self, session: DatasetSession, changes: Changes, text: str,
                 on_change: Callable[[list[str]], None], merge_key: str | None = None):
        super().__init__(text)
        self.session = session
        self.changes = dict(changes)
        self.on_change = on_change
        self.merge_key = merge_key

    def redo(self):
        self.session.apply(self.changes)
        self.on_change(list(self.changes))

    def undo(self):
        self.session.apply(self.changes, undo=True)
        self.on_change(list(self.changes))

    def id(self) -> int:
        return _TYPING_ID if self.merge_key else -1

    def mergeWith(self, other) -> bool:
        if not isinstance(other, ChangeCommand) or other.merge_key != self.merge_key or self.merge_key is None:
            return False
        for key, (_old, new) in other.changes.items():
            first_old = self.changes.get(key, (_old, new))[0]
            self.changes[key] = (first_old, new)
        return True
