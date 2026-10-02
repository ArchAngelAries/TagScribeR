"""Qt item models over a DatasetSession."""
from __future__ import annotations

from PySide6.QtCore import QAbstractListModel, QModelIndex, QSortFilterProxyModel, Qt

from core import captions
from core.dataset import natural_key
from core.dataset_session import DatasetSession, Entry
from core.query import compile_query

ENTRY_ROLE = Qt.UserRole + 1
KEY_ROLE = Qt.UserRole + 2


class DatasetModel(QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.session: DatasetSession | None = None

    def set_session(self, session: DatasetSession | None) -> None:
        self.beginResetModel()
        self.session = session
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() or self.session is None else len(self.session)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid() or self.session is None:
            return None
        e: Entry = self.session.entries[index.row()]
        if role == ENTRY_ROLE:
            return e
        if role == KEY_ROLE:
            return e.key
        if role == Qt.DisplayRole:
            return e.name
        if role == Qt.ToolTipRole:
            dims = f"{e.width} × {e.height}\n" if e.width else ""
            text = e.text.strip() or "(no caption)"
            if len(text) > 400:
                text = text[:400] + "…"
            return f"{e.name}\n{dims}{text}"
        return None

    def refresh(self, keys) -> None:
        """Repaint rows whose entry changed."""
        if self.session is None:
            return
        rows = sorted(r for r in (self.session.row_of(k) for k in keys) if r is not None)
        if not rows:
            return
        # Coalesce into contiguous ranges to keep signal traffic low on big batches.
        start = prev = rows[0]
        for r in rows[1:] + [None]:
            if r is not None and r == prev + 1:
                prev = r
                continue
            self.dataChanged.emit(self.index(start), self.index(prev))
            if r is not None:
                start = prev = r

    def refresh_all(self) -> None:
        if self.session and len(self.session):
            self.dataChanged.emit(self.index(0), self.index(len(self.session) - 1))

    def remove_keys(self, keys) -> None:
        if self.session is None:
            return
        self.beginResetModel()
        self.session.remove(keys)
        self.endResetModel()


SORT_MODES = {
    "Name": lambda e: natural_key(e.name),
    "Caption length": lambda e: len(e.text),
    "Tag count": lambda e: len(captions.split_tags(e.text)),
    "Width": lambda e: e.width,
    "Height": lambda e: e.height,
    "Megapixels": lambda e: e.width * e.height,
    "Aspect ratio": lambda e: (e.width / e.height) if e.height else 0,
    "Date modified": lambda e: e.mtime,
    "File size": lambda e: e.file_size,
    "Unsaved first": lambda e: (not e.dirty, natural_key(e.name)),
    "Missing captions first": lambda e: (e.has_caption, natural_key(e.name)),
}


class FilterProxy(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.query = compile_query("")
        self.query_text = ""
        self.sort_key = SORT_MODES["Name"]

    def set_query(self, text: str) -> list[str]:
        self.query_text = text
        self.query = compile_query(text)
        self.invalidateFilter()
        return self.query.errors

    def set_sort_mode(self, name: str, descending: bool = False) -> None:
        self.sort_key = SORT_MODES.get(name, SORT_MODES["Name"])
        self.invalidate()
        self.sort(0, Qt.DescendingOrder if descending else Qt.AscendingOrder)

    def filterAcceptsRow(self, row: int, parent: QModelIndex) -> bool:
        if not self.query_text.strip():
            return True
        e = self.sourceModel().index(row, 0, parent).data(ENTRY_ROLE)
        return e is not None and self.query.predicate(e)

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        a, b = left.data(ENTRY_ROLE), right.data(ENTRY_ROLE)
        if a is None or b is None:
            return False
        return self.sort_key(a) < self.sort_key(b)
