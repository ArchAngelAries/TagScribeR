"""Qt item models over a DatasetSession."""
from __future__ import annotations

from PySide6.QtCore import QAbstractListModel, QModelIndex, QSortFilterProxyModel, Qt, Signal

from core import captions
from core.dataset import natural_key
from core.dataset_session import DatasetSession, Entry
from core.query import compile_query

ENTRY_ROLE = Qt.UserRole + 1
KEY_ROLE = Qt.UserRole + 2

_HAS_CAPTION, _DIRTY = 1, 2


def _flags_of(e: Entry) -> int:
    return (_HAS_CAPTION if e.has_caption else 0) | (_DIRTY if e.dirty else 0)


class DatasetModel(QAbstractListModel):
    # Every row that refresh() is about to report, sent before the first dataChanged: a proxy drops its cached sort
    # keys for all of them at once (the entries have already changed, so a range reported first would otherwise be
    # compared against the stale keys of a range reported later).
    rows_changing = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.session: DatasetSession | None = None
        # Running totals for the status line: one flag per row, adjusted whenever refresh() reports a row changed,
        # so reading the counts never walks the folder.
        self._flags: list[int] = []
        self._n_captioned = 0
        self._n_dirty = 0

    def set_session(self, session: DatasetSession | None) -> None:
        self.beginResetModel()
        self.session = session
        self._recount()
        self.endResetModel()

    # -- running counts ----------------------------------------------------------------------------------------------
    def _recount(self) -> None:
        self._flags = [_flags_of(e) for e in self.session.entries] if self.session is not None else []
        self._n_captioned = sum(1 for f in self._flags if f & _HAS_CAPTION)
        self._n_dirty = sum(1 for f in self._flags if f & _DIRTY)

    def counts(self) -> dict[str, int]:
        """The numbers of DatasetSession.stats() (images, captioned, missing, unsaved), from the running totals."""
        if self.session is None:
            return {"images": 0, "captioned": 0, "missing": 0, "unsaved": 0}
        if len(self._flags) != len(self.session.entries):      # the entry list changed behind our back: start over
            self._recount()
        n = len(self._flags)
        return {"images": n, "captioned": self._n_captioned, "missing": n - self._n_captioned,
                "unsaved": self._n_dirty}

    def _update_flags(self, rows) -> None:
        entries = self.session.entries
        flags = self._flags
        if len(flags) != len(entries):
            self._recount()
            return
        for r in rows:
            new = _flags_of(entries[r])
            old = flags[r]
            if new != old:
                self._n_captioned += (1 if new & _HAS_CAPTION else 0) - (1 if old & _HAS_CAPTION else 0)
                self._n_dirty += (1 if new & _DIRTY else 0) - (1 if old & _DIRTY else 0)
                flags[r] = new

    # -- Qt model ----------------------------------------------------------------------------------------------------
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
        self._update_flags(rows)
        self.rows_changing.emit(rows)
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
            self._recount()
            self.dataChanged.emit(self.index(0), self.index(len(self.session) - 1))

    def remove_keys(self, keys) -> None:
        if self.session is None:
            return
        self.beginResetModel()
        self.session.remove(keys)
        self._recount()
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

_MISS = object()


class FilterProxy(QSortFilterProxyModel):
    """Filter and sort over the shared model.

    Sorting compares cached keys: a row's key for a sort mode is computed once, on first use, and dropped when the
    model reports that row changed, so ``lessThan`` is two dict reads and a compare instead of two Python round
    trips into the model plus the key function. The order is the one the uncached comparison gave.

    The proxy of a hidden browser is ``deferred``: a change of the data throws its mapping away (cheap) and marks it
    ``stale`` instead of re-sorting; nothing is sorted until the browser is shown or read.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.query = compile_query("")
        self.query_text = ""
        self.sort_mode = "Name"
        self.sort_key = SORT_MODES["Name"]
        self.deferred = False
        self.stale = False
        self.lessthan_calls = 0             # for tests: a hidden browser must not sort
        self._caches: dict[str, dict[int, object]] = {"Name": {}}
        self._cache = self._caches["Name"]

    # -- the model under us ------------------------------------------------------------------------------------------
    def setSourceModel(self, model) -> None:
        if model is not None:               # connected before the base class connects its own, so they run first
            model.dataChanged.connect(self._source_changed)
            model.modelAboutToBeReset.connect(self._source_about_to_reset)
            if hasattr(model, "rows_changing"):
                model.rows_changing.connect(self._rows_changing)
        super().setSourceModel(model)

    def _rows_changing(self, rows: list) -> None:
        for cache in self._caches.values():
            if len(rows) >= len(cache):
                cache.clear()
            else:
                for r in rows:
                    cache.pop(r, None)

    def _source_changed(self, top: QModelIndex, bottom: QModelIndex, *_roles) -> None:
        a, b = top.row(), bottom.row()
        for cache in self._caches.values():
            if not cache:
                continue
            if b - a + 1 >= len(cache):
                cache.clear()
            else:
                for r in range(a, b + 1):
                    cache.pop(r, None)
        if self.deferred and not self.stale:
            self.stale = True
            self.invalidate()               # lazy: the mapping is rebuilt on the first read after the browser is shown
            self.sort(-1)                   # no sort column: a later reset or change of the data sorts nothing (free)

    def _source_about_to_reset(self) -> None:
        for cache in self._caches.values():
            cache.clear()
        if self.deferred and not self.stale:
            self.sort(-1)                   # Qt sorts at once on a reset when a sort column is set: not for a hidden one
        self.stale = True                   # the rows of the new data are not worked out yet

    # -- configuration -----------------------------------------------------------------------------------------------
    def _select_mode(self, name: str) -> None:
        self.sort_mode = name if name in SORT_MODES else "Name"
        self.sort_key = SORT_MODES[self.sort_mode]
        self._cache = self._caches.setdefault(self.sort_mode, {})

    def set_query(self, text: str, apply: bool = True) -> list[str]:
        self.query_text = text
        self.query = compile_query(text)
        if apply:
            self.invalidateFilter()
        return self.query.errors

    def set_sort_mode(self, name: str, descending: bool = False, apply: bool = True) -> None:
        self._select_mode(name)
        if apply:
            self.invalidate()
            self.sort(0, Qt.DescendingOrder if descending else Qt.AscendingOrder)

    def configure(self, name: str, descending: bool, text: str) -> list[str]:
        """Sort mode, direction and filter text in one step (one lazy re-map); returns the filter's errors."""
        self._select_mode(name)
        self.set_query(text, apply=False)
        self.invalidate()
        self.sort(0, Qt.DescendingOrder if descending else Qt.AscendingOrder)
        self.stale = False
        return self.query.errors

    # -- the two callbacks Qt uses -----------------------------------------------------------------------------------
    def filterAcceptsRow(self, row: int, parent: QModelIndex) -> bool:
        if not self.query_text.strip():
            return True
        e = self.sourceModel().index(row, 0, parent).data(ENTRY_ROLE)
        return e is not None and self.query.predicate(e)

    def _key_of(self, row: int):
        session = self.sourceModel().session
        if session is None or not 0 <= row < len(session.entries):
            return None
        key = self._cache[row] = self.sort_key(session.entries[row])
        return key

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        self.lessthan_calls += 1
        cache = self._cache
        a = cache.get(left.row(), _MISS)
        if a is _MISS:
            a = self._key_of(left.row())
        b = cache.get(right.row(), _MISS)
        if b is _MISS:
            b = self._key_of(right.row())
        if a is None or b is None:
            return False
        return a < b
