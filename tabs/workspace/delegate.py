"""Paints dataset cards in the grid (no per-image widgets, so thousands scroll smoothly)."""
from __future__ import annotations

from PySide6.QtCore import QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QStyle, QStyledItemDelegate

from tabs.workspace.model import ENTRY_ROLE

CARD_BG = QColor("#2b2b2b")
CARD_HOVER = QColor("#333638")
THUMB_BG = QColor("#1e1e1e")
ACCENT = QColor("#00b894")
UNSAVED = QColor("#fdcb6e")
MISSING = QColor("#d63031")
TEXT = QColor("#d0d0d0")
SUBTLE = QColor("#8a8a8a")


class CardDelegate(QStyledItemDelegate):
    TEXT_LINES = 3

    def __init__(self, loader, parent=None, job_state: dict | None = None):
        super().__init__(parent)
        self.loader = loader
        self.job_state = job_state if job_state is not None else {}
        self.card_width = 220
        self.show_captions = True
        self._font = QFont()
        self._font.setPointSizeF(8.5)
        self._small = QFont()
        self._small.setPointSizeF(7.5)

    def set_card_width(self, w: int) -> None:
        self.card_width = w
        # Text grows gently with the card (readability when zoomed in) but is
        # bounded so every card keeps the same, predictable layout.
        scale = max(1.0, min(1.6, (w / 220) ** 0.5))
        self._font.setPointSizeF(8.5 * scale)
        self._small.setPointSizeF(7.5 * scale)

    def text_height(self) -> int:
        if not self.show_captions:
            return QFontMetrics(self._small).height() + 8
        return QFontMetrics(self._font).lineSpacing() * self.TEXT_LINES + QFontMetrics(self._small).height() + 14

    def sizeHint(self, option, index) -> QSize:
        return QSize(self.card_width, self.card_width + self.text_height())

    def paint(self, painter: QPainter, option, index) -> None:
        e = index.data(ENTRY_ROLE)
        if e is None:
            return
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        r = option.rect.adjusted(4, 4, -4, -4)
        selected = bool(option.state & QStyle.State_Selected)
        hover = bool(option.state & QStyle.State_MouseOver)

        path = QPainterPath()
        path.addRoundedRect(QRectF(r), 8, 8)
        painter.fillPath(path, CARD_HOVER if hover and not selected else CARD_BG)
        job = self.job_state.get(e.key)
        if job == "working":
            painter.setPen(QPen(UNSAVED, 2.5))
            painter.drawPath(path)
        elif job == "failed":
            painter.setPen(QPen(MISSING, 2.5))
            painter.drawPath(path)
        elif selected:
            painter.setPen(QPen(ACCENT, 2.5))
            painter.drawPath(path)

        # Thumbnail
        side = r.width() - 10
        thumb_rect = QRect(r.left() + 5, r.top() + 5, side, side)
        painter.fillRect(thumb_rect, THUMB_BG)
        dpr = painter.device().devicePixelRatioF() if painter.device() else 1.0
        pm = self.loader.pixmap(e.key, e.mtime, e.file_size, int(side * dpr))
        if pm is not None:
            scaled = pm.size().scaled(thumb_rect.size(), Qt.KeepAspectRatio)
            target = QRect(0, 0, scaled.width(), scaled.height())
            target.moveCenter(thumb_rect.center())
            painter.drawPixmap(target, pm)
        else:
            painter.setPen(SUBTLE)
            painter.setFont(self._small)
            label = "Unreadable image" if self.loader.is_failed(e.key) else "…"
            painter.drawText(thumb_rect, Qt.AlignCenter, label)

        # Badges
        painter.setFont(self._small)
        if e.dirty:
            painter.setBrush(UNSAVED)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(thumb_rect.left() + 6, thumb_rect.top() + 6, 10, 10)
        if not e.has_caption:
            self._pill(painter, "NO CAPTION", MISSING, thumb_rect.left() + (22 if e.dirty else 6), thumb_rect.top() + 4)
        if job:
            label = {"working": "AI WORKING…", "queued": "QUEUED", "failed": "FAILED"}.get(job, job.upper())
            color = {"working": QColor("#b7791f"), "queued": QColor(60, 60, 60, 220), "failed": MISSING}.get(job, SUBTLE)
            fm = QFontMetrics(self._small)
            self._pill(painter, label, color, thumb_rect.right() - fm.horizontalAdvance(label) - 14, thumb_rect.top() + 4)
        if e.width:
            dims = f"{e.width}×{e.height}"
            fm = QFontMetrics(self._small)
            self._pill(painter, dims, QColor(0, 0, 0, 150), thumb_rect.right() - fm.horizontalAdvance(dims) - 14,
                       thumb_rect.bottom() - fm.height() - 6)

        # Text
        y = thumb_rect.bottom() + 6
        fm_small = QFontMetrics(self._small)
        painter.setPen(SUBTLE)
        painter.drawText(QRect(r.left() + 6, y, r.width() - 12, fm_small.height()), Qt.AlignLeft | Qt.AlignVCenter,
                         fm_small.elidedText(e.name, Qt.ElideMiddle, r.width() - 12))
        if self.show_captions:
            y += fm_small.height() + 2
            painter.setFont(self._font)
            painter.setPen(TEXT if e.has_caption else SUBTLE)
            fm = QFontMetrics(self._font)
            text_rect = QRect(r.left() + 6, y, r.width() - 12, fm.lineSpacing() * self.TEXT_LINES)
            painter.drawText(text_rect, Qt.TextWordWrap | Qt.AlignTop | Qt.AlignLeft,
                             self._elide_lines(e.text.strip() or "—", fm, text_rect.width()))
        painter.restore()

    def _pill(self, painter: QPainter, text: str, color: QColor, x: int, y: int) -> None:
        fm = QFontMetrics(self._small)
        rect = QRectF(x, y, fm.horizontalAdvance(text) + 10, fm.height() + 2)
        p = QPainterPath()
        p.addRoundedRect(rect, 4, 4)
        painter.fillPath(p, color)
        painter.setPen(Qt.white)
        painter.drawText(rect, Qt.AlignCenter, text)

    def _elide_lines(self, text: str, fm: QFontMetrics, width: int) -> str:
        """Cheap multi-line elision: cut the text to what fits in TEXT_LINES lines."""
        text = " ".join(text.split())
        budget = width * self.TEXT_LINES
        if fm.horizontalAdvance(text) <= budget * 0.92:
            return text
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if fm.horizontalAdvance(text[:mid]) <= budget * 0.88:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo].rstrip() + "…"
