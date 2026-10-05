"""The memory bar at the bottom of the window: live VRAM and RAM use with a peak marker, as Fizgig's trainer shows it
(lora_trainer_gui.py: _draw_status_segment, _poll_status_bar, reset_status_peaks, _toggle_status_bar).

Both figures are for the whole machine, not only the training process, so other apps holding memory are included.
The numbers come from core/vram_monitor.py on a background thread; this widget only draws, once a second.
Fizgig's bar belongs to its trainer window; here it sits under every tab, so it also shows what a captioning model
takes. Peaks reset when a training run starts.
"""
from __future__ import annotations

import weakref

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from core.vram_monitor import MemoryReader

GIB = 1073741824          # binary GB: matches the "20 GB" on the card's box
VISIBLE_KEY = "ui.stats_bar_visible"
_bars: weakref.WeakSet = weakref.WeakSet()


def reset_all_peaks() -> None:
    """Zero the peak markers of every live bar. The Train tab calls this when a run starts."""
    for bar in list(_bars):
        bar.reset_peaks()


class _Segment(QWidget):
    """One bar: a track, a gradient fill up to the current use, a white tick at the peak, and the figures."""

    def __init__(self, label: str, c_start: str, c_end: str, tip: str):
        super().__init__()
        self.label, self.c_start, self.c_end = label, QColor(c_start), QColor(c_end)
        self.used = self.total = self.peak = 0
        self.available = None                       # None until the first sample; False = no reader on this machine
        self.setMinimumSize(250, 24)
        self.setMaximumHeight(28)
        self.setToolTip(tip)

    def set_reading(self, reading) -> None:
        if reading:
            self.used, self.total = reading
            self.peak = max(self.peak, self.used)
            self.available = True
        else:
            self.available = False                  # say so rather than leave a stale bar
        self.update()

    def text(self) -> str:
        if not self.available:
            return f"{self.label} stats unavailable"
        return f"{self.label}  {self.used / GIB:.1f} / {self.total / GIB:.1f} GB · peak {self.peak / GIB:.1f}"

    def paintEvent(self, _event):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#15171c"))
        font = QFont(self.font())
        font.setBold(bool(self.available))
        p.setFont(font)
        if not self.available:
            p.setPen(QColor("#8b93a1"))
            p.drawText(QRectF(10, 0, w - 10, h), Qt.AlignVCenter | Qt.AlignLeft, self.text())
            return
        frac = max(0.0, min(1.0, self.used / self.total)) if self.total else 0.0
        # the colour runs c_start -> c_end across the FULL width and is drawn up to the fill: fuller = closer to c_end
        grad = QLinearGradient(0, 0, w, 0)
        grad.setColorAt(0.0, self.c_start)
        grad.setColorAt(1.0, self.c_end)
        p.fillRect(QRectF(0, 1, w * frac, h - 2), grad)
        if self.peak and self.total:
            px = int(w * max(0.0, min(1.0, self.peak / self.total)))
            p.setPen(QPen(QColor("#FFFFFF"), 2))
            p.drawLine(px, 0, px, h)
        p.setPen(QColor("#FFFFFF"))
        p.drawText(QRectF(10, 0, w - 10, h), Qt.AlignVCenter | Qt.AlignLeft, self.text())


class MemoryBar(QWidget):
    """VRAM and RAM bars side by side. `reset_peaks()` at the start of a run; `shutdown()` when the app closes."""

    def __init__(self, reader: MemoryReader | None = None, parent=None):
        super().__init__(parent)
        self.reader = reader or MemoryReader()
        self.vram = _Segment("VRAM", "#3FB950", "#E5534B",                      # green -> red
                             "Video memory in use on the GPU, out of its total, and the highest it has reached "
                             "since the last training run started or the app opened (the white mark). This is "
                             "the whole card: other apps holding VRAM are included.")
        self.ram = _Segment("RAM", "#3B82F6", "#EAC54F",                        # blue -> yellow
                            "System memory in use, out of the total, and the highest it has reached since the last "
                            "training run started or the app opened (the white mark). This is the whole machine, "
                            "not only TagScribeR.")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.vram, 1)
        lay.addWidget(self.ram, 1)
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self.poll)
        _bars.add(self)

    def showEvent(self, event):
        """Sampling starts the first time the bar is actually on screen, so a tab that is never opened costs nothing."""
        super().showEvent(event)
        self.reader.start()
        self._timer.start()
        self.poll()

    def poll(self) -> None:
        """Peaks are tracked whether or not the bar is shown, so showing it mid-run still has the run's peak."""
        if not self.reader.samples:
            return                                  # nothing read yet: not the same as unavailable
        vram, ram = self.reader.latest
        self.vram.set_reading(vram)
        self.ram.set_reading(ram)

    def reset_peaks(self) -> None:
        self.vram.peak = self.ram.peak = 0

    def shutdown(self) -> None:
        self._timer.stop()
        self.reader.stop()


class MemoryStrip(QWidget):
    """The bar plus its Hide stats / Show stats button, for the bottom of the main window. The choice is remembered."""

    def __init__(self, cfg, reader: MemoryReader | None = None, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.bar = MemoryBar(reader)
        self.btn = QPushButton()
        self.btn.setFlat(True)
        self.btn.setToolTip("Show or hide the VRAM / RAM bars. Peaks are still tracked while hidden.")
        self.btn.clicked.connect(self.toggle)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 3, 8, 3)
        lay.addWidget(self.bar, 1)
        lay.addWidget(self.btn)
        # the key was training.stats_bar_visible while the bar lived in the Train tab
        self.set_shown(bool(cfg.get(VISIBLE_KEY, cfg.get("training.stats_bar_visible", True))))

    def set_shown(self, visible: bool) -> None:
        self.bar.setVisible(visible)
        self.btn.setText("Hide stats" if visible else "Show stats")

    def toggle(self) -> None:
        visible = self.bar.isHidden()
        self.set_shown(visible)
        self.cfg.set(VISIBLE_KEY, visible)

    def shutdown(self) -> None:
        self.bar.shutdown()
