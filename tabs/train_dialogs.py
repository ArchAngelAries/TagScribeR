"""Train tab windows: loss chart, Problem Images, live sample override."""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QSplitter, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from tabs.common import hint_label
from training import pipeline


class LossChart(QWidget):
    """Epoch-average loss (line) and learning rate (dashed, own scale), drawn with QPainter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.points: list[tuple[int, float, float]] = []     # (epoch, loss, lr)
        self.marks: dict[int, str] = {}                      # epoch -> adaptive action
        self.setMinimumHeight(150)
        self.setToolTip("Average training loss per epoch (green) and the learning rate (dashed orange, its own "
                        "scale). Adaptive LR decisions are marked: up = probe, down = reduce, ! = rollback. Loss "
                        "going down is not the whole story - judge by the sample previews.")

    def clear(self):
        self.points, self.marks = [], {}
        self.update()

    def add(self, epoch, loss, lr):
        self.points = [p for p in self.points if p[0] != epoch] + [(epoch, loss, lr)]
        self.points.sort()
        self.update()

    def mark(self, epoch, action):
        self.marks[epoch] = action
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(40, 10, -40, -22)
        p.fillRect(self.rect(), QColor("#1e1e1e"))
        p.setPen(QColor("#555"))
        p.drawRect(r)
        if not self.points:
            p.setPen(QColor("#888"))
            p.drawText(r, Qt.AlignCenter, "The loss chart fills in as epochs finish.")
            return
        eps = [e for e, _, _ in self.points]
        e0, e1 = min(eps), max(max(eps), min(eps) + 1)
        losses = [l for _, l, _ in self.points]
        lo, hi = min(losses), max(losses)
        if hi - lo < 1e-9:
            lo, hi = lo - 0.01, hi + 0.01
        lrs = [x for _, _, x in self.points if x > 0]
        lr_lo, lr_hi = (min(lrs), max(lrs)) if lrs else (0, 1)
        if lr_hi - lr_lo < 1e-12:
            lr_lo, lr_hi = lr_lo * 0.9, lr_hi * 1.1 + 1e-12

        def xy(e, v, a, b):
            return QPointF(r.left() + (e - e0) / (e1 - e0) * r.width(), r.bottom() - (v - a) / (b - a) * r.height())
        p.setPen(QColor("#aaa"))
        p.drawText(2, r.top() + 10, f"{hi:.4f}")
        p.drawText(2, r.bottom(), f"{lo:.4f}")
        p.drawText(r.left(), self.height() - 4, f"epoch {e0}")
        p.drawText(r.right() - 60, self.height() - 4, f"epoch {e1}")
        if lrs:
            p.setPen(QPen(QColor("#e17055"), 1, Qt.DashLine))
            pts = [xy(e, x, lr_lo, lr_hi) for e, _, x in self.points if x > 0]
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)
            p.drawText(r.right() + 2, r.top() + 10, f"{lr_hi:.1e}")
        p.setPen(QPen(QColor("#00b894"), 2))
        pts = [xy(e, l, lo, hi) for e, l, _ in self.points]
        for a, b in zip(pts, pts[1:]):
            p.drawLine(a, b)
        for pt in pts:
            p.drawEllipse(pt, 2.5, 2.5)
        p.setPen(QColor("#fdcb6e"))
        for e, act in self.marks.items():
            sym = "!" if "ROLLBACK" in act else ("↑" if "UP" in act else ("↓" if act.startswith("REDUCE") else ""))
            if sym:
                p.drawText(xy(e, hi, lo, hi) + QPointF(-3, 12), sym)


VERDICT_HELP = {
    "stuck": "persistently harder than the rest and not improving - check the caption and the image",
    "suspect": "unusually high loss - often a wrong or very different caption",
    "exhausted": "learned early, then stopped improving while still above average",
    "excluded": "set aside after two failed AI recaptions",
    "watch": "looked stuck this epoch; not confirmed yet",
    "learning": "harder than average but still improving",
    "easy": "easier than average",
    "mid": "average",
    "warmup": "look outlier on a learning-rate warm-up",
}


class ProblemImagesDialog(QDialog):
    """Fizgig's Problem Images window: per-image verdicts from loss_log/problem_images.json, refreshed every few
    seconds, with a caption editor. A fix is saved to the caption file AND queued for the running trainer, which
    re-encodes it at the next epoch boundary."""

    def __init__(self, run_dir: str, caption_ext: str, image_folder: str, parent=None, on_saved=None):
        super().__init__(parent)
        self.run_dir, self.ext, self.folder, self.on_saved = run_dir, caption_ext, image_folder, on_saved
        self.setWindowTitle("Problem Images")
        self.resize(1000, 640)
        lay = QVBoxLayout(self)
        lay.addWidget(hint_label("Images the loss watch flags while training. Stuck and suspect images usually have "
                                 "a wrong, missing or misleading caption. Edit it here and click Save fix: the "
                                 "caption file is updated and the run picks the new text up at the next epoch."))
        self.lbl_status = QLabel("Waiting for the first epoch report…")
        lay.addWidget(self.lbl_status)
        split = QSplitter()
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Image", "Verdict", "LR x", "Residual", "Fixes"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 200)
        self.table.itemSelectionChanged.connect(self._show)
        split.addWidget(self.table)
        right = QWidget()
        rl = QVBoxLayout(right)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(320, 320)
        self.lbl_verdict = QLabel()
        self.lbl_verdict.setWordWrap(True)
        self.edit = QPlainTextEdit()
        self.edit.setPlaceholderText("Select an image")
        self.btn_save = QPushButton("Save fix")
        self.btn_save.setToolTip("Write this caption to the caption file (with a backup) and queue it for the "
                                 "running trainer")
        self.btn_save.clicked.connect(self._save)
        for w in (self.preview, self.lbl_verdict, self.edit, self.btn_save):
            rl.addWidget(w)
        split.addWidget(right)
        split.setSizes([560, 420])
        lay.addWidget(split, 1)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(4000)
        self._mtime = None
        self.refresh()

    def _image_path(self, key):
        from training.dataset import IMAGE_EXTENSIONS
        for e in IMAGE_EXTENSIONS:
            p = os.path.join(self.folder, key + e)
            if os.path.exists(p):
                return p
        return None

    def refresh(self):
        p = Path(self.run_dir, "loss_log", "problem_images.json")
        try:
            m = p.stat().st_mtime
        except OSError:
            return
        if m == self._mtime:
            return
        self._mtime = m
        rep = pipeline.read_problem_images(self.run_dir)
        applied = pipeline.read_applied_captions(self.run_dir)
        imgs = rep.get("images", {})
        order = {"excluded": 0, "stuck": 1, "suspect": 2, "exhausted": 3, "watch": 4, "learning": 5, "warmup": 6,
                 "mid": 7, "easy": 8}
        rows = sorted(imgs.items(), key=lambda kv: (order.get(kv[1].get("verdict", "mid"), 9), kv[0]))
        sel = self._selected_key()
        self.table.setRowCount(len(rows))        # rows stay in problem-first order (no column sorting)
        for i, (k, s) in enumerate(rows):
            vals = [os.path.basename(k), s.get("verdict", ""), f"{s.get('multiplier', 1.0):.2f}",
                    f"{s.get('mean_residual', 0.0):+.4f}", str(len(applied.get(k, [])) or "")]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setData(Qt.UserRole, k)
                if s.get("verdict") in ("stuck", "excluded", "suspect"):
                    it.setForeground(QColor("#ff7675"))
                self.table.setItem(i, c, it)
            if k == sel:
                self.table.selectRow(i)
        n_bad = sum(1 for _, s in rows if s.get("verdict") in ("stuck", "suspect", "excluded"))
        extra = ""
        if rep.get("plateaued"):
            extra = f" · training has plateaued - best epoch about {rep.get('best_epoch_estimate')}"
        self.lbl_status.setText(f"Epoch {rep.get('epoch', '?')}: {len(rows)} image(s) tracked, {n_bad} flagged{extra}")

    def _selected_key(self):
        items = self.table.selectedItems()
        return items[0].data(Qt.UserRole) if items else None

    def _show(self):
        k = self._selected_key()
        if not k:
            return
        rep = pipeline.read_problem_images(self.run_dir).get("images", {}).get(k, {})
        v = rep.get("verdict", "")
        self.lbl_verdict.setText(f"<b>{v}</b>: {VERDICT_HELP.get(v, '')}")
        img = self._image_path(k)
        if img:
            pm = QPixmap(img)
            self.preview.setPixmap(pm.scaled(320, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            try:
                from core.caption_io import read_caption
                self.edit.setPlainText(read_caption(img, self.ext))
            except OSError:
                self.edit.setPlainText("")

    def _save(self):
        k = self._selected_key()
        text = " ".join(self.edit.toPlainText().split())
        img = self._image_path(k) if k else None
        if not k or not img or not text:
            return
        from core.caption_io import write_caption
        write_caption(img, text, self.ext)
        pipeline.queue_caption_updates(self.run_dir, {k: text})
        if self.on_saved:
            self.on_saved([img])
        QMessageBox.information(self, "Queued", "Saved. The run re-encodes this caption at the next epoch boundary.")


class SampleOverrideDialog(QDialog):
    """The live sample override: the next preview renders this prompt instead of the configured ones."""

    def __init__(self, run_dir: str, parent=None):
        super().__init__(parent)
        self.run_dir = run_dir
        self.setWindowTitle("Preview override")
        cur = pipeline.read_sample_override(run_dir) or {}
        f = QFormLayout(self)
        f.addRow(hint_label("While set, every preview round renders this prompt instead of the run's preview "
                            "prompts. It is encoded mid-run (the model may briefly move to system RAM)."))
        self.inp = QLineEdit(cur.get("prompt", ""))
        self.seed, self.w, self.h = QSpinBox(), QSpinBox(), QSpinBox()
        self.seed.setRange(0, 2 ** 31 - 1)
        self.seed.setValue(int(cur.get("seed", 1234)))
        for s, v in ((self.w, cur.get("width", 1024)), (self.h, cur.get("height", 1024))):
            s.setRange(64, 4096)
            s.setSingleStep(32)
            s.setValue(int(v))
        f.addRow("Prompt:", self.inp)
        f.addRow("Seed:", self.seed)
        row = QHBoxLayout()
        row.addWidget(self.w)
        row.addWidget(QLabel("×"))
        row.addWidget(self.h)
        f.addRow("Size:", row)
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.btn_clear = bb.addButton("Clear override", QDialogButtonBox.ResetRole)
        bb.accepted.connect(self._save)
        bb.rejected.connect(self.reject)
        self.btn_clear.clicked.connect(self._clear)
        f.addRow(bb)

    def _save(self):
        if self.inp.text().strip():
            pipeline.write_sample_override(self.run_dir, self.inp.text().strip(), self.seed.value(), self.w.value(),
                                           self.h.value())
        else:
            pipeline.clear_sample_override(self.run_dir)
        self.accept()

    def _clear(self):
        pipeline.clear_sample_override(self.run_dir)
        self.accept()
