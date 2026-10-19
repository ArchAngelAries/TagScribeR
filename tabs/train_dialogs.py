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
    """The run's loss, drawn with QPainter: the loss of every step (faint) and its smoothed line when the step log is
    there (`set_steps`, from pipeline.read_step_log), the epoch average (green, dots), the learning rate (dashed, its
    own scale) and a faint line at each epoch start. Without a step log (an older run, a read-only folder) the chart is
    the epoch averages alone, as it always was."""

    SMOOTH = 0.9                                             # the smoothed line's moving-average weight

    def __init__(self, parent=None):
        super().__init__(parent)
        self.points: list[tuple[int, float, float]] = []     # (epoch, loss, lr) - the epoch averages
        self.marks: dict[int, str] = {}                      # epoch -> adaptive action
        self.steps: dict | None = None                       # pipeline.read_step_log's data
        self._hover: int | None = None                       # index into steps["points"] under the mouse
        self.setMinimumHeight(150)
        self.setMouseTracking(True)
        self.setToolTip("Training loss. Faint line: the loss of every step; bright line: the same smoothed; green "
                        "dots: the average of each epoch; dashed orange: the learning rate (its own scale); grey "
                        "lines: where each epoch starts. Adaptive LR decisions are marked: up = probe, down = "
                        "reduce, ! = rollback. Point at the graph to read a step. Loss going down is not the whole "
                        "story - judge by the sample previews.")

    def clear(self):
        self.points, self.marks, self.steps, self._hover = [], {}, None, None
        self.update()

    def add(self, epoch, loss, lr):
        self.points = [p for p in self.points if p[0] != epoch] + [(epoch, loss, lr)]
        self.points.sort()
        self.update()

    def mark(self, epoch, action):
        self.marks[epoch] = action
        self.update()

    def set_steps(self, data):
        """The per-step curve (pipeline.read_step_log's dict) or None. Redraws only when it changed."""
        if data is not None and not data.get("points"):
            data = None
        if data == self.steps:
            return
        self.steps = data
        self._hover = None
        self.update()

    # ---- geometry ---------------------------------------------------------------------------------
    def _plot_rect(self):
        return self.rect().adjusted(40, 10, -40, -22)

    def _epoch_span(self):
        """epoch -> (first step, last step) from the step log."""
        eps = (self.steps or {}).get("epochs") or []
        pts = (self.steps or {}).get("points") or []
        last = pts[-1][0] if pts else 0
        return {e: (first, eps[i + 1][1] - 1 if i + 1 < len(eps) else last) for i, (e, first) in enumerate(eps)}

    def _smoothed(self):
        out, a = [], None
        for pt in self.steps["points"]:
            a = pt[2] if a is None else self.SMOOTH * a + (1 - self.SMOOTH) * pt[2]
            out.append(a)
        return out

    def _loss_range(self, extra):
        vals = list(extra)
        if self.steps:
            raw = sorted(pt[2] for pt in self.steps["points"])
            if len(raw) > 20:                                # one spike does not flatten the rest
                cut = int(len(raw) * 0.01)
                raw = raw[cut:len(raw) - cut]
            vals += raw + self._smoothed()
        lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
        if hi - lo < 1e-9:
            lo, hi = lo - 0.01, hi + 0.01
        return lo, hi

    def mouseMoveEvent(self, e):
        if not self.steps:
            return
        r = self._plot_rect()
        pts = self.steps["points"]
        x0, x1 = pts[0][0], max(pts[-1][0], pts[0][0] + 1)
        step = x0 + (e.position().x() - r.left()) / max(1, r.width()) * (x1 - x0)
        best = min(range(len(pts)), key=lambda i: abs(pts[i][0] - step))
        if best != self._hover:
            self._hover = best
            self.update()

    def leaveEvent(self, _e):
        if self._hover is not None:
            self._hover = None
            self.update()

    def readout(self, i: int) -> str:
        """The text for step point i: step, epoch, loss (and smoothed), learning rate when logged."""
        s, ep, loss, lr = self.steps["points"][i]
        return (f"step {s} · epoch {ep} · loss {loss:.4f} (smoothed {self._smoothed()[i]:.4f})"
                + (f" · LR {lr:.2e}" if lr else ""))

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self._plot_rect()
        p.fillRect(self.rect(), QColor("#1e1e1e"))
        p.setPen(QColor("#555"))
        p.drawRect(r)
        steps = self.steps["points"] if self.steps else []
        if not self.points and not steps:
            p.setPen(QColor("#888"))
            p.drawText(r, Qt.AlignCenter, "The loss chart fills in as the run trains.")
            return
        span = self._epoch_span() if steps else {}
        if steps:                                            # x = steps; an epoch's average sits mid-epoch
            x0, x1 = steps[0][0], max(steps[-1][0], steps[0][0] + 1)
            avg = [(sum(span[e]) / 2, l, lr) for e, l, lr in self.points if e in span]
        else:                                                # x = epochs
            eps = [e for e, _, _ in self.points]
            x0, x1 = min(eps), max(max(eps), min(eps) + 1)
            avg = [(e, l, lr) for e, l, lr in self.points]
        lo, hi = self._loss_range([l for _, l, _ in avg])
        lr_pts = [(pt[0], pt[3]) for pt in steps if pt[3]]
        if len(lr_pts) < 2:
            lr_pts = [(x, lr) for x, _, lr in avg if lr > 0]
        lrs = [v for _, v in lr_pts]
        lr_lo, lr_hi = (min(lrs), max(lrs)) if lrs else (0, 1)
        if lr_hi - lr_lo < 1e-12:
            lr_lo, lr_hi = lr_lo * 0.9, lr_hi * 1.1 + 1e-12

        def xy(x, v, a, b):
            v = min(max(v, a), b)
            return QPointF(r.left() + (x - x0) / (x1 - x0) * r.width(), r.bottom() - (v - a) / (b - a) * r.height())
        if steps:                                            # faint epoch starts, labels thinned to fit
            every = max(1, -(-len(span) // 10))
            for i, (e, (first, _last)) in enumerate(sorted(span.items())):
                x = xy(first, lo, lo, hi).x()
                p.setPen(QPen(QColor("#333"), 1))
                p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
                if i % every == 0:
                    p.setPen(QColor("#777"))
                    p.drawText(QPointF(x + 2, r.bottom() - 3), f"e{e}")
        p.setPen(QColor("#aaa"))
        p.drawText(2, r.top() + 10, f"{hi:.4f}")
        p.drawText(2, r.bottom(), f"{lo:.4f}")
        p.drawText(r.left(), self.height() - 4, f"step {x0:g}" if steps else f"epoch {x0:g}")
        right = f"step {x1:g}" if steps else f"epoch {x1:g}"
        p.drawText(r.right() - 10 - p.fontMetrics().horizontalAdvance(right), self.height() - 4, right)
        if lrs:
            p.setPen(QPen(QColor("#e17055"), 1, Qt.DashLine))
            pts = [xy(x, v, lr_lo, lr_hi) for x, v in lr_pts]
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)
            p.drawText(r.right() + 2, r.top() + 10, f"{lr_hi:.1e}")
        if steps:
            p.setPen(QPen(QColor(0, 184, 148, 70), 1))
            raw = [xy(pt[0], pt[2], lo, hi) for pt in steps]
            for a, b in zip(raw, raw[1:]):
                p.drawLine(a, b)
            p.setPen(QPen(QColor("#81ecec"), 2))
            sm = [xy(pt[0], v, lo, hi) for pt, v in zip(steps, self._smoothed())]
            for a, b in zip(sm, sm[1:]):
                p.drawLine(a, b)
        p.setPen(QPen(QColor("#00b894"), 2))
        pts = [xy(x, l, lo, hi) for x, l, _ in avg]
        if not steps:
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)
        for pt in pts:
            p.drawEllipse(pt, 2.5, 2.5)
        p.setPen(QColor("#fdcb6e"))
        for e, act in self.marks.items():
            sym = "!" if "ROLLBACK" in act else ("↑" if "UP" in act else ("↓" if act.startswith("REDUCE") else ""))
            ex = (span[e][0] if e in span else None) if steps else e
            if sym and ex is not None:
                p.drawText(xy(ex, hi, lo, hi) + QPointF(-3, 12), sym)
        note = Qt.AlignTop | Qt.AlignHCenter
        if steps and self._hover is not None and self._hover < len(steps):
            x = xy(steps[self._hover][0], lo, lo, hi).x()
            p.setPen(QPen(QColor("#dfe6e9"), 1, Qt.DotLine))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.setPen(QColor("#dfe6e9"))
            p.drawText(r.adjusted(6, 2, -6, 0), note, self.readout(self._hover))
        elif steps:
            p.setPen(QColor("#888"))
            stride = self.steps.get("stride", 1)
            p.drawText(r.adjusted(6, 2, -6, 0), note,
                       f"{self.steps['steps']} steps" + (f" · every {stride}th drawn" if stride > 1 else ""))


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
        self.lbl_verdict.setText(f"<b>{v}</b>: {VERDICT_HELP.get(v, '')}" + (
            "<br>Excluded by an earlier run: training normally; if it gets stuck again it is excluded at once."
            if rep.get("earlier_exclusion") else ""))
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


class SliderPromptsDialog(QDialog):
    """A prompt-pair slider's live prompts (Fizgig 8.0.3): what the picture is, what each end adds and the push
    strength, edited while the run trains. The trainer takes them at the next epoch boundary and re-renders the
    practice pictures when the base prompt changed; the dialog shows what it last applied."""

    def __init__(self, run_dir: str, current: dict, parent=None, is_live=None):
        super().__init__(parent)
        self.run_dir = run_dir
        self.is_live = is_live            # () -> whether a prompt-slider run is still going
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Slider prompts")
        self.setMinimumWidth(520)
        applied = pipeline.read_applied_slider_prompts(run_dir) or {}
        cur = {**current, **{k: applied[k] for k in ("base", "pos", "neg", "guidance") if k in applied}}
        f = QFormLayout(self)
        f.addRow(hint_label("Change what the slider trains on while it runs. The new prompts are taken at the end of "
                            "the epoch in progress. Changing the base prompt makes the model render its practice "
                            "pictures again (a short pause); the other fields change straight away."))
        self.base = QLineEdit(str(cur.get("base", "")))
        self.pos = QLineEdit(str(cur.get("pos", "")))
        self.neg = QLineEdit(str(cur.get("neg", "")))
        self.guidance = QLineEdit(f"{float(cur.get('guidance') or 0):g}")
        self.base.setToolTip("What the picture is, e.g. 'a photo of a woman'. Previews show the dial on this prompt.")
        self.pos.setToolTip("What the +1 end adds to the base prompt, e.g. 'smiling'.")
        self.neg.setToolTip("What the -1 end adds to the base prompt, e.g. 'frowning'.")
        self.guidance.setToolTip("How hard each end is pushed away from the base. Larger = a stronger slider that "
                                 "overcooks sooner. Must be above 0.")
        f.addRow("Base prompt:", self.base)
        f.addRow("+1 end adds:", self.pos)
        f.addRow("-1 end adds:", self.neg)
        f.addRow("Push strength:", self.guidance)
        self.lbl_state = QLabel()
        self.lbl_state.setWordWrap(True)
        f.addRow(self.lbl_state)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        self.btn_apply = bb.addButton("Apply at next epoch", QDialogButtonBox.AcceptRole)
        self.btn_apply.setToolTip("Hand these prompts to the running trainer (taken at the end of this epoch)")
        self.btn_apply.clicked.connect(self.apply)
        bb.rejected.connect(self.reject)
        f.addRow(bb)
        self._sent = None
        self._start = {"base": str(current.get("base", "")).strip(), "pos": str(current.get("pos", "")).strip(),
                       "neg": str(current.get("neg", "")).strip(), "guidance": float(current.get("guidance") or 0)}
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_state)
        self.timer.start(3000)
        self.refresh_state()

    def done(self, result):
        self.timer.stop()
        super().done(result)

    def closeEvent(self, e):
        self.timer.stop()
        super().closeEvent(e)

    def prompts(self):
        """The four fields as the trainer wants them, or None when one is missing (the trainer ignores a partial set)."""
        try:
            g = float(self.guidance.text().strip())
        except ValueError:
            g = 0.0
        out = {"base": self.base.text().strip(), "pos": self.pos.text().strip(), "neg": self.neg.text().strip(),
               "guidance": g}
        return out if out["base"] and out["pos"] and out["neg"] and g > 0 else None

    def _in_effect(self):
        """The prompts the trainer is using as far as we know: what it last applied, else what the run started with."""
        a = pipeline.read_applied_slider_prompts(self.run_dir)
        return ({k: a[k] for k in ("base", "pos", "neg", "guidance")} if a and all(
            k in a for k in ("base", "pos", "neg", "guidance")) else self._start)

    def apply(self):
        if self.is_live is not None and not self.is_live():
            QMessageBox.information(self, "Slider prompts", "No prompt slider is training right now, so there is "
                                    "nothing to hand these prompts to.")
            return False
        p = self.prompts()
        if p is None:
            QMessageBox.information(self, "Slider prompts", "All four fields are needed: the base prompt, what each "
                                    "end adds, and a push strength above 0.")
            return False
        if p != self._in_effect():
            pipeline.write_slider_prompts(self.run_dir, p)
            self._sent = p
        self.refresh_state()
        return True

    def refresh_state(self):
        a = pipeline.read_applied_slider_prompts(self.run_dir)
        if self._sent and self._sent != self._in_effect():
            self.lbl_state.setText("Waiting for the end of this epoch...")
        elif a:
            self.lbl_state.setText(
                f"Training on these prompts since epoch {int(a.get('epoch', 0)) + 1}"
                + (" (the practice pictures were rendered again)." if a.get("practice_rerendered") else "."))
        else:
            self.lbl_state.setText("Training on the prompts the run started with.")
