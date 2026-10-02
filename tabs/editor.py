"""Image Editor: batch rotate / flip / resize / crop / convert with a live before-after preview.

Works on the shared workspace folder. Copies (default) go to
<Image Edits>/<source folder>/ with unique names and their captions; overwriting
originals asks first. Processing runs in the background and can be cancelled.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
                               QProgressBar, QPushButton, QRadioButton, QScrollArea, QSlider, QSpinBox, QSplitter,
                               QStackedWidget, QVBoxLayout, QWidget)

from core import fileops, image_ops, paths
from core.caption_io import caption_path
from core.config import settings
from core.image_ops import ASPECTS, FOCUS_POINTS, FORMATS, Operation
from core.image_utils import pil_to_qimage
from tabs.common import confirm, hint_label, run_in_background, show_job_summary
from tabs.workspace.browser import DatasetBrowser
from tabs.workspace.context import workspace
from tabs.workspace.jobs import start_file_job

log = logging.getLogger(__name__)


def edits_root() -> str:
    return str(settings().get("paths.edits_dir") or paths.DEFAULT_EDITS_DIR)


class EditorTab(QWidget):
    def __init__(self):
        super().__init__()
        self.ctx = workspace()
        self.job = None
        self.preview_op: Operation | None = None
        self._preview_key = ""

        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        splitter = QSplitter(Qt.Horizontal)
        self.browser = DatasetBrowser(self.ctx, size_key="ui.editor_thumbnail_size")
        self.browser.selection_changed.connect(self._on_selection)
        self.browser.current_changed.connect(lambda _k: self._schedule_preview())
        splitter.addWidget(self.browser)

        panel = QWidget()
        pl = QVBoxLayout(panel)

        # Preview
        g = QGroupBox("Preview")
        gl = QVBoxLayout(g)
        row = QHBoxLayout()
        self.lbl_before = self._preview_label("Before")
        self.lbl_after = self._preview_label("After")
        row.addWidget(self.lbl_before)
        row.addWidget(self.lbl_after)
        gl.addLayout(row)
        self.lbl_preview_info = hint_label("Select an image and adjust an operation to preview it.")
        gl.addWidget(self.lbl_preview_info)
        pl.addWidget(g)

        # Output
        g = QGroupBox("Output")
        gl = QVBoxLayout(g)
        self.rad_copies = QRadioButton("Save edited copies (originals untouched)")
        self.rad_copies.setChecked(True)
        self.rad_overwrite = QRadioButton("Overwrite originals (asks first)")
        self.rad_copies.setToolTip("Copies keep their caption files and never overwrite anything.")
        self.rad_overwrite.setToolTip("Replaces the original files. Color profile and EXIF data are kept.")
        out_row = QHBoxLayout()
        self.lbl_out = hint_label("")
        b_open = QPushButton("Open")
        b_open.setToolTip("Open the output folder")
        b_open.clicked.connect(self._open_output)
        out_row.addWidget(self.lbl_out, 1)
        out_row.addWidget(b_open)
        gl.addWidget(self.rad_copies)
        gl.addLayout(out_row)
        gl.addWidget(self.rad_overwrite)
        pl.addWidget(g)

        # Rotate / flip (instant actions)
        g = QGroupBox("Rotate / flip")
        from PySide6.QtWidgets import QGridLayout
        gl = QGridLayout(g)
        for label, tip, op in (("⟲ Left", "Rotate 90° left (Ctrl+Shift+R)", Operation("rotate", {"direction": "ccw"})),
                               ("⟳ Right", "Rotate 90° right (Ctrl+R)", Operation("rotate", {"direction": "cw"})),
                               ("180°", "Rotate 180°", Operation("rotate", {"direction": "180"})),
                               ("⇋ Flip H", "Mirror left-right", Operation("flip", {"axis": "h"})),
                               ("⇵ Flip V", "Mirror top-bottom", Operation("flip", {"axis": "v"}))):
            b = QPushButton(label)
            b.setToolTip(tip + " — applies to the selected images")
            b.clicked.connect(lambda _=False, o=op: self.run(o))
            n = gl.count()
            gl.addWidget(b, n // 3, n % 3)
        pl.addWidget(g)

        # Resize
        g = QGroupBox("Resize")
        f = QFormLayout(g)
        self.combo_resize = QComboBox()
        for label, val in (("Longest side", "longest"), ("Shortest side", "shortest"),
                           ("Exact size", "force"), ("Percent", "scale")):
            self.combo_resize.addItem(label, val)
        self.stack_resize = QStackedWidget()
        self.spin_side = self._spin(64, 8192, 1024, " px")
        self.spin_side2 = self._spin(64, 8192, 1024, " px")
        exact = QWidget()
        el = QHBoxLayout(exact)
        el.setContentsMargins(0, 0, 0, 0)
        self.spin_w = self._spin(16, 8192, 1024, " w")
        self.spin_h = self._spin(16, 8192, 1024, " h")
        el.addWidget(self.spin_w)
        el.addWidget(self.spin_h)
        self.spin_pct = self._spin(1, 800, 50, " %")
        for w in (self.spin_side, self.spin_side2, exact, self.spin_pct):
            self.stack_resize.addWidget(w)
        self.combo_resize.currentIndexChanged.connect(self.stack_resize.setCurrentIndex)
        self.chk_upscale = QCheckBox("Allow upscaling")
        self.chk_upscale.setToolTip("Off: images already smaller than the target are skipped, never enlarged.")
        b = QPushButton("Apply resize")
        b.clicked.connect(lambda: self.run(self._resize_op()))
        f.addRow("Mode:", self.combo_resize)
        f.addRow("Size:", self.stack_resize)
        f.addRow(self.chk_upscale)
        f.addRow(b)
        self._watch(self._resize_op, self.combo_resize, self.spin_side, self.spin_side2, self.spin_w, self.spin_h,
                    self.spin_pct, self.chk_upscale)
        pl.addWidget(g)

        # Crop
        g = QGroupBox("Crop")
        f = QFormLayout(g)
        self.combo_crop_mode = QComboBox()
        self.combo_crop_mode.addItem("To aspect ratio (keeps as much as possible)", "aspect")
        self.combo_crop_mode.addItem("To exact size", "exact")
        self.stack_crop = QStackedWidget()
        self.combo_aspect = QComboBox()
        self.combo_aspect.addItems(list(ASPECTS))
        self.combo_aspect.setToolTip("Training buckets: 1:1 for SD1.5/SDXL squares, 2:3 / 3:2 portraits & landscapes…")
        exact = QWidget()
        el = QHBoxLayout(exact)
        el.setContentsMargins(0, 0, 0, 0)
        self.spin_cw = self._spin(16, 8192, 1024, " w")
        self.spin_ch = self._spin(16, 8192, 1024, " h")
        el.addWidget(self.spin_cw)
        el.addWidget(self.spin_ch)
        self.stack_crop.addWidget(self.combo_aspect)
        self.stack_crop.addWidget(exact)
        self.combo_crop_mode.currentIndexChanged.connect(self.stack_crop.setCurrentIndex)
        self.combo_focus = QComboBox()
        self.combo_focus.addItems(list(FOCUS_POINTS))
        self.combo_focus.setToolTip("Which part of the image to keep")
        b = QPushButton("Apply crop")
        b.clicked.connect(lambda: self.run(self._crop_op()))
        f.addRow("Mode:", self.combo_crop_mode)
        f.addRow("Target:", self.stack_crop)
        f.addRow("Keep:", self.combo_focus)
        f.addRow(b)
        self._watch(self._crop_op, self.combo_crop_mode, self.combo_aspect, self.spin_cw, self.spin_ch, self.combo_focus)
        pl.addWidget(g)

        # Convert
        g = QGroupBox("Convert format")
        f = QFormLayout(g)
        self.combo_format = QComboBox()
        self.combo_format.addItems(list(FORMATS))
        self.slider_quality = QSlider(Qt.Horizontal)
        self.slider_quality.setRange(50, 100)
        self.slider_quality.setValue(92)
        self.lbl_quality = QLabel("92")
        self.slider_quality.valueChanged.connect(lambda v: self.lbl_quality.setText(str(v)))
        qrow = QHBoxLayout()
        qrow.addWidget(self.slider_quality, 1)
        qrow.addWidget(self.lbl_quality)
        b = QPushButton("Apply conversion")
        b.clicked.connect(lambda: self.run(self._convert_op()))
        f.addRow("Format:", self.combo_format)
        f.addRow("Quality:", qrow)
        f.addRow(hint_label("Quality applies to JPG/WEBP. Converted files sit next to the originals when "
                            "overwriting is chosen — originals are never deleted."))
        f.addRow(b)
        pl.addWidget(g)

        # Progress
        prow = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(lambda: self.job and self.job.cancel())
        prow.addWidget(self.progress, 1)
        prow.addWidget(self.btn_cancel)
        pl.addLayout(prow)
        self.lbl_status = hint_label("")
        pl.addWidget(self.lbl_status)
        pl.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(400)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1100, 420])
        root.addWidget(splitter)

        self._preview_timer = QTimer(self, singleShot=True, interval=300)
        self._preview_timer.timeout.connect(self._render_preview)
        self.ctx.session_changed.connect(self._update_output_label)
        self._update_output_label()
        for seq, op in (("Ctrl+R", Operation("rotate", {"direction": "cw"})),
                        ("Ctrl+Shift+R", Operation("rotate", {"direction": "ccw"}))):
            s = QShortcut(QKeySequence(seq), self)
            s.setContext(Qt.WidgetWithChildrenShortcut)
            s.activated.connect(lambda o=op: self.run(o))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _spin(lo, hi, val, suffix):
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setSuffix(suffix)
        return s

    @staticmethod
    def _preview_label(text):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setMinimumSize(120, 150)
        from PySide6.QtWidgets import QSizePolicy
        lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        lbl.setFixedHeight(170)
        lbl.setStyleSheet("background-color: #1a1a1a; border-radius: 6px; color: #777;")
        return lbl

    def _watch(self, op_factory, *widgets):
        """Changing any control in a group previews that group's operation."""
        def touched(*_):
            self.preview_op = op_factory()
            self._schedule_preview()
        for w in widgets:
            for sig in ("valueChanged", "currentIndexChanged", "toggled"):
                if hasattr(w, sig):
                    getattr(w, sig).connect(touched)
                    break

    def _resize_op(self) -> Operation:
        mode = self.combo_resize.currentData()
        p = {"mode": mode, "allow_upscale": self.chk_upscale.isChecked()}
        if mode == "longest":
            p["size"] = self.spin_side.value()
        elif mode == "shortest":
            p["size"] = self.spin_side2.value()
        elif mode == "force":
            p.update(w=self.spin_w.value(), h=self.spin_h.value())
        else:
            p["percent"] = self.spin_pct.value()
        return Operation("resize", p)

    def _crop_op(self) -> Operation:
        focus = self.combo_focus.currentText()
        if self.combo_crop_mode.currentData() == "aspect":
            return Operation("crop_aspect", {"aspect": self.combo_aspect.currentText(), "focus": focus})
        return Operation("crop", {"w": self.spin_cw.value(), "h": self.spin_ch.value(), "focus": focus})

    def _convert_op(self) -> Operation:
        return Operation("convert", {"format": self.combo_format.currentText(), "quality": self.slider_quality.value()})

    def _output_dir(self) -> Path:
        folder = os.path.basename(os.path.normpath(self.ctx.folder)) if self.ctx.folder else "edits"
        return Path(edits_root()) / folder

    def _update_output_label(self):
        self.lbl_out.setText(f"→ {self._output_dir()}")

    def _open_output(self):
        d = self._output_dir()
        d.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))

    def _on_selection(self, keys):
        n = len(keys)
        self.lbl_status.setText(f"{n} image(s) selected" if n else "Select images to edit.")
        self._schedule_preview()

    # ------------------------------------------------------------------ preview
    def _schedule_preview(self):
        self._preview_timer.start()

    def _render_preview(self):
        keys = self.browser.selected_keys()
        cur = self.browser.view.currentIndex()
        from tabs.workspace.model import KEY_ROLE
        key = cur.data(KEY_ROLE) if cur.isValid() and cur.data(KEY_ROLE) in keys else (keys[0] if keys else "")
        if not key:
            self.lbl_before.setPixmap(QPixmap())
            self.lbl_before.setText("Before")
            self.lbl_after.setPixmap(QPixmap())
            self.lbl_after.setText("After")
            return
        op = self.preview_op
        self._preview_key = key

        def work():
            img = image_ops.open_upright(key)
            before = img.copy()
            before.thumbnail((340, 340))
            after, size, note = None, None, ""
            if op is not None:
                try:
                    after, size = image_ops.preview(img, op, 340)
                except image_ops.SkipImage as e:
                    note = f"Skipped for this image: {e}"
            return pil_to_qimage(before), img.size, (pil_to_qimage(after) if after else None), size, note

        def done(res):
            if self._preview_key != key:
                return
            qb, osize, qa, nsize, note = res
            self._set_pix(self.lbl_before, qb)
            if qa is not None:
                self._set_pix(self.lbl_after, qa)
                self.lbl_preview_info.setText(f"{op.describe()}:  {osize[0]}×{osize[1]}  →  {nsize[0]}×{nsize[1]}")
            else:
                self.lbl_after.setPixmap(QPixmap())
                self.lbl_after.setText("After")
                self.lbl_preview_info.setText(note or f"{osize[0]}×{osize[1]} — adjust Resize, Crop or Convert "
                                                      "to preview the result.")

        run_in_background(work, done, lambda e: self.lbl_preview_info.setText(f"Can't preview: {e}"))

    @staticmethod
    def _set_pix(label: QLabel, qimg):
        pm = QPixmap.fromImage(qimg)
        label.setPixmap(pm.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # ------------------------------------------------------------------ run
    def run(self, op: Operation):
        if self.job is not None:
            QMessageBox.information(self, "Busy", "An edit is already running — cancel it or wait.")
            return
        keys = self.browser.selected_keys()
        if not keys:
            self.lbl_status.setText("⚠ Select images first.")
            return
        overwrite = self.rad_overwrite.isChecked()
        convert = op.kind == "convert"
        if overwrite and not convert and not confirm(
                self, "Overwrite originals?",
                f"{op.describe()} will permanently change {len(keys)} original image(s). This can't be undone.\n\n"
                "Tip: choose 'Save edited copies' to keep originals untouched.", destructive=True):
            return
        out_dir = self._output_dir()
        fmt = op.params.get("format") if convert else None
        quality = op.params.get("quality", 92)

        def process(src: str) -> Path:
            srcp = Path(src)
            img = image_ops.open_upright(srcp)
            result = image_ops.apply(img, op)
            if overwrite:
                dest = srcp
                if fmt and FORMATS[fmt][1] != srcp.suffix.lower():
                    dest = fileops.unique_path(srcp.with_suffix(FORMATS[fmt][1]))
            else:
                ext = FORMATS[fmt][1] if fmt else srcp.suffix
                dest = fileops.unique_path(out_dir / f"{srcp.stem}{ext}")
            saved = image_ops.save(result, dest, fmt, quality)
            if saved != srcp:  # derived file: keep the caption paired with it
                cap, dcap = caption_path(srcp), caption_path(saved)
                if cap.is_file() and not dcap.exists():
                    shutil.copy2(cap, dcap)
            return saved

        self.progress.setMaximum(len(keys))
        self.progress.setValue(0)
        self.btn_cancel.setEnabled(True)
        self.lbl_status.setText(f"{op.describe()} — {len(keys)} image(s)…")
        self._last_overwrite = overwrite and not convert
        self._created_in_place = overwrite and convert
        self._modified: list[str] = []
        self._skipped: list[tuple[str, str]] = []
        self.job = start_file_job(keys, process, op.describe())
        self.job.item_skipped.connect(lambda src, why: self._skipped.append((src, why)))
        self.job.progress.connect(lambda d, t: self.progress.setValue(d))
        self.job.item_done.connect(lambda src, dest: self._modified.append(src) if dest == src else None)
        self.job.finished.connect(self._finished)

    def _finished(self, summary):
        self.job = None
        self.btn_cancel.setEnabled(False)
        if self._last_overwrite and self._modified:
            self.ctx.images_modified(self._modified)
        where = "" if self._last_overwrite or self._created_in_place else f" → {self._output_dir()}"
        parts = [f"{summary.done} done"]
        if summary.skipped:
            parts.append(f"{summary.skipped} skipped")
        if summary.failed:
            parts.append(f"{len(summary.failed)} failed")
        self.lbl_status.setText(f"{summary.title}: " + ", ".join(parts) + where +
                                (" (cancelled)" if summary.cancelled else ""))
        self._schedule_preview()
        show_job_summary(self, summary)
        if self._skipped:
            lines = "\n".join(f"• {Path(s).name}: {why}" for s, why in self._skipped[:15])
            more = f"\n…and {len(self._skipped) - 15} more" if len(self._skipped) > 15 else ""
            QMessageBox.information(self, "Some images were skipped",
                                    f"{len(self._skipped)} image(s) were left unchanged:\n\n{lines}{more}")
        if self._created_in_place and summary.done:
            if confirm(self, "Converted files created",
                       f"{summary.done} converted file(s) were saved next to the originals. Reload the folder to "
                       "show them?"):
                self.ctx.rescan(self)
