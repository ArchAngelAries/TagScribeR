"""Export for training: options dialog + background job."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QLineEdit, QMessageBox, QProgressBar, QPushButton, QSpinBox, QVBoxLayout)

from core.config import settings
from core.export import ExportOptions, export_one, summarize
from core.image_ops import FORMATS
from tabs.common import hint_label, show_job_summary
from tabs.workspace.jobs import start_file_job


class ExportDialog(QDialog):
    def __init__(self, keys: list[str], subject: str = "", parent=None):
        super().__init__(parent)
        self.keys = keys
        self.cfg = settings()
        self.job = None
        self.setWindowTitle(f"Export {len(keys)} image(s) for training")
        self.resize(560, 600)
        lay = QVBoxLayout(self)
        lay.addWidget(hint_label("Creates a training-ready copy: resized, cleaned and captioned. Your originals are "
                                 "never changed and nothing in the output folder is overwritten."))

        g = QGroupBox("Destination")
        f = QFormLayout(g)
        row = QHBoxLayout()
        self.inp_out = QLineEdit(self.cfg.get("export.out_dir", ""))
        self.inp_out.setPlaceholderText("Choose an output folder")
        b = QPushButton("Browse…")
        b.clicked.connect(self._browse)
        row.addWidget(self.inp_out, 1)
        row.addWidget(b)
        f.addRow("Folder:", row)
        self.chk_kohya = QCheckBox("kohya-style subfolder  <repeats>_<concept>")
        self.chk_kohya.setChecked(self.cfg.get("export.kohya", True))
        self.chk_kohya.setToolTip("e.g. 10_ohwx — the folder name tells kohya/sd-scripts (and compatible trainers) "
                                  "how many times to repeat these images per epoch")
        self.spin_repeats = QSpinBox()
        self.spin_repeats.setRange(1, 1000)
        self.spin_repeats.setValue(self.cfg.get("export.repeats", 10))
        self.inp_concept = QLineEdit(subject.split(",")[0].strip().replace(" ", "_") or self.cfg.get("export.concept", "dataset"))
        f.addRow(self.chk_kohya)
        f.addRow("Repeats:", self.spin_repeats)
        f.addRow("Concept:", self.inp_concept)
        lay.addWidget(g)

        g = QGroupBox("Images")
        f = QFormLayout(g)
        self.combo_resize = QComboBox()
        for label, val in (("Fit training buckets (scale + center crop)", "bucket"),
                           ("Limit longest side (keep shape)", "longest"), ("Keep original size", "none")):
            self.combo_resize.addItem(label, val)
        self.combo_resize.setCurrentIndex(max(0, self.combo_resize.findData(self.cfg.get("export.resize", "bucket"))))
        self.spin_res = QSpinBox()
        self.spin_res.setRange(256, 4096)
        self.spin_res.setSingleStep(64)
        self.spin_res.setSuffix(" px")
        self.spin_res.setValue(self.cfg.get("export.resolution", self.cfg.get("health.resolution", 1024)))
        self.chk_upscale = QCheckBox("Allow upscaling small images (otherwise they're skipped)")
        self.combo_fmt = QComboBox()
        self.combo_fmt.addItem("Keep format", "")
        for k in FORMATS:
            self.combo_fmt.addItem(k, k)
        self.combo_fmt.setCurrentIndex(max(0, self.combo_fmt.findData(self.cfg.get("export.format", ""))))
        self.chk_strip = QCheckBox("Strip metadata (keeps colour profile)")
        self.chk_strip.setChecked(self.cfg.get("export.strip", True))
        self.chk_rename = QCheckBox("Rename sequentially")
        self.inp_prefix = QLineEdit("img")
        self.inp_prefix.setMaximumWidth(140)
        rrow = QHBoxLayout()
        rrow.addWidget(self.chk_rename)
        rrow.addWidget(self.inp_prefix)
        rrow.addStretch()
        f.addRow("Resize:", self.combo_resize)
        f.addRow("Resolution:", self.spin_res)
        f.addRow(self.chk_upscale)
        f.addRow("Format:", self.combo_fmt)
        f.addRow(self.chk_strip)
        f.addRow(rrow)
        lay.addWidget(g)

        g = QGroupBox("Captions")
        f = QFormLayout(g)
        self.inp_trigger = QLineEdit(subject)
        self.inp_trigger.setPlaceholderText("optional trigger word added to the start of every caption")
        self.combo_ext = QComboBox()
        self.combo_ext.addItems([".txt", ".caption"])
        self.chk_skip = QCheckBox("Skip images without a caption")
        self.chk_skip.setChecked(True)
        f.addRow("Trigger:", self.inp_trigger)
        f.addRow("Caption file:", self.combo_ext)
        f.addRow(self.chk_skip)
        lay.addWidget(g)

        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m")
        self.progress.setVisible(False)
        lay.addWidget(self.progress)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.btn_export = self.buttons.addButton("Export", QDialogButtonBox.AcceptRole)
        self.btn_export.clicked.connect(self._start)
        self.buttons.rejected.connect(self._cancel)
        lay.addWidget(self.buttons)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Export to folder", self.inp_out.text())
        if d:
            self.inp_out.setText(d)

    def options(self) -> ExportOptions:
        return ExportOptions(
            out_dir=self.inp_out.text().strip(), kohya_folder=self.chk_kohya.isChecked(),
            repeats=self.spin_repeats.value(), concept=self.inp_concept.text().strip() or "dataset",
            resize=self.combo_resize.currentData(), resolution=self.spin_res.value(),
            allow_upscale=self.chk_upscale.isChecked(), fmt=self.combo_fmt.currentData(),
            strip_metadata=self.chk_strip.isChecked(), rename=self.chk_rename.isChecked(),
            name_prefix=self.inp_prefix.text().strip() or "img", caption_ext=self.combo_ext.currentText(),
            trigger=self.inp_trigger.text().strip(), skip_uncaptioned=self.chk_skip.isChecked())

    def _start(self):
        opts = self.options()
        if not opts.out_dir:
            QMessageBox.information(self, "Export", "Choose an output folder.")
            return
        src_dirs = {str(Path(k).parent.resolve()).lower() for k in self.keys}
        if str(opts.target_dir().resolve()).lower() in src_dirs:
            QMessageBox.warning(self, "Export", "Choose an output folder different from the source folder.")
            return
        self.cfg.update({"export.out_dir": opts.out_dir, "export.kohya": opts.kohya_folder,
                         "export.repeats": opts.repeats, "export.concept": opts.concept, "export.resize": opts.resize,
                         "export.resolution": opts.resolution, "export.format": opts.fmt,
                         "export.strip": opts.strip_metadata})
        index = {k: i + 1 for i, k in enumerate(self.keys)}
        self.btn_export.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setMaximum(len(self.keys))
        self._skipped: list[tuple[str, str]] = []
        self.job = start_file_job(self.keys, lambda k: export_one(k, index[k], opts), "Export for training")
        self.job.progress.connect(lambda d, t: self.progress.setValue(d))
        self.job.item_skipped.connect(lambda k, why: self._skipped.append((k, why)))
        self.job.finished.connect(lambda s: self._done(s, opts))

    def _done(self, summary, opts: ExportOptions):
        self.job = None
        show_job_summary(self, summary)
        msg = summarize(opts, summary.done)
        if self._skipped:
            reasons: dict[str, int] = {}
            for _k, why in self._skipped:
                reasons[why.split(" (")[0]] = reasons.get(why.split(" (")[0], 0) + 1
            msg += "\n\nSkipped: " + "; ".join(f"{n} × {r}" for r, n in reasons.items())
        box = QMessageBox(QMessageBox.Information, "Export finished", msg, QMessageBox.Ok, self)
        open_btn = box.addButton("Open folder", QMessageBox.ActionRole)
        box.exec()
        if box.clickedButton() is open_btn:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(opts.target_dir())))
        self.accept()

    def _cancel(self):
        if self.job is not None:
            self.job.cancel()
        else:
            self.reject()
