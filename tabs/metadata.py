"""Metadata: inspect, audit, clean, sign and reuse embedded prompts.

* Inspect — everything stored in the selected image, a privacy summary, and
  editing of text fields (PNG text / EXIF artist, copyright…).
* Privacy & cleanup — audit a whole selection, then strip location, device
  IDs, AI generation data, XMP history and more. Lossless: pixels untouched.
* Authorship — apply your own artist / copyright / description / software
  from reusable templates.
* Prompts → captions — reuse A1111/ComfyUI/NovelAI/InvokeAI prompts as captions.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QMessageBox, QPushButton, QRadioButton, QScrollArea, QSplitter, QTableWidget,
                               QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from core import metadata as md
from core.presets import PresetError, metadata_presets
from tabs.common import confirm, hint_label, run_in_background, show_job_summary
from tabs.workspace.browser import DatasetBrowser
from tabs.workspace.context import workspace
from tabs.workspace.jobs import start_file_job

LEVEL_COLORS = {"high": "#ff7675", "medium": "#fdcb6e", "low": "#9a9a9a"}


def _scroll(w: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    sa.setWidget(w)
    return sa


class MetadataTab(QWidget):
    def __init__(self):
        super().__init__()
        self.ctx = workspace()
        self.job = None
        self.current_key = ""
        self.report: md.MetaReport | None = None

        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        splitter = QSplitter(Qt.Horizontal)
        self.browser = DatasetBrowser(self.ctx, size_key="ui.metadata_thumbnail_size")
        self.browser.current_changed.connect(self.load_metadata)
        self.browser.selection_changed.connect(self._on_selection)
        splitter.addWidget(self.browser)

        self.side = QTabWidget()
        self.side.setMinimumWidth(420)
        self.side.addTab(self._build_inspect(), "Inspect")
        self.side.addTab(_scroll(self._build_clean()), "Privacy")
        self.side.addTab(_scroll(self._build_author()), "Authorship")
        self.side.addTab(_scroll(self._build_prompts()), "Prompts")
        for i, tip in enumerate(("Everything stored in the selected image",
                                 "Audit and remove location, device IDs, AI data and more (lossless)",
                                 "Write your artist / copyright / license to images",
                                 "Use embedded generation prompts as captions")):
            self.side.setTabToolTip(i, tip)
        from tabs.help import help_button
        self.side.setCornerWidget(help_button("metadata", self.side, "How the metadata tools work"), Qt.TopRightCorner)
        splitter.addWidget(self.side)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([1000, 520])
        root.addWidget(splitter)

    # ------------------------------------------------------------------ build
    def _scope_row(self, layout) -> tuple[QRadioButton, QRadioButton]:
        row = QHBoxLayout()
        row.addWidget(QLabel("Apply to:"))
        sel, shown = QRadioButton("Selected"), QRadioButton("All shown")
        sel.setChecked(True)
        row.addWidget(sel)
        row.addWidget(shown)
        row.addStretch()
        layout.addLayout(row)
        return sel, shown

    def _build_inspect(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.lbl_file = QLabel("Select an image.")
        self.lbl_file.setWordWrap(True)
        lay.addWidget(self.lbl_file)
        self.list_findings = QListWidget()
        self.list_findings.setMaximumHeight(150)
        self.list_findings.setWordWrap(True)
        self.list_findings.setToolTip("What this file reveals. Red = sensitive, yellow = worth knowing.")
        lay.addWidget(self.list_findings)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Field", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setWordWrap(True)
        self.table.itemChanged.connect(lambda _i: self.btn_save.setEnabled(True))
        lay.addWidget(self.table, 1)
        lay.addWidget(hint_label("Double-click a white value (text fields, artist, copyright…) to edit it."))
        row = QHBoxLayout()
        self.btn_save = QPushButton("💾 Save field edits")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_edits)
        self.btn_use_prompt = QPushButton("Prompt → caption")
        self.btn_use_prompt.setToolTip("Copies the generation prompt into this image's caption (unsaved, undoable)")
        self.btn_use_prompt.clicked.connect(lambda: self.prompts_to_captions([self.current_key], replace=True))
        row.addWidget(self.btn_save)
        row.addWidget(self.btn_use_prompt)
        lay.addLayout(row)
        return w

    def _build_clean(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(hint_label("Check what your images reveal before sharing them, then remove it. Cleaning is "
                                 "lossless (pixels are never re-encoded), keeps photo orientation, and keeps the "
                                 "color profile unless you choose otherwise."))
        self.rad_clean_sel, self.rad_clean_shown = self._scope_row(lay)
        b_audit = QPushButton("🔍 Audit")
        b_audit.setToolTip("Count what the chosen images contain, without changing anything")
        b_audit.clicked.connect(self.audit)
        lay.addWidget(b_audit)
        self.lbl_audit = hint_label("")
        lay.addWidget(self.lbl_audit)

        g = QGroupBox("Remove")
        gl = QVBoxLayout(g)
        self.chk_location = QCheckBox("GPS location")
        self.chk_device = QCheckBox("Camera / device make, model, serial numbers, owner name")
        self.chk_ai = QCheckBox("AI generation data (prompts, seeds, models, ComfyUI workflows)")
        self.chk_xmp = QCheckBox("XMP editing history")
        self.chk_time = QCheckBox("Timestamps")
        self.chk_software = QCheckBox("Software name")
        for c in (self.chk_location, self.chk_device, self.chk_ai, self.chk_xmp):
            c.setChecked(True)
        self.chk_all = QCheckBox("Everything (keeps only orientation and color profile)")
        self.chk_all.toggled.connect(self._toggle_all)
        self.chk_icc = QCheckBox("Also remove the color profile (not recommended — colors may shift)")
        for c in (self.chk_location, self.chk_device, self.chk_ai, self.chk_xmp, self.chk_time, self.chk_software,
                  self.chk_all, self.chk_icc):
            gl.addWidget(c)
        lay.addWidget(g)
        b_clean = QPushButton("🧹 Clean metadata…")
        b_clean.setStyleSheet("background-color: #0984e3; color: white; font-weight: bold; padding: 6px;")
        b_clean.clicked.connect(self.clean)
        lay.addWidget(b_clean)
        lay.addWidget(hint_label("Supported: JPEG, PNG, WebP. Captions (.txt) are separate files and aren't affected."))
        lay.addStretch()
        return w

    def _build_author(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(hint_label("Credit your work. Values are written exactly as you enter them to the standard "
                                 "EXIF and PNG fields. (EXIF only stores plain ASCII, so © is written as (c) there; "
                                 "PNG keeps the exact text.)"))
        self.presets = metadata_presets()
        row = QHBoxLayout()
        self.combo_author = QComboBox()
        self.combo_author.setToolTip("Your saved authorship templates")
        self.combo_author.activated.connect(lambda _i: self._load_author(self.combo_author.currentData()))
        b_save = QPushButton("Save…")
        b_save.setToolTip("Save these fields as a template (kept until you delete it)")
        b_save.clicked.connect(self._save_author)
        b_del = QPushButton("Delete")
        b_del.clicked.connect(self._delete_author)
        row.addWidget(self.combo_author, 1)
        row.addWidget(b_save)
        row.addWidget(b_del)
        lay.addLayout(row)
        f = QFormLayout()
        self.inp_artist = QLineEdit()
        self.inp_artist.setPlaceholderText("Your name or handle")
        self.inp_copyright = QLineEdit()
        self.inp_copyright.setPlaceholderText("e.g. © 2026 Your Name — CC BY-NC 4.0")
        self.inp_desc = QLineEdit()
        self.inp_desc.setPlaceholderText("Optional description")
        self.inp_software = QLineEdit()
        self.inp_software.setPlaceholderText("Tools you actually used (optional)")
        f.addRow("Artist:", self.inp_artist)
        f.addRow("Copyright / license:", self.inp_copyright)
        f.addRow("Description:", self.inp_desc)
        f.addRow("Software:", self.inp_software)
        lay.addLayout(f)
        self.rad_auth_sel, self.rad_auth_shown = self._scope_row(lay)
        b_apply = QPushButton("✍ Apply authorship…")
        b_apply.setStyleSheet("background-color: #6c5ce7; color: white; font-weight: bold; padding: 6px;")
        b_apply.clicked.connect(self.apply_author)
        lay.addWidget(b_apply)
        lay.addWidget(hint_label("Blank fields are left unchanged. Combine with Privacy & cleanup to remove other data."))
        lay.addStretch()
        self._fill_author_presets()
        return w

    def _build_prompts(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(hint_label("Images made with A1111 / Forge, ComfyUI, NovelAI or InvokeAI often carry the prompt "
                                 "that generated them. Use it as a starting caption, then refine it."))
        self.rad_pr_sel, self.rad_pr_shown = self._scope_row(lay)
        self.rad_pr_missing = QRadioButton("Only images without a caption")
        self.rad_pr_missing.setChecked(True)
        self.rad_pr_replace = QRadioButton("Replace existing captions")
        lay.addWidget(self.rad_pr_missing)
        lay.addWidget(self.rad_pr_replace)
        b = QPushButton("Use embedded prompts as captions")
        b.clicked.connect(lambda: self.prompts_to_captions(
            self._scope_keys(self.rad_pr_shown), replace=self.rad_pr_replace.isChecked()))
        lay.addWidget(b)
        lay.addWidget(hint_label("Results are unsaved and undoable in every tab until you press Save."))
        self.lbl_prompts = hint_label("")
        lay.addWidget(self.lbl_prompts)
        lay.addStretch()
        return w

    # ------------------------------------------------------------------ helpers
    def _scope_keys(self, shown_radio: QRadioButton) -> list[str]:
        return self.browser.keys_for_scope("shown" if shown_radio.isChecked() else "selected")

    def _on_selection(self, keys):
        if keys and self.current_key not in keys:
            self.load_metadata(keys[0])

    def _toggle_all(self, on: bool):
        for c in (self.chk_location, self.chk_device, self.chk_ai, self.chk_xmp, self.chk_time, self.chk_software):
            c.setEnabled(not on)

    def _strip_options(self) -> md.StripOptions:
        return md.StripOptions(everything=self.chk_all.isChecked(), location=self.chk_location.isChecked(),
                               device=self.chk_device.isChecked(), ai_data=self.chk_ai.isChecked(),
                               timestamps=self.chk_time.isChecked(), software=self.chk_software.isChecked(),
                               xmp=self.chk_xmp.isChecked(), remove_icc=self.chk_icc.isChecked())

    # ------------------------------------------------------------------ inspect
    def load_metadata(self, path: str):
        """Also the slot for the Gallery's image_selected signal."""
        if not path:
            return
        self.current_key = path
        key = path

        def done(report):
            if self.current_key != key:
                return
            self.report = report
            self._show_report(report)

        run_in_background(lambda: md.read_metadata(key), done)

    def _show_report(self, r: md.MetaReport):
        self.lbl_file.setText(f"<b>{Path(r.path).name}</b>" + (f"  — error: {r.error}" if r.error else ""))
        self.list_findings.clear()
        findings = md.privacy_findings(r)
        if not findings:
            item = QListWidgetItem("✓ Nothing sensitive found")
            item.setForeground(QColor("#00b894"))
            self.list_findings.addItem(item)
        for f in findings:
            item = QListWidgetItem(f"● {f.label} — {f.detail}")
            item.setForeground(QColor(LEVEL_COLORS[f.level]))
            item.setToolTip(f.detail)
            self.list_findings.addItem(item)
        self.table.blockSignals(True)
        rows = r.rows()
        self.table.setRowCount(len(rows))
        for i, (key, value, edit) in enumerate(rows):
            k = QTableWidgetItem(key)
            k.setFlags(k.flags() & ~Qt.ItemIsEditable)
            v = QTableWidgetItem(value)
            v.setData(Qt.UserRole, edit)
            if not edit:
                v.setFlags(v.flags() & ~Qt.ItemIsEditable)
                v.setForeground(QColor("#9a9a9a"))
            self.table.setItem(i, 0, k)
            self.table.setItem(i, 1, v)
        self.table.resizeRowsToContents()
        self.table.blockSignals(False)
        self.btn_save.setEnabled(False)
        self.btn_use_prompt.setEnabled(md.extract_prompt(r) is not None)

    def save_edits(self):
        if not self.report:
            return
        changed = []
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 1)
            edit = item.data(Qt.UserRole)
            if not edit:
                continue
            kind, field = edit.split(":", 1)
            old = self.report.text.get(field) if kind == "text" else md._val(self.report.exif.get(int(field), ""))
            if item.text() != old:
                changed.append((kind, field, item.text()))
        try:
            for kind, field, value in changed:
                if kind == "text":
                    md.set_text_field(self.report.path, field, value)
                else:
                    md.set_exif_text(self.report.path, int(field), md.exif_ascii(value))
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
        self.load_metadata(self.report.path)

    # ------------------------------------------------------------------ audit / clean
    def audit(self):
        keys = self._scope_keys(self.rad_clean_shown)
        if not keys:
            self.lbl_audit.setText("Select images first.")
            return
        self.lbl_audit.setText(f"Auditing {len(keys)} image(s)…")

        def work():
            counts: Counter = Counter()
            for k in keys:
                for f in md.privacy_findings(md.read_metadata(k)):
                    counts[(f.level, f.label)] += 1
            return counts

        def done(counts):
            if not counts:
                self.lbl_audit.setText(f"✓ Nothing sensitive found in {len(keys)} image(s).")
                return
            order = {"high": 0, "medium": 1, "low": 2}
            lines = [f"<span style='color:{LEVEL_COLORS[lv]}'>● {label}: {n} of {len(keys)}</span>"
                     for (lv, label), n in sorted(counts.items(), key=lambda kv: (order[kv[0][0]], -kv[1]))]
            self.lbl_audit.setText("<br>".join(lines))

        run_in_background(work, done, lambda e: self.lbl_audit.setText(f"Audit failed: {e}"))

    def _run_file_job(self, keys, fn, title):
        if self.job is not None:
            QMessageBox.information(self, "Busy", "A metadata job is already running.")
            return
        self.job = start_file_job(keys, fn, title)

        def finished(summary):
            self.job = None
            self.browser.flash(summary.text())
            show_job_summary(self, summary)
            if self.current_key:
                self.load_metadata(self.current_key)

        self.job.finished.connect(finished)

    def clean(self):
        keys = self._scope_keys(self.rad_clean_shown)
        if not keys:
            QMessageBox.information(self, "Clean metadata", "Select images first.")
            return
        opts = self._strip_options()
        what = "all metadata except orientation" + ("" if opts.remove_icc else " and color profile") \
            if opts.everything else ", ".join(c.text().split(" (")[0] for c in (
                self.chk_location, self.chk_device, self.chk_ai, self.chk_xmp, self.chk_time, self.chk_software)
                if c.isChecked())
        if not what:
            QMessageBox.information(self, "Clean metadata", "Tick at least one kind of data to remove.")
            return
        if not confirm(self, "Clean metadata", f"Remove {what} from {len(keys)} image file(s)?\n\n"
                       "Pixels are not touched, but removed metadata can't be restored.", destructive=True):
            return
        self._run_file_job(keys, lambda k: (md.strip_metadata(k, opts), None)[1], "Metadata cleanup")

    # ------------------------------------------------------------------ authorship
    def _author(self) -> md.Authorship:
        return md.Authorship(artist=self.inp_artist.text(), copyright=self.inp_copyright.text(),
                             description=self.inp_desc.text(), software=self.inp_software.text())

    def _fill_author_presets(self, select: str = ""):
        self.combo_author.clear()
        self.combo_author.addItem("(no template)", "")
        for name in self.presets.names():
            self.combo_author.addItem(name, name)
        self.combo_author.setCurrentIndex(max(0, self.combo_author.findData(select)))

    def _load_author(self, name: str):
        data = self.presets.get(name) if name else None
        if data:
            self.inp_artist.setText(data.get("artist", ""))
            self.inp_copyright.setText(data.get("copyright", ""))
            self.inp_desc.setText(data.get("description", ""))
            self.inp_software.setText(data.get("software", ""))

    def _save_author(self):
        current = self.combo_author.currentData() or ""
        name, ok = QInputDialog.getText(self, "Save authorship template", "Template name:", text=current)
        if not ok or not name.strip():
            return
        try:
            saved = self.presets.save(name, {k: v for k, v in vars(self._author()).items()})
        except PresetError as e:
            QMessageBox.warning(self, "Save template", str(e))
            return
        self._fill_author_presets(saved)

    def _delete_author(self):
        name = self.combo_author.currentData()
        if name and confirm(self, "Delete template", f"Delete the authorship template '{name}'?"):
            self.presets.delete(name)
            self._fill_author_presets()

    def apply_author(self):
        a = self._author()
        if not any(v.strip() for v in vars(a).values()):
            QMessageBox.information(self, "Authorship", "Fill in at least one field.")
            return
        keys = self._scope_keys(self.rad_auth_shown)
        if not keys:
            QMessageBox.information(self, "Authorship", "Select images first.")
            return
        if confirm(self, "Apply authorship", f"Write these authorship fields to {len(keys)} image file(s)?"):
            self._run_file_job(keys, lambda k: (md.apply_authorship(k, a), None)[1], "Authorship")

    # ------------------------------------------------------------------ prompts → captions
    def prompts_to_captions(self, keys: list[str], replace: bool):
        keys = [k for k in keys if k]
        s = self.ctx.session
        if not keys or not s:
            return

        def work():
            return {k: md.extract_prompt(md.read_metadata(k)) for k in keys}

        def done(found: dict):
            changes = {}
            for k, prompt in found.items():
                e = s.get(k)
                if not prompt or e is None or (e.has_caption and not replace):
                    continue
                if prompt != e.text:
                    changes[k] = (e.text, prompt)
            n = self.ctx.push(changes, f"Prompts → captions ({len(changes)})")
            with_prompt = sum(1 for p in found.values() if p)
            msg = (f"{n} caption(s) set from embedded prompts — unsaved; press Save to keep. "
                   f"({with_prompt} of {len(keys)} image(s) had a prompt.)")
            self.lbl_prompts.setText(msg)
            self.browser.flash(msg)

        run_in_background(work, done, lambda e: self.lbl_prompts.setText(f"Failed: {e}"))
