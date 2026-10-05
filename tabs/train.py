"""Train tab: native LoRA training on the open dataset (training/ - Fizgig's family layer).

The tab never trains in-process. Start freezes a run folder (training.pipeline.build_run) and launches the cache and
train stages as child processes with QProcess, reading their console output for progress, the loss chart and the
samples strip. Pause / preview override / caption fixes are files in the run folder; Stop kills the process tree.
"""
from __future__ import annotations

import logging
import os
import re
import webbrowser
from pathlib import Path

from PySide6.QtCore import QProcess, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap, QTextCursor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox,
                               QSplitter, QTabWidget, QVBoxLayout, QWidget)

from core.config import settings
from tabs.common import CollapsibleSection, confirm, hint_label
from tabs.memory_bar import MemoryBar
from tabs.train_dialogs import LossChart, ProblemImagesDialog, SampleOverrideDialog
from tabs.workspace.context import workspace
from training import params as P
from training import pipeline, presets
from training.progress import TrainingProgressTracker
from training.registry import training_families

# the trainer's tqdm step bar, including its first `0/1200 [00:00<?, ?it/s]` line (no rate yet, so the progress
# tracker does not parse it)
_STEP_BAR_RE = re.compile(r"^\s*steps:\s+\d+%\|")
# tqdm leaves its bar without a newline, so the trainer's next log line arrives glued to the bar's tail:
# `steps: 0%|   | 0/1200 [00:00<?, ?it/s][warm-up] ...`. Group 1 is the bar, group 2 the message.
_GLUED_RE = re.compile(r"^(\s*steps:\s+\d+%\|[^|]*\|\s*\d+/\d+\s+\[[^\]]*\])(.*\S.*)$")

log = logging.getLogger(__name__)

IDLE, RUNNING, PAUSING, PAUSED = "idle", "running", "pausing", "paused"
# What "simple mode" shows; the preset decides everything else.
SIMPLE_KEYS = {"LORA_OUTPUT_DIR", "LORA_NAME", "MAX_TRAIN_EPOCHS", "DATASET_MEGAPIXELS", "SAMPLE_ENABLED",
               "SAMPLE_PROMPT"}
_ACTIVE: list = []          # running TrainTabs (main.py asks before quitting)


def any_running() -> bool:
    return any(t.state in (RUNNING, PAUSING) for t in _ACTIVE)


class TrainTab(QWidget):
    def __init__(self):
        super().__init__()
        _ACTIVE.append(self)
        self.cfg = settings()
        self.ctx = workspace()
        self.families = training_families()
        self.desc = None
        self.values: dict = P.defaults()
        self.memory: dict = {}             # family key -> values this session (Fizgig's per-family memory)
        self.widgets: dict = {}            # param key -> (widget, row widgets)
        self.model_edits: dict = {}
        self.proc: QProcess | None = None
        self.run: pipeline.Run | None = None
        self.stage_index = 0
        self.state = IDLE
        self.tracker = TrainingProgressTracker()
        self.queue: list = presets.load_queue()
        self._buf = ""
        self._log_file = None
        self._progress_line = ""       # the trainer's step bar, shown as ONE console line that updates in place
        self._stopped = False          # Stop was pressed: the kill's exit code is not a failure
        self._loading = False

        root = QHBoxLayout(self)
        split = QSplitter()
        root.addWidget(split)
        split.addWidget(self._build_settings())
        split.addWidget(self._build_run_panel())
        split.setSizes([620, 760])

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh_samples)
        self.timer.start(5000)
        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.timeout.connect(self._persist_values)
        self.ctx.session_changed.connect(self._update_dataset_label)
        from PySide6.QtGui import QKeySequence, QShortcut
        sc = QShortcut(QKeySequence("Ctrl+Return"), self)
        sc.setContext(Qt.WidgetWithChildrenShortcut)
        sc.activated.connect(self.start)

        last = self.cfg.get("training.family", "")
        idx = next((i for i, d in enumerate(self.families) if d.key == last), 0)
        if self.families:
            self.combo_family.setCurrentIndex(idx)
            self._on_family(idx)
        self._update_dataset_label()
        self._refresh_queue()
        self._set_state(IDLE)

    # ======================================================================================== settings side
    def _build_settings(self) -> QWidget:
        outer = QWidget()
        ol = QVBoxLayout(outer)
        ol.setContentsMargins(0, 0, 0, 0)
        top = QGroupBox("Model and preset")
        tf = QFormLayout(top)
        self.combo_family = QComboBox()
        self.combo_family.setToolTip("The base model family to train a LoRA for. Each family brings its own model "
                                     "files, presets and settings.")
        for d in self.families:
            self.combo_family.addItem(d.display_name + (" (experimental)" if d.experimental else ""), d.key)
        self.combo_family.currentIndexChanged.connect(self._on_family)
        tf.addRow("Family:", self.combo_family)
        row = QHBoxLayout()
        self.combo_preset = QComboBox()
        self.combo_preset.setToolTip("✨ = built-in presets (Fizgig's measured recipes). Your own presets follow. "
                                     "Loading a preset changes only the settings it contains.")
        self.combo_preset.setMinimumWidth(260)
        row.addWidget(self.combo_preset, 1)
        for label, tip, fn in (("Load", "Apply the selected preset", self._load_preset),
                               ("Save…", "Save the current settings as your own preset", self._save_preset),
                               ("Delete", "Delete the selected user preset (to the Recycle Bin)", self._delete_preset),
                               ("Import…", "Import a preset file (TagScribeR or Fizgig .json)", self._import_preset),
                               ("Last run", "Load the settings of the last training run", self._load_last_run)):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        tf.addRow("Preset:", row)
        self.lbl_family_note = hint_label("")
        tf.addRow(self.lbl_family_note)
        self.chk_all = QCheckBox("Show all settings")
        self.chk_all.setChecked(bool(self.cfg.get("training.show_all", False)))
        self.chk_all.setToolTip("Off: only the essentials are shown and the preset decides everything else - the "
                                "simple way to train. On: every setting (optimizer, memory, loss watch, timesteps, "
                                "metadata, resume...) for full control. Hidden settings keep their values.")
        self.chk_all.toggled.connect(self._toggle_all)
        tf.addRow(self.chk_all)
        ol.addWidget(top)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        self.form_layout = QVBoxLayout(body)
        scroll.setWidget(body)
        ol.addWidget(scroll, 1)

        # dataset
        g = QGroupBox("Dataset")
        f = QFormLayout(g)
        self.lbl_dataset = QLabel()
        self.lbl_dataset.setWordWrap(True)
        f.addRow("Folder:", self.lbl_dataset)
        self.inp_trigger = QLineEdit()
        self.inp_trigger.setPlaceholderText("e.g. ohwx woman (from Auto Caption's subject)")
        self.inp_trigger.setToolTip("Your trigger word. Recorded as the LoRA's trigger phrase, and put at the start "
                                    "(or end) of auto-recaptions. It is not added to your captions - write it into "
                                    "them (Auto Caption > Subject) if the LoRA should learn it.")
        f.addRow("Trigger:", self.inp_trigger)
        self.combo_trigger_pos = QComboBox()
        self.combo_trigger_pos.addItems(["start", "end"])
        self.combo_trigger_pos.setToolTip("Where auto-recaptions get the trigger word")
        f.addRow("Trigger position:", self.combo_trigger_pos)
        self.chk_cache = QCheckBox("Enable cache preparation")
        self.chk_cache.setChecked(self.cfg.get("training.enable_cache", True))
        self.chk_cache.setToolTip("Encode latents and captions before training (only what changed is re-encoded). "
                                  "Turn off only if the cache is known to be current.")
        f.addRow(self.chk_cache)
        self.form_layout.addWidget(g)

        # model files
        self.grp_models = QGroupBox("Model files")
        self.models_form = QFormLayout(self.grp_models)
        self.form_layout.addWidget(self.grp_models)

        # parameter groups
        self.sections: dict = {}
        for group in P.GROUPS:
            if group == "Samples":
                continue
            sec = CollapsibleSection(group, expanded=group in ("Output", "Training Parameters"))
            fl = QFormLayout()
            sec.body_layout.addLayout(fl)
            self.sections[group] = (sec, fl)
            self.form_layout.addWidget(sec)
        sec = CollapsibleSection("Samples (previews)", expanded=False)
        fl = QFormLayout()
        sec.body_layout.addLayout(fl)
        self.sections["Samples"] = (sec, fl)
        self.form_layout.addWidget(sec)
        for p in P.PARAMS:
            self._make_widget(p, self.sections[p.group][1])

        # auto-recaption captioner
        g = self.grp_captioner = QGroupBox("Auto-recaption captioner")
        f = QFormLayout(g)
        f.addRow(hint_label("Used only when 'Auto-recaption stuck images' is on. Runs between epochs, then is "
                            "unloaded before training continues."))
        self.combo_cap_kind = QComboBox()
        self.combo_cap_kind.addItem("Local vision model (Transformers)", "transformers")
        self.combo_cap_kind.addItem("WD tagger (booru tags)", "wd_tagger")
        self.combo_cap_kind.setCurrentIndex(max(0, self.combo_cap_kind.findData(
            self.cfg.get("training.captioner_kind", "transformers"))))
        self.inp_cap_model = QLineEdit(self.cfg.get("training.captioner_model", "") or
                                       self.cfg.get("caption.last_model", ""))
        self.inp_cap_model.setPlaceholderText("model folder or Hugging Face repo id")
        self.inp_cap_model.setToolTip("Defaults to the model last used in Auto Caption")
        self.inp_cap_prompt = QLineEdit(self.cfg.get("training.captioner_prompt", "") or
                                        "Describe this image in detail for an image-generation training caption.")
        f.addRow("Captioner:", self.combo_cap_kind)
        f.addRow("Model:", self.inp_cap_model)
        f.addRow("Instruction:", self.inp_cap_prompt)
        self.form_layout.addWidget(g)

        # resume
        g = self.grp_resume = QGroupBox("Resume")
        f = QFormLayout(g)
        row = QHBoxLayout()
        self.inp_resume = QLineEdit()
        self.inp_resume.setPlaceholderText("empty = a fresh run")
        self.inp_resume.setToolTip("A saved state folder (<name>-000012-state) to continue from - e.g. to train more "
                                   "epochs on a finished LoRA. Raise Epochs above the state's epoch.")
        b = QPushButton("Browse…")
        b.clicked.connect(lambda: self._browse_dir(self.inp_resume))
        b2 = QPushButton("Latest")
        b2.setToolTip("The newest saved state of this LoRA name in the output folder")
        b2.clicked.connect(self._latest_state)
        row.addWidget(self.inp_resume, 1)
        row.addWidget(b)
        row.addWidget(b2)
        f.addRow("State:", row)
        self.form_layout.addWidget(g)
        self.form_layout.addStretch(1)
        return outer

    def _make_widget(self, p: P.Param, form: QFormLayout):
        if p.kind == P.BOOL:
            w = QCheckBox(p.label)
            w.toggled.connect(self._changed)
            form.addRow(w)
            label = None
        else:
            if p.kind == P.INT:
                w = QSpinBox()
                w.setRange(int(p.minimum if p.minimum is not None else -10 ** 9),
                           int(p.maximum if p.maximum is not None else 10 ** 9))
                w.valueChanged.connect(self._changed)
            elif p.kind == P.FLOAT:
                w = QLineEdit()
                w.textChanged.connect(self._changed)
            elif p.kind == P.CHOICE:
                w = QComboBox()
                w.setEditable(not p.strict)
                w.addItems([str(o) for o in p.options])
                w.currentTextChanged.connect(self._changed)
            elif p.kind == P.MULTILINE:
                w = QPlainTextEdit()
                w.setMaximumHeight(90)
                w.textChanged.connect(self._changed)
            elif p.kind in (P.PATH, P.DIR):
                w = QLineEdit()
                w.textChanged.connect(self._changed)
                holder = QWidget()
                hl = QHBoxLayout(holder)
                hl.setContentsMargins(0, 0, 0, 0)
                hl.addWidget(w, 1)
                b = QPushButton("Browse")
                b.clicked.connect(lambda _=False, e=w, d=p.kind == P.DIR: self._browse_dir(e) if d
                                  else self._browse_file(e))
                hl.addWidget(b)
            else:
                w = QLineEdit()
                w.textChanged.connect(self._changed)
            label = QLabel(p.label + ":")
            label.setToolTip(p.tip)
            form.addRow(label, holder if p.kind in (P.PATH, P.DIR) else w)
        w.setToolTip(p.tip + (" (not part of presets)" if not p.preset else ""))
        if p.key == "LORA_OUTPUT_DIR":
            w.setPlaceholderText(str(pipeline.default_output_dir()))
        self.widgets[p.key] = (w, label)

    # ---- values <-> widgets -------------------------------------------------------------------------
    def _set_widget(self, key, value):
        p = P.BY_KEY[key]
        w, _ = self.widgets[key]
        if p.kind == P.BOOL:
            w.setChecked(bool(value))
        elif p.kind == P.INT:
            try:
                w.setValue(int(float(value)))
            except (TypeError, ValueError):
                w.setValue(int(p.default))
        elif p.kind == P.CHOICE:
            hit = P.match_option(value, [w.itemText(i) for i in range(w.count())])
            if hit is not None:
                w.setCurrentText(hit)
            elif w.isEditable():
                w.setEditText(str(value))
        elif p.kind == P.MULTILINE:
            w.setPlainText(str(value))
        elif p.kind == P.FLOAT:
            w.setText(f"{value:g}" if isinstance(value, float) else str(value))
        else:
            w.setText(str(value))

    def _get_widget(self, key):
        p = P.BY_KEY[key]
        w, _ = self.widgets[key]
        if p.kind == P.BOOL:
            return w.isChecked()
        if p.kind == P.INT:
            return w.value()
        if p.kind == P.CHOICE:
            return w.currentText()
        if p.kind == P.MULTILINE:
            return w.toPlainText()
        if p.kind == P.FLOAT:
            try:
                return float(w.text().strip())
            except ValueError:
                return w.text().strip()
        return w.text()

    def _apply_values(self, values: dict):
        self._loading = True
        try:
            for k, v in values.items():
                if k in self.widgets:
                    self._set_widget(k, v)
        finally:
            self._loading = False
        self.values = self.collect()
        self._update_visibility()

    def collect(self) -> dict:
        return {k: self._get_widget(k) for k in self.widgets}

    def _changed(self, *_):
        if self._loading:
            return
        self.values = self.collect()
        self._update_visibility()
        self.save_timer.start(800)

    def _persist_values(self):
        if self.desc:
            self.cfg.set(f"training.values.{self.desc.key}", self.collect())

    def _toggle_all(self, on):
        self.cfg.set("training.show_all", bool(on))
        self._update_visibility()

    def _update_visibility(self):
        simple = not self.chk_all.isChecked()
        shown_groups = set()
        adaptive = bool(self.values.get("ADAPTIVE_LR"))
        lokr = str(self.values.get("NETWORK_TYPE", "")).startswith("LoKR")
        edit = bool(self.values.get("FAMILY_EDIT"))
        for key, (w, label) in self.widgets.items():
            p = P.BY_KEY[key]
            show = P.family_shows(p, self.desc) and (not simple or key in SIMPLE_KEYS)
            if key in ("NETWORK_DIM", "NETWORK_ALPHA"):
                show = show and not lokr
            if key == "LOKR_FACTOR":
                show = show and lokr
            if key in ("FAMILY_EDIT_DIR", "FAMILY_EDIT_REF", "FAMILY_EDIT_CAPTION"):
                show = show and edit
            for x in (w if p.kind not in (P.PATH, P.DIR) else w.parentWidget(), label):
                if x is not None:
                    x.setVisible(show)
            if show:
                shown_groups.add(p.group)
            if key in ("ADAPTIVE_LR_MIN", "ADAPTIVE_LR_MAX"):
                w.setEnabled(adaptive)
            if key in ("LEARNING_RATE", "LR_SCHEDULER", "LR_WARMUP_STEPS"):
                w.setEnabled(not adaptive or key == "LEARNING_RATE")
        for group, (sec, _fl) in self.sections.items():
            sec.setVisible(group in shown_groups)
            if simple and group in shown_groups and not sec.toggle.isChecked():
                sec.toggle.setChecked(True)
        self.grp_captioner.setVisible(not simple)
        self.grp_resume.setVisible(not simple)
        if adaptive:
            try:
                lo = float(P.first_token(self.values.get("ADAPTIVE_LR_MIN")))
                hi = float(P.first_token(self.values.get("ADAPTIVE_LR_MAX")))
                self.widgets["LEARNING_RATE"][0].setToolTip(
                    f"Ignored while Adaptive LR is on: the run starts at sqrt(min x max) = {(lo * hi) ** 0.5:.2e}.")
            except ValueError:
                pass

    # ---- family / presets ---------------------------------------------------------------------------
    def _on_family(self, idx):
        if idx < 0 or idx >= len(self.families):
            return
        if self.desc is not None:
            self.memory[self.desc.key] = self.collect()
            self._persist_values()
        self.desc = self.families[idx]
        self.cfg.set("training.family", self.desc.key)
        self.store = presets.TrainingPresets(self.desc)
        for key in ("OPTIMIZER_TYPE", "NETWORK_TYPE", "FAMILY_PRECISION"):
            w = self.widgets[key][0]
            w.blockSignals(True)
            w.clear()
            w.addItems(list(P.options_for(P.BY_KEY[key], self.desc)))
            w.blockSignals(False)
        self._build_model_rows()
        self._refresh_presets()
        notes = [f"Native resolution about {self.desc.native_megapixels:g} MP", f"{len(self.desc.presets)} presets"]
        if self.desc.training_adapter_note:
            notes.append(self.desc.training_adapter_note)
        self.lbl_family_note.setText(" · ".join(notes))
        saved = self.memory.get(self.desc.key) or self.cfg.get(f"training.values.{self.desc.key}")
        if saved:
            self._apply_values({**P.defaults(), **saved})
        else:                               # first visit: Fizgig applies the family's first built-in preset
            base = {**P.defaults(), "SAMPLE_WIDTH": self.desc.preview_width,
                    "SAMPLE_HEIGHT": self.desc.preview_height, "SAMPLE_STEPS": self.desc.preview_steps,
                    "SAMPLE_CFG_SCALE": self.desc.preview_cfg}
            if self.desc.preview_negative:
                base["SAMPLE_NEGATIVE"] = self.desc.preview_negative
            if self.desc.ema_default:
                base["FAMILY_EMA"] = next((o for o in P.EMA_OPTIONS if P.first_token(o) == self.desc.ema_default),
                                          base["FAMILY_EMA"])
            self._apply_values(base)
            if self.store.default_name:
                self._apply_preset(self.store.default_name, quiet=True)
        if not self.values.get("LORA_NAME") or self.values.get("LORA_NAME") == "my_lora":
            self._set_widget("LORA_NAME", f"my_lora_{self.desc.lora_name_suffix}")

    def _build_model_rows(self):
        while self.models_form.rowCount():
            self.models_form.removeRow(0)
        self.model_edits = {}
        for f in self.desc.model_files:
            e = QLineEdit(self.cfg.get(pipeline.model_setting_key(f.pref_key), ""))
            e.setPlaceholderText("required" if f.required else ("empty = use the checkpoint's own" if f.default_to else "optional"))
            tip = f.note + (f"\nDownload: huggingface.co/{f.repo} - {f.path} ({f.size_gb:g} GB)" if f.repo else "")
            e.setToolTip(tip)
            e.textChanged.connect(lambda t, k=f.pref_key: self.cfg.set(pipeline.model_setting_key(k), t.strip()))
            holder = QWidget()
            hl = QHBoxLayout(holder)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.addWidget(e, 1)
            b = QPushButton("Browse")
            b.setToolTip("Choose the file")
            b.clicked.connect(lambda _=False, ed=e: self._browse_file(ed, "Model files (*.safetensors *.json);;All (*)"))
            hl.addWidget(b)
            if f.repo:
                g = QPushButton("Get")
                g.setToolTip(f"Open huggingface.co/{f.repo} in the browser ({f.path})")
                g.clicked.connect(lambda _=False, r=f.repo, pth=f.path: webbrowser.open(
                    f"https://huggingface.co/{r}/blob/main/{pth}" if pth else f"https://huggingface.co/{r}"))
                hl.addWidget(g)
            lab = QLabel(f.label + ("" if f.required else " (optional)") + ":")
            lab.setToolTip(tip)
            self.models_form.addRow(lab, holder)
            self.model_edits[f.pref_key] = e

    def models(self) -> dict:
        return {k: e.text().strip() for k, e in self.model_edits.items()}

    def _refresh_presets(self, select: str | None = None):
        self.combo_preset.clear()
        for n in self.store.names():
            self.combo_preset.addItem(n)
        if select:
            self.combo_preset.setCurrentText(select)

    def _apply_preset(self, name, quiet=False):
        try:
            data = self.store.load(name)
        except presets.PresetError as e:
            QMessageBox.warning(self, "Preset", str(e))
            return
        new, rep = presets.apply(data, self.collect(), self.desc)
        self._apply_values(new)
        for m in rep.messages():
            self._console(m)
        if not quiet:
            self._console(f"[preset] applied '{name}' ({len(rep.applied)} setting(s))")
            if rep.refused:
                QMessageBox.information(self, "Preset", "Some saved values aren't offered for this family and were "
                                        "kept as they were:\n\n" + "\n".join(rep.messages()))

    def _load_preset(self):
        if self.combo_preset.currentText():
            self._apply_preset(self.combo_preset.currentText())

    def _save_preset(self):
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:",
                                        text=self.combo_preset.currentText().replace("✨ ", ""))
        if not ok or not name.strip():
            return
        try:
            if self.store.exists(name) and not self.store.is_builtin(name) and not confirm(
                    self, "Overwrite preset", f"Replace your preset '{name}'?"):
                return
            self.store.save(name, self.collect())
        except presets.PresetError as e:
            QMessageBox.warning(self, "Preset", str(e))
            return
        self._refresh_presets(name.strip())

    def _delete_preset(self):
        name = self.combo_preset.currentText()
        if not name:
            return
        if self.store.is_builtin(name):
            QMessageBox.information(self, "Preset", "Built-in presets can't be deleted.")
            return
        if confirm(self, "Delete preset", f"Move '{name}' to the Recycle Bin?", destructive=True):
            self.store.delete(name)
            self._refresh_presets()

    def _import_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import preset", "", "Preset (*.json)")
        if path:
            try:
                name = self.store.import_file(Path(path))
            except presets.PresetError as e:
                QMessageBox.warning(self, "Import preset", str(e))
                return
            self._refresh_presets(name)

    def _load_last_run(self):
        snap = presets.load_last_run()
        if not snap:
            QMessageBox.information(self, "Last run", "No training run has been started yet.")
            return
        fam = snap.get(presets.ARCH_KEY)
        if fam and fam != self.desc.key:
            i = self.combo_family.findData(fam)
            if i >= 0:
                self.combo_family.setCurrentIndex(i)
        new, rep = presets.apply(snap, self.collect(), self.desc)
        self._apply_values(new)
        for m in rep.messages():
            self._console(m)
        self._console("[preset] loaded the last run's settings")

    # ---- helpers --------------------------------------------------------------------------------------
    def _browse_dir(self, edit):
        d = QFileDialog.getExistingDirectory(self, "Choose folder", edit.text() or "")
        if d:
            edit.setText(d)

    def _browse_file(self, edit, filt="All files (*)"):
        f, _ = QFileDialog.getOpenFileName(self, "Choose file", os.path.dirname(edit.text() or ""), filt)
        if f:
            edit.setText(f)

    def _latest_state(self):
        from training.train_utils import latest_state_dir
        rd = pipeline.run_dir_for(self.collect())
        s = latest_state_dir(str(rd), str(self.values.get("LORA_NAME", "")).strip())
        if s:
            self.inp_resume.setText(s)
        else:
            QMessageBox.information(self, "Resume", f"No saved state for this LoRA name in {rd}.")

    def _update_dataset_label(self):
        folder = self.ctx.folder
        if not folder:
            self.lbl_dataset.setText("<i>No folder open - open one in the Gallery (Ctrl+O).</i>")
            return
        n = len(self.ctx.session) if self.ctx.session is not None else 0
        self.lbl_dataset.setText(f"{folder}" + (f"  ({n} images)" if n else ""))
        if self.ctx.project is not None and not self.inp_trigger.text().strip():
            self.inp_trigger.setText(str(self.ctx.project.get("subject", "") or "").split(",")[0].strip())

    def captioner(self) -> dict:
        return {"kind": self.combo_cap_kind.currentData(), "model": self.inp_cap_model.text().strip(),
                "prompt": self.inp_cap_prompt.text().strip(),
                "prompt_detailed": self.inp_cap_prompt.text().strip() + " Be exhaustive: describe every visible "
                                   "detail, the subject, clothing, pose, setting, lighting and style."}

    # ======================================================================================== run side
    def _build_run_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        self.btn_start = QPushButton("🚀 Start Training")
        self.btn_start.setToolTip("Check the settings, freeze them into the run folder and start (Ctrl+Enter). While "
                                  "a run is going, this queues the current settings as the next run.")
        self.btn_start.clicked.connect(self.start)
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setToolTip("Finish the current epoch, save a resumable state and stop cleanly. Resume "
                                  "continues exactly where it left off.")
        self.btn_pause.clicked.connect(self.pause)
        self.btn_resume = QPushButton("Resume")
        self.btn_resume.setToolTip("Continue the paused run from its saved state")
        self.btn_resume.clicked.connect(self.resume_paused)
        self.btn_stop = QPushButton("🛑 Stop")
        self.btn_stop.setToolTip("Stop immediately (nothing more is saved; epochs already saved stay). Use Pause to "
                                 "stop cleanly.")
        self.btn_stop.clicked.connect(self.stop)
        for b in (self.btn_start, self.btn_pause, self.btn_resume, self.btn_stop):
            row.addWidget(b)
        lay.addLayout(row)
        row = QHBoxLayout()
        for label, tip, fn in (("Problem Images", "Images the loss watch flags, with a caption editor",
                                self.open_problem_images),
                               ("Preview override…", "Render a different prompt at the next preview",
                                self.open_override),
                               ("📂 Open run folder", "Checkpoints, samples and logs of this run", self.open_run_folder)):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        self.lbl_status = QLabel("Idle")
        self.lbl_status.setWordWrap(True)
        lay.addWidget(self.lbl_status)
        self.progress = QProgressBar()
        lay.addWidget(self.progress)
        # VRAM / RAM with this run's peak, as Fizgig's trainer shows it (Hide stats is remembered across launches)
        stats = QHBoxLayout()
        self.memory_bar = MemoryBar()
        stats.addWidget(self.memory_bar, 1)
        self.btn_stats = QPushButton()
        self.btn_stats.setFlat(True)
        self.btn_stats.setToolTip("Show or hide the VRAM / RAM bars. Peaks are still tracked while hidden.")
        self.btn_stats.clicked.connect(self._toggle_stats)
        stats.addWidget(self.btn_stats)
        lay.addLayout(stats)
        self._show_stats(bool(self.cfg.get("training.stats_bar_visible", True)))
        tabs = QTabWidget()
        self.chart = LossChart()
        tabs.addTab(self.chart, "Loss")
        self.samples = QListWidget()
        self.samples.setViewMode(QListWidget.IconMode)
        self.samples.setIconSize(QSize(192, 192))
        self.samples.setResizeMode(QListWidget.Adjust)
        self.samples.setToolTip("Preview images by epoch (newest first). Double-click to open.")
        self.samples.itemDoubleClicked.connect(lambda it: QDesktopServices.openUrl(QUrl.fromLocalFile(
            it.data(Qt.UserRole))))
        tabs.addTab(self.samples, "Samples")
        qw = QWidget()
        ql = QVBoxLayout(qw)
        ql.addWidget(hint_label("Runs waiting to start. Start while a run is going adds the current settings here; "
                                "the next one starts when the current run finishes. The queue is never started "
                                "automatically when the app opens."))
        self.list_queue = QListWidget()
        ql.addWidget(self.list_queue)
        qr = QHBoxLayout()
        for label, fn in (("Remove", self._queue_remove), ("Load into settings", self._queue_load),
                          ("Start next now", self._queue_start)):
            b = QPushButton(label)
            b.clicked.connect(fn)
            qr.addWidget(b)
        ql.addLayout(qr)
        tabs.addTab(qw, "Queue")
        split = QSplitter(Qt.Vertical)
        split.addWidget(tabs)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(4000)
        self.console.setToolTip("The training processes' output (also saved as run.log in the run folder)")
        split.addWidget(self.console)
        split.setSizes([380, 300])
        lay.addWidget(split, 1)
        return w

    def _show_stats(self, visible):
        self.memory_bar.setVisible(visible)
        self.btn_stats.setText("Hide stats" if visible else "Show stats")

    def _toggle_stats(self):
        visible = self.memory_bar.isHidden()
        self._show_stats(visible)
        self.cfg.set("training.stats_bar_visible", visible)

    def _console_progress(self, text):
        """The trainer's `steps: N/total` bar: one console line, replaced by each update (printing it once and never
        again left a stale `0/1200` on screen for the whole run). run.log gets the last one only."""
        if not self._progress_line:
            self.console.appendPlainText(text)
        else:
            cur = self.console.textCursor()
            cur.movePosition(QTextCursor.End)
            cur.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
            cur.insertText(text)
        self._progress_line = text

    def _console(self, text):
        if self._progress_line:
            last, self._progress_line = self._progress_line, ""
            self._log(last)
        self.console.appendPlainText(text)
        self._log(text)

    def _log(self, text):
        if self._log_file:
            try:
                self._log_file.write(text + "\n")
                self._log_file.flush()
            except OSError:
                pass

    def _set_state(self, state):
        self.state = state
        running = state in (RUNNING, PAUSING)
        from tabs import icons
        self.btn_start.setText("Queue this run" if running else "Start Training")
        icons.set(self.btn_start, "add" if running else "run")
        self.btn_pause.setEnabled(state == RUNNING and self.stage_label() == "Training")
        self.btn_stop.setEnabled(running)
        self.btn_resume.setVisible(state == PAUSED)
        self.combo_family.setEnabled(not running)

    def stage_label(self):
        if self.run and self.stage_index < len(self.run.stages):
            return self.run.stages[self.stage_index][0]
        return ""

    # ---- start / queue --------------------------------------------------------------------------------
    def start(self):
        values = self.collect()
        folder = self.ctx.folder
        resume = self.inp_resume.text().strip()
        if self.state in (RUNNING, PAUSING):
            self.queue.append(presets.queue_item(self.desc.key, values, folder))
            presets.save_queue(self.queue)
            self._refresh_queue()
            self._console(f"[queue] added {values.get('LORA_NAME')} (runs after the current one)")
            return
        if self.ctx.has_unsaved():
            if not confirm(self, "Unsaved captions", "Some captions have unsaved changes. Save them before "
                                                      "training? (Training reads the caption files.)"):
                return
            self.ctx.save(self)
        self._launch(self.desc, values, folder, resume)

    def models_for(self, desc) -> dict:
        return {f.pref_key: str(self.cfg.get(pipeline.model_setting_key(f.pref_key), "") or "").strip()
                for f in desc.model_files}

    def _launch(self, desc, values, folder, resume=""):
        models = self.models_for(desc)
        checks = pipeline.preflight(desc, values, folder, models, captioner=self.captioner(), resume=resume)
        errors = [c.message for c in checks if c.level == "error"]
        warnings = [c.message for c in checks if c.level == "warning"]
        if errors:
            QMessageBox.warning(self, "Can't start yet", "\n\n".join(errors))
            return False
        if warnings and not confirm(self, "Start training?", "\n\n".join(warnings) + "\n\nStart anyway?"):
            return False
        if not resume and pipeline.checkpoints(pipeline.run_dir_for(values), str(values.get("LORA_NAME")).strip()):
            if not confirm(self, "LoRA name in use", f"The run folder for '{values.get('LORA_NAME')}' already has "
                           "checkpoints. A fresh run overwrites files with the same names. Continue?"):
                return False
        try:
            self.run = pipeline.build_run(desc, values, folder, models, captioner=self.captioner(),
                                          trigger=self.inp_trigger.text().strip(),
                                          trigger_position=self.combo_trigger_pos.currentText(),
                                          resume=resume, enable_cache=self.chk_cache.isChecked())
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Can't start", str(e))
            return False
        presets.save_last_run(desc.key, values, folder)
        self._run_ctx = (desc, dict(values), folder)
        self.cfg.set("training.enable_cache", self.chk_cache.isChecked())
        self.cfg.update({"training.captioner_kind": self.combo_cap_kind.currentData(),
                         "training.captioner_model": self.inp_cap_model.text().strip(),
                         "training.captioner_prompt": self.inp_cap_prompt.text().strip()})
        pipeline.clear_paused(self.run.run_dir)
        try:
            self._log_file = open(self.run.run_dir / "run.log", "a", encoding="utf-8")
        except OSError:
            self._log_file = None
        if not resume:
            self.chart.clear()
        self.tracker.reset(self.run.total_epochs)
        self._progress_line, self._stopped = "", False
        self.memory_bar.reset_peaks()
        for c in checks:
            if c.level == "info":
                self._console(f"[check] {c.message}")
        self._console(f"=== {desc.display_name}: {self.run.output_name} -> {self.run.run_dir}")
        self.stage_index = 0
        self._set_state(RUNNING)
        self._start_stage()
        return True

    def _start_stage(self):
        label, argv = self.run.stages[self.stage_index]
        self.lbl_status.setText(f"{label}…")
        self.progress.setRange(0, 0)
        self._console(f"--- {label}")
        from core import paths
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.setWorkingDirectory(str(paths.APP_ROOT))
        env = self.proc.processEnvironment()
        env.clear()
        for k, v in pipeline.child_env().items():
            env.insert(k, v)
        self.proc.setProcessEnvironment(env)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._finished)
        self.proc.errorOccurred.connect(lambda e: self._console(f"[process] error: {e}"))
        import time
        self._stage_started = time.time() - 1
        self.proc.start(argv[0], argv[1:])
        if self.proc.waitForStarted(5000):
            pipeline.lower_priority(int(self.proc.processId()))
        self._set_state(self.state)

    def _read(self):
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._buf += data
        parts = self._buf.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        self._buf = parts.pop()
        lines = []
        for line in parts:
            glued = _GLUED_RE.match(line)       # a log line stuck to the step bar: keep both, the message as its own line
            lines.extend(glued.groups() if glued else (line,))
        for line in lines:
            if not line.strip():
                continue
            u = self.tracker.consume(line)
            if (u is not None and u["kind"] == "training") or _STEP_BAR_RE.match(line):
                self._console_progress(line.strip())
            else:
                self._console(line)
            self._on_update(u)

    def _on_update(self, u):
        if not u:
            return
        k = u["kind"]
        if k == "cache":
            self.progress.setRange(0, max(1, u["total"]))
            self.progress.setValue(u["done"])
            self.lbl_status.setText(f"Caching {u['stage']}: {u['done']}/{u['total']}")
        elif k == "training":
            self.progress.setRange(0, u["total_steps"])
            self.progress.setValue(u["step"])
            loss = f" · loss {u['average_loss_text']}" if u.get("average_loss_text") else ""
            self.lbl_status.setText(f"Training - epoch {u['epoch']}/{u['total_epochs']} · step {u['step']}/"
                                    f"{u['total_steps']} · {u['speed_text']} · ETA {u['eta_text']}{loss}"
                                    + (" · pausing after this epoch" if self.state == PAUSING else ""))
        elif k == "epoch":
            self.chart.add(u["epoch"], u["loss"], u["lr"])
        elif k == "adaptive":
            self.chart.mark(u["epoch"], u["action"])
        elif k == "preview" and u["phase"] == "complete":
            self._refresh_samples()

    def _finished(self, code, _status):
        if self._buf.strip():
            u = self.tracker.consume(self._buf)
            if (u is not None and u["kind"] == "training") or _STEP_BAR_RE.match(self._buf):
                self._console_progress(self._buf.strip())
            else:
                self._console(self._buf)
            self._on_update(u)
        self._buf = ""
        label = self.stage_label()
        final = self.run.run_dir / f"{self.run.output_name}.safetensors"
        finished = final.exists() and final.stat().st_mtime >= self._stage_started
        paused = label == "Training" and code == 0 and pipeline.pause_requested(self.run.run_dir) and not finished
        if self._stopped:
            self._stopped = False
            self._end(f"Stopped during {label.lower()}. Epochs already saved are kept.")
            return
        if code != 0:
            self._console(f"[{label}] stopped with exit code {code}")
            self._end(f"{label} failed (exit code {code}) - see the console. Epochs already saved are kept.")
            return
        if paused:
            from training.train_utils import latest_state_dir
            st = latest_state_dir(str(self.run.run_dir), self.run.output_name)
            pipeline.clear_pause(self.run.run_dir)
            desc, values, folder = self._run_ctx
            pipeline.write_paused(self.run.run_dir, {"state": st, "family": desc.key, "values": values,
                                                     "folder": folder})
            self._paused_state = st
            self._end(f"Paused - state saved ({os.path.basename(st or '')}). Resume continues from there.",
                      state=PAUSED)
            return
        self.stage_index += 1
        if self.stage_index < len(self.run.stages):
            self._start_stage()
            return
        self._refresh_samples()
        self._end(f"Done - {self.run.run_dir / (self.run.output_name + '.safetensors')}")
        self._next_in_queue()

    def _end(self, message, state=IDLE):
        self.lbl_status.setText(message)
        self._console(f"=== {message}")
        self.progress.setRange(0, 1)
        self.progress.setValue(1 if state == IDLE else 0)
        self.proc = None
        if self._log_file:
            self._log_file.close()
            self._log_file = None
        self._set_state(state)

    def pause(self):
        if self.run and self.state == RUNNING:
            pipeline.request_pause(self.run.run_dir)
            self._set_state(PAUSING)
            self._console("[pause] requested - the run saves its state and stops after the current epoch")

    def resume_paused(self):
        info = pipeline.read_paused(self.run.run_dir) if self.run else None
        st = (info or {}).get("state") or getattr(self, "_paused_state", None)
        if not st:
            QMessageBox.information(self, "Resume", "No paused state was found.")
            return
        from training.registry import get
        desc = get((info or {}).get("family", "")) or self.desc
        self._launch(desc, (info or {}).get("values") or self.collect(), (info or {}).get("folder") or self.ctx.folder,
                     resume=st)

    def stop(self):
        if not self.proc:
            return
        if not confirm(self, "Stop training", "Stop the run now? The current epoch is lost; saved epochs stay. "
                                              "(Pause stops cleanly at the end of the epoch.)", destructive=True):
            return
        self._stopped = True
        pipeline.kill_tree(int(self.proc.processId()))
        self._console("[stop] process tree stopped")

    def shutdown(self):
        """Called by main.py on quit (after the user confirmed)."""
        self.memory_bar.shutdown()
        if self.proc:
            pipeline.kill_tree(int(self.proc.processId()))
            self.proc.waitForFinished(5000)

    # ---- queue ----------------------------------------------------------------------------------------
    def _refresh_queue(self):
        self.list_queue.clear()
        for it in self.queue:
            self.list_queue.addItem(f"{it.get('LORA_NAME')} · {it.get(presets.ARCH_KEY)} · "
                                    f"{os.path.basename(str(it.get(presets.DATASET_KEY) or ''))}")

    def _queue_remove(self):
        r = self.list_queue.currentRow()
        if r >= 0:
            self.queue.pop(r)
            presets.save_queue(self.queue)
            self._refresh_queue()

    def _queue_load(self):
        r = self.list_queue.currentRow()
        if r < 0:
            return
        it = self.queue[r]
        i = self.combo_family.findData(it.get(presets.ARCH_KEY))
        if i >= 0:
            self.combo_family.setCurrentIndex(i)
        new, _ = presets.apply(it, self.collect(), self.desc)
        self._apply_values(new)

    def _queue_start(self):
        if self.state in (RUNNING, PAUSING):
            QMessageBox.information(self, "Queue", "A run is going; the queue continues when it finishes.")
            return
        self._next_in_queue()

    def _next_in_queue(self):
        while self.queue:
            it = self.queue.pop(0)
            presets.save_queue(self.queue)
            self._refresh_queue()
            fam = next((d for d in self.families if d.key == it.get(presets.ARCH_KEY)), None)
            folder = str(it.get(presets.DATASET_KEY) or "")
            if fam is None or not os.path.isdir(folder):
                self._console(f"[queue] skipped {it.get('LORA_NAME')}: family or dataset folder missing")
                continue
            values, rep = presets.apply(it, P.defaults(), fam)
            values.update({k: v for k, v in it.items() if k in P.BY_KEY and not P.BY_KEY[k].preset})
            self._console(f"[queue] starting {values.get('LORA_NAME')}")
            if self._launch(fam, values, folder):
                return

    # ---- windows --------------------------------------------------------------------------------------
    def _run_dir(self):
        return self.run.run_dir if self.run else pipeline.run_dir_for(self.collect())

    def open_problem_images(self):
        rd = self._run_dir()

        def saved(paths):
            try:
                self.ctx.reload_from_disk(paths)
            except Exception:
                pass
        dlg = ProblemImagesDialog(str(rd), str(self.values.get("DATASET_CAPTION_EXT") or ".txt"), self.ctx.folder,
                                  self, on_saved=saved)
        dlg.show()

    def open_override(self):
        SampleOverrideDialog(str(self._run_dir()), self).exec()

    def open_run_folder(self):
        rd = self._run_dir()
        rd.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(rd)))

    def _refresh_samples(self):
        if not self.isVisible():
            return
        files = list(reversed(pipeline.samples(self._run_dir())))[:60]
        have = [self.samples.item(i).data(Qt.UserRole) for i in range(self.samples.count())]
        if have == [str(p) for p in files]:
            return
        self.samples.clear()
        for p in files:
            parsed = pipeline.parse_sample_name(p.name)
            it = QListWidgetItem(QIcon(QPixmap(str(p)).scaled(192, 192, Qt.KeepAspectRatio,
                                                               Qt.SmoothTransformation)),
                                 f"epoch {parsed[0]} #{parsed[1]}" if parsed else p.name)
            it.setData(Qt.UserRole, str(p))
            it.setToolTip(f"{p.name}\nseed {parsed[2]}" if parsed else p.name)
            self.samples.addItem(it)

    def showEvent(self, e):
        super().showEvent(e)
        self._update_dataset_label()
        self._refresh_samples()

