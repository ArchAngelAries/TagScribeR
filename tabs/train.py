"""Train tab: native LoRA training on the open dataset (training/ - Fizgig's family layer).

The tab never trains in-process. Start freezes a run folder (training.pipeline.build_run) and launches the cache and
train stages as child processes with QProcess, reading their console output for progress, the loss chart and the
samples strip. Pause / preview override / caption fixes are files in the run folder; Stop kills the process tree.
"""
from __future__ import annotations

import json
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
from tabs.common import ClickToDeselect, CollapsibleSection, confirm, hint_label
from tabs import icons, memory_bar
from tabs.train_dialogs import LossChart, ProblemImagesDialog, SampleOverrideDialog, SliderPromptsDialog
from tabs.workspace.context import workspace
from training import params as P
from training import pipeline, presets, step_log
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
# the Edit rows' labels when extra pictures are chosen (Fizgig families/form.py:376-389 and :884-891)
EDIT_STEP_LABELS = {"FAMILY_EDIT_DIR": "1. Originals folder (before editing)",
                    "FAMILY_EDIT_EXTRAS_DIR": "2. Extra pictures folder",
                    "FAMILY_EDIT_CAPTION": "4. Edit instruction", "FAMILY_EDIT_REF": "5. Test photo (optional)"}
SAMPLE_NOW_TIP = ("Render a preview round right after the current step, with the LoRA as it is now. The pictures show "
                   "under the epoch in progress in Samples (Ctrl+Shift+S). Needs previews set up: Every N epochs "
                   "above 0 and a prompt. Not while the dataset is being cached.")
SAVE_NOW_TIP = ("Save the LoRA as it is right after the current step, as <name>-stepNNNNNNN.safetensors beside the "
                "epoch checkpoints (Ctrl+Shift+D). It is a checkpoint only: it carries no resume state. Not while "
                "the dataset is being cached.")
_ACTIVE: list = []          # running TrainTabs (main.py asks before quitting)


def run_previews_set_up(run_dir) -> bool:
    """Whether a run folder's frozen config has previews: a Sample Every N above 0 and at least one prompt (what Sample
    now can render). False for a missing or unreadable folder."""
    try:
        cfg = json.loads(Path(run_dir, "train_config.json").read_text(encoding="utf-8"))
        every = int((cfg.get("train") or {}).get("sample_every_n_epochs") or 0)
        prompts = Path(run_dir, "sample_prompts.txt").read_text(encoding="utf-8").strip()
    except (OSError, ValueError, TypeError):
        return False
    return every > 0 and bool(prompts)


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
        self._loss_sig = None          # (path, mtime, size) of the step log the chart shows
        self._chart_key = None         # the run (folder, name) the chart is drawn for
        self._follow_live = None       # whether the Distilled tick was following the checkpoint's recipe last time
        self._samples_sig = None
        self._slider_dlg = None

        root = QHBoxLayout(self)
        split = QSplitter()
        root.addWidget(split)
        split.addWidget(self._build_settings())
        split.addWidget(self._build_run_panel())
        split.setSizes([620, 760])

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh_samples)
        self.timer.start(5000)
        self.loss_timer = QTimer(self)                  # the per-step loss graph while a run trains
        self.loss_timer.timeout.connect(self._refresh_loss)
        self.loss_timer.start(4000)
        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.timeout.connect(self._persist_values)
        self.ctx.session_changed.connect(self._update_dataset_label)
        from PySide6.QtGui import QKeySequence, QShortcut
        sc = QShortcut(QKeySequence("Ctrl+Return"), self)
        sc.setContext(Qt.WidgetWithChildrenShortcut)
        sc.activated.connect(self.start)
        # Sample now / Save now / slider prompts: only while their button is enabled
        for seq, btn in (("Ctrl+Shift+S", self.btn_sample_now), ("Ctrl+Shift+D", self.btn_save_now),
                         ("Ctrl+Shift+L", self.btn_slider)):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda b=btn: b.click() if b.isEnabled() and b.isVisibleTo(self) else None)

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
            if p.key == "SAMPLE_USE_DISTILLED":
                w.toggled.connect(self._on_distilled_toggled)      # Steps / CFG follow the tick
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
        self._follow_update(loading=True)
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
        extras = str(self.values.get("FAMILY_EDIT_PICTURES") or "").strip() == P.EDIT_WITH_EXTRAS
        for key, (w, label) in self.widgets.items():
            p = P.BY_KEY[key]
            show = P.family_shows(p, self.desc) and (not simple or key in SIMPLE_KEYS)
            if key in ("NETWORK_DIM", "NETWORK_ALPHA"):
                show = show and not lokr
            if key == "LOKR_FACTOR":
                show = show and lokr
            if key in ("FAMILY_EDIT_DIR", "FAMILY_EDIT_REF", "FAMILY_EDIT_CAPTION", "FAMILY_EDIT_PICTURES"):
                show = show and edit
            if key == "FAMILY_EDIT_EXTRAS_DIR":          # only with "The photo + extra pictures" chosen
                show = show and edit and extras
            if key in EDIT_STEP_LABELS and label is not None:      # Fizgig's numbered steps once extras join
                label.setText((EDIT_STEP_LABELS[key] if extras else P.BY_KEY[key].label) + ":")
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

    # ---- the Distilled tick's Steps / CFG follow (Fizgig form.py "follow" + app.js turboFollow) ---------------
    def _follow_sides(self):
        """({key: value} when the preview checkpoint renders, {key: value} when the training model does), or None for
        a family without a preview checkpoint recipe."""
        d = self.desc
        ck = getattr(d, "preview_checkpoint_sampling", None) if d is not None else None
        if d is None or not d.train_preview_checkpoint or ck is None:
            return None
        return ({"SAMPLE_STEPS": ck.steps, "SAMPLE_CFG_SCALE": ck.cfg},
                {"SAMPLE_STEPS": d.preview_steps, "SAMPLE_CFG_SCALE": d.preview_cfg})

    def _checkpoint_live(self) -> bool:
        """The preview checkpoint will render the previews: its tick is on and its file is set."""
        d = self.desc
        ck = d.preview_checkpoint() if d is not None else None
        if ck is None or "SAMPLE_USE_DISTILLED" not in self.widgets:
            return False
        e = self.model_edits.get(ck[0].pref_key)
        return bool(self._get_widget("SAMPLE_USE_DISTILLED") and e is not None and os.path.isfile(e.text().strip()))

    def _follow_update(self, loading=False):
        """Steps and CFG follow the Distilled tick: the checkpoint's own recipe while it renders the previews (tick on
        and its file set), the training model's when not. Only a box that is empty (0 = the family default) or still on
        one of those two recipes moves, so a value typed by hand is never replaced. The boxes move when the switch
        changes, and when a form loads (which also puts right one saved on the other side's default)."""
        sides = self._follow_sides()
        if sides is None or "SAMPLE_STEPS" not in self.widgets:
            self._follow_live = None
            return
        live = self._checkpoint_live()
        if not loading and live == self._follow_live:
            return
        self._follow_live = live
        want = sides[0] if live else sides[1]
        for key, val in want.items():
            cur = self._get_widget(key)
            try:
                cur_f = float(cur)
            except (TypeError, ValueError):
                cur_f = 0.0                                  # empty / not a number: nothing typed
            known = [float(sides[0][key]), float(sides[1][key])]
            if cur_f <= 0 or (abs(cur_f - float(val)) > 1e-9 and any(abs(cur_f - k) < 1e-9 for k in known)):
                self._set_widget(key, val)

    def _on_distilled_toggled(self, *_):
        if not self._loading:
            self._follow_update()

    def _apply_tick_label(self):
        """The Samples tick's text from the family (its description's preview_checkpoint_tick); Klein's wording
        otherwise."""
        w = self.widgets["SAMPLE_USE_DISTILLED"][0]
        w.setText(P.preview_checkpoint_tick(self.desc))

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
        self._apply_tick_label()
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
            if f.role == "preview_dit":                  # the Distilled's file being set is part of the tick's follow
                e.textChanged.connect(lambda _t: self._follow_update())
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
        for m in rep.messages():
            self._console(m)
        if rep.blocked:
            QMessageBox.warning(self, "Preset", f"'{name}' is a Fizgig preset for "
                                f"{' and '.join(sorted({w for _, w in rep.blocked}))}, which TagScribeR cannot train "
                                "yet.\n\nNothing was changed: applied anyway, it would train an ordinary LoRA.")
            return
        self._apply_values(new)
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
        rd = self._run_dir_or_say()
        if rd is None:
            return
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
        row = QHBoxLayout()                       # between-epoch actions on the running trainer (control files)
        self.btn_sample_now = QPushButton("Sample now")
        self.btn_sample_now.setToolTip(SAMPLE_NOW_TIP)
        self.btn_sample_now.clicked.connect(self.sample_now)
        self.btn_save_now = QPushButton("Save now")
        self.btn_save_now.setToolTip(SAVE_NOW_TIP)
        self.btn_save_now.clicked.connect(self.save_now)
        self.btn_slider = QPushButton("Slider prompts…")
        self.btn_slider.setToolTip("Change the base prompt, what each end adds and the push strength of the prompt "
                                   "slider that is training (taken at the end of the epoch) (Ctrl+Shift+L)")
        self.btn_slider.clicked.connect(self.open_slider_prompts)
        for b, ic in ((self.btn_sample_now, "camera"), (self.btn_save_now, "save"), (self.btn_slider, "tag")):
            icons.set(b, ic)
            row.addWidget(b)
        lay.addLayout(row)
        self.lbl_status = QLabel("Idle")
        self.lbl_status.setWordWrap(True)
        lay.addWidget(self.lbl_status)
        self.progress = QProgressBar()
        lay.addWidget(self.progress)
        tabs = QTabWidget()
        self.chart = LossChart()
        tabs.addTab(self.chart, "Loss")
        self.samples = QListWidget()
        self.samples.setViewMode(QListWidget.IconMode)
        self.samples.setIconSize(QSize(192, 192))
        self.samples.setResizeMode(QListWidget.Adjust)
        self.samples.setToolTip("Preview images by epoch (newest first). Double-click to open; click a selected "
                                "image again to deselect it.")
        self._samples_deselect = ClickToDeselect(self.samples)
        self.samples.itemDoubleClicked.connect(lambda it: QDesktopServices.openUrl(QUrl.fromLocalFile(
            it.data(Qt.UserRole))))
        tabs.addTab(self.samples, "Samples")
        self.checkpoints = QListWidget()
        self.checkpoints.setToolTip("The LoRA files in the run folder: the final one, each epoch's, and the ones Save "
                                    "now made (<name>-step0000123). Double-click to show the folder.")
        self.checkpoints.itemDoubleClicked.connect(lambda _it: self.open_run_folder())
        tabs.addTab(self.checkpoints, "Checkpoints")
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
        self.btn_start.setText("Queue this run" if running else "Start Training")
        icons.set(self.btn_start, "add" if running else "run")
        self.btn_pause.setEnabled(state == RUNNING and self.stage_label() == "Training")
        self.btn_stop.setEnabled(running)
        self.btn_resume.setVisible(state == PAUSED)
        self.combo_family.setEnabled(not running)
        self._update_now_buttons()

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
        """The model files set for `desc`. A stored path to a file a newer one replaces (ModelFile.replaces, e.g. Qwen's
        training adapter v1 -> v2) moves to the new file once it is on disk, and the setting is updated."""
        out = {}
        for f in desc.model_files:
            key = pipeline.model_setting_key(f.pref_key)
            stored = str(self.cfg.get(key, "") or "").strip()
            moved = f.migrate(stored, os.path.isfile)
            if moved != stored:
                self.cfg.set(key, moved)
                if f.pref_key in self.model_edits:          # the Model files row shows (and keeps) the new path
                    self.model_edits[f.pref_key].setText(moved)
                log.info("[models] %s: %s -> %s (the newer file replaces it)", f.label, stored, moved)
            out[f.pref_key] = moved
        return out

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
        if not resume:                  # a fresh run starts on its own slider prompts, not the last run's
            for name in (pipeline.SLIDER_LIVE_FILE, pipeline.SLIDER_LIVE_APPLIED_FILE):
                try:
                    Path(self.run.run_dir, name).unlink()
                except OSError:
                    pass
        try:
            self._log_file = open(self.run.run_dir / "run.log", "a", encoding="utf-8")
        except OSError:
            self._log_file = None
        self._chart_key, self._loss_sig = (str(self.run.run_dir), self.run.output_name), None
        if not resume:
            self.chart.clear()
        else:
            self.chart.set_steps(pipeline.read_step_log(self.run.run_dir, self.run.output_name))
        self.tracker.reset(self.run.total_epochs)
        self._progress_line, self._stopped = "", False
        memory_bar.reset_all_peaks()      # the window's VRAM / RAM bar: peaks are per training run
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
        self.proc = None
        self._refresh_loss(force=True)        # the last steps the trainer flushed on its way out
        self.lbl_status.setText(message)
        self._console(f"=== {message}")
        self.progress.setRange(0, 1)
        self.progress.setValue(1 if state == IDLE else 0)
        if self._log_file:
            self._log_file.close()
            self._log_file = None
        self._set_state(state)

    def pause(self):
        if self.run and self.state == RUNNING:
            pipeline.request_pause(self.run.run_dir)
            self._set_state(PAUSING)
            self._console("[pause] requested - the run saves its state and stops after the current epoch")

    # ---- Sample now / Save now / slider prompts (control files the trainer takes between steps) ---------
    def _training_stage(self) -> bool:
        """The train stage (not the cache stages) of a live run is going."""
        return self.run is not None and self.state in (RUNNING, PAUSING) and self.stage_label() == "Training"

    def _slider_prompt_run(self) -> bool:
        ctx = getattr(self, "_run_ctx", None)
        return bool(ctx and self._training_stage() and pipeline.slider_on(ctx[0], ctx[1], "prompts"))

    def _update_now_buttons(self):
        live = self._training_stage()
        previews = live and run_previews_set_up(self.run.run_dir)
        self.btn_save_now.setEnabled(live)
        self.btn_sample_now.setEnabled(previews)
        self.btn_sample_now.setToolTip(SAMPLE_NOW_TIP if previews or not live else
                                       "Previews are not set up for this run (Every N epochs is 0 or there is no "
                                       "prompt), so there is nothing to render. " + SAMPLE_NOW_TIP)
        slider = self._slider_prompt_run()
        self.btn_slider.setVisible(slider)
        self.btn_slider.setEnabled(slider)
        if not slider and self._slider_dlg is not None:      # the slider run is over: its window goes too
            self._slider_dlg.close()

    def sample_now(self):
        if not self.btn_sample_now.isEnabled() or self.run is None:
            return
        pipeline.request_sample_now(self.run.run_dir)
        self._console("[sample now] requested - previews are rendered after the current step and show under the "
                      "epoch in progress")

    def save_now(self):
        if not self.btn_save_now.isEnabled() or self.run is None:
            return
        pipeline.request_save_now(self.run.run_dir)
        self._console(f"[save now] requested - the LoRA is saved after the current step as "
                      f"{self.run.output_name}-stepNNNNNNN.safetensors")
        QTimer.singleShot(6000, self, self._refresh_checkpoints)

    def open_slider_prompts(self):
        ctx = getattr(self, "_run_ctx", None)
        if not self._slider_prompt_run() or ctx is None:
            return
        desc, v, _folder = ctx
        try:
            g = float(v.get("FAMILY_SLIDER_GUIDANCE") or 0)
        except (TypeError, ValueError):
            g = 0.0
        cur = {"base": str(v.get("FAMILY_SLIDER_BASE") or ""), "pos": str(v.get("FAMILY_SLIDER_POS") or ""),
               "neg": str(v.get("FAMILY_SLIDER_NEG") or ""), "guidance": g if g > 0 else desc.slider_guidance}
        if self._slider_dlg is not None:                     # one window: bring the open one forward
            self._slider_dlg.raise_()
            self._slider_dlg.activateWindow()
            return
        dlg = self._slider_dlg = SliderPromptsDialog(str(self.run.run_dir), cur, self,
                                                     is_live=self._slider_prompt_run)
        dlg.destroyed.connect(lambda *_: setattr(self, "_slider_dlg", None))
        dlg.show()

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
            if rep.blocked:
                self._console(f"[queue] skipped {it.get('LORA_NAME')}: " + " ".join(rep.messages()))
                continue
            values.update({k: v for k, v in it.items() if k in P.BY_KEY and not P.BY_KEY[k].preset})
            self._console(f"[queue] starting {values.get('LORA_NAME')}")
            if self._launch(fam, values, folder):
                return

    # ---- windows --------------------------------------------------------------------------------------
    def _run_dir(self):
        """The run folder for the current LoRA name, or None while the name is not a valid one (half typed)."""
        if self.run:
            return self.run.run_dir
        try:
            return pipeline.run_dir_for(self.collect())
        except ValueError:
            return None

    def _run_dir_or_say(self):
        rd = self._run_dir()
        if rd is None:
            QMessageBox.information(self, "LoRA name", "Give the LoRA a valid name first (letters, digits, '-' and "
                                    "'_'; no folders).")
        return rd

    def open_problem_images(self):
        rd = self._run_dir_or_say()
        if rd is None:
            return

        def saved(paths):
            try:
                self.ctx.reload_from_disk(paths)
            except Exception:
                pass
        dlg = ProblemImagesDialog(str(rd), str(self.values.get("DATASET_CAPTION_EXT") or ".txt"), self.ctx.folder,
                                  self, on_saved=saved)
        dlg.show()

    def open_override(self):
        rd = self._run_dir_or_say()
        if rd is not None:
            SampleOverrideDialog(str(rd), self).exec()

    def open_run_folder(self):
        rd = self._run_dir_or_say()
        if rd is None:
            return
        rd.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(rd)))

    def _output_name(self) -> str:
        return self.run.output_name if self.run else str(self.values.get("LORA_NAME") or "").strip()

    def _refresh_loss(self, force=False):
        """The per-step loss curve from the run folder's step log: polled while a run trains (the log changes every
        few steps), and read once for a finished, paused or earlier run that is shown. Nothing is read while the
        cache stages run (the file is the previous run's until the trainer restarts it) or when it has not changed."""
        if not force and not self.isVisible():
            return
        if self.state in (RUNNING, PAUSING) and self.proc is not None and self.stage_label() != "Training":
            return
        rd, name = self._run_dir(), self._output_name()
        if rd is None or not name:
            return
        key = (str(rd), name)
        if key != self._chart_key:                 # another run is shown: its own curve, not the last one's
            self._chart_key, self._loss_sig = key, None
            self.chart.clear()
        path = step_log.loss_path(str(rd), name)
        try:
            st = os.stat(path)
            sig = (path, st.st_mtime_ns, st.st_size)
        except OSError:
            sig = (path, None, None)
        if sig == self._loss_sig:
            return
        self._loss_sig = sig
        self.chart.set_steps(pipeline.read_step_log(rd, name) if sig[1] is not None else None)

    def _refresh_checkpoints(self):
        rd, name = self._run_dir(), self._output_name()
        if rd is None or not name:
            return
        rows = []
        for f in pipeline.checkpoints(rd, name):
            try:
                rows.append(f"{f.name}   ({f.stat().st_size / 1024 ** 2:.0f} MB)")
            except OSError:
                pass
        if rows != [self.checkpoints.item(i).text() for i in range(self.checkpoints.count())]:
            self.checkpoints.clear()
            self.checkpoints.addItems(rows)

    def _refresh_samples(self):
        if not self.isVisible():
            return
        rd = self._run_dir()
        if rd is None:
            return
        self._refresh_checkpoints()
        files = list(reversed(pipeline.samples(rd)))[:60]
        prompts = pipeline.read_preview_prompts(rd, self._output_name())     # {epoch: [the prompts that round used]}
        sig = ([str(p) for p in files], sorted(prompts.items()))
        if sig == self._samples_sig:
            return
        self._samples_sig = sig
        self.samples.clear()
        for p in files:
            parsed = pipeline.parse_sample_name(p.name)
            prompt = ""
            if parsed:
                used = prompts.get(parsed[0]) or []
                prompt = used[parsed[1]] if parsed[1] < len(used) else ""
            label = f"epoch {parsed[0]} #{parsed[1]}" if parsed else p.name
            if prompt:
                label += "\n" + (prompt if len(prompt) <= 48 else prompt[:45] + "...")
            it = QListWidgetItem(QIcon(QPixmap(str(p)).scaled(192, 192, Qt.KeepAspectRatio,
                                                               Qt.SmoothTransformation)), label)
            it.setData(Qt.UserRole, str(p))
            it.setToolTip((f"{p.name}\nseed {parsed[2]}" if parsed else p.name)
                          + (f"\nPrompt: {prompt}" if prompt else ""))
            self.samples.addItem(it)

    def showEvent(self, e):
        super().showEvent(e)
        self._update_dataset_label()
        self._refresh_samples()
        self._refresh_loss()

