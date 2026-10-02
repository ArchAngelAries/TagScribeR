"""Auto Caption tab: caption images with local VLMs or OpenAI-compatible APIs.

All inference runs through ``inference`` (providers + ModelManager +
BatchWorker); this module is only UI. Loaded models are reused across runs.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QMenu, QToolButton, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                               QMessageBox, QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter,
                               QTabWidget, QTextEdit, QVBoxLayout, QWidget)

from core import paths
from core.config import settings
from core.secrets import ApiProfile, ApiProfileStore
from inference import models as model_catalog
from inference.base import CaptionRequest, GenerationParams, ProviderSpec
from inference.manager import manager
from core.presets import PresetError, caption_presets
from inference.prompts import DEFAULT_PRESET
from inference.worker import (SAVE_APPEND, SAVE_OVERWRITE, SAVE_PREPEND, SAVE_SKIP_EXISTING, BatchJob,
                              start_job)
from tabs.common import CollapsibleSection, confirm, hint_label, run_in_background, show_job_summary
from tabs.workspace.browser import DatasetBrowser
from tabs.workspace.context import JOB_FAILED, JOB_QUEUED, JOB_WORKING, workspace

log = logging.getLogger(__name__)

SAVE_MODES = [
    ("Overwrite existing captions", SAVE_OVERWRITE),
    ("Skip images that already have a caption", SAVE_SKIP_EXISTING),
    ("Append to existing captions", SAVE_APPEND),
    ("Prepend to existing captions", SAVE_PREPEND),
]


class CaptionTab(QWidget):
    def __init__(self):
        super().__init__()
        self.cfg = settings()
        self.ctx = workspace()
        self.selected_paths: list[str] = []
        self.worker = None
        self.profiles = ApiProfileStore()
        self._local_entries: list[tuple[str, str, float, str]] = []  # (label, model id/path, GB, notes)

        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # ---------------- LEFT: shared dataset browser ----------------
        self.browser = DatasetBrowser(self.ctx, size_key="ui.caption_thumbnail_size")
        self.browser.selection_changed.connect(self.on_selection)
        self.btn_select_uncaptioned = QPushButton("Select Uncaptioned")
        self.btn_select_uncaptioned.setToolTip("Select every image that has no caption yet")
        self.btn_select_uncaptioned.clicked.connect(self.select_uncaptioned)
        self.browser.extra_toolbar.addWidget(self.btn_select_uncaptioned)
        left = self.browser

        # ---------------- RIGHT: controls ----------------
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right = QWidget()
        right.setMinimumWidth(360)
        right_scroll.setMinimumWidth(392)  # content + scrollbar: never clip controls
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right.setMaximumWidth(460)
        rl = QVBoxLayout(right)
        right_scroll.setWidget(right)

        # 1. Model source
        self.tab_source = QTabWidget()
        self.tab_source.addTab(self._build_local_tab(), "Local model")
        self.tab_source.addTab(self._build_api_tab(), "API / Server")
        from tabs.help import HelpDialog, help_button
        corner = help_button("models", self.tab_source, "Which model should I use? / How do I connect a server?")
        corner.clicked.disconnect()
        corner.clicked.connect(lambda: HelpDialog.open_topic(self.window(),
                                                             ("models", "api")[self.tab_source.currentIndex()]))
        self.tab_source.setCornerWidget(corner, Qt.TopRightCorner)
        rl.addWidget(self.tab_source)

        # 2. Instructions
        grp_prompt = QGroupBox("Instructions")
        lp = QVBoxLayout(grp_prompt)
        row_help = QHBoxLayout()
        row_help.addWidget(hint_label("Tell the model what to write. Presets are tuned for common dataset styles."), 1)
        row_help.addWidget(help_button("caption", grp_prompt, "How Auto Caption works"))
        lp.addLayout(row_help)
        self.presets = caption_presets()
        preset_row = QHBoxLayout()
        self.combo_template = QComboBox()
        self.combo_template.setToolTip("Caption preset. Built-in presets are listed first; ★ marks your own.\n"
                                       "Edit the text below freely, then 💾 to save it as your own preset.")
        self.btn_preset_save = QToolButton(text="💾")
        self.btn_preset_save.setToolTip("Save the current instructions as a preset (yours are kept until you delete them)")
        self.btn_preset_save.clicked.connect(self.save_preset)
        self.btn_preset_menu = QToolButton(text="⋯")
        self.btn_preset_menu.setToolTip("Rename, delete, revert, import or export presets")
        self.btn_preset_menu.setPopupMode(QToolButton.InstantPopup)
        self.btn_preset_menu.setMenu(QMenu(self.btn_preset_menu))
        self.btn_preset_menu.menu().aboutToShow.connect(self._fill_preset_menu)
        preset_row.addWidget(self.combo_template, 1)
        preset_row.addWidget(self.btn_preset_save)
        preset_row.addWidget(self.btn_preset_menu)
        lp.addLayout(preset_row)
        self.lbl_preset_state = hint_label("")
        lp.addWidget(self.lbl_preset_state)
        self.prompt_input = QTextEdit()
        self.prompt_input.setFixedHeight(90)
        self.prompt_input.setPlaceholderText("What should the model write about each image?")
        lp.addWidget(self.prompt_input)
        self.chk_tag_output = QCheckBox("Output is a tag list")
        self.chk_tag_output.setToolTip("Tick for presets that produce comma-separated tags: appending/prepending then "
                                       "merges tags without duplicates instead of adding a sentence.")
        lp.addWidget(self.chk_tag_output)
        self.sec_system = CollapsibleSection("System prompt (optional)")
        self.system_input = QTextEdit()
        self.system_input.setFixedHeight(60)
        self.system_input.setPlaceholderText("e.g. You are an expert dataset captioner. Never refuse.")
        self.system_input.textChanged.connect(self._update_preset_state)
        self.sec_system.body_layout.addWidget(self.system_input)
        lp.addWidget(self.sec_system)
        self.sec_subject = CollapsibleSection("Subject & tag hints")
        fs = QFormLayout()
        self.inp_subject = QLineEdit()
        self.inp_subject.setPlaceholderText("e.g. ohwx woman, or a character name")
        self.inp_subject.setToolTip("The model calls the subject this instead of 'a woman' / 'a person'. "
                                    "Use your LoRA trigger word.")
        self.chk_subject_first = QCheckBox("Start every caption with it")
        self.chk_subject_first.setToolTip("Guaranteed: if the model forgets, TagScribeR adds it at the start.")
        self.chk_tag_hints = QCheckBox("Use each image's current tags as hints")
        self.chk_tag_hints.setToolTip("Tag → caption recipe: run Auto Tag in the Gallery first; those tags then guide "
                                      "the model toward accurate details. Pick 'Append' to keep the tags too.")
        fs.addRow("Subject:", self.inp_subject)
        fs.addRow(self.chk_subject_first)
        fs.addRow(self.chk_tag_hints)
        self.sec_subject.body_layout.addLayout(fs)
        self.sec_subject.body_layout.addWidget(hint_label(
            "Recipe: Gallery → Batch → Auto Tag the images, then caption here with tag hints on."))
        lp.addWidget(self.sec_subject)
        rl.addWidget(grp_prompt)

        # 3. Generation
        grp_gen = QGroupBox("Generation")
        lg = QFormLayout(grp_gen)
        self.spin_tokens = QSpinBox()
        self.spin_tokens.setRange(16, 32768)
        self.spin_tokens.setToolTip("Upper limit on caption length (in tokens). Raise it for very detailed captions "
                                    "or for reasoning models.")
        self.spin_temp = QDoubleSpinBox()
        self.spin_temp.setRange(0.0, 2.0)
        self.spin_temp.setSingleStep(0.1)
        self.spin_temp.setToolTip("0 = deterministic. Higher = more varied wording.")
        self.combo_save = QComboBox()
        for label, mode in SAVE_MODES:
            self.combo_save.addItem(label, mode)
        lg.addRow("Max tokens:", self.spin_tokens)
        lg.addRow("Temperature:", self.spin_temp)
        lg.addRow("If captioned:", self.combo_save)
        self.combo_save.setToolTip("What to do with images that already have a caption")
        self.sec_adv_gen = CollapsibleSection("Advanced sampling")
        fa = QFormLayout()
        self.spin_top_p = QDoubleSpinBox()
        self.spin_top_p.setRange(0.05, 1.0)
        self.spin_top_p.setSingleStep(0.05)
        self.spin_top_k = QSpinBox()
        self.spin_top_k.setRange(0, 500)
        self.spin_top_k.setToolTip("0 = model default")
        self.spin_rep = QDoubleSpinBox()
        self.spin_rep.setRange(1.0, 2.0)
        self.spin_rep.setSingleStep(0.05)
        self.spin_max_side = QSpinBox()
        self.spin_max_side.setRange(256, 4096)
        self.spin_max_side.setSingleStep(128)
        self.spin_max_side.setSuffix(" px")
        self.spin_max_side.setToolTip("Images are downscaled to this longest side before inference. "
                                      "Lower = faster and less VRAM; higher = more fine detail.")
        self.btn_model_defaults = QPushButton("Use model's recommended settings")
        self.btn_model_defaults.clicked.connect(self.apply_model_defaults)
        fa.addRow("Top P:", self.spin_top_p)
        fa.addRow("Top K:", self.spin_top_k)
        fa.addRow("Repetition penalty:", self.spin_rep)
        fa.addRow("Max image size:", self.spin_max_side)
        fa.addRow(self.btn_model_defaults)
        self.sec_adv_gen.body_layout.addLayout(fa)
        lg.addRow(self.sec_adv_gen)
        rl.addWidget(grp_gen)

        # 4. Run
        self.btn_run = QPushButton()
        self.btn_run.setFixedHeight(48)
        self.btn_run.clicked.connect(self.toggle_process_state)
        rl.addWidget(self.btn_run)
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setFormat("%v / %m")
        rl.addWidget(self.progress_bar)
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #aaa;")
        rl.addWidget(self.lbl_status)

        self.sec_log = CollapsibleSection("Log")
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(150)
        self.sec_log.body_layout.addWidget(self.log_box)
        rl.addWidget(self.sec_log)
        rl.addStretch()

        splitter.addWidget(left)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        self._restore_settings()
        self.combo_template.currentIndexChanged.connect(self.apply_template)
        self.presets.subscribe(lambda: self._fill_presets(self.combo_template.currentData()))
        self.prompt_input.textChanged.connect(self._on_prompt_edited)
        self.refresh_models()
        self.reload_profiles()
        self._update_run_button()
        self.setup_hotkeys()

        self.idle_timer = QTimer(self)
        self.idle_timer.timeout.connect(self._check_idle_unload)
        self.idle_timer.start(60_000)

    # ------------------------------------------------------------------ UI build
    def _build_local_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        self.combo_model = QComboBox()
        self.combo_model.setMinimumWidth(200)
        self.combo_model.currentIndexChanged.connect(self._on_model_changed)
        self.btn_refresh = QPushButton("🔄")
        self.btn_refresh.setFixedWidth(32)
        self.btn_refresh.setToolTip("Rescan model folders")
        self.btn_refresh.clicked.connect(self.refresh_models)
        self.btn_browse_path = QPushButton("📂")
        self.btn_browse_path.setFixedWidth(32)
        self.btn_browse_path.setToolTip("Use a model folder from anywhere on disk")
        self.btn_browse_path.clicked.connect(self.browse_model_path)
        row.addWidget(self.combo_model, 1)
        row.addWidget(self.btn_refresh)
        row.addWidget(self.btn_browse_path)
        lay.addLayout(row)
        self.lbl_model_info = hint_label("")
        lay.addWidget(self.lbl_model_info)
        btns = QHBoxLayout()
        self.btn_download = QPushButton("⬇️ Download")
        self.btn_download.clicked.connect(self.start_download)
        self.btn_unload = QPushButton("🧹 Free VRAM")
        self.btn_unload.setToolTip("Unload the model from GPU memory (it reloads on the next run)")
        self.btn_unload.clicked.connect(self.force_cleanup)
        btns.addWidget(self.btn_download)
        btns.addWidget(self.btn_unload)
        lay.addLayout(btns)
        self.lbl_loaded = hint_label("No model loaded")
        lay.addWidget(self.lbl_loaded)

        self.sec_adv_model = CollapsibleSection("Model options")
        f = QFormLayout()
        self.combo_device = QComboBox()
        self.combo_device.addItem("Auto (best GPU)", "auto")
        self.combo_dtype = QComboBox()
        for label, val in (("Auto", "auto"), ("bfloat16", "bfloat16"), ("float16", "float16"), ("float32", "float32")):
            self.combo_dtype.addItem(label, val)
        self.combo_quant = QComboBox()
        for label, val in (("None", "none"), ("8-bit (bitsandbytes)", "8bit"), ("4-bit (bitsandbytes)", "4bit")):
            self.combo_quant.addItem(label, val)
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 32)
        self.spin_batch.setToolTip("Images processed together on the GPU. 2–8 is usually faster; "
                                   "reduce if you run out of memory (TagScribeR retries one-by-one automatically).")
        self.chk_thinking = QCheckBox("Allow reasoning ('thinking') mode")
        self.chk_thinking.setToolTip("Qwen3.5+ models can reason before answering. Slower and rarely better for captions.")
        self.chk_trust = QCheckBox("Allow custom model code")
        self.chk_trust.setToolTip("Some models ship their own Python code. Only enable for sources you trust.")
        f.addRow("Device:", self.combo_device)
        f.addRow("Precision:", self.combo_dtype)
        f.addRow("Quantization:", self.combo_quant)
        f.addRow("Batch size:", self.spin_batch)
        f.addRow(self.chk_thinking)
        f.addRow(self.chk_trust)
        self.sec_adv_model.body_layout.addLayout(f)
        lay.addWidget(self.sec_adv_model)
        lay.addStretch()
        run_in_background(self._list_devices, self._fill_devices)
        return w

    def _build_api_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        self.combo_presets = QComboBox()
        self.combo_presets.currentIndexChanged.connect(self.apply_api_preset)
        self.btn_save_preset = QPushButton("💾")
        self.btn_save_preset.setFixedWidth(32)
        self.btn_save_preset.setToolTip("Save as profile (the key goes to Windows Credential Manager)")
        self.btn_save_preset.clicked.connect(self.save_api_preset)
        self.btn_del_preset = QPushButton("🗑️")
        self.btn_del_preset.setFixedWidth(32)
        self.btn_del_preset.setToolTip("Delete profile")
        self.btn_del_preset.clicked.connect(self.delete_api_preset)
        row.addWidget(self.combo_presets, 1)
        row.addWidget(self.btn_save_preset)
        row.addWidget(self.btn_del_preset)
        lay.addLayout(row)
        form = QFormLayout()
        self.inp_api_url = QLineEdit("http://127.0.0.1:1234/v1")
        self.inp_api_key = QLineEdit()
        self.inp_api_key.setEchoMode(QLineEdit.Password)
        self.inp_api_key.setPlaceholderText("not needed for local servers")
        model_row = QHBoxLayout()
        self.inp_api_model = QComboBox()
        self.inp_api_model.setEditable(True)
        self.inp_api_model.setToolTip("Model name/ID on the server")
        self.btn_fetch_models = QPushButton("List")
        self.btn_fetch_models.setToolTip("Ask the server which models it offers (also tests the connection)")
        self.btn_fetch_models.clicked.connect(self.fetch_api_models)
        model_row.addWidget(self.inp_api_model, 1)
        model_row.addWidget(self.btn_fetch_models)
        self.spin_concurrency = QSpinBox()
        self.spin_concurrency.setRange(1, 16)
        self.spin_concurrency.setToolTip("Parallel requests. Raise for cloud APIs; keep 1 for a single local GPU.")
        form.addRow("URL:", self.inp_api_url)
        form.addRow("Key:", self.inp_api_key)
        form.addRow("Model:", model_row)
        form.addRow("Parallel:", self.spin_concurrency)
        lay.addLayout(form)
        lay.addWidget(hint_label("Works with LM Studio, Ollama, llama.cpp server, KoboldCpp, vLLM and cloud "
                                 "APIs. Use this for GGUF models (e.g. JoyCaption GGUF in LM Studio)."))
        lay.addStretch()
        return w

    # ------------------------------------------------------------------ settings
    def _restore_settings(self):
        c = self.cfg
        preset = c.get("caption.prompt_template")
        if self.presets.get(preset) is None:
            preset = "My custom prompt" if self.presets.get("My custom prompt") else DEFAULT_PRESET
        self._fill_presets(preset)
        data = self.presets.get(preset) or {}
        working = c.get("caption.working_prompt")
        self.prompt_input.setPlainText(working or data.get("prompt", ""))
        self.system_input.setPlainText(c.get("caption.system_prompt"))
        self.chk_tag_output.setChecked(c.get("caption.tag_output", data.get("output") == "tags"))
        self.inp_subject.setText(c.get("caption.subject", ""))
        self.chk_subject_first.setChecked(c.get("caption.subject_first", False))
        self.chk_tag_hints.setChecked(c.get("caption.tag_hints", False))
        if self.inp_subject.text() or self.chk_tag_hints.isChecked():
            self.sec_subject.toggle.setChecked(True)
        self.spin_tokens.setValue(c.get("caption.max_tokens"))
        self.spin_temp.setValue(c.get("caption.temperature"))
        self.spin_top_p.setValue(c.get("caption.top_p"))
        self.spin_top_k.setValue(c.get("caption.top_k", 0))
        self.spin_rep.setValue(c.get("caption.repetition_penalty", 1.05))
        self.spin_max_side.setValue(c.get("caption.max_image_side"))
        self.spin_batch.setValue(c.get("local.batch_size", 1))
        self.spin_concurrency.setValue(c.get("api.concurrency", 1))
        self.chk_thinking.setChecked(c.get("local.thinking", False))
        self.chk_trust.setChecked(c.get("local.trust_remote_code", False))
        i = self.combo_save.findData(c.get("caption.save_mode", SAVE_OVERWRITE))
        self.combo_save.setCurrentIndex(max(0, i))
        for combo, key in ((self.combo_dtype, "local.dtype"), (self.combo_quant, "local.quantization")):
            j = combo.findData(c.get(key))
            combo.setCurrentIndex(max(0, j))
        self.tab_source.setCurrentIndex(1 if c.get("caption.source", "local") == "api" else 0)

    def _save_settings(self):
        self.cfg.update({
            "caption.prompt_template": self.combo_template.currentData() or DEFAULT_PRESET,
            "caption.working_prompt": self.prompt_input.toPlainText(),
            "caption.tag_output": self.chk_tag_output.isChecked(),
            "caption.subject": self.inp_subject.text().strip(),
            "caption.subject_first": self.chk_subject_first.isChecked(),
            "caption.tag_hints": self.chk_tag_hints.isChecked(),
            "caption.system_prompt": self.system_input.toPlainText(),
            "caption.max_tokens": self.spin_tokens.value(),
            "caption.temperature": self.spin_temp.value(),
            "caption.top_p": self.spin_top_p.value(),
            "caption.top_k": self.spin_top_k.value(),
            "caption.repetition_penalty": self.spin_rep.value(),
            "caption.max_image_side": self.spin_max_side.value(),
            "caption.save_mode": self.combo_save.currentData(),
            "caption.source": "api" if self.tab_source.currentIndex() == 1 else "local",
            "caption.last_model": self.combo_model.currentData() or "",
            "local.batch_size": self.spin_batch.value(),
            "local.dtype": self.combo_dtype.currentData(),
            "local.quantization": self.combo_quant.currentData(),
            "local.device": self.combo_device.currentData() or "auto",
            "local.thinking": self.chk_thinking.isChecked(),
            "local.trust_remote_code": self.chk_trust.isChecked(),
            "api.concurrency": self.spin_concurrency.value(),
        })

    # ------------------------------------------------------------------ hotkeys
    def setup_hotkeys(self):
        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(self.run_process)
        QShortcut(QKeySequence("Ctrl+Enter"), self).activated.connect(self.run_process)
        QShortcut(QKeySequence("Esc"), self).activated.connect(self.abort_process)

    # ------------------------------------------------------------------ devices / models
    @staticmethod
    def _list_devices():
        from core import hardware
        return [(d.label, d.id) for d in hardware.detect_devices()]

    def _fill_devices(self, devices):
        for label, dev_id in devices:
            self.combo_device.addItem(label, dev_id)
        i = self.combo_device.findData(self.cfg.get("local.device"))
        self.combo_device.setCurrentIndex(max(0, i))

    def refresh_models(self):
        self.combo_model.blockSignals(True)
        self.combo_model.clear()
        self.combo_model.addItem("Scanning model folders…")
        self.combo_model.blockSignals(False)
        extra = self.cfg.get("local.model_dirs")
        run_in_background(lambda: model_catalog.discover_local_models(extra), self._fill_models,
                          lambda e: self.log(f"Model scan failed: {e}"))

    def _fill_models(self, local_models):
        last = self.cfg.get("caption.last_model", "")
        self.combo_model.blockSignals(True)
        self.combo_model.clear()
        self._entries = {}
        installed_paths = set()
        if local_models:
            self.combo_model.addItem("── Installed ──")
            self.combo_model.model().item(self.combo_model.count() - 1).setEnabled(False)
            for m in local_models:
                key = str(m.path)
                installed_paths.add(m.path.name.lower())
                self.combo_model.addItem(f"📁 {m.name}  ({m.approx_gb:.1f} GB)", key)
                self._entries[key] = {"path": key, "gb": m.approx_gb,
                                      "notes": f"{m.model_type} · {m.path.parent}", "catalog": None}
        custom = self.cfg.get("caption.custom_model_path", "")
        if custom and Path(custom).is_dir() and custom not in self._entries:
            self.combo_model.addItem(f"📁 {Path(custom).name} (custom)", custom)
            self._entries[custom] = {"path": custom, "gb": 0, "notes": custom, "catalog": None}
        self.combo_model.addItem("── Download ──")
        self.combo_model.model().item(self.combo_model.count() - 1).setEnabled(False)
        for c in model_catalog.CATALOG:
            local = model_catalog.catalog_local_path(c.repo_id)
            cached = local is None and model_catalog.hf_fully_cached(c.repo_id)
            if local is not None and str(local) in self._entries:
                self._entries[str(local)]["catalog"] = c
                continue
            target = str(local) if local else c.repo_id
            mark = "✅" if (local or cached) else "☁️"
            self.combo_model.addItem(f"{mark} {c.label}  ({c.approx_gb:.0f} GB)", target)
            self._entries[target] = {"path": target, "gb": c.approx_gb, "notes": c.notes, "catalog": c,
                                     "downloaded": bool(local or cached)}
        self.combo_model.blockSignals(False)
        idx = self.combo_model.findData(last) if last else -1
        if idx < 0:
            idx = next((i for i in range(self.combo_model.count()) if self.combo_model.itemData(i)), 0)
        self.combo_model.setCurrentIndex(idx)
        self._on_model_changed()
        self.log(f"Found {len(local_models)} local model(s) in: " +
                 ", ".join(str(d) for d in model_catalog.model_search_dirs(self.cfg.get("local.model_dirs"))))

    def _current_entry(self) -> dict | None:
        return getattr(self, "_entries", {}).get(self.combo_model.currentData())

    def _on_model_changed(self, *_):
        e = self._current_entry()
        if not e:
            self.lbl_model_info.setText("")
            self.btn_download.setEnabled(False)
            return
        is_remote = not Path(e["path"]).is_dir() and not e.get("downloaded", False)
        self.btn_download.setEnabled(is_remote)
        self.btn_download.setText("⬇️ Download" if is_remote else "✅ Available")
        vram = ""
        try:
            from core import hardware
            dev = hardware.detect_devices()[0] if hardware.detect_devices.cache_info().currsize else None
            if dev and dev.total_memory and e["gb"]:
                fits = e["gb"] + 1.5 <= dev.total_memory / 2**30
                vram = "  ✓ fits your GPU" if fits else "  ⚠ likely too large for your GPU in bf16"
        except Exception:
            pass
        self.lbl_model_info.setText((e["notes"] or "") + vram)
        self.btn_model_defaults.setEnabled(e.get("catalog") is not None)

    def apply_model_defaults(self):
        e = self._current_entry()
        c = e.get("catalog") if e else None
        if not c:
            return
        self.spin_temp.setValue(c.temperature)
        self.spin_top_p.setValue(c.top_p)
        self.spin_top_k.setValue(c.top_k)
        self.log(f"Applied recommended sampling for {c.label}.")

    def browse_model_path(self):
        folder = QFileDialog.getExistingDirectory(self, "Select a model folder (containing config.json)")
        if not folder:
            return
        if not (Path(folder) / "config.json").is_file():
            QMessageBox.warning(self, "Not a model folder", "That folder has no config.json — pick the folder "
                                "that directly contains the model files.")
            return
        self.cfg.set("caption.custom_model_path", folder)
        self.cfg.set("caption.last_model", folder)
        self.refresh_models()

    def start_download(self):
        e = self._current_entry()
        if not e or not e.get("catalog"):
            return
        repo = e["catalog"].repo_id
        self.btn_download.setEnabled(False)
        self.btn_download.setText("Checking size…")

        def on_size(size):
            size_txt = f"{size / 2**30:.1f} GB" if size else f"about {e['gb']:.0f} GB"
            target = paths.DEFAULT_MODELS_DIR / repo.split("/")[-1]
            if not confirm(self, "Download model",
                           f"Download {e['catalog'].label} ({size_txt}) from Hugging Face into\n{target}?"):
                self._on_model_changed()
                return
            self.btn_download.setText("Downloading…")
            self.log(f"Downloading {repo} → {target} ({size_txt}). This can take a while.")
            self.set_status(f"Downloading {e['catalog'].label} ({size_txt})…")

            def done(_):
                self.log("✅ Download complete.")
                self.set_status("Download complete")
                self.cfg.set("caption.last_model", str(target))
                self.refresh_models()

            def failed(err):
                self.log(f"❌ Download failed: {err}")
                self.set_status("Download failed — partial files are resumed next time")
                self._on_model_changed()

            run_in_background(lambda: model_catalog.download_model(repo, target), done, failed)

        run_in_background(lambda: model_catalog.remote_size_bytes(repo), on_size, lambda _: on_size(None))

    def force_cleanup(self):
        if manager().unload_all():
            self.log("🧹 Models unloaded; GPU memory released.")
            self.lbl_loaded.setText("No model loaded")
        else:
            QMessageBox.information(self, "Busy", "A job is running. Abort it first, then free VRAM.")

    def _check_idle_unload(self):
        minutes = self.cfg.get("local.idle_unload_minutes")
        if manager().unload_if_idle(minutes):
            self.log(f"Model unloaded after {minutes} idle minutes.")
            self.lbl_loaded.setText("No model loaded")

    def cleanup_worker(self):
        """Called on app exit."""
        if self.worker:
            self.worker.cancel()

    # ------------------------------------------------------------------ API profiles
    def reload_profiles(self):
        self.profiles.load()
        self.combo_presets.blockSignals(True)
        self.combo_presets.clear()
        self.combo_presets.addItem("Select profile…")
        self.combo_presets.addItems(sorted(self.profiles.profiles))
        self.combo_presets.blockSignals(False)
        last = self.cfg.get("api.last_profile", "")
        if last in self.profiles.profiles:
            self.combo_presets.setCurrentText(last)

    def apply_api_preset(self):
        p = self.profiles.profiles.get(self.combo_presets.currentText())
        if p:
            self.inp_api_url.setText(p.base_url)
            self.inp_api_key.setText(p.api_key)
            self.inp_api_model.setCurrentText(p.model)
            self.cfg.set("api.last_profile", p.name)

    def save_api_preset(self):
        current = self.combo_presets.currentText()
        default = current if current in self.profiles.profiles else ""
        name, ok = QInputDialog.getText(self, "Save API profile", "Profile name:", text=default)
        if not ok or not name.strip():
            return
        self.profiles.upsert(ApiProfile(name=name.strip(), base_url=self.inp_api_url.text().strip(),
                                        model=self.inp_api_model.currentText().strip(),
                                        api_key=self.inp_api_key.text()))
        self.reload_profiles()
        self.combo_presets.setCurrentText(name.strip())

    def delete_api_preset(self):
        name = self.combo_presets.currentText()
        if name in self.profiles.profiles and confirm(self, "Delete profile", f"Delete API profile '{name}'?"):
            self.profiles.delete(name)
            self.reload_profiles()

    def fetch_api_models(self):
        url, key = self.inp_api_url.text().strip(), self.inp_api_key.text() or "not-needed"
        self.btn_fetch_models.setEnabled(False)

        def fetch():
            from openai import OpenAI
            client = OpenAI(base_url=url, api_key=key, timeout=15, max_retries=0)
            return sorted(m.id for m in client.models.list())

        def done(ids):
            self.btn_fetch_models.setEnabled(True)
            current = self.inp_api_model.currentText()
            self.inp_api_model.clear()
            self.inp_api_model.addItems(ids)
            if current:
                self.inp_api_model.setCurrentText(current)
            self.log(f"✅ Connected: {len(ids)} model(s) available at {url}")

        def failed(err):
            self.btn_fetch_models.setEnabled(True)
            self.log(f"❌ Could not list models at {url}: {err}")
            QMessageBox.warning(self, "Connection failed", f"Could not reach {url}\n\n{err}")

        run_in_background(fetch, done, failed)

    # ------------------------------------------------------------------ prompt
    def _fill_presets(self, select: str | None = None):
        self.combo_template.blockSignals(True)
        self.combo_template.clear()
        for name in self.presets.names():
            self.combo_template.addItem(name if self.presets.is_builtin(name) else f"★ {name}", name)
        i = self.combo_template.findData(select) if select else -1
        self.combo_template.setCurrentIndex(i if i >= 0 else max(0, self.combo_template.findData(DEFAULT_PRESET)))
        self.combo_template.blockSignals(False)
        self._update_preset_state()

    def apply_template(self):
        """Load the chosen preset into the editor (and its saved generation settings, if any)."""
        name = self.combo_template.currentData()
        data = self.presets.get(name) if name else None
        if not data:
            return
        self.prompt_input.blockSignals(True)
        self.prompt_input.setPlainText(data.get("prompt", ""))
        self.prompt_input.blockSignals(False)
        if not self.presets.is_builtin(name):
            self.system_input.setPlainText(data.get("system_prompt", ""))  # built-ins leave yours alone
        self.chk_tag_output.setChecked(data.get("output") == "tags")
        for key, widget in (("max_tokens", self.spin_tokens), ("temperature", self.spin_temp),
                            ("top_p", self.spin_top_p), ("top_k", self.spin_top_k),
                            ("repetition_penalty", self.spin_rep)):
            if key in data:
                widget.setValue(data[key])
        if "save_mode" in data:
            i = self.combo_save.findData(data["save_mode"])
            if i >= 0:
                self.combo_save.setCurrentIndex(i)
        self._update_preset_state()

    def _on_prompt_edited(self):
        self._update_preset_state()

    def _update_preset_state(self):
        if not hasattr(self, "lbl_preset_state"):
            return
        name = self.combo_template.currentData()
        data = self.presets.get(name) if name else None
        builtin = bool(name) and self.presets.is_builtin(name)
        edited = bool(data) and (self.prompt_input.toPlainText() != data.get("prompt", "") or
                                 (not builtin and self.system_input.toPlainText() != data.get("system_prompt", "")))
        if edited:
            self.lbl_preset_state.setText("● Edited — 💾 saves it as " + ("a new preset." if builtin else
                                                                          "this preset or a new one."))
        else:
            self.lbl_preset_state.setText("Built-in preset" if builtin else ("Your preset" if data else ""))

    def _current_preset_data(self, include_generation: bool) -> dict:
        data = {"prompt": self.prompt_input.toPlainText().strip(),
                "system_prompt": self.system_input.toPlainText().strip(),
                "output": "tags" if self.chk_tag_output.isChecked() else "prose"}
        if include_generation:
            data.update(max_tokens=self.spin_tokens.value(), temperature=self.spin_temp.value(),
                        top_p=self.spin_top_p.value(), top_k=self.spin_top_k.value(),
                        repetition_penalty=self.spin_rep.value(), save_mode=self.combo_save.currentData())
        return data

    def save_preset(self):
        if not self.prompt_input.toPlainText().strip():
            QMessageBox.information(self, "Save preset", "Write some instructions first.")
            return
        current = self.combo_template.currentData() or ""
        suggested = current if current and not self.presets.is_builtin(current) else \
            self.presets.unique_name(f"{current} (mine)" if current else "My preset")
        dlg = QDialog(self)
        dlg.setWindowTitle("Save caption preset")
        form = QFormLayout(dlg)
        name_edit = QLineEdit(suggested)
        name_edit.selectAll()
        chk_gen = QCheckBox("Also save generation settings (tokens, temperature, sampling, existing-caption mode)")
        chk_gen.setChecked(bool(self.presets.get(current) and "temperature" in self.presets.get(current)))
        form.addRow("Name:", name_edit)
        form.addRow(chk_gen)
        form.addRow(hint_label("Your presets are stored in user_data\\caption_presets.json and stay until you delete "
                               "them. Built-in presets can't be overwritten."))
        btns = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)
        if not dlg.exec():
            return
        name = name_edit.text().strip()
        if self.presets.get(name) is not None and not self.presets.is_builtin(name) and name != current:
            if not confirm(self, "Replace preset?", f"A preset named '{name}' already exists. Replace it?"):
                return
        try:
            saved = self.presets.save(name, self._current_preset_data(chk_gen.isChecked()))
        except PresetError as e:
            QMessageBox.warning(self, "Save preset", str(e))
            return
        self._fill_presets(saved)
        self.log(f"Saved caption preset '{saved}'.")

    def _fill_preset_menu(self):
        menu = self.btn_preset_menu.menu()
        menu.clear()
        name = self.combo_template.currentData()
        mine = bool(name) and not self.presets.is_builtin(name)
        a = menu.addAction("Rename preset…", self.rename_preset)
        a.setEnabled(mine)
        a = menu.addAction("Delete preset…", self.delete_preset)
        a.setEnabled(mine)
        menu.addAction("Revert text to saved preset", self.apply_template)
        menu.addSeparator()
        menu.addAction("Import presets…", self.import_presets)
        a = menu.addAction("Export my presets…", self.export_presets)
        a.setEnabled(bool(self.presets.user))

    def rename_preset(self):
        old = self.combo_template.currentData()
        new, ok = QInputDialog.getText(self, "Rename preset", "New name:", text=old)
        if ok and new.strip() and new.strip() != old:
            try:
                self._fill_presets(self.presets.rename(old, new))
            except PresetError as e:
                QMessageBox.warning(self, "Rename preset", str(e))

    def delete_preset(self):
        name = self.combo_template.currentData()
        if name and confirm(self, "Delete preset", f"Delete your preset '{name}'? This can't be undone.",
                            destructive=True):
            try:
                self.presets.delete(name)
            except PresetError as e:
                QMessageBox.warning(self, "Delete preset", str(e))
                return
            self._fill_presets(DEFAULT_PRESET)
            self.apply_template()

    def import_presets(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import caption presets", "", "TagScribeR presets (*.json)")
        if not path:
            return
        try:
            added = self.presets.import_file(Path(path))
        except PresetError as e:
            QMessageBox.warning(self, "Import presets", str(e))
            return
        QMessageBox.information(self, "Import presets",
                                f"Imported {len(added)} preset(s)." + ("\n" + "\n".join(added) if added else
                                                                     "\nAll of them were already present."))

    def export_presets(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export my caption presets", "tagscriber_caption_presets.json",
                                              "TagScribeR presets (*.json)")
        if path:
            n = self.presets.export(Path(path))
            self.log(f"Exported {n} preset(s) to {path}")

    # ------------------------------------------------------------------ grid
    def load_folder(self):
        self.browser.select_folder()

    def open_folder(self, folder: str):
        if self.ctx.confirm_discard(self):
            self.ctx.open_folder(folder, self.browser.chk_recursive.isChecked(), self)

    def on_selection(self, keys: list[str]):
        self.selected_paths = keys
        self._update_run_button()

    def select_all(self):
        self.browser.select_all()

    def select_uncaptioned(self):
        self.browser.select_matching("missing:caption")

    def has_unsaved_changes(self) -> bool:
        return self.ctx.has_unsaved()

    # ------------------------------------------------------------------ run
    def _update_run_button(self):
        if self.worker is not None:
            self.btn_run.setText("🛑 Abort  (Esc)")
            self.btn_run.setStyleSheet("background-color: #ff4757; font-weight: bold; font-size: 14px;")
        else:
            n = len(self.selected_paths)
            self.btn_run.setText(f"🚀 Caption {n} Selected  (Ctrl+Enter)" if n else "Select images to caption")
            self.btn_run.setStyleSheet("background-color: #d63031; font-weight: bold; font-size: 14px;")
        self.browser.btn_open.setEnabled(self.worker is None)

    def abort_process(self):
        if self.worker is not None:
            self.worker.cancel()
            self.set_status("Stopping after the current image…")
            self.log("🛑 Abort requested.")

    def toggle_process_state(self):
        if self.worker is not None:
            self.abort_process()
        else:
            self.run_process()

    def _build_spec(self) -> ProviderSpec | None:
        if self.tab_source.currentIndex() == 0:
            e = self._current_entry()
            if not e:
                QMessageBox.information(self, "No model", "Choose a model first.")
                return None
            if not Path(e["path"]).is_dir() and not e.get("downloaded"):
                QMessageBox.information(self, "Download needed",
                                        f"{e['catalog'].label if e.get('catalog') else e['path']} isn't downloaded "
                                        "yet. Click Download first.")
                return None
            return ProviderSpec.make(
                "transformers", e["path"], device=self.combo_device.currentData() or "auto",
                dtype=self.combo_dtype.currentData(), quantization=self.combo_quant.currentData(),
                batch_size=self.spin_batch.value(), thinking=self.chk_thinking.isChecked(),
                trust_remote_code=self.chk_trust.isChecked(), attention=self.cfg.get("local.attention"))
        url = self.inp_api_url.text().strip()
        model = self.inp_api_model.currentText().strip()
        if not url or not model:
            QMessageBox.information(self, "API settings", "Enter the server URL and model name.")
            return None
        return ProviderSpec.make("openai", model, base_url=url, api_key=self.inp_api_key.text(),
                                 concurrency=self.spin_concurrency.value(), timeout=180.0, max_retries=3)

    def run_process(self):
        if self.worker is not None:
            return
        if not self.selected_paths:
            self.set_status("Select some images first.")
            return
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.information(self, "Instructions", "Enter instructions for the model.")
            return
        spec = self._build_spec()
        if spec is None:
            return
        mode = self.combo_save.currentData()
        ordered = list(self.selected_paths)  # grid order
        session = self.ctx.session
        self.ctx.commit_pending_edits()
        dirty = [k for k in ordered if session.get(k) and session.get(k).dirty]
        if dirty:
            # Merge modes must build on what the user sees, so unsaved edits are saved first.
            if not confirm(self, "Unsaved edits",
                           f"{len(dirty)} selected caption(s) have unsaved edits. Save them before captioning?"):
                return
            if not self.ctx.save(self, dirty):
                return
        if mode == SAVE_OVERWRITE:
            existing = sum(1 for k in ordered if session.get(k) and session.get(k).has_caption)
            if existing and not confirm(
                    self, "Overwrite captions?",
                    f"{existing} of the {len(ordered)} selected images already have captions, which will be "
                    "replaced.\n\nPrevious versions are backed up to user_data\\caption_backups.\n\nContinue?",
                    destructive=True):
                return
        self._save_settings()
        request = CaptionRequest(
            prompt=prompt, system_prompt=self.system_input.toPlainText(),
            params=GenerationParams(max_new_tokens=self.spin_tokens.value(), temperature=self.spin_temp.value(),
                                    top_p=self.spin_top_p.value(), top_k=self.spin_top_k.value(),
                                    repetition_penalty=self.spin_rep.value()),
            max_image_side=self.spin_max_side.value(), strip_thinking=self.cfg.get("caption.strip_thinking"))
        tag_mode = self.chk_tag_output.isChecked()
        subject = self.inp_subject.text().strip()
        prompts = None
        if subject or self.chk_tag_hints.isChecked():
            from inference.prompts import build_prompt
            prompts = {}
            for k in ordered:
                e = session.get(k)
                hints = e.text if (self.chk_tag_hints.isChecked() and e is not None) else ""
                prompts[k] = build_prompt(prompt, subject=subject,
                                          start_with_subject=self.chk_subject_first.isChecked(), tags=hints)
        job = BatchJob(paths=[Path(p) for p in ordered], spec=spec, request=request, save_mode=mode,
                       tag_mode=tag_mode, title="Captioning", prompts=prompts,
                       ensure_prefix=subject if (subject and self.chk_subject_first.isChecked()) else "")
        self.ctx.set_job_state(ordered, JOB_QUEUED)
        self._job_keys = ordered
        self._job_generation = self.ctx.generation
        self.progress_bar.setMaximum(len(ordered))
        self.progress_bar.setValue(0)
        self.worker = start_job(job)
        self.worker.status.connect(self.set_status)
        self.worker.status.connect(self.log)
        self.worker.progress.connect(lambda done, total: self.progress_bar.setValue(done))
        self.worker.item_done.connect(self.on_item_done)
        self.worker.item_failed.connect(self.on_item_failed)
        self.worker.item_skipped.connect(lambda p, r: self._mark(p, None))
        self.worker.items_started.connect(lambda keys: self._mark_many(keys, JOB_WORKING))
        self.worker.finished.connect(self.on_job_finished)
        self._update_run_button()
        self.log(f"▶ Captioning {len(ordered)} image(s) with {spec.model}")

    def _same_folder(self) -> bool:
        return self.ctx.generation == getattr(self, "_job_generation", -1)

    def _mark(self, path, state):
        if self._same_folder():
            self.ctx.set_job_state([path], state)

    def _mark_many(self, keys, state):
        if self._same_folder():
            self.ctx.set_job_state(keys, state)

    def on_item_done(self, path, text):
        if self._same_folder():
            self.ctx.reload_from_disk([path])  # the worker already saved it safely
            self.ctx.set_job_state([path], None)

    def on_item_failed(self, path, reason):
        self._mark(path, JOB_FAILED)
        self.log(f"❌ {Path(path).name}: {reason}")

    def on_job_finished(self, summary):
        self.worker = None
        if self._same_folder():
            leftover = [k for k in getattr(self, "_job_keys", []) if self.ctx.job_state.get(k) in (JOB_QUEUED, JOB_WORKING)]
            self.ctx.set_job_state(leftover, None)
        self._update_run_button()
        self.set_status(summary.text())
        self.log(("✅ " if not summary.failed and not summary.fatal else "⚠ ") + summary.text())
        name = manager().loaded_model_name()
        self.lbl_loaded.setText(f"Loaded: {name}" if name else "No model loaded")
        show_job_summary(self, summary)

    # ------------------------------------------------------------------ manual actions
    def save_edits(self):
        return self.browser.save()

    # ------------------------------------------------------------------ log
    def set_status(self, msg: str):
        self.lbl_status.setText(msg)

    def log(self, msg: str):
        self.log_box.append(msg)

