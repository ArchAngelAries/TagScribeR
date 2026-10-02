"""Auto Caption tab: caption images with local VLMs or OpenAI-compatible APIs.

All inference runs through ``inference`` (providers + ModelManager +
BatchWorker); this module is only UI. Loaded models are reused across runs.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from PySide6.QtCore import Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
                               QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                               QMessageBox, QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter,
                               QTabWidget, QTextEdit, QVBoxLayout, QWidget)

from core import caption_io, dataset, fileops, paths
from core.config import settings
from core.secrets import ApiProfile, ApiProfileStore
from inference import models as model_catalog
from inference.base import CaptionRequest, GenerationParams, ProviderSpec
from inference.manager import manager
from inference.prompts import DEFAULT_PRESET, PROMPT_PRESETS
from inference.worker import (SAVE_APPEND, SAVE_OVERWRITE, SAVE_PREPEND, SAVE_SKIP_EXISTING, BatchJob,
                              start_job)
from tabs.common import (CollapsibleSection, ThumbnailWorker, confirm, hint_label, run_in_background,
                         show_job_summary)

log = logging.getLogger(__name__)

SAVE_MODES = [
    ("Overwrite existing captions", SAVE_OVERWRITE),
    ("Skip images that already have a caption", SAVE_SKIP_EXISTING),
    ("Append to existing captions", SAVE_APPEND),
    ("Prepend to existing captions", SAVE_PREPEND),
]


class CaptionCard(QFrame):
    selection_changed = Signal(str, bool)

    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.is_selected = False
        self.setFixedSize(240, 340)
        self.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        self.lbl_image = QLabel("Loading…")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("background-color: #1e1e1e; border-radius: 4px;")
        self.lbl_image.setFixedHeight(180)
        self.txt_caption = QTextEdit()
        self.txt_caption.setPlaceholderText("No caption yet")
        self.txt_caption.setStyleSheet("background-color: #151515; border: 1px solid #333; color: #ccc;")
        layout.addWidget(self.lbl_image)
        layout.addWidget(self.txt_caption)
        self.reload_text()
        self.update_style()

    def set_image(self, _path, pix):
        if not pix.isNull():
            self.lbl_image.setPixmap(pix)
            self.lbl_image.setText("")
        else:
            self.lbl_image.setText("Unreadable image")

    def reload_text(self):
        self.txt_caption.setPlainText(caption_io.read_caption(self.path))
        self.txt_caption.document().setModified(False)

    def set_status(self, state: str):
        colors = {"working": "#fdcb6e", "failed": "#d63031", "done": None}
        self._status_color = colors.get(state)
        self.update_style()

    def show_result(self, text: str):
        self.txt_caption.setPlainText(text)
        self.txt_caption.document().setModified(False)

    @property
    def dirty(self) -> bool:
        return self.txt_caption.document().isModified()

    def save_text(self) -> bool:
        """Persist manual edits. Returns True if the file changed."""
        changed = caption_io.write_caption(self.path, self.txt_caption.toPlainText())
        self.txt_caption.document().setModified(False)
        return changed

    def toggle_selection(self, state=None):
        self.is_selected = (not self.is_selected) if state is None else state
        self.update_style()
        self.selection_changed.emit(self.path, self.is_selected)

    def update_style(self):
        border = getattr(self, "_status_color", None) or ("#00b894" if self.is_selected else "transparent")
        self.setStyleSheet(f"CaptionCard {{ background-color: #2b2b2b; border: 2px solid {border}; border-radius: 8px; }}")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.toggle_selection()
        super().mousePressEvent(event)


class CaptionTab(QWidget):
    def __init__(self):
        super().__init__()
        self.cfg = settings()
        self.cards: dict[str, CaptionCard] = {}
        self.selected_paths: set[str] = set()
        self.thumb_pool = QThreadPool(self)
        self.thumb_pool.setMaxThreadCount(max(2, (os.cpu_count() or 4) // 2))
        self.worker = None
        self.profiles = ApiProfileStore()
        self._local_entries: list[tuple[str, str, float, str]] = []  # (label, model id/path, GB, notes)

        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # ---------------- LEFT: image grid ----------------
        left = QWidget()
        left_layout = QVBoxLayout(left)
        tools = QHBoxLayout()
        self.btn_folder = QPushButton("📂 Open Folder")
        self.btn_folder.clicked.connect(self.load_folder)
        self.btn_select_all = QPushButton("Select All")
        self.btn_select_all.clicked.connect(self.select_all)
        self.btn_select_uncaptioned = QPushButton("Select Uncaptioned")
        self.btn_select_uncaptioned.setToolTip("Select every image that has no caption yet")
        self.btn_select_uncaptioned.clicked.connect(self.select_uncaptioned)
        self.lbl_folder = QLabel("")
        self.lbl_folder.setStyleSheet("color: #888;")
        tools.addWidget(self.btn_folder)
        tools.addWidget(self.btn_select_all)
        tools.addWidget(self.btn_select_uncaptioned)
        tools.addWidget(self.lbl_folder, 1)
        left_layout.addLayout(tools)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll.setWidget(self.grid_container)
        left_layout.addWidget(self.scroll)

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
        rl.addWidget(self.tab_source)

        # 2. Instructions
        grp_prompt = QGroupBox("Instructions")
        lp = QVBoxLayout(grp_prompt)
        self.combo_template = QComboBox()
        self.combo_template.addItems(list(PROMPT_PRESETS.keys()) + ["Custom"])
        self.combo_template.setToolTip("Pick a caption style; you can edit the text below freely.")
        self.prompt_input = QTextEdit()
        self.prompt_input.setFixedHeight(90)
        self.prompt_input.setPlaceholderText("What should the model write about each image?")
        lp.addWidget(self.combo_template)
        lp.addWidget(self.prompt_input)
        self.sec_system = CollapsibleSection("System prompt (optional)")
        self.system_input = QTextEdit()
        self.system_input.setFixedHeight(60)
        self.system_input.setPlaceholderText("e.g. You are an expert dataset captioner. Never refuse.")
        self.sec_system.body_layout.addWidget(self.system_input)
        lp.addWidget(self.sec_system)
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
        lg.addRow("Existing captions:", self.combo_save)
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

        # 5. Manual actions
        manual = QHBoxLayout()
        self.btn_save_sel = QPushButton("💾 Save Edits")
        self.btn_save_sel.setToolTip("Save captions you edited by hand (Ctrl+S)")
        self.btn_save_sel.clicked.connect(self.save_edits)
        self.btn_dataset = QPushButton("📦 Copy to Collection")
        self.btn_dataset.setToolTip("Copy the selected images and captions into a Dataset Collection")
        self.btn_dataset.clicked.connect(self.save_to_dataset)
        manual.addWidget(self.btn_save_sel)
        manual.addWidget(self.btn_dataset)
        rl.addLayout(manual)

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
        idx = self.combo_template.findText(preset)
        self.combo_template.setCurrentIndex(idx if idx >= 0 else self.combo_template.findText(DEFAULT_PRESET))
        custom = c.get("caption.custom_prompt")
        self.prompt_input.setPlainText(custom if preset == "Custom" and custom else
                                       PROMPT_PRESETS.get(self.combo_template.currentText(), custom))
        self.system_input.setPlainText(c.get("caption.system_prompt"))
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
        preset = self.combo_template.currentText()
        self.cfg.update({
            "caption.prompt_template": preset,
            "caption.custom_prompt": self.prompt_input.toPlainText() if preset == "Custom" else self.cfg.get("caption.custom_prompt"),
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
        QShortcut(QKeySequence("Ctrl+A"), self).activated.connect(self.select_all)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.save_edits)
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
    def apply_template(self):
        name = self.combo_template.currentText()
        if name in PROMPT_PRESETS:
            self.prompt_input.blockSignals(True)
            self.prompt_input.setPlainText(PROMPT_PRESETS[name])
            self.prompt_input.blockSignals(False)
        elif name == "Custom":
            self.prompt_input.setPlainText(self.cfg.get("caption.custom_prompt"))

    def _on_prompt_edited(self):
        name = self.combo_template.currentText()
        if name in PROMPT_PRESETS and self.prompt_input.toPlainText() != PROMPT_PRESETS[name]:
            self.combo_template.blockSignals(True)
            self.combo_template.setCurrentText("Custom")
            self.combo_template.blockSignals(False)

    # ------------------------------------------------------------------ grid
    def load_folder(self):
        if self._has_unsaved() and not confirm(self, "Unsaved edits", "You have unsaved caption edits. "
                                               "Discard them and open another folder?", destructive=True):
            return
        start = self.cfg.get("ui.last_folder") or ""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", start)
        if folder:
            self.open_folder(folder)

    def open_folder(self, folder: str):
        self.thumb_pool.clear()
        while self.grid_layout.count():
            w = self.grid_layout.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.cards.clear()
        self.selected_paths.clear()
        try:
            files = dataset.scan_images(folder)
        except OSError as e:
            QMessageBox.critical(self, "Error", f"Could not read folder:\n{e}")
            return
        self.cfg.set("ui.last_folder", folder)
        self.lbl_folder.setText(f"{folder}  —  {len(files)} images")
        cols = 3
        for i, path in enumerate(files):
            p = str(path)
            card = CaptionCard(p)
            card.selection_changed.connect(self.on_selection)
            self.grid_layout.addWidget(card, i // cols, i % cols)
            self.cards[p] = card
            worker = ThumbnailWorker(p, (230, 170))
            worker.signals.loaded.connect(card.set_image)
            self.thumb_pool.start(worker)
        self._update_run_button()

    def on_selection(self, path, is_selected):
        if is_selected:
            self.selected_paths.add(path)
        else:
            self.selected_paths.discard(path)
        self._update_run_button()

    def select_all(self):
        target = not (self.cards and len(self.selected_paths) == len(self.cards))
        for card in self.cards.values():
            card.toggle_selection(target)

    def select_uncaptioned(self):
        for card in self.cards.values():
            card.toggle_selection(not card.txt_caption.toPlainText().strip())

    def _has_unsaved(self) -> bool:
        return any(c.dirty for c in self.cards.values())

    def has_unsaved_changes(self) -> bool:
        return self._has_unsaved()

    # ------------------------------------------------------------------ run
    def _update_run_button(self):
        if self.worker is not None:
            self.btn_run.setText("🛑 Abort  (Esc)")
            self.btn_run.setStyleSheet("background-color: #ff4757; font-weight: bold; font-size: 14px;")
        else:
            n = len(self.selected_paths)
            self.btn_run.setText(f"🚀 Caption {n} Selected  (Ctrl+Enter)" if n else "Select images to caption")
            self.btn_run.setStyleSheet("background-color: #d63031; font-weight: bold; font-size: 14px;")
        self.btn_folder.setEnabled(self.worker is None)

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
        ordered = [p for p in self.cards if p in self.selected_paths]  # grid order
        # Hand-edited captions are saved first so merge modes build on what the user sees.
        for p in ordered:
            if self.cards[p].dirty:
                try:
                    self.cards[p].save_text()
                except OSError as e:
                    self.log(f"Could not save edit for {Path(p).name}: {e}")
        if mode == SAVE_OVERWRITE:
            existing = sum(1 for p in ordered if self.cards[p].txt_caption.toPlainText().strip())
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
        tag_mode = self.combo_template.currentText() in ("Booru Tags", "Stable Diffusion Tags")
        job = BatchJob(paths=[Path(p) for p in ordered], spec=spec, request=request, save_mode=mode,
                       tag_mode=tag_mode, title="Captioning")
        for p in ordered:
            self.cards[p].set_status("working")
        self.progress_bar.setMaximum(len(ordered))
        self.progress_bar.setValue(0)
        self.worker = start_job(job)
        self.worker.status.connect(self.set_status)
        self.worker.status.connect(self.log)
        self.worker.progress.connect(lambda done, total: self.progress_bar.setValue(done))
        self.worker.item_done.connect(self.on_item_done)
        self.worker.item_failed.connect(self.on_item_failed)
        self.worker.item_skipped.connect(lambda p, r: self._card_status(p, "done"))
        self.worker.finished.connect(self.on_job_finished)
        self._update_run_button()
        self.log(f"▶ Captioning {len(ordered)} image(s) with {spec.model}")

    def _card_status(self, path, state):
        if path in self.cards:
            self.cards[path].set_status(state)

    def on_item_done(self, path, text):
        card = self.cards.get(path)
        if card:
            card.show_result(text)
            card.set_status("done")

    def on_item_failed(self, path, reason):
        self._card_status(path, "failed")
        self.log(f"❌ {Path(path).name}: {reason}")

    def on_job_finished(self, summary):
        self.worker = None
        for card in self.cards.values():
            if getattr(card, "_status_color", None) == "#fdcb6e":  # still marked working (cancelled)
                card.set_status("done")
        self._update_run_button()
        self.set_status(summary.text())
        self.log(("✅ " if not summary.failed and not summary.fatal else "⚠ ") + summary.text())
        name = manager().loaded_model_name()
        self.lbl_loaded.setText(f"Loaded: {name}" if name else "No model loaded")
        show_job_summary(self, summary)

    # ------------------------------------------------------------------ manual actions
    def save_edits(self):
        changed = 0
        try:
            for card in self.cards.values():
                if card.dirty and card.save_text():
                    changed += 1
        except OSError as e:
            QMessageBox.critical(self, "Save failed", str(e))
        self.set_status(f"Saved {changed} edited caption(s)." if changed else "No unsaved edits.")

    def save_to_dataset(self):
        if not self.selected_paths:
            QMessageBox.information(self, "No selection", "Select images first.")
            return
        root = Path(self.cfg.get("paths.collections_dir") or paths.DEFAULT_COLLECTIONS_DIR)
        existing = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
        name, ok = QInputDialog.getItem(self, "Copy to collection", "Collection name (new or existing):",
                                        existing, editable=True)
        if not ok or not name.strip():
            return
        for p in self.selected_paths:
            if self.cards[p].dirty:
                self.cards[p].save_text()
        report = fileops.CopyReport()
        for p in sorted(self.selected_paths):
            fileops.copy_image_with_caption(Path(p), root / name.strip(), report=report)
        self.set_status(f"Collection '{name.strip()}': {report.summary()}")
        if report.failed:
            QMessageBox.warning(self, "Some copies failed",
                                "\n".join(f"{Path(p).name}: {e}" for p, e in report.failed[:20]))

    # ------------------------------------------------------------------ log
    def set_status(self, msg: str):
        self.lbl_status.setText(msg)

    def log(self, msg: str):
        self.log_box.append(msg)

