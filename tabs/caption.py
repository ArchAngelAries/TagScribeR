import os
import shutil
import json
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, 
    QLabel, QProgressBar, QComboBox, QFileDialog, QScrollArea, 
    QGridLayout, QFrame, QSplitter, QMessageBox, QCheckBox, QGroupBox, 
    QSpinBox, QDoubleSpinBox, QFormLayout, QInputDialog, QTabWidget, QLineEdit
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QRunnable, QThreadPool, QObject
from PySide6.QtGui import QPixmap, QShortcut, QKeySequence
from core.ai_backend import QwenWorker, DownloadWorker
from core.image_utils import load_thumbnail

API_PRESETS_FILE = "api_presets.json"

KNOWN_MODELS = {
    "Qwen3-VL-8B-Instruct (Best Quality)": "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen3-VL-4B-Instruct (Balanced)": "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen3-VL-2B-Instruct (Fastest)": "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen2.5-VL-3B-Instruct": "Qwen/Qwen2.5-VL-3B-Instruct"
}

PROMPT_TEMPLATES = {
    "Detailed Description": "Describe this image in detail, focusing on visual elements, colors, lighting, and composition.",
    "Stable Diffusion Tags": "Describe this image using comma-separated tags (e.g., 1girl, solo, sunset, detailed background).",
    "Accessibility Caption": "Provide a brief, literal description of the main subject for accessibility purposes.",
    "Short Summary": "Summarize the image in one short sentence."
}

class ThumbnailSignals(QObject):
    loaded = Signal(str, QPixmap)

class ThumbnailWorker(QRunnable):
    def __init__(self, path, size):
        super().__init__()
        self.path = path
        self.size = size
        self.signals = ThumbnailSignals()

    @Slot()
    def run(self):
        pix = load_thumbnail(self.path, self.size)
        self.signals.loaded.emit(self.path, pix)

class CaptionCard(QFrame):
    selection_changed = Signal(str, bool)

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.is_selected = False
        self.setFixedWidth(240)
        self.setFixedHeight(340)
        self.setFrameShape(QFrame.StyledPanel)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5,5,5,5)
        
        self.lbl_image = QLabel("Loading...")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("background-color: #1e1e1e; border-radius: 4px;")
        self.lbl_image.setFixedHeight(180)
        
        self.txt_caption = QTextEdit()
        self.txt_caption.setPlaceholderText("Waiting for AI...")
        self.txt_caption.setStyleSheet("background-color: #151515; border: 1px solid #333; color: #ccc;")
        
        layout.addWidget(self.lbl_image)
        layout.addWidget(self.txt_caption)
        
        self.load_current_text()
        self.update_style()

    def set_image(self, path, pix):
        if not pix.isNull():
            self.lbl_image.setPixmap(pix)
            self.lbl_image.setText("")

    def load_current_text(self):
        txt_path = os.path.splitext(self.path)[0] + ".txt"
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    self.txt_caption.setText(f.read())
            except: pass

    def update_caption_from_ai(self, text):
        self.txt_caption.setText(text)
        self.save_text()

    def save_text(self):
        txt_path = os.path.splitext(self.path)[0] + ".txt"
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(self.txt_caption.toPlainText())
            return True
        except: return False

    def toggle_selection(self, state=None):
        if state is not None:
            self.is_selected = state
        else:
            self.is_selected = not self.is_selected
        self.update_style()
        self.selection_changed.emit(self.path, self.is_selected)

    def update_style(self):
        border = "#00b894" if self.is_selected else "transparent"
        self.setStyleSheet(f"CaptionCard {{ background-color: #2b2b2b; border: 2px solid {border}; border-radius: 8px; }}")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.toggle_selection()
        super().mousePressEvent(event)

class CaptionTab(QWidget):
    def __init__(self):
        super().__init__()
        self.cards = {}
        self.selected_paths = set()
        self.thread_pool = QThreadPool()
        self.is_processing = False 
        self.api_presets = {}
        self.worker = None
        
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # LEFT
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        grid_tools = QHBoxLayout()
        self.btn_folder = QPushButton("📂 Open Folder")
        self.btn_folder.clicked.connect(self.load_folder)
        self.btn_select_all = QPushButton("Select All")
        self.btn_select_all.clicked.connect(self.select_all)
        grid_tools.addWidget(self.btn_folder)
        grid_tools.addWidget(self.btn_select_all)
        grid_tools.addStretch()
        left_layout.addLayout(grid_tools)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll.setWidget(self.grid_container)
        left_layout.addWidget(self.scroll)

        # RIGHT
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_widget = QWidget()
        right_widget.setFixedWidth(340)
        right_layout = QVBoxLayout(right_widget)
        right_scroll.setWidget(right_widget)

        # 1. Model Source Tabs
        self.tab_source = QTabWidget()
        # Increased height to fit new custom path options
        self.tab_source.setFixedHeight(280) 
        
        # TAB 1: Local
        tab_local = QWidget()
        lay_local = QVBoxLayout(tab_local)
        
        # Standard Dropdown Area
        self.combo_model = QComboBox()
        self.combo_model.currentIndexChanged.connect(self.check_download_status)
        btn_box = QHBoxLayout()
        self.btn_refresh = QPushButton("🔄")
        self.btn_refresh.setFixedWidth(30)
        self.btn_refresh.clicked.connect(self.refresh_models)
        self.btn_download = QPushButton("⬇️ Download")
        self.btn_download.clicked.connect(self.start_download)
        self.btn_download.setEnabled(False) 
        self.btn_download.setStyleSheet("background-color: #0984e3; color: white;")
        
        # NEW: Unload Button
        self.btn_unload = QPushButton("🧹 Free VRAM")
        self.btn_unload.setToolTip("Unloads model from GPU memory")
        self.btn_unload.clicked.connect(self.cleanup_worker)
        self.btn_unload.setStyleSheet("background-color: #636e72; color: white;")

        btn_box.addWidget(self.btn_refresh)
        btn_box.addWidget(self.btn_download)
        
        lay_local.addWidget(QLabel("Preset / Downloaded Models:"))
        lay_local.addWidget(self.combo_model)
        lay_local.addLayout(btn_box)
        lay_local.addWidget(self.btn_unload)
        
        # NEW: Custom Path Area
        lay_local.addSpacing(10)
        self.chk_custom_path = QCheckBox("Use External Model Path")
        self.chk_custom_path.toggled.connect(self.toggle_custom_path)
        lay_local.addWidget(self.chk_custom_path)
        
        self.wid_custom_path = QWidget()
        path_box = QHBoxLayout(self.wid_custom_path)
        path_box.setContentsMargins(0,0,0,0)
        self.line_custom_path = QLineEdit()
        self.line_custom_path.setPlaceholderText("C:/AI/Models/Qwen-VL...")
        self.btn_browse_path = QPushButton("📂")
        self.btn_browse_path.setFixedWidth(30)
        self.btn_browse_path.clicked.connect(self.browse_model_path)
        
        path_box.addWidget(self.line_custom_path)
        path_box.addWidget(self.btn_browse_path)
        
        lay_local.addWidget(self.wid_custom_path)
        self.wid_custom_path.setVisible(False) # Hidden by default
        
        lay_local.addStretch()
        
        # TAB 2: API
        tab_api = QWidget()
        lay_api = QVBoxLayout(tab_api)
        
        preset_box = QHBoxLayout()
        self.combo_presets = QComboBox()
        self.combo_presets.addItem("Select Preset...")
        self.combo_presets.currentIndexChanged.connect(self.apply_api_preset)
        self.btn_save_preset = QPushButton("💾")
        self.btn_save_preset.setFixedWidth(30)
        self.btn_save_preset.clicked.connect(self.save_api_preset)
        self.btn_del_preset = QPushButton("🗑️")
        self.btn_del_preset.setFixedWidth(30)
        self.btn_del_preset.setStyleSheet("background-color: #d63031;")
        self.btn_del_preset.clicked.connect(self.delete_api_preset)
        preset_box.addWidget(self.combo_presets)
        preset_box.addWidget(self.btn_save_preset)
        preset_box.addWidget(self.btn_del_preset)
        lay_api.addLayout(preset_box)

        form_api = QFormLayout()
        self.inp_api_url = QLineEdit("http://localhost:1234/v1")
        self.inp_api_key = QLineEdit("lm-studio")
        self.inp_api_key.setEchoMode(QLineEdit.Password)
        self.inp_api_model = QLineEdit("qwen-vl")
        self.inp_api_model.setPlaceholderText("Model ID (e.g. gpt-4o)")
        form_api.addRow("URL:", self.inp_api_url)
        form_api.addRow("Key:", self.inp_api_key)
        form_api.addRow("ID:", self.inp_api_model)
        lay_api.addLayout(form_api)
        
        self.tab_source.addTab(tab_local, "Local (GPU)")
        self.tab_source.addTab(tab_api, "API / Remote")
        
        right_layout.addWidget(self.tab_source)

        # 2. Params
        grp_params = QGroupBox("2. Parameters")
        lyt_params = QFormLayout(grp_params)
        self.spin_tokens = QSpinBox(); self.spin_tokens.setRange(64, 4096); self.spin_tokens.setValue(512)
        self.spin_temp = QDoubleSpinBox(); self.spin_temp.setRange(0.0, 2.0); self.spin_temp.setSingleStep(0.1); self.spin_temp.setValue(0.7)
        self.spin_top_p = QDoubleSpinBox(); self.spin_top_p.setRange(0.0, 1.0); self.spin_top_p.setSingleStep(0.05); self.spin_top_p.setValue(0.9)
        lyt_params.addRow("Max Tokens:", self.spin_tokens)
        lyt_params.addRow("Temperature:", self.spin_temp)
        lyt_params.addRow("Top P:", self.spin_top_p)
        right_layout.addWidget(grp_params)

        # 3. Prompt
        grp_prompt = QGroupBox("3. Prompting")
        lyt_prompt = QVBoxLayout(grp_prompt)
        self.combo_template = QComboBox()
        self.combo_template.addItems(list(PROMPT_TEMPLATES.keys()))
        self.combo_template.currentIndexChanged.connect(self.apply_template)
        self.chk_override = QCheckBox("Enable Manual Override")
        self.chk_override.toggled.connect(self.toggle_prompt_edit)
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Enter custom prompt...")
        self.prompt_input.setFixedHeight(80)
        self.prompt_input.setReadOnly(True) 
        lyt_prompt.addWidget(QLabel("Caption Style:"))
        lyt_prompt.addWidget(self.combo_template)
        lyt_prompt.addWidget(self.chk_override)
        lyt_prompt.addWidget(self.prompt_input)
        right_layout.addWidget(grp_prompt)

        # 4. Actions
        self.btn_run = QPushButton("🚀 Caption Selected")
        self.btn_run.setFixedHeight(50)
        self.btn_run.setStyleSheet("background-color: #d63031; font-weight: bold; font-size: 14px;")
        self.btn_run.clicked.connect(self.toggle_process_state)
        right_layout.addWidget(self.btn_run)

        # Save Actions
        right_layout.addWidget(QLabel("<b>Manual Actions</b>"))
        save_layout = QHBoxLayout()
        self.btn_save_sel = QPushButton("Save Selected")
        self.btn_save_sel.clicked.connect(self.save_selected)
        self.btn_save_all = QPushButton("Save All")
        self.btn_save_all.clicked.connect(self.save_all)
        save_layout.addWidget(self.btn_save_sel)
        save_layout.addWidget(self.btn_save_all)
        right_layout.addLayout(save_layout)
        
        self.btn_dataset = QPushButton("📦 Save Selected to Dataset")
        self.btn_dataset.clicked.connect(self.save_to_dataset)
        self.btn_dataset.setStyleSheet("background-color: #6c5ce7; font-weight: bold;")
        right_layout.addWidget(self.btn_dataset)
        
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("font-weight: bold; color: #aaa;")
        right_layout.addWidget(self.lbl_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.progress_bar)
        
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(150)
        right_layout.addWidget(self.log_box)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        
        self.apply_template()
        self.refresh_models()
        self.load_api_presets()
        self.setup_hotkeys()

    def setup_hotkeys(self):
        QShortcut(QKeySequence("Ctrl+A"), self).activated.connect(self.select_all)
        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(self.run_process)
        QShortcut(QKeySequence("Ctrl+Enter"), self).activated.connect(self.run_process)
        QShortcut(QKeySequence("Esc"), self).activated.connect(self.abort_process)

    # --- CUSTOM PATH LOGIC ---
    def toggle_custom_path(self, checked):
        self.wid_custom_path.setVisible(checked)
        self.combo_model.setDisabled(checked)
        self.btn_download.setDisabled(checked)
        if not checked:
            self.check_download_status() # Re-enable logic

    def browse_model_path(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Model Folder (containing config.json)")
        if folder:
            self.line_custom_path.setText(folder)

    def cleanup_worker(self):
        if self.worker:
            self.worker.stop()
            del self.worker
            self.worker = None
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()
            self.log_box.append("🧹 VRAM/RAM Cleared.")
        except: pass

    # ... [Rest of Existing Logic] ...
    
    def abort_process(self):
        if self.is_processing:
            self.toggle_process_state()

    def load_api_presets(self):
        if os.path.exists(API_PRESETS_FILE):
            try:
                with open(API_PRESETS_FILE, 'r') as f:
                    self.api_presets = json.load(f)
            except: self.api_presets = {}
        self.combo_presets.clear()
        self.combo_presets.addItem("Select Preset...")
        self.combo_presets.addItems(list(self.api_presets.keys()))

    def save_api_preset(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset Name:")
        if ok and name:
            preset_data = {
                "base_url": self.inp_api_url.text(),
                "api_key": self.inp_api_key.text(),
                "model_name": self.inp_api_model.text()
            }
            self.api_presets[name] = preset_data
            with open(API_PRESETS_FILE, 'w') as f: json.dump(self.api_presets, f, indent=4)
            self.load_api_presets()
            self.combo_presets.setCurrentText(name)

    def delete_api_preset(self):
        name = self.combo_presets.currentText()
        if name in self.api_presets:
            del self.api_presets[name]
            with open(API_PRESETS_FILE, 'w') as f: json.dump(self.api_presets, f, indent=4)
            self.load_api_presets()

    def apply_api_preset(self):
        name = self.combo_presets.currentText()
        if name in self.api_presets:
            data = self.api_presets[name]
            self.inp_api_url.setText(data.get("base_url", ""))
            self.inp_api_key.setText(data.get("api_key", ""))
            self.inp_api_model.setText(data.get("model_name", ""))

    def apply_template(self):
        if not self.chk_override.isChecked():
            key = self.combo_template.currentText()
            self.prompt_input.setText(PROMPT_TEMPLATES[key])

    def toggle_prompt_edit(self, checked):
        self.prompt_input.setReadOnly(not checked)
        if not checked:
            self.apply_template()

    def refresh_models(self):
        self.combo_model.clear()
        for name in KNOWN_MODELS.keys():
            self.combo_model.addItem(f"☁️ {name}", KNOWN_MODELS[name])
        models_dir = os.path.join(os.getcwd(), "models")
        if os.path.exists(models_dir):
            for d in os.listdir(models_dir):
                if os.path.isdir(os.path.join(models_dir, d)):
                    self.combo_model.addItem(f"📁 {d}", d)
        self.check_download_status()

    def check_download_status(self):
        data = self.combo_model.currentData()
        models_dir = os.path.join(os.getcwd(), "models")
        local_path_a = os.path.join(models_dir, data)
        folder_name = data.split("/")[-1]
        local_path_b = os.path.join(models_dir, folder_name)
        if os.path.exists(local_path_a) or os.path.exists(local_path_b):
            self.btn_download.setEnabled(False)
            self.btn_download.setText("✅ Installed")
        else:
            self.btn_download.setEnabled(True)
            self.btn_download.setText("⬇️ Download Model")

    def start_download(self):
        repo_id = self.combo_model.currentData()
        folder_name = repo_id.split("/")[-1]
        target_dir = os.path.join(os.getcwd(), "models", folder_name)
        self.btn_download.setEnabled(False)
        self.log_box.append(f"Starting download: {repo_id} -> {target_dir}")
        self.dl_thread = QThread()
        self.dl_worker = DownloadWorker(repo_id, target_dir)
        self.dl_worker.moveToThread(self.dl_thread)
        self.dl_thread.started.connect(self.dl_worker.run)
        self.dl_worker.progress.connect(self.log_box.append)
        self.dl_worker.finished.connect(self.on_download_finished)
        self.dl_thread.start()

    def on_download_finished(self):
        self.dl_thread.quit()
        self.refresh_models()
        self.log_box.append("Download process finished.")

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder: return
        for i in reversed(range(self.grid_layout.count())): 
            self.grid_layout.itemAt(i).widget().setParent(None)
        self.cards.clear()
        self.selected_paths.clear()
        exts = ('.jpg', '.png', '.jpeg', '.webp')
        files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
        cols = 3
        for i, f in enumerate(files):
            path = os.path.join(folder, f)
            card = CaptionCard(path)
            card.selection_changed.connect(self.on_selection)
            self.grid_layout.addWidget(card, i // cols, i % cols)
            self.cards[path] = card
            worker = ThumbnailWorker(path, (230, 170))
            worker.signals.loaded.connect(card.set_image)
            self.thread_pool.start(worker)

    def on_selection(self, path, is_selected):
        if is_selected: self.selected_paths.add(path)
        else: self.selected_paths.discard(path)
        if not self.is_processing:
            self.btn_run.setText(f"🚀 Caption Selected ({len(self.selected_paths)})")

    def select_all(self):
        all_selected = len(self.selected_paths) == len(self.cards) and len(self.cards) > 0
        target = not all_selected
        for card in self.cards.values(): card.toggle_selection(target)

    def toggle_process_state(self):
        if self.is_processing:
            self.cleanup_worker()
            self.log_box.append("🛑 Process Aborted.")
            self.set_processing_ui(False)
        else:
            self.run_process()

    def set_processing_ui(self, running):
        self.is_processing = running
        if running:
            self.btn_run.setText("🛑 ABORT")
            self.btn_run.setStyleSheet("background-color: #ff4757; font-weight: bold; font-size: 14px;")
            self.btn_select_all.setEnabled(False)
            self.btn_folder.setEnabled(False)
        else:
            self.btn_run.setText(f"🚀 Caption Selected ({len(self.selected_paths)})")
            self.btn_run.setStyleSheet("background-color: #d63031; font-weight: bold; font-size: 14px;")
            self.btn_select_all.setEnabled(True)
            self.btn_folder.setEnabled(True)
            self.lbl_status.setText("Ready")

    def run_process(self):
        if not self.selected_paths:
            self.log_box.append("⚠️ No images selected!")
            return
        
        # Cleanup any existing worker first
        self.cleanup_worker()
        
        is_local = self.tab_source.currentIndex() == 0
        model_path = ""
        api_config = None
        
        if is_local:
            # Check for Custom Path
            if self.chk_custom_path.isChecked():
                custom_path = self.line_custom_path.text().strip()
                if not os.path.exists(custom_path):
                    self.log_box.append("❌ Custom path does not exist.")
                    return
                # Verify it looks like a model folder (basic check)
                if not os.path.exists(os.path.join(custom_path, "config.json")):
                    self.log_box.append("⚠️ Warning: config.json not found in custom path. It might fail.")
                model_path = custom_path
            else:
                # Standard Logic
                data = self.combo_model.currentData()
                models_dir = os.path.join(os.getcwd(), "models")
                path = os.path.join(models_dir, data)
                if not os.path.exists(path):
                    path = os.path.join(models_dir, data.split("/")[-1])
                    if not os.path.exists(path):
                        self.log_box.append("❌ Local model not found.")
                        return
                model_path = path
        else:
            api_config = {
                "base_url": self.inp_api_url.text(),
                "api_key": self.inp_api_key.text(),
                "model_name": self.inp_api_model.text()
            }
            model_path = "API"

        self.set_processing_ui(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(self.selected_paths))
        self.progress_bar.setFormat("%p% - %v/%m")
        
        params = {
            "max_tokens": self.spin_tokens.value(),
            "temperature": self.spin_temp.value(),
            "top_p": self.spin_top_p.value()
        }
        
        self.worker = QwenWorker(model_path, list(self.selected_paths), self.prompt_input.toPlainText(), params, api_config)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_single_finished)
        self.worker.progress.connect(self.update_log_status)
        self.worker.error.connect(self.log_box.append)
        self.worker.finished.connect(self.check_if_done)
        self.thread.start()

    def on_single_finished(self, path, caption):
        if path in self.cards: self.cards[path].update_caption_from_ai(caption)
        self.progress_bar.setValue(self.progress_bar.value() + 1)

    def update_log_status(self, msg):
        self.log_box.append(msg)
        self.lbl_status.setText(msg)

    def check_if_done(self):
        if self.progress_bar.value() >= self.progress_bar.maximum():
            self.log_box.append("✅ Batch Complete!")
            self.set_processing_ui(False)
            self.thread.quit()

    def save_selected(self):
        c = sum(1 for p in self.selected_paths if self.cards[p].save_text())
        QMessageBox.information(self, "Saved", f"Saved {c} captions.")

    def save_all(self):
        c = sum(1 for card in self.cards.values() if card.save_text())
        QMessageBox.information(self, "Saved", f"Saved {c} captions.")

    def save_to_dataset(self):
        if not self.selected_paths:
            QMessageBox.warning(self, "No Selection", "Please select images.")
            return
        name, ok = QInputDialog.getText(self, "New Dataset", "Enter Dataset Collection Name:")
        if not ok or not name.strip(): return
        
        dataset_root = os.path.join(os.getcwd(), "Dataset Collections")
        target_dir = os.path.join(dataset_root, name.strip())
        if not os.path.exists(target_dir): os.makedirs(target_dir)
            
        count = 0
        for path in self.selected_paths:
            if path in self.cards: self.cards[path].save_text()
            try:
                shutil.copy2(path, target_dir)
                txt_path = os.path.splitext(path)[0] + ".txt"
                if os.path.exists(txt_path): shutil.copy2(txt_path, target_dir)
                count += 1
            except Exception as e: self.log_box.append(f"Copy Error: {e}")
        
        QMessageBox.information(self, "Success", f"Copied {count} pairs to '{name}'.")