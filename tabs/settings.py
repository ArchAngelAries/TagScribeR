"""Settings tab: appearance, AI defaults, model folders, quick tags, diagnostics."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QListWidget, QMessageBox, QPlainTextEdit, QPushButton,
                               QScrollArea, QSpinBox, QVBoxLayout, QWidget)
from qt_material import apply_stylesheet, list_themes

from core import paths
from core.config import settings
from inference.models import stability_matrix_llm_dirs
from tabs.common import hint_label, run_in_background

APP_STYLE_TWEAKS = """
    QStackedWidget { background-color: #1e1e1e; }
    QScrollBar:vertical { width: 12px; }
    QLineEdit, QTextEdit, QPlainTextEdit { border-radius: 4px; }
"""


def apply_theme(theme: str) -> None:
    app = QApplication.instance()
    apply_stylesheet(app, theme=theme)
    app.setStyleSheet(app.styleSheet() + APP_STYLE_TWEAKS)


class SettingsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.cfg = settings()
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        scroll.setWidget(container)
        main = QVBoxLayout(container)
        main.setSpacing(16)
        main.setAlignment(Qt.AlignTop)
        outer.addWidget(scroll)

        # --- Appearance ---
        grp_app = QGroupBox("Appearance")
        f_app = QFormLayout(grp_app)
        self.combo_theme = QComboBox()
        self.combo_theme.addItems([t for t in list_themes() if "dark" in t or "light" in t])
        i = self.combo_theme.findText(self.cfg.get("ui.theme"))
        if i >= 0:
            self.combo_theme.setCurrentIndex(i)
        self.combo_theme.currentTextChanged.connect(self._on_theme)
        f_app.addRow("Theme:", self.combo_theme)
        self.combo_scale = QComboBox()
        for pct in (100, 115, 125, 150, 175, 200):
            self.combo_scale.addItem(f"{pct}%", pct / 100)
        current = self.cfg.get("ui.scale")
        self.combo_scale.setCurrentIndex(min(range(self.combo_scale.count()),
                                             key=lambda i: abs(self.combo_scale.itemData(i) - current)))
        self.combo_scale.setToolTip("Enlarges all text, buttons and inputs. Takes effect after restarting TagScribeR.")
        self.combo_scale.currentIndexChanged.connect(self._on_scale)
        f_app.addRow("Interface scale:", self.combo_scale)
        f_app.addRow(hint_label("Thumbnail size is adjusted in the Gallery (Size slider, Ctrl + mouse wheel, "
                                "Ctrl+= / Ctrl+-)."))
        main.addWidget(grp_app)

        # --- AI defaults ---
        grp_ai = QGroupBox("AI defaults")
        f_ai = QFormLayout(grp_ai)
        self.spin_idle = QSpinBox()
        self.spin_idle.setRange(0, 240)
        self.spin_idle.setSuffix(" min")
        self.spin_idle.setSpecialValueText("Never")
        self.spin_idle.setValue(self.cfg.get("local.idle_unload_minutes"))
        self.spin_idle.setToolTip("Free GPU memory automatically after this long without captioning.")
        self.spin_idle.valueChanged.connect(lambda v: self.cfg.set("local.idle_unload_minutes", v))
        self.combo_tagger_device = QComboBox()
        self.combo_tagger_device.addItem("Auto (GPU if available)", "auto")
        self.combo_tagger_device.addItem("CPU only", "cpu")
        self.combo_tagger_device.setCurrentIndex(max(0, self.combo_tagger_device.findData(self.cfg.get("tagger.device"))))
        self.combo_tagger_device.currentIndexChanged.connect(
            lambda: self.cfg.set("tagger.device", self.combo_tagger_device.currentData()))
        f_ai.addRow("Unload idle model after:", self.spin_idle)
        f_ai.addRow("Tagger device:", self.combo_tagger_device)
        f_ai.addRow(hint_label("Captioning parameters (tokens, temperature, prompt) are remembered "
                               "automatically from the Auto Caption tab."))
        main.addWidget(grp_ai)

        # --- Model folders ---
        grp_models = QGroupBox("Model folders")
        lm = QVBoxLayout(grp_models)
        lm.addWidget(hint_label(
            "TagScribeR scans these folders (and subfolders) for Hugging Face vision models. "
            f"Always included: {paths.DEFAULT_MODELS_DIR}"
            + "".join(f"\nDetected Stability Matrix: {d}" for d in stability_matrix_llm_dirs())))
        self.list_dirs = QListWidget()
        self.list_dirs.setMaximumHeight(110)
        self.list_dirs.addItems(self.cfg.get("local.model_dirs"))
        lm.addWidget(self.list_dirs)
        row = QHBoxLayout()
        b_add = QPushButton("Add folder…")
        b_add.clicked.connect(self._add_dir)
        b_rm = QPushButton("Remove")
        b_rm.clicked.connect(self._remove_dir)
        row.addWidget(b_add)
        row.addWidget(b_rm)
        row.addStretch()
        lm.addLayout(row)
        main.addWidget(grp_models)

        # --- Quick tags ---
        grp_tags = QGroupBox("Quick tags")
        lt = QVBoxLayout(grp_tags)
        lt.addWidget(hint_label("One-click tags shown in Gallery → Batch. Saved immediately and kept until you "
                                "remove them. An entry can hold several tags (\"ohwx, 1girl\")."))
        from tabs.common import QuickTagList
        self.quick_tags = QuickTagList()
        self.quick_tags.list.setMaximumHeight(200)
        lt.addWidget(self.quick_tags)
        main.addWidget(grp_tags)

        # --- Diagnostics ---
        grp_diag = QGroupBox("System && diagnostics")
        ld = QVBoxLayout(grp_diag)
        self.txt_env = QPlainTextEdit("Detecting hardware…")
        self.txt_env.setReadOnly(True)
        self.txt_env.setMaximumHeight(170)
        ld.addWidget(self.txt_env)
        row = QHBoxLayout()
        b_logs = QPushButton("Open log folder")
        b_logs.clicked.connect(lambda: self._open(paths.LOG_DIR))
        b_backups = QPushButton("Open caption backups")
        b_backups.clicked.connect(lambda: self._open(paths.CAPTION_BACKUP_DIR))
        b_copy = QPushButton("Copy report")
        b_copy.clicked.connect(lambda: QApplication.clipboard().setText(self.txt_env.toPlainText()))
        for b in (b_logs, b_backups, b_copy):
            row.addWidget(b)
        row.addStretch()
        ld.addLayout(row)
        main.addWidget(grp_diag)

        from core import hardware
        run_in_background(hardware.describe_environment, self.txt_env.setPlainText,
                          lambda e: self.txt_env.setPlainText(f"Hardware detection failed: {e}"))

    # --- handlers ---
    def _on_theme(self, theme: str):
        apply_theme(theme)
        self.cfg.set("ui.theme", theme)

    def _on_scale(self):
        value = self.combo_scale.currentData()
        if abs(value - self.cfg.get("ui.scale")) < 0.01:
            return
        self.cfg.set("ui.scale", value)
        QMessageBox.information(self, "Interface scale",
                                f"Interface scale set to {int(value * 100)}%. Restart TagScribeR to apply it.")

    def _add_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Add model folder")
        if d:
            dirs = list(dict.fromkeys(self.cfg.get("local.model_dirs") + [d]))
            self.cfg.set("local.model_dirs", dirs)
            self.list_dirs.clear()
            self.list_dirs.addItems(dirs)

    def _remove_dir(self):
        item = self.list_dirs.currentItem()
        if item:
            dirs = [d for d in self.cfg.get("local.model_dirs") if d != item.text()]
            self.cfg.set("local.model_dirs", dirs)
            self.list_dirs.takeItem(self.list_dirs.row(item))

    @staticmethod
    def _open(folder):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
