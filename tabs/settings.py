import os
import json
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QComboBox, QGroupBox, QSpinBox, QDoubleSpinBox, QListWidget, 
    QMessageBox, QApplication, QFormLayout, QScrollArea
)
from PySide6.QtCore import Qt
from qt_material import list_themes, apply_stylesheet

CONFIG_FILE = "config.json"
TAG_FILE = "user_tags.txt"

DEFAULT_CONFIG = {
    "theme": "dark_teal.xml",
    "ai_max_tokens": 512,
    "ai_temperature": 0.7,
    "ai_top_p": 0.9,
    "default_prompt_template": "Detailed Description"
}

class SettingsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.config = self.load_config()
        
        layout = QVBoxLayout(self)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        scroll.setWidget(container)
        
        main_layout = QVBoxLayout(container)
        main_layout.setSpacing(20)
        main_layout.setAlignment(Qt.AlignTop)

        # --- 1. APPEARANCE ---
        grp_app = QGroupBox("1. Appearance & Theme")
        lyt_app = QFormLayout(grp_app)
        
        self.combo_theme = QComboBox()
        # Filter for nice themes only
        themes = [t for t in list_themes() if 'dark' in t or 'light' in t]
        self.combo_theme.addItems(themes)
        
        # Set current theme in dropdown
        current_theme = self.config.get("theme", "dark_teal.xml")
        index = self.combo_theme.findText(current_theme)
        if index >= 0: self.combo_theme.setCurrentIndex(index)
        
        self.btn_apply_theme = QPushButton("Apply Theme")
        self.btn_apply_theme.clicked.connect(self.apply_theme)
        self.btn_apply_theme.setStyleSheet("background-color: #6c5ce7; color: white;")
        
        lyt_app.addRow("Interface Theme:", self.combo_theme)
        lyt_app.addRow("", self.btn_apply_theme)
        main_layout.addWidget(grp_app)

        # --- 2. AI DEFAULTS ---
        grp_ai = QGroupBox("2. AI Default Parameters")
        lyt_ai = QFormLayout(grp_ai)
        
        self.spin_tokens = QSpinBox()
        self.spin_tokens.setRange(64, 4096)
        self.spin_tokens.setValue(self.config.get("ai_max_tokens", 512))
        
        self.spin_temp = QDoubleSpinBox()
        self.spin_temp.setRange(0.0, 2.0)
        self.spin_temp.setSingleStep(0.1)
        self.spin_temp.setValue(self.config.get("ai_temperature", 0.7))
        
        self.spin_top = QDoubleSpinBox()
        self.spin_top.setRange(0.0, 1.0)
        self.spin_top.setSingleStep(0.05)
        self.spin_top.setValue(self.config.get("ai_top_p", 0.9))
        
        self.btn_save_ai = QPushButton("Save Defaults")
        self.btn_save_ai.clicked.connect(self.save_settings)
        self.btn_save_ai.setStyleSheet("background-color: #00b894; color: white;")
        
        lyt_ai.addRow("Default Max Tokens:", self.spin_tokens)
        lyt_ai.addRow("Default Temperature:", self.spin_temp)
        lyt_ai.addRow("Default Top P:", self.spin_top)
        lyt_ai.addRow("", self.btn_save_ai)
        main_layout.addWidget(grp_ai)

        # --- 3. TAG MANAGER ---
        grp_tags = QGroupBox("3. Custom Tag Manager")
        lyt_tags = QVBoxLayout(grp_tags)
        
        self.list_tags = QListWidget()
        self.refresh_tag_list()
        
        btn_tag_box = QHBoxLayout()
        self.btn_del_tag = QPushButton("Delete Selected Tag")
        self.btn_del_tag.clicked.connect(self.delete_selected_tag)
        self.btn_del_tag.setStyleSheet("background-color: #d63031; color: white;")
        
        self.btn_refresh_tags = QPushButton("Refresh List")
        self.btn_refresh_tags.clicked.connect(self.refresh_tag_list)
        
        btn_tag_box.addWidget(self.btn_refresh_tags)
        btn_tag_box.addWidget(self.btn_del_tag)
        
        lyt_tags.addWidget(QLabel("Manage your persistent 'Quick Tags' here:"))
        lyt_tags.addWidget(self.list_tags)
        lyt_tags.addLayout(btn_tag_box)
        main_layout.addWidget(grp_tags)

        layout.addWidget(scroll)

    # --- LOGIC ---
    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except:
                return DEFAULT_CONFIG
        return DEFAULT_CONFIG

    def save_settings(self):
        self.config["theme"] = self.combo_theme.currentText()
        self.config["ai_max_tokens"] = self.spin_tokens.value()
        self.config["ai_temperature"] = self.spin_temp.value()
        self.config["ai_top_p"] = self.spin_top.value()
        
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=4)
        
        QMessageBox.information(self, "Saved", "Settings saved successfully.")

    def apply_theme(self):
        selected_theme = self.combo_theme.currentText()
        app = QApplication.instance()
        
        # Apply theme
        apply_stylesheet(app, theme=selected_theme)
        
        # Re-apply custom tweaks that the theme might have overwritten
        app.setStyleSheet(app.styleSheet() + """
            QStackedWidget { background-color: #1e1e1e; }
            QListWidget { border: none; }
            QLineEdit, QTextEdit { border-radius: 4px; }
        """)
        
        self.save_settings()

    # --- TAG MANAGER LOGIC ---
    def refresh_tag_list(self):
        self.list_tags.clear()
        if os.path.exists(TAG_FILE):
            with open(TAG_FILE, 'r', encoding='utf-8') as f:
                tags = [line.strip() for line in f.readlines() if line.strip()]
                self.list_tags.addItems(tags)

    def delete_selected_tag(self):
        row = self.list_tags.currentRow()
        if row < 0: return
        
        item = self.list_tags.takeItem(row)
        tag_to_remove = item.text()
        
        # Rewrite file
        if os.path.exists(TAG_FILE):
            with open(TAG_FILE, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            with open(TAG_FILE, 'w', encoding='utf-8') as f:
                for line in lines:
                    if line.strip() != tag_to_remove:
                        f.write(line)
                        
        QMessageBox.information(self, "Deleted", f"Removed tag: {tag_to_remove}")