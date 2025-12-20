import sys
import json
import os
import ctypes
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QListWidget, QStackedWidget, QPushButton
)
from PySide6.QtGui import QIcon, QShortcut, QKeySequence
from qt_material import apply_stylesheet

# Clean imports
from tabs import GalleryTab, CaptionTab, EditorTab, MetadataTab, SettingsTab, DatasetsTab, HelpDialog

CONFIG_FILE = "config.json"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TagScribeR v2.1 - Qwen Edition")
        self.resize(1600, 900)
        
        # Window Icon logic
        basedir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(basedir, "resources", "logo.ico")
        if not os.path.exists(icon_path):
            icon_path = os.path.join(basedir, "resources", "logo.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            self.icon_path = icon_path
        
        # Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0,0,0,0)
        main_layout.setSpacing(0)

        # --- SIDEBAR ---
        sidebar_container = QWidget()
        sidebar_container.setFixedWidth(200)
        sidebar_container.setStyleSheet("background-color: #232323;")
        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(0,0,0,0)
        
        self.sidebar = QListWidget()
        self.sidebar.addItem("🖼️ Gallery")
        self.sidebar.addItem("🤖 Auto Caption")
        self.sidebar.addItem("✏️ Image Editor")
        self.sidebar.addItem("📂 Datasets") 
        self.sidebar.addItem("ℹ️ Metadata")
        self.sidebar.addItem("⚙️ Settings")
        self.sidebar.currentRowChanged.connect(self.change_tab)
        
        self.sidebar.setStyleSheet("""
            QListWidget { border: none; background-color: #232323; font-size: 15px; }
            QListWidget::item { padding: 15px; border-bottom: 1px solid #2c2c2c; }
            QListWidget::item:selected { background-color: #00b894; color: white; }
        """)
        
        # Help Button
        btn_help = QPushButton("❓ Help / Manual")
        btn_help.setStyleSheet("""
            QPushButton { background-color: #2d3436; color: #aaa; border: none; padding: 15px; text-align: left; }
            QPushButton:hover { background-color: #333; color: white; }
        """)
        btn_help.clicked.connect(self.show_help)
        
        sidebar_layout.addWidget(self.sidebar)
        sidebar_layout.addWidget(btn_help)

        # --- STACK ---
        self.stack = QStackedWidget()
        self.tab_gallery = GalleryTab()
        self.tab_caption = CaptionTab()
        self.tab_editor = EditorTab()
        self.tab_datasets = DatasetsTab() 
        self.tab_metadata = MetadataTab()
        self.tab_settings = SettingsTab()
        
        self.tab_gallery.image_selected.connect(self.tab_metadata.load_metadata)
        
        self.stack.addWidget(self.tab_gallery)
        self.stack.addWidget(self.tab_caption)
        self.stack.addWidget(self.tab_editor)
        self.stack.addWidget(self.tab_datasets) 
        self.stack.addWidget(self.tab_metadata)
        self.stack.addWidget(self.tab_settings)

        main_layout.addWidget(sidebar_container)
        main_layout.addWidget(self.stack)
        self.sidebar.setCurrentRow(0)
        
        self.setup_hotkeys()

    def change_tab(self, index):
        self.stack.setCurrentIndex(index)

    def show_help(self):
        dlg = HelpDialog(self)
        dlg.exec()

    def setup_hotkeys(self):
        for i in range(6):
            QShortcut(QKeySequence(f"Ctrl+{i+1}"), self).activated.connect(lambda idx=i: self.sidebar.setCurrentRow(idx))
        QShortcut(QKeySequence("F1"), self).activated.connect(self.show_help)

    def showEvent(self, event):
        super().showEvent(event)
        if os.name == 'nt' and hasattr(self, 'icon_path'):
            try: ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            except: pass

    # --- CLEANUP ON EXIT ---
    def closeEvent(self, event):
        # Explicitly ask caption tab to kill threads and free VRAM
        if hasattr(self.tab_caption, 'cleanup_worker'):
            self.tab_caption.cleanup_worker()
        
        # Accept closing
        event.accept()

def load_theme_from_config():
    default_theme = 'dark_teal.xml'
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f).get("theme", default_theme)
        except: pass
    return default_theme

myappid = 'ArchAngelAries.TagScribeR.Pro.Final.v4' 

if __name__ == "__main__":
    if os.name == 'nt':
        try: ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except: pass

    app = QApplication(sys.argv)
    
    basedir = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(basedir, "resources", "logo.ico")
    if not os.path.exists(icon_path):
        icon_path = os.path.join(basedir, "resources", "logo.png")
    
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    startup_theme = load_theme_from_config()
    apply_stylesheet(app, theme=startup_theme)
    
    app.setStyleSheet(app.styleSheet() + """
        QStackedWidget { background-color: #1e1e1e; }
        QScrollBar:vertical { width: 12px; }
        QLineEdit, QTextEdit { border-radius: 4px; }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())