import sys
import json
import os
import ctypes
from ctypes import wintypes
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QListWidget, QStackedWidget)
from PySide6.QtGui import QIcon
from qt_material import apply_stylesheet

from tabs import GalleryTab, CaptionTab, EditorTab, MetadataTab, SettingsTab, DatasetsTab

CONFIG_FILE = "config.json"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TagScribeR v2.0 - Qwen Edition")
        self.resize(1600, 900)
        
        # --- ICON LOGIC ---
        # We try to use the ICO first, then PNG
        basedir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(basedir, "resources", "logo.ico")
        if not os.path.exists(icon_path):
            icon_path = os.path.join(basedir, "resources", "logo.png")
            
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            # Store path for global access
            self.icon_path = icon_path 
        
        # Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0,0,0,0)
        main_layout.setSpacing(0)

        # Sidebar
        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(200)
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
        
        # Stack
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

        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.stack)
        self.sidebar.setCurrentRow(0)

    def change_tab(self, index):
        self.stack.setCurrentIndex(index)

    # --- NATIVE WINDOWS ICON FORCE ---
    def showEvent(self, event):
        super().showEvent(event)
        # Only run on Windows
        if os.name == 'nt' and hasattr(self, 'icon_path'):
            try:
                # Force Windows to refresh the taskbar icon mapping for this window handle
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            except:
                pass

def load_theme_from_config():
    default_theme = 'dark_teal.xml'
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f).get("theme", default_theme)
        except: pass
    return default_theme

# --- GLOBAL APP ID ---
# Changed again to force cache refresh
myappid = 'ArchAngelAries.TagScribeR.Pro.Final.v4' 

if __name__ == "__main__":
    if os.name == 'nt':
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception as e:
            print(f"AppID Error: {e}")

    app = QApplication(sys.argv)
    
    # 1. Load Icon Global
    basedir = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(basedir, "resources", "logo.ico")
    if not os.path.exists(icon_path):
        icon_path = os.path.join(basedir, "resources", "logo.png")
    
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
        print(f"Icon loaded from: {icon_path}")
    else:
        print("❌ Icon file not found in resources/")

    startup_theme = load_theme_from_config()
    apply_stylesheet(app, theme=startup_theme)
    
    app.setStyleSheet(app.styleSheet() + """
        QStackedWidget { background-color: #1e1e1e; }
        QScrollBar:vertical { width: 12px; }
        QLineEdit, QTextEdit { border-radius: 4px; }
    """)

    window = MainWindow()
    window.show()
    
    # One last trick: Force update
    if os.name == 'nt':
        window.setWindowIcon(QIcon(icon_path))
    
    sys.exit(app.exec())