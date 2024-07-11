import os
import sys
import json
import qdarkstyle
import ctypes
import win32gui
import win32api
import win32con
from PyQt5.QtWidgets import QApplication, QTabWidget, QMainWindow, QStyleFactory
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QFile, QTextStream, Qt
from gui.gallery import GalleryWindow
from gui.auto_captioning import AutoCaptioningWindow
from gui.image_editing import ImageEditingWindow
from gui.settings import SettingsWindow

def set_app_id():
    app_id = "TagScribeR.ArchAngelAried.0.03a"  # Change this to a unique identifier for your app
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)

class MainApplicationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TagScribeR")
        self.setGeometry(100, 100, 1200, 800)

        # Set application icon
        icon_path = self.get_icon_path()
        if icon_path:
            self.setWindowIcon(QIcon(icon_path))
            self.set_taskbar_icon(icon_path)
        else:
            print("Warning: Logo.png not found in the resources folder.")

        # Create Tab Widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Add tabs
        self.galleryWindow = GalleryWindow()
        self.autoCaptioningWindow = AutoCaptioningWindow()
        self.imageEditingWindow = ImageEditingWindow()
        self.settingsWindow = SettingsWindow()

        self.tabs.addTab(self.galleryWindow, "Gallery")
        self.tabs.addTab(self.autoCaptioningWindow, "Blip-2 Auto Captioning")
        self.tabs.addTab(self.imageEditingWindow, "Image Editing")
        self.tabs.addTab(self.settingsWindow, "Settings")

        # Connect the themeChanged signal to a slot
        self.settingsWindow.themeChanged.connect(self.changeTheme)

        # Load initial theme
        self.loadInitialTheme()

    def loadInitialTheme(self):
        try:
            with open('settings.json', 'r') as f:
                settings = json.load(f)
                self.changeTheme(settings.get('theme', 'Default'))
        except FileNotFoundError:
            self.changeTheme('Default')

    def changeTheme(self, theme):
        if theme == 'Dark':
            self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        elif theme == 'Light':
            # Reset to the default style
            QApplication.setStyle(QStyleFactory.create('Fusion'))
            self.setStyleSheet("")
        else:  # Default theme
            QApplication.setStyle(QStyleFactory.create('Fusion'))
            self.setStyleSheet("")
        
        # Refresh the application to apply changes
        QApplication.processEvents()

        # Update each tab's theme
        for i in range(self.tabs.count()):
            widget = self.tabs.widget(i)
            if hasattr(widget, 'applyTheme'):
                widget.applyTheme(theme)
            
    def get_icon_path(self):
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the path to the resources folder
        resources_dir = os.path.join(current_dir, 'resources')
        # Construct the full path to the Logo.png file
        icon_path = os.path.join(resources_dir, 'Logo.png')
        
        # Check if the file exists
        if os.path.isfile(icon_path):
            return icon_path
        else:
            return None        
            
    def set_taskbar_icon(self, icon_path):
        # Convert the icon path to the format Windows expects
        icon_path = os.path.abspath(icon_path)
        
        # Load the icon
        icon_flags = win32con.LR_LOADFROMFILE | win32con.LR_DEFAULTSIZE
        try:
            hicon = win32gui.LoadImage(None, icon_path, win32con.IMAGE_ICON, 0, 0, icon_flags)
        except:
            return  # If it fails, just return and use the default icon
        
        # Find the Windows handle for this window
        hwnd = self.winId().__int__()
        
        # Set the icon for the window
        win32gui.SendMessage(hwnd, win32con.WM_SETICON, win32con.ICON_BIG, hicon)
        win32gui.SendMessage(hwnd, win32con.WM_SETICON, win32con.ICON_SMALL, hicon)            

def main():
    # Set the app ID for Windows
    set_app_id()

    # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    
    mainWindow = MainApplicationWindow()

    # Set the application icon
    icon_path = mainWindow.get_icon_path()
    if icon_path:
        app_icon = QIcon(icon_path)
        app.setWindowIcon(app_icon)
        mainWindow.setWindowIcon(app_icon)

    mainWindow.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()