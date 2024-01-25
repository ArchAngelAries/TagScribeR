import sys
from PyQt5.QtWidgets import QApplication, QTabWidget, QMainWindow
from gui.gallery import GalleryWindow  # Make sure these paths are correct
from gui.auto_captioning import AutoCaptioningWindow
# from gui.tag_management import TagManagementWindow  # As an example, add imports for your other windows
# from gui.image_editing import ImageEditingWindow
# from gui.settings import SettingsWindow

class MainApplicationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TagScribeR")
        self.setGeometry(100, 100, 1200, 800)

        # Create Tab Widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Add tabs
        self.tabs.addTab(GalleryWindow(), "Gallery")
        self.tabs.addTab(AutoCaptioningWindow(), "Blip-2 Auto Captioning")
        # self.tabs.addTab(TagManagementWindow(), "Tag Saving/Sharing")  # Add these as you implement each window
        # self.tabs.addTab(ImageEditingWindow(), "Image Cropping/Resizing")
        # self.tabs.addTab(SettingsWindow(), "Settings")

def main():
    app = QApplication(sys.argv)
    mainWindow = MainApplicationWindow()
    mainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
