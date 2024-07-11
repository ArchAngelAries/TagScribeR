import json
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton
from PyQt5.QtCore import pyqtSignal

class SettingsWindow(QWidget):
    themeChanged = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.initUI()
        self.loadSettings()

    def initUI(self):
        self.setWindowTitle("TagScribeR - Settings")
        layout = QVBoxLayout()

        # Theme selection
        themeLayout = QHBoxLayout()
        themeLabel = QLabel("UI Theme:")
        self.themeCombo = QComboBox()
        self.themeCombo.addItems(["Default", "Dark", "Light"])
        themeLayout.addWidget(themeLabel)
        themeLayout.addWidget(self.themeCombo)
        layout.addLayout(themeLayout)

        # Save button
        saveButton = QPushButton("Save Settings")
        saveButton.clicked.connect(self.saveSettings)
        layout.addWidget(saveButton)

        self.setLayout(layout)

    def loadSettings(self):
        try:
            with open('settings.json', 'r') as f:
                settings = json.load(f)
                self.themeCombo.setCurrentText(settings.get('theme', 'Default'))
        except FileNotFoundError:
            pass  # Use default settings if file not found

    def saveSettings(self):
        settings = {
            'theme': self.themeCombo.currentText()
        }
        with open('settings.json', 'w') as f:
            json.dump(settings, f)
        self.themeChanged.emit(settings['theme'])