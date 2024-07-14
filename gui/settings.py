import json
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QPushButton, QTableWidget, QTableWidgetItem, QDialog, 
    QLineEdit, QMessageBox, QHeaderView, QAbstractItemView
)
from PyQt5.QtCore import pyqtSignal, Qt, QSize
from PyQt5.QtGui import QKeySequence

class RemapDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Remap Shortcut")
        layout = QVBoxLayout(self)
        self.keySequenceEdit = QLineEdit(self)
        self.keySequenceEdit.setPlaceholderText("Press new shortcut")
        self.keySequenceEdit.setReadOnly(True)
        layout.addWidget(self.keySequenceEdit)
        
        buttonLayout = QHBoxLayout()
        okButton = QPushButton("OK")
        cancelButton = QPushButton("Cancel")
        buttonLayout.addWidget(okButton)
        buttonLayout.addWidget(cancelButton)
        layout.addLayout(buttonLayout)
        
        okButton.clicked.connect(self.accept)
        cancelButton.clicked.connect(self.reject)
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.reject()
        else:
            key = QKeySequence(event.key() | int(event.modifiers())).toString()
            self.keySequenceEdit.setText(key)
        
    def getKeySequence(self):
        return self.keySequenceEdit.text()

class SettingsWindow(QWidget):
    themeChanged = pyqtSignal(str)
    shortcutsChanged = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.shortcuts = {}
        self.custom_shortcuts = {}
        self.initUI()
        self.loadSettings()

    def initUI(self):
        self.setWindowTitle("TagScribeR - Settings")
        self.mainLayout = QVBoxLayout()

        # Theme selection
        themeLayout = QHBoxLayout()
        themeLabel = QLabel("UI Theme:")
        self.themeCombo = QComboBox()
        self.themeCombo.addItems(["Default", "Dark", "Light"])
        themeLayout.addWidget(themeLabel)
        themeLayout.addWidget(self.themeCombo)
        self.mainLayout.addLayout(themeLayout)
        
        # Shortcuts table
        self.shortcutsLabel = QLabel("Keyboard Shortcuts:")
        self.mainLayout.addWidget(self.shortcutsLabel)

        self.shortcutsTable = QTableWidget()
        self.shortcutsTable.setColumnCount(3)
        self.shortcutsTable.setHorizontalHeaderLabels(["Action", "Default", "Custom"])
        self.shortcutsTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.shortcutsTable.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.shortcutsTable.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.mainLayout.addWidget(self.shortcutsTable)

        # Reset Shortcuts Button
        self.resetShortcutsButton = QPushButton("Reset Shortcuts to Default")
        self.resetShortcutsButton.clicked.connect(self.resetShortcutsToDefault)
        self.mainLayout.addWidget(self.resetShortcutsButton)

        # Save button
        self.saveButton = QPushButton("Save Settings")
        self.saveButton.clicked.connect(self.saveSettings)
        self.mainLayout.addWidget(self.saveButton)

        self.setLayout(self.mainLayout)

    def loadSettings(self):
        try:
            with open('settings.json', 'r') as f:
                settings = json.load(f)
                self.themeCombo.setCurrentText(settings.get('theme', 'Default'))
                self.custom_shortcuts = settings.get('custom_shortcuts', {})
        except FileNotFoundError:
            self.custom_shortcuts = {}

    def saveSettings(self):
        settings = {
            'theme': self.themeCombo.currentText(),
            'custom_shortcuts': self.custom_shortcuts
        }
        with open('settings.json', 'w') as f:
            json.dump(settings, f)
        self.themeChanged.emit(settings['theme'])
        self.shortcutsChanged.emit(self.custom_shortcuts)
        QMessageBox.information(self, "Settings Saved", "Your settings have been saved successfully.")

    def updateShortcuts(self, shortcuts):
        self.shortcuts = shortcuts
        self.shortcutsTable.setRowCount(len(shortcuts))
        for row, (key, (_, desc)) in enumerate(shortcuts.items()):
            self.shortcutsTable.setItem(row, 0, QTableWidgetItem(desc))
            self.shortcutsTable.setItem(row, 1, QTableWidgetItem(key))
            
            custom_key = self.custom_shortcuts.get(desc, "Not Remapped")
            remapButton = QPushButton(custom_key)
            remapButton.clicked.connect(lambda _, r=row: self.remapShortcut(r))
            self.shortcutsTable.setCellWidget(row, 2, remapButton)
        
        self.shortcutsTable.resizeRowsToContents()

    def remapShortcut(self, row):
        action = self.shortcutsTable.item(row, 0).text()
        dialog = RemapDialog(self)
        if dialog.exec_():
            new_shortcut = dialog.getKeySequence()
            if new_shortcut:
                # Check if the new shortcut is already in use
                for existing_action, existing_shortcut in self.custom_shortcuts.items():
                    if existing_shortcut == new_shortcut and existing_action != action:
                        QMessageBox.warning(self, "Shortcut Conflict", 
                                            f"The shortcut '{new_shortcut}' is already assigned to '{existing_action}'.")
                        return
                
                self.custom_shortcuts[action] = new_shortcut
                remapButton = self.shortcutsTable.cellWidget(row, 2)
                remapButton.setText(new_shortcut)
            else:
                QMessageBox.warning(self, "Invalid Shortcut", "The entered shortcut is invalid.")

    def resetShortcutsToDefault(self):
        reply = QMessageBox.question(self, 'Reset Shortcuts', 
                                     "Are you sure you want to reset all shortcuts to their default values?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.custom_shortcuts.clear()
            self.updateShortcuts(self.shortcuts)
            QMessageBox.information(self, "Reset Complete", "All shortcuts have been reset to their default values.")
                