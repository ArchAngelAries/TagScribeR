import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QTableWidget, QTableWidgetItem, QFileDialog,
                             QListWidget, QSplitter, QMessageBox, QProgressDialog,
                             QCheckBox)
from PyQt5.QtCore import Qt, QSettings
from exif import Image as ExifImage

class MetadataEditor(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_directory = ""
        self.image_files = []
        self.current_image = None
        self.settings = QSettings("YourCompany", "TagScribeR")
        self.initUI()
        self.disclaimer_shown = False

    def initUI(self):
        layout = QVBoxLayout()

        # Buttons for loading
        buttonLayout = QHBoxLayout()
        loadDirButton = QPushButton("Load Directory")
        loadDirButton.clicked.connect(self.loadDirectory)
        loadFilesButton = QPushButton("Load Files")
        loadFilesButton.clicked.connect(self.loadFiles)
        buttonLayout.addWidget(loadDirButton)
        buttonLayout.addWidget(loadFilesButton)
        layout.addLayout(buttonLayout)

        # Splitter for file list and metadata
        splitter = QSplitter(Qt.Horizontal)

        # File list
        self.fileList = QListWidget()
        self.fileList.itemSelectionChanged.connect(self.onFileSelected)
        splitter.addWidget(self.fileList)

        # Metadata table
        self.metadataTable = QTableWidget()
        self.metadataTable.setColumnCount(2)
        self.metadataTable.setHorizontalHeaderLabels(["Field", "Value"])
        self.metadataTable.itemChanged.connect(self.onMetadataItemChanged)
        splitter.addWidget(self.metadataTable)

        layout.addWidget(splitter)

        # Save button
        saveButton = QPushButton("Save Changes")
        saveButton.clicked.connect(self.saveChanges)
        layout.addWidget(saveButton)

        self.setLayout(layout)

    def showEvent(self, event):
        super().showEvent(event)
        if not self.disclaimer_shown and not self.settings.value("hide_metadata_disclaimer", False, type=bool):
            self.showDisclaimer()

    def showDisclaimer(self):
        disclaimer_text = (
            "EXIF Metadata Editing Disclaimer:\n\n"
            "1. This tool allows editing of EXIF metadata in JPEG images.\n"
            "2. Some EXIF tags may not be editable due to technical limitations.\n"
            "3. Editing metadata incorrectly can potentially corrupt image files.\n"
            "4. Always keep backups of your original images before editing metadata.\n"
            "5. This tool is provided as-is, without any guarantees or warranty.\n\n"
            "By using this tool, you acknowledge these limitations and risks."
        )

        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(disclaimer_text)
        msg_box.setWindowTitle("Metadata Editor Disclaimer")
        
        dont_show_checkbox = QCheckBox("Don't show this message again")
        msg_box.setCheckBox(dont_show_checkbox)

        msg_box.exec_()

        if dont_show_checkbox.isChecked():
            self.settings.setValue("hide_metadata_disclaimer", True)

        self.disclaimer_shown = True

    def loadDirectory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            self.current_directory = dir_path
            self.loadImagesFromDirectory(dir_path)

    def loadFiles(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Image(s)", "", "Images (*.jpg *.jpeg)")
        if files:
            self.image_files = files
            self.updateFileList()

    def loadImagesFromDirectory(self, dir_path):
        self.image_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) 
                            if f.lower().endswith(('.jpg', '.jpeg'))]
        self.updateFileList()

    def updateFileList(self):
        self.fileList.clear()
        for file in self.image_files:
            self.fileList.addItem(os.path.basename(file))

    def onFileSelected(self):
        selected_items = self.fileList.selectedItems()
        if selected_items:
            file_name = selected_items[0].text()
            file_path = os.path.join(self.current_directory, file_name)
            self.loadMetadata(file_path)

    def loadMetadata(self, image_path):
        try:
            with open(image_path, 'rb') as img_file:
                self.current_image = ExifImage(img_file)
            
            self.metadataTable.setRowCount(0)
            for tag in self.current_image.list_all():
                try:
                    value = getattr(self.current_image, tag)
                    row = self.metadataTable.rowCount()
                    self.metadataTable.insertRow(row)
                    self.metadataTable.setItem(row, 0, QTableWidgetItem(str(tag)))
                    self.metadataTable.setItem(row, 1, QTableWidgetItem(str(value)))
                except AttributeError:
                    pass  # Skip attributes that can't be read
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load metadata: {str(e)}")

    def onMetadataItemChanged(self, item):
        if item.column() == 1:  # Value column
            row = item.row()
            tag = self.metadataTable.item(row, 0).text()
            new_value = item.text()
            try:
                setattr(self.current_image, tag, new_value)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to update {tag}: {str(e)}")
                self.loadMetadata(self.current_image._filehandle.name)  # Reload to revert changes

    def saveChanges(self):
        if self.current_image:
            try:
                image_path = self.current_image._filehandle.name
                with open(image_path, 'wb') as new_image_file:
                    new_image_file.write(self.current_image.get_file())
                QMessageBox.information(self, "Success", "Metadata saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save metadata: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "No image loaded.")

    def closeEvent(self, event):
        # Save the state of the checkbox when closing the widget
        self.settings.setValue("hide_metadata_disclaimer", self.disclaimer_widget.hide_checkbox.isChecked())
        super().closeEvent(event)