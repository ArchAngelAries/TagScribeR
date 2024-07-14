import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QTableWidget, QTableWidgetItem, QFileDialog,
                             QListWidget, QSplitter, QMessageBox, QProgressDialog,
                             QCheckBox, QListWidgetItem, QAbstractItemView)
from PyQt5.QtCore import Qt, QSettings, QSize
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PIL import Image
from PIL.ExifTags import TAGS
import piexif

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
        loadFilesButton = QPushButton("Load Single File")
        loadFilesButton.clicked.connect(self.loadFiles)
        buttonLayout.addWidget(loadDirButton)
        buttonLayout.addWidget(loadFilesButton)
        layout.addLayout(buttonLayout)

        # Splitter for file list and metadata
        splitter = QSplitter(Qt.Horizontal)

        # File list with thumbnails
        self.fileList = QListWidget()
        self.fileList.setIconSize(QSize(100, 100))
        self.fileList.setResizeMode(QListWidget.Adjust)
        self.fileList.setViewMode(QListWidget.IconMode)
        self.fileList.setSelectionMode(QAbstractItemView.ExtendedSelection)
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

    def loadDirectory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            self.current_directory = dir_path
            self.loadImagesFromDirectory(dir_path)

    def loadFiles(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Image(s)", "", "Images (*.jpg *.jpeg *.png *.gif *.bmp)")
        if files:
            self.current_directory = os.path.dirname(files[0])
            self.image_files = files
            self.updateFileList()

    def loadImagesFromDirectory(self, dir_path):
        self.image_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
        self.updateFileList()

    def updateFileList(self):
        self.fileList.clear()
        for file in self.image_files:
            item = QListWidgetItem()
            thumbnail = self.createThumbnail(file)
            item.setIcon(QIcon(thumbnail))
            item.setText(os.path.basename(file))
            item.setSizeHint(QSize(120, 120))  # Adjust size as needed
            self.fileList.addItem(item)

    def createThumbnail(self, image_path):
        try:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                return pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            else:
                print(f"Failed to load image: {image_path}")
                return QPixmap()
        except Exception as e:
            print(f"Error creating thumbnail for {image_path}: {e}")
            return QPixmap()

    def onFileSelected(self):
        selected_items = self.fileList.selectedItems()
        if selected_items:
            file_name = selected_items[0].text()
            file_path = os.path.join(self.current_directory, file_name)
            self.loadMetadata(file_path)

    def loadMetadata(self, image_path):
        try:
            exif_dict = piexif.load(image_path)
            self.metadataTable.setRowCount(0)
            for ifd in ("0th", "Exif", "GPS", "1st"):
                for tag_id, value in exif_dict[ifd].items():
                    tag = TAGS.get(tag_id, str(tag_id))
                    if isinstance(value, bytes):
                        try:
                            value = value.decode()
                        except:
                            value = value.hex()
                    row = self.metadataTable.rowCount()
                    self.metadataTable.insertRow(row)
                    self.metadataTable.setItem(row, 0, QTableWidgetItem(str(tag)))
                    self.metadataTable.setItem(row, 1, QTableWidgetItem(str(value)))
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load metadata: {str(e)}")

    def onMetadataItemChanged(self, item):
        if item.column() == 1:  # Value column
            row = item.row()
            tag = self.metadataTable.item(row, 0).text()
            new_value = item.text()
            # Here you would update the exif_dict with the new value
            # This part requires careful implementation to map back to the correct IFD and tag ID

    def saveChanges(self):
        selected_items = self.fileList.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "No image selected.")
            return
    
        progress = QProgressDialog("Saving metadata changes...", "Abort", 0, len(selected_items), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
    
        success_count = 0
        for i, item in enumerate(selected_items):
            if progress.wasCanceled():
                break
            file_name = item.text()
            file_path = os.path.join(self.current_directory, file_name)
            if self.saveMetadataChanges(file_path):
                success_count += 1
            progress.setValue(i + 1)
    
        progress.close()
        QMessageBox.information(self, "Save Complete", f"Metadata saved successfully for {success_count} out of {len(selected_items)} images.")

    def showEvent(self, event):
        super().showEvent(event)
        if not self.disclaimer_shown and not self.settings.value("hide_metadata_disclaimer", False, type=bool):
            self.showDisclaimer()

    def showDisclaimer(self):
        disclaimer_text = (
            "EXIF Metadata Editing Disclaimer:\n\n"
            "1. This tool allows editing of metadata in various image formats.\n"
            "2. Some metadata fields may not be editable due to technical limitations.\n"
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
        
    def getExifValue(self, exif_dict, ifd, tag):
        try:
            value = exif_dict[ifd][tag]
            if isinstance(value, bytes):
                try:
                    return value.decode()
                except:
                    return value.hex()
            return value
        except KeyError:
            return None

    def setExifValue(self, exif_dict, ifd, tag, value):
        try:
            if isinstance(exif_dict[ifd][tag], bytes):
                exif_dict[ifd][tag] = value.encode()
            else:
                exif_dict[ifd][tag] = value
        except KeyError:
            print(f"Warning: Unable to set value for tag {tag} in IFD {ifd}")
    
    def updateExifData(self, image_path):
        exif_dict = piexif.load(image_path)
        for row in range(self.metadataTable.rowCount()):
            tag = self.metadataTable.item(row, 0).text()
            value = self.metadataTable.item(row, 1).text()
            for ifd in ("0th", "Exif", "GPS", "1st"):
                for tag_id, tag_name in TAGS.items():
                    if tag_name == tag:
                        self.setExifValue(exif_dict, ifd, tag_id, value)
                        break
        return exif_dict
    
    def saveMetadataChanges(self, image_path):
        try:
            exif_dict = self.updateExifData(image_path)
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, image_path)
            return True
        except Exception as e:
            print(f"Error saving metadata for {image_path}: {e}")
            return False    

    def closeEvent(self, event):
        # Save any settings if needed
        super().closeEvent(event)