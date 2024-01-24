import logging
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, 
    QTextEdit, QGridLayout, QScrollArea, QSlider, QMessageBox, QLineEdit, QDockWidget, QProgressDialog, QMenu
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal
import os
from PIL import Image, ImageOps

# Constants
MAX_DISPLAY_SIZE = 809600  # maximum size (in pixels) for the longest side of the displayed image

# Logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

# Function to convert PIL image to QPixmap
def pil2pixmap(image):
    image_rgb = image.convert('RGB')
    data = image_rgb.tobytes('raw', 'RGB')
    qimage = QImage(data, image.size[0], image.size[1], QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qimage)
    return pixmap

# Custom QLabel for selectable images
class SelectableImageLabel(QLabel):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super(SelectableImageLabel, self).__init__(parent)
        self.selected = False
        self.setStyleSheet("border: 2px solid black;")  # Default style

    def mousePressEvent(self, event):
        self.clicked.emit()  # Emit clicked signal
        super(SelectableImageLabel, self).mousePressEvent(event)

# Auto Captioning window
class AutoCaptioningWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TagScribeR - Auto Captioning")
        self.setGeometry(100, 100, 800, 600)
        self.imageTextEdits = {}  # Dictionary to map image paths to their QTextEdits
        self.imageLabels = {}  # Dictionary to map image paths to their QLabel widgets
        self.selectedImages = {}  # Dictionary to track selected images
        self.setupUI()

    def setupUI(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        self.sizeSlider = QSlider(Qt.Horizontal, self)
        self.sizeSlider.setMinimum(50)
        self.sizeSlider.setMaximum(256)
        self.sizeSlider.setValue(100)
        self.sizeSlider.valueChanged.connect(self.updateThumbnails)
        self.layout.addWidget(self.sizeSlider)
        self.loadDirButton = QPushButton("Load Directory")
        self.loadDirButton.clicked.connect(self.loadDirectory)
        self.layout.addWidget(self.loadDirButton)
        self.autoCaptionButton = QPushButton("Auto Caption Selected Images")
        self.autoCaptionButton.clicked.connect(self.autoCaptionImages)
        self.layout.addWidget(self.autoCaptionButton)
        self.gridLayout = QGridLayout()
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.container = QWidget()
        self.container.setLayout(self.gridLayout)
        self.scrollArea.setWidget(self.container)
        self.layout.addWidget(self.scrollArea)

    def loadDirectory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            logging.info(f"Directory loaded: {dir_path}")
            self.displayImagesFromDirectory(dir_path)
        else:
            logging.warning("No directory was selected.")

    def displayImagesFromDirectory(self, dir_path):
        # Setup progress dialog
        image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        file_count = len(image_files)
        progress = QProgressDialog("Loading images...", "Abort", 0, file_count, self)
        progress.setWindowTitle("Loading...")
        progress.setCancelButton(None)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        try:
            for i in reversed(range(self.gridLayout.count())): 
                widget = self.gridLayout.itemAt(i).widget()
                if widget is not None: 
                    widget.deleteLater()
            self.imageTextEdits.clear()
            self.imageLabels.clear()
            for index, filename in enumerate(image_files):
                if progress.wasCanceled():
                    break
                progress.setValue(index)
                QApplication.processEvents()
                
                image_path = os.path.join(dir_path, filename)
                self.loadImageAndSetupUI(image_path, filename, index, file_count)
                
            progress.setValue(file_count)
        except Exception as e:
            logging.critical(f"Failed to display images from directory {dir_path}: {e}")
            QMessageBox.critical(self, "Error", "Could not display images from the directory.")

    def loadImageAndSetupUI(self, image_path, filename, index, file_count):
        try:
            image = Image.open(image_path)
            if max(image.size) > MAX_DISPLAY_SIZE:
                image = ImageOps.exif_transpose(image)
                image.thumbnail((MAX_DISPLAY_SIZE, MAX_DISPLAY_SIZE), Image.Resampling.LANCZOS)

            label = SelectableImageLabel(self)
            label.setPixmap(pil2pixmap(image).scaled(self.sizeSlider.value(), self.sizeSlider.value(), Qt.KeepAspectRatio))
            label.clicked.connect(lambda path=image_path: self.toggleImageSelection(path))
            self.imageLabels[image_path] = label
            self.selectedImages[image_path] = label

            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_file_path = os.path.join(image_path, txt_filename)
            description = ""
            try:
                if os.path.isfile(txt_file_path):
                    with open(txt_file_path, 'r') as file:
                        description = file.read()
            except Exception as e:
                logging.error(f"Error reading text file {txt_file_path}: {e}")

            textEdit = QTextEdit(self)
            textEdit.setText(description)
            self.imageTextEdits[image_path] = textEdit

            row = index // 4
            col = index % 4
            self.gridLayout.addWidget(label, row * 2, col)
            self.gridLayout.addWidget(textEdit, row * 2 + 1, col)
        except Exception as e:
            logging.error(f"Failed to process image {image_path}: {e}")
            QMessageBox.critical(self, "Error", f"Could not process the image: {os.path.basename(image_path)}")

    def updateThumbnails(self):
        try:
            size = self.sizeSlider.value()
            for image_path, label in self.imageLabels.items():
                try:
                    image = Image.open(image_path)
                    # Downscale the image for display if it's too large
                    if max(image.size) > MAX_DISPLAY_SIZE:
                        image.thumbnail((MAX_DISPLAY_SIZE, MAX_DISPLAY_SIZE), Image.Resampling.LANCZOS)
                    label.setPixmap(pil2pixmap(image).scaled(size, size, Qt.KeepAspectRatio))
                except Exception as e:
                    logging.error(f"Failed to update thumbnail for image {image_path}: {e}")
        except Exception as e:
            logging.critical("Failed to update thumbnails: {e}")

    def toggleImageSelection(self, image_path):
        if image_path in self.selectedImages:
            label = self.selectedImages[image_path]
            if label.selected:
                label.setStyleSheet("border: 2px solid black;")
                label.selected = False
            else:
                label.setStyleSheet("border: 2px solid blue;")  # Change border color to indicate selection
                label.selected = True
        else:
            logging.error(f"Image path not found in selectedImages: {image_path}")

    def autoCaptionImages(self):
        # This function should implement the auto-captioning feature
        logging.info("Auto-captioning started.")
        QMessageBox.information(self, "Info", "Auto Captioning will be implemented here.")

    def getSelectedImages(self):
        return [path for path, label in self.selectedImages.items() if label.selected]

    def updateCaption(self, image_path, caption):
        try:
            if image_path in self.imageTextEdits:
                self.imageTextEdits[image_path].setText(caption)
            else:
                QMessageBox.warning(self, "Warning", f"No text field found for image: {image_path}")
        except Exception as e:
            logging.error(f"Failed to update caption for image {image_path}: {e}")
