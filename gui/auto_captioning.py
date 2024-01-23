import logging
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, 
    QTextEdit, QGridLayout, QScrollArea, QSlider, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import os
from PIL import Image, ImageOps

# Add this constant near the top of your script
MAX_DISPLAY_SIZE = 8096  # maximum size (in pixels) for the longest side of the displayed image

# Setup basic configuration for logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

def pil2pixmap(image):
    """Convert a PIL image to a QPixmap"""
    image_rgb = image.convert('RGB')
    data = image_rgb.tobytes('raw', 'RGB')
    qimage = QImage(data, image.size[0], image.size[1], QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qimage)
    return pixmap

class AutoCaptioningWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TagScribeR - Auto Captioning")
        self.setGeometry(100, 100, 800, 600)
        self.imageTextEdits = {}
        self.imageLabels = {}
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
        try:
            for i in reversed(range(self.gridLayout.count())): 
                widget = self.gridLayout.itemAt(i).widget()
                if widget is not None: 
                    widget.deleteLater()
            self.imageTextEdits.clear()
            self.imageLabels.clear()
            for index, filename in enumerate(sorted(os.listdir(dir_path))):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(dir_path, filename)
                    try:
                        image = Image.open(image_path)
                        # Downscale the image for display if it's too large
                        if max(image.size) > MAX_DISPLAY_SIZE:
                            image = ImageOps.exif_transpose(image)
                            image.thumbnail((MAX_DISPLAY_SIZE, MAX_DISPLAY_SIZE), Image.Resampling.LANCZOS)
                        label = QLabel(self)
                        pixmap = pil2pixmap(image)
                        label.setPixmap(pixmap.scaled(self.sizeSlider.value(), self.sizeSlider.value(), Qt.KeepAspectRatio))
                        self.imageLabels[image_path] = label
                        textEdit = QTextEdit(self)
                        textEdit.setText("")
                        self.imageTextEdits[image_path] = textEdit
                        row = index // 4
                        col = index % 4
                        self.gridLayout.addWidget(label, row * 2, col)
                        self.gridLayout.addWidget(textEdit, row * 2 + 1, col)
                    except Exception as e:
                        logging.error(f"Failed to process image {image_path}: {e}")
                        QMessageBox.critical(self, "Error", f"Could not process the image: {os.path.basename(image_path)}")
        except Exception as e:
            logging.critical(f"Failed to display images from directory {dir_path}: {e}")
            QMessageBox.critical(self, "Error", "Could not display images from the directory.")

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

    def autoCaptionImages(self):
        # This function should implement the auto-captioning feature
        logging.info("Auto-captioning started.")
        QMessageBox.information(self, "Info", "Auto Captioning will be implemented here.")

    def getSelectedImages(self):
        return list(self.imageTextEdits.keys())

    def updateCaption(self, image_path, caption):
        try:
            if image_path in self.imageTextEdits:
                self.imageTextEdits[image_path].setText(caption)
            else:
                QMessageBox.warning(self, "Warning", f"No text field found for image: {image_path}")
        except Exception as e:
            logging.error(f"Failed to update caption for image {image_path}: {e}")
