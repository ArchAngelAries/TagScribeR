import logging
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel,
    QTextEdit, QGridLayout, QScrollArea, QSlider, QMessageBox, QLineEdit, QDockWidget, QProgressDialog, QComboBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image, ImageOps
import clip_interrogator  # Ensure you have clip_interrogator properly installed and imported

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

MAX_DISPLAY_SIZE = 809600  # maximum size (in pixels) for the longest side of the displayed image

def pil2pixmap(image):
    image_rgb = image.convert('RGB')
    data = image_rgb.tobytes('raw', 'RGB')
    qimage = QImage(data, image.size[0], image.size[1], QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qimage)
    return pixmap

class SelectableImageLabel(QLabel):
    def __init__(self, parent=None):
        super(SelectableImageLabel, self).__init__(parent)
        self.selected = False
        self.setStyleSheet("border: 2px solid black;")

    def mousePressEvent(self, event):
        self.selected = not self.selected
        self.setStyleSheet("border: 2px solid blue;" if self.selected else "border: 2px solid black;")
        super(SelectableImageLabel, self).mousePressEvent(event)

class AutoCaptioningWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TagScribeR - Auto Captioning")
        self.setGeometry(100, 100, 800, 600)
        self.imageTextEdits = {}
        self.imageLabels = {}
        self.selectedImages = {}
        self.allSelected = False
        self.originalCaptions = {}  # To store original captions for undo functionality
        self.thumbnailSize = 150  # Default thumbnail size
        self.setupUI()
        self.interrogator = None

    def setupUI(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.mainLayout = QVBoxLayout(self.central_widget)

        self.loadDirButton = QPushButton("Load Directory")
        self.loadDirButton.clicked.connect(self.loadDirectory)
        self.mainLayout.addWidget(self.loadDirButton)
        
        self.captionMethodLabel = QLabel("Caption Method:")
        self.mainLayout.addWidget(self.captionMethodLabel)
        
        self.captionMethodDropdown = QComboBox()
        self.captionMethodDropdown.addItems(["BLIP-2", "Other Method 1", "Other Method 2"])  # Add other methods as needed
        self.mainLayout.addWidget(self.captionMethodDropdown)

        self.captionModelLabel = QLabel("Caption Model:")
        self.mainLayout.addWidget(self.captionModelLabel)

        self.captionModelDropdown = QComboBox()
        self.captionModelDropdown.addItems(clip_interrogator.list_caption_models())
        self.mainLayout.addWidget(self.captionModelDropdown)
        
        self.clipModelLabel = QLabel("CLIP Model:")
        self.mainLayout.addWidget(self.clipModelLabel)
        
        self.clipModelDropdown = QComboBox()
        self.clipModelDropdown.addItems(clip_interrogator.list_clip_models())
        self.mainLayout.addWidget(self.clipModelDropdown)
        
        self.captionModelDropdown.currentTextChanged.connect(self.initialize_interrogator)
        self.clipModelDropdown.currentTextChanged.connect(self.initialize_interrogator)

        self.captionButton = QPushButton("Caption Selected Images")
        self.captionButton.clicked.connect(self.captionImages)
        self.mainLayout.addWidget(self.captionButton)
        
        self.toggleSelectButton = QPushButton("Select/Deselect All")
        self.toggleSelectButton.clicked.connect(self.toggleSelectAllImages)
        self.mainLayout.addWidget(self.toggleSelectButton)
        
        self.sizeSliderLayout = QHBoxLayout()
        self.thumbnailSizeLabel = QLabel("Thumbnail Size: 100")
        self.sizeSlider = QSlider(Qt.Horizontal)
        self.sizeSlider.setMinimum(50)
        self.sizeSlider.setMaximum(353)
        self.sizeSlider.setValue(100)
        self.sizeSlider.valueChanged.connect(self.updateThumbnails)
        self.sizeSlider.valueChanged.connect(lambda value: self.thumbnailSizeLabel.setText(f"Thumbnail Size: {value}"))
        self.sizeSliderLayout.addWidget(self.thumbnailSizeLabel)
        self.sizeSliderLayout.addWidget(self.sizeSlider)
        self.mainLayout.addLayout(self.sizeSliderLayout)

        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.container = QWidget()
        self.gridLayout = QGridLayout(self.container)
        self.scrollArea.setWidget(self.container)
        self.mainLayout.addWidget(self.scrollArea)
        
        self.saveEditsButton = QPushButton("Save Edits")
        self.saveEditsButton.clicked.connect(self.saveEdits)
        self.mainLayout.addWidget(self.saveEditsButton)
        
        self.undoButton = QPushButton("Undo")
        self.undoButton.clicked.connect(self.undoLastEdit)
        self.mainLayout.addWidget(self.undoButton)

    def loadDirectory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            logging.info(f"Directory loaded: {dir_path}")
            self.displayImagesFromDirectory(dir_path)
        else:
            logging.warning("No directory was selected.")

    def displayImagesFromDirectory(self, dir_path):
        progress = QProgressDialog("Loading images...", "Abort", 0, len(os.listdir(dir_path)), self)
        progress.setWindowTitle("Loading...")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        # Clear existing widgets from the layout and dictionaries
        self.clearLayout(self.gridLayout)
        self.imageTextEdits.clear()
        self.imageLabels.clear()
        self.selectedImages.clear()

        for index, filename in enumerate(os.listdir(dir_path)):
            if progress.wasCanceled():
                break
            progress.setValue(index)
            QApplication.processEvents()

            image_path = os.path.join(dir_path, filename)
            self.loadImageAndSetupUI(image_path, filename, index)

        progress.setValue(len(os.listdir(dir_path)))
        
    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()    
        
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

    def loadImageAndSetupUI(self, image_path, filename, index):
        try:
            image = Image.open(image_path)
            if max(image.size) > MAX_DISPLAY_SIZE:
                image = ImageOps.exif_transpose(image)
                image.thumbnail((MAX_DISPLAY_SIZE, MAX_DISPLAY_SIZE), Image.Resampling.LANCZOS)

            # Convert PIL image to QPixmap
            pixmap = pil2pixmap(image)

            # Maintain aspect ratio
            pixmap = pixmap.scaled(self.thumbnailSize, self.thumbnailSize, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            label = SelectableImageLabel(self)
            label.setPixmap(pixmap)
            self.imageLabels[image_path] = label
            self.selectedImages[image_path] = label

            textEdit = QTextEdit(self)
            self.imageTextEdits[image_path] = textEdit

            row = index // 4
            col = index % 4
            self.gridLayout.addWidget(label, row * 2, col)
            self.gridLayout.addWidget(textEdit, row * 2 + 1, col)
        except Exception as e:
            logging.error(f"Failed to process image {image_path}: {e}")
            QMessageBox.critical(self, "Error", f"Could not process the image: {os.path.basename(image_path)}")

    def toggleSelectAllImages(self):
        self.allSelected = not self.allSelected  # Toggle the selection state
        for label in self.selectedImages.values():
            label.selected = self.allSelected
            label.setStyleSheet("border: 2px solid blue;" if self.allSelected else "border: 2px solid black;")
        # Update the button text based on the current state
        self.toggleSelectButton.setText("Deselect All" if self.allSelected else "Select All")

    def saveEdits(self):
        for image_path, textEdit in self.imageTextEdits.items():
            with open(image_path.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt'), 'w') as file:
                file.write(textEdit.toPlainText())
            logging.info(f"Saved edits for {image_path}")

    def undoLastEdit(self):
        # Implement the functionality to undo the last edit in the active QTextEdit
        # You might need to keep track of the edits or leverage the undo functionality of QTextEdit
        activeTextEdit = self.central_widget.focusWidget()
        if isinstance(activeTextEdit, QTextEdit):
            activeTextEdit.undo()

    def initialize_interrogator(self):
        config = clip_interrogator.Config(
            # Set the model names based on the UI selection
            caption_model_name=self.captionModelDropdown.currentText(),
            clip_model_name=self.clipModelDropdown.currentText(),
            # Other config settings
            # ...
        )
        self.interrogator = clip_interrogator.Interrogator(config)

    def captionImages(self):
        if not self.interrogator:
            self.initialize_interrogator()

        selected_images = [path for path, label in self.selectedImages.items() if label.selected]

        progress = QProgressDialog("Captioning images...", "Abort", 0, len(selected_images), self)
        progress.setWindowTitle("Captioning in progress...")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        for idx, image_path in enumerate(selected_images):
            if progress.wasCanceled():
                break
            progress.setValue(idx)
            QApplication.processEvents()  # Process events to keep the UI responsive

            pil_image = Image.open(image_path).convert("RGB")
            caption = self.interrogator.generate_caption(pil_image)
            self.imageTextEdits[image_path].setText(caption)

            # Save the caption to a text file
            self.save_caption_to_file(image_path, caption)

        progress.setValue(len(selected_images))

    def save_caption_to_file(self, image_path, caption):
        # Create the .txt filename based on the image filename
        txt_filename = os.path.splitext(image_path)[0] + '.txt'

        try:
            # Write the caption to the .txt file
            with open(txt_filename, 'w') as txt_file:
                txt_file.write(caption)
            logging.info(f"Caption saved to {txt_filename}")
        except Exception as e:
            logging.error(f"Error saving caption to file {txt_filename}: {e}")
            QMessageBox.critical(self, "Error", f"Could not save caption to file: {os.path.basename(txt_filename)}")

if __name__ == "__main__":
    app = QApplication([])
    window = AutoCaptioningWindow()
    window.show()
    app.exec_()
