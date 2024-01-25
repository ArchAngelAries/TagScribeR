import logging
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel,
    QTextEdit, QGridLayout, QScrollArea, QSlider, QMessageBox, QLineEdit, QDockWidget, QProgressDialog, QMenu, QComboBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal
from PIL import Image, ImageOps
import torch
from blip2 import BLIP2
from lavis.models import load_model_and_preprocess

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

class BLIP2:
    def __init__(self, model_type):
        # setup device to use
        self.device = torch.device("directml") if "directml" in torch.__version__ else "cpu"
        self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2", model_type=model_type, is_eval=True, device=self.device)

def generate_caption(
        self, 
        image: Image, 
        num_beams: int = 9,
        use_nucleus_sampling: bool = False,
        max_length: int = 40,
        min_length: int = 30,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ):
    # prepare the image
    image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)

    # Ensure `do_sample` is set correctly based on the sampling method
    do_sample = use_nucleus_sampling or top_p < 1.0

    captions = self.model.generate(
        {"image": image},
        num_beams=num_beams,
        do_sample=do_sample,  # Added to address the warning
        use_nucleus_sampling=use_nucleus_sampling,
        max_length=max_length,
        min_length=min_length,
        top_p=top_p if do_sample else None,  # Ensure top_p is used only when do_sample is True
        repetition_penalty=repetition_penalty,
    )
    return captions

    def unload(self):
        # Ensure all references to the model and processors are deleted
        if self.model:
            del self.model
        if self.vis_processors:
            del self.vis_processors
        logging.info("BLIP2 model resources have been released")

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
        self.blip2_model = BLIP2(model_type='pretrain')
        self.setupUI()

    def setupUI(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.mainLayout = QVBoxLayout(self.central_widget)

        # Controls for image selection and captioning
        self.controlsLayout = QHBoxLayout()

        # Create and add a size slider
        self.sizeSlider = QSlider(Qt.Horizontal)
        self.sizeSlider.setMinimum(50)
        self.sizeSlider.setMaximum(256)
        self.sizeSlider.setValue(100)
        self.sizeSlider.valueChanged.connect(self.updateThumbnails)
        self.controlsLayout.addWidget(self.sizeSlider)

        # Create and add a directory load button
        self.loadDirButton = QPushButton("Load Directory")
        self.loadDirButton.clicked.connect(self.loadDirectory)
        self.controlsLayout.addWidget(self.loadDirButton)

        # Create and add the caption button
        self.autoCaptionButton = QPushButton("Auto Caption Selected Images")
        self.autoCaptionButton.clicked.connect(self.autoCaptionImages)
        self.controlsLayout.addWidget(self.autoCaptionButton)
        self.autoCaptionButton.clicked.connect(self.onAutoCaptionButtonClicked)

        # Add controls layout to the main layout
        self.mainLayout.addLayout(self.controlsLayout)

        # Additional controls for BLIP-2 settings
        self.settingsLayout = QHBoxLayout()

        # Create and add the sampling method dropdown
        self.samplingMethodDropdown = QComboBox()
        self.samplingMethodDropdown.addItems(["Nucleus", "Top-K"])
        self.settingsLayout.addWidget(self.samplingMethodDropdown)

        # Create and add the topP slider
        self.topPSlider = QSlider(Qt.Horizontal)
        self.topPSlider.setMinimum(0)
        self.topPSlider.setMaximum(100)
        self.topPSlider.setValue(90)  # For top_p = 0.9
        self.settingsLayout.addWidget(self.topPSlider)

        # Create and add the numBeams slider
        self.numBeamsSlider = QSlider(Qt.Horizontal)
        self.numBeamsSlider.setMinimum(1)
        self.numBeamsSlider.setMaximum(10)
        self.numBeamsSlider.setValue(3)
        self.settingsLayout.addWidget(self.numBeamsSlider)

        # Add settings layout to the main layout
        self.mainLayout.addLayout(self.settingsLayout)

        # Setup scroll area and image grid layout
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.container = QWidget()
        self.gridLayout = QGridLayout(self.container)
        self.scrollArea.setWidget(self.container)
        self.mainLayout.addWidget(self.scrollArea)
        
                # NEW: Add a model selection dropdown
        self.modelSelectionDropdown = QComboBox(self)
        self.modelSelectionDropdown.addItems(["coco", "pretrain"])
        self.modelSelectionDropdown.currentTextChanged.connect(self.loadSelectedModel)
        self.controlsLayout.addWidget(self.modelSelectionDropdown)

        # NEW: Add an unload button
        self.unloadModelButton = QPushButton("Unload Model")
        self.unloadModelButton.clicked.connect(self.unloadModel)
        self.controlsLayout.addWidget(self.unloadModelButton)

        # NEW: Add labels and value indicators for sliders
        self.topPLabel = QLabel("Top P: 0.90")  # Initial value indicator
        self.numBeamsLabel = QLabel("Num Beams: 3")  # Initial value indicator
        self.topPSlider.valueChanged.connect(lambda value: self.topPLabel.setText(f"Top P: {value / 100:.2f}"))
        self.numBeamsSlider.valueChanged.connect(lambda value: self.numBeamsLabel.setText(f"Num Beams: {value}"))
        self.settingsLayout.addWidget(self.topPLabel)
        self.settingsLayout.addWidget(self.numBeamsLabel)
        
            # NEW: Method to load the selected model
    def loadSelectedModel(self, model_type):
        try:
            self.blip2_model = BLIP2(model_type=model_type)
            logging.info(f"Loaded model: {model_type}")
        except Exception as e:
            logging.error(f"Error loading model {model_type}: {e}")
            QMessageBox.critical(self, "Error", f"Could not load the model: {model_type}")
            
     # NEW: Method to unload the model
    def unloadModel(self):
        # Safely unload the model
        try:
            if self.blip2_model:
                self.blip2_model.unload()
                self.blip2_model = None
                logging.info("Model unloaded successfully")
            else:
                logging.info("No model is currently loaded")
        except Exception as e:
            logging.error(f"Error unloading model: {e}")
            QMessageBox.critical(self, "Error", "Could not unload the model")

        
    def onAutoCaptionButtonClicked(self):
        top_p = self.topPSlider.value() / 100.0
        num_beams = self.numBeamsSlider.value()
        use_nucleus_sampling = self.samplingMethodDropdown.currentText() == "Nucleus"
        # Call the captioning function with the parameters
        self.autoCaptionImages(top_p, num_beams, use_nucleus_sampling)

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
            

    def autoCaptionImages(self, top_p, num_beams, use_nucleus_sampling):
        selected_images = self.getSelectedImages()
        progress = QProgressDialog("Auto-Captioning images...", "Abort", 0, len(selected_images), self)
        progress.setWindowTitle("Auto-Captioning...")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        for idx, image_path in enumerate(selected_images):
            if progress.wasCanceled():
                break
            progress.setValue(idx)
            QApplication.processEvents()

            caption = self.generate_caption(image_path, num_beams, use_nucleus_sampling, top_p)
            self.updateCaption(image_path, caption)

        progress.setValue(len(selected_images))

    def generate_caption(self, image_path, num_beams, use_nucleus_sampling, top_p):
        max_length = 40  # Consider making this a UI component
        min_length = 30  # Consider making this a UI component

        try:
            image = Image.open(image_path).convert("RGB")
            caption = self.blip2_model.generate_caption(
                image,
                num_beams=num_beams,
                use_nucleus_sampling=use_nucleus_sampling,
                max_length=max_length,
                min_length=min_length,
                top_p=top_p,
            )
            return caption
        except Exception as e:
            logging.error(f"Error generating caption for {image_path}: {e}")
            return "Caption generation failed."

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
