import logging
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QFileDialog, QLabel, QTextEdit, QGridLayout, QScrollArea, QSlider, 
    QMessageBox, QLineEdit, QDockWidget, QProgressDialog, QComboBox, QShortcut
)
from PyQt5.QtGui import QPixmap, QImage, QKeySequence
from PyQt5.QtCore import Qt, pyqtSignal, QRunnable, QThreadPool, QObject
from PIL import Image, ImageOps
import clip_interrogator
import io
import torch

def detect_gpu_acceleration():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'directml') and torch.backends.directml.is_available():
        return "directml"
    elif hasattr(torch.backends, 'zluda') and torch.backends.zluda.is_available():
        return "zluda"
    else:
        return "cpu"

def get_device(acceleration_method):
    if acceleration_method == "cuda":
        return torch.device("cuda")
    elif acceleration_method == "directml":
        return torch.device("directml")
    elif acceleration_method == "zluda":
        return torch.device("zluda")
    else:
        return torch.device("cpu")

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='auto_captioning_log.txt', filemode='w')

MAX_DISPLAY_SIZE = 809600
THUMBNAIL_SIZE = (200, 200)

class ThumbnailLoader(QRunnable):
    class Signals(QObject):
        finished = pyqtSignal(str, QPixmap)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self.signals = self.Signals()

    def run(self):
        try:
            with Image.open(self.image_path) as img:
                img.thumbnail(THUMBNAIL_SIZE)
                byte_array = io.BytesIO()
                img.save(byte_array, format='PNG')
                pixmap = QPixmap()
                pixmap.loadFromData(byte_array.getvalue())
                self.signals.finished.emit(self.image_path, pixmap)
        except Exception as e:
            logging.error(f"Failed to load thumbnail for {self.image_path}: {e}")

class SelectableImageLabel(QLabel):
    clicked = pyqtSignal(str)

    def __init__(self, imagePath='', parent=None):
        super(SelectableImageLabel, self).__init__(parent)
        self.selected = False
        self.imagePath = imagePath
        self.setStyleSheet("border: 2px solid black;")
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)

    def mousePressEvent(self, event):
        self.clicked.emit(self.imagePath)
        super(SelectableImageLabel, self).mousePressEvent(event)

class AutoCaptioningWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.acceleration_method = detect_gpu_acceleration()
        self.device = get_device(self.acceleration_method)
        
        if self.acceleration_method == "cpu":
            logging.warning("No GPU acceleration found. Using CPU.")
            QMessageBox.warning(self, "GPU Not Found", "No GPU acceleration found. The program will run on CPU, which may be slower.")
        else:
            logging.info(f"Using GPU acceleration: {self.acceleration_method}")
        
        self.setWindowTitle("TagScribeR - Auto Captioning")
        self.setGeometry(100, 100, 800, 600)
        self.imageTextEdits = {}
        self.imageLabels = {}
        self.selectedImages = {}
        self.thumbnail_cache = {}
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(4)
        self.setupUI()
        self.interrogator = None
        
        self.editHistory = {}
        self.redoHistory = {}
        
        self.shortcuts = {
            "Ctrl+S": (self.saveEdits, "Save all edits"),
            "Ctrl+Z": (self.undoLastEdit, "Undo last action"),
            "Ctrl+Shift+Z": (self.redoLastEdit, "Redo last action"),
            "Ctrl+A": (self.toggleSelectAllImages, "Toggle select/deselect all"),
            "Del": (self.clearSelectedCaptions, "Clear selected captions")
        }
        self.setupShortcuts()
        
    def setupShortcuts(self):
        for key, (func, _) in self.shortcuts.items():
            QShortcut(QKeySequence(key), self).activated.connect(func)

    def updateCustomShortcuts(self, custom_shortcuts):
        for action, new_key in custom_shortcuts.items():
            for key, (func, desc) in self.shortcuts.items():
                if desc == action:
                    try:
                        QShortcut(QKeySequence(new_key), self).activated.connect(func)
                    except Exception as e:
                        logging.error(f"Failed to update shortcut for {action}: {str(e)}")
                    break    
    
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
            QApplication.processEvents()

            try:
                pil_image = Image.open(image_path).convert("RGB")
                caption = self.interrogator.generate_caption(pil_image)
                textEdit = self.imageTextEdits[image_path]
                currentText = textEdit.toPlainText()
                
                # Update edit history
                if image_path not in self.editHistory:
                    self.editHistory[image_path] = []
                self.editHistory[image_path].append(currentText)
                
                if image_path not in self.redoHistory:
                    self.redoHistory[image_path] = []
                self.redoHistory[image_path].clear()
                
                textEdit.setText(caption)
                logging.info(f"Caption generated for {image_path}")
                
                if self.acceleration_method != "cpu":
                    if self.acceleration_method == "cuda":
                        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
                    elif self.acceleration_method == "directml":
                        print("DirectML acceleration in use")
                    elif self.acceleration_method == "zluda":
                        print("ZLUDA acceleration in use")
                
            except Exception as e:
                logging.error(f"Error captioning image {image_path}: {str(e)}")
                QMessageBox.warning(self, "Captioning Error", f"Failed to caption {os.path.basename(image_path)}: {str(e)}")

        progress.setValue(len(selected_images))
        QMessageBox.information(self, "Captioning Complete", "Selected images have been captioned.")

    def saveEdits(self):
        for image_path, textEdit in self.imageTextEdits.items():
            self.saveTextToFile(image_path, textEdit)
        QMessageBox.information(self, "Save Complete", "All edits have been saved successfully.")

    def undoLastEdit(self):
        for image_path, textEdit in self.imageTextEdits.items():
            if self.selectedImages[image_path].selected and self.editHistory.get(image_path):
                currentText = textEdit.toPlainText()
                lastText = self.editHistory[image_path].pop()
                textEdit.setText(lastText)
                if image_path not in self.redoHistory:
                    self.redoHistory[image_path] = []
                self.redoHistory[image_path].append(currentText)

    def redoLastEdit(self):
        for image_path, textEdit in self.imageTextEdits.items():
            if self.selectedImages[image_path].selected and self.redoHistory.get(image_path):
                currentText = textEdit.toPlainText()
                nextText = self.redoHistory[image_path].pop()
                textEdit.setText(nextText)
                if image_path not in self.editHistory:
                    self.editHistory[image_path] = []
                self.editHistory[image_path].append(currentText)

    def toggleSelectAllImages(self):
        all_selected = all(label.selected for label in self.selectedImages.values())
        for label in self.selectedImages.values():
            label.selected = not all_selected
            label.setStyleSheet("border: 2px solid blue;" if label.selected else "border: 2px solid black;")

    def clearSelectedCaptions(self):
        for image_path, label in self.selectedImages.items():
            if label.selected:
                textEdit = self.imageTextEdits[image_path]
                currentText = textEdit.toPlainText()
                if image_path not in self.editHistory:
                    self.editHistory[image_path] = []
                self.editHistory[image_path].append(currentText)
                textEdit.clear()
                if image_path not in self.redoHistory:
                    self.redoHistory[image_path] = []
                self.redoHistory[image_path].clear()

    # Make sure you have this method
    def initialize_interrogator(self):
        config = clip_interrogator.Config(
            caption_model_name=self.captionModelDropdown.currentText(),
            clip_model_name=self.clipModelDropdown.currentText(),
            device=self.device
        )
        self.interrogator = clip_interrogator.Interrogator(config)
        logging.info(f"Interrogator initialized with device: {self.device}")

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
        self.captionMethodDropdown.addItems(["BLIP-2", "Other Method 1", "Other Method 2"])
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

    def loadDirectory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            logging.info(f"Directory loaded: {dir_path}")
            self.displayImagesFromDirectory(dir_path)
        else:
            logging.warning("No directory was selected.")

    def displayImagesFromDirectory(self, dir_path):
        self.clearLayout(self.gridLayout)
        self.imageLabels.clear()
        self.imageTextEdits.clear()
        self.selectedImages.clear()
        self.thumbnail_cache.clear()

        image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        
        progress = QProgressDialog("Loading images...", "Abort", 0, len(image_files), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        for index, filename in enumerate(image_files):
            if progress.wasCanceled():
                break
            progress.setValue(index)
            QApplication.processEvents()

            image_path = os.path.join(dir_path, filename)
            self.createImagePlaceholder(image_path, index)
            self.queueThumbnailLoad(image_path)

        progress.setValue(len(image_files))

    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def createImagePlaceholder(self, image_path, index):
        label = SelectableImageLabel(imagePath=image_path)
        label.setFixedSize(THUMBNAIL_SIZE[0], THUMBNAIL_SIZE[1])
        label.setText("Loading...")
        label.clicked.connect(self.toggleImageSelection)
        
        row = index // 4
        col = index % 4
        self.gridLayout.addWidget(label, row * 2, col)
        
        self.imageLabels[image_path] = label
        self.selectedImages[image_path] = label
        
        textEdit = QTextEdit(self)
        self.imageTextEdits[image_path] = textEdit
        self.gridLayout.addWidget(textEdit, row * 2 + 1, col)

    def queueThumbnailLoad(self, image_path):
        loader = ThumbnailLoader(image_path)
        loader.signals.finished.connect(self.onThumbnailLoaded)
        self.thread_pool.start(loader)

    def onThumbnailLoaded(self, image_path, pixmap):
        if image_path in self.imageLabels:
            label = self.imageLabels[image_path]
            label.setPixmap(pixmap)
            label.setText("")
            self.thumbnail_cache[image_path] = pixmap

    def toggleImageSelection(self, image_path):
        if image_path in self.selectedImages:
            label = self.selectedImages[image_path]
            label.selected = not label.selected
            label.setStyleSheet("border: 2px solid blue;" if label.selected else "border: 2px solid black;")
            logging.info(f"Image selection toggled: {image_path}, Selected: {label.selected}")
        else:
            logging.error(f"Image path not found in selectedImages: {image_path}")

    def updateThumbnails(self):
        size = self.sizeSlider.value()
        for image_path, label in self.imageLabels.items():
            if image_path in self.thumbnail_cache:
                pixmap = self.thumbnail_cache[image_path]
                scaled_pixmap = pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label.setPixmap(scaled_pixmap)

    def saveEdits(self):
        for image_path, textEdit in self.imageTextEdits.items():
            self.saveTextToFile(image_path, textEdit)
        QMessageBox.information(self, "Save Complete", "All edits have been saved successfully.")

    def saveTextToFile(self, image_path, textEdit):
        txt_file_path = os.path.splitext(image_path)[0] + '.txt'
        try:
            with open(txt_file_path, 'w') as file:
                file.write(textEdit.toPlainText())
            logging.info(f"Saved text for {image_path}")
        except Exception as e:
            logging.error(f"Failed to save text for {image_path}: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save text for {os.path.basename(image_path)}: {str(e)}")
        

if __name__ == "__main__":
    app = QApplication([])
    window = AutoCaptioningWindow()
    window.show()
    app.exec_()