import logging
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, 
                             QFileDialog, QLabel, QTextEdit, QGridLayout, QScrollArea, QSlider, 
                             QMessageBox, QLineEdit, QDockWidget)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QMenu
import os
from PIL import Image, ImageOps

# Add this constant near the top of your script
MAX_DISPLAY_SIZE = 809600  # maximum size (in pixels) for the longest side of the displayed image

# Setup basic configuration for logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

def pil2pixmap(image):
    """Convert a PIL image to a QPixmap"""
    image_rgb = image.convert('RGB')
    data = image_rgb.tobytes('raw', 'RGB')
    qimage = QImage(data, image.size[0], image.size[1], QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qimage)
    return pixmap

class SelectableImageLabel(QLabel):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super(SelectableImageLabel, self).__init__(parent)
        self.selected = False
        self.setStyleSheet("border: 2px solid black;")  # Default style

    def mousePressEvent(self, event):
        self.clicked.emit()  # Emit clicked signal
        super(SelectableImageLabel, self).mousePressEvent(event)

class GalleryWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TagScribeR - Gallery")
        self.setGeometry(100, 100, 800, 600)
        self.imageTextEdits = {}  # Dictionary to map image paths to their QTextEdits
        self.imageLabels = {}  # Dictionary to map image paths to their QLabel widgets
        self.selectedImages = {}  # {image_path: SelectableImageLabel}
        self.setupUI()
        self.setupTagsPanel()

    def setupTagsPanel(self):
        self.tagsDockWidget = QDockWidget("Tags", self)  # Dockable tags panel
        self.tagsPanel = QWidget()  # This is the main panel for tags
        self.tagsPanelLayout = QVBoxLayout()
        self.tagsPanel.setLayout(self.tagsPanelLayout)
        
        # New tag text field
        self.newTagLineEdit = QLineEdit()
        self.tagsPanelLayout.addWidget(self.newTagLineEdit)
        
        # Button to add new tag
        self.addTagButton = QPushButton("Add Tag", self)
        self.addTagButton.clicked.connect(self.addTag)
        self.tagsPanelLayout.addWidget(self.addTagButton)

        # Scrollable area for tag buttons
        self.tagsScrollArea = QScrollArea()
        self.tagsContainer = QWidget()
        self.tagsContainerLayout = QVBoxLayout()
        self.tagsContainer.setLayout(self.tagsContainerLayout)
        self.tagsScrollArea.setWidget(self.tagsContainer)
        self.tagsScrollArea.setWidgetResizable(True)
        self.tagsPanelLayout.addWidget(self.tagsScrollArea)
        
        self.tagsDockWidget.setWidget(self.tagsPanel)
        self.addDockWidget(Qt.RightDockWidgetArea, self.tagsDockWidget)
        
    def loadTags(self):
        try:
            with open('tags.txt', 'r') as file:
                for line in file:
                    tagText = line.strip()
                    self.createTagButton(tagText)
        except FileNotFoundError:
            pass  # File doesn't exist yet

    def saveTags(self):
        with open('tags.txt', 'w') as file:
            for i in range(self.tagsContainerLayout.count()):
                widget = self.tagsContainerLayout.itemAt(i).widget()
                if isinstance(widget, QPushButton):
                    file.write(widget.text() + '\n')

    def createTagButton(self, tagText):
        tagButton = QPushButton(tagText)
        tagButton.clicked.connect(lambda: self.tagButtonClicked(tagText))
        tagButton.setContextMenuPolicy(Qt.CustomContextMenu)
        tagButton.customContextMenuRequested.connect(lambda pos, btn=tagButton: self.tagButtonContextMenu(pos, btn))
        self.tagsContainerLayout.addWidget(tagButton)

    def addTag(self):
        tagText = self.newTagLineEdit.text().strip()
        if tagText:
            self.createTagButton(tagText)
            self.newTagLineEdit.clear()
            self.saveTags()  # Save tags to file

    def tagButtonContextMenu(self, pos, btn):
        # Create a context menu for the tag button
        menu = QMenu()
        deleteAction = menu.addAction("Delete Tag")
        action = menu.exec_(btn.mapToGlobal(pos))
        if action == deleteAction:
            self.tagsContainerLayout.removeWidget(btn)
            btn.deleteLater()
            self.saveTags()  # Update tags file after deletion

    def tagButtonClicked(self, tagText):
        for image_path, textEdit in self.imageTextEdits.items():
            if self.selectedImages[image_path].selected:
                currentText = textEdit.toPlainText()
                newText = f"{currentText}, {tagText}" if currentText else tagText
                textEdit.setText(newText)
                # Update text file
                txt_file_path = os.path.splitext(image_path)[0] + '.txt'
                with open(txt_file_path, 'w') as file:
                    file.write(newText)
        self.saveTags()  # Optional: Save tags after modification, if needed

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
        # Clear existing items in the grid layout
        for i in reversed(range(self.gridLayout.count())): 
            widget = self.gridLayout.itemAt(i).widget()
            if widget is not None: 
                widget.deleteLater()

        self.imageTextEdits.clear()
        self.imageLabels.clear()
        self.selectedImages.clear()  # Clear previously selected images

        # Load and display images
        for index, filename in enumerate(sorted(os.listdir(dir_path))):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Corrected indentation
                image_path = os.path.join(dir_path, filename)
                try:
                    image = Image.open(image_path)
                    
                    # Downscale the image for display if it's too large
                    if max(image.size) > MAX_DISPLAY_SIZE:
                        image = ImageOps.exif_transpose(image)  # Correct the orientation of the image
                        image.thumbnail((MAX_DISPLAY_SIZE, MAX_DISPLAY_SIZE), Image.Resampling.LANCZOS)

                    label = SelectableImageLabel(self)  # Use SelectableImageLabel
                    label.clicked.connect(lambda path=image_path: self.toggleImageSelection(path))
                    pixmap = pil2pixmap(image)
                    label.setPixmap(pixmap.scaled(self.sizeSlider.value(), self.sizeSlider.value(), Qt.KeepAspectRatio))
                    self.imageLabels[image_path] = label
                    self.selectedImages[image_path] = label  # Add to selectedImages dictionary

                    txt_filename = os.path.splitext(filename)[0] + '.txt'
                    txt_file_path = os.path.join(dir_path, txt_filename)
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
                QMessageBox.critical(self, "Error", f"Could not process the image: {os.path.basename(image_path)}")

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
