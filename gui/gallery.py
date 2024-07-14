import shutil
import json
import logging
import sys
from functools import lru_cache
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QFileDialog, QLabel, QTextEdit, QGridLayout, QScrollArea, QSlider, 
    QMessageBox, QLineEdit, QDockWidget, QSizePolicy, QProgressDialog, QMenu, QListWidget, QListWidgetItem, QShortcut, QInputDialog
)
from PyQt5.QtGui import QPixmap, QImage, QKeySequence
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtCore import pyqtSlot
import os
from PIL import Image, ImageOps
import subprocess
import gc
from PyQt5.QtCore import QRunnable, QThreadPool, pyqtSignal, QObject
import io

@lru_cache(maxsize=100)
def load_image(image_path):
    try:
        image = Image.open(image_path)
        if max(image.size) > MAX_DISPLAY_SIZE:
            image = ImageOps.exif_transpose(image)
            image.thumbnail((MAX_DISPLAY_SIZE, MAX_DISPLAY_SIZE), Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        logging.error(f"Failed to load image {image_path}: {e}")
        return None

# Add this constant near the top of your script
MAX_DISPLAY_SIZE = 809600  # maximum size (in pixels) for the longest side of the displayed image
THUMBNAIL_SIZE = (200, 200)  # Adjust as needed

# Setup basic configuration for logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='gallery_log.txt',
                    filemode='w')

def pil2pixmap(image):
    """Convert a PIL image to a QPixmap"""
    image_rgb = image.convert('RGB')
    data = image_rgb.tobytes('raw', 'RGB')
    qimage = QImage(data, image.size[0], image.size[1], QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qimage)
    return pixmap
    
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

    def setPixmap(self, pixmap):
        super().setPixmap(pixmap)
        self.setText("")  # Clear any existing text when setting a pixmap

    def setText(self, text):
        super().setText(text)
        if text:
            self.setPixmap(QPixmap())  # Clear any existing pixmap when setting text
            
    def mousePressEvent(self, event):
        self.clicked.emit(self.imagePath)
        super(SelectableImageLabel, self).mousePressEvent(event)       

class GalleryWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TagScribeR - Gallery")
        self.setGeometry(100, 100, 800, 600)
        self.imageTextEdits = {}  # Dictionary to map image paths to their QTextEdits
        self.imageLabels = {}  # Dictionary to map image paths to their QLabel widgets
        self.selectedImages = {}  # {image_path: SelectableImageLabel}

        self.collectionFolderPath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "Dataset Collections"))
        self.collectionsFilePath = os.path.join(self.collectionFolderPath, "collections.json")

        self.setupUI()
        self.setupTagsPanel()
        self.loadTags()
        self.setupCollectionsPanel()
        self.loadCollections()

        self.editButtonsLayout = QHBoxLayout()  # Layout to hold edit-related buttons

        # Add a Save Edits button
        self.saveEditsButton = QPushButton("Save Edits", self)
        self.saveEditsButton.setFixedSize(100, 30)  # Make the button smaller
        self.saveEditsButton.clicked.connect(self.saveAllEdits)
        self.editButtonsLayout.addWidget(self.saveEditsButton)
        
        # Undo Button
        self.undoButton = QPushButton("Undo", self)
        self.undoButton.setFixedSize(100, 30)  # Make the button smaller
        self.undoButton.clicked.connect(self.undoLastAction)
        self.undoButton.clicked.connect(self.undoLastEdit)
        self.editButtonsLayout.addWidget(self.undoButton)
        
        # Redo Button
        self.redoButton = QPushButton("Redo", self)
        self.redoButton.setFixedSize(100, 30)  # Make the button smaller
        self.redoButton.clicked.connect(self.redoLastEdit)
        self.editButtonsLayout.addWidget(self.redoButton)
        
        # Add the new Clear Captions button
        self.clearCaptionsButton = QPushButton("Clear Selected Captions", self)
        self.clearCaptionsButton.setFixedSize(120, 40)  # Make the button smaller
        self.clearCaptionsButton.clicked.connect(self.clearSelectedCaptions)
        self.editButtonsLayout.addWidget(self.clearCaptionsButton)     
        
        self.mainLayout.addLayout(self.editButtonsLayout)
        
        self.editHistory = {image_path: [] for image_path in self.imageTextEdits.keys()}
        self.redoHistory = {image_path: [] for image_path in self.imageTextEdits.keys()}

        self.loadCollections()
        self.thumbnail_cache = {}
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(4)  # Adjust based on your system
        
        if os.path.exists(self.collectionsFilePath):
            with open(self.collectionsFilePath, 'r') as file:
                collections = json.load(file)
                for collection in collections:
                    self.collectionsList.addItem(collection)
        else:
            with open(self.collectionsFilePath, 'w') as file:
                json.dump([], file)
                
        self.shortcuts = {
            "Ctrl+S": (self.saveAllEdits, "Save all edits"),
            "Ctrl+Z": (self.undoLastEdit, "Undo last action"),
            "Ctrl+Shift+Z": (self.redoLastEdit, "Redo last action"),
            "Ctrl+A": (self.toggleSelectAllImages, "Toggle select/deselect all"),
            "Ctrl+F": (self.focusSearchBar, "Focus main search"),
            "Del": (self.clearSelectedCaptions, "Clear selected captions"),
            "Ctrl+L": (self.loadDirectory, "Load directory"),
            "Ctrl+C": (self.copySelectedToCollection, "Save selected to collection")
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

    def toggleSelectAllImages(self):
        if all(label.selected for label in self.imageLabels.values()):
            for label in self.imageLabels.values():
                label.selected = False
                label.setStyleSheet("border: 2px solid black;")
            self.selectAllButton.setText("Select All")
        else:
            for label in self.imageLabels.values():
                label.selected = True
                label.setStyleSheet("border: 2px solid blue;")
            self.selectAllButton.setText("Deselect All")
                
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
        # Load existing caption text if available
        txt_file_path = os.path.splitext(image_path)[0] + '.txt'
        description = self.load_description(txt_file_path)
        textEdit.setText(description)
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
            label.setText("")  # Clear the "Loading..." text
            self.thumbnail_cache[image_path] = pixmap
    
    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()    
        
    def deselectAllImages(self):
        for label in self.imageLabels.values():
            label.selected = False
            label.setStyleSheet("border: 2px solid black;")
    
    def focusSearchBar(self):
        self.searchBox.setFocus()
    
    def copySelectedToCollection(self):
        selected_images = [path for path, label in self.selectedImages.items() if label.selected]
        if not selected_images:
            QMessageBox.warning(self, "Warning", "No images selected.")
            return
    
        # Get list of collections
        collections = [self.collectionsList.item(i).text() for i in range(self.collectionsList.count())]
        
        # Let user choose a collection
        collection, ok = QInputDialog.getItem(self, "Select Collection", 
                                            "Choose a collection to copy to:", 
                                            collections, 0, False)
        if ok and collection:
            collection_path = os.path.join(self.collectionFolderPath, collection)
            for image_path in selected_images:
                try:
                    shutil.copy(image_path, collection_path)
                    # If you also want to copy associated text files:
                    txt_file_path = os.path.splitext(image_path)[0] + '.txt'
                    if os.path.exists(txt_file_path):
                        shutil.copy(txt_file_path, collection_path)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to copy {os.path.basename(image_path)} to the collection {collection}: {str(e)}")
            
            QMessageBox.information(self, "Success", f"Selected images have been copied to the collection: {collection}")               

    def saveCollections(self):
        collections = [self.collectionsList.item(i).text() for i in range(self.collectionsList.count())]
        with open(self.collectionsFilePath, 'w') as file:
            json.dump(collections, file)

    def addCollection(self):
        collectionName = self.newCollectionLineEdit.text().strip()
        if collectionName:
            collectionPath = os.path.join(self.collectionFolderPath, collectionName)
            try:
                os.makedirs(collectionPath, exist_ok=True)
                self.collectionsList.addItem(collectionName)
                self.newCollectionLineEdit.clear()
                self.saveCollections()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not create the collection: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Collection name cannot be empty.")

    def setupCollectionsPanel(self):
        self.collectionsDockWidget = QDockWidget("Collections", self)
        self.collectionsDockWidget.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.collectionsPanel = QWidget()
        self.collectionsPanelLayout = QVBoxLayout()
        self.collectionsPanel.setLayout(self.collectionsPanelLayout)
        
        self.newCollectionLineEdit = QLineEdit()
        self.collectionsPanelLayout.addWidget(self.newCollectionLineEdit)
        
        self.addCollectionButton = QPushButton("Add Collection", self)
        self.addCollectionButton.clicked.connect(self.addCollection)
        self.collectionsPanelLayout.addWidget(self.addCollectionButton)
        
        self.collectionsList = QListWidget(self)
        self.collectionsList.setSelectionMode(QListWidget.ExtendedSelection)
        self.collectionsList.itemClicked.connect(self.onCollectionClicked)
        self.collectionsList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.collectionsList.customContextMenuRequested.connect(self.onCustomContextMenuRequested)
        self.collectionsPanelLayout.addWidget(self.collectionsList)
    
        self.openCollectionsFolderButton = QPushButton("Open Dataset Collection Folder", self)
        self.openCollectionsFolderButton.clicked.connect(self.open_dataset_collection_folder)
        self.collectionsPanelLayout.addWidget(self.openCollectionsFolderButton)
        
        self.collectionsDockWidget.setWidget(self.collectionsPanel)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.collectionsDockWidget)

    def onCollectionClicked(self, item):
        # Toggle the selection state of the clicked item
        item.setSelected(not item.isSelected())

    def onCustomContextMenuRequested(self, position):
        selected_items = self.collectionsList.selectedItems()
        if not selected_items:
            return  # No item is selected, do not show the context menu

        contextMenu = QMenu(self)
        copyAction = contextMenu.addAction("Copy selected dataset files")
        deleteAction = contextMenu.addAction("Delete selected collections")
        loadCollectionAction = contextMenu.addAction("Load Collection")  # New action

        action = contextMenu.exec_(self.collectionsList.mapToGlobal(position))
        if action == copyAction:
            self.copySelectedDatasets()
        elif action == deleteAction:
            self.deleteSelectedCollections()
        elif action == loadCollectionAction:
            self.loadSelectedCollection()

    def loadCollections(self):
        self.collectionsList.clear()
        if not os.path.exists(self.collectionFolderPath):
            os.makedirs(self.collectionFolderPath)

    def onSelectionChanged(self):
        selected_items = self.collectionsList.selectedItems()
        if selected_items:
            # Handle selection changed if needed
            pass

    def copySelectedDatasets(self):
        selected_collections = [item.text() for item in self.collectionsList.selectedItems()]
        selected_images = [path for path, label in self.imageLabels.items() if label.selected]
        
        # Check if any collection and image are selected
        if not selected_collections or not selected_images:
            QMessageBox.warning(self, "Warning", "Please select at least one collection and one image.")
            return

        for collection in selected_collections:
            collection_path = os.path.join(self.collectionFolderPath, collection)
            for image_path in selected_images:
                try:
                    shutil.copy(image_path, collection_path)
                    # If you also want to copy associated text files:
                    txt_file_path = os.path.splitext(image_path)[0] + '.txt'
                    if os.path.exists(txt_file_path):
                        shutil.copy(txt_file_path, collection_path)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to copy files to the collection {collection}: {str(e)}")

        QMessageBox.information(self, "Success", "Selected images have been copied to the selected collections.")

    def loadSelectedCollection(self):
        selected_collections = [item.text() for item in self.collectionsList.selectedItems()]
        if not selected_collections:
            QMessageBox.warning(self, "Warning", "Please select at least one collection.")
            return

        # Clear existing items in the grid layout
        for i in reversed(range(self.gridLayout.count())):
            widget = self.gridLayout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        self.imageTextEdits.clear()
        self.imageLabels.clear()
        self.selectedImages.clear()

        # Process each selected collection
        for collection_name in selected_collections:
            collection_path = os.path.join(self.collectionFolderPath, collection_name)
            image_files = [f for f in os.listdir(collection_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            file_count = len(image_files)

            # Setup progress dialog
            progress = QProgressDialog("Loading images from collection...", "Abort", 0, file_count, self)
            progress.setWindowTitle("Loading Collection...")
            progress.setCancelButton(None)  # Disable the cancel button
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            for index, filename in enumerate(image_files):
                if progress.wasCanceled():
                    break  # Break the loop if the operation was canceled
                progress.setValue(index)
                QApplication.processEvents()  # Process UI events

                # Proceed with loading the image and setting up UI
                self.loadImageAndSetupUI(collection_path, filename, index, file_count)

            progress.setValue(file_count)  # Complete the progress

    def deleteSelectedCollections(self):
        selected_items = self.collectionsList.selectedItems()
        if selected_items:
            # Implement the logic to delete the selected collections
            for item in selected_items:
                collection_name = item.text()
                collection_path = os.path.join(self.collectionFolderPath, collection_name)
                try:
                    # Remove the collection folder
                    shutil.rmtree(collection_path)
                    # Remove the item from the list
                    self.collectionsList.takeItem(self.collectionsList.row(item))
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Could not delete the collection {collection_name}: {str(e)}")
            self.saveCollections()

    def open_dataset_collection_folder(self):
        if not os.path.exists(self.collectionFolderPath):
            os.makedirs(self.collectionFolderPath)
        subprocess.Popen(f'explorer "{self.collectionFolderPath}"')

    def setupTagsPanel(self):
        self.tagsDockWidget = QDockWidget("Tags", self)
        self.tagsDockWidget.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.tagsPanel = QWidget()
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
        self.tagsScrollArea.setWidgetResizable(True)
        self.tagsScrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tagsScrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    
        self.tagsContainer = QWidget()
        self.tagsContainerLayout = QVBoxLayout()
        self.tagsContainerLayout.setAlignment(Qt.AlignTop)
        self.tagsContainer.setLayout(self.tagsContainerLayout)
    
        self.tagsScrollArea.setWidget(self.tagsContainer)
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
        tagButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        tagButton.setFixedHeight(30)  # Set a fixed height for the button
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
        for image_path, label in self.selectedImages.items():
            if label.selected:
                textEdit = self.imageTextEdits[image_path]
                currentText = textEdit.toPlainText()
                newText = f"{currentText}, {tagText}" if currentText else tagText
                
                # Ensure the image_path exists in editHistory and redoHistory
                if image_path not in self.editHistory:
                    self.editHistory[image_path] = []
                if image_path not in self.redoHistory:
                    self.redoHistory[image_path] = []
                
                # Append current state to editHistory
                self.editHistory[image_path].append(currentText)
                
                # Clear redoHistory as a new edit is being made
                self.redoHistory[image_path].clear()
                
                # Set the new text
                textEdit.setText(newText)
        
        self.saveTags()

    def setupUI(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.mainLayout = QVBoxLayout(self.central_widget)

        self.topBarLayout = QHBoxLayout()
        self.searchBox = QLineEdit(self)
        self.searchBox.setPlaceholderText("Search by tag or caption...")
        self.searchBox.returnPressed.connect(self.filterImages)
        self.topBarLayout.addWidget(self.searchBox)
        self.mainLayout.addLayout(self.topBarLayout)
        self.topBarLayout.setAlignment(Qt.AlignTop)

        self.sizeSlider = QSlider(Qt.Horizontal, self)
        self.sizeSlider.setMinimum(50)
        self.sizeSlider.setMaximum(353)
        self.sizeSlider.setValue(100)
        self.sizeSlider.valueChanged.connect(self.updateThumbnails)
        self.mainLayout.addWidget(self.sizeSlider)
        
        self.selectAllButton = QPushButton("Select/Deselect All", self)
        self.selectAllButton.clicked.connect(self.toggleSelectDeselectAll)
        self.mainLayout.addWidget(self.selectAllButton)
        self.loadDirButton = QPushButton("Load Directory")
        self.loadDirButton.clicked.connect(self.loadDirectory)
        self.mainLayout.addWidget(self.loadDirButton)
        self.gridLayout = QGridLayout()
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.container = QWidget()
        self.container.setLayout(self.gridLayout)
        self.scrollArea.setWidget(self.container)
        self.mainLayout.addWidget(self.scrollArea)

    def selectAllImages(self):
        for label in self.imageLabels.values():
            label.selected = True
            label.setStyleSheet("border: 2px solid blue;")
            
    def toggleSelectDeselectAll(self):
        # Check if any image is selected
        any_selected = any(label.selected for label in self.imageLabels.values())
    
        # If any image is selected, deselect all, else select all
        if any_selected:
            for label in self.imageLabels.values():
                label.selected = False
                label.setStyleSheet("border: 2px solid black;")
        else:
            for label in self.imageLabels.values():
                label.selected = True
                label.setStyleSheet("border: 2px solid blue;")        

    def filterImages(self):
        query = self.searchBox.text().lower().strip()

        # Hide all images and text fields first
        for label in self.imageLabels.values(): 
            label.hide()
        for textEdit in self.imageTextEdits.values():
            textEdit.hide()

        # If query is empty, show all images and text fields
        if not query:
            for label in self.imageLabels.values():
                label.show()
            for textEdit in self.imageTextEdits.values():
                textEdit.show()
            return  # End the function here

        # Show only images and text fields that match the query
        for image_path, textEdit in self.imageTextEdits.items():
            caption = textEdit.toPlainText()
            if query in caption.lower() or query in os.path.basename(image_path).lower():
                label = self.imageLabels[image_path]
                label.show()
                textEdit.show()

    def saveAllEdits(self):
        for image_path, textEdit in self.imageTextEdits.items():
            self.saveTextToFile(image_path, textEdit)
        QMessageBox.information(self, "Save Complete", "All edits have been saved successfully.")

    def saveTextToFile(self, image_path, textEdit):
        txt_file_path = os.path.splitext(image_path)[0] + '.txt'
        try:
            with open(txt_file_path, 'w', encoding='utf-8') as file:
                file.write(textEdit.toPlainText())
            logging.info(f"Saved text for {image_path}")
        except Exception as e:
            logging.error(f"Failed to save text for {image_path}: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save text for {os.path.basename(image_path)}: {str(e)}")
            
    def loadImageAndSetupUI(self, dir_path, filename, index, file_count):
        image_path = os.path.join(dir_path, filename)
        try:
            # Check if the file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"File not found: {image_path}")

            # Check file size
            file_size = os.path.getsize(image_path)
            if file_size == 0:
                raise ValueError(f"File is empty: {image_path}")

            # Attempt to open the image
            try:
                image = Image.open(image_path)
            except Image.UnidentifiedImageError:
                raise ValueError(f"Unidentified image format: {image_path}")

            # Check if the image can be read
            try:
                image.verify()
                image = Image.open(image_path)  # Reopen the image after verify
            except Exception as e:
                raise ValueError(f"Corrupt image file: {image_path}. Error: {str(e)}")

            if max(image.size) > MAX_DISPLAY_SIZE:
                image = ImageOps.exif_transpose(image)
                image.thumbnail((MAX_DISPLAY_SIZE, MAX_DISPLAY_SIZE), Image.Resampling.LANCZOS)

            label = SelectableImageLabel(imagePath=image_path)
            label.clicked.connect(self.toggleImageSelection)
            pixmap = self.pil2pixmap(image)
            label.setPixmap(pixmap.scaled(self.sizeSlider.value(), self.sizeSlider.value(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.imageLabels[image_path] = label
            self.selectedImages[image_path] = label

            # Load existing caption text if available
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_file_path = os.path.join(dir_path, txt_filename)
            description = self.load_description(txt_file_path)

            textEdit = QTextEdit(self)
            textEdit.setText(description)
            self.imageTextEdits[image_path] = textEdit

            row = index // 4
            col = index % 4
            self.gridLayout.addWidget(label, row * 2, col)
            self.gridLayout.addWidget(textEdit, row * 2 + 1, col)
        
            logging.info(f"Successfully loaded image and caption: {image_path}")
        except Exception as e:
            logging.error(f"Failed to process image {image_path}: {str(e)}")
            QMessageBox.warning(self, "Image Loading Error", f"Could not process the image: {os.path.basename(image_path)}\nError: {str(e)}")
            return False
        return True

    def load_description(self, txt_file_path):
        try:
            if os.path.isfile(txt_file_path):
                with open(txt_file_path, 'r', encoding='utf-8') as file:
                    return file.read().strip()
        except Exception as e:
            logging.error(f"Error reading text file {txt_file_path}: {e}")
        return "" 

    def undoLastAction(self):
        # Get the currently selected image(s)
        selected_images = [path for path, label in self.selectedImages.items() if label.selected]
    
        # If there's a selected image, call its undo method
        for image_path in selected_images:
            text_edit_widget = self.imageTextEdits.get(image_path)
            if text_edit_widget:
                text_edit_widget.undo()
                
    def undoLastEdit(self):
        for image_path, textEdit in self.imageTextEdits.items():
            if self.selectedImages[image_path].selected and self.editHistory[image_path]:
                currentState = textEdit.toPlainText()
                lastState = self.editHistory[image_path].pop()
                textEdit.setText(lastState)
                self.redoHistory[image_path].append(currentState)

    def redoLastEdit(self):
        for image_path, textEdit in self.imageTextEdits.items():
            if self.selectedImages[image_path].selected and self.redoHistory[image_path]:
                currentState = textEdit.toPlainText()
                nextState = self.redoHistory[image_path].pop()
                textEdit.setText(nextState)
                self.editHistory[image_path].append(currentState)        

    def clearSelectedCaptions(self):
        selected_images = [path for path, label in self.selectedImages.items() if label.selected]
        
        if not selected_images:
            QMessageBox.warning(self, "Warning", "No images selected. Please select images to clear captions.")
            return
    
        reply = QMessageBox.question(self, 'Clear Selected Captions', 
                                    f"Are you sure you want to clear captions for {len(selected_images)} selected image(s)?",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    
        if reply == QMessageBox.Yes:
            for image_path in selected_images:
                textEdit = self.imageTextEdits[image_path]
                current_text = textEdit.toPlainText()
                
                # Update edit history
                if image_path not in self.editHistory:
                    self.editHistory[image_path] = [current_text]
                else:
                    self.editHistory[image_path].append(current_text)
                
                # Clear the text
                textEdit.clear()
            
            QMessageBox.information(self, "Captions Cleared", f"Captions for {len(selected_images)} image(s) have been cleared.")                

    def loadDirectory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            logging.info(f"Directory loaded: {dir_path}")
        
            self.clearLayout(self.gridLayout)
            self.thumbnail_cache.clear()
            self.imageLabels.clear()
            self.imageTextEdits.clear()
            self.selectedImages.clear()
            self.editHistory.clear()  # Clear edit history
            self.redoHistory.clear()  # Clear redo history
            gc.collect()  # Force garbage collection
    
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
                # Use os.path.normpath to ensure consistent path format
                image_path = os.path.normpath(image_path)
                self.createImagePlaceholder(image_path, index)
                self.queueThumbnailLoad(image_path)
                
                # Initialize edit and redo history for this image
                self.editHistory[image_path] = []
                self.redoHistory[image_path] = []
    
            progress.setValue(len(image_files))
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

    def updateThumbnails(self):
        try:
            size = self.sizeSlider.value()
            for image_path, label in self.imageLabels.items():
                if image_path in self.thumbnail_cache:
                    pixmap = self.thumbnail_cache[image_path]
                else:
                    try:
                        image = load_image(image_path)
                        if image is not None:
                            pixmap = pil2pixmap(image)
                            self.thumbnail_cache[image_path] = pixmap
                        else:
                            continue  # Skip this image if it couldn't be loaded
                    except Exception as e:
                        logging.error(f"Error loading image {image_path}: {str(e)}")
                        continue  # Skip this image if there was an error
                
                scaled_pixmap = pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label.setPixmap(scaled_pixmap)
        except Exception as e:
            logging.error(f"Error in updateThumbnails: {str(e)}", exc_info=True)
            QMessageBox.warning(self, "Error", f"Failed to update thumbnails: {str(e)}")

    def toggleImageSelection(self, image_path):
        if image_path in self.selectedImages:
            label = self.selectedImages[image_path]
            label.selected = not label.selected
            label.setStyleSheet("border: 2px solid blue;" if label.selected else "border: 2px solid black;")
            logging.info(f"Image selection toggled: {image_path}, Selected: {label.selected}")
        else:
            logging.error(f"Image path not found in selectedImages: {image_path}")
            
    def closeEvent(self, event):
        self.thumbnail_cache.clear()
        gc.collect()
        super().closeEvent(event)            

