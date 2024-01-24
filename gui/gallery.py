import shutil
import json
import logging
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QFileDialog, QLabel, QTextEdit, QGridLayout, QScrollArea, QSlider, 
    QMessageBox, QLineEdit, QDockWidget, QProgressDialog, QMenu, QListWidget, QListWidgetItem
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal
import os
from PIL import Image, ImageOps
import subprocess

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
        
        self.mainLayout.addLayout(self.editButtonsLayout)
        
        self.editHistory = {image_path: [] for image_path in self.imageTextEdits.keys()}

        self.loadCollections()
        
        if os.path.exists(self.collectionsFilePath):
            with open(self.collectionsFilePath, 'r') as file:
                collections = json.load(file)
                for collection in collections:
                    self.collectionsList.addItem(collection)
        else:
            with open(self.collectionsFilePath, 'w') as file:
                json.dump([], file)

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
        self.collectionsPanel = QWidget()
        self.collectionsPanelLayout = QVBoxLayout()
        self.collectionsPanel.setLayout(self.collectionsPanelLayout)
        
        self.newCollectionLineEdit = QLineEdit()
        self.collectionsPanelLayout.addWidget(self.newCollectionLineEdit)
        
        self.addCollectionButton = QPushButton("Add Collection", self)
        self.addCollectionButton.clicked.connect(self.addCollection)
        self.collectionsPanelLayout.addWidget(self.addCollectionButton)
        
        self.collectionsList = QListWidget(self)
        self.collectionsList.setSelectionMode(QListWidget.ExtendedSelection)  # Enable multiple selection
        self.collectionsList.itemClicked.connect(self.onCollectionClicked)  # Connect the item clicked signal
        self.collectionsList.setContextMenuPolicy(Qt.CustomContextMenu)  # Enable custom context menu
        self.collectionsList.customContextMenuRequested.connect(self.onCustomContextMenuRequested)  # Connect the context menu signal
        self.collectionsPanelLayout.addWidget(self.collectionsList)

        self.openCollectionsFolderButton = QPushButton("Open Dataset Collection Folder", self)
        self.openCollectionsFolderButton.clicked.connect(self.open_dataset_collection_folder)
        self.collectionsPanelLayout.addWidget(self.openCollectionsFolderButton)
        
        self.collectionsDockWidget.setWidget(self.collectionsPanel)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.collectionsDockWidget)
        self.collectionsList.itemClicked.connect(self.onCollectionClicked)

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

                # Update the edit history for undo functionality
                if image_path not in self.editHistory:
                    self.editHistory[image_path] = [currentText]  # Initialize with a list containing the current text
                else:
                    self.editHistory[image_path].append(currentText)  # Append the current text to the history list

                # Update text file
                txt_file_path = os.path.splitext(image_path)[0] + '.txt'
                with open(txt_file_path, 'w') as file:
                    file.write(newText)

        # Optionally, save tags after modification, if needed
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
        self.sizeSlider.setMaximum(256)
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

    def saveTextToFile(self, image_path, textEdit):
        txt_file_path = os.path.splitext(image_path)[0] + '.txt'
        with open(txt_file_path, 'w') as file:
            file.write(textEdit.toPlainText())
    def loadImageAndSetupUI(self, dir_path, filename, index, file_count):
        image_path = os.path.join(dir_path, filename)
        try:
            image = Image.open(image_path)
            if max(image.size) > MAX_DISPLAY_SIZE:
                image = ImageOps.exif_transpose(image)
                image.thumbnail((MAX_DISPLAY_SIZE, MAX_DISPLAY_SIZE), Image.Resampling.LANCZOS)

            label = SelectableImageLabel(self)
            label.clicked.connect(lambda path=image_path: self.toggleImageSelection(path))
            pixmap = pil2pixmap(image)
            label.setPixmap(pixmap.scaled(self.sizeSlider.value(), self.sizeSlider.value(), Qt.KeepAspectRatio))
            self.imageLabels[image_path] = label
            self.selectedImages[image_path] = label

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
                # Pop the last state from the stack and set it as the current text
                lastState = self.editHistory[image_path].pop()
                textEdit.setText(lastState)                

    def loadDirectory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            logging.info(f"Directory loaded: {dir_path}")
            
            # Get the count of image files for progress indication
            image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            file_count = len(image_files)
            
            # Setup progress dialog
            progress = QProgressDialog("Loading images...", "Abort", 0, file_count, self)
            progress.setWindowTitle("Loading...")
            progress.setCancelButton(None)  # Disable the cancel button
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            for index, filename in enumerate(image_files):
                if progress.wasCanceled():
                    break  # Break the loop if the operation was canceled
                progress.setValue(index)
                QApplication.processEvents()  # Process UI events
                
                # Proceed with loading the image and setting up UI
                self.loadImageAndSetupUI(dir_path, filename, index, file_count)
                
            progress.setValue(file_count)  # Complete the progress
            
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

