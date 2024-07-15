import shutil
import json
import logging
import sys
from functools import lru_cache
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QFileDialog, QLabel, QTextEdit, QGridLayout, QScrollArea, QSlider, 
    QMessageBox, QLineEdit, QDockWidget, QSizePolicy, QProgressDialog, QMenu, 
    QListWidget, QListWidgetItem, QShortcut, QInputDialog, QTreeWidget, QTreeWidgetItem, QCheckBox
)
from PyQt5.QtGui import QPixmap, QImage, QKeySequence
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtCore import pyqtSlot
import os
from PIL import Image, ImageOps
import subprocess
import gc
from PyQt5.QtCore import QRunnable, QThreadPool, pyqtSignal, QObject
import io

def alphanumeric_sort(text):
    return ''.join([format(int(c), '03d') if c.isdigit() else c for c in text])  

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
        self.setGeometry(100, 100, 1200, 800)
        self.imageTextEdits = {}
        self.imageLabels = {}
        self.selectedImages = {}
    
        self.collectionFolderPath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "Dataset Collections"))
        self.collectionsFilePath = os.path.join(self.collectionFolderPath, "collections.json")
    
        self.editButtonsLayout = QHBoxLayout()
    
        # Add a Save Edits button
        self.saveEditsButton = QPushButton("Save Edits", self)
        self.saveEditsButton.setFixedSize(100, 30)
        self.saveEditsButton.clicked.connect(self.saveAllEdits)
        self.editButtonsLayout.addWidget(self.saveEditsButton)
        
        # Undo Button
        self.undoButton = QPushButton("Undo", self)
        self.undoButton.setFixedSize(100, 30)
        self.undoButton.clicked.connect(self.undoLastAction)
        self.undoButton.clicked.connect(self.undoLastEdit)
        self.editButtonsLayout.addWidget(self.undoButton)
        
        # Redo Button
        self.redoButton = QPushButton("Redo", self)
        self.redoButton.setFixedSize(100, 30)
        self.redoButton.clicked.connect(self.redoLastEdit)
        self.editButtonsLayout.addWidget(self.redoButton)
        
        # Add the new Clear Captions button
        self.clearCaptionsButton = QPushButton("Clear Selected Captions", self)
        self.clearCaptionsButton.setFixedSize(120, 40)
        self.clearCaptionsButton.clicked.connect(self.clearSelectedCaptions)
        self.editButtonsLayout.addWidget(self.clearCaptionsButton)
    
        self.setupUI()
        
        # Add dock widgets
        self.addDockWidget(Qt.LeftDockWidgetArea, self.collectionsDockWidget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.tagsDockWidget)
        self.setCorner(Qt.TopLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.TopRightCorner, Qt.RightDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)
    
        self.loadCollections()
        self.loadTags()
        
        # Initialize theme
        self.updateTheme()
    
        self.thumbnail_cache = {}
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(4)
    
        self.editHistory = {image_path: [] for image_path in self.imageTextEdits.keys()}
        self.redoHistory = {image_path: [] for image_path in self.imageTextEdits.keys()}
    
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
        
        self.organizeMode = False
        
    def updateTheme(self):
        app = QApplication.instance()
        if app:
            dark_mode = app.property("darkMode")
            for i in range(self.tagsTreeWidget.topLevelItemCount()):
                self.updateItemStyle(self.tagsTreeWidget.topLevelItem(i), dark_mode)    

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
        self.collectionsDockWidget.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable)
        self.collectionsDockWidget.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.collectionsPanel = QWidget()
        self.collectionsPanelLayout = QVBoxLayout(self.collectionsPanel)
        
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
        self.collectionsDockWidget.setMinimumWidth(10)  # Smaller minimum width
        self.collectionsDockWidget.setMaximumWidth(300)  # Maximum width

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
        self.tagsDockWidget.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable)
        self.tagsDockWidget.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.tagsPanel = QWidget()
        self.tagsPanelLayout = QVBoxLayout(self.tagsPanel)
        
        # New tag text field
        self.newTagLineEdit = QLineEdit()
        self.tagsPanelLayout.addWidget(self.newTagLineEdit)
        
        # Button to add new tag
        self.addTagButton = QPushButton("Add Tag", self)
        self.addTagButton.clicked.connect(self.addTag)
        self.tagsPanelLayout.addWidget(self.addTagButton)
    
        # Organization mode checkbox
        self.organizeModeCheckbox = QCheckBox("Organization Mode")
        self.organizeModeCheckbox.stateChanged.connect(self.toggleOrganizeMode)
        self.tagsPanelLayout.addWidget(self.organizeModeCheckbox)
        
        # Add search box for tags
        self.tagSearchBox = QLineEdit()
        self.tagSearchBox.setPlaceholderText("Search tags...")
        self.tagSearchBox.textChanged.connect(self.filterTags)
        self.tagsPanelLayout.addWidget(self.tagSearchBox)
    
        # Tree widget for tags and categories
        self.tagsTreeWidget = QTreeWidget()
    
        # Tree widget for tags and categories
        self.tagsTreeWidget = QTreeWidget()
        self.tagsTreeWidget.setHeaderLabels(["Tags"])
        self.tagsTreeWidget.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.tagsTreeWidget.itemSelectionChanged.connect(self.onTagSelectionChanged)
        self.tagsTreeWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.updateAllItemsContextMenuPolicy()
        self.tagsTreeWidget.customContextMenuRequested.connect(self.showTagContextMenu)
        self.tagsTreeWidget.setIndentation(20)
        self.tagsTreeWidget.setStyleSheet("""
            QTreeWidget::item { padding: 5px; }
            QTreeWidget::item:has-children { font-weight: bold; }
        """)
        
        self.tagsPanelLayout.addWidget(self.tagsTreeWidget)
        
        self.tagsDockWidget.setWidget(self.tagsPanel)
        self.tagsDockWidget.setMinimumWidth(10)  # Smaller minimum width
        self.tagsDockWidget.setMaximumWidth(300)  # Maximum width
        
    def toggleOrganizeMode(self, state):
        self.organizeMode = state == Qt.Checked
        self.toggleAllItemsFunctionality(not self.organizeMode)
        for i in range(self.tagsTreeWidget.topLevelItemCount()):
            self.updateItemContextMenuPolicy(self.tagsTreeWidget.topLevelItem(i))
        
        # Set the context menu policy for the tree widget itself
        if self.organizeMode:
            self.tagsTreeWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        else:
            self.tagsTreeWidget.setContextMenuPolicy(Qt.DefaultContextMenu)
            
        # Enable category expansion in both modes
        self.tagsTreeWidget.setItemsExpandable(True)
        self.tagsTreeWidget.setExpandsOnDoubleClick(True)

    def updateAllItemsContextMenuPolicy(self):
        for i in range(self.tagsTreeWidget.topLevelItemCount()):
            self.updateItemContextMenuPolicy(self.tagsTreeWidget.topLevelItem(i))
    
    def updateItemContextMenuPolicy(self, item):
        button = self.tagsTreeWidget.itemWidget(item, 0)
        if button:
            button.setContextMenuPolicy(Qt.CustomContextMenu)
        for i in range(item.childCount()):
            self.updateItemContextMenuPolicy(item.child(i))
    
    def toggleAllItemsFunctionality(self, enable):
        for i in range(self.tagsTreeWidget.topLevelItemCount()):
            item = self.tagsTreeWidget.topLevelItem(i)
            self.toggleItemFunctionality(item, enable)
    
    def toggleItemFunctionality(self, item, enable):
        button = self.tagsTreeWidget.itemWidget(item, 0)
        if button:
            button.setEnabled(enable)
        else:
            # It's a category, set it expandable/collapsible based on organize mode
            item.setFlags(item.flags() | Qt.ItemIsEnabled if enable else item.flags() & ~Qt.ItemIsEnabled)
        for i in range(item.childCount()):
            self.toggleItemFunctionality(item.child(i), enable)
    
    def onTagSelectionChanged(self):
        if not self.organizeMode:
            selected_tags = []
            for item in self.tagsTreeWidget.selectedItems():
                if item.parent() is None and self.tagsTreeWidget.itemWidget(item, 0):
                    selected_tags.append(self.tagsTreeWidget.itemWidget(item, 0).text())
            self.applyTagFilter(selected_tags)
    
    def applyTagFilter(self, selected_tags):
        for image_path, label in self.imageLabels.items():
            textEdit = self.imageTextEdits[image_path]
            image_tags = textEdit.toPlainText().split(', ')
            if all(tag in image_tags for tag in selected_tags):
                label.show()
                textEdit.show()
            else:
                label.hide()
                textEdit.hide()
                
    def filterTags(self, filter_text):
        for i in range(self.tagsTreeWidget.topLevelItemCount()):
            item = self.tagsTreeWidget.topLevelItem(i)
            if item.childCount() == 0:  # It's an uncategorized tag
                button = self.tagsTreeWidget.itemWidget(item, 0)
                if button:
                    item.setHidden(filter_text.lower() not in button.text().lower())
            else:  # It's a category
                visible_children = self.filterTagsInCategory(item, filter_text)
                item.setHidden(visible_children == 0)
    
    def filterTagsInCategory(self, category_item, filter_text):
        visible_children = 0
        for i in range(category_item.childCount()):
            child = category_item.child(i)
            button = self.tagsTreeWidget.itemWidget(child, 0)
            if button:
                if filter_text.lower() in button.text().lower():
                    child.setHidden(False)
                    visible_children += 1
                else:
                    child.setHidden(True)
        return visible_children            
    
    def showTagContextMenu(self, position):
        item = self.tagsTreeWidget.itemAt(position)
        if not item:
            if self.organizeMode:
                self.showEmptySpaceContextMenu(position)
            return
    
        # Check if the item is a tag button or a category
        widget = self.tagsTreeWidget.itemWidget(item, 0)
        if isinstance(widget, TagButton):
            if self.organizeMode:
                self.showTagButtonContextMenu(position, widget)
        elif self.organizeMode:
            self.showCategoryContextMenu(position, item)
        
    def showTagButtonContextMenu(self, position, button):
        if not self.organizeMode:
            return
    
        menu = QMenu()
        move_to_category_action = menu.addAction("Move to Category")
        rename_action = menu.addAction("Rename Tag")
        delete_action = menu.addAction("Delete Tag")
    
        # Use the tagsTreeWidget's viewport to get the global position
        global_pos = self.tagsTreeWidget.viewport().mapToGlobal(position)
        action = menu.exec_(global_pos)
    
        if action == move_to_category_action:
            self.moveTagsToCategory([self.tagsTreeWidget.itemAt(position)])
        elif action == rename_action:
            self.renameTag(button)
        elif action == delete_action:
            self.deleteTag(button)
    
        # Prevent image filtering
        self.onTagSelectionChanged()
    
    def showCategoryContextMenu(self, position, item):
        if not self.organizeMode:
            return
    
        menu = QMenu()
        rename_action = menu.addAction("Rename Category")
        delete_action = menu.addAction("Delete Category")
    
        action = menu.exec_(self.tagsTreeWidget.mapToGlobal(position))
    
        if action == rename_action:
            self.renameCategory(item)
        elif action == delete_action:
            self.deleteCategory(item)
    
    def showEmptySpaceContextMenu(self, position):
        menu = QMenu()
        create_category_action = menu.addAction("Create New Category")
    
        action = menu.exec_(self.tagsTreeWidget.mapToGlobal(position))
    
        if action == create_category_action:
            self.createNewCategory()
            
    def createCategory(self, category_name):
        category_item = QTreeWidgetItem(self.tagsTreeWidget)
        category_item.setText(0, category_name)
        category_item.setFlags(category_item.flags() | Qt.ItemIsUserCheckable)
        category_item.setCheckState(0, Qt.Unchecked)
        return category_item        
    
    def createNewCategory(self):
        category_name, ok = QInputDialog.getText(self, "New Category", "Enter category name:")
        if ok and category_name:
            # Find the correct position to insert the new category
            insert_index = 0
            for i in range(self.tagsTreeWidget.topLevelItemCount()):
                item = self.tagsTreeWidget.topLevelItem(i)
                if self.tagsTreeWidget.itemWidget(item, 0) is None:  # It's a category
                    if alphanumeric_sort(item.text(0)) > alphanumeric_sort(category_name):
                        break
                insert_index += 1
            
            # Insert the new category at the correct position
            new_category = QTreeWidgetItem()
            new_category.setText(0, category_name)
            new_category.setFlags(new_category.flags() | Qt.ItemIsUserCheckable)
            new_category.setCheckState(0, Qt.Unchecked)
            self.tagsTreeWidget.insertTopLevelItem(insert_index, new_category)
            
            self.saveTags()
    
    def renameTag(self, button):
        old_name = button.text()
        new_name, ok = QInputDialog.getText(self, "Rename Tag", "Enter new name:", text=old_name)
        if ok and new_name:
            button.setText(new_name)
            self.saveTags()
    
    def renameCategory(self, item):
        old_name = item.text(0)
        new_name, ok = QInputDialog.getText(self, "Rename Category", "Enter new name:", text=old_name)
        if ok and new_name:
            item.setText(0, new_name)
            self.saveTags()
    
    def deleteTag(self, button):
        reply = QMessageBox.question(self, "Delete Tag", f"Are you sure you want to delete the tag '{button.text()}'?",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            item = self.tagsTreeWidget.itemAt(button.pos())
            if item.parent():
                item.parent().removeChild(item)
            else:
                index = self.tagsTreeWidget.indexOfTopLevelItem(item)
                self.tagsTreeWidget.takeTopLevelItem(index)
            self.saveTags()
    
    def deleteCategory(self, item):
        reply = QMessageBox.question(self, "Delete Category", f"Are you sure you want to delete the category '{item.text(0)}' and all its tags?",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            index = self.tagsTreeWidget.indexOfTopLevelItem(item)
            self.tagsTreeWidget.takeTopLevelItem(index)
            self.saveTags()
    
    def moveTagsToCategory(self, items):
        if not items:
            return
    
        categories = [self.tagsTreeWidget.topLevelItem(i).text(0) 
                    for i in range(self.tagsTreeWidget.topLevelItemCount())
                    if self.tagsTreeWidget.itemWidget(self.tagsTreeWidget.topLevelItem(i), 0) is None]
        
        category, ok = QInputDialog.getItem(self, "Select Category", "Move to category:", categories, 0, False)
        
        if ok and category:
            category_item = self.findCategoryItem(category)
            if category_item:
                for item in items:
                    if isinstance(item, QTreeWidgetItem):
                        button = self.tagsTreeWidget.itemWidget(item, 0)
                        if button:
                            tag_text = button.text()
                            if item.parent():
                                item.parent().removeChild(item)
                            else:
                                index = self.tagsTreeWidget.indexOfTopLevelItem(item)
                                self.tagsTreeWidget.takeTopLevelItem(index)
                            new_tag_item = self.createTagButton(tag_text, category_item)
                    elif isinstance(item, TagButton):
                        tag_text = item.text()
                        parent_item = self.tagsTreeWidget.itemAt(item.pos()).parent()
                        if parent_item:
                            parent_item.removeChild(self.tagsTreeWidget.itemAt(item.pos()))
                        else:
                            index = self.tagsTreeWidget.indexOfTopLevelItem(self.tagsTreeWidget.itemAt(item.pos()))
                            self.tagsTreeWidget.takeTopLevelItem(index)
                        new_tag_item = self.createTagButton(tag_text, category_item)
            self.saveTags()
    
    def findCategoryItem(self, category_name):
        for i in range(self.tagsTreeWidget.topLevelItemCount()):
            item = self.tagsTreeWidget.topLevelItem(i)
            if item.text(0) == category_name:
                return item
        return None    
        
    def loadTags(self):
        try:
            categories = {}
            uncategorized_tags = []
            
            with open('tags.txt', 'r') as file:
                for line in file:
                    line = line.strip()
                    if line.startswith("category:"):
                        category_name = line[9:]
                        categories[category_name] = []
                    elif line.startswith("tag_in_category:"):
                        _, category_name, tag_text = line.split(":", 2)
                        if category_name in categories:
                            categories[category_name].append(tag_text)
                    elif line.startswith("tag:"):
                        tag_text = line[4:]
                        uncategorized_tags.append(tag_text)
                    else:
                        # Handle old format (just tag text)
                        uncategorized_tags.append(line)
            
            # Sort categories
            sorted_categories = sorted(categories.keys(), key=alphanumeric_sort)
            
            # Add sorted uncategorized tags
            for tag in sorted(uncategorized_tags, key=alphanumeric_sort):
                self.createTagButton(tag)
            
            # Add sorted categories with their sorted tags
            for category in sorted_categories:
                category_item = self.createCategory(category)
                for tag in sorted(categories[category], key=alphanumeric_sort):
                    self.createTagButton(tag, category_item)
            
            self.tagsTreeWidget.expandAll()  # Expand all categories
        except FileNotFoundError:
            pass  # File doesn't exist yet
        
        # Apply the current theme
        app = QApplication.instance()
        if app:
            dark_mode = app.property("darkMode")
            self.updateTagButtonStyles(dark_mode)
            
    def applyTheme(self, theme):
        app = QApplication.instance()
        dark_mode = (theme == 'Dark')
        app.setProperty("darkMode", dark_mode)
        self.tagsTreeWidget.clear()  # Clear existing tags
        self.loadTags()  # Reload tags with new theme
        
    def updateTagButtonStyles(self, dark_mode):
        def update_item(item):
            button = self.tagsTreeWidget.itemWidget(item, 0)
            if isinstance(button, TagButton):
                button.updateStyle(dark_mode)
            for i in range(item.childCount()):
                update_item(item.child(i))

        for i in range(self.tagsTreeWidget.topLevelItemCount()):
            update_item(self.tagsTreeWidget.topLevelItem(i))
    
    def updateItemStyle(self, item, dark_mode):
        button = self.tagsTreeWidget.itemWidget(item, 0)
        if isinstance(button, TagButton):
            button.updateStyle(dark_mode)
        for i in range(item.childCount()):
            self.updateItemStyle(item.child(i), dark_mode)                

    def saveTags(self):
        with open('tags.txt', 'w') as file:
            # Save categories and their tags
            for i in range(self.tagsTreeWidget.topLevelItemCount()):
                item = self.tagsTreeWidget.topLevelItem(i)
                if self.tagsTreeWidget.itemWidget(item, 0) is None:  # It's a category
                    file.write(f"category:{item.text(0)}\n")
                    for j in range(item.childCount()):
                        tag_item = item.child(j)
                        button = self.tagsTreeWidget.itemWidget(tag_item, 0)
                        if button:
                            file.write(f"tag_in_category:{item.text(0)}:{button.text()}\n")
                else:  # It's a top-level tag
                    button = self.tagsTreeWidget.itemWidget(item, 0)
                    if button:
                        file.write(f"tag:{button.text()}\n")
    
    def saveTagsInCategory(self, category_item, file):
        for i in range(category_item.childCount()):
            tag_item = category_item.child(i)
            button = self.tagsTreeWidget.itemWidget(tag_item, 0)
            if button:
                file.write(f"tag_in_category:{category_item.text(0)}:{button.text()}\n")
                
    def createTagButton(self, tag_text, parent=None):
        if parent and parent != self.tagsTreeWidget:  # It's a categorized tag
            # Find the correct position to insert the new tag within the category
            insert_index = 0
            for i in range(parent.childCount()):
                child = parent.child(i)
                button = self.tagsTreeWidget.itemWidget(child, 0)
                if alphanumeric_sort(button.text()) > alphanumeric_sort(tag_text):
                    break
                insert_index += 1
            
            tag_item = QTreeWidgetItem()
            parent.insertChild(insert_index, tag_item)
        else:  # It's an uncategorized tag
            tag_item = QTreeWidgetItem(self.tagsTreeWidget)
        
        tag_button = TagButton(tag_text, self)
        app = QApplication.instance()
        dark_mode = app.property("darkMode") if app else False
        tag_button.updateStyle(dark_mode)
        tag_button.clicked.connect(lambda: self.tagButtonClicked(tag_text))
        tag_button.setContextMenuPolicy(Qt.CustomContextMenu)
        tag_button.customContextMenuRequested.connect(lambda pos: self.showTagButtonContextMenu(tag_button.mapToGlobal(pos), tag_button))
        self.tagsTreeWidget.setItemWidget(tag_item, 0, tag_button)
        tag_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return tag_item

    def addTag(self):
        tagText = self.newTagLineEdit.text().strip()
        if tagText:
            # Find the correct position to insert the new tag
            insert_index = 0
            for i in range(self.tagsTreeWidget.topLevelItemCount()):
                item = self.tagsTreeWidget.topLevelItem(i)
                if self.tagsTreeWidget.itemWidget(item, 0) is None:  # It's a category
                    continue
                button = self.tagsTreeWidget.itemWidget(item, 0)
                if alphanumeric_sort(button.text()) > alphanumeric_sort(tagText):
                    break
                insert_index += 1
            
            # Insert the new tag at the correct position
            new_item = QTreeWidgetItem()
            self.tagsTreeWidget.insertTopLevelItem(insert_index, new_item)
            self.createTagButton(tagText, new_item)
            
            self.newTagLineEdit.clear()
            self.saveTags()  # Save tags to file

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

    def saveAllEdits(self):
        for image_path, textEdit in self.imageTextEdits.items():
            self.saveTextToFile(image_path, textEdit)
        QMessageBox.information(self, "Save Complete", "All edits have been saved successfully.")            

    def setupUI(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.mainLayout = QVBoxLayout(self.central_widget)
    
        # Top bar with search
        self.topBarLayout = QHBoxLayout()
        self.searchBox = QLineEdit(self)
        self.searchBox.setPlaceholderText("Search by tag or caption...")
        self.searchBox.returnPressed.connect(self.filterImages)
        self.topBarLayout.addWidget(self.searchBox)
        self.mainLayout.addLayout(self.topBarLayout)
    
        # Slider for thumbnail size
        self.sizeSlider = QSlider(Qt.Horizontal, self)
        self.sizeSlider.setMinimum(50)
        self.sizeSlider.setMaximum(353)
        self.sizeSlider.setValue(100)
        self.sizeSlider.valueChanged.connect(self.updateThumbnails)
        self.mainLayout.addWidget(self.sizeSlider)
    
        # Viewer window (center)
        self.viewerLayout = QVBoxLayout()
        self.selectAllButton = QPushButton("Select/Deselect All", self)
        self.selectAllButton.clicked.connect(self.toggleSelectDeselectAll)
        self.viewerLayout.addWidget(self.selectAllButton)
        
        self.loadDirButton = QPushButton("Load Directory")
        self.loadDirButton.clicked.connect(self.loadDirectory)
        self.viewerLayout.addWidget(self.loadDirButton)
        
        self.gridLayout = QGridLayout()
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.container = QWidget()
        self.container.setLayout(self.gridLayout)
        self.scrollArea.setWidget(self.container)
        self.viewerLayout.addWidget(self.scrollArea)
        
        self.mainLayout.addLayout(self.viewerLayout)
    
        # Add edit buttons layout
        self.mainLayout.addLayout(self.editButtonsLayout)
    
        # Setup dock widgets
        self.setupCollectionsPanel()
        self.setupTagsPanel()

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

class TagButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.updateStyle()

    def updateStyle(self, dark_mode=False):
        if dark_mode:
            self.setStyleSheet("""
                QPushButton { 
                    text-align: left; 
                    padding: 5px; 
                    margin: 2px 0; 
                    min-height: 30px;
                    background-color: #455364;
                    color: white;
                    border: 1px solid #3a4654;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #3a4654;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton { 
                    text-align: left; 
                    padding: 5px; 
                    margin: 2px 0; 
                    min-height: 30px;
                    background-color: #f2f2f2;
                    color: black;
                    border: 1px solid #d9d9d9;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #e6e6e6;
                }
            """)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.customContextMenuRequested.emit(event.pos())
        else:
            super().mousePressEvent(event)      

