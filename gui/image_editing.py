import cv2
import logging
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QScrollArea, QSlider, QMessageBox, QFileDialog, QDialog, QLineEdit, QComboBox, QGridLayout, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import os
import traceback

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

def cv2pixmap(cv_img):
    """Convert from an OpenCV image to QPixmap"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    height, width, channel = rgb_image.shape
    bytes_per_line = channel * width
    qimage = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage)

class ResizeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        self.layout = QVBoxLayout(self)

        # Width and Height Input
        self.layout.addWidget(QLabel("Width:"))
        self.widthInput = QLineEdit(self)
        self.layout.addWidget(self.widthInput)

        self.layout.addWidget(QLabel("Height:"))
        self.heightInput = QLineEdit(self)
        self.layout.addWidget(self.heightInput)

        # Aspect Ratio Selection
        self.layout.addWidget(QLabel("Or select aspect ratio:"))
        self.aspectRatioComboBox = QComboBox(self)
        self.aspectRatioComboBox.addItems(["16:9", "4:3", "1:1", "3:2", "2:3", "4:5", "5:4", "7:5", "16:10", "Custom"])
        self.layout.addWidget(self.aspectRatioComboBox)

        # Resize Button
        self.resizeButton = QPushButton("Resize", self)
        self.resizeButton.clicked.connect(self.onResizeClicked)
        self.layout.addWidget(self.resizeButton)

    def onResizeClicked(self):
        self.width = self.widthInput.text()
        self.height = self.heightInput.text()
        self.aspect_ratio = self.aspectRatioComboBox.currentText()
        self.accept()  # Close the dialog

    def getValues(self):
        return self.width, self.height, self.aspect_ratio

class SelectableImageLabel(QLabel):
    def __init__(self, parent=None):
        super(SelectableImageLabel, self).__init__(parent)
        self.selected = False
        self.setStyleSheet("border: 2px solid black;")

    def mousePressEvent(self, event):
        self.selected = not self.selected
        self.setStyleSheet("border: 2px solid blue;" if self.selected else "border: 2px solid black;")
        super(SelectableImageLabel, self).mousePressEvent(event)

class ImageEditingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.imageLabels = {}
        self.selectedImages = {}
        self.allSelected = False
        self.thumbnailSize = 150  # Default thumbnail size
        self.setupUI()
        self.workingCopies = {}  # Dictionary to store working copies of images

    def setupUI(self):
        self.layout = QVBoxLayout(self)

        # Load Directory Button
        self.loadDirButton = QPushButton("Load Directory")
        self.loadDirButton.clicked.connect(self.loadDirectory)
        self.layout.addWidget(self.loadDirButton)
        
        self.toggleSelectButton = QPushButton("Select/Deselect All")
        self.toggleSelectButton.clicked.connect(self.toggleSelectAllImages)
        self.layout.addWidget(self.toggleSelectButton)
        
        # Resize Button
        self.resizeButton = QPushButton("Resize Image(s)")
        self.resizeButton.clicked.connect(self.openResizeDialog)
        self.layout.addWidget(self.resizeButton)
        
        # Rotate Button
        self.rotateButton = QPushButton("Rotate Selected Images")
        self.rotateButton.clicked.connect(self.rotateSelectedImages)
        self.layout.addWidget(self.rotateButton)

        # Slider for resizing thumbnails
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
        self.layout.addLayout(self.sizeSliderLayout)

        # Scroll Area for displaying images
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.container = QWidget()
        self.gridLayout = QGridLayout(self.container)
        self.scrollArea.setWidget(self.container)
        self.layout.addWidget(self.scrollArea)
        
        # Save Edits Button
        self.saveEditsButton = QPushButton("Save Edits")
        self.saveEditsButton.clicked.connect(self.saveEdits)
        self.layout.addWidget(self.saveEditsButton)

        # Add buttons for other functionalities like cropping, resizing, rotation, saving changes, etc.
        # ...
        
    def openResizeDialog(self):
        # This method will be implemented later
        pass    

    def loadDirectory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            # Extract the name of the current directory for naming the subdirectory
            dir_name = os.path.basename(dir_path)

            # Define the path for 'Image Edits' directory and subdirectory
            edits_path = os.path.join(os.getcwd(), "Image Edits", dir_name)

            # Create 'Image Edits' and subdirectory if they don't exist
            os.makedirs(edits_path, exist_ok=True)

            # Store the edits path in a variable for later use
            self.edits_dir_path = edits_path

            self.displayImagesFromDirectory(dir_path)
        else:
            QMessageBox.warning(self, "Warning", "No directory was selected.")

    def displayImagesFromDirectory(self, dir_path):
        self.clearLayout(self.gridLayout)
        self.imageLabels.clear()

        for index, filename in enumerate(sorted(os.listdir(dir_path))):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(dir_path, filename)
                self.loadImageAndSetupUI(image_path, index)        

    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def loadImageAndSetupUI(self, image_path, index):
        image = cv2.imread(image_path)
        if image is not None:
            pixmap = cv2pixmap(image)
            resized_pixmap = pixmap.scaled(self.thumbnailSize, self.thumbnailSize, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            label = self.imageLabels.get(image_path)
            if label is None:
                label = SelectableImageLabel(self)
                self.imageLabels[image_path] = label
                row, col = divmod(index, 4)
                self.gridLayout.addWidget(label, row, col)

            label.setPixmap(resized_pixmap)
            label.setFixedSize(resized_pixmap.size())

    def toggleSelectAllImages(self):
        self.allSelected = not self.allSelected
        for label in self.imageLabels.values():
            label.selected = self.allSelected
            label.setStyleSheet("border: 2px solid blue;" if label.selected else "border: 2px solid black;")
        # Update the button text based on the current state
        self.toggleSelectButton.setText("Deselect All" if self.allSelected else "Select All")

    def updateThumbnails(self):
        size = self.sizeSlider.value()
        for image_path, label in self.imageLabels.items():
            image = cv2.imread(image_path)
            if image is not None:
                # Calculate aspect ratio and new size
                height, width = image.shape[:2]
                aspect_ratio = width / height
                new_width = int(size * aspect_ratio) if width > height else size
                new_height = int(size / aspect_ratio) if height > width else size
    
                # Resize the image
                resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                pixmap = cv2pixmap(resized_image)
    
                # Update label
                label.setPixmap(pixmap)
                label.setFixedSize(pixmap.size())
                label.update()
            
    def rotateSelectedImages(self):
        try:  # Start of the try block
            max_size = self.thumbnailSize
            for image_path, label in self.imageLabels.items():
                if label.selected:
                    # Load the image either from workingCopies or from the file
                    image = self.workingCopies.get(image_path, cv2.imread(image_path))
                    if image is not None:
                        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
                        # Calculate new size while maintaining aspect ratio
                        height, width = rotated_image.shape[:2]
                        aspect_ratio = width / height
                        if width > height:  # Landscape orientation
                            new_width = min(width, max_size)
                            new_height = int(new_width / aspect_ratio)
                        else:  # Portrait orientation
                            new_height = min(height, max_size)
                            new_width = int(new_height * aspect_ratio)
    
                        resized_rotated_image = cv2.resize(rotated_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                        pixmap = cv2pixmap(resized_rotated_image)
    
                        label.setPixmap(pixmap)
                        label.setFixedSize(pixmap.size())
                        label.update()
    
                        # Update the working copy with the newly rotated image
                        self.workingCopies[image_path] = rotated_image
    
        except Exception as e:  # Start of the except block
            logging.error(f"Failed to rotate images: {e}")
            QMessageBox.critical(self, "Error", f"Failed to rotate one or more images. Error: {e}")
        
                    
    def saveEdits(self):
        newWorkingCopies = {}
        for image_path, image in self.workingCopies.items():
            save_path = os.path.join(self.edits_dir_path, os.path.basename(image_path))
            cv2.imwrite(save_path, image)
    
            # Update the label pixmap to the saved image
            if image_path in self.imageLabels:
                label = self.imageLabels[image_path]
                pixmap = cv2pixmap(image)
                resized_pixmap = pixmap.scaled(self.thumbnailSize, self.thumbnailSize, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label.setPixmap(resized_pixmap)
                label.setFixedSize(resized_pixmap.size())
    
            # Update the working copies to reflect the saved state
            newWorkingCopies[save_path] = image
    
        # Replace the old working copies with the new state
        self.workingCopies = newWorkingCopies
        QMessageBox.information(self, "Save Complete", "Edited images have been saved successfully.")                
        
    def openResizeDialog(self):
        dialog = ResizeDialog(self)
        if dialog.exec_():
            width, height, aspect_ratio = dialog.getValues()
            self.resizeImage(width, height, aspect_ratio)

    def resizeImage(self, width_str, height_str, aspect_ratio):
        try:
            width = int(width_str) if width_str.isdigit() else None
            height = int(height_str) if height_str.isdigit() else None
    
            for image_path, label in self.imageLabels.items():
                if label.selected:
                    image = self.workingCopies.get(image_path, cv2.imread(image_path))
                    if image is not None:
                        original_height, original_width = image.shape[:2]
    
                        if aspect_ratio != "Custom":
                            ratio_w, ratio_h = map(int, aspect_ratio.split(':'))
                            if not width and not height:
                                # Calculate new dimensions based on the original size and aspect ratio
                                if original_width / original_height > ratio_w / ratio_h:
                                    width = int(original_height * ratio_w / ratio_h)
                                    height = original_height
                                else:
                                    width = original_width
                                    height = int(original_width * ratio_h / ratio_w)
                            elif width and not height:
                                height = int(width * ratio_h / ratio_w)
                            elif height and not width:
                                width = int(height * ratio_w / ratio_h)
                        elif not width or not height:
                            # Skip if any one of the dimensions is missing for 'Custom'
                            continue
    
                        resized_image = cv2.resize(image, (width, height))
                        self.workingCopies[image_path] = resized_image
                        pixmap = cv2pixmap(resized_image)
                        resized_pixmap = pixmap.scaled(self.thumbnailSize, self.thumbnailSize, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        label.setPixmap(resized_pixmap)
                        label.setFixedSize(resized_pixmap.size())
                        label.update()
        except Exception as e:
            logging.error(f"Error in resizing images: {e}")
            QMessageBox.critical(self, "Error", f"Error occurred while resizing images. Error: {e}")


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    app = QApplication([])
    window = ImageEditingWindow()
    window.show()
    app.exec_()
    
    