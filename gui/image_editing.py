import cv2
import numpy as np
import logging
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QPushButton, QLabel,
    QScrollArea, QSlider, QMessageBox, QFileDialog, QComboBox, QDialog, QProgressDialog, QLineEdit, QShortcut)
from PyQt5.QtGui import QPixmap, QImage, QKeySequence
from PyQt5.QtCore import Qt
import os

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

    def toggleSelectAllImages(self):
        if all(label.selected for label in self.imageLabels.values()):
            for label in self.imageLabels.values():
                label.selected = False
                label.setStyleSheet("border: 2px solid black;")
        else:
            for label in self.imageLabels.values():
                label.selected = True
                label.setStyleSheet("border: 2px solid blue;")

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
        self.aspectRatioComboBox.addItems(["16:9", "9:16", "4:3", "1:1", "3:2", "2:3", "4:5", "5:4", "7:5", "16:10"])
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

class CropDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        self.layout = QVBoxLayout(self)

        # Crop Dimensions Input
        self.layout.addWidget(QLabel("Crop Width:"))
        self.cropWidthInput = QLineEdit(self)
        self.layout.addWidget(self.cropWidthInput)

        self.layout.addWidget(QLabel("Crop Height:"))
        self.cropHeightInput = QLineEdit(self)
        self.layout.addWidget(self.cropHeightInput)

        # Aspect Ratio Selection
        self.layout.addWidget(QLabel("Aspect Ratio:"))
        self.aspectRatioComboBox = QComboBox(self)
        self.aspectRatioComboBox.addItems(["None", "16:9", "9:16", "4:3", "1:1", "3:2", "2:3", "4:5", "5:4", "7:5", "16:10"])
        self.layout.addWidget(self.aspectRatioComboBox)

        # Crop Button
        self.cropButton = QPushButton("Crop", self)
        self.cropButton.clicked.connect(self.onCropClicked)
        self.layout.addWidget(self.cropButton)
        
        # Add focus point selection in setupUI method
        self.layout.addWidget(QLabel("Focus Point:"))
        self.focusPointComboBox = QComboBox(self)
        self.focusPointComboBox.addItems(["Center", "Top-Left", "Top-Center", "Top-Right", "Left-Center", "Right-Center", "Bottom-Left", "Bottom-Center", "Bottom-Right"])
        self.layout.addWidget(self.focusPointComboBox)

    def onCropClicked(self):
        self.width = self.cropWidthInput.text()
        self.height = self.cropHeightInput.text()
        self.aspect_ratio = self.aspectRatioComboBox.currentText()
        self.accept()

    def getValues(self):
        focus_point = self.focusPointComboBox.currentText()  # Retrieve the selected focus point
        return self.cropWidthInput.text(), self.cropHeightInput.text(), self.aspectRatioComboBox.currentText(), focus_point

class SelectableImageLabel(QLabel):
    def __init__(self, imagePath='', parent=None):
        super(SelectableImageLabel, self).__init__(parent)
        self.selected = False
        self.imagePath = imagePath  # Ensure this line is correctly added
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
        self.workingCopies = {}  # Dictionary to store working copies of images
        self.setupUI()
        self.edits_dir_path = ""
        
        self.shortcuts = {
            "Ctrl+S": (self.saveEdits, "Save all edits"),
            "Ctrl+A": (self.toggleSelectAllImages, "Toggle select/deselect all")
        }
        self.setupShortcuts()

    def setupShortcuts(self):
        for key, (func, _) in self.shortcuts.items():
            QShortcut(QKeySequence(key), self).activated.connect(func)

    def updateCustomShortcuts(self, custom_shortcuts):
        for action, new_key in custom_shortcuts.items():
            for key, (func, desc) in self.shortcuts.items():
                if desc == action:
                    QShortcut(QKeySequence(new_key), self).activated.connect(func)
                    break

    def setupUI(self):
        self.layout = QVBoxLayout(self)

        # Load Directory Button
        self.loadDirButton = QPushButton("Load Directory")
        self.loadDirButton.clicked.connect(self.loadDirectory)
        self.layout.addWidget(self.loadDirButton)

        self.toggleSelectButton = QPushButton("Select/Deselect All")
        self.toggleSelectButton.clicked.connect(self.toggleSelectAllImages)
        self.layout.addWidget(self.toggleSelectButton)

        # Crop Button
        self.cropButton = QPushButton("Crop Selected Image(s)")
        self.cropButton.clicked.connect(self.openCropDialog)
        self.layout.addWidget(self.cropButton)

        # Resize Button
        self.resizeButton = QPushButton("Resize Selected Image(s)")
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

    def openCropDialog(self):
        selectedImages = [label for label in self.imageLabels.values() if label.selected]
        if selectedImages:
            dialog = CropDialog(self)
            if dialog.exec_():
                width, height, aspect_ratio, focus_point = dialog.getValues()
                self.processImagesSequentially(selectedImages, width, height, aspect_ratio, focus_point)
                
    def processImagesSequentially(self, images, width, height, aspect_ratio, focus_point):
        progress_dialog = QProgressDialog("Processing images...", "Cancel", 0, len(images), self)
        progress_dialog.setModal(True)
        progress_dialog.show()

        for i, label in enumerate(images):
            if progress_dialog.wasCanceled():
                break
            self.cropImage(label, width, height, aspect_ratio, focus_point)
            progress_dialog.setValue(i + 1)

        progress_dialog.close()            

    def cropImage(self, label, width_str, height_str, aspect_ratio_str, focus_point):
        # Load the original full-size image from the file path
        image_path = label.imagePath
        original_image = cv2.imread(image_path)

        # Handle width and height inputs
        if aspect_ratio_str == "None":
            # If 'None' is selected, use the specified width and height
            width = int(width_str) if width_str.isdigit() else None
            height = int(height_str) if height_str.isdigit() else None
            if width is None or height is None:
                QMessageBox.warning(self, "Warning", "Invalid width or height provided.")
                return
        else:
            # If an aspect ratio is selected, calculate width and height based on the ratio
            aspect_ratio = self.convertAspectRatio(aspect_ratio_str)
            img_height, img_width = original_image.shape[:2]
            if not width_str and not height_str:
                # Default to full image dimensions if width and height are not specified
                if aspect_ratio >= 1:  # Width is greater than height
                    width = img_width
                    height = int(img_width / aspect_ratio)
                else:  # Height is greater than width
                    height = img_height
                    width = int(img_height * aspect_ratio)
            elif width_str and height_str:
                # If both width and height are specified, use them directly
                width = int(width_str) if width_str.isdigit() else None
                height = int(height_str) if height_str.isdigit() else None
            elif width_str:
                # If only width is specified
                width = int(width_str) if width_str.isdigit() else None
                height = int(width / aspect_ratio) if width else None
            elif height_str:
                # If only height is specified
                height = int(height_str) if height_str.isdigit() else None
                width = int(height * aspect_ratio) if height else None

        # Check for valid width and height
        if width is None or height is None:
            QMessageBox.warning(self, "Warning", "Invalid width or height provided.")
            return

        cropped_img = self.calculateCrop(original_image, width, height, focus_point)
        if cropped_img is not None:
            # Update QLabel with the resized cropped image
            resized_cropped_pixmap = cv2pixmap(cropped_img).scaled(self.thumbnailSize, self.thumbnailSize, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(resized_cropped_pixmap)

            # Update the working copy with the full-size cropped image
            self.workingCopies[image_path] = cropped_img
        else:
            QMessageBox.warning(self, "Warning", "Invalid cropping parameters.")

    def qimageToCv2(self, qimage):
        """Convert QImage to cv2 image format"""
        temp_image = qimage.convertToFormat(QImage.Format_RGB888)
        ptr = temp_image.bits()
        ptr.setsize(temp_image.byteCount())
        arr = np.array(ptr).reshape(temp_image.height(), temp_image.width(), 3)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)        
    
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
                
    def calculateCrop(self, image, width_str, height_str, aspect_ratio_str, focus_point):
        try:
            width = int(width_str) if width_str.isdigit() else None
            height = int(height_str) if height_str.isdigit() else None

            aspect_ratio = self.convertAspectRatio(aspect_ratio_str)

            # Ensure width and height don't exceed image dimensions
            width = min(width, image.shape[1])
            height = min(height, image.shape[0])

            # Calculate crop position based on focus point
            x, y = self.calculateFocusPoint(image, width, height, focus_point)

            # Cropping the image
            cropped_image = image[y:y + height, x:x + width]
            return cropped_image

        except Exception as e:
            logging.error(f"Cropping error: {e}")
            return None    

    def calculateFocusPoint(self, image, crop_width, crop_height, focus_point):
        img_height, img_width = image.shape[:2]
    
        # Calculate center points
        center_x, center_y = img_width // 2, img_height // 2
        x, y = center_x - crop_width // 2, center_y - crop_height // 2
    
        # Adjust x and y based on focus point
        if focus_point == "Top-Left":
            x, y = 0, 0
        elif focus_point == "Top-Center":
            x = center_x - crop_width // 2
            y = 0
        elif focus_point == "Top-Right":
            x = img_width - crop_width
            y = 0
        elif focus_point == "Left-Center":
            x = 0
            y = center_y - crop_height // 2
        elif focus_point == "Center":
            x = center_x - crop_width // 2
            y = center_y - crop_height // 2
        elif focus_point == "Right-Center":
            x = img_width - crop_width
            y = center_y - crop_height // 2
        elif focus_point == "Bottom-Left":
            x = 0
            y = img_height - crop_height
        elif focus_point == "Bottom-Center":
            x = center_x - crop_width // 2
            y = img_height - crop_height
        elif focus_point == "Bottom-Right":
            x = img_width - crop_width
            y = img_height - crop_height
    
        # Ensure x and y are within the image boundaries
        x = max(0, min(x, img_width - crop_width))
        y = max(0, min(y, img_height - crop_height))
    
        return x, y

    def convertAspectRatio(self, aspect_ratio_str):
        if aspect_ratio_str == "Custom":
            return 1.0  # Default value, adjust as needed
        width, height = map(int, aspect_ratio_str.split(':'))
        return width / height        

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
                label = SelectableImageLabel(imagePath=image_path)  # Check this line
                self.imageLabels[image_path] = label
                row, col = divmod(index, 4)
                self.gridLayout.addWidget(label, row, col)
    
            label.setPixmap(resized_pixmap)
            label.setFixedSize(resized_pixmap.size())

    def toggleSelectAllImages(self):
        if all(label.selected for label in self.imageLabels.values()):
            for label in self.imageLabels.values():
                label.selected = False
                label.setStyleSheet("border: 2px solid black;")
            self.toggleSelectButton.setText("Select All")
        else:
            for label in self.imageLabels.values():
                label.selected = True
                label.setStyleSheet("border: 2px solid blue;")
            self.toggleSelectButton.setText("Deselect All")

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
    
    