import cv2
import numpy as np
from PIL import Image, ImageOps
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt

def cv2_to_qpixmap(cv_img):
    """Convert OpenCV BGR image to QPixmap."""
    if cv_img is None: return QPixmap()
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    # Create QImage and COPY the data to ensure it persists
    qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)

def pil_to_qpixmap(pil_img):
    """Convert PIL image to QPixmap safely."""
    if pil_img is None: return QPixmap()
    
    # Convert to RGBA (Handles transparency and 32-bit alignment better in Qt)
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    
    data = pil_img.tobytes("raw", "RGBA")
    
    # Create QImage
    qimg = QImage(data, pil_img.width, pil_img.height, QImage.Format_RGBA8888)
    
    # .copy() is CRITICAL here. 
    # Without it, 'data' gets garbage collected, resulting in black/noise images.
    return QPixmap.fromImage(qimg.copy())

def load_thumbnail(path, size=(300, 300)):
    """Efficiently load a thumbnail."""
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img) # Fix rotation
        img.thumbnail(size, Image.Resampling.LANCZOS)
        return pil_to_qpixmap(img)
    except Exception as e:
        print(f"Error loading thumbnail {path}: {e}")
        return QPixmap()