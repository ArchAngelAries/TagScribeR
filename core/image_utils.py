import cv2
import numpy as np
import base64
from io import BytesIO
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

def image_to_base64(image_path):
    """Converts an image file to a base64 string for API usage."""
    try:
        with Image.open(image_path) as img:
            # Fix rotation based on EXIF
            img = ImageOps.exif_transpose(img)
            
            # Convert to RGB to ensure compatibility
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Resize if too massive (optional, but good for APIs)
            # Most VLMs choke on > 4096px
            if max(img.size) > 4096:
                img.thumbnail((4096, 4096))

            buff = BytesIO()
            img.save(buff, format="JPEG", quality=90)
            return base64.b64encode(buff.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Base64 Conversion Error: {e}")
        return None