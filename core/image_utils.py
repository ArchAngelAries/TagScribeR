import base64
import logging

import cv2
from io import BytesIO
from PIL import Image, ImageOps
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt

log = logging.getLogger(__name__)

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

def pil_to_qimage(pil_img) -> QImage:
    """PIL -> QImage. Safe to call from worker threads (unlike QPixmap)."""
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    data = pil_img.tobytes("raw", "RGBA")
    return QImage(data, pil_img.width, pil_img.height, QImage.Format_RGBA8888).copy()


def load_qimage(path, size=(300, 300)):
    """Upright, downscaled QImage for display; returns (QImage, (orig_w, orig_h)). Thread-safe."""
    with Image.open(path) as img:
        img.draft("RGB", (size[0] * 2, size[1] * 2))
        img = ImageOps.exif_transpose(img)
        orig = img.size
        img.thumbnail(size, Image.Resampling.LANCZOS)
        return pil_to_qimage(img), orig


def load_thumbnail(path, size=(300, 300)):
    """Efficiently load an upright thumbnail (JPEG draft decoding avoids full-size decodes)."""
    try:
        with Image.open(path) as img:
            img.draft("RGB", (size[0] * 2, size[1] * 2))  # no-op for non-JPEG formats
            img = ImageOps.exif_transpose(img)
            img.thumbnail(size, Image.Resampling.LANCZOS)
            return pil_to_qpixmap(img)
    except Exception as e:
        log.warning("Could not load thumbnail %s: %s", path, e)
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