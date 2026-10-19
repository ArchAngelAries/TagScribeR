import base64
import logging

import cv2
from io import BytesIO
from PIL import Image, ImageOps
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt

from core.image_ops import to_8bit

log = logging.getLogger(__name__)

# Thumbnails and previews refuse to decode anything above this many pixels (a PNG header can claim 400 Mpx
# while the file is 1 MB; decoding it costs gigabytes per thumbnail worker). JPEGs are first reduced with
# Image.draft, so only their reduced size counts. The Editor's explicit open is not limited by this.
MAX_THUMBNAIL_PIXELS = 80_000_000


class ImageTooLarge(OSError):
    """The image is too big to decode for a thumbnail or preview."""


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
    
    pil_img = to_8bit(pil_img)   # 16-bit greyscale would be clipped by convert()
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
    pil_img = to_8bit(pil_img)
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    data = pil_img.tobytes("raw", "RGBA")
    return QImage(data, pil_img.width, pil_img.height, QImage.Format_RGBA8888).copy()


def _open_for_display(img, size):
    """Draft-reduce, check the pixel budget, apply EXIF orientation, 8-bit and downscale. Returns (image, orig)."""
    w, h = img.size
    img.draft("RGB", (size[0] * 2, size[1] * 2))  # no-op for non-JPEG formats
    if img.size[0] * img.size[1] > MAX_THUMBNAIL_PIXELS:
        raise ImageTooLarge(f"Image is {w}\u00d7{h} ({w * h / 1e6:.0f} Mpx), too large to preview")
    if img.getexif().get(0x0112) in (5, 6, 7, 8):
        w, h = h, w  # orientation swaps the axes
    img = to_8bit(ImageOps.exif_transpose(img))       # transpose first: to_8bit's copy carries no EXIF
    img.thumbnail(size, Image.Resampling.LANCZOS)
    return img, (w, h)


def load_qimage(path, size=(300, 300)):
    """Upright, downscaled QImage for display; returns (QImage, (orig_w, orig_h)). Thread-safe.

    Raises ImageTooLarge instead of decoding an image above MAX_THUMBNAIL_PIXELS.
    """
    with Image.open(path) as img:
        img, orig = _open_for_display(img, size)
        return pil_to_qimage(img), orig


def load_thumbnail(path, size=(300, 300)):
    """Efficiently load an upright thumbnail (JPEG draft decoding avoids full-size decodes)."""
    try:
        with Image.open(path) as img:
            img, _ = _open_for_display(img, size)
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