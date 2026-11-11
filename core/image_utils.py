import base64
import logging
import math

from io import BytesIO
from PIL import Image, ImageOps
from PySide6.QtGui import QImage, QImageReader, QPixmap
from PySide6.QtCore import QSize

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
    import cv2          # imported here: it is only needed by this helper and costs start-up time
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


def _thumbnail_size(w: int, h: int, size) -> tuple[int, int]:
    """The size Image.thumbnail gives a w x h picture for the box `size`: the same arithmetic (Pillow 12), so the Qt
    path and the Pillow path end with the same pixel dimensions."""
    x, y = (math.floor(v) for v in size)
    if x >= w and y >= h:
        return w, h

    def round_aspect(number, key):
        return max(min(math.floor(number), math.ceil(number), key=key), 1)

    aspect = w / h
    if x / y >= aspect:
        x = round_aspect(y * aspect, key=lambda n: abs(aspect - n / y))
    else:
        y = round_aspect(x / aspect, key=lambda n: 0 if n == 0 else abs(aspect - x / n))
    return x, y


# What the Qt reader is trusted with: plain 8-bit pictures of these types (see _qt_load_qimage).
_QT_MODES = ("RGB", "L", "RGBA")
_QT_FORMATS = ("JPEG", "PNG", "WEBP")


def _qt_load_qimage(path, size):
    """Fast decode with Qt (a scaled read; the reader applies the EXIF orientation) for plain 8-bit JPEG, PNG and WebP
    files. Returns (QImage, (orig_w, orig_h)) like the Pillow path, or None for anything else (16-bit, palette,
    CMYK, animated, a colour profile, a gamma or a transparency key that Pillow and Qt could treat differently) or
    when Qt gives another size than expected: the caller then uses the Pillow path. Raises ImageTooLarge like it."""
    with Image.open(path) as img:
        if (img.format not in _QT_FORMATS or img.mode not in _QT_MODES or getattr(img, "is_animated", False)
                or "icc_profile" in img.info or "gamma" in img.info or "transparency" in img.info):
            return None
        w, h = img.size
        img.draft("RGB", (size[0] * 2, size[1] * 2))          # JPEG only; its reduced size is what the guard counts
        dw, dh = img.size
        if dw * dh > MAX_THUMBNAIL_PIXELS:
            raise ImageTooLarge(f"Image is {w}×{h} ({w * h / 1e6:.0f} Mpx), too large to preview")
        orientation = img.getexif().get(0x0112, 1)
    swap = orientation in (5, 6, 7, 8)
    if swap:
        w, h, dw, dh = h, w, dh, dw                           # the sizes once the picture is upright
    tw, th = _thumbnail_size(dw, dh, size)                    # Pillow works on the drafted picture, so do the same
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    if (tw, th) != (w, h):
        reader.setScaledSize(QSize(th, tw) if swap else QSize(tw, th))   # scaled before the reader turns it upright
    qimg = reader.read()
    if qimg.isNull() or (qimg.width(), qimg.height()) != (tw, th):
        return None
    return qimg.convertToFormat(QImage.Format_RGBA8888), (w, h)


def load_qimage(path, size=(300, 300)):
    """Upright, downscaled QImage for display; returns (QImage, (orig_w, orig_h)). Thread-safe.

    Raises ImageTooLarge instead of decoding an image above MAX_THUMBNAIL_PIXELS.
    """
    try:
        fast = _qt_load_qimage(path, size)
    except ImageTooLarge:
        raise
    except Exception:
        fast = None                     # anything unusual: the Pillow path below decides
    if fast is not None:
        return fast
    return _pillow_load_qimage(path, size)


def _pillow_load_qimage(path, size):
    """The Pillow decode path: handles every format and mode, and is the reference the Qt path is tested against."""
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