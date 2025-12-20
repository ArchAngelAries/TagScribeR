import os
import cv2
import numpy as np
import traceback
import gc
from PIL import Image
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QScrollArea, QGridLayout, QFrame, QSplitter, QFileDialog, 
    QMessageBox, QGroupBox, QSpinBox, QComboBox, QRadioButton, 
    QCheckBox, QProgressBar, QTextEdit, QFormLayout, QSlider,
    QProgressDialog, QApplication
)
from PySide6.QtCore import Qt, Signal, QRunnable, QThreadPool, QObject, Slot
from PySide6.QtGui import QPixmap, QShortcut, QKeySequence
from core.image_utils import load_thumbnail

# Determine Root Directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDIT_DIR = os.path.join(BASE_DIR, "Image Edits")

# --- WORKER FOR THUMBNAILS (Keep this threaded, it's read-only and safe) ---
class ThumbnailSignals(QObject):
    loaded = Signal(str, QPixmap, str) 

class ThumbnailWorker(QRunnable):
    def __init__(self, path, size):
        super().__init__()
        self.path = path
        self.size = size
        self.signals = ThumbnailSignals()

    @Slot()
    def run(self):
        pix = load_thumbnail(self.path, self.size)
        info = "?"
        try:
            with Image.open(self.path) as img:
                w, h = img.size
                info = f"{w} x {h}"
        except: pass
        self.signals.loaded.emit(self.path, pix, info)

# --- VISUAL CARD ---
class EditorCard(QFrame):
    selection_changed = Signal(str, bool)

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.is_selected = False
        self.setFixedWidth(240)
        self.setFixedHeight(300)
        self.setFrameShape(QFrame.StyledPanel)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5,5,5,5)
        
        self.lbl_res = QLabel("Loading...")
        self.lbl_res.setStyleSheet("color: #aaa; font-size: 11px; font-weight: bold;")
        self.lbl_res.setAlignment(Qt.AlignCenter)
        
        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("background-color: #1e1e1e; border-radius: 4px;")
        self.lbl_image.setFixedHeight(220)
        
        self.lbl_name = QLabel(os.path.basename(path))
        self.lbl_name.setAlignment(Qt.AlignCenter)
        self.lbl_name.setStyleSheet("color: #888; font-size: 10px;")
        
        layout.addWidget(self.lbl_res)
        layout.addWidget(self.lbl_image)
        layout.addWidget(self.lbl_name)
        self.update_style()

    def set_data(self, path, pix, info):
        if not pix.isNull():
            self.lbl_image.setPixmap(pix)
        self.lbl_res.setText(info)

    def toggle_selection(self, state=None):
        if state is not None:
            self.is_selected = state
        else:
            self.is_selected = not self.is_selected
        self.update_style()
        self.selection_changed.emit(self.path, self.is_selected)

    def update_style(self):
        border = "#e17055" if self.is_selected else "transparent"
        self.setStyleSheet(f"EditorCard {{ background-color: #2b2b2b; border: 2px solid {border}; border-radius: 8px; }}")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.toggle_selection()
        super().mousePressEvent(event)

# --- MAIN EDITOR TAB ---
class EditorTab(QWidget):
    def __init__(self):
        super().__init__()
        self.cards = {}
        self.selected_paths = set()
        self.thread_pool = QThreadPool()
        self.current_folder = ""
        
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # LEFT
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        grid_tools = QHBoxLayout()
        self.btn_folder = QPushButton("📂 Open Folder")
        self.btn_folder.clicked.connect(self.load_folder)
        self.btn_select_all = QPushButton("Select All")
        self.btn_select_all.clicked.connect(self.select_all)
        grid_tools.addWidget(self.btn_folder)
        grid_tools.addWidget(self.btn_select_all)
        grid_tools.addStretch()
        left_layout.addLayout(grid_tools)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll.setWidget(self.grid_container)
        left_layout.addWidget(self.scroll)

        # RIGHT
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_widget = QWidget()
        right_widget.setFixedWidth(340)
        right_layout = QVBoxLayout(right_widget)
        right_scroll.setWidget(right_widget)

        grp_out = QGroupBox("1. Output Settings")
        lyt_out = QVBoxLayout(grp_out)
        self.rad_folder = QRadioButton("Save to 'Image Edits' (In App Root)")
        self.rad_folder.setChecked(True)
        self.rad_overwrite = QRadioButton("Overwrite Originals")
        lyt_out.addWidget(self.rad_folder)
        lyt_out.addWidget(self.rad_overwrite)
        right_layout.addWidget(grp_out)

        grp_rot = QGroupBox("2. Batch Rotate")
        lyt_rot = QHBoxLayout(grp_rot)
        self.btn_ccw = QPushButton("⟲ Left")
        self.btn_cw = QPushButton("⟳ Right")
        self.btn_ccw.clicked.connect(lambda: self.run_batch_main_thread('rotate', {'direction': 'ccw'}))
        self.btn_cw.clicked.connect(lambda: self.run_batch_main_thread('rotate', {'direction': 'cw'}))
        lyt_rot.addWidget(self.btn_ccw)
        lyt_rot.addWidget(self.btn_cw)
        right_layout.addWidget(grp_rot)

        grp_res = QGroupBox("3. Batch Resize")
        lyt_res = QVBoxLayout(grp_res)
        self.rad_longest = QRadioButton("Scale Longest Side")
        self.rad_longest.setChecked(True)
        self.rad_force = QRadioButton("Force Dimensions")
        self.spin_long = QSpinBox(); self.spin_long.setRange(64, 8192); self.spin_long.setValue(1024); self.spin_long.setSuffix(" px")
        self.wid_force = QWidget(); l_force = QHBoxLayout(self.wid_force)
        self.spin_w = QSpinBox(); self.spin_w.setRange(64, 8192); self.spin_w.setValue(512); self.spin_w.setPrefix("W: ")
        self.spin_h = QSpinBox(); self.spin_h.setRange(64, 8192); self.spin_h.setValue(512); self.spin_h.setPrefix("H: ")
        l_force.addWidget(self.spin_w); l_force.addWidget(self.spin_h)
        self.wid_force.hide()
        self.rad_longest.toggled.connect(lambda: (self.spin_long.show(), self.wid_force.hide()))
        self.rad_force.toggled.connect(lambda: (self.spin_long.hide(), self.wid_force.show()))
        self.btn_resize = QPushButton("Apply Resize")
        self.btn_resize.clicked.connect(self.prep_resize)
        self.btn_resize.setStyleSheet("background-color: #e17055; color: white;")
        lyt_res.addWidget(self.rad_longest); lyt_res.addWidget(self.rad_force); lyt_res.addWidget(self.spin_long); lyt_res.addWidget(self.wid_force); lyt_res.addWidget(self.btn_resize)
        right_layout.addWidget(grp_res)

        grp_crop = QGroupBox("4. Batch Crop")
        lyt_crop = QVBoxLayout(grp_crop)
        crop_dim = QHBoxLayout()
        self.spin_cw = QSpinBox(); self.spin_cw.setRange(64, 8192); self.spin_cw.setValue(512); self.spin_cw.setPrefix("W: ")
        self.spin_ch = QSpinBox(); self.spin_ch.setRange(64, 8192); self.spin_ch.setValue(512); self.spin_ch.setPrefix("H: ")
        crop_dim.addWidget(self.spin_cw); crop_dim.addWidget(self.spin_ch)
        self.combo_focus = QComboBox(); self.combo_focus.addItems(["Center", "Top-Left", "Top-Center", "Top-Right", "Bottom-Center"])
        self.btn_crop = QPushButton("Apply Crop")
        self.btn_crop.clicked.connect(self.prep_crop)
        self.btn_crop.setStyleSheet("background-color: #d63031; color: white;")
        lyt_crop.addLayout(crop_dim); lyt_crop.addWidget(QLabel("Focus:")); lyt_crop.addWidget(self.combo_focus); lyt_crop.addWidget(self.btn_crop)
        right_layout.addWidget(grp_crop)

        grp_conv = QGroupBox("5. Format Converter")
        lyt_conv = QVBoxLayout(grp_conv)
        self.combo_format = QComboBox(); self.combo_format.addItems(["JPG", "PNG", "WEBP", "BMP", "TIFF"])
        self.slider_quality = QSlider(Qt.Horizontal); self.slider_quality.setRange(1, 100); self.slider_quality.setValue(90)
        self.lbl_quality = QLabel("Quality: 90")
        self.slider_quality.valueChanged.connect(lambda v: self.lbl_quality.setText(f"Quality: {v}"))
        self.btn_convert = QPushButton("Apply Conversion")
        self.btn_convert.clicked.connect(self.prep_convert)
        self.btn_convert.setStyleSheet("background-color: #6c5ce7; color: white;")
        lyt_conv.addWidget(QLabel("Format:")); lyt_conv.addWidget(self.combo_format); lyt_conv.addWidget(self.lbl_quality); lyt_conv.addWidget(self.slider_quality); lyt_conv.addWidget(self.btn_convert)
        right_layout.addWidget(grp_conv)

        right_layout.addStretch()
        
        self.progress = QProgressBar()
        right_layout.addWidget(self.progress)
        self.log_box = QTextEdit()
        self.log_box.setPlaceholderText("Log...")
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(100)
        right_layout.addWidget(self.log_box)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        
        self.setup_hotkeys()

    def setup_hotkeys(self):
        QShortcut(QKeySequence("Ctrl+A"), self).activated.connect(self.select_all)
        QShortcut(QKeySequence("Ctrl+R"), self).activated.connect(lambda: self.run_batch_main_thread('rotate', {'direction': 'cw'}))
        QShortcut(QKeySequence("Ctrl+Shift+R"), self).activated.connect(lambda: self.run_batch_main_thread('rotate', {'direction': 'ccw'}))

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder: return
        self.current_folder = folder
        self.refresh_grid()

    def refresh_grid(self):
        for i in reversed(range(self.grid_layout.count())): 
            self.grid_layout.itemAt(i).widget().setParent(None)
        self.cards.clear()
        self.selected_paths.clear()
        
        exts = ('.jpg', '.png', '.jpeg', '.webp', '.bmp', '.tiff')
        files = [f for f in os.listdir(self.current_folder) if f.lower().endswith(exts)]
        cols = 3
        for i, f in enumerate(files):
            path = os.path.join(self.current_folder, f)
            card = EditorCard(path)
            card.selection_changed.connect(self.on_selection)
            self.grid_layout.addWidget(card, i // cols, i % cols)
            self.cards[path] = card
            self.reload_card_thumbnail(path)

    def reload_card_thumbnail(self, path):
        if path in self.cards:
            worker = ThumbnailWorker(path, (220, 220))
            worker.signals.loaded.connect(self.cards[path].set_data)
            self.thread_pool.start(worker)

    def on_selection(self, path, is_selected):
        if is_selected: self.selected_paths.add(path)
        else: self.selected_paths.discard(path)

    def select_all(self):
        target = not (len(self.selected_paths) == len(self.cards) and len(self.cards) > 0)
        for card in self.cards.values(): card.toggle_selection(target)

    def prep_resize(self):
        params = {}
        if self.rad_longest.isChecked():
            params['mode'] = 'longest'
            params['size'] = self.spin_long.value()
        else:
            params['mode'] = 'force'
            params['w'] = self.spin_w.value()
            params['h'] = self.spin_h.value()
        self.run_batch_main_thread('resize', params)

    def prep_crop(self):
        params = {
            'w': self.spin_cw.value(),
            'h': self.spin_ch.value(),
            'focus': self.combo_focus.currentText()
        }
        self.run_batch_main_thread('crop', params)

    def prep_convert(self):
        params = {
            'format': self.combo_format.currentText(),
            'quality': self.slider_quality.value()
        }
        self.run_batch_main_thread('convert', params)

    def run_batch_main_thread(self, operation, params):
        if not self.selected_paths:
            self.log_box.append("⚠️ No images selected!")
            return

        mode = 'overwrite' if self.rad_overwrite.isChecked() else 'folder'
        output_folder = EDIT_DIR if mode == 'folder' else ""
        
        if mode == 'folder':
            if not os.path.exists(output_folder):
                try: os.makedirs(output_folder)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Could not create output folder:\n{e}")
                    return

        paths = list(self.selected_paths)
        count = len(paths)
        progress = QProgressDialog(f"Running {operation}...", "Abort", 0, count, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        processed_count = 0
        
        for i, path in enumerate(paths):
            if progress.wasCanceled():
                self.log_box.append("🛑 Batch Aborted.")
                break
            
            progress.setValue(i)
            QApplication.processEvents()
            
            try:
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    # Unicode fallback
                    stream = np.fromfile(path, dtype=np.uint8)
                    img = cv2.imdecode(stream, cv2.IMREAD_UNCHANGED)
                
                if img is None:
                    self.log_box.append(f"❌ Failed load: {os.path.basename(path)}")
                    continue

                h, w = img.shape[:2]
                res_img = img
                new_ext = None
                save_params = []

                if operation == 'rotate':
                    direction = params.get('direction', 'cw')
                    if direction == 'cw': res_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    else: res_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                elif operation == 'resize':
                    mode_r = params.get('mode', 'longest')
                    new_w, new_h = w, h
                    if mode_r == 'longest':
                        target = int(params['size'])
                        scale = target / max(h, w)
                        new_w, new_h = int(w * scale), int(h * scale)
                    else:
                        new_w, new_h = int(params['w']), int(params['h'])
                    
                    new_w = max(1, new_w); new_h = max(1, new_h)
                    if new_w != w or new_h != h:
                        res_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                elif operation == 'crop':
                    t_w, t_h = int(params['w']), int(params['h'])
                    focus = params.get('focus', 'Center')
                    if w < t_w or h < t_h:
                        self.log_box.append(f"⚠️ Too small: {os.path.basename(path)}")
                        continue
                    
                    x = (w - t_w) // 2
                    y = (h - t_h) // 2
                    
                    if focus == 'Top-Left': x, y = 0, 0
                    elif focus == 'Top-Center': x, y = (w - t_w)//2, 0
                    elif focus == 'Top-Right': x, y = w - t_w, 0
                    elif focus == 'Center-Left': x, y = 0, (h - t_h)//2
                    elif focus == 'Center-Right': x, y = w - t_w, (h - t_h)//2
                    elif focus == 'Bottom-Left': x, y = 0, h - t_h
                    elif focus == 'Bottom-Center': x, y = (w - t_w)//2, h - t_h
                    elif focus == 'Bottom-Right': x, y = w - t_w, h - t_h
                    
                    x = max(0, min(x, w - t_w))
                    y = max(0, min(y, h - t_h))
                    res_img = img[y:y+t_h, x:x+t_w]

                elif operation == 'convert':
                    fmt = params['format'].lower()
                    quality = params['quality']
                    new_ext = f".{fmt}"
                    if fmt in ['jpg', 'jpeg']:
                        save_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                        if len(res_img.shape) == 3 and res_img.shape[2] == 4:
                            res_img = cv2.cvtColor(res_img, cv2.COLOR_BGRA2BGR)
                    elif fmt == 'webp':
                        save_params = [int(cv2.IMWRITE_WEBP_QUALITY), quality]

                filename = os.path.basename(path)
                if new_ext:
                    base = os.path.splitext(filename)[0]
                    filename = f"{base}{new_ext}"
                
                if mode == 'folder':
                    save_path = os.path.join(output_folder, filename)
                else:
                    save_path = os.path.join(os.path.dirname(path), filename)

                ext = os.path.splitext(save_path)[1]
                if not ext: ext = os.path.splitext(path)[1]
                
                success, encoded_img = cv2.imencode(ext, res_img, save_params)
                if success:
                    with open(save_path, "wb") as f:
                        encoded_img.tofile(f)
                    processed_count += 1
                    if save_path == path:
                        self.reload_card_thumbnail(path)
                else:
                    self.log_box.append(f"❌ Save failed: {filename}")

                gc.collect()

            except Exception as e:
                self.log_box.append(f"❌ Error {os.path.basename(path)}: {e}")

        progress.setValue(count)
        self.log_box.append(f"✅ Finished. Processed {processed_count} images.")