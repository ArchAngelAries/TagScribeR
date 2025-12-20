import os
import shutil
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QPushButton, 
    QLineEdit, QGridLayout, QTextEdit, QSplitter, QFileDialog, QMessageBox, 
    QFrame, QListWidget, QProgressBar, QApplication, QInputDialog, QRadioButton
)
from PySide6.QtCore import Qt, Signal, QRunnable, QThreadPool, QObject, Slot
from PySide6.QtGui import QPixmap, QShortcut, QKeySequence, QIcon
from core.image_utils import load_thumbnail

TAG_FILE = "user_tags.txt"

# --- WORKER ---
class ThumbnailSignals(QObject):
    loaded = Signal(str, QPixmap) 

class ThumbnailWorker(QRunnable):
    def __init__(self, path, size):
        super().__init__()
        self.path = path
        self.size = size
        self.signals = ThumbnailSignals()

    @Slot()
    def run(self):
        pix = load_thumbnail(self.path, self.size)
        self.signals.loaded.emit(self.path, pix)

# --- IMAGE CARD ---
class ImageCard(QFrame):
    selection_changed = Signal(str, bool)

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path
        self.is_selected = False
        
        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedWidth(260) 
        self.setFixedHeight(380)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(5)

        self.lbl_image = QLabel("Loading...")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("background-color: #1e1e1e; border-radius: 4px; color: #888;")
        self.lbl_image.setFixedHeight(200) 
        
        self.txt_caption = QTextEdit()
        self.txt_caption.setPlaceholderText("Caption...")
        self.txt_caption.setStyleSheet("QTextEdit { background-color: #1e1e1e; border: 1px solid #333; border-radius: 4px; padding: 5px; color: #ddd; }")
        
        self.layout.addWidget(self.lbl_image)
        self.layout.addWidget(self.txt_caption)

        self.load_text()
        self.update_style()

    def set_image(self, path, pixmap):
        if not pixmap.isNull():
            self.lbl_image.setPixmap(pixmap)
            self.lbl_image.setText("")
        else:
            self.lbl_image.setText("Error")

    def load_text(self):
        txt_path = os.path.splitext(self.path)[0] + ".txt"
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    self.txt_caption.setText(f.read())
            except: pass

    def save_text(self):
        txt_path = os.path.splitext(self.path)[0] + ".txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(self.txt_caption.toPlainText())

    def clear_text(self):
        self.txt_caption.clear()

    def toggle_selection(self, force_state=None):
        if force_state is not None:
            self.is_selected = force_state
        else:
            self.is_selected = not self.is_selected
        self.update_style()
        self.selection_changed.emit(self.path, self.is_selected)

    def update_style(self):
        border_color = "#00b894" if self.is_selected else "transparent"
        bg_color = "#2b2b2b"
        self.setStyleSheet(f"ImageCard {{ background-color: {bg_color}; border-radius: 8px; border: 2px solid {border_color}; }}")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.toggle_selection()
        super().mousePressEvent(event)

# --- GALLERY TAB ---
class GalleryTab(QWidget):
    image_selected = Signal(str) 

    def __init__(self):
        super().__init__()
        self.current_folder = ""
        self.image_cards = {} 
        self.selected_paths = set()
        self.thread_pool = QThreadPool() 

        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # --- LEFT PANEL ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        toolbar = QHBoxLayout()
        self.btn_open = QPushButton("📂 Open Folder")
        self.btn_open.clicked.connect(self.select_folder)
        self.btn_open.setStyleSheet("background-color: #00b894; color: white; font-weight: bold;")
        
        self.btn_select_all = QPushButton("Select All")
        self.btn_select_all.clicked.connect(self.select_all)
        
        # Undo/Redo Buttons (Visual placeholders for now)
        self.btn_undo = QPushButton("↩️")
        self.btn_undo.setToolTip("Undo (Ctrl+Z) - Logic coming next")
        self.btn_redo = QPushButton("↪️")
        self.btn_redo.setToolTip("Redo (Ctrl+Y) - Logic coming next")
        
        self.btn_save_all = QPushButton("💾 Save")
        self.btn_save_all.clicked.connect(self.save_all)
        self.btn_save_all.setStyleSheet("background-color: #0984e3; color: white; font-weight: bold;")

        self.btn_dataset = QPushButton("📦 Save to Dataset")
        self.btn_dataset.clicked.connect(self.save_to_dataset)
        self.btn_dataset.setStyleSheet("background-color: #6c5ce7; color: white; font-weight: bold;")

        toolbar.addWidget(self.btn_open)
        toolbar.addWidget(self.btn_select_all)
        toolbar.addWidget(self.btn_undo)
        toolbar.addWidget(self.btn_redo)
        toolbar.addWidget(self.btn_save_all)
        toolbar.addWidget(self.btn_dataset)
        left_layout.addLayout(toolbar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll.setWidget(self.grid_container)
        left_layout.addWidget(self.scroll)

        # --- RIGHT PANEL ---
        right_widget = QWidget()
        right_widget.setFixedWidth(300)
        right_layout = QVBoxLayout(right_widget)

        right_layout.addWidget(QLabel("<b>Quick Tags</b>"))
        
        mode_layout = QHBoxLayout()
        self.rad_append = QRadioButton("Append")
        self.rad_prepend = QRadioButton("Prepend")
        self.rad_append.setChecked(True)
        mode_layout.addWidget(self.rad_append)
        mode_layout.addWidget(self.rad_prepend)
        right_layout.addLayout(mode_layout)

        tag_add_layout = QHBoxLayout()
        self.inp_new_tag = QLineEdit()
        self.inp_new_tag.setPlaceholderText("New tag...")
        self.inp_new_tag.returnPressed.connect(self.add_custom_tag)
        self.btn_add_tag = QPushButton("+")
        self.btn_add_tag.setFixedWidth(30)
        self.btn_add_tag.clicked.connect(self.add_custom_tag)
        tag_add_layout.addWidget(self.inp_new_tag)
        tag_add_layout.addWidget(self.btn_add_tag)
        right_layout.addLayout(tag_add_layout)

        self.tag_list = QListWidget()
        self.tag_list.itemClicked.connect(self.apply_tag_to_selection)
        self.load_tags() 
        right_layout.addWidget(self.tag_list)

        right_layout.addWidget(QLabel("<i>Click a tag to add it to selected images.</i>"))
        right_layout.addStretch()

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)
        
        self.setup_hotkeys()

    def setup_hotkeys(self):
        # Toggle Select All
        QShortcut(QKeySequence("Ctrl+A"), self).activated.connect(self.select_all)
        # Save All
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.save_all)
        # Clear Selection Text
        QShortcut(QKeySequence("Del"), self).activated.connect(self.delete_text_selection)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.current_folder = folder
            self.load_grid()

    def load_grid(self):
        for i in reversed(range(self.grid_layout.count())): 
            self.grid_layout.itemAt(i).widget().setParent(None)
        self.image_cards.clear()
        self.selected_paths.clear()

        exts = ('.jpg', '.jpeg', '.png', '.webp')
        try:
            files = [f for f in os.listdir(self.current_folder) if f.lower().endswith(exts)]
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read directory: {e}")
            return

        cols = 4
        for i, f in enumerate(files):
            path = os.path.join(self.current_folder, f)
            card = ImageCard(path)
            card.selection_changed.connect(self.on_card_selection)
            self.grid_layout.addWidget(card, i // cols, i % cols)
            self.image_cards[path] = card
            worker = ThumbnailWorker(path, (250, 200))
            worker.signals.loaded.connect(card.set_image)
            self.thread_pool.start(worker)

    def on_card_selection(self, path, is_selected):
        if is_selected:
            self.selected_paths.add(path)
            self.image_selected.emit(path) 
        else:
            self.selected_paths.discard(path)

    def select_all(self):
        all_selected = len(self.selected_paths) == len(self.image_cards) and len(self.image_cards) > 0
        toggle_to = not all_selected
        for card in self.image_cards.values():
            card.toggle_selection(toggle_to)

    def delete_text_selection(self):
        # Don't delete if focusing input field
        if self.inp_new_tag.hasFocus(): return
        
        # Check specific focus within cards
        focus_widget = QApplication.focusWidget()
        if isinstance(focus_widget, QTextEdit): return

        if self.selected_paths:
            reply = QMessageBox.question(self, "Clear Captions", f"Clear text for {len(self.selected_paths)} images?", QMessageBox.Yes|QMessageBox.No)
            if reply == QMessageBox.Yes:
                for path in self.selected_paths:
                    if path in self.image_cards:
                        self.image_cards[path].clear_text()

    def save_all(self):
        count = 0
        for card in self.image_cards.values():
            card.save_text()
            count += 1
        QMessageBox.information(self, "Saved", f"Saved captions for {count} images.")

    def save_to_dataset(self):
        if not self.selected_paths:
            QMessageBox.warning(self, "No Selection", "Please select images.")
            return
        name, ok = QInputDialog.getText(self, "New Dataset", "Enter Dataset Collection Name:")
        if not ok or not name.strip(): return
        
        dataset_root = os.path.join(os.getcwd(), "Dataset Collections")
        target_dir = os.path.join(dataset_root, name.strip())
        if not os.path.exists(target_dir): os.makedirs(target_dir)
            
        count = 0
        for path in self.selected_paths:
            if path in self.image_cards: self.image_cards[path].save_text()
            try:
                shutil.copy2(path, target_dir)
                txt_path = os.path.splitext(path)[0] + ".txt"
                if os.path.exists(txt_path): shutil.copy2(txt_path, target_dir)
                count += 1
            except Exception as e: print(f"Error copying {path}: {e}")
        QMessageBox.information(self, "Success", f"Successfully copied {count} items to '{name}'.")

    # --- TAG LOGIC ---
    def load_tags(self):
        defaults = ["masterpiece", "best quality", "4k", "photo", "illustration", 
                    "scenery", "portrait", "simple background", "solo", "1girl", "1boy"]
        tags_to_load = defaults
        if os.path.exists(TAG_FILE):
            try:
                with open(TAG_FILE, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    tags_to_load = [line.strip() for line in lines if line.strip()]
            except: pass
        self.tag_list.clear()
        self.tag_list.addItems(tags_to_load)

    def save_tags_to_file(self, tags):
        try:
            with open(TAG_FILE, "w", encoding="utf-8") as f:
                for tag in tags: f.write(f"{tag}\n")
        except: pass

    def add_custom_tag(self):
        tag = self.inp_new_tag.text().strip()
        if tag:
            items = [self.tag_list.item(i).text() for i in range(self.tag_list.count())]
            if tag not in items:
                self.tag_list.addItem(tag)
                try:
                    with open(TAG_FILE, "a", encoding="utf-8") as f: f.write(f"{tag}\n")
                except: pass
            self.inp_new_tag.clear()

    def apply_tag_to_selection(self, item):
        tag = item.text()
        if not self.selected_paths:
            QMessageBox.warning(self, "No Selection", "Select images in the grid first.")
            return

        for path in self.selected_paths:
            card = self.image_cards.get(path)
            if card:
                current_text = card.txt_caption.toPlainText().strip()
                if not current_text:
                    new_text = tag
                else:
                    tags = [t.strip() for t in current_text.split(',')]
                    if tag in tags: continue
                    if self.rad_prepend.isChecked():
                        new_text = f"{tag}, {current_text}"
                    else:
                        new_text = f"{current_text}, {tag}"
                card.txt_caption.setText(new_text)