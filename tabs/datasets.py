import os
import shutil
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QScrollArea, QGridLayout, QFrame, QSplitter, QFileDialog, 
    QMessageBox, QListWidget, QLineEdit, QInputDialog, QTextEdit,
    QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QRunnable, QThreadPool, QObject, Slot
from PySide6.QtGui import QPixmap, QShortcut, QKeySequence
from core.image_utils import load_thumbnail

# Root folder for collections
DATASETS_ROOT = os.path.join(os.getcwd(), "Dataset Collections")

class ThumbnailWorker(QRunnable):
    class Signals(QObject):
        loaded = Signal(str, QPixmap)
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.signals = self.Signals()
    @Slot()
    def run(self):
        pix = load_thumbnail(self.path, (200, 200))
        self.signals.loaded.emit(self.path, pix)

class DatasetCard(QFrame):
    selection_changed = Signal(str, bool)
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.caption_text = ""
        self.is_selected = False
        
        self.setFixedSize(200, 320)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("background-color: #2b2b2b; border-radius: 6px;")
        
        lay = QVBoxLayout(self)
        lay.setContentsMargins(5,5,5,5)
        lay.setSpacing(2)
        
        self.lbl_img = QLabel("...")
        self.lbl_img.setAlignment(Qt.AlignCenter)
        self.lbl_img.setStyleSheet("background-color: #1e1e1e; border-radius: 4px;")
        self.lbl_img.setFixedHeight(150)
        
        self.lbl_name = QLabel(os.path.basename(path))
        self.lbl_name.setStyleSheet("color: #888; font-size: 9px; padding-bottom: 2px;")
        self.lbl_name.setWordWrap(False) 
        
        self.txt_caption = QTextEdit()
        self.txt_caption.setReadOnly(True)
        self.txt_caption.setPlaceholderText("No caption file...")
        self.txt_caption.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a; 
                color: #aaa; 
                border: 1px solid #333; 
                border-radius: 4px; 
                font-size: 10px;
            }
        """)
        
        lay.addWidget(self.lbl_img)
        lay.addWidget(self.lbl_name)
        lay.addWidget(self.txt_caption)
        
        self.load_caption()

    def load_caption(self):
        txt_path = os.path.splitext(self.path)[0] + ".txt"
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    self.caption_text = f.read().strip()
                    self.txt_caption.setText(self.caption_text)
            except: pass

    def set_pixmap(self, path, pix):
        if not pix.isNull(): self.lbl_img.setPixmap(pix)

    def toggle(self, state=None):
        if state is not None: self.is_selected = state
        else: self.is_selected = not self.is_selected
        
        border = "#0984e3" if self.is_selected else "transparent"
        self.setStyleSheet(f"background-color: #2b2b2b; border-radius: 6px; border: 2px solid {border};")
        self.selection_changed.emit(self.path, self.is_selected)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton: self.toggle()

class DatasetsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.cards = {}
        self.selected_paths = set()
        self.thread_pool = QThreadPool()
        self.current_view_folder = "" 
        
        if not os.path.exists(DATASETS_ROOT):
            os.makedirs(DATASETS_ROOT)

        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # --- LEFT: Grid ---
        left_widget = QWidget()
        left_lay = QVBoxLayout(left_widget)
        
        # Tools
        tools = QHBoxLayout()
        self.btn_load = QPushButton("📂 Load Source")
        self.btn_load.clicked.connect(self.load_folder_dialog)
        
        self.inp_filter = QLineEdit()
        self.inp_filter.setPlaceholderText("Filter tags...")
        self.inp_filter.textChanged.connect(self.apply_filter)
        
        self.btn_sel_all = QPushButton("Select All")
        self.btn_sel_all.setMinimumWidth(80) 
        self.btn_sel_all.clicked.connect(self.select_all)
        
        tools.addWidget(self.btn_load, 0)
        tools.addWidget(self.inp_filter, 1) 
        tools.addWidget(self.btn_sel_all, 0)
        left_lay.addLayout(tools)
        
        # Delete Images Button
        self.btn_del_imgs = QPushButton("🗑️ Delete Selected Images from Disk")
        self.btn_del_imgs.setStyleSheet("background-color: #d63031; color: white; font-weight: bold;")
        self.btn_del_imgs.clicked.connect(self.delete_selected_images)
        self.btn_del_imgs.hide()
        left_lay.addWidget(self.btn_del_imgs)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll.setWidget(self.grid_container)
        left_lay.addWidget(self.scroll)

        # --- RIGHT: Collections ---
        right_widget = QWidget()
        right_widget.setMinimumWidth(250) 
        right_lay = QVBoxLayout(right_widget)
        
        right_lay.addWidget(QLabel("<b>Dataset Collections</b>"))
        
        self.list_datasets = QListWidget()
        self.list_datasets.itemClicked.connect(self.update_buttons)
        self.list_datasets.itemDoubleClicked.connect(self.load_dataset_into_grid)
        right_lay.addWidget(self.list_datasets)
        
        # Collection Management Buttons
        col_tools = QHBoxLayout()
        self.btn_new = QPushButton("New")
        self.btn_new.clicked.connect(self.create_collection)
        
        self.btn_del_col = QPushButton("🗑️ Delete")
        self.btn_del_col.setMinimumWidth(70) 
        self.btn_del_col.setStyleSheet("background-color: #d63031; color: white;")
        self.btn_del_col.clicked.connect(self.delete_collection)
        
        self.btn_refresh = QPushButton("🔄")
        self.btn_refresh.setFixedWidth(30)
        self.btn_refresh.clicked.connect(self.refresh_collections)
        
        col_tools.addWidget(self.btn_new)
        col_tools.addWidget(self.btn_del_col)
        col_tools.addWidget(self.btn_refresh)
        right_lay.addLayout(col_tools)
        
        self.btn_add = QPushButton("➕ Add Selected to Collection")
        self.btn_add.clicked.connect(self.add_to_collection)
        self.btn_add.setStyleSheet("background-color: #00b894; color: white; font-weight: bold; height: 40px;")
        self.btn_add.setEnabled(False)
        right_lay.addWidget(self.btn_add)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        
        self.refresh_collections()
        self.setup_hotkeys()

    def setup_hotkeys(self):
        QShortcut(QKeySequence("Ctrl+A"), self).activated.connect(self.select_all)
        QShortcut(QKeySequence("Ctrl+F"), self).activated.connect(self.inp_filter.setFocus)
        QShortcut(QKeySequence("Ctrl+N"), self).activated.connect(self.create_collection)
        QShortcut(QKeySequence("F5"), self).activated.connect(self.refresh_collections)
        QShortcut(QKeySequence("Del"), self).activated.connect(self.delete_selected_images)
        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(self.add_to_collection)
        QShortcut(QKeySequence("Ctrl+Enter"), self).activated.connect(self.add_to_collection)

    def load_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Source Images")
        if folder: self.load_grid(folder)

    def load_dataset_into_grid(self, item):
        col_name = item.text()
        folder = os.path.join(DATASETS_ROOT, col_name)
        self.load_grid(folder)

    def load_grid(self, folder):
        self.current_view_folder = folder
        for i in reversed(range(self.grid_layout.count())): 
            self.grid_layout.itemAt(i).widget().setParent(None)
        self.cards.clear()
        self.selected_paths.clear()
        self.btn_del_imgs.hide()
        
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.png','.jpeg','.webp'))]
        cols = 4
        
        for i, f in enumerate(files):
            path = os.path.join(folder, f)
            card = DatasetCard(path)
            card.selection_changed.connect(self.on_selection)
            self.grid_layout.addWidget(card, i//cols, i%cols)
            self.cards[path] = card
            
            worker = ThumbnailWorker(path)
            worker.signals.loaded.connect(card.set_pixmap)
            self.thread_pool.start(worker)

    def apply_filter(self, text):
        search = text.lower().strip()
        for path, card in self.cards.items():
            match = (search in card.caption_text.lower()) or (search in os.path.basename(path).lower())
            if not search: match = True
            card.setVisible(match)

    def on_selection(self, path, state):
        if state: self.selected_paths.add(path)
        else: self.selected_paths.discard(path)
        
        if len(self.selected_paths) > 0:
            self.btn_del_imgs.show()
            self.btn_del_imgs.setText(f"🗑️ Delete {len(self.selected_paths)} Images")
        else:
            self.btn_del_imgs.hide()
            
        self.update_buttons()

    def select_all(self):
        visible_cards = [c for c in self.cards.values() if not c.isHidden()]
        all_vis_selected = all(c.is_selected for c in visible_cards)
        target = not all_vis_selected
        for c in visible_cards: c.toggle(target)

    def refresh_collections(self):
        self.list_datasets.clear()
        if os.path.exists(DATASETS_ROOT):
            folders = [f for f in os.listdir(DATASETS_ROOT) if os.path.isdir(os.path.join(DATASETS_ROOT, f))]
            self.list_datasets.addItems(folders)

    def create_collection(self):
        name, ok = QInputDialog.getText(self, "New Collection", "Name:")
        if ok and name:
            path = os.path.join(DATASETS_ROOT, name)
            if not os.path.exists(path):
                os.makedirs(path)
                self.refresh_collections()

    def delete_collection(self):
        item = self.list_datasets.currentItem()
        if not item: return
        
        col_name = item.text()
        reply = QMessageBox.question(self, "Delete Collection", 
                                     f"Are you sure you want to delete '{col_name}'?\nThis will delete the folder and all files inside it!",
                                     QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            path = os.path.join(DATASETS_ROOT, col_name)
            try:
                shutil.rmtree(path)
                self.refresh_collections()
                if self.current_view_folder == path:
                    self.load_grid(DATASETS_ROOT) 
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete: {e}")

    def delete_selected_images(self):
        if not self.selected_paths: return
        
        reply = QMessageBox.question(self, "Delete Images", 
                                     f"Are you sure you want to delete {len(self.selected_paths)} images?\nThis will permanently remove them from disk.",
                                     QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            for path in list(self.selected_paths):
                try:
                    os.remove(path)
                    txt_path = os.path.splitext(path)[0] + ".txt"
                    if os.path.exists(txt_path):
                        os.remove(txt_path)
                    
                    if path in self.cards:
                        card = self.cards.pop(path)
                        card.setParent(None)
                        card.deleteLater()
                    
                    self.selected_paths.discard(path)
                except Exception as e:
                    print(f"Error deleting {path}: {e}")
            
            self.btn_del_imgs.hide()
            self.update_buttons()

    def update_buttons(self):
        has_sel = len(self.selected_paths) > 0
        has_col = self.list_datasets.currentItem() is not None
        self.btn_add.setEnabled(has_sel and has_col)
        self.btn_del_col.setEnabled(has_col)
        if has_sel:
            self.btn_add.setText(f"➕ Add {len(self.selected_paths)} to Collection")
        else:
            self.btn_add.setText("➕ Add Selected to Collection")

    def add_to_collection(self):
        item = self.list_datasets.currentItem()
        if not item: return
        col_name = item.text()
        dest_dir = os.path.join(DATASETS_ROOT, col_name)
        
        count = 0
        for path in self.selected_paths:
            try:
                shutil.copy2(path, dest_dir)
                txt_path = os.path.splitext(path)[0] + ".txt"
                if os.path.exists(txt_path):
                    shutil.copy2(txt_path, dest_dir)
                count += 1
            except Exception as e:
                print(f"Copy error: {e}")
        
        QMessageBox.information(self, "Success", f"Added {count} images to '{col_name}'.")
        for c in self.cards.values(): c.toggle(False)