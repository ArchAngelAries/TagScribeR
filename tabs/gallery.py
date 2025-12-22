import os
import shutil
import unicodedata
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QPushButton, 
    QLineEdit, QGridLayout, QTextEdit, QSplitter, QFileDialog, QMessageBox, 
    QFrame, QListWidget, QProgressBar, QApplication, QInputDialog, QRadioButton,
    QGroupBox, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QRunnable, QThreadPool, QObject, Slot, QEvent
from PySide6.QtGui import QPixmap, QShortcut, QKeySequence, QIcon, QUndoStack, QUndoCommand
from core.image_utils import load_thumbnail
from core.tagger import WD14Tagger
from core.widgets import TagEditorWidget, AutoTagDialog

TAG_FILE = "user_tags.txt"

# --- UNDO COMMANDS ---
class UpdateCaptionCommand(QUndoCommand):
    def __init__(self, card, old_text, new_text):
        super().__init__()
        self.card = card
        self.old_text = old_text
        self.new_text = new_text
        self.setText(f"Edit: {os.path.basename(card.path)}")

    def redo(self):
        self.card.txt_caption.setPlainText(self.new_text)
        if self.card.is_selected:
            self.card.text_changed_internal.emit(self.card.path, self.new_text)

    def undo(self):
        self.card.txt_caption.setPlainText(self.old_text)
        if self.card.is_selected:
            self.card.text_changed_internal.emit(self.card.path, self.old_text)

class BatchUpdateCommand(QUndoCommand):
    def __init__(self, cards, new_texts, description="Batch Update"):
        super().__init__(description)
        self.cards = cards
        self.new_texts = new_texts
        self.old_texts = [c.txt_caption.toPlainText() for c in cards]

    def redo(self):
        for i, card in enumerate(self.cards):
            card.txt_caption.setPlainText(self.new_texts[i])
            if card.is_selected:
                card.text_changed_internal.emit(card.path, self.new_texts[i])

    def undo(self):
        for i, card in enumerate(self.cards):
            card.txt_caption.setPlainText(self.old_texts[i])
            if card.is_selected:
                card.text_changed_internal.emit(card.path, self.old_texts[i])

# --- WORKERS ---
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

class TaggerSignals(QObject):
    finished = Signal(object, list)

class TaggerWorker(QRunnable):
    def __init__(self, tagger, card, settings):
        super().__init__()
        self.tagger = tagger
        self.card = card
        self.settings = settings
        self.signals = TaggerSignals()
        
    @Slot()
    def run(self):
        tags = self.tagger.tag_image(
            self.card.path, 
            threshold=self.settings['threshold'],
            max_tags=self.settings['max_tags'],
            blacklist=self.settings['blacklist']
        )
        self.signals.finished.emit(self.card, tags)

# --- IMAGE CARD ---
class ImageCard(QFrame):
    selection_changed = Signal(str, bool)
    text_changed_internal = Signal(str, str) 
    undo_req = Signal(str, str, str) 

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path
        self.is_selected = False
        self._cached_text = "" 
        
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
        self.txt_caption.installEventFilter(self)

        self.layout.addWidget(self.lbl_image)
        self.layout.addWidget(self.txt_caption)

        self.load_text()
        self.update_style()

    def eventFilter(self, obj, event):
        if obj == self.txt_caption:
            if event.type() == QEvent.FocusIn:
                self._cached_text = self.txt_caption.toPlainText()
            elif event.type() == QEvent.FocusOut:
                new_text = self.txt_caption.toPlainText()
                if self._cached_text != new_text:
                    self.undo_req.emit(self.path, self._cached_text, new_text)
                    self.text_changed_internal.emit(self.path, new_text)
        return super().eventFilter(obj, event)

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
                    content = f.read()
                    self.txt_caption.setPlainText(content)
                    self._cached_text = content
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
        self.undo_stack = QUndoStack(self)
        self.tagger = WD14Tagger() 
        self.active_card = None 

        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # LEFT PANEL
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Tools
        toolbar = QHBoxLayout()
        self.btn_open = QPushButton("📂 Open Folder")
        self.btn_open.clicked.connect(self.select_folder)
        self.btn_open.setStyleSheet("background-color: #00b894; color: white; font-weight: bold;")
        
        # Filter (Stretch 1)
        self.inp_filter = QLineEdit()
        self.inp_filter.setPlaceholderText("Filter tags/names...")
        self.inp_filter.textChanged.connect(self.apply_filter)
        
        self.btn_select_all = QPushButton("Select All")
        self.btn_select_all.clicked.connect(self.select_all)
        self.btn_select_all.setMinimumWidth(80)
        
        # Undo/Redo Buttons (Text Based for Visibility)
        self.btn_undo = QPushButton("Undo")
        self.btn_undo.setToolTip("Undo (Ctrl+Z)")
        self.btn_undo.clicked.connect(self.undo_stack.undo)
        self.btn_undo.setEnabled(False) 
        self.btn_undo.setMinimumWidth(50)
        
        self.btn_redo = QPushButton("Redo")
        self.btn_redo.setToolTip("Redo (Ctrl+Y)")
        self.btn_redo.clicked.connect(self.undo_stack.redo)
        self.btn_redo.setEnabled(False)
        self.btn_redo.setMinimumWidth(50)

        self.undo_stack.canUndoChanged.connect(self.btn_undo.setEnabled)
        self.undo_stack.canRedoChanged.connect(self.btn_redo.setEnabled)
        
        self.btn_sanitize = QPushButton("Sanitize")
        self.btn_sanitize.setToolTip("Convert special characters (ä->a)")
        self.btn_sanitize.clicked.connect(self.sanitize_selection)
        self.btn_sanitize.setStyleSheet("background-color: #e17055; color: white;")
        self.btn_sanitize.setMinimumWidth(70)

        self.btn_save_all = QPushButton("💾 Save")
        self.btn_save_all.clicked.connect(self.save_all)
        self.btn_save_all.setStyleSheet("background-color: #0984e3; color: white; font-weight: bold;")

        self.btn_dataset = QPushButton("📦 Dataset")
        self.btn_dataset.setToolTip("Save selected to Dataset")
        self.btn_dataset.clicked.connect(self.save_to_dataset)
        self.btn_dataset.setStyleSheet("background-color: #6c5ce7; color: white; font-weight: bold;")

        # Add to Layout with stretch
        toolbar.addWidget(self.btn_open)
        toolbar.addWidget(self.inp_filter, 1) # Give filter max space
        toolbar.addWidget(self.btn_select_all)
        toolbar.addWidget(self.btn_undo)
        toolbar.addWidget(self.btn_redo)
        toolbar.addWidget(self.btn_sanitize)
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

        # RIGHT PANEL (INSPECTOR)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_widget = QWidget()
        right_widget.setFixedWidth(320)
        right_layout = QVBoxLayout(right_widget)
        right_scroll.setWidget(right_widget)

        # 1. Inspector
        grp_inspector = QGroupBox("Selected Image Tags")
        lyt_inspector = QVBoxLayout(grp_inspector)
        self.tag_editor = TagEditorWidget()
        self.tag_editor.tagsChanged.connect(self.sync_tags_from_inspector)
        lyt_inspector.addWidget(self.tag_editor)
        right_layout.addWidget(grp_inspector)

        # 2. Auto Tagger
        grp_auto = QGroupBox("🤖 Auto Tagger")
        lyt_auto = QVBoxLayout(grp_auto)
        self.btn_auto_tag = QPushButton("✨ Auto Tag Selected...")
        self.btn_auto_tag.clicked.connect(self.run_auto_tagger)
        self.btn_auto_tag.setStyleSheet("background-color: #6c5ce7; color: white; font-weight: bold; padding: 6px;")
        lyt_auto.addWidget(self.btn_auto_tag)
        right_layout.addWidget(grp_auto)

        # 3. Quick Tags
        right_layout.addSpacing(10)
        right_layout.addWidget(QLabel("<b>Quick Tags (Presets)</b>"))
        
        mode_layout = QHBoxLayout()
        self.rad_append = QRadioButton("Append")
        self.rad_prepend = QRadioButton("Prepend")
        self.rad_append.setChecked(True)
        mode_layout.addWidget(self.rad_append)
        mode_layout.addWidget(self.rad_prepend)
        right_layout.addLayout(mode_layout)

        tag_add_layout = QHBoxLayout()
        self.inp_new_tag = QLineEdit()
        self.inp_new_tag.setPlaceholderText("New preset...")
        self.inp_new_tag.returnPressed.connect(self.add_custom_tag)
        self.btn_add_tag = QPushButton("+")
        self.btn_add_tag.setFixedWidth(30)
        self.btn_add_tag.clicked.connect(self.add_custom_tag)
        tag_add_layout.addWidget(self.inp_new_tag)
        tag_add_layout.addWidget(self.btn_add_tag)
        right_layout.addLayout(tag_add_layout)

        self.tag_list = QListWidget()
        self.tag_list.itemClicked.connect(self.apply_tag_to_selection)
        self.tag_list.setFixedHeight(200) 
        self.load_tags() 
        right_layout.addWidget(self.tag_list)

        right_layout.addWidget(QLabel("<i>Click a preset to add to selection.</i>"))
        right_layout.addStretch()

        splitter.addWidget(left_widget)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)
        
        self.setup_hotkeys()
        self.pending_tag_updates = {}
        self.total_tag_jobs = 0
        self.auto_tag_settings = {}

    def setup_hotkeys(self):
        QShortcut(QKeySequence("Ctrl+A"), self).activated.connect(self.select_all)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.save_all)
        QShortcut(QKeySequence("Del"), self).activated.connect(self.delete_text_selection)
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self.undo_stack.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self.undo_stack.redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self).activated.connect(self.undo_stack.redo)
        QShortcut(QKeySequence("Ctrl+F"), self).activated.connect(self.inp_filter.setFocus)

    # --- FILTER LOGIC ---
    def apply_filter(self, text):
        search = text.lower().strip()
        for path, card in self.image_cards.items():
            # Filter by Filename OR Caption content
            match = (search in os.path.basename(path).lower()) or \
                    (search in card.txt_caption.toPlainText().lower())
            
            if not search: match = True
            card.setVisible(match)

    def select_all(self):
        """Modified to respect Filter (Visibility)"""
        # Get visible cards only
        visible_cards = [c for c in self.image_cards.values() if not c.isHidden()]
        
        if not visible_cards: return

        # If all visible are selected, deselect all visible. Otherwise select all visible.
        all_vis_selected = all(c.is_selected for c in visible_cards)
        target = not all_vis_selected
        
        for card in visible_cards:
            card.toggle_selection(target)

    # --- TAG EDITOR SYNC ---
    def on_card_selection(self, path, is_selected):
        if is_selected:
            self.selected_paths.add(path)
            self.image_selected.emit(path)
            self.active_card = self.image_cards[path]
            self.tag_editor.set_tags(self.active_card.txt_caption.toPlainText())
        else:
            self.selected_paths.discard(path)
            # If deselecting active card, verify if we should clear editor or switch
            if self.active_card and self.active_card.path == path:
                # Pick another selected card if available, else clear
                if self.selected_paths:
                    next_path = list(self.selected_paths)[-1]
                    self.active_card = self.image_cards[next_path]
                    self.tag_editor.set_tags(self.active_card.txt_caption.toPlainText())
                else:
                    self.active_card = None
                    self.tag_editor.set_tags("")

    def sync_tags_from_inspector(self, new_text):
        if self.active_card:
            old_text = self.active_card.txt_caption.toPlainText()
            if old_text != new_text:
                self.active_card.txt_caption.setPlainText(new_text)
                cmd = UpdateCaptionCommand(self.active_card, old_text, new_text)
                self.undo_stack.push(cmd)

    def handle_manual_text_change(self, path, old, new):
        if path in self.image_cards:
            cmd = UpdateCaptionCommand(self.image_cards[path], old, new)
            self.undo_stack.push(cmd)
            if self.active_card and self.active_card.path == path:
                self.tag_editor.set_tags(new)

    # --- AUTO TAGGER ---
    def run_auto_tagger(self):
        if not self.selected_paths:
            QMessageBox.warning(self, "No Selection", "Please select images to tag.")
            return

        dlg = AutoTagDialog(self)
        if dlg.exec():
            self.auto_tag_settings = dlg.get_settings()
            self.btn_auto_tag.setText("⏳ Tagging...")
            self.btn_auto_tag.setEnabled(False)
            self.pending_tag_updates = {}
            self.total_tag_jobs = len(self.selected_paths)

            for path in self.selected_paths:
                if path in self.image_cards:
                    card = self.image_cards[path]
                    worker = TaggerWorker(self.tagger, card, self.auto_tag_settings)
                    worker.signals.finished.connect(self.on_tagger_finished)
                    self.thread_pool.start(worker)

    def on_tagger_finished(self, card, tags):
        current_text = card.txt_caption.toPlainText().strip()
        settings = self.auto_tag_settings
        
        all_tags = settings['prepend'] + tags + settings['append']
        
        if settings['mode'] == 'overwrite':
            final_text = ", ".join(all_tags)
        elif settings['mode'] == 'ignore':
            if current_text: final_text = current_text
            else: final_text = ", ".join(all_tags)
        else: 
            if current_text:
                final_text = f"{current_text}, {', '.join(all_tags)}"
            else:
                final_text = ", ".join(all_tags)
        
        final_text = final_text.replace(", ,", ",").replace(" , ", ", ").strip(", ")

        self.pending_tag_updates[card] = final_text
        card.txt_caption.setPlainText(final_text)
        
        if self.active_card == card:
            self.tag_editor.set_tags(final_text)

        if len(self.pending_tag_updates) >= self.total_tag_jobs:
            self.finalize_auto_tagging()

    def finalize_auto_tagging(self):
        cards = list(self.pending_tag_updates.keys())
        new_texts = list(self.pending_tag_updates.values())
        cmd = BatchUpdateCommand(cards, new_texts, "Auto Tag (WD14)")
        self.undo_stack.push(cmd)
        self.btn_auto_tag.setText("✨ Auto Tag Selected...")
        self.btn_auto_tag.setEnabled(True)
        self.pending_tag_updates = {}

    # --- STANDARD LOGIC ---
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.current_folder = folder
            self.load_grid()

    def load_grid(self):
        self.undo_stack.clear()
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
            card.undo_req.connect(self.handle_manual_text_change)
            self.grid_layout.addWidget(card, i // cols, i % cols)
            self.image_cards[path] = card
            worker = ThumbnailWorker(path, (250, 200))
            worker.signals.loaded.connect(card.set_image)
            self.thread_pool.start(worker)

    def delete_text_selection(self):
        if self.tag_editor.inp_add.hasFocus() or self.inp_new_tag.hasFocus() or self.inp_filter.hasFocus(): return
        focus_widget = QApplication.focusWidget()
        if isinstance(focus_widget, QTextEdit): return

        if self.selected_paths:
            cards = []
            new_texts = []
            for path in self.selected_paths:
                if path in self.image_cards:
                    cards.append(self.image_cards[path])
                    new_texts.append("")
            
            if cards:
                cmd = BatchUpdateCommand(cards, new_texts, "Clear Captions")
                self.undo_stack.push(cmd)

    def sanitize_selection(self):
        if not self.selected_paths: return
        cards = []
        new_texts = []
        for path in self.selected_paths:
            if path in self.image_cards:
                card = self.image_cards[path]
                org = card.txt_caption.toPlainText()
                clean = unicodedata.normalize('NFKD', org).encode('ascii', 'ignore').decode('ascii')
                if org != clean:
                    cards.append(card)
                    new_texts.append(clean)
        
        if cards:
            cmd = BatchUpdateCommand(cards, new_texts, "Sanitize Text")
            self.undo_stack.push(cmd)
            QMessageBox.information(self, "Sanitized", f"Cleaned {len(cards)} captions.")

    def apply_tag_to_selection(self, item):
        tag = item.text()
        if not self.selected_paths:
            QMessageBox.warning(self, "No Selection", "Select images in the grid first.")
            return

        cards = []
        new_texts = []

        for path in self.selected_paths:
            card = self.image_cards.get(path)
            if card:
                current = card.txt_caption.toPlainText().strip()
                new_t = ""
                
                if not current:
                    new_t = tag
                else:
                    tags = [t.strip() for t in current.split(',')]
                    if tag in tags: continue # Skip if exists
                    
                    if self.rad_prepend.isChecked():
                        new_t = f"{tag}, {current}"
                    else:
                        new_t = f"{current}, {tag}"
                
                cards.append(card)
                new_texts.append(new_t)
        
        if cards:
            cmd = BatchUpdateCommand(cards, new_texts, f"Add Tag: {tag}")
            self.undo_stack.push(cmd)

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

    # --- TAG MANAGER LOGIC ---
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