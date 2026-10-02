import os
import unicodedata
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QPushButton,
    QLineEdit, QGridLayout, QTextEdit, QSplitter, QFileDialog, QMessageBox,
    QFrame, QListWidget, QProgressBar, QApplication, QInputDialog, QRadioButton,
    QGroupBox, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QThreadPool, QEvent, QTimer
from PySide6.QtGui import QPixmap, QShortcut, QKeySequence, QIcon, QUndoStack, QUndoCommand
from pathlib import Path

from core import caption_io, captions, dataset, fileops, paths
from core.config import load_quick_tags, save_quick_tags, settings
from core.widgets import TagEditorWidget, AutoTagDialog
from inference.base import CaptionRequest, ProviderSpec
from inference.worker import SAVE_NONE, BatchJob, start_job
from tabs.common import ThumbnailWorker, confirm, show_job_summary

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
    def __init__(self, cards, new_texts, description="Batch Update", old_texts=None):
        super().__init__(description)
        self.cards = cards
        self.new_texts = new_texts
        # Callers that already applied the change must pass the true previous texts.
        self.old_texts = old_texts if old_texts is not None else [c.txt_caption.toPlainText() for c in cards]

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
        content = caption_io.read_caption(self.path)
        self.txt_caption.setPlainText(content)
        self._cached_text = content
        self._saved_text = content

    @property
    def dirty(self) -> bool:
        return self.txt_caption.toPlainText() != self._saved_text

    def save_text(self) -> bool:
        """Write the caption if it changed (atomic, backed up). Raises OSError on failure."""
        text = self.txt_caption.toPlainText()
        if text == self._saved_text:
            return False
        caption_io.write_caption(self.path, text)
        self._saved_text = text
        return True

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
        self.thread_pool = QThreadPool(self)
        self.undo_stack = QUndoStack(self)
        self.tag_worker = None
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
        right_scroll.setMinimumWidth(342)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
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
        if self.tag_worker is not None:
            self.tag_worker.cancel()
            self.btn_auto_tag.setText("Stopping…")
            return
        if not self.selected_paths:
            QMessageBox.warning(self, "No Selection", "Please select images to tag.")
            return

        dlg = AutoTagDialog(self)
        if not dlg.exec():
            return
        st = self.auto_tag_settings = dlg.get_settings()
        fmt = dict(general_threshold=st["threshold"], character_threshold=st["character_threshold"],
                   max_tags=st["max_tags"], underscores_to_spaces=st["underscores_to_spaces"],
                   escape_parentheses=st["escape_parentheses"], include_rating=st["include_rating"],
                   blacklist=tuple(st["blacklist"]))
        spec = ProviderSpec.make("wd_tagger", st["model"], format=tuple(sorted(fmt.items())),
                                 device=settings().get("tagger.device"))
        ordered = [p for p in self.image_cards if p in self.selected_paths]
        if st["mode"] == "ignore":
            ordered = [p for p in ordered if not self.image_cards[p].txt_caption.toPlainText().strip()]
            if not ordered:
                QMessageBox.information(self, "Nothing to tag", "All selected images already have tags.")
                return
        # Tags come back in memory (SAVE_NONE) so the result is undoable; Save writes them.
        job = BatchJob(paths=[Path(p) for p in ordered], spec=spec,
                       request=CaptionRequest(prompt="", max_image_side=0),
                       save_mode=SAVE_NONE, tag_mode=True, title="Auto tagging")
        self.pending_tag_updates = {}
        self.total_tag_jobs = len(ordered)
        self.btn_auto_tag.setText(f"⏳ Tagging 0/{len(ordered)}…  (click to stop)")
        self.tag_worker = start_job(job)
        self.tag_worker.item_done.connect(self.on_tagger_finished)
        self.tag_worker.progress.connect(
            lambda d, t: self.btn_auto_tag.setText(f"⏳ Tagging {d}/{t}…  (click to stop)"))
        self.tag_worker.finished.connect(self.finalize_auto_tagging)

    def on_tagger_finished(self, path, tag_text):
        card = self.image_cards.get(path)
        if card is None:
            return
        st = self.auto_tag_settings
        current = card.txt_caption.toPlainText()
        final_text = captions.merge_generated_tags(current, captions.split_tags(tag_text), st["mode"],
                                                   prepend=st["prepend"], append=st["append"])
        if card not in self.pending_tag_updates:
            self.pending_tag_updates[card] = current  # remember the true original for undo
        card.txt_caption.setPlainText(final_text)
        if self.active_card == card:
            self.tag_editor.set_tags(final_text)

    def finalize_auto_tagging(self, summary=None):
        self.tag_worker = None
        cards = list(self.pending_tag_updates.keys())
        if cards:
            old_texts = list(self.pending_tag_updates.values())
            new_texts = [c.txt_caption.toPlainText() for c in cards]
            self.undo_stack.push(BatchUpdateCommand(cards, new_texts, "Auto Tag", old_texts=old_texts))
        self.btn_auto_tag.setText("✨ Auto Tag Selected...")
        self.pending_tag_updates = {}
        if summary is not None:
            show_job_summary(self, summary)
            if cards and not summary.fatal:
                self.btn_auto_tag.setToolTip(f"{summary.text()} — remember to Save (Ctrl+S)")

    # --- STANDARD LOGIC ---
    def has_unsaved_changes(self) -> bool:
        return any(c.dirty for c in self.image_cards.values())

    def select_folder(self):
        if self.has_unsaved_changes() and not confirm(
                self, "Unsaved changes", "You have unsaved caption changes in this folder.\n"
                "Discard them and open another folder?", destructive=True):
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", settings().get("ui.last_folder"))
        if folder:
            settings().set("ui.last_folder", folder)
            self.current_folder = folder
            self.load_grid()

    def load_grid(self):
        self.undo_stack.clear()
        self.thread_pool.clear()
        while self.grid_layout.count():
            w = self.grid_layout.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.image_cards.clear()
        self.selected_paths.clear()
        self.active_card = None
        self.tag_editor.set_tags("")

        try:
            files = dataset.scan_images(self.current_folder)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read directory: {e}")
            return

        cols = 4
        for i, f in enumerate(files):
            path = str(f)
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
        saved, failed = 0, []
        for card in self.image_cards.values():
            try:
                if card.save_text():
                    saved += 1
            except OSError as e:
                failed.append(f"{os.path.basename(card.path)}: {e}")
        if failed:
            QMessageBox.critical(self, "Some captions were not saved", "\n".join(failed[:20]))
        self.btn_save_all.setText(f"💾 Saved {saved}" if saved else "💾 No changes")
        QTimer.singleShot(2500, lambda: self.btn_save_all.setText("💾 Save"))

    def save_to_dataset(self):
        if not self.selected_paths:
            QMessageBox.warning(self, "No Selection", "Please select images.")
            return
        root = Path(settings().get("paths.collections_dir") or paths.DEFAULT_COLLECTIONS_DIR)
        existing = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
        name, ok = QInputDialog.getItem(self, "Copy to Collection", "Collection name (new or existing):",
                                        existing, 0, True)
        if not ok or not name.strip():
            return
        report = fileops.CopyReport()
        for path in sorted(self.selected_paths):
            card = self.image_cards.get(path)
            try:
                if card:
                    card.save_text()  # copy what the user sees
            except OSError as e:
                report.failed.append((Path(path), f"caption not saved: {e}"))
                continue
            fileops.copy_image_with_caption(Path(path), root / name.strip(), report=report)
        msg = f"Collection '{name.strip()}': {report.summary()}."
        if report.failed:
            msg += "\n\n" + "\n".join(f"{p.name}: {e}" for p, e in report.failed[:20])
            QMessageBox.warning(self, "Copy to Collection", msg)
        else:
            QMessageBox.information(self, "Copy to Collection", msg)

    # --- TAG MANAGER LOGIC ---
    def load_tags(self):
        self.tag_list.clear()
        self.tag_list.addItems(load_quick_tags())

    def add_custom_tag(self):
        tag = self.inp_new_tag.text().strip()
        if tag:
            items = [self.tag_list.item(i).text() for i in range(self.tag_list.count())]
            if tag not in items:
                self.tag_list.addItem(tag)
                try:
                    save_quick_tags(items + [tag])
                except OSError as e:
                    QMessageBox.warning(self, "Quick Tags", f"Could not save quick tags: {e}")
            self.inp_new_tag.clear()
