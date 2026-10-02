from PySide6.QtWidgets import (
    QWidget, QLayout, QSizePolicy, QLabel, QPushButton, QHBoxLayout, 
    QVBoxLayout, QFrame, QLineEdit, QDialog, QComboBox, QSpinBox, 
    QDoubleSpinBox, QGroupBox, QFormLayout, QDialogButtonBox, QRadioButton,
    QButtonGroup, QCheckBox
)
from PySide6.QtCore import Qt, QRect, QSize, QPoint, Signal

# --- CUSTOM FLOW LAYOUT (For wrapping bubbles) ---
class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=0, hSpacing=5, vSpacing=5):
        super().__init__(parent)
        self._hSpace = hSpacing
        self._vSpace = vSpacing
        self._items = []
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item): self._items.append(item)
    def count(self): return len(self._items)
    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None
    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self): return Qt.Orientations(0)
    def hasHeightForWidth(self): return True
    def heightForWidth(self, width):
        return self.doLayout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self.doLayout(rect, False)

    def sizeHint(self): return self.minimumSize()
    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return size + QSize(2 * self.contentsMargins().top(), 2 * self.contentsMargins().top())

    def doLayout(self, rect, testOnly):
        x, y = rect.x(), rect.y()
        lineHeight = 0
        
        for item in self._items:
            wid = item.widget()
            spaceX = self._hSpace
            spaceY = self._vSpace
            nextX = x + item.sizeHint().width() + spaceX
            
            if nextX - spaceX > rect.right() and lineHeight > 0:
                x = rect.x()
                y = y + lineHeight + spaceY
                nextX = x + item.sizeHint().width() + spaceX
                lineHeight = 0

            if not testOnly:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = nextX
            lineHeight = max(lineHeight, item.sizeHint().height())

        return y + lineHeight - rect.y()

# --- TAG BUBBLE ---
class TagBubble(QFrame):
    deleteRequested = Signal(str) # Emits tag text when X is clicked

    def __init__(self, text):
        super().__init__()
        self.text = text
        self.setStyleSheet("""
            TagBubble {
                background-color: #2b2b2b;
                border: 1px solid #3c3c3c;
                border-radius: 12px;
                padding: 2px;
            }
            QLabel { color: #dddddd; font-weight: bold; padding-left: 5px; }
            QPushButton {
                background-color: transparent;
                color: #888;
                border: none;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #d63031; color: white; }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(5)
        
        lbl = QLabel(text)
        btn = QPushButton("×")
        btn.setFixedSize(16, 16)
        btn.clicked.connect(lambda: self.deleteRequested.emit(self.text))
        
        layout.addWidget(lbl)
        layout.addWidget(btn)

# --- TAG EDITOR WIDGET ---
class TagEditorWidget(QWidget):
    tagsChanged = Signal(str) # Emits comma-separated string

    def __init__(self):
        super().__init__()
        self.tags = []
        
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0,0,0,0)
        
        # Flow Container
        self.flow_widget = QWidget()
        self.flow_layout = FlowLayout(self.flow_widget)
        self.main_layout.addWidget(self.flow_widget)
        
        # Input Field
        self.inp_add = QLineEdit()
        self.inp_add.setPlaceholderText("Add tag (Press Enter)...")
        self.inp_add.returnPressed.connect(self.add_from_input)
        self.main_layout.addWidget(self.inp_add)

    def set_tags(self, tag_string):
        """Populates bubbles from 'tag1, tag2' string"""
        # Clear layout
        while self.flow_layout.count():
            item = self.flow_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
            
        self.tags = [t.strip() for t in tag_string.split(',') if t.strip()]
        
        for tag in self.tags:
            bubble = TagBubble(tag)
            bubble.deleteRequested.connect(self.remove_tag)
            self.flow_layout.addWidget(bubble)
        
        # Force redraw logic
        self.flow_widget.updateGeometry()

    def add_from_input(self):
        txt = self.inp_add.text().strip()
        if txt and txt not in self.tags:
            self.tags.append(txt)
            self.emit_change()
            # Refresh view
            self.set_tags(", ".join(self.tags))
        self.inp_add.clear()

    def remove_tag(self, tag_text):
        if tag_text in self.tags:
            self.tags.remove(tag_text)
            self.emit_change()
            self.set_tags(", ".join(self.tags))

    def emit_change(self):
        self.tagsChanged.emit(", ".join(self.tags))

# --- ADVANCED AUTO TAG DIALOG ---
class AutoTagDialog(QDialog):
    """Options for WD booru tagging. Values persist in settings between runs."""

    MODES = (("ignore", "Skip images that already have tags"),
             ("append", "Append new tags (no duplicates)"),
             ("overwrite", "Replace existing tags"))

    def __init__(self, parent=None):
        super().__init__(parent)
        from core.config import settings
        from inference.wd_tagger import KNOWN_TAGGERS
        self.cfg = settings()
        self.setWindowTitle("Auto Tag Settings")
        self.resize(460, 520)
        layout = QVBoxLayout(self)

        form_model = QFormLayout()
        self.combo_model = QComboBox()
        for repo, label in KNOWN_TAGGERS.items():
            self.combo_model.addItem(label, repo)
        idx = self.combo_model.findData(self.cfg.get("tagger.model"))
        self.combo_model.setCurrentIndex(max(0, idx))
        self.combo_model.setToolTip("Downloaded automatically on first use (~0.3-1.2 GB).")
        form_model.addRow("Tagger model:", self.combo_model)
        layout.addLayout(form_model)

        grp_mode = QGroupBox("Existing Tags")
        vbox_mode = QVBoxLayout(grp_mode)
        self.bg_mode = QButtonGroup(self)
        saved_mode = self.cfg.get("tagger.mode", "append")
        for i, (key, label) in enumerate(self.MODES):
            rb = QRadioButton(label)
            rb.setChecked(key == saved_mode)
            self.bg_mode.addButton(rb, i)
            vbox_mode.addWidget(rb)
        if self.bg_mode.checkedId() < 0:
            self.bg_mode.button(1).setChecked(True)
        layout.addWidget(grp_mode)

        form = QFormLayout()
        self.spin_max = QSpinBox()
        self.spin_max.setRange(1, 200)
        self.spin_max.setValue(self.cfg.get("tagger.max_tags"))
        self.spin_thresh = QDoubleSpinBox()
        self.spin_thresh.setRange(0.05, 1.0)
        self.spin_thresh.setSingleStep(0.05)
        self.spin_thresh.setValue(self.cfg.get("tagger.general_threshold"))
        self.spin_thresh.setToolTip("Minimum confidence for general tags. Lower = more tags, more mistakes.")
        self.spin_char = QDoubleSpinBox()
        self.spin_char.setRange(0.05, 1.0)
        self.spin_char.setSingleStep(0.05)
        self.spin_char.setValue(self.cfg.get("tagger.character_threshold"))
        self.spin_char.setToolTip("Minimum confidence for named-character tags (kept high to avoid false names).")
        self.line_blacklist = QLineEdit(self.cfg.get("tagger.blacklist"))
        self.line_blacklist.setPlaceholderText("tag1, tag2 (comma separated)")
        self.line_prepend = QLineEdit(self.cfg.get("tagger.prepend", ""))
        self.line_prepend.setPlaceholderText("e.g. your trigger word — always first")
        self.line_append = QLineEdit(self.cfg.get("tagger.append", ""))
        self.line_append.setPlaceholderText("Tags to force at the end")
        self.chk_underscores = QCheckBox("Replace underscores with spaces")
        self.chk_underscores.setChecked(self.cfg.get("tagger.underscores_to_spaces"))
        self.chk_escape = QCheckBox(r"Escape parentheses  ( → \( )")
        self.chk_escape.setChecked(self.cfg.get("tagger.escape_parentheses"))
        self.chk_rating = QCheckBox("Include rating tag (general / sensitive / ...)")
        self.chk_rating.setChecked(self.cfg.get("tagger.include_rating"))
        form.addRow("Max tags:", self.spin_max)
        form.addRow("General threshold:", self.spin_thresh)
        form.addRow("Character threshold:", self.spin_char)
        form.addRow("Blacklist:", self.line_blacklist)
        form.addRow("Force prepend:", self.line_prepend)
        form.addRow("Force append:", self.line_append)
        form.addRow(self.chk_underscores)
        form.addRow(self.chk_escape)
        form.addRow(self.chk_rating)
        layout.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    @staticmethod
    def _split(text):
        return [t.strip() for t in text.split(',') if t.strip()]

    def get_settings(self):
        mode = self.MODES[max(0, self.bg_mode.checkedId())][0]
        self.cfg.update({
            "tagger.model": self.combo_model.currentData(),
            "tagger.mode": mode,
            "tagger.max_tags": self.spin_max.value(),
            "tagger.general_threshold": self.spin_thresh.value(),
            "tagger.character_threshold": self.spin_char.value(),
            "tagger.blacklist": self.line_blacklist.text(),
            "tagger.prepend": self.line_prepend.text(),
            "tagger.append": self.line_append.text(),
            "tagger.underscores_to_spaces": self.chk_underscores.isChecked(),
            "tagger.escape_parentheses": self.chk_escape.isChecked(),
            "tagger.include_rating": self.chk_rating.isChecked(),
        })
        return {
            "model": self.combo_model.currentData(),
            "mode": mode,
            "max_tags": self.spin_max.value(),
            "threshold": self.spin_thresh.value(),
            "character_threshold": self.spin_char.value(),
            "blacklist": self._split(self.line_blacklist.text()),
            "prepend": self._split(self.line_prepend.text()),
            "append": self._split(self.line_append.text()),
            "underscores_to_spaces": self.chk_underscores.isChecked(),
            "escape_parentheses": self.chk_escape.isChecked(),
            "include_rating": self.chk_rating.isChecked(),
        }
