from PySide6.QtWidgets import (
    QWidget, QLayout, QSizePolicy, QLabel, QPushButton, QHBoxLayout, 
    QVBoxLayout, QFrame, QLineEdit, QDialog, QComboBox, QSpinBox, 
    QDoubleSpinBox, QGroupBox, QFormLayout, QDialogButtonBox, QRadioButton,
    QButtonGroup
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto Label Settings")
        self.resize(400, 500)
        
        layout = QVBoxLayout(self)
        
        # 1. Mode
        grp_mode = QGroupBox("Existing Tags")
        vbox_mode = QVBoxLayout(grp_mode)
        self.bg_mode = QButtonGroup()
        
        self.rad_ignore = QRadioButton("Ignore (Skip if exists)")
        self.rad_append = QRadioButton("Append (Add to end)")
        self.rad_overwrite = QRadioButton("Overwrite (Replace all)")
        self.rad_append.setChecked(True)
        
        self.bg_mode.addButton(self.rad_ignore, 0)
        self.bg_mode.addButton(self.rad_append, 1)
        self.bg_mode.addButton(self.rad_overwrite, 2)
        
        vbox_mode.addWidget(self.rad_ignore)
        vbox_mode.addWidget(self.rad_append)
        vbox_mode.addWidget(self.rad_overwrite)
        layout.addWidget(grp_mode)
        
        # 2. Settings
        form = QFormLayout()
        
        self.spin_max = QSpinBox()
        self.spin_max.setRange(1, 100)
        self.spin_max.setValue(20)
        
        self.spin_thresh = QDoubleSpinBox()
        self.spin_thresh.setRange(0.01, 1.0)
        self.spin_thresh.setSingleStep(0.05)
        self.spin_thresh.setValue(0.35)
        
        self.line_blacklist = QLineEdit()
        self.line_blacklist.setPlaceholderText("tag1, tag2 (comma separated)")
        
        self.line_prepend = QLineEdit()
        self.line_prepend.setPlaceholderText("Tags to force at start...")
        
        self.line_append = QLineEdit()
        self.line_append.setPlaceholderText("Tags to force at end...")
        
        form.addRow("Max Tags:", self.spin_max)
        form.addRow("Min Threshold:", self.spin_thresh)
        form.addRow("Blacklist:", self.line_blacklist)
        form.addRow("Force Prepend:", self.line_prepend)
        form.addRow("Force Append:", self.line_append)
        
        layout.addLayout(form)
        
        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_settings(self):
        mode_id = self.bg_mode.checkedId()
        mode_str = "append"
        if mode_id == 0: mode_str = "ignore"
        elif mode_id == 2: mode_str = "overwrite"
        
        return {
            "mode": mode_str,
            "max_tags": self.spin_max.value(),
            "threshold": self.spin_thresh.value(),
            "blacklist": [t.strip() for t in self.line_blacklist.text().split(',') if t.strip()],
            "prepend": [t.strip() for t in self.line_prepend.text().split(',') if t.strip()],
            "append": [t.strip() for t in self.line_append.text().split(',') if t.strip()]
        }