import os
import piexif
from PIL import Image, PngImagePlugin
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit, 
    QSplitter, QFileDialog, QMessageBox, QGroupBox, QListWidget,
    QAbstractItemView, QFrame
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from core.image_utils import load_thumbnail

# Common tags to map to the "Quick Editor"
PROMPT_KEYS = ["parameters", "UserComment", "ImageDescription", "Description", "Comment"]

class MetadataTab(QWidget):
    def __init__(self):
        super().__init__()
        self.current_folder = ""
        self.current_image_path = None
        self.exif_dict = {}     # For JPG
        self.png_info = {}      # For PNG
        self.is_png = False
        
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # --- LEFT: File List ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        self.btn_folder = QPushButton("📂 Open Folder")
        self.btn_folder.clicked.connect(self.select_folder)
        self.btn_folder.setStyleSheet("background-color: #00b894; color: white; font-weight: bold;")
        
        self.list_files = QListWidget()
        self.list_files.currentItemChanged.connect(self.on_file_selected)
        
        left_layout.addWidget(self.btn_folder)
        left_layout.addWidget(self.list_files)

        # --- CENTER: Preview & Quick Edit ---
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        
        # Image Preview
        self.lbl_image = QLabel("No Image Selected")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("background-color: #1e1e1e; border-radius: 8px;")
        self.lbl_image.setMinimumHeight(300)
        
        # Generation Parameters (Prompt) Editor
        grp_prompt = QGroupBox("Generation Parameters / User Comment")
        lyt_prompt = QVBoxLayout(grp_prompt)
        self.txt_prompt = QTextEdit()
        self.txt_prompt.setPlaceholderText("Stable Diffusion generation parameters usually appear here...")
        lyt_prompt.addWidget(self.txt_prompt)
        
        center_layout.addWidget(self.lbl_image)
        center_layout.addWidget(grp_prompt)
        center_layout.setStretch(0, 1) # Image takes available space
        center_layout.setStretch(1, 1) # Text takes available space

        # --- RIGHT: Raw Data & Actions ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Raw Table
        grp_raw = QGroupBox("Raw Metadata Tags")
        lyt_raw = QVBoxLayout(grp_raw)
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Key / Tag ID", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        lyt_raw.addWidget(self.table)
        
        # Actions
        btn_layout = QVBoxLayout()
        self.btn_save = QPushButton("💾 Save Changes")
        self.btn_save.clicked.connect(self.save_metadata)
        self.btn_save.setStyleSheet("background-color: #0984e3; color: white; font-weight: bold; padding: 10px;")
        
        self.btn_strip = QPushButton("🗑️ Strip All Metadata")
        self.btn_strip.clicked.connect(self.strip_metadata)
        self.btn_strip.setStyleSheet("background-color: #d63031; color: white;")
        
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_strip)
        
        right_layout.addWidget(grp_raw)
        right_layout.addLayout(btn_layout)

        # Add to Splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(center_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 2)
        
        layout.addWidget(splitter)

    # --- LOADING LOGIC ---
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.current_folder = folder
            self.load_file_list()

    def load_file_list(self):
        self.list_files.clear()
        exts = ('.jpg', '.jpeg', '.png', '.webp', '.tiff')
        files = [f for f in os.listdir(self.current_folder) if f.lower().endswith(exts)]
        self.list_files.addItems(files)

    def on_file_selected(self, item, prev):
        if not item: return
        self.current_image_path = os.path.join(self.current_folder, item.text())
        self.load_image_data()

    def load_metadata(self, path):
        # Slot for external signals (e.g. from Gallery)
        self.current_image_path = path
        # Update list selection visually if possible
        if self.current_folder and os.path.dirname(path) == self.current_folder:
            items = self.list_files.findItems(os.path.basename(path), Qt.MatchExactly)
            if items: self.list_files.setCurrentItem(items[0])
        else:
            # If from different folder, just load data directly
            self.load_image_data()

    def load_image_data(self):
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            return

        # 1. Load Preview
        pix = load_thumbnail(self.current_image_path, (500, 500))
        self.lbl_image.setPixmap(pix)
        
        # 2. Reset Data
        self.table.setRowCount(0)
        self.txt_prompt.clear()
        self.exif_dict = {}
        self.png_info = {}
        self.is_png = self.current_image_path.lower().endswith('.png')

        try:
            if self.is_png:
                self.load_png_metadata()
            else:
                self.load_jpg_metadata()
        except Exception as e:
            print(f"Metadata load error: {e}")

    # --- FORMAT SPECIFIC LOADERS ---
    def load_png_metadata(self):
        img = Image.open(self.current_image_path)
        img.load() # Force load to get info
        self.png_info = img.info
        
        row = 0
        for key, value in self.png_info.items():
            # Filter out binary or unreadable keys if necessary
            str_val = str(value)
            
            # Populate Prompt Box if it matches
            if key in PROMPT_KEYS:
                self.txt_prompt.setText(str_val)
            
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(key))
            self.table.setItem(row, 1, QTableWidgetItem(str_val))
            row += 1

    def load_jpg_metadata(self):
        try:
            self.exif_dict = piexif.load(self.current_image_path)
        except:
            return # No EXIF or error reading it

        row = 0
        # Iterate over IFDs (0th, Exif, GPS, 1st)
        for ifd in ("0th", "Exif", "GPS", "1st"):
            if ifd in self.exif_dict:
                for tag, value in self.exif_dict[ifd].items():
                    tag_name = piexif.TAGS[ifd].get(tag, {"name": str(tag)})["name"]
                    
                    # Decode Bytes
                    if isinstance(value, bytes):
                        try:
                            # Try generic decode
                            str_val = value.decode('utf-8').strip('\x00')
                            # Handle EXIF UserComment specifically (often has 'UNICODE' or 'ASCII' header)
                            if tag_name == "UserComment":
                                if value.startswith(b'UNICODE'):
                                    str_val = value[8:].decode('utf-16').strip('\x00')
                                elif value.startswith(b'ASCII'):
                                    str_val = value[5:].decode('ascii').strip('\x00')
                        except:
                            str_val = "<Binary Data>"
                    else:
                        str_val = str(value)

                    # Populate Prompt Box
                    if tag_name in PROMPT_KEYS:
                        self.txt_prompt.setText(str_val)

                    self.table.insertRow(row)
                    # Store IFD and Tag ID in UserRole for saving later
                    key_item = QTableWidgetItem(f"{ifd} - {tag_name}")
                    key_item.setData(Qt.UserRole, (ifd, tag)) 
                    self.table.setItem(row, 0, key_item)
                    self.table.setItem(row, 1, QTableWidgetItem(str_val))
                    row += 1

    # --- SAVING LOGIC ---
    def save_metadata(self):
        if not self.current_image_path: return
        
        try:
            if self.is_png:
                self.save_png()
            else:
                self.save_jpg()
            QMessageBox.information(self, "Saved", "Metadata updated successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")

    def save_png(self):
        # 1. Create new PngInfo object
        target_info = PngImagePlugin.PngInfo()
        
        # 2. Add data from the Table
        # Note: We prioritize the Prompt Box for the 'parameters' key if it exists in table
        prompt_text = self.txt_prompt.toPlainText()
        prompt_key_found = False
        
        for row in range(self.table.rowCount()):
            key = self.table.item(row, 0).text()
            val = self.table.item(row, 1).text()
            
            # If this key is the prompt key, use the text box value instead of table value
            if key in PROMPT_KEYS:
                target_info.add_text(key, prompt_text)
                prompt_key_found = True
            else:
                target_info.add_text(key, val)
        
        # If no prompt key was in table but text box has text, add it as 'parameters'
        if not prompt_key_found and prompt_text:
            target_info.add_text("parameters", prompt_text)

        # 3. Save (Re-saving PNG is required to update chunks)
        # We assume the user is okay with re-saving. PNG is lossless so quality is preserved.
        img = Image.open(self.current_image_path)
        img.save(self.current_image_path, pnginfo=target_info)

    def save_jpg(self):
        # 1. Update self.exif_dict from Table
        prompt_text = self.txt_prompt.toPlainText()
        
        for row in range(self.table.rowCount()):
            key_item = self.table.item(row, 0)
            val_str = self.table.item(row, 1).text()
            
            tag_data = key_item.data(Qt.UserRole) # (ifd, tag_id)
            
            if tag_data:
                ifd, tag_id = tag_data
                
                # Check if this is UserComment (Prompt)
                tag_name = piexif.TAGS[ifd].get(tag_id, {}).get("name", "")
                
                if tag_name == "UserComment":
                    # Use the Prompt Box value
                    # EXIF UserComment requires specific encoding
                    # We'll use UNICODE header for compatibility
                    payload = b'UNICODE\x00' + prompt_text.encode('utf-16le')
                    self.exif_dict[ifd][tag_id] = payload
                else:
                    # Attempt to convert string back to int/bytes if needed
                    # (Simplified: For robust generic EXIF editing, we assume strings for now
                    # or skip complex binary types that shouldn't be hand-edited)
                    try:
                        # If original was int, try converting
                        original_type = type(self.exif_dict[ifd][tag_id])
                        if original_type == int:
                            self.exif_dict[ifd][tag_id] = int(val_str)
                        elif original_type == bytes:
                            self.exif_dict[ifd][tag_id] = val_str.encode('utf-8')
                        else:
                            self.exif_dict[ifd][tag_id] = val_str
                    except:
                        pass # Keep original if conversion fails

        # 2. Dump and Insert (Lossless operation)
        exif_bytes = piexif.dump(self.exif_dict)
        piexif.insert(exif_bytes, self.current_image_path)

    def strip_metadata(self):
        if not self.current_image_path: return
        
        reply = QMessageBox.question(self, "Strip Metadata", 
                                     "Are you sure you want to remove ALL metadata? This cannot be undone.",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.No: return

        try:
            img = Image.open(self.current_image_path)
            
            # To strip PNG, we save a new image with no info
            # To strip JPG, we use piexif.remove
            
            if self.is_png:
                data = list(img.getdata())
                clean_img = Image.new(img.mode, img.size)
                clean_img.putdata(data)
                clean_img.save(self.current_image_path)
            else:
                piexif.remove(self.current_image_path)
            
            QMessageBox.information(self, "Success", "Metadata stripped.")
            self.load_image_data() # Reload to show empty table
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to strip: {e}")