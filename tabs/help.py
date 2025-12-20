from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, 
    QHeaderView, QPushButton, QTabWidget, QTextBrowser, QWidget
)
from PySide6.QtCore import Qt

class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TagScribeR - Help Center")
        self.resize(800, 600)
        
        layout = QVBoxLayout(self)
        
        # Tabs
        self.tabs = QTabWidget()
        
        # Tab 1: Hotkeys
        self.tab_hotkeys = QWidget()
        self.init_hotkeys(self.tab_hotkeys)
        
        # Tab 2: Walkthrough
        self.tab_guide = QWidget()
        self.init_guide(self.tab_guide)
        
        self.tabs.addTab(self.tab_hotkeys, "⌨️ Keyboard Shortcuts")
        self.tabs.addTab(self.tab_guide, "📖 User Manual / Walkthrough")
        
        layout.addWidget(self.tabs)
        
        # Close Button
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

    def init_hotkeys(self, parent):
        layout = QVBoxLayout(parent)
        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Scope", "Shortcut", "Action"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.setAlternatingRowColors(True)
        
        hotkeys = [
            ("Global", "Ctrl+1 - Ctrl+6", "Switch Tabs"),
            ("Global", "F1", "Show this Help Center"),
            ("Gallery", "Ctrl+A", "Toggle Select All / None"),
            ("Gallery", "Del", "Clear Caption Text (Selected Images)"),
            ("Gallery", "Ctrl+S", "Save All Changes to Disk"),
            ("Auto Caption", "Ctrl+A", "Toggle Select All"),
            ("Auto Caption", "Ctrl+Enter", "Run Captioning"),
            ("Auto Caption", "Esc", "Abort Processing"),
            ("Image Editor", "Ctrl+A", "Toggle Select All"),
            ("Image Editor", "Ctrl+R", "Rotate CW"),
            ("Image Editor", "Ctrl+Shift+R", "Rotate CCW"),
            ("Datasets", "Ctrl+A", "Toggle Select All"),
            ("Datasets", "Ctrl+F", "Focus Filter Bar"),
            ("Datasets", "Ctrl+N", "New Collection"),
            ("Datasets", "Ctrl+Enter", "Add Selected to Collection"),
            ("Datasets", "F5", "Refresh Collections"),
            ("Datasets", "Del", "Delete Selected Images"),
        ]
        
        table.setRowCount(len(hotkeys))
        for i, (scope, key, action) in enumerate(hotkeys):
            table.setItem(i, 0, QTableWidgetItem(scope))
            table.setItem(i, 1, QTableWidgetItem(key))
            table.setItem(i, 2, QTableWidgetItem(action))
            
        layout.addWidget(table)

    def init_guide(self, parent):
        layout = QVBoxLayout(parent)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        
        # HTML Guide Content
        html = """
        <style>
            h2 { color: #00b894; margin-top: 20px; }
            h3 { color: #74b9ff; margin-top: 10px; }
            p { font-size: 14px; line-height: 1.4; }
            li { font-size: 14px; margin-bottom: 5px; }
            .warning { color: #ff7675; font-weight: bold; }
        </style>
        
        <h1>Welcome to TagScribeR v2.1</h1>
        <p>TagScribeR is a professional studio for managing, captioning, and editing image datasets for AI training (LoRA/Checkpoints). Below is a walkthrough of each workspace.</p>
        
        <hr>
        
        <h2>🖼️ Gallery Tab</h2>
        <p>This is your primary workspace for reviewing and manually tagging images.</p>
        <ul>
            <li><b>Loading:</b> Click "Open Folder" to load a directory of images.</li>
            <li><b>Selection:</b> Click images to select/deselect them (Green border = Selected). Use <b>Ctrl+A</b> to toggle all.</li>
            <li><b>Captioning:</b> Type directly into the text box under an image. Changes are in memory until saved.</li>
            <li><b>Quick Tags:</b> Click a tag in the sidebar to append it to ALL selected images. Use the "Prepend/Append" toggle to decide if tags go at the start (triggers) or end.</li>
            <li><b>Saving:</b> Click "Save Changes" (Ctrl+S) to write .txt files to disk.</li>
        </ul>

        <h2>🤖 Auto Caption Tab</h2>
        <p>Use AI to automatically describe your images using Qwen 3-VL or external APIs.</p>
        <h3>Local Mode (GPU)</h3>
        <p>Runs Qwen directly on your hardware. Supports NVIDIA (CUDA) and AMD (ROCm).</p>
        <ul>
            <li>Select a model from the dropdown. If missing, click <b>Download</b>.</li>
            <li>Adjust <b>Max Tokens</b> (length) and <b>Temperature</b> (creativity).</li>
            <li>Click <b>Caption Selected</b> to start. Images process one by one to save VRAM.</li>
        </ul>
        <h3>API Mode (LM Studio / OpenAI)</h3>
        <p>Connects to local or cloud APIs.</p>
        <ul>
            <li>Enter your Base URL (e.g., <code>http://localhost:1234/v1</code>) and Key.</li>
            <li>Save your configuration using the "Floppy Disk" icon in the API tab.</li>
        </ul>

        <h2>✏️ Image Editor Tab</h2>
        <p>Batch process images for training prep.</p>
        <ul>
            <li><b>Output:</b> Choose "Save to Image Edits" (Safe) or "Overwrite Originals" (Destructive).</li>
            <li><b>Resize:</b> "Scale Longest Side" keeps aspect ratio (e.g., 1024px max). "Force Dimensions" stretches the image.</li>
            <li><b>Crop:</b> Smart cropping with focus points (e.g., Top-Center helps preserve heads in portrait crops).</li>
            <li><b>Convert:</b> Batch convert to JPG/PNG/WEBP with quality control.</li>
        </ul>

        <h2>📂 Datasets Tab</h2>
        <p>Organize images into training sets without moving files manually.</p>
        <ul>
            <li><b>Collections:</b> Create named folders (e.g., "Style_A", "Concept_B").</li>
            <li><b>Filtering:</b> Load a massive source folder, then type tags into the filter bar (e.g., "1girl") to see only matching images.</li>
            <li><b>Add to Collection:</b> Select filtered images and click "Add". This <b>copies</b> the image and its caption text file to the collection folder.</li>
        </ul>

        <h2>ℹ️ Metadata Tab</h2>
        <p>View and sanitize image data.</p>
        <ul>
            <li><b>Prompt Reader:</b> Automatically extracts Stable Diffusion Generation data from PNG chunks or JPG UserComments.</li>
            <li><b>Strip:</b> The "Strip All Metadata" button removes all EXIF/PNG info for privacy.</li>
        </ul>
        """
        
        browser.setHtml(html)
        layout.addWidget(browser)