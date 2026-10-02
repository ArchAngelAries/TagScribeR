import ctypes
import logging
import os
import sys

# Must run before anything imports torch: sets ROCm / allocator env defaults.
from core import hardware
from core.log import setup_logging

hardware.apply_runtime_env()
setup_logging()
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from PySide6.QtGui import QIcon, QKeySequence, QShortcut  # noqa: E402
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QListWidget, QMainWindow, QMessageBox,  # noqa: E402
                               QPushButton, QStackedWidget, QVBoxLayout, QWidget)

from core import paths  # noqa: E402
from core.config import settings  # noqa: E402
from tabs import CaptionTab, DatasetsTab, EditorTab, GalleryTab, HelpDialog, MetadataTab, SettingsTab  # noqa: E402
from tabs.settings import apply_theme  # noqa: E402

log = logging.getLogger("tagscriber")
APP_ID = "ArchAngelAries.TagScribeR"
APP_TITLE = "TagScribeR"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1600, 900)
        icon = paths.resource("logo.ico") or paths.resource("logo.png")
        if icon:
            self.setWindowIcon(QIcon(str(icon)))

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        sidebar_container = QWidget()
        sidebar_container.setFixedWidth(200)
        sidebar_container.setStyleSheet("background-color: #232323;")
        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        self.sidebar = QListWidget()
        for label in ("🖼️ Gallery", "🤖 Auto Caption", "✏️ Image Editor", "📂 Datasets", "ℹ️ Metadata", "⚙️ Settings"):
            self.sidebar.addItem(label)
        self.sidebar.currentRowChanged.connect(self.change_tab)
        self.sidebar.setStyleSheet("""
            QListWidget { border: none; background-color: #232323; font-size: 15px; }
            QListWidget::item { padding: 15px; border-bottom: 1px solid #2c2c2c; }
            QListWidget::item:selected { background-color: #00b894; color: white; }
        """)
        btn_help = QPushButton("❓ Help / Manual")
        btn_help.setStyleSheet("""
            QPushButton { background-color: #2d3436; color: #aaa; border: none; padding: 15px; text-align: left; }
            QPushButton:hover { background-color: #333; color: white; }
        """)
        btn_help.setToolTip("Help Center — guides for every tool (F1 opens help for the current tab)")
        btn_help.clicked.connect(lambda: self.show_help("start"))
        sidebar_layout.addWidget(self.sidebar)
        sidebar_layout.addWidget(btn_help)

        self.stack = QStackedWidget()
        self.tab_gallery = GalleryTab()
        self.tab_caption = CaptionTab()
        self.tab_editor = EditorTab()
        self.tab_datasets = DatasetsTab()
        self.tab_metadata = MetadataTab()
        self.tab_settings = SettingsTab()
        self.tab_gallery.image_selected.connect(self.tab_metadata.load_metadata)
        for tab in (self.tab_gallery, self.tab_caption, self.tab_editor, self.tab_datasets,
                    self.tab_metadata, self.tab_settings):
            self.stack.addWidget(tab)

        main_layout.addWidget(sidebar_container)
        main_layout.addWidget(self.stack)
        self.sidebar.setCurrentRow(0)
        self.setup_hotkeys()

    def change_tab(self, index):
        self.stack.setCurrentIndex(index)

    def show_help(self, topic: str | None = None):
        from tabs.help_content import TAB_TOPICS
        HelpDialog.open_topic(self, topic or TAB_TOPICS[self.stack.currentIndex()])

    def setup_hotkeys(self):
        for i in range(6):
            QShortcut(QKeySequence(f"Ctrl+{i + 1}"), self).activated.connect(lambda idx=i: self.sidebar.setCurrentRow(idx))
        QShortcut(QKeySequence("F1"), self).activated.connect(self.show_help)

    def closeEvent(self, event):
        from inference import worker
        from inference.manager import manager

        from tabs.workspace.context import workspace
        if not workspace().confirm_discard(self):  # one shared dataset, so one prompt
            event.ignore()
            return
        from tabs.workspace import jobs
        if worker.any_running() or jobs.any_running():
            reply = QMessageBox.question(
                self, "Job running", "A captioning/tagging job is still running. Stop it and quit?\n"
                "Images finished so far are already saved.", QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                event.ignore()
                return
        worker.cancel_all()
        jobs.cancel_all()
        manager().unload_all(blocking=True)
        event.accept()


def main():
    scale = settings().get("ui.scale")
    if scale and abs(scale - 1.0) > 0.01 and "QT_SCALE_FACTOR" not in os.environ:
        os.environ["QT_SCALE_FACTOR"] = f"{max(0.75, min(2.5, scale)):.2f}"
    if os.name == "nt":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
        except Exception:
            pass
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    icon = paths.resource("logo.ico") or paths.resource("logo.png")
    if icon:
        app.setWindowIcon(QIcon(str(icon)))
    apply_theme(settings().get("ui.theme"))
    window = MainWindow()
    window.show()
    log.info("TagScribeR started")
    code = app.exec()
    # Let short background tasks finish (e.g. a first-time torch import started at
    # launch) so interpreter shutdown doesn't tear modules down mid-import.
    from PySide6.QtCore import QThreadPool
    QThreadPool.globalInstance().waitForDone(10000)
    sys.exit(code)


if __name__ == "__main__":
    main()
