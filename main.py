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
        for seq in ("Ctrl+K", "Ctrl+Shift+P"):
            QShortcut(QKeySequence(seq), self).activated.connect(self.open_palette)

    # -- command palette -----------------------------------------------------
    def _goto(self, index: int):
        self.sidebar.setCurrentRow(index)

    def _active_browser(self):
        """The image grid of the visible tab (Gallery's if the tab has none)."""
        tab = self.stack.currentWidget()
        return getattr(tab, "browser", None) or self.tab_gallery.browser

    def palette_commands(self):
        from core.projects import BUILTIN_FILTERS
        from tabs.help_content import TOPICS
        from tabs.palette import Command
        from tabs.workspace.context import workspace
        ctx = workspace()
        has_folder = lambda: ctx.session is not None  # noqa: E731
        b = self._active_browser
        cmds = [Command(name, lambda i=i: self._goto(i), "Go to", f"Ctrl+{i + 1}")
                for i, name in enumerate(("Gallery", "Auto Caption", "Image Editor", "Datasets", "Metadata",
                                          "Settings"))]
        cmds += [
            Command("Open folder…", lambda: b().select_folder(), "File", "Ctrl+O"),
            Command("Save changed captions", lambda: b().save(), "File", "Ctrl+S", enabled=has_folder),
            Command("Undo", ctx.undo, "Edit", "Ctrl+Z", enabled=has_folder),
            Command("Redo", ctx.redo, "Edit", "Ctrl+Y", enabled=has_folder),
            Command("Select all shown images", lambda: b().select_all(), "Select", "Ctrl+A", enabled=has_folder),
            Command("Select images missing captions", lambda: b().select_matching("missing:caption"), "Select",
                    enabled=has_folder),
            Command("Select unsaved images", lambda: b().select_matching("is:unsaved"), "Select", enabled=has_folder),
            Command("Clear filter", lambda: b().set_filter(""), "Filter", enabled=has_folder),
            Command("Zoom thumbnails in", lambda: b().zoom(1), "View", "Ctrl+="),
            Command("Zoom thumbnails out", lambda: b().zoom(-1), "View", "Ctrl+-"),
            Command("Auto tag selected images (WD)…", lambda: (self._goto(0), self.tab_gallery.run_auto_tagger("selected")),
                    "AI", keywords="booru wd tagger", enabled=has_folder),
            Command("Caption selected images", lambda: (self._goto(1), self.tab_caption.run_process()), "AI",
                    "Ctrl+Enter", keywords="vlm describe", enabled=has_folder),
            Command("Review AI caption proposals", lambda: (self._goto(1), self.tab_caption.open_review()), "AI",
                    enabled=lambda: bool(self.tab_caption.pending)),
            Command("Free GPU memory (unload model)", self.tab_caption.force_cleanup, "AI", keywords="vram unload"),
            Command("Export for training…", lambda: (self._goto(3), self.tab_datasets.export_for_training()),
                    "Dataset", keywords="kohya buckets prepare", enabled=has_folder),
            Command("Scan dataset health", lambda: (self._goto(0), self.tab_gallery.side.setCurrentIndex(3),
                                                   self.tab_gallery.scan_health()),
                    "Dataset", keywords="duplicates blur buckets", enabled=has_folder),
        ]
        cmds += [Command(f"Open recent: {f}", lambda f=f: ctx.confirm_discard(self) and ctx.open_folder(
            f, ctx.cfg.get("ui.recursive_scan", False), self), "File") for f in ctx.recent_folders()[:6]]
        cmds += [Command(name, lambda q=q: b().set_filter(q), "Filter", keywords=q, enabled=has_folder)
                 for name, q in BUILTIN_FILTERS.items()]
        if ctx.project is not None:
            cmds += [Command(f"★ {name}", lambda q=q: b().set_filter(q), "Filter", keywords=q)
                     for name, q in ctx.project.filters().items()]
        cmds += [Command(title, lambda t=tid: self.show_help(t), "Help") for tid, (title, _g, _h) in TOPICS.items()]
        cmds.append(Command("Keyboard shortcuts", lambda: self.show_help("hotkeys"), "Help"))
        return cmds

    def open_palette(self):
        from tabs.palette import CommandPalette
        CommandPalette(self.palette_commands(), self).show()

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


def preload_ai_runtime(app) -> None:
    """Import PyTorch once on the main thread, before any background work starts.

    A first-time torch import on a worker thread while the main thread is busy
    importing other modules can fail halfway (torch's operator registry ends up
    inconsistent), leaving a half-initialised module behind. Importing it here,
    with nothing else running, makes every later use safe — and the first model
    load faster.
    """
    if not hardware.torch_installed():
        return
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QSplashScreen
    from PySide6.QtGui import QPixmap
    logo = paths.resource("logo.png")
    pm = QPixmap(str(logo)).scaled(360, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation) if logo else QPixmap(360, 120)
    splash = QSplashScreen(pm)
    splash.showMessage("Loading AI runtime…", Qt.AlignBottom | Qt.AlignHCenter, Qt.white)
    splash.show()
    app.processEvents()
    try:
        hardware.detect_devices()  # imports torch (under the hardware lock) and caches the device list
    except Exception as e:  # never block startup; AI features report the problem when used
        log.warning("AI runtime preload failed: %s", e)
    splash.close()


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
    preload_ai_runtime(app)
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
