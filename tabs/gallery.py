"""Gallery: the dataset workspace (browser + inspector, batch tools and tag statistics).

The open folder, edits, undo history and unsaved state are shared with every
other tab through the WorkspaceContext; nothing is written until Save.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QHBoxLayout, QMessageBox, QSplitter, QTabWidget, QWidget

from core import captions
from core.dataset_session import op_add_tags, op_remove_tags, op_replace_tag
from core.widgets import AutoTagDialog
from inference.base import CaptionRequest, ProviderSpec
from inference.worker import SAVE_NONE, BatchJob, start_job
from tabs.common import confirm, show_job_summary
from tabs.workspace.browser import DatasetBrowser
from tabs.workspace.context import JOB_WORKING, workspace
from tabs.workspace.panels import BatchPanel, InspectorPanel, TagStatsPanel

log = logging.getLogger(__name__)


class GalleryTab(QWidget):
    image_selected = Signal(str)

    def __init__(self):
        super().__init__()
        self.ctx = workspace()
        self.tag_worker = None
        self.pending_tag_updates: dict[str, str] = {}
        self.auto_tag_settings: dict = {}
        self._tag_generation = 0

        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        splitter = QSplitter(Qt.Horizontal)

        self.browser = DatasetBrowser(self.ctx, size_key="ui.thumbnail_size")
        self.browser.selection_changed.connect(self._on_selection_changed)
        self.browser.current_changed.connect(self.image_selected.emit)
        self.browser.view.doubleClicked.connect(lambda _: self.inspector.editor.setFocus())

        self.side = QTabWidget()
        self.side.setMinimumWidth(360)
        self.inspector = InspectorPanel()
        self.inspector.text_edited.connect(
            lambda key, old, new: self.ctx.push({key: (old, new)}, f"Edit {Path(key).name}", merge_key=key))
        self.inspector.tags_added.connect(lambda tags: self.browser.run_operation(
            f"Add tags: {', '.join(tags)}", op_add_tags(tags)))
        self.inspector.tags_removed.connect(lambda tags: self.browser.run_operation(
            f"Remove tags: {', '.join(tags)}", op_remove_tags(tags)))
        self.inspector.navigate.connect(self.browser.navigate)
        self.ctx.add_commit_hook(self.inspector._commit_typing)

        self.batch = BatchPanel()
        self.batch.operation.connect(self.browser.run_operation)
        self.batch.auto_tag.connect(self.run_auto_tagger)
        self.batch.quick_tag.connect(lambda tag, pos: self.browser.run_operation(
            f"Add tag: {tag}", op_add_tags(captions.split_tags(tag), pos)))

        self.tags = TagStatsPanel()
        self.tags.filter_requested.connect(self.browser.set_filter)
        self.tags.rename_requested.connect(lambda old, new: self.browser.run_operation(
            f"Rename tag '{old}' → '{new}'", op_replace_tag(old, new), "all"))
        self.tags.delete_requested.connect(self._delete_tag_everywhere)
        self.tags.add_to_selection.connect(lambda t: self.browser.run_operation(f"Add tag: {t}", op_add_tags([t])))
        self.tags.select_with_tag.connect(lambda t: self.browser.select_matching(f'tag:"{t}"'))
        self.tags.combo_scope.currentIndexChanged.connect(lambda _: self._refresh_tag_stats())

        self.side.addTab(self.inspector, "Inspect")
        self.side.addTab(self.batch, "Batch")
        self.side.addTab(self.tags, "Tags")
        self.side.setTabToolTip(0, "View and edit the selected image's caption")
        self.side.setTabToolTip(1, "Apply tag/text operations to many images at once")
        self.side.setTabToolTip(2, "Tag frequency statistics; rename, merge or delete tags everywhere")
        self.side.currentChanged.connect(lambda i: self._refresh_tag_stats() if i == 2 else None)
        from tabs.help import HelpDialog, help_button
        corner = help_button("inspect", self.side, "Help for this panel")
        corner.clicked.disconnect()
        corner.clicked.connect(lambda: HelpDialog.open_topic(self.window(),
                                                             ("inspect", "batch", "tags")[self.side.currentIndex()]))
        self.side.setCornerWidget(corner, Qt.TopRightCorner)

        splitter.addWidget(self.browser)
        splitter.addWidget(self.side)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1100, 420])
        root.addWidget(splitter)

        self._stats_timer = QTimer(self, singleShot=True, interval=400)
        self._stats_timer.timeout.connect(self._refresh_tag_stats)
        self.ctx.entries_changed.connect(self._on_entries_changed)
        self.ctx.session_changed.connect(self._on_session_changed)
        self.ctx.dimensions_loaded.connect(lambda _k: self.inspector.refresh_from_entries())

        for seq, fn in (("Ctrl+Down", lambda: self.browser.navigate(1)),
                        ("Ctrl+Up", lambda: self.browser.navigate(-1)),
                        ("F2", lambda: self.inspector.editor.setFocus())):
            s = QShortcut(QKeySequence(seq), self)
            s.setContext(Qt.WidgetWithChildrenShortcut)
            s.activated.connect(fn)

    # ------------------------------------------------------------------ compatibility API
    @property
    def session(self):
        return self.ctx.session

    @property
    def current_folder(self) -> str:
        return self.ctx.folder

    def open_folder(self, folder: str):
        self.ctx.open_folder(folder, self.browser.chk_recursive.isChecked(), self)

    def has_unsaved_changes(self) -> bool:
        return self.ctx.has_unsaved()

    def save_all(self) -> bool:
        return self.browser.save()

    def select_all(self):
        self.browser.select_all()

    # ------------------------------------------------------------------ reactions
    def showEvent(self, event):
        super().showEvent(event)
        if self.tag_worker is None:
            self.ctx.pick_up_external_changes()

    def _on_session_changed(self):
        self.inspector.show_entries([])
        self._stats_timer.start()

    def _on_entries_changed(self, _keys):
        self.inspector.refresh_from_entries()
        self._stats_timer.start()

    def _on_selection_changed(self, keys: list[str]):
        s = self.ctx.session
        entries = [s.get(k) for k in keys] if s else []
        self.inspector.show_entries([e for e in entries if e])
        self.batch.set_counts(len(keys), self.browser.proxy.rowCount())
        if self.tags.scope() == "selected" and self.side.currentIndex() == 2:
            self._stats_timer.start()

    def _refresh_tag_stats(self):
        if not self.ctx.session or self.side.currentIndex() != 2:
            return
        keys = self.browser.keys_for_scope(self.tags.scope())
        self.tags.set_counts(self.ctx.session.tag_counts(keys), len(keys))

    def _delete_tag_everywhere(self, tag: str):
        if confirm(self, "Delete tag", f"Remove '{tag}' from every caption in this folder? (You can undo this.)"):
            self.browser.run_operation(f"Delete tag '{tag}'", op_remove_tags([tag]), "all")

    # ------------------------------------------------------------------ auto tagger
    def run_auto_tagger(self, scope: str = "selected"):
        if self.tag_worker is not None:
            self.tag_worker.cancel()
            self.batch.btn_auto.setText("Stopping…")
            return
        keys = self.browser.keys_for_scope(scope)
        if not keys:
            QMessageBox.information(self, "Auto Tag", "Select images to tag first.")
            return
        dlg = AutoTagDialog(self)
        if not dlg.exec():
            return
        st = self.auto_tag_settings = dlg.get_settings()
        if st["mode"] == "ignore":
            keys = [k for k in keys if not self.ctx.session.get(k).has_caption]
            if not keys:
                QMessageBox.information(self, "Nothing to tag", "All chosen images already have captions.")
                return
        fmt = dict(general_threshold=st["threshold"], character_threshold=st["character_threshold"],
                   max_tags=st["max_tags"], underscores_to_spaces=st["underscores_to_spaces"],
                   escape_parentheses=st["escape_parentheses"], include_rating=st["include_rating"],
                   blacklist=tuple(st["blacklist"]))
        spec = ProviderSpec.make("wd_tagger", st["model"], format=tuple(sorted(fmt.items())),
                                 device=self.ctx.cfg.get("tagger.device"))
        job = BatchJob(paths=[Path(k) for k in keys], spec=spec, request=CaptionRequest(prompt="", max_image_side=0),
                       save_mode=SAVE_NONE, tag_mode=True, title="Auto tagging")
        self.ctx.commit_pending_edits()
        self.pending_tag_updates = {}
        self._tag_generation = self.ctx.generation
        self.ctx.set_job_state(keys, JOB_WORKING)
        self.batch.btn_auto.setText(f"⏳ Tagging 0/{len(keys)}…  (click to stop)")
        self.tag_worker = start_job(job)
        self.tag_worker.item_done.connect(self.on_tagger_finished)
        self.tag_worker.item_failed.connect(lambda p, r: self.ctx.set_job_state([p], "failed"))
        self.tag_worker.progress.connect(lambda d, t: self.batch.btn_auto.setText(f"⏳ Tagging {d}/{t}…  (click to stop)"))
        self.tag_worker.finished.connect(lambda s, ks=keys: self.finalize_auto_tagging(s, ks))

    def on_tagger_finished(self, path: str, tag_text: str):
        s = self.ctx.session
        if not s or self.ctx.generation != self._tag_generation:
            return  # folder changed while tagging
        e = s.get(path)
        if e is None:
            return
        st = self.auto_tag_settings
        new = captions.merge_generated_tags(e.text, captions.split_tags(tag_text), st["mode"],
                                            prepend=st["prepend"], append=st["append"])
        self.pending_tag_updates.setdefault(path, e.text)
        e.text = new
        self.ctx.set_job_state([path], None)

    def finalize_auto_tagging(self, summary=None, keys=()):
        self.tag_worker = None
        self.batch.btn_auto.setText("✨ Auto Tag (WD tagger)…")
        s = self.ctx.session
        if s and self.ctx.generation == self._tag_generation:
            leftover = [k for k in keys if self.ctx.job_state.get(k) == JOB_WORKING]
            self.ctx.set_job_state(leftover, None)  # cancelled before reaching these
            if self.pending_tag_updates:
                changes = {k: (old, s.get(k).text) for k, old in self.pending_tag_updates.items() if s.get(k)}
                # Entries already hold the new text; the command records it for undo/redo.
                self.ctx.push(changes, f"Auto tag ({len(changes)})")
        self.pending_tag_updates = {}
        if summary is not None:
            show_job_summary(self, summary)
            if not summary.fatal:
                self.browser.flash(summary.text() + " — unsaved; press Save (Ctrl+S) to keep.")
