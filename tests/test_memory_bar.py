"""The Train tab's VRAM / RAM bar: the readers' parsing, peaks and the unavailable state. No GPU is touched."""
import os

import pytest

from core import vram_monitor as V

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
GIB = 1024 ** 3


@pytest.fixture
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _Reader:
    def __init__(self):
        self.latest, self.samples, self.stopped = (None, None), 0, False

    def start(self):
        pass

    def stop(self):
        self.stopped = True

    def give(self, vram, ram):
        self.latest, self.samples = (vram, ram), self.samples + 1


def test_typeperf_row_takes_the_busiest_adapter():
    row = '"10/05/2026 04:16:45.123","1073741824.000000","19928309760.000000"," "'
    assert V._parse_typeperf_line(row) == 19928309760
    assert V._parse_typeperf_line('"(PDH-CSV 4.0)","GPU Local Adapter Memory(x) Local Usage"') is None
    assert V._parse_typeperf_line("") is None


def test_linux_amd_parsers():
    static = [{"gpu": 0, "vram": {"size": {"value": 20464, "unit": "MB"}}}]
    monitor = [{"gpu": 0, "vram_used": {"value": 15000, "unit": "MB"}}]
    assert V._merge_amd_smi_vram(static, monitor) == (15000 * 1024 ** 2, 20464 * 1024 ** 2)
    text = "GPU[0]: Used Memory (B): 1000\nGPU[0]: Total Memory (B): 4000\n"
    assert V._parse_rocm_smi_text(text) == (1000, 4000)


def test_visible_gpu_index_follows_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    assert V.visible_gpu_index() == 1
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert V.visible_gpu_index() == 0


def test_ram_read():
    used, total = V.read_ram()
    assert 0 < used < total


def test_bar_tracks_peak_and_resets(app):
    from tabs.memory_bar import MemoryBar
    reader = _Reader()
    bar = MemoryBar(reader)
    bar.poll()
    assert bar.vram.available is None                       # nothing sampled yet: not "unavailable"
    reader.give((18 * GIB, 20 * GIB), (12 * GIB, 32 * GIB))
    bar.poll()
    reader.give((10 * GIB, 20 * GIB), (14 * GIB, 32 * GIB))
    bar.poll()
    assert bar.vram.text() == "VRAM  10.0 / 20.0 GB · peak 18.0"
    assert bar.ram.text() == "RAM  14.0 / 32.0 GB · peak 14.0"
    bar.reset_peaks()
    bar.poll()
    assert bar.vram.peak == 10 * GIB
    reader.give(None, (14 * GIB, 32 * GIB))                 # no VRAM reader on this machine
    bar.poll()
    assert bar.vram.text() == "VRAM stats unavailable" and bar.ram.available
    bar.grab()                                              # both states paint
    bar.shutdown()
    assert reader.stopped


def test_train_tab_has_the_bar_and_remembers_hide(app):
    from tabs.train import TrainTab
    tab = TrainTab()
    was = tab.cfg.get("training.stats_bar_visible", True)
    try:
        tab._show_stats(True)
        tab._toggle_stats()
        assert tab.memory_bar.isHidden() and tab.btn_stats.text() == "Show stats"
        assert tab.cfg.get("training.stats_bar_visible") is False
        tab._toggle_stats()
        assert not tab.memory_bar.isHidden() and tab.btn_stats.text() == "Hide stats"
    finally:
        tab.cfg.set("training.stats_bar_visible", was)
        tab.memory_bar.shutdown()
