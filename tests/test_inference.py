import threading

import pytest

from core import caption_io
from inference import manager as mgr_mod
from inference import worker as w
from inference.base import CaptionRequest, InferenceError, Provider, ProviderSpec
from inference.wd_tagger import TagFormat, TagPrediction, format_tags
from tests.conftest import make_image


@pytest.fixture(scope="module", autouse=True)
def qapp():
    from PySide6.QtCore import QCoreApplication
    app = QCoreApplication.instance() or QCoreApplication([])
    yield app


class FakeProvider(Provider):
    instances = 0

    def __init__(self, spec):
        super().__init__(spec)
        FakeProvider.instances += 1
        self.is_loaded = False
        self.preferred_chunk = int(spec.opt("batch_size", 1))

    @property
    def loaded(self):
        return self.is_loaded

    def load(self, report, cancel=None):
        self.is_loaded = True

    def unload(self):
        self.is_loaded = False

    def generate(self, images, request, cancel=None, prompts=None):
        out = []
        for i, img in enumerate(images):
            if img.size == (13, 13):
                out.append(InferenceError("boom"))
            elif prompts:
                out.append(f"{prompts[i]} {img.size[0]}")
            else:
                out.append(f"caption {img.size[0]}")
        return out


@pytest.fixture
def fake_manager(monkeypatch):
    m = mgr_mod.ModelManager()
    monkeypatch.setattr(mgr_mod, "_manager", m)
    monkeypatch.setattr(mgr_mod, "create_provider", lambda spec: FakeProvider(spec))
    FakeProvider.instances = 0
    return m


def run_job(job):
    worker = w.BatchWorker(job)
    events = {"done": [], "failed": [], "skipped": [], "summary": None}
    worker.item_done.connect(lambda p, t: events["done"].append((p, t)))
    worker.item_failed.connect(lambda p, r: events["failed"].append((p, r)))
    worker.item_skipped.connect(lambda p, r: events["skipped"].append(p))
    worker.finished.connect(lambda s: events.__setitem__("summary", s))
    worker.run()  # synchronous: direct signal delivery
    return events


def spec(**opts):
    return ProviderSpec.make("transformers", "fake-model", **opts)


def test_batch_tolerates_failures_and_saves_results(tmp_path, fake_manager):
    good = [make_image(tmp_path / f"g{i}.png", size=(20 + i, 20)) for i in range(3)]
    bad = make_image(tmp_path / "bad.png", size=(13, 13))
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    job = w.BatchJob(paths=[good[0], bad, corrupt, good[1], good[2]], spec=spec(batch_size=2),
                     request=CaptionRequest(prompt="x"))
    ev = run_job(job)
    s = ev["summary"]
    assert s.done == 3 and len(s.failed) == 2 and not s.fatal
    assert caption_io.read_caption(good[1]) == "caption 21"
    assert not (tmp_path / "bad.txt").exists()


def test_skip_existing_does_not_touch_captioned(tmp_path, fake_manager):
    a = make_image(tmp_path / "a.png")
    b = make_image(tmp_path / "b.png")
    caption_io.write_caption(a, "keep me")
    ev = run_job(w.BatchJob(paths=[a, b], spec=spec(), request=CaptionRequest(prompt="x"),
                            save_mode=w.SAVE_SKIP_EXISTING))
    assert caption_io.read_caption(a) == "keep me"
    assert ev["summary"].skipped == 1 and ev["summary"].done == 1


def test_save_none_never_writes(tmp_path, fake_manager):
    a = make_image(tmp_path / "a.png")
    ev = run_job(w.BatchJob(paths=[a], spec=spec(), request=CaptionRequest(prompt="x"), save_mode=w.SAVE_NONE))
    assert ev["done"] and not (tmp_path / "a.txt").exists()


def test_model_reused_between_jobs(tmp_path, fake_manager):
    a = make_image(tmp_path / "a.png")
    run_job(w.BatchJob(paths=[a], spec=spec(batch_size=1), request=CaptionRequest(prompt="x")))
    run_job(w.BatchJob(paths=[a], spec=spec(batch_size=4), request=CaptionRequest(prompt="x")))
    assert FakeProvider.instances == 1  # batch size change must not reload


def test_switching_heavy_model_unloads_previous(fake_manager):
    p1 = fake_manager.acquire(ProviderSpec.make("transformers", "m1"), lambda m: None)
    p2 = fake_manager.acquire(ProviderSpec.make("transformers", "m2"), lambda m: None)
    assert not p1.loaded and p2.loaded


def test_unload_refused_while_job_running(fake_manager):
    entered, release = threading.Event(), threading.Event()

    def job():
        with fake_manager.session(ProviderSpec.make("transformers", "m"), lambda m: None):
            entered.set()
            release.wait(5)

    t = threading.Thread(target=job)
    t.start()
    entered.wait(5)
    assert fake_manager.busy() and fake_manager.unload_all() is False
    release.set()
    t.join()
    assert fake_manager.unload_all() is True


def test_cancel_stops_job(tmp_path, fake_manager):
    paths = [make_image(tmp_path / f"{i}.png") for i in range(5)]
    worker = w.BatchWorker(w.BatchJob(paths=paths, spec=spec(), request=CaptionRequest(prompt="x")))
    worker.cancel()
    result = {}
    worker.finished.connect(lambda s: result.setdefault("s", s))
    worker.run()
    assert result["s"].cancelled and result["s"].done == 0


def test_fatal_load_error_reported(tmp_path, monkeypatch, fake_manager):
    class Broken(FakeProvider):
        def load(self, report, cancel=None):
            raise InferenceError("no model", "download it")
    monkeypatch.setattr(mgr_mod, "create_provider", lambda s: Broken(s))
    ev = run_job(w.BatchJob(paths=[make_image(tmp_path / "a.png")], spec=spec(), request=CaptionRequest(prompt="x")))
    assert "no model" in ev["summary"].fatal


@pytest.mark.parametrize("existing,generated,mode,tag_mode,expected", [
    ("a, b", "b, c", w.SAVE_APPEND, True, "a, b, c"),
    ("a, b", "c", w.SAVE_PREPEND, True, "c, a, b"),
    ("a, b", "c", w.SAVE_OVERWRITE, True, "c"),
    ("A cat.", "Sitting.", w.SAVE_APPEND, False, "A cat. Sitting."),
    ("", "New.", w.SAVE_APPEND, False, "New."),
    ("old", "new", w.SAVE_OVERWRITE, False, "new"),
])
def test_combine(existing, generated, mode, tag_mode, expected):
    assert w.combine(existing, generated, mode, tag_mode) == expected


def test_combine_forced_tags():
    assert w.combine("x", "trigger, y", w.SAVE_APPEND, True, prepend=("trigger",), append=("end",)) == "trigger, x, y, end"


def test_format_tags():
    pred = TagPrediction(rating={"general": 0.9, "explicit": 0.1},
                         general=[("long_hair", 0.9), ("smile", 0.5), ("^_^", 0.6), ("blurry", 0.1)],
                         character=[("hatsune_miku", 0.95), ("other_char", 0.5)])
    tags = format_tags(pred, TagFormat(blacklist=("smile",), include_rating=True))
    assert tags == ["general", "hatsune miku", "long hair", "^_^"]
    assert format_tags(pred, TagFormat(max_tags=1)) == ["hatsune miku"]


def test_build_prompt_subject_and_tags():
    from inference.prompts import build_prompt
    p = build_prompt("Describe the image.", subject="ohwx woman", start_with_subject=True, tags="red hair, smile")
    assert p.startswith("Describe the image.")
    assert '"ohwx woman"' in p and 'Begin the caption with "ohwx woman"' in p
    assert "red hair, smile" in p and "may contain mistakes" in p
    assert build_prompt("Describe.") == "Describe."


def test_per_image_prompts_and_prefix(tmp_path, fake_manager):
    a = make_image(tmp_path / "a.png", size=(21, 20))
    b = make_image(tmp_path / "b.png", size=(22, 20))
    job = w.BatchJob(paths=[a, b], spec=spec(batch_size=2), request=CaptionRequest(prompt="base"),
                     prompts={str(a): "hint-a", str(b): "hint-b"}, ensure_prefix="ohwx")
    run_job(job)
    assert caption_io.read_caption(a) == "ohwx, hint-a 21"
    assert caption_io.read_caption(b) == "ohwx, hint-b 22"


def test_prefix_not_duplicated(tmp_path, fake_manager):
    a = make_image(tmp_path / "a.png", size=(21, 20))
    run_job(w.BatchJob(paths=[a], spec=spec(), request=CaptionRequest(prompt="x"),
                       prompts={str(a): "OHWX woman"}, ensure_prefix="ohwx"))
    assert caption_io.read_caption(a) == "OHWX woman 21"


def test_job_handle_delivers_on_ui_thread(tmp_path, fake_manager, qapp):
    """Lambdas connected to job signals must run on the UI thread (regression)."""
    import time
    from PySide6.QtCore import QThread
    a = make_image(tmp_path / "a.png")
    handle = w.start_job(w.BatchJob(paths=[a], spec=spec(), request=CaptionRequest(prompt="x")))
    seen = {}
    ui_thread = qapp.thread()
    handle.item_done.connect(lambda p, t: seen.setdefault("item", QThread.currentThread() is ui_thread))
    handle.finished.connect(lambda s: seen.setdefault("finished", QThread.currentThread() is ui_thread))
    end = time.time() + 10
    while "finished" not in seen and time.time() < end:
        qapp.processEvents()
        time.sleep(0.01)
    assert seen == {"item": True, "finished": True}


def test_file_job_handle_delivers_on_ui_thread(tmp_path, qapp):
    import time
    from PySide6.QtCore import QThread
    from tabs.workspace.jobs import start_file_job
    handle = start_file_job(["x", "y"], lambda k: None, "t")
    seen = []
    handle.progress.connect(lambda d, t: seen.append(QThread.currentThread() is qapp.thread()))
    done = {}
    handle.finished.connect(lambda s: done.setdefault("ok", QThread.currentThread() is qapp.thread()))
    end = time.time() + 10
    while not done and time.time() < end:
        qapp.processEvents()
        time.sleep(0.01)
    assert done == {"ok": True} and seen and all(seen)
