"""Captioning speed: the parallel-request limiter, the request pipeline, image limits, the loop guard and the
tagger's fast paths. No model, no GPU, no real server: a discrete-event model (tests/caption_sim.py) for the
limiter's decisions and a simulated HTTP server on 127.0.0.1 (tools/bench_caption.py) for the real client.
Every server test has a hard time limit and shuts its server down."""
import importlib
import os
import threading
import time

import numpy as np
import pytest
from PIL import Image

from core import caption_io
from inference import image_budget, image_prep, loopguard
from inference import manager as mgr_mod
from inference import worker as w
from inference.base import Cancelled, CaptionRequest, GenerationParams, InferenceError, Provider, ProviderSpec
from inference.concurrency import AdaptiveLimiter, RateMeter, format_eta
from inference.pipeline import Outcome, PrepareError, run_pipeline
from tests.caption_sim import Sim, simulate
from tests.conftest import make_image

bench = importlib.import_module("tools.bench_caption")


@pytest.fixture(scope="module", autouse=True)
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def within(seconds, fn):
    """Run ``fn`` on a daemon thread and fail the test if it has not returned in ``seconds`` (never hang)."""
    box = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001
            box["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(seconds)
    assert not t.is_alive(), f"did not finish within {seconds} s"
    if "error" in box:
        raise box["error"]
    return box["value"]


def wait_for(condition, seconds=5.0):
    end = time.monotonic() + seconds
    while not condition() and time.monotonic() < end:
        time.sleep(0.02)
    return condition()


# -- the limiter, against the event model ----------------------------------------------------------------------------
def settle(requests=600, start=1, cap=8, **server):
    sim = Sim(**server)
    lim = AdaptiveLimiter(start=start, cap=cap, clock=sim.clock)
    simulate(lim, sim, requests)
    return lim, sim


@pytest.mark.parametrize("seed", range(12))
def test_one_slot_server_settles_at_one(seed):
    lim, sim = settle(slots=1, seed=seed)
    assert lim.limit == 1
    assert max(sim.history) <= 2                           # it only ever tried one more
    assert sim.history.count(2) <= 0.06 * len(sim.history)   # ... briefly: a few tries of six requests each


@pytest.mark.parametrize("slots", [2, 3, 4])
@pytest.mark.parametrize("seed", range(6))
def test_finds_the_servers_slots(slots, seed):
    lim, sim = settle(slots=slots, seed=seed)
    assert lim.limit == slots
    best = Sim(slots=slots, seed=seed)
    simulate(AdaptiveLimiter(start=slots, cap=slots, adaptive=False, clock=best.clock), best, 600)
    assert sim.finished / sim.seconds > 0.93 * best.finished / best.seconds     # the search itself costs little


def test_a_one_slot_server_is_not_slower_than_one_at_a_time():
    lim, sim = settle(slots=1, seed=3)
    alone = Sim(slots=1, seed=3)
    simulate(AdaptiveLimiter(start=1, cap=1, adaptive=False, clock=alone.clock), alone, 600)
    assert sim.finished / sim.seconds >= 0.999 * alone.finished / alone.seconds


def test_no_gain_from_batching_means_one():
    lim, _ = settle(slots=4, penalty=1.0, seed=1)          # four slots that share one stream's speed
    assert lim.limit == 1


@pytest.mark.parametrize("seed", range(6))
def test_a_number_set_too_high_is_walked_down(seed):
    lim, _ = settle(slots=1, start=6, seed=seed)
    assert lim.limit == 1
    lim, _ = settle(slots=4, start=8, seed=seed)
    assert lim.limit == 4


def test_started_at_the_right_number_it_stays():
    lim, sim = settle(slots=4, start=4, seed=2)
    assert lim.limit == 4 and min(sim.history[60:]) >= 3 and max(sim.history) <= 5


def test_cap_and_fixed_mode_are_respected():
    lim, sim = settle(slots=8, penalty=0.0, cap=3, seed=0)
    assert lim.limit == 3 and max(sim.history) == 3
    sim = Sim(slots=8, seed=0)
    fixed = AdaptiveLimiter(start=5, cap=5, adaptive=False, clock=sim.clock)
    simulate(fixed, sim, 100)
    assert set(sim.history) == {5} and fixed.describe() == "5 at a time"
    assert not AdaptiveLimiter(start=1, cap=1).adaptive          # nothing to adapt between 1 and 1


def test_backs_off_on_overload_and_never_returns_to_a_level_that_failed_twice():
    lim, sim = settle(requests=900, slots=8, penalty=0.0, fail_above=3, seed=0)
    assert lim.limit == 3
    assert sim.failed <= 4                                    # level 4 was tried twice, not again and again
    assert any("overloaded" in e for e in lim.events)
    after = sim.history[len(sim.history) // 2:]
    assert max(after) <= 3


def test_does_not_try_more_when_the_queue_could_reach_the_timeout():
    sim = Sim(slots=4, fixed=30.0, seed=0)                    # about 32 s per request
    lim = AdaptiveLimiter(start=1, cap=8, timeout=60.0, clock=sim.clock)
    simulate(lim, sim, 80)
    assert set(sim.history) == {1}                            # two in a queue could take 64 s of a 60 s limit
    sim = Sim(slots=4, fixed=30.0, seed=0)
    lim = AdaptiveLimiter(start=1, cap=8, timeout=600.0, clock=sim.clock)
    simulate(lim, sim, 200)
    assert sim.history.count(4) > 100 and max(sim.history) <= 5      # with room in the time limit it uses the slots


def test_tries_one_fewer_when_requests_become_much_slower():
    sim = Sim(slots=4, seed=1, slow_after=150, slow_factor=2.0, slow_extra_above=3)
    lim = AdaptiveLimiter(start=1, cap=8, clock=sim.clock)
    simulate(lim, sim, 150)
    assert lim.limit == 4
    simulate(lim, sim, 400)                                   # now a fourth busy slot makes everything crawl
    assert lim.limit == 3 and any("slower" in e for e in lim.events)


def test_failed_requests_and_old_requests_are_not_measurements():
    lim = AdaptiveLimiter(start=1, cap=4, clock=lambda: 0.0)
    for _ in range(50):
        lim.done(lim.begin(), 0.001, ok=False)                # fast failures must not look like speed
    assert lim.limit == 1 and not lim.events
    old = lim.begin()
    lim._set(2, "up", "test")
    lim.done(old, 99.0, ok=False, busy=True)                  # belongs to the level already left
    assert lim.limit == 2


def test_rate_meter_and_eta():
    now = [0.0]
    m = RateMeter(window=10, clock=lambda: now[0])
    assert m.per_minute() == 0 and m.eta_seconds(5) is None
    for _ in range(4):
        now[0] += 2.0
        m.tick()
    assert m.per_minute() == pytest.approx(30.0)
    assert m.eta_seconds(15) == pytest.approx(30.0)
    m.pause()
    now[0] += 100.0                                           # a pause is not slowness
    assert m.per_minute() == pytest.approx(30.0)
    m.resume()
    assert m.per_minute() == pytest.approx(30.0)
    for _ in range(30):                                       # only the latest results count
        now[0] += 1.0
        m.tick()
    assert m.per_minute() == pytest.approx(60.0)
    assert format_eta(None) == "" and format_eta(42) == "42 s"
    assert format_eta(125) == "2 min 05 s" and format_eta(3 * 3600 + 120) == "3 h 02 min"


# -- the pipeline ----------------------------------------------------------------------------------------------------
def collect(keys, caption, limiter, prepare=lambda k: f"p{k}", **kw):
    return within(20, lambda: list(run_pipeline(keys, prepare, caption, limiter, **kw)))


def test_pipeline_keeps_the_limit_in_flight_without_chunk_barriers():
    lock, flying, peak, order = threading.Lock(), [0], [0], []
    slow = {0}                                                # the first request takes far longer than the rest

    def caption(prepared, key, stop):
        with lock:
            flying[0] += 1
            peak[0] = max(peak[0], flying[0])
        time.sleep(0.5 if key in slow else 0.03)
        with lock:
            flying[0] -= 1
            order.append(key)
        return f"c{key}", {"size": 10}

    out = collect(range(12), caption, AdaptiveLimiter(start=3, cap=3, adaptive=False))
    assert sorted(o.key for o in out) == list(range(12)) and all(o.result == f"c{o.key}" for o in out)
    assert peak[0] == 3
    assert order.index(0) >= 8        # in chunks of three, images 3..11 would have waited for the slow one


def test_pipeline_is_in_order_and_bounded_at_one():
    prepared_ahead = []
    seen = []

    def prepare(key):
        prepared_ahead.append(key)
        return key

    def caption(prepared, key, stop):
        seen.append((key, len(prepared_ahead)))
        time.sleep(0.01)
        return str(key), {}

    out = collect(range(30), caption, AdaptiveLimiter(start=1, cap=1, adaptive=False), prepare=prepare)
    assert [o.key for o in out] == list(range(30))
    # never more than limit + 1 pictures waiting: memory stays flat however long the list is
    assert max(ahead - key for key, ahead in seen) <= 3


def test_pipeline_failures_resend_and_cancel():
    calls = {}

    def prepare(key):
        if key == 2:
            raise OSError("broken file")
        return key

    def caption(prepared, key, stop):
        calls[key] = calls.get(key, 0) + 1
        if key == 5 and calls[key] == 1:
            return InferenceError("busy", busy=True, resend=True), {}
        if key == 7:
            return InferenceError("timed out", busy=True), {}          # a timeout is never sent again
        if key == 8:
            raise TypeError("boom")
        return f"c{key}", {}

    lim = AdaptiveLimiter(start=2, cap=4)
    out = {o.key: o for o in collect(range(10), caption, lim, prepare=prepare)}
    assert isinstance(out[2].result, PrepareError) and "broken file" in str(out[2].result)
    assert out[5].result == "c5" and out[5].attempts == 2 and calls[5] == 2
    assert isinstance(out[7].result, InferenceError) and calls[7] == 1
    assert isinstance(out[8].result, TypeError)
    assert lim.limit == 1 and any("overloaded" in e for e in lim.events)
    # one at a time: nothing is sent twice (the local-server rule)
    calls.clear()
    out = {o.key: o for o in collect([5], caption, AdaptiveLimiter(start=1, cap=1, adaptive=False))}
    assert isinstance(out[5].result, InferenceError) and calls[5] == 1

    cancel, started = threading.Event(), threading.Event()

    def slow(prepared, key, stop):
        started.set()
        if key == 0:
            return "done", {}
        stop.wait(10)
        return Cancelled(), {}

    got = []

    def run():
        with pytest.raises(Cancelled):
            for o in run_pipeline(range(50), lambda k: k, slow, AdaptiveLimiter(start=2, cap=2, adaptive=False),
                                  cancel=cancel):
                got.append(o)
                cancel.set()
        return True

    assert within(10, run)
    assert [o.key for o in got] == [0]                        # what was answered before the cancel is kept


def test_pipeline_pause_holds_new_requests():
    paused = threading.Event()
    paused.set()
    count = [0]

    def caption(prepared, key, stop):
        count[0] += 1
        return "x", {}

    threading.Timer(0.6, paused.clear).start()
    t0 = time.monotonic()
    out = collect(range(3), caption, AdaptiveLimiter(start=1, cap=1, adaptive=False), paused=paused)
    assert len(out) == 3 and time.monotonic() - t0 >= 0.5


# -- image limits ----------------------------------------------------------------------------------------------------
def test_fit_size():
    f = image_prep.fit_size
    assert f(4000, 3000) == (4000, 3000)
    assert f(4000, 3000, 1536) == (1536, 1152)
    assert f(800, 600, 1536, 1_000_000) == (800, 600)          # never enlarged
    w_, h_ = f(4000, 3000, 1536, 1_000_000)
    assert w_ * h_ <= 1_000_000 * 1.01 and abs(w_ / h_ - 4 / 3) < 0.01
    assert f(4000, 3000, 1024, 5_000_000) == (1024, 768)       # the tighter limit wins
    assert f(10000, 10, 100) == (100, 1)


def _detailed(size, seed=0):
    rng = np.random.default_rng(seed)
    w_, h_ = size
    y, x = np.mgrid[0:h_, 0:w_]
    base = np.stack([x * 255 // w_, y * 255 // h_, (x + y) * 255 // (w_ + h_)], -1).astype(np.float32)
    base += rng.normal(0, 10, (h_, w_, 3))
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")


def test_load_rgb_limits_and_matches_the_old_result(tmp_path):
    png = tmp_path / "a.png"
    _detailed((1600, 1200)).save(png)
    assert image_prep.load_rgb(png).size == (1600, 1200)
    assert image_prep.load_rgb(png, 800).size == (800, 600)
    small = image_prep.load_rgb(png, 1536, max_pixels=500_000)
    assert small.size[0] * small.size[1] <= 505_000 and small.mode == "RGB"
    # same pixels as the plain thumbnail the app made before (no JPEG involved)
    with Image.open(png) as im:
        ref = im.convert("RGB")
        ref.thumbnail((800, 800), Image.Resampling.LANCZOS)
    assert np.array_equal(np.asarray(ref), np.asarray(image_prep.load_rgb(png, 800)))


def test_reduced_jpeg_decode_is_close_to_the_full_decode(tmp_path):
    jpg = tmp_path / "big.jpg"
    _detailed((2400, 1800)).save(jpg, quality=93)
    fast = image_prep.load_rgb(jpg, 512)
    with Image.open(jpg) as im:
        full = im.convert("RGB")
        full.thumbnail((512, 512), Image.Resampling.LANCZOS)
    assert fast.size == full.size == (512, 384)
    mse = float(np.mean((np.asarray(fast, np.float32) - np.asarray(full, np.float32)) ** 2))
    assert 10 * np.log10(255 ** 2 / max(mse, 1e-9)) > 40          # far below what the JPEG sent on adds
    assert image_prep.load_rgb(jpg).size == (2400, 1800)          # no limit: decoded in full


def test_transparency_and_rotation_still_handled(tmp_path):
    p = make_image(tmp_path / "t.png", size=(40, 20), mode="RGBA")
    im = image_prep.load_rgb(p, 0, max_pixels=200)
    assert im.mode == "RGB" and im.size == (20, 10)
    assert image_prep.to_data_url(im).startswith("data:image/jpeg;base64,")


def test_image_token_estimates():
    t = image_budget.image_tokens
    assert t(1536, 1152, "qwen3") == 48 * 36
    assert t(1024, 768, "qwen2") == round(1024 / 28) * round(768 / 28)
    assert t(4000, 3000, "gemma3") == t(200, 100, "gemma3") == 256
    assert t(4000, 3000, "joycaption") == 729
    assert t(336, 336, "llava_next") == 576 and t(672, 672, "llava_next") == 2880
    assert image_budget.guess_family("qwen/qwen3-vl-8b-instruct").key == "qwen3"
    assert image_budget.guess_family("Qwen2.5-VL-7B-Instruct-GGUF").key == "qwen2"
    assert image_budget.guess_family("llama-joycaption-beta-one").key == "joycaption"
    assert image_budget.guess_family("google/gemma-3-12b-it").key == "gemma3"
    assert image_budget.guess_family("some-text-model") is None       # unknown: no advice, never a wrong one
    full = image_budget.tokens_at_budget(1536 * 1152, "qwen3")
    fast = image_budget.tokens_at_budget(600_000, "qwen3")
    assert 0.3 < fast / full < 0.4
    assert image_budget.context_needed(1000, 100, 300, parallel=4) == 4 * (1000 + 100 + 300 + 64)
    line = image_budget.describe_budget("gemma-3-12b", 1536, 0)
    assert "256 image tokens" in line and "loses nothing" in line
    assert "image tokens" in image_budget.describe_budget("qwen3-vl-8b", 1536, 1_000_000)
    assert "image tokens" not in image_budget.describe_budget("mystery", 1536, 0)


# -- the loop guard --------------------------------------------------------------------------------------------------
def test_loop_detection_and_cut():
    assert loopguard.find_loop("A photo of a cat sitting on a red sofa in a bright room.") is None
    tags = "1girl, solo, long hair, blue eyes, smile, " * 2 + "outdoors, sky, cloud, tree, grass"
    assert loopguard.find_loop(tags) is None                   # a phrase said twice is not a loop
    looping = "A woman stands in a room, wearing " + "a red scarf, " * 12
    start, period = loopguard.find_loop(looping)
    assert period == len("a red scarf, ")
    assert loopguard.cut_loop(looping) == "A woman stands in a room, wearing a red scarf"
    assert loopguard.cut_loop("fine text") is None
    assert loopguard.find_loop("An image." + " " * 100) is not None        # runaway blanks
    watch = loopguard.LoopWatch()
    fed = 0
    for piece in ["A woman stands in a room, wearing "] + ["a red ", "scarf, "] * 200:
        fed += 1
        if watch.feed(piece):
            break
    assert fed < 40                                            # seen after a few repeats, not at the token limit


# -- the tagger's fast paths -----------------------------------------------------------------------------------------
def _old_read_scores(tags, probs):
    from inference.wd_tagger import CHARACTER, RATING, TagPrediction
    out = []
    for row in probs:
        pred = TagPrediction()
        for (tag, cat), p in zip(tags, row):
            p = float(p)
            if cat == RATING:
                pred.rating[tag] = p
            elif cat == CHARACTER:
                if p >= 0.2:
                    pred.character.append((tag, p))
            elif p >= 0.05:
                pred.general.append((tag, p))
        pred.general.sort(key=lambda x: -x[1])
        pred.character.sort(key=lambda x: -x[1])
        out.append(pred)
    return out


def test_tagger_scores_are_read_exactly_as_before():
    from inference.wd_tagger import WDTagger
    rng = np.random.default_rng(5)
    cats = [9] * 4 + [0] * 3000 + [4] * 500 + [1] * 20
    tagger = WDTagger(ProviderSpec.make("wd_tagger", "x"))
    tagger.tags = [(f"tag_{i}", c) for i, c in enumerate(cats)]
    probs = (rng.random((5, len(cats))) ** 6).astype(np.float32)
    probs[0, 10:20] = 0.5                                      # ties keep the file's order
    probs[1, 30] = np.float32(0.05)                            # exactly on a floor
    probs[1, 31] = np.float32(0.2)
    probs[2, 3010] = np.float32(0.2)
    new, old = tagger.read_scores(probs), _old_read_scores(tagger.tags, probs)
    for a, b in zip(new, old):
        assert a.rating == b.rating and a.general == b.general and a.character == b.character
    assert tagger.preferred_chunk == 8
    assert WDTagger(ProviderSpec.make("wd_tagger", "x", batch_size=16)).preferred_chunk == 16


def test_tagger_prepare_runs_ahead_and_gives_the_same_input():
    from inference.wd_tagger import WDTagger

    class Session:
        def __init__(self):
            self.batches = []

        def get_inputs(self):
            return [type("I", (), {"name": "in", "shape": [None, 64, 64, 3]})()]

        def run(self, _out, feed):
            self.batches.append(feed["in"])
            return [np.zeros((len(feed["in"]), 3), np.float32)]

    tagger = WDTagger(ProviderSpec.make("wd_tagger", "x"))
    tagger.size, tagger.dynamic_batch = 64, True
    tagger.tags = [("general", 9), ("a", 0), ("b", 4)]
    img = _detailed((200, 120))
    assert tagger.prepare(img, CaptionRequest(prompt="")) is img        # not loaded yet: left alone
    tagger.session = Session()
    ready = tagger.prepare(img, CaptionRequest(prompt=""))
    assert isinstance(ready, np.ndarray) and ready.shape == (64, 64, 3) and ready.dtype == np.float32
    tagger.predict([ready, img])                                       # prepared and plain pictures mix
    batch = tagger.session.batches[0]
    assert np.array_equal(batch[0], batch[1])


# -- the worker ------------------------------------------------------------------------------------------------------
class FakeServer(Provider):
    """A pipelined provider without HTTP: answers after a short sleep, records what it was given."""
    pipelined = True

    def __init__(self, spec):
        super().__init__(spec)
        self.lock = threading.Lock()
        self.flying = self.peak = 0
        self.prepared = []
        self.limiter = AdaptiveLimiter(start=int(spec.opt("concurrency", 1)), cap=int(spec.opt("concurrency", 1)),
                                       adaptive=False)

    def make_limiter(self):
        return self.limiter

    def prepare(self, image, request):
        self.prepared.append(image.size)
        return f"data:{image.size[0]}x{image.size[1]}"

    def generate(self, images, request, cancel=None, prompts=None):
        raise AssertionError("a pipelined provider is not given chunks")

    def caption_prepared(self, prepared, request, cancel=None, prompt=None):
        with self.lock:
            self.flying += 1
            self.peak = max(self.peak, self.flying)
        time.sleep(0.02)
        with self.lock:
            self.flying -= 1
        if prepared == "data:13x13":
            return InferenceError("boom"), {}
        if prepared == "data:14x14":
            raise TypeError("odd <b>reply</b>")
        return f"{prompt or request.prompt} {prepared}", {"size": 5, "cut": prepared == "data:15x15"}


@pytest.fixture
def server_manager(monkeypatch):
    m = mgr_mod.ModelManager()
    monkeypatch.setattr(mgr_mod, "_manager", m)
    made = []

    def make(spec):
        made.append(FakeServer(spec))
        return made[-1]

    monkeypatch.setattr(mgr_mod, "create_provider", make)
    return made


def run_job(job, before=None):
    worker = w.BatchWorker(job)
    ev = {"done": [], "failed": [], "stats": [], "status": [], "started": [], "summary": None}
    worker.item_done.connect(lambda p, t: ev["done"].append((p, t)))
    worker.item_failed.connect(lambda p, r: ev["failed"].append((p, r)))
    worker.stats.connect(ev["stats"].append)
    worker.status.connect(ev["status"].append)
    worker.items_started.connect(ev["started"].extend)
    worker.finished.connect(lambda s: ev.__setitem__("summary", s))
    if before:
        before(worker)
    watchdog = threading.Timer(30, worker.cancel)            # never hang: a stuck job is cancelled and fails below
    watchdog.daemon = True
    watchdog.start()
    try:
        worker.run()        # on this thread, so the signals above are delivered directly
    finally:
        fired = not watchdog.is_alive()
        watchdog.cancel()
    assert not fired, "the job did not finish within 30 s"
    return ev


def test_worker_streams_to_a_server_and_saves_each_caption(tmp_path, server_manager):
    good = [make_image(tmp_path / f"g{i}.png", size=(20 + i, 20)) for i in range(9)]
    bad = make_image(tmp_path / "bad.png", size=(13, 13))
    odd = make_image(tmp_path / "odd.png", size=(14, 14))
    cut = make_image(tmp_path / "cut.png", size=(15, 15))
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    big = make_image(tmp_path / "big.png", size=(400, 200))
    paths = good[:4] + [bad, corrupt, odd, cut, big] + good[4:]
    job = w.BatchJob(paths=paths, spec=ProviderSpec.make("openai", "m", base_url="http://x/v1", concurrency=3),
                     request=CaptionRequest(prompt="base", max_image_side=0, max_image_pixels=20_000),
                     prompts={str(good[0]): "hint"}, ensure_prefix="ohwx")
    ev = run_job(job)
    s = ev["summary"]
    prov = server_manager[0]
    assert not s.fatal and s.done == 11 and len(s.failed) == 3 and s.cut_short == 1 and s.parallel == 3
    assert prov.peak == 3
    assert caption_io.read_caption(good[0]) == "ohwx, hint data:20x20"
    assert caption_io.read_caption(good[8]) == "ohwx, base data:28x20"
    assert caption_io.read_caption(big) == "ohwx, base data:200x100"        # the pixel budget was applied
    reasons = dict(ev["failed"])
    assert "boom" in reasons[str(bad)] and "could not open image" in reasons[str(corrupt)]
    assert "Unexpected error: TypeError" in reasons[str(odd)] and "<b>" not in reasons[str(odd)]
    assert set(ev["started"]) == {str(p) for p in paths if p != corrupt}
    assert any("cut short" in m for m in ev["status"])
    last = ev["stats"][-1]
    assert last["processed"] == last["total"] == len(paths) and last["parallel"] == 3
    assert last["per_minute"] > 0 and last["eta_seconds"] == 0 and last["done"] == 11 and last["failed"] == 3
    assert "cut short" in s.text()


def test_worker_one_at_a_time_is_in_order_and_skips_existing(tmp_path, server_manager):
    paths = [make_image(tmp_path / f"{i}.png", size=(30 + i, 20)) for i in range(6)]
    caption_io.write_caption(paths[2], "keep me")
    ev = run_job(w.BatchJob(paths=paths, spec=ProviderSpec.make("openai", "m", base_url="http://x/v1"),
                            request=CaptionRequest(prompt="p"), save_mode=w.SAVE_SKIP_EXISTING))
    assert [p for p, _ in ev["done"]] == [str(p) for p in paths if p != paths[2]]
    assert caption_io.read_caption(paths[2]) == "keep me" and ev["summary"].skipped == 1
    assert server_manager[0].peak == 1


def test_worker_pause_resume_and_cancel_keep_finished_work(tmp_path, server_manager):
    paths = [make_image(tmp_path / f"{i}.png", size=(30 + i, 20)) for i in range(40)]
    job = w.BatchJob(paths=paths, spec=ProviderSpec.make("openai", "m", base_url="http://x/v1", concurrency=2),
                     request=CaptionRequest(prompt="p"))

    def before(worker):
        worker.pause()
        assert worker.paused
        threading.Timer(0.5, worker.resume).start()
        worker.item_done.connect(lambda p, t: worker.cancel() if p == str(paths[7]) else None)

    t0 = time.monotonic()
    ev = run_job(job, before)
    s = ev["summary"]
    assert time.monotonic() - t0 >= 0.4                       # nothing started while paused
    assert s.cancelled and not s.fatal and 8 <= s.done < 40
    for p, text in ev["done"]:
        assert caption_io.read_caption(p) == text             # every reported caption is on disk


def test_chunked_providers_still_work_and_report_stats(tmp_path, monkeypatch):
    class Local(Provider):
        preferred_chunk = 3

        def __init__(self, spec):
            super().__init__(spec)
            self.seen = []

        def prepare(self, image, request):
            return ("ready", image.size)

        def generate(self, images, request, cancel=None, prompts=None):
            self.seen.append(list(images))
            return [f"c{size[0]}" for _tag, size in images]

    m = mgr_mod.ModelManager()
    monkeypatch.setattr(mgr_mod, "_manager", m)
    made = []
    monkeypatch.setattr(mgr_mod, "create_provider", lambda spec: made.append(Local(spec)) or made[-1])
    paths = [make_image(tmp_path / f"{i}.png", size=(30 + i, 20)) for i in range(7)]
    ev = run_job(w.BatchJob(paths=paths, spec=ProviderSpec.make("transformers", "x"),
                            request=CaptionRequest(prompt="p", max_image_side=0)))
    assert ev["summary"].done == 7 and [len(c) for c in made[0].seen] == [3, 3, 1]
    assert made[0].seen[0][0] == ("ready", (30, 20))          # prepare ran ahead; generate got its result
    assert [p for p, _ in ev["done"]] == [str(p) for p in paths]
    assert len(ev["stats"]) == 3 and ev["stats"][-1]["processed"] == 7 and ev["stats"][-1]["parallel"] == 0


# -- the real client against a simulated server ----------------------------------------------------------------------
def _spec(server, **options):
    return ProviderSpec.make("openai", "mock-vlm", base_url=server.base_url, timeout=20.0, max_retries=0, **options)


def _pictures(tmp_path, count, size=(96, 64)):
    return [make_image(tmp_path / f"s{i:03d}.png", size=(size[0] + i % 5, size[1]), color=(10 * (i % 20), 90, 160))
            for i in range(count)]


def _run(server, paths, request=None, **options):
    from inference.openai_api import OpenAICompatible
    prov = OpenAICompatible(_spec(server, **options))
    prov.load(lambda m: None)
    request = request or CaptionRequest(prompt="Describe this image.", params=GenerationParams(max_new_tokens=200))
    limiter = prov.make_limiter()

    def go():
        try:
            return list(run_pipeline(
                paths, lambda p: prov.prepare(image_prep.load_rgb(p, request.max_image_side,
                                                                  max_pixels=request.max_image_pixels), request),
                lambda prepared, _p, stop: prov.caption_prepared(prepared, request, stop), limiter))
        finally:
            prov.unload()

    return within(60, go), limiter, prov


def test_real_client_fixed_parallel_uses_the_slots(tmp_path):
    with bench.MockServer(slots=4, time_scale=0.03, out_tokens=40) as server:
        out, limiter, prov = _run(server, _pictures(tmp_path, 16), concurrency=4)
        stats = server.stats
    assert all(isinstance(o.result, str) and o.result for o in out) and len(out) == 16
    assert stats["peak_active"] == 4 and stats["peak_waiting"] == 0 and stats["requests"] == 16
    assert prov.totals["requests"] == 16 and prov.totals["completion_tokens"] == stats["reply_tokens"]
    assert stats["cached_tokens"] > 0                         # the instruction comes first: a prefix cache can reuse it


def test_real_client_finds_the_slots_by_measuring(tmp_path):
    with bench.MockServer(slots=3, time_scale=0.1, out_tokens=60, out_cv=0.25, vision_s_per_mp=0.0) as server:
        out, limiter, _ = _run(server, _pictures(tmp_path, 60), auto_parallel=True, max_parallel=6)
        stats = server.stats
    assert all(isinstance(o.result, str) for o in out)
    # Real threads and real sleeps: on a quiet machine this ends at exactly 3 (the event-model tests above prove
    # the exact behaviour); on a machine under heavy load a window can be noisy, so only the essentials are
    # asserted here: it went parallel, and it did not run away past the server's slots.
    assert 2 <= limiter.limit <= 4, list(limiter.events)
    assert 2 <= stats["peak_active"] <= 3 and any("kept" in e for e in limiter.events)


def test_real_client_asks_a_llama_cpp_server_for_its_slots(tmp_path):
    with bench.MockServer(slots=2, time_scale=0.01, out_tokens=20, flavour="llamacpp") as server:
        from inference.openai_api import OpenAICompatible
        from inference.server_probe import advice, probe
        prov = OpenAICompatible(_spec(server, auto_parallel=True))
        assert prov.make_limiter().cap == 2
        info = within(10, lambda: probe(server.base_url, "mock-vlm"))
        assert info.kind == "llama.cpp" and info.slots == 2 and info.loaded and info.vision
        assert any("2 request(s) at a time" in line for line in advice(info, 500, 50, 200, parallel=4))
    with bench.MockServer(slots=2, time_scale=0.01, context=4096) as server:
        info = within(10, lambda: probe(server.base_url, "mock-vlm"))
        assert info.kind == "lmstudio" and info.context == 4096 and info.slots is None and info.vision
        assert "LM Studio" in info.summary() and "context 4096" in info.summary()
        assert any("holds 2" in line for line in advice(info, 1200, 100, 400, parallel=4))
        assert not advice(info, 300, 50, 200, parallel=4)
        assert within(10, lambda: probe(server.base_url, "other")).notes
    assert not within(10, lambda: probe("http://127.0.0.1:9/v1", "m", timeout=0.5)).reachable


def test_real_client_stops_a_repeating_reply_early(tmp_path):
    paths = _pictures(tmp_path, 6)
    request = CaptionRequest(prompt="Describe this image.", params=GenerationParams(max_new_tokens=400))
    with bench.MockServer(slots=2, time_scale=0.01, loop_rate=1.0) as server:
        out, _, prov = _run(server, paths, request, concurrency=2, loop_guard=True)
        assert wait_for(lambda: server.stats["aborted"] == 6)       # the server noticed every closed stream
        guarded = server.stats
    assert all(o.result == "A figure stands in a plain room, wearing a red scarf" for o in out)
    assert all(o.info["cut"] for o in out) and prov.totals["loops_cut"] == 6
    assert guarded["stream_requests"] == 6 and guarded["aborted"] == 6
    assert guarded["reply_tokens"] < 0.35 * 6 * 400            # the server stopped generating
    with bench.MockServer(slots=2, time_scale=0.01, loop_rate=1.0) as server:
        out, _, _ = _run(server, paths[:2], request, concurrency=2)          # the default: the whole reply
        plain = server.stats
    assert plain["reply_tokens"] == 2 * 400 and all("a red scarf, a red scarf" in o.result for o in out)
    with bench.MockServer(slots=2, time_scale=0.01, out_tokens=30) as server:
        out, _, _ = _run(server, paths, concurrency=2, loop_guard=True)      # a normal reply, streamed
    assert all(isinstance(o.result, str) and not o.info["cut"] and o.info["size"] > 20 for o in out)
    assert all("first_text_seconds" in o.info for o in out)


def test_real_client_backs_off_when_the_context_is_too_small_and_loses_nothing(tmp_path):
    # the shared context holds two requests; the third of the three sent at the start is refused by the server
    paths = _pictures(tmp_path, 24)
    request = CaptionRequest(prompt="Describe this image.", params=GenerationParams(max_new_tokens=60))
    with bench.MockServer(slots=4, time_scale=0.08, out_tokens=40, out_cv=0.0, context=110) as server:
        out, limiter, _ = _run(server, paths, request, concurrency=3, auto_parallel=True, max_parallel=4)
        stats = server.stats
    assert all(isinstance(o.result, str) for o in out), [str(o.result) for o in out if not isinstance(o.result, str)]
    assert stats["errors"] >= 1 and max(o.attempts for o in out) == 2
    assert limiter.limit <= 2 and any("overloaded" in e for e in limiter.events)


def test_image_first_and_the_message_layout(tmp_path):
    from inference.openai_api import OpenAICompatible
    req = CaptionRequest(prompt="Describe.", system_prompt=" Be brief. ")
    default = OpenAICompatible(ProviderSpec.make("openai", "m", base_url="http://x/v1"))._messages("data:x", req)
    assert default[0] == {"role": "system", "content": "Be brief."}
    assert [p["type"] for p in default[1]["content"]] == ["text", "image_url"]
    first = OpenAICompatible(ProviderSpec.make("openai", "m", base_url="http://x/v1", image_first=True))
    assert [p["type"] for p in first._messages("data:x", req)[1]["content"]] == ["image_url", "text"]


def test_new_options_do_not_rebuild_the_client():
    a = ProviderSpec.make("openai", "m", base_url="http://x/v1", concurrency=1)
    b = ProviderSpec.make("openai", "m", base_url="http://x/v1", concurrency=4, auto_parallel=True, max_parallel=6,
                          loop_guard=True, image_first=True, jpeg_quality=90)
    assert mgr_mod._cache_key(a) == mgr_mod._cache_key(b)
    assert mgr_mod._cache_key(a) != mgr_mod._cache_key(ProviderSpec.make("openai", "m", base_url="http://y/v1"))


# -- the benchmark tool ----------------------------------------------------------------------------------------------
def test_bench_pictures_and_fact_scoring(tmp_path):
    made = bench.make_pictures(tmp_path / "pics", 4, seed=3, sizes=((160, 120), (120, 160)))
    assert len(made) == 4 and all(p.is_file() for p, _ in made)
    path, facts = made[0]
    with Image.open(path) as im:
        assert im.size == (160, 120)
    perfect = (f"A {facts['colour']} {facts['shape']} on a {facts['background']} background, the text reads "
               f"{facts['word']} and, in small print, serial {facts['small']}.")
    assert bench.facts_found(perfect, facts) == (5, 5)
    assert bench.facts_found(perfect.replace(facts["small"], "..."), facts) == (4, 5)
    assert bench.facts_found("An abstract picture.", facts)[0] == 0
    assert bench.make_pictures(tmp_path / "again", 4, seed=3, sizes=((160, 120), (120, 160)))[0][1] == facts


def test_bench_runs_both_pipelines_against_the_simulated_server(tmp_path):
    paths = [p for p, _ in bench.make_pictures(tmp_path / "pics", 6, seed=1, sizes=((320, 240), (240, 320)))]
    with bench.MockServer(slots=2, time_scale=0.01, out_tokens=30) as server:
        common = dict(base_url=server.base_url, model="mock-vlm", api_key="", prompt="Describe.", max_tokens=100,
                      timeout=20.0)
        old = within(60, lambda: bench.run_legacy(paths, **common))
        new = within(60, lambda: bench.run_new(paths, concurrency=2, max_pixels=40_000, **common))
        stats = server.stats
    assert len(old["results"]) == len(new["results"]) == 6
    assert all(isinstance(t, str) and t for _, t in old["results"] + new["results"])
    assert stats["requests"] == 12 and stats["peak_active"] == 2
    row = bench._row("new", new, None)
    assert row["ok"] == 6 and row["per_minute"] > 0 and row["facts"] == "n/a"
    assert isinstance(Outcome("k", "x"), Outcome)
