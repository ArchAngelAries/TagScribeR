"""Regression tests for audit 10 (inference): per-model custom code, partial downloads, API timeouts and
cancellation, malformed replies, key/host handling, gated models. No real models, no network."""
import json
import os
import threading
import time
from types import SimpleNamespace

import pytest

from core import caption_io, paths
from inference import manager as mgr_mod
from inference import models as catalog
from inference import openai_api, transformers_vlm
from inference import worker as w
from inference.base import Cancelled, CaptionRequest, InferenceError, Provider, ProviderSpec
from tests.conftest import make_image


@pytest.fixture(scope="module", autouse=True)
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


# -- 1. custom code is a per-model decision ------------------------------------------------------------------------
def _model_dir(tmp_path, model_type, auto_map=True):
    d = tmp_path / "m"
    d.mkdir()
    cfg = {"model_type": model_type}
    if auto_map:
        cfg["auto_map"] = {"AutoModelForCausalLM": "modeling_x.X"}
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def test_needs_remote_code_only_for_unknown_architectures():
    pytest.importorskip("transformers")
    assert not transformers_vlm.needs_remote_code({"model_type": "llava", "auto_map": {"A": "x.Y"}})
    assert not transformers_vlm.needs_remote_code({"model_type": "llava"})
    assert transformers_vlm.needs_remote_code({"model_type": "totally_new_arch", "auto_map": {"A": "x.Y"}})
    assert not transformers_vlm.needs_remote_code({"model_type": "totally_new_arch"})


class _Stop(Exception):
    pass


def _load_capturing(monkeypatch, tmp_path, model_type, trust_opt):
    pytest.importorskip("transformers")
    import transformers
    from core import hardware
    seen = {}

    def fake_from_pretrained(*a, **kw):
        seen.update(kw)
        raise _Stop()

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(hardware, "torch_installed", lambda: True)
    d = _model_dir(tmp_path, model_type)
    prov = transformers_vlm.TransformersVLM(ProviderSpec.make("transformers", str(d), trust_remote_code=trust_opt))
    with pytest.raises(InferenceError) as ei:
        prov.load(lambda m: None)
    return seen, str(ei.value)


def test_builtin_architecture_never_gets_trust_remote_code(monkeypatch, tmp_path):
    seen, _ = _load_capturing(monkeypatch, tmp_path, "llava", trust_opt=True)   # even with the opt-in set
    assert seen["trust_remote_code"] is False


def test_custom_architecture_needs_explicit_per_model_opt_in(monkeypatch, tmp_path):
    seen, msg = _load_capturing(monkeypatch, tmp_path, "totally_new_arch", trust_opt=False)
    assert not seen and "custom code" in msg                 # refused before anything was loaded
    (tmp_path / "second").mkdir()
    seen, _ = _load_capturing(monkeypatch, tmp_path / "second", "totally_new_arch", trust_opt=True)
    assert seen["trust_remote_code"] is True


def test_trust_is_remembered_by_model_and_revision():
    key = transformers_vlm.trust_key("org/model", "abc123")
    assert transformers_vlm.is_trusted([key], "org/model", "abc123")
    assert not transformers_vlm.is_trusted([key], "org/model", "other")
    assert not transformers_vlm.is_trusted([key], "org/another", "abc123")
    assert not transformers_vlm.is_trusted([], "org/model")


def test_catalog_download_skips_py_unless_model_needs_remote_code(monkeypatch, tmp_path):
    import huggingface_hub
    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: calls.append(kw))
    catalog.download_model("org/model", tmp_path / "t")
    catalog.download_model("org/model", tmp_path / "t", revision="v1", remote_code=True)
    assert "*.py" not in calls[0]["allow_patterns"] and calls[0]["revision"] is None
    assert "*.py" in calls[1]["allow_patterns"] and calls[1]["revision"] == "v1"
    assert not any(c.remote_code for c in catalog.CATALOG)


# -- 2. partial downloads ---------------------------------------------------------------------------------------------
def _catalog_folder(root, shards=("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"), present=1):
    root.mkdir(parents=True)
    (root / "config.json").write_text('{"model_type": "qwen3_vl"}')
    (root / "preprocessor_config.json").write_text("{}")
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": shards[0], "b": shards[1]}}))
    for s in shards[:present]:
        (root / s).write_bytes(b"x" * 16)
    return root


def test_partial_download_is_not_installed_and_can_resume(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DEFAULT_MODELS_DIR", tmp_path / "models")
    folder = _catalog_folder(tmp_path / "models" / "Qwen3-VL-4B-Instruct")
    assert catalog._is_vlm_folder(folder) is None
    assert catalog.catalog_local_path("Qwen/Qwen3-VL-4B-Instruct") is None
    assert catalog.catalog_partial_path("Qwen/Qwen3-VL-4B-Instruct") == folder
    assert catalog.discover_local_models() == []
    (folder / "model-00002-of-00002.safetensors").write_bytes(b"x" * 16)
    assert catalog._is_vlm_folder(folder) == "qwen3_vl"
    assert catalog.catalog_local_path("Qwen/Qwen3-VL-4B-Instruct") == folder
    assert catalog.catalog_partial_path("Qwen/Qwen3-VL-4B-Instruct") is None
    assert [m.name for m in catalog.discover_local_models()] == ["Qwen3-VL-4B-Instruct"]


def test_empty_shard_pending_file_and_unindexed_shards_are_incomplete(tmp_path):
    f = _catalog_folder(tmp_path / "a", present=2)
    (f / "model-00002-of-00002.safetensors").write_bytes(b"")
    assert not catalog.weights_complete(f)
    f = _catalog_folder(tmp_path / "b", present=2)
    assert catalog.weights_complete(f)
    pending = f / ".cache" / "huggingface" / "download"
    pending.mkdir(parents=True)
    (pending / "x.incomplete").write_bytes(b"1")
    assert catalog.weights_complete(f)         # a leftover of some other file: the index says the weights are whole
    f = tmp_path / "c"
    f.mkdir()
    (f / "model-00001-of-00003.safetensors").write_bytes(b"x")       # no index file: judged by the file names
    (f / "model-00002-of-00003.safetensors").write_bytes(b"x")
    assert not catalog.weights_complete(f)
    (f / "model-00003-of-00003.safetensors").write_bytes(b"x")
    assert catalog.weights_complete(f)
    single = tmp_path / "d"
    single.mkdir()
    (single / "model.safetensors").write_bytes(b"x")
    assert catalog.weights_complete(single)


# -- 3. API timeout / retries / cancellation --------------------------------------------------------------------
def test_retry_defaults_and_profile_values():
    assert openai_api.effective_retries("http://127.0.0.1:1234/v1", None) == 0
    assert openai_api.effective_retries("http://127.0.0.1:1234/v1", 3) == 1          # never a cascade on a local server
    assert openai_api.effective_retries("http://127.0.0.1:1234/v1", 0) == 0
    assert openai_api.effective_retries("https://api.example.com/v1", None) == openai_api.CLOUD_RETRIES
    assert openai_api.effective_retries("https://api.example.com/v1", 5) == 5


def test_client_uses_spec_timeout_and_retries(monkeypatch):
    import openai
    made = {}

    class FakeOpenAI:
        def __init__(self, **kw):
            made.update(kw)

        def close(self):
            pass

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    prov = openai_api.OpenAICompatible(ProviderSpec.make("openai", "m", base_url="http://localhost:1234/v1",
                                                         timeout=42.0, max_retries=3))
    prov.load(lambda m: None)
    assert made["timeout"] == 42.0 and made["max_retries"] == 1
    prov2 = openai_api.OpenAICompatible(ProviderSpec.make("openai", "m", base_url="https://api.example.com/v1",
                                                          timeout=30.0, max_retries=4))
    prov2.load(lambda m: None)
    assert made["max_retries"] == 4


class _Completions:
    def __init__(self, behaviour):
        self.behaviour = behaviour

    def create(self, **kw):
        return self.behaviour(**kw)


class _FakeClient:
    def __init__(self, behaviour):
        self.chat = SimpleNamespace(completions=_Completions(behaviour))
        self.closed = threading.Event()

    def close(self):
        self.closed.set()


def _reply(content, finish="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)])


def _api_provider(behaviour):
    prov = openai_api.OpenAICompatible(ProviderSpec.make("openai", "m", base_url="http://127.0.0.1:1/v1"))
    prov.client = _FakeClient(behaviour)
    return prov


def _img():
    from PIL import Image
    return Image.new("RGB", (8, 8))


REQ = CaptionRequest(prompt="describe")


def test_cancel_aborts_request_in_flight():
    release = threading.Event()
    prov = _api_provider(lambda **kw: release.wait(30) and _reply("late"))
    client = prov.client
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    out = prov.generate([_img()], REQ, cancel)
    release.set()
    assert isinstance(out[0], Cancelled) and time.monotonic() - t0 < 5
    assert client.closed.is_set() and prov.client is None        # connection dropped; load() would rebuild it


def test_list_content_parts_are_joined():
    parts = [{"type": "text", "text": "a cat "}, {"type": "text", "text": "on a mat"}, {"type": "other"}]
    assert _api_provider(lambda **kw: _reply(parts)).generate([_img()], REQ)[0] == "a cat on a mat"
    objs = [SimpleNamespace(text="hello")]
    assert _api_provider(lambda **kw: _reply(objs)).generate([_img()], REQ)[0] == "hello"


@pytest.mark.parametrize("content", [7, {"x": 1}])
def test_malformed_reply_is_an_inference_error_not_a_crash(content):
    out = _api_provider(lambda **kw: _reply(content)).generate([_img()], REQ)
    assert isinstance(out[0], InferenceError) and "malformed" in str(out[0])


def test_manager_session_wait_is_cancellable_and_unload_is_bounded():
    m = mgr_mod.ModelManager()
    held, release = threading.Event(), threading.Event()

    def hold():
        with m._lock:
            held.set()
            release.wait(10)

    t = threading.Thread(target=hold, daemon=True)
    t.start()
    held.wait(2)
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        with m.session(ProviderSpec.make("openai", "m", base_url="http://x/v1"), lambda s: None, cancel):
            pass
    assert time.monotonic() - t0 < 5
    t0 = time.monotonic()
    assert m.unload_all(blocking=True, timeout=0.3) is False and time.monotonic() - t0 < 3
    release.set()
    t.join(2)
    assert m.unload_all(blocking=True, timeout=1.0) is True


# -- 4 + 6. one bad reply fails one image; working badges ------------------------------------------------------------
class _Exploding(Provider):
    preferred_chunk = 1

    def generate(self, images, request, cancel=None, prompts=None):
        if any(i.size == (13, 13) for i in images):
            raise TypeError("content is a list")
        return [f"ok {i.size[0]}" for i in images]


@pytest.mark.parametrize("chunk", [1, 3])
def test_unexpected_provider_error_fails_only_that_image(tmp_path, monkeypatch, chunk):
    m = mgr_mod.ModelManager()
    monkeypatch.setattr(mgr_mod, "_manager", m)

    def make(spec):
        p = _Exploding(spec)
        p.preferred_chunk = chunk
        return p

    monkeypatch.setattr(mgr_mod, "create_provider", make)
    a = make_image(tmp_path / "a.png", size=(20, 20))
    bad = make_image(tmp_path / "bad.png", size=(13, 13))
    b = make_image(tmp_path / "b.png", size=(22, 20))
    job = w.BatchJob(paths=[a, bad, b], spec=ProviderSpec.make("transformers", "x", batch_size=chunk),
                     request=CaptionRequest(prompt="x"))
    worker = w.BatchWorker(job)
    started, failed, done = [], [], []
    worker.items_started.connect(lambda keys: started.extend(keys))
    worker.item_failed.connect(lambda p, r: failed.append((p, r)))
    worker.item_done.connect(lambda p, t: done.append(p))
    summary = []
    worker.finished.connect(summary.append)
    worker.run()
    s = summary[0]
    assert not s.fatal and s.done == 2 and len(s.failed) == 1
    assert failed[0][0] == str(bad) and "Unexpected error" in failed[0][1]
    assert caption_io.read_caption(b) == "ok 22"
    assert set(started) == {str(a), str(bad), str(b)}              # item 8: the "working" badge signal is emitted


# -- 5. key and host ---------------------------------------------------------------------------------------------------
def test_key_exposure_warning():
    f = openai_api.key_exposure_warning
    assert f("http://192.168.1.5:1234/v1", "k") and "192.168.1.5" in f("http://192.168.1.5:1234/v1", "k")
    assert f("http://api.example.com/v1", "k")
    assert not f("http://127.0.0.1:1234/v1", "k") and not f("http://localhost:1234/v1", "k")
    assert not f("https://api.example.com/v1", "k")
    assert not f("http://192.168.1.5:1234/v1", "")


@pytest.fixture
def tab(qapp):
    from tabs.caption import CaptionTab
    t = CaptionTab()
    yield t
    t.close()


def test_editing_the_url_host_clears_the_key(tab):
    from core.secrets import ApiProfile
    tab.profiles.profiles["cloud"] = ApiProfile("cloud", "https://api.example.com/v1", "m", api_key="sk-secret")
    tab.combo_presets.addItem("cloud")
    tab.combo_presets.setCurrentText("cloud")
    assert tab.inp_api_key.text() == "sk-secret"
    tab.inp_api_url.setText("https://api.example.com/v1/")      # programmatic: no textEdited
    tab._on_api_url_edited()                                    # same host: key stays
    assert tab.inp_api_key.text() == "sk-secret"
    tab.inp_api_url.setText("http://192.168.1.9:1234/v1")
    tab._on_api_url_edited()
    assert tab.inp_api_key.text() == ""
    tab.inp_api_key.setText("typed")
    tab._on_api_key_edited()
    assert "192.168.1.9" in tab.lbl_api_warn.text()             # plain http to another machine with a key


def test_api_spec_takes_limits_from_profile(tab):
    from core.secrets import ApiProfile
    tab.profiles.profiles["p"] = ApiProfile("p", "http://127.0.0.1:1234/v1", "m", timeout=77.0, max_retries=0)
    tab.combo_presets.addItem("p")
    tab.combo_presets.setCurrentText("p")
    tab.tab_source.setCurrentIndex(1)
    spec = tab._build_spec()
    assert spec.opt("timeout") == 77.0 and spec.opt("max_retries") == 0


# -- 7 + 8. gated models; revisions ---------------------------------------------------------------------------------
def test_gated_models_are_flagged_and_explained():
    gemma = [c for c in catalog.CATALOG if c.repo_id.startswith("google/gemma")]
    assert gemma and all(c.gated for c in gemma)

    class GatedRepoError(Exception):
        pass

    msg = catalog.describe_download_error(GatedRepoError("401 Client Error. Cannot access gated repo"), gemma[0])
    assert f"https://huggingface.co/{gemma[0].repo_id}" in msg and "hf auth login" in msg and "HF_TOKEN" in msg
    assert catalog.describe_download_error(OSError("disk full"), gemma[0]) == "disk full"


def test_no_catalog_revisions_are_pinned_yet():
    # No commit hashes can be verified offline, so none are invented; the plumbing is in place (CatalogModel.revision).
    assert all(c.revision == "" for c in catalog.CATALOG)


# ---- from the Fable review of this batch ------------------------------------------------------------------
def test_legacy_profile_defaults_become_automatic(tmp_path, monkeypatch):
    import json
    from core import secrets
    f = tmp_path / "api.json"
    f.write_text(json.dumps({"lm": {"base_url": "http://localhost:1234/v1", "model": "m", "timeout": 120.0,
                                    "max_retries": 3},
                             "custom": {"base_url": "https://api.example.com/v1", "model": "m", "timeout": 60.0,
                                        "max_retries": 1}}), encoding="utf-8")
    monkeypatch.setattr(secrets, "get_secret", lambda *_a, **_k: "")
    store = secrets.ApiProfileStore(f)
    lm, custom = store.profiles["lm"], store.profiles["custom"]
    assert (lm.timeout, lm.max_retries) == (180.0, None)          # the old defaults: automatic now
    assert (custom.timeout, custom.max_retries) == (60.0, 1)     # a value someone chose is kept


def test_a_stale_incomplete_file_does_not_hide_a_complete_model(tmp_path):
    from inference import models
    d = tmp_path / "m"
    (d / ".cache" / "huggingface" / "download").mkdir(parents=True)
    (d / ".cache" / "huggingface" / "download" / "abc.etag.incomplete").write_bytes(b"x")
    (d / "model.safetensors").write_bytes(b"weights")
    assert models.weights_complete(d)
