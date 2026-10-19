"""Fizgig 8.x trainer-loop features (the "wave 2" catch-up): resume-state retry, the test compile and eager previews,
the per-step loss log, Sample now / Save now, the recorded preview prompts, slider previews and live slider prompts, and
the preview checkpoint's typed Steps / CFG. Tiny random models only (tests/tiny_family.py); no GPU, no real weights."""
import dataclasses
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests.test_training_smoke import _run_stages, _values  # noqa: E402
from tests.tiny_family import TinyDiT, TinyDriver, make_dataset, register, write_models  # noqa: E402
from training import compile as tcompile  # noqa: E402
from training import pipeline, step_log, train  # noqa: E402


@pytest.fixture
def setup(tmp_path):
    desc = register()
    data = make_dataset(str(tmp_path / "scratch_dataset"))
    models = write_models(str(tmp_path / "models"))
    return desc, data, models, tmp_path


def _quick(tmp, **over):
    """A small run: no Adaptive LR, no loss watch, no previews unless asked."""
    over = {"SAMPLE_ENABLED": False, "SAMPLE_AT_FIRST": False, **over}
    return _values(tmp, ADAPTIVE_LR=False, KREA2_LOSS_WATCH=False, **over)


# ---- 1. the resume state: one retry, a half-written folder removed, a pause still raises ---------------------------
def test_state_save_is_retried_once(setup, monkeypatch, caplog):
    desc, data, models, tmp = setup
    monkeypatch.setattr(train, "STATE_RETRY_WAIT_S", 0.0)
    real, calls = train._save_state, []

    def flaky(*a, **k):
        calls.append(k["epoch"])
        if len(calls) == 1:
            raise OSError("sync program holds the file")
        return real(*a, **k)
    monkeypatch.setattr(train, "_save_state", flaky)
    run = pipeline.build_run(desc, _quick(tmp, MAX_TRAIN_EPOCHS=2), data, models)
    with caplog.at_level("WARNING"):
        _run_stages(run)
    assert calls[:2] == [1, 1] and (run.run_dir / "tiny_lora-000001-state" / "training_state.json").exists()
    assert "trying again" in caplog.text and (run.run_dir / "tiny_lora.safetensors").exists()


def test_state_save_that_keeps_failing_warns_and_removes_the_half_folder(setup, monkeypatch, caplog):
    desc, data, models, tmp = setup
    monkeypatch.setattr(train, "STATE_RETRY_WAIT_S", 0.0)

    def broken(output_dir, output_name, net, optimizer, *, epoch, **k):
        d = Path(output_dir, f"{output_name}-{epoch:06d}-state")
        d.mkdir(parents=True, exist_ok=True)
        (d / "lora.safetensors").write_bytes(b"half")
        raise OSError("disk full")
    monkeypatch.setattr(train, "_save_state", broken)
    run = pipeline.build_run(desc, _quick(tmp, MAX_TRAIN_EPOCHS=2, SAVE_STATE_ON_TRAIN_END=False), data, models)
    with caplog.at_level("WARNING"):
        _run_stages(run)
    assert not list(run.run_dir.glob("*-state")) and "could not be saved" in caplog.text
    assert (run.run_dir / "tiny_lora-000001.safetensors").exists() and (run.run_dir / "tiny_lora.safetensors").exists()


def test_a_pause_needs_its_state(setup, monkeypatch):
    desc, data, models, tmp = setup
    monkeypatch.setattr(train, "STATE_RETRY_WAIT_S", 0.0)

    def broken(output_dir, output_name, net, optimizer, *, epoch, **k):
        Path(output_dir, f"{output_name}-{epoch:06d}-state").mkdir(parents=True, exist_ok=True)
        raise OSError("disk full")
    monkeypatch.setattr(train, "_save_state", broken)
    run = pipeline.build_run(desc, _quick(tmp, MAX_TRAIN_EPOCHS=3, SAVE_STATE=False), data, models)
    pipeline.request_pause(run.run_dir)
    with pytest.raises(OSError):
        _run_stages(run)
    assert not list(run.run_dir.glob("*-state"))


# ---- 2. test compile, eager previews -------------------------------------------------------------------------------
def test_test_compile_failure_names_the_build_tools(monkeypatch, caplog):
    monkeypatch.setattr(tcompile, "_COMPILE_WORKS", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)

    def broken_compile(fn, *a, **k):
        def run(*x):
            raise RuntimeError("Failed to find C compiler")
        return run
    monkeypatch.setattr(torch, "compile", broken_compile)
    with caplog.at_level("WARNING"):
        assert tcompile.compile_works() is False
    assert "vs_BuildTools.exe" in caplog.text and "restart TagScribeR" in caplog.text
    caplog.clear()
    assert tcompile.compile_works() is False and not caplog.text      # asked once per process
    # compile_blocks stops before touching a block
    monkeypatch.setattr(tcompile, "_triton_importable", lambda: True)
    monkeypatch.setattr(tcompile, "triton_matches_torch", lambda *a: (True, ""))
    monkeypatch.setattr(tcompile, "find_host_compiler", lambda: True)
    holder = type("D", (), {"blocks": torch.nn.ModuleList([torch.nn.Linear(2, 2)])})()
    assert tcompile.compile_blocks(holder, 0) == 0 and isinstance(holder.blocks[0], torch.nn.Linear)
    monkeypatch.setattr(tcompile, "_COMPILE_WORKS", None)             # do not leak the cached False


def test_test_compile_passes_without_a_gpu(monkeypatch):
    monkeypatch.setattr(tcompile, "_COMPILE_WORKS", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert tcompile.compile_works() is True
    monkeypatch.setattr(tcompile, "_COMPILE_WORKS", None)


def _render(tmp_path, **over):
    from types import SimpleNamespace
    from training.driver import FamilyDriver
    gen = []
    driver = SimpleNamespace(generate=lambda *a, **k: gen.append(k) or None,
                             decode=lambda *a: SimpleNamespace(save=lambda p: None),
                             block_swap_mode=lambda *a, **k: None, park_for=lambda *a: None, unpark=lambda *a: None,
                             save_preview=lambda r, p: FamilyDriver.save_preview(None, r, p))
    net = SimpleNamespace(set_enabled=lambda *a: None, move_adapter=lambda *a: None,
                          set_trainable_multiplier=lambda *a: None)
    kw = dict(output_name="x", steps=1, cfg=1.0, neg=None, width=8, height=8, seed=1)
    kw.update(over)
    train._render_previews(driver, torch.nn.Linear(2, 2), net, None, [{}], str(tmp_path), 1, **kw)
    return gen


def test_previews_render_under_force_eager(tmp_path, monkeypatch):
    events = []

    class Stance:
        def __init__(self, mode):
            events.append(("set", mode))

        def __enter__(self):
            events.append("enter")

        def __exit__(self, *a):
            events.append("exit")

    monkeypatch.setattr(torch.compiler, "set_stance", Stance, raising=False)
    _render(tmp_path)
    assert events == [("set", "force_eager"), "enter", "exit"]
    events.clear()
    from types import SimpleNamespace

    def oom(*a, **k):
        raise RuntimeError("oom")
    with pytest.raises(RuntimeError):                       # a failed render still leaves the stance
        train._render_previews(SimpleNamespace(generate=oom), torch.nn.Linear(2, 2),
                               SimpleNamespace(set_enabled=lambda *a: None), None, [{}], str(tmp_path), 1,
                               output_name="x", steps=1, cfg=1.0, neg=None, width=8, height=8, seed=1)
    assert events[-1] == "exit"


def test_previews_work_on_a_torch_without_set_stance(tmp_path, monkeypatch):
    monkeypatch.delattr(torch.compiler, "set_stance", raising=False)
    assert len(_render(tmp_path)) == 1


# ---- 3. the per-step loss log -------------------------------------------------------------------------------------
def test_step_log_batches_resumes_and_thins(tmp_path):
    log = step_log.StepLog(str(tmp_path), "m")
    for s in range(1, 25):
        log.add(s, 1 if s <= 10 else 2, 0.5 / s, 1e-4 if s % 20 == 0 else None)
    on_disk = Path(log.path).read_text(encoding="utf-8").splitlines()
    assert len(on_disk) == 20 and json.loads(on_disk[19]) == {"s": 20, "e": 2, "l": round(0.5 / 20, 6), "lr": 0.0001}
    log.flush()
    assert len(Path(log.path).read_text(encoding="utf-8").splitlines()) == 24
    assert Path(log.path).name == ".tagscriber_loss_m.jsonl"
    # a resume at step 12 keeps the lines up to it; a fresh run starts empty
    again = step_log.StepLog(str(tmp_path), "m", start_step=12)
    again.add(13, 2, 0.1)
    again.flush()
    rows = [json.loads(x)["s"] for x in Path(again.path).read_text(encoding="utf-8").splitlines()]
    assert rows == list(range(1, 13)) + [13]
    assert Path(step_log.StepLog(str(tmp_path), "m").path).read_text(encoding="utf-8") == ""
    # the graph's data: every k-th above the cap, the last kept, the epoch starts from every line
    big = step_log.StepLog(str(tmp_path), "big")
    for s in range(1, 4502):
        big.add(s, 1 + s // 1000, 1.0)
    big.flush()
    d = pipeline.read_step_log(tmp_path, "big")
    assert d["stride"] == 2 and d["points"][-1][0] == 4501 and len(d["points"]) <= 2252
    assert d["epochs"][0] == [1, 1] and d["epochs"][1] == [2, 1000] and d["steps"] == 4501
    assert pipeline.read_step_log(tmp_path, "nothing") is None


def test_a_run_writes_its_loss_log_and_a_resume_keeps_the_curve(setup):
    desc, data, models, tmp = setup
    vals = _quick(tmp, MAX_TRAIN_EPOCHS=6)
    run = pipeline.build_run(desc, vals, data, models)
    pipeline.request_pause(run.run_dir)
    _run_stages(run)
    d = pipeline.read_step_log(run.run_dir, "tiny_lora")
    assert [p[0] for p in d["points"]] == [1, 2, 3, 4] and {p[1] for p in d["points"]} == {1}
    pipeline.clear_pause(run.run_dir)
    state = str(run.run_dir / "tiny_lora-000001-state")
    _run_stages(pipeline.build_run(desc, vals, data, models, resume=state))
    d = pipeline.read_step_log(run.run_dir, "tiny_lora")
    assert [p[0] for p in d["points"]] == list(range(1, 25)) and d["epochs"] == [[e, 4 * e - 3] for e in range(1, 7)]
    assert any(p[3] for p in d["points"] if p[0] % 20 == 0)               # the rate, now and then


# ---- 4. Sample now / Save now -----------------------------------------------------------------------------------
def _at_step(monkeypatch, step, action):
    orig = train.StepLog.add

    def add(self, s, epoch, loss, lr=None):
        orig(self, s, epoch, loss, lr)
        if s == step:
            action()
    monkeypatch.setattr(train.StepLog, "add", add)


def test_sample_now_and_save_now_are_taken_once_between_steps(setup, monkeypatch, caplog):
    desc, data, models, tmp = setup
    vals = _quick(tmp, MAX_TRAIN_EPOCHS=2, SAMPLE_ENABLED=True, SAMPLE_EVERY_N_EPOCHS=2)
    run = pipeline.build_run(desc, vals, data, models)
    pipeline.request_sample_now(run.run_dir)            # left over from before: not this run's
    pipeline.request_save_now(run.run_dir)
    _at_step(monkeypatch, 2, lambda: (pipeline.request_sample_now(run.run_dir), pipeline.request_save_now(run.run_dir)))
    with caplog.at_level("INFO"):
        _run_stages(run)
    assert sorted(p.name for p in run.run_dir.glob("*-step*.safetensors")) == ["tiny_lora-step0000002.safetensors"]
    assert not (run.run_dir / pipeline.SAMPLE_NOW_FILE).exists() and not (run.run_dir / pipeline.SAVE_NOW_FILE).exists()
    epochs = sorted({pipeline.parse_sample_name(p.name)[0] for p in pipeline.samples(run.run_dir)})
    assert epochs == [1, 2]                              # epoch 1 only by Sample now (every 2 epochs otherwise)
    assert "[sample now] previews at step 2" in caplog.text and "[save now] step 2" in caplog.text
    from safetensors import safe_open
    with safe_open(str(run.run_dir / "tiny_lora-step0000002.safetensors"), framework="pt") as f:
        assert f.metadata()["ss_epoch"] == "1"


def test_save_now_with_accumulation_waits_for_the_optimizer_step(setup, monkeypatch):
    desc, data, models, tmp = setup
    vals = _quick(tmp, MAX_TRAIN_EPOCHS=1, GRADIENT_ACCUMULATION=2)
    run = pipeline.build_run(desc, vals, data, models)
    _at_step(monkeypatch, 1, lambda: pipeline.request_save_now(run.run_dir))   # mid-group: taken after step 2
    _run_stages(run)
    assert [p.name for p in run.run_dir.glob("*-step*.safetensors")] == ["tiny_lora-step0000002.safetensors"]


# ---- 5 / 6. the recorded prompts; a slider ignores the override ---------------------------------------------------
def test_record_prompts_is_atomic_merged_and_forgiving(tmp_path, caplog):
    d = tmp_path / "sample"
    d.mkdir()
    train._record_prompts(str(d), "m", 0, ["a", "b"])
    train._record_prompts(str(d), "m", 3, ["c"])
    train._record_prompts(str(d), "m", 3, ["d"])             # a later round of the same epoch replaces it
    assert json.loads((d / ".prompts_m.json").read_text(encoding="utf-8")) == {"0": ["a", "b"], "3": ["d"]}
    assert not (d / ".prompts_m.json.tmp").exists()
    assert pipeline.read_preview_prompts(tmp_path, "m") == {0: ["a", "b"], 3: ["d"]}      # run_dir/sample/.prompts_m.json
    assert pipeline.read_preview_prompts(tmp_path, "other") == {} and pipeline.read_preview_prompts(tmp_path / "x", "m") == {}
    with caplog.at_level("WARNING"):
        train._record_prompts(str(tmp_path / "missing"), "m", 1, ["x"])
    assert "could not record the preview prompts" in caplog.text


def test_each_round_records_its_prompts_and_an_override_is_seen(setup):
    desc, data, models, tmp = setup
    vals = _quick(tmp, MAX_TRAIN_EPOCHS=2, SAMPLE_ENABLED=True, SAMPLE_AT_FIRST=True)
    run = pipeline.build_run(desc, vals, data, models)
    pipeline.write_sample_override(run.run_dir, "tok, override prompt", seed=7, width=64, height=64)
    _run_stages(run)
    rec = pipeline.read_preview_prompts(run.run_dir, "tiny_lora")
    assert sorted(rec) == [0, 1, 2] and all(v == ["tok, override prompt"] for v in rec.values())
    assert pipeline.read_preview_prompts(run.run_dir, "other") == {}


def _slider_values(tmp, **over):
    over = {"SAMPLE_ENABLED": True, **over}
    return _values(tmp, ADAPTIVE_LR=False, KREA2_LOSS_WATCH=False, FAMILY_SLIDER=True, FAMILY_SLIDER_SOURCE="prompts",
                   FAMILY_SLIDER_BASE="a photo", FAMILY_SLIDER_POS="smiling", FAMILY_SLIDER_NEG="frowning",
                   LORA_NAME="live_slider", SAMPLE_AT_FIRST=False, DATASET_MEGAPIXELS="0.1", **over)


def test_slider_previews_ignore_the_sample_override(setup, caplog):
    desc, data, models, tmp = setup
    run = pipeline.build_run(desc, _slider_values(tmp, MAX_TRAIN_EPOCHS=1), "", models)
    pipeline.write_sample_override(run.run_dir, "something else", seed=7, width=64, height=64)
    with caplog.at_level("INFO"):
        _run_stages(run, slider_bank=2, slider_bank_res=64)
    assert "[sample override] ignored - slider previews always show the slider's own prompt" in caplog.text
    assert "[sample override] active" not in caplog.text
    assert pipeline.read_preview_prompts(run.run_dir, "live_slider") == {1: ["a photo"]}
    assert 7 not in {pipeline.parse_sample_name(p.name)[2] for p in pipeline.samples(run.run_dir)}


# ---- 7. live slider prompts ---------------------------------------------------------------------------------------
def test_slider_prompts_change_at_an_epoch_boundary(setup, monkeypatch, caplog):
    from safetensors import safe_open
    desc, data, models, tmp = setup
    run = pipeline.build_run(desc, _slider_values(tmp, MAX_TRAIN_EPOCHS=3), "", models)
    new = {"base": "a new photo", "pos": "laughing", "neg": "crying", "guidance": 3.0}
    _at_step(monkeypatch, 1, lambda: pipeline.write_slider_prompts(run.run_dir, new))
    with caplog.at_level("INFO"):
        _run_stages(run, slider_bank=2, slider_bank_res=64)
    assert "[slider] new prompts from epoch 2: base 'a new photo' | +1 -> 'a new photo laughing'" in caplog.text
    assert caplog.text.count("practice images rendered at 64x64") == 2          # at the start, and again for the new base
    applied = pipeline.read_applied_slider_prompts(run.run_dir)
    assert applied["epoch"] == 1 and applied["base"] == "a new photo" and applied["practice_rerendered"] is True
    assert applied["guidance"] == 3.0
    rec = pipeline.read_preview_prompts(run.run_dir, "live_slider")
    assert rec == {1: ["a photo"], 2: ["a new photo"], 3: ["a new photo"]}      # the previews follow the new base
    with safe_open(str(run.run_dir / "live_slider.safetensors"), framework="pt") as f:
        md = f.metadata()
    assert json.loads(md["ss_slider_prompts"]) == ["a new photo", "a new photo laughing", "a new photo crying"]
    assert md["ss_slider_guidance"] == "3"
    hist = json.loads(md["ss_slider_prompt_history"])
    assert len(hist) == 1 and hist[0]["epoch"] == 1 and hist[0]["guidance"] == 3.0


def test_slider_guidance_alone_does_not_re_render_and_a_bad_file_is_ignored(setup, monkeypatch, caplog):
    desc, data, models, tmp = setup
    run = pipeline.build_run(desc, _slider_values(tmp, MAX_TRAIN_EPOCHS=2, SAMPLE_ENABLED=False), "", models)
    _at_step(monkeypatch, 1, lambda: pipeline.write_slider_prompts(
        run.run_dir, {"base": "a photo", "pos": "smiling", "neg": "frowning", "guidance": 4.5}))
    with caplog.at_level("INFO"):
        _run_stages(run, slider_bank=2, slider_bank_res=64)
    assert caplog.text.count("practice images rendered") == 1
    assert pipeline.read_applied_slider_prompts(run.run_dir)["practice_rerendered"] is False
    assert train._read_slider_live(str(run.run_dir)) == {"base": "a photo", "pos": "smiling", "neg": "frowning",
                                                         "guidance": 4.5}
    (run.run_dir / ".slider_prompts.json").write_text(json.dumps({"base": "x", "pos": "", "neg": "n", "guidance": 2}))
    assert train._read_slider_live(str(run.run_dir)) is None             # all three ends and a guidance are needed
    (run.run_dir / ".slider_prompts.json").write_text("{not json")
    assert train._read_slider_live(str(run.run_dir)) is None
    assert pipeline.read_applied_slider_prompts(tmp / "nowhere") is None


def test_the_practice_bank_goes_through_the_drivers_practice_latent(setup):
    desc = setup[0]
    drv = TinyDriver()
    drv.description = desc
    torch.manual_seed(0)
    dit = TinyDiT().requires_grad_(False)
    from training.lora import FamilyLoRA
    net = FamilyLoRA(dit, drv)
    net.add_trainable(2, 2)
    bank = train._SliderBank(3, 64)
    seen = []
    vae = drv.load_vae(None, "cpu")
    drv.practice_latent = lambda v, lat, res: seen.append(res) or torch.zeros(1, 4, 8, 8)
    cond = {"hidden_states": torch.randn(1, 5, 8)}
    train._render_slider_bank(drv, dit, net, vae, bank, cond, None, device=torch.device("cpu"), res=64, seed=1,
                              steps=2, cfg=1.0, speed=None, swapped=False)
    assert seen == [64, 64, 64] and all(z is not None for z in bank.latents)
    w = next(iter(net.wrapped.values()))
    assert w.scales["lora"] == pytest.approx(1.0)                # the dial is back at 1
    # the default hook is the cache's own encoding: decode, then encode_images
    z = TinyDriver().practice_latent(vae, torch.randn(1, 4, 8, 8), 64)
    assert z.shape[0] == 1 and z.shape[1] == 4 and z.device.type == "cpu"


# ---- 8. the slider strip's label band ----------------------------------------------------------------------------
@pytest.mark.parametrize("size", [64, 256, 1024])
def test_slider_strip_band_fits_its_text(size):
    from PIL import Image
    frames = [Image.new("RGB", (size, size), (0, 0, 0)) for _ in range(3)]
    strip = train._slider_strip(frames, train.SLIDER_PREVIEW_MULTIPLIERS)
    band = strip.height - size
    assert band >= 12 and strip.width == 3 * size + 16
    px = strip.convert("L")
    bright_rows = [y for y in range(strip.height) if px.crop((0, y, strip.width, y + 1)).getextrema()[1] > 100]
    assert bright_rows and max(bright_rows) < band - 2 and min(bright_rows) >= 2    # the whole text, inside the band
    if size == 1024:
        assert band > 30                                                           # grows with the text, not fixed 30


# ---- 9. the preview checkpoint takes a typed Steps / CFG ---------------------------------------------------------
def test_untouched_boxes_send_nothing_for_a_checkpoint_preview(tmp_path):
    from training import registry
    from training.families.klein.description import KLEIN_9B as desc
    if registry.get(desc.key) is None:
        registry.register(desc)
    ck = tmp_path / "distilled.safetensors"
    ck.write_bytes(b"x")
    base = {**pipeline.P.defaults(), "SAMPLE_ENABLED": True, "SAMPLE_EVERY_N_EPOCHS": 1, "SAMPLE_PROMPT": "a cat"}
    models = {"klein_distilled_dit": str(ck)}

    def kw_for(**over):
        return pipeline.train_kwargs(desc, {**base, **over}, tmp_path / "run", models)[0]
    kw = kw_for()
    assert kw["preview_checkpoint"] == str(ck) and "sample_steps" not in kw and "sample_cfg_scale" not in kw
    kw = kw_for(SAMPLE_STEPS=desc.preview_steps, SAMPLE_CFG_SCALE=desc.preview_cfg, SAMPLE_NEGATIVE="blurry")
    assert "sample_steps" not in kw and "sample_cfg_scale" not in kw and "sample_negative" not in kw
    kw = kw_for(SAMPLE_STEPS=8, SAMPLE_CFG_SCALE=3.0, SAMPLE_NEGATIVE="blurry")
    assert (kw["sample_steps"], kw["sample_cfg_scale"], kw["sample_negative"]) == (8, 3.0, "blurry")
    # without a checkpoint the plain model's values are still sent, as before
    kw = pipeline.train_kwargs(desc, base, tmp_path / "run", {})[0]
    assert kw["sample_cfg_scale"] == desc.preview_cfg and "preview_checkpoint" not in kw


class CkDriver(TinyDriver):
    renders = []

    def park_for_preview(self, dit, device):
        return "token"

    def unpark_after_preview(self, dit, device, token):
        pass

    def load_preview_checkpoint(self, path, device, int8=False):
        return TinyDiT().requires_grad_(False), 0

    def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, **kw):
        CkDriver.renders.append((steps, cfg, neg_cond is not None))
        return super().generate(dit, cond, width, height, steps=steps, seed=seed, cfg=cfg, neg_cond=neg_cond, **kw)


def _ck_family(tmp):
    from tests.tiny_family import TINY
    from training import registry
    from training.description import SamplingSettings
    key = "_tiny_ck"
    if registry.get(key) is None:
        registry.register(dataclasses.replace(
            TINY, key=key, driver="tests.test_trainer_wave2:CkDriver", train_preview_checkpoint=True, preview_cfg=4.0,
            preview_checkpoint_sampling=SamplingSettings("tiny distilled", steps=2, cfg=1.0, source="test")))
    ckpt = tmp / "preview_ck.safetensors"
    ckpt.write_bytes(b"x")
    return registry.get(key), str(ckpt)


def test_the_trainer_renders_a_checkpoint_preview_with_typed_steps_and_cfg(tmp_path):
    desc, ckpt = _ck_family(tmp_path)
    data = make_dataset(str(tmp_path / "scratch_dataset"))
    models = write_models(str(tmp_path / "models"))
    base = _quick(tmp_path, MAX_TRAIN_EPOCHS=1, SAMPLE_ENABLED=True, SAMPLE_STEPS=0)

    def go(name, **train_over):
        CkDriver.renders.clear()
        run = pipeline.build_run(desc, {**base, "LORA_NAME": name}, data, models)
        _run_stages(run, preview_checkpoint=ckpt, preview_checkpoint_cache="off", **train_over)
        return list(CkDriver.renders)
    # nothing typed: the checkpoint's own recipe (2 steps, CFG 1, no negative)
    assert go("a", sample_steps=None, sample_cfg_scale=None) == [(2, 1.0, False)]
    # typed Steps 5, CFG 3: both honoured, the negative comes with a CFG above 1
    assert go("b", sample_steps=5, sample_cfg_scale=3.0, sample_negative="blurry") == [(5, 3.0, True)]
    # typed CFG 1: the recipe's CFG, no negative; steps typed alone leave the recipe's CFG
    assert go("c", sample_steps=3, sample_cfg_scale=1.0) == [(3, 1.0, False)]
    assert go("d", sample_steps=6, sample_cfg_scale=None) == [(6, 1.0, False)]
