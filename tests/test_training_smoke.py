"""End-to-end CPU smoke tests of the training pipeline on the tiny random family (tests/tiny_family.py).

These run the real cache and train entry points (in-process) on a scratch dataset in a temp folder: caching,
the generic loop, Conv2d + Linear LoRA, Adaptive LR, EMA, the loss watch, previews, checkpoints with metadata,
pause and resume, gradient accumulation, batching and the caption-augmentation extension. No real model is
trained; a run takes a few seconds.
"""
import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests.tiny_family import make_dataset, register, write_models  # noqa: E402
from training import params as P  # noqa: E402
from training import pipeline  # noqa: E402


@pytest.fixture
def setup(tmp_path):
    desc = register()
    data = make_dataset(str(tmp_path / "scratch_dataset"))
    models = write_models(str(tmp_path / "models"))
    return desc, data, models, tmp_path


def _values(tmp_path, **over):
    v = P.defaults()
    v.update({"LORA_OUTPUT_DIR": str(tmp_path / "runs"), "LORA_NAME": "tiny_lora", "NETWORK_DIM": 4,
              "NETWORK_ALPHA": 4, "MAX_TRAIN_EPOCHS": 3, "OPTIMIZER_TYPE": "adamw", "LEARNING_RATE": 1e-3,
              "DATASET_MEGAPIXELS": "0.01", "SAMPLE_PROMPT": "tok, a test prompt", "SAMPLE_EVERY_N_EPOCHS": 1,
              "SAMPLE_WIDTH": 64, "SAMPLE_HEIGHT": 64, "SAMPLE_STEPS": 2, "SAMPLE_AT_FIRST": True,
              "KREA2_LOSS_WATCH": True, "ADAPTIVE_LR": True, "ADAPTIVE_LR_MIN": "1e-4", "ADAPTIVE_LR_MAX": "4e-4"})
    v.update(over)
    return v


def _run_stages(run, *, device="cpu"):
    """Run a built run's stages in-process (the same argv the Train tab launches as child processes)."""
    from training import cache, train
    for _label, argv in run.stages:
        mod, args = argv[2], argv[3:]
        if mod == "training.cache":
            cache.main(args + ["--device", device])
        else:
            cfg_path = args[args.index("--config") + 1]
            cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
            cfg["train"]["device"] = device
            Path(cfg_path).write_text(json.dumps(cfg), encoding="utf-8")
            try:
                train.main(args)
            except SystemExit as e:
                assert e.code in (0, None)


def test_preflight_and_build(setup):
    desc, data, models, tmp = setup
    vals = _values(tmp)
    checks = pipeline.preflight(desc, vals, data, models)
    assert not [c for c in checks if c.level == "error"], checks
    run = pipeline.build_run(desc, vals, data, models, trigger="tok")
    assert [lab for lab, _ in run.stages] == ["Caching latents", "Caching text", "Training"]
    cfg = json.loads((run.run_dir / "train_config.json").read_text(encoding="utf-8"))
    kw = cfg["train"]
    assert kw["adaptive_lr"] and kw["adaptive_lr_min"] == 1e-4 and kw["adaptive_lr_max"] == 4e-4
    assert kw["ema_decay"] == 0.98 and kw["blocks_to_swap"] == -1 and kw["precision"] == "bf16"
    assert kw["log_per_image_loss"] and kw["metadata_trigger_phrase"] == "tok"
    ds = json.loads((run.run_dir / "dataset.json").read_text(encoding="utf-8"))
    assert ds["datasets"][0]["image_directory"] == os.path.abspath(data)
    assert ds["caption_shuffle_variants"] == 0 and ds["caption_dropout"] == 0.0   # extension off by default


def test_preflight_errors(setup):
    desc, data, models, tmp = setup
    os.remove(os.path.join(data, "img00.txt"))
    checks = pipeline.preflight(desc, _values(tmp, DATASET_BATCH_SIZE=2, LORA_NAME="bad/name"), data,
                                {**models, "tiny_vae": ""})
    errors = " ".join(c.message for c in checks if c.level == "error")
    assert "no .txt caption" in errors and "batch size 1" in errors and "Tiny VAE" in errors
    assert "must be just a name" in errors


def test_full_run_then_pause_and_resume(setup):
    desc, data, models, tmp = setup
    vals = _values(tmp)
    run = pipeline.build_run(desc, vals, data, models, trigger="tok")
    rd = run.run_dir
    # pause after epoch 1: the sentinel is honoured at the epoch boundary (Fizgig's contract)
    pipeline.request_pause(rd)
    _run_stages(run)
    assert (rd / "tiny_lora-000001.safetensors").exists()
    state = rd / "tiny_lora-000001-state"
    assert (state / "training_state.json").exists() and (state / "optimizer.pt").exists()
    meta = json.loads((state / "training_state.json").read_text(encoding="utf-8"))
    assert meta["epoch"] == 1 and "adaptive_lr_state" in meta and (state / "ema.pt").exists()
    assert not (rd / "tiny_lora.safetensors").exists()
    # caches written with Fizgig's naming
    cache_dir = Path(json.loads((rd / "dataset.json").read_text(encoding="utf-8"))["datasets"][0]["cache_directory"])
    names = sorted(p.name for p in cache_dir.iterdir())
    assert any(n.startswith("img00_0096x0096_tiny") for n in names) and "img00_tiny_te.safetensors" in names
    # resume to the end
    pipeline.clear_pause(rd)
    resumed = pipeline.build_run(desc, vals, data, models, trigger="tok", resume=str(state))
    assert [lab for lab, _ in resumed.stages] == ["Training"]
    _run_stages(resumed)
    final = rd / "tiny_lora.safetensors"
    assert final.exists()
    assert (rd / "tiny_lora-000003.safetensors").exists()      # the last epoch under its number too (Fizgig #176)
    from safetensors import safe_open
    with safe_open(str(final), framework="pt") as f:
        md = f.metadata()
        keys = list(f.keys())
    assert md["modelspec.architecture"] == "Tiny-Test/lora" and md["ss_epoch"] == "3"
    assert md["modelspec.trigger_phrase"] == "tok" and md["modelspec.thumbnail"].startswith("data:image/jpeg")
    assert "diffusion_model.blocks.0.conv.lora_down.weight" in keys
    shots = pipeline.samples(rd)
    epochs = {pipeline.parse_sample_name(p.name)[0] for p in shots}
    assert {0, 1, 2, 3} <= epochs                        # sample at first + every epoch
    assert (rd / "loss_log" / "per_image_loss.jsonl").exists()


def test_sample_override_and_caption_fix(setup):
    desc, data, models, tmp = setup
    vals = _values(tmp, MAX_TRAIN_EPOCHS=3, ADAPTIVE_LR=False, SAMPLE_AT_FIRST=False)
    run = pipeline.build_run(desc, vals, data, models)
    pipeline.write_sample_override(run.run_dir, "tok, override prompt", seed=7, width=64, height=64)
    pipeline.queue_caption_updates(run.run_dir, {"img02": "queued before start"})
    pipeline.request_pause(run.run_dir)
    _run_stages(run)
    # a fresh run clears the previous run's leftovers (Fizgig): edits are queued while a run is going
    assert not (run.run_dir / "loss_log" / "caption_updates.json").exists()
    pipeline.clear_pause(run.run_dir)
    pipeline.queue_caption_updates(run.run_dir, {"img01": "tok, fixed caption"})
    state = str(run.run_dir / "tiny_lora-000001-state")
    _run_stages(pipeline.build_run(desc, vals, data, models, resume=state))
    seeds = {pipeline.parse_sample_name(p.name)[2] for p in pipeline.samples(run.run_dir)}
    assert seeds == {7}
    applied = pipeline.read_applied_captions(run.run_dir)
    assert applied["img01"][0]["caption"] == "tok, fixed caption"
    from safetensors import safe_open
    cache_dir = json.loads((run.run_dir / "dataset.json").read_text(encoding="utf-8"))["datasets"][0][
        "cache_directory"]
    with safe_open(os.path.join(cache_dir, "img01_tiny_te.safetensors"), framework="pt") as f:
        assert f.metadata()["caption1"] == "tok, fixed caption"
    # the dataset's own caption file is untouched by a manual mid-run fix (the UI saves those itself)
    assert Path(data, "img01.txt").read_text(encoding="utf-8").startswith("tok, photo 1")


def test_accumulation_lokr_and_flat_lr(setup):
    desc, data, models, tmp = setup
    vals = _values(tmp, MAX_TRAIN_EPOCHS=2, ADAPTIVE_LR=False, GRADIENT_ACCUMULATION=3, SAMPLE_ENABLED=False,
                   NETWORK_TYPE="LoKR (Kronecker)", LOKR_FACTOR=4, LR_SCHEDULER="cosine", LORA_NAME="lokr_run")
    run = pipeline.build_run(desc, vals, data, models)
    _run_stages(run)
    from safetensors import safe_open
    with safe_open(str(run.run_dir / "lokr_run.safetensors"), framework="pt") as f:
        keys = list(f.keys())
        md = f.metadata()
    assert any(k.endswith(".lokr_w1") for k in keys) and md["ss_lokr_factor"] == "4"
    assert not pipeline.samples(run.run_dir)


def test_batching_family_and_caption_extension(tmp_path):
    from tests.tiny_family import TINY_BATCH
    register()
    data = make_dataset(str(tmp_path / "scratch"), n=6, sizes=[(96, 96)])
    models = write_models(str(tmp_path / "models"), arch="tinyb")
    vals = _values(tmp_path, MAX_TRAIN_EPOCHS=2, DATASET_BATCH_SIZE=2, SAMPLE_ENABLED=False,
                   CAPTION_SHUFFLE_VARIANTS=2, CAPTION_KEEP_TOKENS=1, CAPTION_DROPOUT=0.2, LORA_NAME="batched")
    assert not [c for c in pipeline.preflight(TINY_BATCH, vals, data, models) if c.level == "error"]
    run = pipeline.build_run(TINY_BATCH, vals, data, models)
    _run_stages(run)
    assert (run.run_dir / "batched.safetensors").exists()
    cache_dir = Path(json.loads((run.run_dir / "dataset.json").read_text(encoding="utf-8"))["datasets"][0][
        "cache_directory"])
    names = {p.name for p in cache_dir.iterdir()}
    assert "img00_tinyb_te_s0.safetensors" in names and "img00_tinyb_te_s1.safetensors" in names
    assert "_empty_tinyb_te.safetensors" in names
    assert not (run.run_dir / "loss_log" / "per_image_loss.jsonl").exists()   # loss watch stands down at batch 2


def test_stale_cache_removed_and_latents_skipped(setup):
    desc, data, models, tmp = setup
    vals = _values(tmp, MAX_TRAIN_EPOCHS=1, SAMPLE_ENABLED=False, ADAPTIVE_LR=False)
    run = pipeline.build_run(desc, vals, data, models)
    _run_stages(run)
    cache_dir = Path(json.loads((run.run_dir / "dataset.json").read_text(encoding="utf-8"))["datasets"][0][
        "cache_directory"])
    lat = next(cache_dir.glob("img00_*_tiny.safetensors"))
    mtime = lat.stat().st_mtime_ns
    os.remove(os.path.join(data, "img03.png"))
    os.remove(os.path.join(data, "img03.txt"))
    _run_stages(pipeline.build_run(desc, vals, data, models))
    assert lat.stat().st_mtime_ns == mtime                       # --skip_existing kept the valid latent
    assert not list(cache_dir.glob("img03_*"))                   # the deleted image's caches are gone

    # a cache written before the encoder revision mark (or with an older one) is encoded again
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    from training import cache as C
    assert C.latent_rev(str(lat)) == C.LATENT_REV
    with safe_open(str(lat), framework="pt") as f:
        md = dict(f.metadata())
    md.pop("latent_rev")
    save_file(load_file(str(lat)), str(lat), metadata=md)
    assert C.latent_rev(str(lat)) == ""
    _run_stages(pipeline.build_run(desc, vals, data, models))
    assert C.latent_rev(str(lat)) == C.LATENT_REV


def test_qwen_driver_on_a_tiny_config():
    """The ported Qwen Image 2.1 DiT and driver objective run end to end (forward + backward) on a 2-layer,
    randomly initialised config on CPU - proving the port's plumbing, not the model."""
    from training.families.qwen_image21 import sampling as S
    from training.families.qwen_image21.driver import QwenImage21Driver
    from training.families.qwen_image21.model import QwenImage21DiT
    from training.lora import FamilyLoRA
    from training.registry import get
    torch.manual_seed(0)
    dit = QwenImage21DiT(patch_size=1, in_channels=64, out_channels=64, num_layers=2, attention_head_dim=16,
                         num_attention_heads=2, context_in_dim=32, mlp_ratio=3, axes_dims_rope=(4, 6, 6),
                         eps=1e-6, causal_condition=True).to(torch.bfloat16).requires_grad_(False)
    drv = QwenImage21Driver()
    drv.description = get("qwen_image21")
    dit.enable_gradient_checkpointing(True)
    net = FamilyLoRA(dit, drv)
    net.add_trainable(4, 4)
    assert len(net.trainable_modules()) == 2 * 7
    latents = torch.randn(1, 64, 4, 4)
    cond = {"hidden_states": torch.randn(1, 5, 32)}
    loss, info = drv.training_loss(dit, latents, cond, torch.Generator().manual_seed(0))
    loss.backward()
    assert torch.isfinite(loss) and 0 <= info["t"] <= 1
    assert all(p.grad is not None for p in net.parameters())
    out = drv.generate(dit, {"hidden_states": torch.randn(5, 32)}, 64, 64, steps=2, seed=1)
    assert out.shape == (1, S.latent_hw(64, 64)[0] * S.latent_hw(64, 64)[1], 64)


class _Rec:
    """Records the preview's calls on the network, EMA and driver; `fail` names a call that raises."""
    def __init__(self, fail=None):
        self.calls, self.fail, self.gen = [], fail, []

    def __getattr__(self, name):
        def f(*a, **k):
            self.calls.append((name, a))
            if name == self.fail:
                self.fail = None                        # fails once, as a GPU running out of memory would
                raise RuntimeError("out of memory")
        return f


def _preview(out_dir, fail=None, cfg=1.0, speed_cfg=1.0):
    from types import SimpleNamespace
    from training import train
    net, ema = _Rec(fail), _Rec(fail)
    gen = []
    from training.driver import FamilyDriver
    driver = SimpleNamespace(generate=lambda *a, **k: gen.append(k), decode=lambda *a: SimpleNamespace(save=lambda p: None),
                             block_swap_mode=lambda *a, **k: None, park_for=lambda *a: None, unpark=lambda *a: None,
                             save_preview=lambda r, p: FamilyDriver.save_preview(None, r, p))
    dit = torch.nn.Linear(2, 2)
    speed = SimpleNamespace(cfg=speed_cfg, sigmas=None, options=())
    try:
        train._render_previews(driver, dit, net, None, [{}], str(out_dir), 1, output_name="x", steps=1, cfg=cfg,
                               neg={"n": 1}, width=8, height=8, seed=1, ema=ema, speed=speed)
    except RuntimeError:
        pass
    return net.calls, ema.calls, gen


def test_failed_preview_undoes_exactly_what_it_changed(tmp_path):
    from training.train import ADAPTER, SPEED
    net, ema, _ = _preview(tmp_path, fail="move_adapter")         # the speed LoRA's move to the GPU runs out of memory
    assert ("set_enabled", (ADAPTER, True)) in net and ("set_enabled", (SPEED, False)) in net
    assert ("move_adapter", (SPEED, "cpu")) in net and ema == []        # EMA was never swapped in: not swapped out


def test_turbo_preview_honours_a_higher_cfg(tmp_path):
    *_, gen = _preview(tmp_path, cfg=1.0, speed_cfg=1.0)
    assert gen[0]["cfg"] == 1.0 and gen[0]["neg_cond"] is None         # the speed LoRA's own CFG, no negative
    *_, gen = _preview(tmp_path, cfg=3.0, speed_cfg=1.0)
    assert gen[0]["cfg"] == 3.0 and gen[0]["neg_cond"] == {"n": 1}     # Fizgig 6.8.1: more CFG, with the negative


def test_driver_hooks_reach_the_loop(setup, monkeypatch):
    """Fizgig 7.0.1's driver hooks (training/driver.py) are called by the loop: step_policy, batch_cond,
    after_optimizer_step, run_metadata, frozen_file_added, save_preview."""
    from tests.tiny_family import TinyDriver
    calls = {"policy": 0, "cond": 0, "after": 0, "saved": 0}
    base_cond = TinyDriver.batch_cond

    def policy(self, batch, epoch):
        calls["policy"] += 1
        return calls["policy"] % 4 == 0, 0.5           # every fourth step retired, the rest at half the LR

    def cond(self, batch, device):
        calls["cond"] += 1
        return base_cond(self, batch, device)

    def save(self, result, path):
        calls["saved"] += 1
        result.save(path)
        return [path]
    monkeypatch.setattr(TinyDriver, "step_policy", policy)
    monkeypatch.setattr(TinyDriver, "batch_cond", cond)
    monkeypatch.setattr(TinyDriver, "after_optimizer_step", lambda self: calls.__setitem__("after", calls["after"] + 1))
    monkeypatch.setattr(TinyDriver, "run_metadata", lambda self: {"ss_family_option": "x"})
    monkeypatch.setattr(TinyDriver, "save_preview", save)
    desc, data, models, tmp = setup
    vals = _values(tmp, MAX_TRAIN_EPOCHS=2, ADAPTIVE_LR=False, KREA2_LOSS_WATCH=False)
    run = pipeline.build_run(desc, vals, data, models, trigger="tok")
    _run_stages(run)
    steps = calls["policy"]
    assert steps == 8 and calls["cond"] == 6 and calls["after"] == 6 and calls["saved"] >= 3
    from safetensors import safe_open
    with safe_open(str(run.run_dir / "tiny_lora.safetensors"), framework="pt") as f:
        assert f.metadata()["ss_family_option"] == "x"


def _tiny_qwen():
    from training.families.qwen_image21.driver import QwenImage21Driver
    from training.families.qwen_image21.model import QwenImage21DiT
    from training.registry import get
    torch.manual_seed(0)
    dit = QwenImage21DiT(patch_size=1, in_channels=64, out_channels=64, num_layers=2, attention_head_dim=16,
                         num_attention_heads=2, context_in_dim=32, mlp_ratio=3, axes_dims_rope=(4, 6, 6),
                         eps=1e-6, causal_condition=True).to(torch.bfloat16).requires_grad_(False)
    drv = QwenImage21Driver()
    drv.description = get("qwen_image21")
    return dit, drv


def test_qwen_compiled_block_equals_the_eager_block():
    """Fizgig 7.0.1's Qwen compile hook: the DiT calls a CheckpointedBlock directly; the traced graph (dynamo's eager
    backend, Qwen's fullgraph=False, the checkpoint outside) gives the eager loss and gradients."""
    from training.compile import CheckpointedBlock
    from training.lora import FamilyLoRA
    dit, drv = _tiny_qwen()
    dit.enable_gradient_checkpointing(True)
    net = FamilyLoRA(dit, drv)
    net.add_trainable(4, 4)
    for p in net.parameters():
        p.data.normal_(0, 0.05)
    lat, cond = torch.randn(1, 64, 4, 4), {"hidden_states": torch.randn(1, 5, 32)}

    def run():
        dit.train()
        for p in net.parameters():
            p.grad = None
        loss, _ = drv.training_loss(dit, lat, cond, torch.Generator().manual_seed(0))
        loss.backward()
        return loss.item(), [p.grad.clone() for p in net.parameters()]
    eager_loss, eager_grads = run()
    blocks = drv.compile_targets(dit)
    for i, b in enumerate(list(blocks)):
        blocks[i] = CheckpointedBlock(torch.compile(b, fullgraph=False, backend="eager"), True)
    loss, grads = run()
    assert loss == pytest.approx(eager_loss, rel=1e-5)
    assert all(torch.allclose(a, b, atol=1e-4) for a, b in zip(grads, eager_grads))


def test_qwen_fast_identity_mode_trains_the_identity_blocks_only(tmp_path):
    """Fizgig 7.0.1: FAMILY_FAST_ID sends the description's identity_blocks (Qwen 10-14) as train_blocks; never for an
    Edit LoRA. The built-in preset is Fast at 0.25 MP."""
    from training import presets
    from training.registry import get
    qwen = get("qwen_image21")
    assert qwen.identity_blocks == tuple(f"block_{i}" for i in range(10, 15))
    assert P.family_shows(P.BY_KEY["FAMILY_FAST_ID"], qwen) and not P.family_shows(P.BY_KEY["FAMILY_FAST_ID"],
                                                                                   get("krea2"))
    name, fast_id = qwen.presets[1]
    assert "Fast Identity" in name and fast_id["FAMILY_FAST_ID"] and fast_id["DATASET_MEGAPIXELS"] == "0.25"
    vals, rep = presets.apply(fast_id, P.defaults(), qwen)
    assert not rep.refused and not rep.blocked
    kw, _ = pipeline.train_kwargs(qwen, vals, tmp_path / "run", {})
    assert kw["train_blocks"] == list(qwen.identity_blocks)
    kw, _ = pipeline.train_kwargs(qwen, {**vals, "FAMILY_EDIT": True}, tmp_path / "run", {})
    assert "train_blocks" not in kw
    from training.lora import FamilyLoRA
    dit, drv = _tiny_qwen()                               # two blocks: only block_1 trains
    net = FamilyLoRA(dit, drv)
    net.add_trainable(4, 4, blocks={"block_1"})
    assert len(net.trainable_modules()) == 7 and all(drv.block_of(f) == "block_1" for f, w in net.wrapped.items()
                                                      if "lora" in w.adapters)


def test_train_blocks_reach_the_loop_and_the_metadata(setup):
    desc, data, models, tmp = setup
    vals = _values(tmp, MAX_TRAIN_EPOCHS=1, ADAPTIVE_LR=False, KREA2_LOSS_WATCH=False, SAMPLE_ENABLED=False)
    run = pipeline.build_run(desc, vals, data, models, trigger="tok")
    cfg_path = run.run_dir / "train_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["train"]["train_blocks"] = ["block_1"]
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    _run_stages(run)
    from safetensors import safe_open
    with safe_open(str(run.run_dir / "tiny_lora.safetensors"), framework="pt") as f:
        assert f.metadata()["ss_train_blocks"] == "block_1"
        assert all(".blocks.1." in k for k in f.keys())
