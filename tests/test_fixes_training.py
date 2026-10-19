"""Regression tests for the training-side findings of docs/dev/audit/08_bug_and_security_review.md (#1-#8) and
05_shared_layer_v8.md (#1, block swap through LoRA wrappers). Tiny random models only; nothing here loads weights."""
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from tests import tiny_family as TF  # noqa: E402
from training import cache  # noqa: E402
from training import params as P  # noqa: E402
from training import pipeline, presets  # noqa: E402
from training.dataset import TrainingDataset  # noqa: E402


class RefDriver(TF.TinyDriver):
    """A tiny driver whose reference conditioning carries a ref_mask (as Qwen Image 2.1's does)."""
    supports_references = True

    def load_reference_text_encoder(self, path, device):
        return self.load_text_encoder(path, device)

    @torch.no_grad()
    def encode_text_with_references(self, te, captions, references):
        return [{"hidden_states": te.encode(c), "ref_mask": torch.ones(len(r), dtype=torch.bool)}
                for c, r in zip(captions, references)]


def _pair_dataset(tmp_path, n=2):
    desc = TF.register()
    driver = RefDriver()
    driver.description = desc
    ds_dir = TF.make_dataset(str(tmp_path / "after"), n=n, sizes=[(96, 96)], captions=["shared caption"] * n)
    ctrl_dir = TF.make_dataset(str(tmp_path / "before"), n=n, sizes=[(96, 96)], captions=["shared caption"] * n)
    cfg = {"datasets": [{"image_directory": ds_dir, "cache_directory": str(pipeline.cache_dir_for(ds_dir, tmp_path)),
                         "control_directory": ctrl_dir, "num_repeats": 1}],
           "resolution": [96, 96], "caption_extension": ".txt", "batch_size": 1}
    dataset = TrainingDataset(cfg, desc.arch_id, desc.spatial_factor, desc.bucket_step)
    return desc, driver, dataset, ds_dir, ctrl_dir


def _keys_and_md(path):
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        return sorted(f.keys()), dict(f.metadata())


# ---- 08 #1 / #4: prepare_for_training keeps the partner images and the caption -------------------------------
def test_prepared_items_keep_control_paths_and_caption(tmp_path):
    desc, driver, dataset, ds_dir, ctrl_dir = _pair_dataset(tmp_path)
    with torch.no_grad():
        cache.run_latents(desc, driver, dataset, "x", torch.device("cpu"))
        cache.run_text(desc, driver, dataset, "x", torch.device("cpu"))
    assert dataset.prepare_for_training() == 2
    it = dataset.items()["img00"]
    assert it.caption == "shared caption"
    assert [os.path.basename(p) for p in it.control_paths] == ["img00.png"]

    # a mid-run caption fix re-encodes WITH the reference: the text cache keeps its ref_mask and reference_sizes
    from training.loss_watch import Watch
    w = Watch(str(tmp_path), dataset, driver, te_path="x")
    w._reencode([("img00", it, "fixed caption", 0)], torch.device("cpu"))
    keys, md = _keys_and_md(it.text_encoder_output_cache_path)
    assert "cond__ref_mask" in keys and md.get("reference_sizes") == "96x96" and md["caption1"] == "fixed caption"


def test_slider_fix_is_not_encoded_with_references(tmp_path):
    desc, driver, dataset, *_ = _pair_dataset(tmp_path)
    with torch.no_grad():
        cache.run_latents(desc, driver, dataset, "x", torch.device("cpu"))
        cache.run_text(desc, driver, dataset, "x", torch.device("cpu"), slider=True)
    dataset.prepare_for_training()
    it = dataset.items()["img00"]
    from training.loss_watch import Watch
    Watch(str(tmp_path), dataset, driver, te_path="x", slider=True)._reencode([("img00", it, "again", 0)],
                                                                              torch.device("cpu"))
    keys, _ = _keys_and_md(it.text_encoder_output_cache_path)
    assert "cond__ref_mask" not in keys


def test_pair_slider_caption_from_prepared_items_and_from_files(tmp_path):
    from training.train import _pair_slider_caption
    desc, driver, dataset, ds_dir, _ = _pair_dataset(tmp_path)
    with torch.no_grad():
        cache.run_latents(desc, driver, dataset, "x", torch.device("cpu"))
        cache.run_text(desc, driver, dataset, "x", torch.device("cpu"), slider=True)
    dataset.prepare_for_training()
    assert _pair_slider_caption(dataset) == "shared caption"
    for it in dataset.items().values():            # items that lack their caption: the caption files are read
        it.caption = ""
    assert _pair_slider_caption(dataset) == "shared caption"


# ---- 08 #2: an image edited in place at the same size is encoded again --------------------------------------------
def _latent_path(dataset, key="img00"):
    return next(it for it in dataset.source_items() if it.item_key == key).latent_cache_path


def _recache(desc, driver, dataset):
    with torch.no_grad():
        return cache.run_latents(desc, driver, dataset, "x", torch.device("cpu"), skip_existing=True)


def _overwrite(path, value):
    from PIL import Image
    with Image.open(path) as im:
        size = im.size
    Image.fromarray(np.full((size[1], size[0], 3), value, np.uint8)).save(path)


def test_latent_cache_reencodes_after_an_in_place_edit(tmp_path):
    desc, driver, dataset, ds_dir, ctrl_dir = _pair_dataset(tmp_path)
    assert _recache(desc, driver, dataset) == 2
    assert _recache(desc, driver, dataset) == 0                      # nothing changed: reused
    _, md = _keys_and_md(_latent_path(dataset))
    assert md["source_stamp"] and md["control_stamps"]
    _overwrite(os.path.join(ds_dir, "img00.png"), 0)                 # same pixel size, new pixels
    assert _recache(desc, driver, dataset) == 1
    assert _recache(desc, driver, dataset) == 0
    _overwrite(os.path.join(ctrl_dir, "img01.png"), 255)             # an edited partner image
    assert _recache(desc, driver, dataset) == 1


def test_latent_cache_without_a_stamp_is_judged_like_fizgig_8_4_0(tmp_path):
    """Fizgig re-encodes a picture newer than its cache file and keeps the rest: an old cache is not all redone."""
    from safetensors.torch import load_file, save_file
    desc, driver, dataset, ds_dir, _ = _pair_dataset(tmp_path)
    _recache(desc, driver, dataset)
    for it in dataset.source_items():
        _, md = _keys_and_md(it.latent_cache_path)
        md.pop("source_stamp"), md.pop("control_stamps")
        save_file(load_file(it.latent_cache_path), it.latent_cache_path, metadata=md)
    assert _recache(desc, driver, dataset) == 0                      # an unstamped, up-to-date cache is kept
    img = os.path.join(ds_dir, "img00.png")
    future = os.stat(_latent_path(dataset)).st_mtime + 100
    os.utime(img, (future, future))
    assert _recache(desc, driver, dataset) == 1                      # the picture is newer than its cache


def test_text_cache_for_edit_pairs_notices_an_edited_partner(tmp_path):
    desc, driver, dataset, ds_dir, ctrl_dir = _pair_dataset(tmp_path)

    def text():
        with torch.no_grad():
            return cache.run_text(desc, driver, dataset, "x", torch.device("cpu"), skip_existing=True)

    assert text() == 2 and text() == 0
    _overwrite(os.path.join(ctrl_dir, "img00.png"), 7)
    assert text() == 1


# ---- 08 #3: a caption extension that is an image extension ----------------------------------------------------
@pytest.mark.parametrize("ext", [".png", ".JPG", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".jxl", ".avif"])
def test_image_extension_is_refused_as_caption_extension(ext):
    param = P.BY_KEY["DATASET_CAPTION_EXT"]
    with pytest.raises(ValueError, match="image"):
        P.coerce(param, ext)
    out, rep = presets.apply({"DATASET_CAPTION_EXT": ext}, {"DATASET_CAPTION_EXT": ".txt"})
    assert out["DATASET_CAPTION_EXT"] == ".txt" and rep.refused and rep.refused[0][0] == "DATASET_CAPTION_EXT"
    assert any("overwrite" in m for m in rep.messages())


def test_caption_extension_that_is_fine_still_applies():
    assert P.coerce(P.BY_KEY["DATASET_CAPTION_EXT"], ".caption") == ".caption"
    assert P.coerce(P.BY_KEY["DATASET_CAPTION_EXT"], "") == ""
    with pytest.raises(ValueError):
        P.coerce(P.BY_KEY["DATASET_CAPTION_EXT"], "txt")


def test_preflight_errors_on_an_image_caption_extension(tmp_path):
    desc = TF.register()
    data = TF.make_dataset(str(tmp_path / "scratch_dataset"))
    models = TF.write_models(str(tmp_path / "models"))
    v = P.defaults()
    v.update({"LORA_OUTPUT_DIR": str(tmp_path / "runs"), "LORA_NAME": "tiny_lora", "DATASET_CAPTION_EXT": ".png"})
    checks = pipeline.preflight(desc, v, data, models)
    assert any(c.level == "error" and "image type" in c.message for c in checks)


# ---- 08 #5: the stale sweep only touches this family's own cache names --------------------------------------------
def test_stale_sweep_leaves_another_familys_cache_alone(tmp_path):
    desc, driver, dataset, ds_dir, _ = _pair_dataset(tmp_path)
    cdir = Path(dataset.config["datasets"][0]["cache_directory"])
    cdir.mkdir(parents=True, exist_ok=True)
    other = [cdir / "portrait_tiny_v2_1024x1024_krea2.safetensors", cdir / "portrait_tiny_v2_krea2_te.safetensors",
             cdir / "portrait_tiny_te_s1_krea2_te_s0.safetensors"]
    mine_stale = [cdir / "gone_0096x0096_tiny.safetensors", cdir / "gone_tiny_te.safetensors",
                  cdir / "gone_tiny_te_s0.safetensors"]
    for p in other + mine_stale:
        p.write_bytes(b"x")
    with torch.no_grad():
        cache.run_latents(desc, driver, dataset, "x", torch.device("cpu"))
    assert all(p.exists() for p in other)
    assert not any(p.exists() for p in mine_stale)


# ---- 08 #6: run_dir_for validates the LoRA name -------------------------------------------------------------------
@pytest.mark.parametrize("name", ["..", "../x", "a/b", "..\\..\\x", "a\nb", ""])
def test_run_dir_for_refuses_a_name_that_is_a_path(tmp_path, name):
    values = {"LORA_OUTPUT_DIR": str(tmp_path), "LORA_NAME": name}
    if name == "":                                   # an empty name falls back to the default one
        assert pipeline.run_dir_for(values) == tmp_path / "my_lora"
        return
    with pytest.raises(ValueError):
        pipeline.run_dir_for(values)


def test_run_dir_for_accepts_a_plain_name(tmp_path):
    assert pipeline.run_dir_for({"LORA_OUTPUT_DIR": str(tmp_path), "LORA_NAME": "my_lora_v2"}) == tmp_path / "my_lora_v2"


# ---- 08 #8: resume files are read with weights_only ---------------------------------------------------------------
def test_resume_files_load_weights_only(tmp_path, monkeypatch):
    from training import train
    from training.automagic3 import Automagic3
    seen = []
    real = torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **k: (seen.append(k.get("weights_only")), real(*a, **k))[1])
    for make in (lambda p: torch.optim.AdamW(p, lr=1e-3), lambda p: Automagic3(p, lr=1e-3)):
        w = nn.Parameter(torch.randn(8, 8))
        opt = make([w])
        w.grad = torch.randn(8, 8)
        opt.step()
        torch.save(opt.state_dict(), tmp_path / "optimizer.pt")
        opt2 = make([nn.Parameter(torch.randn(8, 8))])
        opt2.load_state_dict(train._torch_load(str(tmp_path / "optimizer.pt"), "cpu"))
    torch.save({"torch": torch.get_rng_state()}, tmp_path / "rng.pt")
    assert "torch" in train._torch_load(str(tmp_path / "rng.pt"), "cpu")
    assert seen and all(s is True for s in seen)


class _Evil:
    def __reduce__(self):
        return (os.getcwd, ())


def test_resume_refuses_a_pickled_object(tmp_path):
    from training import train
    torch.save({"x": _Evil()}, tmp_path / "optimizer.pt")
    with pytest.raises(RuntimeError, match="could not be read safely"):
        train._torch_load(str(tmp_path / "optimizer.pt"), "cpu")


# ---- 08 #7: a preset that sets a path is named in the report ------------------------------------------------------
def test_preset_that_sets_a_path_is_noted():
    out, rep = presets.apply({"LORA_OUTPUT_DIR": "D:\\Datasets\\demo_out", "FAMILY_EDIT_DIR": "D:\\Datasets\\originals",
                              "CONTEXT_LORA_PATH": "D:\\x.safetensors", "METADATA_THUMBNAIL": "D:\\t.png",
                              "NETWORK_DIM": 8}, P.defaults())
    text = "\n".join(rep.messages())
    for key in ("LORA_OUTPUT_DIR", "FAMILY_EDIT_DIR", "CONTEXT_LORA_PATH", "METADATA_THUMBNAIL"):
        assert key in text
    assert "D:\\Datasets\\demo_out" in text and "NETWORK_DIM" not in text
    _, rep = presets.apply({"METADATA_THUMBNAIL": "off", "LORA_OUTPUT_DIR": ""}, P.defaults())
    assert not rep.notes


# ---- 05 #1: block swap pairs a bare block with a LoRA-wrapped one (Fizgig #199) ----------------------------------
class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(6, 6)
        self.b = nn.Linear(6, 6)

    def forward(self, x):
        return x + self.b(torch.relu(self.a(x)))


def _wrap(block, names):
    from training.lora import LoRALinear
    for n in names:
        lin = getattr(block, n)
        w = LoRALinear(lin)
        w.add("default", 2, 2, True)
        setattr(block, n, w)


def test_swap_pairs_a_bare_block_with_a_wrapped_one():
    """Both blocks sit on the CPU here, so the swap copies one weight over the other (on a card it exchanges them):
    what this pins down is WHICH weights are paired - a.<->a.base, b.<->b.base - and that the adapters stay put."""
    from training.modules.offloading import swap_weight_devices_no_cuda
    torch.manual_seed(0)
    bare, wrapped = _Block(), _Block()
    _wrap(wrapped, ["a", "b"])
    wa, wb = wrapped.a.base.weight.detach().clone(), wrapped.b.base.weight.detach().clone()
    adapter = wrapped.a.adapters["default"][0].weight.detach().clone()
    swap_weight_devices_no_cuda(torch.device("cpu"), bare, wrapped)
    assert torch.equal(bare.a.weight, wa) and torch.equal(bare.b.weight, wb)
    assert torch.equal(wrapped.a.adapters["default"][0].weight, adapter)
    # a half-wrapped block against a bare one, the other way round
    half, bare2 = _Block(), _Block()
    _wrap(half, ["b"])
    ba = bare2.a.weight.detach().clone()
    swap_weight_devices_no_cuda(torch.device("cpu"), half, bare2)
    assert torch.equal(half.a.weight, ba)                      # the unwrapped Linear pairs with the unwrapped one


def test_block_swap_with_lora_on_some_blocks_runs_forward_and_backward():
    from training.modules.offloading import ModelOffloader
    torch.manual_seed(0)
    blocks = nn.ModuleList([_Block() for _ in range(5)])
    _wrap(blocks[3], ["a", "b"])                       # the LoRA covers the later blocks only (an H3-style window)
    _wrap(blocks[4], ["a"])
    ref = nn.ModuleList([_Block() for _ in range(5)])
    ref.load_state_dict({k.replace(".base.", "."): v for k, v in blocks.state_dict().items()
                         if "adapters" not in k}, strict=False)
    x = torch.randn(2, 6)
    expected = x
    for b in ref:
        expected = b(expected)

    off = ModelOffloader("test", list(blocks), 5, 2, True, torch.device("cpu"))
    off.prepare_block_devices_before_forward(list(blocks))
    y = x.clone().requires_grad_(True)
    for i, b in enumerate(blocks):
        off.wait_for_block(i)
        y = b(y)
        off.submit_move_blocks_forward(list(blocks), i)
    assert torch.allclose(y, expected, atol=1e-5)       # every block ran on its own weights
    y.sum().backward()
    off.remove_hooks()
    assert blocks[3].a.adapters["default"][1].weight.grad is not None
