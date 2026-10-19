"""Edit LoRAs with extra reference pictures (Fizgig 8.1.1 / 8.1.2): `edit_references` per family, the extra pictures
folder (photo_1.png, photo_2.png ... for photo.png), extras at their own shape at the target's pixel count, controls
cached one at a time, the cache limit, and the drivers taking several references. On the tiny random family
(tests/tiny_family.py) and the tiny Klein / Qwen configs; no real model is loaded.

A run with zero or one reference must be what it was: the tests here compute the pre-change latents and batches by
hand and compare bit for bit."""
import os
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests import tiny_family as TF  # noqa: E402
from training import cache, registry  # noqa: E402
from training.dataset import (TrainingDataset, _own_shape, control_signature, partner_map, partners,  # noqa: E402
                              resize_image_to_bucket)


class RefDriver(TF.TinyDriver):
    """A tiny driver that takes references: its conditioning carries a ref_mask (one slot per reference, as Qwen
    Image 2.1's does) and its loss adds every reference's mean, so a missing or extra reference changes it."""
    supports_references = True

    def load_reference_text_encoder(self, path, device):
        return self.load_text_encoder(path, device)

    @torch.no_grad()
    def encode_text_with_references(self, te, captions, references):
        return [{"hidden_states": te.encode(c), "ref_mask": torch.ones(len(r), dtype=torch.bool)}
                for c, r in zip(captions, references)]

    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None, diff_ref=None,
                      diff_weight=0.0):
        loss, info = super().training_loss(dit, latents, cond, generator, min_t=min_t, max_t=max_t)
        info["n_refs"] = len(refs or [])
        for r in refs or []:
            loss = loss + r.float().mean()
        return loss, info


def _desc(edit_references):
    from dataclasses import replace
    key = f"_tinye{edit_references}"
    d = registry.get(key)
    if d is None:
        d = replace(TF.TINY, key=key, arch_id=f"tinye{edit_references}", edit_training=True,
                    edit_references=edit_references, driver="tests.test_edit_references:RefDriver")
        registry.register(d)
    return d


def _picture(path, w, h, seed):
    from PIL import Image
    rng = np.random.default_rng(seed)
    Image.fromarray((rng.random((h, w, 3)) * 255).astype(np.uint8)).save(path)


def _folders(tmp_path, extras=None, n=2):
    """after/ (the edited photos, captioned), before/ (the originals, same crop) and extras/ ({photo stem: [(suffix,
    w, h), ...]}) - a scratch dataset, never a user's."""
    after = TF.make_dataset(str(tmp_path / "after"), n=n, sizes=[(96, 96)], captions=["make it blue"] * n)
    before = TF.make_dataset(str(tmp_path / "before"), n=n, sizes=[(96, 96)], captions=["x"] * n)
    xdir = str(tmp_path / "extras")
    os.makedirs(xdir, exist_ok=True)
    for stem, pics in (extras or {}).items():
        for suffix, w, h in pics:
            _picture(os.path.join(xdir, f"{stem}_{suffix}.png"), w, h, hash((stem, suffix)) % 1000)
    return after, before, xdir


def _dataset(desc, tmp_path, after, before, xdir=None, batch_size=1):
    from training import pipeline
    ds = {"image_directory": after, "cache_directory": str(pipeline.cache_dir_for(after, tmp_path)),
          "control_directory": before, "num_repeats": 1}
    if xdir:
        ds["extra_reference_directory"] = xdir
    cfg = {"datasets": [ds], "resolution": [96, 96], "caption_extension": ".txt", "batch_size": batch_size}
    return TrainingDataset(cfg, desc.arch_id, desc.spatial_factor, desc.bucket_step)


def _keys_and_md(path):
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        return sorted(f.keys()), dict(f.metadata())


# ---- the description --------------------------------------------------------------------------------------
def test_edit_references_per_family():
    """Fizgig 8.1.1 / 8.1.2: Klein takes up to 4 before-images per edit (families/klein.py:171), Qwen Image 2.1 up
    to 10 (families/qwen_image.py:161); the default is one original (families/description.py:460)."""
    assert TF.TINY.edit_references == 1
    assert registry.get("klein9b").edit_references == 4
    assert registry.get("qwen_image21").edit_references == 10
    families = registry.training_families(include_hidden=True)
    assert families
    for d in families:
        assert d.edit_references >= 1


# ---- sizes and names ----------------------------------------------------------------------------------------
def test_own_shape_keeps_the_aspect_at_the_targets_pixel_count():
    """Fizgig image_dataset.py:207-214: (width, height) at the bucket's area in the picture's own aspect, snapped to
    the step, never below one step."""
    from PIL import Image
    assert _own_shape(Image.new("RGB", (300, 100)), (96, 96), 32) == (160, 64)
    assert _own_shape(np.zeros((100, 300, 3), np.uint8), (96, 96), 32) == (160, 64)
    assert _own_shape(Image.new("RGB", (100, 300)), (96, 96), 32) == (64, 160)
    assert _own_shape(Image.new("RGB", (1000, 10)), (96, 96), 32) == (960, 32)       # the step is the floor
    assert _own_shape(Image.new("RGB", (1024, 1024)), (1024, 1024), 64) == (1024, 1024)
    assert _own_shape(Image.new("RGB", (1536, 1024)), (1024, 1024), 64) == (1280, 832)


def test_partners_sort_the_original_first_then_by_number():
    """Fizgig image_dataset.py _sort_key: photo.png, then photo_1, photo_2, ... photo_10 (by number, not by name),
    then anything else; case does not matter (IMG_1 pairs with img_1)."""
    names = ["photo_x.png", "photo_10.jpg", "photo_2.png", "PHOTO.png", "photo_1.webp", "other.png", "photo.txt"]
    assert partners("photo", names) == ["PHOTO.png", "photo_1.webp", "photo_2.png", "photo_10.jpg", "photo_x.png"]
    assert partners("Photo", names) == ["PHOTO.png", "photo_1.webp", "photo_2.png", "photo_10.jpg", "photo_x.png"]
    assert partners("other", names) == ["other.png"]


def test_partner_map_lets_the_longest_name_claim_first():
    """Fizgig image_dataset.py:681-696: when photo and photo_1 are both edited photos, photo_1.png is photo_1's own
    original, and photo_1_1.png its extra picture - not photo's."""
    names = ["photo.png", "photo_1.png", "photo_1_1.png", "photo_2.png", "lone.png"]
    assert partner_map(["photo", "photo_1", "nothing"], names) == {
        "photo_1": ["photo_1.png", "photo_1_1.png"], "photo": ["photo.png", "photo_2.png"]}
    assert partner_map([], names) == {}


def test_control_signature_reads_the_cached_shapes():
    assert control_signature(["latent_12x12", "latent_control_1_8x16", "latent_control_0_12x12"]) == \
        (("0", "12x12"), ("1", "8x16"))
    assert control_signature(["latent_12x12"]) == ()
    assert control_signature(None) == ()


# ---- caching with extra pictures -------------------------------------------------------------------------------
def test_cache_extras_at_their_own_shape_one_at_a_time(tmp_path):
    """The extra pictures folder's photo_1, photo_2 ... ride after the original: cached as latent_control_1, _2 at
    their own shape at the target's pixel count (Fizgig cache.py:184-187, image_dataset.py:1080-1085); the text cache
    records every reference's size; training gets them in order."""
    desc = _desc(3)
    driver = RefDriver()
    driver.description = desc
    after, before, xdir = _folders(tmp_path, {"img00": [(1, 128, 64), (2, 64, 128)], "img01": [(1, 96, 96)]})
    dataset = _dataset(desc, tmp_path, after, before, xdir)
    items = {it.item_key: it for it in dataset.source_items()}
    assert [os.path.basename(p) for p in items["img00"].control_paths] == ["img00.png", "img00_1.png", "img00_2.png"]
    assert items["img00"].control_paths[0].startswith(before) and items["img00"].control_paths[1].startswith(xdir)
    assert [os.path.basename(p) for p in items["img01"].control_paths] == ["img01.png", "img01_1.png"]
    calls = []
    real = driver.encode_images

    def one_at_a_time(vae, images):
        calls.append(len(images))
        return real(vae, images)
    driver.encode_images = one_at_a_time
    with torch.no_grad():
        cache.run_latents(desc, driver, dataset, "x", torch.device("cpu"))
        cache.run_text(desc, driver, dataset, "x", torch.device("cpu"))
    assert set(calls) == {1}                                       # every control encoded on its own
    keys, md = _keys_and_md(items["img00"].latent_cache_path)
    assert keys == ["latent_12x12", "latent_control_0_12x12", "latent_control_1_8x16", "latent_control_2_16x8"]
    assert md["control_stamps"].count(",") == 2
    _, tmd = _keys_and_md(items["img00"].text_encoder_output_cache_path)
    assert tmd["reference_sizes"] == "96x96,128x64,64x128"
    keys, _ = _keys_and_md(items["img01"].latent_cache_path)
    assert keys == ["latent_12x12", "latent_control_0_12x12", "latent_control_1_12x12"]

    # training reads them back in order and the driver sees all of them
    assert dataset.prepare_for_training() == 2
    it = dataset.items()["img00"]
    assert [os.path.basename(p) for p in it.control_paths] == ["img00.png", "img00_1.png", "img00_2.png"]
    assert it.control_signature == (("0", "12x12"), ("1", "8x16"), ("2", "16x8"))
    idx = next(i for i, (_, s) in enumerate(dataset.batches) if dataset.buckets[(96, 96)][s].item_key == "img00")
    batch = dataset.get_batch(idx, random.Random(0))
    assert batch["latents_control_0"].shape == (1, 4, 12, 12) and batch["latents_control_1"].shape == (1, 4, 8, 16)
    assert batch["latents_control_2"].shape == (1, 4, 16, 8) and batch["cond__ref_mask"].shape == (1, 3)
    refs = [batch[k] for k in sorted(k for k in batch if k.startswith("latents_control_"))]
    dit = driver.load_dit("none", "cpu")
    _, info = driver.training_loss(dit, batch["latents"], {"hidden_states": batch["cond__hidden_states"]},
                                   torch.Generator().manual_seed(0), refs=refs)
    assert info["n_refs"] == 3

    # the loss watch's re-encode loads the extras at the same shapes the cache used
    dataset.load_pixels(it)
    assert [c.shape[:2] for c in it.control_content] == [(96, 96), (64, 128), (128, 64)]


def test_skip_existing_reencodes_when_an_extra_picture_arrives(tmp_path):
    """A latent cache with one before-image no longer fits once the photo has an extra picture: --skip_existing
    re-encodes it (and leaves the others alone)."""
    desc = _desc(3)
    driver = RefDriver()
    driver.description = desc
    after, before, xdir = _folders(tmp_path)
    dataset = _dataset(desc, tmp_path, after, before, xdir)
    with torch.no_grad():
        assert cache.run_latents(desc, driver, dataset, "x", torch.device("cpu")) == 2
        assert cache.run_latents(desc, driver, dataset, "x", torch.device("cpu"), skip_existing=True) == 0
        _picture(os.path.join(xdir, "img00_1.png"), 64, 64, 7)
        assert cache.run_latents(desc, driver, dataset, "x", torch.device("cpu"), skip_existing=True) == 1
        assert cache.run_latents(desc, driver, dataset, "x", torch.device("cpu"), skip_existing=True) == 0
    it = {i.item_key: i for i in dataset.source_items()}["img00"]
    keys, _ = _keys_and_md(it.latent_cache_path)
    assert keys == ["latent_12x12", "latent_control_0_12x12", "latent_control_1_12x12"]   # 64x64 -> the target's area
    # the text cache too: its reference sizes changed
    with torch.no_grad():
        assert cache.run_text(desc, driver, dataset, "x", torch.device("cpu")) == 2
        assert cache.run_text(desc, driver, dataset, "x", torch.device("cpu"), skip_existing=True) == 0
        os.remove(os.path.join(xdir, "img00_1.png"))
        assert cache.run_text(desc, driver, dataset, "x", torch.device("cpu"), skip_existing=True) == 1


def test_cache_refuses_more_than_the_familys_references(tmp_path):
    """Fizgig cache.py:164-170: an edit takes up to edit_references before-images per photo, a slider exactly one."""
    import json
    desc = _desc(3)
    after, before, xdir = _folders(tmp_path, {"img00": [(1, 96, 96), (2, 96, 96), (3, 96, 96)]})
    cfg = _dataset(desc, tmp_path, after, before, xdir).config
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(SystemExit, match="more than 3 partner.*up to 3"):
        cache.main(["--family", desc.key, "--stage", "latents", "--dataset", str(p), "--model", "x",
                    "--device", "cpu"])
    # within the limit it caches
    os.remove(os.path.join(xdir, "img00_3.png"))
    cache.main(["--family", desc.key, "--stage", "latents", "--dataset", str(p), "--model", "x", "--device", "cpu"])
    # a slider still pairs one with one, whatever the family's edit count
    with pytest.raises(SystemExit, match="more than 1 partner.*one partner per image"):
        cache.main(["--family", desc.key, "--stage", "latents", "--dataset", str(p), "--model", "x",
                    "--device", "cpu", "--slider"])
    # a family with one reference refuses a second (what it did before)
    one = _desc(1)
    cfg1 = _dataset(one, tmp_path, after, before, xdir).config
    p.write_text(json.dumps(cfg1), encoding="utf-8")
    with pytest.raises(SystemExit, match="more than 1 partner"):
        cache.main(["--family", one.key, "--stage", "latents", "--dataset", str(p), "--model", "x",
                    "--device", "cpu"])


# ---- zero and one reference: unchanged -------------------------------------------------------------------------
def _expected_old_latents(driver, vae, item):
    """What the cache wrote before this change: the image and every control at the TARGET's bucket, the controls
    encoded as one batch."""
    from PIL import Image
    imgs = [resize_image_to_bucket(Image.open(p).convert("RGB"), item.bucket_size)
            for p in [item.image_path, *item.control_paths]]
    z = driver.encode_images(vae, imgs[:1])[0]
    ctrl = driver.encode_images(vae, imgs[1:]) if imgs[1:] else []
    return z, ctrl


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("with_pair", [False, True])
def test_zero_or_one_reference_is_bit_identical(tmp_path, with_pair, batch_size):
    """The cache contents, the batches and the loss of a run with no pair or one before-image per photo are exactly
    what they were: same tensors (bit for bit), same keys, same batch order, same loss from the same seed."""
    from safetensors.torch import load_file
    desc = _desc(3)
    driver = RefDriver()
    driver.description = desc
    after, before, _ = _folders(tmp_path, n=3)
    from training import pipeline
    ds = {"image_directory": after, "cache_directory": str(pipeline.cache_dir_for(after, tmp_path)), "num_repeats": 2}
    if with_pair:
        ds["control_directory"] = before
    cfg = {"datasets": [ds], "resolution": [96, 96], "caption_extension": ".txt",
           "batch_size": batch_size}
    dataset = TrainingDataset(cfg, desc.arch_id, desc.spatial_factor, desc.bucket_step)
    with torch.no_grad():
        cache.run_latents(desc, driver, dataset, "x", torch.device("cpu"))
        cache.run_text(desc, driver, dataset, "x", torch.device("cpu"))
        vae = driver.load_vae("x", "cpu")
        for it in dataset.source_items():
            assert len(it.control_paths) == (1 if with_pair else 0)
            z, ctrl = _expected_old_latents(driver, vae, it)
            sd = load_file(it.latent_cache_path)
            assert sorted(sd) == ["latent_12x12"] + (["latent_control_0_12x12"] if with_pair else [])
            assert torch.equal(sd["latent_12x12"], z)
            if with_pair:
                assert torch.equal(sd["latent_control_0_12x12"], ctrl[0])
            _, md = _keys_and_md(it.text_encoder_output_cache_path)
            assert md.get("reference_sizes", "") == ("96x96" if with_pair else "")
    assert dataset.prepare_for_training() == 6
    # the epoch order is the pre-change algorithm's: shuffle the bucket, then fixed-size slices, then the slices
    for seed in (0, 1, 7):
        bucket = list(dataset.buckets[(96, 96)])
        dataset.shuffle(seed)
        rng = random.Random(seed)
        rng.shuffle(bucket)
        batches = [((96, 96), i) for i in range(0, len(bucket), batch_size)]
        rng.shuffle(batches)
        assert dataset.batches == batches
        assert [it.item_key for it in dataset.buckets[(96, 96)]] == [it.item_key for it in bucket]
    if batch_size > 1:          # the order above is what batch size > 1 changes; the step below is one item's
        return
    # the training step: the same refs list and the same loss as a hand-built step from the cache files
    dataset.shuffle(3)
    batch = dataset.get_batch(0, random.Random(0))
    refs = [batch[k] for k in sorted(k for k in batch if k.startswith("latents_control_"))]
    assert len(refs) == (1 if with_pair else 0)
    dit = driver.load_dit("none", "cpu")
    cond = {"hidden_states": batch["cond__hidden_states"]}
    loss, info = driver.training_loss(dit, batch["latents"], cond, torch.Generator().manual_seed(5),
                                      **({"refs": refs} if refs else {}))
    it = dataset.buckets[(96, 96)][dataset.batches[0][1]]
    sd = load_file(it.latent_cache_path)
    hand_refs = [sd["latent_control_0_12x12"][None]] if with_pair else []
    expected, _ = driver.training_loss(dit, sd["latent_12x12"][None], cond, torch.Generator().manual_seed(5),
                                       **({"refs": hand_refs} if hand_refs else {}))
    assert torch.equal(loss, expected) and info["n_refs"] == len(hand_refs)


def test_an_extras_folder_without_pictures_changes_nothing(tmp_path):
    """Pointing at an empty (or absent) extra pictures folder leaves a one-reference run as it was."""
    from safetensors.torch import load_file
    desc = _desc(3)
    driver = RefDriver()
    driver.description = desc
    after, before, xdir = _folders(tmp_path)
    plain = _dataset(desc, tmp_path, after, before)
    with_dir = _dataset(desc, tmp_path, after, before, xdir)
    missing = _dataset(desc, tmp_path, after, before, str(tmp_path / "nowhere"))
    with torch.no_grad():
        cache.run_latents(desc, driver, plain, "x", torch.device("cpu"))
    saved = {it.item_key: load_file(it.latent_cache_path) for it in plain.source_items()}
    for ds in (with_dir, missing):
        assert [it.control_paths for it in ds.source_items()] == [it.control_paths for it in plain.source_items()]
        with torch.no_grad():
            assert cache.run_latents(desc, driver, ds, "x", torch.device("cpu"), skip_existing=True) == 0
            cache.run_latents(desc, driver, ds, "x", torch.device("cpu"))
        for it in ds.source_items():
            sd = load_file(it.latent_cache_path)
            assert sorted(sd) == sorted(saved[it.item_key])
            assert all(torch.equal(sd[k], saved[it.item_key][k]) for k in sd)


# ---- batches above 1 (TagScribeR) -----------------------------------------------------------------------------
def test_batches_above_one_stack_only_matching_reference_shapes(tmp_path):
    """TagScribeR trains batches above 1: photos whose extras differ in count or shape cannot stack, so a bucket is
    dealt out by its cached reference shapes (and a bucket where they all match is batched as before)."""
    desc = _desc(3)
    driver = RefDriver()
    driver.description = desc
    after, before, xdir = _folders(tmp_path, {"img00": [(1, 128, 64)], "img02": [(1, 128, 64)],
                                              "img03": [(1, 64, 128), (2, 96, 96)]}, n=5)
    dataset = _dataset(desc, tmp_path, after, before, xdir, batch_size=2)
    with torch.no_grad():
        cache.run_latents(desc, driver, dataset, "x", torch.device("cpu"))
        cache.run_text(desc, driver, dataset, "x", torch.device("cpu"))
    assert dataset.prepare_for_training() == 5
    for seed in range(4):
        dataset.shuffle(seed)
        seen = []
        for i in range(len(dataset)):
            reso, start = dataset.batches[i]
            batch = dataset.get_batch(i, random.Random(seed))
            chunk = [dataset.items()[k] for k in batch["item_keys"]]
            assert len({it.control_signature for it in chunk}) == 1
            assert chunk[0] is dataset.buckets[reso][start]
            seen += batch["item_keys"]
            assert batch["latents"].shape[0] == len(chunk)
            assert sum(1 for k in batch if k.startswith("latents_control_")) == len(chunk[0].control_paths)
        assert sorted(seen) == ["img00", "img01", "img02", "img03", "img04"]
    sigs = {it.item_key: it.control_signature for it in dataset.items().values()}
    assert sigs["img00"] == sigs["img02"] != sigs["img01"] == sigs["img04"] != sigs["img03"]


# ---- the drivers take several references -----------------------------------------------------------------------
def test_klein_packs_each_reference_at_its_own_time_offset():
    """Fizgig klein/position.py pack_control_latent (:120-145): reference i rides at time coordinate 10 * (i + 1) with
    its own h / w grid, in the order given, after the image tokens (klein/driver.py:176-198); the loss reads only the
    image tokens, however many references there are and whatever their shapes."""
    from tests.test_training_klein import _cond, _dit
    from training.families.klein import sampling as Ks
    from training.families.klein.driver import KleinDriver
    from training.lora import FamilyLoRA
    refs = [torch.randn(1, 16, 6, 4), torch.randn(1, 16, 2, 8), torch.randn(1, 16, 3, 3), torch.randn(1, 16, 4, 2)]
    tok, ids = Ks.pack_refs(refs)
    assert tok.shape == (1, 24 + 16 + 9 + 8, 16) and ids.shape == (1, 57, 4)
    t = ids[0, :, 0].tolist()
    assert t[:24] == [10] * 24 and t[24:40] == [20] * 16 and t[40:49] == [30] * 9 and t[49:] == [40] * 8
    assert ids[0, 24:40, 1].max() == 1 and ids[0, 24:40, 2].max() == 7      # the second reference's own grid
    assert torch.equal(tok[0, 24:40], refs[1].flatten(2).transpose(1, 2)[0])
    drv = KleinDriver()
    drv.description = registry.get("klein9b")
    dit = _dit()
    net = FamilyLoRA(dit, drv)
    net.add_trainable(4, 4)
    lat, cond = torch.randn(1, 16, 6, 4), _cond(1)
    losses = [drv.training_loss(dit, lat, cond, torch.Generator().manual_seed(2), refs=refs[:n])[0].item()
              for n in (0, 1, 2, 4)]
    assert len(set(losses)) == 4 and all(np.isfinite(losses))               # each extra reference reaches the DiT
    loss, _ = drv.training_loss(dit, lat, cond, torch.Generator().manual_seed(2), refs=refs)
    loss.backward()
    assert all(p.grad is not None for p in net.parameters())
    with torch.no_grad():
        out = drv.generate(dit, {"text_embed": cond["text_embed"][0]}, 64, 96, steps=2, seed=1, refs=refs)
    assert out.shape == (1, 16, 6, 4)


def test_qwen_takes_several_references_ahead_of_the_target():
    """Fizgig qwen_image21/driver.py _ref_sequence / training_loss (:174-209): every reference's latents are packed
    ahead of the target in the order the text encoder saw them, each with its own img_shapes entry (its own RoPE
    block, model.py Rope3D), and the ref_mask must hold one vision slot per 2x2 latent tokens of all of them."""
    from training.families.qwen_image21 import sampling as S
    from training.families.qwen_image21.driver import QwenImage21Driver
    from training.families.qwen_image21.model import QwenImage21DiT
    from training.lora import FamilyLoRA
    torch.manual_seed(0)
    dit = QwenImage21DiT(patch_size=1, in_channels=64, out_channels=64, num_layers=2, attention_head_dim=16,
                         num_attention_heads=2, context_in_dim=32, mlp_ratio=3, axes_dims_rope=(4, 6, 6),
                         eps=1e-6, causal_condition=True).to(torch.bfloat16).requires_grad_(False)
    drv = QwenImage21Driver()
    drv.description = registry.get("qwen_image21")
    net = FamilyLoRA(dit, drv)
    net.add_trainable(4, 4)
    refs = [torch.randn(1, 64, 4, 4), torch.randn(1, 64, 2, 6), torch.randn(1, 64, 6, 2)]     # 16 + 12 + 12 tokens
    slots = sum(r.shape[-2] * r.shape[-1] for r in refs) // 4                                 # 10 vision slots
    L = 3 + slots + 2
    mask = torch.zeros(L, dtype=torch.bool)
    mask[3:3 + slots] = True
    cond = {"hidden_states": torch.randn(1, L, 32), "ref_mask": mask[None]}
    tokens, shapes = drv._ref_sequence(refs, mask, "cpu")
    assert tokens.shape == (1, 40, 64) and shapes == [(1, 4, 4), (1, 2, 6), (1, 6, 2)]
    assert torch.equal(tokens[0, 16:28], S.pack(refs[1].float())[0])
    latents = torch.randn(1, 64, 4, 4)
    loss, _ = drv.training_loss(dit, latents, cond, torch.Generator().manual_seed(0), refs=refs)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None for p in net.parameters())
    with pytest.raises(RuntimeError, match="reference size mismatch"):       # two references for ten slots
        drv.training_loss(dit, latents, cond, torch.Generator().manual_seed(0), refs=refs[:2])
    with torch.no_grad():
        out = drv.generate(dit, {"hidden_states": cond["hidden_states"][0], "ref_mask": mask}, 64, 64, steps=2,
                           seed=1, refs=refs)
    assert out.shape == (1, 16, 64)                                           # 4x4 target tokens, refs dropped


def test_preflight_pairs_as_the_dataset_does(tmp_path):
    """photo_1.png in the originals belongs to the edited photo_1, so photo (with no original of its own) is reported
    missing before the run instead of training without a reference (longest stems claim first, as the dataset)."""
    from PIL import Image

    from training import pipeline
    after, before = tmp_path / "after", tmp_path / "before"
    after.mkdir()
    before.mkdir()
    for n in ("photo.png", "photo_1.png"):
        Image.new("RGB", (32, 32)).save(after / n)
    Image.new("RGB", (32, 32)).save(before / "photo_1.png")
    matched, missing, _ = pipeline.edit_pairs(str(after), str(before))
    assert matched == ["photo_1"] and missing == ["photo"]
    assert pipeline.pair_problems(str(after), str(before)) == ([], [])
