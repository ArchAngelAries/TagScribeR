import shutil

import numpy as np
from PIL import Image, ImageFilter

from core import health


def scene(seed: int, size=(512, 384)) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (24, 32, 3), dtype=np.uint8)
    return Image.fromarray(base).resize(size, Image.Resampling.NEAREST)  # blocky but detailed


def test_dhash_survives_resize_and_recompression(tmp_path):
    a = scene(1)
    a.save(tmp_path / "a.png")
    a.resize((256, 192)).save(tmp_path / "small.jpg", quality=70)
    scene(2).save(tmp_path / "other.png")
    s = {p.name: health.analyze(p) for p in tmp_path.iterdir()}
    assert health.hamming(s["a.png"].dhash, s["small.jpg"].dhash) <= 6
    assert health.hamming(s["a.png"].dhash, s["other.png"].dhash) > 10


def test_groups_exact_and_near(tmp_path):
    scene(1).save(tmp_path / "a.png")
    shutil.copy(tmp_path / "a.png", tmp_path / "a_copy.png")
    scene(1).resize((300, 225)).save(tmp_path / "a_small.jpg", quality=80)
    scene(3).save(tmp_path / "b.png")
    stats = [health.analyze(p) for p in sorted(tmp_path.iterdir())]
    exact = health.exact_duplicate_groups(stats)
    assert len(exact) == 1 and len(exact[0]) == 2
    near = health.near_duplicate_groups(stats)
    assert len(near) == 1 and any(k.endswith("a_small.jpg") for k in near[0])
    assert not any(k.endswith("b.png") for g in near for k in g)


def test_blur_detection_relative(tmp_path):
    for i in range(5):
        scene(10 + i).save(tmp_path / f"sharp{i}.png")
    scene(20).filter(ImageFilter.GaussianBlur(6)).save(tmp_path / "soft.png")
    stats = [health.analyze(p) for p in tmp_path.iterdir()]
    flagged = health.blurry(stats)
    assert len(flagged) == 1 and flagged[0].endswith("soft.png")


def test_buckets_and_fit():
    b = health.make_buckets(1024)
    assert (1024, 1024) in b and (832, 1216) in b and (1216, 832) in b
    assert all(w % 64 == 0 and h % 64 == 0 and w * h <= 1024 * 1024 for w, h in b)
    sq = health.fit_bucket(2000, 2000, b)
    assert sq.bucket == (1024, 1024) and sq.crop_fraction < 0.01 and not sq.upscale
    wide = health.fit_bucket(3000, 600, b)
    assert wide.bucket[0] > wide.bucket[1] and wide.crop_fraction > 0.0
    small = health.fit_bucket(400, 400, b)
    assert small.upscale


def test_report_flags(tmp_path):
    scene(1).save(tmp_path / "a.png")
    shutil.copy(tmp_path / "a.png", tmp_path / "b.png")
    scene(5, size=(200, 150)).save(tmp_path / "tiny.png")
    Image.new("RGB", (3000, 500)).save(tmp_path / "pano.png")
    (tmp_path / "broken.png").write_bytes(b"nope")
    stats = [health.analyze(p) for p in sorted(tmp_path.iterdir())]
    r = health.build_report(stats, resolution=1024)
    flags = r.flags()
    assert health.FLAG_DUPLICATE in flags[str(tmp_path / "b.png")]
    assert health.FLAG_LOW_RES in flags[str(tmp_path / "tiny.png")]
    assert health.FLAG_HEAVY_CROP in flags[str(tmp_path / "pano.png")]
    assert r.unreadable == [str(tmp_path / "broken.png")]
    assert sum(r.bucket_counts.values()) == 4


def test_keeper_prefers_original_names():
    k = health.keeper_order
    assert k(["d/exact copy.png", "d/portrait (1).png"])[0] == "d/portrait (1).png"  # numbered datasets OK
    assert k(["d/photo (1).png", "d/photo.png"])[0] == "d/photo.png"
    assert k(["d/img - Copy.png", "d/img_2.png", "d/img.png"])[0] == "d/img.png"
