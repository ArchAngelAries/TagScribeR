"""Dataset health checks: duplicates, near-duplicates, blur, resolution, aspect buckets.

Pure numpy/Pillow so it's testable and reusable by the future training module
(the bucket logic matches the kohya / SDXL convention: buckets in 64 px steps
whose area doesn't exceed the training resolution squared).
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

FLAG_DUPLICATE = "duplicate"
FLAG_NEAR_DUPLICATE = "similar"
FLAG_BLURRY = "blurry"
FLAG_LOW_RES = "lowres"
FLAG_HEAVY_CROP = "crop"


@dataclass
class ImageStats:
    key: str
    width: int = 0
    height: int = 0
    file_hash: str = ""
    dhash: int = 0
    sharpness: float = 0.0
    error: str = ""


def analyze(path: str | os.PathLike) -> ImageStats:
    """Decode once (downscaled) and compute everything the checks need."""
    s = ImageStats(key=str(path))
    try:
        h = hashlib.blake2b(digest_size=16)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        s.file_hash = h.hexdigest()
        with Image.open(path) as im:
            im.draft("L", (768, 768))
            im = ImageOps.exif_transpose(im)
            s.width, s.height = _true_size(path)
            gray = im.convert("L")
            gray.thumbnail((512, 512), Image.Resampling.BILINEAR)
            s.dhash = dhash(gray)
            s.sharpness = laplacian_variance(np.asarray(gray, dtype=np.float32))
    except Exception as e:
        s.error = str(e)
    return s


def _true_size(path) -> tuple[int, int]:
    from core.dataset import read_image_info
    info = read_image_info(path)
    return info.width, info.height


def dhash(gray: Image.Image, size: int = 8) -> int:
    """64-bit difference hash: robust to resizing, re-compression and small edits."""
    small = np.asarray(gray.resize((size + 1, size), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def laplacian_variance(a: np.ndarray) -> float:
    """Variance of the Laplacian — low values mean few sharp edges (blur)."""
    if a.shape[0] < 3 or a.shape[1] < 3:
        return 0.0
    lap = (a[1:-1, :-2] + a[1:-1, 2:] + a[:-2, 1:-1] + a[2:, 1:-1] - 4 * a[1:-1, 1:-1])
    return float(lap.var())


# ---------------------------------------------------------------------------- grouping
_COPY_WORD = re.compile(r"\b(copy|duplicate)\b", re.IGNORECASE)
_NUM_SUFFIX = re.compile(r"(\s*\(\d+\)|_\d+| - \d+)$")


def keeper_order(keys: list[str]) -> list[str]:
    """Order duplicate files so the one to keep comes first.

    "copy"/"duplicate" in the name marks a copy. A numbered suffix ("photo (1)",
    "photo_2") only counts when the same name without it is in the group, so
    datasets that are numbered throughout ("img (1)", "img (2)"…) aren't penalised.
    Ties go to the oldest file, then the shorter name.
    """
    stems = {Path(k).stem.lower() for k in keys}

    def score(k):
        stem = Path(k).stem
        base = _NUM_SUFFIX.sub("", stem).lower()
        numbered_copy = base != stem.lower() and base in stems
        try:
            mtime = os.path.getmtime(k)
        except OSError:
            mtime = float("inf")
        return (bool(_COPY_WORD.search(stem)), numbered_copy, mtime, len(stem), stem)
    return sorted(keys, key=score)


def exact_duplicate_groups(stats: list[ImageStats]) -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for s in stats:
        if s.file_hash:
            groups[s.file_hash].append(s.key)
    return [g for g in groups.values() if len(g) > 1]


def near_duplicate_groups(stats: list[ImageStats], max_distance: int = 6) -> list[list[str]]:
    """Cluster images whose perceptual hashes differ by <= max_distance bits (of 64).

    Exact duplicates are excluded (reported separately). Uses union-find over a
    bucketed comparison so it stays fast for thousands of images.
    """
    items = [s for s in stats if s.file_hash and not s.error]
    seen_hash: dict[str, int] = {}
    uniq = []
    for s in items:  # one representative per exact-duplicate set
        if s.file_hash not in seen_hash:
            seen_hash[s.file_hash] = len(uniq)
            uniq.append(s)
    parent = list(range(len(uniq)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Pigeonhole: if two 64-bit hashes differ in <= d bits, at least one of (d+1) bands matches exactly.
    bands = max_distance + 1
    width = math.ceil(64 / bands)
    for b in range(bands):
        shift = b * width
        mask = (1 << width) - 1
        buckets: dict[int, list[int]] = defaultdict(list)
        for i, s in enumerate(uniq):
            buckets[(s.dhash >> shift) & mask].append(i)
        for members in buckets.values():
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    i, j = members[x], members[y]
                    if find(i) != find(j) and hamming(uniq[i].dhash, uniq[j].dhash) <= max_distance:
                        parent[find(i)] = find(j)
    clusters: dict[int, list[str]] = defaultdict(list)
    for i, s in enumerate(uniq):
        clusters[find(i)].append(s.key)
    return [sorted(c) for c in clusters.values() if len(c) > 1]


def blurry(stats: list[ImageStats], relative: float = 0.3, absolute: float = 40.0) -> list[str]:
    """Images much softer than the dataset's typical sharpness (or very soft in absolute terms)."""
    vals = [s.sharpness for s in stats if not s.error]
    if not vals:
        return []
    median = float(np.median(vals))
    cutoff = max(absolute, median * relative) if median > 0 else absolute
    return [s.key for s in stats if not s.error and s.sharpness < cutoff]


def low_resolution(stats: list[ImageStats], min_side: int) -> list[str]:
    return [s.key for s in stats if not s.error and s.width and min(s.width, s.height) < min_side]


# ---------------------------------------------------------------------------- buckets
def make_buckets(resolution: int, step: int = 64, min_side: int = 256, max_side: int | None = None,
                 max_ratio: float = 4.0) -> list[tuple[int, int]]:
    """SDXL/kohya-style buckets: multiples of ``step`` with area <= resolution²."""
    max_side = max_side or resolution * 2
    area = resolution * resolution
    out = set()
    w = min_side
    while w <= max_side:
        h = min(max_side, (area // w) // step * step)
        if h >= min_side and max(w, h) / min(w, h) <= max_ratio:
            out.add((w, h))
            out.add((h, w))
        w += step
    return sorted(out, key=lambda b: (b[0] / b[1], b[0]))


@dataclass
class BucketFit:
    bucket: tuple[int, int]
    crop_fraction: float     # share of the image lost when scaling to cover the bucket then cropping
    upscale: bool            # image smaller than the bucket


def fit_bucket(width: int, height: int, buckets: list[tuple[int, int]]) -> BucketFit:
    ar = math.log(width / height)
    bw, bh = min(buckets, key=lambda b: abs(math.log(b[0] / b[1]) - ar))
    scale = max(bw / width, bh / height)
    covered_w, covered_h = width * scale, height * scale
    crop = 1 - (bw * bh) / (covered_w * covered_h)
    return BucketFit((bw, bh), max(0.0, crop), scale > 1.0)


@dataclass
class HealthReport:
    stats: dict[str, ImageStats] = field(default_factory=dict)
    exact: list[list[str]] = field(default_factory=list)
    similar: list[list[str]] = field(default_factory=list)
    blurry: list[str] = field(default_factory=list)
    low_res: list[str] = field(default_factory=list)
    heavy_crop: list[str] = field(default_factory=list)
    bucket_counts: dict[tuple[int, int], int] = field(default_factory=dict)
    unreadable: list[str] = field(default_factory=list)

    def flags(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = defaultdict(set)
        for g in self.exact:
            for k in keeper_order(g)[1:]:   # keep the most "original-looking" file, flag the copies
                out[k].add(FLAG_DUPLICATE)
        for g in self.similar:
            for k in g:
                out[k].add(FLAG_NEAR_DUPLICATE)
        for k in self.blurry:
            out[k].add(FLAG_BLURRY)
        for k in self.low_res:
            out[k].add(FLAG_LOW_RES)
        for k in self.heavy_crop:
            out[k].add(FLAG_HEAVY_CROP)
        return out


def build_report(stats: list[ImageStats], resolution: int = 1024, similarity: int = 6,
                 blur_relative: float = 0.3, crop_limit: float = 0.2) -> HealthReport:
    r = HealthReport(stats={s.key: s for s in stats})
    r.unreadable = [s.key for s in stats if s.error]
    r.exact = exact_duplicate_groups(stats)
    r.similar = near_duplicate_groups(stats, similarity)
    r.blurry = blurry(stats, blur_relative)
    r.low_res = low_resolution(stats, min_side=int(resolution * 0.5))
    buckets = make_buckets(resolution)
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for s in stats:
        if s.error or not s.width:
            continue
        fit = fit_bucket(s.width, s.height, buckets)
        counts[fit.bucket] += 1
        if fit.crop_fraction > crop_limit:
            r.heavy_crop.append(s.key)
    r.bucket_counts = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    return r
