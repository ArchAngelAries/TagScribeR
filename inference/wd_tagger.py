"""WD (SmilingWolf) booru taggers via ONNX Runtime.

Improvements over the original tagger:
* Input size read from the model (v2 convnext = 448, v3 = 448; others vary)
  instead of hard-coded.
* Reference preprocessing: pad to a white square, then resize (no squashing).
* Rating / general / character categories with separate thresholds.
* Uses the fastest available ONNX execution provider (DirectML, ROCm/MIGraphX,
  CUDA) and falls back to CPU. Plain ``onnxruntime`` is CPU-only; installing
  ``onnxruntime-directml`` enables AMD/NVIDIA/Intel GPUs on Windows.
* No pandas dependency; thread-safe lazy loading; batched inference.
"""
from __future__ import annotations

import csv
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from core import captions
from inference.base import (CAP_BATCH, CAP_TAGS, Cancelled, CaptionRequest, InferenceError,
                            ProgressFn, Provider, ProviderSpec)

log = logging.getLogger(__name__)

KNOWN_TAGGERS = {
    "SmilingWolf/wd-eva02-large-tagger-v3": "WD EVA02 Large v3 (most accurate, slower)",
    "SmilingWolf/wd-vit-large-tagger-v3": "WD ViT Large v3",
    "SmilingWolf/wd-swinv2-tagger-v3": "WD SwinV2 v3 (balanced)",
    "SmilingWolf/wd-convnext-tagger-v3": "WD ConvNext v3",
    "SmilingWolf/wd-vit-tagger-v3": "WD ViT v3 (fast)",
    "SmilingWolf/wd-v1-4-convnextv2-tagger-v2": "WD ConvNextV2 v2 (legacy default)",
}

RATING, GENERAL, CHARACTER = 9, 0, 4
# Tags where underscores are part of the meaning (kaomoji).
KAOMOJI = {"0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<", "3_3",
           "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||"}

GPU_PROVIDERS = ("CUDAExecutionProvider", "ROCMExecutionProvider", "MIGraphXExecutionProvider",
                 "DmlExecutionProvider", "CoreMLExecutionProvider")


@dataclass
class TagPrediction:
    rating: dict[str, float] = field(default_factory=dict)
    general: list[tuple[str, float]] = field(default_factory=list)
    character: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class TagFormat:
    general_threshold: float = 0.35
    character_threshold: float = 0.85
    max_tags: int = 50
    underscores_to_spaces: bool = True
    escape_parentheses: bool = False
    include_rating: bool = False
    blacklist: tuple[str, ...] = ()


def format_tags(pred: TagPrediction, fmt: TagFormat) -> list[str]:
    """Characters first, then general tags by confidence; thresholds, blacklist, max count."""
    black = {b.strip().lower().replace("_", " ") for b in fmt.blacklist if b.strip()}
    chosen: list[str] = []
    for tag, score in pred.character:
        if score >= fmt.character_threshold:
            chosen.append(tag)
    for tag, score in pred.general:
        if score >= fmt.general_threshold:
            chosen.append(tag)
    out = []
    for t in chosen:
        if fmt.underscores_to_spaces and t not in KAOMOJI:
            t = t.replace("_", " ")
        if t.lower().replace("_", " ") in black:
            continue
        out.append(captions.normalize_tag(t, escape_parentheses=fmt.escape_parentheses))
    out = out[: max(1, fmt.max_tags)]
    if fmt.include_rating and pred.rating:
        out.insert(0, max(pred.rating.items(), key=lambda kv: kv[1])[0])
    return captions.dedupe_tags(out)


class WDTagger(Provider):
    capabilities = frozenset({CAP_TAGS, CAP_BATCH})
    preferred_chunk = 8

    def __init__(self, spec: ProviderSpec):
        super().__init__(spec)
        self._lock = threading.Lock()
        self.session = None
        self.tags: list[tuple[str, int]] = []
        self.size = 448
        self.dynamic_batch = False
        self.fmt = TagFormat(**dict(spec.opt("format", ()) or ()))

    @property
    def display_name(self) -> str:
        return KNOWN_TAGGERS.get(self.spec.model, Path(self.spec.model).name)

    @property
    def loaded(self) -> bool:
        return self.session is not None

    def _files(self, report: ProgressFn) -> tuple[str, str]:
        local = Path(self.spec.model)
        if local.is_dir():
            m, t = local / "model.onnx", local / "selected_tags.csv"
            if not (m.is_file() and t.is_file()):
                raise InferenceError(f"{local} must contain model.onnx and selected_tags.csv.")
            return str(m), str(t)
        from huggingface_hub import hf_hub_download
        files = []
        for name in ("model.onnx", "selected_tags.csv"):
            try:
                files.append(hf_hub_download(self.spec.model, name, local_files_only=True))
            except Exception:
                report(f"Downloading tagger {self.display_name} ({name})…")
                try:
                    files.append(hf_hub_download(self.spec.model, name))
                except Exception as e:
                    raise InferenceError(f"Could not download tagger '{self.spec.model}': {e}",
                                         "Check your internet connection, or choose a local tagger folder.") from e
        return files[0], files[1]

    def load(self, report: ProgressFn, cancel: threading.Event | None = None) -> None:
        with self._lock:
            if self.session is not None:
                return
            try:
                import onnxruntime as ort
            except ImportError as e:
                raise InferenceError("onnxruntime is not installed.", "Run update.bat.") from e
            model_path, tags_path = self._files(report)
            with open(tags_path, newline="", encoding="utf-8") as f:
                self.tags = [(row["name"], int(row["category"])) for row in csv.DictReader(f)]
            available = ort.get_available_providers()
            if self.spec.opt("device", "auto") == "cpu":
                providers = ["CPUExecutionProvider"]
            else:
                providers = [p for p in GPU_PROVIDERS if p in available] + ["CPUExecutionProvider"]
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            try:
                self.session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
            except Exception as e:
                log.warning("Tagger failed with %s (%s); retrying on CPU.", providers, e)
                self.session = ort.InferenceSession(model_path, sess_options=opts,
                                                    providers=["CPUExecutionProvider"])
            shape = self.session.get_inputs()[0].shape  # [batch, H, W, 3]
            self.size = int(shape[1]) if isinstance(shape[1], int) else 448
            self.dynamic_batch = not isinstance(shape[0], int)
            report(f"Tagger ready: {self.display_name} on {self.session.get_providers()[0].replace('ExecutionProvider', '')}")

    def unload(self) -> None:
        with self._lock:
            self.session = None

    def _prep(self, img: Image.Image) -> np.ndarray:
        img = img.convert("RGB")
        side = max(img.size)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
        if side != self.size:
            canvas = canvas.resize((self.size, self.size), Image.Resampling.BICUBIC)
        arr = np.asarray(canvas, dtype=np.float32)
        return arr[:, :, ::-1]  # RGB -> BGR, 0-255, as the models were trained

    def predict(self, images: Sequence[Image.Image]) -> list[TagPrediction]:
        if self.session is None:
            raise InferenceError("Tagger not loaded.")
        batch = np.stack([self._prep(im) for im in images])
        name = self.session.get_inputs()[0].name
        if self.dynamic_batch:
            probs = self.session.run(None, {name: batch})[0]
        else:
            probs = np.concatenate([self.session.run(None, {name: b[None]})[0] for b in batch])
        out = []
        for row in probs:
            pred = TagPrediction()
            for (tag, cat), p in zip(self.tags, row):
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

    def generate(self, images: Sequence[Image.Image], request: CaptionRequest,
                 cancel: threading.Event | None = None,
                 prompts: Sequence[str] | None = None) -> list[str | Exception]:
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        try:
            preds = self.predict(images)
        except InferenceError:
            raise
        except Exception as e:
            return [InferenceError(f"Tagging failed: {e}") for _ in images]
        return [captions.join_tags(format_tags(p, self.fmt)) for p in preds]
