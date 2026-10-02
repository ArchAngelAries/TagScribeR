# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/training/metadata.py
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: architecture strings come only from family descriptions (training.registry); the Klein /
# Krea 2 / MiniMax constants and safetensors hashing helpers are dropped. Field rules are Fizgig's.
"""SAI Model Spec 1.0.0 metadata for LoRA safetensors files (https://github.com/Stability-AI/ModelSpec).

Note for families: the arch ids used in cache filenames must contain no underscore (`qwenimage21`), because the
latent cache filename `<image>_<WxH>_<arch>.safetensors` is parsed with split("_").
"""
import base64
import datetime
import logging
import os
from io import BytesIO
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)

MODELSPEC_TITLE = "modelspec.title"


def build_metadata(architecture: str, timestamp: float, title: Optional[str] = None,
                   reso: Optional[Union[str, int, Tuple[int, int]]] = None, author: Optional[str] = None,
                   description: Optional[str] = None, license: Optional[str] = None, tags: Optional[str] = None,
                   merged_from: Optional[str] = None, timesteps: Optional[Tuple[int, int]] = None,
                   trigger_phrase: Optional[str] = None, thumbnail: Optional[str] = None,
                   usage_hint: Optional[str] = None, is_lora: bool = True) -> dict:
    """SAI ModelSpec fields for a LoRA trained on the family whose arch id is `architecture`."""
    from training.registry import by_arch_id
    fam = by_arch_id(architecture)
    if fam is None:
        raise ValueError(f"Unknown architecture: {architecture}")
    # a reasonable default beats a blank field: only when there is a trigger to name and no explicit hint
    if usage_hint is None and trigger_phrase:
        usage_hint = f"Include '{trigger_phrase}' in your prompt."
    md = {"modelspec.sai_model_spec": "1.0.0",
          "modelspec.architecture": fam.modelspec_arch + ("/lora" if is_lora else ""),
          "modelspec.implementation": fam.implementation,
          MODELSPEC_TITLE: title if title is not None else f"LoRA@{timestamp}"}
    for key, value in (("modelspec.author", author), ("modelspec.description", description),
                       ("modelspec.license", license), ("modelspec.tags", tags),
                       ("modelspec.merged_from", merged_from), ("modelspec.trigger_phrase", trigger_phrase),
                       ("modelspec.thumbnail", thumbnail), ("modelspec.usage_hint", usage_hint)):
        if value is not None:
            md[key] = value
    md["modelspec.date"] = datetime.datetime.fromtimestamp(int(timestamp)).isoformat()
    if reso is None:
        reso = (1024, 1024)
    elif isinstance(reso, str):
        reso = tuple(map(int, reso.split(",")))
    elif isinstance(reso, int):
        reso = (reso, reso)
    if len(reso) == 1:
        reso = (reso[0], reso[0])
    md["modelspec.resolution"] = f"{reso[0]}x{reso[1]}"
    if timesteps is not None:
        if isinstance(timesteps, (str, int)):
            timesteps = (timesteps, timesteps)
        if len(timesteps) == 1:
            timesteps = (timesteps[0], timesteps[0])
        md["modelspec.timestep_range"] = f"{timesteps[0]},{timesteps[1]}"
    return md


def latest_sample_image(output_dir: Optional[str]) -> Optional[str]:
    """Most recently written preview under <output_dir>/sample/, if any (the default `modelspec.thumbnail` source)."""
    if not output_dir:
        return None
    sample_dir = os.path.join(output_dir, "sample")
    if not os.path.isdir(sample_dir):
        return None
    exts = (".png", ".jpg", ".jpeg", ".webp")
    candidates = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.lower().endswith(exts)]
    return max(candidates, key=os.path.getmtime) if candidates else None


def sample_for_epoch(output_dir: Optional[str], output_name: Optional[str], epoch: int) -> Optional[str]:
    """The newest preview written for THIS epoch (`<name>_e{epoch:06d}_*`), or None. The epoch checkpoint is saved
    before its own preview renders, so picking the newest file would give every checkpoint the previous epoch's."""
    if not output_dir or not output_name:
        return None
    sample_dir = os.path.join(output_dir, "sample")
    if not os.path.isdir(sample_dir):
        return None
    prefix = f"{output_name}_e{int(epoch):06d}_"
    exts = (".png", ".jpg", ".jpeg", ".webp")
    candidates = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir)
                  if f.startswith(prefix) and f.lower().endswith(exts)]
    return max(candidates, key=os.path.getmtime) if candidates else None


def refresh_checkpoint_thumbnail(lora_path: str, image_path: str) -> bool:
    """Re-embed `modelspec.thumbnail` in an already-saved checkpoint from `image_path` (after the epoch's preview
    renders). Tensors untouched; rewritten atomically. Best-effort: a failure logs and leaves the file."""
    uri = thumbnail_data_uri(image_path)
    if not uri or not lora_path or not os.path.exists(lora_path):
        return False
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
        tensors, meta = {}, {}
        with safe_open(lora_path, framework="pt") as f:
            meta = dict(f.metadata() or {})
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        if meta.get("modelspec.thumbnail") == uri:
            return True
        meta["modelspec.thumbnail"] = uri
        tmp = lora_path + ".thumb.tmp"
        save_file(tensors, tmp, metadata=meta)
        os.replace(tmp, lora_path)
        logger.info("[thumbnail] %s now carries its own epoch's preview (%s)", os.path.basename(lora_path),
                    os.path.basename(image_path))
        return True
    except Exception:
        logger.warning("could not refresh the thumbnail of %s", lora_path, exc_info=True)
        return False


def thumbnail_data_uri(image_path: Optional[str], max_size: int = 512, quality: int = 85) -> Optional[str]:
    """Downscale an image into a `modelspec.thumbnail` data URI (ComfyUI's model browser card art). Best-effort: a
    missing or broken thumbnail never fails a checkpoint save."""
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_size, max_size))
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=quality)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    except Exception:
        logger.warning(f"could not build metadata thumbnail from {image_path}", exc_info=True)
        return None


def resolve_title(output_name: Optional[str], trigger_phrase: Optional[str]) -> Optional[str]:
    """The metadata title: the output name (the trigger phrase if the name is empty)."""
    return output_name or trigger_phrase


def read_metadata(path: str) -> dict:
    """A safetensors file's metadata header (no tensors loaded)."""
    if not path.endswith(".safetensors"):
        return {}
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        return dict(f.metadata() or {})
