"""Model catalog and discovery of local Hugging Face-format VLM folders."""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from core import paths

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CatalogModel:
    repo_id: str
    label: str
    approx_gb: float        # bf16 weights on disk; add 1-3 GB for activations / image tokens
    notes: str = ""
    family: str = ""
    # Model-card recommended sampling for non-thinking caption use.
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    gated: bool = False     # requires accepting a license / HF login
    revision: str = ""      # pinned commit / tag; empty = the repo's default branch (no pins are known yet)
    remote_code: bool = False   # True only if the repo needs its own *.py to load (none of the catalog does)


# Verified against the Hugging Face Hub on 2026-10-02. Sizes are bf16 weights.
# Larger models (Qwen3.6/3.8 27B, 35B-A3B MoE, Gemma 4 26B/31B) don't fit
# consumer GPUs in bf16; run their GGUF quants through LM Studio / llama-server
# with the API provider instead.
CATALOG: tuple[CatalogModel, ...] = (
    CatalogModel("Qwen/Qwen3-VL-8B-Instruct", "Qwen3-VL 8B Instruct", 17.5,
                 "Detailed captions, no thinking overhead. Same family as Fizgig's text encoder. "
                 "Comfortable on 20-24 GB.", "qwen3_vl"),
    CatalogModel("Qwen/Qwen3-VL-4B-Instruct", "Qwen3-VL 4B Instruct", 8.9,
                 "Mature, fast and AMD-friendly (standard attention). Good default for 12-16 GB.", "qwen3_vl"),
    CatalogModel("Qwen/Qwen3-VL-2B-Instruct", "Qwen3-VL 2B Instruct", 4.3,
                 "Fastest Qwen3-VL; works on 8 GB.", "qwen3_vl"),
    CatalogModel("Qwen/Qwen3.5-9B", "Qwen3.5 9B", 19.3,
                 "Strongest single-GPU Qwen (Feb 2026). Very tight on 20 GB: lower Max image size. "
                 "On AMD its linear-attention layers use a slower PyTorch fallback.", "qwen3_5"),
    CatalogModel("Qwen/Qwen3.5-4B", "Qwen3.5 4B", 9.3,
                 "Newest small Qwen (Feb 2026), strong detail for its size. Slower on AMD than Qwen3-VL "
                 "(no ROCm kernels for its linear attention yet).", "qwen3_5"),
    CatalogModel("Qwen/Qwen3.5-2B", "Qwen3.5 2B", 4.5, "Fast drafts on small GPUs.", "qwen3_5"),
    CatalogModel("google/gemma-4-E4B-it", "Gemma 4 E4B", 16.0,
                 "Google's Mar 2026 multimodal model; fluent natural-language captions. "
                 "Larger on disk than its name suggests.", "gemma4", temperature=1.0, top_p=0.95, top_k=64, gated=True),
    CatalogModel("google/gemma-4-E2B-it", "Gemma 4 E2B", 10.2,
                 "Smaller Gemma 4; good prose on 12 GB GPUs.", "gemma4", temperature=1.0, top_p=0.95, top_k=64, gated=True),
    CatalogModel("fancyfeast/llama-joycaption-beta-one-hf-llava", "JoyCaption Beta One", 17.0,
                 "Captioner built for diffusion training data; strong on uncensored and artistic content.",
                 "llava", temperature=0.6, top_p=0.9, top_k=0),
)


@dataclass(frozen=True)
class LocalModel:
    path: Path
    name: str
    size_bytes: int
    model_type: str

    @property
    def approx_gb(self) -> float:
        return self.size_bytes / 2**30


def stability_matrix_llm_dirs() -> list[Path]:
    """Best-effort guesses for a Stability Matrix shared ``Models/LLM`` folder."""
    candidates = [
        paths.APP_ROOT.parent / "Stability Matrix" / "Data" / "Models" / "LLM",
        paths.APP_ROOT.parent / "StabilityMatrix" / "Data" / "Models" / "LLM",
    ]
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(Path(appdata) / "StabilityMatrix" / "Models" / "LLM")
    return [c for c in candidates if c.is_dir()]


def model_search_dirs(extra: list[str] | None = None) -> list[Path]:
    dirs = [paths.DEFAULT_MODELS_DIR, *[Path(d) for d in (extra or []) if d], *stability_matrix_llm_dirs()]
    seen, out = set(), []
    for d in dirs:
        key = str(d.resolve()).lower() if d.exists() else str(d).lower()
        if key not in seen and d.is_dir():
            seen.add(key)
            out.append(d)
    return out


_SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<idx>\d+)-of-(?P<n>\d+)\.(?P<ext>safetensors|bin)$")


def _nonempty(f: Path) -> bool:
    try:
        return f.is_file() and f.stat().st_size > 0
    except OSError:
        return False


def weights_complete(d: Path) -> bool:
    """True only if every weight file the folder promises is present and non-empty.

    ``snapshot_download`` writes finished files one by one, so an interrupted download can leave
    config + processor + the first shard behind. The shard index (or the ``-0000N-of-0000M`` file names)
    says how many shards there should be; a leftover ``.cache/huggingface/download/*.incomplete`` marks an
    unfinished file too - but a weight file only appears once it is whole, so the index and the shard names
    decide; a stale ``.incomplete`` of some other file (a README, a GGUF) never hides a complete model.
    """
    try:
        for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            index = d / index_name
            if index.is_file():
                try:
                    shards = set(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values())
                except (OSError, ValueError, AttributeError):
                    return False
                return bool(shards) and all(_nonempty(d / s) for s in shards)
        files = [f for pat in ("*.safetensors", "*.bin") for f in d.glob(pat)]
        if not files or not all(_nonempty(f) for f in files):
            return False
        counts: dict[tuple, set[int]] = {}
        totals: dict[tuple, int] = {}
        for f in files:
            m = _SHARD_RE.match(f.name)
            if m:
                key = (m["stem"], m["ext"])
                counts.setdefault(key, set()).add(int(m["idx"]))
                totals[key] = int(m["n"])
        return all(len(counts[k]) >= totals[k] for k in counts)
    except OSError:
        return False


def _vlm_config(d: Path) -> str | None:
    """model_type if ``d`` has config + processor config + some weight file (complete or not)."""
    cfg = d / "config.json"
    if not cfg.is_file():
        return None
    has_processor = any((d / f).is_file() for f in ("preprocessor_config.json", "processor_config.json"))
    if not has_processor:
        return None
    if not any(d.glob("*.safetensors")) and not any(d.glob("*.bin")):
        return None
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
        return str(data.get("model_type") or "unknown")
    except (OSError, ValueError):
        return "unknown"


def _is_vlm_folder(d: Path) -> str | None:
    """model_type of a usable (complete) VLM folder, else None."""
    mtype = _vlm_config(d)
    return mtype if mtype and weights_complete(d) else None


def is_partial_folder(d: Path) -> bool:
    """A model folder whose download was interrupted (resume it with the Download button)."""
    return _vlm_config(d) is not None and not weights_complete(d)


def _folder_size(d: Path) -> int:
    total = 0
    for pat in ("*.safetensors", "*.bin"):
        for f in d.glob(pat):
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def discover_local_models(extra_dirs: list[str] | None = None, max_depth: int = 3) -> list[LocalModel]:
    """Find HF-format vision models (config + processor config + weights) in the search dirs."""
    found: list[LocalModel] = []
    seen: set[str] = set()

    def walk(d: Path, depth: int) -> None:
        try:
            mtype = _is_vlm_folder(d)
        except OSError:
            return
        if mtype:
            key = str(d.resolve()).lower()
            if key not in seen:
                seen.add(key)
                found.append(LocalModel(d, d.name, _folder_size(d), mtype))
            return
        if depth >= max_depth:
            return
        try:
            subdirs = sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith("."))
        except OSError:
            return
        for sub in subdirs:
            walk(sub, depth + 1)

    for root in model_search_dirs(extra_dirs):
        walk(root, 0)
    return found


def hf_fully_cached(repo_id: str) -> bool:
    """True only if the HF cache holds the weights, not just config.json.

    A partial cache (e.g. only metadata fetched) must not count as downloaded,
    or loading would fail offline with a confusing error.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    try:
        index = try_to_load_from_cache(repo_id, "model.safetensors.index.json")
        if isinstance(index, str):
            shards = set(json.loads(Path(index).read_text(encoding="utf-8")).get("weight_map", {}).values())
            return bool(shards) and all(isinstance(try_to_load_from_cache(repo_id, s), str) for s in shards)
        return isinstance(try_to_load_from_cache(repo_id, "model.safetensors"), str)
    except Exception:
        return False


def _catalog_candidates(repo_id: str) -> tuple[Path, Path]:
    return paths.DEFAULT_MODELS_DIR / repo_id.split("/")[-1], paths.DEFAULT_MODELS_DIR / repo_id


def catalog_local_path(repo_id: str) -> Path | None:
    """Where a catalog model lives if it was downloaded (completely) into <app>/models/."""
    for candidate in _catalog_candidates(repo_id):
        if _is_vlm_folder(candidate):
            return candidate
    return None


def catalog_partial_path(repo_id: str) -> Path | None:
    """The folder of an interrupted catalog download, if any (Download resumes into it)."""
    for candidate in _catalog_candidates(repo_id):
        if is_partial_folder(candidate):
            return candidate
    return None


def remote_size_bytes(repo_id: str, revision: str = "") -> int | None:
    """Total size of the weight + config files that a download would fetch."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id, revision=revision or None, files_metadata=True)
        return sum((s.size or 0) for s in (info.siblings or []) if _wanted(s.rfilename))
    except Exception as e:
        log.info("Could not fetch size for %s: %s", repo_id, e)
        return None


def _wanted(filename: str) -> bool:
    low = filename.lower()
    if low.endswith((".gguf", ".onnx", ".msgpack", ".h5", ".ot", ".pt")):
        return False
    if low.endswith(".bin") and "pytorch_model" in low:
        return False  # prefer safetensors
    return "/" not in filename or filename.startswith(("tokenizer", "processor"))


def download_patterns(remote_code: bool = False) -> list[str]:
    """Files a catalog download fetches. ``*.py`` only for a model that needs custom code."""
    patterns = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.tiktoken"]
    return patterns + ["*.py"] if remote_code else patterns


def download_model(repo_id: str, target: Path, progress=None, revision: str = "", remote_code: bool = False) -> Path:
    from huggingface_hub import snapshot_download
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(target), revision=revision or None,
                      allow_patterns=download_patterns(remote_code))
    return target


def describe_download_error(err: BaseException, model: CatalogModel | None = None) -> str:
    """A plain-language reason for a failed download; gated repos get the model page link."""
    text = str(err)
    gated = type(err).__name__ in ("GatedRepoError",) or re.search(r"\b401\b", text) is not None or "gated" in text.lower()
    if gated or (model is not None and model.gated and "403" in text):
        repo = model.repo_id if model else ""
        link = f"https://huggingface.co/{repo}" if repo else "the model page on huggingface.co"
        return (f"This model is gated: it needs a Hugging Face account that has accepted its licence. "
                f"Open {link}, accept the licence, then log in with 'hf auth login' in the TagScribeR "
                "environment (or set the HF_TOKEN environment variable) and download again.")
    return text
