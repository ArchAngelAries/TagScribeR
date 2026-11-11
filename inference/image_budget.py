"""How many image tokens a picture costs, per vision-model family, and what size is worth sending.

The time a vision-language model spends on a picture grows with its image tokens: every token passes through
the vision encoder once and is then read by the language model like a word of the prompt. The families differ
in how pixels become tokens, so "send it smaller" pays off very differently:

* **grid** models (Qwen-VL and its successors, Pixtral): one token per patch of the picture at its own
  resolution. Tokens grow with the pixel count, so a pixel budget cuts time almost proportionally.
* **fixed** models (Gemma 3, LLaVA 1.5, JoyCaption): the picture is squeezed to one fixed size, whatever is
  sent. Pixels above that size are thrown away by the server: sending them only costs upload and decode.
* **tiles** models (LLaVA-NeXT, InternVL, Idefics / SmolVLM, MiniCPM-V): a fixed-size overview plus fixed-size
  tiles; the number of tiles follows the picture's size up to a limit.

The numbers are estimates from the public model cards and processor settings, for planning and for the
benchmark's arithmetic. A server runtime may clamp differently (llama.cpp has its own minimum and maximum
image tokens per model): measure with ``tools/bench_caption.py`` before trusting a figure. No Qt, no torch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VisionFamily:
    key: str
    label: str
    kind: str                   # grid | fixed | tiles
    patch: int = 0              # grid: pixels per token side; tiles / fixed: the side of one tile or of the input
    tokens: int = 0             # fixed: tokens per picture; tiles: tokens per tile
    max_tokens: int = 0         # grid / tiles: the family's own ceiling (0 = none known)
    max_side: int = 0           # grid: the model shrinks a longer picture to this side first (0 = no such limit)
    suggested_pixels: int = 0   # a budget that keeps normal pictures readable for this family (0 = no advice)
    note: str = ""


MP = 1_000_000
FAMILIES: dict[str, VisionFamily] = {f.key: f for f in (
    VisionFamily("qwen3", "Qwen3-VL / Qwen3.5 and later", "grid", patch=32, max_tokens=16384,
                 suggested_pixels=1 * MP, note="One token per 32 x 32 pixels."),
    VisionFamily("qwen2", "Qwen2-VL / Qwen2.5-VL", "grid", patch=28, max_tokens=16384,
                 suggested_pixels=1 * MP, note="One token per 28 x 28 pixels."),
    VisionFamily("pixtral", "Pixtral / Mistral Small 3 vision", "grid", patch=16, max_tokens=4096, max_side=1024,
                 suggested_pixels=590_000, note="One token per 16 x 16 pixels, at most 1024 x 1024 pixels."),
    VisionFamily("gemma3", "Gemma 3 / Gemma 3n", "fixed", patch=896, tokens=256, suggested_pixels=896 * 896,
                 note="Every picture becomes 896 x 896 pixels and 256 tokens."),
    VisionFamily("llava15", "LLaVA 1.5 style (CLIP 336)", "fixed", patch=336, tokens=576,
                 suggested_pixels=672 * 672, note="Every picture becomes 336 x 336 pixels and 576 tokens."),
    VisionFamily("joycaption", "JoyCaption (SigLIP 384)", "fixed", patch=384, tokens=729,
                 suggested_pixels=768 * 768, note="Every picture becomes 384 x 384 pixels and 729 tokens."),
    VisionFamily("llava_next", "LLaVA-NeXT (1.6)", "tiles", patch=336, tokens=576, max_tokens=2880,
                 suggested_pixels=672 * 672, note="An overview plus up to four 336-pixel tiles."),
    VisionFamily("internvl", "InternVL 2 / 3", "tiles", patch=448, tokens=256, max_tokens=3328,
                 suggested_pixels=1 * MP, note="An overview plus up to twelve 448-pixel tiles."),
    VisionFamily("smolvlm", "Idefics3 / SmolVLM", "tiles", patch=384, tokens=81, max_tokens=1377,
                 suggested_pixels=1 * MP, note="An overview plus 384-pixel tiles (512 and 64 tokens for SmolVLM2)."),
    VisionFamily("minicpm", "MiniCPM-V 2.6 / 4", "tiles", patch=448, tokens=64, max_tokens=640,
                 suggested_pixels=1_400_000, note="An overview plus up to nine 448-pixel slices."),
    VisionFamily("gemma4", "Gemma 4", "grid", patch=48, max_tokens=1120, suggested_pixels=1 * MP,
                 note="Variable resolution with a token budget (70 to 1120); the patch size here is a planning figure."),
)}

# Substrings of a model id (lower case), most specific first. A server model name is free text, so this is a
# guess: an unknown name gets no advice, never a wrong budget.
_NAME_HINTS: tuple[tuple[str, str], ...] = (
    ("joycaption", "joycaption"), ("joy-caption", "joycaption"),
    ("qwen2.5-vl", "qwen2"), ("qwen2.5vl", "qwen2"), ("qwen2-vl", "qwen2"), ("qwen2vl", "qwen2"),
    ("qwen3", "qwen3"), ("qwen-3", "qwen3"),
    ("gemma-4", "gemma4"), ("gemma4", "gemma4"),
    ("gemma-3", "gemma3"), ("gemma3", "gemma3"),
    ("pixtral", "pixtral"), ("mistral-small-3", "pixtral"),
    ("internvl", "internvl"),
    ("smolvlm", "smolvlm"), ("idefics3", "smolvlm"),
    ("minicpm", "minicpm"),
    ("llava-v1.6", "llava_next"), ("llava-1.6", "llava_next"), ("llava-next", "llava_next"),
    ("llava-onevision", "llava_next"),
    ("llava", "llava15"), ("bakllava", "llava15"),
)

# Names for a "how much of the picture to send" choice. 0 = no pixel budget (only "Max image size" applies).
BUDGET_CHOICES: tuple[tuple[str, int], ...] = (
    ("Full size", 0),
    ("Detailed (1.6 MP)", 1_600_000),
    ("Balanced (1.0 MP)", 1_000_000),
    ("Fast (0.6 MP)", 600_000),
)


def guess_family(model_name: str) -> VisionFamily | None:
    low = (model_name or "").lower()
    for hint, key in _NAME_HINTS:
        if hint in low:
            return FAMILIES[key]
    return None


def image_tokens(width: int, height: int, family: VisionFamily | str) -> int:
    """Estimated image tokens for a picture sent at ``width`` x ``height``."""
    fam = FAMILIES[family] if isinstance(family, str) else family
    width, height = max(1, int(width)), max(1, int(height))
    if fam.kind == "fixed":
        return fam.tokens
    if fam.kind == "grid":
        if fam.max_side and max(width, height) > fam.max_side:
            scale = fam.max_side / max(width, height)
            width, height = width * scale, height * scale
        n = max(1, round(width / fam.patch)) * max(1, round(height / fam.patch))
        return min(n, fam.max_tokens) if fam.max_tokens else n
    tiles = math.ceil(width / fam.patch) * math.ceil(height / fam.patch)
    n = fam.tokens * (1 + (tiles if tiles > 1 else 0))
    return min(n, fam.max_tokens) if fam.max_tokens else n


def tokens_at_budget(pixels: int, family: VisionFamily | str, aspect: float = 4 / 3) -> int:
    """Image tokens of a picture with ``pixels`` pixels and the given width / height ratio."""
    height = math.sqrt(max(1, pixels) / aspect)
    return image_tokens(round(height * aspect), round(height), family)


def context_needed(image_tokens_: int, prompt_tokens: int, max_new_tokens: int, parallel: int = 1,
                   shared: bool = True) -> int:
    """Context length a server needs so that ``parallel`` such requests fit.

    llama.cpp (and LM Studio on top of it) divides one context between its parallel slots, or lets them share
    it: either way the total has to hold every request in flight. ``shared=False`` is one request.
    """
    one = image_tokens_ + prompt_tokens + max_new_tokens + 64       # 64: chat template and image markers
    return one * (max(1, parallel) if shared else 1)


def describe_budget(model_name: str, max_side: int, max_pixels: int, sample=(1536, 1152)) -> str:
    """One line for a status label: what a typical picture costs with these limits for this model."""
    from inference.image_prep import fit_size
    fam = guess_family(model_name)
    w, h = fit_size(sample[0] * 4, sample[1] * 4, max_side, max_pixels)     # a picture larger than any limit
    size = f"up to {w} x {h} pixels ({w * h / MP:.1f} MP)"
    if fam is None:
        return f"Pictures are sent at {size}."
    n = image_tokens(w, h, fam)
    if fam.kind == "fixed":
        return (f"Pictures are sent at {size}. {fam.label} reads every picture at {fam.patch} x {fam.patch} "
                f"({n} image tokens), so a smaller size loses nothing.")
    return f"Pictures are sent at {size}: about {n} image tokens each for {fam.label}."
