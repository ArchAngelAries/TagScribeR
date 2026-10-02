"""Hardware detection and device / precision selection.

Design notes
------------
* ROCm builds of PyTorch expose AMD GPUs through the ``torch.cuda`` API, so
  "cuda" device strings are correct for both vendors. What differs is feature
  support, so we detect the backend explicitly (``torch.version.hip``) instead
  of assuming CUDA.
* Nothing here imports torch at module import time. The UI can start, and the
  API / ONNX paths can run, on machines without a working torch install.
* Capabilities (bf16, SDPA kernels, bitsandbytes) are probed, never assumed.

The ROCm environment defaults mirror what the Fizgig trainer ships for
Windows ROCm 7.x wheels (MIOpen fast find mode, AOTriton SDPA kernels on RDNA3+,
math-only SDPA on RDNA1/2).
"""
from __future__ import annotations

import gc
import importlib.metadata
import importlib.util
import logging
import os
import platform
from dataclasses import dataclass
from functools import lru_cache

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    id: str               # torch device string: "cuda:0", "cpu", "mps", "xpu:0"
    backend: str          # "rocm" | "cuda" | "mps" | "xpu" | "cpu"
    name: str
    total_memory: int = 0  # bytes (0 = unknown / system RAM)
    arch: str = ""        # e.g. "gfx1100" or "sm_89"
    bf16: bool = False

    @property
    def is_gpu(self) -> bool:
        return self.backend != "cpu"

    @property
    def label(self) -> str:
        if self.backend == "cpu":
            return "CPU"
        mem = f", {self.total_memory / 2**30:.0f} GB" if self.total_memory else ""
        kind = {"rocm": "ROCm", "cuda": "CUDA", "mps": "Metal", "xpu": "XPU"}.get(self.backend, self.backend)
        return f"{self.name} ({kind}{mem})"


CPU = DeviceInfo(id="cpu", backend="cpu", name=platform.processor() or "CPU", bf16=False)


def torch_installed() -> bool:
    return importlib.util.find_spec("torch") is not None


def _installed_torch_version() -> str:
    try:
        return importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return ""


def apply_runtime_env() -> None:
    """Set backend environment defaults. Call before torch is first imported.

    Only fills variables the user hasn't set, so launch scripts and power users
    keep full control.
    """
    if "+rocm" in _installed_torch_version().lower():
        os.environ.setdefault("MIOPEN_FIND_MODE", "2")  # FAST: skip exhaustive kernel search on first use
        os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")  # flash/mem-efficient SDPA on RDNA3+
        os.environ.pop("ROCBLAS_USE_HIPBLASLT_BATCHED", None)
        _expose_pip_rocm_sdk()
    # Reduce fragmentation when models are loaded/unloaded repeatedly.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.8")
    # Don't phone home to the Hub on every from_pretrained when files are cached.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _expose_pip_rocm_sdk() -> None:
    """Point ROCM_PATH/HIP_PATH at the pip-installed ROCm SDK (TheRock wheels).

    bitsandbytes and a few other libraries locate HIP through these variables
    or by running ``hipinfo``; the wheel layout isn't on PATH by default.
    """
    spec = importlib.util.find_spec("_rocm_sdk_core")
    if spec is None or not spec.submodule_search_locations:
        return
    from pathlib import Path
    core = Path(list(spec.submodule_search_locations)[0])
    os.environ.setdefault("ROCM_PATH", str(core))
    os.environ.setdefault("HIP_PATH", os.environ["ROCM_PATH"])
    extra = [core / "bin"]
    devel = importlib.util.find_spec("_rocm_sdk_devel")
    if devel is not None and devel.submodule_search_locations:
        extra.append(Path(list(devel.submodule_search_locations)[0]) / "bin")
    current = os.environ.get("PATH", "")
    prepend = [str(p) for p in extra if p.is_dir() and str(p) not in current]
    if prepend:
        os.environ["PATH"] = os.pathsep.join(prepend + [current])


def _torch():
    import torch  # noqa: PLC0415 - deliberate lazy import
    return torch


def is_rocm() -> bool:
    if not torch_installed():
        return False
    try:
        torch = _torch()
    except Exception:
        return False
    return bool(getattr(torch.version, "hip", None) or getattr(torch.version, "rocm", None)
                or "+rocm" in torch.__version__.lower())


@lru_cache(maxsize=1)
def detect_devices() -> tuple[DeviceInfo, ...]:
    """All usable compute devices, best first; CPU is always last."""
    devices: list[DeviceInfo] = []
    if torch_installed():
        try:
            torch = _torch()
            if torch.cuda.is_available():
                backend = "rocm" if is_rocm() else "cuda"
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    if backend == "rocm":
                        arch = (getattr(props, "gcnArchName", "") or "").split(":")[0]
                    else:
                        arch = f"sm_{props.major}{props.minor}"
                    try:
                        with torch.cuda.device(i):
                            bf16 = bool(torch.cuda.is_bf16_supported())
                    except Exception:
                        bf16 = False
                    devices.append(DeviceInfo(id=f"cuda:{i}", backend=backend, name=props.name,
                                              total_memory=int(props.total_memory), arch=arch, bf16=bf16))
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                devices.append(DeviceInfo(id="mps", backend="mps", name="Apple GPU", bf16=False))
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                for i in range(torch.xpu.device_count()):
                    devices.append(DeviceInfo(id=f"xpu:{i}", backend="xpu",
                                              name=torch.xpu.get_device_name(i), bf16=True))
        except Exception as e:  # a broken driver must not take the app down
            log.warning("GPU detection failed, using CPU: %s", e, exc_info=True)
    devices.append(CPU)
    return tuple(devices)


def resolve_device(preference: str = "auto") -> DeviceInfo:
    devices = detect_devices()
    if preference and preference != "auto":
        for d in devices:
            if d.id == preference or (preference == "cuda" and d.id == "cuda:0"):
                return d
        log.warning("Requested device %r not available; falling back to auto.", preference)
    return devices[0]


def resolve_dtype(device: DeviceInfo, preference: str = "auto"):
    """Pick a torch dtype. 'auto' = bf16 where supported, else fp16 on GPU, fp32 on CPU.

    bf16 is preferred over fp16 because several modern VLMs (Qwen2.5/3-VL in
    particular) overflow in fp16 and produce garbage or NaNs.
    """
    torch = _torch()
    table = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if preference in table:
        if preference == "bfloat16" and device.is_gpu and not device.bf16:
            log.warning("%s does not support bfloat16; using float16.", device.name)
            return torch.float16
        if device.backend == "cpu" and preference == "float16":
            log.warning("float16 is very slow / unsupported on CPU; using float32.")
            return torch.float32
        return table[preference]
    if device.backend == "cpu":
        return torch.float32
    return torch.bfloat16 if device.bf16 else torch.float16


def configure_torch_backends(device: DeviceInfo) -> None:
    """Per-device runtime tweaks applied once a device has been chosen."""
    if device.backend not in ("rocm", "cuda"):
        return
    torch = _torch()
    if device.backend == "rocm" and device.arch.startswith(("gfx101", "gfx103")):
        # RDNA1/2 lack the AOTriton flash / mem-efficient kernels; force math SDPA.
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception as e:
            log.debug("Could not adjust SDPA backends: %s", e)
    if device.backend == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass


def memory_info(device: DeviceInfo) -> tuple[int, int] | None:
    """(free, total) bytes for a GPU, or None if unknown."""
    if device.backend not in ("rocm", "cuda"):
        return None
    try:
        torch = _torch()
        idx = int(device.id.split(":")[1]) if ":" in device.id else 0
        free, total = torch.cuda.mem_get_info(idx)
        return int(free), int(total)
    except Exception:
        return None


def free_memory() -> None:
    """Release cached allocator memory after unloading a model."""
    import sys
    gc.collect()
    try:
        torch = sys.modules.get("torch")
        if torch is None:
            return  # torch never loaded: nothing to free
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if not is_rocm():
                torch.cuda.ipc_collect()
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception as e:
        log.debug("free_memory: %s", e)


def optional_feature(module: str) -> bool:
    """True if an optional acceleration package is importable (no import side effects)."""
    return importlib.util.find_spec(module) is not None


def describe_environment() -> str:
    lines = [f"Python {platform.python_version()} on {platform.system()} {platform.release()}"]
    tv = _installed_torch_version()
    if not tv:
        lines.append("PyTorch: not installed (local models unavailable; API and tagger still work)")
    else:
        lines.append(f"PyTorch: {tv}")
        for d in detect_devices():
            extra = f", arch {d.arch}" if d.arch else ""
            bf = ", bf16" if d.bf16 else ""
            lines.append(f"  - {d.label}{extra}{bf}")
    for mod, label in (("transformers", "Transformers"), ("onnxruntime", "ONNX Runtime"),
                       ("bitsandbytes", "bitsandbytes"), ("flash_attn", "FlashAttention")):
        try:
            v = importlib.metadata.version(mod.replace("_", "-"))
            lines.append(f"{label}: {v}")
        except importlib.metadata.PackageNotFoundError:
            lines.append(f"{label}: not installed")
    try:
        import onnxruntime as ort
        lines.append("ONNX providers: " + ", ".join(ort.get_available_providers()))
    except Exception:
        pass
    return "\n".join(lines)
