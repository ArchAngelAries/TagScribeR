# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/utils/vram_monitor.py and the status-bar
# readers in lora_trainer_gui.py (_visible_gpu_index, _read_vram, _status_reader_loop).
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: the ROCm check is core.hardware.is_rocm (no FIZGIG_GPU_BACKEND variable); the GUI's NVIDIA
# readers (pynvml, then nvidia-smi), the psutil RAM read and the one-second reader thread live here as read_vram,
# read_ram and MemoryReader so the widget (tabs/memory_bar.py) only draws; the card's total VRAM also comes from the
# app's cached device detection when torch has not brought the GPU up. The AMD readers are otherwise unchanged.
"""VRAM and RAM reads for the Train tab's memory bar.

NVIDIA: pynvml, then a one-shot nvidia-smi query. The AMD / ROCm readers below are tried only when those return
nothing. Nothing here imports torch or initialises the GPU: the bar is cosmetic, and a second process querying a
preview ROCm driver beside a training run is what the readers are written to avoid.

Windows ROCm: typeperf ``\\GPU Local Adapter Memory(*)\\Local Usage``.
Linux ROCm: ``amd-smi`` CLI (ROCm Core / amdrocm-amdsmi), legacy ``rocm-smi``,
else torch mem_get_info (allocator-only on HIP).

The PyPI ``amdsmi`` package is stale - AMD SMI ships with ROCm Core SDK / the
``amdrocm-amdsmi`` system package. See:
    https://rocm.docs.amd.com/projects/amdsmi/en/latest/install/install.html
"""

from __future__ import annotations

import atexit
import glob
import json
import logging
import os
import re
import subprocess
import sys
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Windows performance counter — bytes of local VRAM in use per adapter.
_TYPEPERF_COUNTER = r"\GPU Local Adapter Memory(*)\Local Usage"
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _parse_typeperf_line(line: str) -> Optional[int]:
    """Return max adapter Local Usage (bytes) from one typeperf CSV row."""
    line = line.strip()
    if not line or line.startswith('"('):
        return None
    max_usage = 0.0
    for value in line.strip().split('","')[1:]:
        try:
            usage = float(value.strip('"').strip())
            if usage > max_usage:
                max_usage = usage
        except ValueError:
            pass
    return int(max_usage) if max_usage > 0 else None


def _torch_ready():
    """The torch module, only if something else has ALREADY imported it and brought the
    GPU up — never imports torch, never calls is_available(). The status bar is cosmetic:
    on the Windows ROCm preview stack, importing torch / initialising HIP in the GUI process
    beside a training subprocess, then querying it every second, is exactly the kind of
    load a preview driver falls over on (#145). Before a job has the GPU up this returns
    None and the bar reads "unavailable"; once a tool in this process has initialised it,
    the reads are free."""
    t = sys.modules.get("torch")
    if t is None:
        return None
    try:
        if t.cuda.is_initialized():
            return t
    except Exception:
        pass
    return None


def _total_vram_from_torch() -> Optional[int]:
    try:
        torch = _torch_ready()
        if torch is not None:
            return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        pass
    # TagScribeR: the app's own startup detection already knows the card's size. Read it only if it has run (the
    # cache is filled) - this must never be what starts GPU detection.
    try:
        from core import hardware
        if hardware.detect_devices.cache_info().currsize:
            total = int(hardware.detect_devices()[0].total_memory or 0)
            return total or None
    except Exception:
        pass
    return None


def _to_bytes(num: float, unit: Optional[str]) -> int:
    u = (unit or "MB").upper()
    scale = {"B": 1, "KB": 1024, "KIB": 1024, "MB": 1024 ** 2, "MIB": 1024 ** 2,
             "GB": 1024 ** 3, "GIB": 1024 ** 3}.get(u, 1024 ** 2)
    return int(num * scale)


def _parse_vram_mb(value) -> Optional[int]:
    """Parse amd-smi VRAM fields — API reports vram_used/vram_total in megabytes."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return _to_bytes(float(value), "MB")
    if isinstance(value, dict):
        if "value" in value:
            unit = value.get("unit") or value.get("Unit") or "MB"
            try:
                return _to_bytes(float(value["value"]), str(unit))
            except (TypeError, ValueError, KeyError):
                pass
        return None
    if isinstance(value, str):
        m = re.match(r"^([\d.]+)\s*(B|KB|MB|GB|MiB|GiB|KIB|MIB|GIB)?$", value.strip(), re.I)
        if m:
            return _to_bytes(float(m.group(1)), m.group(2) or "MB")
    return None


def _coerce_bytes(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, dict):
        if "value" in value:
            unit = value.get("unit") or value.get("Unit") or "B"
            try:
                return _to_bytes(float(value["value"]), str(unit))
            except (TypeError, ValueError, KeyError):
                pass
        for key in (
            "vram_used", "vram_total", "used", "total",
            "VRAM Used Memory (B)", "VRAM Total Memory (B)",
            "VRAM Total Used Memory (B)", "VRAM_USED", "VRAM_TOTAL",
            "SIZE", "size",
        ):
            if key in value:
                hit = _coerce_bytes(value[key])
                if hit is not None:
                    return hit
        for v in value.values():
            hit = _coerce_bytes(v)
            if hit is not None:
                return hit
        return None
    if isinstance(value, str):
        m = re.match(r"^([\d.]+)\s*(B|KB|MB|GB|MiB|GiB|KIB|MIB|GIB)?$", value.strip(), re.I)
        if m:
            num = float(m.group(1))
            unit = (m.group(2) or "B").upper()
            scale = {"B": 1, "KB": 1024, "KIB": 1024, "MB": 1024 ** 2, "MIB": 1024 ** 2,
                     "GB": 1024 ** 3, "GIB": 1024 ** 3}.get(unit, 1)
            return int(num * scale)
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _vram_pair_from_monitor_node(node: dict) -> Optional[tuple[int, int]]:
    """Extract used/total from one amd-smi monitor GPU object (VRAM fields are MB)."""
    if not isinstance(node, dict):
        return None

    def _pair(used_val, total_val, parser=_parse_vram_mb) -> Optional[tuple[int, int]]:
        used = parser(used_val)
        total = parser(total_val)
        if used is not None and total is not None and total > 0:
            return int(used), int(total)
        return None

    for used_key, total_key, parser in (
        ("vram_used", "vram_total", _parse_vram_mb),
        ("VRAM_USED", "VRAM_TOTAL", _parse_vram_mb),
        ("VRAM Used Memory (B)", "VRAM Total Memory (B)", _coerce_bytes),
    ):
        if used_key in node and total_key in node:
            hit = _pair(node[used_key], node[total_key], parser)
            if hit:
                return hit

    for block_key in ("vram_usage", "memory_usage", "vram", "memory"):
        block = node.get(block_key)
        if not isinstance(block, dict):
            continue
        for used_key, total_key, parser in (
            ("vram_used", "vram_total", _parse_vram_mb),
            ("VRAM_USED", "VRAM_TOTAL", _parse_vram_mb),
        ):
            if used_key in block and total_key in block:
                hit = _pair(block[used_key], block[total_key], parser)
                if hit:
                    return hit
        if isinstance(block.get("size"), dict) and "vram_used" in block:
            hit = _pair(block.get("vram_used"), block["size"], _parse_vram_mb)
            if hit:
                return hit
    return None


def _gpu_index(node: dict) -> Optional[int]:
    gpu = node.get("gpu")
    if isinstance(gpu, int):
        return gpu
    if isinstance(gpu, str) and gpu.isdigit():
        return int(gpu)
    return None


def _monitor_vram_used_by_gpu(data) -> dict[int, int]:
    """gpu index -> used bytes from amd-smi monitor JSON."""
    out: dict[int, int] = {}

    def _scan(node):
        if isinstance(node, dict):
            idx = _gpu_index(node)
            if idx is not None and "vram_used" in node:
                used = _parse_vram_mb(node["vram_used"])
                if used is not None:
                    out[idx] = int(used)
            for v in node.values():
                _scan(v)
        elif isinstance(node, list):
            for item in node:
                _scan(item)

    _scan(data)
    return out


def _static_vram_total_by_gpu(data) -> dict[int, int]:
    """gpu index -> total bytes from amd-smi static --vram JSON."""
    out: dict[int, int] = {}

    def _scan(node):
        if isinstance(node, dict):
            idx = _gpu_index(node)
            vram = node.get("vram")
            if idx is not None and isinstance(vram, dict):
                total = _parse_vram_mb(vram.get("size"))
                if total is not None:
                    out[idx] = int(total)
            for v in node.values():
                _scan(v)
        elif isinstance(node, list):
            for item in node:
                _scan(item)

    _scan(data)
    return out


def _hip_device_total_bytes() -> Optional[int]:
    try:
        torch = _torch_ready()
        if torch is not None:
            return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        pass
    return None


def _pick_amd_gpu_index(totals_by_gpu: dict[int, int]) -> Optional[int]:
    """Pick the GPU index training runs on — matches HIP device 0 VRAM when possible."""
    if not totals_by_gpu:
        return None
    hip_total = _hip_device_total_bytes()
    if hip_total is not None:
        return min(totals_by_gpu, key=lambda g: abs(totals_by_gpu[g] - hip_total))
    return max(totals_by_gpu, key=lambda g: totals_by_gpu[g])


def _merge_amd_smi_vram(static_data, monitor_data) -> Optional[tuple[int, int]]:
    """Use static --vram for capacity (accurate) + monitor for used (live)."""
    totals = _static_vram_total_by_gpu(static_data) if static_data else {}
    used_map = _monitor_vram_used_by_gpu(monitor_data) if monitor_data else {}
    if totals:
        gpu = _pick_amd_gpu_index(totals)
        if gpu is not None:
            total = totals[gpu]
            used = used_map.get(gpu, 0)
            return int(used), int(total)
    # No static — fall back to monitor pairs, still prefer HIP-matched total.
    pairs: dict[int, tuple[int, int]] = {}
    if monitor_data:
        def _scan(node):
            if isinstance(node, dict):
                idx = _gpu_index(node)
                if idx is not None:
                    hit = _vram_pair_from_monitor_node(node)
                    if hit:
                        pairs[idx] = hit
                for v in node.values():
                    _scan(v)
            elif isinstance(node, list):
                for item in node:
                    _scan(item)
        _scan(monitor_data)
    if pairs:
        gpu = _pick_amd_gpu_index({g: t for g, (_u, t) in pairs.items()})
        if gpu is not None:
            return pairs[gpu]
    return _best_vram_pair(list(pairs.values()))


def _collect_amd_smi_monitor_pairs(data) -> list[tuple[int, int]]:
    """Collect VRAM pairs from amd-smi monitor JSON (not static-only size blobs)."""
    hits: list[tuple[int, int]] = []

    def _walk(node):
        if isinstance(node, dict):
            hit = _vram_pair_from_monitor_node(node)
            if hit:
                hits.append(hit)
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(data)
    return hits


def _best_vram_pair(pairs: list[tuple[int, int]], min_total: int = 512 * 1024 * 1024) -> Optional[tuple[int, int]]:
    """Prefer the GPU with the largest VRAM total (skip iGPU / empty entries)."""
    valid = [(u, t) for u, t in pairs if t >= min_total]
    if not valid:
        valid = [(u, t) for u, t in pairs if t > 0]
    if not valid:
        return None
    return max(valid, key=lambda x: x[1])


def _parse_vram_json(data) -> Optional[tuple[int, int]]:
    pairs = _collect_amd_smi_monitor_pairs(data)
    hit = _best_vram_pair(pairs)
    if hit:
        return hit
    # Monitor JSON missing used — do not invent used=0 from unrelated fields.
    return None


def _linux_rocm_cli_env() -> dict[str, str]:
    """PATH/LD_LIBRARY_PATH so amd-smi from ROCm Core SDK is discoverable."""
    env = os.environ.copy()
    path_extra: list[str] = []
    for pattern in ("/opt/rocm/core-*/bin", "/opt/rocm/bin"):
        path_extra.extend(sorted(d for d in glob.glob(pattern) if os.path.isdir(d)))
    if path_extra:
        env["PATH"] = os.pathsep.join(path_extra + [env.get("PATH", "")])
    lib_extra = [p for p in ("/opt/rocm/lib", "/opt/rocm/lib64") if os.path.isdir(p)]
    if lib_extra:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(lib_extra + [env.get("LD_LIBRARY_PATH", "")])
    return env


def _run_linux_cli(args: list[str], timeout: float = 4) -> Optional[str]:
    try:
        out = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout,
            env=_linux_rocm_cli_env(),
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout
    except Exception:
        pass
    return None




_amd_smi_bin: Optional[str] = None


def _find_amd_smi() -> Optional[str]:
    global _amd_smi_bin
    if _amd_smi_bin is not None:
        return _amd_smi_bin or None
    env = _linux_rocm_cli_env()
    for name in ("amd-smi", "rocm-smi"):
        path = env.get("PATH", os.environ.get("PATH", ""))
        for directory in path.split(os.pathsep):
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                if name == "amd-smi":
                    _amd_smi_bin = candidate
                    return candidate
    for pattern in (
        "/opt/rocm/core-*/bin/amd-smi",
        "/opt/rocm/bin/amd-smi",
        "venv/bin/amd-smi",
        "*/venv/bin/amd-smi",
    ):
        for candidate in sorted(glob.glob(pattern)):
            if os.access(candidate, os.X_OK):
                _amd_smi_bin = candidate
                return candidate
    _amd_smi_bin = ""
    return None


def _parse_amd_smi_monitor_text(text: str) -> Optional[tuple[int, int]]:
    """Parse ``amd-smi monitor`` text — pick the GPU row with the largest VRAM total."""
    best: Optional[tuple[int, int]] = None
    for line in text.splitlines():
        if line.startswith("GPU") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        nums: list[tuple[float, Optional[str]]] = []
        i = 0
        while i < len(parts):
            if re.match(r"^[\d.]+$", parts[i]):
                unit = parts[i + 1] if i + 1 < len(parts) else None
                if unit and unit.upper() in {"MB", "GB", "MIB", "GIB", "B", "KB", "KIB"}:
                    nums.append((float(parts[i]), unit))
                    i += 2
                    continue
                nums.append((float(parts[i]), "MB"))
            i += 1
        if len(nums) >= 2:
            used = _to_bytes(nums[-2][0], nums[-2][1])
            total = _to_bytes(nums[-1][0], nums[-1][1])
            if total > 0 and (best is None or total > best[1]):
                best = (used, total)
    return best


def _read_vram_amd_smi_cli() -> Optional[tuple[int, int]]:
    """Linux AMD: amd-smi from ROCm Core SDK / amdrocm-amdsmi (replaces legacy rocm-smi)."""
    if _find_amd_smi() is None:
        return None
    bin_path = _amd_smi_bin
    assert bin_path

    static_text = _run_linux_cli([bin_path, "static", "--vram", "--json"])
    static_data = None
    if static_text:
        try:
            static_data = json.loads(static_text)
        except json.JSONDecodeError:
            static_data = None

    for extra in (
        ["monitor", "-v", "--json"],
        ["monitor", "--vram-usage", "--json"],
        ["monitor", "--vram-usage", "-v", "--json"],
        ["monitor", "--json"],
    ):
        text = _run_linux_cli([bin_path, *extra])
        if not text:
            continue
        try:
            monitor_data = json.loads(text)
            hit = _merge_amd_smi_vram(static_data, monitor_data)
            if hit and hit[1] > 0:
                return hit
        except json.JSONDecodeError:
            pass
        hit = _parse_amd_smi_monitor_text(text)
        if hit and hit[1] > 0:
            return hit

    text = _run_linux_cli([bin_path, "monitor", "--vram-usage"])
    if text:
        hit = _parse_amd_smi_monitor_text(text)
        if hit and hit[1] > 0:
            return hit
    return None


def _parse_rocm_smi_text(text: str) -> Optional[tuple[int, int]]:
    """Parse ``rocm-smi --showmeminfo vram`` plain-text output."""
    used = total = None
    for line in text.splitlines():
        if "Used Memory (B)" in line or "Used Memory (MiB)" in line:
            m = re.search(r":\s*(\d+)", line)
            if m:
                val = int(m.group(1))
                if "MiB" in line:
                    val *= 1024 * 1024
                used = val
        if "Total Memory (B)" in line or "Total Memory (MiB)" in line:
            m = re.search(r":\s*(\d+)", line)
            if m:
                val = int(m.group(1))
                if "MiB" in line:
                    val *= 1024 * 1024
                total = val
    if used is not None and total is not None:
        return used, total
    return None


def _read_vram_rocm_smi() -> Optional[tuple[int, int]]:
    """Linux AMD: legacy rocm-smi CLI (superseded by amd-smi in ROCm 7+)."""
    try:
        text = _run_linux_cli(["rocm-smi", "--showmeminfo", "vram", "--json"])
        if text:
            hit = _parse_vram_json(json.loads(text))
            if hit:
                return hit
    except Exception:
        pass
    try:
        text = _run_linux_cli(["rocm-smi", "--showmeminfo", "vram"])
        if text:
            hit = _parse_rocm_smi_text(text)
            if hit:
                return hit
    except Exception:
        pass
    return None




class _TypeperfVramReader:
    """Background typeperf process — one sample per second, latest value cached."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_used: Optional[int] = None
        self._total: Optional[int] = None
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._failed = False

    def _reader_loop(self) -> None:
        proc = None
        try:
            proc = subprocess.Popen(
                ["typeperf", _TYPEPERF_COUNTER, "-si", "1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                creationflags=_CREATE_NO_WINDOW,
            )
            with self._lock:
                self._proc = proc
            assert proc.stdout is not None
            next(proc.stdout, None)
            next(proc.stdout, None)
            for line in proc.stdout:
                used = _parse_typeperf_line(line)
                if used is not None:
                    with self._lock:
                        self._latest_used = used
        except Exception as exc:
            logger.debug("typeperf VRAM reader stopped: %s", exc)
            with self._lock:
                self._failed = True
        finally:
            with self._lock:
                self._proc = None
                self._latest_used = None      # a dead stream must not keep reporting its last row
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass

    def stop(self) -> None:
        """Kill the background typeperf (app close / interpreter exit)."""
        with self._lock:
            proc, self._proc = self._proc, None
            self._failed = True               # no restart after a stop
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass

    def _ensure_started(self) -> None:
        with self._lock:
            if self._thread is not None or self._failed:
                return
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()

    def read(self) -> Optional[tuple[int, int]]:
        """The latest streamed sample, or None. Never blocks: there used to be a synchronous
        one-shot typeperf here (10 s timeout) on every tick until the stream produced a row —
        on a machine whose counter never yields one, that was a process spawned every second
        for the life of the app (#145). Unavailable is a fine answer for a status bar."""
        if self._total is None:
            self._total = _total_vram_from_torch()
        self._ensure_started()
        with self._lock:
            used = self._latest_used
        if used is None or self._total is None:
            return None
        return used, self._total


def _is_rocm_backend() -> bool:
    try:
        from core.hardware import is_rocm
        return is_rocm()
    except Exception:
        return False


def _read_vram_torch_fallback() -> Optional[tuple[int, int]]:
    """Allocator-visible used/total via HIP (less accurate than typeperf/amd-smi). Only
    when this process already has the GPU up — see _torch_ready."""
    try:
        torch = _torch_ready()
        if torch is not None:
            free_b, total_b = torch.cuda.mem_get_info(0)
            return int(total_b - free_b), int(total_b)
    except Exception:
        pass
    return None


_typeperf_reader: Optional[_TypeperfVramReader] = None
_typeperf_lock = threading.Lock()


def _read_vram_windows_typeperf() -> Optional[tuple[int, int]]:
    global _typeperf_reader
    with _typeperf_lock:
        if _typeperf_reader is None:
            _typeperf_reader = _TypeperfVramReader()
        reader = _typeperf_reader
    return reader.read()


def shutdown() -> None:
    """Stop the background typeperf reader, if one was ever started. Registered with atexit;
    the GUI also calls it on window close so the child never outlives the app."""
    global _typeperf_reader
    with _typeperf_lock:
        reader, _typeperf_reader = _typeperf_reader, None
    if reader is not None:
        reader.stop()


atexit.register(shutdown)


def read_amd_gpu_vram() -> Optional[tuple[int, int]]:
    """AMD-only VRAM read for the GUI fallback / ROCm installer probes.

    Windows: typeperf (then torch). Linux: amd-smi, then rocm-smi, then torch.
    """
    if os.name == "nt":
        # typeperf's stream, or nothing. The HIP fallback that used to sit here ran
        # torch.cuda.mem_get_info every second whenever the counter gave nothing — on the
        # Windows ROCm preview stack that is a second process querying the driver beside the
        # training run, for a cosmetic bar (#145). Unavailable is the right answer.
        return _read_vram_windows_typeperf()

    hit = _read_vram_amd_smi_cli() or _read_vram_rocm_smi()
    if hit:
        return hit
    if _is_rocm_backend():
        return _read_vram_torch_fallback()
    return None


# ---- NVIDIA readers, RAM and the reader thread (Fizgig lora_trainer_gui.py) -------------------------------------

_nvml_handle = None
_nvidia_smi_missing = False


def visible_gpu_index() -> int:
    """The physical GPU index training will actually use.

    NVML and nvidia-smi index every card in the machine; CUDA_VISIBLE_DEVICES does not change that, it only changes
    what torch can see. Torch's cuda:0 is the FIRST entry in the list. A UUID entry (new format) is looked up through
    NVML."""
    raw = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
    if raw.isdigit():
        return int(raw)
    if raw:
        try:
            import pynvml
            pynvml.nvmlInit()
            for i in range(pynvml.nvmlDeviceGetCount()):
                uuid = pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(i))
                if (uuid.decode() if isinstance(uuid, bytes) else uuid) == raw:
                    return i
        except Exception:
            pass
    return 0


def read_vram() -> Optional[tuple[int, int]]:
    """(used_bytes, total_bytes) for the GPU training uses, or None. Prefers pynvml (fast); falls back to a one-shot
    nvidia-smi query. AMD ROCm paths are tried only when the NVIDIA readers return nothing."""
    global _nvml_handle, _nvidia_smi_missing
    try:
        import pynvml
        if _nvml_handle is None:
            pynvml.nvmlInit()
            _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(visible_gpu_index())
        m = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
        return int(m.used), int(m.total)
    except Exception:
        pass
    if not _nvidia_smi_missing:
        try:
            out = subprocess.run(
                ["nvidia-smi", "-i", str(visible_gpu_index()), "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=4, creationflags=_CREATE_NO_WINDOW)
            used, total = out.stdout.strip().splitlines()[0].split(",")
            return int(used) * 1024 * 1024, int(total) * 1024 * 1024
        except FileNotFoundError:
            _nvidia_smi_missing = True         # an AMD box: stop spawning it every second
        except Exception:
            pass
    try:
        return read_amd_gpu_vram()
    except Exception:
        return None


def read_ram() -> Optional[tuple[int, int]]:
    """(used_bytes, total_bytes) of system RAM, or None."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return int(vm.total - vm.available), int(vm.total)
    except Exception:
        return None


class MemoryReader:
    """One sample a second on a daemon thread; `latest` is (vram, ram), each (used, total) or None. The readers can
    block (nvidia-smi, amd-smi), so they never run on the UI thread."""

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self.latest: tuple = (None, None)
        self.samples = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.latest = (read_vram(), read_ram())
            self.samples += 1
            self._stop.wait(self.interval)

    def stop(self) -> None:
        """Stop sampling and the typeperf child (AMD Windows): neither may outlive the window."""
        self._stop.set()
        shutdown()
