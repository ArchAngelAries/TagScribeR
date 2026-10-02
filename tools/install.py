#!/usr/bin/env python3
"""TagScribeR environment installer.

Run inside the target venv (install.bat / install.sh create it and call this).
Mirrors the environment conventions of the Fizgig trainer so both apps can
share one known-good AMD stack:

* Python 3.12, packages installed with ``uv``
* AMD on Windows: AMD's pinned multi-arch ROCm nightly wheels,
  ``torch[device-<gfx>]`` from https://rocm.nightlies.amd.com/whl-multi-arch/
* AMD on Linux:   AMD's stable multi-arch ROCm wheels from repo.amd.com
* NVIDIA:         CUDA 12.8 wheels from download.pytorch.org
* No GPU:         CPU wheels

torch is installed *before* requirements.txt, and requirements.txt never lists
torch, so a CUDA wheel can never replace a ROCm install.

Usage:
  python tools/install.py [--backend auto|rocm|cuda|cpu] [--arch gfx1100]
                          [--experimental] [--deps-only] [--with-bnb] [--dry-run]

Every pin below can be overridden with an environment variable of the same name.
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IS_WINDOWS = platform.system() == "Windows"

# --- Pinned stacks (kept in step with Fizgig's installers) -------------------
ROCM_WIN_INDEX = os.environ.get("ROCM_INDEX", "https://rocm.nightlies.amd.com/whl-multi-arch/")
ROCM_EXPERIMENTAL_INDEX = os.environ.get("ROCM_EXPERIMENTAL_INDEX", "https://nightly.repo.amd.com/rocm/whl-next/")
ROCM_WIN_TORCH_PIN = os.environ.get("TORCH_PIN", "2.12.0+rocm7.15.0a20260728")
ROCM_WIN_VISION_PIN = os.environ.get("TORCHVISION_PIN", "0.27.0+rocm7.15.0a20260728")
ROCM_WIN_SDK_PIN = os.environ.get("ROCM_SDK_DEVEL_PIN", "7.15.0a20260728")

ROCM_LINUX_INDEX = os.environ.get("ROCM_LINUX_INDEX", "https://repo.amd.com/rocm/whl-multi-arch/")
ROCM_LINUX_TORCH_PIN = os.environ.get("ROCM_LINUX_TORCH_PIN", "2.12.0+rocm7.14.0")
ROCM_LINUX_SDK_PIN = os.environ.get("ROCM_LINUX_SDK_PIN", "7.14.0")

CUDA_INDEX = os.environ.get("CUDA_INDEX", "https://download.pytorch.org/whl/cu128")
CUDA_TORCH_PIN = os.environ.get("CUDA_TORCH_PIN", "2.10.0")
CUDA_VISION_PIN = os.environ.get("CUDA_VISION_PIN", "0.25.0")
CPU_INDEX = "https://download.pytorch.org/whl/cpu"

# Community Windows ROCm bitsandbytes build (cp312 only) — optional, for 8/4-bit loading.
BNB_ROCM_WIN_WHEEL = os.environ.get(
    "BNB_WHEEL",
    "https://github.com/0xDELUXA/bitsandbytes_win_rocm/releases/download/"
    "0.50.2.dev0-py3.12-rocm7.16-win_amd64_all/bitsandbytes-0.50.2.dev0-cp312-cp312-win_amd64.whl",
)

# --- AMD architecture detection ---------------------------------------------
# PCI device id -> LLVM gfx target. Public hardware facts; covers the consumer
# RDNA parts ROCm-on-Windows supports. Unknown ids fall back to name matching.
AMD_PCI_TO_GFX = {
    # RDNA4
    "7550": "gfx1201", "7551": "gfx1201", "7590": "gfx1200",
    # RDNA3 / 3.5
    "744c": "gfx1100", "7448": "gfx1100", "745e": "gfx1100",
    "747e": "gfx1101", "7470": "gfx1101", "7460": "gfx1101",
    "7480": "gfx1102", "7483": "gfx1102", "7489": "gfx1102",
    "15bf": "gfx1103", "15c8": "gfx1103", "1900": "gfx1103", "1901": "gfx1103",
    "150e": "gfx1150", "1586": "gfx1151", "1114": "gfx1152",
    # RDNA2
    "73a0": "gfx1030", "73a1": "gfx1030", "73a2": "gfx1030", "73a3": "gfx1030",
    "73a5": "gfx1030", "73ab": "gfx1030", "73af": "gfx1030", "73bf": "gfx1030",
    "73df": "gfx1031", "73ff": "gfx1032", "73ef": "gfx1032", "743f": "gfx1034",
}
AMD_NAME_TO_GFX = [  # first substring match wins, so more specific names go first
    ("9070", "gfx1201"), ("9060", "gfx1200"),
    ("7900", "gfx1100"), ("w7900", "gfx1100"), ("w7800", "gfx1100"),
    ("7800", "gfx1101"), ("7700", "gfx1101"), ("w7700", "gfx1101"),
    ("7600", "gfx1102"), ("w7600", "gfx1102"),
    ("8060s", "gfx1151"), ("8050s", "gfx1151"), ("890m", "gfx1150"), ("880m", "gfx1150"),
    ("780m", "gfx1103"), ("760m", "gfx1103"), ("740m", "gfx1103"),
    ("6950", "gfx1030"), ("6900", "gfx1030"), ("6800", "gfx1030"),
    ("6750", "gfx1031"), ("6700", "gfx1031"), ("6650", "gfx1032"), ("6600", "gfx1032"),
    ("6500", "gfx1034"), ("6400", "gfx1034"),
]


def gfx_from_pnp(pnp_id: str, name: str = "") -> str | None:
    m = re.search(r"VEN_1002&DEV_([0-9A-Fa-f]{4})", pnp_id or "")
    if m and m.group(1).lower() in AMD_PCI_TO_GFX:
        return AMD_PCI_TO_GFX[m.group(1).lower()]
    lname = (name or "").lower().replace(" ", "")
    for key, gfx in AMD_NAME_TO_GFX:
        if key in lname:
            return gfx
    return None


def _run(cmd: list[str], timeout: int = 20) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              creationflags=0x08000000 if IS_WINDOWS else 0).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def detect_gpus() -> list[tuple[str, str, str]]:
    """List (vendor, name, gfx-or-empty) for display adapters."""
    found: list[tuple[str, str, str]] = []
    if IS_WINDOWS:
        out = _run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + $_.PNPDeviceID }"])
        for line in out.splitlines():
            if "|" not in line:
                continue
            name, pnp = line.strip().split("|", 1)
            if "VEN_10DE" in pnp.upper() or "nvidia" in name.lower():
                found.append(("nvidia", name, ""))
            elif "VEN_1002" in pnp.upper() or "radeon" in name.lower():
                found.append(("amd", name, gfx_from_pnp(pnp, name) or ""))
    else:
        drm = Path("/sys/class/drm")
        for dev in sorted(drm.glob("card[0-9]*/device")) if drm.exists() else []:
            try:
                vendor = (dev / "vendor").read_text().strip().lower()
                device = (dev / "device").read_text().strip().lower().removeprefix("0x")
            except OSError:
                continue
            if vendor == "0x10de":
                found.append(("nvidia", f"NVIDIA {device}", ""))
            elif vendor == "0x1002":
                gfx = AMD_PCI_TO_GFX.get(device, "")
                found.append(("amd", f"AMD {device}", gfx))
        if not any(v == "amd" and g for v, _, g in found):
            out = _run(["rocminfo"])
            m = re.search(r"Name:\s+(gfx1\d{2}[0-9a-f])", out)
            if m:
                found.append(("amd", "rocminfo", m.group(1)))
    if not any(v == "nvidia" for v, _, _ in found) and _run(["nvidia-smi", "-L"]).strip():
        found.append(("nvidia", "nvidia-smi", ""))
    return found


def choose_backend(gpus: list[tuple[str, str, str]]) -> tuple[str, str]:
    """Pick (backend, gfx): discrete AMD > NVIDIA > AMD APU > CPU.

    Strix Halo (gfx1151) counts as "discrete-class" given its large unified memory.
    """
    apus = ("gfx1103", "gfx1150", "gfx1152")
    amd = [g[2] for g in gpus if g[0] == "amd" and g[2]]
    discrete_amd = [a for a in amd if a not in apus]
    if discrete_amd:
        return "rocm", discrete_amd[0]
    if any(g[0] == "nvidia" for g in gpus):
        return "cuda", ""
    if amd:
        return "rocm", amd[0]
    return "cpu", ""


# --- install steps -------------------------------------------------------------
def uv(args: list[str], dry: bool) -> None:
    cmd = [sys.executable, "-m", "uv", "pip", "install", *args]
    print("\n> " + " ".join(cmd), flush=True)
    if dry:
        return
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"\nERROR: package installation failed (exit {rc}). See messages above.")


def ensure_uv(dry: bool) -> None:
    if shutil.which("uv") or _has_module("uv"):
        return
    print("Installing uv (fast package installer)...")
    if not dry:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "uv"])


def _has_module(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


def install_torch(backend: str, arch: str, experimental: bool, dry: bool) -> None:
    common = ["--index-strategy", "unsafe-best-match"]
    if backend == "rocm":
        if not arch:
            raise SystemExit("ERROR: could not determine your AMD GPU architecture. "
                             "Re-run with --arch gfxXXXX (e.g. --arch gfx1100 for RX 7900).")
        if IS_WINDOWS:
            if experimental:
                uv(common + ["--prerelease", "allow", "--index-url", ROCM_EXPERIMENTAL_INDEX,
                             f"torch[device-{arch}]", f"torchvision[device-{arch}]", "rocm-sdk-devel"], dry)
            else:
                uv(common + ["--index-url", ROCM_WIN_INDEX,
                             f"torch[device-{arch}]=={ROCM_WIN_TORCH_PIN}",
                             f"torchvision[device-{arch}]=={ROCM_WIN_VISION_PIN}",
                             f"rocm-sdk-devel=={ROCM_WIN_SDK_PIN}"], dry)
        else:
            uv(common + ["--index-url", ROCM_LINUX_INDEX,
                         f"torch[device-{arch}]=={ROCM_LINUX_TORCH_PIN}", f"torchvision[device-{arch}]",
                         f"rocm-sdk-devel=={ROCM_LINUX_SDK_PIN}"], dry)
    elif backend == "cuda":
        uv(["--index-url", CUDA_INDEX, "--extra-index-url", "https://pypi.org/simple",
            f"torch=={CUDA_TORCH_PIN}", f"torchvision=={CUDA_VISION_PIN}"], dry)
    else:
        uv(["--index-url", CPU_INDEX, "--extra-index-url", "https://pypi.org/simple",
            "torch", "torchvision"], dry)


def install_bitsandbytes(backend: str, dry: bool) -> None:
    if backend == "rocm":
        if not IS_WINDOWS:
            print("bitsandbytes on Linux ROCm: install a build matching your ROCm version manually.")
            return
        if sys.version_info[:2] != (3, 12):
            print("Skipping bitsandbytes: the Windows ROCm wheel is cp312-only.")
            return
        uv([BNB_ROCM_WIN_WHEEL], dry)
    elif backend == "cuda":
        uv(["bitsandbytes>=0.48"], dry)


def verify() -> None:
    code = (
        "import torch, sys; print(f'PyTorch {torch.__version__} | Python {sys.version.split()[0]}');"
        "print('HIP', getattr(torch.version,'hip',None), '| CUDA', torch.version.cuda);"
        "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU mode)')"
    )
    subprocess.call([sys.executable, "-c", code])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["auto", "rocm", "cuda", "cpu"], default="auto")
    ap.add_argument("--arch", default=os.environ.get("TAGSCRIBER_GFX", ""), help="AMD gfx target, e.g. gfx1100")
    ap.add_argument("--experimental", action="store_true", help="AMD Windows: floating newest nightlies (unpinned)")
    ap.add_argument("--deps-only", action="store_true", help="Only (re)install requirements.txt; leave torch alone")
    ap.add_argument("--with-bnb", action="store_true", help="Also install bitsandbytes (8/4-bit model loading)")
    ap.add_argument("--dev", action="store_true", help="Also install test tooling")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if sys.prefix == sys.base_prefix and not args.dry_run:
        print("ERROR: run this inside the TagScribeR virtual environment (use install.bat / install.sh).")
        return 1
    if sys.version_info[:2] != (3, 12):
        print(f"WARNING: Python {sys.version.split()[0]} detected. TagScribeR targets Python 3.12 "
              "(matching Fizgig and the ROCm wheels).")

    ensure_uv(args.dry_run)

    if not args.deps_only:
        backend, arch = args.backend, args.arch
        if backend == "auto" or (backend == "rocm" and not arch):
            gpus = detect_gpus()
            for vendor, name, gfx in gpus:
                print(f"Detected GPU: {name} [{vendor}{' ' + gfx if gfx else ''}]")
            auto_backend, auto_arch = choose_backend(gpus)
            if backend == "auto":
                backend = auto_backend
            arch = arch or auto_arch
        print(f"\nCompute backend: {backend.upper()}{' (' + arch + ')' if arch else ''}")
        install_torch(backend, arch, args.experimental, args.dry_run)
        if args.with_bnb:
            install_bitsandbytes(backend, args.dry_run)

    uv(["-r", str(ROOT / "requirements.txt")], args.dry_run)
    if args.dev:
        uv(["-r", str(ROOT / "requirements-dev.txt")], args.dry_run)

    if not args.dry_run:
        print()
        verify()
    print("\nDone. Launch TagScribeR with start.bat (Windows) or ./start.sh (Linux).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
