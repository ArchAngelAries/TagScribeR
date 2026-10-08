# Ported from Fizgig (https://github.com/shootthesound/Fizgig): src/fizgig/krea2/trainer.py (_CheckpointedBlock,
# _find_host_compiler, _compile_blocks, and the Compile Blocks resolution in train_krea2) and src/fizgig/utils/
# capabilities.py (should_compile, compile_boundary, triton_matches_torch, has_host_c_compiler and their measured constants).
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: the base is named by precision ("nf4" / "int8" / "fp8" / "bf16") instead of Fizgig's
# quant_4bit / quant_int8 / fp8_scaled flags; ROCm is read from torch.version.hip and free VRAM from training.quant
# (Fizgig's detect()); init_compile and the SDPA backend settle come from training/modules. Rules, thresholds, messages
# and the fp8 / triton / compiler guards are Fizgig's.
# Brought level with Fizgig 7.0.1 families/compile.py and utils/capabilities.py compile_blocker (commit 1c8ec88): one
# module for every family with compiles=True (it lived in training/families/krea2/), any block signature
# (CheckpointedBlock(*args)), the block list from driver.compile_targets, fullgraph from the description,
# compile_blocker for the generic Auto rule, and the two expected inductor notices silenced.
"""torch.compile of a family's transformer blocks, as Fizgig does it (Krea 2's rules measured first).

Compile Blocks is Auto / On / Off (and "outside", a hand-set power value). Auto is a judgement, off wherever it is not
known to win: on ROCm (recompiles per bucket shape on HIP), with block swap, without triton / a matching triton / a C
compiler, on a base other than INT8 / NF4 (only those are measured), when the run is too short to repay the ~90 s
warm-up, or when it will not fit in VRAM; INT8 falls back to the checkpoint-OUTSIDE boundary before it gives up. On
means on, but only where it can run: it still refuses block swap, a missing triton / compiler, and the fp8 base on
GPUs without fp8 Triton kernels. Compiling a failed or unavailable setup costs speed, never the run.

Each block is compiled AFTER the LoRA wrapped its Linears (the compiled graph is the one that actually runs), as a
_CheckpointedBlock so the gradient checkpoint sits inside the graph ("inside") or around it ("outside").
"""
import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ---- Fizgig utils/capabilities.py -----------------------------------------------------------------------------
# torch major.minor -> the triton major.minor releases it is known to work with (the version the Linux torch wheel
# depends on, and the one before it, which the field has proven). A triton built for a newer torch imports fine and
# then fails or hangs INSIDE torch.compile, so an import check alone lets it through.
_TRITON_FOR_TORCH = {
    "2.8": ("3.3", "3.4"),
    "2.9": ("3.4", "3.5"),
    "2.10": ("3.5", "3.6"),
    "2.11": ("3.6", "3.7"),
    "2.12": ("3.7", "3.8"),
}
_HEADROOM_GB = 1.5
_BATCH_GB_PER_IMAGE = 2.4
# Compile pays ~90 s up front. Doubled for margin: at break-even there is nothing to win, and being wrong should cost a
# few percent, not a run. INT8 saves 0.30 s/step (0.59 -> 0.29), NF4 0.153 s/step (0.71 -> 0.56) - Fizgig measurements.
_COMPILE_WARMUP_S = 90.0
_COMPILE_SAVING_S = {"int8": 0.300, "nf4": 0.153}
_COMPILE_MARGIN = 2.0
# INT8 + compile peaked at 21.7 GB against 17.8 GB for INT8 alone; NF4 + compile is VRAM-neutral (12.9 GB vs 13.6 GB).
# BOTH figures are 0.25 MP measurements.
_INT8_COMPILE_PEAK_GB = 20.0
_NF4_COMPILE_PEAK_GB = 13.0
# Resolution scaling UNDER COMPILE (nothing like the eager 0.25 GB/MP): inductor saves activations between the forward
# and backward graphs, and those scale with token count - a real 0.98 MP INT8+compile run on a 32 GB card OOM'd on the
# first backward. 15 adds slack in the only safe direction (over-declining runs uncompiled; under-declining repeats the
# OOM). The NF4 figure is extrapolated from the INT8 data point.
_COMPILE_GB_PER_MP = 15.0
# The OUTSIDE boundary (#99): checkpoint kept outside the compiled region, so stashes stay at eager checkpointing's
# level. Measured (Krea 2 INT8, 46 imgs @ 1.05 MP, rank 32, RTX 5090): peak ~18.7 GB net vs eager's ~18.0, 2.4 s/step vs
# eager 3.30 (~27% faster), warm-up ~25 s.
_INT8_COMPILE_OUTSIDE_PEAK_GB = 19.0
_COMPILE_OUTSIDE_ANCHOR_MP = 1.05


def _major_minor(v: str) -> str:
    parts = str(v or "").split("+")[0].split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else str(v or "")


def triton_matches_torch(triton_version=None, torch_version=None):
    """(ok, note): whether this triton is one torch is known to work with. Unknown torch versions are not gated
    (ok, ""). Versions default to the installed ones."""
    try:
        if torch_version is None:
            torch_version = torch.__version__
        if triton_version is None:
            import triton as _tr
            triton_version = getattr(_tr, "__version__", "")
    except Exception:
        return True, ""
    want = _TRITON_FOR_TORCH.get(_major_minor(torch_version))
    if not want:
        return True, ""
    have = _major_minor(triton_version)
    if have in want:
        return True, ""
    return False, (f"triton {triton_version} does not pair with torch {torch_version} "
                   f"(torch {_major_minor(torch_version)} needs triton {' or '.join(want)}; "
                   f"a mismatched triton fails inside torch.compile, sometimes silently) - "
                   f"reinstall with: pip install \"triton-windows>={want[0]}.0,<{float(want[1]) + 0.1:.1f}\"")


def has_host_c_compiler(platform: Optional[str] = None) -> bool:
    """POSIX: is a C compiler on PATH? inductor/triton build small host-side stubs with one at runtime. Windows always
    returns True here: MSVC lives outside PATH by design and find_host_compiler's vcvars import handles it."""
    import shutil
    if (platform or os.name) == "nt":
        return True
    return any(shutil.which(c) for c in ("cc", "gcc", "clang"))


def is_rocm() -> bool:
    """AMD ROCm (HIP) PyTorch build (Fizgig utils/gpu_backend.py is_rocm)."""
    if not torch.cuda.is_available():
        return False
    return bool(getattr(torch.version, "rocm", None) or getattr(torch.version, "hip", None)
                or "+rocm" in (getattr(torch, "__version__", "") or "").lower())


def _triton_importable() -> bool:
    try:
        import importlib
        importlib.import_module("triton")
        return True
    except Exception:
        return False


def _free_vram_gb() -> float:
    from training.quant import free_vram_gb
    return free_vram_gb()


def _kind(precision: str):
    return precision if precision in ("nf4", "int8") else None


def compile_boundary(precision: str, vram_gb=None, mp: float = 0.25, batch: int = 1) -> str:
    """'inside' | 'outside' - where the gradient checkpoint sits for an EXPLICIT Compile=On (#99). Never declines (On
    means on); it only places the boundary where it fits: inside-the-graph is 1.19x faster per block but its stashes
    scale hard with tokens (measured >32 GB at 1 MP on INT8), outside stays at eager-level stashes."""
    if _kind(precision) != "int8":
        return "inside"          # NF4 / other: outside unmeasured - behave as before
    vram = vram_gb if vram_gb is not None else _free_vram_gb()
    if not vram:
        # no readable GPU: no basis to pick, so inside (today's behaviour), not "inside doesn't fit -> outside"
        return "inside"
    _step_mp = float(mp) * max(1, int(batch))
    _res_gb = _COMPILE_GB_PER_MP * max(0.0, _step_mp - 0.25)
    if vram < _INT8_COMPILE_PEAK_GB + _res_gb + _HEADROOM_GB:
        return "outside"
    return "inside"


def should_compile(total_steps: int, precision: str, blocks_to_swap: int, vram_gb: Optional[float] = None,
                   mp: float = 0.25, batch: int = 1) -> tuple:
    """Decide whether torch.compile pays for itself on this run. Returns (False | True | "outside", reason).

    `mp` is the run's largest bucket in megapixels, `batch` its batch size: what compiled activation stashes scale with
    is tokens PER STEP, and batch multiplies tokens exactly as resolution does, so the load term is mp x batch."""
    if is_rocm():
        return False, ("ROCm/HIP PyTorch build - Auto leaves torch.compile off "
                       "(recompiles per bucket shape on HIP; set Compile Blocks to On to override)")
    vram = vram_gb if vram_gb is not None else _free_vram_gb()

    if blocks_to_swap:
        return False, ("block swap is active - swapping moves weights between devices every step, "
                       "which compiled graphs cannot tolerate")
    if not _triton_importable():
        return False, "triton is not installed (pip install triton-windows on Windows)"
    _ok, _why = triton_matches_torch()
    if not _ok:
        return False, _why
    if not has_host_c_compiler():
        return False, ("no C compiler on this system - inductor/triton build host-side stubs "
                       "with one at runtime (on Debian/Ubuntu: apt install gcc); "
                       "running uncompiled")

    kind = _kind(precision)
    if kind is None:
        return False, "only measured for the quantised paths (NF4 / INT8); not enabled for fp8 or bf16"
    _step_mp = float(mp) * max(1, int(batch))       # MP of latents per step
    _res_gb = _COMPILE_GB_PER_MP * max(0.0, _step_mp - 0.25)
    _shape = (f" at {mp:.2f} MP" + (f" x batch {batch}" if batch > 1 else "")) if _res_gb else ""
    _fix = (" (lower Target Megapixels or batch size to compile)" if _res_gb else "")
    _boundary = "inside"
    if kind == "int8" and vram < _INT8_COMPILE_PEAK_GB + _res_gb + _HEADROOM_GB:
        # Inside-the-graph doesn't fit at this token load - try the OUTSIDE boundary (#99): same fused kernels,
        # eager-level stashes, measured ~27% faster than eager at 1 MP. Only falls to uncompiled when even that can't
        # fit. Batch is charged at the measured EAGER term, not laundered through the step-MP slope.
        _out_need = (_INT8_COMPILE_OUTSIDE_PEAK_GB
                     + _COMPILE_GB_PER_MP * max(0.0, float(mp) - _COMPILE_OUTSIDE_ANCHOR_MP)
                     + _BATCH_GB_PER_IMAGE * max(0, int(batch) - 1))
        if vram >= _out_need + _HEADROOM_GB:
            _boundary = "outside"
        else:
            return False, (f"INT8 + compile peaks near {_INT8_COMPILE_PEAK_GB + _res_gb:.0f} "
                           f"GB{_shape} (checkpoint-outside still ~{_out_need:.0f} GB) and only "
                           f"{vram:.1f} GB is free - INT8 alone still fits, compile does not" + _fix)
    if kind == "nf4" and _res_gb and vram < _NF4_COMPILE_PEAK_GB + _res_gb + _HEADROOM_GB:
        return False, (f"NF4 + compile peaks near {_NF4_COMPILE_PEAK_GB + _res_gb:.0f} GB{_shape} "
                       f"and only {vram:.1f} GB is free - NF4 alone still fits, compile does not" + _fix)

    needed = int(_COMPILE_WARMUP_S / _COMPILE_SAVING_S[kind] * _COMPILE_MARGIN)
    if total_steps < needed:
        return False, (f"{total_steps} steps is too short - compiling costs ~{_COMPILE_WARMUP_S:.0f} s "
                       f"up front and needs ~{needed} steps on the {kind.upper()} path to pay back")
    if _boundary == "outside":
        # Truthy like True, so bool-minded callers keep working; boundary-aware callers pass it to compile_blocks.
        return "outside", (f"{total_steps} steps on the {kind.upper()} path - inside-the-graph "
                           f"doesn't fit{_shape}, compiling with the checkpoint OUTSIDE the "
                           f"region instead (measured ~27% faster than eager at 1 MP)")
    return True, (f"{total_steps} steps on the {kind.upper()} path - compile pays back within "
                  f"~{needed} steps and this run is longer")


# ---- Fizgig krea2/trainer.py ----------------------------------------------------------------------------------
class CheckpointedBlock(torch.nn.Module):
    """A transformer block that does its own gradient checkpointing.

    Exists so torch.compile can capture the checkpoint inside the graph. `_handles_checkpointing` tells the DiT forward
    not to wrap it a second time."""

    _handles_checkpointing = True

    def __init__(self, block, checkpointing: bool):
        super().__init__()
        self.block = block
        self.checkpointing = checkpointing

    def forward(self, *args):
        if self.checkpointing and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(self.block, *args, use_reentrant=False)
        return self.block(*args)


def find_host_compiler() -> bool:
    """Make sure a host C/C++ compiler exists before torch.compile runs; never crash the run.

    Inductor/triton build small host-side stubs at runtime, so compile without a compiler dies with "Failed to find C
    compiler". POSIX: check PATH for cc/gcc/clang. Windows: `cl.exe` is installed by Visual Studio but only exposed
    inside a developer prompt, so running vcvars64.bat and importing the environment it sets is what a developer prompt
    does; doing it here means the user does not have to know any of this."""
    import shutil
    import subprocess

    if os.name != "nt":
        if has_host_c_compiler():
            return True
        logger.warning("[compile] no C compiler found - torch.compile needs one to build "
                       "inductor/triton host-side stubs (on Debian/Ubuntu: apt install gcc). "
                       "Training continues uncompiled.")
        return False
    if shutil.which("cl"):
        return True

    vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                           "Microsoft Visual Studio", "Installer", "vswhere.exe")
    roots = []
    if os.path.isfile(vswhere):
        try:
            out = subprocess.run([vswhere, "-latest", "-products", "*", "-property", "installationPath"],
                                 capture_output=True, text=True, timeout=30)
            roots += [line.strip() for line in out.stdout.splitlines() if line.strip()]
        except Exception:
            pass
    for pf in (os.environ.get("ProgramFiles", r"C:\Program Files"),
               os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")):
        for year in ("2022", "2019"):
            for ed in ("BuildTools", "Community", "Professional", "Enterprise"):
                roots.append(os.path.join(pf, "Microsoft Visual Studio", year, ed))

    for root in roots:
        vcvars = os.path.join(root, "VC", "Auxiliary", "Build", "vcvars64.bat")
        if not os.path.isfile(vcvars):
            continue
        try:
            # shell=True is intentional (as in Fizgig): vcvars is a path just discovered via vswhere / well-known VS
            # install roots (not external input), and the shell's && sources the .bat file's env vars into `set`.
            out = subprocess.run(f'"{vcvars}" >nul && set', shell=True, capture_output=True,
                                 text=True, timeout=120)
            if out.returncode != 0:
                continue
            for line in out.stdout.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k] = v
            if shutil.which("cl"):
                logger.info("[compile] MSVC found via %s", os.path.basename(root))
                return True
        except Exception:
            continue

    logger.warning("[compile] no MSVC C++ compiler found - torch.compile needs one on Windows to "
                   "build inductor's host-side code. Direct installer: "
                   "https://aka.ms/vs/17/release/vs_BuildTools.exe (tick the 'Desktop development "
                   "with C++' workload), or leave Compile Blocks off. Training continues uncompiled.")
    return False


def compile_blocks(dit, blocks_to_swap: int, fp8_scaled: bool = False, boundary: str = "inside", blocks=None,
                   fullgraph: bool = True) -> int:
    """Compile each transformer block of `blocks` (the driver's ModuleList, replaced in place; default dit.blocks).
    The DiT's forward must call a block that has `_handles_checkpointing` directly, without checkpointing it again.
    Returns the number compiled (0 = refused / unavailable).

    The win is real on the quantised path (inductor fuses the per-matmul quantise/dequantise elementwise work that bounds
    INT8), and small on dense bf16. It costs compile time on the first step, and a recompile for every new latent shape
    a bucketed dataset presents.

    `boundary` places the gradient checkpoint relative to the compiled region (#99): "inside" (default) compiles the
    checkpoint INTO the graph - worth 1.19x per block, but inductor's partitioner stashes far more intermediates as
    tokens grow (measured >32 GB at 1 MP on the INT8 path, vs ~18 GB eager). "outside" compiles the raw block and keeps
    the checkpoint wrapper eager: recompute reruns the compiled graph, stashes stay at eager checkpointing's level.

    Refused under block swap: compiled graphs assume their weights stay put, and swap moves them between CPU and GPU
    every step. Also refused for the fp8 base on pre-Ada GPUs: inductor lowers the fp8 dequant to an fp8e4nv Triton
    kernel that only SM 8.9+ silicon has, and the resulting ValueError escapes dynamo's suppress_errors and kills the run
    before step one (#97, RTX 3090)."""
    if blocks_to_swap > 0:
        logger.warning("[compile] ignored - block swap moves weights between devices every step, "
                       "which invalidates compiled graphs. Quantise instead of swapping if you "
                       "want both.")
        return 0
    if fp8_scaled:
        _cc = None
        try:
            # `import torch as _torch`, NOT the bare name: the `import torch._dynamo` further down makes `torch`
            # function-LOCAL, so referencing it here raises UnboundLocalError - which the except would silently eat and
            # the guard would never fire (Fizgig's #97 regression note)
            import torch as _torch
            if _torch.cuda.is_available():
                _cc = _torch.cuda.get_device_capability()
        except Exception:
            pass
        if _cc is not None and _cc < (8, 9):
            logger.warning("[compile] ignored - the fp8 base needs fp8 Triton kernels "
                           "(fp8e4nv), which need SM 8.9+ (RTX 40-series or newer); this GPU "
                           "is SM %d.%d. Pick INT8 or NF4 Base Precision to compile on this "
                           "card. Training continues uncompiled.", _cc[0], _cc[1])
            return 0
    if not _triton_importable():
        logger.warning("[compile] ignored - triton is not installed (pip install triton-windows "
                       "on Windows, triton on Linux)")
        return 0
    try:
        _ok, _why = triton_matches_torch()
    except Exception:
        _ok, _why = True, ""
    if not _ok:
        # A triton built for another torch imports fine and then fails or hangs INSIDE torch.compile (a preview that
        # never comes back, no log) - say so and run eager.
        logger.warning("[compile] ignored - %s. Training continues uncompiled.", _why)
        return 0
    if not find_host_compiler():
        return 0
    import torch._dynamo
    # Raises the recompile ceiling (default 8, which a bucketed dataset exhausts immediately - after which dynamo
    # silently runs eager) and works around a torch assertion that otherwise aborts inductor mid-run.
    from training.modules.compile_util import init_compile
    init_compile()
    # Settle the SDPA backend global BEFORE tracing: its lazy first-use probe inside a compiled block is exactly what
    # fullgraph=True raises on.
    from training.modules import sdpa as _sdpa
    _sdpa.prime()
    # A compile failure must cost speed, not the run.
    torch._dynamo.config.suppress_errors = True
    # Two inductor notices that are expected here, not problems: TF32 stays off on purpose (the LoRA's fp32 maths
    # would change), and a complex-number op (Qwen's RoPE) runs uncompiled inside the compiled block.
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning,
                            message=r"TensorFloat32 tensor cores for float32 matrix multiplication available")
    warnings.filterwarnings("ignore", category=UserWarning,
                            message=r"Torchinductor does not support code generation for complex operators")
    blocks = dit.blocks if blocks is None else blocks

    # fullgraph=True refuses to compile around a graph break instead of quietly degrading. Each block is wrapped so
    # the GRADIENT CHECKPOINT sits INSIDE the compiled region (1.19x on a real block: 8.817 -> 7.428 ms/block-step).
    checkpointing = bool(getattr(dit, "gradient_checkpointing", False))
    n = 0
    if boundary == "outside":
        for i, block in enumerate(blocks):
            blocks[i] = CheckpointedBlock(torch.compile(block, fullgraph=fullgraph), checkpointing)
            n += 1
        logger.info("[compile] %d blocks compiled (checkpoint OUTSIDE the "
                    "compiled region - recompute reruns the compiled graph, so activation "
                    "stashes stay at eager level; the high-resolution fit) - the first "
                    "step of each new shape pauses to compile", n)
        return n
    for i, block in enumerate(blocks):
        blocks[i] = torch.compile(CheckpointedBlock(block, checkpointing), fullgraph=fullgraph)
        n += 1
    logger.info("[compile] %d blocks compiled (checkpoint inside the graph, "
                "cache_size_limit=8192) - the first step of each new shape pauses to compile", n)
    return n


def compile_blocker(blocks_to_swap: int) -> Optional[str]:
    """Why Auto must not compile on this machine / run (ROCm, block swap, no or mismatched triton, no host C
    compiler), or None (Fizgig utils/capabilities.py). The generic family rule's guard."""
    if is_rocm():
        return ("ROCm/HIP PyTorch build - Auto leaves torch.compile off "
                "(recompiles per bucket shape on HIP; set Compile Blocks to On to override)")
    if blocks_to_swap:
        return "block swap is active - swapping moves weights between devices every step, " \
               "which compiled graphs cannot tolerate"
    if not _triton_importable():
        return "triton is not installed (pip install triton-windows on Windows)"
    _ok, _why = triton_matches_torch()
    if not _ok:
        return _why
    if not has_host_c_compiler():
        return ("no C compiler on this system - inductor/triton build host-side stubs "
                "with one at runtime (on Debian/Ubuntu: apt install gcc); running uncompiled")
    return None


def resolve(setting, *, precision: str, blocks_to_swap: int, total_steps: int, mp: float, batch: int):
    """The Compile Blocks setting (auto | on | off | outside) -> False | True | "outside" (Fizgig train_krea2). "auto"
    weighs the ~90 s warm-up against how long this run actually is; "on" / "off" are explicit overrides; "outside" is
    the hand-set high-resolution boundary (#99). Logs what it decided."""
    s = str(setting or "auto").strip().lower().split(" ")[0]
    do = s in ("1", "true", "on", "yes")
    if s == "outside":
        do = "outside"
    if do is True:
        # Explicit On means ON - but the checkpoint boundary is still placed where it fits (#99): forced inside-the-graph
        # at 1 MP measured >32 GB and OOM'd on a 32 GB card; outside completed in ~18.7 GB at ~27% faster than eager.
        if compile_boundary(precision, mp=mp, batch=batch) == "outside":
            logger.info("[compile] on: inside-the-graph won't fit at this token load - "
                        "compiling with the checkpoint OUTSIDE the region instead.")
            do = "outside"
    if s == "auto":
        do, why = should_compile(total_steps, precision, blocks_to_swap, mp=mp, batch=batch)
        logger.info("[compile] auto: %s - %s",
                    ("ENABLED (checkpoint outside)" if do == "outside" else ("ENABLED" if do else "off")), why)
    if do and blocks_to_swap:
        # the GUI's note (Fizgig lora_trainer_gui.py:34127): On with block swap is ignored this run
        logger.info(f"[compile] ignored this run - block swap is active ({blocks_to_swap} blocks), and compiled "
                    "graphs can't tolerate weights moving between CPU and GPU each step. Use 4-bit (NF4) instead of "
                    "swapping if you want compile as well.")
    return do
