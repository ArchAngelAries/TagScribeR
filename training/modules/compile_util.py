# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/modules/compile_util.py
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: none (verbatim).
"""Making torch.compile survive on this stack.

Two settings decide whether compiling a bucketed diffusion trainer does anything at all, and
neither is the obvious one:

1. **`cache_size_limit`.** PyTorch defaults it to 8. After 8 recompiles of a frame, dynamo gives up
   and silently falls back to eager — permanently, with no error. A bucketed dataset presents more
   than 8 distinct shapes almost immediately, so the first attempt at compiling Krea 2 here paid
   warm-up and guard overhead while running eager, which is why it measured SLOWER than no compile
   (0.794 vs 0.610 s/it) with nothing in the logs to explain it.

2. **A torch bug in symbolic-shape arithmetic.** `torch/utils/_sympy/functions.py` has
   `Mod.eval` raise `AssertionError(p)` for negative `p`, and inductor's tiling reaches it by
   substituting negative test values into expressions containing Mod. On Krea 2 this aborts the
   compile a few steps in and then segfaults the process (exit 139):

       torch._inductor.exc.InductorError: AssertionError: -111500631004807/2000000000000000

   The assertion is the bug — modulo is perfectly well defined for negative operands. Rather than
   vendor a patched copy of the function, `_patch_sympy_mod` wraps torch's own implementation and
   supplies the answer only for the case it wrongly refuses. That keeps this Apache-2.0 file free
   of code copied from elsewhere, and it degrades safely: if a future torch fixes the assertion,
   the wrapper simply never fires.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_patched = False


def _patch_sympy_mod() -> None:
    """Let sympy's Mod accept negative operands instead of asserting."""
    try:
        from torch.utils._sympy.functions import Mod
    except Exception:
        return

    original = Mod.eval.__func__ if hasattr(Mod.eval, "__func__") else Mod.eval

    def eval_allowing_negatives(cls, p, q):
        try:
            return original(cls, p, q)
        except AssertionError:
            # Only rescue the case torch refuses on principle: both literal numbers, a sane
            # modulus, and a negative left operand. Anything else is a real assertion — re-raise.
            if getattr(q, "is_Number", False) and getattr(p, "is_Number", False) and q >= 1 and p < 0:
                return p % q
            raise

    Mod.eval = classmethod(eval_allowing_negatives)


def init_compile(cache_size_limit: int = 8192) -> None:
    """Prepare the process for torch.compile. Safe to call more than once."""
    global _patched
    import torch._dynamo

    # Applies per process; dynamo reads it when it decides whether to recompile or bail to eager.
    torch._dynamo.config.cache_size_limit = cache_size_limit
    # The ACCUMULATED limit (default 256) binds before a raised per-frame limit — and under
    # fullgraph=True hitting it RAISES mid-run instead of falling back to eager, with an
    # error message pointing at the knob already raised. Raise both together.
    if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
        torch._dynamo.config.accumulated_cache_size_limit = max(
            cache_size_limit, torch._dynamo.config.accumulated_cache_size_limit)

    if not _patched:
        _patch_sympy_mod()
        _patched = True
        logger.info("[compile] cache_size_limit=%d; sympy Mod patched to accept negative operands "
                    "(torch asserts on them, which aborts inductor mid-run)", cache_size_limit)
