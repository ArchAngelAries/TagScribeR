# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/minimax/trainer.py (_load_h3_egrid,
# _turbo_adaln_forward, turbo_adaln_patch, turbo_adaln_unpatch and the AdaLN half of _prefilter_frozen_lora).
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# The run-time AdaLN injection and its grid file (assets/h3_silu_temb_grid.safetensors) come, through Fizgig, from
# larryvrh's ComfyUI-MiniMax-H3-Turbo node (Apache-2.0).
# Changes for TagScribeR: the LoRA's backbone Linears ride the family layer's frozen-adapter machinery
# (training/lora.py add_file, which already keeps only the shapes that fit this base), so only the AdaLN rows are read
# here; an AdaLN projection the family layer has wrapped (LoRALinear) is seen through its `.base`.
"""The MiniMax H3 Turbo LoRA's AdaLN rows on the pruned base, for previews.

The Turbo LoRA adapts two kinds of module: the blocks' Linears, which any LoRA loader can host, and the AdaLN
projections, whose LoRA rows are 2688 wide - the FULL model's silu(t_emb) space. The pruned base collapsed its time
embedder into an 8-wide curve table, so it cannot host those rows as weights. They are not discarded: they carry the
per-timestep modulation of the video and audio streams (dropping them is what makes few-step output fall apart). Each
incoming t_emb row is matched to its nearest table row and the full model's precomputed silu(t_emb) row for that grid
point stands in:  x += B @ A @ silu(t_emb).
"""
import logging
import os

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_EGRID_CACHE = [None]


def load_h3_egrid():
    """The full model's silu(t_emb) rows on a 1025-point t grid, [1025, 2688] (Fizgig _load_h3_egrid)."""
    if _EGRID_CACHE[0] is None:
        from safetensors.torch import load_file
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "h3_silu_temb_grid.safetensors")
        _EGRID_CACHE[0] = load_file(p)["silu_t_emb_grid"]
    return _EGRID_CACHE[0]


def _linear_of(adaln):
    """An AdalnProj's Linear, through the family layer's wrapper if it has one."""
    return getattr(adaln.linear, "base", adaln.linear)


def read_adaln_pairs(dit, path, strength):
    """The file's AdaLN LoRA rows this base cannot host as weights: [(AdalnProj, A, B * strength)] on the CPU (Fizgig
    _prefilter_frozen_lora). A row whose shape fits the module's own Linear is left to the ordinary LoRA loader."""
    from safetensors.torch import load_file
    sd = load_file(path)
    parents = {f"{n.replace('.', '_')}_linear": m for n, m in dit.named_modules() if type(m).__name__ == "AdalnProj"}
    pairs = []
    for key in sorted(sd):
        # kohya (lora_unet_<flattened>.lora_down) or PEFT / dotted (<prefix.><dotted>.lora_A): Fizgig converts the
        # latter to kohya names first (networks/lora.py ensure_kohya_lora_state_dict)
        for dn, un in ((".lora_down.weight", ".lora_up.weight"), (".lora_A.weight", ".lora_B.weight")):
            if key.endswith(dn):
                stem = key[:-len(dn)]
                break
        else:
            continue
        flat = stem
        for p in ("lora_unet_", "model.diffusion_model.", "diffusion_model.", "transformer."):
            if flat.startswith(p):
                flat = flat[len(p):]
                break
        ap = parents.get(flat.replace(".", "_"))
        down, up = sd[key], sd.get(stem + un)
        if ap is None or up is None:
            continue
        lin = _linear_of(ap)
        if down.shape[1] == lin.in_features and up.shape[0] == lin.out_features:
            continue                                     # hosted as a weight module
        if up.shape[0] == lin.out_features:
            # Fizgig folds the load strength into B and applies no alpha / rank factor to these rows
            pairs.append((ap, down.clone(), up.clone() * float(strength)))
    return pairs


def _adaln_forward(base, updates, table, egrid):
    """A replacement AdalnProj.forward that adds one or more full-model AdaLN LoRA updates (Fizgig
    _turbo_adaln_forward). `updates` is a list of (A, B): several LoRAs on one module ADD."""
    def forward(t_emb):
        x = base.linear(F.silu(t_emb) if base.apply_silu else t_emb)
        idx = torch.cdist(t_emb.detach().float(), table.to(t_emb.device, torch.float32)).argmin(dim=1)
        st = egrid.to(t_emb.device)[idx].to(x.dtype)
        for A, B in updates:
            x = x + (B.to(x) @ (A.to(x) @ st.T)).T
        x = x.view(x.shape[0] * base.modalities, base.expand * base.hidden)
        return x.chunk(base.expand, dim=-1)
    return forward


def adaln_patch(dit, pairs, device, dtype, egrid=None):
    """Install the AdaLN injection for the preview render. Returns the number of modules patched.

    Instance-attribute forwards: assignment shadows the class method, deletion restores it - the module tree is never
    rebuilt (Fizgig turbo_adaln_patch)."""
    if not pairs or not getattr(dit, "pruned_adaln", False):
        return 0
    egrid = load_h3_egrid() if egrid is None else egrid
    table = dit.adaln_t_table
    if table.shape[0] != egrid.shape[0]:
        logger.warning(f"[turbo] adaln grid rows {egrid.shape[0]} != table rows {table.shape[0]} — adaln injection "
                       f"skipped")
        return 0
    eg = egrid.to(device)
    by_mod = {}
    for mod, A, B in pairs:
        if A.shape[1] != eg.shape[1]:
            continue
        by_mod.setdefault(id(mod), (mod, []))[1].append((A.to(device, dtype), B.to(device, dtype)))
    for mod, updates in by_mod.values():
        mod.forward = _adaln_forward(mod, updates, table, eg)
    return len(by_mod)


def adaln_unpatch(pairs):
    """Remove the injection (idempotent): the class forward comes back, and the GPU copies of A / B / the grid die
    with the closures (Fizgig turbo_adaln_unpatch)."""
    for mod, _a, _b in pairs:
        try:
            del mod.forward
        except AttributeError:
            pass
