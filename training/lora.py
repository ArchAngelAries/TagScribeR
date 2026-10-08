# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/families/lora.py, with the LoKR helpers
# (factorization, _lokr_forward_update, lycoris_scale_from_keys) from src/fizgig/networks/lora.py.
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: Conv2d targets (LoRAConv2d: down = the base conv's kernel, up = 1x1, as Fizgig's kohya
# LoRAModule does for Conv2d), needed by UNet families such as SDXL; `LoRAFormat(kohya=True)` families (Krea 2) write
# and read kohya keys (lora_unet_<flattened path>, a LoKR as diffusion_model.<dotted path>, as Fizgig's
# krea2/trainer.py _save_lora does); a frozen file's `diff_b` bias deltas (the Krea 2 Turbo LoRA's) are applied while the
# adapter is on (Fizgig krea2/trainer.py _apply_turbo_lora); `driver.lora_key_name` lets a kohya family write (and read)
# keys named differently from its module paths (SDXL: LDM names). Behaviour of non-kohya families is unchanged.
# Brought level with Fizgig 7.0.1 (commit 1c8ec88): frozen LoHa files load (LoHa, add_loha), `lora.down` / `lora.up` key
# spelling and the `unet.` prefix are read, and the reader goes through `driver.convert_lora_state_dict` and
# `driver.alias_flat` (another trainer's module names: Krea 2 LoRAs from OneTrainer / AI-Toolkit, Fizgig 6.8.2), and a
# frozen adapter in the model's dtype is added in one fused, once-rounded step.
"""The family layer's LoRA: wraps a family's Linears, trains one adapter and runs any number of frozen ones.

Which Linears (driver.block_map / lora_target_names) and how files are keyed (description.lora: file prefix, down/up
names, alpha key) come from the family, so every described family gets the same adapter machinery and writes its own
ComfyUI-compatible format.

Each wrapped Linear computes W x + sum_i s_i * B_i(A_i(x)). For a frozen adapter
    s = alpha / rank * load_strength * block_strength * (adapter on) * (block on)
where block_strength / block on belong to the block the module is in (driver.block_of); modules outside the block
map (e.g. a speed LoRA's modulation layers) follow the adapter's load strength and on/off only. The trainable adapter
is named "lora"; frozen ones (training adapter, context LoRA, a workbench primary/donor) get their own names.

Files in any common layout are accepted: the family's own keys, kohya (`lora_unet_<flattened>.lora_down/up`), or
PEFT / diffusers (`lora_A/lora_B` or `lora_down/lora_up`, bare or under `transformer.` / `diffusion_model.`).
LyCORIS LoKR files are read (low-rank factors multiplied out), and frozen LoHa files (the Hadamard delta materialised
per forward). Another trainer's module names reach this model through the driver's alias_flat.

Conv2d modules named by the driver are wrapped too (LoRAConv2d). LoKR adapters are Linear-only: a LoKR run leaves a
family's Conv2d targets untrained (logged), the same as LyCORIS' default "LoKR for Linear" preset.
"""
import logging
import math
import re
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

TRAINABLE = "lora"
_PREFIXES = ("transformer.", "unet.", "diffusion_model.", "model.diffusion_model.", "base_model.model.", "")


_ALPHA_SENTINEL_THRESHOLD = 1e6   # alpha >= this in a LyCORIS file means "scale already baked in" (scale = 1)


def factorization(n: int, factor: int) -> tuple:
    """Split n into (small, large) with small * large == n and small the LARGEST divisor of n that is <= factor - the
    LyCORIS convention, so a trained file's shapes look like every other LoKR in the wild. A prime n degenerates to
    (1, n), which is still a valid Kronecker factor."""
    if factor < 1:
        factor = 1
    small = 1
    for cand in range(min(factor, n), 0, -1):
        if n % cand == 0:
            small = cand
            break
    return small, n // small


def _lokr_forward_update(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, a: int, b: int, c: int,
                         d: int) -> torch.Tensor:
    """kron(w1, w2) @ x without materialising it: kron(A, B) @ vec(X) = vec(B @ X @ A.T) (column-major vec).
    x (..., N=b*d) -> (..., M=a*c)."""
    orig = x.shape[:-1]
    X = x.reshape(*orig, b, d).transpose(-1, -2)              # (..., d, b)
    Y = w2 @ X @ w1.transpose(-1, -2)                          # (..., c, a)
    return Y.transpose(-1, -2).reshape(*orig, a * c)


def lycoris_scale_from_keys(mod_keys: Dict[str, torch.Tensor]) -> float:
    """Effective scale of a LoKR/LoHa module's dense delta (the LyCORIS rule): alpha / min(decomposition ranks in use);
    a full-matrix LoKR has dim 1, so scale = alpha."""
    alpha_t = mod_keys.get("alpha")
    alpha = float(alpha_t.item()) if alpha_t is not None else 1.0
    if alpha >= _ALPHA_SENTINEL_THRESHOLD:
        return 1.0
    if mod_keys.get("hada_w1_a") is not None:
        return alpha / max(1, min(int(mod_keys["hada_w1_a"].shape[1]), int(mod_keys["hada_w2_a"].shape[1])))
    dims = []
    if mod_keys.get("lokr_w1_a") is not None:
        dims.append(int(mod_keys["lokr_w1_a"].shape[1]))
    if mod_keys.get("lokr_w2_a") is not None:
        dims.append(int(mod_keys["lokr_w2_a"].shape[1]))
    return alpha / (max(1, min(dims)) if dims else 1)


class LoRAFactor(nn.Linear):
    """One adapter matrix (A or B). Its own class so block-swap offloaders, which stream every module whose class name
    ends in "Linear", leave the adapters resident: trainable weights must never be moved between steps."""


class LoHa(nn.Module):
    """A frozen LoHa (Hadamard) adapter: delta = (w1_a @ w1_b) * (w2_a @ w2_b), materialised per forward as the old
    loaders' LoHaInfModule does (the Hadamard product does not factor); the scale is applied by the LoRALinear."""

    def __init__(self, w1a, w1b, w2a, w2b):
        super().__init__()
        self.hada_w1_a, self.hada_w1_b = nn.Parameter(w1a.clone()), nn.Parameter(w1b.clone())
        self.hada_w2_a, self.hada_w2_b = nn.Parameter(w2a.clone()), nn.Parameter(w2b.clone())

    def delta(self):
        return (self.hada_w1_a.float() @ self.hada_w1_b.float()) * (self.hada_w2_a.float() @ self.hada_w2_b.float())

    def forward(self, x):
        w = (self.hada_w1_a @ self.hada_w1_b) * (self.hada_w2_a @ self.hada_w2_b)
        return x @ w.to(x.dtype).transpose(-1, -2)


class LoKR(nn.Module):
    """A LoKR (Kronecker) adapter: delta = scale * kron(w1, w2), applied without materialising it (Krea 2's trainable
    recipe: full-matrix w2, w1 about factor x factor). Frozen files may carry w1/w2 as low-rank factors; they are
    multiplied out at load."""

    def __init__(self, in_f, out_f, factor=8, w1=None, w2=None):
        super().__init__()
        if w1 is not None:
            self.a, self.b = w1.shape
            self.c, self.d = w2.shape
        else:
            self.a, self.c = factorization(out_f, int(factor))
            self.b, self.d = factorization(in_f, int(factor))
        if self.a * self.c != out_f or self.b * self.d != in_f:
            raise ValueError(f"LoKR factors {self.a}x{self.b} (x) {self.c}x{self.d} do not tile {out_f}x{in_f}")
        self.lokr_w1 = nn.Parameter(torch.empty(self.a, self.b) if w1 is None else w1.clone())
        self.lokr_w2 = nn.Parameter(torch.zeros(self.c, self.d) if w2 is None else w2.clone())
        if w1 is None:
            nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))   # w2 = 0: delta is exactly 0 at step 0

    def forward(self, x):
        """Computed in the input's dtype (bf16 in the DiT): the Kronecker product runs as thousands of small batched
        matmuls, which in fp32 made LoKR 3.7x slower than LoRA on Qwen (7.46 vs 2.04 s/step). The weights stay fp32
        for the optimizer; autograd casts the gradients back."""
        return _lokr_forward_update(x, self.lokr_w1.to(x.dtype), self.lokr_w2.to(x.dtype), self.a, self.b, self.c,
                                    self.d)

    def delta(self):
        return torch.kron(self.lokr_w1.float(), self.lokr_w2.float())


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.adapters = nn.ModuleDict()
        self.scales = {}

    def add(self, name, rank, alpha, trainable, A=None, B=None, strength=1.0, trainable_dtype=torch.float32):
        a = LoRAFactor(self.base.in_features, rank, bias=False)
        b = LoRAFactor(rank, self.base.out_features, bias=False)
        if A is not None:
            a.weight.data.copy_(A)
            b.weight.data.copy_(B)
        else:
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
            nn.init.zeros_(b.weight)
        dev = getattr(self, "home", None) or self.base.weight.device   # a swapped block's base may sit on CPU
        dt = trainable_dtype if trainable else torch.bfloat16
        a.to(dev, dt).requires_grad_(trainable)
        b.to(dev, dt).requires_grad_(trainable)
        self.adapters[name] = nn.Sequential(a, b)
        self.scales[name] = alpha / rank * strength

    def add_lokr(self, name, trainable, factor=8, w1=None, w2=None, scale=1.0, trainable_dtype=torch.float32):
        ad = LoKR(self.base.in_features, self.base.out_features, factor, w1, w2)
        dev = getattr(self, "home", None) or self.base.weight.device
        ad.to(dev, trainable_dtype if trainable else torch.bfloat16).requires_grad_(trainable)
        self.adapters[name] = ad
        self.scales[name] = scale

    def add_loha(self, name, w1a, w1b, w2a, w2b):
        ad = LoHa(w1a, w1b, w2a, w2b)
        dev = getattr(self, "home", None) or self.base.weight.device
        ad.to(dev, torch.bfloat16).requires_grad_(False)
        self.adapters[name] = ad
        self.scales[name] = 1.0

    def forward(self, x):
        out = self.base(x)
        for n, ad in self.adapters.items():
            s = self.scales.get(n, 0.0)
            if s:
                if isinstance(ad, (LoKR, LoHa)):
                    out = out + (s * ad(x)).to(out.dtype)
                    continue
                lx = ad(x.to(ad[0].weight.dtype))
                if lx.dtype == out.dtype:
                    # a frozen adapter in the model's dtype: ONE fused add, the strength formed in fp32 and the sum
                    # rounded once - the old loaders' epilogue, so a preview at strength 0.75 matches them (Fizgig 7.0.1)
                    out = torch.add(out, lx, alpha=float(s))
                else:                                # the fp32 trainable adapter: as before
                    out = out + (s * lx).to(out.dtype)
        return out


class LoRAConvFactor(nn.Conv2d):
    """A Conv2d adapter matrix. Its own class for the same reason as LoRAFactor (offloaders stream base modules)."""


class LoRAConv2d(nn.Module):
    """A Conv2d with adapters: base(x) + sum_i s_i * up_i(down_i(x)), down = the base's kernel / stride / padding /
    dilation with `rank` output channels, up = 1x1 (Fizgig's kohya LoRAModule layout for Conv2d, which ComfyUI and
    A1111 load as 4-D lora_down / lora_up weights)."""

    def __init__(self, base: nn.Conv2d):
        super().__init__()
        self.base = base
        self.adapters = nn.ModuleDict()
        self.scales = {}

    def fits(self, A, B) -> bool:
        k = tuple(self.base.kernel_size)
        return (A.dim() == 4 and B.dim() == 4 and A.shape[1] == self.base.in_channels and tuple(A.shape[2:]) == k
                and B.shape[0] == self.base.out_channels and tuple(B.shape[2:]) == (1, 1) and B.shape[1] == A.shape[0])

    def add(self, name, rank, alpha, trainable, A=None, B=None, strength=1.0):
        c = self.base
        a = LoRAConvFactor(c.in_channels, rank, c.kernel_size, c.stride, c.padding, c.dilation, bias=False)
        b = LoRAConvFactor(rank, c.out_channels, 1, bias=False)
        if A is not None:
            a.weight.data.copy_(A)
            b.weight.data.copy_(B)
        else:
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
            nn.init.zeros_(b.weight)
        dev = getattr(self, "home", None) or c.weight.device
        dt = torch.float32 if trainable else torch.bfloat16
        a.to(dev, dt).requires_grad_(trainable)
        b.to(dev, dt).requires_grad_(trainable)
        self.adapters[name] = nn.Sequential(a, b)
        self.scales[name] = alpha / rank * strength

    def forward(self, x):
        out = self.base(x)
        for n, ad in self.adapters.items():
            s = self.scales.get(n, 0.0)
            if s:
                out = out + (s * ad(x.to(ad[0].weight.dtype))).to(out.dtype)
        return out


def _fits_linear(w, A, B) -> bool:
    return A.dim() == 2 and B.dim() == 2 and A.shape[1] == w.base.in_features and B.shape[0] == w.base.out_features


class FamilyLoRA:
    """A model's adapter set. `wrapped` maps the module name (relative to the model) -> LoRALinear / LoRAConv2d."""

    def __init__(self, dit, driver, device=None):
        """device: where adapters live. Needed when block swap has parked some base weights on the CPU; by default
        each adapter follows its base weight."""
        self.dit = dit
        self.driver = driver
        self.device = torch.device(device) if device is not None else None
        self.desc = driver.description
        self.targets = set(driver.lora_target_names(dit))
        self.linears = {n for n, m in dit.named_modules() if isinstance(m, (nn.Linear, nn.Conv2d))
                        and not isinstance(m, (LoRAFactor, LoRAConvFactor))}
        self._flat = {n.replace(".", "_"): n for n in self.linears}
        # a family whose file key names differ from its module paths (SDXL: LDM names) also reads those
        self._flat.update({driver.lora_key_name(n).replace(".", "_"): n for n in self.linears})
        self.wrapped = {}
        for full in sorted(self.targets):
            self._wrap(full)
        if not self.wrapped:
            raise RuntimeError(f"{self.desc.display_name}: none of the driver's LoRA targets exist in this model")
        # per frozen adapter: {"alpha_rank": {module: alpha/rank}, "load": float, "on": bool,
        #                      "block_mult": {block_id: float}, "block_on": {block_id: bool}}
        self._frozen = {}

    def _stem(self, full, lokr=False):
        """Key stem of a wrapped module in the family's file format: `<file_prefix><dotted path>`, or for kohya families
        `lora_unet_<path with dots as underscores>` (a LoKR: `diffusion_model.<dotted path>`, the LyCORIS standard)."""
        f = self.desc.lora
        if f.kohya:
            return (f"diffusion_model.{full}" if lokr and not f.lokr_kohya_stems
                    else f"{f.file_prefix}{self.driver.lora_key_name(full).replace('.', '_')}")
        return f"{f.file_prefix}{full}"

    def _keys(self, full):
        """(down key, up key, alpha key) of any wrapped module, in the family's file format. Built from the module
        path alone, so modules outside the blocks never need a block id."""
        f = self.desc.lora
        stem = self._stem(full)
        return f"{stem}.{f.down}.weight", f"{stem}.{f.up}.weight", f.alpha_key.format(prefix=stem)

    def _wrap(self, full):
        """Wrap one Linear (or a groups=1 Conv2d) by dotted name (targets at init; frozen files may reach beyond them, e.g. a speed LoRA
        that also patches the modulation / timestep layers). Returns the LoRALinear or None."""
        if full in self.wrapped:
            return self.wrapped[full]
        parent_name, _, leaf = full.rpartition(".")
        parent = self.dit.get_submodule(parent_name) if parent_name else self.dit
        child = getattr(parent, leaf, None)
        if isinstance(child, nn.Conv2d) and child.groups == 1:
            w = LoRAConv2d(child)
        elif isinstance(child, nn.Linear):
            w = LoRALinear(child)
        else:
            return None
        if self.device is not None:
            w.home = self.device
        setattr(parent, leaf, w)
        self.wrapped[full] = w
        return w

    # ---- trainable ------------------------------------------------------------------------------
    def add_trainable(self, rank, alpha, blocks=None, kind="lora", factor=8):
        """blocks: optional set of block ids to train (None = every target). kind "lokr": a Kronecker adapter per
        Linear (w1 about factor x factor, full w2); rank / alpha do not apply to it. The adapters are fp32 unless the
        description's trainable_dtype says otherwise (H3's old trainer trained its LoRA in bf16, Fizgig 7.0.1)."""
        tdt = {"bf16": torch.bfloat16}.get(getattr(self.desc, "trainable_dtype", "fp32"), torch.float32)
        skipped_conv = 0
        for full, w in self.wrapped.items():
            if full not in self.targets:
                continue                    # extra Linears wrapped for a frozen file are never trained
            if blocks is None or self.driver.block_of(full) in blocks:
                if kind == "lokr":
                    if isinstance(w, LoRAConv2d):
                        skipped_conv += 1
                        continue
                    w.add_lokr(TRAINABLE, True, factor, trainable_dtype=tdt)
                elif isinstance(w, LoRAConv2d):
                    w.add(TRAINABLE, rank, alpha, True)
                else:
                    w.add(TRAINABLE, rank, alpha, True, trainable_dtype=tdt)
        if skipped_conv:
            logger.info(f"[lokr] {skipped_conv} Conv2d target(s) left untrained: LoKR adapters are Linear-only")
        self.rank, self.alpha, self.kind, self.factor = rank, alpha, kind, factor

    def trainable_modules(self):
        return nn.ModuleList([w.adapters[TRAINABLE] for w in self.wrapped.values() if TRAINABLE in w.adapters])

    def parameters(self):
        return [p for m in self.trainable_modules() for p in m.parameters()]

    # ---- reading LoRA files in any common layout ------------------------------------------------
    def _module_for(self, stem):
        """A file's module stem (dotted, prefixed, or kohya-flattened) -> a Linear name in this model, or None."""
        if stem.startswith("lora_unet_"):
            flats = [stem[len("lora_unet_"):]]
        else:
            flats = []
            for p in _PREFIXES:
                if p and not stem.startswith(p):
                    continue
                name = stem[len(p):]
                if name in self.linears:
                    return name
                flats.append(name.replace(".", "_"))
        for flat in flats:
            if flat in self._flat:
                return self._flat[flat]
        for flat in flats:                  # another trainer's naming (the driver's renames)
            alias = self.driver.alias_flat(flat)
            if alias in self._flat:
                return self._flat[alias]
        return None

    def read_file(self, path):
        """-> {module name: entry} for every Linear the file adapts in this model. A LoRA entry is
        ("lora", A, B, scale) with scale = alpha / rank; a LoKR entry is ("lokr", w1, w2, scale) with low-rank factors
        multiplied out and the LyCORIS scale rule (lycoris_scale_from_keys). A LoHa entry is
        ("loha", (w1_a, w1_b), (w2_a, w2_b), scale), the LyCORIS scale rule as for LoKR."""
        from safetensors.torch import load_file
        sd = self.driver.convert_lora_state_dict(load_file(path))
        out = {}
        for key in sd:
            m = re.match(r"(.+)\.hada_w1_a$", key)
            if m:
                stem = m.group(1)
                full = self._module_for(stem)
                if full is None:
                    continue
                keys = {k[len(stem) + 1:]: v for k, v in sd.items() if k.startswith(stem + ".")}
                out[full] = ("loha", (keys["hada_w1_a"], keys["hada_w1_b"]), (keys["hada_w2_a"], keys["hada_w2_b"]),
                             lycoris_scale_from_keys(keys))
                continue
            m = re.match(r"(.+)\.lokr_w1(_a)?$", key)
            if m:
                stem = m.group(1)
                full = self._module_for(stem)
                if full is None:
                    continue
                keys = {k[len(stem) + 1:]: v for k, v in sd.items() if k.startswith(stem + ".")}
                w1 = keys["lokr_w1"] if "lokr_w1" in keys else keys["lokr_w1_a"].float() @ keys["lokr_w1_b"].float()
                w2 = keys["lokr_w2"] if "lokr_w2" in keys else keys["lokr_w2_a"].float() @ keys["lokr_w2_b"].float()
                out[full] = ("lokr", w1, w2, lycoris_scale_from_keys(keys))
                continue
            m = re.match(r"(.+)\.(lora_A|lora_down|lora\.down)\.weight$", key)
            if not m:
                continue
            stem, down = m.group(1), m.group(2)
            up = {"lora_A": "lora_B", "lora_down": "lora_up", "lora.down": "lora.up"}[down]
            if f"{stem}.{up}.weight" not in sd:
                continue
            full = self._module_for(stem)
            if full is None:
                continue
            A, B = sd[key], sd[f"{stem}.{up}.weight"]
            alpha = sd.get(f"{stem}.alpha")
            out[full] = ("lora", A, B, (float(alpha.item()) if alpha is not None else float(A.shape[0])) / A.shape[0])
        return out

    # ---- frozen adapters ------------------------------------------------------------------------
    def read_bias_deltas(self, path):
        """[(bias Parameter, delta on CPU)] for a file's `<module>.diff_b` keys (ComfyUI's bias deltas: the Krea 2
        Turbo LoRA carries them on its input / output layers). A key with no matching bias is skipped with a warning."""
        from safetensors import safe_open
        out = []
        with safe_open(path, framework="pt") as f:
            for key in sorted(k for k in f.keys() if k.endswith(".diff_b")):
                mod = key[:-len(".diff_b")]
                for p in _PREFIXES:
                    if p and mod.startswith(p):
                        mod = mod[len(p):]
                        break
                delta = f.get_tensor(key)
                try:
                    m = self.dit.get_submodule(mod)
                    bias = getattr(getattr(m, "base", m), "bias", None)
                    if bias is None or tuple(bias.shape) != tuple(delta.shape):
                        raise AttributeError("no bias or a different shape")
                except AttributeError as e:
                    logger.warning(f"[lora] {key}: no matching bias in the model ({e}) - skipped")
                    continue
                out.append((bias, delta.clone()))
        return out

    def add_file(self, path, name, strength=1.0):
        """Attach a LoRA file frozen under `name` on every Linear it adapts. Returns the number of Linears covered
        (0 = nothing in the file matches this model)."""
        n = 0
        ar = {}
        for full, (kind, P, Q, scale) in self.read_file(path).items():
            w = self._wrap(full)
            if w is None:
                continue
            if kind == "loha":
                if isinstance(w, LoRAConv2d) or P[0].shape[0] != w.base.out_features \
                        or P[1].shape[1] != w.base.in_features:
                    continue                          # LoHa on convs is not read
                w.add_loha(name, P[0], P[1], Q[0], Q[1])
            elif kind == "lokr":
                if isinstance(w, LoRAConv2d) or P.shape[0] * Q.shape[0] != w.base.out_features \
                        or P.shape[1] * Q.shape[1] != w.base.in_features:
                    continue
                w.add_lokr(name, False, w1=P, w2=Q)
            else:
                if not (w.fits(P, Q) if isinstance(w, LoRAConv2d) else _fits_linear(w, P, Q)):
                    continue
                w.add(name, P.shape[0], P.shape[0], False, P, Q)
            ar[full] = scale
            n += 1
        self._frozen[name] = {"alpha_rank": ar, "load": float(strength), "on": True, "block_mult": {},
                              "block_on": {}, "outside_on": True, "path": path,
                              "bias": self.read_bias_deltas(path), "bias_snap": None}
        self._apply(name)
        return n

    def has(self, name):
        return name in self._frozen

    def _apply(self, name):
        st = self._frozen[name]
        for full, ar in st["alpha_rank"].items():
            b = self.driver.block_of(full)
            on = st["on"] and (st["outside_on"] if b is None else st["block_on"].get(b, True))
            mult = 1.0 if b is None else st["block_mult"].get(b, 1.0)
            self.wrapped[full].scales[name] = ar * st["load"] * mult if on else 0.0
        bias = st.get("bias")
        if bias:                    # bias deltas follow the adapter's on / off and load strength, restored exactly
            if st["bias_snap"] is not None:
                for (b, _), snap in zip(bias, st["bias_snap"]):
                    b.data.copy_(snap)
                st["bias_snap"] = None
            if st["on"] and st["load"]:
                st["bias_snap"] = [b.detach().clone() for b, _ in bias]
                for b, d in bias:
                    b.data.add_(d.to(device=b.device, dtype=b.dtype), alpha=st["load"])

    def set_enabled(self, name, enabled: bool):
        """Switch a frozen adapter on (with its load strength and block settings) or fully off - every module,
        inside or outside the block map."""
        if name in self._frozen:
            self._frozen[name]["on"] = bool(enabled)
            self._apply(name)

    def set_strength(self, name, strength):
        """The adapter's whole-file (load) strength."""
        self._frozen[name]["load"] = float(strength)
        self._apply(name)

    def set_outside(self, name, enabled: bool):
        """On/off for the adapter's modules outside the block map (no slider reaches them)."""
        self._frozen[name]["outside_on"] = bool(enabled)
        self._apply(name)

    def set_blocks(self, name, mult=None, enabled=None):
        """Per-block controls for one adapter: mult {block_id: strength}, enabled {block_id: bool}. Blocks not
        named keep their current values."""
        st = self._frozen[name]
        st["block_mult"].update(mult or {})
        st["block_on"].update(enabled or {})
        self._apply(name)

    def adapter_blocks(self, name):
        """Block ids the adapter touches (the workbench greys out the rest)."""
        return {b for b in (self.driver.block_of(f) for f in self._frozen.get(name, {}).get("alpha_rank", {}))
                if b is not None}

    @torch.no_grad()
    def swap_file(self, name, path):
        """Replace an adapter's weights with another file's (epoch scrubbing). In place when the file adapts the
        same modules at the same ranks; otherwise the adapter is rebuilt. Block settings and strength carry over.
        Returns the number of Linears covered."""
        st = self._frozen[name]
        new = self.read_file(path)

        def params(full):
            ad = self.wrapped[full].adapters[name]
            return (ad.lokr_w1, ad.lokr_w2) if isinstance(ad, LoKR) else (ad[0].weight, ad[1].weight)
        same = set(new) == set(st["alpha_rank"]) and not any(
            new[f][0] == "loha" or isinstance(self.wrapped[f].adapters[name], LoHa) for f in new) and all(
            (new[f][0] == "lokr") == isinstance(self.wrapped[f].adapters[name], LoKR)
            and params(f)[0].shape == new[f][1].shape and params(f)[1].shape == new[f][2].shape for f in new)
        if same:
            for full, (_kind, P, Q, scale) in new.items():
                a, b = params(full)
                a.copy_(P.to(a.dtype))
                b.copy_(Q.to(b.dtype))
                st["alpha_rank"][full] = scale
            st["path"] = path
            self._apply(name)
            return len(new)
        keep = {k: st[k] for k in ("load", "on", "block_mult", "block_on", "outside_on")}
        self.remove(name)
        n = self.add_file(path, name, keep["load"])
        self._frozen[name].update(keep)
        self._apply(name)
        return n

    def remove(self, name):
        """Drop a frozen adapter's weights entirely (and put its bias deltas back)."""
        st = self._frozen.get(name)
        if st and st.get("bias_snap") is not None:
            for (b, _), snap in zip(st["bias"], st["bias_snap"]):
                b.data.copy_(snap)
        for w in self.wrapped.values():
            if name in w.adapters:
                del w.adapters[name]
                w.scales.pop(name, None)
        self._frozen.pop(name, None)

    def move_adapter(self, name, device):
        """Move one frozen adapter's weights (e.g. a speed LoRA parked on CPU between previews)."""
        for w in self.wrapped.values():
            if name in w.adapters:
                w.adapters[name].to(device)

    # ---- bake: what is live now, as one standard LoRA in the family format ------------------------
    @torch.no_grad()
    def bake(self, names, dtype=torch.bfloat16):
        """One LoRA state dict equal to the named adapters as currently set (strengths, block settings, on/off),
        in the family's key format. Several adapters on a module are rank-concatenated with their scales folded
        into the up weights, so alpha = total rank (scale 1). Returns (state_dict, {module: rank})."""
        sd, ranks = {}, {}
        for full, w in self.wrapped.items():
            live = [(n, w.scales[n]) for n in names if n in w.adapters and w.scales.get(n, 0.0)]
            if not live:
                continue
            if len(live) == 1 and isinstance(w.adapters[live[0][0]], LoKR):
                # a lone LoKR stays a LoKR: the scale folds into w2 (alpha 1, full matrices -> scale 1 everywhere)
                ad, s1 = w.adapters[live[0][0]], live[0][1]
                stem = self._stem(full, lokr=True)
                sd[f"{stem}.lokr_w1"] = ad.lokr_w1.detach().to("cpu", dtype).contiguous()
                sd[f"{stem}.lokr_w2"] = (ad.lokr_w2.detach().float() * s1).to("cpu", dtype).contiguous()
                sd[f"{stem}.alpha"] = torch.tensor(1.0)
                ranks[full] = 0
                continue
            As, Bs = [], []
            for n, sc in live:
                ad = w.adapters[n]
                if isinstance(ad, (LoKR, LoHa)):  # Kronecker / Hadamard delta: SVD to a rank <= 64 LoRA
                    U, S, Vh = torch.linalg.svd(ad.delta(), full_matrices=False)
                    k = min(64, S.numel())
                    root = S[:k].sqrt()
                    As.append(root[:, None] * Vh[:k])
                    Bs.append(U[:, :k] * root * sc)
                    continue
                a, b = ad
                As.append(a.weight.float())
                Bs.append(b.weight.float() * sc)
            A, B = torch.cat(As, 0), torch.cat(Bs, 1)
            ka, kb, kal = self._keys(full)
            sd[ka] = A.to("cpu", dtype).contiguous()
            sd[kb] = B.to("cpu", dtype).contiguous()
            sd[kal] = torch.tensor(float(A.shape[0]))
            ranks[full] = A.shape[0]
        return sd, ranks

    # ---- save / load the trainable adapter (approved key format) --------------------------------
    def state_dict(self, dtype=torch.bfloat16):
        sd = {}
        for full, w in self.wrapped.items():
            if TRAINABLE in w.adapters and isinstance(w.adapters[TRAINABLE], LoKR):
                ad = w.adapters[TRAINABLE]
                stem = self._stem(full, lokr=True)
                sd[f"{stem}.lokr_w1"] = ad.lokr_w1.detach().to("cpu", dtype).contiguous()
                sd[f"{stem}.lokr_w2"] = ad.lokr_w2.detach().to("cpu", dtype).contiguous()
                sd[f"{stem}.alpha"] = torch.tensor(1.0)     # full matrices: LyCORIS scale = alpha = 1
            elif TRAINABLE in w.adapters:
                a, b = w.adapters[TRAINABLE]
                ka, kb, kal = self._keys(full)
                sd[ka] = a.weight.detach().to("cpu", dtype).contiguous()
                sd[kb] = b.weight.detach().to("cpu", dtype).contiguous()
                sd[kal] = torch.tensor(float(self.alpha))
        return sd

    def save(self, path, metadata=None, dtype=torch.bfloat16):
        from safetensors.torch import save_file
        save_file(self.state_dict(dtype), path, metadata={k: str(v) for k, v in (metadata or {}).items()})

    @torch.no_grad()
    def load_trainable(self, path):
        """Restore the trainable adapter from a file saved by save(). Returns modules matched."""
        from safetensors.torch import load_file
        sd = load_file(path)
        n = 0
        for full, w in self.wrapped.items():
            if TRAINABLE not in w.adapters:
                continue                # frozen-only wraps (training adapter, speed LoRA extras) hold nothing to load
            ad = w.adapters[TRAINABLE]
            if isinstance(ad, LoKR):
                stem = self._stem(full, lokr=True)
                if f"{stem}.lokr_w1" in sd:
                    ad.lokr_w1.copy_(sd[f"{stem}.lokr_w1"].to(ad.lokr_w1.dtype))
                    ad.lokr_w2.copy_(sd[f"{stem}.lokr_w2"].to(ad.lokr_w2.dtype))
                    n += 1
                continue
            ka, kb, _ = self._keys(full)
            if ka in sd:
                a, b = w.adapters[TRAINABLE]
                a.weight.copy_(sd[ka].to(a.weight.dtype))
                b.weight.copy_(sd[kb].to(b.weight.dtype))
                n += 1
        return n
