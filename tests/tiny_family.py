"""A tiny, randomly initialised model family for CPU smoke tests of the training layer.

Nothing here is a real model: a 2-block "DiT" with Conv2d and Linear layers (so Conv2d LoRA is exercised), a toy
VAE (8x average-pool + a fixed 1x1 conv), and a hashing "text encoder" with variable-length output (batch 1, like
Qwen Image) or fixed-length output (the batching variant). Training on it takes seconds and proves the plumbing:
caching, the loop, adapters, Adaptive LR, EMA, the loss watch, previews, saving, pause and resume.
"""
from __future__ import annotations

import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.description import FamilyDescription, LoRAFormat, ModelFile, SamplingSettings
from training.driver import FamilyDriver

DIM = 8


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, DIM, 3, padding=1)
        self.lin = nn.Linear(DIM, DIM)
        self.out = nn.Conv2d(DIM, 4, 1)

    def forward(self, x, ctx):
        h = F.silu(self.conv(x))                                    # (B, D, h, w)
        h = h + self.lin(ctx)[:, :, None, None]                     # text conditioning, broadcast
        return x + self.out(h)


class TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctx = nn.Linear(DIM, DIM)
        self.t_embed = nn.Linear(1, DIM)
        self.blocks = nn.ModuleList([TinyBlock(), TinyBlock()])

    def forward(self, x, cond, t):
        ctx = self.ctx(cond.mean(dim=1)) + self.t_embed(t.view(-1, 1).to(x.dtype))
        for b in self.blocks:
            x = b(x, ctx)
        return x


class TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.enc = nn.Conv2d(3, 4, 1)
        self.dec = nn.Conv2d(4, 3, 1)
        with torch.no_grad():
            for m in (self.enc, self.dec):
                m.weight.copy_(torch.randn(m.weight.shape, generator=g) * 0.5)
                m.bias.zero_()


class TinyTE:
    def __init__(self, fixed_len=None):
        self.fixed_len = fixed_len

    def encode(self, caption: str) -> torch.Tensor:
        words = caption.replace(",", " ").split() or [""]
        if self.fixed_len:
            words = (words + [""] * self.fixed_len)[: self.fixed_len]
        rows = []
        for w in words:
            g = torch.Generator().manual_seed(zlib.crc32(w.encode("utf-8")))
            rows.append(torch.randn(DIM, generator=g))
        return torch.stack(rows)


class TinyDriver(FamilyDriver):
    fixed_len = None

    def load_dit(self, path, device):
        dit = TinyDiT()
        try:
            from safetensors.torch import load_file
            dit.load_state_dict(load_file(path))
        except Exception:
            torch.manual_seed(0)
            dit = TinyDiT()
        return dit.to(device).eval().requires_grad_(False)

    def load_vae(self, path, device):
        return TinyVAE().to(device)

    def load_text_encoder(self, path, device):
        return TinyTE(self.fixed_len)

    def unload_text_encoder(self, te):
        pass

    def enable_gradient_checkpointing(self, dit, on=True):
        pass

    @torch.no_grad()
    def encode_images(self, vae, images):
        x = torch.stack([torch.from_numpy(np.ascontiguousarray(a[..., :3])) for a in images])
        x = x.permute(0, 3, 1, 2).float().div(127.5).sub(1.0)
        z = vae.enc(F.avg_pool2d(x, 8))
        return [zi.cpu() for zi in z]

    @torch.no_grad()
    def encode_text(self, te, captions):
        return [{"hidden_states": te.encode(c)} for c in captions]

    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None):
        x0 = latents.float()
        noise = torch.randn(x0.shape, generator=generator).to(x0.device)
        t = min_t + (max_t - min_t) * torch.sigmoid(torch.randn(1, generator=generator)).item()
        xt = (1 - t) * x0 + t * noise
        tt = torch.full((x0.shape[0],), t, device=x0.device)
        pred = dit(xt, cond["hidden_states"].float(), tt)
        return F.mse_loss(pred.float(), noise - x0), {"t": t}

    def initial_noise(self, seed, width, height):
        g = torch.Generator().manual_seed(int(seed))
        return torch.randn(1, 4, height // 8, width // 8, generator=g)

    @torch.no_grad()
    def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, sigmas=None, options=(),
                 noise=None, on_step=None, refs=None):
        device = next(dit.parameters()).device
        x = (noise if noise is not None else self.initial_noise(seed, width, height)).to(device)
        hs = cond["hidden_states"]
        hs = hs[None] if hs.dim() == 2 else hs
        sig = torch.linspace(1.0, 0.0, steps + 1)
        for i in range(steps):
            v = dit(x, hs.float().to(device), torch.full((1,), float(sig[i]), device=device))
            x = x + (sig[i + 1] - sig[i]) * v
        return x

    @torch.no_grad()
    def decode(self, vae, latents, width, height):
        from PIL import Image
        img = torch.tanh(vae.dec(F.interpolate(latents.float(), size=(height, width))))[0]
        return Image.fromarray(((img.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8))


class TinyBatchDriver(TinyDriver):
    fixed_len = 6
    supports_batching = True


def _preset(rank, adaptive=None, epochs=2):
    lo, hi = adaptive or ("2e-4", "4e-4")
    return {"NETWORK_DIM": rank, "NETWORK_ALPHA": rank, "NETWORK_TYPE": "LoRA (standard)", "LEARNING_RATE": 1e-3,
            "MAX_TRAIN_EPOCHS": epochs, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42, "ADAPTIVE_LR": adaptive is not None,
            "ADAPTIVE_LR_MIN": lo, "ADAPTIVE_LR_MAX": hi, "OPTIMIZER_TYPE": "adamw", "GRADIENT_ACCUMULATION": 1,
            "MAX_GRAD_NORM": 1.0, "DATASET_MEGAPIXELS": "0.25", "FAMILY_EMA": "0.98 (recommended)",
            "KREA2_LOSS_WATCH": True}


def _desc(key, arch, driver):
    return FamilyDescription(
        key=key, arch_id=arch, display_name=f"Tiny test family ({key})", gui_label=key, lora_name_suffix=arch,
        model_files=(ModelFile(f"{arch}_dit", "Tiny DiT", True, role="dit"),
                     ModelFile(f"{arch}_vae", "Tiny VAE", True, role="vae"),
                     ModelFile(f"{arch}_te", "Tiny text encoder", True, role="text_encoder")),
        latent_channels=4, spatial_factor=8, bucket_step=32, n_blocks=2, block_prefix="blocks",
        lora=LoRAFormat(key_template="diffusion_model.blocks.{block}.{module}.{ab}.weight", down="lora_down",
                        up="lora_up", block_modules=("conv", "lin", "out"), file_prefix="diffusion_model.",
                        source="test"),
        driver=f"tests.tiny_family:{driver}", modelspec_arch="Tiny-Test", implementation="https://example.invalid",
        precisions=("bf16",), optimizers=("adamw", "adamw8bit"), network_types=("lora", "lokr"),
        ema_default="0.98", sampling=(SamplingSettings("tiny", steps=3, cfg=1.0, source="test"),),
        preview_steps=3, preview_width=64, preview_height=64,
        presets=(("Tiny Fast (rank 4, adaptive)", _preset(4, adaptive=("2e-4", "4e-4"))),
                 ("Tiny Flat (rank 8)", _preset(8))))


TINY = _desc("_tiny", "tiny", "TinyDriver")
TINY_BATCH = _desc("_tinyb", "tinyb", "TinyBatchDriver")


def register():
    from training import registry
    for d in (TINY, TINY_BATCH):
        if registry.get(d.key) is None:
            registry.register(d)
    return TINY


def make_dataset(folder, n=4, captions=None, sizes=None):
    """Write n small images with captions into `folder` (a scratch copy, never a user dataset)."""
    from PIL import Image
    import os
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(0)
    sizes = sizes or [(96, 96), (128, 96), (96, 128), (112, 112)]
    for i in range(n):
        w, h = sizes[i % len(sizes)]
        arr = (rng.random((h, w, 3)) * 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(folder, f"img{i:02d}.png"))
        cap = (captions[i] if captions else f"tok, photo {i}, red, blue, green, small")
        with open(os.path.join(folder, f"img{i:02d}.txt"), "w", encoding="utf-8") as f:
            f.write(cap)
    return folder


def write_models(folder, arch="tiny"):
    """Dummy model files (the tiny driver builds its own weights; the DiT file holds a real state dict)."""
    import os

    from safetensors.torch import save_file
    os.makedirs(folder, exist_ok=True)
    torch.manual_seed(0)
    dit = os.path.join(folder, "dit.safetensors")
    save_file({k: v.contiguous() for k, v in TinyDiT().state_dict().items()}, dit)
    paths = {f"{arch}_dit": dit}
    for role in ("vae", "te"):
        p = os.path.join(folder, f"{role}.bin")
        with open(p, "wb") as f:
            f.write(b"\0")
        paths[f"{arch}_{role}"] = p
    return paths


