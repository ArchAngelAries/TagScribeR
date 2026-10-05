# SDXL VAE facts and loader (original code).
#   [sdxl-vae]  stabilityai/stable-diffusion-xl-base-1.0 vae/config.json: AutoencoderKL, 4 latent channels, 8x, scaling_factor
#               0.13025, no shift. Vendored so loading is offline.
#   [diffusers-ldm] diffusers loaders/single_file_utils.py convert_ldm_vae_checkpoint converts the original layout
#               (`first_stage_model.*`, or bare `encoder.down.*` in a standalone VAE file) to diffusers names.
# A standalone file may also be in diffusers layout (`encoder.down_blocks.*`, e.g. madebyollin/sdxl-vae-fp16-fix's
# diffusion_pytorch_model.safetensors); both are accepted.
# Precision: the stock SDXL VAE overflows to NaN in fp16, so it is NEVER run in fp16 here. Encoding (the cache stage, nothing
# else on the GPU) runs in float32; decoding previews runs in bf16 on a GPU (the training UNet is resident) and fp32 on CPU.
from __future__ import annotations

import torch

SCALING_FACTOR = 0.13025
SDXL_VAE_CONFIG = {
    "act_fn": "silu", "block_out_channels": [128, 256, 512, 512], "down_block_types": ["DownEncoderBlock2D"] * 4,
    "in_channels": 3, "latent_channels": 4, "layers_per_block": 2, "norm_num_groups": 32, "out_channels": 3,
    "sample_size": 1024, "scaling_factor": SCALING_FACTOR, "up_block_types": ["UpDecoderBlock2D"] * 4,
    "force_upcast": True, "use_quant_conv": True, "use_post_quant_conv": True, "mid_block_add_attention": True,
}


def load_vae(path: str, device="cpu", config: dict = None, dtype=torch.float32):
    import copy

    from diffusers import AutoencoderKL
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
    from safetensors.torch import load_file
    cfg = copy.deepcopy(config or SDXL_VAE_CONFIG)
    sd = load_file(path)
    sd = {(k[len("first_stage_model."):] if k.startswith("first_stage_model.") else k): v for k, v in sd.items()
          if k.startswith("first_stage_model.") or not k.startswith(("model.", "conditioner.", "cond_stage_model."))}
    if any(k.startswith("encoder.down_blocks") for k in sd):
        state = sd                                                    # already diffusers names
    elif any(k.startswith("encoder.down.") for k in sd):
        state = convert_ldm_vae_checkpoint(sd, cfg)
    else:
        raise ValueError(f"{path} has no SDXL VAE tensors (`first_stage_model.*` or `encoder.*`).")
    state = {k: v.to(dtype) for k, v in state.items()}
    vae = AutoencoderKL.from_config(cfg)
    r = vae.load_state_dict(state, strict=False)
    if r.missing_keys or r.unexpected_keys:
        raise ValueError(f"{path} does not match the SDXL VAE: missing {r.missing_keys[:3]}, unexpected "
                         f"{r.unexpected_keys[:3]}")
    return vae.to(device=device, dtype=dtype).eval().requires_grad_(False)
