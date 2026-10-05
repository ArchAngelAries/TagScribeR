# SDXL UNet facts and loader for TagScribeR's training layer (original code; no Fizgig counterpart - Fizgig has no SDXL).
#
# Sources of every architecture fact (cited where used):
#   [diffusers-config]  stabilityai/stable-diffusion-xl-base-1.0 unet/config.json (Hugging Face), the config diffusers
#                       builds a UNet2DConditionModel from. Vendored below so loading works offline.
#   [diffusers-ldm]     diffusers (Apache-2.0) loaders/single_file_utils.py convert_ldm_unet_checkpoint: converts the
#                       original (LDM / SGM) checkpoint keys `model.diffusion_model.*` to diffusers names. Used as a
#                       library call, not copied.
#   [comfyui-keys]      ComfyUI (GPL-3.0) comfy/utils.py unet_to_diffusers + comfy/lora.py model_lora_keys_unet, read ONLY
#                       as a reference for which key NAMES a LoRA must carry (`lora_unet_` + the original LDM module path
#                       with dots as underscores). No ComfyUI code is used here.
#   [kohya-targets]     kohya-ss/sd-scripts networks/lora.py: UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"]
#                       (Linear layers of the transformer blocks), UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 =
#                       ["ResnetBlock2D", "Downsample2D", "Upsample2D"] (3x3 convolutions, only with conv_dim).
"""The SDXL UNet: vendored config, the diffusers <-> original (LDM) name tables, and the offline single-file loader."""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Optional

# [diffusers-config] stabilityai/stable-diffusion-xl-base-1.0 / unet / config.json (the three variants Pony, Illustrious
# and NoobAI are SDXL fine-tunes with this exact architecture).
SDXL_UNET_CONFIG = {
    "act_fn": "silu", "addition_embed_type": "text_time", "addition_embed_type_num_heads": 64,
    "addition_time_embed_dim": 256, "attention_head_dim": [5, 10, 20], "block_out_channels": [320, 640, 1280],
    "center_input_sample": False, "class_embed_type": None, "class_embeddings_concat": False, "conv_in_kernel": 3,
    "conv_out_kernel": 3, "cross_attention_dim": 2048, "cross_attention_norm": None,
    "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"], "downsample_padding": 1,
    "dual_cross_attention": False, "encoder_hid_dim": None, "encoder_hid_dim_type": None, "flip_sin_to_cos": True,
    "freq_shift": 0, "in_channels": 4, "layers_per_block": 2, "mid_block_only_cross_attention": None,
    "mid_block_scale_factor": 1, "mid_block_type": "UNetMidBlock2DCrossAttn", "norm_eps": 1e-05,
    "norm_num_groups": 32, "num_attention_heads": None, "num_class_embeds": None, "only_cross_attention": False,
    "out_channels": 4, "projection_class_embeddings_input_dim": 2816, "resnet_out_scale_factor": 1.0,
    "resnet_skip_time_act": False, "resnet_time_scale_shift": "default", "sample_size": 128, "time_cond_proj_dim": None,
    "time_embedding_act_fn": None, "time_embedding_dim": None, "time_embedding_type": "positional",
    "timestep_post_act": None, "transformer_layers_per_block": [1, 2, 10], "up_block_types": [
        "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"], "upcast_attention": None,
    "use_linear_projection": True,
}

LDM_UNET_PREFIX = "model.diffusion_model."          # [diffusers-ldm] LDM_UNET_KEY

# the Linears inside every transformer block that a LoRA targets [kohya-targets]; names are identical in the diffusers
# and the original (LDM) layouts [comfyui-keys TRANSFORMER_BLOCKS]
_ATTN_LINEARS = ("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
                 "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0")
_FF_LINEARS = ("ff.net.0.proj", "ff.net.2")
# a resnet's modules, diffusers -> LDM [comfyui-keys UNET_MAP_RESNET]
_RESNET_RENAME = {"norm1": "in_layers.0", "conv1": "in_layers.2", "time_emb_proj": "emb_layers.1",
                  "norm2": "out_layers.0", "conv2": "out_layers.3", "conv_shortcut": "skip_connection"}
_RESNET_CONV3X3 = ("conv1", "conv2")                # LoCon targets of a resnet [kohya-targets]


@dataclass(frozen=True)
class Unit:
    """One module group of the UNet: a resnet, an attention (Transformer2DModel) or a sampler, with both its diffusers
    and its original (LDM) prefix and the block it belongs to (IN00..IN08, mid, OUT00..OUT08 in kohya's block terms)."""
    kind: str            # "resnet" | "attn" | "down" | "up"
    diffusers: str       # e.g. "down_blocks.1.attentions.0"
    ldm: str             # e.g. "input_blocks.4.1"
    block: str           # block id: "in_04", "mid", "out_00"
    depth: int = 0       # transformer blocks (attn only)


def _per_block(value, n):
    return list(value) if isinstance(value, (list, tuple)) else [int(value)] * n


def structure(config: dict) -> list:
    """Every Unit of a UNet2DConditionModel config, in the original network's order. [comfyui-keys unet_to_diffusers]
    walks the same layout: input_blocks.0 is conv_in, each down block adds `layers_per_block` (resnet [+ attention])
    blocks and a downsampler block; the middle block is resnet, attention, resnet; output blocks mirror the input
    side with one more layer per block and the upsampler inside the block's last output block."""
    n_down = len(config["down_block_types"])
    L = int(config["layers_per_block"])
    depth = _per_block(config.get("transformer_layers_per_block", 1), n_down)
    units = []
    n = 1
    for x, btype in enumerate(config["down_block_types"]):
        attn = "CrossAttn" in btype
        for i in range(L):
            units.append(Unit("resnet", f"down_blocks.{x}.resnets.{i}", f"input_blocks.{n}.0", f"in_{n:02d}"))
            if attn:
                units.append(Unit("attn", f"down_blocks.{x}.attentions.{i}", f"input_blocks.{n}.1", f"in_{n:02d}",
                                  depth[x]))
            n += 1
        if x != n_down - 1:
            units.append(Unit("down", f"down_blocks.{x}.downsamplers.0", f"input_blocks.{n}.0", f"in_{n:02d}"))
            n += 1
    units.append(Unit("resnet", "mid_block.resnets.0", "middle_block.0", "mid"))
    units.append(Unit("attn", "mid_block.attentions.0", "middle_block.1", "mid", depth[-1]))
    units.append(Unit("resnet", "mid_block.resnets.1", "middle_block.2", "mid"))
    n = 0
    rdepth = depth[::-1]
    for x, btype in enumerate(config["up_block_types"]):
        attn = "CrossAttn" in btype
        for i in range(L + 1):
            units.append(Unit("resnet", f"up_blocks.{x}.resnets.{i}", f"output_blocks.{n}.0", f"out_{n:02d}"))
            if attn:
                units.append(Unit("attn", f"up_blocks.{x}.attentions.{i}", f"output_blocks.{n}.1", f"out_{n:02d}",
                                  rdepth[x]))
            if i == L and x != len(config["up_block_types"]) - 1:
                units.append(Unit("up", f"up_blocks.{x}.upsamplers.0", f"output_blocks.{n}.{2 if attn else 1}",
                                  f"out_{n:02d}"))
            n += 1
    return units


def unit_modules(u: Unit, locon: bool) -> list:
    """(diffusers module name, LDM module name) of the LoRA targets inside one Unit. Always the transformer Linears
    [kohya-targets UNET_TARGET_REPLACE_MODULE]; with `locon` also the 3x3 convolutions of resnets and samplers
    [kohya-targets UNET_TARGET_REPLACE_MODULE_CONV2D_3X3]."""
    out = []
    if u.kind == "attn":
        for name in ("proj_in", "proj_out"):
            out.append((f"{u.diffusers}.{name}", f"{u.ldm}.{name}"))
        for t in range(u.depth):
            for m in _ATTN_LINEARS + _FF_LINEARS:
                out.append((f"{u.diffusers}.transformer_blocks.{t}.{m}", f"{u.ldm}.transformer_blocks.{t}.{m}"))
    elif locon and u.kind == "resnet":
        for m in _RESNET_CONV3X3:
            out.append((f"{u.diffusers}.{m}", f"{u.ldm}.{_RESNET_RENAME[m]}"))
    elif locon and u.kind == "down":
        out.append((f"{u.diffusers}.conv", f"{u.ldm}.op"))
    elif locon and u.kind == "up":
        out.append((f"{u.diffusers}.conv", f"{u.ldm}.conv"))
    return out


def ldm_module_table(config: dict, locon: bool = True) -> dict:
    """{diffusers module name: LDM module name} for every LoRA target module of the config (the kohya key stem is
    `lora_unet_` + the LDM name with dots as underscores)."""
    return {d: l for u in structure(config) for d, l in unit_modules(u, locon)}


def ldm_param_name(name: str, config: dict) -> str:
    """The original (LDM) name of ANY diffusers UNet parameter or module name (used to build test checkpoints and to
    cross-check the converter; training itself only needs the LoRA target names)."""
    basic = {"time_embedding.linear_1": "time_embed.0", "time_embedding.linear_2": "time_embed.2",
             "add_embedding.linear_1": "label_emb.0.0", "add_embedding.linear_2": "label_emb.0.2",
             "conv_in": "input_blocks.0.0", "conv_norm_out": "out.0", "conv_out": "out.2"}
    for k, v in basic.items():
        if name == k or name.startswith(k + "."):
            return v + name[len(k):]
    for u in structure(config):
        if name == u.diffusers or name.startswith(u.diffusers + "."):
            rest = name[len(u.diffusers) + 1:]
            if u.kind == "resnet":
                head, _, tail = rest.partition(".")
                return f"{u.ldm}.{_RESNET_RENAME[head]}.{tail}"
            if u.kind == "down":                       # downsamplers.0.conv.weight -> input_blocks.N.0.op.weight
                return f"{u.ldm}.op.{rest.partition('.')[2]}"
            if u.kind == "up":
                return f"{u.ldm}.conv.{rest.partition('.')[2]}"
            return f"{u.ldm}.{rest}"
    raise KeyError(name)


# ---- loading ----------------------------------------------------------------------------------------------
def read_prefixed(path: str, prefixes, dtype=None) -> dict:
    """Tensors of a safetensors file whose key starts with one of `prefixes` (read lazily: a 7 GB checkpoint is never
    loaded whole), optionally cast to `dtype`."""
    from safetensors import safe_open
    out = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if k.startswith(tuple(prefixes)):
                t = f.get_tensor(k)
                out[k] = t.to(dtype) if dtype is not None and t.is_floating_point() else t
    return out


def load_unet(path: str, device="cpu", config: Optional[dict] = None, dtype=None):
    """The UNet from a single-file SDXL checkpoint (`model.diffusion_model.*` keys), frozen, in `dtype` (bf16 by
    default). No Hugging Face access: the architecture comes from the vendored config. [diffusers-ldm]"""
    import torch
    from diffusers import UNet2DConditionModel
    from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint
    cfg = copy.deepcopy(config or SDXL_UNET_CONFIG)
    dtype = dtype or torch.bfloat16
    ldm = read_prefixed(path, (LDM_UNET_PREFIX,), dtype)
    if not ldm:
        raise ValueError(f"{path} has no `{LDM_UNET_PREFIX}*` tensors - it is not a single-file SDXL checkpoint "
                         f"(a diffusers-folder or UNet-only file is not supported).")
    try:
        state = convert_ldm_unet_checkpoint(ldm, cfg)
    except KeyError as e:
        raise ValueError(f"{path} does not match the SDXL UNet (missing {e}). Is it an SDXL checkpoint (not SD 1.5 / "
                         f"SD 2 / Flux)?") from e
    del ldm
    with torch.device("meta"):
        unet = UNet2DConditionModel.from_config(cfg)
    result = unet.load_state_dict(state, strict=False, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"{path} does not match the SDXL UNet: {len(result.missing_keys)} missing key(s) (e.g. "
                         f"{result.missing_keys[:3]}), {len(result.unexpected_keys)} unexpected (e.g. "
                         f"{result.unexpected_keys[:3]}). Is it an SDXL checkpoint (not SD 1.5 / SD 2 / Flux)?")
    return unet.to(device).eval().requires_grad_(False)


def block_label(block: str) -> str:
    """'in_04' -> 'IN04' (kohya's block-weight names); 'mid' -> 'MID'."""
    m = re.match(r"(in|out)_(\d+)$", block)
    return f"{m.group(1).upper()}{m.group(2)}" if m else block.upper()
