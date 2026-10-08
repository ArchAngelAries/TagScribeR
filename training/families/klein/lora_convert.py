# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/networks/lora.py (commit 1c8ec88):
# _DIFFUSERS_* tables, _is_diffusers_flux_lora, _convert_diffusers_flux_lora, _peft_normalize and the OneTrainer
# lora_transformer_ rename of ensure_kohya_lora_state_dict - what Fizgig 7.0.1's KleinDriver.convert_lora_state_dict
# runs. Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: only the parts a Klein file needs before training/lora.py's reader, which already strips
# the transformer. / diffusion_model. prefixes and reads lora_A / lora_B, LoKR and LoHa keys.
"""Klein LoRA files from other trainers -> kohya names Klein's model has: diffusers Flux files (split q / k / v fused
into Klein's qkv and linear1), OneTrainer's lora_transformer_ prefix, PEFT adapter names."""
import logging
import re
from typing import Dict

import torch

logger = logging.getLogger(__name__)

_DIFFUSERS_DOUBLE_BLOCK_MAP = {
    "attn.to_out.0": "img_attn.proj",
    "attn.to_add_out": "txt_attn.proj",
    "ff.net.0.proj": "img_mlp.0",
    "ff.net.2": "img_mlp.2",
    "ff_context.net.0.proj": "txt_mlp.0",
    "ff_context.net.2": "txt_mlp.2",
    "norm1.linear": "img_mod.lin",
    "norm1_context.linear": "txt_mod.lin",
    # LyCORIS / LoKR variants
    "ff.linear_in": "img_mlp.0",
    "ff.linear_out": "img_mlp.2",
    "ff_context.linear_in": "txt_mlp.0",
    "ff_context.linear_out": "txt_mlp.2",
}

_DIFFUSERS_SINGLE_BLOCK_MAP = {
    "proj_out": "linear2",
    "norm.linear": "modulation.lin",
    "attn.to_out": "linear2",          # Flux 2 variant
    "attn.to_qkv_mlp_proj": "linear1", # Flux 2 fused variant
}

# QKV fusion groups: diffusers separate keys → Klein fused key
# (diffusers_suffix, klein_target, slot_index)
_DIFFUSERS_DOUBLE_QKV_IMG = [
    ("attn.to_q", "img_attn.qkv", 0),
    ("attn.to_k", "img_attn.qkv", 1),
    ("attn.to_v", "img_attn.qkv", 2),
]
_DIFFUSERS_DOUBLE_QKV_TXT = [
    ("attn.add_q_proj", "txt_attn.qkv", 0),
    ("attn.add_k_proj", "txt_attn.qkv", 1),
    ("attn.add_v_proj", "txt_attn.qkv", 2),
]
_DIFFUSERS_SINGLE_QKV = [
    ("attn.to_q", "linear1", 0),
    ("attn.to_k", "linear1", 1),
    ("attn.to_v", "linear1", 2),
    ("proj_mlp", "linear1", 3),  # MLP projection fused into linear1 slot 3
]


def _is_diffusers_flux_lora(weights_sd: Dict[str, torch.Tensor]) -> bool:
    """Check if the state dict uses diffusers-style Flux key names."""
    for k in weights_sd:
        if "transformer_blocks." in k or "single_transformer_blocks." in k:
            return True
    return False


def _convert_diffusers_flux_lora(weights_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert a diffusers-format Flux LoRA to kohya convention.

    Handles QKV fusion: separate Q/K/V LoRA weights are concatenated into
    a single fused LoRA module targeting Klein's fused QKV linear.
    """
    converted: Dict[str, torch.Tensor] = {}
    # Collect QKV groups for fusion: {klein_module_name: {slot: {suffix: tensor}}}
    qkv_pending: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {}

    # Strip transformer. prefix if present
    stripped_sd = {}
    for k, v in weights_sd.items():
        if k.startswith("transformer."):
            stripped_sd[k[len("transformer."):]] = v
        else:
            stripped_sd[k] = v

    for key, tensor in stripped_sd.items():
        # Determine suffix type
        if key.endswith(".lora_down.weight"):
            suffix = "lora_down.weight"
            module_path = key[:-len(".lora_down.weight")]
        elif key.endswith(".lora_up.weight"):
            suffix = "lora_up.weight"
            module_path = key[:-len(".lora_up.weight")]
        elif key.endswith(".lora_A.weight"):
            suffix = "lora_down.weight"
            module_path = key[:-len(".lora_A.weight")]
        elif key.endswith(".lora_B.weight"):
            suffix = "lora_up.weight"
            module_path = key[:-len(".lora_B.weight")]
        elif key.endswith(".alpha"):
            suffix = "alpha"
            module_path = key[:-len(".alpha")]
        elif key.endswith(".dora_scale"):
            continue
        else:
            converted[key] = tensor
            continue

        # Parse block type and index
        klein_module = None
        is_qkv = False

        # Double blocks
        if module_path.startswith("transformer_blocks."):
            rest = module_path[len("transformer_blocks."):]
            dot = rest.find(".")
            if dot == -1:
                continue
            idx = rest[:dot]
            layer_path = rest[dot + 1:]
            block_prefix = f"double_blocks.{idx}"

            # Check QKV fusion groups
            for diff_name, klein_target, slot in _DIFFUSERS_DOUBLE_QKV_IMG + _DIFFUSERS_DOUBLE_QKV_TXT:
                if layer_path == diff_name:
                    klein_module = f"{block_prefix}.{klein_target}"
                    qkv_pending.setdefault(klein_module, {}).setdefault(slot, {})[suffix] = tensor
                    is_qkv = True
                    break
            if is_qkv:
                continue

            # Check 1:1 mappings
            for diff_name, klein_name in _DIFFUSERS_DOUBLE_BLOCK_MAP.items():
                if layer_path == diff_name:
                    klein_module = f"{block_prefix}.{klein_name}"
                    break

        # Single blocks
        elif module_path.startswith("single_transformer_blocks."):
            rest = module_path[len("single_transformer_blocks."):]
            dot = rest.find(".")
            if dot == -1:
                # Could be proj_mlp (no dot after index)
                # e.g. single_transformer_blocks.0  — shouldn't happen with a layer suffix
                continue
            idx = rest[:dot]
            layer_path = rest[dot + 1:]
            block_prefix = f"single_blocks.{idx}"

            # Check QKV fusion groups
            for diff_name, klein_target, slot in _DIFFUSERS_SINGLE_QKV:
                if layer_path == diff_name:
                    klein_module = f"{block_prefix}.{klein_target}"
                    qkv_pending.setdefault(klein_module, {}).setdefault(slot, {})[suffix] = tensor
                    is_qkv = True
                    break
            if is_qkv:
                continue

            # proj_mlp has no dot separator in the layer path for single blocks
            if layer_path == "proj_mlp" or module_path.endswith(".proj_mlp"):
                klein_module = f"{block_prefix}.linear1"
                qkv_pending.setdefault(klein_module, {}).setdefault(3, {})[suffix] = tensor
                continue

            # Check 1:1 mappings
            for diff_name, klein_name in _DIFFUSERS_SINGLE_BLOCK_MAP.items():
                if layer_path == diff_name:
                    klein_module = f"{block_prefix}.{klein_name}"
                    break

        if klein_module is not None and not is_qkv:
            flat = klein_module.replace(".", "_")
            lora_name = f"lora_unet_{flat}"
            converted[f"{lora_name}.{suffix}"] = tensor

    # Fuse QKV groups
    for klein_module, slots in qkv_pending.items():
        flat = klein_module.replace(".", "_")
        lora_name = f"lora_unet_{flat}"

        # Collect lora_down and lora_up from each slot, sorted by slot index
        sorted_slots = sorted(slots.keys())
        downs = [slots[s]["lora_down.weight"] for s in sorted_slots if "lora_down.weight" in slots[s]]
        ups = [slots[s]["lora_up.weight"] for s in sorted_slots if "lora_up.weight" in slots[s]]

        if not downs or not ups:
            # Incomplete QKV group — emit whatever we have as-is
            for s in sorted_slots:
                for suf, t in slots[s].items():
                    converted[f"{lora_name}_slot{s}.{suf}"] = t
            continue

        # All Q/K/V lora_down: (rank, in_dim) each → cat along dim 0 → (N*rank, in_dim)
        fused_down = torch.cat(downs, dim=0)

        # All Q/K/V lora_up: (out_dim, rank) each → build block-diagonal
        # so each slot's output lands in the right slice of the fused output.
        # Each slot's own alpha/rank scale is baked into its columns here, and the
        # fused module ships alpha = total_rank (scale 1.0) — so a source exporting
        # e.g. alpha = rank/2 (OneTrainer, ai-toolkit) keeps its true strength
        # instead of being inflated to scale 1.0.
        slot_ups = [(s, slots[s]["lora_up.weight"]) for s in sorted_slots
                    if "lora_up.weight" in slots[s]]
        ranks = [u.shape[1] for _, u in slot_ups]
        out_dims = [u.shape[0] for _, u in slot_ups]
        total_rank = sum(ranks)
        total_out = sum(out_dims)

        fused_up = torch.zeros(total_out, total_rank, dtype=slot_ups[0][1].dtype)
        r_offset = 0
        o_offset = 0
        for i, (s, up) in enumerate(slot_ups):
            alpha_t = slots[s].get("alpha")
            # PEFT convention when alpha is absent: alpha = rank → scale 1.0
            slot_scale = (float(alpha_t) / ranks[i]) if alpha_t is not None else 1.0
            if slot_scale != 1.0:
                up = (up.to(torch.float32) * slot_scale).to(up.dtype)
            fused_up[o_offset:o_offset + out_dims[i], r_offset:r_offset + ranks[i]] = up
            r_offset += ranks[i]
            o_offset += out_dims[i]

        converted[f"{lora_name}.lora_down.weight"] = fused_down
        converted[f"{lora_name}.lora_up.weight"] = fused_up
        converted[f"{lora_name}.alpha"] = torch.tensor(float(total_rank))

    n_modules = len([k for k in converted if k.endswith(".lora_down.weight")])
    logger.info(f"[diffusers→kohya] converted {len(weights_sd)} keys → {len(converted)} "
                f"({n_modules} LoRA modules, QKV groups fused)")
    return converted


def _peft_normalize(weights_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Two PEFT export quirks, straightened before conversion:
      - the adapter name left in the key (PEFT's own state dict): blocks.0.attn.qkv_proj.lora_A.default.weight
        -> blocks.0.attn.qkv_proj.lora_A.weight
      - a module saved twice, once under diffusion_model. / transformer. and once bare: the prefixed one is kept (both
        map to the same module, and a file seen in the wild held DIFFERENT weights under the bare copy)."""
    pat = re.compile(r"\.(lora_[AB])\.(?!weight$)[^.]+\.weight$")
    out: Dict[str, torch.Tensor] = {}
    renamed = 0
    for k, v in weights_sd.items():
        nk = pat.sub(lambda m: f".{m.group(1)}.weight", k)
        renamed += nk != k
        out[nk] = v
    if renamed:
        logger.info(f"[peft→kohya] {renamed} keys carried a PEFT adapter name (lora_A.<name>.weight) - read as lora_A/B")
    prefixed = {k[len(p):] for k in out for p in ("diffusion_model.", "transformer.") if k.startswith(p)}
    dup = [k for k in out if k in prefixed]
    if dup:
        mods = {k.rsplit(".lora_", 1)[0].rsplit(".alpha", 1)[0] for k in dup}
        logger.warning(f"[peft→kohya] {len(mods)} module(s) are in the file twice, prefixed and bare (e.g. "
                       f"{sorted(mods)[0]}) - the prefixed copy is used, the bare one ignored")
        for k in dup:
            del out[k]
    return out


def convert(weights_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """A Klein LoRA file's tensors in the names training/lora.py reads (Fizgig ensure_kohya_lora_state_dict, the
    Klein-relevant steps)."""
    if any(k.startswith("lora_transformer_") for k in weights_sd):
        n = sum(1 for k in weights_sd if k.startswith("lora_transformer_"))
        weights_sd = {("lora_unet_" + k[len("lora_transformer_"):]) if k.startswith("lora_transformer_") else k: v
                      for k, v in weights_sd.items()}
        logger.info("OneTrainer legacy format detected - renamed %d lora_transformer_ keys to lora_unet_", n)
    weights_sd = _peft_normalize(weights_sd)
    if _is_diffusers_flux_lora(weights_sd):
        logger.info("Diffusers-format Flux LoRA detected (transformer_blocks) - converting with QKV fusion.")
        return _convert_diffusers_flux_lora(weights_sd)
    return weights_sd
