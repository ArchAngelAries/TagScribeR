# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/krea2/embedder.py (Qwen3VLConditioner,
# the vendored Qwen3-VL-4B config, the checkpoint loader).
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: ENCODE ONLY - no captioning, reference images or style-caption prompts (the vision tower and
# LM head are dropped at load: the language model's hidden states are all Krea 2 conditions on); an fp8-scaled
# checkpoint is dequantised to bf16 at load (Fizgig keeps it fp8 behind a dequantising forward), or the language
# model's Linears are INT8 on small cards (the Qwen Image 2.1 encoder's weight-only per-group scheme); the tokenizer
# comes from a qwen3vl_tokenizer/ folder next to the checkpoint or the Hugging Face cache (Fizgig also bundles
# the files).
"""Krea 2 text conditioning: Qwen3-VL-4B-Instruct, a stack of 12 hidden-state layers per token.

The prompt goes into the fixed descriptor template (byte-identical to ComfyUI's Text-Encode-(Krea2) node; changing it
would silently alter every cached embedding), is padded to max_length (512 tokens after the template) and the
hidden states of layers (2, 5, ..., 35) are stacked: (batch, 512, 12, 2560) plus the validity mask. The mask is
[valid prompt, padding, valid suffix]; the DiT's text-fusion transformer folds the 12 layers into one stream.
"""
import logging
import os

import torch

logger = logging.getLogger(__name__)

QWEN3_VL_4B_INSTRUCT_REPO_ID = "Qwen/Qwen3-VL-4B-Instruct"
QWEN3_VL_TOKENIZER_FILES = (
    "chat_template.json", "generation_config.json", "merges.txt", "preprocessor_config.json", "tokenizer.json",
    "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json",
)

# Vendored Qwen3-VL-4B-Instruct config.json (Fizgig krea2/embedder.py QWEN3_VL_4B_INSTRUCT_CONFIG): the encoder is
# built without fetching a config from the Hub.
QWEN3_VL_4B_INSTRUCT_CONFIG = {
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "image_token_id": 151655,
    "model_type": "qwen3_vl",
    "text_config": {
        "attention_bias": False, "attention_dropout": 0.0, "bos_token_id": 151643, "dtype": "bfloat16",
        "eos_token_id": 151645, "head_dim": 128, "hidden_act": "silu", "hidden_size": 2560, "initializer_range": 0.02,
        "intermediate_size": 9728, "max_position_embeddings": 262144, "model_type": "qwen3_vl_text",
        "num_attention_heads": 32, "num_hidden_layers": 36, "num_key_value_heads": 8, "rms_norm_eps": 1e-06,
        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"},
        "rope_theta": 5000000, "tie_word_embeddings": True, "use_cache": True, "vocab_size": 151936,
    },
    "tie_word_embeddings": True,
    "transformers_version": "4.57.0.dev0",
    "video_token_id": 151656,
    "vision_config": {
        "deepstack_visual_indexes": [5, 11, 17], "depth": 24, "hidden_act": "gelu_pytorch_tanh", "hidden_size": 1024,
        "in_channels": 3, "initializer_range": 0.02, "intermediate_size": 4096, "model_type": "qwen3_vl",
        "num_heads": 16, "num_position_embeddings": 2304, "out_hidden_size": 2560, "patch_size": 16,
        "spatial_merge_size": 2, "temporal_patch_size": 2,
    },
    "vision_end_token_id": 151653,
    "vision_start_token_id": 151652,
}

MAX_LENGTH = 512
SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
# Must stay byte-identical to ComfyUI's Text-Encode-(Krea2) node (Fizgig krea2/embedder.py ENCODE_SYSTEM_DESCRIPTOR).
ENCODE_SYSTEM_DESCRIPTOR = ("Describe the image by detailing the color, shape, size, texture, "
                            "quantity, text, spatial relationships of the objects and background:")
PREFIX = "<|im_start|>system\n" + ENCODE_SYSTEM_DESCRIPTOR + "<|im_end|>\n<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PREFIX_START_IDX = 34             # tokens of PREFIX (checked against the tokenizer at load)
SUFFIX_START_IDX = 5              # tokens of SUFFIX


def tokenizer_source(model_path: str) -> str:
    """A qwen3vl_tokenizer/ folder next to the checkpoint (Fizgig's override / offline route: the files in
    QWEN3_VL_TOKENIZER_FILES from the Qwen3-VL-4B-Instruct repo), else the repo id (Hugging Face cache first)."""
    local = os.path.join(os.path.dirname(os.path.abspath(model_path)), "qwen3vl_tokenizer")
    if os.path.isfile(os.path.join(local, "tokenizer_config.json")):
        return local
    return QWEN3_VL_4B_INSTRUCT_REPO_ID


def dequantize_prequantized(sd: dict, dtype=torch.bfloat16) -> dict:
    """ComfyUI fp8_scaled -> plain weights: each `X.weight` with an `X.weight_scale` becomes weight * scale (a
    per-tensor scalar or a per-output-channel vector), the scale and `.comfy_quant` marker keys are dropped. A file
    without scales passes through. (Fizgig applies the scales per matmul instead; the values are the same.)"""
    if not any(k.endswith((".weight_scale", ".scale_weight")) for k in sd):
        return sd
    out = {}
    for k, v in sd.items():
        if k.endswith(".comfy_quant"):
            continue
        if k.endswith((".weight_scale", ".scale_weight")):
            continue
        stem = k[:-len(".weight")] if k.endswith(".weight") else None
        scale = sd.get(stem + ".weight_scale", sd.get(stem + ".scale_weight")) if stem else None
        if scale is not None:
            s = scale.float()
            s = s.reshape(1) if s.ndim == 0 else (s.unsqueeze(1) if s.ndim == 1 else s)
            out[k] = (v.float() * s).to(dtype)
        else:
            out[k] = v
    return out


class Krea2TextEncoder:
    """Loads the Qwen3-VL-4B language model and encodes prompts into the Krea 2 conditioning stack."""

    def __init__(self, model_path=None, device="cuda", dtype=torch.bfloat16, int8=False, *, model=None,
                 tokenizer=None, max_length=MAX_LENGTH, select_layers=SELECT_LAYERS):
        """model_path: a Qwen3-VL-4B safetensors (ComfyUI or official key layout; bf16 or fp8_scaled). int8: the
        language model's Linears as 8-bit weights (about 4.5 GB instead of 8.3), for cards that cannot hold bf16.
        `model` / `tokenizer` inject ready objects (tests)."""
        self.device, self.dtype = torch.device(device), dtype
        self.max_length, self.select_layers = max_length, tuple(select_layers)
        if model is None:
            model, tokenizer = self._load(model_path, dtype, int8, self.device)
        self.model = model.to(self.device).eval().requires_grad_(False)
        self.tokenizer = tokenizer
        n_prefix = len(tokenizer(PREFIX).input_ids)
        if n_prefix != PREFIX_START_IDX:
            raise RuntimeError(f"the tokenizer encodes the Krea 2 prompt template in {n_prefix} tokens, not "
                               f"{PREFIX_START_IDX} - is it the Qwen3-VL tokenizer?")

    @staticmethod
    def _load(model_path, dtype, int8, device):
        from accelerate import init_empty_weights
        from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration

        from training.families.qwen_image21.embedder import (_disable_broken_hf_transfer, _int8_weights,
                                                             convert_comfyui_qwen3vl_state_dict, load_split_weights)
        _disable_broken_hf_transfer()
        source = tokenizer_source(model_path)
        try:
            tokenizer = AutoTokenizer.from_pretrained(source)
        except Exception as e:
            raise RuntimeError(
                f"Couldn't load the Qwen3-VL tokenizer from {source} ({type(e).__name__}: {e}). Offline: download "
                f"{', '.join(QWEN3_VL_TOKENIZER_FILES)} from https://huggingface.co/{QWEN3_VL_4B_INSTRUCT_REPO_ID}"
                f" into a qwen3vl_tokenizer/ folder next to {model_path}.") from e
        config = Qwen3VLConfig.from_dict(QWEN3_VL_4B_INSTRUCT_CONFIG)
        with init_empty_weights():
            model = Qwen3VLForConditionalGeneration._from_config(config)
        model.lm_head = None                        # tied to the embeddings, never needed to encode
        model.model.visual = None                   # reference images / captioning are not used here
        logger.info(f"Loading Krea 2 text encoder (Qwen3-VL-4B) weights from {model_path}")
        sd = convert_comfyui_qwen3vl_state_dict(load_split_weights(str(model_path)))
        sd = {k: v for k, v in sd.items() if not k.startswith(("model.visual.", "lm_head."))}
        sd = dequantize_prequantized(sd, dtype)
        sd = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in sd.items()}
        info = model.load_state_dict(sd, strict=False, assign=True)
        if info.unexpected_keys or info.missing_keys:
            raise RuntimeError(f"Qwen3-VL-4B checkpoint mismatch: missing={info.missing_keys[:8]}, "
                               f"unexpected={info.unexpected_keys[:8]} - is this the Krea 2 text encoder?")
        del sd
        if int8:
            n = _int8_weights(model, "language_model.layers.", device)
            logger.info(f"[text encoder] 8-bit weights: {n} language-model Linears (low-VRAM card); matmuls stay bf16")
        return model, tokenizer

    @torch.no_grad()
    def encode(self, prompts):
        """-> (hiddens (B, max_length, n_layers, 2560), mask (B, max_length) bool), on the encoder's device. Fizgig's
        Qwen3VLConditioner._forward_text."""
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        text = [PREFIX + p for p in prompts]
        suffix = self.tokenizer([SUFFIX] * len(text), return_tensors="pt").to(self.device)
        inputs = self.tokenizer(text, truncation=True, return_length=False, return_overflowing_tokens=False,
                                padding="max_length", max_length=self.max_length + PREFIX_START_IDX - SUFFIX_START_IDX,
                                return_tensors="pt").to(self.device)
        input_ids = torch.cat([inputs["input_ids"], suffix["input_ids"]], dim=1)
        mask = torch.cat([inputs["attention_mask"].bool(), suffix["attention_mask"].bool()], dim=1)
        # the base model, not the LM wrapper: same hidden states, no vocabulary-wide logits. The selected layers
        # (<= 35) are intermediate states, so the final-norm quirk of the last hidden state does not apply.
        states = self.model.model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)
        hiddens = torch.stack([states.hidden_states[i] for i in self.select_layers], dim=2)
        return hiddens[:, PREFIX_START_IDX:], mask[:, PREFIX_START_IDX:]

    def unload(self):
        self.model.to("cpu")
        del self.model
        torch.cuda.empty_cache()
