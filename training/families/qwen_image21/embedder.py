# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/qwen_image21/embedder.py
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: the ComfyUI key converter and shard loader are inlined (they lived in Fizgig's krea2 package); otherwise unchanged.
"""Qwen Image 2.1 text conditioning: Qwen3-VL-8B, last decoder layer BEFORE the final RMSNorm.

Follows the reference pipeline (diffusers pipeline_qwenimage21.py `_get_qwen_prompt_embeds`): the prompt goes into a
raw template string (not apply_chat_template - "the two tokenize differently and the checkpoint expects this one"),
left padding, `hidden_states[-1]` with the final norm neutralised by a forward hook, the system-turn tokens dropped,
an empty prompt becomes " ". Text-to-image only for now (no reference images).

Weights: the ComfyUI single file (bare `model.` / `model.visual.` keys, converted by convert_comfyui_qwen3vl_state_dict)
or official shards; the vision tower and LM head are dropped after loading. Tokenizer/processor: the Qwen-Image-2.1 `processor/` files
(downloaded once from the Hugging Face Hub).
"""
import json
import logging
import os

import torch

logger = logging.getLogger(__name__)

TOKENIZER_REPO = "Qwen/Qwen-Image-2.1"
SYS_PROMPT = "Comprehend and analyze the provided prompt."
TEMPLATE_T2I = (f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n"
                "<|im_start|>user\n{}<|im_end|>\n"
                "<|im_start|>assistant\n")


def _disable_broken_hf_transfer():
    """Pod images often export HF_HUB_ENABLE_HF_TRANSFER=1 without the hf_transfer package; huggingface_hub then
    refuses every download (here: the tokenizer files). Fall back to the normal downloader instead."""
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") not in ("", "0", "false", "False"):
        try:
            import hf_transfer  # noqa: F401
        except ImportError:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
            try:
                from huggingface_hub import constants
                constants.HF_HUB_ENABLE_HF_TRANSFER = False
            except Exception:
                pass


def _text_encoder_config(config_path=None) -> dict:
    here = os.path.join(os.path.dirname(__file__), "qwen3vl_8b_config.json")
    with open(config_path or here, encoding="utf-8") as f:
        return json.load(f)


def convert_comfyui_qwen3vl_state_dict(sd: dict) -> dict:
    """Map a ComfyUI-style (bare ``model.`` / ``visual.``) Qwen3-VL state dict onto the HF
    ``Qwen3VLForConditionalGeneration`` layout. Official HF checkpoints already use the
    ``model.language_model.`` / ``model.visual.`` layout and pass through unchanged.
    (Fizgig src/fizgig/krea2/embedder.py: _convert_comfyui_qwen3vl_state_dict.)"""
    converted = {}
    for key, value in sd.items():
        if key.startswith("model.language_model.") or key.startswith("model.visual."):
            new_key = key
        elif key.startswith("visual."):
            new_key = "model.visual." + key[len("visual."):]
        elif key.startswith("language_model."):
            new_key = "model." + key
        elif key.startswith("model."):
            new_key = "model.language_model." + key[len("model."):]
        else:
            new_key = key
        converted[new_key] = value
    return converted


def load_split_weights(path: str, dtype=None) -> dict:
    """All tensors of a .safetensors file on CPU - every shard when the name ends in -00001-of-0000N (the shards'
    convention), so official multi-file checkpoints load from their first file."""
    import glob
    import re

    from safetensors.torch import load_file
    m = re.match(r"(.+)-\d{5}-of-(\d{5})\.safetensors$", path)
    files = sorted(glob.glob(glob.escape(m.group(1)) + f"-*-of-{m.group(2)}.safetensors")) if m else [path]
    sd = {}
    for f in files:
        sd.update(load_file(f, device="cpu"))
    if dtype is not None:
        sd = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in sd.items()}
    return sd


def _w8_forward(self, x):
    w = (self.weight.view(self.weight.shape[0], -1, _W8_GROUP).to(x.dtype) * self._w8_scale.to(x.dtype))
    return torch.nn.functional.linear(x, w.view(self.weight.shape), self.bias)


_W8_GROUP = 64


@torch.no_grad()
def _int8_weights(model, prefix, device):
    """Weight-only 8-bit with one scale per 64 weights, dequantised per matmul; activations stay bf16. Measured on
    Qwen3-VL-8B hidden states against bf16 (27 Sep 2026): W8A8 was 35% off (cosine 0.93: activation outliers), weight-
    only with per-channel scales 8% (0.996); per-group scales are the fix for the per-channel error."""
    n = 0
    for name, m in model.named_modules():
        if not (isinstance(m, torch.nn.Linear) and prefix in name) or m.in_features % _W8_GROUP:
            continue
        w = m.weight.data.to(device).float().view(m.out_features, -1, _W8_GROUP)
        scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
        m.weight.requires_grad_(False)
        m.weight.data = (w / scale).round_().clamp_(-127, 127).to(torch.int8).view(m.out_features, m.in_features)
        m.register_buffer("_w8_scale", scale.to(torch.bfloat16), persistent=False)
        m.forward = _w8_forward.__get__(m, type(m))
        n += 1
    return n


class Qwen21TextEncoder:
    def __init__(self, model_path, tokenizer_dir=None, device="cuda", dtype=torch.bfloat16, config_path=None,
                 int8=False, vision=False):
        """The LM head is dropped at load (captioning is TagScribeR's captioners' job), and the vision tower too
        unless `vision` (edit training: reference images are encoded with the prompt, see encode_with_references). int8: the language model's Linears as INT8 (about 8 GB
        in all instead of 16), for cards that cannot hold the bf16 encoder - small enough for a 10 GB card. Quantised
        one Linear at a time from CPU, so the bf16 model is never resident."""
        from accelerate import init_empty_weights
        from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration


        self.device, self.dtype = torch.device(device), dtype
        _disable_broken_hf_transfer()
        # Default: the official repo's processor/ files (a few MB, cached by huggingface_hub; the fetcher
        # warms the cache so this works offline).
        self.tokenizer = (AutoTokenizer.from_pretrained(tokenizer_dir) if tokenizer_dir else
                          AutoTokenizer.from_pretrained(TOKENIZER_REPO, subfolder="processor"))
        config = Qwen3VLConfig.from_dict(_text_encoder_config(config_path))
        with init_empty_weights():
            model = Qwen3VLForConditionalGeneration._from_config(config)
        sd = convert_comfyui_qwen3vl_state_dict(load_split_weights(model_path, dtype=dtype))
        info = model.load_state_dict(sd, strict=False, assign=True)
        if info.unexpected_keys or info.missing_keys:
            raise RuntimeError(f"Qwen3-VL-8B checkpoint mismatch: missing={info.missing_keys[:8]}, "
                               f"unexpected={info.unexpected_keys[:8]}")
        model.lm_head = None
        if not vision:
            model.model.visual = None
        self.vision = vision
        del sd
        if int8:
            n = _int8_weights(model, "language_model.layers.", self.device)
            logger.info(f"[text encoder] 8-bit weights: {n} language-model Linears (low-VRAM card); matmuls stay bf16")
        self.model = model.to(self.device).eval().requires_grad_(False)
        # Number of leading system-turn tokens to drop (the reference derives it from the tokenized system message).
        sys_ids = self.tokenizer(f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n", add_special_tokens=False).input_ids
        self.drop_idx = len(sys_ids)

    @torch.no_grad()
    def encode(self, prompts):
        """-> list of [L_i, 4096] tensors (variable length, padding removed), in the model dtype, on CPU."""
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        texts = [TEMPLATE_T2I.format(p if p else " ") for p in prompts]
        tok = self.tokenizer(texts, padding=True, padding_side="left", return_tensors="pt").to(self.device)
        lm = getattr(self.model.model, "language_model", self.model.model)
        # hidden_states[-1] must be the last decoder layer BEFORE the final RMSNorm (what the DiT was trained on).
        # transformers 4.x already returns that; the hook makes it hold on 5.x too.
        handle = lm.norm.register_forward_hook(lambda m, args, out: args[0])
        try:
            # the base model, not the LM wrapper: same hidden states, no vocabulary-wide logits
            out = self.model.model(input_ids=tok.input_ids, attention_mask=tok.attention_mask,
                                   output_hidden_states=True)
        finally:
            handle.remove()
        hs = out.hidden_states[-1]
        res = []
        for h, m in zip(hs, tok.attention_mask.bool()):
            res.append(h[m][self.drop_idx:].to("cpu"))
        return res

    @torch.no_grad()
    def encode_with_references(self, prompt, images):
        """Edit conditioning: the prompt with its reference images in the template (<imageN> + a vision block per image
        before the prompt, as the reference pipeline and ComfyUI build it). `images` are PIL RGB images already at
        multiples of 32 px (the same size their VAE latents use, so each vision token covers 2x2 latent tokens).
        -> ([L, 4096] hidden states on CPU, [L] bool True at the reference images' vision tokens)."""
        if not self.vision:
            raise RuntimeError("this Qwen3-VL encoder was loaded without its vision tower (vision=False)")
        from transformers import Qwen3VLProcessor
        if getattr(self, "_processor", None) is None:
            self._processor = Qwen3VLProcessor.from_pretrained(TOKENIZER_REPO, subfolder="processor")
        refs = " ".join(f"<image{i + 1}><|vision_start|><|image_pad|><|vision_end|>" for i in range(len(images)))
        text = TEMPLATE_T2I.format(refs + (prompt if prompt else " "))
        inputs = self._processor(text=[text], images=list(images), return_tensors="pt",
                                 images_kwargs={"do_resize": False}).to(self.device)
        lm = getattr(self.model.model, "language_model", self.model.model)
        handle = lm.norm.register_forward_hook(lambda m, args, out: args[0])
        try:
            out = self.model.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                   pixel_values=inputs["pixel_values"].to(self.dtype),
                                   image_grid_thw=inputs["image_grid_thw"], output_hidden_states=True)
        finally:
            handle.remove()
        pad_id = self._processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        ids = inputs["input_ids"][0][self.drop_idx:]
        return out.hidden_states[-1][0][self.drop_idx:].to("cpu"), (ids == pad_id).to("cpu")

    def unload(self):
        self.model.to("cpu")
        del self.model
        torch.cuda.empty_cache()
