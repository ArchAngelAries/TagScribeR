# Ported from Fizgig (https://github.com/shootthesound/Fizgig) src/fizgig/klein/embedder.py (Qwen3Embedder, the vendored
# Qwen3-8B config, load_qwen3) and scripts/cache_text.py (the caching call).
# Copyright 2026 Peter Neill. Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
# Changes for TagScribeR: the LM head is dropped and the base model is called directly (no vocabulary-wide logits);
# Fizgig's fp8 text-encoder option (float8 weights + patched norms) is replaced by weight-only INT8 with per-group
# scales on small cards (the Qwen Image 2.1 encoder's scheme, measured there against bf16) - bf16 otherwise; an
# fp8-scaled checkpoint is dequantised at load; einops is replaced by a reshape; the tokenizer comes from a
# qwen3_tokenizer/ folder next to the checkpoint or the Hugging Face cache (Fizgig loads Qwen2Tokenizer from the
# cache first, then the Hub; AutoTokenizer resolves to the same Qwen tokenizer).
"""Klein 9B text conditioning: Qwen3-8B, the hidden states of layers 9, 18 and 27 (each passed through the model's final
RMSNorm, as ComfyUI's Klein pipeline does), concatenated per token: (batch, 512, 12288). The prompt goes through the
chat template (user turn, generation prompt, thinking off) and is padded to 512 tokens; every position is kept - there
is no mask on the DiT side, exactly as Fizgig caches it (about 12.6 MB per caption in bf16)."""
import json
import logging
import os

import torch

logger = logging.getLogger(__name__)

TOKENIZER_REPO = "Qwen/Qwen3-8B"
TOKENIZER_FILES = ("generation_config.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json")

# Klein 9B uses Qwen3-8B with hidden states from these layers (Fizgig klein/embedder.py:20-21)
OUTPUT_LAYERS_QWEN3 = (9, 18, 27)
MAX_LENGTH = 512

# Fizgig klein/embedder.py QWEN3_8B_CONFIG_JSON, verbatim
QWEN3_8B_CONFIG_JSON = """{
  "architectures": ["Qwen3ForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 12288,
  "max_position_embeddings": 40960,
  "max_window_layers": 36,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 36,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.0",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}"""


def tokenizer_source(model_path: str) -> str:
    """A qwen3_tokenizer/ folder next to the checkpoint (offline route: TOKENIZER_FILES from Qwen/Qwen3-8B), else the
    repo id (Hugging Face cache first)."""
    local = os.path.join(os.path.dirname(os.path.abspath(model_path)), "qwen3_tokenizer")
    if os.path.isfile(os.path.join(local, "tokenizer_config.json")):
        return local
    return TOKENIZER_REPO


def build_config(**overrides):
    from transformers import Qwen3Config
    cfg = json.loads(QWEN3_8B_CONFIG_JSON)
    cfg.update(overrides)
    return Qwen3Config(**cfg)


class KleinTextEncoder:
    """Loads Qwen3-8B and encodes prompts into Klein's conditioning."""

    def __init__(self, model_path=None, device="cuda", dtype=torch.bfloat16, int8=False, *, model=None,
                 tokenizer=None, max_length=MAX_LENGTH, layers=OUTPUT_LAYERS_QWEN3):
        """model_path: Qwen3-8B safetensors (ComfyUI's qwen_3_8b.safetensors; bf16 or fp8-scaled). int8: the language
        model's Linears as 8-bit weights (about 9 GB instead of 16.4) for cards that cannot hold bf16. `model` /
        `tokenizer` inject ready objects (tests): a Qwen3ForCausalLM-like object with `.model` (base) and
        `.model.norm`."""
        self.device, self.dtype = torch.device(device), dtype
        self.max_length, self.layers = max_length, tuple(layers)
        if model is None:
            model, tokenizer = self._load(model_path, dtype, int8, self.device)
        self.model = model.to(self.device).eval().requires_grad_(False)
        self.tokenizer = tokenizer

    @staticmethod
    def _load(model_path, dtype, int8, device):
        from accelerate import init_empty_weights
        from transformers import AutoTokenizer, Qwen3ForCausalLM

        from training.families.krea2.embedder import dequantize_prequantized
        from training.families.qwen_image21.embedder import _disable_broken_hf_transfer, _int8_weights, load_split_weights
        _disable_broken_hf_transfer()
        source = tokenizer_source(model_path)
        try:
            try:
                tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)   # cache first (Fizgig)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(source)
        except Exception as e:
            raise RuntimeError(
                f"Couldn't load the Qwen3 tokenizer from {source} ({type(e).__name__}: {e}). Offline: download "
                f"{', '.join(TOKENIZER_FILES)} from https://huggingface.co/{TOKENIZER_REPO} into a qwen3_tokenizer/ "
                f"folder next to {model_path}.") from e
        with init_empty_weights():
            model = Qwen3ForCausalLM._from_config(build_config())
        model.lm_head = None                        # tied to the embeddings, never needed to encode
        logger.info(f"Loading Klein text encoder (Qwen3-8B) weights from {model_path}")
        sd = load_split_weights(str(model_path))
        sd = {k: v for k, v in sd.items() if not k.startswith("lm_head.")}
        sd = dequantize_prequantized(sd, dtype)
        sd = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in sd.items()}
        info = model.load_state_dict(sd, strict=False, assign=True)
        if info.unexpected_keys or info.missing_keys:
            raise RuntimeError(f"Qwen3-8B checkpoint mismatch: missing={info.missing_keys[:8]}, "
                               f"unexpected={info.unexpected_keys[:8]} - is this qwen_3_8b.safetensors?")
        del sd
        if int8:
            n = _int8_weights(model, "model.layers.", device)
            logger.info(f"[text encoder] 8-bit weights: {n} language-model Linears (low-VRAM card); matmuls stay bf16")
        return model, tokenizer

    @torch.no_grad()
    def encode(self, prompts):
        """-> (B, max_length, 3 * hidden) on the encoder's device. Fizgig Qwen3Embedder.forward."""
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        ids, masks = [], []
        for prompt in prompts:
            text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                      add_generation_prompt=True, enable_thinking=False)
            tok = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True,
                                 max_length=self.max_length)
            ids.append(tok["input_ids"])
            masks.append(tok["attention_mask"])
        input_ids = torch.cat(ids, dim=0).to(self.device)
        attention_mask = torch.cat(masks, dim=0).to(self.device)
        base = self.model.model
        out = base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
        # Stack hidden states from the target layers, then apply the final RMSNorm (ComfyUI's
        # final_layer_norm_intermediate; without it raw states carry huge outliers - Fizgig klein/embedder.py:111-118)
        stack = torch.stack([out.hidden_states[k] for k in self.layers], dim=1)           # (B, 3, L, D)
        stack = base.norm(stack)
        b, c, l, d = stack.shape
        return stack.permute(0, 2, 1, 3).reshape(b, l, c * d)                              # b c l d -> b l (c d)

    def unload(self):
        self.model.to("cpu")
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
