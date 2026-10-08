# Facts from kohya sd-scripts (Apache-2.0) library/strategy_anima.py + library/anima_utils.py and diffusion-pipe
# models/cosmos_predict2.py (the Qwen3-0.6B + T5-tokenizer conditioning recipe), cross-checked against ComfyUI's
# comfy/text_encoders/anima.py (GPL: facts only - the pad token id and the right-padded / unmasked inference
# convention). The vendored config is Qwen/Qwen3-0.6B's config.json (Apache-2.0). Code is TagScribeR's own.
# See THIRD_PARTY_NOTICES.md.
"""Anima text conditioning: Qwen3-0.6B (base) hidden states + T5 token ids.

The prompt is tokenised TWICE: Qwen3 tokens go through the Qwen3 model (its last hidden state after the final norm,
padding zeroed); the T5 tokens are NOT encoded by any T5 model - their ids are the `target` sequence the DiT's LLM
adapter embeds and lets cross-attend to the Qwen3 states. Both are padded to 512 tokens with a validity mask, so a
cached caption is {prompt_embeds (512, 1024) bf16, attn_mask (512,), t5_ids (512,), t5_mask (512,)} (~1 MB).
"""
import logging
import os

import torch

logger = logging.getLogger(__name__)

MAX_LENGTH = 512                               # sd-scripts --qwen3_max_token_length / --t5_max_token_length defaults
QWEN3_REPO = "Qwen/Qwen3-0.6B"                 # the base model shares the instruct model's tokenizer
T5_REPO = "google/t5-v1_1-xxl"                 # sd-scripts docs: "vocabulary from google/t5-v1_1-xxl", files only
QWEN3_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
T5_TOKENIZER_FILES = ("spiece.model", "tokenizer_config.json", "special_tokens_map.json")

# Qwen/Qwen3-0.6B config.json (the Anima text encoder is Qwen3-0.6B-Base: same architecture)
QWEN3_06B_CONFIG = {
    "attention_bias": False, "attention_dropout": 0.0, "bos_token_id": 151643, "eos_token_id": 151645,
    "head_dim": 128, "hidden_act": "silu", "hidden_size": 1024, "initializer_range": 0.02,
    "intermediate_size": 3072, "max_position_embeddings": 40960, "max_window_layers": 28, "model_type": "qwen3",
    "num_attention_heads": 16, "num_hidden_layers": 28, "num_key_value_heads": 8, "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0, "sliding_window": None, "tie_word_embeddings": True, "use_cache": False,
    "use_sliding_window": False, "vocab_size": 151936,
}


def tokenizer_source(model_path: str, folder: str, repo: str, marker: str) -> str:
    """A `folder`/ next to the text-encoder file (the offline route: put the repo's tokenizer files in it), else the
    Hugging Face repo id (the local cache first)."""
    local = os.path.join(os.path.dirname(os.path.abspath(model_path)), folder)
    return local if os.path.isfile(os.path.join(local, marker)) else repo


def _tokenize(tokenizer, captions, max_length):
    enc = tokenizer(list(captions), return_tensors="pt", truncation=True, padding="max_length",
                    max_length=max_length)
    return enc["input_ids"], enc["attention_mask"].bool()


class AnimaTextEncoder:
    """Qwen3-0.6B language model + both tokenizers; `encode(captions)` -> (embeds, qwen_mask, t5_ids, t5_mask)."""

    def __init__(self, model_path=None, device="cuda", dtype=torch.bfloat16, *, model=None, qwen_tokenizer=None,
                 t5_tokenizer=None, max_length=MAX_LENGTH):
        """model_path: the Qwen3-0.6B safetensors (ComfyUI `qwen_3_06b_base.safetensors`). `model` / tokenizers inject
        ready objects (tests)."""
        self.device, self.dtype, self.max_length = torch.device(device), dtype, max_length
        if model is None:
            model, qwen_tokenizer, t5_tokenizer = self._load(model_path, dtype)
        self.model = model.to(self.device).eval().requires_grad_(False)
        self.qwen_tokenizer, self.t5_tokenizer = qwen_tokenizer, t5_tokenizer
        # RIGHT padding: the real tokens sit at positions 0..n-1 exactly as in unpadded inference (ComfyUI feeds no
        # padding at all). A left-padding tokenizer default would shift every position.
        self.qwen_tokenizer.padding_side = "right"
        if self.qwen_tokenizer.pad_token is None:
            self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token

    @staticmethod
    def _load(model_path, dtype):
        from accelerate import init_empty_weights
        from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

        from training.families.qwen_image21.embedder import load_split_weights
        from training.hf_cache import from_pretrained_cache_first
        sources = {"qwen": tokenizer_source(model_path, "qwen3_tokenizer", QWEN3_REPO, "tokenizer_config.json"),
                   "t5": tokenizer_source(model_path, "t5_tokenizer", T5_REPO, "tokenizer_config.json")}
        toks = {}
        for name, src in sources.items():
            try:
                toks[name] = from_pretrained_cache_first(AutoTokenizer, src)    # offline once cached (Fizgig #174)
            except Exception as e:
                repo, files, folder = ((QWEN3_REPO, QWEN3_TOKENIZER_FILES, "qwen3_tokenizer") if name == "qwen"
                                       else (T5_REPO, T5_TOKENIZER_FILES, "t5_tokenizer"))
                raise RuntimeError(
                    f"Couldn't load the {'Qwen3' if name == 'qwen' else 'T5'} tokenizer from {src} "
                    f"({type(e).__name__}: {e}). Offline: download {', '.join(files)} from "
                    f"https://huggingface.co/{repo} into a {folder}/ folder next to {model_path}. (Only the "
                    f"tokenizer files are needed, not the model weights.)") from e
        with init_empty_weights():
            model = Qwen3Model(Qwen3Config.from_dict(QWEN3_06B_CONFIG))
        logger.info(f"Loading Anima text encoder (Qwen3-0.6B) weights from {model_path}")
        sd = {}
        for k, v in load_split_weights(str(model_path)).items():
            k = k[len("model."):] if k.startswith("model.") else k
            if not k.startswith("lm_head."):
                sd[k] = v.to(dtype) if v.is_floating_point() else v
        info = model.load_state_dict(sd, strict=False, assign=True)
        if info.missing_keys or info.unexpected_keys:
            raise RuntimeError(f"Qwen3-0.6B checkpoint mismatch: missing={info.missing_keys[:8]}, "
                               f"unexpected={info.unexpected_keys[:8]} - is this qwen_3_06b_base.safetensors?")
        return model, toks["qwen"], toks["t5"]

    @torch.no_grad()
    def encode(self, captions):
        """-> embeds (B, L, 1024) in `dtype`, qwen_mask (B, L) bool, t5_ids (B, L) long, t5_mask (B, L) bool, on CPU."""
        captions = [captions] if isinstance(captions, str) else list(captions)
        ids, mask = _tokenize(self.qwen_tokenizer, captions, self.max_length)
        empty = ~mask.any(dim=1)
        if empty.any():
            # An empty caption has no Qwen3 token, and a fully masked attention row is NaN. ComfyUI encodes an empty
            # prompt as one pad token (min_length 1), so do the same (the caption-dropout / negative-prompt case).
            ids[empty, 0] = self.qwen_tokenizer.pad_token_id
            mask[empty, 0] = True
        t5_ids, t5_mask = _tokenize(self.t5_tokenizer, captions, self.max_length)
        states = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).last_hidden_state
        states = states.masked_fill(~mask.to(self.device)[..., None], 0.0)       # padding zeroed (both trainers)
        return states.to(self.dtype).cpu(), mask, t5_ids, t5_mask

    def unload(self):
        self.model.to("cpu")
        del self.model
        torch.cuda.empty_cache()
