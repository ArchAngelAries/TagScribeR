# SDXL text conditioning for TagScribeR's training layer (original code; Fizgig has no SDXL, its caching design - encode
# once, store a fixed-shape dict, unload the encoder - is what this follows).
#
# Architecture facts and their sources:
#   [sdxl-te]      stabilityai/stable-diffusion-xl-base-1.0 text_encoder/config.json (CLIP ViT-L/14: 12 layers, width 768,
#                  quick_gelu) and text_encoder_2/config.json (OpenCLIP ViT-bigG/14: 32 layers, width 1280, 20 heads,
#                  MLP 5120, gelu, projection 1280); vendored below so no config download is needed.
#   [sdxl-cond]    diffusers StableDiffusionXLPipeline.encode_prompt: the UNet's cross-attention input is the PENULTIMATE
#                  hidden state (hidden_states[-2], before the final layer norm) of both encoders, concatenated on the
#                  feature axis (768 + 1280 = 2048); the pooled vector is OpenCLIP-bigG's projected EOS embedding (1280).
#   [chunking]     A1111 / ComfyUI convention for prompts over 75 tokens: split the token ids into chunks of 75, wrap each
#                  as [BOS] + ids + [EOS] + padding (77 tokens), encode every chunk on its own and concatenate the hidden
#                  states along the sequence axis -> (77 * k, 2048); the pooled vector comes from the first chunk.
#                  (kohya's --max_token_length 225 instead strips the inner BOS/EOS and trains on 227 tokens; this port
#                  keeps the 77k form that A1111 and ComfyUI feed the UNet at inference.)
#   [pad]          CLIP-L pads with its EOS id (49407); OpenCLIP-bigG pads with id 0 ("!"), as open_clip does.
#   [diffusers-ldm] diffusers loaders/single_file_utils.py convert_ldm_clip_checkpoint / convert_open_clip_checkpoint turn
#                  the checkpoint's `conditioner.embedders.0.transformer.*` (CLIP-L, Hugging Face layout) and
#                  `conditioner.embedders.1.model.*` (bigG, open_clip layout) into transformers' CLIPTextModel keys.
#   Tokenizer: CLIP-L's BPE vocabulary (openai/clip-vit-large-patch14: vocab.json + merges.txt). OpenCLIP-bigG uses the same
#   vocabulary (open_clip's bpe_simple_vocab_16e6), so one tokenizer serves both; only the padding id differs.
"""SDXL's two CLIP text encoders, cached as one fixed-shape conditioning dict per caption."""
from __future__ import annotations

import os
from typing import Optional

import torch

CHUNKS = 3                       # 3 x 75 = 225 tokens, kohya's usual --max_token_length 225; FIXED so items stack
CHUNK_TOKENS = 75
SEQ_LEN = CHUNKS * (CHUNK_TOKENS + 2)          # 231
CROSS_DIM = 768 + 1280
POOLED_DIM = 1280
TOKENIZER_REPO = "openai/clip-vit-large-patch14"
TOKENIZER_DIRNAME = "clip_tokenizer"            # folder next to the checkpoint holding vocab.json + merges.txt

CLIP_L_CONFIG = dict(vocab_size=49408, hidden_size=768, intermediate_size=3072, num_hidden_layers=12,
                     num_attention_heads=12, max_position_embeddings=77, hidden_act="quick_gelu", projection_dim=768)
CLIP_G_CONFIG = dict(vocab_size=49408, hidden_size=1280, intermediate_size=5120, num_hidden_layers=32,
                     num_attention_heads=20, max_position_embeddings=77, hidden_act="gelu", projection_dim=1280)

PREFIX_L = ("cond_stage_model.transformer.", "conditioner.embedders.0.transformer.")
PREFIX_G = "conditioner.embedders.1.model."


def load_tokenizer(checkpoint_path: Optional[str] = None):
    """CLIP's tokenizer, offline-capable. Order: a `clip_tokenizer/` folder next to the checkpoint (vocab.json +
    merges.txt), then the local Hugging Face cache (`openai/clip-vit-large-patch14`, fetched once with the helper
    models), and only then the Hub."""
    from transformers import CLIPTokenizer
    if checkpoint_path:
        d = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), TOKENIZER_DIRNAME)
        if os.path.isfile(os.path.join(d, "vocab.json")) and os.path.isfile(os.path.join(d, "merges.txt")):
            return CLIPTokenizer.from_pretrained(d)
    try:
        return CLIPTokenizer.from_pretrained(TOKENIZER_REPO, local_files_only=True)
    except Exception:
        pass
    try:
        return CLIPTokenizer.from_pretrained(TOKENIZER_REPO)
    except Exception as e:
        raise RuntimeError(
            f"The CLIP tokenizer files are not available offline. Download vocab.json and merges.txt from "
            f"huggingface.co/{TOKENIZER_REPO} into a '{TOKENIZER_DIRNAME}' folder next to your SDXL checkpoint "
            f"(or let the model downloader fetch them once). ({e})") from e


def chunk_ids(ids: list, bos: int, eos: int, pad: int, chunks: int = CHUNKS) -> list:
    """Token ids (no special tokens) -> `chunks` rows of 77: [BOS] + up to 75 ids + [EOS] + padding. Longer captions
    are cut at chunks * 75 tokens; shorter ones leave the later chunks empty ([BOS] [EOS] padding)."""
    ids = list(ids)[: chunks * CHUNK_TOKENS]
    rows = []
    for c in range(chunks):
        part = ids[c * CHUNK_TOKENS:(c + 1) * CHUNK_TOKENS]
        rows.append([bos] + part + [eos] + [pad] * (CHUNK_TOKENS - len(part)))
    return rows


def build_encoders(l_cfg: dict = None, g_cfg: dict = None):
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection
    return (CLIPTextModel(CLIPTextConfig(**(l_cfg or CLIP_L_CONFIG))),
            CLIPTextModelWithProjection(CLIPTextConfig(**(g_cfg or CLIP_G_CONFIG))))


def load_encoders(path: str, l_cfg: dict = None, g_cfg: dict = None, dtype=torch.float32):
    """Both CLIPs from a single-file SDXL checkpoint, frozen and in eval mode."""
    from diffusers.loaders.single_file_utils import convert_ldm_clip_checkpoint, convert_open_clip_checkpoint
    from training.families.sdxl.unet import read_prefixed
    ckpt = read_prefixed(path, PREFIX_L + (PREFIX_G,), dtype)
    if not any(k.startswith(PREFIX_G) for k in ckpt):
        raise ValueError(f"{path} has no SDXL text encoders (`conditioner.embedders.*`). Point the text-encoder row "
                         f"at a full SDXL checkpoint.")
    clip_l, clip_g = build_encoders(l_cfg, g_cfg)
    for model, state, what in ((clip_l, convert_ldm_clip_checkpoint(ckpt), "CLIP-L"),
                               (clip_g, convert_open_clip_checkpoint(clip_g, ckpt, prefix=PREFIX_G), "OpenCLIP-bigG")):
        r = model.load_state_dict(state, strict=False)
        missing = [k for k in r.missing_keys if "position_ids" not in k]
        unexpected = [k for k in r.unexpected_keys if "position_ids" not in k]
        if missing or unexpected:
            raise ValueError(f"{path}: {what} weights do not match ({len(missing)} missing, e.g. {missing[:3]}; "
                             f"{len(unexpected)} unexpected, e.g. {unexpected[:3]})")
    del ckpt
    return (clip_l.to(dtype).eval().requires_grad_(False), clip_g.to(dtype).eval().requires_grad_(False))


class SDXLTextEncoders:
    """CLIP-L + OpenCLIP-bigG + tokenizer. encode(captions) -> crossattn (B, 231, 2048) and pooled (B, 1280)."""

    def __init__(self, clip_l, clip_g, tokenizer, device="cpu", dtype=torch.float32):
        self.clip_l, self.clip_g, self.tokenizer = clip_l, clip_g, tokenizer
        self.device, self.dtype = torch.device(device), dtype
        self.clip_l.to(self.device)
        self.clip_g.to(self.device)

    def token_rows(self, caption: str):
        tok = self.tokenizer
        ids = tok(caption or "", add_special_tokens=False, truncation=False)["input_ids"]
        bos, eos = int(tok.bos_token_id), int(tok.eos_token_id)
        return chunk_ids(ids, bos, eos, pad=eos), chunk_ids(ids, bos, eos, pad=0)      # [pad]: L pads EOS, bigG 0

    @torch.no_grad()
    def encode(self, captions: list):
        rows_l, rows_g = zip(*[self.token_rows(c) for c in captions])
        b = len(captions)
        ids_l = torch.tensor([r for rows in rows_l for r in rows], device=self.device)     # (B * 3, 77)
        ids_g = torch.tensor([r for rows in rows_g for r in rows], device=self.device)
        hl = self.clip_l(input_ids=ids_l, output_hidden_states=True).hidden_states[-2]     # [sdxl-cond] penultimate
        out_g = self.clip_g(input_ids=ids_g, output_hidden_states=True)
        hg = out_g.hidden_states[-2]
        cross = torch.cat([hl, hg], dim=-1).reshape(b, CHUNKS * 77, -1)                    # chunks side by side
        pooled = out_g.text_embeds.reshape(b, CHUNKS, -1)[:, 0]                            # first chunk's EOS embedding
        return cross.float(), pooled.float()

    def unload(self):
        self.clip_l = self.clip_g = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
