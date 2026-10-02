"""Local vision-language models through Hugging Face Transformers.

One generic provider covers every model that ships a processor chat template
and loads with ``AutoModelForImageTextToText`` — Qwen2.5-VL / Qwen3-VL,
Gemma 3, LLaVA-family models such as JoyCaption, SmolVLM, Idefics3, InternVL
(-hf), Mistral 3 / Pixtral and more. No model-family-specific helper packages
are required.

Hardware handling (see core/hardware.py): device and dtype are resolved from
probed capabilities — bf16 on RDNA3+/Ampere+, fp16 otherwise, fp32 on CPU —
and attention defaults to PyTorch SDPA, which uses AOTriton kernels on ROCm
and flash/mem-efficient kernels on CUDA without extra packages.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Sequence

from PIL import Image

from core import captions, hardware
from inference.base import (CAP_BATCH, CAP_PROMPT, CAP_SAMPLING, CAP_SYSTEM_PROMPT, CAP_UNLOADABLE,
                            Cancelled, CaptionRequest, InferenceError, ProgressFn, Provider,
                            ProviderSpec)

log = logging.getLogger(__name__)


def read_local_config(model: str) -> dict:
    p = Path(model) / "config.json"
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
    return {}


def needs_remote_code(cfg: dict) -> bool:
    """True if the checkpoint relies on custom modeling code not built into transformers."""
    if not cfg.get("auto_map"):
        return False
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        return cfg.get("model_type") not in CONFIG_MAPPING
    except Exception:
        return True


def is_cached_or_local(model: str) -> bool:
    if Path(model).is_dir():
        return True
    from inference.models import hf_fully_cached
    return hf_fully_cached(model)


class TransformersVLM(Provider):
    capabilities = frozenset({CAP_PROMPT, CAP_SYSTEM_PROMPT, CAP_SAMPLING, CAP_BATCH, CAP_UNLOADABLE})

    def __init__(self, spec: ProviderSpec):
        super().__init__(spec)
        self.model = None
        self.processor = None
        self.device: hardware.DeviceInfo | None = None
        self.dtype = None
        self.preferred_chunk = max(1, int(spec.opt("batch_size", 1)))

    @property
    def display_name(self) -> str:
        return Path(self.spec.model).name if Path(self.spec.model).is_dir() else self.spec.model

    @property
    def loaded(self) -> bool:
        return self.model is not None

    # -- loading -----------------------------------------------------------
    def load(self, report: ProgressFn, cancel: threading.Event | None = None) -> None:
        if self.model is not None:
            return
        if not hardware.torch_installed():
            raise InferenceError("PyTorch is not installed, so local models can't run.",
                                 "Run install.bat, or use an API / LM Studio endpoint instead.")
        from transformers import AutoProcessor

        model_id = self.spec.model
        if not is_cached_or_local(model_id) and not self.spec.opt("allow_download", False):
            raise InferenceError(f"Model '{model_id}' has not been downloaded yet.",
                                 "Use the Download button first (the size is shown before anything is fetched).")

        cfg = read_local_config(model_id)
        trust = bool(self.spec.opt("trust_remote_code", False))
        if cfg and needs_remote_code(cfg) and not trust:
            raise InferenceError(
                f"'{self.display_name}' uses custom code ({cfg.get('model_type', 'unknown type')}) that "
                "isn't built into Transformers.",
                "Only if you trust its source, enable 'Allow custom model code' in the model options.")

        self.device = hardware.resolve_device(self.spec.opt("device", "auto"))
        self.dtype = hardware.resolve_dtype(self.device, self.spec.opt("dtype", "auto"))
        hardware.configure_torch_backends(self.device)
        report(f"Loading {self.display_name} on {self.device.label} ({str(self.dtype).replace('torch.', '')})…")

        common = {"trust_remote_code": trust}
        local_only = {} if Path(model_id).is_dir() else {"local_files_only": not self.spec.opt("allow_download", False)}

        t0 = time.monotonic()
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, **common, **local_only)
        except Exception as e:
            raise InferenceError(f"Could not load the processor for {self.display_name}: {e}",
                                 "Make sure the folder contains a complete Hugging Face model "
                                 "(config.json, preprocessor/processor config, tokenizer files).") from e
        tok = getattr(self.processor, "tokenizer", None)
        if tok is not None:
            tok.padding_side = "left"  # required for correct batched generation
        if hasattr(self.processor, "padding_side"):
            self.processor.padding_side = "left"

        kwargs = dict(common, **local_only, low_cpu_mem_usage=True)
        dtype_key = "dtype" if _tf_version() >= (4, 56) else "torch_dtype"
        kwargs[dtype_key] = self.dtype

        attn = self.spec.opt("attention", "auto")
        if attn == "auto":
            attn = "sdpa"
        if attn == "flash_attention_2" and not hardware.optional_feature("flash_attn"):
            log.warning("flash_attn is not installed; using SDPA.")
            attn = "sdpa"
        kwargs["attn_implementation"] = attn

        quant = self.spec.opt("quantization", "none")
        if quant in ("8bit", "4bit"):
            if not self.device.is_gpu or not hardware.optional_feature("bitsandbytes"):
                raise InferenceError(f"{quant} loading needs a GPU and the bitsandbytes package.",
                                     "Install with: install.bat --with-bnb   (or set Quantization to None).")
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = (
                BitsAndBytesConfig(load_in_8bit=True) if quant == "8bit" else
                BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=self.dtype,
                                   bnb_4bit_quant_type="nf4"))

        if self.device.is_gpu:
            kwargs["device_map"] = {"": self.device.id}  # stream weights straight onto the GPU

        self.model = self._load_model(model_id, kwargs, attn)
        if "device_map" not in kwargs and self.device.backend != "cpu":
            self.model.to(self.device.id)
        self.model.eval()
        if cancel is not None and cancel.is_set():
            self.unload()
            raise Cancelled()
        mem = ""
        if self.device.is_gpu:
            info = hardware.memory_info(self.device)
            if info:
                mem = f" — {(info[1] - info[0]) / 2**30:.1f} / {info[1] / 2**30:.1f} GB VRAM in use"
        report(f"Model ready in {time.monotonic() - t0:.0f}s{mem}")

    def _load_model(self, model_id: str, kwargs: dict, attn: str):
        import torch
        from transformers import AutoModelForImageTextToText
        classes = [AutoModelForImageTextToText]
        import transformers
        # Gemma 4 registers under AutoModelForMultimodalLM (Transformers 5.x);
        # AutoModelForVision2Seq only exists in 4.x.
        for name in ("AutoModelForMultimodalLM", "AutoModelForVision2Seq"):
            cls = getattr(transformers, name, None)
            if cls is not None:
                classes.append(cls)
        last: Exception | None = None
        for cls in classes:
            for attempt_attn in dict.fromkeys([attn, "eager"]):
                kwargs["attn_implementation"] = attempt_attn
                try:
                    return cls.from_pretrained(model_id, **kwargs)
                except torch.OutOfMemoryError as e:
                    hardware.free_memory()
                    raise InferenceError(
                        f"Not enough GPU memory to load {self.display_name}.",
                        "Try a smaller model, 8-bit/4-bit quantization, or close other GPU apps "
                        "(ComfyUI/Forge keep models in VRAM).") from e
                except ValueError as e:
                    msg = str(e)
                    last = e
                    if "attn_implementation" in msg or ("does not support" in msg and "attention" in msg):
                        log.info("Attention '%s' unsupported for this model; retrying.", attempt_attn)
                        continue
                    if "Unrecognized configuration class" in msg:
                        break  # try the next auto class
                    raise InferenceError(f"Could not load {self.display_name}: {e}") from e
                except Exception as e:
                    raise InferenceError(f"Could not load {self.display_name}: {e}") from e
        raise InferenceError(f"{self.display_name} is not a supported vision-language model: {last}",
                             "It may be a text-only model, or need a newer Transformers version.")

    def unload(self) -> None:
        if self.model is None and self.processor is None:
            return
        self.model = None
        self.processor = None
        hardware.free_memory()

    # -- inference ---------------------------------------------------------
    def _conversation(self, img: Image.Image, req: CaptionRequest) -> list[dict]:
        conv = []
        if req.system_prompt.strip():
            conv.append({"role": "system", "content": [{"type": "text", "text": req.system_prompt.strip()}]})
        conv.append({"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": req.prompt}]})
        return conv

    def generate(self, images: Sequence[Image.Image], request: CaptionRequest,
                 cancel: threading.Event | None = None) -> list[str | Exception]:
        if self.model is None:
            raise InferenceError("Model is not loaded.")
        import torch
        try:
            return self._generate_batch(list(images), request, cancel)
        except torch.OutOfMemoryError:
            hardware.free_memory()
            if len(images) > 1:
                log.warning("Out of memory with batch of %d; retrying one image at a time.", len(images))
                out: list[str | Exception] = []
                for img in images:
                    out.extend(self.generate([img], request, cancel))
                return out
            return [InferenceError("Ran out of GPU memory on this image.",
                                   "Lower 'Max image size' or 'Max tokens', or use quantization.")]

    def _generate_batch(self, images: list[Image.Image], req: CaptionRequest,
                        cancel: threading.Event | None) -> list[str | Exception]:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        convs = [self._conversation(img, req) for img in images]
        template_kwargs = {}
        template = getattr(self.processor, "chat_template", None) or ""
        if isinstance(template, str) and "enable_thinking" in template:
            # Reasoning burns tokens and rarely improves captions; off unless the user opts in.
            template_kwargs["enable_thinking"] = bool(self.spec.opt("thinking", False))
        if len(convs) > 1:
            if _tf_version() >= (5, 0):
                template_kwargs["processor_kwargs"] = {"padding": True}
            else:
                template_kwargs["padding"] = True
        inputs = self.processor.apply_chat_template(
            convs if len(convs) > 1 else convs[0], add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", **template_kwargs)
        inputs = inputs.to(self.model.device)
        if "pixel_values" in inputs and inputs["pixel_values"].is_floating_point():
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

        p = req.params
        gen = {"max_new_tokens": p.max_new_tokens, "repetition_penalty": p.repetition_penalty}
        if p.temperature > 0:
            gen.update(do_sample=True, temperature=p.temperature, top_p=p.top_p)
            if p.top_k > 0:
                gen["top_k"] = p.top_k
        else:
            gen.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        if p.seed is not None:
            torch.manual_seed(p.seed)

        class _Cancel(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                flag = bool(cancel is not None and cancel.is_set())
                return torch.full((input_ids.shape[0],), flag, dtype=torch.bool, device=input_ids.device)

        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen, stopping_criteria=StoppingCriteriaList([_Cancel()]))
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        texts = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if req.strip_thinking:
            texts = [captions.clean_model_output(t) for t in texts]
        return [t.strip() if t.strip() else InferenceError("The model returned an empty caption.") for t in texts]


def _tf_version() -> tuple[int, int]:
    try:
        import transformers
        major, minor = transformers.__version__.split(".")[:2]
        return int(major), int(minor)
    except Exception:
        return (0, 0)
