# TagScribeR

**TagScribeR v2** is a modern, GPU-accelerated local image captioning and dataset management suite. Rebuilt from the ground up using **PySide6** and powered by modern local Vision-Language Models (Qwen3-VL, Qwen3.5, Gemma 4, JoyCaption) with optional API support, it offers a "Studio" workflow for preparing AI training datasets.

<img width="512" height="512" alt="Logo" src="https://github.com/user-attachments/assets/c94898af-b851-49f0-9f72-f40587b739b8" />

<img width="3250" height="1888" alt="1" src="https://github.com/user-attachments/assets/fd63cc0e-ad96-44a9-8051-ac360936cae5" />

<img width="3824" height="2056" alt="2" src="https://github.com/user-attachments/assets/2afb6ae8-003e-4632-81cc-39f629060b9f" />

<img width="3831" height="2058" alt="Screenshot 2025-12-19 123602" src="https://github.com/user-attachments/assets/592ba435-13be-4e1a-977d-113354e8fdc0" />

<img width="3829" height="2066" alt="3" src="https://github.com/user-attachments/assets/78423ea1-5d91-4017-95e0-5e7f1ea655e1" />

<img width="3839" height="2066" alt="4" src="https://github.com/user-attachments/assets/69e019fe-16e0-458e-80e4-0065dc901159" />

<img width="3839" height="2067" alt="5" src="https://github.com/user-attachments/assets/f9382a3a-2b9e-43c8-b625-18e4469687c7" />

<img width="3839" height="2060" alt="6" src="https://github.com/user-attachments/assets/bf22af80-d7ac-47aa-bf5b-3025a77f726c" />

## ✨ Key Features

*   **🖼️ Dataset workspace (Gallery):** a fast grid that handles thousands of images, with zoomable thumbnails (Ctrl + mouse wheel), search-style filters (`tag:1girl -tag:blurry missing:caption res:<768`), saved filters per folder, sorting, multi-select editing of shared tags, batch tag and text operations, tag statistics with rename/merge, and undo/redo. One folder is shared by every tab, and nothing is written until you Save.
*   **🤖 AI captioning with current VLMs:** local models through Hugging Face Transformers (Qwen3-VL, Qwen3.5, Gemma 4, JoyCaption and other image-text-to-text models) or any OpenAI-compatible server (LM Studio, llama.cpp, Ollama, cloud APIs; use this for GGUF models).
    *   **AMD first-class:** native ROCm on Windows (RX 7000/9000, Strix Halo), plus NVIDIA CUDA and CPU fallback, with precision chosen from your GPU's capabilities.
    *   **Fast and safe:** models stay loaded between runs and images are batched on the GPU. You choose to overwrite, skip, append or prepend; overwritten captions are backed up, and one bad image never stops a batch.
    *   **Review before applying:** compare each AI caption with the current one (word-level diff), edit, and accept or reject per image or all at once.
    *   **Recipes:** a subject / trigger word the model must use (optionally starting every caption), and each image's existing tags passed as hints (tag first, then caption).
    *   **Presets:** built-in instruction presets plus your own, with save, rename, delete and import/export.
*   **🏷️ WD auto-tagging:** SmilingWolf v3 taggers with general and character thresholds, blacklist, trigger words and named presets.
*   **🩺 Dataset health:** exact and near-duplicate detection, blurry and low-resolution flags, and an aspect-ratio bucket preview for your training resolution.
*   **✏️ Image Editor:** live before/after preview; rotate, flip, resize (never upscales by accident), crop to an exact size or an aspect ratio with focus points, convert. Copies are saved by default and keep their captions, and colour profiles and EXIF are preserved.
*   **ℹ️ Metadata tools:** a privacy audit (GPS, device serials, AI prompts and workflows, editing history), lossless cleanup that never re-encodes pixels, honest authorship templates, and embedded A1111/ComfyUI/NovelAI/InvokeAI prompts turned into captions.
*   **🎓 Native LoRA training (Train tab, experimental):** train a LoRA on the open dataset with the [Fizgig](https://github.com/shootthesound/Fizgig) trainer's engine: its presets (plus your own, and Fizgig preset files), Adaptive LR, EMA, a per-image loss watch with a Problem Images window and AI recaptioning of stuck images, live sample previews, pause / resume and a run queue. First family: Qwen Image 2.1; SDXL / Pony / Illustrious / NoobAI, Anima, Krea 2, MiniMax H3 and FLUX.2 Klein are planned ([plan](docs/TRAINING_PLAN.md)).
*   **📂 Dataset Collections:** gather finished images and captions into training folders. Nothing is overwritten, and deletes go to the Recycle Bin.
*   **⌨️ Built for speed and accessibility:** a Ctrl+K command palette, a context-aware Help Center (F1), tooltips everywhere, and interface scaling up to 200%.
*   **📁 Model discovery:** finds models in `models/`, in your **Stability Matrix** shared `Models/LLM` folder, and in any folders you add.

---

## 🚀 Installation

**Requirements:** Windows 10/11 or Linux, **Python 3.12**, Git.

```cmd
git clone https://github.com/ArchAngelAries/TagScribeR.git
cd TagScribeR
install.bat
```

The installer creates `venv\`, detects your GPU and installs the matching PyTorch build:

| GPU | PyTorch build |
|---|---|
| AMD Radeon (Windows) | AMD's native ROCm wheels, `torch 2.12.0+rocm7.15` for your exact chip (e.g. `gfx1100` for RX 7900) |
| AMD Radeon (Linux) | AMD multi-arch ROCm 7.14 wheels |
| NVIDIA | CUDA 12.8 (RTX 20–50 series) |
| None | CPU (captioning works but is slow; API mode is recommended) |

Useful options:

```cmd
install.bat --arch gfx1201        :: force an AMD architecture
install.bat --backend cpu         :: skip GPU detection
install.bat --experimental        :: AMD: newest unpinned ROCm nightlies
install.bat --with-bnb            :: add bitsandbytes for 8-bit / 4-bit loading
```

Launch with **`start.bat`** (or `./start.sh`). Update with **`update.bat`**: it pulls the code and refreshes dependencies without touching your PyTorch build.

> Upgrading from TagScribeR 2.1? Run `install.bat` again. Your old Python 3.10/3.11 venv is renamed to `venv-old\` (not deleted). Your settings, quick tags and API presets are imported automatically on first launch. API keys move into Windows Credential Manager. After confirming that, you can delete the old `api_presets.json`, which held them in plain text.

---

## 🛠️ Usage Guide

### Auto Captioning
1.  Open the **Auto Caption** tab and pick a model. Installed models are listed first. ☁️ entries can be downloaded; the size is shown before anything downloads.
2.  Open a folder, then select images (or **Select Uncaptioned**).
3.  Pick an instruction preset (training caption, booru tags, character, clothing, composition…) or write your own.
4.  Choose what happens to existing captions, then press **Caption** (Ctrl+Enter). Esc aborts; finished images are already saved.

**Which model?** (20 GB GPU, bf16)
*   **Qwen3-VL 8B / 4B:** detailed, fast and the most AMD-friendly. Recommended default.
*   **Qwen3.5 4B / 9B:** newest Qwen (2026). Strong detail, but slower on AMD for now (see Troubleshooting).
*   **Gemma 4 E4B / E2B:** fluent natural-language captions.
*   **JoyCaption Beta One:** built for diffusion training captions.
*   Larger models (27B+): run a GGUF quant in LM Studio and use the **API / Server** tab.

### Dataset Management
1.  In **Datasets**, create a collection and load a source folder.
2.  Filter by tag or file name, select images, and **Add to Collection**. Images and captions are copied, and name clashes are renamed, never overwritten.

---

## 🔧 Troubleshooting

*   **Logs:** `user_data\logs\tagscriber.log`. Settings → *System & diagnostics* shows the detected GPU, PyTorch, ROCm and Transformers versions (*Copy report*).
*   **Recovering a caption:** `user_data\caption_backups\<date>\` holds the first version of every caption overwritten that day.
*   **First image is slow (~20 s) on AMD:** one-time kernel selection after a model loads. Later images take about 1 s.
*   **Qwen3.5 logs "falling back to its reference PyTorch implementation":** expected on ROCm. Its linear-attention kernels have no AMD build yet. Output is correct. Use batch size 2–4, or Qwen3-VL for top speed.
*   **Out of GPU memory:** lower *Max image size* (Advanced sampling), use a smaller model, or close ComfyUI/Forge, which keep models in VRAM. TagScribeR automatically retries a failed batch one image at a time.
*   **Training:** each run lives in `<output folder>/<LoRA name>/` (checkpoints, `sample/`, `loss_log/`, `run.log`). AdamW 8-bit needs the `bitsandbytes` package; without it runs fall back to AdamW. Help → *Training a LoRA* explains every setting.
*   **A model says it needs custom code:** enable *Allow custom model code* in Model options, but only for sources you trust.

Architecture and developer notes: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Run the tests with `venv\Scripts\python -m pytest`.

---

## 🤝 Credits & License

*   **GUI Framework:** [PySide6](https://pypi.org/project/PySide6/) & [qt-material](https://pypi.org/project/qt-material/)
*   **AI Backend:** [HuggingFace Transformers](https://huggingface.co/docs/transformers/index), [ONNX Runtime](https://onnxruntime.ai/), [WD Taggers](https://huggingface.co/SmilingWolf)
*   **Training engine:** ported from [Fizgig](https://github.com/shootthesound/Fizgig) (Apache-2.0, © 2026 Peter Neill), whose ROCm stack the environment also follows. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
*   **AMD Support:** [ROCm for Windows](https://github.com/ROCm/TheRock)

Created by **ArchAngelAries**. Code Assisted by **Google's Gemini Pro 3**.
```