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

*   **🖼️ Gallery Studio:** multi-select grid, tag bubbles, quick tags, find-by-text filter and undo/redo for batch caption edits. Saves only what changed.
*   **🤖 AI Captioning with current VLMs:** local models through Hugging Face Transformers: Qwen3-VL, Qwen3.5, Gemma 4, JoyCaption and any other image-text-to-text model.
    *   **AMD first-class:** native ROCm on Windows (RX 7000/9000, Strix Halo), plus NVIDIA CUDA and CPU fallback. Precision is chosen from your GPU's capabilities (bf16 on RDNA3+).
    *   **Fast batches:** models stay loaded between runs, images are batched on the GPU, and the next images are decoded while the GPU works.
    *   **Safe:** choose to overwrite, skip, append to or prepend to existing captions. Overwritten captions are backed up. One bad image never stops a batch.
    *   **API / server mode:** LM Studio, llama.cpp server, Ollama, KoboldCpp, vLLM or cloud APIs. Use this for GGUF models. API keys are stored in Windows Credential Manager.
*   **🏷️ WD Auto-Tagging:** SmilingWolf v3 taggers (EVA02 / ViT / SwinV2 / ConvNext) with general and character thresholds, blacklist and trigger words.
*   **📁 Model discovery:** finds models in `models/`, in your **Stability Matrix** shared `Models/LLM` folder, and in any folders you add.
*   **✏️ Batch Editor:** resize, crop, rotate and convert formats. Copies are saved by default and keep their captions. EXIF rotation is respected.
*   **📂 Dataset Collections:** gather images and captions into training folders without overwriting anything. Deletes go to the Recycle Bin.
*   **ℹ️ Metadata Editor:** view and edit EXIF / PNG text, including Stable Diffusion generation parameters.

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
*   **A model says it needs custom code:** enable *Allow custom model code* in Model options, but only for sources you trust.

Architecture and developer notes: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Run the tests with `venv\Scripts\python -m pytest`.

---

## 🤝 Credits & License

*   **GUI Framework:** [PySide6](https://pypi.org/project/PySide6/) & [qt-material](https://pypi.org/project/qt-material/)
*   **AI Backend:** [HuggingFace Transformers](https://huggingface.co/docs/transformers/index), [ONNX Runtime](https://onnxruntime.ai/), [WD Taggers](https://huggingface.co/SmilingWolf)
*   **Environment conventions:** aligned with the [Fizgig](https://github.com/shootthesound/Fizgig) trainer's ROCm stack
*   **AMD Support:** [ROCm for Windows](https://github.com/ROCm/TheRock)

Created by **ArchAngelAries**. Code Assisted by **Google's Gemini Pro 3**.
```