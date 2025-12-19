# TagScribeR v2

**TagScribeR v2** is a modern, GPU-accelerated local image captioning and dataset management suite. Rebuilt from the ground up using **PySide6** and powered by **Qwen 3-VL** (Vision-Language) models, it offers a "Studio" workflow for preparing AI training datasets.

<img width="512" height="512" alt="Logo" src="https://github.com/user-attachments/assets/c94898af-b851-49f0-9f72-f40587b739b8" />

<img width="3250" height="1888" alt="1" src="https://github.com/user-attachments/assets/fd63cc0e-ad96-44a9-8051-ac360936cae5" />

<img width="3824" height="2056" alt="2" src="https://github.com/user-attachments/assets/2afb6ae8-003e-4632-81cc-39f629060b9f" />

<img width="3829" height="2066" alt="3" src="https://github.com/user-attachments/assets/78423ea1-5d91-4017-95e0-5e7f1ea655e1" />

<img width="3839" height="2066" alt="4" src="https://github.com/user-attachments/assets/69e019fe-16e0-458e-80e4-0065dc901159" />

<img width="3839" height="2067" alt="5" src="https://github.com/user-attachments/assets/f9382a3a-2b9e-43c8-b625-18e4469687c7" />

<img width="3839" height="2060" alt="6" src="https://github.com/user-attachments/assets/bf22af80-d7ac-47aa-bf5b-3025a77f726c" />

## ✨ Key Features

*   **🖼️ Gallery Studio:** Multi-select visual grid, instant tagging, and batch caption editing.
*   **🤖 Qwen 3-VL Captioning:** State-of-the-art vision model integration.
    *   **GPU Accelerated:** Supports NVIDIA (CUDA) and AMD (ROCm) on Windows.
    *   **Real-time Preview:** Watch captions appear as they generate.
    *   **Custom Prompts:** Use templates or natural language (e.g., "Describe the lighting in detail").
    *   **API Mode:** Connect to **LM Studio**, **Ollama**, or **OpenAI** (GPT-4o) to offload processing to another machine or the cloud.
*   **✏️ Batch Editor:** Resize, Crop (with focus points), Rotate, and Convert formats in bulk.
*   **📂 Dataset Manager:** Create, sort, filter, and organize image collections without duplicating files manually.
*   **ℹ️ Metadata Editor:** View and edit EXIF data, specifically targeting Stable Diffusion generation parameters.

---

## 🚀 Installation

### 1. Prerequisites
*   **Python 3.10 or 3.11** installed.
*   **Git** installed.

### 2. Setup
Clone the repository and run the installer:

```cmd
git clone https://github.com/ArchAngelAries/TagScribeR.git
cd TagScribeR
install.bat
```

The installer will ask for your GPU type (NVIDIA, AMD, or CPU).

---

## 🔴 AMD GPU Users (Windows ROCm)

TagScribeR v2 supports AMD GPUs (RX 6000/7000 series) on Windows via **ROCm Nightlies**.

The `install.bat` attempts to install a generic ROCm build (`gfx1100`). However, if you experience issues or have a specific card variant, you may need to install the specific PyTorch build for your architecture manually.

**Please visit "TheRock" (AMD ROCm for Windows) Release Page:**
*   [View Supported GPU List (TheRock Releases)](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)

**Common Manual Install Commands (Run inside `venv`):**

**For RX 7900 Series / 7000 Series (gfx110X):**
```powershell
pip install --force-reinstall --pre torch torchvision torchaudio --index-url https://rocm.nightlies.amd.com/v2/gfx1100-all/
```

**For RX 7900 XTX / XT specific (gfx1100):**
```powershell
pip install --force-reinstall --pre torch torchvision torchaudio --index-url https://rocm.nightlies.amd.com/v2/gfx1100-dgpu/
```

> **⚠️ Important:** Do not run `pip install torch` afterwards, or it will overwrite the AMD version with the generic CPU version.

---

## 🛠️ Usage Guide

### 1. Auto Captioning (Qwen)
1.  Go to the **Auto Caption** tab.
2.  **Download a Model:** Select a preset (e.g., Qwen 2.5-VL-3B) and click **Download**.
3.  **Load Images:** Open a folder containing your dataset.
4.  **Select Images:** Click individual images or "Select All".
5.  **Run:** Click "🚀 Caption Selected".

### 2. Dataset Management
1.  Go to the **Datasets** tab.
2.  **Create Collection:** Click "New" to create a named folder in `Dataset Collections`.
3.  **Filter Source:** Load a source folder and type tags (e.g., `1girl`, `outdoors`) to find specific images.
4.  **Add:** Select the images and click "➕ Add Selected to Collection". This **copies** the images and their text files safely.

### 3. Settings
*   **Themes:** Choose from various Material Design themes (Dark Teal, Dark Amber, Light Blue, etc.).
*   **Defaults:** Set your preferred AI temperature and token limits.

---

## 🤝 Credits & License

*   **GUI Framework:** [PySide6](https://pypi.org/project/PySide6/) & [qt-material](https://pypi.org/project/qt-material/)
*   **AI Backend:** [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) & [Qwen-VL](https://github.com/QwenLM/Qwen-VL)
*   **AMD Support:** [ROCm for Windows](https://github.com/ROCm/TheRock)

Created by **ArchAngelAries**.
```
