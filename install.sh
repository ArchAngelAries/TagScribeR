#!/bin/bash

echo "==================================================="
echo "       TagScribeR v2.1 - Linux Installer"
echo "==================================================="
echo ""

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found."
    exit 1
fi

# 2. Check System Dependencies (The fix the user mentioned)
echo "[INFO] Checking for Qt dependencies..."
if dpkg -l | grep -q libxcb-cursor0; then
    echo " - libxcb-cursor0 is installed."
else
    echo " - Missing libxcb-cursor0. Attempting install (requires sudo)..."
    sudo apt-get update && sudo apt-get install -y libxcb-cursor0 libxcb-xinerama0
fi

# 3. Create Venv
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv
fi

# 4. Activate
source venv/bin/activate

# 5. Install Core Requirements (No Torch yet)
echo ""
echo "[INFO] Installing core dependencies..."
pip install -r requirements.txt

# 6. GPU Detection
echo ""
echo "[INFO] Detecting Hardware..."

if command -v nvidia-smi &> /dev/null; then
    echo "[DETECTED] NVIDIA GPU"
    echo "[INFO] Installing PyTorch (CUDA 12.4)..."
    pip install torch torchvision torchaudio
elif command -v rocminfo &> /dev/null; then
    echo "[DETECTED] AMD GPU (ROCm)"
    echo "[INFO] Installing PyTorch (ROCm 6.2)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
else
    echo "[WARNING] No dedicated GPU detected. Defaulting to CPU."
    pip install torch torchvision torchaudio
fi

echo ""
echo "==================================================="
echo "   Installation Complete! Run: python3 main.py"
echo "==================================================="