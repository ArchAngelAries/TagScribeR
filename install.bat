@echo off
title TagScribeR v2 Installer
cls

echo ===================================================
echo       TagScribeR v2 - Environment Installer
echo ===================================================
echo.

REM 1. Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10 or 3.11 from python.org.
    pause
    exit /b
)

REM 2. Create Venv
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
) else (
    echo [INFO] Virtual environment already exists.
)

REM 3. Activate Venv
call venv\Scripts\activate

REM 4. Install Core Requirements
echo.
echo [INFO] Installing core dependencies (GUI, Utils)...
pip install -r requirements.txt
echo.

REM 5. GPU Selection Menu
echo ===================================================
echo           SELECT YOUR HARDWARE ACCELERATOR
echo ===================================================
echo 1. NVIDIA GPU (CUDA 12.4) - Recommended for RTX Cards
echo 2. AMD GPU (ROCm) - For RX 6000/7000 Series (Windows)
echo 3. CPU Only (Slow, fallback)
echo.
set /p gpu_choice="Enter number (1-3): "

if "%gpu_choice%"=="1" goto install_nvidia
if "%gpu_choice%"=="2" goto install_amd
if "%gpu_choice%"=="3" goto install_cpu
goto end

:install_nvidia
echo.
echo [INFO] Installing PyTorch for NVIDIA (CUDA 12.4)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
goto end

:install_amd
echo.
echo [INFO] Installing PyTorch for AMD (ROCm Nightly)...
echo [NOTE] Using generic gfx1100 install. If you have a specific card,
echo        please check the README for "TheRock" github links.
echo.
pip install --pre torch torchvision torchaudio --index-url https://rocm.nightlies.amd.com/v2/gfx1100-all/
goto end

:install_cpu
echo.
echo [INFO] Installing PyTorch for CPU...
pip install torch torchvision torchaudio
goto end

:end
echo.
echo ===================================================
echo    Installation Complete! Run start.bat to launch.
echo ===================================================
pause