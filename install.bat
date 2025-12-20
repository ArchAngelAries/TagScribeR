@echo off
title TagScribeR v2.1 Smart Installer
cls

echo ===================================================
echo       TagScribeR v2.1 - Auto-Detect Installer
echo ===================================================
echo.

REM --- 1. Python Check ---
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10 or 3.11.
    pause
    exit /b
)

REM --- 2. Venv Setup ---
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate

REM --- 3. Core Requirements ---
echo.
echo [INFO] Installing GUI and AI dependencies...
pip install -r requirements.txt
echo.

REM --- 4. Hardware Auto-Detection ---
echo [INFO] Detecting Hardware...

REM Check for NVIDIA (Linear Logic)
powershell -Command "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name" | findstr /i "NVIDIA" >nul
if %errorlevel% equ 0 goto nvidia_found

REM Check for AMD
powershell -Command "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name" | findstr /i "Radeon AMD" >nul
if %errorlevel% equ 0 goto amd_found

REM Fallback
goto cpu_fallback

:nvidia_found
echo.
echo [DETECTED] NVIDIA GPU.
REM Check specifically for RTX 50 Series (Blackwell) -> CUDA 12.8
powershell -Command "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name" | findstr /i "RTX 50" >nul
if %errorlevel% equ 0 (
    echo [INFO] RTX 50 Series detected!
    echo [INFO] Installing PyTorch Nightly (CUDA 12.8) + Required Deps...
    
    REM Install critical dependencies first to prevent 'module not found' errors
    pip install torchgen sympy networkx jinja2 fsspec pyyaml
    
    REM Force install the nightly build
    pip install --force-reinstall --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
) else (
    echo [INFO] Standard NVIDIA card detected. Installing CUDA 12.4 (cu124)...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
)
goto finish

:amd_found
echo.
echo [DETECTED] AMD Radeon GPU.

REM Default to 7000 series
set "amd_url=https://rocm.nightlies.amd.com/v2/gfx110X-all/"
set "arch=gfx110X-all (RX 7000 Series / 780M)"

REM Check RX 9000
powershell -Command "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name" | findstr /i "RX 90" >nul
if %errorlevel% equ 0 (
    set "amd_url=https://rocm.nightlies.amd.com/v2/gfx120X-all/"
    set "arch=gfx120X (RX 9000 Series)"
)

REM Check Strix Halo
powershell -Command "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name" | findstr /i "Strix Halo" >nul
if %errorlevel% equ 0 (
    set "amd_url=https://rocm.nightlies.amd.com/v2/gfx1151/"
    set "arch=gfx1151 (Strix Halo)"
)

REM Check Workstation
powershell -Command "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name" | findstr /i "MI300" >nul
if %errorlevel% equ 0 (
    set "amd_url=https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/"
    set "arch=gfx94X (MI300)"
)

echo [INFO] Architecture detected: %arch%
echo [INFO] Installing ROCm SDK...
pip install --index-url %amd_url% "rocm[libraries,devel]"
echo [INFO] Installing PyTorch for ROCm...
pip install --index-url %amd_url% --pre torch torchvision torchaudio
goto finish

:cpu_fallback
echo [WARNING] No dedicated GPU detected.
echo [INFO] Installing CPU-only PyTorch...
pip install torch torchvision torchaudio
goto finish

:finish
echo.
echo ===================================================
echo    Installation Complete! Run start.bat to launch.
echo ===================================================
pause