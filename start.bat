@echo off
title TagScribeR
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found. Please run install.bat first.
    pause
    exit /b 1
)
REM ROCm / allocator environment defaults are applied inside the app
REM (core/hardware.py) before PyTorch loads, so no extra env script is needed.
"venv\Scripts\python.exe" main.py %*
if errorlevel 1 (
    echo.
    echo TagScribeR exited with an error. Details: user_data\logs\tagscriber.log
    pause
)
