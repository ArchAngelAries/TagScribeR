@echo off
title TagScribeR Updater
cd /d "%~dp0"

git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed or not in PATH.
    pause
    exit /b 1
)

echo [INFO] Pulling latest changes...
git pull --ff-only
if errorlevel 1 (
    echo.
    echo [ERROR] git pull failed - you may have local changes. Your settings in
    echo user_data\ are not tracked by git and are safe.
    pause
    exit /b 1
)

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found. Please run install.bat first.
    pause
    exit /b 1
)

echo [INFO] Updating dependencies (your PyTorch / GPU build is left untouched)...
"venv\Scripts\python.exe" tools\install.py --deps-only
if errorlevel 1 (
    pause
    exit /b 1
)
echo.
echo Update complete.
pause
