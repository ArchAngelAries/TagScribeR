@echo off
title TagScribeR Updater
cls

echo ===================================================
echo          TagScribeR Update Utility
echo ===================================================
echo.

REM 1. Check for Git
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed or not in PATH.
    echo Please install Git to use the auto-updater.
    pause
    exit /b
)

REM 2. Pull latest code
echo [INFO] Pulling latest changes from GitHub...
git pull origin main
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Git pull failed. You might have local file changes.
    echo Try backing up your config files and re-cloning if this persists.
    pause
    exit /b
)

REM 3. Check Venv
if not exist "venv" (
    echo [ERROR] Virtual environment not found. Please run install.bat first.
    pause
    exit /b
)

REM 4. Update Dependencies (Safe Mode)
echo.
echo [INFO] Checking for new dependencies...
echo (This will NOT overwrite your existing PyTorch/GPU setup)
call venv\Scripts\activate
pip install -r requirements.txt

echo.
echo ===================================================
echo           Update Complete! 
echo ===================================================
pause