@echo off
title TagScribeR v2
if not exist "venv" (
    echo [ERROR] Virtual environment not found. Please run install.bat first.
    pause
    exit /b
)

call venv\Scripts\activate
python main.py
pause