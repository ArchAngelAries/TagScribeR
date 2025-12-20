@echo off
title TagScribeR Debug Mode
echo ===================================================
echo       TagScribeR Debug Launcher
echo ===================================================
echo.
echo This window will stay open after a crash.
echo Please copy any error messages below and report them.
echo.

call venv\Scripts\activate
python main.py

echo.
echo ===================================================
echo       PROGRAM CRASHED / CLOSED
echo ===================================================
pause