@echo off
REM TagScribeR installer (Windows). Finds Python 3.12, creates venv\, then hands
REM off to tools\install.py which detects the GPU and installs the matching
REM PyTorch build (AMD ROCm for Windows / NVIDIA CUDA / CPU) plus dependencies.
REM
REM Options are passed through, e.g.:
REM   install.bat --backend rocm --arch gfx1100
REM   install.bat --experimental      (AMD: newest unpinned ROCm nightlies)
REM   install.bat --with-bnb          (also install bitsandbytes for 8/4-bit loading)
setlocal enabledelayedexpansion
title TagScribeR Installer
cd /d "%~dp0"

echo ============================================================
echo   TagScribeR Installer
echo ============================================================
echo.

set "PY312="
where py >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%P in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do set "PY312=%%P"
)
if not defined PY312 (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        if not defined PY312 (
            "%%P" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) else 1)" >nul 2>&1
            if not errorlevel 1 set "PY312=%%P"
        )
    )
)
if not defined PY312 (
    echo ERROR: Python 3.12 was not found.
    echo Install it from https://www.python.org/downloads/windows/  ^(or: py install 3.12^)
    echo then re-run this script.
    pause
    exit /b 1
)
echo Using Python 3.12: !PY312!

if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) else 1)" >nul 2>&1
    if errorlevel 1 (
        echo.
        echo Your existing venv\ uses a different Python version.
        echo It will be renamed to venv-old\ ^(not deleted^) and a fresh 3.12 venv created.
        set /p "OK=Continue? (Y/n): "
        if /I "!OK!"=="n" exit /b 1
        if exist "venv-old" (
            echo ERROR: venv-old\ already exists. Remove or rename it first.
            pause
            exit /b 1
        )
        move "venv" "venv-old" >nul
    )
)

if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    "!PY312!" -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

"venv\Scripts\python.exe" -m pip install --upgrade pip uv
if errorlevel 1 goto :failed
"venv\Scripts\python.exe" tools\install.py %*
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo   Installation complete. Launch with start.bat
echo ============================================================
pause
exit /b 0

:failed
echo.
echo ERROR: Installation failed. See the messages above.
pause
exit /b 1
