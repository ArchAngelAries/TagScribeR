#!/usr/bin/env bash
# TagScribeR installer (Linux). Creates venv/ with Python 3.12, then runs
# tools/install.py which detects the GPU and installs the matching PyTorch build
# (AMD ROCm multi-arch / NVIDIA CUDA / CPU). Options pass through, e.g.
#   ./install.sh --backend rocm --arch gfx1100
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for c in python3.12 python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) else 1)'; then
        PY="$c"; break
    fi
done
if [[ -z "$PY" ]]; then
    echo "ERROR: Python 3.12 not found (try your distro's python3.12 package or uv python install 3.12)." >&2
    exit 1
fi

# Qt needs libxcb-cursor0 on X11 / Debian-based distros.
if command -v dpkg >/dev/null 2>&1 && ! dpkg -s libxcb-cursor0 >/dev/null 2>&1; then
    echo "Note: Qt needs libxcb-cursor0. Install it with: sudo apt-get install libxcb-cursor0"
fi

if [[ -x venv/bin/python ]] && ! venv/bin/python -c 'import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) else 1)'; then
    echo "Existing venv/ uses another Python version; moving it to venv-old/."
    [[ -e venv-old ]] && { echo "ERROR: venv-old/ exists; remove it first." >&2; exit 1; }
    mv venv venv-old
fi
[[ -x venv/bin/python ]] || "$PY" -m venv venv

venv/bin/python -m pip install --upgrade pip uv
venv/bin/python tools/install.py "$@"
echo "Installation complete. Launch with ./start.sh"
