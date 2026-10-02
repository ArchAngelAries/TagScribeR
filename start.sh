#!/usr/bin/env bash
cd "$(dirname "$0")"
if [[ ! -x venv/bin/python ]]; then
    echo "Virtual environment not found. Run ./install.sh first." >&2
    exit 1
fi
exec venv/bin/python main.py "$@"
