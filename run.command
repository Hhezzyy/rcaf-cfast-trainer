#!/bin/zsh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"

cd "$SCRIPT_DIR" || exit 1
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing virtual environment at .venv/bin/python"
  echo "From $SCRIPT_DIR run:"
  echo "  python3 -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  python -m pip install --upgrade pip"
  echo "  python -m pip install -r requirements-3d.txt"
  echo
  echo "Then run this shortcut again."
  read -r "?Press Enter to close..."
  exit 1
fi

exec "$PYTHON_BIN" -m cfast_trainer "$@"
