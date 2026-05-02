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
  echo "  python -m pip install -r requirements.txt"
  echo
  echo "Then run this shortcut again."
  read -r "?Press Enter to close..."
  exit 1
fi

GODOT_ENSURE="$SCRIPT_DIR/scripts/ensure_godot_macos.sh"
GODOT_BIN="/Applications/Godot.app/Contents/MacOS/Godot"
if [[ -x "$GODOT_ENSURE" ]]; then
  "$GODOT_ENSURE" || echo "Godot companion unavailable; continuing with pygame fallback."
fi
if [[ -x "$GODOT_BIN" ]]; then
  export CFAST_GODOT_BIN="$GODOT_BIN"
fi

exec "$PYTHON_BIN" -m cfast_trainer "$@"
