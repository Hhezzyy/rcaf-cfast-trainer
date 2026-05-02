#!/bin/zsh
set -euo pipefail

GODOT_VERSION="4.6.2"
GODOT_APP="/Applications/Godot.app"
GODOT_BIN="$GODOT_APP/Contents/MacOS/Godot"
DOWNLOAD_URL="https://github.com/godotengine/godot/releases/download/${GODOT_VERSION}-stable/Godot_v${GODOT_VERSION}-stable_macos.universal.zip"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_PATH="$REPO_ROOT/godot/cfast_3d"

function import_project() {
  if [[ -f "$PROJECT_PATH/project.godot" ]]; then
    echo "Importing CFAST Godot project..."
    "$GODOT_BIN" --headless --path "$PROJECT_PATH" --import
  fi
}

function verify_godot() {
  echo "Godot binary: $GODOT_BIN"
  "$GODOT_BIN" --version
}

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This helper installs Godot to /Applications on macOS only."
  exit 1
fi

if [[ -x "$GODOT_BIN" ]]; then
  verify_godot
  import_project
  exit 0
fi

if [[ -e "$GODOT_APP" ]]; then
  echo "$GODOT_APP exists, but $GODOT_BIN is not executable."
  echo "Move or repair the existing app, then rerun this script."
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

ZIP_PATH="$TMP_DIR/godot.zip"
echo "Downloading Godot ${GODOT_VERSION} macOS Universal..."
/usr/bin/curl -L --fail --show-error --output "$ZIP_PATH" "$DOWNLOAD_URL"

echo "Unzipping Godot..."
/usr/bin/unzip -q "$ZIP_PATH" -d "$TMP_DIR/unpacked"

FOUND_APP="$(find "$TMP_DIR/unpacked" -maxdepth 2 -name "Godot*.app" -type d | head -n 1)"
if [[ -z "$FOUND_APP" ]]; then
  echo "Downloaded archive did not contain a Godot .app bundle."
  exit 1
fi

echo "Installing Godot to $GODOT_APP..."
if ! /usr/bin/ditto "$FOUND_APP" "$GODOT_APP" 2>/dev/null; then
  SRC_QUOTED="$(printf "%q" "$FOUND_APP")"
  DST_QUOTED="$(printf "%q" "$GODOT_APP")"
  /usr/bin/osascript -e "do shell script \"ditto $SRC_QUOTED $DST_QUOTED\" with administrator privileges"
fi

/usr/bin/xattr -dr com.apple.quarantine "$GODOT_APP" 2>/dev/null || true

verify_godot
import_project

