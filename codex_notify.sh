#!/bin/zsh
set -euo pipefail

ROOT_DIR=${0:A:h}
VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"

if [[ -x "${VENV_PYTHON}" ]]; then
  exec "${VENV_PYTHON}" -m cfast_trainer.macos_notify "$@"
fi

exec python3 -m cfast_trainer.macos_notify "$@"
