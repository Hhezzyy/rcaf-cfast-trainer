from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


_SOUND_BY_EVENT = {
    "response_ready": "/System/Library/Sounds/Glass.aiff",
    "user_input_required": "/System/Library/Sounds/Submarine.aiff",
}

_TITLE_BY_EVENT = {
    "response_ready": "Codex Response Ready",
    "user_input_required": "Codex Needs Input",
}

_MESSAGE_BY_EVENT = {
    "response_ready": "The latest Codex response is ready.",
    "user_input_required": "Codex is waiting for your input.",
}


def notify(event: str) -> int:
    key = str(event).strip().lower()
    if key not in _TITLE_BY_EVENT:
        return 1

    _send_macos_notification(
        title=_TITLE_BY_EVENT[key],
        message=_MESSAGE_BY_EVENT[key],
    )
    _play_sound(_SOUND_BY_EVENT[key])
    return 0


def _send_macos_notification(*, title: str, message: str) -> None:
    if shutil.which("osascript") is None:
        return
    script = f'display notification "{message}" with title "{title}"'
    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return


def _play_sound(sound_path: str) -> None:
    if shutil.which("afplay") is None:
        return
    path = Path(sound_path)
    if not path.exists():
        return
    try:
        subprocess.run(
            ["afplay", str(path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("Usage: python -m cfast_trainer.macos_notify <response_ready|user_input_required>")
        return 1
    return notify(args[0])


if __name__ == "__main__":
    raise SystemExit(main())
