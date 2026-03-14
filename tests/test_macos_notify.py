from __future__ import annotations

from pathlib import Path

from cfast_trainer import macos_notify


def test_notify_invokes_notification_and_sound(monkeypatch, tmp_path) -> None:
    sound_path = tmp_path / "notify.aiff"
    sound_path.write_bytes(b"sound")

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs) -> None:
        calls.append(list(cmd))

    monkeypatch.setattr(
        macos_notify,
        "_SOUND_BY_EVENT",
        {
            "response_ready": str(sound_path),
            "user_input_required": str(sound_path),
        },
    )
    monkeypatch.setattr(macos_notify.shutil, "which", lambda _cmd: "/usr/bin/tool")
    monkeypatch.setattr(macos_notify.subprocess, "run", fake_run)

    assert macos_notify.notify("response_ready") == 0
    assert calls[0][0] == "osascript"
    assert calls[1] == ["afplay", str(sound_path)]


def test_notify_rejects_unknown_event() -> None:
    assert macos_notify.notify("unknown") == 1
