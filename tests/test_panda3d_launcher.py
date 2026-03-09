from __future__ import annotations

import json
from pathlib import Path

from cfast_trainer.panda3d_launcher import build_runtime_command, launch_runtime
from cfast_trainer.panda3d_protocol import Panda3DRequest, Panda3DResult, Panda3DScene


def test_build_runtime_command_includes_headless_flag(tmp_path) -> None:
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"

    cmd = build_runtime_command(
        request_path=request_path,
        result_path=result_path,
        python_executable="/tmp/python",
        headless=True,
    )

    assert cmd == [
        "/tmp/python",
        "-m",
        "cfast_trainer.panda3d_runtime",
        "--request",
        str(request_path),
        "--result",
        str(result_path),
        "--headless",
    ]


def test_launch_runtime_returns_runtime_result_when_subprocess_succeeds(
    monkeypatch,
    tmp_path,
) -> None:
    request = Panda3DRequest(scene=Panda3DScene.RAPID_TRACKING, duration_s=0.1)
    expected = Panda3DResult(
        ok=True,
        scene=Panda3DScene.RAPID_TRACKING,
        summary="ok",
        metrics={"duration_s": 0.1},
    )

    monkeypatch.setattr("cfast_trainer.panda3d_launcher.panda3d_available", lambda: True)

    def fake_run(cmd, *, check, capture_output, text, timeout):
        _ = (cmd, check, capture_output, text, timeout)
        result_path = Path(cmd[cmd.index("--result") + 1])
        result_path.write_text(json.dumps(expected.to_dict()), encoding="utf-8")

        class Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        return Completed()

    monkeypatch.setattr("subprocess.run", fake_run)

    result = launch_runtime(request, python_executable="/tmp/python", headless=True, timeout_s=1.0)

    assert result == expected


def test_launch_runtime_returns_install_message_when_panda3d_missing(monkeypatch) -> None:
    monkeypatch.setattr("cfast_trainer.panda3d_launcher.panda3d_available", lambda: False)

    result = launch_runtime(Panda3DRequest(scene=Panda3DScene.SPATIAL_INTEGRATION))

    assert result.ok is False
    assert "not installed" in result.summary.lower()
