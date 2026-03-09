from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from .panda3d_protocol import Panda3DRequest, Panda3DResult, Panda3DScene


def panda3d_available() -> bool:
    return importlib.util.find_spec("direct.showbase.ShowBase") is not None


def build_runtime_command(
    *,
    request_path: Path,
    result_path: Path,
    python_executable: str | None = None,
    headless: bool = False,
) -> list[str]:
    cmd = [
        python_executable or sys.executable,
        "-m",
        "cfast_trainer.panda3d_runtime",
        "--request",
        str(request_path),
        "--result",
        str(result_path),
    ]
    if headless:
        cmd.append("--headless")
    return cmd


def launch_runtime(
    request: Panda3DRequest,
    *,
    python_executable: str | None = None,
    headless: bool = False,
    timeout_s: float | None = None,
) -> Panda3DResult:
    if not panda3d_available():
        return Panda3DResult(
            ok=False,
            scene=request.scene,
            summary="Panda3D is not installed in the active environment.",
            metrics={},
        )

    with tempfile.TemporaryDirectory(prefix="cfast-panda3d-") as tmp_dir:
        tmp = Path(tmp_dir)
        request_path = tmp / "request.json"
        result_path = tmp / "result.json"
        request_path.write_text(json.dumps(request.to_dict(), indent=2), encoding="utf-8")
        cmd = build_runtime_command(
            request_path=request_path,
            result_path=result_path,
            python_executable=python_executable,
            headless=headless,
        )
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            detail = stderr or stdout or f"Runtime exited with code {completed.returncode}."
            return Panda3DResult(
                ok=False,
                scene=request.scene,
                summary=detail,
                metrics={},
            )
        if not result_path.exists():
            return Panda3DResult(
                ok=False,
                scene=request.scene,
                summary="Panda3D runtime completed without producing a result file.",
                metrics={},
            )
        raw = json.loads(result_path.read_text(encoding="utf-8"))
        return Panda3DResult.from_dict(raw)


def default_request_for(scene: Panda3DScene) -> Panda3DRequest:
    return Panda3DRequest(scene=scene)
