from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cfast_trainer.trace_test_2_panda3d import (
    TraceTest2Panda3DRenderer,
    panda3d_trace_test_2_rendering_available,
)


_HELPER = Path(__file__).with_name("_panda3d_runtime_probe.py")


def _run_probe(scene_name: str) -> dict[str, object]:
    env = dict(os.environ)
    env.pop("SDL_VIDEODRIVER", None)
    env.setdefault("SDL_AUDIODRIVER", "dummy")
    env.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    result = subprocess.run(
        [sys.executable, str(_HELPER), scene_name],
        capture_output=True,
        text=True,
        env=env,
        cwd=Path(__file__).resolve().parents[1],
    )
    if result.returncode == 77:
        pytest.skip(result.stdout.strip() or result.stderr.strip() or "Panda3D unavailable")
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines, "subprocess produced no output"
    return json.loads(lines[-1])


def test_panda3d_trace_test_2_rendering_disabled_for_dummy_video(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.delenv("CFAST_TRACE_TEST_2_RENDERER", raising=False)

    assert panda3d_trace_test_2_rendering_available() is False


def test_panda3d_trace_test_2_rendering_disabled_when_forced_to_pygame(monkeypatch) -> None:
    monkeypatch.setenv("CFAST_TRACE_TEST_2_RENDERER", "pygame")
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)

    assert panda3d_trace_test_2_rendering_available() is False


def test_trace_test_2_aircraft_hpr_points_nose_along_motion_tangent() -> None:
    east = TraceTest2Panda3DRenderer._aircraft_hpr_from_tangent(tangent=(1.0, 0.0, 0.0))
    north = TraceTest2Panda3DRenderer._aircraft_hpr_from_tangent(tangent=(0.0, 1.0, 0.0))
    climb = TraceTest2Panda3DRenderer._aircraft_hpr_from_tangent(tangent=(0.0, 1.0, 1.0))

    assert east[0] == pytest.approx(90.0)
    assert north[0] == pytest.approx(0.0)
    assert climb[1] < 0.0


def test_trace_test_2_screen_prefers_panda3d_runtime() -> None:
    probe = _run_probe("trace_test_2")

    assert probe["kind"] == "trace_test_2"
    assert probe["renderer_type"] == "TraceTest2Panda3DRenderer"
    assert probe["gl_scene_type"] is None
    assert probe["renderer_size"][0] > 0
    assert probe["renderer_size"][1] > 0
    assert probe["aircraft_count"] > 0
    assert probe["orientation_count"] > 0
    assert sum(probe["avg_color"]) > 60
