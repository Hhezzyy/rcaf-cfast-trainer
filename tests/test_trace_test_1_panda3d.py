from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cfast_trainer.trace_test_1_panda3d import panda3d_trace_test_1_rendering_available


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


def test_panda3d_trace_test_1_rendering_disabled_for_dummy_video(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.delenv("CFAST_TRACE_TEST_1_RENDERER", raising=False)

    assert panda3d_trace_test_1_rendering_available() is False


def test_panda3d_trace_test_1_rendering_disabled_when_forced_to_pygame(monkeypatch) -> None:
    monkeypatch.setenv("CFAST_TRACE_TEST_1_RENDERER", "pygame")
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)

    assert panda3d_trace_test_1_rendering_available() is False


def test_trace_test_1_screen_prefers_panda3d_runtime() -> None:
    probe = _run_probe("trace_test_1")

    assert probe["kind"] == "trace_test_1"
    assert probe["renderer_type"] == "TraceTest1Panda3DRenderer"
    assert probe["gl_scene_type"] is None
    assert probe["renderer_size"][0] > 0
    assert probe["renderer_size"][1] > 0
    assert probe["aircraft_count"] >= 2
    assert probe["blue_count"] >= 1
    assert len(probe["red_hpr"]) == 3
    assert probe["blue_hpr_count"] == probe["blue_count"]
    assert sum(probe["avg_color"]) > 60
