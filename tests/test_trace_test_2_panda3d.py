from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pygame
import pytest

from cfast_trainer.aircraft_art import fixed_wing_heading_from_screen_heading
from cfast_trainer.trace_test_2 import (
    TraceTest2AircraftTrack,
    TraceTest2MotionKind,
    TraceTest2Point3,
    trace_test_2_track_tangent,
)
from cfast_trainer.trace_test_2_gl import project_point, screen_heading_deg
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


def _project_world_point(
    renderer: TraceTest2Panda3DRenderer,
    *,
    point: TraceTest2Point3,
) -> tuple[float, float]:
    from panda3d.core import Point2, Point3

    relative = renderer._base.cam.getRelativePoint(
        renderer._base.render,
        Point3(point.x, point.y, point.z),
    )
    projected = Point2()
    assert renderer._base.camLens.project(relative, projected)
    return (
        (float(projected.x) + 1.0) * 0.5 * renderer.size[0],
        (1.0 - float(projected.y)) * 0.5 * renderer.size[1],
    )


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


def test_trace_test_2_panda_hpr_prefers_apparent_screen_motion_during_lateral_turn() -> None:
    track = TraceTest2AircraftTrack(
        code=1,
        color_name="RED",
        color_rgb=(220, 64, 72),
        waypoints=(
            TraceTest2Point3(0.0, 38.0, 9.0),
            TraceTest2Point3(0.0, 92.0, 9.0),
            TraceTest2Point3(34.0, 142.0, 9.0),
        ),
        motion_kind=TraceTest2MotionKind.RIGHT,
        direction_changed=True,
        ended_screen_x=18.0,
        ended_altitude_z=9.0,
    )
    progress = 0.5
    size = (960, 540)
    tangent = trace_test_2_track_tangent(track=track, progress=progress)
    world_hpr = TraceTest2Panda3DRenderer._aircraft_hpr_from_tangent(tangent=tangent)
    hpr = TraceTest2Panda3DRenderer._aircraft_hpr_for_track(
        track=track,
        progress=progress,
        size=size,
        tangent=tangent,
    )
    expected_heading = fixed_wing_heading_from_screen_heading(
        screen_heading_deg(track=track, progress=progress, size=size)
    )

    assert hpr[0] == pytest.approx(expected_heading)
    assert abs(hpr[0] - world_hpr[0]) >= 1.0
    assert hpr[2] == pytest.approx(world_hpr[2])


def test_trace_test_2_camera_matches_gl_centering_and_ordering_contract() -> None:
    if not panda3d_trace_test_2_rendering_available():
        pytest.skip("Panda3D unavailable")

    pygame.init()
    renderer = TraceTest2Panda3DRenderer(size=(960, 540))
    try:
        center = TraceTest2Point3(0.0, 96.0, 9.0)
        left = TraceTest2Point3(-30.0, 96.0, 9.0)
        right = TraceTest2Point3(30.0, 96.0, 9.0)
        low = TraceTest2Point3(0.0, 96.0, 0.0)
        high = TraceTest2Point3(0.0, 96.0, 20.0)

        panda_center = _project_world_point(renderer, point=center)
        panda_left = _project_world_point(renderer, point=left)
        panda_right = _project_world_point(renderer, point=right)
        panda_low = _project_world_point(renderer, point=low)
        panda_high = _project_world_point(renderer, point=high)

        gl_center = project_point(center, size=(960, 540))
        gl_left = project_point(left, size=(960, 540))
        gl_right = project_point(right, size=(960, 540))
        gl_low = project_point(low, size=(960, 540))
        gl_high = project_point(high, size=(960, 540))

        assert abs(panda_center[0] - (renderer.size[0] * 0.5)) <= 24.0
        assert abs(gl_center[0] - 480.0) <= 0.01
        assert panda_left[0] < panda_center[0] < panda_right[0]
        assert gl_left[0] < gl_center[0] < gl_right[0]
        assert panda_high[1] < panda_low[1]
        assert gl_high[1] < gl_low[1]
    finally:
        renderer.close()
        pygame.quit()


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
