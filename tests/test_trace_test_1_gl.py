from __future__ import annotations

import math

import pytest

from cfast_trainer.trace_test_1 import TraceTest1Generator, TraceTest1PromptPlan
from cfast_trainer.trace_test_1_gl import (
    aircraft_hpr_for_frame,
    build_scene_frames,
    project_scene_position,
    screen_heading_deg,
)


def _sample_prompt(*, difficulty: float = 0.82) -> TraceTest1PromptPlan:
    payload = TraceTest1Generator(seed=44).next_problem(difficulty=difficulty).payload
    assert isinstance(payload, TraceTest1PromptPlan)
    return payload


def test_trace_test_1_aircraft_hpr_matches_heading_pitch_and_roll() -> None:
    prompt = _sample_prompt()
    scene = build_scene_frames(prompt=prompt, progress=0.78)
    hpr = aircraft_hpr_for_frame(scene.red_frame)

    assert len(hpr) == 3
    assert hpr[0] == scene.red_frame.travel_heading_deg
    assert hpr[1] == scene.red_frame.attitude.pitch_deg
    assert hpr[2] == scene.red_frame.attitude.roll_deg


def test_trace_test_1_projection_keeps_red_on_screen_and_projects_blues() -> None:
    prompt = _sample_prompt(difficulty=0.95)
    scene = build_scene_frames(prompt=prompt, progress=0.48)

    red_center, red_scale = project_scene_position(scene.red_frame.position, size=(960, 540))
    assert 0.0 <= red_center[0] <= 960.0
    assert 0.0 <= red_center[1] <= 540.0
    assert red_scale > 0.0

    for blue_frame in scene.blue_frames:
        blue_center, blue_scale = project_scene_position(blue_frame.position, size=(960, 540))
        assert math.isfinite(blue_center[0])
        assert math.isfinite(blue_center[1])
        assert blue_scale > 0.0


def test_trace_test_1_screen_heading_tracks_current_motion_vector() -> None:
    prompt = _sample_prompt()
    scene = build_scene_frames(prompt=prompt, progress=0.62)

    heading = screen_heading_deg(scene.red_frame, size=(960, 540))

    assert math.isfinite(heading)
    assert abs(heading) > 1.0


def test_trace_test_1_screen_headings_stay_cardinal() -> None:
    prompt = _sample_prompt(difficulty=0.95)
    scene = build_scene_frames(prompt=prompt, progress=0.78)

    for frame in (scene.red_frame, *scene.blue_frames):
        heading = screen_heading_deg(frame, size=(960, 540))
        candidates = (-180.0, -90.0, 0.0, 90.0, 180.0)
        assert min(abs(heading - candidate) for candidate in candidates) <= 0.01


def test_trace_test_1_forward_depth_changes_scale_not_vertical_position() -> None:
    near_center, near_scale = project_scene_position((0.0, 8.0, 12.0), size=(960, 540))
    far_center, far_scale = project_scene_position((0.0, 40.0, 12.0), size=(960, 540))

    assert near_center[0] == pytest.approx(far_center[0], abs=0.01)
    assert near_center[1] == pytest.approx(far_center[1], abs=0.01)
    assert near_scale > far_scale


def test_trace_test_1_altitude_changes_vertical_position_without_depth_scale_shift() -> None:
    low_center, low_scale = project_scene_position((0.0, 20.0, 8.0), size=(960, 540))
    high_center, high_scale = project_scene_position((0.0, 20.0, 18.0), size=(960, 540))

    assert low_center[0] == pytest.approx(high_center[0], abs=0.01)
    assert low_center[1] > high_center[1]
    assert low_scale == pytest.approx(high_scale, abs=0.01)
