from __future__ import annotations

import math

from cfast_trainer.trace_test_1 import TraceTest1Attitude, TraceTest1SceneFrame
from cfast_trainer.trace_test_1_gl import (
    aircraft_hpr_for_frame,
    build_scene_frames,
    project_scene_position,
    screen_heading_deg,
)


def test_trace_test_1_aircraft_hpr_matches_heading_pitch_and_roll() -> None:
    frame = TraceTest1SceneFrame(
        position=(0.0, 0.0, 0.0),
        attitude=TraceTest1Attitude(roll_deg=18.0, pitch_deg=12.0, yaw_deg=0.0),
        travel_heading_deg=90.0,
    )

    hpr = aircraft_hpr_for_frame(frame)

    assert hpr == (90.0, 12.0, 18.0)


def test_trace_test_1_projection_keeps_target_anchor_centered() -> None:
    target_frame, distractor_frames = build_scene_frames(
        reference=TraceTest1Attitude(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0),
        candidate=TraceTest1Attitude(roll_deg=18.0, pitch_deg=10.0, yaw_deg=0.0),
        correct_code=1,
        progress=0.48,
        scene_turn_index=2,
    )

    center, scale = project_scene_position(
        target_frame.position,
        anchor=target_frame.position,
        size=(960, 540),
    )

    assert center == (480.0, 313.2)
    assert 0.54 <= scale <= 1.2
    assert len(distractor_frames) == 3


def test_trace_test_1_screen_heading_tracks_motion() -> None:
    target_frame, _ = build_scene_frames(
        reference=TraceTest1Attitude(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0),
        candidate=TraceTest1Attitude(roll_deg=-16.0, pitch_deg=6.0, yaw_deg=0.0),
        correct_code=2,
        progress=0.35,
        scene_turn_index=1,
    )
    future_target_frame, _ = build_scene_frames(
        reference=TraceTest1Attitude(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0),
        candidate=TraceTest1Attitude(roll_deg=-16.0, pitch_deg=6.0, yaw_deg=0.0),
        correct_code=2,
        progress=0.38,
        scene_turn_index=1,
    )

    heading = screen_heading_deg(
        target_frame,
        future_target_frame,
        anchor=target_frame.position,
        size=(960, 540),
    )

    assert math.isfinite(heading)
    assert abs(heading) > 1.0
