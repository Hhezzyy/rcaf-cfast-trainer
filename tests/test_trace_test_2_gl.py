from __future__ import annotations

import math

import pytest

from cfast_trainer.trace_test_2 import (
    TraceTest2AircraftTrack,
    TraceTest2Generator,
    TraceTest2MotionKind,
    TraceTest2Payload,
    TraceTest2Point3,
    build_trace_test_2_test,
)
from cfast_trainer.trace_test_2_gl import (
    aircraft_hpr_from_tangent,
    aircraft_screen_pose_for_track,
    project_point,
    screen_heading_deg,
    screen_heading_deg_for_tangent,
    tangent_for_track,
)


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t


def _angle_delta_deg(current: float, reference: float) -> float:
    return ((float(current) - float(reference) + 180.0) % 360.0) - 180.0


def _turn_track(*, motion_kind: TraceTest2MotionKind) -> TraceTest2AircraftTrack:
    end_x = -34.0 if motion_kind is TraceTest2MotionKind.LEFT else 34.0
    return TraceTest2AircraftTrack(
        code=1,
        color_name="RED",
        color_rgb=(220, 64, 72),
        waypoints=(
            TraceTest2Point3(0.0, 38.0, 9.0),
            TraceTest2Point3(0.0, 92.0, 9.0),
            TraceTest2Point3(end_x, 142.0, 9.0),
        ),
        motion_kind=motion_kind,
        direction_changed=True,
        ended_screen_x=end_x,
        ended_altitude_z=9.0,
    )


def test_trace_test_2_aircraft_hpr_points_nose_along_motion_tangent() -> None:
    east = aircraft_hpr_from_tangent((1.0, 0.0, 0.0))
    north = aircraft_hpr_from_tangent((0.0, 1.0, 0.0))
    climb = aircraft_hpr_from_tangent((0.0, 1.0, 1.0))

    assert east[0] == pytest.approx(90.0)
    assert north[0] == pytest.approx(0.0)
    assert climb[1] < 0.0


def test_trace_test_2_projection_and_heading_follow_payload_tracks() -> None:
    clock = _FakeClock()
    engine = build_trace_test_2_test(clock=clock, seed=71, difficulty=0.58)
    engine.start_practice()
    payload = engine.snapshot().payload

    assert payload is not None
    assert payload.aircraft

    for track in payload.aircraft:
        tangent = tangent_for_track(track=track, progress=0.5)
        hpr = aircraft_hpr_from_tangent(tangent)
        start = project_point(track.waypoints[0], size=(960, 540))
        heading = screen_heading_deg(track=track, progress=0.5, size=(960, 540))

        assert len(hpr) == 3
        assert math.isfinite(start[0])
        assert math.isfinite(start[1])
        assert math.isfinite(heading)


def test_trace_test_2_screen_heading_rejects_zero_motion() -> None:
    with pytest.raises(ValueError, match="non-degenerate"):
        screen_heading_deg_for_tangent(
            point=TraceTest2Point3(0.0, 38.0, 9.0),
            tangent=(0.0, 0.0, 0.0),
            size=(960, 540),
        )


def test_trace_test_2_left_turn_screen_heading_changes_in_correct_direction() -> None:
    track = _turn_track(motion_kind=TraceTest2MotionKind.LEFT)
    headings = [
        screen_heading_deg(track=track, progress=progress, size=(960, 540))
        for progress in (0.25, 0.40, 0.55, 0.70)
    ]
    deltas = [_angle_delta_deg(heading, -90.0) for heading in headings]

    assert all(later < earlier for earlier, later in zip(deltas, deltas[1:], strict=False))
    assert all(3.0 <= abs(later - earlier) <= 40.0 for earlier, later in zip(headings, headings[1:], strict=False))
    assert all(min(abs((heading % 90.0)), abs((heading % 90.0) - 90.0)) > 2.0 for heading in headings)


def test_trace_test_2_right_turn_screen_heading_changes_in_correct_direction() -> None:
    track = _turn_track(motion_kind=TraceTest2MotionKind.RIGHT)
    headings = [
        screen_heading_deg(track=track, progress=progress, size=(960, 540))
        for progress in (0.25, 0.40, 0.55, 0.70)
    ]
    deltas = [_angle_delta_deg(heading, -90.0) for heading in headings]

    assert all(later > earlier for earlier, later in zip(deltas, deltas[1:], strict=False))
    assert all(3.0 <= abs(later - earlier) <= 40.0 for earlier, later in zip(headings, headings[1:], strict=False))
    assert all(min(abs((heading % 90.0)), abs((heading % 90.0) - 90.0)) > 2.0 for heading in headings)


def test_trace_test_2_multi_aircraft_screen_pose_matches_lattice_motion_kind() -> None:
    payload = TraceTest2Generator(seed=71).next_problem(difficulty=0.58).payload

    assert isinstance(payload, TraceTest2Payload)

    for track in payload.aircraft:
        pose = aircraft_screen_pose_for_track(track=track, progress=0.5, size=(960, 540))
        tangent = tangent_for_track(track=track, progress=0.5)

        assert math.isfinite(pose[0])
        assert pose[2] == pytest.approx(aircraft_hpr_from_tangent(tangent)[2])
        assert pose[1] == pytest.approx(aircraft_hpr_from_tangent(tangent)[1])
        assert pose[2] == pytest.approx(0.0, abs=0.01)

        if track.motion_kind is TraceTest2MotionKind.LEFT:
            assert pose[0] == pytest.approx(-180.0)
        elif track.motion_kind is TraceTest2MotionKind.RIGHT:
            assert pose[0] == pytest.approx(0.0)
        elif track.motion_kind is TraceTest2MotionKind.CLIMB:
            assert pose[1] < 0.0
        elif track.motion_kind is TraceTest2MotionKind.DESCEND:
            assert pose[1] > 0.0
        else:
            assert abs(pose[1]) <= 0.01


def test_trace_test_2_screen_pose_sampling_is_deterministic_for_same_seed() -> None:
    progresses = (0.18, 0.42, 0.66, 0.84)
    payload_a = TraceTest2Generator(seed=71).next_problem(difficulty=0.58).payload
    payload_b = TraceTest2Generator(seed=71).next_problem(difficulty=0.58).payload

    assert isinstance(payload_a, TraceTest2Payload)
    assert isinstance(payload_b, TraceTest2Payload)
    assert payload_a.aircraft == payload_b.aircraft

    poses_a = tuple(
        tuple(
            aircraft_screen_pose_for_track(
                track=track,
                progress=progress,
                size=(960, 540),
            )
            for track in payload_a.aircraft
        )
        for progress in progresses
    )
    poses_b = tuple(
        tuple(
            aircraft_screen_pose_for_track(
                track=track,
                progress=progress,
                size=(960, 540),
            )
            for track in payload_b.aircraft
        )
        for progress in progresses
    )

    assert poses_a == poses_b
