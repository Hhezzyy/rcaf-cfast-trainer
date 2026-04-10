from __future__ import annotations

import pytest

from cfast_trainer.aircraft_art import screen_motion_heading_deg
from cfast_trainer.trace_lattice import TraceLatticeAction
from cfast_trainer.trace_scene_3d import (
    build_trace_test_2_scene3d,
    trace_test_2_track_sample_points,
)
from cfast_trainer.trace_test_2 import (
    TraceTest2Generator,
    TraceTest2MotionKind,
    TraceTest2Payload,
    trace_test_2_track_position,
)
from cfast_trainer.trace_test_2_gl import aircraft_screen_pose_for_track, project_point


def _angle_delta_deg(current: float, reference: float) -> float:
    return ((float(current) - float(reference) + 180.0) % 360.0) - 180.0


def _sample_payload(*, difficulty: float = 0.58) -> TraceTest2Payload:
    payload = TraceTest2Generator(seed=71).next_problem(difficulty=difficulty).payload
    assert isinstance(payload, TraceTest2Payload)
    return payload


def test_trace_test_2_scene3d_builder_is_deterministic_for_same_payload() -> None:
    payload = _sample_payload()

    first = build_trace_test_2_scene3d(payload=payload, practice_mode=False)
    second = build_trace_test_2_scene3d(payload=payload, practice_mode=False)

    assert first == second
    assert len(first.aircraft) == len(payload.aircraft)
    assert first.camera.target[1] > first.camera.position[1]


def test_trace_test_2_practice_scene_adds_ghost_trails_without_moving_live_aircraft() -> None:
    payload = _sample_payload()

    scored = build_trace_test_2_scene3d(payload=payload, practice_mode=False)
    practice = build_trace_test_2_scene3d(payload=payload, practice_mode=True)

    assert practice.aircraft == scored.aircraft
    assert len(practice.ghosts) > 0
    assert all(ghost.scale[0] < practice.aircraft[0].scale[0] for ghost in practice.ghosts)


def test_trace_test_2_track_sample_points_match_track_endpoints() -> None:
    payload = _sample_payload()

    for track in payload.aircraft:
        samples = trace_test_2_track_sample_points(track=track)
        assert samples[0] == pytest.approx(
            (track.waypoints[0].x, track.waypoints[0].y, track.waypoints[0].z)
        )
        end = track.waypoints[-1]
        assert samples[-1] == pytest.approx((end.x, end.y, end.z))


def test_trace_test_2_projected_aircraft_centers_move_between_observe_samples() -> None:
    payload = _sample_payload()

    for track in payload.aircraft:
        early = project_point(
            trace_test_2_track_position(track=track, progress=0.18),
            size=(640, 360),
        )
        late = project_point(
            trace_test_2_track_position(track=track, progress=0.82),
            size=(640, 360),
        )

        assert early != pytest.approx(late)


def test_trace_test_2_projected_turning_tracks_move_during_turn_window() -> None:
    payload = _sample_payload()

    for track in payload.aircraft:
        if track.motion_kind is TraceTest2MotionKind.STRAIGHT:
            continue
        assert track.lattice_path is not None
        turn_index = next(
            idx
            for idx, step in enumerate(track.lattice_path.steps)
            if step.effective_action is not TraceLatticeAction.STRAIGHT
        )
        step_count = len(track.lattice_path.steps)
        early = project_point(
            trace_test_2_track_position(
                track=track,
                progress=(float(turn_index) + 0.08) / float(step_count),
            ),
            size=(640, 360),
        )
        mid = project_point(
            trace_test_2_track_position(
                track=track,
                progress=(float(turn_index) + 0.20) / float(step_count),
            ),
            size=(640, 360),
        )

        assert mid != pytest.approx(early)


def test_trace_test_2_screen_pose_matches_projected_motion_for_representative_tracks() -> None:
    payload = _sample_payload()
    tracks = {track.motion_kind: track for track in payload.aircraft}

    for motion_kind in (
        TraceTest2MotionKind.STRAIGHT,
        TraceTest2MotionKind.LEFT,
        TraceTest2MotionKind.RIGHT,
        TraceTest2MotionKind.CLIMB,
    ):
        track = tracks[motion_kind]
        start = project_point(
            trace_test_2_track_position(track=track, progress=0.45),
            size=(640, 360),
        )
        end = project_point(
            trace_test_2_track_position(track=track, progress=0.55),
            size=(640, 360),
        )
        motion_heading = screen_motion_heading_deg(start, end, minimum_distance=0.01)
        assert motion_heading is not None

        pose = aircraft_screen_pose_for_track(
            track=track,
            progress=0.50,
            size=(640, 360),
        )

        assert _angle_delta_deg(pose[0], motion_heading) == pytest.approx(0.0, abs=1.1)


def test_trace_test_2_screen_pose_progresses_toward_turn_heading_during_turn_window() -> None:
    payload = _sample_payload()
    right_track = next(track for track in payload.aircraft if track.motion_kind is TraceTest2MotionKind.RIGHT)
    assert right_track.lattice_path is not None
    turn_index = next(
        idx
        for idx, step in enumerate(right_track.lattice_path.steps)
        if step.effective_action is not TraceLatticeAction.STRAIGHT
    )
    step_count = len(right_track.lattice_path.steps)
    early_progress = (float(turn_index) + 0.08) / float(step_count)
    mid_progress = (float(turn_index) + 0.20) / float(step_count)
    early_pose = aircraft_screen_pose_for_track(
        track=right_track,
        progress=early_progress,
        size=(640, 360),
    )
    mid_pose = aircraft_screen_pose_for_track(
        track=right_track,
        progress=mid_progress,
        size=(640, 360),
    )

    assert abs(_angle_delta_deg(mid_pose[0], 0.0)) < abs(_angle_delta_deg(early_pose[0], 0.0))
