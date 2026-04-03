from __future__ import annotations

import pytest

from cfast_trainer.trace_scene_3d import (
    build_trace_test_2_scene3d,
    trace_test_2_track_sample_points,
)
from cfast_trainer.trace_test_2 import TraceTest2Generator, TraceTest2Payload


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
