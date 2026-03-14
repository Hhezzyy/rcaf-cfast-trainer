from __future__ import annotations

import math

import pytest

from cfast_trainer.trace_test_2 import build_trace_test_2_test
from cfast_trainer.trace_test_2_gl import (
    aircraft_hpr_from_tangent,
    project_point,
    screen_heading_deg,
    tangent_for_track,
)


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t


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
