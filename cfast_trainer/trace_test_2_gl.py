from __future__ import annotations

import math

from .aircraft_art import panda3d_fixed_wing_hpr_from_tangent
from .trace_test_2 import TraceTest2AircraftTrack, TraceTest2Point3, trace_test_2_track_position


def aircraft_hpr_from_tangent(tangent: tuple[float, float, float]) -> tuple[float, float, float]:
    return panda3d_fixed_wing_hpr_from_tangent(tangent=tangent)


def project_point(point: TraceTest2Point3, *, size: tuple[int, int]) -> tuple[float, float]:
    width = max(1, int(size[0]))
    height = max(1, int(size[1]))
    horizon = height * 0.58
    forward_scale_x = max(2.8, width / 128.0)
    depth_parallax_x = max(0.20, width / 520.0)
    altitude_scale = max(2.0, height / 50.0)
    return (
        (width * 0.5) + (((point.y - 88.0) * forward_scale_x) + (point.x * depth_parallax_x)),
        horizon - ((point.z - 8.0) * altitude_scale),
    )


def tangent_for_track(
    *,
    track: TraceTest2AircraftTrack,
    progress: float,
) -> tuple[float, float, float]:
    pos = trace_test_2_track_position(track=track, progress=progress)
    future = trace_test_2_track_position(track=track, progress=min(1.0, progress + 0.03))
    dx = future.x - pos.x
    dy = future.y - pos.y
    dz = future.z - pos.z
    if (dx * dx) + (dy * dy) + (dz * dz) <= 1e-8:
        past = trace_test_2_track_position(track=track, progress=max(0.0, progress - 0.03))
        dx = pos.x - past.x
        dy = pos.y - past.y
        dz = pos.z - past.z
    return (float(dx), float(dy), float(dz))


def screen_heading_deg(
    *,
    track: TraceTest2AircraftTrack,
    progress: float,
    size: tuple[int, int],
) -> float:
    pos = trace_test_2_track_position(track=track, progress=progress)
    tangent = tangent_for_track(track=track, progress=progress)
    future = TraceTest2Point3(
        x=pos.x + tangent[0],
        y=pos.y + tangent[1],
        z=pos.z + tangent[2],
    )
    px, py = project_point(pos, size=size)
    fx, fy = project_point(future, size=size)
    if abs(fx - px) + abs(fy - py) < 0.01:
        return 0.0
    return float(math.degrees(math.atan2(fy - py, fx - px)))
