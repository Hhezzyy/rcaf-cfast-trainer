from __future__ import annotations

import math

from .aircraft_art import fixed_wing_hpr_from_world_tangent, screen_motion_heading_deg
from .trace_test_2 import (
    TraceTest2AircraftTrack,
    TraceTest2Point3,
    trace_test_2_track_position,
    trace_test_2_track_tangent,
)


def aircraft_hpr_from_tangent(tangent: tuple[float, float, float]) -> tuple[float, float, float]:
    return fixed_wing_hpr_from_world_tangent(
        tangent=tangent,
        roll_deg=0.0,
    )


def aircraft_screen_pose_for_track(
    *,
    track: TraceTest2AircraftTrack,
    progress: float,
    size: tuple[int, int],
    tangent: tuple[float, float, float] | None = None,
) -> tuple[float, float, float]:
    tangent_value = tangent_for_track(track=track, progress=progress) if tangent is None else tangent
    hpr = aircraft_hpr_from_tangent(tangent_value)
    return (
        float(
            screen_heading_deg_for_tangent(
                point=trace_test_2_track_position(track=track, progress=progress),
                tangent=tangent_value,
                size=size,
            )
        ),
        float(hpr[1]),
        float(hpr[2]),
    )


def project_point(point: TraceTest2Point3, *, size: tuple[int, int]) -> tuple[float, float]:
    width = max(1, int(size[0]))
    height = max(1, int(size[1]))
    horizon = height * 0.76
    depth = max(24.0, float(point.y))
    depth_factor = 1.0 / (1.0 + ((depth - 62.0) / 136.0))
    return (
        (width * 0.5) + (point.x * width * 0.015 * depth_factor),
        horizon - (point.z * height * 0.030) - ((depth - 62.0) * height * 0.010),
    )


def tangent_for_track(
    *,
    track: TraceTest2AircraftTrack,
    progress: float,
) -> tuple[float, float, float]:
    return trace_test_2_track_tangent(track=track, progress=progress)


def screen_heading_deg_for_tangent(
    *,
    point: TraceTest2Point3,
    tangent: tuple[float, float, float],
    size: tuple[int, int],
) -> float:
    width = max(1, int(size[0]))
    height = max(1, int(size[1]))
    depth = max(24.0, float(point.y))
    depth_factor = 1.0 / (1.0 + ((depth - 62.0) / 136.0))
    x_scale = width * 0.015 * depth_factor
    d_depth_factor_dy = 0.0 if float(point.y) <= 24.0 else (-136.0 / ((depth + 74.0) ** 2))
    screen_dx = (float(tangent[0]) * x_scale) + (
        float(point.x) * width * 0.015 * d_depth_factor_dy * float(tangent[1])
    )
    screen_dy = -(float(tangent[2]) * height * 0.030) - (
        (0.0 if float(point.y) <= 24.0 else float(tangent[1])) * height * 0.010
    )
    heading = screen_motion_heading_deg(
        (0.0, 0.0),
        (screen_dx, screen_dy),
        minimum_distance=0.01,
    )
    if heading is None:
        raise ValueError("trace_test_2 screen heading requires a non-degenerate tangent")
    return float(heading)


def screen_heading_deg(
    *,
    track: TraceTest2AircraftTrack,
    progress: float,
    size: tuple[int, int],
) -> float:
    pos = trace_test_2_track_position(track=track, progress=progress)
    tangent = tangent_for_track(track=track, progress=progress)
    return screen_heading_deg_for_tangent(
        point=pos,
        tangent=tangent,
        size=size,
    )
