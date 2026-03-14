from __future__ import annotations

import math

from .aircraft_art import panda3d_fixed_wing_hpr
from .trace_test_1 import (
    TraceTest1Attitude,
    TraceTest1SceneFrame,
    trace_test_1_scene_frames,
)


def aircraft_hpr_for_frame(frame: TraceTest1SceneFrame) -> tuple[float, float, float]:
    return panda3d_fixed_wing_hpr(
        heading_deg=float(frame.travel_heading_deg),
        pitch_deg=float(frame.attitude.pitch_deg),
        roll_deg=float(frame.attitude.roll_deg),
    )


def build_scene_frames(
    *,
    reference: TraceTest1Attitude,
    candidate: TraceTest1Attitude,
    correct_code: int,
    progress: float,
    scene_turn_index: int,
) -> tuple[TraceTest1SceneFrame, tuple[TraceTest1SceneFrame, ...]]:
    return trace_test_1_scene_frames(
        reference=reference,
        candidate=candidate,
        correct_code=correct_code,
        progress=progress,
        scene_turn_index=scene_turn_index,
    )


def project_scene_position(
    position: tuple[float, float, float],
    *,
    anchor: tuple[float, float, float],
    size: tuple[int, int],
) -> tuple[tuple[float, float], float]:
    width = max(1, int(size[0]))
    height = max(1, int(size[1]))
    horizon = height * 0.58
    rel_x = float(position[0] - anchor[0])
    rel_forward = float(position[1] - anchor[1])
    rel_alt = float(position[2] - anchor[2])
    forward_scale_x = max(2.0, width / 176.0)
    depth_parallax_x = max(0.12, width / 760.0)
    altitude_scale = max(1.45, height / 84.0)
    center = (
        (width * 0.5) + ((rel_forward * forward_scale_x) + (rel_x * depth_parallax_x)),
        horizon - (rel_alt * altitude_scale),
    )
    scale = max(0.54, min(1.20, 0.82 - (rel_x / 210.0)))
    return (float(center[0]), float(center[1])), float(scale)


def screen_heading_deg(
    frame: TraceTest1SceneFrame,
    future_frame: TraceTest1SceneFrame,
    *,
    anchor: tuple[float, float, float],
    size: tuple[int, int],
) -> float:
    heading_rad = math.radians(float(frame.travel_heading_deg))
    ahead_world = (
        frame.position[0] + (math.sin(heading_rad) * 5.4),
        frame.position[1] + (math.cos(heading_rad) * 5.4),
        frame.position[2],
    )
    center, _ = project_scene_position(frame.position, anchor=anchor, size=size)
    ahead_center, _ = project_scene_position(ahead_world, anchor=anchor, size=size)
    _ = future_frame
    dx = float(ahead_center[0] - center[0])
    dy = float(ahead_center[1] - center[1])
    if abs(dx) + abs(dy) < 0.5:
        return 0.0
    return float(math.degrees(math.atan2(dy, dx)))
