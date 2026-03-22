from __future__ import annotations

import math

from .aircraft_art import screen_motion_heading_deg
from .trace_test_1 import (
    TraceTest1PromptPlan,
    TraceTest1SceneFrame,
    TraceTest1SceneSnapshot,
    trace_test_1_aircraft_hpr,
    trace_test_1_normalized_position,
    trace_test_1_scene_frames,
)


def _screen_heading_from_world_heading(heading_deg: float) -> float:
    wrapped = float(heading_deg) % 360.0
    cardinal = int(round(wrapped / 90.0)) % 4
    return (-90.0, 0.0, 90.0, 180.0)[cardinal]


def aircraft_hpr_for_frame(frame: TraceTest1SceneFrame) -> tuple[float, float, float]:
    return trace_test_1_aircraft_hpr(frame)


def build_scene_frames(
    *,
    prompt: TraceTest1PromptPlan,
    progress: float,
) -> TraceTest1SceneSnapshot:
    return trace_test_1_scene_frames(prompt=prompt, progress=progress)


def project_scene_position(
    position: tuple[float, float, float],
    *,
    anchor: tuple[float, float, float] | None = None,
    size: tuple[int, int],
) -> tuple[tuple[float, float], float]:
    _ = anchor
    width = max(1, int(size[0]))
    height = max(1, int(size[1]))
    nx, ny = trace_test_1_normalized_position(position)
    center = (
        float(nx * width),
        float(ny * height),
    )
    depth_t = max(0.0, min(1.0, (float(position[1]) - 4.0) / 44.0))
    scale = 1.22 - (depth_t * 0.50)
    return center, float(scale)


def screen_heading_deg(
    frame: TraceTest1SceneFrame,
    future_frame: TraceTest1SceneFrame | None = None,
    *,
    anchor: tuple[float, float, float] | None = None,
    size: tuple[int, int],
) -> float:
    world_heading = float(frame.travel_heading_deg) % 360.0
    if abs((world_heading / 90.0) - round(world_heading / 90.0)) <= 1e-4:
        return _screen_heading_from_world_heading(world_heading)

    center, _ = project_scene_position(frame.position, anchor=anchor, size=size)
    if future_frame is not None:
        future_center, _ = project_scene_position(future_frame.position, anchor=anchor, size=size)
        heading = screen_motion_heading_deg(center, future_center)
        if heading is not None:
            return heading

    heading_rad = math.radians(float(frame.travel_heading_deg))
    ahead_world = (
        frame.position[0] + (math.sin(heading_rad) * 4.0),
        frame.position[1] + (math.cos(heading_rad) * 4.0),
        frame.position[2] + (math.sin(math.radians(float(frame.attitude.pitch_deg))) * 1.4),
    )
    ahead_center, _ = project_scene_position(ahead_world, anchor=anchor, size=size)
    heading = screen_motion_heading_deg(center, ahead_center)
    if heading is None:
        return 0.0
    return heading
