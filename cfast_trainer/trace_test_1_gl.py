from __future__ import annotations

from .aircraft_art import screen_heading_deg_from_world_tangent
from .trace_test_1 import (
    TraceTest1Command,
    TraceTest1Payload,
    TraceTest1PromptPlan,
    TraceTest1SceneFrame,
    TraceTest1SceneSnapshot,
    trace_test_1_aircraft_hpr,
    trace_test_1_scene_frames,
)


_LEFT_SCREEN_HEADING_DEG = -160.0
_RIGHT_SCREEN_HEADING_DEG = 0.0
_PUSH_SCREEN_HEADING_DEG = 90.0
_PULL_SCREEN_HEADING_DEG = -90.0
_STRAIGHT_SCREEN_HEADING_DEG = -90.0
_WORLD_X_SCREEN_NORMALIZER = 48.0
_WORLD_Z_SCREEN_CENTER = 12.0
_WORLD_Z_SCREEN_NORMALIZER = 36.0
_WORLD_DEPTH_SCREEN_CENTER = 26.0
_WORLD_DEPTH_X_MIX = 0.0
_WORLD_DEPTH_Y_MIX = 0.31


def _command_screen_heading(command: TraceTest1Command) -> float:
    if command is TraceTest1Command.LEFT:
        return _LEFT_SCREEN_HEADING_DEG
    if command is TraceTest1Command.RIGHT:
        return _RIGHT_SCREEN_HEADING_DEG
    if command is TraceTest1Command.PUSH:
        return _PUSH_SCREEN_HEADING_DEG
    return _PULL_SCREEN_HEADING_DEG


def _visible_screen_heading(
    *,
    command: TraceTest1Command,
    observe_progress: float,
    answer_open_progress: float,
) -> float:
    if float(observe_progress) < float(answer_open_progress):
        return _STRAIGHT_SCREEN_HEADING_DEG
    return _command_screen_heading(command)


def aircraft_hpr_for_frame(frame: TraceTest1SceneFrame) -> tuple[float, float, float]:
    return trace_test_1_aircraft_hpr(frame)


def aircraft_screen_pose_for_frame(
    frame: TraceTest1SceneFrame,
    future_frame: TraceTest1SceneFrame | None = None,
    *,
    command: TraceTest1Command,
    observe_progress: float,
    answer_open_progress: float,
    anchor: tuple[float, float, float] | None = None,
    size: tuple[int, int],
) -> tuple[float, float, float]:
    return (
        float(
            screen_heading_deg(
                frame,
                future_frame,
                command=command,
                observe_progress=observe_progress,
                answer_open_progress=answer_open_progress,
                anchor=anchor,
                size=size,
            )
        ),
        float(frame.attitude.pitch_deg),
        float(frame.attitude.roll_deg),
    )


def aircraft_screen_poses_for_payload(
    payload: TraceTest1Payload,
    *,
    size: tuple[int, int],
) -> tuple[tuple[float, float, float], tuple[tuple[float, float, float], ...]]:
    red_pose = aircraft_screen_pose_for_frame(
        payload.scene.red_frame,
        command=payload.active_command,
        observe_progress=payload.observe_progress,
        answer_open_progress=payload.answer_open_progress,
        size=size,
    )
    blue_poses = tuple(
        aircraft_screen_pose_for_frame(
            frame,
            command=command,
            observe_progress=payload.observe_progress,
            answer_open_progress=payload.answer_open_progress,
            size=size,
        )
        for frame, command in zip(payload.scene.blue_frames, payload.blue_commands, strict=True)
    )
    return red_pose, blue_poses


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
    world_x = float(position[0])
    world_y = float(position[1])
    world_z = float(position[2])
    depth_offset = world_y - _WORLD_DEPTH_SCREEN_CENTER
    center = (
        float(
            (width * 0.5)
            + (
                (
                    world_x
                    + (depth_offset * _WORLD_DEPTH_X_MIX)
                )
                * (width / _WORLD_X_SCREEN_NORMALIZER)
            )
        ),
        float(
            (height * 0.64)
            - (
                (
                    (world_z - _WORLD_Z_SCREEN_CENTER)
                    + (depth_offset * _WORLD_DEPTH_Y_MIX)
                )
                * (height / _WORLD_Z_SCREEN_NORMALIZER)
            )
        ),
    )
    depth_t = max(0.0, min(1.0, (world_y - 4.0) / 44.0))
    scale = 1.22 - (depth_t * 0.50)
    return center, float(scale)


def screen_heading_deg(
    frame: TraceTest1SceneFrame,
    future_frame: TraceTest1SceneFrame | None = None,
    *,
    command: TraceTest1Command,
    observe_progress: float,
    answer_open_progress: float,
    anchor: tuple[float, float, float] | None = None,
    size: tuple[int, int],
) -> float:
    _ = (anchor, future_frame, size, command, observe_progress, answer_open_progress)
    heading = screen_heading_deg_from_world_tangent(frame.world_tangent)
    if heading is not None:
        return float(heading)
    return _visible_screen_heading(
        command=command,
        observe_progress=observe_progress,
        answer_open_progress=answer_open_progress,
    )
